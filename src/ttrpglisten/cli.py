"""CLI interface and main orchestration for TTRPGListen."""

from __future__ import annotations

import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue

from rich.console import Console
from rich.table import Table

from .audio import DualAudioCapture, list_devices
from .config import load_config, resolve_device
from .display import TranscriptDisplay
from .pipeline import StreamingPipeline
from .transcribe import TranscriptionEngine


def print_devices():
    """Print available audio devices in a formatted table."""
    console = Console()
    devices = list_devices()
    table = Table(title="Available Audio Devices")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Name", style="white")
    table.add_column("Input Ch", justify="right")
    table.add_column("Output Ch", justify="right")
    table.add_column("Sample Rate", justify="right")
    table.add_column("Host API", style="dim")

    for dev in devices:
        if dev["max_input_channels"] > 0 or dev["max_output_channels"] > 0:
            table.add_row(
                str(dev["index"]),
                dev["name"],
                str(dev["max_input_channels"]),
                str(dev["max_output_channels"]),
                str(int(dev["default_samplerate"])),
                dev["hostapi"],
            )

    console.print(table)
    console.print("\n[dim]Tip: Loopback devices capture system audio. "
                  "Use --loopback-device ID for remote players, --mic-device ID for your mic.[/dim]")


def _save_live_transcript(
    cfg, timestamp: str, lines: list[str], console: Console, label: str = "Live transcript saved",
) -> Path:
    """Save streaming transcript lines to a text file."""
    output_dir = Path(cfg.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"session_{timestamp}_live.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    console.print(f"\n[green]{label}:[/green] {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ttrpglisten",
        description="Local AI-powered live transcription for TTRPG sessions",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--loopback-device", type=int, default=None,
        help="Audio device ID for system loopback (remote players)",
    )
    parser.add_argument(
        "--mic-device", type=int, default=None,
        help="Audio device ID for microphone (your voice)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--preset", choices=["low", "medium", "high"], default=None,
        help="Quality preset (overrides config model settings)",
    )
    parser.add_argument(
        "--no-postprocess", action="store_true",
        help="Skip post-session processing",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device: auto, cpu, cuda, mps",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_devices:
        print_devices()
        return

    console = Console()

    # Load config
    cfg = load_config(config_path=args.config, preset=args.preset)

    # CLI overrides
    if args.loopback_device is not None:
        cfg.audio.loopback_device = args.loopback_device
    if args.mic_device is not None:
        cfg.audio.mic_device = args.mic_device
    if args.device:
        cfg.streaming.device = args.device

    # Resolve compute device
    device = resolve_device(cfg.streaming.device)
    console.print(f"[bold]Compute device:[/bold] {device}")

    # Session setup
    session_start = datetime.now()
    timestamp = session_start.strftime("%Y-%m-%d_%H%M")
    wav_path = Path(cfg.output.directory) / f"session_{timestamp}.wav"

    # Queues connecting the pipeline stages
    audio_queue: Queue = Queue(maxsize=200)   # raw audio chunks
    text_queue: Queue = Queue(maxsize=100)    # streaming text updates

    # Initialize components
    console.print("[bold]Loading transcription model...[/bold]")
    engine = TranscriptionEngine(cfg.streaming.model, device=device)
    engine.load()
    console.print(f"[green]Model loaded:[/green] {cfg.streaming.model}")

    capture = DualAudioCapture(audio_queue, cfg.audio, wav_path=wav_path)
    pipeline = StreamingPipeline(engine, audio_queue, text_queue, sample_rate=cfg.audio.sample_rate)
    display = TranscriptDisplay(text_queue)

    # Shutdown handler
    shutdown_count = 0

    def signal_handler(signum, frame):
        nonlocal shutdown_count
        shutdown_count += 1
        if shutdown_count == 1:
            console.print("\n[yellow]Stopping capture... (Ctrl+C again to skip post-processing)[/yellow]")
            capture.stop()
            pipeline.stop()
            display.stop()
        elif shutdown_count >= 2:
            console.print("\n[red]Skipping post-processing.[/red]")
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start pipeline
    console.print("[bold green]Starting live transcription...[/bold green]\n")

    try:
        capture.start()
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        del engine
        return

    # Log active devices
    lb_name = capture.loopback_name
    if lb_name:
        console.print(f"  Loopback: [cyan]{lb_name}[/cyan]")
    else:
        console.print("  [yellow]No loopback device found (system audio not captured)[/yellow]")
        console.print("  [dim]Install pyaudiowpatch for WASAPI loopback: pip install pyaudiowpatch[/dim]")

    mic_name = capture.mic_name
    if mic_name:
        console.print(f"  Mic: [cyan]{mic_name}[/cyan]")
    else:
        console.print("  [yellow]No mic device (your voice not captured)[/yellow]")

    console.print()

    pipeline.start()
    display.start()

    try:
        # Wait for shutdown signal
        try:
            signal.pause()
        except AttributeError:
            import time
            while shutdown_count == 0:
                time.sleep(0.5)

        # Post-processing
        if not args.no_postprocess and shutdown_count < 2:
            console.print("\n[bold]Running post-session processing...[/bold]")
            try:
                from .postprocess import postprocess
                postprocess(wav_path, cfg, session_start, display.get_full_transcript())
            except Exception as e:
                console.print(f"[red]Post-processing failed:[/red] {e}")
                _save_live_transcript(
                    cfg, timestamp, display.get_full_transcript(), console,
                    label="Live transcript saved as fallback",
                )
        elif shutdown_count < 2:
            _save_live_transcript(cfg, timestamp, display.get_full_transcript(), console)
    finally:
        if shutdown_count == 0:
            capture.stop()
            pipeline.stop()
            display.stop()

    console.print("[bold]Session ended.[/bold]")
