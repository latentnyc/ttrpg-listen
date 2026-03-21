"""Post-session processing: high-quality re-transcription with optional diarization."""

from __future__ import annotations

import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config, resolve_device
from .transcribe import TranscriptionEngine


def load_wav_channels(wav_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load stereo WAV and return (loopback, mic, sample_rate) channels."""
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels == 2:
        loopback = data[0::2]
        mic = data[1::2]
    else:
        loopback = data
        mic = data

    return loopback, mic, sample_rate


def postprocess(
    wav_path: Path,
    config: Config,
    session_start: datetime,
    streaming_lines: list[str] | None = None,
) -> Path:
    """Run post-processing: re-transcribe with larger model, optionally diarize.

    Returns path to the output transcript file.
    """
    console = Console()
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = session_start.strftime("%Y-%m-%d_%H%M")
    output_path = output_dir / f"session_{timestamp}.txt"

    device = resolve_device(config.streaming.device)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load audio
        task = progress.add_task("Loading recorded audio...", total=None)
        loopback, mic, sample_rate = load_wav_channels(wav_path)
        # Mix for transcription
        if len(loopback) == len(mic):
            full_audio = (loopback + mic) * 0.5
        else:
            full_audio = loopback if len(loopback) > len(mic) else mic
        progress.update(task, description="Audio loaded", completed=True)

        # Load larger model
        task2 = progress.add_task(
            f"Loading model {config.postprocess.model}...", total=None
        )
        engine = TranscriptionEngine(config.postprocess.model, device=device)
        engine.load()
        progress.update(task2, description="Model loaded", completed=True)

        # Transcribe in chunks (60-second windows with overlap)
        task3 = progress.add_task("Transcribing full session...", total=None)
        chunk_size = 60 * sample_rate  # 60 seconds
        overlap = 2 * sample_rate  # 2-second overlap
        segments = []
        pos = 0

        while pos < len(full_audio):
            end = min(pos + chunk_size, len(full_audio))
            chunk = full_audio[pos:end]
            text = engine.transcribe(chunk, sample_rate)
            if text:
                time_offset = pos / sample_rate
                segments.append((time_offset, text))
            pos = end - overlap if end < len(full_audio) else end

        progress.update(task3, description="Transcription complete", completed=True)

    # Write output
    duration_s = len(full_audio) / sample_rate
    duration_m = int(duration_s // 60)
    duration_h = int(duration_m // 60)
    duration_m = duration_m % 60

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Session: {session_start.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Duration: {duration_h}h {duration_m}m\n")
        f.write(f"Model: {config.postprocess.model}\n")
        f.write("\n---\n\n")

        for time_offset, text in segments:
            minutes = int(time_offset // 60)
            seconds = int(time_offset % 60)
            session_time = session_start.replace(
                minute=session_start.minute + minutes,
                second=seconds,
            )
            ts = f"{int(time_offset // 3600):02d}:{minutes % 60:02d}:{seconds:02d}"
            f.write(f"[{ts}] {text}\n\n")

    console.print(f"\n[bold green]Transcript saved to:[/bold green] {output_path}")
    return output_path
