"""Post-session processing: high-quality re-transcription with optional diarization."""

from __future__ import annotations

import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config, resolve_device
from .diarize import align_transcript_with_speakers, apply_speaker_names, diarize_wav, infer_speaker_names
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
        engine = TranscriptionEngine(
            config.postprocess.model,
            device=device,
            language=config.postprocess.language,
        )
        engine.load()
        progress.update(task2, description="Model loaded", completed=True)

        # Whisper has a 30-second context window; Moonshine handles longer chunks
        task3 = progress.add_task("Transcribing full session...", total=None)
        chunk_seconds = 30 if engine.is_whisper else 60
        chunk_size = chunk_seconds * sample_rate
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

        # Speaker diarization (optional)
        speaker_segments = []
        if config.postprocess.diarization:
            task4 = progress.add_task("Running speaker diarization...", total=None)
            try:
                speaker_segments = diarize_wav(
                    wav_path,
                    min_speakers=config.postprocess.min_speakers,
                    max_speakers=config.postprocess.max_speakers,
                )
            except Exception as e:
                console.print(f"[yellow]Diarization failed, continuing without speaker labels:[/yellow] {e}")
                speaker_segments = []
            if speaker_segments:
                progress.update(task4, description=f"Diarization complete ({len(set(s['speaker'] for s in speaker_segments))} speakers)", completed=True)
            else:
                progress.update(task4, description="Diarization skipped", completed=True)

    # Align transcript with speakers and infer names
    if speaker_segments:
        aligned = align_transcript_with_speakers(segments, speaker_segments)
        names = infer_speaker_names(aligned)
        if names:
            aligned = apply_speaker_names(aligned, names)
    else:
        aligned = [(t, text, None) for t, text in segments]

    # Write output
    duration_s = len(full_audio) / sample_rate
    duration_m = int(duration_s // 60)
    duration_h = int(duration_m // 60)
    duration_m = duration_m % 60

    # Collect unique speakers
    speakers = sorted(set(s for _, _, s in aligned if s))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Session: {session_start.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Duration: {duration_h}h {duration_m}m\n")
        f.write(f"Model: {config.postprocess.model}\n")
        if speakers:
            f.write(f"Speakers: {', '.join(speakers)}\n")
        f.write("\n---\n\n")

        for time_offset, text, speaker in aligned:
            minutes = int(time_offset // 60)
            seconds = int(time_offset % 60)
            ts = f"{int(time_offset // 3600):02d}:{minutes % 60:02d}:{seconds:02d}"
            if speaker:
                f.write(f"[{ts}] {speaker}: {text}\n\n")
            else:
                f.write(f"[{ts}] {text}\n\n")

    console.print(f"\n[bold green]Transcript saved to:[/bold green] {output_path}")
    return output_path
