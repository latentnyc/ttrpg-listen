"""Speaker diarization using pyannote-audio (optional)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def diarize_wav(wav_path: Path, min_speakers: int = 2, max_speakers: int = 8) -> list[dict]:
    """Run speaker diarization on a WAV file.

    Returns a list of segments: [{"start": float, "end": float, "speaker": str}, ...]
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("[warning] pyannote-audio not installed. Skipping diarization.")
        print("  Install with: pip install pyannote-audio")
        return []

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    diarization = pipeline(
        str(wav_path),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    return segments


def align_transcript_with_speakers(
    transcript_segments: list[tuple[float, str]],
    speaker_segments: list[dict],
) -> list[tuple[float, str, str]]:
    """Align transcription timestamps with speaker diarization.

    Returns: [(time_offset, text, speaker_label), ...]
    """
    if not speaker_segments:
        return [(t, text, "Unknown") for t, text in transcript_segments]

    result = []
    for time_offset, text in transcript_segments:
        # Find the speaker active at this time
        best_speaker = "Unknown"
        best_overlap = 0.0

        for seg in speaker_segments:
            # Check overlap between transcript time and speaker segment
            overlap_start = max(time_offset, seg["start"])
            overlap_end = min(time_offset + 10.0, seg["end"])  # assume ~10s per segment
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]

        result.append((time_offset, text, best_speaker))

    return result
