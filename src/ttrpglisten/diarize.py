"""Speaker diarization using pyannote-audio (optional)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def diarize_wav(wav_path: Path, min_speakers: int = 2, max_speakers: int = 8) -> list[dict]:
    """Run speaker diarization on a WAV file.

    Requires pyannote-audio and a HuggingFace token (set HF_TOKEN env var).
    Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1

    Returns a list of segments: [{"start": float, "end": float, "speaker": str}, ...]
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("[warning] pyannote-audio not installed. Skipping diarization.")
        print("  Install with: pip install pyannote-audio")
        return []

    import os
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[warning] HF_TOKEN not set. Speaker diarization requires a HuggingFace token.")
        print("  1. Create token at: https://huggingface.co/settings/tokens")
        print("  2. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("  3. Set: export HF_TOKEN=your_token_here")
        return []

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception as e:
        print(f"[warning] Could not load diarization model: {e}")
        return []

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


_NAME_PATTERNS = [
    re.compile(r"\bmy name is (\w+)", re.IGNORECASE),
    re.compile(r"\bmy name'?s (\w+)", re.IGNORECASE),
    re.compile(r"\bi'?m (\w+)[,.]", re.IGNORECASE),
    re.compile(r"\bcall me (\w+)", re.IGNORECASE),
    re.compile(r"\bthis is (\w+) speaking", re.IGNORECASE),
    re.compile(r"\bhey,? i'?m (\w+)", re.IGNORECASE),
    re.compile(r"\bhi,? i'?m (\w+)", re.IGNORECASE),
]

# Words that match the patterns but aren't names
_NOT_NAMES = {
    "a", "the", "not", "so", "just", "here", "there", "back", "going",
    "gonna", "done", "fine", "good", "okay", "ok", "sorry", "sure",
    "like", "very", "really", "also", "still", "now", "then",
}


def infer_speaker_names(
    aligned: list[tuple[float, str, str]],
) -> dict[str, str]:
    """Scan transcript for self-introductions and map speaker labels to names.

    Returns a dict like {"SPEAKER_00": "Tim", "SPEAKER_01": "Sarah"}.
    """
    label_to_name: dict[str, str] = {}

    for _, text, speaker in aligned:
        if not speaker or speaker in label_to_name:
            continue
        for pattern in _NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip()
                if name.lower() not in _NOT_NAMES and len(name) > 1:
                    label_to_name[speaker] = name.capitalize()
                    break

    return label_to_name


def apply_speaker_names(
    aligned: list[tuple[float, str, str]],
    names: dict[str, str],
) -> list[tuple[float, str, str]]:
    """Replace speaker labels with inferred names."""
    return [
        (t, text, names.get(speaker, speaker))
        for t, text, speaker in aligned
    ]
