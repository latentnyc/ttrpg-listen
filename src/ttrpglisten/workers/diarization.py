"""Speaker diarization with overlapping 60s windows for boundary correction.

Processes 60s windows every 30s. Each chunk boundary gets diarized twice:
once at the edge of window N, and once in the middle of window N+1.
The second pass corrects any speaker misattributions from the first.

Pipeline per window:
1. pyannote speaker diarization on 60s audio
2. whisperx.assign_word_speakers to merge with aligned transcription
3. Cross-check overlapping regions and correct speaker labels
"""

from __future__ import annotations

import os
import threading
from collections import deque

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QThread, Signal

from ..audio.recorder import SharedAudioBuffer
from ..models.selector import detect_vram_gb, select_compute_device


class DiarizationWorker(QObject):
    # text, speaker, start, end - fully attributed segments
    attributed_segment = Signal(str, str, float, float)
    # old_speaker -> new_speaker corrections for segments already displayed
    speaker_correction = Signal(float, float, str)  # start, end, corrected_speaker
    status_message = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        shared_buffer: SharedAudioBuffer,
        min_speakers: int = 2,
        max_speakers: int = 8,
    ):
        super().__init__()
        self._shared_buffer = shared_buffer
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._stop_requested = False

        self._pipeline = None
        self._device = None

        # Thread-safe queue: (aligned_result, chunk_start_time)
        self._pending_chunks: deque[tuple[dict, float]] = deque()
        self._queue_lock = threading.Lock()

        # History of diarization results for cross-checking overlaps.
        # Maps absolute_time -> speaker for each word from previous windows.
        self._speaker_history: dict[float, str] = {}


    def request_stop(self):
        self._stop_requested = True

    def enqueue_chunk(self, aligned_result: dict, chunk_start_time: float):
        """Thread-safe enqueue from main thread."""
        with self._queue_lock:
            self._pending_chunks.append((aligned_result, chunk_start_time))

    def run(self):
        # Lower OS priority - diarization is background work
        try:
            import ctypes
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, -1)
        except Exception:
            pass

        try:
            self._load_pipeline()
        except Exception as e:
            self.error_occurred.emit(f"Diarization load error: {e}")
            return

        while not self._stop_requested:
            item = None
            with self._queue_lock:
                if self._pending_chunks:
                    item = self._pending_chunks.popleft()

            if item is None:
                QThread.msleep(1000)
                continue

            aligned_result, chunk_start_time = item

            try:
                self._process_with_overlap(aligned_result, chunk_start_time)
            except Exception as e:
                self.error_occurred.emit(f"Diarization error: {e}")

        self.status_message.emit("Diarization stopped")

    def _load_pipeline(self):
        import io
        import sys
        import torch

        real_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            from pyannote.audio import Pipeline
        finally:
            sys.stderr = real_stderr

        self._device = select_compute_device()
        hf_token = os.environ.get("HF_TOKEN")

        self.status_message.emit("Loading diarization pipeline...")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=hf_token
        )
        if self._device == "cuda":
            self._pipeline = self._pipeline.to(torch.device("cuda"))
        self.status_message.emit("Diarization pipeline loaded")

    def _process_with_overlap(self, aligned_result: dict, chunk_start_time: float):
        """Diarize a 60s window centered around the new chunk.

        If we have enough audio history, we extend 30s back before chunk_start_time
        to create a 60s window. This means the boundary region (where the previous
        chunk ended) is now in the middle of the window where diarization is best.
        """
        sr = self._shared_buffer.sample_rate
        total_available = self._shared_buffer.total_samples / sr

        segments = aligned_result.get("segments", [])
        if not segments:
            return

        chunk_end_time = chunk_start_time + max(s.get("end", 0) for s in segments)

        # Build a 60s window: extend 30s before chunk_start if possible
        window_start = max(0.0, chunk_start_time - 30.0)
        window_end = min(total_available, chunk_end_time)
        window_duration = window_end - window_start

        if window_duration < 5.0:
            return

        # Get audio for the full window
        start_sample = int(window_start * sr)
        end_sample = int(window_end * sr)
        lb, mic = self._shared_buffer.get_channels(start_sample, end_sample)
        audio = (lb * 0.5 + mic * 0.5).astype(np.float32)

        if len(audio) < sr * 5:
            return

        self.status_message.emit(
            f"Diarizing {window_start:.0f}s-{window_end:.0f}s ({window_duration:.0f}s window)"
        )

        # Run pyannote on the full 60s window
        diar_df = self._run_pyannote(audio, sr)
        if diar_df is None or diar_df.empty:
            return

        n_speakers = diar_df["speaker"].nunique()

        # Offset diarization timestamps to absolute session time
        diar_df["start"] = diar_df["start"] + window_start
        diar_df["end"] = diar_df["end"] + window_start

        # Build aligned result covering the full window for whisperx assignment.
        # We need the transcription segments offset to window-local time for alignment,
        # but the diarization is now in absolute time.
        window_aligned = self._build_window_aligned(
            aligned_result, chunk_start_time, window_start
        )

        if not window_aligned.get("segments"):
            return

        # Assign speakers to words
        import whisperx
        # whisperx expects diar_df times relative to audio start (window-local)
        diar_local = diar_df.copy()
        diar_local["start"] = diar_local["start"] - window_start
        diar_local["end"] = diar_local["end"] - window_start

        attributed = whisperx.assign_word_speakers(diar_local, window_aligned)

        self.status_message.emit(f"  -> {n_speakers} speakers identified")

        # Process attributed segments: emit new ones, correct old ones
        for seg in attributed.get("segments", []):
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "Speaker")
            start = seg.get("start", 0) + window_start  # to absolute time
            end = seg.get("end", start + 1) + window_start

            if not text:
                continue

            # Check if this segment was in the overlap region (already displayed)
            prev_speaker = self._speaker_history.get(round(start, 1))

            if prev_speaker is not None:
                # This segment was already shown. If speaker changed, emit correction.
                if prev_speaker != speaker:
                    self.speaker_correction.emit(start, end, speaker)
            else:
                # New segment from the current chunk
                self.attributed_segment.emit(text, speaker, start, end)

            # Record in history for future cross-checks
            self._speaker_history[round(start, 1)] = speaker

    def _run_pyannote(self, audio: np.ndarray, sr: int):
        """Run pyannote diarization, return DataFrame."""
        import torch

        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        diar_output = self._pipeline(
            {"waveform": waveform, "sample_rate": sr},
            min_speakers=self._min_speakers,
            max_speakers=self._max_speakers,
        )

        annotation = getattr(diar_output, "speaker_diarization", diar_output)

        rows = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            rows.append({"start": turn.start, "end": turn.end, "speaker": speaker})

        if not rows:
            return None

        return pd.DataFrame(rows)

    def _build_window_aligned(self, aligned_result, chunk_start_time, window_start):
        """Build an aligned result dict with times relative to window start.

        The aligned_result from the transcription worker has times relative to
        chunk_start_time. We need to adjust them so they're relative to
        window_start for whisperx.assign_word_speakers to work correctly.
        """
        import copy
        offset = chunk_start_time - window_start
        result = copy.deepcopy(aligned_result)

        for seg in result.get("segments", []):
            seg["start"] = seg.get("start", 0) + offset
            seg["end"] = seg.get("end", 0) + offset
            for word in seg.get("words", []):
                if "start" in word:
                    word["start"] = word["start"] + offset
                if "end" in word:
                    word["end"] = word["end"] + offset

        return result
