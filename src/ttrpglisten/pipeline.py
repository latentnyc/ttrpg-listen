"""Streaming pipeline: transcribes audio on a fixed interval for near-real-time subtitles.

Instead of waiting for speech pauses (VAD-gated), transcribes every ~2 seconds
of audio. VAD is only used to skip silent windows (prevent hallucinations).
"""

from __future__ import annotations

import threading
from queue import Empty, Queue

import numpy as np
import torch
from silero_vad import load_silero_vad

from .transcribe import TranscriptionEngine


class StreamingPipeline:
    """Transcribes audio on a fixed interval for near-real-time display.

    Reads 100ms audio chunks from audio_queue, accumulates them, and
    transcribes every `interval_s` seconds. Skips transcription if
    VAD detects no speech in the window.
    """

    def __init__(
        self,
        engine: TranscriptionEngine,
        audio_queue: Queue,
        text_queue: Queue,
        sample_rate: int = 16000,
        interval_s: float = 2.0,
        vad_threshold: float = 0.3,
    ):
        self.engine = engine
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self.sample_rate = sample_rate
        self.interval_samples = int(interval_s * sample_rate)
        self.vad_threshold = vad_threshold
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _has_speech(self, audio: np.ndarray, vad_model) -> bool:
        """Check if audio contains speech using Silero VAD."""
        # Process in 512-sample frames, return True if any frame has speech
        frame_size = 512
        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i : i + frame_size]
            tensor = torch.from_numpy(frame).float()
            prob = vad_model(tensor, self.sample_rate).item()
            if prob >= self.vad_threshold:
                return True
        return False

    def _process_loop(self):
        """Main loop: accumulate audio, transcribe at fixed intervals."""
        vad_model = load_silero_vad()
        buffer = np.zeros(0, dtype=np.float32)

        while not self._stop_event.is_set():
            # Drain audio queue into buffer
            try:
                chunk = self.audio_queue.get(timeout=0.05)
                buffer = np.concatenate([buffer, chunk])
            except Empty:
                pass

            # Transcribe when we have enough audio
            if len(buffer) >= self.interval_samples:
                window = buffer[: self.interval_samples]
                buffer = buffer[self.interval_samples :]

                # Skip silent windows
                if not self._has_speech(window, vad_model):
                    vad_model.reset_states()
                    continue

                vad_model.reset_states()

                try:
                    text = self.engine.transcribe(window, self.sample_rate)
                    if text:
                        self.text_queue.put(text)
                except Exception as e:
                    self.text_queue.put(f"[error: {e}]")

        # Flush remaining audio
        if len(buffer) >= self.sample_rate:  # at least 1 second
            if self._has_speech(buffer, vad_model):
                try:
                    text = self.engine.transcribe(buffer, self.sample_rate)
                    if text:
                        self.text_queue.put(text)
                except Exception:
                    pass

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
