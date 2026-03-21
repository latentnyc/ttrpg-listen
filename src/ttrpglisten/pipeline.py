"""Streaming pipeline: growing-window Moonshine transcription.

Transcribes audio with an expanding window (1s, 2s, 3s... up to 10s),
re-transcribing each time with more context for better accuracy.
The display updates in place as accuracy improves, then moves on.
"""

from __future__ import annotations

import threading
from queue import Empty, Queue

import numpy as np

from .transcribe import TranscriptionEngine


class StreamingPipeline:
    """Growing-window transcription for near-real-time subtitles.

    Accumulates audio and re-transcribes at each 1-second tick with the
    full window so far. After max_window_s seconds, finalizes the text
    and starts a new window. Gives immediate results that improve as
    more context arrives.

    Sends (text, is_final) tuples to text_queue:
      - (text, False) = partial result, replace current display line
      - (text, True)  = final result, commit to transcript and start new line
    """

    def __init__(
        self,
        engine: TranscriptionEngine,
        audio_queue: Queue,
        text_queue: Queue,
        sample_rate: int = 16000,
        tick_s: float = 1.0,
        max_window_s: float = 10.0,
    ):
        self.engine = engine
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self.sample_rate = sample_rate
        self.tick_samples = int(tick_s * sample_rate)
        self.max_window_samples = int(max_window_s * sample_rate)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _process_loop(self):
        buffer = np.zeros(0, dtype=np.float32)
        window = np.zeros(0, dtype=np.float32)

        while not self._stop_event.is_set():
            # Drain audio queue into buffer
            try:
                while True:
                    chunk = self.audio_queue.get_nowait()
                    buffer = np.concatenate([buffer, chunk])
            except Empty:
                pass

            # Wait until we have a full tick of new audio
            if len(buffer) < self.tick_samples:
                self._stop_event.wait(0.02)
                continue

            # Take one tick of audio and add to window
            tick_audio = buffer[: self.tick_samples]
            buffer = buffer[self.tick_samples :]
            window = np.concatenate([window, tick_audio])

            # Skip if window is too quiet
            rms = np.sqrt(np.mean(window ** 2))
            if rms < 0.003:
                # If quiet and window is growing, reset to avoid stale audio
                if len(window) > self.tick_samples * 3:
                    window = np.zeros(0, dtype=np.float32)
                continue

            # Transcribe the full window so far
            try:
                text = self.engine.transcribe(window, self.sample_rate)
            except Exception as e:
                text = f"[error: {e}]"

            if text:
                # Check if window is full -> finalize
                if len(window) >= self.max_window_samples:
                    self.text_queue.put((text, True))
                    window = np.zeros(0, dtype=np.float32)
                else:
                    self.text_queue.put((text, False))

        # Flush remaining window
        if len(window) >= self.sample_rate // 2:
            rms = np.sqrt(np.mean(window ** 2))
            if rms >= 0.003:
                try:
                    text = self.engine.transcribe(window, self.sample_rate)
                    if text:
                        self.text_queue.put((text, True))
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
