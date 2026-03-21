"""Streaming pipeline: connects audio capture, VAD, transcription, and display."""

from __future__ import annotations

import threading
from queue import Empty, Queue

from .transcribe import TranscriptionEngine


class StreamingPipeline:
    """Orchestrates the streaming transcription pipeline.

    Audio → VAD → Transcription → Display

    The audio capture and VAD are managed externally. This class handles
    the transcription worker that reads speech chunks and produces text.
    """

    def __init__(
        self,
        engine: TranscriptionEngine,
        speech_queue: Queue,
        text_queue: Queue,
        sample_rate: int = 16000,
    ):
        self.engine = engine
        self.speech_queue = speech_queue
        self.text_queue = text_queue
        self.sample_rate = sample_rate
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _transcribe_loop(self):
        """Worker loop: take speech chunks, transcribe, output text."""
        while not self._stop_event.is_set():
            try:
                audio_chunk = self.speech_queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                text = self.engine.transcribe(audio_chunk, self.sample_rate)
                if text:
                    self.text_queue.put(text)
            except Exception as e:
                self.text_queue.put(f"[transcription error: {e}]")

    def start(self):
        """Start the transcription worker thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the transcription worker."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
