"""Voice Activity Detection for streaming audio chunking."""

from __future__ import annotations

import threading
from queue import Empty, Queue

import numpy as np
import torch
from silero_vad import load_silero_vad


class VadChunker:
    """Uses Silero VAD to segment streaming audio into speech chunks.

    Consumes raw audio from audio_queue, detects speech boundaries,
    and puts complete speech segments onto speech_queue for transcription.
    """

    def __init__(
        self,
        audio_queue: Queue,
        speech_queue: Queue,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 600,
        speech_pad_ms: int = 200,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = 10.0,
    ):
        self.audio_queue = audio_queue
        self.speech_queue = speech_queue
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_samples = int(min_silence_duration_ms * sample_rate / 1000)
        self.speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)
        self.min_speech_samples = int(min_speech_duration_ms * sample_rate / 1000)
        self.max_speech_samples = int(max_speech_duration_s * sample_rate)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Silero VAD model
        self._vad_model = None

    def _load_vad(self):
        """Load Silero VAD model."""
        self._vad_model = load_silero_vad()

    def _get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """Get speech probability for a 32ms audio chunk (512 samples at 16kHz)."""
        tensor = torch.from_numpy(audio_chunk).float()
        return self._vad_model(tensor, self.sample_rate).item()

    def _process_loop(self):
        """Main processing loop: accumulate audio, detect speech, emit chunks."""
        self._load_vad()

        buffer = np.zeros(0, dtype=np.float32)
        speech_start = -1  # -1 means not in speech
        silence_counter = 0
        # Silero VAD requires exactly 512 samples at 16kHz (32ms)
        vad_frame_size = 512 if self.sample_rate == 16000 else 256

        while not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except Empty:
                continue

            buffer = np.concatenate([buffer, chunk])

            # Process in 32ms VAD frames (512 samples at 16kHz)
            while len(buffer) >= vad_frame_size:
                frame = buffer[:vad_frame_size]
                buffer = buffer[vad_frame_size:]

                prob = self._get_speech_prob(frame)

                if prob >= self.threshold:
                    if speech_start == -1:
                        # Speech just started
                        speech_start = max(0, len(buffer) - self.speech_pad_samples)
                        self._speech_buffer = np.zeros(0, dtype=np.float32)
                    silence_counter = 0
                    self._speech_buffer = np.concatenate([self._speech_buffer, frame])
                else:
                    if speech_start != -1:
                        silence_counter += vad_frame_size
                        self._speech_buffer = np.concatenate([self._speech_buffer, frame])

                        if silence_counter >= self.min_silence_samples:
                            # Speech ended - emit chunk
                            if len(self._speech_buffer) >= self.min_speech_samples:
                                self.speech_queue.put(self._speech_buffer.copy())
                            speech_start = -1
                            silence_counter = 0

                # Force-emit if speech too long (avoid waiting forever on monologues)
                if speech_start != -1 and len(self._speech_buffer) >= self.max_speech_samples:
                    self.speech_queue.put(self._speech_buffer.copy())
                    self._speech_buffer = np.zeros(0, dtype=np.float32)
                    speech_start = -1
                    silence_counter = 0
                    self._vad_model.reset_states()

        # Flush remaining speech
        if speech_start != -1 and hasattr(self, "_speech_buffer"):
            if len(self._speech_buffer) >= self.min_speech_samples:
                self.speech_queue.put(self._speech_buffer.copy())

    def start(self):
        """Start VAD processing in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop VAD processing."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
