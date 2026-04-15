"""Whisper transcription worker using whisperx for accurate word timestamps.

Pipeline per chunk:
1. whisperx.load_model().transcribe() - VAD + batched Whisper
2. whisperx.align() - wav2vec2 forced alignment for word-level timestamps
3. Emit segments with precise timing for speaker assignment
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from ..audio.recorder import SharedAudioBuffer
from ..models.selector import detect_vram_gb

CHUNK_INTERVAL_S = 30.0


def _set_low_priority():
    """Set the current thread to below-normal priority on Windows.

    This ensures Whisper/diarization don't starve game audio, Discord,
    or the GUI thread.
    """
    try:
        # Windows: THREAD_PRIORITY_BELOW_NORMAL = -1
        handle = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(handle, -1)
    except Exception:
        # Non-Windows or permission error - continue at normal priority
        pass


def _configure_cuda_for_background():
    """Configure CUDA to play nice with other GPU users (games, Discord).

    - yield mode: GPU yields to other processes between kernel launches
    - limits memory growth so games/Discord keep their VRAM
    """
    try:
        import torch
        if torch.cuda.is_available():
            # CUDA_DEVICE_SCHEDULE_YIELD - yields CPU thread while waiting
            # for GPU, reducing CPU contention
            os.environ.setdefault("CUDA_DEVICE_SCHEDULE", "YIELD")
            # Limit PyTorch CUDA memory to 70% to leave headroom
            torch.cuda.set_per_process_memory_fraction(0.7)
    except Exception:
        pass


class TranscriptionWorker(QObject):
    # Loopback segments (remote players) - speaker TBD from diarization
    segment_ready = Signal(str, float, float)
    # Mic segments (local player) - speaker is always "Microphone"
    mic_segment_ready = Signal(str, float, float)
    # Full aligned result for diarization worker to consume
    aligned_result_ready = Signal(object, float)
    status_message = Signal(str)
    model_loaded = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        shared_buffer: SharedAudioBuffer,
        game_prompt: str = "",
        language: str = "en",
    ):
        super().__init__()
        self._shared_buffer = shared_buffer
        self._game_prompt = game_prompt
        self._language = language
        self._stop_requested = False

        self._whisper_model = None
        self._align_model = None
        self._align_meta = None
        self._device = "cpu"

        self._last_processed_sample = 0

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        # Lower this thread's OS priority so Whisper doesn't starve
        # the game audio, Discord, or the GUI
        _set_low_priority()

        try:
            self._load_models()
            self.model_loaded.emit()
        except Exception as e:
            self.error_occurred.emit(f"Model load error: {e}")
            return

        sample_rate = self._shared_buffer.sample_rate
        min_new_samples = int(CHUNK_INTERVAL_S * sample_rate)

        while not self._stop_requested:
            total = self._shared_buffer.total_samples
            new_samples = total - self._last_processed_sample

            if new_samples < min_new_samples:
                QThread.msleep(1000)
                continue

            try:
                self._process_chunk(total, sample_rate)
            except Exception as e:
                self.error_occurred.emit(f"Transcription error: {e}")
                self._last_processed_sample = total

        # Process remaining audio on stop
        total = self._shared_buffer.total_samples
        if total > self._last_processed_sample + self._shared_buffer.sample_rate:
            try:
                self._process_chunk(total, self._shared_buffer.sample_rate)
            except Exception:
                pass

        self._unload_models()
        self.status_message.emit("Transcription stopped")

    def _load_models(self):
        import whisperx

        _configure_cuda_for_background()

        vram = detect_vram_gb()
        if vram >= 12:
            model_size = "large-v3"
        elif vram >= 6:
            model_size = "large-v3-turbo"
        elif vram >= 4:
            model_size = "medium"
        elif vram >= 2:
            model_size = "small"
        else:
            model_size = "tiny"

        device = "cuda" if vram > 0 else "cpu"
        compute = "float16" if device == "cuda" else "int8"

        self.status_message.emit(f"Loading whisperx {model_size} on {device} ({vram:.0f}GB)...")

        self._whisper_model = whisperx.load_model(
            model_size, device, compute_type=compute, language=self._language
        )

        self.status_message.emit("Loading alignment model...")
        self._align_model, self._align_meta = whisperx.load_align_model(
            language_code=self._language, device=device
        )
        self._device = device

        self.status_message.emit(f"Models loaded: whisper {model_size} + wav2vec2 alignment")

    def _unload_models(self):
        import torch

        del self._whisper_model
        del self._align_model
        self._whisper_model = None
        self._align_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _process_chunk(self, total_samples: int, sample_rate: int):
        """Process both channels separately.

        - Loopback (system audio): transcribe + align, send to diarization
          for speaker identification of remote players.
        - Mic: transcribe + align, attribute directly as 'Microphone'
          (no diarization needed - we know who's at the mic).
        """
        import whisperx

        chunk_start = self._last_processed_sample
        chunk_end = total_samples
        chunk_start_time = chunk_start / sample_rate

        loopback, mic = self._shared_buffer.get_channels(chunk_start, chunk_end)

        if len(loopback) < sample_rate:
            return

        duration = len(loopback) / sample_rate
        self.status_message.emit(
            f"Whisper: {chunk_start_time:.0f}s-{chunk_start_time + duration:.0f}s"
        )

        # --- Process loopback (remote players) ---
        lb_rms = float(np.sqrt(np.mean(loopback ** 2)))
        if lb_rms >= 0.003:
            lb_result = self._transcribe_and_align(loopback, sample_rate)
            if lb_result and lb_result.get("segments"):
                n_segs = len(lb_result["segments"])
                self.status_message.emit(f"  -> loopback: {n_segs} segments")

                # Emit for immediate display (speaker TBD from diarization)
                for seg in lb_result["segments"]:
                    text = seg.get("text", "").strip()
                    start = seg.get("start", 0)
                    end = seg.get("end", start + 1)
                    if text:
                        self.segment_ready.emit(
                            text, chunk_start_time + start, chunk_start_time + end
                        )

                # Send to diarization for speaker assignment
                self.aligned_result_ready.emit(lb_result, chunk_start_time)

        # --- Process mic (local player) ---
        mic_rms = float(np.sqrt(np.mean(mic ** 2)))
        if mic_rms >= 0.008:  # noise floor after gain boost
            mic_result = self._transcribe_and_align(mic, sample_rate)
            if mic_result and mic_result.get("segments"):
                n_segs = len(mic_result["segments"])
                self.status_message.emit(f"  -> mic: {n_segs} segments")

                # Emit mic segments directly as "Microphone" speaker
                for seg in mic_result["segments"]:
                    text = seg.get("text", "").strip()
                    start = seg.get("start", 0)
                    end = seg.get("end", start + 1)
                    if text:
                        self.mic_segment_ready.emit(
                            text, chunk_start_time + start, chunk_start_time + end
                        )

        self._last_processed_sample = total_samples

    def _transcribe_and_align(self, audio: np.ndarray, sample_rate: int) -> dict | None:
        """Run whisperx transcription + forced alignment on an audio array."""
        import whisperx

        result = self._whisper_model.transcribe(
            audio, batch_size=16, initial_prompt=self._game_prompt or None
        )
        if not result.get("segments"):
            return None

        result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_meta,
            audio,
            self._device,
        )
        return result
