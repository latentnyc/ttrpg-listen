"""SharedAudioBuffer and crash-safe WAV writer."""

from __future__ import annotations

import os
import struct
import threading
import time
from pathlib import Path

import numpy as np

_TARGET_RMS = 0.08


_MIC_NOISE_FLOOR = 0.008  # Default RMS below which mic is considered ambient


def smart_mix(
    loopback: np.ndarray,
    mic: np.ndarray,
    noise_floor: float = _MIC_NOISE_FLOOR,
) -> np.ndarray:
    """Mix loopback and mic audio for Whisper transcription.

    Mic audio is already gain-boosted at capture time. We mix it in when the
    mic RMS is above `noise_floor`, otherwise we use loopback only to keep
    Whisper input clean. Callers should pass `AppConfig.mic_sensitivity`
    so the user's configured value is honored.
    """
    mic_rms = float(np.sqrt(np.mean(mic ** 2)))

    if mic_rms < noise_floor:
        return loopback.copy()

    # Mix at natural levels - mic is already gain-boosted
    mixed = loopback + mic
    peak = np.abs(mixed).max()
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)
    return mixed


_CONSOLIDATE_INTERVAL = 1000  # Merge chunk lists into single arrays every N appends


class SharedAudioBuffer:
    """Thread-safe growing audio buffer for Whisper/diarization to read from.

    AudioCaptureWorker appends chunks. TranscriptionWorker and DiarizationWorker
    read ranges without consuming them.

    Memory: 4 hours at 16kHz stereo float32 ≈ 1.8 GB. Acceptable for a
    desktop session. The WAV file on disk serves as the persistent backup.
    """

    def __init__(self, sample_rate: int = 16000):
        self._lock = threading.Lock()
        self._loopback: np.ndarray = np.array([], dtype=np.float32)
        self._mic: np.ndarray = np.array([], dtype=np.float32)
        self._loopback_pending: list[np.ndarray] = []
        self._mic_pending: list[np.ndarray] = []
        self._total_samples: int = 0
        self._append_count: int = 0
        self.sample_rate = sample_rate

    def append(self, loopback: np.ndarray, mic: np.ndarray):
        """Append synchronized audio chunks."""
        n = min(len(loopback), len(mic))
        with self._lock:
            self._loopback_pending.append(loopback[:n].copy())
            self._mic_pending.append(mic[:n].copy())
            self._total_samples += n
            self._append_count += 1

            # Periodically consolidate pending chunks into the main arrays
            # to avoid O(n) concat on every get_channels call
            if self._append_count >= _CONSOLIDATE_INTERVAL:
                self._consolidate()

    def _consolidate(self):
        """Merge pending chunks into the main arrays. Must hold _lock."""
        if self._loopback_pending:
            parts = [self._loopback] if len(self._loopback) > 0 else []
            parts.extend(self._loopback_pending)
            self._loopback = np.concatenate(parts)
            self._loopback_pending.clear()

            parts = [self._mic] if len(self._mic) > 0 else []
            parts.extend(self._mic_pending)
            self._mic = np.concatenate(parts)
            self._mic_pending.clear()

            self._append_count = 0

    def append_single(self, audio: np.ndarray, source: str):
        """Append audio from a single source, zero-padding the other channel."""
        zeros = np.zeros_like(audio)
        if source == "loopback":
            self.append(audio, zeros)
        else:
            self.append(zeros, audio)

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total_samples

    @property
    def duration_seconds(self) -> float:
        return self.total_samples / self.sample_rate

    def get_mixed(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Get mixed (loopback + mic) audio for a sample range."""
        lb, mic = self.get_channels(start_sample, end_sample)
        return smart_mix(lb, mic)

    def get_channels(
        self, start_sample: int, end_sample: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get separate loopback and mic channels for a sample range."""
        with self._lock:
            # Consolidate pending chunks first
            if self._loopback_pending:
                self._consolidate()

            lb_all = self._loopback
            mic_all = self._mic

        start = max(0, start_sample)
        end = min(len(lb_all), end_sample)
        return lb_all[start:end].copy(), mic_all[start:end].copy()


class CrashSafeWavWriter:
    """Writes stereo WAV with periodic flushing and header updates for crash safety.

    - Data flushed every 5 seconds via flush() + fsync()
    - WAV header rewritten every 30 seconds so file is always valid
    - Placeholder header on open so file is playable even if never finalized
    """

    def __init__(self, path: Path, sample_rate: int = 16000, channels: int = 2):
        self._path = path
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = 2  # 16-bit
        self._lock = threading.Lock()
        self._file = None
        self._data_size = 0
        self._last_flush = 0.0
        self._last_header_update = 0.0

    def open(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "wb")
        self._data_size = 0
        # Write placeholder header with large size so it's playable even if we crash.
        # Use 0xFFFFFFFF - 36 so the RIFF size (36 + data_size) fits in uint32.
        self._write_header(0xFFFFFFFF - 36)
        self._last_flush = time.monotonic()
        self._last_header_update = time.monotonic()

    def write_stereo(self, loopback: np.ndarray, mic: np.ndarray):
        """Write interleaved stereo frames."""
        if self._file is None:
            return

        left = np.clip(loopback * 32767, -32768, 32767).astype(np.int16)
        right = np.clip(mic * 32767, -32768, 32767).astype(np.int16)
        stereo = np.empty(len(left) * 2, dtype=np.int16)
        stereo[0::2] = left
        stereo[1::2] = right
        data = stereo.tobytes()

        now = time.monotonic()

        try:
            with self._lock:
                self._file.write(data)
                self._data_size += len(data)

                if now - self._last_flush >= 5.0:
                    self._file.flush()
                    os.fsync(self._file.fileno())
                    self._last_flush = now

                if now - self._last_header_update >= 30.0:
                    self._do_update_header()
                    self._last_header_update = now
        except OSError:
            # Disk full or I/O error - close the file to preserve what we have
            with self._lock:
                if self._file:
                    try:
                        self._do_update_header()
                        self._file.close()
                    except Exception:
                        pass
                    self._file = None

    def _write_header(self, data_size: int):
        """Write a complete WAV header at the beginning of the file."""
        byte_rate = self._sample_rate * self._channels * self._sample_width
        block_align = self._channels * self._sample_width
        riff_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", riff_size, b"WAVE",
            b"fmt ", 16,
            1,  # PCM
            self._channels,
            self._sample_rate,
            byte_rate,
            block_align,
            self._sample_width * 8,
            b"data", data_size,
        )
        self._file.write(header)

    def _do_update_header(self):
        """Rewrite the header with current data size. Must hold _lock."""
        if self._file is None:
            return
        pos = self._file.tell()
        riff_size = 36 + self._data_size
        self._file.seek(4)
        self._file.write(struct.pack("<I", riff_size))
        self._file.seek(40)
        self._file.write(struct.pack("<I", self._data_size))
        self._file.seek(pos)
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self):
        with self._lock:
            if self._file is None:
                return
            # Final header update with actual data size
            pos = self._file.tell()
            riff_size = 36 + self._data_size
            self._file.seek(4)
            self._file.write(struct.pack("<I", riff_size))
            self._file.seek(40)
            self._file.write(struct.pack("<I", self._data_size))
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
