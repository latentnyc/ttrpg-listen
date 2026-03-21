"""Audio capture: dual-source (loopback + mic) with WAV recording."""

from __future__ import annotations

import struct
import threading
import wave
from pathlib import Path
from queue import Queue

import numpy as np
import sounddevice as sd

from .config import AudioConfig


def list_devices() -> list[dict]:
    """Return all audio devices with their indices and properties."""
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        result.append({
            "index": i,
            "name": dev["name"],
            "max_input_channels": dev["max_input_channels"],
            "max_output_channels": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
            "hostapi": sd.query_hostapis(dev["hostapi"])["name"],
        })
    return result


def find_loopback_device() -> int | None:
    """Auto-detect a WASAPI loopback device on Windows."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        hostapi = sd.query_hostapis(dev["hostapi"])["name"]
        if "WASAPI" in hostapi and dev["max_input_channels"] > 0:
            name_lower = dev["name"].lower()
            if "loopback" in name_lower:
                return i
    # Fallback: look for any stereo mix or similar
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            name_lower = dev["name"].lower()
            if "stereo mix" in name_lower or "what u hear" in name_lower:
                return i
    return None


def find_default_mic() -> int | None:
    """Find the default input (microphone) device."""
    try:
        default = sd.query_devices(kind="input")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["name"] == default["name"]:
                return i
    except sd.PortAudioError:
        pass
    return None


class DualAudioCapture:
    """Captures audio from both loopback and microphone, mixes them."""

    def __init__(
        self,
        audio_queue: Queue,
        config: AudioConfig,
        wav_path: Path | None = None,
    ):
        self.audio_queue = audio_queue
        self.config = config
        self.sample_rate = config.sample_rate
        self.wav_path = wav_path
        self._stop_event = threading.Event()
        self._streams: list[sd.InputStream] = []
        self._wav_file: wave.Wave_write | None = None
        self._wav_lock = threading.Lock()

        # Resolve devices
        self.loopback_id = config.loopback_device
        self.mic_id = config.mic_device

        if self.loopback_id is None:
            self.loopback_id = find_loopback_device()
        if self.mic_id is None:
            self.mic_id = find_default_mic()

        # Buffers for mixing (accumulate from both sources)
        self._loopback_buffer = np.zeros(0, dtype=np.float32)
        self._mic_buffer = np.zeros(0, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        # Chunk size for output (100ms)
        self._chunk_samples = int(self.sample_rate * 0.1)

    def _open_wav(self):
        if self.wav_path:
            self.wav_path.parent.mkdir(parents=True, exist_ok=True)
            self._wav_file = wave.open(str(self.wav_path), "wb")
            # Stereo: left=loopback, right=mic
            self._wav_file.setnchannels(2)
            self._wav_file.setsampwidth(2)  # 16-bit
            self._wav_file.setframerate(self.sample_rate)

    def _write_wav_stereo(self, loopback: np.ndarray, mic: np.ndarray):
        """Write interleaved stereo frames to WAV (left=loopback, right=mic)."""
        if self._wav_file is None:
            return
        # Convert float32 [-1,1] to int16
        left = np.clip(loopback * 32767, -32768, 32767).astype(np.int16)
        right = np.clip(mic * 32767, -32768, 32767).astype(np.int16)
        # Interleave
        stereo = np.empty(len(left) * 2, dtype=np.int16)
        stereo[0::2] = left
        stereo[1::2] = right
        with self._wav_lock:
            self._wav_file.writeframes(stereo.tobytes())

    def _make_callback(self, source: str):
        """Create a sounddevice callback for a given source."""
        def callback(indata, frames, time_info, status):
            if self._stop_event.is_set():
                return
            # Convert to mono float32
            audio = indata[:, 0].copy() if indata.shape[1] > 1 else indata.flatten().copy()

            with self._buffer_lock:
                if source == "loopback":
                    self._loopback_buffer = np.concatenate([self._loopback_buffer, audio])
                else:
                    self._mic_buffer = np.concatenate([self._mic_buffer, audio])

                # When we have enough from both sources (or one if the other isn't available)
                min_len = self._chunk_samples
                lb_len = len(self._loopback_buffer)
                mic_len = len(self._mic_buffer)

                if lb_len >= min_len and mic_len >= min_len:
                    n = min(lb_len, mic_len, min_len)
                    lb_chunk = self._loopback_buffer[:n]
                    mic_chunk = self._mic_buffer[:n]
                    self._loopback_buffer = self._loopback_buffer[n:]
                    self._mic_buffer = self._mic_buffer[n:]

                    # Mix for transcription (average)
                    mixed = (lb_chunk + mic_chunk) * 0.5
                    self.audio_queue.put(mixed)
                    self._write_wav_stereo(lb_chunk, mic_chunk)

                elif self._single_source and (lb_len >= min_len or mic_len >= min_len):
                    # Only one source active
                    if lb_len >= min_len:
                        chunk = self._loopback_buffer[:min_len]
                        self._loopback_buffer = self._loopback_buffer[min_len:]
                        silence = np.zeros(min_len, dtype=np.float32)
                        self.audio_queue.put(chunk)
                        self._write_wav_stereo(chunk, silence)
                    elif mic_len >= min_len:
                        chunk = self._mic_buffer[:min_len]
                        self._mic_buffer = self._mic_buffer[min_len:]
                        silence = np.zeros(min_len, dtype=np.float32)
                        self.audio_queue.put(chunk)
                        self._write_wav_stereo(silence, chunk)

        return callback

    @property
    def _single_source(self) -> bool:
        return self.loopback_id is None or self.mic_id is None

    def start(self):
        """Start capturing audio from available sources."""
        self._open_wav()

        if self.loopback_id is not None:
            try:
                stream = sd.InputStream(
                    device=self.loopback_id,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=int(self.sample_rate * 0.03),  # 30ms blocks
                    callback=self._make_callback("loopback"),
                )
                stream.start()
                self._streams.append(stream)
            except sd.PortAudioError as e:
                print(f"[warning] Could not open loopback device {self.loopback_id}: {e}")
                self.loopback_id = None

        if self.mic_id is not None:
            try:
                stream = sd.InputStream(
                    device=self.mic_id,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=int(self.sample_rate * 0.03),
                    callback=self._make_callback("mic"),
                )
                stream.start()
                self._streams.append(stream)
            except sd.PortAudioError as e:
                print(f"[warning] Could not open mic device {self.mic_id}: {e}")
                self.mic_id = None

        if not self._streams:
            raise RuntimeError(
                "No audio devices available. Use --list-devices to see options, "
                "then specify --loopback-device and/or --mic-device."
            )

    def stop(self):
        """Stop all audio streams and close WAV file."""
        self._stop_event.set()
        for stream in self._streams:
            stream.stop()
            stream.close()
        self._streams.clear()

        # Flush remaining buffers
        with self._buffer_lock:
            self._loopback_buffer = np.zeros(0, dtype=np.float32)
            self._mic_buffer = np.zeros(0, dtype=np.float32)

        with self._wav_lock:
            if self._wav_file:
                self._wav_file.close()
                self._wav_file = None
