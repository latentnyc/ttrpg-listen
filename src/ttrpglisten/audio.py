"""Audio capture: WASAPI loopback (pyaudiowpatch) + mic (sounddevice).

Uses pyaudiowpatch for system audio loopback (the only reliable way on
Windows to capture from Bluetooth/USB/HDMI outputs). Uses sounddevice
for microphone input. These use separate PortAudio instances but
coexist fine when started in sequence.
"""

from __future__ import annotations

import threading
import wave
from math import gcd
from pathlib import Path
from queue import Queue

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from .config import AudioConfig

_TARGET_RMS = 0.08


def _normalize_and_mix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalize two audio arrays to similar RMS levels, then mix."""
    def _scale(x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2))
        if rms < 1e-6:
            return x
        return x * (_TARGET_RMS / rms)

    mixed = _scale(a) + _scale(b)
    peak = np.abs(mixed).max()
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)
    return mixed


def list_devices() -> list[dict]:
    """Return all sounddevice audio devices."""
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


def _find_wasapi_loopback() -> dict | None:
    """Find the WASAPI loopback device for the default output via pyaudiowpatch."""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return None

    p = pyaudio.PyAudio()
    try:
        wasapi_info = None
        for i in range(p.get_host_api_count()):
            info = p.get_host_api_info_by_index(i)
            if "WASAPI" in info["name"]:
                wasapi_info = info
                break
        if wasapi_info is None:
            return None

        default_output = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        output_name = default_output["name"]

        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice") and output_name in dev["name"]:
                return {
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["maxInputChannels"],
                    "rate": int(dev["defaultSampleRate"]),
                }
        return None
    finally:
        p.terminate()


def _find_default_mic() -> int | None:
    """Find the default microphone via sounddevice."""
    try:
        default = sd.query_devices(kind="input")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["name"] == default["name"]:
                return i
    except sd.PortAudioError:
        pass
    return None


class _LoopbackThread:
    """Captures system audio via pyaudiowpatch WASAPI loopback on a dedicated thread.

    Uses blocking reads with small chunks to avoid long hangs when no audio
    is playing. The thread is a daemon so it won't prevent process exit.
    """

    def __init__(self, loopback_info: dict, on_audio, target_rate: int):
        self._info = loopback_info
        self._on_audio = on_audio
        self._target_rate = target_rate
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        native_rate = loopback_info["rate"]
        self._need_resample = native_rate != target_rate
        if self._need_resample:
            g = gcd(native_rate, target_rate)
            self._up = target_rate // g
            self._down = native_rate // g

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=self._info["channels"],
                rate=self._info["rate"],
                input=True,
                input_device_index=self._info["index"],
                frames_per_buffer=512,
            )

            channels = self._info["channels"]
            while not self._stop_event.is_set():
                try:
                    data = stream.read(512, exception_on_overflow=False)
                except OSError:
                    if self._stop_event.is_set():
                        break
                    continue
                audio = np.frombuffer(data, dtype=np.float32)
                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)
                if self._need_resample:
                    audio = resample_poly(audio, self._up, self._down).astype(np.float32)
                self._on_audio(audio)

            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        finally:
            pa.terminate()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            # Don't wait too long -- blocking read may be stuck if no audio playing
            self._thread.join(timeout=2)
            self._thread = None


class DualAudioCapture:
    """Captures from WASAPI loopback (system audio) + sounddevice (mic)."""

    def __init__(self, audio_queue: Queue, config: AudioConfig, wav_path: Path | None = None):
        self.audio_queue = audio_queue
        self.config = config
        self.sample_rate = config.sample_rate
        self.wav_path = wav_path
        self._loopback_thread: _LoopbackThread | None = None
        self._mic_stream: sd.InputStream | None = None
        self._wav_file: wave.Wave_write | None = None
        self._wav_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Resolved in start()
        self.loopback_info: dict | None = None
        self.mic_id: int | None = None

        # Buffers for mixing
        self._loopback_buffer = np.zeros(0, dtype=np.float32)
        self._mic_buffer = np.zeros(0, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._chunk_samples = int(self.sample_rate * 0.1)

    @property
    def loopback_name(self) -> str | None:
        return self.loopback_info["name"] if self.loopback_info else None

    @property
    def mic_name(self) -> str | None:
        if self.mic_id is not None:
            return sd.query_devices(self.mic_id)["name"]
        return None

    def _open_wav(self):
        if self.wav_path:
            self.wav_path.parent.mkdir(parents=True, exist_ok=True)
            self._wav_file = wave.open(str(self.wav_path), "wb")
            self._wav_file.setnchannels(2)
            self._wav_file.setsampwidth(2)
            self._wav_file.setframerate(self.sample_rate)

    def _write_wav_stereo(self, loopback: np.ndarray, mic: np.ndarray):
        if self._wav_file is None:
            return
        left = np.clip(loopback * 32767, -32768, 32767).astype(np.int16)
        right = np.clip(mic * 32767, -32768, 32767).astype(np.int16)
        stereo = np.empty(len(left) * 2, dtype=np.int16)
        stereo[0::2] = left
        stereo[1::2] = right
        with self._wav_lock:
            self._wav_file.writeframes(stereo.tobytes())

    def _push_audio(self, source: str, audio: np.ndarray):
        with self._buffer_lock:
            if source == "loopback":
                self._loopback_buffer = np.concatenate([self._loopback_buffer, audio])
            else:
                self._mic_buffer = np.concatenate([self._mic_buffer, audio])

            min_len = self._chunk_samples
            lb_len = len(self._loopback_buffer)
            mic_len = len(self._mic_buffer)

            if self._has_both and lb_len >= min_len and mic_len >= min_len:
                n = min(lb_len, mic_len, min_len)
                lb_chunk = self._loopback_buffer[:n]
                mic_chunk = self._mic_buffer[:n]
                self._loopback_buffer = self._loopback_buffer[n:]
                self._mic_buffer = self._mic_buffer[n:]
                self._write_wav_stereo(lb_chunk, mic_chunk)
                mixed = _normalize_and_mix(lb_chunk, mic_chunk)
                self.audio_queue.put(mixed)

            elif not self._has_both:
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

    @property
    def _has_both(self) -> bool:
        return self._loopback_thread is not None and self._mic_stream is not None

    def start(self):
        """Start capturing audio."""
        self._open_wav()

        # 1) Start mic first (sounddevice)
        mic_id = self.config.mic_device
        if mic_id is None:
            mic_id = _find_default_mic()
        self.mic_id = mic_id

        if self.mic_id is not None:
            try:
                native_rate = int(sd.query_devices(self.mic_id)["default_samplerate"])
                need_resample = native_rate != self.sample_rate
                if need_resample:
                    g = gcd(native_rate, self.sample_rate)
                    up, down = self.sample_rate // g, native_rate // g

                def mic_cb(indata, frames, time_info, status):
                    if self._stop_event.is_set():
                        return
                    audio = indata[:, 0].copy() if indata.shape[1] > 1 else indata.flatten().copy()
                    if need_resample:
                        audio = resample_poly(audio, up, down).astype(np.float32)
                    self._push_audio("mic", audio)

                self._mic_stream = sd.InputStream(
                    device=self.mic_id,
                    samplerate=native_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=int(native_rate * 0.03),
                    callback=mic_cb,
                )
                self._mic_stream.start()
            except sd.PortAudioError as e:
                print(f"[warning] Could not open mic device {self.mic_id}: {e}")
                self.mic_id = None

        # 2) Start loopback (pyaudiowpatch, separate PortAudio instance in thread)
        if self.config.loopback_device is None:
            self.loopback_info = _find_wasapi_loopback()
        # TODO: support user-specified loopback device index

        if self.loopback_info is not None:
            try:
                self._loopback_thread = _LoopbackThread(
                    self.loopback_info,
                    lambda audio: self._push_audio("loopback", audio),
                    self.sample_rate,
                )
                self._loopback_thread.start()
            except Exception as e:
                print(f"[warning] Could not start loopback: {e}")
                self._loopback_thread = None

        if self._loopback_thread is None and self._mic_stream is None:
            raise RuntimeError(
                "No audio devices available. Use --list-devices to see options."
            )

    def stop(self):
        self._stop_event.set()
        if self._mic_stream:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None
        if self._loopback_thread:
            self._loopback_thread.stop()
            self._loopback_thread = None
        with self._buffer_lock:
            self._loopback_buffer = np.zeros(0, dtype=np.float32)
            self._mic_buffer = np.zeros(0, dtype=np.float32)
        with self._wav_lock:
            if self._wav_file:
                self._wav_file.close()
                self._wav_file = None
