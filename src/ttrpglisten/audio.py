"""Audio capture: dual-source (WASAPI loopback + mic) with WAV recording."""

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


def find_wasapi_loopback() -> dict | None:
    """Find the WASAPI loopback device for the default output using pyaudiowpatch.

    Returns dict with 'index', 'name', 'channels', 'rate' or None.
    """
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return None

    p = pyaudio.PyAudio()
    try:
        # Find WASAPI host API
        wasapi_info = None
        for i in range(p.get_host_api_count()):
            info = p.get_host_api_info_by_index(i)
            if "WASAPI" in info["name"]:
                wasapi_info = info
                break
        if wasapi_info is None:
            return None

        # Get default output device name
        default_output = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        output_name = default_output["name"]

        # Find matching loopback device
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


def find_default_mic() -> int | None:
    """Find the default input (microphone) device via sounddevice."""
    try:
        default = sd.query_devices(kind="input")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["name"] == default["name"]:
                return i
    except sd.PortAudioError:
        pass
    return None


class LoopbackStream:
    """WASAPI loopback capture using pyaudiowpatch."""

    def __init__(self, loopback_info: dict, on_audio):
        """
        loopback_info: dict from find_wasapi_loopback()
        on_audio: callable(np.ndarray) - receives mono float32 at native rate
        """
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        self._info = loopback_info
        self._on_audio = on_audio
        self._stream = None

    def start(self):
        import pyaudiowpatch as pyaudio

        def callback(in_data, frame_count, time_info, status):
            audio = np.frombuffer(in_data, dtype=np.float32)
            # Convert to mono if multi-channel
            channels = self._info["channels"]
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            self._on_audio(audio)
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._info["channels"],
            rate=self._info["rate"],
            input=True,
            input_device_index=self._info["index"],
            frames_per_buffer=1024,
            stream_callback=callback,
        )
        self._stream.start_stream()

    def stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None


class DualAudioCapture:
    """Captures audio from WASAPI loopback and microphone, mixes them."""

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
        self._mic_stream: sd.InputStream | None = None
        self._loopback_stream: LoopbackStream | None = None
        self._wav_file: wave.Wave_write | None = None
        self._wav_lock = threading.Lock()

        # Resolve devices
        self.loopback_info: dict | None = None
        self.mic_id = config.mic_device

        if config.loopback_device is None:
            self.loopback_info = find_wasapi_loopback()
        # If user specified a sounddevice ID, we'll use it as a plain input device
        self._loopback_sd_id = config.loopback_device if isinstance(config.loopback_device, int) else None

        if self.mic_id is None:
            self.mic_id = find_default_mic()

        # Buffers for mixing (accumulate from both sources)
        self._loopback_buffer = np.zeros(0, dtype=np.float32)
        self._mic_buffer = np.zeros(0, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        # Chunk size for output (100ms at target sample rate)
        self._chunk_samples = int(self.sample_rate * 0.1)

        # Resampling state for loopback
        self._lb_resample_up = 1
        self._lb_resample_down = 1
        self._lb_need_resample = False

        # Resampling state for mic
        self._mic_resample_up = 1
        self._mic_resample_down = 1
        self._mic_need_resample = False

    def _open_wav(self):
        if self.wav_path:
            self.wav_path.parent.mkdir(parents=True, exist_ok=True)
            self._wav_file = wave.open(str(self.wav_path), "wb")
            self._wav_file.setnchannels(2)  # stereo: left=loopback, right=mic
            self._wav_file.setsampwidth(2)  # 16-bit
            self._wav_file.setframerate(self.sample_rate)

    def _write_wav_stereo(self, loopback: np.ndarray, mic: np.ndarray):
        """Write interleaved stereo frames to WAV (left=loopback, right=mic)."""
        if self._wav_file is None:
            return
        left = np.clip(loopback * 32767, -32768, 32767).astype(np.int16)
        right = np.clip(mic * 32767, -32768, 32767).astype(np.int16)
        stereo = np.empty(len(left) * 2, dtype=np.int16)
        stereo[0::2] = left
        stereo[1::2] = right
        with self._wav_lock:
            self._wav_file.writeframes(stereo.tobytes())

    def _setup_resampler(self, native_rate: int) -> tuple[bool, int, int]:
        """Compute resampling parameters from native_rate to self.sample_rate."""
        if native_rate == self.sample_rate:
            return False, 1, 1
        g = gcd(native_rate, self.sample_rate)
        return True, self.sample_rate // g, native_rate // g

    def _on_loopback_audio(self, audio: np.ndarray):
        """Called from loopback stream with mono float32 at native rate."""
        if self._stop_event.is_set():
            return
        if self._lb_need_resample:
            audio = resample_poly(audio, self._lb_resample_up, self._lb_resample_down).astype(np.float32)
        self._push_audio("loopback", audio)

    def _make_mic_callback(self, native_rate: int):
        """Create sounddevice callback for the microphone."""
        need_resample, up, down = self._setup_resampler(native_rate)

        def callback(indata, frames, time_info, status):
            if self._stop_event.is_set():
                return
            audio = indata[:, 0].copy() if indata.shape[1] > 1 else indata.flatten().copy()
            if need_resample:
                audio = resample_poly(audio, up, down).astype(np.float32)
            self._push_audio("mic", audio)

        return callback

    def _push_audio(self, source: str, audio: np.ndarray):
        """Accumulate audio from a source and emit mixed chunks."""
        with self._buffer_lock:
            if source == "loopback":
                self._loopback_buffer = np.concatenate([self._loopback_buffer, audio])
            else:
                self._mic_buffer = np.concatenate([self._mic_buffer, audio])

            min_len = self._chunk_samples
            lb_len = len(self._loopback_buffer)
            mic_len = len(self._mic_buffer)

            if lb_len >= min_len and mic_len >= min_len:
                n = min(lb_len, mic_len, min_len)
                lb_chunk = self._loopback_buffer[:n]
                mic_chunk = self._mic_buffer[:n]
                self._loopback_buffer = self._loopback_buffer[n:]
                self._mic_buffer = self._mic_buffer[n:]

                mixed = (lb_chunk + mic_chunk) * 0.5
                self.audio_queue.put(mixed)
                self._write_wav_stereo(lb_chunk, mic_chunk)

            elif self._single_source and (lb_len >= min_len or mic_len >= min_len):
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
    def _single_source(self) -> bool:
        has_loopback = self.loopback_info is not None or self._loopback_sd_id is not None
        has_mic = self.mic_id is not None
        return not (has_loopback and has_mic)

    @property
    def loopback_name(self) -> str | None:
        if self.loopback_info:
            return self.loopback_info["name"]
        if self._loopback_sd_id is not None:
            return sd.query_devices(self._loopback_sd_id)["name"]
        return None

    def start(self):
        """Start capturing audio from available sources."""
        self._open_wav()
        started_loopback = False
        started_mic = False

        # Start WASAPI loopback (via pyaudiowpatch)
        if self.loopback_info is not None:
            try:
                native_rate = self.loopback_info["rate"]
                self._lb_need_resample, self._lb_resample_up, self._lb_resample_down = (
                    self._setup_resampler(native_rate)
                )
                self._loopback_stream = LoopbackStream(self.loopback_info, self._on_loopback_audio)
                self._loopback_stream.start()
                started_loopback = True
            except Exception as e:
                print(f"[warning] Could not open WASAPI loopback: {e}")
                self.loopback_info = None

        # Fallback: user specified a sounddevice loopback device ID
        elif self._loopback_sd_id is not None:
            try:
                native_rate = int(sd.query_devices(self._loopback_sd_id)["default_samplerate"])
                need_resample, up, down = self._setup_resampler(native_rate)

                def lb_cb(indata, frames, time_info, status):
                    if self._stop_event.is_set():
                        return
                    audio = indata[:, 0].copy() if indata.shape[1] > 1 else indata.flatten().copy()
                    if need_resample:
                        audio = resample_poly(audio, up, down).astype(np.float32)
                    self._push_audio("loopback", audio)

                stream = sd.InputStream(
                    device=self._loopback_sd_id,
                    samplerate=native_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=int(native_rate * 0.03),
                    callback=lb_cb,
                )
                stream.start()
                self._mic_stream = stream  # reuse field for cleanup
                started_loopback = True
            except sd.PortAudioError as e:
                print(f"[warning] Could not open loopback device {self._loopback_sd_id}: {e}")
                self._loopback_sd_id = None

        # Start microphone (via sounddevice)
        if self.mic_id is not None:
            try:
                native_rate = int(sd.query_devices(self.mic_id)["default_samplerate"])
                self._mic_stream = sd.InputStream(
                    device=self.mic_id,
                    samplerate=native_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=int(native_rate * 0.03),
                    callback=self._make_mic_callback(native_rate),
                )
                self._mic_stream.start()
                started_mic = True
            except sd.PortAudioError as e:
                print(f"[warning] Could not open mic device {self.mic_id}: {e}")
                self.mic_id = None

        if not started_loopback and not started_mic:
            raise RuntimeError(
                "No audio devices available. Use --list-devices to see options, "
                "then specify --loopback-device and/or --mic-device."
            )

    def stop(self):
        """Stop all audio streams and close WAV file."""
        self._stop_event.set()

        if self._loopback_stream:
            try:
                self._loopback_stream.stop()
            except Exception:
                pass
            self._loopback_stream = None

        if self._mic_stream:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

        with self._buffer_lock:
            self._loopback_buffer = np.zeros(0, dtype=np.float32)
            self._mic_buffer = np.zeros(0, dtype=np.float32)

        with self._wav_lock:
            if self._wav_file:
                self._wav_file.close()
                self._wav_file = None
