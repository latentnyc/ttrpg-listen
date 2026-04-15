"""Audio capture worker - WASAPI loopback + sounddevice mic on a QThread."""

from __future__ import annotations

import traceback
import threading
from collections import deque
from math import gcd
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, QThread, Signal

from ..audio.devices import find_default_loopback, find_default_mic
from ..audio.recorder import CrashSafeWavWriter, SharedAudioBuffer


class AudioCaptureWorker(QObject):
    audio_level_input = Signal(np.ndarray)
    audio_level_output = Signal(np.ndarray)
    status_message = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        mic_device_idx: int | None,
        loopback_device_idx: int | None,
        sample_rate: int,
        wav_path: Path | None,
        shared_buffer: SharedAudioBuffer,
    ):
        super().__init__()
        self._mic_device_idx = mic_device_idx
        self._loopback_device_idx = loopback_device_idx
        self._sample_rate = sample_rate
        self._wav_path = wav_path
        self._shared_buffer = shared_buffer
        self._stop_event = threading.Event()

        self._wav_writer: CrashSafeWavWriter | None = None
        self._mic_stream: sd.InputStream | None = None
        self._loopback_thread = None

        # Buffers for synchronized writing
        self._loopback_buf = np.zeros(0, dtype=np.float32)
        self._mic_buf = np.zeros(0, dtype=np.float32)
        self._buf_lock = threading.Lock()
        self._chunk_samples = int(sample_rate * 0.1)  # 100ms chunks

        self._has_both = False
        self._loopback_info: dict | None = None

        # Raw mic audio queue - callback pushes here, main loop resamples
        self._mic_raw_queue: deque[np.ndarray] = deque()
        self._mic_need_resample = False
        self._mic_up = 1
        self._mic_down = 1

        # For equalizer updates (~30fps)
        self._eq_counter = 0

    def request_stop(self):
        self._stop_event.set()

    def run(self):
        """Start audio capture. Called from QThread."""
        try:
            self._setup_wav()
            self._setup_mic()
            self._setup_loopback()

            self._has_both = self._mic_stream is not None and self._loopback_thread is not None

            if self._mic_stream is None and self._loopback_thread is None:
                self.error_occurred.emit("No audio devices available")
                return

            self.status_message.emit("Audio capture started")

            # Main loop: drain raw mic queue and resample on this thread
            # (scipy.resample_poly can't run inside the sounddevice callback
            # because it triggers a torch import cycle via array API detection)
            from scipy.signal import resample_poly

            # Software gain for quiet USB mics (Blue Snowball etc)
            MIC_GAIN = 3.0

            while not self._stop_event.is_set():
                while self._mic_raw_queue:
                    raw = self._mic_raw_queue.popleft()
                    if self._mic_need_resample:
                        raw = resample_poly(raw, self._mic_up, self._mic_down).astype(np.float32)
                    # Apply gain boost and clip to prevent distortion
                    raw = np.clip(raw * MIC_GAIN, -1.0, 1.0)
                    self._on_audio("mic", raw)

                self._stop_event.wait(0.02)

        except Exception as e:
            self.error_occurred.emit(f"Audio capture error: {e}\n{traceback.format_exc()}")
        finally:
            self._cleanup()

    def _setup_wav(self):
        if self._wav_path:
            self._wav_writer = CrashSafeWavWriter(
                self._wav_path, self._sample_rate, channels=2
            )
            self._wav_writer.open()

    def _setup_mic(self):
        mic_id = self._mic_device_idx
        if mic_id is None:
            mic_id = find_default_mic()

        if mic_id is None:
            self.status_message.emit("No microphone found")
            return

        try:
            dev_info = sd.query_devices(mic_id)
            dev_name = dev_info["name"]
            native_rate = int(dev_info["default_samplerate"])
            max_channels = int(dev_info["max_input_channels"])

            if max_channels < 1:
                self.error_occurred.emit(f"Mic '{dev_name}' has no input channels")
                return

            self._mic_need_resample = native_rate != self._sample_rate
            if self._mic_need_resample:
                g = gcd(native_rate, self._sample_rate)
                self._mic_up = self._sample_rate // g
                self._mic_down = native_rate // g

            def mic_cb(indata, frames, time_info, status):
                if self._stop_event.is_set():
                    return
                audio = indata[:, 0].copy() if indata.shape[1] > 1 else indata.flatten().copy()
                # Push raw audio to queue; resampling happens on the worker thread
                self._mic_raw_queue.append(audio)

            # Some devices don't support mono - try mono first, fall back to max channels
            channels = 1
            blocksize = max(256, int(native_rate * 0.03))
            try:
                self._mic_stream = sd.InputStream(
                    device=mic_id,
                    samplerate=native_rate,
                    channels=channels,
                    dtype="float32",
                    blocksize=blocksize,
                    callback=mic_cb,
                )
            except sd.PortAudioError:
                channels = min(max_channels, 2)
                self.status_message.emit(
                    f"Mic mono failed, trying {channels}ch..."
                )
                self._mic_stream = sd.InputStream(
                    device=mic_id,
                    samplerate=native_rate,
                    channels=channels,
                    dtype="float32",
                    blocksize=blocksize,
                    callback=mic_cb,
                )

            self._mic_stream.start()
            self.status_message.emit(
                f"Mic: {dev_name} ({native_rate}Hz, {channels}ch)"
            )
        except Exception as e:
            self.error_occurred.emit(f"Mic error: {e}")

    def _setup_loopback(self):
        loopback_info = None

        if self._loopback_device_idx is not None:
            # User selected a specific loopback device
            try:
                import pyaudiowpatch as pyaudio
                p = pyaudio.PyAudio()
                try:
                    self.status_message.emit(
                        f"Opening loopback device idx={self._loopback_device_idx}..."
                    )
                    dev = p.get_device_info_by_index(self._loopback_device_idx)
                    loopback_info = {
                        "index": self._loopback_device_idx,
                        "name": dev["name"],
                        "channels": max(1, int(dev["maxInputChannels"])),
                        "rate": int(dev["defaultSampleRate"]),
                    }
                finally:
                    p.terminate()
            except Exception as e:
                self.error_occurred.emit(f"Loopback device error: {e}")

        if loopback_info is None:
            loopback_info = find_default_loopback()

        if loopback_info is None:
            self.status_message.emit("No loopback device found")
            return

        self._loopback_info = loopback_info

        try:
            self._loopback_thread = _LoopbackThread(
                loopback_info,
                lambda audio: self._on_audio("loopback", audio),
                lambda msg: self.error_occurred.emit(msg),
                self._sample_rate,
                self._stop_event,
            )
            self._loopback_thread.start()
            self.status_message.emit(f"Loopback: {loopback_info['name']}")
        except Exception as e:
            self.error_occurred.emit(f"Loopback error: {e}")
            self._loopback_thread = None

    def _on_audio(self, source: str, audio: np.ndarray):
        """Handle incoming audio from mic or loopback."""
        # Emit for equalizer (~30fps = every 3rd chunk at 100ms chunks)
        self._eq_counter += 1
        if self._eq_counter % 3 == 0:
            if source == "mic":
                self.audio_level_input.emit(audio.copy())
            else:
                self.audio_level_output.emit(audio.copy())

        with self._buf_lock:
            if source == "loopback":
                self._loopback_buf = np.concatenate([self._loopback_buf, audio])
            else:
                self._mic_buf = np.concatenate([self._mic_buf, audio])

            self._drain_buffers()

    def _drain_buffers(self):
        """Process buffered audio - must be called with _buf_lock held."""
        min_len = self._chunk_samples

        if self._has_both:
            lb_len = len(self._loopback_buf)
            mic_len = len(self._mic_buf)
            if lb_len >= min_len and mic_len >= min_len:
                n = min(lb_len, mic_len, min_len)
                lb_chunk = self._loopback_buf[:n]
                mic_chunk = self._mic_buf[:n]
                self._loopback_buf = self._loopback_buf[n:]
                self._mic_buf = self._mic_buf[n:]

                if self._wav_writer:
                    self._wav_writer.write_stereo(lb_chunk, mic_chunk)
                self._shared_buffer.append(lb_chunk, mic_chunk)
        else:
            if len(self._loopback_buf) >= min_len:
                chunk = self._loopback_buf[:min_len]
                self._loopback_buf = self._loopback_buf[min_len:]
                silence = np.zeros(min_len, dtype=np.float32)
                if self._wav_writer:
                    self._wav_writer.write_stereo(chunk, silence)
                self._shared_buffer.append(chunk, silence)
            elif len(self._mic_buf) >= min_len:
                chunk = self._mic_buf[:min_len]
                self._mic_buf = self._mic_buf[min_len:]
                silence = np.zeros(min_len, dtype=np.float32)
                if self._wav_writer:
                    self._wav_writer.write_stereo(silence, chunk)
                self._shared_buffer.append(silence, chunk)

    def _cleanup(self):
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

        if self._wav_writer:
            self._wav_writer.close()
            self._wav_writer = None

        self.status_message.emit("Audio capture stopped")


class _LoopbackThread:
    """Captures system audio via pyaudiowpatch WASAPI loopback on a daemon thread."""

    def __init__(
        self,
        loopback_info: dict,
        on_audio,
        on_error,
        target_rate: int,
        stop_event: threading.Event,
    ):
        self._info = loopback_info
        self._on_audio = on_audio
        self._on_error = on_error
        self._target_rate = target_rate
        self._stop_event = stop_event
        self._thread: threading.Thread | None = None

        native_rate = loopback_info["rate"]
        self._need_resample = native_rate != target_rate
        if self._need_resample:
            g = gcd(native_rate, target_rate)
            self._up = target_rate // g
            self._down = native_rate // g

    def start(self):
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        import pyaudiowpatch as pyaudio
        from scipy.signal import resample_poly

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
        except Exception as e:
            self._on_error(f"Loopback thread error: {e}")
        finally:
            pa.terminate()

    def stop(self):
        # stop_event is shared, already set by parent
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
