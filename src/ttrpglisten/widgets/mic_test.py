"""Discord-style mic test dialog: gain, sensitivity, live meter, playback test.

Opens via the "Mic Settings..." button in the Controls panel. While open, the
dialog runs its own lightweight sounddevice.InputStream so the meter and
"Test Mic" playback work even when the main recording worker isn't active.
"""

from __future__ import annotations

import math
import threading
from collections import deque

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# Gain slider maps 5..100 -> 0.5x..10x (linear)
_GAIN_MIN = 0.5
_GAIN_MAX = 10.0

# Sensitivity slider maps 1..100 -> 0.001..0.1 (logarithmic)
_SENS_MIN = 0.001
_SENS_MAX = 0.1


def _gain_from_slider(v: int) -> float:
    # slider 5..100 -> 0.5..10
    return max(_GAIN_MIN, min(_GAIN_MAX, v / 10.0))


def _slider_from_gain(g: float) -> int:
    return int(round(max(_GAIN_MIN, min(_GAIN_MAX, g)) * 10.0))


def _sens_from_slider(v: int) -> float:
    # Log-scale 1..100 -> 0.001..0.1
    t = max(0, min(100, v - 1)) / 99.0
    log_min, log_max = math.log10(_SENS_MIN), math.log10(_SENS_MAX)
    return 10.0 ** (log_min + t * (log_max - log_min))


def _slider_from_sens(s: float) -> int:
    s = max(_SENS_MIN, min(_SENS_MAX, s))
    log_min, log_max = math.log10(_SENS_MIN), math.log10(_SENS_MAX)
    t = (math.log10(s) - log_min) / (log_max - log_min)
    return int(round(1 + t * 99))


def _describe_sensitivity(s: float) -> str:
    if s < 0.003:
        return "very sensitive (picks up whispers)"
    if s < 0.008:
        return "sensitive (quiet speech)"
    if s < 0.02:
        return "normal voice"
    if s < 0.05:
        return "loud voice"
    return "shout / close-talk only"


class LevelMeter(QWidget):
    """Simple vertical(ish) horizontal bar meter with a threshold line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0  # 0..1 RMS (post-gain)
        self._threshold = 0.008
        self.setMinimumHeight(24)
        self.setMinimumWidth(200)

    def set_level(self, rms: float):
        # Clamp and smooth a tiny bit
        self._level = max(0.0, min(1.0, rms))
        self.update()

    def set_threshold(self, t: float):
        self._threshold = max(0.0, min(1.0, t))
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        try:
            w, h = self.width(), self.height()
            # Background
            p.fillRect(0, 0, w, h, QColor("#1a1a2e"))
            # Filled level (green up to ~0.7, yellow ~0.85, red above)
            filled = int(w * self._level)
            level_color = QColor("#81c784")
            if self._level > 0.7:
                level_color = QColor("#ffb74d")
            if self._level > 0.9:
                level_color = QColor("#e57373")
            p.fillRect(0, 2, filled, h - 4, level_color)
            # Threshold line
            tx = int(w * self._threshold)
            pen = QPen(QColor("#fff176"))
            pen.setWidth(2)
            p.setPen(pen)
            p.drawLine(tx, 0, tx, h)
            # Border
            p.setPen(QColor("#3e3e42"))
            p.drawRect(0, 0, w - 1, h - 1)
        finally:
            p.end()


class MicTestDialog(QDialog):
    """Modal dialog for mic gain + sensitivity with a live meter and playback test."""

    def __init__(
        self,
        mic_device_idx: int | None,
        initial_gain: float,
        initial_sensitivity: float,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Mic Settings")
        self.setModal(True)
        self.resize(460, 360)

        self._mic_device_idx = mic_device_idx
        self._gain = float(initial_gain)
        self._sensitivity = float(initial_sensitivity)

        # Live sample queue (filled by sounddevice callback, drained by timer)
        self._sample_queue: deque[np.ndarray] = deque()
        self._queue_lock = threading.Lock()
        self._stream: sd.InputStream | None = None

        # Test-mic state machine
        self._test_buffer: list[np.ndarray] = []
        self._test_state = "idle"  # idle | recording | playing
        self._test_samplerate = 16000

        self._build_ui()
        self._start_stream()

        # UI refresh timer (~30 Hz)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # Test-recording clock
        self._test_timer = QTimer(self)
        self._test_timer.setSingleShot(True)
        self._test_timer.timeout.connect(self._finish_test_recording)

    # ---------- UI ----------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Device label
        dev_label = QLabel(self._device_label_text())
        dev_label.setStyleSheet("color: #8a8a8a;")
        layout.addWidget(dev_label)

        # Gain
        gain_row = QVBoxLayout()
        gain_row.setSpacing(2)
        self._gain_value_label = QLabel()
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(5, 100)
        self._gain_slider.setValue(_slider_from_gain(self._gain))
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        gain_row.addWidget(QLabel("Mic gain"))
        gain_row.addWidget(self._gain_slider)
        gain_row.addWidget(self._gain_value_label)
        layout.addLayout(gain_row)

        # Sensitivity
        sens_row = QVBoxLayout()
        sens_row.setSpacing(2)
        self._sens_value_label = QLabel()
        self._sens_slider = QSlider(Qt.Orientation.Horizontal)
        self._sens_slider.setRange(1, 100)
        self._sens_slider.setValue(_slider_from_sens(self._sensitivity))
        self._sens_slider.valueChanged.connect(self._on_sens_changed)
        sens_row.addWidget(QLabel("Sensitivity (noise floor)"))
        sens_row.addWidget(self._sens_slider)
        sens_row.addWidget(self._sens_value_label)
        layout.addLayout(sens_row)

        # Meter + help text
        layout.addWidget(QLabel("Live level (yellow line = sensitivity threshold):"))
        self._meter = LevelMeter(self)
        layout.addWidget(self._meter)

        # Test button
        test_row = QHBoxLayout()
        self._test_btn = QPushButton("Test Mic (records 3s, plays back)")
        self._test_btn.setStyleSheet(
            "QPushButton { background-color: #37474f; color: #d4d4d4; "
            "border-radius: 4px; padding: 6px 14px; } "
            "QPushButton:hover { background-color: #455a64; }"
        )
        self._test_btn.clicked.connect(self._start_test_recording)
        test_row.addWidget(self._test_btn)
        test_row.addStretch()
        layout.addLayout(test_row)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh_labels()

    def _device_label_text(self) -> str:
        try:
            if self._mic_device_idx is None:
                dev = sd.query_devices(kind="input")
            else:
                dev = sd.query_devices(self._mic_device_idx)
            return f"Device: {dev['name']}"
        except Exception:
            return "Device: (default)"

    # ---------- Slider handlers ----------

    @Slot(int)
    def _on_gain_changed(self, v: int):
        self._gain = _gain_from_slider(v)
        self._refresh_labels()

    @Slot(int)
    def _on_sens_changed(self, v: int):
        self._sensitivity = _sens_from_slider(v)
        self._meter.set_threshold(self._sensitivity)
        self._refresh_labels()

    def _refresh_labels(self):
        self._gain_value_label.setText(f"{self._gain:.2f}x")
        self._sens_value_label.setText(
            f"{self._sensitivity:.4f} — {_describe_sensitivity(self._sensitivity)}"
        )
        self._meter.set_threshold(self._sensitivity)

    # ---------- Live stream ----------

    def _start_stream(self):
        try:
            if self._mic_device_idx is None:
                dev_info = sd.query_devices(kind="input")
            else:
                dev_info = sd.query_devices(self._mic_device_idx)
            native_rate = int(dev_info["default_samplerate"])
            self._test_samplerate = native_rate
            channels = 1 if dev_info["max_input_channels"] >= 1 else 1

            def cb(indata, frames, time_info, status):
                audio = indata[:, 0].copy() if indata.shape[1] > 1 else indata.flatten().copy()
                with self._queue_lock:
                    self._sample_queue.append(audio)

            self._stream = sd.InputStream(
                device=self._mic_device_idx,
                samplerate=native_rate,
                channels=channels,
                dtype="float32",
                blocksize=max(256, int(native_rate * 0.03)),
                callback=cb,
            )
            self._stream.start()
        except Exception as e:
            # Not fatal — user can still tweak sliders blind.
            self._gain_value_label.setText(
                f"{self._gain:.2f}x  (meter unavailable: {e})"
            )

    def _stop_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ---------- Timer tick: drain queue, update meter, append test buffer ----------

    def _tick(self):
        with self._queue_lock:
            samples = list(self._sample_queue)
            self._sample_queue.clear()

        if not samples:
            return

        block = np.concatenate(samples)
        # Apply current gain + clip for meter display (matches runtime pipeline)
        amplified = np.clip(block * self._gain, -1.0, 1.0)

        rms = float(np.sqrt(np.mean(amplified ** 2)))
        self._meter.set_level(rms)

        if self._test_state == "recording":
            self._test_buffer.append(amplified.copy())

    # ---------- Test recording + playback ----------

    def _start_test_recording(self):
        if self._test_state != "idle":
            return
        if self._stream is None:
            return
        self._test_buffer = []
        self._test_state = "recording"
        self._test_btn.setEnabled(False)
        self._test_btn.setText("Recording... (3s)")
        self._test_timer.start(3000)

    def _finish_test_recording(self):
        if self._test_state != "recording":
            return
        self._test_state = "playing"
        self._test_btn.setText("Playing back...")

        if not self._test_buffer:
            self._end_test()
            return

        audio = np.concatenate(self._test_buffer)
        try:
            # Play back on the default output. If the user has made BlackHole
            # the default output they won't hear it — that's documented in
            # the README.
            sd.play(audio, samplerate=self._test_samplerate, blocking=False)
            # Schedule a callback when playback finishes. sd.wait() would block
            # the UI thread, so use a timer approximated from audio length.
            QTimer.singleShot(
                int(len(audio) / self._test_samplerate * 1000) + 200,
                self._end_test,
            )
        except Exception as e:
            self._test_btn.setText(f"Playback failed: {e}")
            QTimer.singleShot(1500, self._end_test)

    def _end_test(self):
        self._test_state = "idle"
        self._test_buffer = []
        self._test_btn.setEnabled(True)
        self._test_btn.setText("Test Mic (records 3s, plays back)")

    # ---------- Results ----------

    def result_gain(self) -> float:
        return float(self._gain)

    def result_sensitivity(self) -> float:
        return float(self._sensitivity)

    # ---------- Close/teardown ----------

    def closeEvent(self, event):
        self._timer.stop()
        self._test_timer.stop()
        try:
            sd.stop()
        except Exception:
            pass
        self._stop_stream()
        super().closeEvent(event)

    def done(self, code: int):
        self._timer.stop()
        self._test_timer.stop()
        try:
            sd.stop()
        except Exception:
            pass
        self._stop_stream()
        super().done(code)
