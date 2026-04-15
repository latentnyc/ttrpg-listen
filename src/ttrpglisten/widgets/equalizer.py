"""Real-time audio frequency visualizer using FFT."""

import numpy as np
from PySide6.QtCore import QTimer, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

BAND_COUNT = 16
UPDATE_INTERVAL_MS = 33  # ~30 fps


class AudioEqualizerWidget(QWidget):
    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)

        self._magnitudes = np.zeros(BAND_COUNT, dtype=np.float32)
        self._label_text = label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if label:
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #8a8a8a; font-size: 9px; padding: 0px;")
            layout.addWidget(lbl)

        self._canvas = _EqualizerCanvas(self._magnitudes, self)
        layout.addWidget(self._canvas)

        self.setMinimumSize(160, 60)
        self.setMaximumHeight(80)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._canvas.update)
        self._timer.start(UPDATE_INTERVAL_MS)

    @Slot(np.ndarray)
    def update_levels(self, audio_chunk: np.ndarray):
        if len(audio_chunk) < 64:
            return

        # Remove DC offset, apply window, normalize FFT by chunk length
        centered = audio_chunk - np.mean(audio_chunk)
        windowed = centered * np.hanning(len(centered))
        fft = np.abs(np.fft.rfft(windowed)) * (2.0 / len(windowed))
        n_bins = len(fft)

        # Logarithmic frequency binning
        freq_edges = np.logspace(
            np.log10(1), np.log10(n_bins), BAND_COUNT + 1, dtype=int
        )
        freq_edges = np.clip(freq_edges, 0, n_bins - 1)

        new_mags = np.zeros(BAND_COUNT, dtype=np.float32)
        for i in range(BAND_COUNT):
            lo = freq_edges[i]
            hi = max(freq_edges[i + 1], lo + 1)
            new_mags[i] = np.mean(fft[lo:hi])

        # Convert to dB scale relative to a fixed reference, then map to 0-1.
        # This gives absolute levels instead of always normalizing to peak.
        # -60 dB floor (silence) to 0 dB (loud) -> 0.0 to 1.0
        ref = 1.0  # full-scale float32 audio
        new_mags = np.clip(new_mags, 1e-10, None)
        db = 20.0 * np.log10(new_mags / ref)
        new_mags = np.clip((db + 60.0) / 60.0, 0.0, 1.0).astype(np.float32)

        # Exponential smoothing - fast attack, slow decay
        rising = new_mags > self._magnitudes
        self._magnitudes[rising] = 0.4 * self._magnitudes[rising] + 0.6 * new_mags[rising]
        self._magnitudes[~rising] = 0.85 * self._magnitudes[~rising] + 0.15 * new_mags[~rising]
        self._canvas._magnitudes = self._magnitudes

    def reset(self):
        self._magnitudes[:] = 0
        self._canvas._magnitudes = self._magnitudes


class _EqualizerCanvas(QWidget):
    def __init__(self, magnitudes: np.ndarray, parent=None):
        super().__init__(parent)
        self._magnitudes = magnitudes
        self.setMinimumHeight(40)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor("#1a1a2e"))

        bar_width = max(1, (w - (BAND_COUNT - 1)) // BAND_COUNT)
        gap = 1

        for i in range(BAND_COUNT):
            mag = float(self._magnitudes[i])
            bar_height = max(1, int(mag * h * 0.95))
            x = i * (bar_width + gap)
            y = h - bar_height

            # Color gradient: green -> yellow -> red
            if mag < 0.5:
                r = int(255 * mag * 2)
                g = 200
            else:
                r = 255
                g = int(200 * (1 - (mag - 0.5) * 2))
            color = QColor(r, g, 40)

            painter.fillRect(x, y, bar_width, bar_height, color)

        painter.end()
