"""Bottom control panel - buttons, device selectors, equalizers, status log."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..utils.config import GAME_PROMPTS
from ..widgets.equalizer import AudioEqualizerWidget


class ControlPanel(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    layout_toggle_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recording = False

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)

        # --- Left: Buttons + Game system ---
        left_group = QGroupBox("Controls")
        left_group.setStyleSheet(
            "QGroupBox { color: #8a8a8a; border: 1px solid #3e3e42; "
            "border-radius: 4px; margin-top: 6px; padding-top: 10px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        left_layout = QVBoxLayout(left_group)

        self._start_stop_btn = QPushButton("Start Recording")
        self._start_stop_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self._start_stop_btn.setMinimumHeight(36)
        self._start_stop_btn.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "border-radius: 4px; padding: 6px 16px; } "
            "QPushButton:hover { background-color: #388e3c; } "
            "QPushButton:pressed { background-color: #1b5e20; }"
        )
        self._start_stop_btn.clicked.connect(self._on_start_stop)
        left_layout.addWidget(self._start_stop_btn)

        self._layout_btn = QPushButton("Toggle Layout")
        self._layout_btn.setMinimumHeight(28)
        self._layout_btn.setStyleSheet(
            "QPushButton { background-color: #37474f; color: #d4d4d4; "
            "border-radius: 4px; padding: 4px 12px; } "
            "QPushButton:hover { background-color: #455a64; }"
        )
        self._layout_btn.clicked.connect(self.layout_toggle_requested.emit)
        left_layout.addWidget(self._layout_btn)

        # Game system selector
        left_layout.addWidget(QLabel("Game System:"))
        self._game_combo = QComboBox()
        for name in GAME_PROMPTS:
            self._game_combo.addItem(name)
        left_layout.addWidget(self._game_combo)

        left_layout.addStretch()
        main_layout.addWidget(left_group)

        # --- Middle: Device selectors ---
        mid_group = QGroupBox("Audio Devices")
        mid_group.setStyleSheet(left_group.styleSheet())
        mid_layout = QVBoxLayout(mid_group)

        mid_layout.addWidget(QLabel("Microphone:"))
        self._mic_combo = QComboBox()
        mid_layout.addWidget(self._mic_combo)

        mid_layout.addWidget(QLabel("System Audio:"))
        self._loopback_combo = QComboBox()
        mid_layout.addWidget(self._loopback_combo)

        mid_layout.addStretch()
        main_layout.addWidget(mid_group)

        # --- Right: Equalizers + Status ---
        right_group = QGroupBox("Monitor")
        right_group.setStyleSheet(left_group.styleSheet())
        right_layout = QVBoxLayout(right_group)

        eq_row = QHBoxLayout()
        self.input_equalizer = AudioEqualizerWidget("Mic In")
        self.output_equalizer = AudioEqualizerWidget("System Out")
        eq_row.addWidget(self.input_equalizer)
        eq_row.addWidget(self.output_equalizer)
        right_layout.addLayout(eq_row)

        self._status_log = QPlainTextEdit()
        self._status_log.setReadOnly(True)
        self._status_log.setMaximumHeight(60)
        self._status_log.setFont(QFont("Consolas", 8))
        self._status_log.setStyleSheet(
            "QPlainTextEdit { background-color: #1a1a2e; color: #8a8a8a; "
            "border: 1px solid #3e3e42; border-radius: 2px; }"
        )
        right_layout.addWidget(self._status_log)
        main_layout.addWidget(right_group)

    def _on_start_stop(self):
        if self._recording:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def set_recording_state(self, recording: bool):
        self._recording = recording
        if recording:
            self._start_stop_btn.setText("Stop Recording")
            self._start_stop_btn.setStyleSheet(
                "QPushButton { background-color: #c62828; color: white; "
                "border-radius: 4px; padding: 6px 16px; } "
                "QPushButton:hover { background-color: #d32f2f; } "
                "QPushButton:pressed { background-color: #b71c1c; }"
            )
        else:
            self._start_stop_btn.setText("Start Recording")
            self._start_stop_btn.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; "
                "border-radius: 4px; padding: 6px 16px; } "
                "QPushButton:hover { background-color: #388e3c; } "
                "QPushButton:pressed { background-color: #1b5e20; }"
            )

        # Lock/unlock controls during recording
        self._mic_combo.setEnabled(not recording)
        self._loopback_combo.setEnabled(not recording)
        self._game_combo.setEnabled(not recording)

    def log_message(self, msg: str):
        self._status_log.appendPlainText(msg)
        # Auto-scroll to bottom so latest message is visible
        scrollbar = self._status_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def populate_devices(self, mic_devices: list[dict], loopback_devices: list[dict]):
        self._mic_combo.clear()
        for dev in mic_devices:
            self._mic_combo.addItem(dev["name"], dev["index"])

        self._loopback_combo.clear()
        for dev in loopback_devices:
            self._loopback_combo.addItem(dev["name"], dev["index"])

    def select_device_by_index(self, device_type: str, device_index: int):
        """Pre-select a device in the combo box by its device index (stored as item data)."""
        combo = self._mic_combo if device_type == "mic" else self._loopback_combo
        for i in range(combo.count()):
            if combo.itemData(i) == device_index:
                combo.setCurrentIndex(i)
                return

    def selected_mic_index(self) -> int | None:
        idx = self._mic_combo.currentIndex()
        if idx < 0:
            return None
        return self._mic_combo.itemData(idx)

    def selected_loopback_index(self) -> int | None:
        idx = self._loopback_combo.currentIndex()
        if idx < 0:
            return None
        return self._loopback_combo.itemData(idx)

    def selected_game_system(self) -> str:
        return self._game_combo.currentText()

    def set_game_system(self, name: str):
        idx = self._game_combo.findText(name)
        if idx >= 0:
            self._game_combo.setCurrentIndex(idx)
