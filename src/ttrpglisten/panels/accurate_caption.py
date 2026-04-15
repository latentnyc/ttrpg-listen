"""Accurate Caption panel - Whisper transcription with color-coded speakers."""

from PySide6.QtCore import QTimer, Slot
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

SPEAKER_COLORS = [
    "#4fc3f7",  # light blue
    "#81c784",  # green
    "#ffb74d",  # orange
    "#e57373",  # red
    "#ba68c8",  # purple
    "#4dd0e1",  # cyan
    "#fff176",  # yellow
    "#f48fb1",  # pink
]


class AccurateCaptionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._speaker_colors: dict[str, str] = {}
        self._next_color_idx: int = 0
        self._segments: list[dict] = []

        # Debounce timer for rebuild_display - batches rapid corrections
        # into a single UI update (200ms delay)
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(200)
        self._rebuild_timer.timeout.connect(self._do_rebuild)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QHBoxLayout()
        title = QLabel("Accurate Caption")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setStyleSheet("color: #81c784; padding: 2px;")
        header.addWidget(title)
        header.addStretch()

        copy_btn = QPushButton("Copy")
        copy_btn.setFixedSize(50, 22)
        copy_btn.setStyleSheet(
            "QPushButton { background-color: #37474f; color: #d4d4d4; "
            "border-radius: 3px; font-size: 9px; } "
            "QPushButton:hover { background-color: #455a64; }"
        )
        copy_btn.clicked.connect(self._copy_to_clipboard)
        header.addWidget(copy_btn)

        layout.addLayout(header)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Segoe UI", 10))
        self._text_edit.setStyleSheet(
            "QTextEdit { background-color: #1a1a2e; color: #d4d4d4; "
            "border: 1px solid #3e3e42; border-radius: 4px; padding: 4px; }"
        )
        layout.addWidget(self._text_edit)

    def _get_speaker_color(self, speaker: str) -> str:
        if speaker not in self._speaker_colors:
            color = SPEAKER_COLORS[self._next_color_idx % len(SPEAKER_COLORS)]
            self._speaker_colors[speaker] = color
            self._next_color_idx += 1
        return self._speaker_colors[speaker]

    @Slot(str, str, float, float)
    def add_segment(self, text: str, speaker: str, timestamp: float, end_time: float = 0.0):
        color = self._get_speaker_color(speaker)
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)

        self._segments.append({
            "text": text,
            "speaker": speaker,
            "timestamp": timestamp,
            "end_time": end_time,
        })

        html = (
            f'<span style="color:{color}">'
            f"<b>[{minutes:02d}:{seconds:02d}] {speaker}:</b> {text}"
            f"</span><br>"
        )
        self._text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self._text_edit.insertHtml(html)
        self._text_edit.moveCursor(QTextCursor.MoveOperation.End)

    @Slot(dict)
    def update_speaker_colors(self, speaker_mapping: dict):
        """Re-render all segments with updated speaker assignments.

        speaker_mapping: dict of old_speaker -> new_speaker for merges/splits.
        """
        for seg in self._segments:
            old = seg["speaker"]
            if old in speaker_mapping:
                seg["speaker"] = speaker_mapping[old]

        self.rebuild_display()

    def rebuild_display(self):
        """Schedule a debounced rebuild. Multiple rapid calls collapse into one."""
        if not self._rebuild_timer.isActive():
            self._rebuild_timer.start()

    def _do_rebuild(self):
        """Actually rebuild the display. Called by debounce timer."""
        scrollbar = self._text_edit.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10

        self._text_edit.clear()
        for seg in self._segments:
            color = self._get_speaker_color(seg["speaker"])
            minutes = int(seg["timestamp"] // 60)
            seconds = int(seg["timestamp"] % 60)
            html = (
                f'<span style="color:{color}">'
                f'<b>[{minutes:02d}:{seconds:02d}] {seg["speaker"]}:</b> {seg["text"]}'
                f"</span><br>"
            )
            self._text_edit.moveCursor(QTextCursor.MoveOperation.End)
            self._text_edit.insertHtml(html)

        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            self._text_edit.moveCursor(QTextCursor.MoveOperation.End)

    def _copy_to_clipboard(self):
        lines = []
        for seg in self._segments:
            minutes = int(seg["timestamp"] // 60)
            seconds = int(seg["timestamp"] % 60)
            lines.append(f"[{minutes:02d}:{seconds:02d}] {seg['speaker']}: {seg['text']}")
        QApplication.clipboard().setText("\n".join(lines))

    def get_segments(self) -> list[dict]:
        return list(self._segments)
