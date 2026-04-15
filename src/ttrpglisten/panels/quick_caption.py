"""Quick Caption panel - displays Windows Live Captions history.

Handles three behaviors of Live Captions:
1. New words appended at the end (speech continues)
2. Retroactive corrections anywhere in the visible window (re-recognition)
3. Old text scrolling off the top (window is fixed size)

The panel archives text that scrolls off and always shows:
    [archived frozen text] + [current LC visible text]
Corrections within the visible window update in place.
"""

from difflib import SequenceMatcher

from PySide6.QtCore import Slot
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)


class QuickCaptionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Archived words that have scrolled off the LC window
        self._frozen_words: list[str] = []
        # The previous visible snapshot (as words) for diffing
        self._last_visible_words: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QHBoxLayout()
        title = QLabel("Quick Caption")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setStyleSheet("color: #4fc3f7; padding: 2px;")
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

    @Slot(str)
    def on_caption_snapshot(self, visible_text: str):
        """Handle a full visible-text snapshot from Live Captions."""
        new_words = visible_text.split()
        if not new_words:
            return

        old_words = self._last_visible_words

        if not old_words:
            # First snapshot
            self._last_visible_words = new_words
            self._refresh_display()
            return

        # Use SequenceMatcher to find what's common between old and new.
        # This handles retroactive corrections (replacements), scrolling
        # (deletions from the front of old), and new text (insertions at
        # the end of new).
        sm = SequenceMatcher(None, old_words, new_words, autojunk=False)
        blocks = sm.get_matching_blocks()

        # Find how many words scrolled off the front of the old visible text.
        # These are words at the start of old_words that have no match in
        # new_words. The first matching block tells us where alignment starts.
        scrolled_off_count = 0
        if blocks:
            first_match = blocks[0]
            if first_match.size > 0:
                # first_match.a = index in old_words where first match starts
                # Everything before that in old_words has scrolled off.
                scrolled_off_count = first_match.a

        # Archive scrolled-off words
        if scrolled_off_count > 0:
            self._frozen_words.extend(old_words[:scrolled_off_count])

        self._last_visible_words = new_words
        self._refresh_display()

    def _refresh_display(self):
        """Rebuild the display: frozen archive + current visible."""
        all_words = self._frozen_words + self._last_visible_words
        full_text = " ".join(all_words)

        # Preserve scroll position if user scrolled up
        scrollbar = self._text_edit.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10

        self._text_edit.setPlainText(full_text)

        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def _copy_to_clipboard(self):
        QApplication.clipboard().setText(self._text_edit.toPlainText())

    def get_full_text(self) -> str:
        return self._text_edit.toPlainText()
