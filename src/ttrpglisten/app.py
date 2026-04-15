"""Application bootstrap - QApplication setup with dark Fusion theme."""

import atexit
import os
import signal
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def _build_dark_palette() -> QPalette:
    palette = QPalette()

    # Base colors
    dark = QColor("#1e1e1e")
    mid_dark = QColor("#2d2d30")
    mid = QColor("#3e3e42")
    text_color = QColor("#d4d4d4")
    bright_text = QColor("#ffffff")
    accent = QColor("#569cd6")
    disabled_text = QColor("#6a6a6a")
    highlight = QColor("#264f78")

    palette.setColor(QPalette.ColorRole.Window, dark)
    palette.setColor(QPalette.ColorRole.WindowText, text_color)
    palette.setColor(QPalette.ColorRole.Base, QColor("#1a1a2e"))
    palette.setColor(QPalette.ColorRole.AlternateBase, mid_dark)
    palette.setColor(QPalette.ColorRole.ToolTipBase, mid_dark)
    palette.setColor(QPalette.ColorRole.ToolTipText, text_color)
    palette.setColor(QPalette.ColorRole.Text, text_color)
    palette.setColor(QPalette.ColorRole.Button, mid_dark)
    palette.setColor(QPalette.ColorRole.ButtonText, text_color)
    palette.setColor(QPalette.ColorRole.BrightText, bright_text)
    palette.setColor(QPalette.ColorRole.Link, accent)
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, bright_text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, disabled_text)
    palette.setColor(QPalette.ColorRole.Mid, mid)
    palette.setColor(QPalette.ColorRole.Dark, dark)
    palette.setColor(QPalette.ColorRole.Light, mid)

    # Disabled state
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text
    )

    return palette


def _write_pid_file():
    """Write PID file next to the script/package for the shutdown script."""
    # Look for .pid location: prefer the repo root (where start.bat lives)
    # Fall back to CWD
    pid_path = Path(__file__).resolve().parents[2] / ".pid"
    try:
        pid_path.write_text(str(os.getpid()))
        atexit.register(lambda: pid_path.unlink(missing_ok=True))
    except OSError:
        pass



def _load_env():
    """Load .env file from the repo root if it exists."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    _load_env()
    # Warning suppression is in __init__.py (runs at import time)
    _write_pid_file()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_build_dark_palette())
    app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName("TTRPG Listen")

    # Let Ctrl+C kill the app. Qt's C++ event loop swallows SIGINT by default,
    # so we use a timer to give Python a chance to run its signal handler.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    _keepalive = QTimer()
    _keepalive.timeout.connect(lambda: None)
    _keepalive.start(200)

    window = MainWindow()
    window.setWindowTitle("TTRPG Listen")
    window.resize(1200, 800)
    window.show()

    return app.exec()
