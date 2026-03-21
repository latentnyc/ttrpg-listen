"""Rich terminal display for live transcription."""

from __future__ import annotations

import threading
from datetime import datetime
from queue import Empty, Queue

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class TranscriptDisplay:
    """Displays live transcription in a scrolling Rich terminal panel."""

    def __init__(self, text_queue: Queue, max_lines: int = 30):
        self.text_queue = text_queue
        self.max_lines = max_lines
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lines: list[str] = []
        self._console = Console()
        self._silence_since: datetime | None = None

    def _render(self) -> Panel:
        """Render the current transcript state as a Rich Panel."""
        text = Text()
        for line in self._lines[-self.max_lines :]:
            text.append(line + "\n")

        if self._silence_since:
            elapsed = (datetime.now() - self._silence_since).seconds
            if elapsed > 60:
                text.append(f"\n(silence for {elapsed}s...)", style="dim italic")

        return Panel(
            text,
            title="[bold green]TTRPGListen - Live Transcript[/bold green]",
            subtitle="[dim]Ctrl+C to stop and generate full transcript[/dim]",
            border_style="green",
            padding=(0, 1),
        )

    def _display_loop(self):
        """Main display loop using Rich Live."""
        with Live(self._render(), console=self._console, refresh_per_second=4) as live:
            while not self._stop_event.is_set():
                updated = False
                try:
                    while True:
                        text = self.text_queue.get_nowait()
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        self._lines.append(f"[{timestamp}] {text}")
                        self._silence_since = None
                        updated = True
                except Empty:
                    if not self._lines:
                        if self._silence_since is None:
                            self._silence_since = datetime.now()
                    elif self._silence_since is None:
                        self._silence_since = datetime.now()

                if updated or self._stop_event.wait(0.25):
                    live.update(self._render())

    def start(self):
        """Start the display thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the display thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def get_full_transcript(self) -> list[str]:
        """Return all captured transcript lines."""
        return list(self._lines)
