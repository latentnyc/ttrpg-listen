"""Rich terminal display for live streaming transcription."""

from __future__ import annotations

import threading
from datetime import datetime
from queue import Empty, Queue

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class TranscriptDisplay:
    """Displays live streaming transcription in a Rich terminal panel.

    Handles partial results from sherpa-onnx: each text update replaces
    the current line until an endpoint is detected (new line starts).
    """

    def __init__(self, text_queue: Queue, max_lines: int = 30):
        self.text_queue = text_queue
        self.max_lines = max_lines
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._completed_lines: list[str] = []
        self._current_partial: str = ""
        self._console = Console()
        self._silence_since: datetime | None = None

    def _render(self) -> Panel:
        text = Text()
        for line in self._completed_lines[-self.max_lines :]:
            text.append(line + "\n")

        # Show current partial result (being typed)
        if self._current_partial:
            timestamp = datetime.now().strftime("%H:%M:%S")
            text.append(f"[{timestamp}] {self._current_partial}", style="dim")

        if self._silence_since and not self._current_partial:
            elapsed = (datetime.now() - self._silence_since).seconds
            if elapsed > 30:
                text.append(f"\n(silence for {elapsed}s...)", style="dim italic")

        return Panel(
            text,
            title="[bold green]TTRPGListen - Live Transcript[/bold green]",
            subtitle="[dim]Ctrl+C to stop and generate full transcript[/dim]",
            border_style="green",
            padding=(0, 1),
        )

    def _display_loop(self):
        with Live(self._render(), console=self._console, refresh_per_second=8) as live:
            while not self._stop_event.is_set():
                updated = False
                try:
                    while True:
                        text = self.text_queue.get_nowait()
                        self._silence_since = None
                        updated = True

                        if text is None:
                            # Sentinel: endpoint detected, finalize current line
                            if self._current_partial:
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                self._completed_lines.append(
                                    f"[{timestamp}] {self._current_partial}"
                                )
                                self._current_partial = ""
                        else:
                            # Partial or updated result -- replace current line
                            self._current_partial = text

                except Empty:
                    if not self._current_partial and self._silence_since is None:
                        self._silence_since = datetime.now()

                if updated or self._stop_event.wait(0.1):
                    live.update(self._render())

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        # Finalize any remaining partial
        if self._current_partial:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._completed_lines.append(f"[{timestamp}] {self._current_partial}")
            self._current_partial = ""
        if self._thread:
            self._thread.join(timeout=3)

    def get_full_transcript(self) -> list[str]:
        return list(self._completed_lines)
