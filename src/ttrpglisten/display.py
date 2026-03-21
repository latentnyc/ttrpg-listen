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
    """Displays live transcription with growing-window updates.

    Receives (text, is_final) tuples from the pipeline:
      - (text, False) = partial result, updates current line in place
      - (text, True)  = final result, commits line and starts new one
    """

    def __init__(self, text_queue: Queue, max_lines: int = 30):
        self.text_queue = text_queue
        self.max_lines = max_lines
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._completed_lines: list[str] = []
        self._current_partial: str = ""
        self._current_timestamp: str = ""
        self._console = Console()

    def _render(self) -> Panel:
        text = Text()
        for line in self._completed_lines[-self.max_lines :]:
            text.append(line + "\n")

        if self._current_partial:
            text.append(f"[{self._current_timestamp}] {self._current_partial}", style="bold")

        return Panel(
            text,
            title="[bold green]TTRPGListen - Live Transcript[/bold green]",
            subtitle="[dim]Ctrl+C to stop and generate full transcript[/dim]",
            border_style="green",
            padding=(0, 1),
        )

    def _display_loop(self):
        with Live(self._render(), console=self._console, refresh_per_second=4) as live:
            while not self._stop_event.is_set():
                updated = False
                try:
                    while True:
                        msg = self.text_queue.get_nowait()
                        text, is_final = msg
                        updated = True

                        if not self._current_timestamp:
                            self._current_timestamp = datetime.now().strftime("%H:%M:%S")

                        if is_final:
                            self._completed_lines.append(
                                f"[{self._current_timestamp}] {text}"
                            )
                            self._current_partial = ""
                            self._current_timestamp = ""
                        else:
                            self._current_partial = text
                except Empty:
                    pass

                if updated or self._stop_event.wait(0.15):
                    live.update(self._render())

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._current_partial:
            ts = self._current_timestamp or datetime.now().strftime("%H:%M:%S")
            self._completed_lines.append(f"[{ts}] {self._current_partial}")
            self._current_partial = ""
        if self._thread:
            self._thread.join(timeout=3)

    def get_full_transcript(self) -> list[str]:
        return list(self._completed_lines)
