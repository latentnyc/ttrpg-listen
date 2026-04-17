"""Cross-platform helpers."""

from __future__ import annotations

import os
import sys


def set_low_priority() -> None:
    """Lower the current thread/process priority so heavy AI work doesn't
    starve the GUI, game audio, or Discord.

    On Windows this lowers the OS-level *thread* priority. On POSIX the
    standard library only exposes a process-level `os.nice`, so the whole
    process niceness moves. That's acceptable here: the only other Python
    threads are the Qt GUI (mostly idle while models run) and the audio
    capture threads (which are PortAudio-managed and run at their own
    priority inside their callback loop)."""
    if sys.platform == "win32":
        try:
            import ctypes

            # THREAD_PRIORITY_BELOW_NORMAL = -1
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, -1)
        except Exception:
            pass
        return

    try:
        os.nice(5)
    except (AttributeError, OSError, PermissionError):
        pass
