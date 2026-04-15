"""Worker that reads Windows 11 Live Captions via UI Automation."""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal


class LiveCaptionWorker(QObject):
    caption_snapshot = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, poll_interval_ms: int = 300):
        super().__init__()
        self._poll_interval_ms = poll_interval_ms
        self._stop_requested = False
        self._prev_text = ""
        self._debug_dumped = False
        self._consecutive_errors = 0

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception as e:
            self.error_occurred.emit(f"COM init failed: {e}")
            return

        try:
            self._poll_loop()
        except Exception as e:
            self.error_occurred.emit(f"Live Caption error: {e}")
        finally:
            try:
                import comtypes
                comtypes.CoUninitialize()
            except Exception:
                pass

    def _poll_loop(self):
        from comtypes import CLSCTX_INPROC_SERVER, CoCreateInstance, GUID

        CLSID_CUIAutomation = GUID("{FF48DBA4-60EF-4201-AA87-54103EEF594E}")

        try:
            from comtypes.gen.UIAutomationClient import (
                IUIAutomation,
                TreeScope_Descendants,
                UIA_ClassNamePropertyId,
                UIA_NamePropertyId,
            )
        except ImportError:
            import comtypes.client
            comtypes.client.GetModule("UIAutomationCore.dll")
            from comtypes.gen.UIAutomationClient import (
                IUIAutomation,
                TreeScope_Descendants,
                UIA_ClassNamePropertyId,
                UIA_NamePropertyId,
            )

        uia = CoCreateInstance(
            CLSID_CUIAutomation, interface=IUIAutomation, clsctx=CLSCTX_INPROC_SERVER
        )

        window = None
        self.error_occurred.emit(
            "Searching for Live Captions window... "
            "Enable via Win+Ctrl+L or Settings > Accessibility > Captions"
        )

        while not self._stop_requested:
            try:
                # Re-find the window every time. This is cheap and avoids
                # stale COM pointer issues when Live Captions recreates its UI.
                root = uia.GetRootElement()
                window = self._find_live_captions_window(uia, root)
                if window is None:
                    if self._consecutive_errors == 0:
                        self.error_occurred.emit("Live Captions window not found")
                    self._consecutive_errors += 1
                    QThread.msleep(2000)
                    continue

                if self._consecutive_errors > 0:
                    self.error_occurred.emit("Live Captions window found")
                    self._consecutive_errors = 0
                    self._debug_dumped = False

                text = self._read_caption_text(uia, window)
                if text and text != self._prev_text:
                    self.caption_snapshot.emit(text)
                    self._prev_text = text

            except Exception as e:
                self._consecutive_errors += 1
                # Log first error and then every 30th to avoid flooding
                if self._consecutive_errors == 1 or self._consecutive_errors % 30 == 0:
                    self.error_occurred.emit(
                        f"LC error ({self._consecutive_errors}x): {e}"
                    )
                window = None

            QThread.msleep(self._poll_interval_ms)

    def _find_live_captions_window(self, uia, root):
        from comtypes.gen.UIAutomationClient import (
            TreeScope_Descendants,
            UIA_ClassNamePropertyId,
            UIA_NamePropertyId,
        )

        for class_name in ["LiveCaptionsDesktopWindow", "LiveCaptions"]:
            try:
                condition = uia.CreatePropertyCondition(
                    UIA_ClassNamePropertyId, class_name
                )
                el = root.FindFirst(TreeScope_Descendants, condition)
                if el is not None:
                    return el
            except Exception:
                continue

        for name in ["Live Captions", "Live captions"]:
            try:
                condition = uia.CreatePropertyCondition(UIA_NamePropertyId, name)
                el = root.FindFirst(TreeScope_Descendants, condition)
                if el is not None:
                    return el
            except Exception:
                continue

        return None

    def _read_caption_text(self, uia, window) -> str:
        """Read caption text from the Live Captions window."""
        from comtypes.gen.UIAutomationClient import TreeScope_Descendants

        # FindAll can raise NULL COM pointer if window was destroyed.
        # Let that propagate so the poll loop re-acquires the window.
        condition = uia.CreateTrueCondition()
        elements = window.FindAll(TreeScope_Descendants, condition)
        if elements is None:
            return ""

        try:
            count = elements.Length
        except Exception:
            return ""

        # Debug dump once
        if not self._debug_dumped and count > 0:
            self._debug_dumped = True
            self.error_occurred.emit(f"LC tree: {count} elements")

        skip_lower = frozenset({
            "live captions", "live captions ", "settings", "pause",
            "close", "minimize", "maximize", "more options",
            "caption controls",
        })

        texts = []
        for i in range(count):
            try:
                el = elements.GetElement(i)
                if el is None:
                    continue
                name = el.CurrentName
                if name and name.strip():
                    text = name.strip()
                    if text.lower() not in skip_lower:
                        texts.append(text)
            except Exception:
                continue

        if not texts:
            return ""

        longest = max(texts, key=len)
        if len(longest) >= 8:
            return longest

        return " ".join(texts)
