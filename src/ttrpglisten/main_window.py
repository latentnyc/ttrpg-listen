"""Main window - platform-aware layout + worker lifecycle management.

Windows: 2 caption panels (Quick from Live Captions + Accurate from whisperx)
         in a togglable splitter, plus the Controls panel below.
macOS:   only the Accurate caption panel + Controls (no Live Captions API).
"""

import sys
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtWidgets import QMainWindow, QSplitter, QVBoxLayout, QWidget

from .panels.accurate_caption import AccurateCaptionPanel
from .panels.controls import ControlPanel
from .utils.config import AppConfig

IS_WINDOWS = sys.platform == "win32"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self._config = AppConfig()
        self._recording = False

        # Worker references
        self._live_caption_worker = None
        self._live_caption_thread = None
        self._audio_capture_worker = None
        self._audio_capture_thread = None
        self._transcription_worker = None
        self._transcription_thread = None
        self._diarization_worker = None
        self._diarization_thread = None
        self._shared_buffer = None

        # UI references (only set when their panel exists)
        self._quick_panel = None
        self._splitter = None

        self._setup_ui()
        self._restore_settings()
        if IS_WINDOWS:
            self._start_live_captions()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        self._accurate_panel = AccurateCaptionPanel()

        if IS_WINDOWS:
            from .panels.quick_caption import QuickCaptionPanel

            self._quick_panel = QuickCaptionPanel()
            self._splitter = QSplitter(Qt.Orientation.Horizontal)
            self._splitter.addWidget(self._quick_panel)
            self._splitter.addWidget(self._accurate_panel)
            self._splitter.setStretchFactor(0, 1)
            self._splitter.setStretchFactor(1, 1)
            main_layout.addWidget(self._splitter, stretch=1)
        else:
            main_layout.addWidget(self._accurate_panel, stretch=1)

        # Bottom: control panel
        self._control_panel = ControlPanel(show_layout_toggle=IS_WINDOWS)
        self._control_panel.setMaximumHeight(220)
        main_layout.addWidget(self._control_panel, stretch=0)

        # Wire control panel signals
        self._control_panel.start_requested.connect(self._start_recording)
        self._control_panel.stop_requested.connect(self._stop_recording)
        self._control_panel.mic_settings_changed.connect(self._on_mic_settings_changed)
        if IS_WINDOWS:
            self._control_panel.layout_toggle_requested.connect(self._toggle_layout)

        # Populate devices
        self._populate_devices()

    def _restore_settings(self):
        if self._splitter is not None:
            orientation = self._config.layout_orientation
            if orientation == "vertical":
                self._splitter.setOrientation(Qt.Orientation.Vertical)
            else:
                self._splitter.setOrientation(Qt.Orientation.Horizontal)
        self._control_panel.set_game_system(self._config.game_system)

    def _populate_devices(self):
        try:
            from .audio.devices import (
                enumerate_input_devices,
                enumerate_loopback_devices,
                find_default_loopback,
                find_default_mic,
            )

            mic_devices = enumerate_input_devices()
            loopback_devices = enumerate_loopback_devices()
            self._control_panel.populate_devices(mic_devices, loopback_devices)

            # Pre-select system defaults
            default_mic = find_default_mic()
            if default_mic is not None:
                self._control_panel.select_device_by_index("mic", default_mic)

            default_lb = find_default_loopback()
            if default_lb is not None:
                self._control_panel.select_device_by_index("loopback", default_lb["index"])

            self._control_panel.log_message(
                f"Found {len(mic_devices)} mic(s), {len(loopback_devices)} loopback(s)"
            )
        except Exception as e:
            self._control_panel.log_message(f"Device enumeration error: {e}")

    def _toggle_layout(self):
        if self._splitter is None:
            return
        current = self._splitter.orientation()
        if current == Qt.Orientation.Horizontal:
            self._splitter.setOrientation(Qt.Orientation.Vertical)
            self._config.layout_orientation = "vertical"
        else:
            self._splitter.setOrientation(Qt.Orientation.Horizontal)
            self._config.layout_orientation = "horizontal"

    def _start_live_captions(self):
        """Windows-only: start the Live Captions UI Automation reader."""
        try:
            from .workers.live_caption import LiveCaptionWorker

            self._live_caption_thread = QThread()
            self._live_caption_worker = LiveCaptionWorker()
            self._live_caption_worker.moveToThread(self._live_caption_thread)

            self._live_caption_thread.started.connect(self._live_caption_worker.run)
            if self._quick_panel is not None:
                self._live_caption_worker.caption_snapshot.connect(
                    self._quick_panel.on_caption_snapshot
                )
            self._live_caption_worker.error_occurred.connect(self._control_panel.log_message)

            self._live_caption_thread.start()
            self._control_panel.log_message("Live Captions reader started")
        except Exception as e:
            self._control_panel.log_message(f"Live Captions error: {e}")

    @Slot()
    def _start_recording(self):
        if self._recording:
            return

        self._recording = True
        self._control_panel.set_recording_state(True)

        # Save game system selection
        self._config.game_system = self._control_panel.selected_game_system()

        mic_idx = self._control_panel.selected_mic_index()
        loopback_idx = self._control_panel.selected_loopback_index()

        # Create output directory and WAV path
        transcript_dir = Path(self._config.transcript_directory)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        wav_path = transcript_dir / f"session_{timestamp}.wav"

        self._control_panel.log_message(f"Recording to {wav_path}")

        try:
            self._start_audio_capture(mic_idx, loopback_idx, wav_path)
            self._start_transcription()
            self._start_diarization()
        except Exception as e:
            self._control_panel.log_message(f"Start error: {e}")
            self._stop_recording()

    def _start_audio_capture(self, mic_idx, loopback_idx, wav_path):
        from .audio.recorder import SharedAudioBuffer
        from .workers.audio_capture import AudioCaptureWorker

        self._shared_buffer = SharedAudioBuffer(sample_rate=self._config.sample_rate)

        self._audio_capture_thread = QThread()
        self._audio_capture_worker = AudioCaptureWorker(
            mic_device_idx=mic_idx,
            loopback_device_idx=loopback_idx,
            sample_rate=self._config.sample_rate,
            wav_path=wav_path,
            shared_buffer=self._shared_buffer,
            mic_gain=self._config.mic_gain,
            mic_sensitivity=self._config.mic_sensitivity,
        )
        self._audio_capture_worker.moveToThread(self._audio_capture_thread)

        self._audio_capture_thread.started.connect(self._audio_capture_worker.run)
        self._audio_capture_worker.audio_level_input.connect(
            self._control_panel.input_equalizer.update_levels
        )
        self._audio_capture_worker.audio_level_output.connect(
            self._control_panel.output_equalizer.update_levels
        )
        self._audio_capture_worker.status_message.connect(self._control_panel.log_message)
        self._audio_capture_worker.error_occurred.connect(self._control_panel.log_message)

        self._audio_capture_thread.start()

    def _start_transcription(self):
        from .workers.transcription import TranscriptionWorker

        self._transcription_thread = QThread()
        self._transcription_worker = TranscriptionWorker(
            shared_buffer=self._shared_buffer,
            game_prompt=self._config.game_prompt,
            language=self._config.language,
            mic_sensitivity=self._config.mic_sensitivity,
        )
        self._transcription_worker.moveToThread(self._transcription_thread)

        self._transcription_thread.started.connect(self._transcription_worker.run)
        self._transcription_worker.segment_ready.connect(self._on_transcription_segment)
        self._transcription_worker.mic_segment_ready.connect(self._on_mic_segment)
        self._transcription_worker.aligned_result_ready.connect(self._on_aligned_result)
        self._transcription_worker.status_message.connect(self._control_panel.log_message)
        self._transcription_worker.error_occurred.connect(self._control_panel.log_message)

        self._transcription_thread.start()

    def _start_diarization(self):
        from .workers.diarization import DiarizationWorker

        self._diarization_thread = QThread()
        self._diarization_worker = DiarizationWorker(
            shared_buffer=self._shared_buffer,
            min_speakers=self._config.min_speakers,
            max_speakers=self._config.max_speakers,
        )
        self._diarization_worker.moveToThread(self._diarization_thread)

        self._diarization_thread.started.connect(self._diarization_worker.run)
        self._diarization_worker.attributed_segment.connect(self._on_attributed_segment)
        self._diarization_worker.speaker_correction.connect(self._on_speaker_correction)
        self._diarization_worker.status_message.connect(self._control_panel.log_message)
        self._diarization_worker.error_occurred.connect(self._control_panel.log_message)

        self._diarization_thread.start()

    @Slot()
    def _stop_recording(self):
        if not self._recording:
            return

        self._recording = False
        self._control_panel.set_recording_state(False)
        self._control_panel.log_message("Stopping recording...")

        for worker, thread, name in [
            (self._diarization_worker, self._diarization_thread, "Diarization"),
            (self._transcription_worker, self._transcription_thread, "Transcription"),
            (self._audio_capture_worker, self._audio_capture_thread, "Audio capture"),
        ]:
            if worker and thread:
                worker.request_stop()
                thread.quit()
                if not thread.wait(5000):
                    self._control_panel.log_message(f"{name} thread did not stop cleanly")
                    thread.terminate()
                    thread.wait(2000)

        self._audio_capture_worker = None
        self._audio_capture_thread = None
        self._transcription_worker = None
        self._transcription_thread = None
        self._diarization_worker = None
        self._diarization_thread = None

        self._control_panel.input_equalizer.reset()
        self._control_panel.output_equalizer.reset()

        self._save_transcript()
        self._control_panel.log_message("Recording stopped")

    @Slot(float, float)
    def _on_mic_settings_changed(self, gain: float, sensitivity: float):
        """Persist new mic settings and push to live workers if recording."""
        self._config.mic_gain = gain
        self._config.mic_sensitivity = sensitivity
        if self._audio_capture_worker is not None:
            self._audio_capture_worker.set_mic_gain(gain)
            self._audio_capture_worker.set_mic_sensitivity(sensitivity)
        if self._transcription_worker is not None:
            self._transcription_worker.set_mic_sensitivity(sensitivity)
        self._control_panel.log_message(
            f"Mic settings updated: gain={gain:.2f}x, sensitivity={sensitivity:.4f}"
        )

    @Slot(str, float, float)
    def _on_transcription_segment(self, text: str, start_time: float, end_time: float):
        """Loopback transcription with generic speaker label; replaced by diarization."""
        self._accurate_panel.add_segment(text, "Speaker", start_time, end_time)

    @Slot(str, float, float)
    def _on_mic_segment(self, text: str, start_time: float, end_time: float):
        """Mic transcription attributed directly as 'Microphone'."""
        self._accurate_panel.add_segment(text, "Microphone", start_time, end_time)

    @Slot(object, float)
    def _on_aligned_result(self, aligned_result: dict, chunk_start_time: float):
        if self._diarization_worker:
            self._diarization_worker.enqueue_chunk(aligned_result, chunk_start_time)

    @Slot(str, str, float, float)
    def _on_attributed_segment(self, text: str, speaker: str, start: float, end: float):
        """Replace generic-speaker segments with speaker-attributed ones from diarization."""
        updated = False
        for seg in self._accurate_panel.get_segments():
            if seg["speaker"] != "Speaker":
                continue
            seg_start = seg["timestamp"]
            seg_end = seg.get("end_time", seg_start + 2.0)
            overlap_start = max(start, seg_start)
            overlap_end = min(end, seg_end)
            if overlap_end - overlap_start > 0.3:
                seg["speaker"] = speaker
                updated = True

        if updated:
            self._accurate_panel.rebuild_display()

    @Slot(float, float, str)
    def _on_speaker_correction(self, start: float, end: float, corrected_speaker: str):
        updated = False
        for seg in self._accurate_panel.get_segments():
            seg_start = seg["timestamp"]
            seg_end = seg.get("end_time", seg_start + 2.0)
            overlap_start = max(start, seg_start)
            overlap_end = min(end, seg_end)
            if overlap_end - overlap_start > 0.3:
                if seg["speaker"] != corrected_speaker:
                    seg["speaker"] = corrected_speaker
                    updated = True

        if updated:
            self._accurate_panel.rebuild_display()

    def _save_transcript(self):
        segments = self._accurate_panel.get_segments()
        if not segments:
            return

        try:
            transcript_dir = Path(self._config.transcript_directory)
            transcript_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            txt_path = transcript_dir / f"session_{timestamp}.txt"

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("TTRPG Listen - Session Transcript\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"Game: {self._control_panel.selected_game_system()}\n")
                f.write(f"{'=' * 50}\n\n")

                for seg in segments:
                    minutes = int(seg["timestamp"] // 60)
                    seconds = int(seg["timestamp"] % 60)
                    f.write(f"[{minutes:02d}:{seconds:02d}] {seg['speaker']}: {seg['text']}\n\n")

            self._control_panel.log_message(f"Transcript saved to {txt_path}")
        except OSError as e:
            self._control_panel.log_message(f"Failed to save transcript: {e}")

    def closeEvent(self, event):
        if self._recording:
            self._stop_recording()

        if self._live_caption_worker:
            self._live_caption_worker.request_stop()
        if self._live_caption_thread:
            self._live_caption_thread.quit()
            if not self._live_caption_thread.wait(5000):
                self._live_caption_thread.terminate()
                self._live_caption_thread.wait(2000)

        event.accept()
