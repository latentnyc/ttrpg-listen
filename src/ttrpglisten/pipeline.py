"""True streaming pipeline using sherpa-onnx for frame-by-frame transcription.

Audio flows in continuously. Text appears as words are recognized.
No chunking, no waiting for pauses. Fully non-blocking.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from queue import Empty, Queue

import sherpa_onnx


_MODEL_DIR_NAME = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{_MODEL_DIR_NAME}.tar.bz2"
)


def _find_model_dir() -> Path:
    """Locate the sherpa-onnx model directory."""
    # Check relative to package, then relative to cwd
    candidates = [
        Path(__file__).parent.parent.parent / "models" / _MODEL_DIR_NAME,
        Path.cwd() / "models" / _MODEL_DIR_NAME,
        Path.home() / ".cache" / "ttrpglisten" / _MODEL_DIR_NAME,
    ]
    for p in candidates:
        if (p / "tokens.txt").is_file():
            return p
    return candidates[-1]  # return default for download


def _ensure_model() -> Path:
    """Download model if not present. Returns model directory path."""
    model_dir = _find_model_dir()
    if (model_dir / "tokens.txt").is_file():
        return model_dir

    print(f"Downloading streaming model ({_MODEL_DIR_NAME})...")
    model_dir.mkdir(parents=True, exist_ok=True)

    import tarfile
    import urllib.request

    archive = model_dir.parent / f"{_MODEL_DIR_NAME}.tar.bz2"
    urllib.request.urlretrieve(_MODEL_URL, str(archive))
    with tarfile.open(str(archive), "r:bz2") as tar:
        tar.extractall(str(model_dir.parent))
    archive.unlink()
    print("Model downloaded.")
    return model_dir


def _create_recognizer(model_dir: Path, provider: str = "cpu") -> sherpa_onnx.OnlineRecognizer:
    """Create a sherpa-onnx streaming recognizer."""
    d = str(model_dir)
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{d}/tokens.txt",
        encoder=f"{d}/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        decoder=f"{d}/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        joiner=f"{d}/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        num_threads=4,
        sample_rate=16000,
        provider=provider,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=0.8,   # long pause -> endpoint
        rule2_min_trailing_silence=0.4,   # short pause after speech -> endpoint
        rule3_min_utterance_length=15.0,  # force endpoint after 15s
    )


class StreamingPipeline:
    """True streaming transcription: feed audio, get text immediately.

    Reads audio from audio_queue, feeds to sherpa-onnx frame-by-frame,
    and pushes recognized text to text_queue as soon as it's available.
    Fully non-blocking -- runs on a dedicated thread.
    """

    def __init__(
        self,
        audio_queue: Queue,
        text_queue: Queue,
        sample_rate: int = 16000,
        provider: str = "cpu",
    ):
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self.sample_rate = sample_rate
        self.provider = provider
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def _process_loop(self):
        model_dir = _ensure_model()
        recognizer = _create_recognizer(model_dir, self.provider)
        stream = recognizer.create_stream()
        last_text = ""

        while not self._stop_event.is_set():
            # Drain audio queue (non-blocking batch)
            got_audio = False
            try:
                while True:
                    chunk = self.audio_queue.get_nowait()
                    stream.accept_waveform(self.sample_rate, chunk)
                    got_audio = True
            except Empty:
                pass

            if not got_audio:
                # No audio available, brief sleep to avoid busy-waiting
                self._stop_event.wait(0.02)
                continue

            # Decode whatever is ready
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            # Get current partial result
            result = recognizer.get_result(stream)
            if result and result != last_text:
                # Emit the new text
                self.text_queue.put(result)
                last_text = result

            # Check if endpoint detected (speaker paused)
            if recognizer.is_endpoint(stream):
                if result:
                    # Send None sentinel to tell display to finalize the line
                    self.text_queue.put(None)
                    last_text = ""
                recognizer.reset(stream)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
