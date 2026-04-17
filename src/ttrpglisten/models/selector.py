"""Device + model selection for Whisper and diarization.

Returns whisperx-compatible short names (e.g. "large-v3-turbo") rather than
HuggingFace paths. The transcription worker passes these straight to
whisperx.load_model().
"""

from __future__ import annotations

import sys


def detect_vram_gb() -> float:
    """Detect available CUDA VRAM in GB. Returns 0 on non-CUDA hosts."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def detect_system_memory_gb() -> float:
    """Best-effort total-system memory in GB (used for Apple Silicon sizing).

    On macOS we use sysctl so we don't take a psutil dependency. Falls back
    to sysconf on other POSIX hosts, or 0 if none of those are available."""
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL
            )
            return int(out.strip()) / (1024 ** 3)
        except Exception:
            pass
    try:
        import os

        pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else -1
        page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else -1
        if pages > 0 and page_size > 0:
            return (pages * page_size) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def select_compute_device() -> str:
    """Resolve the best PyTorch device string: "cuda", "mps", or "cpu"."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def select_whisper_model(vram_gb: float | None = None) -> str:
    """Pick the best whisperx model size for this host.

    Returns whisperx short names ("large-v3", "large-v3-turbo", "medium",
    "small", "tiny"), not HuggingFace paths.
    """
    device = select_compute_device()

    if device == "cuda":
        if vram_gb is None:
            vram_gb = detect_vram_gb()
        if vram_gb >= 12.0:
            return "large-v3"
        if vram_gb >= 6.0:
            return "large-v3-turbo"
        if vram_gb >= 4.0:
            return "medium"
        if vram_gb >= 2.0:
            return "small"
        return "tiny"

    if device == "mps":
        # Apple Silicon: whisperx itself runs on CPU (CTranslate2 has no MPS
        # backend), but Apple's NEON/AMX int8 is fast enough that
        # large-v3-turbo is usable on 16GB+ unified memory. Fall back for
        # smaller machines.
        mem = detect_system_memory_gb()
        if mem >= 16.0:
            return "large-v3-turbo"
        if mem >= 12.0:
            return "medium"
        if mem >= 8.0:
            return "small"
        return "tiny"

    # CPU-only host: be conservative.
    mem = detect_system_memory_gb()
    if mem >= 16.0:
        return "medium"
    if mem >= 8.0:
        return "small"
    return "tiny"


def select_diarization_strategy(vram_gb: float | None = None) -> dict:
    """Select diarization parameters based on available compute."""
    device = select_compute_device()

    if device == "cuda":
        if vram_gb is None:
            vram_gb = detect_vram_gb()
        if vram_gb >= 12.0:
            return {
                "keep_both_loaded": True,
                "process_interval_s": 60,
                "window_size_s": 600,
                "enabled": True,
            }
        if vram_gb >= 8.0:
            return {
                "keep_both_loaded": False,
                "process_interval_s": 60,
                "window_size_s": 300,
                "enabled": True,
            }
        if vram_gb >= 6.0:
            return {
                "keep_both_loaded": False,
                "process_interval_s": 90,
                "window_size_s": 180,
                "enabled": True,
            }
        return {
            "keep_both_loaded": False,
            "process_interval_s": 120,
            "window_size_s": 120,
            "enabled": vram_gb >= 4.0,
        }

    if device == "mps":
        return {
            "keep_both_loaded": False,
            "process_interval_s": 60,
            "window_size_s": 180,
            "enabled": True,
        }

    return {
        "keep_both_loaded": False,
        "process_interval_s": 180,
        "window_size_s": 120,
        "enabled": detect_system_memory_gb() >= 8.0,
    }


def get_torch_dtype(device: str):
    """Get optimal dtype for the device."""
    import torch
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32
