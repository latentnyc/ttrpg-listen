"""VRAM-aware model selection for Whisper and diarization."""

from __future__ import annotations


def detect_vram_gb() -> float:
    """Detect available VRAM in GB. Returns 0 if no CUDA GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def select_whisper_model(vram_gb: float | None = None) -> str:
    """Select optimal Whisper model for available VRAM."""
    if vram_gb is None:
        vram_gb = detect_vram_gb()

    if vram_gb >= 12.0:
        return "openai/whisper-large-v3"
    elif vram_gb >= 6.0:
        return "openai/whisper-large-v3-turbo"
    elif vram_gb >= 4.0:
        return "openai/whisper-medium"
    elif vram_gb >= 2.0:
        return "openai/whisper-small"
    else:
        return "openai/whisper-tiny"


def select_compute_device() -> str:
    """Resolve the best compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def select_diarization_strategy(vram_gb: float | None = None) -> dict:
    """Select diarization parameters based on VRAM."""
    if vram_gb is None:
        vram_gb = detect_vram_gb()

    if vram_gb >= 12.0:
        return {
            "keep_both_loaded": True,
            "process_interval_s": 60,
            "window_size_s": 600,
            "enabled": True,
        }
    elif vram_gb >= 8.0:
        return {
            "keep_both_loaded": False,
            "process_interval_s": 60,
            "window_size_s": 300,
            "enabled": True,
        }
    elif vram_gb >= 6.0:
        return {
            "keep_both_loaded": False,
            "process_interval_s": 90,
            "window_size_s": 180,
            "enabled": True,
        }
    else:
        return {
            "keep_both_loaded": False,
            "process_interval_s": 120,
            "window_size_s": 120,
            "enabled": vram_gb >= 4.0,
        }


def get_torch_dtype(device: str):
    """Get optimal dtype for the device."""
    import torch
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32
