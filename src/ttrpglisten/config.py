"""Configuration management for TTRPGListen."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    loopback_device: int | str | None = None  # WASAPI loopback (remote players)
    mic_device: int | str | None = None  # Microphone (your voice)
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class StreamingConfig:
    model: str = "usefulsensors/moonshine-streaming-small"
    device: str = "auto"  # auto, cpu, cuda, mps
    language: str = "en"


@dataclass
class PostprocessConfig:
    model: str = "openai/whisper-large-v3-turbo"
    language: str = "en"
    diarization: bool = True
    min_speakers: int = 2
    max_speakers: int = 8


@dataclass
class OutputConfig:
    directory: str = "./transcripts"


@dataclass
class VadConfig:
    threshold: float = 0.5
    min_silence_duration_ms: int = 400
    speech_pad_ms: int = 150
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = 5.0


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    vad: VadConfig = field(default_factory=VadConfig)


PRESETS = {
    "low": {
        "streaming": {"model": "usefulsensors/moonshine-streaming-tiny"},
        "postprocess": {"model": "openai/whisper-large-v3-turbo", "diarization": False},
    },
    "medium": {
        "streaming": {"model": "usefulsensors/moonshine-streaming-small"},
        "postprocess": {"model": "openai/whisper-large-v3-turbo", "diarization": True},
    },
    "high": {
        "streaming": {"model": "usefulsensors/moonshine-streaming-medium"},
        "postprocess": {"model": "openai/whisper-large-v3-turbo", "diarization": True},
    },
}


def _apply_dict(obj, data: dict):
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def resolve_device(requested: str) -> str:
    """Resolve 'auto' device to the best available accelerator."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _validate_config(cfg: Config):
    """Validate config values and raise ValueError on invalid settings."""
    if cfg.audio.sample_rate <= 0:
        raise ValueError(f"audio.sample_rate must be positive, got {cfg.audio.sample_rate}")
    if cfg.postprocess.min_speakers < 1:
        raise ValueError(f"postprocess.min_speakers must be >= 1, got {cfg.postprocess.min_speakers}")
    if cfg.postprocess.max_speakers < cfg.postprocess.min_speakers:
        raise ValueError(
            f"postprocess.max_speakers ({cfg.postprocess.max_speakers}) "
            f"must be >= min_speakers ({cfg.postprocess.min_speakers})"
        )
    if cfg.streaming.device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"streaming.device must be auto/cpu/cuda/mps, got {cfg.streaming.device!r}")
    if cfg.vad.threshold < 0.0 or cfg.vad.threshold > 1.0:
        raise ValueError(f"vad.threshold must be between 0 and 1, got {cfg.vad.threshold}")
    if not cfg.postprocess.language or not isinstance(cfg.postprocess.language, str):
        raise ValueError(f"postprocess.language must be a non-empty string, got {cfg.postprocess.language!r}")


def load_config(config_path: str | Path | None = None, preset: str | None = None) -> Config:
    """Load config from YAML file, apply preset overrides, return Config."""
    cfg = Config()

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"[warning] Failed to parse config file {config_path}: {e}")
            data = {}
        for section_name in ("audio", "streaming", "postprocess", "output", "vad"):
            if section_name in data:
                _apply_dict(getattr(cfg, section_name), data[section_name])

    if preset and preset in PRESETS:
        for section_name, overrides in PRESETS[preset].items():
            _apply_dict(getattr(cfg, section_name), overrides)

    _validate_config(cfg)
    return cfg
