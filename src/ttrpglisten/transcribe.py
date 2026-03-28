"""Transcription engine supporting Moonshine and Whisper models."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


# TTRPG vocabulary prompt to improve domain recognition (used with Whisper models)
TTRPG_PROMPT = (
    "Dungeons & Dragons, TTRPG session. d20, armor class, hit points, initiative, "
    "saving throw, spell slot, cantrip, perception check, attack roll, natural 20, "
    "dungeon master, player character, non-player character, skill check, "
    "advantage, disadvantage, proficiency bonus."
)


class TranscriptionEngine:
    """Manages model loading and transcription for Moonshine and Whisper models."""

    def __init__(self, model_name: str, device: str = "cpu", language: str = "en"):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.model = None
        self.processor = None
        self.is_whisper = "whisper" in model_name.lower()
        self._prompt_ids = None

    def load(self):
        """Load the model and processor. Call this once at startup."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                dtype=self.dtype,
            ).to(self.device)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            if self.device != "cpu":
                print(f"[warning] {self.device.upper()} out of memory, falling back to CPU")
                if self.model is not None:
                    del self.model
                    self.model = None
                torch.cuda.empty_cache()
                self.device = "cpu"
                self.dtype = torch.float32
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    dtype=torch.float32,
                ).to("cpu")
            else:
                raise

        if self.is_whisper:
            try:
                self._prompt_ids = self.processor.get_prompt_ids(
                    TTRPG_PROMPT, return_tensors="pt"
                ).to(self.device)
            except Exception:
                print("[warning] Could not encode TTRPG vocabulary prompt; continuing without it")
                self._prompt_ids = None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a numpy audio array (float32, mono) to text."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if len(audio) == 0:
            return ""

        audio = audio.astype(np.float32)

        inputs = self.processor(
            audio,
            return_tensors="pt",
            sampling_rate=sample_rate,
        )

        if self.is_whisper:
            return self._transcribe_whisper(inputs, audio, sample_rate)
        return self._transcribe_moonshine(inputs, audio, sample_rate)

    def _transcribe_whisper(
        self, inputs, audio: np.ndarray, sample_rate: int
    ) -> str:
        """Transcribe using Whisper-specific generate parameters."""
        inputs = inputs.to(self.device, self.dtype)
        max_length = max(int(len(audio) / sample_rate * 4.5), 10)

        generate_kwargs = {
            "input_features": inputs.input_features,
            "max_length": max_length,
            "language": self.language,
        }
        if self._prompt_ids is not None:
            generate_kwargs["prompt_ids"] = self._prompt_ids

        with torch.no_grad():
            generated_ids = self.model.generate(**generate_kwargs)

        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    def _transcribe_moonshine(
        self, inputs, audio: np.ndarray, sample_rate: int
    ) -> str:
        """Transcribe using Moonshine models."""
        inputs = inputs.to(self.device)
        max_length = max(int(len(audio) / sample_rate * 6.5), 10)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
            )

        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()
