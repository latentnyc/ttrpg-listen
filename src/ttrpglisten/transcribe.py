"""Transcription engine using Moonshine streaming models."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


# TTRPG vocabulary prompt to improve domain recognition
TTRPG_PROMPT = (
    "Dungeons & Dragons, TTRPG session. d20, armor class, hit points, initiative, "
    "saving throw, spell slot, cantrip, perception check, attack roll, natural 20, "
    "dungeon master, player character, non-player character, skill check, "
    "advantage, disadvantage, proficiency bonus."
)


class TranscriptionEngine:
    """Manages Moonshine model loading and transcription."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.model = None
        self.processor = None

    def load(self):
        """Load the model and processor. Call this once at startup."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                dtype=self.dtype,
            ).to(self.device)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # Fall back to CPU if GPU OOM
            if self.device != "cpu":
                print(f"[warning] {self.device.upper()} out of memory, falling back to CPU")
                self.device = "cpu"
                self.dtype = torch.float32
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    dtype=torch.float32,
                ).to("cpu")
            else:
                raise

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a numpy audio array (float32, mono) to text."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if len(audio) == 0:
            return ""

        # Ensure float32 for processor
        audio = audio.astype(np.float32)

        inputs = self.processor(
            audio,
            return_tensors="pt",
            sampling_rate=sample_rate,
        )
        inputs = inputs.to(self.device, self.dtype)

        # Estimate max tokens based on audio length
        seq_len = inputs.attention_mask.sum(dim=-1) if hasattr(inputs, "attention_mask") else None
        max_length = max(int(len(audio) / sample_rate * 6.5), 10)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
            )

        text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return text.strip()
