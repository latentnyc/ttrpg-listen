"""TTRPGListen - Local AI-powered live transcription for TTRPG sessions."""

# Suppress cosmetic warnings from dependencies BEFORE they get imported.
# Must happen here (package init) because pyannote emits warnings at import time.
import logging as _logging
import warnings as _warnings

# pyannote's torchcodec warning (we pass waveform tensors, never decode files)
_warnings.filterwarnings("ignore", message="(?s).*torchcodec.*")
# pyannote TF32 reproducibility (informational only)
_warnings.filterwarnings("ignore", message="(?s).*TensorFloat-32.*")
# pyannote pooling edge case
_warnings.filterwarnings("ignore", message="(?s).*std\\(\\): degrees of freedom.*")
# HuggingFace cache symlinks on Windows
_warnings.filterwarnings("ignore", message="(?s).*cache-system uses symlinks.*")
# transformers sequential pipeline info
_warnings.filterwarnings("ignore", message="(?s).*pipelines sequentially on GPU.*")
# transformers duplicate logits processor (Whisper's config creates them AND generate()
# creates them again - the config ones take precedence which is correct behavior)
_warnings.filterwarnings("ignore", message="(?s).*custom logits processor.*")
# triton not found (profiling only)
_logging.getLogger("torch.utils.flop_counter").setLevel(_logging.ERROR)

__version__ = "0.2.0"
