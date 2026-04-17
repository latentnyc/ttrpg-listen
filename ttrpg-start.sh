#!/usr/bin/env bash
# TTRPG Listen - macOS startup script
# Mirrors ttrpg-start.bat: creates a venv, installs deps on first run, launches the app.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  TTRPG Listen - Starting up..."
echo "============================================"
echo

# ---- Locate Python (prefer 3.11+) ----
PYTHON=""
for cand in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYTHON="$cand"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python not found on PATH."
    echo "        Install Python 3.11+ (e.g. 'brew install python@3.12')."
    exit 1
fi

PYVER="$("$PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>&1)"
echo "  Python: $PYVER ($PYTHON)"

# Require >= 3.11
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' || {
    echo "[ERROR] Python 3.11+ is required. Found $PYVER."
    exit 1
}

# ---- Virtual environment ----
VENV_DIR="$SCRIPT_DIR/.venv"
FRESH_VENV=0

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo
    echo "  Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR" || {
        echo "[ERROR] Failed to create virtual environment."
        exit 1
    }
    FRESH_VENV=1
    echo "  Virtual environment created."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "  Activated: $VENV_DIR"

# ---- Install dependencies ----
needs_install=0
if [ "$FRESH_VENV" = "1" ]; then
    needs_install=1
else
    python -c 'import ttrpglisten' >/dev/null 2>&1 || needs_install=1
    if [ "$needs_install" = "0" ]; then
        # Reinstall if pyproject.toml is newer than the installed egg-info/dist-info.
        installed_marker="$(find "$VENV_DIR/lib" -maxdepth 4 -name 'ttrpg_listen*.dist-info' -print -quit 2>/dev/null)"
        if [ -n "$installed_marker" ] && [ "$SCRIPT_DIR/pyproject.toml" -nt "$installed_marker" ]; then
            needs_install=1
        fi
    fi
fi

if [ "$needs_install" = "1" ]; then
    echo
    echo "  Installing dependencies (first run can take several minutes)..."
    echo "  - PyTorch (with MPS on Apple Silicon), Whisper, pyannote-audio, PySide6..."
    echo

    # PyTorch first. Official macOS wheels include Metal Performance Shaders support.
    python -c 'import torch; import sys; sys.exit(0 if torch.__version__.startswith("2.") else 1)' >/dev/null 2>&1 \
        || pip install --quiet "torch>=2.1,<3" "torchaudio>=2.1,<3"

    # Package + transitive deps
    pip install -e . --quiet || {
        echo
        echo "[ERROR] Dependency installation failed."
        exit 1
    }
    echo "  Dependencies installed."
fi

# ---- Compute device check ----
echo
python -c '
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("  Device: CUDA %s (%.1f GB)" % (p.name, p.total_memory / (1024 ** 3)))
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("  Device: Metal Performance Shaders (Apple Silicon)")
else:
    print("  Device: CPU")
' 2>/dev/null || true

# ---- Ensure transcripts directory exists ----
mkdir -p "$SCRIPT_DIR/transcripts"

# ---- Warn if BlackHole is missing ----
if ! system_profiler SPAudioDataType 2>/dev/null | grep -qi 'BlackHole'; then
    echo
    echo "  [INFO] BlackHole not detected. Install it to capture system audio:"
    echo "         brew install blackhole-2ch"
    echo "         (See README for Audio MIDI Setup instructions.)"
fi

echo
echo "============================================"
echo "  Launching TTRPG Listen..."
echo "  Close the window or press Ctrl+C to stop."
echo "============================================"
echo

# PID file is written by the app itself and cleaned up via atexit
python -m ttrpglisten
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    echo
    echo "  TTRPG Listen exited with code $EXIT_CODE."
fi

exit "$EXIT_CODE"
