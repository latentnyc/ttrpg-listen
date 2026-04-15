# TTRPG Listen

Real-time AI-powered transcription for tabletop roleplaying sessions. Windows 11 desktop app with a dark-mode GUI that captures system audio and microphone, transcribes with speaker identification, and keeps a full session history.

Runs entirely on your machine using your GPU. No cloud services, no data leaves your computer.

## What it does

**Three-panel interface:**

- **Quick Caption** (left) -- reads Windows 11 Live Captions in real time, keeps the full session history that Live Captions normally discards. Handles retroactive corrections from the speech recognizer without duplicating text.

- **Accurate Caption** (right) -- high-quality transcription using [whisperx](https://github.com/m-bain/whisperX) (Whisper + wav2vec2 forced alignment) with color-coded speaker identification via [pyannote-audio](https://github.com/pyannote/pyannote-audio). Processes audio in 30-second chunks with overlapping 60-second diarization windows to correct speaker assignments at chunk boundaries.

- **Controls** (bottom) -- start/stop recording, toggle horizontal/vertical layout, select game system (D&D or Blades in the Dark vocabulary prompts), choose audio devices, real-time FFT equalizer displays for mic and system audio, scrollable status log.

**Key features:**

- Dual audio capture: WASAPI loopback (system audio) + microphone as separate channels
- Mic channel transcribed independently and attributed as "Microphone" (no diarization needed for the local player)
- System audio channel transcribed and speaker-diarized to identify remote players
- VRAM-aware model selection (tiny through large-v3 based on GPU memory)
- Crash-safe WAV recording with periodic header updates
- Game-system-specific vocabulary prompts for better recognition of TTRPG terms
- Low-priority GPU/CPU scheduling so the app doesn't interfere with gameplay

## Requirements

- **Windows 11** (for Live Captions and WASAPI loopback)
- **Python 3.11+**
- **NVIDIA GPU** with 6+ GB VRAM recommended (works on CPU but much slower)
  - GTX 1070 (8 GB): whisper-large-v3-turbo
  - RTX 3090 (24 GB): whisper-large-v3
- **HuggingFace account** with accepted terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## Quick start

### Option 1: Start script (recommended)

```batch
git clone https://github.com/latentnyc/ttrpg-listen.git
cd ttrpg-listen
```

Create a `.env` file with your HuggingFace token (required for speaker diarization):

```
HF_TOKEN=hf_your_token_here
```

Then double-click `ttrpg-start.bat` or run it from a terminal. It will:
1. Create a virtual environment
2. Install PyTorch with CUDA 12.6
3. Install all dependencies
4. Launch the app

To stop: close the window, press Ctrl+C in the terminal, or run `ttrpg-stop.bat`.

### Option 2: Manual install

```bash
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch with CUDA (must be 2.8.x for whisperx compatibility)
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install the app
pip install -e .

# Run
python -m ttrpglisten
```

### Enable Live Captions

Press **Win+Ctrl+L** or go to **Settings > Accessibility > Captions** and enable Live Captions. The app reads the Live Captions window via Windows UI Automation.

## How it works

```
                                    +---> Quick Caption panel
Windows Live Captions ----------->  |     (full session history)
  (UI Automation polling)           |

System Audio (WASAPI loopback) ---> AudioCaptureWorker --+-> WAV file (stereo)
Microphone (WASAPI) ------------>   (separate channels)  |   L=system, R=mic
                                                         |
                                                         +-> SharedAudioBuffer
                                                              |          |
                                               +--------------+          |
                                               |                         |
                                    TranscriptionWorker          DiarizationWorker
                                     (whisperx pipeline)          (pyannote + whisperx
                                               |                  speaker assignment)
                                     +---------+---------+              |
                                     |                   |              |
                              Loopback segments    Mic segments   Speaker corrections
                              (speaker TBD)        ("Microphone")       |
                                     |                   |              |
                                     +----> Accurate Caption panel <----+
                                            (color-coded speakers)
```

### Transcription pipeline (per 30s chunk)

1. **whisperx.transcribe()** -- VAD-segmented batched Whisper with game-specific vocabulary prompt
2. **whisperx.align()** -- wav2vec2 forced alignment for precise word-level timestamps
3. **Loopback segments** displayed immediately with generic "Speaker" label
4. **Mic segments** displayed immediately as "Microphone"

### Diarization pipeline (60s overlapping windows)

1. **pyannote speaker-diarization-3.1** on the loopback audio
2. **whisperx.assign_word_speakers()** merges word timestamps with speaker segments
3. Retroactively updates "Speaker" labels to "SPEAKER_00", "SPEAKER_01", etc.
4. Overlapping windows cross-check speaker assignments at chunk boundaries

### Audio capture

- **System loopback** via [pyaudiowpatch](https://github.com/s0d3s/PyAudioWPatch) -- captures from whatever Windows output device is active (speakers, Bluetooth, USB, HDMI)
- **Microphone** via [sounddevice](https://python-sounddevice.readthedocs.io/) -- WASAPI devices only for best quality
- Both resampled to 16kHz, mic boosted with 3x software gain
- Stereo WAV written with crash-safe periodic flushing (data every 5s, header every 30s)

## Game system prompts

The app uses domain-specific vocabulary prompts to improve Whisper's recognition of game terminology. Select the game system in the Controls panel before recording.

**Dungeons & Dragons** -- d20, armor class, hit points, initiative, saving throw, spell slot, cantrip, perception check, attack roll, natural 20, advantage, disadvantage, proficiency bonus, etc.

**Blades in the Dark** -- action roll, resistance roll, position, effect, stress, trauma, flashback, score, heist, downtime, engagement roll, fortune roll, devil's bargain, Doskvol, Duskwall, ghost field, electroplasm, Spirit Wardens, etc.

## VRAM usage

| GPU VRAM | Whisper model | Diarization |
|----------|--------------|-------------|
| < 2 GB | whisper-tiny | disabled |
| 2-4 GB | whisper-small | enabled |
| 4-6 GB | whisper-medium | enabled |
| 6-12 GB | whisper-large-v3-turbo | enabled |
| 12+ GB | whisper-large-v3 | enabled |

The app runs transcription and diarization at below-normal thread priority and configures CUDA to yield mode, so it won't interfere with games or Discord running on the same GPU.

## Output

Recordings are saved to `transcripts/`:

- **WAV** -- `session_YYYY-MM-DD_HHMM.wav` -- stereo (left = system audio, right = mic with gain boost)
- **Transcript** -- `session_YYYY-MM-DD_HHMM.txt` -- timestamped text with speaker labels

Example transcript:
```
TTRPG Listen - Session Transcript
Date: 2026-04-14 21:13
Game: Dungeons & Dragons
==================================================

[00:00] SPEAKER_03: We're going to go back downstairs.

[00:02] SPEAKER_03: Is anybody reacting in this moment?

[00:06] SPEAKER_00: I'm still holding my cocktail.

[00:08] SPEAKER_02: Push past this nun, because hell no.

[00:12] Microphone: Hey, do you guys think that we should?
```

## Project structure

```
src/ttrpglisten/
  app.py                 -- QApplication, dark Fusion theme, entry point
  main_window.py         -- 3-panel layout, worker lifecycle, signal wiring
  panels/
    quick_caption.py     -- Live Captions history with retroactive corrections
    accurate_caption.py  -- Color-coded speaker-attributed transcription
    controls.py          -- Buttons, device selectors, equalizers, status log
  widgets/
    equalizer.py         -- Real-time FFT bar visualizer (16 bands, 30fps)
  workers/
    live_caption.py      -- Windows UI Automation Live Captions reader
    audio_capture.py     -- WASAPI loopback + sounddevice mic capture
    transcription.py     -- whisperx transcription + forced alignment
    diarization.py       -- pyannote diarization + speaker assignment
  audio/
    devices.py           -- WASAPI device enumeration
    recorder.py          -- SharedAudioBuffer + crash-safe WAV writer
  models/
    selector.py          -- VRAM detection, model selection
  utils/
    config.py            -- QSettings persistence, game system prompts
```

## License

MIT
