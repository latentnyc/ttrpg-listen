# TTRPG Listen

Real-time AI-powered transcription for tabletop roleplaying sessions. Dark-mode desktop GUI for **macOS** and **Windows 11** that captures system audio and microphone, transcribes with speaker identification, and keeps a full session history.

Runs entirely on your machine using your local GPU (CUDA) or Apple Silicon (Metal / CPU). No cloud services, no data leaves your computer.

## What it does

**Accurate Caption panel** — high-quality transcription using [whisperx](https://github.com/m-bain/whisperX) (Whisper + wav2vec2 forced alignment) with color-coded speaker identification via [pyannote-audio](https://github.com/pyannote/pyannote-audio). Processes audio in 30-second chunks with overlapping 60-second diarization windows to correct speaker assignments at chunk boundaries.

**Quick Caption panel (Windows only)** — reads Windows 11 Live Captions in real time, keeps the full session history that Live Captions normally discards. Handles retroactive corrections from the speech recognizer without duplicating text. Sits next to the Accurate Caption panel in a togglable horizontal/vertical splitter. Not available on macOS because there is no equivalent system API.

**Controls panel** — start/stop recording, select game system (D&D or Blades in the Dark vocabulary prompts), choose audio devices, real-time FFT equalizer displays for mic and system audio, scrollable status log, and a **Mic Settings** button for gain/sensitivity tuning with a live meter + Discord-style playback test.

**Key features:**

- Dual audio capture: system audio + microphone as separate channels
- Mic channel transcribed independently and attributed as "Microphone" (no diarization needed for the local player)
- System audio channel transcribed and speaker-diarized to identify remote players
- Compute-aware model selection (tiny through large-v3 based on available CUDA VRAM or Apple Silicon unified memory)
- Crash-safe WAV recording with periodic header updates
- Game-system-specific vocabulary prompts for better recognition of TTRPG terms
- Low-priority GPU/CPU scheduling so the app doesn't interfere with gameplay

## Requirements

- **Python 3.11+**
- **HuggingFace account** with accepted terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

### macOS

- **macOS 12+** (Apple Silicon recommended — transcription runs on CPU via CTranslate2, alignment + diarization run on MPS)
- **[BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole)** to capture system audio (`brew install blackhole-2ch`). BlackHole is a virtual audio driver — route your system output through it via **Audio MIDI Setup → Multi-Output Device** (built-in speakers + BlackHole 2ch) so you hear audio *and* the app can capture it. Then select `BlackHole 2ch` as "System Audio" in the Controls panel.

### Windows 11

- **NVIDIA GPU** with 6+ GB VRAM recommended (works on CPU but much slower)
  - GTX 1070 (8 GB): whisper-large-v3-turbo
  - RTX 3090 (24 GB): whisper-large-v3
- System audio captured via WASAPI loopback — no extra setup required.
- **Live Captions** (Quick Caption panel): enable via **Win+Ctrl+L** or **Settings → Accessibility → Captions**. The app reads the Live Captions window via Windows UI Automation.

## Quick start

```bash
git clone https://github.com/latentnyc/ttrpg-listen.git
cd ttrpg-listen
```

Create a `.env` file with your HuggingFace token (required for speaker diarization):

```
HF_TOKEN=hf_your_token_here
```

### macOS

```bash
./ttrpg-start.sh
```

The script creates a virtual environment, installs PyTorch (macOS wheels include MPS), installs the app, and launches it. First run downloads ~3–4 GB of models (Whisper + wav2vec2 + pyannote) — be patient and make sure you're on a good network. To stop: close the window, press Ctrl+C in the terminal, or run `./ttrpg-stop.sh`.

#### One-time macOS audio setup

1. **Install BlackHole 2ch** — `brew install blackhole-2ch`, then **reboot** (the audio HAL only reloads drivers at boot).
2. **Open Audio MIDI Setup** (`/Applications/Utilities/Audio MIDI Setup.app`).
3. Click the **+** in the lower-left → **Create Multi-Output Device**. In the right-hand pane:
   - Check your speakers/headphones **first** (e.g. "MacBook Pro Speakers" or your USB/Bluetooth output) — the first checkbox is the master clock.
   - Check **BlackHole 2ch** second.
   - Optional: check **Drift Correction** on BlackHole 2ch if the first-listed device is USB/Bluetooth.
4. Rename the Multi-Output Device to something like "Game + BlackHole" so it's obvious later.
5. **Set it as your system output** (Apple menu → System Settings → Sound → Output), or right-click it in Audio MIDI Setup → **Use This Device For Sound Output**.
6. In TTRPG Listen's Controls panel, pick **BlackHole 2ch** as "System Audio" (it should auto-select on first launch when detected).

Result: whatever is playing on your Mac routes through *both* your speakers (so you hear it) *and* BlackHole (so TTRPG Listen can capture it).

#### macOS caveats

- **Apple Silicon recommended.** Intel Macs work but will be slow — CTranslate2 falls back to AVX on CPU, and MPS isn't available, so everything runs on CPU. Prefer the `small` or `tiny` Whisper model manually if needed.
- **Transcription runs on CPU even on Apple Silicon.** whisperx uses CTranslate2 which has no MPS backend. On M1/M2/M3 the CPU int8 path is fast enough for `large-v3-turbo` at 30s chunks; alignment and diarization still run on MPS.
- **First run is slow.** The initial model downloads happen on the first recording start, not on app launch. Expect a ~1–2 minute stall on the first Start Recording while HuggingFace pulls the pyannote + wav2vec2 weights into `~/.cache/huggingface`.
- **The Mic Settings playback test** uses the system default output. If BlackHole is set as the *default* output (without a Multi-Output Device), you won't hear the playback — it'll be routed silently into BlackHole. Use the Multi-Output Device from the setup above as your default, or temporarily switch back to speakers for the test.
- **Microphone permission**: on first launch macOS will prompt "TTRPG Listen would like to access the microphone." You'll also get a second prompt for the BlackHole input (system audio counts as a separate mic permission). Grant both. If you accidentally denied them, toggle in **System Settings → Privacy & Security → Microphone** and the Terminal/Python that launched the app must be listed.
- **Some pyannote ops silently fall back to CPU on MPS.** You'll see a status-log message like "MPS pipeline unavailable … falling back to CPU" if that happens — diarization still works, just slower. This is normal on some macOS/PyTorch version combos.
- **No Live Captions panel on Mac.** macOS has nothing equivalent to Windows 11's Live Captions, so the left-hand "Quick Caption" panel doesn't exist on Mac. The Accurate Caption panel (whisperx) is still fully functional.
- **Discord / game audio**: you can still use Discord on macOS while this runs. Set Discord's output to the Multi-Output Device (or a separate headphone output) so party chat is routed into BlackHole and gets transcribed alongside the game.

### Windows

Double-click `ttrpg-start.bat` or run it from a terminal. It will:

1. Create a virtual environment
2. Install PyTorch with CUDA 12.6
3. Install all dependencies
4. Launch the app

To stop: close the window, press Ctrl+C, or run `ttrpg-stop.bat`.

### Manual install (any platform)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# macOS: plain PyTorch wheels include MPS support
pip install torch torchaudio

# Windows: PyTorch with CUDA (must be 2.8.x for whisperx compatibility)
# pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install the app
pip install -e .

# Run
python -m ttrpglisten
```

## Mic Settings

Click **Mic Settings...** next to the microphone dropdown to open a modal with:

- **Gain** slider (0.5× – 10×) — software boost applied after resampling, before the transcription pipeline
- **Sensitivity** slider (0.001 – 0.1 RMS) — noise floor below which mic audio is treated as silence
- **Live level meter** with a yellow threshold line — watch your voice cross the line to confirm sensitivity is tuned
- **Test Mic** button — records 3 seconds and plays it back through the default output so you can hear what the pipeline hears

Settings persist via `QSettings` and apply immediately even mid-recording.

> macOS note: if BlackHole is set as the *default* output, the playback test won't be audible. Use a Multi-Output Device (BlackHole + built-in speakers) as your default output, or temporarily switch back to speakers for the test.

## How it works

```
Windows Live Captions ----> Quick Caption panel    (Windows only)
  (UI Automation polling)     (full session history)

Microphone ------+                            +--> mic segments
                 v                            |    ("Microphone")
             AudioCaptureWorker --> SharedAudioBuffer
                 ^                            |
System Audio ----+                            +--> TranscriptionWorker
 (Windows: WASAPI loopback)                   |      (whisperx)
 (macOS:   BlackHole input)                   |            |
                                              +--> DiarizationWorker
                                                      (pyannote)
                                                          |
                                        speaker-attributed segments
                                                          |
                                                          v
                                                Accurate Caption panel
                                                (color-coded speakers)
```

### Transcription pipeline (per 30s chunk)

1. **whisperx.transcribe()** — VAD-segmented batched Whisper with game-specific vocabulary prompt
2. **whisperx.align()** — wav2vec2 forced alignment for precise word-level timestamps
3. **Loopback segments** displayed immediately with generic "Speaker" label
4. **Mic segments** displayed immediately as "Microphone"

### Diarization pipeline (60s overlapping windows)

1. **pyannote speaker-diarization-3.1** on the loopback audio
2. **whisperx.assign_word_speakers()** merges word timestamps with speaker segments
3. Retroactively updates "Speaker" labels to "SPEAKER_00", "SPEAKER_01", etc.
4. Overlapping windows cross-check speaker assignments at chunk boundaries

### Audio capture

- **Windows system loopback** via [pyaudiowpatch](https://github.com/s0d3s/PyAudioWPatch) — captures from whatever Windows output device is active (speakers, Bluetooth, USB, HDMI)
- **macOS system audio** via [BlackHole](https://github.com/ExistentialAudio/BlackHole) as a regular Core Audio input device (requires Multi-Output Device setup)
- **Microphone** via [sounddevice](https://python-sounddevice.readthedocs.io/) (WASAPI on Windows, Core Audio on macOS)
- Both resampled to 16kHz, mic multiplied by the user-configured gain
- Stereo WAV written with crash-safe periodic flushing (data every 5s, header every 30s)

## Game system prompts

The app uses domain-specific vocabulary prompts to improve Whisper's recognition of game terminology. Select the game system in the Controls panel before recording.

**Dungeons & Dragons** — d20, armor class, hit points, initiative, saving throw, spell slot, cantrip, perception check, attack roll, natural 20, advantage, disadvantage, proficiency bonus, etc.

**Blades in the Dark** — action roll, resistance roll, position, effect, stress, trauma, flashback, score, heist, downtime, engagement roll, fortune roll, devil's bargain, Doskvol, Duskwall, ghost field, electroplasm, Spirit Wardens, etc.

## Model sizing

| Host | Whisper model | Notes |
|------|---------------|-------|
| CUDA, 12+ GB VRAM | whisper-large-v3 | Diarization keeps both models loaded |
| CUDA, 6–12 GB VRAM | whisper-large-v3-turbo | Standard gaming-GPU config |
| CUDA, 4–6 GB VRAM | whisper-medium | Diarization enabled |
| CUDA, 2–4 GB VRAM | whisper-small | Diarization enabled |
| CUDA, < 2 GB VRAM | whisper-tiny | Diarization disabled |
| Apple Silicon, 16+ GB | whisper-large-v3-turbo | whisperx on CPU (int8); alignment + diarization on MPS |
| Apple Silicon, 12–16 GB | whisper-medium | " |
| Apple Silicon, 8–12 GB | whisper-small | " |
| CPU only | whisper-small / tiny | Slow; fine for testing |

Transcription and diarization run at below-normal thread priority (Windows) or increased nice value (macOS/Linux), and CUDA is configured to yield mode so the app doesn't interfere with games or Discord on the same GPU.

## Output

Recordings are saved to `transcripts/`:

- **WAV** — `session_YYYY-MM-DD_HHMM.wav` — stereo (left = system audio, right = mic with gain boost)
- **Transcript** — `session_YYYY-MM-DD_HHMM.txt` — timestamped text with speaker labels

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
  main_window.py         -- Platform-aware layout (Windows: 2 panels; macOS: 1), worker lifecycle
  panels/
    quick_caption.py     -- Live Captions history with retroactive corrections (Windows only)
    accurate_caption.py  -- Color-coded speaker-attributed transcription
    controls.py          -- Buttons, device selectors, equalizers, status log, Mic Settings button
  widgets/
    equalizer.py         -- Real-time FFT bar visualizer (16 bands, 30fps)
    mic_test.py          -- Discord-style mic test dialog (gain/sensitivity/playback)
  workers/
    live_caption.py      -- Windows UI Automation Live Captions reader (Windows only)
    audio_capture.py     -- Windows WASAPI loopback + sounddevice mic, or sounddevice-only (Mac)
    transcription.py     -- whisperx transcription + forced alignment
    diarization.py       -- pyannote diarization + speaker assignment
  audio/
    devices.py           -- Cross-platform device enumeration
    recorder.py          -- SharedAudioBuffer + crash-safe WAV writer
  models/
    selector.py          -- CUDA VRAM / Apple Silicon memory detection, model selection
  utils/
    config.py            -- QSettings persistence, game system prompts
    platform.py          -- Cross-platform low-priority helper
```

## License

MIT
