# ttrpg-listen

Local AI-powered live transcription for online TTRPG sessions. Runs entirely on your machine -- no cloud services, no data leaves your computer.

Built for players who need live captions or want a searchable record of their sessions across any voice client (Discord, Foundry VTT, Roll20, etc.).

## What it does

- **Live streaming captions** -- text updates in real time as people speak using a growing-window transcription approach with [Moonshine](https://github.com/usefulsensors/moonshine) models
- **Dual audio capture** -- WASAPI loopback (system audio from any output device including Bluetooth) + your microphone
- **Post-session transcript** -- after you stop, re-processes the full recording with [Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) for significantly higher accuracy
- **Speaker diarization** (optional) -- labels who said what in the final transcript using pyannote-audio, with automatic name inference from self-introductions
- **TTRPG vocabulary boost** -- Whisper post-processing uses a domain-specific prompt to improve recognition of D&D terms (armor class, spell slot, initiative, etc.)
- **Quality presets** -- low/medium/high presets to balance speed vs. accuracy for both live and post-processing

## Quick start

**Requirements:** Python 3.11+, Windows (WASAPI loopback), a microphone

```bash
# Clone and install
git clone https://github.com/latentnyc/ttrpg-listen.git
cd ttrpg-listen
pip install -e .

# For CUDA (NVIDIA GPU) -- recommended for Whisper post-processing
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# For speaker diarization (optional, pulls in ~2GB of models)
pip install -e ".[diarization]"
```

On first run, models download automatically from HuggingFace (Moonshine ~30MB for live, Whisper Turbo ~800MB for post-processing).

```bash
# List your audio devices
ttrpglisten --list-devices

# Start transcribing (auto-detects system audio loopback + mic)
ttrpglisten

# Or specify devices explicitly
ttrpglisten --loopback-device 10 --mic-device 1

# Use a quality preset
ttrpglisten --preset high
```

Press **Ctrl+C** to stop. The tool will then run a quality pass over the full recording and save a transcript to `./transcripts/`. Press **Ctrl+C** again to skip post-processing and exit immediately.

## How it works

```
System Audio (WASAPI loopback) --+
                                 +-- normalize + mix --> Moonshine -------> Live Terminal
Microphone ----------------------+     (RMS balancing)   (growing-window    Display
         |                                                transcription)   (Rich panel)
         |
         +-- stereo WAV ---------> (on Ctrl+C) --> Whisper Turbo + Diarization --> Transcript
             (left=system,                          (30s chunked       (pyannote)
              right=mic)                             re-transcription)
```

**Live stream:** Audio from both sources is normalized to equal RMS and mixed. Moonshine processes audio with an expanding window (1s, 2s, 3s... up to 10s), re-transcribing at each tick so accuracy improves as more context arrives. The display updates the current line in place with partial results, then commits the line and starts a new one when the window fills.

**Post-session:** Whisper Large v3 Turbo re-transcribes the full WAV recording in 30-second chunks (with 2s overlap) for much better accuracy (~2.5% WER vs ~6.65% for Moonshine). A TTRPG-specific vocabulary prompt helps Whisper correctly recognize game terms. If diarization is enabled, pyannote-audio identifies individual speakers and the tool scans for self-introductions to map speaker labels to names.

## Audio capture

The tool captures two audio sources simultaneously:

- **System loopback** via [pyaudiowpatch](https://github.com/s0d3s/PyAudioWPatch) -- captures from whatever output device Windows is using (speakers, Bluetooth headphones, USB audio, HDMI). This is how you hear remote players.
- **Microphone** via [sounddevice](https://python-sounddevice.readthedocs.io/) -- captures your voice.

Both are resampled to 16kHz and normalized to equal RMS before mixing, so a quiet mic isn't drowned out by louder system audio. Audio is recorded to a stereo WAV file for post-session processing.

## Configuration

Copy `config.example.yaml` to `config.yaml` and adjust as needed:

```yaml
audio:
  loopback_device: null   # null = auto-detect, or device ID
  mic_device: null
  sample_rate: 16000

streaming:
  model: "usefulsensors/moonshine-streaming-small"
  device: "auto"   # auto, cpu, cuda, mps

postprocess:
  model: "openai/whisper-large-v3-turbo"
  language: "en"
  diarization: true
  min_speakers: 2
  max_speakers: 8

output:
  directory: "./transcripts"
```

### Presets

| Preset | Live model | Post-processing model | Diarization |
|--------|-----------|----------------------|-------------|
| `low` | moonshine-streaming-tiny | whisper-large-v3-turbo | off |
| `medium` (default) | moonshine-streaming-small | whisper-large-v3-turbo | on |
| `high` | moonshine-streaming-medium | whisper-large-v3-turbo | on |

## CLI reference

```
ttrpglisten [options]

  --list-devices              Show available audio devices
  --loopback-device ID        System audio device (remote players)
  --mic-device ID             Microphone device (your voice)
  --config PATH               Path to config.yaml
  --preset {low,medium,high}  Quality preset
  --no-postprocess            Skip the quality pass after stopping
  --device {auto,cpu,cuda,mps}  Compute device
```

## Models

**Live streaming:** [Moonshine](https://github.com/usefulsensors/moonshine) streaming models via HuggingFace transformers. Optimized for low-latency on-device transcription:

| Model | Params | Use case |
|-------|--------|----------|
| `usefulsensors/moonshine-streaming-tiny` | Smallest | Fastest live captions, lower accuracy |
| `usefulsensors/moonshine-streaming-small` | 49M | Default for live -- good balance of speed and accuracy |
| `usefulsensors/moonshine-streaming-medium` | 245M | Best live accuracy, higher latency |

**Post-session:** [Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (809M params) -- OpenAI's fastest large Whisper variant with ~2.5% word error rate. Processes audio in 30-second chunks with a TTRPG vocabulary prompt for improved domain recognition. Uses CUDA or Metal acceleration if available, falls back to CPU automatically.

**Speaker diarization (optional):** [pyannote-audio](https://github.com/pyannote/pyannote-audio) speaker-diarization-3.1 for identifying who said what.

## Output

Transcripts are saved to the `transcripts/` directory:

- **WAV file:** `session_YYYY-MM-DD_HHMM.wav` -- stereo recording (left=system audio, right=mic)
- **Transcript:** `session_YYYY-MM-DD_HHMM.txt` -- timestamped text with optional speaker labels

Example transcript:
```
Session: 2026-03-21 13:55
Duration: 1h 30m
Model: openai/whisper-large-v3-turbo
Speakers: Tim, Sarah

---

[00:00:05] Tim: Can you hear me?

[00:00:08] Sarah: Yes, loud and clear.
```

## License

MIT
