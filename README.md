# ttrpg-listen

Local AI-powered live transcription for online TTRPG sessions. Runs entirely on your machine -- no cloud services, no data leaves your computer.

Built for players who need live captions or want a searchable record of their sessions across any voice client (Discord, Foundry VTT, Roll20, etc.).

## What it does

- **Live captions** in your terminal as people speak (~100ms latency with Moonshine streaming models)
- **Dual audio capture** -- records both system audio (other players) and your microphone simultaneously
- **Post-session transcript** -- after you stop, re-processes the full recording with a larger model for higher accuracy
- **Speaker diarization** (optional) -- labels who said what in the final transcript using pyannote-audio
- **Hardware flexible** -- auto-detects CUDA, Metal, or falls back to CPU

## Quick start

**Requirements:** Python 3.11+, a microphone, and system audio output (speakers or headphones)

```bash
# Clone and install
git clone https://github.com/latentnyc/ttrpg-listen.git
cd ttrpg-listen
pip install -e .

# For CUDA (NVIDIA GPU) -- install the CUDA PyTorch build
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# For speaker diarization (optional, pulls in ~2GB of models)
pip install -e ".[diarization]"
```

```bash
# List your audio devices to find the right IDs
ttrpglisten --list-devices

# Start transcribing (auto-detects loopback + mic)
ttrpglisten

# Or specify devices explicitly
ttrpglisten --loopback-device 10 --mic-device 1
```

Press **Ctrl+C** to stop. The tool will then run a quality pass over the full recording and save a transcript to `./transcripts/`.

## How it works

```
System Audio (loopback) ──┐
                          ├── mix ──> Silero VAD ──> Moonshine ASR ──> Live Terminal
Microphone ───────────────┘                                              Display
         │
         └── WAV recording ──> (on Ctrl+C) ──> Larger Model + Diarization ──> Transcript File
```

**Pass 1 (live):** A small Moonshine streaming model transcribes speech in real-time as you play. Both audio sources are mixed for transcription and recorded to a stereo WAV (left = system audio, right = mic).

**Pass 2 (post-session):** A larger model re-transcribes the full recording for better accuracy. If diarization is enabled, pyannote-audio identifies individual speakers and labels the transcript.

## Configuration

Copy `config.example.yaml` to `config.yaml` to customize:

```yaml
audio:
  loopback_device: null   # null = auto-detect
  mic_device: null
  sample_rate: 16000

streaming:
  model: "usefulsensors/moonshine-streaming-tiny"  # tiny/small/medium
  device: "auto"   # auto, cpu, cuda, mps

postprocess:
  model: "usefulsensors/moonshine-streaming-medium"
  diarization: true
  min_speakers: 2
  max_speakers: 8

output:
  directory: "./transcripts"
```

Or use presets from the command line:

| Preset | Live Model | Post Model | Diarization | Best for |
|--------|-----------|------------|-------------|----------|
| `--preset low` | tiny | small | no | Older hardware, CPU-only |
| `--preset medium` | tiny | medium | yes | Default, balanced |
| `--preset high` | small | medium | yes | Good GPU, best quality |

## CLI reference

```
ttrpglisten [options]

  --list-devices              Show available audio devices
  --loopback-device ID        System audio device (remote players)
  --mic-device ID             Microphone device (your voice)
  --config PATH               Path to config.yaml
  --preset {low,medium,high}  Quality preset
  --no-postprocess            Skip the quality pass after stopping
  --device {auto,cpu,cuda,mps}  Force a compute device
```

## Models

Uses [Moonshine](https://github.com/usefulsensors/moonshine) streaming models by Useful Sensors:

| Model | Parameters | Speed | Use case |
|-------|-----------|-------|----------|
| moonshine-streaming-tiny | 34M | Fastest | Live captions on any hardware |
| moonshine-streaming-small | 123M | Balanced | Live captions with better accuracy |
| moonshine-streaming-medium | 245M | Best quality | Post-session transcription |

Models are downloaded automatically from HuggingFace on first run (~50MB-500MB depending on size).

## License

MIT
