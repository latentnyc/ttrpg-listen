# ttrpg-listen

Local AI-powered live transcription for online TTRPG sessions. Runs entirely on your machine -- no cloud services, no data leaves your computer.

Built for players who need live captions or want a searchable record of their sessions across any voice client (Discord, Foundry VTT, Roll20, etc.).

## What it does

- **True streaming captions** -- text appears word-by-word as people speak (~300ms updates via sherpa-onnx)
- **Dual audio capture** -- WASAPI loopback (system audio from any output device including Bluetooth) + your microphone
- **Post-session transcript** -- after you stop, re-processes the full recording with a larger model for higher accuracy
- **Speaker diarization** (optional) -- labels who said what in the final transcript using pyannote-audio
- **Lightweight** -- the streaming recognizer uses only ~5ms of CPU per 100ms of audio (20x faster than real-time)

## Quick start

**Requirements:** Python 3.11+, Windows (WASAPI loopback), a microphone

```bash
# Clone and install
git clone https://github.com/latentnyc/ttrpg-listen.git
cd ttrpg-listen
pip install -e .

# For CUDA (NVIDIA GPU) -- only needed for post-session quality pass
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# For speaker diarization (optional, pulls in ~2GB of models)
pip install -e ".[diarization]"
```

On first run, the streaming model (~30MB) downloads automatically.

```bash
# List your audio devices
ttrpglisten --list-devices

# Start transcribing (auto-detects system audio loopback + mic)
ttrpglisten

# Or specify devices explicitly
ttrpglisten --loopback-device 10 --mic-device 1
```

Press **Ctrl+C** to stop. The tool will then run a quality pass over the full recording and save a transcript to `./transcripts/`.

## How it works

```
System Audio (WASAPI loopback) ──┐
                                 ├── normalize + mix ──> sherpa-onnx ──> Live Terminal
Microphone ──────────────────────┘     (frame-by-frame     (streaming     Display
         │                              RMS balancing)      zipformer)   (word-by-word)
         │
         └── WAV recording ──> (on Ctrl+C) ──> Moonshine + Diarization ──> Transcript File
```

**Live stream:** Audio from both sources is normalized to equal volume and mixed. sherpa-onnx processes audio frame-by-frame with a streaming zipformer transducer model -- text appears as words are recognized, not after sentences finish. Endpoint detection automatically separates utterances when speakers pause.

**Post-session:** A larger Moonshine model re-transcribes the full WAV recording for better accuracy. If diarization is enabled, pyannote-audio identifies individual speakers. The stereo WAV (left=system audio, right=mic) helps distinguish your voice from remote players.

## Audio capture

The tool captures two audio sources simultaneously:

- **System loopback** via [pyaudiowpatch](https://github.com/s0d3s/PyAudioWPatch) -- captures from whatever output device Windows is using (speakers, Bluetooth headphones, USB audio, HDMI). This is how you hear remote players.
- **Microphone** via [sounddevice](https://python-sounddevice.readthedocs.io/) -- captures your voice.

Both are resampled to 16kHz and normalized to equal RMS before mixing, so a quiet mic isn't drowned out by louder system audio.

## CLI reference

```
ttrpglisten [options]

  --list-devices              Show available audio devices
  --loopback-device ID        System audio device (remote players)
  --mic-device ID             Microphone device (your voice)
  --config PATH               Path to config.yaml
  --preset {low,medium,high}  Quality preset
  --no-postprocess            Skip the quality pass after stopping
  --device {auto,cpu,cuda,mps}  Compute device for post-processing
```

## Models

**Live streaming:** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with a zipformer transducer model (int8 quantized, ~30MB). Runs on CPU at 20x real-time speed. Downloaded automatically on first run.

**Post-session quality:** [Moonshine](https://github.com/usefulsensors/moonshine) streaming models via HuggingFace transformers. Uses CUDA/Metal if available for faster processing.

## License

MIT
