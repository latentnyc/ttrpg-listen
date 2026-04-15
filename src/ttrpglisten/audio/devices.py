"""Audio device enumeration for mic and WASAPI loopback."""

from __future__ import annotations

import sounddevice as sd


def enumerate_input_devices() -> list[dict]:
    """List available WASAPI microphone/input devices only."""
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            api_name = sd.query_hostapis(dev["hostapi"])["name"]
            if "WASAPI" not in api_name:
                continue
            if "loopback" in dev["name"].lower():
                continue
            result.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "rate": int(dev["default_samplerate"]),
            })
    return result


def enumerate_loopback_devices() -> list[dict]:
    """List available WASAPI loopback devices via pyaudiowpatch."""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return []

    result = []
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice"):
                result.append({
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["maxInputChannels"],
                    "rate": int(dev["defaultSampleRate"]),
                })
    finally:
        p.terminate()
    return result


def find_default_loopback() -> dict | None:
    """Find the WASAPI loopback device for the default output."""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return None

    p = pyaudio.PyAudio()
    try:
        wasapi_info = None
        for i in range(p.get_host_api_count()):
            info = p.get_host_api_info_by_index(i)
            if "WASAPI" in info["name"]:
                wasapi_info = info
                break
        if wasapi_info is None:
            return None

        default_output = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        output_name = default_output["name"]

        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice") and output_name in dev["name"]:
                return {
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["maxInputChannels"],
                    "rate": int(dev["defaultSampleRate"]),
                }
        return None
    finally:
        p.terminate()


def find_default_mic() -> int | None:
    """Find the default WASAPI microphone device index."""
    try:
        default = sd.query_devices(kind="input")
        default_name = default["name"]
        devices = sd.query_devices()
        # Find the WASAPI version of the default device
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                api_name = sd.query_hostapis(dev["hostapi"])["name"]
                if "WASAPI" in api_name and default_name in dev["name"]:
                    return i
        # Fallback: any WASAPI input device
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                api_name = sd.query_hostapis(dev["hostapi"])["name"]
                if "WASAPI" in api_name and "loopback" not in dev["name"].lower():
                    return i
    except sd.PortAudioError:
        pass
    return None
