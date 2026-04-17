"""Audio device enumeration for mic and system-audio loopback.

Platform notes:

- On Windows, system audio is captured via WASAPI loopback (pyaudiowpatch).
  The "loopback" list is a distinct set of devices that doesn't overlap with
  the regular input list.

- On macOS, there is no OS-level loopback API. System audio is captured by
  routing it through a virtual audio device like BlackHole 2ch, which then
  appears as a regular Core Audio input device. The "loopback" list on Mac
  is therefore the same device list as the mic list; the user picks the
  BlackHole entry manually.
"""

from __future__ import annotations

import sys

import sounddevice as sd

IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"


def _is_virtual_loopback_name(name: str) -> bool:
    """Names of macOS virtual audio drivers that typically represent loopback,
    so we can exclude them from the mic list and prefer them for loopback."""
    lname = name.lower()
    return (
        "blackhole" in lname
        or "loopback" in lname
        or "soundflower" in lname
        or "virtual audio" in lname
    )


def enumerate_input_devices() -> list[dict]:
    """List available microphone/input devices.

    Windows: WASAPI input devices (excluding loopback variants).
    macOS:   Core Audio input devices (excluding virtual loopback drivers).
    """
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        api_name = sd.query_hostapis(dev["hostapi"])["name"]
        if IS_WINDOWS:
            if "WASAPI" not in api_name:
                continue
            if "loopback" in dev["name"].lower():
                continue
        else:
            # On Mac, default to Core Audio; if the user has multiple hostapis,
            # sounddevice exposes them all — we prefer Core Audio.
            if IS_MAC and "Core Audio" not in api_name:
                continue
            if _is_virtual_loopback_name(dev["name"]):
                continue
        result.append({
            "index": i,
            "name": dev["name"],
            "channels": dev["max_input_channels"],
            "rate": int(dev["default_samplerate"]),
            "hostapi": api_name,
        })
    return result


def enumerate_loopback_devices() -> list[dict]:
    """List devices usable as a system-audio source.

    Windows: WASAPI loopback devices enumerated via pyaudiowpatch.
    macOS:   Core Audio input devices (which includes BlackHole / Loopback
             / Soundflower when installed). The user picks the virtual device.
    """
    if IS_WINDOWS:
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

    # macOS / other POSIX: all Core Audio input devices show up here.
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        api_name = sd.query_hostapis(dev["hostapi"])["name"]
        if IS_MAC and "Core Audio" not in api_name:
            continue
        result.append({
            "index": i,
            "name": dev["name"],
            "channels": dev["max_input_channels"],
            "rate": int(dev["default_samplerate"]),
            "hostapi": api_name,
        })
    return result


def find_default_loopback() -> dict | None:
    """Find the best default loopback source.

    Windows: the WASAPI loopback paired with the default output device.
    macOS:   the first device whose name starts with BlackHole (if installed).
    """
    if IS_WINDOWS:
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

    # macOS: prefer BlackHole, fall back to any other virtual loopback.
    devices = sd.query_devices()
    candidates: list[tuple[int, int, dict]] = []  # (priority, index, dict)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        lname = dev["name"].lower()
        priority = None
        if lname.startswith("blackhole"):
            priority = 0
        elif "loopback" in lname:
            priority = 1
        elif "soundflower" in lname:
            priority = 2
        if priority is None:
            continue
        candidates.append((priority, i, {
            "index": i,
            "name": dev["name"],
            "channels": dev["max_input_channels"],
            "rate": int(dev["default_samplerate"]),
        }))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][2]


def find_default_mic() -> int | None:
    """Find the default input device index.

    Windows: the WASAPI version of the system default input.
    macOS:   the Core Audio version of the system default input, skipping
             virtual loopback devices.
    """
    try:
        default = sd.query_devices(kind="input")
        default_name = default["name"]
        devices = sd.query_devices()

        if IS_WINDOWS:
            # Find the WASAPI version of the default device
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    api_name = sd.query_hostapis(dev["hostapi"])["name"]
                    if "WASAPI" in api_name and default_name in dev["name"]:
                        return i
            # Fallback: any WASAPI input device that isn't a loopback
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    api_name = sd.query_hostapis(dev["hostapi"])["name"]
                    if "WASAPI" in api_name and "loopback" not in dev["name"].lower():
                        return i
            return None

        # macOS / POSIX: prefer the Core Audio version of the default input.
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] <= 0:
                continue
            api_name = sd.query_hostapis(dev["hostapi"])["name"]
            if IS_MAC and "Core Audio" not in api_name:
                continue
            if _is_virtual_loopback_name(dev["name"]):
                continue
            if default_name in dev["name"]:
                return i
        # Fallback: first non-loopback Core Audio input
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] <= 0:
                continue
            api_name = sd.query_hostapis(dev["hostapi"])["name"]
            if IS_MAC and "Core Audio" not in api_name:
                continue
            if _is_virtual_loopback_name(dev["name"]):
                continue
            return i
    except sd.PortAudioError:
        pass
    return None
