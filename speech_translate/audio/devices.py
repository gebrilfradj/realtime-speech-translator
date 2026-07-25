"""Audio device discovery and selection.

Fixes the hardcoded ``MIC_INDEX = 1`` ("Realtek mic") that made the original
project work on exactly one laptop. ``sounddevice`` is imported lazily so the
package still imports on machines and CI runners with no audio stack at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "AudioDevice",
    "AudioUnavailableError",
    "list_input_devices",
    "list_output_devices",
    "resolve_input_device",
    "format_device_table",
]


class AudioUnavailableError(RuntimeError):
    """Raised when no usable audio backend or device exists."""


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    channels: int
    default_samplerate: float
    is_default: bool = False

    def __str__(self) -> str:
        marker = " (default)" if self.is_default else ""
        return f"[{self.index}] {self.name} - {self.channels}ch @ {self.default_samplerate:.0f} Hz{marker}"


def _sounddevice():
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing on the host
        raise AudioUnavailableError(
            "PortAudio is not available. Install it (Linux: 'apt-get install "
            "libportaudio2') or run with --no-audio / the web UI's file-upload mode."
        ) from exc
    except ImportError as exc:
        raise AudioUnavailableError(
            "The 'sounddevice' package is not installed. Run: pip install sounddevice"
        ) from exc
    return sd


def _default_indices() -> tuple[int | None, int | None]:
    sd = _sounddevice()
    try:
        default = sd.default.device
        return (
            default[0] if isinstance(default, (list, tuple)) else None,
            default[1] if isinstance(default, (list, tuple)) else None,
        )
    except Exception:
        return None, None


def _list_devices(kind: str) -> list[AudioDevice]:
    sd = _sounddevice()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    default_in, default_out = _default_indices()
    default_index = default_in if kind == "input" else default_out

    devices: list[AudioDevice] = []
    for index, info in enumerate(sd.query_devices()):
        channels = int(info.get(key, 0))
        if channels <= 0:
            continue
        devices.append(
            AudioDevice(
                index=index,
                name=str(info.get("name", f"device {index}")).strip(),
                channels=channels,
                default_samplerate=float(info.get("default_samplerate", 0) or 0),
                is_default=(index == default_index),
            )
        )
    return devices


def list_input_devices() -> list[AudioDevice]:
    """Every device that can capture audio."""
    return _list_devices("input")


def list_output_devices() -> list[AudioDevice]:
    """Every device that can play audio."""
    return _list_devices("output")


def resolve_input_device(requested: int | str | None) -> int | None:
    """Turn a user-supplied device index or name fragment into an index.

    ``None`` means "use the system default", which is what the OS mixer says
    and is nearly always what the user wants.
    """
    if requested is None:
        return None

    devices = list_input_devices()
    if not devices:
        raise AudioUnavailableError(
            "No input devices found. Check that a microphone is connected and "
            "that this application has microphone permission."
        )

    if isinstance(requested, int) or (isinstance(requested, str) and requested.lstrip("-").isdigit()):
        index = int(requested)
        valid = {d.index for d in devices}
        if index not in valid:
            raise AudioUnavailableError(
                f"Input device {index} does not exist or has no input channels.\n"
                f"{format_device_table(devices)}"
            )
        return index

    needle = str(requested).lower()
    matches = [d for d in devices if needle in d.name.lower()]
    if not matches:
        raise AudioUnavailableError(
            f"No input device matching {requested!r}.\n{format_device_table(devices)}"
        )
    if len(matches) > 1:
        logger.warning(
            "%d devices match %r; using %s", len(matches), requested, matches[0].name
        )
    return matches[0].index


def format_device_table(devices: list[AudioDevice] | None = None) -> str:
    """Printable listing for ``--list-devices``."""
    try:
        inputs = devices if devices is not None else list_input_devices()
    except AudioUnavailableError as exc:
        return str(exc)
    if not inputs:
        return "No input devices found."
    lines = ["Available input devices:"]
    lines.extend(f"  {device}" for device in inputs)
    lines.append("")
    lines.append("Select one with --input-device <index|name fragment>.")
    return "\n".join(lines)
