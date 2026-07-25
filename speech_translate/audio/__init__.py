"""Audio capture, segmentation, playback and file I/O."""

from __future__ import annotations

from .devices import (
    AudioDevice,
    AudioUnavailableError,
    format_device_table,
    list_input_devices,
    list_output_devices,
    resolve_input_device,
)
from .vad import EnergyVAD, Utterance, UtteranceSegmenter, segment_audio
from .wav import read_wav, resample, rms, to_mono, write_wav

__all__ = [
    "AudioDevice",
    "AudioUnavailableError",
    "EnergyVAD",
    "Utterance",
    "UtteranceSegmenter",
    "format_device_table",
    "list_input_devices",
    "list_output_devices",
    "read_wav",
    "resample",
    "resolve_input_device",
    "rms",
    "segment_audio",
    "to_mono",
    "write_wav",
]
