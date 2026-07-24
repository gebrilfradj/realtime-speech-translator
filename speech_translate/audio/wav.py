"""WAV I/O and resampling.

Uses ``soundfile`` rather than ``pydub`` so there is no hidden dependency on an
ffmpeg binary being on PATH, and audio stays as float32 numpy the whole way
through the pipeline instead of being round-tripped via temp files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["read_wav", "write_wav", "resample", "to_mono", "rms", "float_to_int16"]


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse an (n, channels) array to mono; pass 1-D arrays through."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample. Adequate for 48k -> 16k speech; no SciPy required."""
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[0] / src_rate
    dst_len = int(round(duration * dst_rate))
    if dst_len <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0.0, audio.shape[0] - 1, num=dst_len, dtype=np.float64)
    return np.interp(src_idx, np.arange(audio.shape[0]), audio).astype(np.float32)


def read_wav(path: str | Path, target_rate: int | None = 16_000) -> tuple[np.ndarray, int]:
    """Read any soundfile-supported audio as mono float32.

    Returns:
        ``(audio, sample_rate)`` where ``sample_rate`` is ``target_rate`` when
        one was requested.
    """
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = to_mono(np.asarray(audio, dtype=np.float32))
    if target_rate and rate != target_rate:
        audio = resample(audio, rate, target_rate)
        rate = target_rate
    return audio, rate


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int = 16_000) -> Path:
    """Write mono float32 audio to a 16-bit PCM WAV file."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
    return path


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16 with clipping."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def int16_to_float(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32) / 32768.0


def rms(audio: np.ndarray) -> float:
    """Root-mean-square level; the cheap energy signal the VAD gate uses."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
