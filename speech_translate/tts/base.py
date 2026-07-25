"""The interface every TTS backend implements.

Kept deliberately small so a new engine -- XTTS v2 for voice cloning, an API
based voice, a stub in tests -- is a single class with two methods. Coqui TTS
being abandoned is exactly why this abstraction exists.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["SpeechAudio", "TTSBackend", "TTSUnavailableError"]


class TTSUnavailableError(RuntimeError):
    """Raised when a backend cannot serve a language or is not installed."""


@dataclass
class SpeechAudio:
    """Synthesised speech as float32 samples in [-1, 1]."""

    audio: np.ndarray
    sample_rate: int
    #: Backend-specific voice identifier actually used.
    voice: str = ""

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0

    def __bool__(self) -> bool:
        return bool(len(self.audio))

    def save(self, path: str | Path) -> Path:
        from ..audio.wav import write_wav

        return write_wav(path, self.audio, self.sample_rate)


class TTSBackend(abc.ABC):
    """Synthesises text into speech for a fixed target language."""

    #: Short identifier used by ``--tts`` and in logs.
    name: str = "base"

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""

    @property
    def voice(self) -> str:
        """Identifier of the loaded voice, for display."""
        return ""

    @abc.abstractmethod
    def synthesize(self, text: str) -> SpeechAudio:
        """Render ``text`` to audio. Empty text yields empty audio."""

    def load(self) -> TTSBackend:
        """Eagerly initialise. Default is a no-op for lazy backends."""
        return self

    def close(self) -> None:  # noqa: B027 - optional hook, not every backend holds resources
        """Release resources. Default is a no-op."""

    def __enter__(self) -> TTSBackend:
        return self.load()

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class NullTTS(TTSBackend):
    """Subtitles-only mode: translate and display, never speak.

    Also the safety net when no voice exists for the target language, so a
    missing voice degrades to text instead of crashing the pipeline.
    """

    name = "none"

    def __init__(self, sample_rate: int = 22_050, reason: str = "") -> None:
        self._sample_rate = sample_rate
        self.reason = reason

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def voice(self) -> str:
        return "none (subtitles only)"

    def synthesize(self, text: str) -> SpeechAudio:
        return SpeechAudio(audio=np.zeros(0, dtype=np.float32), sample_rate=self._sample_rate)
