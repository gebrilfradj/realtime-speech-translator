"""Shared fixtures. Nothing here downloads a model or opens a device."""

from __future__ import annotations

import numpy as np
import pytest

from speech_translate.asr import Transcript
from speech_translate.mt import Translation
from speech_translate.tts.base import SpeechAudio, TTSBackend


class FakeRecognizer:
    """Stand-in for faster-whisper that returns a scripted transcript."""

    def __init__(self, text: str = "Hello there.", language: str = "eng_Latn") -> None:
        self.text = text
        self.language = language
        self.calls: list[tuple[int, str]] = []

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> Transcript:
        self.calls.append((len(audio), language))
        return Transcript(
            text=self.text,
            language=self.language,
            duration=len(audio) / 16_000,
        )


class FakeTranslator:
    """Records the language pair it was asked for -- the point of most tests."""

    def __init__(self, output: str = "Hola.") -> None:
        self.output = output
        self.calls: list[tuple[str, str, str]] = []

    def translate(self, text: str, src: str, tgt: str) -> Translation:
        self.calls.append((text, src, tgt))
        return Translation(text=self.output, src=src, tgt=tgt)


class FakeTTS(TTSBackend):
    name = "fake"

    def __init__(self, sample_rate: int = 22_050) -> None:
        self._sample_rate = sample_rate
        self.calls: list[str] = []

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def voice(self) -> str:
        return "fake-voice"

    def synthesize(self, text: str) -> SpeechAudio:
        self.calls.append(text)
        samples = np.zeros(self._sample_rate // 10, dtype=np.float32)
        return SpeechAudio(audio=samples, sample_rate=self._sample_rate, voice=self.voice)


@pytest.fixture
def fake_recognizer() -> FakeRecognizer:
    return FakeRecognizer()


@pytest.fixture
def fake_translator() -> FakeTranslator:
    return FakeTranslator()


@pytest.fixture
def fake_tts() -> FakeTTS:
    return FakeTTS()


@pytest.fixture
def silence() -> np.ndarray:
    return np.zeros(16_000, dtype=np.float32)


def tone(seconds: float, sample_rate: int = 16_000, amplitude: float = 0.3) -> np.ndarray:
    """A loud-enough signal for the energy VAD to call it speech."""
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def speech_like(seconds: float, sample_rate: int = 16_000, amplitude: float = 0.4) -> np.ndarray:
    """Audio with a syllable-rate envelope, closer to real speech than a tone.

    A constant sine is *correctly* treated as background noise by a
    minimum-statistics VAD once it has seen a few seconds of it. Real speech
    has strongly varying energy, which is what keeps the noise-floor percentile
    well below the peaks, so tests covering multi-second speech use this.
    """
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    carrier = (
        np.sin(2 * np.pi * 140 * t)
        + 0.5 * np.sin(2 * np.pi * 280 * t)
        + 0.25 * np.sin(2 * np.pi * 560 * t)
    )
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)  # ~4 syllables/second
    return (amplitude * envelope * carrier / 1.75).astype(np.float32)


def quiet(seconds: float, sample_rate: int = 16_000) -> np.ndarray:
    return np.zeros(int(sample_rate * seconds), dtype=np.float32)
