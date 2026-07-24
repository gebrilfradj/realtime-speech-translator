"""Piper text-to-speech.

Piper is a small ONNX VITS runtime: fast enough to synthesise faster than
real time on a CPU, actively maintained, and with 170 voices across 49
languages published in ``rhasspy/piper-voices``.

Voice selection order:

1. an explicit ``--tts-voice``,
2. the curated default for the target language,
3. any voice in the published catalogue for that language family.

Voices are downloaded on first use into a cache directory and reused after.
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path

import numpy as np

from ..config import TTSSettings
from ..languages import piper_voice_for
from .base import SpeechAudio, TTSBackend, TTSUnavailableError

logger = logging.getLogger(__name__)

__all__ = ["PiperTTS", "default_download_dir"]

_LANG_PREFIX = {
    "eng": "en", "spa": "es", "fra": "fr", "deu": "de", "ita": "it",
    "por": "pt", "rus": "ru", "zho": "zh", "nld": "nl", "pol": "pl",
    "tur": "tr", "swe": "sv", "ukr": "uk", "arb": "ar", "ces": "cs",
    "dan": "da", "ell": "el", "fin": "fi", "hun": "hu", "ron": "ro",
    "cat": "ca", "vie": "vi", "kat": "ka", "pes": "fa", "srp": "sr",
    "slk": "sk", "slv": "sl", "nob": "no", "isl": "is", "swh": "sw",
    "npi": "ne", "hin": "hi", "kaz": "kk", "lvs": "lv", "cym": "cy",
    "ben": "bn", "bul": "bg", "eus": "eu", "heb": "he", "hye": "hy",
    "ind": "id", "kor": "ko", "ltz": "lb", "mal": "ml", "mar": "mr",
    "tel": "te", "urd": "ur", "als": "sq",
}


def default_download_dir() -> Path:
    """Where downloaded voices live (``SPEECH_TRANSLATE_VOICES`` overrides)."""
    import os

    override = os.environ.get("SPEECH_TRANSLATE_VOICES")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "speech-translate" / "piper-voices"


class PiperTTS(TTSBackend):
    """Piper backend for one target language."""

    name = "piper"

    def __init__(self, target_language: str, settings: TTSSettings | None = None) -> None:
        self.target_language = target_language
        self.settings = settings or TTSSettings()
        self._voice_name = self.settings.voice or piper_voice_for(target_language)
        self._voice = None
        self._sample_rate = 22_050

    # -- discovery -------------------------------------------------------
    @staticmethod
    def is_installed() -> bool:
        try:
            import piper  # noqa: F401
        except Exception:
            return False
        return True

    @classmethod
    def supports(cls, target_language: str, settings: TTSSettings | None = None) -> bool:
        if not cls.is_installed():
            return False
        if settings is not None and settings.voice:
            return True
        return piper_voice_for(target_language) is not None

    @property
    def voice(self) -> str:
        return self._voice_name or ""

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    # -- loading ---------------------------------------------------------
    def _resolve_voice_name(self) -> str:
        if self._voice_name:
            return self._voice_name
        prefix = _LANG_PREFIX.get(self.target_language.split("_")[0])
        if prefix:
            catalogue = self._catalogue()
            candidates = [
                key
                for key, info in catalogue.items()
                if info.get("language", {}).get("family") == prefix
            ]
            if candidates:
                quality = {"medium": 0, "high": 1, "low": 2, "x_low": 3}
                candidates.sort(key=lambda k: quality.get(catalogue[k].get("quality"), 9))
                logger.info("No curated Piper voice for %s; using %s", self.target_language, candidates[0])
                return candidates[0]
        raise TTSUnavailableError(
            f"Piper has no voice for {self.target_language}. "
            "Use --tts system, --tts none (subtitles only), or pass --tts-voice."
        )

    @staticmethod
    def _catalogue() -> dict:
        try:
            from piper.download_voices import list_voices  # type: ignore

            return list_voices()
        except Exception:
            try:
                import json
                import urllib.request

                from piper.download_voices import VOICES_JSON  # type: ignore

                with urllib.request.urlopen(VOICES_JSON, timeout=20) as response:
                    return json.load(response)
            except Exception:
                logger.debug("Could not fetch the Piper voice catalogue", exc_info=True)
                return {}

    def load(self) -> PiperTTS:
        if self._voice is not None:
            return self
        try:
            from piper import PiperVoice
            from piper.download_voices import download_voice
        except Exception as exc:
            raise TTSUnavailableError(
                "piper-tts is not installed. Run: pip install piper-tts"
            ) from exc

        name = self._resolve_voice_name()
        download_dir = Path(self.settings.download_dir or default_download_dir())
        download_dir.mkdir(parents=True, exist_ok=True)
        model_path = download_dir / f"{name}.onnx"

        if not model_path.exists():
            logger.info("Downloading Piper voice %s ...", name)
            try:
                download_voice(name, download_dir)
            except Exception as exc:
                raise TTSUnavailableError(
                    f"Could not download the Piper voice {name!r}: {exc}"
                ) from exc

        try:
            self._voice = PiperVoice.load(model_path, use_cuda=self.settings.use_cuda)
        except Exception as exc:
            raise TTSUnavailableError(f"Could not load the Piper voice {name!r}: {exc}") from exc

        self._voice_name = name
        config = getattr(self._voice, "config", None)
        self._sample_rate = int(getattr(config, "sample_rate", 22_050) or 22_050)
        logger.info("Piper voice %s ready (%d Hz)", name, self._sample_rate)
        return self

    # -- synthesis -------------------------------------------------------
    def synthesize(self, text: str) -> SpeechAudio:
        text = (text or "").strip()
        if not text:
            return SpeechAudio(np.zeros(0, dtype=np.float32), self._sample_rate, self.voice)
        self.load()

        syn_config = self._synthesis_config()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        buffer.seek(0)
        with wave.open(buffer, "rb") as wav_file:
            rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        self._sample_rate = rate
        return SpeechAudio(audio=audio, sample_rate=rate, voice=self.voice)

    def _synthesis_config(self):
        try:
            from piper import SynthesisConfig

            return SynthesisConfig(length_scale=self.settings.length_scale)
        except Exception:
            return None
