"""Operating-system text-to-speech via pyttsx3 (SAPI5 / NSSpeech / espeak).

The fallback when no Piper voice exists for a language. Quality is mediocre but
it needs no download and works offline on every desktop OS.

Everything is wrapped in a timeout. pyttsx3 drives a platform speech engine
through a blocking event loop, and on a headless or remote session that loop
can block *forever* -- observed on a Windows machine with no audio endpoint,
where it hung the whole pipeline. A speech backend that cannot speak must
degrade to subtitles, not deadlock the application.
"""

from __future__ import annotations

import logging
import queue
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np

from ..config import TTSSettings
from ..languages import flores_to_whisper
from .base import SpeechAudio, TTSBackend, TTSUnavailableError

logger = logging.getLogger(__name__)

__all__ = ["SystemTTS"]

T = TypeVar("T")

#: Seconds before a platform speech call is declared hung.
DEFAULT_TIMEOUT = 15.0


def _with_timeout(func: Callable[[], T], timeout: float, what: str) -> T:
    """Run ``func`` on a daemon thread, raising if it outlives ``timeout``.

    The thread is deliberately abandoned rather than killed -- Python cannot
    interrupt a blocking C call -- but as a daemon it will not keep the process
    alive at exit.
    """
    box: queue.Queue = queue.Queue(maxsize=1)

    def runner() -> None:
        try:
            box.put(("ok", func()))
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
            box.put(("error", exc))

    thread = threading.Thread(target=runner, name=f"system-tts-{what}", daemon=True)
    thread.start()
    try:
        status, payload = box.get(timeout=timeout)
    except queue.Empty:
        raise TTSUnavailableError(
            f"The system speech engine hung during {what} (>{timeout:.0f}s). "
            "This usually means there is no audio output device. "
            "Use --tts piper or --tts none."
        ) from None
    if status == "error":
        raise TTSUnavailableError(f"The system speech engine failed during {what}: {payload}")
    return payload


class SystemTTS(TTSBackend):
    """pyttsx3 backend. Synthesises to a temp WAV and reads it back."""

    name = "system"

    def __init__(
        self,
        target_language: str = "eng_Latn",
        settings: TTSSettings | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.target_language = target_language
        self.settings = settings or TTSSettings()
        self.timeout = timeout
        self._engine = None
        self._voice_name = self.settings.voice or ""
        self._sample_rate = 22_050

    @staticmethod
    def is_installed() -> bool:
        try:
            import pyttsx3  # noqa: F401
        except Exception:
            return False
        return True

    @classmethod
    def supports(cls, target_language: str, settings: TTSSettings | None = None) -> bool:
        return cls.is_installed()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def voice(self) -> str:
        return self._voice_name or "system default"

    def load(self) -> SystemTTS:
        if self._engine is not None:
            return self
        try:
            import pyttsx3  # noqa: F401
        except Exception as exc:
            raise TTSUnavailableError(
                "pyttsx3 is not installed. Run: pip install pyttsx3"
            ) from exc

        # Prove the engine can actually produce audio before the caller relies
        # on it, so a broken engine falls back at set-up rather than mid-demo.
        audio, rate = self._synthesize_to_array("ok")
        self._sample_rate = rate
        self._engine = True
        return self

    def _configure(self, engine) -> None:  # noqa: ANN001
        try:
            engine.setProperty("rate", int(175 / max(self.settings.length_scale, 0.1)))
        except Exception:
            logger.debug("Could not set the speech rate", exc_info=True)
        self._select_voice(engine)

    def _select_voice(self, engine) -> None:  # noqa: ANN001
        """Pick an installed OS voice matching the target language, if any."""
        if self.settings.voice:
            try:
                engine.setProperty("voice", self.settings.voice)
                self._voice_name = self.settings.voice
            except Exception:
                logger.warning("System voice %r not available", self.settings.voice)
            return
        iso = flores_to_whisper(self.target_language)
        if not iso:
            return
        try:
            voices = engine.getProperty("voices")
        except Exception:
            return
        for candidate in voices or []:
            languages = [
                lang.decode("utf-8", "ignore") if isinstance(lang, bytes) else str(lang)
                for lang in (getattr(candidate, "languages", None) or [])
            ]
            haystack = " ".join(
                [
                    str(getattr(candidate, "id", "")),
                    str(getattr(candidate, "name", "")),
                    *languages,
                ]
            ).lower()
            if f"{iso}-" in haystack or f"{iso}_" in haystack or haystack.endswith(iso):
                try:
                    engine.setProperty("voice", candidate.id)
                    self._voice_name = str(getattr(candidate, "name", candidate.id))
                except Exception:
                    logger.debug("Could not set system voice", exc_info=True)
                return
        logger.warning(
            "No system voice installed for %s; output will use the default voice "
            "and may be mispronounced.",
            self.target_language,
        )

    def _synthesize_to_array(self, text: str) -> tuple[np.ndarray, int]:
        from ..audio.wav import read_wav

        def work() -> tuple[np.ndarray, int]:
            import pyttsx3

            # A fresh engine per call. pyttsx3's run loop cannot reliably be
            # driven twice -- the second runAndWait() blocks forever on some
            # platforms -- and a hung loop is exactly what this class exists
            # to avoid.
            engine = pyttsx3.init()
            try:
                self._configure(engine)
                with tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp) / "speech.wav"
                    engine.save_to_file(text, str(out))
                    engine.runAndWait()
                    if not out.exists() or out.stat().st_size == 0:
                        raise RuntimeError("the engine produced no audio")
                    return read_wav(out, target_rate=None)
            finally:
                try:
                    engine.stop()
                except Exception:  # pragma: no cover
                    pass

        return _with_timeout(work, self.timeout, "synthesis")

    def synthesize(self, text: str) -> SpeechAudio:
        text = (text or "").strip()
        if not text:
            return SpeechAudio(np.zeros(0, dtype=np.float32), self._sample_rate, self.voice)
        self.load()
        audio, rate = self._synthesize_to_array(text)
        self._sample_rate = rate
        return SpeechAudio(audio=audio, sample_rate=rate, voice=self.voice)

    def close(self) -> None:
        self._engine = None
