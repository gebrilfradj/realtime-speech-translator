"""Text-to-speech backends and the factory that picks one.

Adding a new engine (e.g. XTTS v2 for voice cloning) means writing a
:class:`~speech_translate.tts.base.TTSBackend` subclass and adding one line to
:func:`create_tts_backend` -- nothing else in the pipeline changes.
"""

from __future__ import annotations

import logging

from ..config import TTSSettings
from .base import NullTTS, SpeechAudio, TTSBackend, TTSUnavailableError
from .piper import PiperTTS
from .system import SystemTTS

logger = logging.getLogger(__name__)

__all__ = [
    "NullTTS",
    "PiperTTS",
    "SpeechAudio",
    "SystemTTS",
    "TTSBackend",
    "TTSUnavailableError",
    "create_tts_backend",
    "available_backends",
]

_BACKENDS: dict[str, type[TTSBackend]] = {
    "piper": PiperTTS,
    "system": SystemTTS,
}


def available_backends() -> list[str]:
    """Backend names usable on this machine right now."""
    names = ["none"]
    if PiperTTS.is_installed():
        names.insert(0, "piper")
    if SystemTTS.is_installed():
        names.insert(-1, "system")
    return names


def create_tts_backend(
    target_language: str, settings: TTSSettings | None = None
) -> TTSBackend:
    """Build a backend for ``target_language``.

    ``backend='auto'`` tries Piper, then the OS voice, then subtitles-only, so
    a missing voice degrades gracefully instead of taking down the pipeline.
    An explicitly requested backend that cannot serve the language raises,
    because silently substituting a different engine would be surprising.
    """
    settings = settings or TTSSettings()
    choice = settings.backend

    if choice == "none":
        return NullTTS(reason="requested")

    if choice != "auto":
        backend_cls = _BACKENDS.get(choice)
        if backend_cls is None:
            raise TTSUnavailableError(
                f"Unknown TTS backend {choice!r}. Available: {', '.join(available_backends())}"
            )
        return backend_cls(target_language, settings)

    for name in ("piper", "system"):
        backend_cls = _BACKENDS[name]
        if not backend_cls.supports(target_language, settings):  # type: ignore[attr-defined]
            continue
        backend = backend_cls(target_language, settings)
        try:
            return backend.load()
        except TTSUnavailableError as exc:
            logger.warning("TTS backend %s unavailable (%s); trying the next one.", name, exc)

    logger.warning(
        "No speech backend could serve %s - running in subtitles-only mode.",
        target_language,
    )
    return NullTTS(reason=f"no voice available for {target_language}")
