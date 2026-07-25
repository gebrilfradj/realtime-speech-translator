"""Real-time multilingual speech-to-speech translation.

A modular cascade of open-source models:

======  ==========================================  ===============================
Stage   Model                                       Why
======  ==========================================  ===============================
ASR     faster-whisper (CTranslate2)                Whisper accuracy, far less compute
MT      NLLB-200 distilled 600M                     200 languages, actively maintained
TTS     Piper (ONNX VITS)                           Faster than real time on CPU
======  ==========================================  ===============================

Typical use::

    from speech_translate import Pipeline, Settings

    pipeline = Pipeline(Settings(src="auto", tgt="spa_Latn"))
    result = pipeline.process_file("sample.wav")
    print(result.transcript, "->", result.translation)

Importing this package loads no models and opens no audio devices.
"""

from __future__ import annotations

__version__ = "2.0.0"

from .config import (
    ASRSettings,
    AudioSettings,
    MTSettings,
    Settings,
    TTSSettings,
    VADSettings,
)
from .languages import (
    UnsupportedLanguageError,
    language_name,
    languages_with_speech,
    resolve_language,
    supported_languages,
)
from .pipeline import Pipeline, PipelineResult, StageTimings
from .segmentation import SentenceBuffer

__all__ = [
    "ASRSettings",
    "AudioSettings",
    "MTSettings",
    "Pipeline",
    "PipelineResult",
    "SentenceBuffer",
    "Settings",
    "StageTimings",
    "TTSSettings",
    "UnsupportedLanguageError",
    "VADSettings",
    "__version__",
    "language_name",
    "languages_with_speech",
    "resolve_language",
    "supported_languages",
]


def __getattr__(name: str):
    """Expose the heavier entry points without importing them eagerly."""
    if name == "RealtimeSession":
        from .realtime import RealtimeSession

        return RealtimeSession
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
