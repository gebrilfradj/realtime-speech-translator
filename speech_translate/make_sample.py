"""Generate sample audio for testing and benchmarking.

Dogfoods the project's own TTS layer, so a sample can be produced in any of the
48 languages that have a voice -- useful for checking that ``--src auto``
really detects the language it claims to.

    python -m speech_translate.make_sample --lang en --out sample.wav
    python -m speech_translate.make_sample --lang es --text "Hola, buenos dias."
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .config import TTSSettings
from .languages import UnsupportedLanguageError, language_name, resolve_language

__all__ = ["create_sample", "main"]

# Multi-sentence on purpose: it exercises the sentence buffer and gives the
# benchmark something more representative than a single short phrase.
DEFAULT_TEXTS: dict[str, str] = {
    "eng_Latn": (
        "Hello, how are you today? "
        "This is a test of the real time speech translation system."
    ),
    "spa_Latn": "Hola, ¿cómo estás hoy? Esta es una prueba del sistema de traducción.",
    "fra_Latn": "Bonjour, comment allez-vous aujourd'hui? Ceci est un test du système.",
    "deu_Latn": "Hallo, wie geht es dir heute? Dies ist ein Test des Übersetzungssystems.",
    "ita_Latn": "Ciao, come stai oggi? Questo è un test del sistema di traduzione.",
    "por_Latn": "Olá, como você está hoje? Este é um teste do sistema de tradução.",
}


def create_sample(
    out_path: str = "sample.wav",
    language: str = "eng_Latn",
    text: str | None = None,
    backend: str = "auto",
    voice: str | None = None,
) -> str:
    """Synthesise ``text`` in ``language`` and write it to ``out_path``."""
    from .tts import create_tts_backend

    language = resolve_language(language)
    text = text or DEFAULT_TEXTS.get(language) or DEFAULT_TEXTS["eng_Latn"]

    engine = create_tts_backend(language, TTSSettings(backend=backend, voice=voice))
    speech = engine.synthesize(text)
    if not speech:
        raise RuntimeError(
            f"No speech backend could synthesise {language_name(language)}. "
            "Install piper-tts or pyttsx3."
        )
    speech.save(out_path)
    engine.close()
    print(
        f"Wrote {out_path}: {speech.duration:.1f}s of {language_name(language)} "
        f"at {speech.sample_rate} Hz (voice: {speech.voice or engine.name})"
    )
    print(f'  text: "{text}"')
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="speech-translate-sample",
        description="Generate a sample WAV file for testing the pipeline.",
    )
    parser.add_argument("--out", default="sample.wav", help="Output path (default: sample.wav).")
    parser.add_argument("--lang", default="en", help="Language of the sample (default: en).")
    parser.add_argument("--text", default=None, help="Text to speak (default: a built-in phrase).")
    parser.add_argument(
        "--tts",
        default="auto",
        choices=("auto", "piper", "system"),
        help="Speech backend to synthesise with.",
    )
    parser.add_argument("--voice", default=None, help="Explicit voice name.")
    args = parser.parse_args(argv)

    try:
        create_sample(args.out, args.lang, args.text, args.tts, args.voice)
    except UnsupportedLanguageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
