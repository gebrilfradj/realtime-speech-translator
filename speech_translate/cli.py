"""Command-line interface.

Argument parsing happens inside :func:`main`, not at import time. The original
``main.py`` ran ``argparse`` at module scope, so merely importing it -- as the
CI smoke test did -- executed the parser and could exit the process.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from . import __version__
from .config import (
    ASRSettings,
    AudioSettings,
    MTSettings,
    Settings,
    TTSSettings,
    VADSettings,
)
from .languages import (
    AUTO,
    UnsupportedLanguageError,
    language_name,
    languages_with_speech,
    resolve_language,
)

logger = logging.getLogger(__name__)

__all__ = ["main", "build_parser", "settings_from_args"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speech-translate",
        description="Real-time speech-to-speech translation (faster-whisper -> NLLB-200 -> Piper).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  speech-translate --tgt es                  translate anything you say into Spanish\n"
            "  speech-translate --src en --tgt ja         pin the source language (a little faster)\n"
            "  speech-translate --tgt fr --no-speak       subtitles only, no synthesised audio\n"
            "  speech-translate --list-devices            show microphones\n"
            "  speech-translate --list-languages          show language codes\n"
            "  speech-translate --file talk.wav --tgt de  translate a recording instead of the mic\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    languages = parser.add_argument_group("languages")
    languages.add_argument(
        "--src",
        default=AUTO,
        help="Source language: 'auto', an ISO code (en), a FLORES code (eng_Latn) or a name.",
    )
    languages.add_argument(
        "--tgt",
        help="Target language (required unless using --list-devices/--list-languages).",
    )

    models = parser.add_argument_group("models")
    models.add_argument(
        "--asr-model",
        default="base",
        help="faster-whisper model. base (default) keeps up with live speech on a "
        "CPU; small is more accurate; distil-small.en is a fast English-only option.",
    )
    models.add_argument(
        "--mt-model",
        default="facebook/nllb-200-distilled-600M",
        help="Hugging Face translation model.",
    )
    models.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Compute device for ASR and MT.",
    )
    models.add_argument(
        "--compute-type",
        default="auto",
        help="CTranslate2 compute type (auto, int8, int8_float16, float16, float32).",
    )
    models.add_argument("--beam-size", type=int, default=1, help="ASR beam size (default: 1).")

    speech = parser.add_argument_group("speech output")
    speech.add_argument(
        "--tts",
        default="auto",
        choices=("auto", "piper", "system", "none"),
        help="Speech backend (default: auto -> Piper, then the OS voice).",
    )
    speech.add_argument("--tts-voice", default=None, help="Explicit voice, e.g. en_US-lessac-medium.")
    speech.add_argument(
        "--speech-rate",
        type=float,
        default=1.0,
        help="Speaking rate multiplier; >1 is slower (default: 1.0).",
    )
    speech.add_argument(
        "--no-speak",
        action="store_true",
        help="Subtitles-only: translate and print, never synthesise.",
    )

    audio = parser.add_argument_group("audio")
    audio.add_argument(
        "--input-device",
        default=None,
        help="Microphone index or name fragment (default: system default).",
    )
    audio.add_argument("--output-device", type=int, default=None, help="Playback device index.")
    audio.add_argument("--list-devices", action="store_true", help="List microphones and exit.")
    audio.add_argument("--file", default=None, help="Translate an audio file instead of the mic.")

    realtime = parser.add_argument_group("real-time tuning")
    realtime.add_argument("--no-vad", action="store_true", help="Disable VAD; use fixed windows.")
    realtime.add_argument(
        "--silence-ms",
        type=int,
        default=600,
        help="Silence that ends an utterance (default: 600).",
    )
    realtime.add_argument(
        "--max-utterance-ms",
        type=int,
        default=12_000,
        help="Hard cap on one utterance (default: 12000).",
    )
    realtime.add_argument(
        "--vad-threshold",
        type=float,
        default=0.015,
        help="Minimum RMS energy treated as speech (default: 0.015).",
    )
    realtime.add_argument(
        "--no-sentence-buffer",
        action="store_true",
        help="Speak every fragment immediately instead of waiting for sentence ends.",
    )

    misc = parser.add_argument_group("misc")
    misc.add_argument("--list-languages", action="store_true", help="List language codes and exit.")
    misc.add_argument("--save-transcript", default=None, help="Write the session transcript here.")
    misc.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug.")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Build :class:`Settings` from parsed arguments."""
    return Settings(
        src=args.src,
        tgt=args.tgt,
        audio=AudioSettings(
            input_device=args.input_device,
            output_device=args.output_device,
        ),
        vad=VADSettings(
            enabled=not args.no_vad,
            threshold=args.vad_threshold,
            min_silence_ms=args.silence_ms,
            max_utterance_ms=args.max_utterance_ms,
        ),
        asr=ASRSettings(
            model=args.asr_model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
        ),
        mt=MTSettings(model=args.mt_model, device=args.device),
        tts=TTSSettings(
            backend="none" if args.no_speak else args.tts,
            voice=args.tts_voice,
            length_scale=args.speech_rate,
        ),
        buffer_to_sentences=not args.no_sentence_buffer,
    )


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8.

    A translator whose console mangles 'cómo' into 'c?mo' is not much of a
    translator. Windows terminals still default to a legacy code page, so the
    streams are reconfigured explicitly rather than left to chance.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def _print_languages() -> None:
    speakable = dict(languages_with_speech())
    from .languages import supported_languages

    print(f"{'FLORES-200':<12} {'Language':<26} Speech")
    print("-" * 52)
    for code, name in supported_languages():
        print(f"{code:<12} {name:<26} {'yes' if code in speakable else '-'}")
    print()
    print(f"{len(supported_languages())} languages translatable, {len(speakable)} with a voice.")
    print("Pass any of these to --src/--tgt; ISO codes like 'es' and names like 'Spanish' also work.")


def _format_result(result) -> str:  # noqa: ANN001
    arrow = f"{result.source_language_name} -> {result.target_language_name}"
    lines = [f"\n[{arrow}]  {result.timings.total * 1000:.0f} ms"]
    lines.append(f"  heard : {result.transcript}")
    lines.append(f"  says  : {result.translation}")
    return "\n".join(lines)


def _run_file(settings: Settings, path: str, save_transcript: str | None) -> int:
    import time

    from .pipeline import Pipeline

    pipeline = Pipeline(settings)
    # Load the models before timing anything. Charging one-off model loading to
    # the first utterance is how you end up reporting a 55-second "latency".
    load_start = time.perf_counter()
    pipeline.warmup()
    load_seconds = time.perf_counter() - load_start

    result = pipeline.process_file(path, speak=settings.tts.backend != "none")
    if not result.transcript:
        print(f"No speech found in {path} ({result.skipped_reason or 'silence'}).")
        pipeline.close()
        return 1
    print(_format_result(result))
    print(
        f"  stages: asr {result.timings.asr * 1000:.0f} ms | "
        f"mt {result.timings.mt * 1000:.0f} ms | tts {result.timings.tts * 1000:.0f} ms"
    )
    print(
        f"  audio {result.audio_duration:.1f} s | RTF {result.real_time_factor:.2f} "
        f"({'faster' if result.real_time_factor < 1 else 'slower'} than real time) | "
        f"models loaded in {load_seconds:.1f} s"
    )
    if result.speech:
        out = "translated.wav"
        result.speech.save(out)
        print(f"  saved : {out}")
    if save_transcript:
        _save_transcript(save_transcript, [result])
    pipeline.close()
    return 0


def _save_transcript(path: str, results) -> None:  # noqa: ANN001
    from pathlib import Path

    lines = []
    for result in results:
        if not result.transcript:
            continue
        lines.append(f"[{result.source_language_name}] {result.transcript}")
        lines.append(f"[{result.target_language_name}] {result.translation}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Transcript written to {path}")


def _run_realtime(settings: Settings, save_transcript: str | None) -> int:
    from .audio.devices import AudioUnavailableError
    from .realtime import RealtimeSession

    def on_result(result) -> None:  # noqa: ANN001
        print(_format_result(result), flush=True)

    def on_status(message: str) -> None:
        print(f">>> {message}", flush=True)

    session = RealtimeSession(settings, on_result=on_result, on_status=on_status,
                              speak=settings.tts.backend != "none")
    try:
        session.start()
    except AudioUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f">>> {language_name(settings.src) if settings.src != AUTO else 'Auto-detect'}"
        f" -> {language_name(settings.tgt)}"
        f" | ASR {settings.asr.model} on {settings.asr.resolved_device()}"
        f" | voice {getattr(session.pipeline.tts, 'voice', 'n/a')}"
    )
    try:
        session.wait()
    except KeyboardInterrupt:
        print("\n>>> Stopping ...")
    stats = session.stop()

    if stats.utterances:
        print(
            f">>> {stats.utterances} utterances | average {stats.average_latency * 1000:.0f} ms"
            f" | RTF {stats.real_time_factor:.2f}"
        )
        if stats.dropped_utterances or stats.dropped_clips:
            print(
                f">>> dropped {stats.dropped_utterances} utterances and "
                f"{stats.dropped_clips} clips to stay near real time."
            )
    if save_transcript and stats.history:
        _save_transcript(save_transcript, stats.history)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _force_utf8_output()
    _configure_logging(args.verbose)

    if args.list_languages:
        _print_languages()
        return 0

    if args.list_devices:
        from .audio.devices import format_device_table

        print(format_device_table())
        return 0

    if not args.tgt:
        parser.error("--tgt is required (e.g. --tgt es). Use --list-languages to see the options.")

    try:
        settings = settings_from_args(args)
    except UnsupportedLanguageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.src != AUTO:
        logger.info("Source pinned to %s", language_name(resolve_language(args.src)))

    if args.file:
        return _run_file(settings, args.file, args.save_transcript)
    return _run_realtime(settings, args.save_transcript)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
