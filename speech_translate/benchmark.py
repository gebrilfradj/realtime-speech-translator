"""Latency benchmark for the cascade.

Reports per-stage latency and the real-time factor (RTF = processing seconds
per second of audio; below 1.0 means the system keeps up with a live speaker).

Two things the original benchmark got wrong and that make numbers meaningless:

* it timed the **first** call, so one-off model loading (tens of seconds) was
  charged to the measurement, and
* it reported only an end-to-end average, which cannot tell you *which* stage
  to optimise.

``--legacy`` runs the pre-modernisation stack (openai-whisper + M2M100-418M) on
the same audio and machine, so the speed-up claim in the README is reproducible
rather than asserted::

    python -m speech_translate.benchmark --audio sample.wav --tgt es
    python -m speech_translate.benchmark --audio sample.wav --tgt es --legacy
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from .config import ASRSettings, MTSettings, Settings, TTSSettings
from .languages import UnsupportedLanguageError

__all__ = ["BenchmarkResult", "run_benchmark", "main"]


@dataclass
class BenchmarkResult:
    label: str
    asr_model: str
    mt_model: str
    tts_backend: str
    device: str
    audio_seconds: float
    repeats: int
    load_seconds: float
    asr_ms: list[float] = field(default_factory=list)
    mt_ms: list[float] = field(default_factory=list)
    tts_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)
    transcript: str = ""
    translation: str = ""

    @staticmethod
    def _median(values: list[float]) -> float:
        return statistics.median(values) if values else 0.0

    @property
    def median_total_ms(self) -> float:
        return self._median(self.total_ms)

    @property
    def real_time_factor(self) -> float:
        if self.audio_seconds <= 0:
            return 0.0
        return self.median_total_ms / 1000 / self.audio_seconds

    def summary(self) -> str:
        lines = [
            f"{self.label}",
            f"  asr   {self._median(self.asr_ms):8.0f} ms   ({self.asr_model})",
            f"  mt    {self._median(self.mt_ms):8.0f} ms   ({self.mt_model})",
            f"  tts   {self._median(self.tts_ms):8.0f} ms   ({self.tts_backend})",
            f"  total {self.median_total_ms:8.0f} ms   RTF {self.real_time_factor:.2f}",
        ]
        if len(self.total_ms) > 1:
            lines.append(
                f"  spread {min(self.total_ms):.0f}-{max(self.total_ms):.0f} ms "
                f"over {self.repeats} runs (median reported)"
            )
        lines.append(f"  models loaded in {self.load_seconds:.1f} s (one-off, excluded above)")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["median_total_ms"] = round(self.median_total_ms, 1)
        data["real_time_factor"] = round(self.real_time_factor, 3)
        return data


def _describe_device(device: str) -> str:
    if device == "cuda":
        try:
            import torch

            return torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover
            return "cuda"
    return f"cpu ({platform.processor() or platform.machine()})"


def run_benchmark(
    audio_path: str,
    settings: Settings,
    repeats: int = 5,
    speak: bool = True,
) -> BenchmarkResult:
    """Benchmark the modern pipeline."""
    from .audio.wav import read_wav
    from .pipeline import Pipeline

    audio, _ = read_wav(audio_path, target_rate=settings.audio.sample_rate)
    pipeline = Pipeline(settings)

    load_start = time.perf_counter()
    pipeline.warmup()
    load_seconds = time.perf_counter() - load_start

    result = BenchmarkResult(
        label="faster-whisper + NLLB-200 + Piper",
        asr_model=settings.asr.model,
        mt_model=settings.mt.model.split("/")[-1],
        tts_backend=getattr(pipeline.tts, "voice", None) or pipeline.tts.name,
        device=_describe_device(settings.asr.resolved_device()),
        audio_seconds=len(audio) / settings.audio.sample_rate,
        repeats=repeats,
        load_seconds=load_seconds,
    )

    for index in range(repeats):
        run = pipeline.process(audio, speak=speak)
        result.asr_ms.append(run.timings.asr * 1000)
        result.mt_ms.append(run.timings.mt * 1000)
        result.tts_ms.append(run.timings.tts * 1000)
        result.total_ms.append(run.timings.total * 1000)
        if index == 0:
            result.transcript = run.transcript
            result.translation = run.translation
        print(f"  run {index + 1}/{repeats}: {run.timings.total * 1000:.0f} ms", flush=True)

    pipeline.close()
    return result


def run_legacy_benchmark(
    audio_path: str, settings: Settings, repeats: int = 5
) -> BenchmarkResult:
    """Benchmark the original stack (openai-whisper + M2M100) for comparison.

    TTS is excluded: Coqui TTS is unmaintained and no longer installs on
    current Python versions, which is precisely why it was replaced. ASR and MT
    are compared like for like on the same audio, machine and text.
    """
    import torch

    from .audio.wav import read_wav
    from .languages import flores_to_whisper

    try:
        import whisper  # openai-whisper
    except ImportError as exc:  # pragma: no cover - optional comparison path
        raise SystemExit(
            "The legacy comparison needs the original packages:\n"
            "  pip install openai-whisper transformers\n"
            f"({exc})"
        ) from exc
    if not hasattr(whisper, "load_model"):
        raise SystemExit(
            "The installed 'whisper' module has no load_model(). You have the "
            "PyPI package 'whisper' (a Graphite time-series database), not "
            "OpenAI's. Install 'openai-whisper' instead -- this is the exact "
            "dependency bug this project used to ship."
        )
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    audio, _ = read_wav(audio_path, target_rate=16_000)
    device = settings.asr.resolved_device()

    load_start = time.perf_counter()
    asr = whisper.load_model(settings.asr.model, device=device)
    mt_name = "facebook/m2m100_418M"
    mt = M2M100ForConditionalGeneration.from_pretrained(mt_name).to(device).eval()
    tokenizer = M2M100Tokenizer.from_pretrained(mt_name)
    # Warm up so loading is not charged to the first run, same as the modern path.
    asr.transcribe(audio[: 16_000 // 2], fp16=(device == "cuda"))
    load_seconds = time.perf_counter() - load_start

    tgt_iso = flores_to_whisper(settings.tgt) or "es"
    result = BenchmarkResult(
        label="openai-whisper + M2M100-418M (original stack)",
        asr_model=settings.asr.model,
        mt_model="m2m100_418M",
        tts_backend="excluded (Coqui TTS no longer installs)",
        device=_describe_device(device),
        audio_seconds=len(audio) / 16_000,
        repeats=repeats,
        load_seconds=load_seconds,
    )

    for index in range(repeats):
        start = time.perf_counter()
        transcription = asr.transcribe(audio, fp16=(device == "cuda"))
        asr_ms = (time.perf_counter() - start) * 1000
        text = transcription.get("text", "").strip()
        src_iso = transcription.get("language", "en")

        start = time.perf_counter()
        tokenizer.src_lang = src_iso
        encoded = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = mt.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(tgt_iso),
                max_length=200,
                num_beams=1,
            )
        translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        mt_ms = (time.perf_counter() - start) * 1000

        result.asr_ms.append(asr_ms)
        result.mt_ms.append(mt_ms)
        result.tts_ms.append(0.0)
        result.total_ms.append(asr_ms + mt_ms)
        if index == 0:
            result.transcript = text
            result.translation = translated
        print(f"  run {index + 1}/{repeats}: {asr_ms + mt_ms:.0f} ms", flush=True)

    return result


def _print_comparison(modern: BenchmarkResult, legacy: BenchmarkResult) -> None:
    print("\n" + "=" * 62)
    print("BEFORE / AFTER (same machine, same audio, TTS excluded from both)")
    print("=" * 62)
    rows = [
        ("ASR", statistics.median(legacy.asr_ms), statistics.median(modern.asr_ms)),
        ("MT", statistics.median(legacy.mt_ms), statistics.median(modern.mt_ms)),
    ]
    legacy_total = sum(value for _, value, _ in rows)
    modern_total = sum(value for _, _, value in rows)
    rows.append(("TOTAL", legacy_total, modern_total))

    print(f"{'stage':<8}{'before (ms)':>14}{'after (ms)':>14}{'speed-up':>12}")
    print("-" * 62)
    for name, before, after in rows:
        speedup = (before / after) if after else 0.0
        print(f"{name:<8}{before:>14.0f}{after:>14.0f}{speedup:>11.2f}x")
    print("-" * 62)
    print(
        f"RTF (audio-normalised): {legacy_total / 1000 / legacy.audio_seconds:.2f}"
        f"  ->  {modern_total / 1000 / modern.audio_seconds:.2f}"
    )
    print(f"Device: {modern.device}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speech-translate-bench",
        description="Measure per-stage latency of the speech translation cascade.",
    )
    parser.add_argument("--audio", required=True, help="Path to a sample WAV file.")
    parser.add_argument("--src", default="auto", help="Source language (default: auto).")
    parser.add_argument("--tgt", default="es", help="Target language (default: es).")
    parser.add_argument("--repeats", type=int, default=5, help="Timed runs (default: 5).")
    parser.add_argument("--asr-model", default="base", help="faster-whisper model.")
    parser.add_argument(
        "--mt-model", default="facebook/nllb-200-distilled-600M", help="Translation model."
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--compute-type", default="auto", help="CTranslate2 compute type.")
    parser.add_argument(
        "--tts", default="auto", choices=("auto", "piper", "system", "none"), help="Speech backend."
    )
    parser.add_argument("--no-tts", action="store_true", help="Skip the synthesis stage.")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Also benchmark the original openai-whisper + M2M100 stack and compare.",
    )
    parser.add_argument("--json", default=None, help="Write results to this JSON file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover
                pass

    try:
        settings = Settings(
            src=args.src,
            tgt=args.tgt,
            asr=ASRSettings(
                model=args.asr_model, device=args.device, compute_type=args.compute_type
            ),
            mt=MTSettings(model=args.mt_model, device=args.device),
            tts=TTSSettings(backend="none" if args.no_tts else args.tts),
        )
    except UnsupportedLanguageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Benchmarking on {args.audio} ({args.repeats} runs after warm-up)\n")
    print("modern stack:")
    modern = run_benchmark(args.audio, settings, args.repeats, speak=not args.no_tts)
    print()
    print(modern.summary())
    print(f'\n  heard: "{modern.transcript}"')
    print(f'  says : "{modern.translation}"')

    payload: dict[str, object] = {"modern": modern.as_dict()}

    if args.legacy:
        print("\nlegacy stack:")
        legacy = run_legacy_benchmark(args.audio, settings, args.repeats)
        print()
        print(legacy.summary())
        _print_comparison(modern, legacy)
        payload["legacy"] = legacy.as_dict()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
