"""Browser demo (Gradio).

Two ways in:

* **Translate** -- record or upload a clip and get transcript, translation and
  synthesised speech back. Works on a hosted Space, where the server has no
  microphone of its own, because the browser does the capture.
* **Live** -- streams microphone audio, segments it with VAD and translates
  each utterance as you pause, building a running transcript.

The heavy models are loaded once and shared. Only the TTS voice is swapped when
the target language changes, because ASR and MT are language-independent and
reloading a 600M-parameter translator per request would dominate latency.

    python -m speech_translate.webui --share
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ASRSettings, MTSettings, Settings, TTSSettings
from .languages import AUTO, languages_with_speech, supported_languages

logger = logging.getLogger(__name__)

__all__ = ["build_demo", "main"]

SAMPLE_RATE = 16_000

_DESCRIPTION = """
# Real-Time Speech Translator

Speak in one language, hear it in another. A fully open-source, local cascade:

**faster-whisper** (speech recognition) -> **NLLB-200** (translation, 200 languages) -> **Piper** (speech synthesis)

Nothing is sent to a third-party API.
"""


@dataclass
class _ModelCache:
    """Process-wide model cache, guarded because Gradio serves concurrently."""

    settings: Settings
    lock: threading.Lock = field(default_factory=threading.Lock)
    _pipeline: object | None = None
    _tts_language: str = ""

    def pipeline(self, target_language: str, speak: bool):
        from .pipeline import Pipeline
        from .tts import create_tts_backend

        with self.lock:
            if self._pipeline is None:
                self.settings = self.settings.with_languages(tgt=target_language)
                self._pipeline = Pipeline(self.settings)
                self._pipeline.warmup()
                self._tts_language = target_language

            pipeline = self._pipeline
            pipeline.settings.tgt = target_language

            needs_voice = speak and (
                self._tts_language != target_language or pipeline._tts is None
            )
            if needs_voice:
                if pipeline._tts is not None:
                    pipeline._tts.close()
                backend = create_tts_backend(target_language, self.settings.tts)
                try:
                    # Warm the new voice here rather than inside the user's
                    # first request; Piper's first synthesis builds its graph.
                    backend.synthesize("Hola.")
                except Exception:
                    logger.debug("Voice warm-up failed (harmless)", exc_info=True)
                pipeline._tts = backend
                self._tts_language = target_language
            return pipeline


def _decode_gradio_audio(payload) -> tuple[np.ndarray, int]:  # noqa: ANN001
    """Normalise Gradio's ``(sample_rate, ndarray)`` into mono float32."""
    from .audio.wav import resample, to_mono

    if payload is None:
        return np.zeros(0, dtype=np.float32), SAMPLE_RATE
    sample_rate, audio = payload
    audio = np.asarray(audio)
    if audio.dtype.kind in "iu":
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    audio = to_mono(audio.astype(np.float32))
    if sample_rate != SAMPLE_RATE:
        audio = resample(audio, sample_rate, SAMPLE_RATE)
    return audio, SAMPLE_RATE


def _format_stats(result) -> str:  # noqa: ANN001
    if not result or not result.transcript:
        return "_No speech detected._"
    rtf = result.real_time_factor
    verdict = "faster than real time" if 0 < rtf < 1 else "slower than real time"
    return (
        f"**{result.source_language_name} → {result.target_language_name}**  \n"
        f"ASR `{result.timings.asr * 1000:.0f} ms` · "
        f"MT `{result.timings.mt * 1000:.0f} ms` · "
        f"TTS `{result.timings.tts * 1000:.0f} ms` · "
        f"**total `{result.timings.total * 1000:.0f} ms`**  \n"
        f"{result.audio_duration:.1f} s of audio → RTF `{rtf:.2f}` ({verdict})"
    )


def _write_transcript(history: list[dict]) -> str | None:
    if not history:
        return None
    lines = []
    for entry in history:
        lines.append(f"[{entry['src']}] {entry['transcript']}")
        lines.append(f"[{entry['tgt']}] {entry['translation']}")
        lines.append("")
    path = Path(tempfile.gettempdir()) / "speech_translate_transcript.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _render_history(history: list[dict], subtitles_only: bool) -> str:
    if not history:
        return "_Transcript will appear here._"
    blocks = []
    for entry in reversed(history[-30:]):
        if subtitles_only:
            blocks.append(f"**{entry['translation']}**")
        else:
            blocks.append(f"**{entry['translation']}**  \n<sub>{entry['transcript']}</sub>")
    return "\n\n".join(blocks)


def build_demo(settings: Settings | None = None, preload: bool = False):
    """Construct the Gradio Blocks app."""
    import gradio as gr

    settings = settings or Settings(src=AUTO, tgt="spa_Latn")
    cache = _ModelCache(settings)
    if preload:
        cache.pipeline(settings.tgt, speak=True)

    speakable = dict(languages_with_speech())
    target_choices = [
        (f"{name}{'' if code in speakable else '  (text only)'}", code)
        for code, name in supported_languages()
    ]
    source_choices = [("Detect automatically", AUTO)] + [
        (name, code) for code, name in supported_languages()
    ]

    # -- callbacks -------------------------------------------------------
    def translate_clip(payload, src, tgt, speak, subtitles_only, history):  # noqa: ANN001
        history = list(history or [])
        if payload is None:
            return (
                "_Record or upload something first._",
                "",
                "",
                None,
                _render_history(history, subtitles_only),
                history,
                None,
            )
        audio, _ = _decode_gradio_audio(payload)
        if audio.size == 0:
            return (
                "_Empty recording._",
                "",
                "",
                None,
                _render_history(history, subtitles_only),
                history,
                None,
            )

        pipeline = cache.pipeline(tgt, speak=speak and not subtitles_only)
        pipeline.settings.src = src
        result = pipeline.process(audio, speak=speak and not subtitles_only)

        audio_out = None
        if result.speech and len(result.speech.audio):
            audio_out = (result.speech.sample_rate, result.speech.audio)

        if result.transcript:
            history.append(
                {
                    "transcript": result.transcript,
                    "translation": result.translation,
                    "src": result.source_language_name,
                    "tgt": result.target_language_name,
                }
            )
        return (
            _format_stats(result),
            result.transcript or "",
            result.translation or "",
            audio_out,
            _render_history(history, subtitles_only),
            history,
            _write_transcript(history),
        )

    def stream_audio(chunk, src, tgt, speak, subtitles_only, state, history):  # noqa: ANN001
        """Handle one streamed microphone chunk."""
        from .audio.vad import UtteranceSegmenter

        history = list(history or [])
        state = state or {}
        if state.get("segmenter") is None:
            state["segmenter"] = UtteranceSegmenter(settings.audio, settings.vad)

        if chunk is None:
            return _render_history(history, subtitles_only), state, history, 0.0, None

        audio, _ = _decode_gradio_audio(chunk)
        if audio.size == 0:
            return _render_history(history, subtitles_only), state, history, 0.0, None

        level = float(np.sqrt(np.mean(np.square(audio))))
        segmenter = state["segmenter"]
        block = settings.audio.block_frames
        utterances = []
        for start in range(0, len(audio), block):
            frame = audio[start : start + block]
            if frame.size < block:
                frame = np.pad(frame, (0, block - frame.size))
            utterances.extend(segmenter.push(frame))

        audio_out = None
        for utterance in utterances:
            pipeline = cache.pipeline(tgt, speak=speak and not subtitles_only)
            pipeline.settings.src = src
            result = pipeline.process(utterance.audio, speak=speak and not subtitles_only)
            if not result.transcript:
                continue
            history.append(
                {
                    "transcript": result.transcript,
                    "translation": result.translation,
                    "src": result.source_language_name,
                    "tgt": result.target_language_name,
                }
            )
            if result.speech and len(result.speech.audio):
                audio_out = (result.speech.sample_rate, result.speech.audio)

        return (
            _render_history(history, subtitles_only),
            state,
            history,
            round(min(level * 8, 1.0), 3),
            audio_out,
        )

    def clear_history(subtitles_only):  # noqa: ANN001
        return _render_history([], subtitles_only), [], None

    # -- layout ----------------------------------------------------------
    with gr.Blocks(title="Real-Time Speech Translator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            source_dropdown = gr.Dropdown(
                choices=source_choices, value=AUTO, label="I am speaking", scale=2
            )
            target_dropdown = gr.Dropdown(
                choices=target_choices, value=settings.tgt, label="Translate into", scale=2
            )
            speak_toggle = gr.Checkbox(value=True, label="Speak the translation", scale=1)
            subtitles_toggle = gr.Checkbox(value=False, label="Subtitles only", scale=1)

        history_state = gr.State([])

        with gr.Tab("Translate a clip"):
            gr.Markdown(
                "Record yourself or upload a file, then press **Translate**. "
                "Works without a microphone if you upload audio."
            )
            with gr.Row():
                with gr.Column():
                    clip_input = gr.Audio(
                        sources=["microphone", "upload"], type="numpy", label="Your speech"
                    )
                    translate_button = gr.Button("Translate", variant="primary")
                with gr.Column():
                    stats_output = gr.Markdown("_Latency will appear here._")
                    transcript_output = gr.Textbox(label="Heard (source language)", lines=3)
                    translation_output = gr.Textbox(label="Translation", lines=3)
                    speech_output = gr.Audio(label="Spoken translation", autoplay=True)

        with gr.Tab("Live"):
            gr.Markdown(
                "Streams your microphone and translates each utterance as you pause. "
                "Speech is detected automatically; there is no fixed chunk size."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    stream_input = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="Live microphone",
                    )
                    level_meter = gr.Slider(
                        0, 1, value=0, label="Microphone level", interactive=False
                    )
                    live_speech = gr.Audio(label="Spoken translation", autoplay=True)
                with gr.Column(scale=2):
                    live_transcript = gr.Markdown("_Transcript will appear here._")
            stream_state = gr.State({})

        with gr.Row():
            transcript_view = gr.Markdown("_Transcript will appear here._")
        with gr.Row():
            download_button = gr.DownloadButton("Download transcript", size="sm")
            clear_button = gr.Button("Clear transcript", size="sm")

        gr.Markdown(
            "---\n"
            "Models: `faster-whisper small` · `facebook/nllb-200-distilled-600M` · `Piper`  \n"
            "[Source on GitHub](https://github.com/gebrilfradj/realtime-speech-translator)"
        )

        translate_button.click(
            translate_clip,
            inputs=[
                clip_input,
                source_dropdown,
                target_dropdown,
                speak_toggle,
                subtitles_toggle,
                history_state,
            ],
            outputs=[
                stats_output,
                transcript_output,
                translation_output,
                speech_output,
                transcript_view,
                history_state,
                download_button,
            ],
        )

        stream_input.stream(
            stream_audio,
            inputs=[
                stream_input,
                source_dropdown,
                target_dropdown,
                speak_toggle,
                subtitles_toggle,
                stream_state,
                history_state,
            ],
            outputs=[live_transcript, stream_state, history_state, level_meter, live_speech],
            show_progress="hidden",
        )

        clear_button.click(
            clear_history,
            inputs=[subtitles_toggle],
            outputs=[transcript_view, history_state, download_button],
        )

    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="speech-translate-web",
        description="Launch the browser demo for the speech translator.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument("--tgt", default="es", help="Initial target language.")
    parser.add_argument("--src", default=AUTO, help="Initial source language.")
    parser.add_argument("--asr-model", default="base", help="faster-whisper model.")
    parser.add_argument(
        "--mt-model", default="facebook/nllb-200-distilled-600M", help="Translation model."
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--preload", action="store_true", help="Load models at start-up instead of on first use."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    settings = Settings(
        src=args.src,
        tgt=args.tgt,
        asr=ASRSettings(model=args.asr_model, device=args.device),
        mt=MTSettings(model=args.mt_model, device=args.device),
        tts=TTSSettings(backend="auto"),
    )

    started = time.perf_counter()
    demo = build_demo(settings, preload=args.preload)
    if args.preload:
        logger.info("Models ready in %.1f s", time.perf_counter() - started)
    demo.queue().launch(
        server_name=args.host, server_port=args.port, share=args.share, show_api=False
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
