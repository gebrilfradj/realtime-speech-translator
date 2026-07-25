"""Smoke tests for the Gradio demo.

The web UI is the project's public face, so it is worth proving that it at
least constructs and that its audio-decoding helper handles what a browser
actually sends. Skipped when the optional ``web`` extra is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

gradio = pytest.importorskip("gradio", reason="the 'web' extra is not installed")

from speech_translate.config import Settings  # noqa: E402
from speech_translate.webui import (  # noqa: E402
    _decode_gradio_audio,
    _format_stats,
    _render_history,
    _write_transcript,
    build_demo,
)


class TestAudioDecoding:
    def test_int16_input_is_normalised_to_float(self) -> None:
        raw = np.array([32767, -32768, 0], dtype=np.int16)
        audio, rate = _decode_gradio_audio((16_000, raw))
        assert audio.dtype == np.float32
        assert rate == 16_000
        assert -1.01 <= float(audio.min()) and float(audio.max()) <= 1.01

    def test_float_input_is_passed_through(self) -> None:
        raw = np.array([0.5, -0.5], dtype=np.float32)
        audio, _ = _decode_gradio_audio((16_000, raw))
        assert audio.tolist() == pytest.approx([0.5, -0.5])

    def test_browser_rate_is_resampled_to_16k(self) -> None:
        """Browsers commonly capture at 48 kHz; Whisper wants 16 kHz."""
        raw = np.zeros(48_000, dtype=np.float32)
        audio, rate = _decode_gradio_audio((48_000, raw))
        assert rate == 16_000
        assert len(audio) == pytest.approx(16_000, rel=0.01)

    def test_stereo_is_collapsed_to_mono(self) -> None:
        raw = np.zeros((1_000, 2), dtype=np.float32)
        audio, _ = _decode_gradio_audio((16_000, raw))
        assert audio.ndim == 1

    def test_none_payload_is_handled(self) -> None:
        audio, _ = _decode_gradio_audio(None)
        assert audio.size == 0


class TestRendering:
    def test_empty_history_renders_a_placeholder(self) -> None:
        assert "_" in _render_history([], subtitles_only=False)

    def test_subtitles_only_hides_the_source_text(self) -> None:
        history = [
            {"transcript": "Hello", "translation": "Hola", "src": "English", "tgt": "Spanish"}
        ]
        assert "Hello" not in _render_history(history, subtitles_only=True)
        assert "Hola" in _render_history(history, subtitles_only=True)

    def test_full_mode_shows_both(self) -> None:
        history = [
            {"transcript": "Hello", "translation": "Hola", "src": "English", "tgt": "Spanish"}
        ]
        rendered = _render_history(history, subtitles_only=False)
        assert "Hello" in rendered and "Hola" in rendered

    def test_transcript_export_writes_both_languages(self) -> None:
        history = [
            {"transcript": "Hello", "translation": "Hola", "src": "English", "tgt": "Spanish"}
        ]
        path = _write_transcript(history)
        assert path is not None
        content = open(path, encoding="utf-8").read()
        assert "Hello" in content and "Hola" in content

    def test_no_transcript_file_without_history(self) -> None:
        assert _write_transcript([]) is None

    def test_stats_placeholder_when_nothing_was_heard(self) -> None:
        from speech_translate.pipeline import PipelineResult

        assert "No speech" in _format_stats(PipelineResult())


def test_demo_builds_without_loading_models() -> None:
    """build_demo must not download a 600M-parameter model at import time."""
    demo = build_demo(Settings(src="auto", tgt="spa_Latn"), preload=False)
    assert demo is not None
