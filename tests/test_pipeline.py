"""The Pipeline class, driven entirely with fakes -- no model downloads."""

from __future__ import annotations

import numpy as np
import pytest

from speech_translate.asr import Transcript
from speech_translate.config import Settings, TTSSettings
from speech_translate.pipeline import Pipeline, PipelineResult, StageTimings, timed
from speech_translate.tts import NullTTS, TTSUnavailableError

from .conftest import FakeRecognizer, FakeTranslator, FakeTTS


def make_pipeline(**kwargs) -> tuple[Pipeline, FakeRecognizer, FakeTranslator, FakeTTS]:
    recognizer = kwargs.pop("recognizer", None) or FakeRecognizer()
    translator = kwargs.pop("translator", None) or FakeTranslator()
    tts = kwargs.pop("tts", None) or FakeTTS()
    settings = kwargs.pop("settings", None) or Settings(src="auto", tgt="spa_Latn")
    pipeline = Pipeline(settings, recognizer=recognizer, translator=translator, tts=tts)
    return pipeline, recognizer, translator, tts


class TestAutoSourceLanguage:
    """Regression tests for the original ``--src auto`` bug.

    The old code passed the literal string ``"auto"`` straight through to the
    translator, which then produced garbage or crashed.
    """

    def test_detected_language_reaches_the_translator(self) -> None:
        recognizer = FakeRecognizer(text="Bonjour tout le monde.", language="fra_Latn")
        pipeline, _, translator, _ = make_pipeline(recognizer=recognizer)

        result = pipeline.process(np.zeros(16_000, dtype=np.float32))

        assert translator.calls, "the translator was never called"
        _, src, tgt = translator.calls[0]
        assert src == "fra_Latn", "the detected language must replace 'auto'"
        assert tgt == "spa_Latn"
        assert result.source_language == "fra_Latn"
        assert result.source_language_name == "French"

    def test_auto_never_leaks_into_the_translator(self) -> None:
        pipeline, _, translator, _ = make_pipeline()
        pipeline.process(np.zeros(16_000, dtype=np.float32))
        assert all(src != "auto" for _, src, _ in translator.calls)

    def test_explicit_source_is_forwarded_to_the_recognizer(self) -> None:
        settings = Settings(src="de", tgt="en")
        pipeline, recognizer, _, _ = make_pipeline(settings=settings)
        pipeline.process(np.zeros(16_000, dtype=np.float32))
        assert recognizer.calls[0][1] == "deu_Latn"


class TestPipelineBehaviour:
    def test_returns_transcript_translation_and_speech(self) -> None:
        pipeline, _, _, tts = make_pipeline()
        result = pipeline.process(np.zeros(16_000, dtype=np.float32))

        assert result.transcript == "Hello there."
        assert result.translation == "Hola."
        assert result.speech is not None
        assert len(result.speech.audio) > 0
        assert tts.calls == ["Hola."]

    def test_silence_short_circuits_before_translation(self) -> None:
        recognizer = FakeRecognizer(text="")
        pipeline, _, translator, tts = make_pipeline(recognizer=recognizer)

        result = pipeline.process(np.zeros(16_000, dtype=np.float32))

        assert not result.transcript
        assert result.skipped_reason == "no speech detected"
        assert translator.calls == [], "silence must not reach the translator"
        assert tts.calls == [], "silence must not reach the synthesiser"

    def test_same_language_is_not_round_tripped_through_mt(self) -> None:
        settings = Settings(src="auto", tgt="eng_Latn")
        recognizer = FakeRecognizer(text="Already English.", language="eng_Latn")
        pipeline, _, translator, _ = make_pipeline(settings=settings, recognizer=recognizer)

        result = pipeline.process(np.zeros(16_000, dtype=np.float32))

        assert result.translation == "Already English."
        assert translator.calls == []
        assert "same" in result.skipped_reason

    def test_speak_false_skips_synthesis(self) -> None:
        pipeline, _, _, tts = make_pipeline()
        result = pipeline.process(np.zeros(16_000, dtype=np.float32), speak=False)
        assert result.speech is None
        assert tts.calls == []

    def test_timings_are_recorded_for_each_stage(self) -> None:
        pipeline, _, _, _ = make_pipeline()
        result = pipeline.process(np.zeros(16_000, dtype=np.float32))
        assert result.timings.asr >= 0
        assert result.timings.mt >= 0
        assert result.timings.tts >= 0
        assert result.timings.total == pytest.approx(
            result.timings.asr + result.timings.mt + result.timings.tts
        )

    def test_audio_duration_and_rtf(self) -> None:
        pipeline, _, _, _ = make_pipeline()
        result = pipeline.process(np.zeros(32_000, dtype=np.float32))
        assert result.audio_duration == pytest.approx(2.0)
        assert result.real_time_factor == pytest.approx(result.timings.total / 2.0)

    def test_rtf_is_zero_when_duration_is_unknown(self) -> None:
        assert PipelineResult().real_time_factor == 0.0


class TestStageTimings:
    def test_total_sums_the_stages(self) -> None:
        timings = StageTimings(asr=0.1, mt=0.2, tts=0.3)
        assert timings.total == pytest.approx(0.6)
        assert timings.as_dict()["total"] == pytest.approx(0.6)

    def test_timed_accumulates_rather_than_overwrites(self) -> None:
        """Two MT calls for one utterance must add up, not clobber."""
        timings = StageTimings()
        with timed(timings, "mt"):
            pass
        first = timings.mt
        with timed(timings, "mt"):
            pass
        assert timings.mt >= first


class TestLazyConstruction:
    def test_components_are_not_built_until_used(self) -> None:
        """Constructing a Pipeline must not load a 600M-parameter model."""
        pipeline = Pipeline(Settings(tgt="es"))
        assert pipeline._recognizer is None
        assert pipeline._translator is None
        assert pipeline._tts is None

    def test_warmup_exercises_every_stage(self) -> None:
        """Regression: warm-up skipped TTS, so the first real utterance paid
        Piper's one-off ONNX graph build (measured at ~3 s)."""
        pipeline, recognizer, translator, tts = make_pipeline()
        pipeline.warmup()
        assert recognizer.calls, "ASR was not warmed"
        assert translator.calls, "MT was not warmed"
        assert tts.calls, "TTS was not warmed"

    def test_warmup_survives_a_failing_component(self) -> None:
        class ExplodingTTS(FakeTTS):
            def synthesize(self, text: str):  # noqa: ANN201
                raise RuntimeError("no voice")

        pipeline, _, _, _ = make_pipeline(tts=ExplodingTTS())
        pipeline.warmup()  # must not raise

    def test_none_backend_yields_silent_speech(self) -> None:
        settings = Settings(tgt="es", tts=TTSSettings(backend="none"))
        pipeline = Pipeline(
            settings, recognizer=FakeRecognizer(), translator=FakeTranslator()
        )
        result = pipeline.process(np.zeros(16_000, dtype=np.float32))
        assert result.translation == "Hola."
        assert result.speech is not None
        assert len(result.speech.audio) == 0


class TestSynthesisFailureDegradesGracefully:
    """A broken speech engine must not take translation down with it.

    Found in practice: the pyttsx3 fallback blocks forever on a machine with no
    audio endpoint. The timeout turns that into an exception -- which then
    crashed the whole run until this degradation path existed.
    """

    class ExplodingTTS(FakeTTS):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def synthesize(self, text: str):  # noqa: ANN201
            self.attempts += 1
            raise TTSUnavailableError("no audio device")

    def test_translation_still_returned_when_synthesis_fails(self) -> None:
        tts = self.ExplodingTTS()
        pipeline, _, _, _ = make_pipeline(tts=tts)
        result = pipeline.process(np.zeros(16_000, dtype=np.float32))
        assert result.translation == "Hola.", "translation must survive a TTS failure"
        assert result.speech is None
        assert tts.attempts == 1

    def test_failure_is_paid_once_not_every_utterance(self) -> None:
        tts = self.ExplodingTTS()
        pipeline, _, _, _ = make_pipeline(tts=tts)
        for _ in range(3):
            pipeline.process(np.zeros(16_000, dtype=np.float32))
        assert tts.attempts == 1, "the failed backend must be swapped out, not retried"
        assert isinstance(pipeline._tts, NullTTS)

    def test_unexpected_exceptions_are_also_contained(self) -> None:
        class WeirdTTS(FakeTTS):
            def synthesize(self, text: str):  # noqa: ANN201
                raise ValueError("something odd")

        pipeline, _, _, _ = make_pipeline(tts=WeirdTTS())
        result = pipeline.process(np.zeros(16_000, dtype=np.float32))
        assert result.translation == "Hola."
        assert result.speech is None


def test_transcript_truthiness() -> None:
    assert not Transcript(text="", language="eng_Latn")
    assert Transcript(text="hi", language="eng_Latn")
