"""The threaded real-time session, driven without a microphone.

``RealtimeSession`` is the piece that actually runs during live translation, so
its queueing, backpressure and sentence-buffering logic is exercised here
directly with fake components rather than left to manual testing.
"""

from __future__ import annotations

import queue

import numpy as np
import pytest

from speech_translate.audio.vad import Utterance
from speech_translate.config import Settings
from speech_translate.pipeline import Pipeline
from speech_translate.realtime import RealtimeSession, SessionStats

from .conftest import FakeRecognizer, FakeTranslator, FakeTTS


def make_utterance(seconds: float = 1.0, truncated: bool = False) -> Utterance:
    samples = np.zeros(int(16_000 * seconds), dtype=np.float32)
    return Utterance(
        audio=samples,
        sample_rate=16_000,
        duration=seconds,
        peak_level=0.3,
        truncated=truncated,
    )


def make_session(
    text: str = "Hello there.",
    language: str = "eng_Latn",
    tgt: str = "spa_Latn",
    **settings_kwargs,
) -> tuple[RealtimeSession, list, FakeTranslator, FakeTTS]:
    recognizer = FakeRecognizer(text=text, language=language)
    translator = FakeTranslator()
    tts = FakeTTS()
    settings = Settings(src="auto", tgt=tgt, **settings_kwargs)
    pipeline = Pipeline(settings, recognizer=recognizer, translator=translator, tts=tts)
    results: list = []
    session = RealtimeSession(
        settings, pipeline=pipeline, on_result=results.append, speak=False
    )
    return session, results, translator, tts


class TestBackpressure:
    """The old pipeline used an unbounded queue behind a blocking playback
    call, so a slow machine drifted further behind for as long as it ran."""

    def test_queue_is_bounded_by_settings(self) -> None:
        session, _, _, _ = make_session(max_pending_utterances=2)
        assert session._utterances.maxsize == 2

    def test_overflow_drops_the_oldest_not_the_newest(self) -> None:
        session, _, _, _ = make_session(max_pending_utterances=2)
        first, second, third = (make_utterance(1.0) for _ in range(3))

        session._enqueue(first)
        session._enqueue(second)
        session._enqueue(third)

        assert session.stats.dropped_utterances == 1
        remaining = [session._utterances.get_nowait() for _ in range(2)]
        assert remaining[0] is second, "the oldest item should have been dropped"
        assert remaining[1] is third, "the newest item must be kept"

    def test_no_drops_when_keeping_up(self) -> None:
        session, _, _, _ = make_session(max_pending_utterances=4)
        for _ in range(4):
            session._enqueue(make_utterance())
        assert session.stats.dropped_utterances == 0


class TestSentenceBuffering:
    def test_complete_utterance_is_translated_immediately(self) -> None:
        session, results, translator, _ = make_session(text="Hello there.")
        session._handle(make_utterance(truncated=False))
        assert len(results) == 1
        assert translator.calls[0][0] == "Hello there."

    def test_truncated_fragment_is_held_until_the_sentence_ends(self) -> None:
        """A truncated utterance means the speaker was cut off mid-sentence."""
        session, results, translator, _ = make_session(text="I went to the")
        session._handle(make_utterance(truncated=True))
        assert results == [], "an incomplete fragment must not be translated yet"
        assert translator.calls == []

    def test_held_fragment_is_flushed_when_the_speaker_pauses(self) -> None:
        session, results, translator, _ = make_session(text="I went to the")
        session._handle(make_utterance(truncated=True))
        session.pipeline._recognizer.text = "store yesterday."
        session._handle(make_utterance(truncated=False))

        assert len(results) == 1
        assert translator.calls[0][0] == "I went to the store yesterday."

    def test_punctuated_utterance_is_never_dropped(self) -> None:
        """Regression: _handle discarded what add() returned and kept only
        flush(), so any utterance ending in punctuation -- which is nearly
        everything Whisper produces -- was silently never translated."""
        session, results, translator, _ = make_session(text="This is a complete sentence.")
        session._handle(make_utterance(truncated=False))
        assert len(results) == 1, "a punctuated utterance must reach the translator"
        assert translator.calls[0][0] == "This is a complete sentence."

    def test_multiple_sentences_in_one_utterance_all_emit(self) -> None:
        session, results, _, _ = make_session(text="First one. Second one. Third one.")
        session._handle(make_utterance(truncated=False))
        assert len(results) == 3

    def test_trailing_fragment_after_a_sentence_is_still_released_on_pause(self) -> None:
        session, results, translator, _ = make_session(text="Done. And then")
        session._handle(make_utterance(truncated=False))
        spoken = [call[0] for call in translator.calls]
        assert spoken == ["Done.", "And then"]
        assert len(results) == 2

    def test_buffering_can_be_disabled(self) -> None:
        session, results, translator, _ = make_session(
            text="I went to the", buffer_to_sentences=False
        )
        session._handle(make_utterance(truncated=True))
        assert len(results) == 1
        assert translator.calls[0][0] == "I went to the"

    def test_silence_produces_no_result(self) -> None:
        session, results, translator, _ = make_session(text="")
        session._handle(make_utterance())
        assert results == []
        assert translator.calls == []
        assert session.stats.utterances == 0


class TestLanguageHandling:
    def test_detected_language_is_forwarded_not_auto(self) -> None:
        session, _, translator, _ = make_session(text="Bonjour.", language="fra_Latn")
        session._handle(make_utterance())
        assert translator.calls[0][1] == "fra_Latn"

    def test_matching_source_and_target_skips_translation(self) -> None:
        session, results, translator, _ = make_session(
            text="Already Spanish.", language="spa_Latn", tgt="spa_Latn"
        )
        session._handle(make_utterance())
        assert translator.calls == []
        assert results[0].translation == "Already Spanish."

    def test_flushing_a_stranded_fragment_uses_the_last_known_language(self) -> None:
        """Flushing must not fall back to 'auto', which the translator rejects."""
        session, results, translator, _ = make_session(
            text="unfinished thought", language="deu_Latn"
        )
        session._handle(make_utterance(truncated=True))
        assert translator.calls == []
        session._flush_buffer(force=True)
        assert translator.calls, "the buffered fragment should have been flushed"
        assert translator.calls[0][1] == "deu_Latn"


class TestStats:
    def test_latency_and_audio_are_accumulated(self) -> None:
        session, _, _, _ = make_session()
        session._handle(make_utterance(seconds=2.0))
        session._handle(make_utterance(seconds=2.0))
        assert session.stats.utterances == 2
        assert session.stats.total_audio == pytest.approx(4.0)
        assert session.stats.average_latency >= 0
        assert len(session.stats.history) == 2

    def test_asr_time_is_recorded(self) -> None:
        session, results, _, _ = make_session()
        session._handle(make_utterance())
        assert results[0].timings.asr > 0

    def test_empty_stats_do_not_divide_by_zero(self) -> None:
        stats = SessionStats()
        assert stats.average_latency == 0.0
        assert stats.real_time_factor == 0.0

    def test_level_is_zero_without_a_capture_device(self) -> None:
        session, _, _, _ = make_session()
        assert session.level == 0.0


class TestWorkerLoop:
    def test_sentinel_stops_the_worker(self) -> None:
        session, _, _, _ = make_session()
        session._utterances.put(None)
        session._worker_loop()  # must return rather than block

    def test_worker_survives_a_failing_utterance(self) -> None:
        session, _, _, _ = make_session()

        def explode(_utterance):  # noqa: ANN001, ANN202
            raise RuntimeError("bad audio")

        session._handle = explode
        session._utterances.put(make_utterance())
        session._utterances.put(None)
        session._worker_loop()  # an exception must not kill the loop

    def test_enqueue_after_full_with_empty_queue_is_safe(self) -> None:
        session, _, _, _ = make_session(max_pending_utterances=1)
        session._utterances = queue.Queue(maxsize=1)
        session._enqueue(make_utterance())
        session._enqueue(make_utterance())
        assert session.stats.dropped_utterances == 1
