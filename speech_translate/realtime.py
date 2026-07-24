"""The threaded real-time session.

Three stages run concurrently so none blocks the others:

* the **capture** thread owns the microphone and the VAD state machine,
* the **worker** thread runs ASR -> MT -> TTS,
* the **playback** thread speaks.

Backpressure is explicit. The utterance queue is bounded and drops the
*oldest* pending item when full: in a live translator, being current matters
more than being complete, and the original unbounded queue meant a slow
machine drifted further behind for as long as it ran.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from .audio.capture import MicrophoneCapture
from .audio.playback import PlaybackWorker
from .audio.vad import Utterance
from .config import Settings
from .pipeline import Pipeline, PipelineResult, StageTimings, timed
from .segmentation import SentenceBuffer

logger = logging.getLogger(__name__)

__all__ = ["RealtimeSession", "SessionStats"]


@dataclass
class SessionStats:
    utterances: int = 0
    spoken: int = 0
    dropped_utterances: int = 0
    dropped_clips: int = 0
    total_latency: float = 0.0
    total_audio: float = 0.0
    history: list[PipelineResult] = field(default_factory=list)

    @property
    def average_latency(self) -> float:
        return self.total_latency / self.utterances if self.utterances else 0.0

    @property
    def real_time_factor(self) -> float:
        return self.total_latency / self.total_audio if self.total_audio else 0.0


class RealtimeSession:
    """Wires microphone -> :class:`Pipeline` -> speakers."""

    def __init__(
        self,
        settings: Settings | None = None,
        pipeline: Pipeline | None = None,
        on_result: Callable[[PipelineResult], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        speak: bool = True,
    ) -> None:
        self.settings = settings or Settings()
        self.pipeline = pipeline or Pipeline(self.settings)
        self.on_result = on_result
        self.on_status = on_status
        self.speak = speak

        self.stats = SessionStats()
        self._utterances: queue.Queue[Utterance | None] = queue.Queue(
            maxsize=self.settings.max_pending_utterances
        )
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._capture: MicrophoneCapture | None = None
        self._playback: PlaybackWorker | None = None
        self._buffer = SentenceBuffer()
        self._last_language = self.settings.src

    # -- lifecycle -------------------------------------------------------
    def start(self) -> RealtimeSession:
        self._status("Loading models ...")
        self.pipeline.warmup()

        self._capture = MicrophoneCapture(self.settings.audio, self.settings.vad)
        self._capture.start()

        if self.speak:
            self._playback = PlaybackWorker(
                sample_rate=self.pipeline.tts.sample_rate,
                device=self.settings.audio.output_device,
            ).start()

        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._worker_loop, name="worker", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._status("Listening. Speak into the microphone; Ctrl+C to stop.")
        return self

    def stop(self) -> SessionStats:
        self._stop.set()
        if self._capture is not None:
            self._capture.stop()
        try:
            self._utterances.put_nowait(None)
        except queue.Full:
            pass
        for thread in self._threads:
            thread.join(timeout=3.0)
        if self._playback is not None:
            self._playback.stop()
            self.stats.dropped_clips = self._playback.dropped
        self.pipeline.close()
        return self.stats

    def wait(self) -> None:
        """Block until interrupted; the caller handles KeyboardInterrupt."""
        while not self._stop.is_set():
            self._stop.wait(0.2)

    @property
    def level(self) -> float:
        """Live microphone level in [0, 1], for a UI meter."""
        return self._capture.level if self._capture else 0.0

    # -- threads ---------------------------------------------------------
    def _capture_loop(self) -> None:
        assert self._capture is not None
        try:
            for utterance in self._capture.utterances():
                if self._stop.is_set():
                    break
                self._enqueue(utterance)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.exception("Capture stopped: %s", exc)
            self._status(f"Capture error: {exc}")
            self._stop.set()

    def _enqueue(self, utterance: Utterance) -> None:
        try:
            self._utterances.put_nowait(utterance)
        except queue.Full:
            try:
                self._utterances.get_nowait()
                self.stats.dropped_utterances += 1
                self._utterances.put_nowait(utterance)
                logger.warning(
                    "Translation is behind real time; dropped an older utterance "
                    "(%d total).",
                    self.stats.dropped_utterances,
                )
            except (queue.Empty, queue.Full):  # pragma: no cover - race
                pass

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                utterance = self._utterances.get(timeout=0.2)
            except queue.Empty:
                self._flush_expired_buffer()
                continue
            if utterance is None:
                break
            try:
                self._handle(utterance)
            except Exception as exc:
                logger.exception("Failed to process an utterance: %s", exc)
                self._status(f"Error: {exc}")
        self._flush_buffer(force=True)

    # -- per-utterance work ---------------------------------------------
    def _handle(self, utterance: Utterance) -> None:
        timings = StageTimings()
        with timed(timings, "asr"):
            transcript = self.pipeline.transcribe(utterance.audio)
        if not transcript.text:
            return

        self.stats.utterances += 1
        self.stats.total_audio += utterance.duration
        self._last_language = transcript.language

        if not self.settings.buffer_to_sentences:
            self._translate_and_emit(
                transcript.text, transcript.language, utterance.duration, timings
            )
            return

        # A truncated utterance means the speaker was cut off mid-sentence by
        # the length cap, so hold the fragment and wait for the rest. A natural
        # pause is already a good boundary, so flush immediately.
        if utterance.truncated:
            ready = self._buffer.add(transcript.text)
        else:
            self._buffer.add(transcript.text)
            ready = self._buffer.flush()

        for index, sentence in enumerate(ready):
            # ASR cost is charged once, to the first sentence it produced.
            self._translate_and_emit(
                sentence,
                transcript.language,
                utterance.duration if index == 0 else 0.0,
                timings if index == 0 else StageTimings(),
            )

    def _flush_expired_buffer(self) -> None:
        for sentence in self._buffer.flush_if_expired():
            self._translate_and_emit(sentence, self._last_language, 0.0, StageTimings())

    def _flush_buffer(self, force: bool = False) -> None:
        if force:
            for sentence in self._buffer.flush():
                try:
                    self._translate_and_emit(
                        sentence, self._last_language, 0.0, StageTimings()
                    )
                except Exception:  # pragma: no cover - shutdown best effort
                    logger.debug("Failed to flush the sentence buffer", exc_info=True)

    def _translate_and_emit(
        self,
        text: str,
        source_language: str,
        audio_duration: float,
        timings: StageTimings,
    ) -> None:
        self._last_language = source_language
        result = PipelineResult(
            transcript=text,
            source_language=source_language,
            target_language=self.settings.tgt,
            audio_duration=audio_duration,
            timings=timings,
        )

        if source_language == self.settings.tgt:
            result.translation = text
            result.skipped_reason = "source and target language are the same"
        else:
            with timed(timings, "mt"):
                result.translation = self.pipeline.translator.translate(
                    text, source_language, self.settings.tgt
                ).text

        if self.speak and result.translation:
            with timed(timings, "tts"):
                speech = self.pipeline.synthesize(result.translation)
            result.speech = speech
            if speech and self._playback is not None:
                if self._playback.play(speech.audio, speech.sample_rate):
                    self.stats.spoken += 1

        self.stats.total_latency += timings.total
        self.stats.history.append(result)
        if self.on_result is not None:
            self.on_result(result)

    def _status(self, message: str) -> None:
        logger.info(message)
        if self.on_status is not None:
            self.on_status(message)

    def __enter__(self) -> RealtimeSession:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
