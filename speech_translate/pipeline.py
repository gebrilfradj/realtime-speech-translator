"""The speech-translation cascade as one importable object.

:class:`Pipeline` is stateless with respect to audio transport: give it a
waveform, get back transcript, translation, synthesised speech and per-stage
timings. The CLI, the web UI and the benchmark all drive this same class, so
none of them can drift from each other.

Importing this module loads no models and opens no devices; models load on
first use, or eagerly via :meth:`Pipeline.warmup`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .asr import FasterWhisperASR, SpeechRecognizer, Transcript
from .config import Settings
from .languages import AUTO, language_name
from .mt import NLLBTranslator, Translator
from .segmentation import SentenceBuffer
from .tts import SpeechAudio, TTSBackend, TTSUnavailableError, create_tts_backend

logger = logging.getLogger(__name__)

__all__ = ["Pipeline", "PipelineResult", "StageTimings", "timed"]


@contextmanager
def timed(timings: StageTimings, stage: str) -> Iterator[None]:
    """Record wall-clock seconds for one stage onto ``timings``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        setattr(timings, stage, getattr(timings, stage) + time.perf_counter() - start)


@dataclass
class StageTimings:
    """Wall-clock seconds per stage."""

    asr: float = 0.0
    mt: float = 0.0
    tts: float = 0.0

    @property
    def total(self) -> float:
        return self.asr + self.mt + self.tts

    def as_dict(self) -> dict[str, float]:
        return {"asr": self.asr, "mt": self.mt, "tts": self.tts, "total": self.total}


@dataclass
class PipelineResult:
    """Everything one utterance produced."""

    transcript: str = ""
    translation: str = ""
    source_language: str = AUTO
    target_language: str = ""
    speech: SpeechAudio | None = None
    timings: StageTimings = field(default_factory=StageTimings)
    #: Seconds of input audio, used for the real-time factor.
    audio_duration: float = 0.0
    skipped_reason: str = ""

    @property
    def real_time_factor(self) -> float:
        """<1.0 means the cascade runs faster than the speech it processes."""
        if self.audio_duration <= 0:
            return 0.0
        return self.timings.total / self.audio_duration

    @property
    def source_language_name(self) -> str:
        return language_name(self.source_language)

    @property
    def target_language_name(self) -> str:
        return language_name(self.target_language)

    def __bool__(self) -> bool:
        return bool(self.translation or self.transcript)


class Pipeline:
    """ASR -> MT -> TTS, with the components injectable for testing."""

    def __init__(
        self,
        settings: Settings | None = None,
        recognizer: SpeechRecognizer | None = None,
        translator: Translator | None = None,
        tts: TTSBackend | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._recognizer = recognizer
        self._translator = translator
        self._tts = tts

    # -- lazily constructed components ----------------------------------
    @property
    def recognizer(self) -> SpeechRecognizer:
        if self._recognizer is None:
            self._recognizer = FasterWhisperASR(self.settings.asr)
        return self._recognizer

    @property
    def translator(self) -> Translator:
        if self._translator is None:
            self._translator = NLLBTranslator(self.settings.mt)
        return self._translator

    @property
    def tts(self) -> TTSBackend:
        if self._tts is None:
            self._tts = create_tts_backend(self.settings.tgt, self.settings.tts)
        return self._tts

    def warmup(self, sample_rate: int = 16_000) -> Pipeline:
        """Load every model and run one tiny inference.

        First inference includes lazy graph construction and memory allocation,
        which would otherwise be charged to the first real utterance and make
        any latency measurement meaningless.
        """
        logger.info("Warming up models ...")
        started = time.perf_counter()
        for component in (self.recognizer, self.translator, self.tts):
            loader = getattr(component, "load", None)
            if callable(loader):
                loader()
        silence = np.zeros(sample_rate // 2, dtype=np.float32)
        try:
            self.recognizer.transcribe(silence, self.settings.src)
        except Exception:
            logger.debug("Warm-up transcription failed (harmless)", exc_info=True)
        try:
            self.translator.translate("Hello.", "eng_Latn", self.settings.tgt)
        except Exception:
            logger.debug("Warm-up translation failed (harmless)", exc_info=True)
        try:
            # Piper builds its ONNX graph on the first synthesis, which would
            # otherwise add seconds to the first thing the user actually says.
            self.tts.synthesize("Hello.")
        except Exception:
            logger.debug("Warm-up synthesis failed (harmless)", exc_info=True)
        logger.info("Warm-up finished in %.1f s", time.perf_counter() - started)
        return self

    # -- the actual work -------------------------------------------------
    def transcribe(self, audio: np.ndarray) -> Transcript:
        return self.recognizer.transcribe(audio, self.settings.src)

    def process(self, audio: np.ndarray, *, speak: bool = True) -> PipelineResult:
        """Run one utterance end to end."""
        timings = StageTimings()
        duration = len(audio) / self.settings.audio.sample_rate
        result = PipelineResult(
            target_language=self.settings.tgt,
            audio_duration=duration,
            timings=timings,
        )

        with timed(timings, "asr"):
            transcript = self.recognizer.transcribe(audio, self.settings.src)

        result.transcript = transcript.text
        # This is the ``--src auto`` fix: the detected language, not the literal
        # string "auto", is what flows into the translator.
        result.source_language = transcript.language

        if not transcript.text:
            result.skipped_reason = "no speech detected"
            return result

        if transcript.language == self.settings.tgt:
            # Speaking the target language already; echoing it back is noise.
            result.translation = transcript.text
            result.skipped_reason = "source and target language are the same"
            return result

        with timed(timings, "mt"):
            translation = self.translator.translate(
                transcript.text, transcript.language, self.settings.tgt
            )
        result.translation = translation.text

        if speak and translation.text:
            with timed(timings, "tts"):
                result.speech = self.synthesize(translation.text)

        return result

    def synthesize(self, text: str) -> SpeechAudio | None:
        """Synthesise, degrading to subtitles-only if the backend fails.

        A speech engine that dies mid-session must not take translation with
        it. The backend is replaced with the null one so the failure is paid
        once rather than on every subsequent utterance.
        """
        from .tts.base import NullTTS

        try:
            return self.tts.synthesize(text)
        except TTSUnavailableError as exc:
            logger.warning(
                "Speech synthesis failed (%s); continuing in subtitles-only mode.", exc
            )
        except Exception:
            logger.warning(
                "Speech synthesis raised unexpectedly; continuing in subtitles-only mode.",
                exc_info=True,
            )
        try:
            self._tts.close()
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to close the speech backend", exc_info=True)
        self._tts = NullTTS(reason="the previous backend failed")
        return None

    def process_file(self, path: str, *, speak: bool = True) -> PipelineResult:
        """Run one audio file end to end (any format soundfile can read)."""
        from .audio.wav import read_wav

        audio, _ = read_wav(path, target_rate=self.settings.audio.sample_rate)
        return self.process(audio, speak=speak)

    def new_sentence_buffer(self) -> SentenceBuffer:
        return SentenceBuffer()

    def close(self) -> None:
        if self._tts is not None:
            self._tts.close()
