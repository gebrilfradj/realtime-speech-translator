"""Voice activity detection and utterance segmentation.

The original pipeline cut the microphone into fixed 3-second chunks. That is
bad in three separate ways: it slices words in half at chunk boundaries, it
pays the full ASR cost on pure silence, and it adds up to 3 s of latency even
when the speaker stopped talking immediately.

This module replaces it with a speech-triggered state machine. Segmentation
here is intentionally cheap (adaptive RMS energy with hangover) because it runs
on every 30 ms frame in the capture callback; the precise trimming is left to
faster-whisper's built-in Silero VAD, which runs once per utterance inside the
model. Cheap gate first, accurate gate second.

Everything is a pure state machine over numpy frames, so it is fully testable
without a microphone.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..config import AudioSettings, VADSettings
from .wav import rms

logger = logging.getLogger(__name__)

__all__ = ["EnergyVAD", "Utterance", "UtteranceSegmenter"]


class _State(Enum):
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass
class Utterance:
    """One contiguous stretch of speech, ready for ASR."""

    audio: np.ndarray
    sample_rate: int
    #: Seconds of speech (excluding pre-roll padding).
    duration: float
    #: Peak RMS observed, handy as a UI mic-level indicator.
    peak_level: float
    #: True when emitted because ``max_utterance_ms`` was hit rather than
    #: because the speaker actually paused.
    truncated: bool = False


class EnergyVAD:
    """RMS gate whose threshold floats above the observed noise floor.

    A fixed threshold fails the moment you move from a quiet room to a cafe, so
    the background level is estimated with *minimum statistics*: the noise floor
    is a low percentile of recent frame energies. Speech is bursty, so the
    bottom decile of a few seconds of audio reflects the background even while
    someone is talking.

    Two failure modes this specifically avoids, both found by the test-suite:

    * **Seeding from the first frame.** If the user is already speaking when
      capture starts, the floor is set to *speech* level and the gate never
      opens. During a short warm-up the static threshold is used instead.
    * **Adapting only on silence.** In a continuously noisy room every frame
      looks like speech, so the floor never updates and the gate jams open.
      A percentile over all frames has no such dependency.
    """

    #: Frames of history used for the estimate (~3 s at 30 ms).
    WINDOW_FRAMES = 100
    #: Until this many frames have been seen, trust the configured threshold.
    WARMUP_FRAMES = 33
    #: Percentile of the window treated as background.
    NOISE_PERCENTILE = 10.0
    #: Speech must exceed the background by this factor.
    MARGIN = 3.0

    def __init__(self, settings: VADSettings | None = None) -> None:
        self.settings = settings or VADSettings()
        self._levels: deque[float] = deque(maxlen=self.WINDOW_FRAMES)
        self._noise_floor = 0.0

    @property
    def noise_floor(self) -> float:
        """Current background-level estimate."""
        return self._noise_floor

    @property
    def threshold(self) -> float:
        """Current absolute decision threshold."""
        return max(self.settings.threshold, self._noise_floor * self.MARGIN)

    def reset(self) -> None:
        self._levels.clear()
        self._noise_floor = 0.0

    def is_speech(self, frame: np.ndarray) -> bool:
        level = rms(frame)
        self._levels.append(level)
        if len(self._levels) >= self.WARMUP_FRAMES:
            self._noise_floor = float(
                np.percentile(np.fromiter(self._levels, dtype=np.float64), self.NOISE_PERCENTILE)
            )
        return level > self.threshold


class UtteranceSegmenter:
    """Turns a stream of fixed-size frames into variable-length utterances.

    Usage::

        segmenter = UtteranceSegmenter(audio_settings, vad_settings)
        for frame in frames:
            for utterance in segmenter.push(frame):
                ...
        for utterance in segmenter.flush():
            ...
    """

    def __init__(
        self,
        audio: AudioSettings | None = None,
        vad: VADSettings | None = None,
        detector: EnergyVAD | None = None,
    ) -> None:
        self.audio = audio or AudioSettings()
        self.settings = vad or VADSettings()
        self.detector = detector or EnergyVAD(self.settings)

        frame_ms = self.audio.block_ms
        self._frames_for_speech = max(1, round(self.settings.min_speech_ms / frame_ms))
        self._frames_for_silence = max(1, round(self.settings.min_silence_ms / frame_ms))
        self._max_frames = max(1, round(self.settings.max_utterance_ms / frame_ms))
        pre_roll_frames = max(0, round(self.settings.pre_roll_ms / frame_ms))

        self._state = _State.SILENCE
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames or 1)
        self._buffer: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._peak = 0.0

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def is_speaking(self) -> bool:
        return self._state is _State.SPEECH

    def push(self, frame: np.ndarray) -> Iterator[Utterance]:
        """Feed one frame; yields an utterance when one completes."""
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return

        if not self.settings.enabled:
            # VAD off: behave like the old fixed-window chunker.
            self._buffer.append(frame)
            self._peak = max(self._peak, rms(frame))
            if len(self._buffer) >= self._max_frames:
                yield self._emit(truncated=True)
            return

        speech = self.detector.is_speech(frame)
        self._peak = max(self._peak, rms(frame))

        if self._state is _State.SILENCE:
            self._pre_roll.append(frame)
            if speech:
                self._speech_frames += 1
                if self._speech_frames >= self._frames_for_speech:
                    self._state = _State.SPEECH
                    self._buffer = list(self._pre_roll)
                    self._pre_roll.clear()
                    self._silence_frames = 0
            else:
                self._speech_frames = 0
            return

        # _State.SPEECH
        self._buffer.append(frame)
        if speech:
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            if self._silence_frames >= self._frames_for_silence:
                yield self._emit(truncated=False)
                return
        if len(self._buffer) >= self._max_frames:
            yield self._emit(truncated=True)

    def flush(self) -> Iterator[Utterance]:
        """Emit any buffered speech, e.g. when the stream stops."""
        if self._buffer and self._has_enough_speech():
            yield self._emit(truncated=False)
        else:
            self._reset_buffers()

    def _has_enough_speech(self) -> bool:
        speech_ms = len(self._buffer) * self.audio.block_ms
        return speech_ms >= self.settings.min_speech_ms

    def _emit(self, *, truncated: bool) -> Utterance:
        audio = (
            np.concatenate(self._buffer)
            if self._buffer
            else np.zeros(0, dtype=np.float32)
        )
        # Trailing silence is dead weight for the ASR; keep a little for
        # natural word endings but drop the rest.
        if not truncated and self._silence_frames > 1:
            keep = max(0, self._silence_frames - 1) * self.audio.block_frames
            if keep and keep < audio.size:
                audio = audio[: audio.size - keep]
        utterance = Utterance(
            audio=audio,
            sample_rate=self.audio.sample_rate,
            duration=audio.size / self.audio.sample_rate,
            peak_level=self._peak,
            truncated=truncated,
        )
        self._reset_buffers()
        if truncated:
            # A truncated utterance means the speaker is mid-sentence, so stay
            # in SPEECH and keep capturing rather than requiring a re-trigger.
            self._state = _State.SPEECH
        return utterance

    def _reset_buffers(self) -> None:
        self._buffer = []
        self._pre_roll.clear()
        self._state = _State.SILENCE
        self._speech_frames = 0
        self._silence_frames = 0
        self._peak = 0.0


def segment_audio(
    audio: np.ndarray,
    audio_settings: AudioSettings | None = None,
    vad_settings: VADSettings | None = None,
) -> list[Utterance]:
    """Run the segmenter over a complete waveform (used by tests and the UI)."""
    audio_settings = audio_settings or AudioSettings()
    segmenter = UtteranceSegmenter(audio_settings, vad_settings)
    block = audio_settings.block_frames
    out: list[Utterance] = []
    for start in range(0, len(audio), block):
        frame = audio[start : start + block]
        if frame.size < block:
            frame = np.pad(frame, (0, block - frame.size))
        out.extend(segmenter.push(frame))
    out.extend(segmenter.flush())
    return out
