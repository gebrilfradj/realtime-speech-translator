"""Microphone capture.

A ``sounddevice`` input stream feeds a bounded queue; the consumer pulls frames
and runs them through :class:`~speech_translate.audio.vad.UtteranceSegmenter`.

The queue is bounded on purpose. The old code used an unbounded
``queue.Queue`` with a blocking playback call downstream, so if translation
fell behind real time the backlog grew without limit and the "real-time"
translator drifted further behind for as long as it ran. Here, overflow drops
the *oldest* audio and says so.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator

import numpy as np

from ..config import AudioSettings, VADSettings
from .devices import AudioUnavailableError, resolve_input_device
from .vad import Utterance, UtteranceSegmenter
from .wav import to_mono

logger = logging.getLogger(__name__)

__all__ = ["MicrophoneCapture"]


class MicrophoneCapture:
    """Yields :class:`Utterance` objects from the default (or chosen) mic."""

    def __init__(
        self,
        audio: AudioSettings | None = None,
        vad: VADSettings | None = None,
        max_queued_frames: int = 200,  # ~6 s at 30 ms frames
    ) -> None:
        self.audio = audio or AudioSettings()
        self.vad = vad or VADSettings()
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max_queued_frames)
        self._stop = threading.Event()
        self._stream = None
        self._dropped_frames = 0
        self._level = 0.0

    @property
    def level(self) -> float:
        """Most recent frame RMS, for a live mic-level meter."""
        return self._level

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("Audio callback status: %s", status)
        mono = to_mono(np.asarray(indata, dtype=np.float32)).copy()
        try:
            self._frames.put_nowait(mono)
        except queue.Full:
            # Drop the oldest frame so we stay near the live edge.
            self._dropped_frames += 1
            try:
                self._frames.get_nowait()
                self._frames.put_nowait(mono)
            except (queue.Empty, queue.Full):
                pass

    def start(self) -> None:
        import sounddevice as sd

        device = resolve_input_device(self.audio.input_device)
        try:
            self._stream = sd.InputStream(
                samplerate=self.audio.sample_rate,
                channels=self.audio.channels,
                dtype="float32",
                blocksize=self.audio.block_frames,
                device=device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            raise AudioUnavailableError(
                f"Could not open the microphone ({exc}). "
                "Run with --list-devices to see what is available."
            ) from exc
        logger.info(
            "Microphone open (device=%s, %d Hz, %d ms blocks)",
            device if device is not None else "default",
            self.audio.sample_rate,
            self.audio.block_ms,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover - teardown best effort
                logger.debug("Error closing audio stream", exc_info=True)
            self._stream = None
        self._frames.put(None)

    def utterances(self) -> Iterator[Utterance]:
        """Blocking generator of complete utterances."""
        segmenter = UtteranceSegmenter(self.audio, self.vad)
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                break
            self._level = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
            yield from segmenter.push(frame)
        yield from segmenter.flush()

    def __enter__(self) -> MicrophoneCapture:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
