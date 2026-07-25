"""Non-blocking audio playback.

The original pipeline called ``pydub.playback.play()`` directly on the
processing thread. That blocks for the entire duration of the synthesised
clip, so while the translator was speaking it was not translating -- every
sentence spoken added its own length to the backlog.

Playback here runs on its own thread with a bounded queue, so synthesis and
speaking overlap with recognition.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["PlaybackWorker"]


class PlaybackWorker:
    """Plays float32 numpy clips in order, on a background thread."""

    def __init__(
        self,
        sample_rate: int = 22_050,
        device: int | None = None,
        max_queued: int = 4,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._queue: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(maxsize=max_queued)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._dropped = 0

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def dropped(self) -> int:
        """Clips discarded because playback could not keep up."""
        return self._dropped

    def start(self) -> PlaybackWorker:
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="playback", daemon=True)
        self._thread.start()
        return self

    def play(self, audio: np.ndarray, sample_rate: int | None = None) -> bool:
        """Queue a clip. Returns False if it was dropped due to backpressure."""
        if audio is None or len(audio) == 0:
            return False
        item = (np.asarray(audio, dtype=np.float32), sample_rate or self.sample_rate)
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self._dropped += 1
            logger.warning(
                "Playback queue full; dropped a clip to stay near real time "
                "(%d dropped so far).",
                self._dropped,
            )
            return False

    def wait_until_idle(self, timeout: float | None = None) -> None:
        """Block until every queued clip has finished playing."""
        try:
            self._queue.join()
        except Exception:  # pragma: no cover
            pass
        if timeout is not None and self._speaking.is_set():
            self._speaking.wait(timeout)

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            logger.warning("Playback unavailable (%s); audio output disabled.", exc)
            self._drain()
            return

        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                break
            audio, rate = item
            self._speaking.set()
            try:
                sd.play(audio, rate, device=self.device, blocking=True)
            except Exception:
                logger.warning("Playback failed", exc_info=True)
            finally:
                self._speaking.clear()
                self._queue.task_done()

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> PlaybackWorker:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
