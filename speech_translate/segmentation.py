"""Sentence-boundary buffering.

Feeding TTS one ASR fragment at a time produces choppy, badly-prosodied speech:
"I went to the" / "store yesterday" is spoken as two separate utterances with a
hard stop in between. Holding text until a sentence boundary gives the
synthesiser a full clause to work with.

The trade-off is latency, so the buffer also flushes on a timeout and on a
length cap -- a speaker who never pauses still gets output.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

__all__ = ["SentenceBuffer"]

# Sentence-final punctuation across the scripts we support: Latin, CJK
# (。！？), Arabic (؟ ،), Devanagari (।), Greek question mark (;).
_SENTENCE_END = re.compile(r"[.!?。！？।؟;]+[\"')\]}»”’]*\s*")
_ABBREVIATIONS = frozenset(
    {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "e.g.", "i.e."}
)


@dataclass
class SentenceBuffer:
    """Accumulates text fragments and releases complete sentences.

    Args:
        max_chars: flush even without punctuation past this length.
        max_wait: flush after this many seconds, so a trailing fragment is
            never stranded in the buffer.
    """

    max_chars: int = 240
    max_wait: float = 2.0
    _pending: str = field(default="", init=False)
    _first_added_at: float | None = field(default=None, init=False)

    @property
    def pending(self) -> str:
        """Text held back, useful for showing a live partial transcript."""
        return self._pending.strip()

    def add(self, text: str) -> list[str]:
        """Add a fragment; returns any complete sentences ready to speak."""
        text = (text or "").strip()
        if not text:
            return []
        if self._first_added_at is None:
            self._first_added_at = time.monotonic()
        self._pending = f"{self._pending} {text}".strip() if self._pending else text
        return self._extract()

    def _extract(self) -> list[str]:
        sentences: list[str] = []
        while True:
            match = self._find_boundary(self._pending)
            if match is None:
                break
            end = match.end()
            sentence = self._pending[:end].strip()
            if not sentence:
                break
            sentences.append(sentence)
            self._pending = self._pending[end:].lstrip()

        if len(self._pending) >= self.max_chars:
            sentences.append(self._pending.strip())
            self._pending = ""

        if sentences:
            self._first_added_at = time.monotonic() if self._pending else None
        return [s for s in sentences if s]

    @staticmethod
    def _find_boundary(text: str) -> re.Match[str] | None:
        """First sentence end that is not an abbreviation's full stop."""
        for match in _SENTENCE_END.finditer(text):
            head = text[: match.start() + 1]
            last_word = head.split()[-1].lower() if head.split() else ""
            if last_word in _ABBREVIATIONS:
                continue
            # A lone "." after a single letter is probably an initial (J. Smith).
            if len(last_word) == 2 and last_word[0].isalpha() and last_word[1] == ".":
                continue
            return match
        return None

    def expired(self, now: float | None = None) -> bool:
        """True when pending text has waited longer than ``max_wait``."""
        if not self._pending or self._first_added_at is None:
            return False
        return (now or time.monotonic()) - self._first_added_at >= self.max_wait

    def flush(self) -> list[str]:
        """Release whatever is pending, complete sentence or not."""
        text = self._pending.strip()
        self._pending = ""
        self._first_added_at = None
        return [text] if text else []

    def flush_if_expired(self, now: float | None = None) -> list[str]:
        return self.flush() if self.expired(now) else []

    def reset(self) -> None:
        self._pending = ""
        self._first_added_at = None
