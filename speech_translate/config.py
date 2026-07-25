"""Configuration for the speech-translation cascade.

All tunables live here as plain dataclasses. Nothing in this module reads
``sys.argv``, opens a device, or touches the network -- importing
``speech_translate.config`` is free and side-effect-free, which is what makes
the package testable and safe to import from a web worker.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

from .languages import AUTO, resolve_language

Device = Literal["auto", "cpu", "cuda"]
TTSBackendName = Literal["auto", "piper", "system", "none"]


def _default_device() -> Device:
    return "auto"


@dataclass
class AudioSettings:
    """Capture/playback format. 16 kHz mono is what Whisper wants natively."""

    sample_rate: int = 16_000
    channels: int = 1
    #: Size of one capture block. 30 ms is the granularity VAD operates on.
    block_ms: int = 30
    input_device: int | None = None
    output_device: int | None = None

    @property
    def block_frames(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)


@dataclass
class VADSettings:
    """Voice-activity detection: when does an utterance start and stop?

    Replaces the old fixed 3-second chunking, which cut words in half and paid
    full ASR cost on silence.
    """

    enabled: bool = True
    #: RMS energy threshold, adaptively raised to the observed noise floor.
    threshold: float = 0.015
    #: Ignore blips shorter than this; they are keyboard clicks, not speech.
    min_speech_ms: int = 250
    #: Silence this long ends the utterance.
    min_silence_ms: int = 600
    #: Hard cap so a monologue still produces output.
    max_utterance_ms: int = 12_000
    #: Audio kept from *before* the trigger so we don't clip the first phoneme.
    pre_roll_ms: int = 300


@dataclass
class ASRSettings:
    """faster-whisper (CTranslate2) settings.

    ``base`` is the default rather than ``small`` because it is the largest
    model that stays comfortably ahead of live speech on a CPU: measured
    end-to-end RTF is 0.61 with ``base`` versus 1.09 with ``small`` on the same
    machine and clip. Accuracy is one flag away (``--asr-model small``).
    """

    model: str = "base"
    device: Device = field(default_factory=_default_device)
    #: ``auto`` picks int8 on CPU and float16 on GPU.
    compute_type: str = "auto"
    #: Greedy decoding: beams buy little for short utterances and cost latency.
    beam_size: int = 1
    #: faster-whisper's built-in Silero VAD, applied inside the model.
    vad_filter: bool = True
    #: Drop segments the model itself flags as probably-not-speech.
    no_speech_threshold: float = 0.6
    #: Drop low-confidence segments (a classic hallucination signature).
    log_prob_threshold: float = -1.0
    #: Whisper loops ("Thank you. Thank you. Thank you.") when it can see its
    #: own previous output. In a chunked real-time loop it is pure downside.
    condition_on_previous_text: bool = False
    cpu_threads: int = 0

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if _cuda_available() else "cpu"

    def resolved_compute_type(self) -> str:
        if self.compute_type != "auto":
            return self.compute_type
        return "float16" if self.resolved_device() == "cuda" else "int8"


@dataclass
class MTSettings:
    """NLLB-200 settings."""

    model: str = "facebook/nllb-200-distilled-600M"
    device: Device = field(default_factory=_default_device)
    num_beams: int = 1
    max_new_tokens: int = 256

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if _cuda_available() else "cpu"


@dataclass
class TTSSettings:
    """Speech synthesis settings.

    ``backend='auto'`` prefers Piper and degrades to the OS voice, then to
    subtitles-only, so the pipeline never hard-fails just because a voice is
    missing.
    """

    backend: TTSBackendName = "auto"
    #: Explicit Piper voice name (e.g. ``en_US-lessac-medium``).
    voice: str | None = None
    #: >1.0 is slower speech. Piper calls this ``length_scale``.
    length_scale: float = 1.0
    use_cuda: bool = False
    download_dir: str | None = None


@dataclass
class Settings:
    """Everything one run of the pipeline needs."""

    src: str = AUTO
    tgt: str = "spa_Latn"
    audio: AudioSettings = field(default_factory=AudioSettings)
    vad: VADSettings = field(default_factory=VADSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    mt: MTSettings = field(default_factory=MTSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    #: Drop the oldest pending utterance when the queue is this deep. Staying
    #: near-live matters more than translating every backlogged utterance.
    max_pending_utterances: int = 4
    #: Hold partial text until sentence-final punctuation before synthesising.
    buffer_to_sentences: bool = True

    def __post_init__(self) -> None:
        self.src = resolve_language(self.src, allow_auto=True)
        self.tgt = resolve_language(self.tgt)

    def with_languages(self, src: str | None = None, tgt: str | None = None) -> Settings:
        """Copy with different languages, validating them."""
        return replace(
            self,
            src=resolve_language(src, allow_auto=True) if src else self.src,
            tgt=resolve_language(tgt) if tgt else self.tgt,
        )


def _cuda_available() -> bool:
    """Check for CUDA without importing torch at module import time."""
    if os.environ.get("SPEECH_TRANSLATE_FORCE_CPU"):
        return False
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
