"""VAD segmentation: the replacement for fixed 3-second chunking."""

from __future__ import annotations

import numpy as np
import pytest

from speech_translate.audio.vad import EnergyVAD, UtteranceSegmenter, segment_audio
from speech_translate.config import AudioSettings, VADSettings

from .conftest import quiet, speech_like, tone


@pytest.fixture
def audio_settings() -> AudioSettings:
    return AudioSettings(sample_rate=16_000, block_ms=30)


@pytest.fixture
def vad_settings() -> VADSettings:
    return VADSettings(
        enabled=True,
        threshold=0.01,
        min_speech_ms=90,
        min_silence_ms=300,
        max_utterance_ms=3_000,
        pre_roll_ms=90,
    )


class TestEnergyVAD:
    def test_silence_is_not_speech(self) -> None:
        detector = EnergyVAD(VADSettings(threshold=0.01))
        assert not detector.is_speech(quiet(0.03))

    def test_loud_audio_is_speech(self) -> None:
        detector = EnergyVAD(VADSettings(threshold=0.01))
        assert detector.is_speech(tone(0.03, amplitude=0.5))

    def test_speech_is_detected_on_the_very_first_frame(self) -> None:
        """Regression: seeding the noise floor from frame 1 swallowed the
        first word whenever the user was already speaking at start-up."""
        detector = EnergyVAD(VADSettings(threshold=0.01))
        assert detector.is_speech(tone(0.03, amplitude=0.5))

    def test_threshold_floats_above_the_noise_floor(self) -> None:
        """A constant hiss must stop registering as speech once learned."""
        detector = EnergyVAD(VADSettings(threshold=0.001))
        rng = np.random.default_rng(0)
        noise = (rng.normal(0, 0.02, 480)).astype(np.float32)
        for _ in range(200):
            detector.is_speech(noise)
        assert detector.threshold > 0.001
        assert not detector.is_speech(noise)

    def test_speech_still_detected_over_a_noisy_floor(self) -> None:
        detector = EnergyVAD(VADSettings(threshold=0.001))
        rng = np.random.default_rng(1)
        noise = (rng.normal(0, 0.01, 480)).astype(np.float32)
        for _ in range(200):
            detector.is_speech(noise)
        assert detector.is_speech(tone(0.03, amplitude=0.6))

    def test_continuous_noise_does_not_jam_the_gate_open(self) -> None:
        """With silence-only adaptation, a constantly noisy room meant every
        frame looked like speech and the floor never updated."""
        detector = EnergyVAD(VADSettings(threshold=0.001))
        rng = np.random.default_rng(2)
        noise = (rng.normal(0, 0.05, 480)).astype(np.float32)
        results = [detector.is_speech(noise) for _ in range(400)]
        assert not results[-1], "the gate must eventually close on steady noise"


class TestUtteranceSegmenter:
    def _feed(self, segmenter: UtteranceSegmenter, audio: np.ndarray, block: int) -> list:
        out = []
        for start in range(0, len(audio), block):
            frame = audio[start : start + block]
            if frame.size < block:
                frame = np.pad(frame, (0, block - frame.size))
            out.extend(segmenter.push(frame))
        return out

    def test_pure_silence_produces_nothing(self, audio_settings, vad_settings) -> None:
        """The old chunker sent 3s of silence to Whisper every 3 seconds."""
        segmenter = UtteranceSegmenter(audio_settings, vad_settings)
        emitted = self._feed(segmenter, quiet(5.0), audio_settings.block_frames)
        emitted.extend(segmenter.flush())
        assert emitted == []

    def test_one_burst_of_speech_yields_one_utterance(self, audio_settings, vad_settings) -> None:
        segmenter = UtteranceSegmenter(audio_settings, vad_settings)
        audio = np.concatenate([quiet(0.3), tone(1.0), quiet(1.0)])
        emitted = self._feed(segmenter, audio, audio_settings.block_frames)
        assert len(emitted) == 1
        assert emitted[0].duration > 0.5
        assert not emitted[0].truncated

    def test_two_bursts_separated_by_a_pause_yield_two_utterances(
        self, audio_settings, vad_settings
    ) -> None:
        segmenter = UtteranceSegmenter(audio_settings, vad_settings)
        audio = np.concatenate(
            [quiet(0.3), tone(0.8), quiet(0.8), tone(0.8), quiet(0.8)]
        )
        emitted = self._feed(segmenter, audio, audio_settings.block_frames)
        assert len(emitted) == 2

    def test_long_speech_is_truncated_and_flagged(self, audio_settings, vad_settings) -> None:
        segmenter = UtteranceSegmenter(audio_settings, vad_settings)
        audio = np.concatenate([quiet(0.2), speech_like(8.0)])
        emitted = self._feed(segmenter, audio, audio_settings.block_frames)
        assert emitted, "a long monologue must still produce output"
        assert emitted[0].truncated
        assert emitted[0].duration <= vad_settings.max_utterance_ms / 1000 + 0.1

    def test_sustained_tone_is_treated_as_background_noise(
        self, audio_settings, vad_settings
    ) -> None:
        """A constant hum is noise, not speech; the VAD must learn that."""
        segmenter = UtteranceSegmenter(audio_settings, vad_settings)
        audio = np.concatenate([quiet(0.2), tone(10.0, amplitude=0.05)])
        emitted = self._feed(segmenter, audio, audio_settings.block_frames)
        emitted.extend(segmenter.flush())
        total = sum(u.duration for u in emitted)
        assert total < 4.0, "a 10 s hum must not be captured as 10 s of speech"

    def test_pre_roll_keeps_the_start_of_the_word(self, audio_settings, vad_settings) -> None:
        """Without pre-roll the first phoneme is lost while VAD is triggering."""
        segmenter = UtteranceSegmenter(audio_settings, vad_settings)
        audio = np.concatenate([quiet(0.5), tone(1.0), quiet(1.0)])
        emitted = self._feed(segmenter, audio, audio_settings.block_frames)
        assert len(emitted) == 1
        # Speech is 1.0 s; the captured segment must be at least that long
        # because pre-roll frames are prepended.
        assert emitted[0].duration >= 1.0

    def test_flush_emits_buffered_speech(self, audio_settings, vad_settings) -> None:
        segmenter = UtteranceSegmenter(audio_settings, vad_settings)
        self._feed(segmenter, np.concatenate([quiet(0.2), tone(1.0)]), audio_settings.block_frames)
        emitted = list(segmenter.flush())
        assert len(emitted) == 1

    def test_flush_on_empty_buffer_is_safe(self, audio_settings, vad_settings) -> None:
        segmenter = UtteranceSegmenter(audio_settings, vad_settings)
        assert list(segmenter.flush()) == []

    def test_state_reported_for_ui(self, audio_settings, vad_settings) -> None:
        segmenter = UtteranceSegmenter(audio_settings, vad_settings)
        assert segmenter.state == "silence"
        assert not segmenter.is_speaking
        self._feed(segmenter, tone(1.0), audio_settings.block_frames)
        assert segmenter.is_speaking

    def test_disabled_vad_falls_back_to_fixed_windows(self, audio_settings) -> None:
        settings = VADSettings(enabled=False, max_utterance_ms=1_000)
        segmenter = UtteranceSegmenter(audio_settings, settings)
        emitted = self._feed(segmenter, quiet(3.0), audio_settings.block_frames)
        assert len(emitted) >= 2
        assert all(u.truncated for u in emitted)

    def test_empty_frames_are_ignored(self, audio_settings, vad_settings) -> None:
        segmenter = UtteranceSegmenter(audio_settings, vad_settings)
        assert list(segmenter.push(np.zeros(0, dtype=np.float32))) == []


class TestSegmentAudioHelper:
    def test_extracts_speech_from_a_waveform(self, audio_settings, vad_settings) -> None:
        audio = np.concatenate([quiet(0.3), tone(1.0), quiet(1.0)])
        utterances = segment_audio(audio, audio_settings, vad_settings)
        assert len(utterances) == 1

    def test_peak_level_is_reported(self, audio_settings, vad_settings) -> None:
        audio = np.concatenate([quiet(0.3), tone(1.0, amplitude=0.4), quiet(1.0)])
        utterances = segment_audio(audio, audio_settings, vad_settings)
        assert 0 < utterances[0].peak_level <= 1.0
