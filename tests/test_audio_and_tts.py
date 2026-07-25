"""WAV I/O, resampling and the TTS backend factory."""

from __future__ import annotations

import numpy as np
import pytest

from speech_translate.audio.wav import (
    float_to_int16,
    int16_to_float,
    read_wav,
    resample,
    rms,
    to_mono,
    write_wav,
)
from speech_translate.config import TTSSettings
from speech_translate.tts import (
    NullTTS,
    PiperTTS,
    SystemTTS,
    TTSUnavailableError,
    available_backends,
    create_tts_backend,
)
from speech_translate.tts.base import SpeechAudio

from .conftest import tone


class TestWavIO:
    def test_round_trip_preserves_audio(self, tmp_path) -> None:
        original = tone(0.5)
        path = tmp_path / "clip.wav"
        write_wav(path, original, 16_000)
        loaded, rate = read_wav(path)
        assert rate == 16_000
        assert loaded.shape == original.shape
        # 16-bit PCM quantisation is the only expected loss.
        assert np.max(np.abs(loaded - original)) < 1e-3

    def test_read_resamples_to_the_target_rate(self, tmp_path) -> None:
        path = tmp_path / "clip.wav"
        write_wav(path, tone(1.0, sample_rate=48_000), 48_000)
        loaded, rate = read_wav(path, target_rate=16_000)
        assert rate == 16_000
        assert len(loaded) == pytest.approx(16_000, rel=0.01)

    def test_write_creates_missing_directories(self, tmp_path) -> None:
        path = tmp_path / "nested" / "dir" / "clip.wav"
        write_wav(path, tone(0.1), 16_000)
        assert path.exists()


class TestAudioHelpers:
    def test_to_mono_averages_channels(self) -> None:
        stereo = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert to_mono(stereo).tolist() == [0.5, 0.5]

    def test_to_mono_passes_through_mono(self) -> None:
        mono = np.array([0.1, 0.2], dtype=np.float32)
        assert to_mono(mono) is mono

    def test_resample_changes_length_proportionally(self) -> None:
        audio = tone(1.0, sample_rate=16_000)
        assert len(resample(audio, 16_000, 8_000)) == pytest.approx(8_000, rel=0.01)

    def test_resample_is_a_noop_at_the_same_rate(self) -> None:
        audio = tone(0.1)
        assert np.array_equal(resample(audio, 16_000, 16_000), audio)

    def test_resample_handles_empty_input(self) -> None:
        assert resample(np.zeros(0, dtype=np.float32), 16_000, 8_000).size == 0

    def test_rms_of_silence_is_zero(self) -> None:
        assert rms(np.zeros(100, dtype=np.float32)) == 0.0

    def test_rms_of_empty_is_zero(self) -> None:
        assert rms(np.zeros(0, dtype=np.float32)) == 0.0

    def test_int16_round_trip(self) -> None:
        audio = np.array([-1.0, 0.0, 0.5], dtype=np.float32)
        assert np.max(np.abs(int16_to_float(float_to_int16(audio)) - audio)) < 1e-3

    def test_float_to_int16_clips(self) -> None:
        assert float_to_int16(np.array([2.0, -2.0], dtype=np.float32)).tolist() == [32767, -32767]


class TestTTSFactory:
    def test_none_backend_returns_null(self) -> None:
        backend = create_tts_backend("spa_Latn", TTSSettings(backend="none"))
        assert isinstance(backend, NullTTS)
        assert backend.name == "none"

    def test_null_backend_produces_no_audio(self) -> None:
        speech = NullTTS().synthesize("hola")
        assert len(speech.audio) == 0
        assert not speech

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(TTSUnavailableError, match="Unknown TTS backend"):
            create_tts_backend("spa_Latn", TTSSettings(backend="nope"))  # type: ignore[arg-type]

    def test_available_backends_always_includes_none(self) -> None:
        assert "none" in available_backends()

    def test_piper_reports_language_support_from_the_voice_map(self) -> None:
        if not PiperTTS.is_installed():
            pytest.skip("piper-tts is not installed")
        assert PiperTTS.supports("spa_Latn")
        # Tibetan has no Piper voice.
        assert not PiperTTS.supports("bod_Tibt")

    def test_explicit_voice_overrides_language_support(self) -> None:
        if not PiperTTS.is_installed():
            pytest.skip("piper-tts is not installed")
        assert PiperTTS.supports("bod_Tibt", TTSSettings(voice="en_US-lessac-medium"))

    def test_auto_falls_back_to_null_for_a_language_with_no_voice(self, monkeypatch) -> None:
        monkeypatch.setattr(PiperTTS, "supports", classmethod(lambda cls, lang, s=None: False))
        monkeypatch.setattr(SystemTTS, "supports", classmethod(lambda cls, lang, s=None: False))
        backend = create_tts_backend("bod_Tibt", TTSSettings(backend="auto"))
        assert isinstance(backend, NullTTS)


class TestSpeechAudio:
    def test_duration(self) -> None:
        speech = SpeechAudio(np.zeros(22_050, dtype=np.float32), 22_050)
        assert speech.duration == pytest.approx(1.0)

    def test_zero_sample_rate_does_not_divide_by_zero(self) -> None:
        assert SpeechAudio(np.zeros(10, dtype=np.float32), 0).duration == 0.0

    def test_save_writes_a_file(self, tmp_path) -> None:
        path = SpeechAudio(tone(0.2, sample_rate=22_050), 22_050).save(tmp_path / "out.wav")
        assert path.exists()
