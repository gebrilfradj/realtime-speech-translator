"""CLI parsing, config plumbing and import hygiene."""

from __future__ import annotations

import subprocess
import sys

import pytest

from speech_translate.cli import build_parser, main, settings_from_args
from speech_translate.config import ASRSettings, Settings
from speech_translate.languages import UnsupportedLanguageError


class TestParser:
    def test_parsing_does_not_require_tgt_for_listing(self) -> None:
        args = build_parser().parse_args(["--list-devices"])
        assert args.list_devices
        assert args.tgt is None

    def test_defaults(self) -> None:
        args = build_parser().parse_args(["--tgt", "es"])
        assert args.src == "auto"
        assert args.asr_model == "base"
        assert args.tts == "auto"
        assert not args.no_speak

    def test_settings_from_args_resolves_languages(self) -> None:
        args = build_parser().parse_args(["--tgt", "Spanish", "--src", "en"])
        settings = settings_from_args(args)
        assert settings.tgt == "spa_Latn"
        assert settings.src == "eng_Latn"

    def test_no_speak_selects_the_null_backend(self) -> None:
        args = build_parser().parse_args(["--tgt", "es", "--no-speak"])
        assert settings_from_args(args).tts.backend == "none"

    def test_vad_flags_are_wired_through(self) -> None:
        args = build_parser().parse_args(
            ["--tgt", "es", "--no-vad", "--silence-ms", "900", "--max-utterance-ms", "5000"]
        )
        settings = settings_from_args(args)
        assert not settings.vad.enabled
        assert settings.vad.min_silence_ms == 900
        assert settings.vad.max_utterance_ms == 5_000

    def test_invalid_language_raises(self) -> None:
        args = build_parser().parse_args(["--tgt", "notalanguage"])
        with pytest.raises(UnsupportedLanguageError):
            settings_from_args(args)


class TestMain:
    def test_list_languages_exits_zero(self, capsys) -> None:
        assert main(["--list-languages"]) == 0
        out = capsys.readouterr().out
        assert "spa_Latn" in out
        assert "FLORES-200" in out

    def test_missing_target_is_a_usage_error(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2

    def test_bad_language_returns_two(self, capsys) -> None:
        assert main(["--tgt", "notalanguage"]) == 2
        assert "Unsupported language" in capsys.readouterr().err


class TestImportHygiene:
    """The original main.py ran argparse at import time.

    That is why the old CI smoke test (``python -c "import asr, translate,
    tts"``) was so fragile, and why importing the package had side effects.
    """

    def test_importing_the_package_has_no_side_effects(self) -> None:
        code = (
            "import sys; sys.argv = ['pytest-probe'];"
            "import speech_translate;"
            "import speech_translate.cli, speech_translate.pipeline;"
            "print(speech_translate.__version__)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
        )
        assert result.returncode == 0, result.stderr
        assert "2.0.0" in result.stdout

    def test_importing_does_not_open_an_audio_device(self) -> None:
        """Old audio_utils.py called pyaudio.PyAudio() at module scope."""
        code = (
            "import speech_translate.audio as a;"
            "assert not hasattr(a, 'audio_interface');"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
        )
        assert result.returncode == 0, result.stderr


class TestSettings:
    def test_settings_validate_languages_on_construction(self) -> None:
        with pytest.raises(UnsupportedLanguageError):
            Settings(tgt="nonsense")

    def test_auto_source_is_allowed(self) -> None:
        assert Settings(src="auto", tgt="es").src == "auto"

    def test_with_languages_returns_a_copy(self) -> None:
        original = Settings(src="auto", tgt="es")
        updated = original.with_languages(tgt="fr")
        assert original.tgt == "spa_Latn"
        assert updated.tgt == "fra_Latn"

    def test_block_frames_matches_the_block_duration(self) -> None:
        settings = Settings(tgt="es")
        assert settings.audio.block_frames == int(16_000 * 30 / 1000)

    def test_compute_type_auto_resolves_by_device(self) -> None:
        assert ASRSettings(device="cpu").resolved_compute_type() == "int8"
        assert ASRSettings(device="cuda").resolved_compute_type() == "float16"
        assert ASRSettings(device="cpu", compute_type="float32").resolved_compute_type() == "float32"

    def test_force_cpu_env_var_is_respected(self, monkeypatch) -> None:
        monkeypatch.setenv("SPEECH_TRANSLATE_FORCE_CPU", "1")
        assert ASRSettings(device="auto").resolved_device() == "cpu"
