"""Language mapping: the layer that turns Whisper's ISO codes into NLLB's."""

from __future__ import annotations

import pytest

from speech_translate.languages import (
    _PIPER_VOICES,
    _WHISPER_TO_FLORES,
    AUTO,
    UnsupportedLanguageError,
    flores_to_whisper,
    language_name,
    languages_with_speech,
    piper_voice_for,
    resolve_language,
    supported_languages,
    whisper_to_flores,
)


class TestResolveLanguage:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("es", "spa_Latn"),
            ("ES", "spa_Latn"),
            ("spa_Latn", "spa_Latn"),
            ("Spanish", "spa_Latn"),
            ("spanish", "spa_Latn"),
            ("es-ES", "spa_Latn"),
            ("es_MX", "spa_Latn"),
            ("en", "eng_Latn"),
            ("zh", "zho_Hans"),
            ("  fr  ", "fra_Latn"),
        ],
    )
    def test_accepts_every_reasonable_spelling(self, value: str, expected: str) -> None:
        assert resolve_language(value) == expected

    def test_auto_allowed_only_for_source(self) -> None:
        assert resolve_language("auto", allow_auto=True) == AUTO
        with pytest.raises(UnsupportedLanguageError, match="only valid for the source"):
            resolve_language("auto")

    def test_unknown_language_raises_with_a_suggestion(self) -> None:
        with pytest.raises(UnsupportedLanguageError) as excinfo:
            resolve_language("spanich")
        assert "spa_Latn" in str(excinfo.value)

    def test_unknown_language_without_close_match_still_explains(self) -> None:
        with pytest.raises(UnsupportedLanguageError, match="ISO 639-1"):
            resolve_language("qqqqqq")

    def test_none_is_rejected(self) -> None:
        with pytest.raises(UnsupportedLanguageError):
            resolve_language(None)  # type: ignore[arg-type]


class TestWhisperRoundTrip:
    def test_every_whisper_code_maps_to_a_named_flores_code(self) -> None:
        """A mapped code with no name would print as a bare code in the UI."""
        for iso, flores in _WHISPER_TO_FLORES.items():
            assert language_name(flores) != flores, f"{iso} -> {flores} has no display name"

    def test_flores_codes_have_the_expected_shape(self) -> None:
        for flores in _WHISPER_TO_FLORES.values():
            lang, _, script = flores.partition("_")
            assert len(lang) == 3, flores
            assert script and script[0].isupper(), flores

    @pytest.mark.parametrize("iso", ["en", "es", "ja", "zh", "ar"])
    def test_round_trip_is_stable(self, iso: str) -> None:
        assert flores_to_whisper(whisper_to_flores(iso)) == iso

    def test_traditional_chinese_maps_back_to_whisper_chinese(self) -> None:
        assert flores_to_whisper("zho_Hant") == "zh"

    def test_auto_has_no_whisper_code(self) -> None:
        assert flores_to_whisper(AUTO) is None

    def test_unknown_flores_code_returns_none(self) -> None:
        assert flores_to_whisper("xxx_Yyyy") is None


class TestPiperVoices:
    def test_every_voice_targets_a_known_language(self) -> None:
        known = {code for code, _ in supported_languages()}
        assert set(_PIPER_VOICES) <= known

    def test_voice_names_look_like_piper_identifiers(self) -> None:
        for code, voice in _PIPER_VOICES.items():
            assert voice.count("-") >= 2, f"{code}: {voice}"
            locale = voice.split("-")[0]
            assert "_" in locale, f"{code}: {voice}"

    def test_english_and_spanish_can_be_spoken(self) -> None:
        assert piper_voice_for("eng_Latn")
        assert piper_voice_for("spa_Latn")

    def test_language_without_a_voice_returns_none(self) -> None:
        assert piper_voice_for("bod_Tibt") is None

    def test_speakable_languages_are_a_subset_of_translatable(self) -> None:
        speakable = {code for code, _ in languages_with_speech()}
        translatable = {code for code, _ in supported_languages()}
        assert speakable <= translatable
        assert len(speakable) >= 40


def test_supported_languages_is_sorted_by_name() -> None:
    names = [name for _, name in supported_languages()]
    assert names == sorted(names)


def test_language_name_falls_back_to_the_code() -> None:
    assert language_name("xxx_Yyyy") == "xxx_Yyyy"
