"""Language identity for the cascade.

Three vocabularies have to agree with each other:

* **Whisper / faster-whisper** detects and accepts ISO 639-1 codes (``en``, ``es``).
* **NLLB-200** speaks FLORES-200 codes that carry a script suffix (``eng_Latn``).
* **Piper** ships voices keyed by locale (``en_US-lessac-medium``).

Everything in this project is normalised to a FLORES-200 code as early as
possible, because that is the only vocabulary of the three that is unambiguous
(``zh`` is not a language you can translate into -- ``zho_Hans`` and
``zho_Hant`` are).
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

__all__ = [
    "UnsupportedLanguageError",
    "AUTO",
    "resolve_language",
    "whisper_to_flores",
    "flores_to_whisper",
    "language_name",
    "supported_languages",
    "piper_voice_for",
]

AUTO = "auto"


class UnsupportedLanguageError(ValueError):
    """Raised when a language code cannot be mapped onto FLORES-200."""


# ISO 639-1 (what Whisper emits) -> FLORES-200 (what NLLB expects).
# Whisper languages with no FLORES-200 counterpart (``la``, ``br``, ``haw``)
# are deliberately absent so they fail loudly rather than silently mistranslate.
_WHISPER_TO_FLORES: dict[str, str] = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "as": "asm_Beng",
    "az": "azj_Latn", "ba": "bak_Cyrl", "be": "bel_Cyrl", "bg": "bul_Cyrl",
    "bn": "ben_Beng", "bo": "bod_Tibt", "bs": "bos_Latn", "ca": "cat_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn",
    "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn",
    "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fo": "fao_Latn",
    "fr": "fra_Latn", "gl": "glg_Latn", "gu": "guj_Gujr", "ha": "hau_Latn",
    "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "ht": "hat_Latn",
    "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn", "is": "isl_Latn",
    "it": "ita_Latn", "ja": "jpn_Jpan", "jw": "jav_Latn", "ka": "kat_Geor",
    "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda", "ko": "kor_Hang",
    "lb": "ltz_Latn", "ln": "lin_Latn", "lo": "lao_Laoo", "lt": "lit_Latn",
    "lv": "lvs_Latn", "mg": "plt_Latn", "mi": "mri_Latn", "mk": "mkd_Cyrl",
    "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva", "ms": "zsm_Latn",
    "mt": "mlt_Latn", "my": "mya_Mymr", "ne": "npi_Deva", "nl": "nld_Latn",
    "nn": "nno_Latn", "no": "nob_Latn", "oc": "oci_Latn", "pa": "pan_Guru",
    "pl": "pol_Latn", "ps": "pbt_Arab", "pt": "por_Latn", "ro": "ron_Latn",
    "ru": "rus_Cyrl", "sa": "san_Deva", "sd": "snd_Arab", "si": "sin_Sinh",
    "sk": "slk_Latn", "sl": "slv_Latn", "sn": "sna_Latn", "so": "som_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "su": "sun_Latn", "sv": "swe_Latn",
    "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "tg": "tgk_Cyrl",
    "th": "tha_Thai", "tk": "tuk_Latn", "tl": "tgl_Latn", "tr": "tur_Latn",
    "tt": "tat_Cyrl", "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn",
    "vi": "vie_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn", "yue": "yue_Hant",
    "zh": "zho_Hans",
}

_FLORES_TO_WHISPER: dict[str, str] = {v: k for k, v in _WHISPER_TO_FLORES.items()}
# ``zho_Hant`` also has to route back to Whisper's ``zh``; the inverted dict
# above only keeps the Hans variant.
_FLORES_TO_WHISPER.setdefault("zho_Hant", "zh")

# Human-readable names, keyed by FLORES code. Used by the CLI error messages
# and by the web UI language pickers.
_LANGUAGE_NAMES: dict[str, str] = {
    "afr_Latn": "Afrikaans", "als_Latn": "Albanian", "amh_Ethi": "Amharic",
    "arb_Arab": "Arabic", "asm_Beng": "Assamese", "azj_Latn": "Azerbaijani",
    "bak_Cyrl": "Bashkir", "bel_Cyrl": "Belarusian", "ben_Beng": "Bengali",
    "bod_Tibt": "Tibetan", "bos_Latn": "Bosnian", "bul_Cyrl": "Bulgarian",
    "cat_Latn": "Catalan", "ces_Latn": "Czech", "cym_Latn": "Welsh",
    "dan_Latn": "Danish", "deu_Latn": "German", "ell_Grek": "Greek",
    "eng_Latn": "English", "est_Latn": "Estonian", "eus_Latn": "Basque",
    "fao_Latn": "Faroese", "fin_Latn": "Finnish", "fra_Latn": "French",
    "glg_Latn": "Galician", "guj_Gujr": "Gujarati", "hat_Latn": "Haitian Creole",
    "hau_Latn": "Hausa", "heb_Hebr": "Hebrew", "hin_Deva": "Hindi",
    "hrv_Latn": "Croatian", "hun_Latn": "Hungarian", "hye_Armn": "Armenian",
    "ind_Latn": "Indonesian", "isl_Latn": "Icelandic", "ita_Latn": "Italian",
    "jav_Latn": "Javanese", "jpn_Jpan": "Japanese", "kan_Knda": "Kannada",
    "kat_Geor": "Georgian", "kaz_Cyrl": "Kazakh", "khk_Cyrl": "Mongolian",
    "khm_Khmr": "Khmer", "kor_Hang": "Korean", "lao_Laoo": "Lao",
    "lin_Latn": "Lingala", "lit_Latn": "Lithuanian", "ltz_Latn": "Luxembourgish",
    "lvs_Latn": "Latvian", "mal_Mlym": "Malayalam", "mar_Deva": "Marathi",
    "mkd_Cyrl": "Macedonian", "mlt_Latn": "Maltese", "mri_Latn": "Maori",
    "mya_Mymr": "Burmese", "nld_Latn": "Dutch", "nno_Latn": "Norwegian Nynorsk",
    "nob_Latn": "Norwegian Bokmal", "npi_Deva": "Nepali", "oci_Latn": "Occitan",
    "pan_Guru": "Punjabi", "pbt_Arab": "Pashto", "pes_Arab": "Persian",
    "plt_Latn": "Malagasy", "pol_Latn": "Polish", "por_Latn": "Portuguese",
    "ron_Latn": "Romanian", "rus_Cyrl": "Russian", "san_Deva": "Sanskrit",
    "sin_Sinh": "Sinhala", "slk_Latn": "Slovak", "slv_Latn": "Slovenian",
    "sna_Latn": "Shona", "snd_Arab": "Sindhi", "som_Latn": "Somali",
    "spa_Latn": "Spanish", "srp_Cyrl": "Serbian", "sun_Latn": "Sundanese",
    "swe_Latn": "Swedish", "swh_Latn": "Swahili", "tam_Taml": "Tamil",
    "tat_Cyrl": "Tatar", "tel_Telu": "Telugu", "tgk_Cyrl": "Tajik",
    "tgl_Latn": "Tagalog", "tha_Thai": "Thai", "tuk_Latn": "Turkmen",
    "tur_Latn": "Turkish", "ukr_Cyrl": "Ukrainian", "urd_Arab": "Urdu",
    "uzn_Latn": "Uzbek", "vie_Latn": "Vietnamese", "ydd_Hebr": "Yiddish",
    "yor_Latn": "Yoruba", "yue_Hant": "Cantonese", "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)", "zsm_Latn": "Malay",
}

_NAME_TO_FLORES: dict[str, str] = {
    name.lower(): code for code, name in _LANGUAGE_NAMES.items()
}

# Default Piper voice per FLORES code. Generated from the official
# ``rhasspy/piper-voices`` catalogue, preferring `medium` quality because
# `high` roughly doubles synthesis time for little perceptual gain in a
# real-time loop. Override per run with ``--tts-voice``.
_PIPER_VOICES: dict[str, str] = {
    "arb_Arab": "ar_JO-kareem-medium",
    "bul_Cyrl": "bg_BG-dimitar-medium",
    "ben_Beng": "bn_BD-google-medium",
    "cat_Latn": "ca_ES-upc_ona-medium",
    "ces_Latn": "cs_CZ-jirka-medium",
    "cym_Latn": "cy_GB-bu_tts-medium",
    "dan_Latn": "da_DK-talesyntese-medium",
    "deu_Latn": "de_DE-thorsten-medium",
    "ell_Grek": "el_GR-rapunzelina-low",
    "eng_Latn": "en_US-lessac-medium",
    "spa_Latn": "es_ES-davefx-medium",
    "eus_Latn": "eu_ES-antton-medium",
    "pes_Arab": "fa_IR-amir-medium",
    "fin_Latn": "fi_FI-harri-medium",
    "fra_Latn": "fr_FR-siwis-medium",
    "heb_Hebr": "he_IL-saspeech-medium",
    "hin_Deva": "hi_IN-pratham-medium",
    "hun_Latn": "hu_HU-anna-medium",
    "hye_Armn": "hy_AM-gor-medium",
    "ind_Latn": "id_ID-news_tts-medium",
    "isl_Latn": "is_IS-bui-medium",
    "ita_Latn": "it_IT-paola-medium",
    "kat_Geor": "ka_GE-natia-medium",
    "kaz_Cyrl": "kk_KZ-issai-high",
    "kor_Hang": "ko_KR-kss-medium",
    "ltz_Latn": "lb_LU-marylux-medium",
    "lvs_Latn": "lv_LV-aivars-medium",
    "mal_Mlym": "ml_IN-arjun-medium",
    "mar_Deva": "mr_IN-google-medium",
    "npi_Deva": "ne_NP-chitwan-medium",
    "nld_Latn": "nl_BE-nathalie-medium",
    "nob_Latn": "no_NO-nvcc-medium",
    "pol_Latn": "pl_PL-darkman-medium",
    "por_Latn": "pt_BR-cadu-medium",
    "ron_Latn": "ro_RO-mihai-medium",
    "rus_Cyrl": "ru_RU-denis-medium",
    "slk_Latn": "sk_SK-lili-medium",
    "slv_Latn": "sl_SI-artur-medium",
    "als_Latn": "sq_AL-edon-medium",
    "srp_Cyrl": "sr_RS-serbski_institut-medium",
    "swe_Latn": "sv_SE-alma-medium",
    "swh_Latn": "sw_CD-lanfrica-medium",
    "tel_Telu": "te_IN-maya-medium",
    "tur_Latn": "tr_TR-dfki-medium",
    "ukr_Cyrl": "uk_UA-mykyta-high",
    "urd_Arab": "ur_PK-aegis_female-medium",
    "vie_Latn": "vi_VN-vais1000-medium",
    "zho_Hans": "zh_CN-chaowen-medium",
}


def _normalise(code: str) -> str:
    return code.strip().replace("-", "_")


def resolve_language(code: str, *, allow_auto: bool = False) -> str:
    """Coerce any reasonable spelling of a language into a FLORES-200 code.

    Accepts a FLORES code (``spa_Latn``), an ISO 639-1 code (``es``), a
    locale-ish code (``es-ES``) or an English name (``Spanish``).

    Raises:
        UnsupportedLanguageError: if the code has no FLORES-200 equivalent.
            The message includes close matches, because a silently wrong
            language code is the single easiest way to get fluent nonsense
            out of an MT model.
    """
    if code is None:
        raise UnsupportedLanguageError("No language given.")

    raw = _normalise(code)
    if raw.lower() == AUTO:
        if allow_auto:
            return AUTO
        raise UnsupportedLanguageError(
            "'auto' is only valid for the source language; "
            "the target language must be explicit."
        )

    if raw in _LANGUAGE_NAMES:
        return raw

    lowered = raw.lower()
    # ISO 639-1, optionally with a region suffix we can discard (es_ES -> es).
    base = lowered.split("_", 1)[0]
    if lowered in _WHISPER_TO_FLORES:
        return _WHISPER_TO_FLORES[lowered]
    if base in _WHISPER_TO_FLORES:
        return _WHISPER_TO_FLORES[base]
    if lowered.replace("_", " ") in _NAME_TO_FLORES:
        return _NAME_TO_FLORES[lowered.replace("_", " ")]

    raise UnsupportedLanguageError(
        f"Unsupported language {code!r}. {_suggest(code)}"
    )


def _suggest(code: str) -> str:
    pool: list[str] = list(_LANGUAGE_NAMES) + list(_WHISPER_TO_FLORES)
    close = difflib.get_close_matches(_normalise(code).lower(), pool, n=3, cutoff=0.5)
    if close:
        return "Did you mean: " + ", ".join(close) + "?"
    return "Use an ISO 639-1 code (e.g. 'es'), a FLORES-200 code (e.g. 'spa_Latn') or a name (e.g. 'Spanish')."


def whisper_to_flores(code: str) -> str:
    """Map a language code detected by Whisper onto FLORES-200."""
    return resolve_language(code)


def flores_to_whisper(code: str) -> str | None:
    """Map FLORES-200 back to the ISO 639-1 code Whisper understands.

    Returns ``None`` when Whisper cannot be constrained to that language, in
    which case the caller should let Whisper auto-detect.
    """
    if code == AUTO:
        return None
    return _FLORES_TO_WHISPER.get(code)


def language_name(code: str) -> str:
    """Human-readable name for a FLORES-200 code."""
    return _LANGUAGE_NAMES.get(code, code)


def supported_languages() -> list[tuple[str, str]]:
    """``(flores_code, english_name)`` pairs sorted by name, for UI pickers."""
    return sorted(_LANGUAGE_NAMES.items(), key=lambda kv: kv[1])


def piper_voice_for(code: str) -> str | None:
    """Default Piper voice for a FLORES code, or ``None`` if none ships."""
    return _PIPER_VOICES.get(code)


def languages_with_speech() -> list[tuple[str, str]]:
    """Languages we can both translate into *and* speak aloud."""
    return sorted(
        ((c, _LANGUAGE_NAMES[c]) for c in _PIPER_VOICES if c in _LANGUAGE_NAMES),
        key=lambda kv: kv[1],
    )


def validate_against_tokenizer(tokenizer_codes: Iterable[str]) -> list[str]:
    """Return FLORES codes we advertise that a given NLLB tokenizer rejects.

    Used by the test-suite to keep the table above honest as model versions
    move; an empty list means every advertised language really works.
    """
    known = set(tokenizer_codes)
    return sorted(c for c in _LANGUAGE_NAMES if c not in known)
