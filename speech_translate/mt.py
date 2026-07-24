"""Machine translation on NLLB-200 (distilled 600M).

Replaces M2M100-418M. NLLB is better maintained, covers 200 languages, and the
distilled 600M checkpoint is stronger than M2M100-418M at comparable cost.

The API difference that matters: M2M100 used ``tokenizer.get_lang_id(tgt)`` with
ISO codes, NLLB uses FLORES-200 codes (``spa_Latn``) resolved through the
tokenizer's vocabulary. Getting this wrong yields fluent output in the *wrong*
language, which is why :meth:`NLLBTranslator.translate` validates up front.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .config import MTSettings
from .languages import AUTO, UnsupportedLanguageError, language_name, resolve_language

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

__all__ = ["Translation", "Translator", "NLLBTranslator"]


@dataclass
class Translation:
    text: str
    src: str
    tgt: str

    def __bool__(self) -> bool:
        return bool(self.text)


class Translator(Protocol):
    def translate(self, text: str, src: str, tgt: str) -> Translation: ...


class NLLBTranslator:
    """NLLB-200 translator with lazy model loading."""

    def __init__(self, settings: MTSettings | None = None) -> None:
        self.settings = settings or MTSettings()
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    # -- loading ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self.settings.resolved_device()
        logger.info("Loading MT model %s (device=%s)", self.settings.model, device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.settings.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.model)
        self._model = model.to(device).eval()

    def load(self) -> NLLBTranslator:
        """Force the model to load now (keeps warm-up out of latency numbers)."""
        self._ensure_loaded()
        return self

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        self._ensure_loaded()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        self._ensure_loaded()
        assert self._model is not None
        return self._model

    # -- language plumbing ----------------------------------------------
    def _lang_token_id(self, flores_code: str) -> int:
        """Vocabulary id of a FLORES language token.

        ``convert_tokens_to_ids`` returns the *unknown* id for a code the
        checkpoint does not know, which would silently translate into a random
        language -- so an unknown code is turned into a loud error instead.
        """
        tokenizer = self.tokenizer
        token_id = tokenizer.convert_tokens_to_ids(flores_code)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise UnsupportedLanguageError(
                f"{flores_code!r} is not a language known to {self.settings.model!r}."
            )
        return int(token_id)

    def supported_codes(self) -> set[str]:
        """FLORES codes this checkpoint actually knows."""
        extra = getattr(self.tokenizer, "additional_special_tokens", None) or []
        return {t for t in extra if "_" in t}

    # -- translation -----------------------------------------------------
    def translate(self, text: str, src: str, tgt: str) -> Translation:
        """Translate ``text`` from ``src`` to ``tgt`` (FLORES-200 codes)."""
        text = (text or "").strip()
        if not text:
            return Translation(text="", src=src, tgt=tgt)

        src_code = resolve_language(src, allow_auto=True)
        tgt_code = resolve_language(tgt)

        if src_code == AUTO:
            # The pipeline should have substituted the detected language by
            # now. Failing loudly beats guessing and mistranslating.
            raise UnsupportedLanguageError(
                "Source language is still 'auto'; pass the language detected by "
                "the recogniser (Transcript.language) instead."
            )

        if src_code == tgt_code:
            return Translation(text=text, src=src_code, tgt=tgt_code)

        import torch

        tokenizer = self.tokenizer
        tokenizer.src_lang = src_code
        device = self.settings.resolved_device()
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=self._lang_token_id(tgt_code),
                max_new_tokens=self.settings.max_new_tokens,
                num_beams=self.settings.num_beams,
            )
        out = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        logger.debug(
            "Translated %s -> %s: %r -> %r",
            language_name(src_code),
            language_name(tgt_code),
            text,
            out,
        )
        return Translation(text=out, src=src_code, tgt=tgt_code)


def load_translation_model(settings: MTSettings | None = None) -> NLLBTranslator:
    """Backwards-compatible helper mirroring the original API."""
    return NLLBTranslator(settings)


def translate_text(text: str, src: str, tgt: str, translator: NLLBTranslator) -> str:
    """Backwards-compatible helper returning only the text."""
    return translator.translate(text, src, tgt).text
