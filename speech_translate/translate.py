# translate.py
# Handles text translation using M2M100 multilingual model

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loads the Facebook M2M100 translation model and tokenizer
def load_translation_model():
    name = "facebook/m2m100_418M"
    model = M2M100ForConditionalGeneration.from_pretrained(name).to(DEVICE)
    tokenizer = M2M100Tokenizer.from_pretrained(name)
    return model, tokenizer

# Translates text from the source language to the target language
def translate_text(text: str, src: str, tgt: str,
                   model, tokenizer) -> str:
    tokenizer.src_lang = src
    encoded = tokenizer(text, return_tensors="pt").to(DEVICE)
    bos = tokenizer.get_lang_id(tgt)
    tokens = model.generate(
        **encoded, forced_bos_token_id=bos,
        max_length=200, num_beams=1, early_stopping=True
    )
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

