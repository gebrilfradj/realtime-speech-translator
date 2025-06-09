# tts.py
# Performs text-to-speech using pre-trained models from the TTS library
from TTS.api import TTS
from config import LANGUAGE_TTS_MODELS
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loads the appropriate TTS model for a given language
def load_tts(lang):
    name = LANGUAGE_TTS_MODELS.get(lang, LANGUAGE_TTS_MODELS["en"])
    return TTS(model_name=name, gpu=(DEVICE == "cuda"))

# Synthesizes speech and saves to a WAV file; uses the first speaker if applicable
def synthesize_speech(text, engine, out_path):
    if engine.speakers:
        engine.tts_to_file(text=text, file_path=out_path, speaker=engine.speakers[0])
    else:
        engine.tts_to_file(text=text, file_path=out_path)