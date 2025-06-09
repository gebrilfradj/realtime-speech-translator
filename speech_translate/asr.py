# asr.py

import whisper
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loads the Whisper ASR model with the specified size
def load_asr_model(size="small"):
    return whisper.load_model(size, device=DEVICE)

#Transcribes speech from an audio file using the ASR model and optionally a language
def transcribe(audio_path, model, language="auto"):
    if language == "auto":
        result = model.transcribe(audio_path)
    else:
        result = model.transcribe(audio_path, language=language)
    return result.get("text", "").strip()