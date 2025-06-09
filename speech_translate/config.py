# config.py
# Constants and model settings

LANGUAGE_TTS_MODELS = {
    "en": "tts_models/en/vctk/vits",
    "es": "tts_models/es/css10/vits",
    "fr": "tts_models/fr/css10/vits"
}

CHUNK_DURATION = 2    # seconds per audio chunk
RATE = 16000          # sampling rate
CHANNELS = 1
FORMAT = None         # will be set dynamically in main.py
CHUNK = 1024