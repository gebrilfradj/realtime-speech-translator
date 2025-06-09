# audio_utils.py
# Audio utility functions for recording and saving WAV files

import pyaudio
import wave
from config import CHANNELS, RATE, CHUNK

# Initialize PyAudio interface
audio_interface = pyaudio.PyAudio()

# Writes audio data to a WAV file using predefined audio settings
def write_wav_file(filename, audio_data):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio_interface.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(audio_data)
    wf.close()