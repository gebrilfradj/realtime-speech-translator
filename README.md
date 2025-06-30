# 🎙️ Real-Time Multilingual Speech Translator

This project was developed as part of my AI research at the University of Florida. It performs real-time speech-to-speech translation using open-source models for ASR (Whisper), machine translation (M2M100), and TTS (TTS by Coqui). The goal was to explore low-latency multilingual communication for applications in global accessibility, human-computer interaction, and real-time dialogue systems.

The final system captures live audio from a microphone, transcribes it to text, translates it into a target language, and synthesizes natural-sounding speech in under 1.5 seconds.

[![CI](https://github.com/gebrilfradj/realtime-speech-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/gebrilfradj/realtime-speech-translator/actions/workflows/ci.yml) 
---

## 🚀 Features

- 🎧 Live microphone input recording
- 🧠 Speech-to-text using OpenAI Whisper
- 🌍 Translation using Facebook's M2M100 multilingual model
- 🔊 Text-to-speech output via Coqui TTS
- 📊 Benchmarking script to measure average end-to-end latency
- 🧵 Multi-threaded real-time pipeline
- 🧪 Sample generation utility

---

## 📦 Tech Stack

- **Speech Recognition (ASR):** [OpenAI Whisper](https://github.com/openai/whisper)
- **Machine Translation:** [Facebook M2M100 (418M)](https://huggingface.co/facebook/m2m100_418M)
- **Text-to-Speech (TTS):** [Coqui TTS](https://github.com/coqui-ai/TTS)
- **Audio Playback & Recording:** PyAudio, PyDub
- **Hardware**: Tested on RTX 3050 Ti (laptop)  


---

## 📂 Project Structure

- ├── main.py # Real-time pipeline (mic → translation → speech)
- ├── benchmark.py # Measures average latency for translation loop
- ├── asr.py # ASR model loader and transcription
- ├── translate.py # Text translation functions
- ├── tts.py # Text-to-speech synthesis
- ├── audio_utils.py # Audio recording & WAV file handling
- ├── config.py # Global constants (chunk size, sampling rate, etc.)
- ├── make_sample.py # Optional: Generate sample audio for testing
- ├── requirements.txt # Dependency list
- └── .gitignore

---

## 🧪 Quickstart

### 1. Clone the repo

- git clone
- cd speech_translate

### 2. Set up the environment

- python -m venv .venv
- source .venv/bin/activate  # or .venv\Scripts\activate on Windows
- pip install -r requirements.txt

### 3. Run real-time translation

- python main.py --tgt es   # Example: translate from any language to Spanish

### 4. Run benchmark on sample audio

- python benchmark.py --audio sample.wav --src en --tgt fr
- (Optional: use make_sample.py to generate sample.wav.)
- Example Output: Avg latency: 1.41s on NVIDIA GeForce RTX 3050 Ti Laptop GPU

---

### 🌍 Supported Languages
Update LANGUAGE_TTS_MODELS in config.py to expand.
Supported by default:

- en – English

- es – Spanish

- fr – French

### 🎯 Use Cases
- AI demo for real-time language translation

- Offline translator for travel or accessibility

- Showcase of end-to-end speech pipeline using open-source AI models

### 🧠 Citation
Based on research by:
Fradj et al., Real-time Multilingual Speech Translation with Open-Source Models,
UF Journal of Undergraduate Research, Spring 2025.

### 📌 Notes
- Performance is optimized for systems with a GPU, but CPU fallback is supported.

- Language support depends on the pre-trained models used (Whisper, M2M100, Coqui TTS).


