# 🎙️ Real-Time Multilingual Speech Translator

This project is an **offline**, **real-time** speech-to-speech translation system that transcribes audio from a microphone, translates it to a target language, and speaks the translation aloud — all with average latency under 1.5 seconds.

Built with **Whisper**, **M2M100**, and **Coqui TTS**, this project showcases real-time AI integration, audio processing, and multi-threaded design.

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

### 🎯 Use Cases
- AI demo for real-time language translation

- Offline translator for travel or accessibility

- Showcase of end-to-end speech pipeline using open-source AI models

### 📌 Notes
- Performance is optimized for systems with a GPU, but CPU fallback is supported.

- Language support depends on the pre-trained models used (Whisper, M2M100, Coqui TTS).
