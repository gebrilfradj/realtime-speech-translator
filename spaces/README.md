---
title: Real-Time Speech Translator
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Speech-to-speech translation with faster-whisper, NLLB-200 and Piper
---

# Real-Time Speech Translator

Speak in one language, hear it in another. A fully open-source, local cascade —
nothing is sent to a third-party API.

**[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** (speech recognition)
→ **[NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)** (translation, 200 languages)
→ **[Piper](https://github.com/OHF-Voice/piper1-gpl)** (speech synthesis, 48 voices)

## How to use it

* **Translate a clip** — record yourself or upload an audio file, pick a target
  language, press Translate. Works without a microphone if you upload.
* **Live** — streams your microphone and translates each utterance as you pause.
  Speech boundaries are detected automatically; there is no fixed chunk size.

Set *Subtitles only* to skip speech synthesis and just read the translation.

## Notes

* The first request loads ~1.5 GB of model weights, so it is slow. Everything
  after that is warm.
* This Space runs on free CPU hardware. On a local machine with a GPU the
  cascade is considerably faster — measured real-time factor **0.61 on CPU**
  (2.7 s to process 4.5 s of speech, synthesis included).
* 48 of the 200 target languages have a Piper voice. The rest fall back to
  subtitles rather than failing.

## Source and benchmarks

[github.com/gebrilfradj/realtime-speech-translator](https://github.com/gebrilfradj/realtime-speech-translator)

The repository has the full latency breakdown, a reproducible before/after
benchmark against the original openai-whisper + M2M100 stack, a Dockerfile, and
188 tests.
