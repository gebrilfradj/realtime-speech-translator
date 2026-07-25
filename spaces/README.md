---
title: Real-Time Speech Translator
emoji: 🎙️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Speech-to-speech translation with faster-whisper, NLLB-200 and Piper
---

# Real-Time Speech Translator

Speak in one language and get it back in another. Everything runs on the Space
itself, nothing is sent to an external API.

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition,
[NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) for translation
across 200 languages, and [Piper](https://github.com/OHF-Voice/piper1-gpl) for speech.

## Two ways to use it

**Translate a clip.** Record yourself or upload an audio file, pick a target
language, hit Translate. Works without a microphone if you upload.

**Live.** Streams your mic and translates each utterance as you pause. Speech
boundaries are detected automatically, there is no fixed chunk size.

Tick *Subtitles only* to skip synthesis and just read the translation.

## Notes

The first request loads about 1.5 GB of weights, so it is slow. Everything after
that is warm.

This Space runs on free CPU hardware. On a local machine the same pipeline takes
2.7 s to process 4.5 s of speech, synthesis included.

48 of the 200 target languages have a Piper voice. The rest fall back to
subtitles rather than failing.

## Source

[github.com/gebrilfradj/realtime-speech-translator](https://github.com/gebrilfradj/realtime-speech-translator)

The repo has the full latency breakdown, a benchmark against the original
openai-whisper and M2M100 stack, a Dockerfile, and 188 tests.
