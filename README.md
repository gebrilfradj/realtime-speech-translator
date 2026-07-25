# Real-Time Speech Translator

Speech-to-speech translation that runs on your own machine. You talk, it transcribes,
translates, and speaks the result back in another language. No cloud APIs.

I built this for an undergraduate research project at the University of Florida, then
rewrote it to be faster and to actually install correctly.

[![CI](https://github.com/gebrilfradj/realtime-speech-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/gebrilfradj/realtime-speech-translator/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**2.7 s to process 4.5 s of speech on a CPU with no GPU, speech synthesis included.
That is a real-time factor of 0.61, so it keeps up with a live speaker.**

![Translating English speech into Spanish in the browser](assets/demo.gif)

## Quickstart

```bash
git clone https://github.com/gebrilfradj/realtime-speech-translator.git
cd realtime-speech-translator

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m speech_translate.webui    # browser demo on http://127.0.0.1:7860
```

Models download on first run (about 1.5 GB) and are cached after that.

Docker works too:

```bash
docker build -t speech-translate .
docker run --rm -p 7860:7860 -v st-cache:/home/app/.cache speech-translate
```

## What it uses

| Stage | Model | Why |
|---|---|---|
| Speech recognition | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Whisper accuracy, int8 on CPU, and it can load distilled checkpoints that `openai-whisper` can't |
| Translation | [nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | 200 languages, still maintained, better than M2M100-418M |
| Speech synthesis | [Piper](https://github.com/OHF-Voice/piper1-gpl) | Faster than real time on CPU, 170 voices across 49 languages |

97 source languages, 200 translation targets, 48 of those with a voice. Each stage sits
behind a small interface, so swapping a model is one class rather than a rewrite.

## Usage

```bash
python -m speech_translate --tgt es               # auto-detect what you speak, reply in Spanish
python -m speech_translate --src en --tgt ja      # pin the source language, slightly faster
python -m speech_translate --tgt fr --no-speak    # subtitles only
python -m speech_translate --tgt de --asr-model small   # more accuracy, more latency

python -m speech_translate --file talk.wav --tgt de     # translate a recording
python -m speech_translate --list-devices               # pick a microphone
python -m speech_translate --list-languages             # language codes
```

Language codes are flexible. `es`, `spa_Latn`, `es-ES` and `Spanish` all work, and an
unknown code fails immediately with a suggestion instead of quietly mistranslating.

As a library:

```python
from speech_translate import Pipeline, Settings

pipeline = Pipeline(Settings(src="auto", tgt="spa_Latn")).warmup()
result = pipeline.process_file("sample.wav")

print(result.source_language_name)   # English, detected
print(result.translation)            # Hola, ¿cómo estás hoy? ...
result.speech.save("out.wav")
```

Importing the package doesn't load models or open audio devices.

## Performance

CPU-only machine, no GPU. Same 4.5 s clip, median of 5 runs after warm-up. Reproduce with
`python -m speech_translate.benchmark --audio sample.wav --tgt es --legacy`.

| Setup | ASR | MT | TTS | Total | RTF |
|---|---:|---:|---:|---:|---:|
| Original (`openai-whisper small` + M2M100-418M) | 3438 ms | 1235 ms | n/a | 4673 ms | 1.04 |
| faster-whisper `small` + NLLB + Piper | 3209 ms | 1432 ms | 253 ms | 4894 ms | 1.09 |
| **faster-whisper `base` + NLLB + Piper (default)** | **1119 ms** | **1365 ms** | **257 ms** | **2741 ms** | **0.61** |

Worth being clear about where the speedup came from. Swapping the ASR runtime by itself
only bought about 1.07x on this CPU, since CTranslate2's bigger wins need a GPU or
AVX-512 VNNI, and NLLB-600M is roughly 18% slower than M2M100 (a deliberate trade for
quality and language coverage). The real gain is that faster-whisper makes the smaller
checkpoints practical, so the default drops to `base`:

| ASR model | Latency | RTF |
|---|---:|---:|
| `tiny` | 600 ms | 0.13 |
| **`base` (default)** | **1097 ms** | **0.24** |
| `distil-small.en` | 1529 ms | 0.34 |
| `small` | 3270 ms | 0.73 |

## Notes from the rewrite

The original version had a few problems worth naming, since they shaped the design:

- `requirements.txt` listed `whisper`, which on PyPI is [Graphite's time-series
  database](https://pypi.org/project/whisper/), not OpenAI's. It imports fine and then has
  no `load_model()`, so CI passed while nothing worked. CI now fails if it reappears.
- `--src auto` threw away the detected language and passed the string `"auto"` to the
  translator as a language code.
- Audio was cut into fixed 3-second chunks, which split words and ran the recognizer on
  silence. Voice activity detection replaced it, so 5 s of silence now costs nothing.
- Playback blocked the processing thread, so the translator stopped translating while it
  was speaking.

Full detail is in [PR #1](https://github.com/gebrilfradj/realtime-speech-translator/pull/1).

## Development

```bash
pip install -e ".[dev,audio,tts,web]"
pytest tests -q                       # 188 tests, models mocked, no downloads
ruff check speech_translate tests
python scripts/check_dependencies.py  # imports must match declared dependencies
```

CI runs lint, the test suite on Linux and Windows across Python 3.10 to 3.12, a dependency
audit, and a build.

`assets/demo.gif` is generated by `scripts/record_demo.py`, which drives a real browser
against a running server, so it stays in sync with the actual UI.

## Hosted demo

Hugging Face Spaces needs a README with `sdk: gradio` front matter and its dependencies in
`requirements.txt` at the Space root, neither of which matches this repo's layout. Those
live in [`spaces/`](spaces/) and get put in place by:

```bash
python scripts/deploy_space.py --push https://huggingface.co/spaces/<user>/<name>
```

## Limitations

- Runs fine on CPU, which is what all the numbers above were measured on. A GPU
  (`--device cuda`) helps recognition and translation a lot.
- Translation quality is NLLB's, so low-resource languages are weaker.
- 48 of the 200 target languages have a Piper voice. The rest fall back to the system voice
  and then to subtitles, so a missing voice never crashes anything.
- The `--legacy` benchmark leaves synthesis out of both sides, because Coqui TTS is
  unmaintained and no longer installs on current Python. That is why it was replaced.

## Citation

Fradj et al., *Real-time Multilingual Speech Translation with Open-Source Models*,
UF Journal of Undergraduate Research, Spring 2025.

## License

MIT, see [LICENSE](LICENSE).
