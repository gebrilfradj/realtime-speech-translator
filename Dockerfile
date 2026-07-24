# Real-time speech translator: faster-whisper -> NLLB-200 -> Piper
#
#   docker build -t speech-translate .
#   docker run --rm -p 7860:7860 speech-translate
#
# Then open http://localhost:7860. Microphone capture happens in the browser,
# so the container needs no audio device.
#
# Models are downloaded on first use into /home/app/.cache. Mount a volume
# there to avoid re-downloading on every run:
#   docker run --rm -p 7860:7860 -v st-cache:/home/app/.cache speech-translate

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    SPEECH_TRANSLATE_VOICES=/home/app/.cache/piper-voices \
    GRADIO_SERVER_NAME=0.0.0.0

# libportaudio2 -> sounddevice (optional here, but keeps --list-devices working)
# libsndfile1   -> soundfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# CPU-only torch: the full CUDA wheel is several gigabytes and unnecessary for
# a container that is usually run on a CPU host.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY speech_translate ./speech_translate
RUN pip install --no-cache-dir ".[audio,tts,web,system-tts]"

USER app
RUN mkdir -p /home/app/.cache

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/').read()" || exit 1

ENTRYPOINT ["python", "-m", "speech_translate.webui"]
CMD ["--host", "0.0.0.0", "--port", "7860"]
