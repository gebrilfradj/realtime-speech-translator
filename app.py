"""Hugging Face Spaces entry point.

Spaces runs ``app.py`` at the repository root, so this file exists purely to
launch the same demo the CLI does. All the logic lives in
``speech_translate/webui.py``; there is no duplicated UI code.

To deploy:

1. Create a Space (SDK: Gradio, hardware: CPU basic is enough).
2. Push this repository to it, or add the Space as a git remote:

       git remote add space https://huggingface.co/spaces/<user>/<space>
       git push space main

3. The Space installs ``requirements-spaces.txt`` and runs this file.
"""

from __future__ import annotations

import os

from speech_translate.config import ASRSettings, MTSettings, Settings, TTSSettings
from speech_translate.webui import build_demo

# Spaces CPUs are modest, so default to the fastest sensible ASR model and let
# it be overridden with a Space variable.
settings = Settings(
    src="auto",
    tgt=os.environ.get("TARGET_LANGUAGE", "spa_Latn"),
    asr=ASRSettings(model=os.environ.get("ASR_MODEL", "base"), device="cpu"),
    mt=MTSettings(model="facebook/nllb-200-distilled-600M", device="cpu"),
    tts=TTSSettings(backend="auto"),
)

demo = build_demo(settings, preload=False)

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=7860)
