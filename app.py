"""Hugging Face Spaces entry point.

Spaces runs ``app.py`` at the repository root, so this file exists purely to
launch the same demo the CLI does. All the logic lives in
``speech_translate/webui.py``; there is no duplicated UI code.

A Space also needs a ``README.md`` with YAML front matter declaring
``sdk: gradio``, and its dependencies in ``requirements.txt`` **at the Space
root** -- any other filename is ignored. Those live in ``spaces/`` and are put
in the right places by::

    python scripts/deploy_space.py --push https://huggingface.co/spaces/<user>/<name>

Do not simply push this repository to a Space: the root ``requirements.txt``
pins torch without the CPU wheel index and would pull the multi-gigabyte CUDA
build onto free CPU hardware.
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
