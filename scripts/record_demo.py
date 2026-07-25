#!/usr/bin/env python
"""Record the README demo GIF by driving the real web UI in a browser.

This is a genuine recording: it starts the actual Gradio app, uploads a real
audio clip, clicks Translate and screenshots the live result. Nothing is
mocked or staged, so the GIF cannot drift away from what the code does.

    python scripts/record_demo.py --url http://127.0.0.1:7860 --audio sample.wav

Requires ``playwright`` and ``pillow``::

    pip install playwright pillow && playwright install chromium
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

FRAME_DIR = Path("assets/_frames")


def record(url: str, audio: str, out: str, target_label: str, hold: float) -> int:
    from playwright.sync_api import sync_playwright

    audio_path = Path(audio).resolve()
    if not audio_path.exists():
        print(f"error: {audio_path} does not exist", file=sys.stderr)
        return 1

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for stale in FRAME_DIR.glob("*.png"):
        stale.unlink()

    frames: list[Path] = []

    def shoot(page, name: str) -> None:
        path = FRAME_DIR / f"{len(frames):02d}_{name}.png"
        page.screenshot(path=str(path))
        frames.append(path)
        print(f"  captured {path.name}")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--use-fake-ui-for-media-stream"])
        page = browser.new_page(viewport={"width": 1280, "height": 780})
        print(f"opening {url}")
        # Gradio holds an SSE connection open, so 'networkidle' never fires;
        # wait for the app's own markup instead.
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_selector("text=Real-Time Speech Translator", timeout=120_000)
        page.get_by_role("button", name="Translate", exact=True).wait_for(timeout=120_000)
        page.wait_for_timeout(3_000)
        shoot(page, "loaded")

        # Pick the target language.
        try:
            page.get_by_label("Translate into").click()
            page.wait_for_timeout(500)
            page.get_by_role("option", name=target_label, exact=False).first.click()
            page.wait_for_timeout(800)
            shoot(page, "language")
        except Exception as exc:  # pragma: no cover - UI drift
            print(f"  (language picker skipped: {exc})")

        # Upload the clip. Gradio's audio component starts on the "record"
        # source; switching to "upload" is what mounts the file input, which
        # stays visually hidden behind the drop zone.
        page.get_by_role("button", name="Upload file").click()
        page.wait_for_selector("input[type=file]", state="attached", timeout=30_000)
        page.locator("input[type=file]").first.set_input_files(str(audio_path))
        page.wait_for_timeout(3_500)
        shoot(page, "uploaded")

        # Translate.
        page.get_by_role("button", name="Translate", exact=True).click()
        print("  translating ...")
        for _ in range(120):
            page.wait_for_timeout(1_000)
            text = page.inner_text("body")
            if "RTF" in text and "ms" in text:
                break
        # The audio players redraw their waveforms after the response lands;
        # screenshotting too early catches them blank.
        page.wait_for_timeout(4_000)
        shoot(page, "result")
        page.wait_for_timeout(int(hold * 1000))
        shoot(page, "result_hold")
        browser.close()

    return _write_gif(frames, out)


def _write_gif(frames: list[Path], out: str) -> int:
    from PIL import Image

    if not frames:
        print("error: no frames captured", file=sys.stderr)
        return 1
    images = [Image.open(f).convert("RGB") for f in frames]
    width = min(img.width for img in images)
    scale = min(1.0, 900 / width)
    images = [
        img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        for img in images
    ]
    # Palette-quantise so the GIF stays small enough for a README.
    images = [img.quantize(colors=128, method=Image.MEDIANCUT) for img in images]

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    durations = [1400] * len(images)
    durations[-1] = 2600
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({len(images)} frames, {size_kb:.0f} KB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860", help="Running web UI.")
    parser.add_argument("--audio", default="sample.wav", help="Clip to upload.")
    parser.add_argument("--out", default="assets/demo.gif", help="Output GIF path.")
    parser.add_argument("--target", default="Spanish", help="Target language label to select.")
    parser.add_argument("--hold", type=float, default=1.0, help="Extra seconds on the result.")
    args = parser.parse_args()
    started = time.time()
    code = record(args.url, args.audio, args.out, args.target, args.hold)
    print(f"done in {time.time() - started:.0f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
