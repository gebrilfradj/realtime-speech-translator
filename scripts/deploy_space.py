#!/usr/bin/env python
"""Assemble (and optionally push) a Hugging Face Space from this repository.

A Space is not just "the repo with an app.py". Hugging Face requires two things
that a normal GitHub repo does not have:

* ``README.md`` must open with YAML front matter declaring ``sdk: gradio`` and
  ``app_file``. Without it the Space will not build as a Gradio app.
* dependencies must be in ``requirements.txt`` **at the Space root**. Any other
  filename is ignored -- and this project's root ``requirements.txt`` pins
  ``torch`` without the CPU index, which would drag the multi-gigabyte CUDA
  build onto free CPU hardware.

So the Space gets its own README and requirements from ``spaces/``, and this
script builds a tree with those in the right places.

    python scripts/deploy_space.py
    python scripts/deploy_space.py --push https://huggingface.co/spaces/<user>/<name>

Pushing needs a Hugging Face token with write access; ``huggingface-cli login``
or a ``HF_TOKEN`` environment variable is the usual way.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "build" / "space"

# (source, destination-inside-the-Space)
FILES = [
    ("spaces/README.md", "README.md"),
    ("spaces/requirements.txt", "requirements.txt"),
    ("app.py", "app.py"),
    ("LICENSE", "LICENSE"),
]
PACKAGE = "speech_translate"


def build(out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for source, destination in FILES:
        src = ROOT / source
        if not src.exists():
            raise SystemExit(f"error: {source} is missing from the repository")
        target = out_dir / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        print(f"  {source} -> {destination}")

    shutil.copytree(
        ROOT / PACKAGE,
        out_dir / PACKAGE,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    print(f"  {PACKAGE}/ -> {PACKAGE}/")

    _verify(out_dir)
    return out_dir


def _verify(out_dir: Path) -> None:
    """Fail loudly on the mistakes that produce a silently broken Space."""
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    if not readme.startswith("---"):
        raise SystemExit("error: the Space README must start with YAML front matter")
    header = readme.split("---")[1]
    for key in ("sdk:", "app_file:", "sdk_version:"):
        if key not in header:
            raise SystemExit(f"error: the Space README front matter is missing {key!r}")

    requirements = (out_dir / "requirements.txt").read_text(encoding="utf-8")
    if "torch" in requirements and "cpu" not in requirements:
        raise SystemExit(
            "error: requirements.txt pins torch without the CPU wheel index; "
            "the CUDA build will not fit on free Space hardware"
        )
    if not (out_dir / PACKAGE / "__init__.py").exists():
        raise SystemExit("error: the speech_translate package did not copy correctly")
    print("Space layout verified.")


def push(out_dir: Path, remote: str, message: str) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=out_dir, check=True)

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("-c", "user.name=deploy", "-c", "user.email=deploy@localhost", "commit", "-q", "-m", message)
    git("remote", "add", "origin", remote)
    print(f"\nPushing to {remote} ...")
    # A Space's history is not interesting; force-push a single clean commit.
    git("push", "--force", "origin", "main")
    print("Pushed. The Space will start building now.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Where to assemble the Space.")
    parser.add_argument("--push", default=None, metavar="URL", help="Space git remote to push to.")
    parser.add_argument("--message", default="Deploy speech translator", help="Commit message.")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    print(f"Assembling Space in {out_dir}")
    build(out_dir)

    if args.push:
        push(out_dir, args.push, args.message)
    else:
        print("\nNothing pushed. To deploy:")
        print(f"  python {Path(__file__).relative_to(ROOT)} --push https://huggingface.co/spaces/<user>/<name>")
        print("or commit the contents of the directory above to your Space by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
