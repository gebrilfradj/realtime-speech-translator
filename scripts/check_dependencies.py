#!/usr/bin/env python
"""Fail if the package imports anything that is not a declared dependency.

The original project shipped ``pyttsx3`` usage with no matching requirement,
and an ASR module importing a package that was never really installed. Both
are the same class of bug: the import graph and the dependency list drifting
apart. This check keeps them together.

Run directly, or via the ``dependency-audit`` CI job::

    python scripts/check_dependencies.py
"""

from __future__ import annotations

import ast
import pathlib
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 fallback
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "speech_translate"

# Import name -> distribution name, where the two differ.
ALIASES = {
    "faster_whisper": "faster-whisper",
    "piper": "piper-tts",
    "whisper": "openai-whisper",  # optional, legacy benchmark comparison only
    "yaml": "pyyaml",
}

# Imported only inside the optional ``--legacy`` benchmark path, guarded by a
# try/except that prints installation instructions.
OPTIONAL = {"openai-whisper"}


def declared_dependencies() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    specs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)

    names = set()
    for spec in specs:
        name = spec.split("[")[0]
        for separator in (">=", "<=", "==", "!=", "~=", ">", "<", ";"):
            name = name.split(separator)[0]
        names.add(name.strip())
    return names


def imported_packages() -> dict[str, set[pathlib.Path]]:
    found: dict[str, set[pathlib.Path]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import: inside this package
                    continue
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name:
                    found.setdefault(name, set()).add(path.relative_to(ROOT))
    return found


def main() -> int:
    declared = declared_dependencies()
    stdlib = set(sys.stdlib_module_names)
    missing: list[str] = []

    for name, paths in sorted(imported_packages().items()):
        if name in stdlib or name == "speech_translate":
            continue
        distribution = ALIASES.get(name, name)
        if distribution in declared or distribution in OPTIONAL:
            continue
        where = ", ".join(str(p) for p in sorted(paths))
        missing.append(f"  - {name} (distribution: {distribution}) imported by {where}")

    if missing:
        print("Undeclared third-party imports found:")
        print("\n".join(missing))
        print("\nAdd them to [project.dependencies] or an extra in pyproject.toml.")
        return 1

    print(f"All third-party imports are declared ({len(declared)} distributions).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
