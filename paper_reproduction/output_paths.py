"""Choose a writable result folder without writing into the research data archive."""

from __future__ import annotations

import os
from pathlib import Path


def prepare_output_directory(data_root: Path, path: Path) -> Path:
    root = Path(data_root).resolve(strict=True)
    # abspath normalizes relative paths and '..' before checking redirection.
    requested = Path(os.path.abspath(Path(path).expanduser()))
    target = requested.resolve()
    if os.path.normcase(str(target)) != os.path.normcase(str(requested)):
        raise ValueError("Output path must not be redirected by a filesystem link")
    if target == root or target.is_relative_to(root):
        raise ValueError("Output must be outside the research data archive")
    target.mkdir(parents=True, exist_ok=True)
    for directory, subdirectories, files in os.walk(target, followlinks=False):
        for name in subdirectories + files:
            item = Path(directory) / name
            if item.resolve() != item:
                raise ValueError("Output directory contains a redirected filesystem path")
    return target
