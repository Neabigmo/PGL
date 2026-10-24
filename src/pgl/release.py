"""Single source of truth for the released PGL model.

``release.json`` records the model identifier, the feature schema, the sample
lists used for evaluation, the metric definitions, the checkpoint layout and
the released result tables. Tests compare it against the code, the package
metadata and the bundled data files, so a change in any of them has to update
the release record in the same commit.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

RELEASE_FILE = "release.json"


def load_release() -> dict:
    """Load the release record from the package data."""
    return json.loads(files("pgl").joinpath(RELEASE_FILE).read_text(encoding="utf-8"))


def repo_root() -> Path | None:
    """Repository root of a source checkout, or ``None`` for an installed wheel."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else None


def release_file(relative: str) -> Path:
    """Resolve a path recorded in ``release.json`` against the source checkout."""
    root = repo_root()
    if root is None:
        raise FileNotFoundError(
            "released data files live in the repository checkout, not in the installed package"
        )
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"released file is missing: {path}")
    return path


RELEASE = load_release()
MODEL_ID = RELEASE["model"]["model_id"]
RELEASE_VERSION = RELEASE["release"]["version"]
FEATURE_BLOCKS = [(block["name"], block["dim"]) for block in RELEASE["feature_schema"]["blocks"]]
FEATURE_DIM_TOTAL = RELEASE["feature_schema"]["total"]
SEEDS = tuple(RELEASE["model"]["seeds"])

__all__ = [
    "FEATURE_BLOCKS",
    "FEATURE_DIM_TOTAL",
    "MODEL_ID",
    "RELEASE",
    "RELEASE_VERSION",
    "SEEDS",
    "load_release",
    "release_file",
    "repo_root",
]
