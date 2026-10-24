"""Shared fixtures for the test suite."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY = REPO_ROOT / "examples" / "toy"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def toy_manifest() -> Path:
    return TOY / "manifest.csv"


@pytest.fixture(scope="session")
def toy_labels() -> Path:
    return TOY / "labels.csv"


@pytest.fixture(scope="session")
def toy_pair() -> tuple[Path, Path]:
    """Protein and ligand files of the first toy complex."""
    protein = TOY / "complexes" / "toy01_protein.pdb"
    ligand = TOY / "complexes" / "toy01_ligand.sdf"
    if not protein.is_file() or not ligand.is_file():
        pytest.skip("toy complexes are missing; run python examples/make_toy.py")
    return protein, ligand


@pytest.fixture(scope="session")
def toy_tokens(toy_manifest):
    """Contact tokens of every toy complex, computed once per session."""
    from pgl.features import contact_features

    ids, rows = [], []
    with open(toy_manifest, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ids.append(row["sample_id"])
            rows.append(contact_features(row["protein_path"], row["ligand_path"]).vector)
    return np.asarray(ids), np.stack(rows).astype(np.float32)


@pytest.fixture(scope="session")
def toy_labels_map(toy_labels) -> dict[str, float]:
    with open(toy_labels, encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: float(row["label"]) for row in csv.DictReader(handle)}
