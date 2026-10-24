"""Reading and writing the intermediate files used by the command line.

Two artefacts are exchanged between stages:

* a tokens archive: ``ids`` (str), ``tokens`` (n, 287), ``feature_names``;
* a features archive: ``ids`` (str), ``features`` (n, d), ``feature_names``.

Labels are read from a CSV with one row per sample and the columns
``sample_id`` and ``label`` (``pK`` is accepted as an alias).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM, feature_names


def save_tokens(path: str | Path, ids, tokens: np.ndarray) -> Path:
    """Write a tokens archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != FEATURE_DIM:
        raise ValueError(f"tokens must have shape (n, {FEATURE_DIM}), got {tokens.shape}")
    np.savez_compressed(
        path,
        ids=np.asarray([str(sample_id) for sample_id in ids]),
        tokens=tokens,
        feature_names=np.asarray(feature_names()),
    )
    return path


def load_tokens(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a tokens archive and return ``(ids, tokens)``."""
    with np.load(path, allow_pickle=False) as data:
        if "ids" not in data.files or "tokens" not in data.files:
            raise ValueError(f"{path}: expected 'ids' and 'tokens' arrays")
        ids = np.asarray([str(value) for value in data["ids"]])
        tokens = np.asarray(data["tokens"], dtype=np.float32)
    if tokens.shape != (len(ids), FEATURE_DIM):
        raise ValueError(f"{path}: tokens shape {tokens.shape} does not match {len(ids)} ids")
    return ids, tokens


def save_features(path: str | Path, ids, features: np.ndarray, names=None) -> Path:
    """Write a features archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features, dtype=np.float32)
    np.savez_compressed(
        path,
        ids=np.asarray([str(sample_id) for sample_id in ids]),
        features=features,
        feature_names=np.asarray(
            names if names is not None else [f"f{i}" for i in range(features.shape[1])]
        ),
    )
    return path


def load_features(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a features archive and return ``(ids, features)``.

    A tokens archive is accepted as well, so the readout can be fitted on
    contact tokens alone.
    """
    with np.load(path, allow_pickle=False) as data:
        if "ids" not in data.files:
            raise ValueError(f"{path}: expected an 'ids' array")
        if "features" in data.files:
            features = np.asarray(data["features"], dtype=np.float32)
        elif "tokens" in data.files:
            features = np.asarray(data["tokens"], dtype=np.float32)
        else:
            raise ValueError(f"{path}: expected a 'features' or 'tokens' array")
        ids = np.asarray([str(value) for value in data["ids"]])
    if features.ndim != 2 or features.shape[0] != len(ids):
        raise ValueError(f"{path}: feature shape {features.shape} does not match {len(ids)} ids")
    return ids, features


def load_labels(path: str | Path) -> dict[str, float]:
    """Read ``sample_id -> label`` from a CSV file."""
    path = Path(path)
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        id_field = _first_present(fields, ("sample_id", "id"))
        label_field = _first_present(fields, ("label", "pK", "label_pK"))
        if id_field is None or label_field is None:
            raise ValueError(
                f"{path}: expected an id column (sample_id) and a label column (label or pK); "
                f"found {fields}"
            )
        labels = {}
        for row in reader:
            sample_id = str(row[id_field]).strip()
            value = row[label_field]
            if sample_id == "":
                continue
            labels[sample_id] = float(value)
    if not labels:
        raise ValueError(f"{path}: no labelled rows")
    return labels


def load_id_list(path: str | Path) -> list[str]:
    """Read an ordered ``sample_id`` list from a CSV file.

    The file may contain a single ``sample_id`` column or a full manifest; the
    first column named ``sample_id``/``id`` is used and the row order is kept.
    """
    path = Path(path)
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        field = _first_present(fields, ("sample_id", "id"))
        if field is None:
            raise ValueError(f"{path}: expected a sample_id column; found {fields}")
        ids = [str(row[field]).strip() for row in reader]
    if not ids or any(sample_id == "" for sample_id in ids):
        raise ValueError(f"{path}: empty sample_id in the id list")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicated sample_id in the id list")
    return ids


def _first_present(fields, candidates) -> str | None:
    for name in candidates:
        if name in fields:
            return name
    return None


__all__ = [
    "load_features",
    "load_id_list",
    "load_labels",
    "load_tokens",
    "save_features",
    "save_tokens",
]
