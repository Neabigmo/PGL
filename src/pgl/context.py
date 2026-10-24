"""Frozen protein and ligand context embeddings.

The contact-token representation is combined with two frozen context blocks
that are produced outside this package:

* ``unimol`` (512 dimensions) from the ligand 3D encoder,
* ``saprot`` (480 dimensions) from the structure-aware protein encoder.

This module only defines the file interface (``ids`` + ``vectors``) and the
assembly of both blocks with the contact tokens. It never trains or modifies
the encoders, and rows missing a block are filled with NaN rather than
imputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM, feature_names

CONTEXT_DIMS = {"unimol": 512, "saprot": 480}
CONTEXT_ORDER = ("unimol", "saprot")
FEATURE_DIM_WITH_CONTEXT = FEATURE_DIM + sum(CONTEXT_DIMS.values())  # 1279


@dataclass
class ContextEmbedding:
    """One frozen context block with its sample identifiers."""

    tag: str
    ids: np.ndarray
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if self.tag not in CONTEXT_DIMS:
            raise ValueError(
                f"unknown context tag {self.tag!r}; expected one of {sorted(CONTEXT_DIMS)}"
            )
        self.ids = np.asarray([str(value) for value in self.ids])
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError(f"context vectors must be 2-D, got {self.vectors.shape}")
        if self.vectors.shape[1] != CONTEXT_DIMS[self.tag]:
            raise ValueError(
                f"context '{self.tag}' must have {CONTEXT_DIMS[self.tag]} columns, "
                f"got {self.vectors.shape[1]}"
            )
        if self.vectors.shape[0] != len(self.ids):
            raise ValueError(
                f"context '{self.tag}' has {self.vectors.shape[0]} vectors for {len(self.ids)} ids"
            )
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"context '{self.tag}' has duplicated ids")

    @classmethod
    def load(cls, path: str | Path, tag: str | None = None) -> ContextEmbedding:
        """Load a context block from an ``.npz`` file.

        The archive must contain ``ids`` and ``vectors``. As a convenience,
        ``vectors`` may be stored under the tag name (``unimol``/``saprot``).
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if "ids" not in data.files:
                raise ValueError(f"{path}: context file has no 'ids' array")
            ids = np.asarray([str(value) for value in data["ids"]])
            if "vectors" in data.files:
                vectors = np.asarray(data["vectors"])
            elif tag is not None and tag in data.files:
                vectors = np.asarray(data[tag])
            else:
                candidates = [name for name in data.files if name != "ids"]
                if len(candidates) != 1:
                    raise ValueError(
                        f"{path}: cannot identify the vector array; expected 'vectors'"
                    )
                vectors = np.asarray(data[candidates[0]])
        if tag is None:
            if vectors.shape[1] == CONTEXT_DIMS["unimol"]:
                tag = "unimol"
            elif vectors.shape[1] == CONTEXT_DIMS["saprot"]:
                tag = "saprot"
            else:
                raise ValueError(
                    f"{path}: cannot infer the context tag from width {vectors.shape[1]}"
                )
        return cls(tag=tag, ids=ids, vectors=vectors)

    @classmethod
    def from_mapping(cls, tag: str, vectors: dict[str, np.ndarray]) -> ContextEmbedding:
        """Build a block from a ``{sample_id: vector}`` mapping."""
        ids = np.asarray([str(key) for key in vectors])
        matrix = np.stack([np.asarray(value, dtype=np.float32) for value in vectors.values()])
        return cls(tag=tag, ids=ids, vectors=matrix)

    def lookup(self, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return the block for ``ids`` (NaN where missing) and a presence mask."""
        index = {sample_id: row for row, sample_id in enumerate(self.ids)}
        block = np.full((len(ids), self.vectors.shape[1]), np.nan, dtype=np.float32)
        present = np.zeros(len(ids), dtype=bool)
        for row, sample_id in enumerate(ids):
            position = index.get(str(sample_id))
            if position is not None:
                block[row] = self.vectors[position]
                present[row] = True
        return block, present


def build_feature_matrix(
    tokens: np.ndarray,
    ids,
    context: dict[str, ContextEmbedding] | None = None,
    missing_flags: bool = False,
    feature_dim: int | None = None,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Assemble contact tokens and context blocks into one feature matrix.

    The column order is contact tokens, then ``unimol``, then ``saprot``.
    Missing context rows are NaN; with ``missing_flags`` one indicator column
    per attached block is appended after the blocks.

    Parameters
    ----------
    tokens:
        Contact tokens of shape ``(n, 287)``.
    ids:
        Sample identifiers in row order.
    context:
        Mapping from tag to :class:`ContextEmbedding`. Tags not present are
        skipped and the corresponding columns are absent from the output.
    missing_flags:
        Append a per-block presence indicator.
    feature_dim:
        Expected width (287 or 1279 with both blocks and no flags). Checked
        when given and otherwise inferred.

    Returns
    -------
    matrix, names, block_dims
        ``matrix`` is ``float32``; ``block_dims`` maps block name to width.
    """
    tokens = np.asarray(tokens, dtype=np.float32)
    ids = [str(sample_id) for sample_id in ids]
    if tokens.ndim != 2 or tokens.shape[1] != FEATURE_DIM:
        raise ValueError(f"tokens must have shape (n, {FEATURE_DIM}), got {tokens.shape}")
    if tokens.shape[0] != len(ids):
        raise ValueError(f"tokens have {tokens.shape[0]} rows for {len(ids)} ids")
    context = dict(context or {})
    unknown = sorted(set(context) - set(CONTEXT_DIMS))
    if unknown:
        raise ValueError(f"unknown context tags: {unknown}")

    columns = [tokens]
    names = feature_names()
    block_dims = {"contact_tokens": FEATURE_DIM}
    present_blocks: dict[str, np.ndarray] = {}
    for tag in CONTEXT_ORDER:
        block = context.get(tag)
        if block is None:
            continue
        matrix, present = block.lookup(ids)
        columns.append(matrix)
        present_blocks[tag] = present
        names += [f"ctx_{tag}_{i}" for i in range(CONTEXT_DIMS[tag])]
        block_dims[f"ctx_{tag}"] = CONTEXT_DIMS[tag]
    if missing_flags:
        for tag, present in present_blocks.items():
            columns.append(present.astype(np.float32).reshape(-1, 1))
            names.append(f"ctx_{tag}_present")
            block_dims[f"ctx_{tag}_present"] = 1
    matrix = np.concatenate(columns, axis=1).astype(np.float32)
    if feature_dim is not None and matrix.shape[1] != feature_dim:
        raise ValueError(f"feature matrix has {matrix.shape[1]} columns, expected {feature_dim}")
    return matrix, names, block_dims


__all__ = [
    "CONTEXT_DIMS",
    "CONTEXT_ORDER",
    "ContextEmbedding",
    "FEATURE_DIM_WITH_CONTEXT",
    "build_feature_matrix",
]
