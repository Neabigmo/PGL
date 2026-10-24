"""Tests of the frozen context interface and feature assembly."""

from __future__ import annotations

import numpy as np
import pytest

from pgl.context import (
    CONTEXT_DIMS,
    FEATURE_DIM_WITH_CONTEXT,
    ContextEmbedding,
    build_feature_matrix,
)
from pgl.features import FEATURE_DIM

IDS = ["a", "b", "c"]


def _tokens(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, FEATURE_DIM)).astype(np.float32)


def _context(tag: str, ids: list[str]) -> ContextEmbedding:
    rng = np.random.default_rng(1 if tag == "unimol" else 2)
    return ContextEmbedding(
        tag=tag,
        ids=np.asarray(ids),
        vectors=rng.normal(size=(len(ids), CONTEXT_DIMS[tag])).astype(np.float32),
    )


def test_full_context_assembly_has_1279_columns():
    matrix, names, dims = build_feature_matrix(
        _tokens(3), IDS, {"unimol": _context("unimol", IDS), "saprot": _context("saprot", IDS)}
    )
    assert matrix.shape == (3, FEATURE_DIM_WITH_CONTEXT) == (3, 1279)
    assert len(names) == matrix.shape[1]
    assert dims == {"contact_tokens": 287, "ctx_unimol": 512, "ctx_saprot": 480}


def test_context_columns_follow_the_contact_tokens():
    tokens = _tokens(3)
    matrix, _names, _dims = build_feature_matrix(tokens, IDS, {"unimol": _context("unimol", IDS)})
    np.testing.assert_array_equal(matrix[:, :FEATURE_DIM], tokens)
    assert matrix.shape[1] == FEATURE_DIM + CONTEXT_DIMS["unimol"]


def test_missing_rows_are_nan_instead_of_imputed():
    context = _context("unimol", ["a", "c"])
    matrix, _names, _dims = build_feature_matrix(_tokens(3), IDS, {"unimol": context})
    assert np.all(np.isfinite(matrix[0, FEATURE_DIM:]))
    assert np.all(np.isnan(matrix[1, FEATURE_DIM:]))
    assert np.all(np.isfinite(matrix[2, FEATURE_DIM:]))


def test_present_flags_mark_available_context():
    context = _context("saprot", ["a", "c"])
    matrix, names, dims = build_feature_matrix(
        _tokens(3), IDS, {"saprot": context}, missing_flags=True
    )
    flags = matrix[:, -1]
    np.testing.assert_array_equal(flags, [1.0, 0.0, 1.0])
    assert names[-1] == "ctx_saprot_present"
    assert dims["ctx_saprot_present"] == 1
    assert matrix.shape[1] == FEATURE_DIM + CONTEXT_DIMS["saprot"] + 1


def test_context_embedding_round_trip(tmp_path):
    original = _context("unimol", IDS)
    path = tmp_path / "unimol.npz"
    np.savez(path, ids=original.ids, vectors=original.vectors)
    loaded = ContextEmbedding.load(path)
    assert loaded.tag == "unimol"
    np.testing.assert_array_equal(loaded.ids, original.ids)
    np.testing.assert_array_equal(loaded.vectors, original.vectors)


def test_context_embedding_rejects_wrong_width():
    with pytest.raises(ValueError):
        ContextEmbedding(tag="unimol", ids=["a"], vectors=np.zeros((1, 480), dtype=np.float32))


def test_context_embedding_requires_ids(tmp_path):
    path = tmp_path / "vectors.npz"
    np.savez(path, vectors=np.zeros((2, 512), dtype=np.float32))
    with pytest.raises(ValueError):
        ContextEmbedding.load(path, "unimol")
