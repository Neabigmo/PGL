"""Tests of the 287-dimensional contact-token representation."""

from __future__ import annotations

import numpy as np

from pgl.features import (
    ATOMIC_DIM,
    BLOCK_SLICES,
    CHEMISTRY_DIM,
    FEATURE_DIM,
    TOKEN_DIM,
    TOKEN_STAT_DIM,
    contact_features,
    feature_names,
    token_names,
)


def test_block_widths_sum_to_feature_dim():
    assert CHEMISTRY_DIM + TOKEN_STAT_DIM + ATOMIC_DIM == FEATURE_DIM == 287
    assert BLOCK_SLICES["chemistry"] == (0, 15)
    assert BLOCK_SLICES["token_statistics"] == (15, 129)
    assert BLOCK_SLICES["atomic_contacts"] == (129, 287)


def test_feature_names_cover_every_column():
    names = feature_names()
    assert len(names) == FEATURE_DIM
    assert len(set(names)) == FEATURE_DIM
    assert len(token_names()) == TOKEN_DIM == 57


def test_toy_tokens_are_finite(toy_pair):
    result = contact_features(*toy_pair)
    assert result.vector.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(result.vector))
    assert result.token_matrix.shape[1] == TOKEN_DIM
    assert result.counts["n_contacts"] > 0


def test_contact_tokens_are_deterministic(toy_pair):
    first = contact_features(*toy_pair).vector
    second = contact_features(*toy_pair).vector
    np.testing.assert_array_equal(first, second)


def test_contact_tokens_are_label_free_across_calls(toy_pair):
    """The representation depends only on the structure, not on call order."""
    other = contact_features(*toy_pair)
    again = contact_features(*toy_pair)
    np.testing.assert_allclose(other.token_matrix, again.token_matrix, rtol=0, atol=0)
