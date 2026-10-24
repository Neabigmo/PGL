"""Tests of the XGBoost readout."""

from __future__ import annotations

import numpy as np
import pytest

from pgl.metrics import all_metrics
from pgl.readout import DEFAULT_SEED, XGB_PRESET, XGBReadout

SMALL = {"n_estimators": 40, "max_depth": 3}


def _fit(tokens, labels, seed: int = DEFAULT_SEED) -> XGBReadout:
    return XGBReadout(seed=seed, n_threads=1, hyperparameters=SMALL).fit(tokens, labels)


def test_default_preset_matches_the_paper_configuration():
    readout = XGBReadout()
    assert readout.hyperparameters == XGB_PRESET
    assert readout.hyperparameters["n_estimators"] == 708
    assert readout.seed == DEFAULT_SEED


def test_fit_predict_shapes(toy_tokens, toy_labels_map):
    ids, tokens = toy_tokens
    labels = np.asarray([toy_labels_map[sample_id] for sample_id in ids])
    readout = _fit(tokens, labels)
    predictions = readout.predict(tokens)
    assert predictions.shape == (len(ids),)
    assert np.all(np.isfinite(predictions))
    assert readout.feature_dim == tokens.shape[1]
    assert np.isfinite(readout.training_metrics["rmse"])


def test_readout_is_deterministic(toy_tokens, toy_labels_map):
    ids, tokens = toy_tokens
    labels = np.asarray([toy_labels_map[sample_id] for sample_id in ids])
    first = _fit(tokens, labels).predict(tokens)
    second = _fit(tokens, labels).predict(tokens)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_save_and_load_round_trip(toy_tokens, toy_labels_map, tmp_path):
    ids, tokens = toy_tokens
    labels = np.asarray([toy_labels_map[sample_id] for sample_id in ids])
    readout = _fit(tokens, labels)
    readout.save(tmp_path / "model")
    assert (tmp_path / "model" / "readout.json").is_file()
    assert (tmp_path / "model" / "config.json").is_file()
    loaded = XGBReadout.load(tmp_path / "model")
    np.testing.assert_allclose(loaded.predict(tokens), readout.predict(tokens), rtol=0, atol=1e-6)
    assert loaded.feature_dim == readout.feature_dim


def test_predict_rejects_feature_dimension_mismatch(toy_tokens, toy_labels_map):
    ids, tokens = toy_tokens
    labels = np.asarray([toy_labels_map[sample_id] for sample_id in ids])
    readout = _fit(tokens[:, :100], labels)
    with pytest.raises(ValueError):
        readout.predict(tokens)


def test_metrics_on_a_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = all_metrics(y, y)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["spearman"] == pytest.approx(1.0)
