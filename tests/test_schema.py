"""Tests of the ordered feature schema and its use by the readout."""

from __future__ import annotations

import json

import numpy as np
import pytest

import pgl
from pgl.readout import XGBReadout
from pgl.schema import (
    BLOCKS,
    build_schema,
    load_schema,
    ordered_feature_names,
    schema_payload,
    schema_sha256,
    validate_schema,
)


def _random_features(rows: int, columns: int = 1279) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(rows, columns)).astype(np.float32)
    labels = 5.0 + 0.5 * features[:, 0] + 0.2 * rng.normal(size=rows)
    return features, labels


def test_schema_has_ordered_names_for_every_column():
    schema = build_schema()
    names = ordered_feature_names()
    assert schema["total"] == 1279 == len(names)
    assert [block["name"] for block in schema["blocks"]] == [name for name, _ in BLOCKS]
    assert schema["blocks"][0]["names"] == pgl.feature_names()
    assert names[:287] == schema["blocks"][0]["names"]
    assert names[287] == "ctx_unimol_0" and names[798] == "ctx_unimol_511"
    assert names[799] == "ctx_saprot_0" and names[-1] == "ctx_saprot_479"
    assert [block["offset"] for block in schema["blocks"]] == [0, 287, 799]


def test_schema_hash_is_stable_and_matches_the_release_copy(repo_root):
    schema = build_schema()
    assert schema_sha256() == schema["sha256"] == schema_sha256(build_schema())
    released = json.loads((repo_root / "release" / "schema.json").read_text(encoding="utf-8"))
    assert released == schema
    validate_schema(released)


def test_schema_validation_rejects_wrong_order():
    schema = schema_payload()
    schema["blocks"][1], schema["blocks"][2] = schema["blocks"][2], schema["blocks"][1]
    with pytest.raises(ValueError):
        validate_schema(schema)


def test_load_schema_from_release_file(repo_root):
    schema = load_schema(repo_root / "release" / "schema.json")
    assert schema["total"] == 1279
    assert schema["sha256"] == schema_sha256(schema)


def test_readout_accepts_matching_schema(tmp_path):
    schema = build_schema()
    features, labels = _random_features(40)
    readout = XGBReadout(seed=7, n_threads=1, schema=schema).fit(features, labels)
    directory = readout.save(tmp_path / "model")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    assert config["feature_schema_sha256"] == schema["sha256"]
    assert config["feature_schema_total"] == 1279
    loaded = XGBReadout.load(directory, schema=schema)
    np.testing.assert_allclose(loaded.predict(features), readout.predict(features), atol=1e-6)


def test_readout_rejects_mismatched_schema(tmp_path, repo_root):
    schema = build_schema()
    features, labels = _random_features(30)
    directory = (
        XGBReadout(seed=7, n_threads=1, schema=schema)
        .fit(features, labels)
        .save(tmp_path / "model")
    )
    other = json.loads((repo_root / "release" / "schema.json").read_text(encoding="utf-8"))
    other["blocks"][0]["names"] = list(reversed(other["blocks"][0]["names"]))
    with pytest.raises(ValueError):
        XGBReadout.load(directory, schema=other)


def test_readout_rejects_schema_width_mismatch():
    schema = build_schema()
    features, labels = _random_features(20, columns=287)
    with pytest.raises(ValueError):
        XGBReadout(seed=7, n_threads=1, schema=schema).fit(features, labels)
