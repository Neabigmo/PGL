"""Consistency tests for the release record and the released data files."""

from __future__ import annotations

import csv
import hashlib
import json
import re

import numpy as np
import pytest

import pgl
from pgl.context import (
    CONTEXT_DIMS,
    FEATURE_DIM_WITH_CONTEXT,
    ContextEmbedding,
    build_feature_matrix,
)
from pgl.features import FEATURE_DIM, feature_names, token_names
from pgl.readout import XGB_PRESET
from pgl.release import (
    FEATURE_BLOCKS,
    FEATURE_DIM_TOTAL,
    MODEL_ID,
    RELEASE,
    RELEASE_VERSION,
    SEEDS,
    repo_root,
)

EXPECTED_BLOCKS = [("contact_tokens", 287), ("ctx_unimol", 512), ("ctx_saprot", 480)]


def _root():
    root = repo_root()
    if root is None:
        pytest.skip("released data files require a source checkout")
    return root


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_record_matches_code():
    assert RELEASE_VERSION == pgl.__version__ == "1.0.0"
    assert MODEL_ID == "PGL-1.0.0"
    assert FEATURE_BLOCKS == EXPECTED_BLOCKS
    assert FEATURE_DIM_TOTAL == 1279 == FEATURE_DIM + sum(CONTEXT_DIMS.values())
    assert FEATURE_DIM == 287
    assert len(feature_names()) == FEATURE_DIM
    assert len(token_names()) == 57
    assert CONTEXT_DIMS == {"unimol": 512, "saprot": 480}
    assert tuple(SEEDS) == (20261022, 20261023, 20261024)
    assert XGB_PRESET == RELEASE["model"]["hyperparameters"]


def test_release_record_block_order_and_subblocks():
    schema = RELEASE["feature_schema"]
    assert [block["order"] for block in schema["blocks"]] == [0, 1, 2]
    assert [block["dim"] for block in schema["blocks"]] == [287, 512, 480]
    subblocks = schema["blocks"][0]["subblocks"]
    assert [(item["name"], item["dim"]) for item in subblocks] == [
        ("chemistry", 15),
        ("token_statistics", 114),
        ("atomic_contacts", 158),
    ]
    assert sum(item["dim"] for item in subblocks) == 287


def test_versions_agree_across_metadata():
    root = _root()
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(rf"^version:\s*{re.escape(pgl.__version__)}\s*$", citation, re.MULTILINE)
    assert f'version = "{pgl.__version__}"' in pyproject
    assert RELEASE["release"]["paper_title"] in citation


def test_feature_matrix_is_1279_columns(toy_tokens):
    ids, tokens = toy_tokens
    assert tokens.shape[1] == 287
    rng = np.random.default_rng(0)
    context = {
        tag: ContextEmbedding(tag, ids, rng.normal(size=(len(ids), dim)).astype(np.float32))
        for tag, dim in CONTEXT_DIMS.items()
    }
    matrix, names, blocks = build_feature_matrix(tokens, ids, context)
    assert matrix.shape == (len(ids), FEATURE_DIM_WITH_CONTEXT) == (len(ids), 1279)
    assert len(names) == matrix.shape[1]
    assert list(blocks) == ["contact_tokens", "ctx_unimol", "ctx_saprot"]


def test_sample_lists_match_the_release_record():
    root = _root()
    splits = RELEASE["sample_lists"]["splits"]
    assert splits, "release record must list the evaluation splits"
    for name, entry in splits.items():
        path = root / entry["path"]
        assert path.is_file(), f"{name}: missing {entry['path']}"
        rows = _read_csv(path)
        ids = [row["sample_id"] for row in rows]
        assert len(ids) == entry["n"], f"{name}: row count changed"
        assert len(ids) == len(set(ids)), f"{name}: duplicated sample id"
        assert all(sample_id.strip() for sample_id in ids), f"{name}: empty sample id"
        assert _digest(path) == entry["sha256"], f"{name}: split hash changed"


def test_released_ids_are_unique_across_files():
    root = _root()
    casf_ids = {
        row["sample_id"]
        for row in _read_csv(root / RELEASE["sample_lists"]["splits"]["casf2016_test"]["path"])
    }
    predictions = _read_csv(
        root / RELEASE["results"]["predictions"]["casf2016_seed20261022"]["path"]
    )
    predicted_ids = [row["sample_id"] for row in predictions]
    assert len(predicted_ids) == len(set(predicted_ids)) == 285
    assert set(predicted_ids) == casf_ids
    values = np.asarray([float(row["y_pred"]) for row in predictions])
    assert np.all(np.isfinite(values))
    assert values.std() > 0.1


def test_every_released_identifier_column_is_unique():
    """Sample, target and pose identifiers are unique in every released table."""
    root = _root()
    identifier_columns = {"sample_id", "target", "target_id", "pose", "pose_id"}
    checked = 0
    for path in sorted((root / "release").rglob("*.csv")):
        rows = _read_csv(path)
        if not rows:
            continue
        columns = [name for name in rows[0] if name.lower() in identifier_columns]
        for column in columns:
            values = [row[column] for row in rows]
            assert len(values) == len(set(values)), f"{path.name}: duplicated {column}"
            assert all(value.strip() for value in values), f"{path.name}: empty {column}"
            checked += 1
    assert checked > 0


def test_figure_source_matches_result_tables():
    root = _root()
    main_table = root / RELEASE["results"]["tables"]["exp1_main_casf2016"]["path"]
    figure = root / RELEASE["results"]["figure_source"]["fig2_casf2016"]["path"]
    table_rows = [row for row in _read_csv(main_table) if row["status"] == "ok"]
    figure_rows = _read_csv(figure)
    assert list(figure_rows[0]) == [
        "method",
        "n_test",
        "rmse_mean",
        "rmse_std",
        "pearson_mean",
        "spearman_mean",
    ]
    assert len(figure_rows) == len(table_rows) == 17
    assert [row["method"] for row in figure_rows] == [row["method"] for row in table_rows]
    for figure_row, table_row in zip(figure_rows, table_rows, strict=True):
        for column in ("rmse_mean", "rmse_std", "pearson_mean", "spearman_mean"):
            assert float(figure_row[column]) == pytest.approx(float(table_row[column]), abs=1e-12)
    pgl_row = next(row for row in figure_rows if row["method"] == "pgl")
    assert float(pgl_row["rmse_mean"]) == pytest.approx(
        RELEASE["results"]["benchmarks"]["casf2016"]["rmse_mean"], abs=1e-12
    )
    assert min(float(row["rmse_mean"]) for row in figure_rows) == pytest.approx(
        float(pgl_row["rmse_mean"]), abs=1e-12
    )


@pytest.mark.parametrize(
    "benchmark,table",
    [
        ("casf2016", "exp1_main_casf2016"),
        ("bdb2020plus", "exp2_bdb2020plus"),
        ("lp_pdbbind", "exp2_lp_pdbbind"),
        ("pdbbind2013", "exp2_pdbbind2013"),
    ],
)
def test_benchmark_values_match_tables(benchmark, table):
    root = _root()
    path = root / RELEASE["results"]["tables"][table]["path"]
    row = next(item for item in _read_csv(path) if item["method"] == "pgl")
    recorded = RELEASE["results"]["benchmarks"][benchmark]
    assert int(row["seeds"]) == int(recorded["seeds"]) == len(SEEDS)
    assert float(row["rmse_mean"]) == pytest.approx(recorded["rmse_mean"], abs=1e-12)
    assert float(row["rmse_std"]) == pytest.approx(recorded["rmse_std"], abs=1e-12)
    assert float(row["pearson_mean"]) == pytest.approx(recorded["pearson_mean"], abs=1e-12)


def test_released_files_are_hashed():
    root = _root()
    for group in ("tables", "figure_source"):
        for entry in RELEASE["results"][group].values():
            assert _digest(root / entry["path"]) == entry["sha256"]
    for entry in RELEASE["results"]["predictions"].values():
        assert _digest(root / entry["path"]) == entry["sha256"]


def test_readme_states_the_released_model():
    root = _root()
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "1,279" in readme
    assert RELEASE_VERSION in readme
    assert "src/pgl/release.json" in readme
    assert "1.301 ± 0.017" in readme
    assert "287 contact-token" in readme
    assert "legacy/" in readme
    assert RELEASE["release"]["tag"] in readme


def test_release_json_is_valid_json():
    root = _root()
    record = json.loads((root / "src/pgl/release.json").read_text(encoding="utf-8"))
    assert record["release"]["version"] == pgl.__version__
    assert record["model"]["checkpoint_files"] == {"model": "readout.json", "config": "config.json"}
    assert record["metrics"]["aggregation"]
