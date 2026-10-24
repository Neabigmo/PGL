"""End-to-end numerical regression on the bundled toy complex.

The test pins the chain structure -> contact tokens -> context assembly ->
schema -> public readout -> prediction. It uses the small fixture readout in
`release/fixtures/`, so it checks the interface between the released code, the
feature layout and a checkpoint without re-running any benchmark.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pgl.context import ContextEmbedding, build_feature_matrix
from pgl.features import contact_features
from pgl.readout import XGBReadout
from pgl.schema import load_schema, ordered_feature_names


@pytest.fixture(scope="module")
def fixture(repo_root):
    path = repo_root / "release" / "fixtures" / "toy01_reference.json"
    return repo_root, json.loads(path.read_text(encoding="utf-8"))


def test_toy_tokens_match_the_reference_fixture(fixture):
    repo_root, reference = fixture
    result = contact_features(
        repo_root / "examples" / "toy" / "complexes" / "toy01_protein.pdb",
        repo_root / "examples" / "toy" / "complexes" / "toy01_ligand.sdf",
    )
    assert result.vector.shape == (reference["feature_dim"],) == (287,)
    np.testing.assert_allclose(
        result.vector, np.asarray(reference["values"], dtype=np.float32), rtol=0, atol=1e-5
    )


def test_toy_end_to_end_prediction_matches_the_reference(fixture):
    repo_root, reference = fixture
    result = contact_features(
        repo_root / "examples" / "toy" / "complexes" / "toy01_protein.pdb",
        repo_root / "examples" / "toy" / "complexes" / "toy01_ligand.sdf",
    )
    context = reference["context"]
    blocks = {
        "unimol": ContextEmbedding(
            "unimol", ["toy01"], np.asarray([context["unimol"]], dtype=np.float32)
        ),
        "saprot": ContextEmbedding(
            "saprot", ["toy01"], np.asarray([context["saprot"]], dtype=np.float32)
        ),
    }
    matrix, names, block_dims = build_feature_matrix(
        np.asarray([result.vector], dtype=np.float32), ["toy01"], blocks
    )
    assert matrix.shape == (1, 1279)
    assert names == ordered_feature_names()
    assert block_dims == {"contact_tokens": 287, "ctx_unimol": 512, "ctx_saprot": 480}

    schema = load_schema(repo_root / "release" / "schema.json")
    readout = XGBReadout.load(repo_root / "release" / "fixtures" / "toy_readout", schema=schema)
    prediction = float(readout.predict(matrix)[0])
    assert prediction == pytest.approx(float(reference["y_pred"]), abs=1e-4)
