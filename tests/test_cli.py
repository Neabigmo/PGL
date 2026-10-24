"""Tests of the command line interface and of the README quickstart."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from pgl.cli import main
from pgl.features import FEATURE_DIM


def test_features_single_complex(toy_pair, tmp_path):
    protein, ligand = toy_pair
    out = tmp_path / "toy01.npz"
    code = main(
        [
            "features",
            "--protein",
            str(protein),
            "--ligand",
            str(ligand),
            "--id",
            "toy01",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    with np.load(out, allow_pickle=False) as data:
        assert list(data["ids"]) == ["toy01"]
        assert data["tokens"].shape == (1, FEATURE_DIM)


def test_readme_quickstart(tmp_path, capsys, monkeypatch, repo_root):
    """Run the three quickstart commands on the bundled toy dataset."""
    monkeypatch.chdir(repo_root)
    tokens = tmp_path / "tokens.npz"
    model = tmp_path / "readout"
    predictions = tmp_path / "predictions.csv"

    assert (
        main(
            [
                "features",
                "--manifest",
                "examples/toy/manifest.csv",
                "--out",
                str(tokens),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "train",
                "--features",
                str(tokens),
                "--labels",
                "examples/toy/labels.csv",
                "--out",
                str(model),
                "--threads",
                "1",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "predict",
                "--model",
                str(model),
                "--features",
                str(tokens),
                "--out",
                str(predictions),
            ]
        )
        == 0
    )

    with np.load(tokens, allow_pickle=False) as data:
        assert data["tokens"].shape[1] == FEATURE_DIM
    with open(predictions, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12 * 5
    assert all(row["sample_id"] for row in rows)
    assert all(np.isfinite(float(row["y_pred"])) for row in rows)


def test_features_manifest_skips_unusable_complex(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,protein_path,ligand_path\n" "bad,missing_protein.pdb,missing_ligand.sdf\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        main(["features", "--manifest", str(manifest), "--out", str(tmp_path / "x.npz")])


def test_train_requires_labels_for_every_sample(toy_tokens, tmp_path, capsys):
    ids, tokens = toy_tokens
    features = tmp_path / "features.npz"
    np.savez(features, ids=ids, tokens=tokens)
    labels = tmp_path / "labels.csv"
    labels.write_text("sample_id,label\n" + f"{ids[0]},5.0\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "train",
                "--features",
                str(features),
                "--labels",
                str(labels),
                "--out",
                str(tmp_path / "m"),
            ]
        )


def test_train_with_id_list(toy_tokens, toy_labels_map, tmp_path, capsys):
    ids, tokens = toy_tokens
    features = tmp_path / "features.npz"
    np.savez(features, ids=ids, tokens=tokens)
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "sample_id,label\n"
        + "\n".join(f"{sample_id},{toy_labels_map[sample_id]}" for sample_id in ids)
        + "\n",
        encoding="utf-8",
    )
    keep = ids[:10]
    id_list = tmp_path / "ids.csv"
    id_list.write_text("sample_id\n" + "\n".join(keep) + "\n", encoding="utf-8")
    model = tmp_path / "model"
    assert (
        main(
            [
                "train",
                "--features",
                str(features),
                "--labels",
                str(labels),
                "--train-manifest",
                str(id_list),
                "--out",
                str(model),
                "--threads",
                "1",
            ]
        )
        == 0
    )
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    assert config["n_train"] == 10
    assert config["feature_dim"] == FEATURE_DIM


def test_train_with_id_list_rejects_unknown_ids(toy_tokens, toy_labels_map, tmp_path):
    ids, tokens = toy_tokens
    features = tmp_path / "features.npz"
    np.savez(features, ids=ids, tokens=tokens)
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "sample_id,label\n" + f"{ids[0]},{toy_labels_map[ids[0]]}\n", encoding="utf-8"
    )
    id_list = tmp_path / "ids.csv"
    id_list.write_text("sample_id\ntoy01\nnot-a-sample\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "train",
                "--features",
                str(features),
                "--labels",
                str(labels),
                "--train-manifest",
                str(id_list),
                "--out",
                str(tmp_path / "model"),
            ]
        )


def test_info_prints_feature_layout(capsys):
    assert main(["info"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["contact_tokens"] == FEATURE_DIM
    assert printed["features_with_context"] == 1279
    assert len(printed["feature_schema_sha256"]) == 64
