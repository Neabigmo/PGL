"""Command line interface: ``pgl features|context|train|predict``."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import __version__
from .context import CONTEXT_DIMS, ContextEmbedding, build_feature_matrix
from .data import load_features, load_id_list, load_labels, load_tokens, save_features, save_tokens
from .features import FEATURE_DIM, contact_features
from .readout import DEFAULT_SEED, XGB_PRESET, XGBReadout
from .schema import load_schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pgl",
        description="Persistent Gauss-linking contact tokens for protein-ligand affinity.",
    )
    parser.add_argument("--version", action="version", version=f"pgl {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    features = subparsers.add_parser(
        "features", help="compute 287-D contact tokens from protein-ligand structures"
    )
    features.add_argument("--protein", help="protein structure (PDB)")
    features.add_argument("--ligand", help="ligand structure (SDF or MOL2)")
    features.add_argument(
        "--manifest",
        help="CSV with columns sample_id, protein_path, ligand_path (batch mode)",
    )
    features.add_argument("--out", required=True, help="output tokens archive (.npz)")
    features.add_argument("--id", help="sample id for single-complex mode")

    context = subparsers.add_parser(
        "context", help="attach frozen 512/480 context embeddings to tokens"
    )
    context.add_argument("--tokens", required=True, help="tokens archive")
    context.add_argument("--unimol", help="ligand context archive (512 columns)")
    context.add_argument("--saprot", help="protein context archive (480 columns)")
    context.add_argument("--out", required=True, help="output features archive (.npz)")
    context.add_argument(
        "--present-flags",
        action="store_true",
        help="append one presence indicator column per context block",
    )

    train = subparsers.add_parser("train", help="fit the XGBoost readout")
    train.add_argument("--features", required=True, help="features archive (.npz)")
    train.add_argument("--labels", required=True, help="CSV with sample_id and label")
    train.add_argument(
        "--train-manifest",
        help="CSV with a sample_id column; train only on these rows",
    )
    train.add_argument("--schema", help="feature schema JSON to bind to the checkpoint")
    train.add_argument("--out", required=True, help="output model directory")
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--threads", type=int, default=8)

    predict = subparsers.add_parser("predict", help="predict pK with a fitted readout")
    predict.add_argument("--model", required=True, help="model directory")
    predict.add_argument("--features", required=True, help="features archive (.npz)")
    predict.add_argument("--schema", help="feature schema JSON to check the checkpoint against")
    predict.add_argument("--out", required=True, help="output predictions CSV")

    subparsers.add_parser("info", help="print feature layout and default settings")

    args = parser.parse_args(argv)
    if args.command == "features":
        return _features(args)
    if args.command == "context":
        return _context(args)
    if args.command == "train":
        return _train(args)
    if args.command == "predict":
        return _predict(args)
    if args.command == "info":
        return _info()
    return 1


def _features(args: argparse.Namespace) -> int:
    pairs = _read_manifest(args.manifest) if args.manifest else None
    if pairs is None:
        if not args.protein or not args.ligand:
            raise SystemExit("features: give --manifest or both --protein and --ligand")
        sample_id = args.id or Path(args.protein).stem
        pairs = [(sample_id, args.protein, args.ligand)]
    ids, rows, skipped = [], [], []
    for sample_id, protein_path, ligand_path in pairs:
        try:
            result = contact_features(protein_path, ligand_path)
        except (ValueError, OSError) as error:
            skipped.append({"sample_id": sample_id, "reason": str(error)})
            continue
        ids.append(sample_id)
        rows.append(result.vector)
    if not rows:
        raise SystemExit(f"features: no complex could be featurized ({len(skipped)} failed)")
    tokens = np.stack(rows).astype(np.float32)
    save_tokens(args.out, ids, tokens)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n": len(ids),
                "dim": int(tokens.shape[1]),
                "skipped": skipped,
            },
            indent=2,
        )
    )
    return 0


def _read_manifest(path: str) -> list[tuple[str, str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for required in ("sample_id", "protein_path", "ligand_path"):
            if required not in fields:
                raise SystemExit(f"features: {path} is missing the column {required!r}")
        pairs = [
            (str(row["sample_id"]), str(row["protein_path"]), str(row["ligand_path"]))
            for row in reader
        ]
    if not pairs:
        raise SystemExit(f"features: {path} has no rows")
    return pairs


def _context(args: argparse.Namespace) -> int:
    ids, tokens = load_tokens(args.tokens)
    blocks: dict[str, ContextEmbedding] = {}
    for tag, path in (("unimol", args.unimol), ("saprot", args.saprot)):
        if path:
            blocks[tag] = ContextEmbedding.load(path, tag)
    if not blocks:
        raise SystemExit("context: give --unimol and/or --saprot")
    matrix, names, dims = build_feature_matrix(
        tokens, ids, blocks, missing_flags=args.present_flags
    )
    save_features(args.out, ids, matrix, names)
    print(json.dumps({"out": str(args.out), "n": len(ids), "blocks": dims}, indent=2))
    return 0


def _train(args: argparse.Namespace) -> int:
    ids, features = load_features(args.features)
    labels = load_labels(args.labels)
    if args.train_manifest:
        wanted = load_id_list(args.train_manifest)
        index = {sample_id: row for row, sample_id in enumerate(ids)}
        missing_ids = [sample_id for sample_id in wanted if sample_id not in index]
        if missing_ids:
            raise SystemExit(
                f"train: {len(missing_ids)} ids from --train-manifest are absent from the "
                f"feature archive (first: {missing_ids[:3]})"
            )
        rows = np.asarray([index[sample_id] for sample_id in wanted], dtype=int)
        ids = [ids[row] for row in rows]
        features = features[rows]
    missing = [sample_id for sample_id in ids if sample_id not in labels]
    if missing:
        raise SystemExit(f"train: {len(missing)} samples have no label (first: {missing[:3]})")
    labels_vector = np.asarray([labels[sample_id] for sample_id in ids], dtype=np.float64)
    schema = load_schema(args.schema) if args.schema else None
    readout = XGBReadout(seed=args.seed, n_threads=args.threads, schema=schema)
    readout.fit(features, labels_vector)
    readout.save(args.out)
    print(json.dumps({"out": str(args.out), **readout.config()}, indent=2))
    return 0


def _predict(args: argparse.Namespace) -> int:
    schema = load_schema(args.schema) if args.schema else None
    readout = XGBReadout.load(args.model, schema=schema)
    ids, features = load_features(args.features)
    predictions = readout.predict(features)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "y_pred"])
        for sample_id, value in zip(ids, predictions, strict=False):
            writer.writerow([sample_id, f"{float(value):.6f}"])
    print(
        json.dumps(
            {
                "out": str(out),
                "n": len(ids),
                "y_pred_mean": float(np.mean(predictions)),
                "y_pred_std": float(np.std(predictions)),
            },
            indent=2,
        )
    )
    return 0


def _info() -> int:
    from .release import MODEL_ID, RELEASE, RELEASE_VERSION
    from .schema import schema_sha256

    print(
        json.dumps(
            {
                "version": __version__,
                "release": RELEASE_VERSION,
                "model_id": MODEL_ID,
                "checkpoint_id": RELEASE["model"]["checkpoint_id"],
                "contact_tokens": FEATURE_DIM,
                "context": CONTEXT_DIMS,
                "features_with_context": FEATURE_DIM + sum(CONTEXT_DIMS.values()),
                "feature_schema_sha256": schema_sha256(),
                "readout": "xgboost",
                "seeds": RELEASE["model"]["seeds"],
                "hyperparameters": XGB_PRESET,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
