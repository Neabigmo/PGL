"""Small command-line interface for feature extraction and inspection."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from . import PGLConfig, PGLRep, PGLVocabulary


def main() -> None:
    parser = argparse.ArgumentParser(prog="pglrep")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("featurize")
    f.add_argument("--protein", required=True)
    f.add_argument("--ligand", required=True)
    f.add_argument("--vocabulary", required=True)
    f.add_argument("--output", required=True)
    i = sub.add_parser("inspect")
    i.add_argument("--feature", required=True)
    args = parser.parse_args()
    if args.command == "featurize":
        result = PGLRep(PGLConfig(), PGLVocabulary.load(args.vocabulary)).from_complex(args.protein, args.ligand)
        np.save(args.output, result.vector)
        print(json.dumps({"schema": result.schema, "metadata": result.metadata}, indent=2))
    else:
        x = np.load(args.feature)
        print(json.dumps({"shape": list(x.shape), "dtype": str(x.dtype), "finite": bool(np.isfinite(x).all())}, indent=2))
