"""Public wrapper around the extracted canonical implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from .config import PGLConfig
from .vocabulary import PGLVocabulary
from . import _canonical_parser as canonical


@dataclass
class PGLResult:
    vector: np.ndarray
    schema: dict
    metadata: dict
    edge_tokens: np.ndarray | None = None
    spectra: list[np.ndarray] | None = None
    gauss_matrix: list[np.ndarray] | None = None


class PGLRep:
    """Canonical PGL-Rep feature extractor with a public Python API."""

    def __init__(self, config: PGLConfig | None = None,
                 vocabulary: PGLVocabulary | None = None, device: str = "cpu"):
        self.config = config or PGLConfig()
        self.vocabulary = vocabulary
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

    def from_complex(self, protein_path: str | Path, ligand_path: str | Path) -> PGLResult:
        if self.vocabulary is None:
            raise ValueError("a train-only PGLVocabulary is required for the 1137-D readout")
        protein_path, ligand_path = Path(protein_path), Path(ligand_path)
        with tempfile.TemporaryDirectory(prefix="pglrep-") as tmp:
            root = Path(tmp)
            shutil.copy2(protein_path, root / "protein.pdb")
            shutil.copy2(ligand_path, root / "ligand.sdf")
            vocab = self.vocabulary.to_canonical(self.device, self.config.dtype)
            raw = canonical.run_once(root, self.device, self.config, vocab,
                                     return_features=True)
        vector = np.asarray(raw["rep1137"], dtype=np.float32)
        schema = {"global": 84, "abundance": 15, "chemical_response": 360,
                  "token_statistics": 114, "vlad": 456, "context": 108,
                  "total": 1137}
        if vector.shape != (1137,):
            raise RuntimeError(f"canonical readout returned {vector.shape}, expected (1137,)")
        return PGLResult(vector=vector, schema=schema, metadata={"device": str(self.device),
                             "config": {"gauss_eps": self.config.gauss_eps,
                                         "gauss_lambda": self.config.gauss_lambda,
                                         "spectral_mode": self.config.spectral_mode,
                                         "spectral_dtype": str(self.config.spectral_dtype),
                                         "vocabulary_fit_scope": "train_only"},
                             **raw["meta"]},
                         edge_tokens=np.asarray(raw["tok57"], dtype=np.float32),
                         spectra=[np.asarray(x, dtype=np.float32) for x in raw["diagnostic_eigenvalues"]],
                         gauss_matrix=[np.asarray(x, dtype=np.float32) for x in raw["diagnostic_gauss"]])
