"""Train-only token vocabulary used by the canonical 1137-D readout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
import torch

from ._canonical_pipeline import Vocabulary


class PGLVocabulary:
    """Persisted K=8 codebook fitted exclusively on training tokens."""

    def __init__(self, token_mean: np.ndarray, token_std: np.ndarray,
                 centers: np.ndarray, tau: float, metadata: dict | None = None):
        self.token_mean = np.asarray(token_mean, dtype=np.float32)
        self.token_std = np.asarray(token_std, dtype=np.float32)
        self.centers = np.asarray(centers, dtype=np.float32)
        self.tau = float(tau)
        self.metadata = dict(metadata or {})
        if self.token_mean.shape != (57,) or self.token_std.shape != (57,):
            raise ValueError("canonical token statistics must have shape (57,)")
        if self.centers.shape != (8, 57):
            raise ValueError("canonical vocabulary centers must have shape (8, 57)")
        if not np.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be finite and strictly positive")

    @classmethod
    def fit(cls, training_tokens: np.ndarray, seed: int = 0) -> "PGLVocabulary":
        tokens = np.asarray(training_tokens, dtype=np.float32)
        if tokens.ndim != 2 or tokens.shape[1] != 57 or tokens.shape[0] < 8:
            raise ValueError("training_tokens must have shape (n>=8, 57)")
        mean = tokens.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = tokens.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        z = (tokens - mean[None, :]) / std[None, :]
        km = KMeans(n_clusters=8, init="k-means++", random_state=seed,
                    n_init=20, max_iter=300)
        km.fit(z)
        centers = km.cluster_centers_.astype(np.float32)
        d2 = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2, dtype=np.float64)
        tau = float(np.median(d2.min(axis=1)))
        meta = {"k": 8, "seed": seed, "n_init": 20, "max_iter": 300,
                "tau": tau, "fit_scope": "train_only",
                "train_token_count": int(tokens.shape[0]), "token_dim": 57,
                "implementation_revision": "canonical"}
        return cls(mean, std, centers, tau, meta)

    def to_canonical(self, device: torch.device, dtype: torch.dtype) -> Vocabulary:
        return Vocabulary(token_mean=torch.as_tensor(self.token_mean, device=device, dtype=dtype),
                          token_std=torch.as_tensor(self.token_std, device=device, dtype=dtype),
                          centers=torch.as_tensor(self.centers, device=device, dtype=dtype),
                          tau=self.tau)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, token_mean=self.token_mean, token_std=self.token_std,
                 centers=self.centers, tau=np.asarray(self.tau, dtype=np.float64),
                 k=np.asarray(8), seed=np.asarray(self.metadata.get("seed", 0)),
                 n_init=np.asarray(20), max_iter=np.asarray(300))
        path.with_suffix(".json").write_text(json.dumps({**self.metadata, "tau": self.tau}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PGLVocabulary":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            required = {"token_mean", "token_std", "centers", "tau"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"vocabulary is missing required fields: {missing}")
            meta_path = path.with_suffix(".json")
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            return cls(data["token_mean"], data["token_std"], data["centers"],
                       float(np.asarray(data["tau"]).reshape(())), meta)
