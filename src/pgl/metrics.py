"""Regression metrics used to report readout accuracy."""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 3 or y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 3 or y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return float("nan")
    from scipy.stats import spearmanr

    value = spearmanr(y_true, y_pred).statistic
    return float(value) if value == value else float("nan")


def all_metrics(y_true, y_pred) -> dict[str, float]:
    """Return n, RMSE, MAE, Pearson r and Spearman rho."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return {
        "n": int(y_true.size),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
    }


__all__ = ["all_metrics", "mae", "pearson", "rmse", "spearman"]
