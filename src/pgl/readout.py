"""XGBoost readout trained on contact tokens and frozen context.

The readout uses one predeclared hyper-parameter set. The values were fixed
before the benchmark runs and are not tuned per dataset or per split; every
dataset is fitted with the same settings and three fixed seeds.

A readout can be bound to a feature schema (see :mod:`pgl.schema`). The schema
hash is written next to the model and checked again when the model is loaded
with the same schema, so a checkpoint cannot be silently applied to a different
feature layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import all_metrics
from .schema import schema_sha256

XGB_PRESET = {
    "n_estimators": 708,
    "max_depth": 6,
    "learning_rate": 0.052911547044925414,
    "subsample": 0.7406949320228273,
    "colsample_bytree": 0.9008575050404631,
    "min_child_weight": 7.087190990869927,
    "reg_lambda": 7.479441772354108,
    "reg_alpha": 0.28096217359611636,
    "objective": "reg:squarederror",
    "tree_method": "hist",
}

DEFAULT_SEED = 20261022


class XGBReadout:
    """Gradient-boosted regression readout with a fixed configuration.

    Parameters
    ----------
    seed:
        Learner seed. The paper reports three seeds: 20261022, 20261023 and
        20261024.
    n_threads:
        Number of boosting threads.
    hyperparameters:
        Optional overrides of :data:`XGB_PRESET`. Only intended for ablation
        runs; the released benchmark numbers use the preset unchanged.
    schema:
        Optional ordered feature schema. When given, the schema hash is stored
        with the checkpoint and the feature width is checked against the schema.
    """

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        n_threads: int = 8,
        hyperparameters: dict | None = None,
        schema: dict | None = None,
    ) -> None:
        self.seed = int(seed)
        self.n_threads = int(n_threads)
        self.hyperparameters = dict(XGB_PRESET)
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        self.model = None
        self.feature_dim: int | None = None
        self.n_train: int | None = None
        self.training_metrics: dict | None = None
        self.schema_total: int | None = int(schema["total"]) if schema is not None else None
        self.schema_sha256: str | None = schema_sha256(schema) if schema is not None else None

    def fit(self, features, labels) -> XGBReadout:
        """Fit the readout on a feature matrix and pK labels."""
        import xgboost as xgb

        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float64).reshape(-1)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got {features.shape}")
        if self.schema_total is not None and features.shape[1] != self.schema_total:
            raise ValueError(
                f"feature schema has {self.schema_total} columns, "
                f"features have {features.shape[1]}"
            )
        if features.shape[0] != labels.size:
            raise ValueError(f"{features.shape[0]} feature rows for {labels.size} labels")
        if not np.all(np.isfinite(labels)):
            raise ValueError("labels must be finite")
        parameters = dict(self.hyperparameters)
        parameters.update(random_state=self.seed, n_jobs=self.n_threads)
        self.model = xgb.XGBRegressor(**parameters)
        self.model.fit(features, labels)
        self.feature_dim = int(features.shape[1])
        self.n_train = int(features.shape[0])
        self.training_metrics = all_metrics(labels, self.model.predict(features))
        return self

    def predict(self, features) -> np.ndarray:
        """Predict pK for a feature matrix."""
        if self.model is None:
            raise RuntimeError("readout is not fitted")
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got {features.shape}")
        if self.feature_dim is not None and features.shape[1] != self.feature_dim:
            raise ValueError(
                f"readout expects {self.feature_dim} features, got {features.shape[1]}"
            )
        return np.asarray(self.model.predict(features), dtype=np.float64).reshape(-1)

    def config(self) -> dict:
        """Serialisable description of the readout."""
        return {
            "model": "xgboost",
            "seed": self.seed,
            "n_threads": self.n_threads,
            "feature_dim": self.feature_dim,
            "feature_schema_sha256": self.schema_sha256,
            "feature_schema_total": self.schema_total,
            "n_train": self.n_train,
            "hyperparameters": self.hyperparameters,
            "training_metrics": self.training_metrics,
        }

    def save(self, directory: str | Path) -> Path:
        """Write ``readout.json`` and ``config.json`` into a directory."""
        if self.model is None:
            raise RuntimeError("readout is not fitted")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(directory / "readout.json"))
        (directory / "config.json").write_text(
            json.dumps(self.config(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path, schema: dict | None = None) -> XGBReadout:
        """Load a readout written by :meth:`save`.

        When ``schema`` is given, the schema hash recorded with the checkpoint
        must match it (checkpoints written without a schema are checked on the
        feature width only).
        """
        import xgboost as xgb

        directory = Path(directory)
        config_path = directory / "config.json"
        model_path = directory / "readout.json"
        if not config_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"{directory} does not contain readout.json and config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        readout = cls(
            seed=int(config.get("seed", DEFAULT_SEED)),
            n_threads=int(config.get("n_threads", 8)),
            hyperparameters=config.get("hyperparameters"),
        )
        readout.model = xgb.XGBRegressor()
        readout.model.load_model(str(model_path))
        readout.feature_dim = config.get("feature_dim")
        readout.n_train = config.get("n_train")
        readout.training_metrics = config.get("training_metrics")
        readout.schema_sha256 = config.get("feature_schema_sha256")
        readout.schema_total = config.get("feature_schema_total")
        if schema is not None:
            expected = schema_sha256(schema)
            if readout.feature_dim is not None and int(schema["total"]) != int(readout.feature_dim):
                raise ValueError(
                    f"checkpoint expects {readout.feature_dim} features, "
                    f"schema has {schema['total']}"
                )
            if readout.schema_sha256 is not None and readout.schema_sha256 != expected:
                raise ValueError("checkpoint feature schema does not match the supplied schema")
            readout.schema_sha256 = expected
            readout.schema_total = int(schema["total"])
        return readout


__all__ = ["DEFAULT_SEED", "XGBReadout", "XGB_PRESET"]
