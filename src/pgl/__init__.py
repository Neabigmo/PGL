"""PGL: persistent Gauss-linking contact tokens for protein-ligand affinity."""

__version__ = "1.0.0"

from .context import (
    CONTEXT_DIMS,
    FEATURE_DIM_WITH_CONTEXT,
    ContextEmbedding,
    build_feature_matrix,
)
from .features import FEATURE_DIM, ContactTokens, contact_features, feature_names
from .operator import Complex, build_windows, parse_complex
from .readout import XGB_PRESET, XGBReadout
from .release import FEATURE_BLOCKS, MODEL_ID, RELEASE, RELEASE_VERSION, load_release

__all__ = [
    "CONTEXT_DIMS",
    "Complex",
    "ContactTokens",
    "ContextEmbedding",
    "FEATURE_BLOCKS",
    "FEATURE_DIM",
    "FEATURE_DIM_WITH_CONTEXT",
    "MODEL_ID",
    "RELEASE",
    "RELEASE_VERSION",
    "XGB_PRESET",
    "XGBReadout",
    "build_feature_matrix",
    "build_windows",
    "contact_features",
    "feature_names",
    "load_release",
    "parse_complex",
    "__version__",
]
