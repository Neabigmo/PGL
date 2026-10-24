"""Ordered feature schema of the 1,279-column model.

The schema lists every feature name in column order: the 287 contact tokens,
then the 512 Uni-Mol context columns, then the 480 SaProt context columns. Its
SHA-256 pins the block order, the names and the missing-value policy, so a
checkpoint can be checked against the feature layout it was trained on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .context import CONTEXT_DIMS
from .features import FEATURE_DIM

SCHEMA_VERSION = "pgl-1.0.0"
BLOCKS = (
    ("contact_tokens", FEATURE_DIM),
    ("ctx_unimol", CONTEXT_DIMS["unimol"]),
    ("ctx_saprot", CONTEXT_DIMS["saprot"]),
)
HASH_DEFINITION = (
    "sha256 of the schema object without its 'sha256' field, serialised as "
    "canonical JSON (sort_keys=True, separators=(',', ':'))"
)


def ordered_feature_names() -> list[str]:
    """Feature names in column order, length 1,279."""
    from .features import feature_names

    names = list(feature_names())
    for tag in ("unimol", "saprot"):
        names.extend(f"ctx_{tag}_{i}" for i in range(CONTEXT_DIMS[tag]))
    return names


def schema_payload() -> dict:
    """Schema body without the self hash."""
    names = ordered_feature_names()
    blocks = []
    offset = 0
    for name, dim in BLOCKS:
        blocks.append(
            {"name": name, "dim": dim, "offset": offset, "names": names[offset : offset + dim]}
        )
        offset += dim
    return {
        "version": SCHEMA_VERSION,
        "total": offset,
        "dtype": "float32",
        "missing_value_policy": "NaN for absent context, no imputation, no presence columns",
        "blocks": blocks,
    }


def schema_sha256(schema: dict | None = None) -> str:
    """SHA-256 of a schema, defined by :data:`HASH_DEFINITION`."""
    body = schema_payload() if schema is None else schema
    body = {key: value for key, value in body.items() if key not in ("sha256", "hash_definition")}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_schema() -> dict:
    """Return the schema with its hash fields."""
    schema = schema_payload()
    schema["hash_definition"] = HASH_DEFINITION
    schema["sha256"] = schema_sha256(schema)
    return schema


def validate_schema(schema: dict) -> None:
    """Raise if a schema is inconsistent with the code."""
    blocks = schema.get("blocks", [])
    if [block.get("name") for block in blocks] != [name for name, _ in BLOCKS]:
        raise ValueError("schema block order does not match the released feature layout")
    if [int(block.get("dim", 0)) for block in blocks] != [dim for _, dim in BLOCKS]:
        raise ValueError("schema block dimensions do not match the released feature layout")
    if int(schema.get("total", 0)) != sum(dim for _, dim in BLOCKS):
        raise ValueError("schema total does not match the sum of the block dimensions")
    names = [name for block in blocks for name in block.get("names", [])]
    expected = ordered_feature_names()
    if names != expected:
        raise ValueError("schema feature names do not match the released feature order")
    if "sha256" in schema and schema_sha256(schema) != schema["sha256"]:
        raise ValueError("schema hash does not match its content")


def load_schema(path: str | Path | None = None) -> dict:
    """Load and validate the released schema.

    Without a path the copy in the source checkout is used; an installed wheel
    falls back to the schema generated from the code, which is identical.
    """
    if path is None:
        from .release import release_file

        try:
            path = release_file("release/schema.json")
        except FileNotFoundError:
            schema = build_schema()
            validate_schema(schema)
            return schema
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_schema(data)
    return data


__all__ = [
    "BLOCKS",
    "HASH_DEFINITION",
    "SCHEMA_VERSION",
    "build_schema",
    "load_schema",
    "ordered_feature_names",
    "schema_payload",
    "schema_sha256",
    "validate_schema",
]
