"""Index/model compatibility checks without guessing missing provenance."""

from __future__ import annotations


def validate_embedding_model(metadata: dict | None, expected: str) -> bool:
    """Reject recorded mismatches; return False for legacy indexes without a model ID."""
    recorded = (metadata or {}).get("embedding_model")
    if recorded and recorded != expected:
        raise ValueError(
            "Index embedding model differs; rebuild in a separate directory"
        )
    return bool(recorded)
