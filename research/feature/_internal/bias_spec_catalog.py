"""Bias spec catalog helpers kept away from phase entrypoints."""
from __future__ import annotations

from typing import Any


def first_bias_spec(
    bias_spec: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    """First dict in a catalog (or pass-through) for metadata / timeframe defaults."""

    if isinstance(bias_spec, list):
        if not bias_spec:
            raise ValueError("bias_spec list must be non-empty")
        return bias_spec[0]
    return bias_spec
