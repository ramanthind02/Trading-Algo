"""Shared string labels for walkforward param grids and reporting."""

from __future__ import annotations


def canonical_param_label(params: dict[str, object]) -> str:
    return "|".join(f"{key}={params[key]}" for key in sorted(params))
