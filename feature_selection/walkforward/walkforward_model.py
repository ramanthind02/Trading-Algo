"""Removed walk-forward shim."""

from __future__ import annotations

from typing import Any

from feature_selection.base_models.base_model import LEGACY_BINNING_REMOVED_ERROR


class WalkForwardSplitter:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)


class WalkForwardModel:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)


def generate_rolling_windows(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)


def apply_function_to_walkforward(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)


def apply_function_to_rolling_windows(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)
