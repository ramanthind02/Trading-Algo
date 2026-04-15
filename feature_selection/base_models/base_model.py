"""Legacy binning symbols; instantiation raises (domain-discrete contract)."""

from __future__ import annotations

from abc import ABC


LEGACY_BINNING_REMOVED_ERROR = "Removed runtime; use the signed-signal contract."


class BinningModelBase(ABC):
    model_type = "legacy_binning"

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)


class ContinuousBinningModel:
    model_type = "removed"

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)


class RuleBasedModel:
    model_type = "removed"

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)
