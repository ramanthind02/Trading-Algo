"""Legacy binning diagnostics removed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


LEGACY_BINNING_REMOVED_ERROR = (
    "Legacy binning diagnostics removed; use frozen signed-signal validation."
)


@dataclass(frozen=True)
class BinningSuccessCriteria:
    metric_threshold: float
    t_threshold: float
    min_region_width: int


@dataclass(frozen=True)
class RegionMetadata:
    start_bin: int
    end_bin: int
    bins: list[int]
    mean_sharpe: float
    mean_t_stat: float
    sample_count: int
    feature_range: tuple[float, float]


def _raise_legacy_binning_removed() -> None:
    raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)


def extract_region_metadata(*_args: object, **_kwargs: object) -> list[RegionMetadata]:
    _raise_legacy_binning_removed()


def validate_binning_success(*_args: object, **_kwargs: object) -> bool:
    _raise_legacy_binning_removed()


def detect_region_shape(*_args: object, **_kwargs: object) -> Literal["tail", "hump"]:
    _raise_legacy_binning_removed()


def calculate_coverage(*_args: object, **_kwargs: object) -> float:
    _raise_legacy_binning_removed()

