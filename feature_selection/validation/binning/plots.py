"""Legacy binning plots removed."""

from __future__ import annotations

from matplotlib.figure import Figure

LEGACY_BINNING_REMOVED_ERROR = (
    "Legacy binning plots removed; use frozen signed-signal validation."
)


def _raise_legacy_binning_removed() -> None:
    raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)


def plot_bin_heatmap(*_args: object, **_kwargs: object) -> Figure:
    _raise_legacy_binning_removed()


def plot_region_boundaries(*_args: object, **_kwargs: object) -> Figure:
    _raise_legacy_binning_removed()


def plot_position_multiplier_curve(*_args: object, **_kwargs: object) -> Figure:
    _raise_legacy_binning_removed()


def create_diagnostic_panel(*_args: object, **_kwargs: object) -> Figure:
    _raise_legacy_binning_removed()

