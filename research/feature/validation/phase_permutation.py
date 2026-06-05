"""Canonical signed-signal phase permutation runner."""

from research.feature.validation.research_permutation_runner import (
    SIGNED_SIGNAL_FEATURE_TYPE,
    _prepare_candles_for_shuffler,
    _two_unit_train_windows,
    load_research_data,
    run_candle_shuffle_null,
    run_permutation_for_phase,
)

__all__ = [
    "SIGNED_SIGNAL_FEATURE_TYPE",
    "_prepare_candles_for_shuffler",
    "_two_unit_train_windows",
    "load_research_data",
    "run_candle_shuffle_null",
    "run_permutation_for_phase",
]
