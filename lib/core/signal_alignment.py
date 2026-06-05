"""Align feature/signal series to a target index for permutation and validation."""

from __future__ import annotations

import pandas as pd


def align_signal_to_target(signal: pd.Series, target: pd.Series) -> tuple[pd.Series, pd.Series]:
    aligned_signal = signal.reindex(target.index).dropna()
    aligned_target = target.reindex(aligned_signal.index).dropna()
    aligned_signal = aligned_signal.reindex(aligned_target.index)
    return aligned_signal, aligned_target
