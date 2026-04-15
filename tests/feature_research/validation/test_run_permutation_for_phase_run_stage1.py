"""Guardrails for feature_research walkforward permutation (vector shuffle only)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime

import pytest

from feature_research.validation.phase_permutation import run_permutation_for_phase


@dataclass(frozen=True)
class _MiniWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True)
class _MiniPerm:
    run_vector_shuffle: bool
    random_seed: int = 42
    alpha: float = 0.05
    n_jobs_reps: int = 1


@dataclass(frozen=True)
class _MiniCfg:
    permutation: _MiniPerm
    validation_window: _MiniWindow


def test_run_permutation_for_phase_raises_when_vector_shuffle_disabled() -> None:
    cfg = _MiniCfg(
        permutation=_MiniPerm(run_vector_shuffle=False),
        validation_window=_MiniWindow(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2021, 1, 1),
            test_start=datetime(2021, 1, 2),
            test_end=datetime(2022, 1, 1),
        ),
    )
    args = argparse.Namespace(nreps=None, seed=None, n_jobs=1, output_dir=None)
    with pytest.raises(ValueError, match="run_vector_shuffle"):
        run_permutation_for_phase(phase="validation", config=cfg, args=args)
