"""Tests for portfolio_research.weight_layer_cv."""

from __future__ import annotations

import pandas as pd

from research.portfolio.weight_layer_cv import (
    CvMetrics,
    CvRunRow,
    build_cv_arms,
    build_expanding_cv_folds,
    summarize_cv_rows,
)


def test_build_cv_arms_includes_sr_l2_inv() -> None:
    names = {arm.name for arm in build_cv_arms()}
    assert "SR_L2_inv" in names
    assert "SR_L1_inv" in names


def test_expanding_folds_cover_is_span() -> None:
    folds = build_expanding_cv_folds(
        is_start=pd.Timestamp("2000-01-01"),
        is_end=pd.Timestamp("2022-12-31"),
    )
    assert len(folds) == 4
    assert folds[-1].test_end == pd.Timestamp("2022-12-31")


def test_summarize_cv_rows_ranks_by_score() -> None:
    rows = [
        CvRunRow("low", "production_validation", CvMetrics(0.4, 0.3, 0.01, -0.05, 0.15, 100)),
        CvRunRow("high", "production_validation", CvMetrics(0.8, 0.7, 0.03, -0.04, 0.20, 100)),
    ]
    summary = summarize_cv_rows(rows)
    assert str(summary.iloc[0]["arm"]) == "high"
