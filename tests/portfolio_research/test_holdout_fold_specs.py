from __future__ import annotations

from datetime import datetime

import pandas as pd

from portfolio_research.config import (
    HoldoutFoldRole,
    PortfolioFitMode,
    PortfolioResearchConfig,
    ResearchWindow,
    build_holdout_fold_specs,
)
from utils.core.enums import Ticker, TimeFrame


def _minimal_config() -> PortfolioResearchConfig:
    return PortfolioResearchConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2026, 1, 1),
        use_cache=False,
        train_window=ResearchWindow(datetime(2000, 1, 1), datetime(2017, 12, 31)),
        validation_window=ResearchWindow(datetime(2018, 1, 1), datetime(2022, 12, 31)),
        test_window=ResearchWindow(datetime(2023, 1, 1), datetime(2025, 12, 31)),
        ensemble_dirs={"demo": "vault/D/demo"},
        portfolio_fit_mode=PortfolioFitMode.ROLLING_HOLDOUT,
    )


def test_build_holdout_fold_specs_two_folds() -> None:
    specs = build_holdout_fold_specs(_minimal_config())
    assert len(specs) == 2

    validation, holdout_test = specs
    assert validation.role == HoldoutFoldRole.VALIDATION
    assert validation.fit_start == datetime(2000, 1, 1)
    assert validation.fit_end == datetime(2017, 12, 31)
    assert validation.test_start == datetime(2018, 1, 1)
    assert validation.test_end == datetime(2022, 12, 31)

    train_span = pd.Timestamp(validation.fit_end) - pd.Timestamp(validation.fit_start)
    expected_roll_start = pd.Timestamp(holdout_test.fit_end) - train_span
    assert holdout_test.role == HoldoutFoldRole.HOLDOUT_TEST
    assert pd.Timestamp(holdout_test.fit_start) == expected_roll_start
    assert holdout_test.fit_end == datetime(2022, 12, 31)
    assert holdout_test.test_start == datetime(2023, 1, 1)
    assert holdout_test.test_end == datetime(2025, 12, 31)
