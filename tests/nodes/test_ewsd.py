from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.ewsd import EWSDNode
from utils.core.enums import Ticker, TimeFrame
from utils.compute.fast_volatility import compute_ewsd_annualized_from_closes
from utils.core.models import Candle


def _make_candle(close: float, day_index: int) -> Candle:
    dt = datetime(2020, 1, 1) + timedelta(days=day_index)
    return Candle(
        datetime=dt,
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=1000.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_ewsd_long_run_window_defaults_to_2520() -> None:
    node = EWSDNode(ticker=Ticker.ES, tf=TimeFrame.D)
    assert node.long_run_window == 2520
    assert node.returns_history.maxlen == 2520


def test_ewsd_outputs_valid_value_from_second_candle() -> None:
    node = EWSDNode(ticker=Ticker.ES, tf=TimeFrame.D)
    closes = [4000.0, 4010.0]

    outputs = [node._compute_candle(_make_candle(close, i)) for i, close in enumerate(closes)]
    first_daily = outputs[0][0] / 100.0
    second_daily = outputs[1][0] / 100.0
    second_annual = outputs[1][1] / 100.0

    assert first_daily > 0.0
    assert second_daily == pytest.approx(0.0025, abs=0.001)
    assert second_annual == pytest.approx(second_daily * 16.0)


def test_ewsd_updates_long_run_estimate_without_20_observation_guard() -> None:
    node = EWSDNode(ticker=Ticker.ES, tf=TimeFrame.D)
    closes = [100.0, 103.0, 99.0, 104.0]

    for i, close in enumerate(closes):
        node._compute_candle(_make_candle(close, i))

    assert node.sigma_long != pytest.approx(0.01)
    assert np.isfinite(node.sigma_long)
    assert node.sigma_long > 0.0


def test_ewsd_never_outputs_nan() -> None:
    np.random.seed(99)
    node = EWSDNode(ticker=Ticker.ES, tf=TimeFrame.D)
    price = 4000.0

    for i in range(100):
        price *= 1.0 + np.random.normal(0.0, 0.01)
        ewsd_daily_pct, ewsd_annual_pct = node._compute_candle(_make_candle(price, i))
        assert np.isfinite(ewsd_daily_pct)
        assert np.isfinite(ewsd_annual_pct)
        assert ewsd_daily_pct > 0.0
        assert ewsd_annual_pct == pytest.approx(ewsd_daily_pct * 16.0)


def test_ewsd_matches_fast_volatility_blending() -> None:
    np.random.seed(3)
    closes = 4000.0 * np.cumprod(1.0 + np.random.normal(0.0, 0.01, 500))
    expected_annual = compute_ewsd_annualized_from_closes(closes, long_run_window=2520)

    node = EWSDNode(ticker=Ticker.ES, tf=TimeFrame.D)
    out: list[float] | None = None
    for i, close in enumerate(closes):
        out = node._compute_candle(_make_candle(float(close), i))

    assert out is not None
    node_annual = out[1] / 100.0
    rel_err = abs(node_annual - expected_annual) / expected_annual
    assert rel_err < 0.20


def test_log_return_ewsd_coverage_is_close_to_log_return_when_cache_available() -> None:
    from dataclasses import replace
    from feature_research.config import FeatureType

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "ohlc_data"
    if not data_dir.exists():
        pytest.skip(f"Skipping coverage check, dataset missing at {data_dir}")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from feature_research.in_sample.config import load_config
    from feature_research.in_sample.data_loader import (
        expand_bias_specs,
        load_features_for_combo,
    )

    from feature_research.config import InSampleDefaultsCatalog, InSamplePhaseDefaultsConfig

    config = None
    try:
        config = load_config()
    except ValueError as exc:
        pytest.skip(f"Skipping coverage check, config unavailable: {exc}")
    assert config is not None

    active_defaults = config.in_sample_defaults.for_feature_type(config.feature_type)
    defaults_ewsd = replace(active_defaults, target_col="log_return_ewsd")
    defaults_log = replace(active_defaults, target_col="log_return")
    if config.feature_type == FeatureType.CONTINUOUS:
        catalog_ewsd = replace(config.in_sample_defaults, continuous=defaults_ewsd)
        catalog_log = replace(config.in_sample_defaults, continuous=defaults_log)
    else:
        catalog_ewsd = replace(config.in_sample_defaults, rule_based=defaults_ewsd)
        catalog_log = replace(config.in_sample_defaults, rule_based=defaults_log)

    config_ewsd = replace(config, in_sample_defaults=catalog_ewsd, populate_cache=False)
    config_log = replace(config, in_sample_defaults=catalog_log, populate_cache=False)

    single_spec = expand_bias_specs(config.bias_spec)[0]
    data_ewsd = None
    data_log = None
    try:
        data_ewsd = load_features_for_combo(single_spec, config_ewsd)
        data_log = load_features_for_combo(single_spec, config_log)
    except Exception as exc:
        pytest.skip(f"Skipping coverage check, feature cache unavailable: {exc}")
    if data_ewsd is None or data_log is None:
        pytest.skip("Skipping coverage check, could not load feature/target data")
    assert data_ewsd is not None
    assert data_log is not None

    _, target_ewsd, _ = data_ewsd
    _, target_log, _ = data_log

    n_ewsd = len(target_ewsd)
    n_log = len(target_log)
    if n_log == 0:
        pytest.skip("Skipping coverage check, log_return target length is zero")

    assert n_ewsd / n_log > 0.95
