import json
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENSEMBLE_DIR = _REPO_ROOT / "ensemble"
_ENSEMBLE_PKG = types.ModuleType("ensemble")
_ENSEMBLE_PKG.__path__ = [str(_ENSEMBLE_DIR)]
sys.modules.setdefault("ensemble", _ENSEMBLE_PKG)

_WEIGHT_LAYER_SPEC = importlib.util.spec_from_file_location(
    "ensemble.weight_layer", _ENSEMBLE_DIR / "weight_layer.py"
)
if _WEIGHT_LAYER_SPEC is None or _WEIGHT_LAYER_SPEC.loader is None:
    raise RuntimeError("Unable to load ensemble.weight_layer")
_WEIGHT_LAYER_MODULE = importlib.util.module_from_spec(_WEIGHT_LAYER_SPEC)
sys.modules["ensemble.weight_layer"] = _WEIGHT_LAYER_MODULE
_WEIGHT_LAYER_SPEC.loader.exec_module(_WEIGHT_LAYER_MODULE)

_PORTFOLIO_PATH = _ENSEMBLE_DIR / "portfolio.py"
_PORTFOLIO_SPEC = importlib.util.spec_from_file_location("ensemble.portfolio", _PORTFOLIO_PATH)
if _PORTFOLIO_SPEC is None or _PORTFOLIO_SPEC.loader is None:
    raise RuntimeError(f"Unable to load Portfolio module from {_PORTFOLIO_PATH}")
_PORTFOLIO_MODULE = importlib.util.module_from_spec(_PORTFOLIO_SPEC)
sys.modules["ensemble.portfolio"] = _PORTFOLIO_MODULE
_PORTFOLIO_SPEC.loader.exec_module(_PORTFOLIO_MODULE)
Portfolio = _PORTFOLIO_MODULE.Portfolio


def _write_json(tmp_path: Path, filename: str, payload: dict) -> str:
    file_path = tmp_path / filename
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    return str(file_path)


def test_sector_config_overrides_instrument_weights(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path,
        "sector.json",
        {
            "weight": 1.0,
            "children": [
                {"weight": 0.8, "tickers": ["ES"]},
                {"weight": 0.2, "tickers": ["NQ"]},
            ],
        },
    )

    portfolio = Portfolio(
        instrument_weights={"ES": 0.5, "NQ": 0.5},
        sector_allocation_config_path=config_path,
    )

    assert portfolio.instrument_weights == pytest.approx({"ES": 0.8, "NQ": 0.2})


def test_no_sector_config_uses_instrument_weights() -> None:
    portfolio = Portfolio(instrument_weights={"ES": 0.7, "NQ": 0.3})
    assert portfolio.instrument_weights == {"ES": 0.7, "NQ": 0.3}


def test_no_sector_or_instrument_weights_defaults_equal() -> None:
    portfolio = Portfolio()
    weighted = portfolio._apply_instrument_weights(
        pd.DataFrame({"ticker": ["ES", "NQ", "YM"], "forecast_score": [1.0, 1.0, 1.0]})
    )
    assert weighted["instrument_weight"].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_resolve_nested_sector_tree_equal_split_matches_spec_example(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path,
        "nested.json",
        {
            "weight": 1.0,
            "children": [
                {
                    "weight": 0.5,
                    "children": [
                        {"weight": 0.6, "tickers": ["ES", "NQ", "YM", "RTY"]},
                        {"weight": 0.3, "tickers": ["DAX"]},
                        {"weight": 0.1, "tickers": ["N225"]},
                    ],
                },
                {"weight": 0.3, "tickers": ["GC", "SI"]},
                {"weight": 0.2, "tickers": ["CL"]},
            ],
        },
    )

    portfolio = Portfolio(sector_allocation_config_path=config_path)
    expected = {
        "ES": 0.075,
        "NQ": 0.075,
        "YM": 0.075,
        "RTY": 0.075,
        "DAX": 0.15,
        "N225": 0.05,
        "GC": 0.15,
        "SI": 0.15,
        "CL": 0.20,
    }
    assert portfolio.instrument_weights == pytest.approx(expected)


def test_resolve_leaf_ticker_weights_distribution(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path,
        "leaf_weights.json",
        {
            "weight": 1.0,
            "children": [
                {
                    "weight": 1.0,
                    "tickers": ["ES", "NQ", "YM"],
                    "ticker_weights": {"ES": 0.5, "NQ": 0.3, "YM": 0.2},
                }
            ],
        },
    )

    portfolio = Portfolio(sector_allocation_config_path=config_path)
    assert portfolio.instrument_weights == pytest.approx({"ES": 0.5, "NQ": 0.3, "YM": 0.2})


def test_final_weights_sum_to_one_after_resolution(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path,
        "sum_one.json",
        {
            "weight": 1.0,
            "children": [
                {"weight": 3.0, "tickers": ["ES", "NQ"]},
                {"weight": 7.0, "tickers": ["YM"]},
            ],
        },
    )
    portfolio = Portfolio(sector_allocation_config_path=config_path)
    assert np.isclose(sum(portfolio.instrument_weights.values()), 1.0)


def test_invalid_json_raises_value_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        Portfolio(sector_allocation_config_path=str(broken))


def test_missing_config_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        Portfolio(sector_allocation_config_path="missing_file.json")


@pytest.mark.parametrize(
    "payload, error_pattern",
    [
        (
            {"weight": 1.0, "children": [{"weight": 1.0, "tickers": ["ES"], "children": []}]},
            "exactly one",
        ),
        ({"weight": 1.0}, "exactly one"),
        ({"weight": 0.0, "tickers": ["ES"]}, "must be > 0"),
        ({"weight": 1.0, "tickers": []}, "at least one ticker"),
        (
            {
                "weight": 1.0,
                "children": [
                    {"weight": 0.5, "tickers": ["ES"]},
                    {"weight": 0.5, "tickers": ["ES"]},
                ],
            },
            "Duplicate ticker",
        ),
        (
            {
                "weight": 1.0,
                "tickers": ["ES", "NQ"],
                "ticker_weights": {"ES": 1.0},
            },
            "must match",
        ),
        (
            {
                "weight": 1.0,
                "tickers": ["ES"],
                "ticker_weights": {"ES": 0.0},
            },
            "ticker_weights",
        ),
    ],
)
def test_validation_errors_raise_value_error(
    tmp_path: Path, payload: dict, error_pattern: str
) -> None:
    config_path = _write_json(tmp_path, "invalid_sector.json", payload)
    with pytest.raises(ValueError, match=error_pattern):
        Portfolio(sector_allocation_config_path=config_path)


def test_missing_ticker_fallback_is_consistent_across_paths() -> None:
    portfolio = Portfolio(instrument_weights={"ES": 0.7}, max_position_pct=10.0)

    predict_df = pd.DataFrame({"ticker": ["ES", "NQ"], "forecast_score": [1.0, 1.0]})
    weighted = portfolio._apply_instrument_weights(predict_df)
    by_predict_path = dict(zip(weighted["ticker"], weighted["instrument_weight"]))
    assert by_predict_path == pytest.approx({"ES": 0.7, "NQ": 0.3})

    risk_df = pd.DataFrame(
        {
            "ticker": ["ES", "NQ"],
            "datetime": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
            "forecast_score": [1.0, 1.0],
        }
    )
    candles_df = pd.DataFrame(
        {
            "ticker": ["ES", "NQ"],
            "datetime": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
        }
    )
    risk_result = portfolio._apply_risk_management_to_forecasts(risk_df, volatility={}, candles_df=candles_df)
    by_risk_path = dict(zip(risk_result["ticker"], risk_result["position_fraction"]))
    assert by_risk_path == pytest.approx({"ES": 0.7, "NQ": 0.3})
