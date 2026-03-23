"""Regression tests for cached BaseModel prediction ticker normalization."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_selection.base_models.feature_base_model import BaseModel
from utils.core.enums import Ticker, TimeFrame


class _DummyBinningModel:
    model_type = "rule_based"

    def predict(self, feature_data: pd.Series, strategy=None):  # noqa: ANN001
        return pd.Series(1.0, index=feature_data.index)


class _DummyBiasNode:
    module_name = "buy_hold"
    params: dict[str, object] = {}

    def __init__(self, values: pd.Series) -> None:
        self._values = values

    def get_cached_values(self, start=None, end=None, require_cache=True):  # noqa: ANN001
        result = self._values
        if start is not None:
            result = result[result.index >= pd.Timestamp(start)]
        if end is not None:
            result = result[result.index <= pd.Timestamp(end)]
        return result


def test_vectorized_predict_matches_enum_model_tickers_to_string_candle_tickers() -> None:
    model = BaseModel(
        feature_config={
            "bias_node_spec": {
                "module_name": "buy_hold",
                "timeframes": [TimeFrame.D],
                "params": {},
            }
        },
        tickers=[Ticker.ES],
        binning_model=_DummyBinningModel(),
        use_cache=True,
    )

    cached_index = pd.date_range("2024-01-01", periods=3, freq="D")
    model.bias_nodes[(Ticker.ES, TimeFrame.D)] = _DummyBiasNode(
        pd.Series([1.0, 1.0, 1.0], index=cached_index)
    )

    candles_df = pd.DataFrame(
        {
            "datetime": cached_index,
            "ticker": ["ES", "ES", "ES"],
        }
    )

    result = model.vectorized_predict(
        candles_df,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 3),
    )

    assert len(result) == 3
    assert result.index.tolist() == list(cached_index)
