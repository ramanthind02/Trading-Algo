from datetime import datetime

import numpy as np
import pandas as pd

from utils.enums import Ticker, TimeFrame
from utils.permutation_test.candle_shuffle import IntradayGapConfig
from utils.permutation_test.permutation_engine import BarPermutationStrategy


def _make_price_df() -> pd.DataFrame:
    dt = pd.date_range("2025-01-06", periods=8, freq="B")
    open_prices = np.linspace(100.0, 107.0, num=len(dt))
    close_prices = open_prices + np.where(np.arange(len(dt)) % 2 == 0, 0.5, -0.3)
    high_prices = np.maximum(open_prices, close_prices) + 1.0
    low_prices = np.minimum(open_prices, close_prices) - 1.0

    return pd.DataFrame(
        {
            "datetime": dt,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
        }
    )


def test_bar_permutation_strategy_uses_correct_import_and_passes_shuffle_kwargs(monkeypatch) -> None:
    import utils.permutation_test.permute_bars as permute_bars_module

    price_df = _make_price_df()
    gap_cfg = IntradayGapConfig(maintenance_prev_hour=16, maintenance_next_hour=17)

    bar_data = {
        "price_df": price_df,
        "ticker": Ticker.ES,
        "start": datetime(2020, 1, 1),
        "end": datetime(2020, 12, 31),
        "base_tf": TimeFrame.D,
        "atr_feature": "atr_252_D_atr_pct_252",
        "feature_cols": ["feature_a"],
    }

    captured: dict[str, object] = {}

    class StubBarPermute:
        def __init__(
            self,
            df: pd.DataFrame,
            permute_start_idx: int,
            shuffle_mode: str,
            intraday_gap_config: IntradayGapConfig,
            random_seed: int,
        ) -> None:
            captured["df_rows"] = len(df)
            captured["permute_start_idx"] = permute_start_idx
            captured["shuffle_mode"] = shuffle_mode
            captured["intraday_gap_config"] = intraday_gap_config
            captured["random_seed"] = random_seed
            self._df = df

        def permute(self) -> pd.DataFrame:
            return self._df.copy()

    def fake_extract_features_from_bars(**kwargs):
        df = kwargs["df"]
        idx = pd.to_datetime(df["datetime"])
        features_df = pd.DataFrame({"feature_a": np.ones(len(df))}, index=idx)
        permuted_price_df = df.set_index("datetime")[["open", "high", "low", "close"]].copy()
        return features_df, permuted_price_df

    monkeypatch.setattr(permute_bars_module, "BarPermute", StubBarPermute)
    monkeypatch.setattr(permute_bars_module, "_extract_features_from_bars", fake_extract_features_from_bars)

    strategy = BarPermutationStrategy()
    result = strategy.permute(
        data=pd.DataFrame({"x": [1.0]}),
        random_seed=123,
        permute_start_idx=3,
        bar_data=bar_data,
        shuffle_mode="intraday",
        intraday_gap_config=gap_cfg,
    )

    assert len(result) == len(price_df)
    assert {"open", "high", "low", "close", "feature_a"}.issubset(result.columns)

    assert captured["permute_start_idx"] == 3
    assert captured["shuffle_mode"] == "intraday"
    assert captured["intraday_gap_config"] == gap_cfg
    assert captured["random_seed"] == 123


def test_bar_permutation_strategy_uses_walkforward_permuter_when_train_windows_set(monkeypatch) -> None:
    import utils.permutation_test.permute_bars as permute_bars_module

    price_df = _make_price_df()

    bar_data = {
        "price_df": price_df,
        "ticker": Ticker.ES,
        "start": datetime(2020, 1, 1),
        "end": datetime(2020, 12, 31),
        "base_tf": TimeFrame.D,
        "atr_feature": "atr_252_D_atr_pct_252",
        "feature_cols": ["feature_a"],
    }

    called = {"walkforward": False}

    class StubWalkForward:
        def __init__(self, df: pd.DataFrame, train_windows):
            self._df = df
            self._train_windows = train_windows

        def permute(self) -> pd.DataFrame:
            called["walkforward"] = True
            return self._df.copy()

    def fake_extract_features_from_bars(**kwargs):
        df = kwargs["df"]
        idx = pd.to_datetime(df["datetime"])
        features_df = pd.DataFrame({"feature_a": np.ones(len(df))}, index=idx)
        permuted_price_df = df.set_index("datetime")[["open", "high", "low", "close"]].copy()
        return features_df, permuted_price_df

    monkeypatch.setattr(permute_bars_module, "BarPermuteWalkForward", StubWalkForward)
    monkeypatch.setattr(permute_bars_module, "_extract_features_from_bars", fake_extract_features_from_bars)

    strategy = BarPermutationStrategy()
    _ = strategy.permute(
        data=pd.DataFrame({"x": [1.0]}),
        random_seed=77,
        bar_data=bar_data,
        train_windows=[(pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-10"))],
    )

    assert called["walkforward"] is True
