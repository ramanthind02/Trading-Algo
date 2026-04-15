from __future__ import annotations

from datetime import datetime
import logging
from typing import Optional

import pandas as pd

from utils.core import helpers
from utils.core.enums import Direction, DirectionInput, Ticker, TimeFrame, coerce_direction
from utils.core.models import Candle

logger = logging.getLogger(__name__)


def _normalize_tickers(tickers: Ticker | list[Ticker]) -> list[Ticker]:
    return [tickers] if isinstance(tickers, Ticker) else list(tickers)


def _project_signal_to_strategy(signal: pd.Series, strategy: Direction) -> pd.Series:
    if strategy is Direction.LONG:
        return signal.clip(lower=0.0)
    if strategy is Direction.SHORT:
        return signal.clip(upper=0.0)
    return signal


def _prepare_candle_rows(candles_df: pd.DataFrame) -> pd.DataFrame:
    rows = candles_df.copy()
    if "datetime" in rows.columns:
        rows["datetime"] = pd.to_datetime(rows["datetime"])
        sort_columns = ["datetime", "ticker"] if "ticker" in rows.columns else ["datetime"]
        rows = rows.sort_values(sort_columns)
    return rows


def _coerce_timeframe(raw: object) -> TimeFrame:
    if isinstance(raw, TimeFrame):
        return raw
    if isinstance(raw, str):
        return TimeFrame[raw]
    raise TypeError(f"timeframe must be TimeFrame or str, got {type(raw).__name__}")


def _coerce_ticker(raw: object) -> Ticker:
    if isinstance(raw, Ticker):
        return raw
    if isinstance(raw, str):
        return Ticker[raw]
    raise TypeError(f"ticker must be Ticker or str, got {type(raw).__name__}")


class BaseModel:
    """Thin node-backed adapter for native signed-signal bias nodes."""

    def __init__(
        self,
        feature_config: dict[str, object],
        tickers: Ticker | list[Ticker],
        binning_model: Optional[object] = None,
        use_cache: bool = True,
    ) -> None:
        if binning_model is not None:
            raise ValueError("BaseModel no longer accepts fitted or learned binning models.")

        raw_bias_node_spec = feature_config.get("bias_node_spec")
        if not isinstance(raw_bias_node_spec, dict):
            raise ValueError("feature_config must include a bias_node_spec dict")
        module_name = str(raw_bias_node_spec.get("module_name") or "").strip()
        if not module_name:
            raise ValueError("bias_node_spec.module_name must be non-empty")
        if module_name == "domain_discrete":
            raise ValueError("BaseModel no longer supports domain_discrete bias nodes.")

        raw_timeframes = raw_bias_node_spec.get("timeframes", [])
        if not isinstance(raw_timeframes, list) or not raw_timeframes:
            raise ValueError("bias_node_spec.timeframes must be a non-empty list")
        params = raw_bias_node_spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("bias_node_spec.params must be a dictionary")

        self.feature_config = dict(feature_config)
        self.bias_node_spec = {
            "module_name": module_name,
            "timeframes": [_coerce_timeframe(tf) for tf in raw_timeframes],
            "params": dict(params),
        }
        self.tickers = _normalize_tickers(tickers)
        self.ticker = self.tickers[0] if self.tickers else None
        self.use_cache = use_cache
        self.requires_fit = False
        self.is_fitted_ = True
        self.strategy = coerce_direction(
            feature_config.get("strategy", Direction.LONG_SHORT),
            field_name="strategy",
        )
        self.model_type = str(feature_config.get("model_type", "signed_signal"))
        self.members: list[tuple[str, object]] = []
        self._member_feature_columns: dict[str, str] = {}
        self.feature_column = feature_config.get("feature_column")
        self.bias_nodes: dict[tuple[Ticker, TimeFrame], object] = {}
        self._latest_feature_series = pd.Series(dtype=float)

    def add_member(
        self,
        name: str,
        member: object,
        feature_column: Optional[str] = None,
    ) -> None:
        self.members.append((name, member))
        if feature_column is not None:
            self._member_feature_columns[name] = feature_column

    def get_member_feature_columns(self) -> list[str]:
        columns: list[str] = []
        if self.feature_column:
            columns.append(str(self.feature_column))
        for column in self._member_feature_columns.values():
            if column not in columns:
                columns.append(column)
        return columns

    def _instantiate_bias_node(self, ticker: Ticker, tf: TimeFrame) -> object:
        base_spec = self.bias_node_spec
        return helpers.create_fresh_bias_node(
            str(base_spec["module_name"]),
            ticker,
            tf,
            dict(base_spec.get("params", {})),
        )

    def _build_bias_nodes(self) -> dict[tuple[Ticker, TimeFrame], object]:
        return {
            (ticker, timeframe): self._instantiate_bias_node(ticker, timeframe)
            for ticker in self.tickers
            for timeframe in self.bias_node_spec["timeframes"]
        }

    def _extract_feature_series(self, candles_df: pd.DataFrame) -> pd.Series:
        nodes = self._build_bias_nodes()
        self.bias_nodes = nodes

        rows = _prepare_candle_rows(candles_df)
        values: list[float] = []
        index: list[pd.Timestamp] = []
        column_name: str | None = None

        for row in rows.itertuples(index=False):
            candle = Candle.from_row_fast(row)
            ticker = _coerce_ticker(getattr(candle, "ticker"))
            tf = _coerce_timeframe(getattr(candle, "tf"))
            key = (ticker, tf)
            if key not in nodes:
                continue
            result = nodes[key].add_candle(candle)
            if column_name is None and hasattr(nodes[key], "get_column_names"):
                names = nodes[key].get_column_names()
                if names:
                    column_name = str(names[0])
            values.append(float(result[0]) if result else 0.0)
            index.append(pd.Timestamp(candle.datetime))

        if column_name is None:
            column_name = str(self.feature_column or "signed_signal")

        feature_series = pd.Series(values, index=pd.DatetimeIndex(index), name=column_name, dtype=float)
        if feature_series.index.duplicated().any():
            feature_series = feature_series.groupby(level=0).mean().clip(-1, 1)
        self.feature_column = column_name
        self._latest_feature_series = feature_series
        return feature_series

    def add_candle(self, candle: Candle, tf: TimeFrame, ticker: Optional[Ticker] = None) -> None:
        raise RuntimeError("BaseModel.add_candle() is no longer used in the signed-signal path")

    def get_feature(self) -> pd.Series:
        return self._latest_feature_series.copy()

    def fit(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BaseModel:
        del target_data, start_date, end_date
        self._extract_feature_series(candles_df)
        self.is_fitted_ = True
        return self

    def vectorized_fit(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BaseModel:
        return self.fit(candles_df, target_data, start_date, end_date)

    def stream_fit(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series,
    ) -> BaseModel:
        return self.fit(candles_df, target_data)

    def predict(
        self,
        candles_df: pd.DataFrame,
        strategy: DirectionInput | None = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.Series:
        del start_date, end_date
        signal = self._extract_feature_series(candles_df)
        requested_strategy = self.strategy if strategy is None else coerce_direction(strategy, field_name="strategy")
        return _project_signal_to_strategy(signal, requested_strategy)

    def stream_predict(
        self,
        candles_df: pd.DataFrame,
        strategy: DirectionInput | None = None,
    ) -> pd.Series:
        return self.predict(candles_df, strategy=strategy)

    def vectorized_predict(
        self,
        candles_df: pd.DataFrame,
        strategy: DirectionInput | None = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.Series:
        return self.predict(candles_df, strategy=strategy, start_date=start_date, end_date=end_date)

    def save_to_vault(
        self,
        ensemble_dir: Optional[str] = None,
        tickers: Optional[list[Ticker]] = None,
    ) -> str:
        from ensemble.vault_manager import add_feature_to_ensemble

        if self.feature_column is None:
            first_timeframe = self.bias_node_spec["timeframes"][0]
            self.feature_column = helpers.build_feature_column_name(
                module=str(self.bias_node_spec["module_name"]),
                feature="signal",
                tf=first_timeframe,
                params=dict(self.bias_node_spec.get("params", {})),
            )

        return add_feature_to_ensemble(
            feature_name=str(self.feature_column),
            bias_node_spec=dict(self.bias_node_spec),
            base_model=self,
            ensemble_dir=ensemble_dir,
            tickers=tickers or list(self.tickers),
        )

    def update_fitted_params_in_vault(
        self,
        ensemble_dir: str,
        model_id: str,
        train_start: str,
        train_end: str,
    ) -> None:
        del ensemble_dir, model_id, train_start, train_end
        raise ValueError("Signed-signal base models do not store fitted params in the vault.")
