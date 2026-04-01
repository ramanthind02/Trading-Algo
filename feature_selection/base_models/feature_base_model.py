from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Optional

import pandas as pd

from feature_selection.domain_discrete import (
    MIGRATION_ERROR_MESSAGE,
    build_domain_discrete_bias_node_spec,
    load_domain_discrete_spec,
    raise_legacy_feature_artifact,
)
from filters import FilterSpec, create_filter
from nodes.filtered import FilteredBiasNode
from utils.core import helpers
from utils.core.enums import Direction, DirectionInput, Ticker, TimeFrame, coerce_direction
from utils.core.models import Candle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BiasNodeSpec:
    module_name: str
    timeframes: list[TimeFrame]
    params: dict[str, object]
    filters: tuple = ()


@dataclass
class DomainDiscreteCompatibilityState:
    strategy: Direction
    n_bins: int
    active_bins_by_strategy_: dict[str, list[int]]
    model_type: str = "domain_discrete"
    is_fitted_: bool = True
    bin_stats_: dict[int, dict[str, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.bin_stats_ is None:
            self.bin_stats_ = {}

    def get_fitted_params(self) -> dict[str, Any]:
        raise ValueError(MIGRATION_ERROR_MESSAGE)


def _normalize_tickers(tickers: Ticker | list[Ticker]) -> list[Ticker]:
    return [tickers] if isinstance(tickers, Ticker) else list(tickers)


def _coerce_ticker(raw: object) -> Ticker:
    if isinstance(raw, Ticker):
        return raw
    if isinstance(raw, str):
        return Ticker[raw]
    raise TypeError(f"ticker must be Ticker or str, got {type(raw).__name__}")


def _coerce_timeframe(raw: object) -> TimeFrame:
    if isinstance(raw, TimeFrame):
        return raw
    if isinstance(raw, str):
        return TimeFrame[raw]
    raise TypeError(f"timeframe must be TimeFrame or str, got {type(raw).__name__}")


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


def _fallback_feature_column_name(
    nodes: dict[tuple[Ticker, TimeFrame], object],
    tickers: list[Ticker],
    timeframes: list[TimeFrame],
) -> str:
    first_key = (tickers[0], timeframes[0])
    fallback_node = nodes[first_key]
    names = fallback_node.get_column_names() if hasattr(fallback_node, "get_column_names") else []
    return names[0] if names else "domain_discrete_signal"


def _build_default_feature_column(
    source_module_name: str,
    direction: Direction,
    spec_version: str,
    first_timeframe: TimeFrame,
) -> str:
    return helpers.build_feature_column_name(
        module="domain_discrete",
        feature="signal",
        tf=first_timeframe,
        params={
            "sourceModule": source_module_name,
            "direction": direction.value,
            "specVersion": spec_version,
        },
    )


def _feature_name_from_column(feature_column: str) -> str:
    parsed = helpers.parse_feature_column_name(feature_column)
    tf = parsed.get("tf")
    tf_name = tf.name if hasattr(tf, "name") else str(tf)
    if parsed.get("module") and parsed.get("feature") and tf_name:
        return f"{parsed.get('module')}_{parsed.get('feature')}_{tf_name}"
    return feature_column


class BaseModel:
    """Thin node-backed adapter for frozen domain-discrete signed signals."""

    def __init__(
        self,
        feature_config: dict[str, object],
        tickers: Ticker | list[Ticker],
        binning_model: Optional[object] = None,
        use_cache: bool = True,
    ) -> None:
        if binning_model is not None:
            raise_legacy_feature_artifact(
                "BaseModel no longer accepts fitted/learned binning_model instances."
            )

        raw_bias_node_spec = feature_config.get("bias_node_spec")
        if not isinstance(raw_bias_node_spec, dict):
            raise ValueError("feature_config must include a bias_node_spec dict")
        if raw_bias_node_spec.get("module_name") != "domain_discrete":
            raise_legacy_feature_artifact(
                "BaseModel requires bias_node_spec.module_name='domain_discrete'."
            )

        params = raw_bias_node_spec.get("params", {})
        self.domain_discrete_spec = load_domain_discrete_spec(params)
        self.feature_config = dict(feature_config)
        self.bias_node_spec = {
            "module_name": "domain_discrete",
            "timeframes": [
                _coerce_timeframe(tf)
                for tf in raw_bias_node_spec.get("timeframes", self.domain_discrete_spec.source_bias_node_spec.timeframes)
            ],
            "params": self.domain_discrete_spec.to_mapping(),
            "filters": tuple(raw_bias_node_spec.get("filters", ())),
        }

        self.tickers = _normalize_tickers(tickers)
        disallowed_tickers = [
            ticker.name
            for ticker in self.tickers
            if not self.domain_discrete_spec.ticker_scope.allows(ticker)
        ]
        if disallowed_tickers:
            raise ValueError(
                f"Tickers {disallowed_tickers} are outside ticker_scope "
                f"{[item.name for item in self.domain_discrete_spec.ticker_scope.tickers]}"
            )
        self.ticker = self.tickers[0] if self.tickers else None
        self.use_cache = use_cache
        self.requires_fit = False
        self.is_fitted_ = True
        self.strategy = self.domain_discrete_spec.direction
        self.model_type = "domain_discrete"
        self.binning_model = DomainDiscreteCompatibilityState(
            strategy=self.strategy,
            n_bins=self.domain_discrete_spec.n_bins,
            active_bins_by_strategy_={
                "long": list(self.domain_discrete_spec.long_bins),
                "short": list(self.domain_discrete_spec.short_bins),
                "long_short": sorted(
                    set(self.domain_discrete_spec.long_bins) | set(self.domain_discrete_spec.short_bins)
                ),
            },
        )
        self.members: list[tuple[str, object]] = []
        self._member_feature_columns: dict[str, str] = {}
        self.feature_column = feature_config.get("feature_column")
        self.bias_nodes: dict[tuple[Ticker, TimeFrame], object] = {}
        self._latest_feature_series = pd.Series(dtype=float)

    def add_member(
        self,
        name: str,
        binning_model: object,
        feature_column: Optional[str] = None,
    ) -> None:
        self.members.append((name, binning_model))
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
        params = dict(base_spec.get("params", {}))
        node = helpers.create_fresh_bias_node(
            base_spec["module_name"],
            ticker,
            tf,
            params,
        )
        filter_specs = tuple(base_spec.get("filters", ()))
        if not filter_specs:
            return node

        filters = tuple(
            create_filter(spec if isinstance(spec, FilterSpec) else FilterSpec(**spec))
            for spec in filter_specs
        )
        return FilteredBiasNode(node, filters)

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
                    column_name = names[0]
            values.append(float(result[0]) if result else 0.0)
            index.append(pd.Timestamp(candle.datetime))

        if column_name is None:
            column_name = _fallback_feature_column_name(
                nodes,
                self.tickers,
                self.bias_node_spec["timeframes"],
            )

        feature_series = pd.Series(values, index=pd.DatetimeIndex(index), name=column_name, dtype=float)
        if feature_series.index.duplicated().any():
            feature_series = feature_series.groupby(level=0).mean().round().clip(-1, 1)
        self.feature_column = column_name
        self._latest_feature_series = feature_series
        return feature_series

    def add_candle(self, candle: Candle, tf: TimeFrame, ticker: Optional[Ticker] = None) -> None:
        raise RuntimeError("BaseModel.add_candle() is no longer used in the domain-discrete path")

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
            self.feature_column = _build_default_feature_column(
                source_module_name=self.domain_discrete_spec.source_bias_node_spec.module_name,
                direction=self.domain_discrete_spec.direction,
                spec_version=self.domain_discrete_spec.spec_version,
                first_timeframe=self.bias_node_spec["timeframes"][0],
            )
        feature_name = _feature_name_from_column(str(self.feature_column))

        return add_feature_to_ensemble(
            feature_name=feature_name,
            bias_node_spec=build_domain_discrete_bias_node_spec(self.domain_discrete_spec),
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
        raise ValueError(MIGRATION_ERROR_MESSAGE)
