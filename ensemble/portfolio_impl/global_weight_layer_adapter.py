"""Helpers for the synthetic `__GLOBAL__` WeightLayer adapter."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import Any, Dict, List, Optional

import pandas as pd

from lib.core.enums import TimeFrame

_GLOBAL_WEIGHT_LAYER_TICKER = "__GLOBAL__"
_UNKNOWN_GROUP = "__unknown__"


def build_group_equal_stream_weights(
    stream_ids: Collection[str],
    stream_id_to_group: Mapping[str, str],
) -> pd.Series:
    """Equal total weight per group (1/n_groups), split equally among streams in each group.

    Only groups that have at least one stream in ``stream_ids`` receive a share. When streams
    are removed from a group, that group's budget is redistributed only among the remaining
    streams in that group; if a group becomes empty it is dropped and remaining groups each get
    ``1 / n_groups`` with the new group count.
    """
    ids = sorted({str(s) for s in stream_ids if str(s)})
    if not ids:
        return pd.Series(dtype=float)

    by_group: dict[str, list[str]] = defaultdict(list)
    for sid in ids:
        g = stream_id_to_group.get(sid, _UNKNOWN_GROUP)
        by_group[str(g)].append(sid)

    active_groups = sorted(by_group.keys())
    n_groups = len(active_groups)
    w_per_group = 1.0 / n_groups if n_groups else 0.0
    weights: dict[str, float] = {}
    for g in active_groups:
        streams = by_group[g]
        n_s = len(streams)
        w_each = w_per_group / n_s if n_s else 0.0
        for sid in streams:
            weights[sid] = w_each
    return pd.Series(weights, dtype=float)


def build_global_model_name(
    timeframe: TimeFrame,
    ensemble_idx: int,
    model_name: str,
) -> str:
    """Stable global model namespace with explicit strategy + timeframe tags."""
    return f"{model_name}__{timeframe.name}::ensemble_{ensemble_idx}"


def parse_timeframe_from_global_model_name(model_name: str) -> str:
    """Extract timeframe tag from global names produced by ``build_global_model_name``."""
    left = str(model_name).split("::", 1)[0]
    if "__" in left:
        _, tf = left.rsplit("__", 1)
        return tf
    return left


def build_global_stream_id(ticker: str, timeframe: str, model_name: str) -> str:
    """Build a stable global stream id for adapter-encoded WeightLayer inputs."""
    return f"{ticker}::{timeframe}::{model_name}"


def decode_global_stream_id(stream_id: str) -> Dict[str, str]:
    """Decode stream id created by ``build_global_stream_id``."""
    parts = str(stream_id).split("::", 2)
    if len(parts) != 3:
        raise ValueError(
            "Invalid global stream_id; expected format "
            "'{ticker}::{timeframe}::{model_name}', got "
            f"'{stream_id}'"
        )
    ticker, timeframe, model_name = parts
    return {
        "ticker": ticker,
        "timeframe": timeframe,
        "original_model_name": model_name,
    }


def encode_forecast_vectors_for_global_weight_layer(
    forecast_vectors: List[pd.DataFrame],
) -> tuple[List[pd.DataFrame], Dict[str, Dict[str, str]]]:
    """Encode per-stream forecasts into one synthetic global WeightLayer ticker."""
    if not forecast_vectors:
        return [], {}

    combined = pd.concat(forecast_vectors, ignore_index=True)
    if combined.empty:
        return [], {}

    required_cols = {"ticker", "datetime", "model_name", "forecast", "signal", "timeframe"}
    if not required_cols.issubset(set(combined.columns)):
        return [], {}

    encoded = combined.copy()
    encoded["stream_id"] = encoded.apply(
        lambda row: build_global_stream_id(
            ticker=str(row["ticker"]),
            timeframe=str(row["timeframe"]),
            model_name=str(row["model_name"]),
        ),
        axis=1,
    )
    encoded["ticker"] = _GLOBAL_WEIGHT_LAYER_TICKER
    encoded["model_name"] = encoded["stream_id"]

    stream_ids = encoded["stream_id"].drop_duplicates(keep="first").tolist()
    decode_map = {
        stream_id: decode_global_stream_id(stream_id)
        for stream_id in stream_ids
    }
    encoded_df = encoded[["ticker", "datetime", "model_name", "forecast", "signal"]].copy()
    return [encoded_df], decode_map


def build_global_adapter_rollups(
    stream_decode_map: Dict[str, Dict[str, str]],
    global_weights: Optional[pd.Series],
) -> Dict[str, Any]:
    """Build ticker and timeframe rollups from synthetic global stream weights."""
    if global_weights is None or global_weights.empty:
        return {
            "synthetic_ticker": _GLOBAL_WEIGHT_LAYER_TICKER,
            "stream_decode_map": stream_decode_map,
            "stream_weights": {},
            "ticker_rollups": {},
            "timeframe_rollups": {},
        }

    stream_weights = {
        str(stream_id): float(weight)
        for stream_id, weight in global_weights.to_dict().items()
    }
    ticker_rollups: Dict[str, float] = {}
    timeframe_rollups: Dict[str, float] = {}
    for stream_id, weight in stream_weights.items():
        decoded = stream_decode_map.get(stream_id)
        if decoded is None:
            continue
        ticker = str(decoded["ticker"])
        timeframe = str(decoded["timeframe"])
        ticker_rollups[ticker] = ticker_rollups.get(ticker, 0.0) + float(weight)
        timeframe_rollups[timeframe] = timeframe_rollups.get(timeframe, 0.0) + float(weight)

    return {
        "synthetic_ticker": _GLOBAL_WEIGHT_LAYER_TICKER,
        "stream_decode_map": stream_decode_map,
        "stream_weights": stream_weights,
        "ticker_rollups": dict(sorted(ticker_rollups.items())),
        "timeframe_rollups": dict(sorted(timeframe_rollups.items())),
    }


def decode_global_weight_layer_output(
    encoded_vectors: List[pd.DataFrame],
    stream_decode_map: Dict[str, Dict[str, str]],
    global_weights: Optional[pd.Series],
    global_fdm: float = 1.0,
    stream_id_to_group: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Decode synthetic global outputs back to ticker-level forecast scores.

    If ``stream_id_to_group`` is set, weights are **equal per vault weight-hierarchy group**
    (each group gets ``1/n_groups`` among groups present in the frame), with equal split
    within each group. Otherwise ``global_weights`` is used, or flat equal weight across
    streams when ``global_weights`` is None.
    """
    if not encoded_vectors:
        return pd.DataFrame(columns=["ticker", "datetime", "forecast_score"])

    combined = pd.concat(encoded_vectors, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["ticker", "datetime", "forecast_score"])

    available_models = combined["model_name"].unique()

    if stream_id_to_group is not None:
        resolved_weights = build_group_equal_stream_weights(
            available_models,
            stream_id_to_group,
        )
    else:
        resolved_weights = global_weights
        if resolved_weights is None:
            n_models = len(available_models)
            equal_weight = 1.0 / n_models if n_models > 0 else 1.0
            resolved_weights = pd.Series(
                {model: equal_weight for model in available_models},
                dtype=float,
            )

    weighted = combined.copy()
    weighted["weight"] = weighted["model_name"].map(resolved_weights)
    missing_models = weighted[weighted["weight"].isna()]["model_name"].unique()
    if len(missing_models) > 0:
        n_known = len(resolved_weights)
        fallback_weight = (
            1.0 / (n_known + len(missing_models))
            if n_known > 0
            else 1.0 / len(missing_models)
        )
        weighted["weight"] = weighted["weight"].fillna(fallback_weight)

    weighted["weighted_forecast"] = weighted["forecast"] * weighted["weight"]
    decoded_lookup = weighted["model_name"].map(stream_decode_map)
    weighted["ticker"] = decoded_lookup.map(
        lambda value: str(value["ticker"]) if isinstance(value, dict) else ""
    )
    weighted = weighted[weighted["ticker"] != ""].copy()
    if weighted.empty:
        return pd.DataFrame(columns=["ticker", "datetime", "forecast_score"])

    grouped = (
        weighted.groupby(["ticker", "datetime"], as_index=False)["weighted_forecast"]
        .sum()
        .sort_values(["ticker", "datetime"])
    )
    grouped["forecast_score"] = (grouped["weighted_forecast"] * float(global_fdm)).clip(
        lower=-2.0,
        upper=2.0,
    )
    return grouped[["ticker", "datetime", "forecast_score"]].reset_index(drop=True)
