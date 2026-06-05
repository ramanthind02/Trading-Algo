"""Pure transforms: quantile bins, per-bin metrics, rolling threshold stability."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from research.feature.in_sample.metric_helpers import compute_param_sensitivity_metric
from lib.core.enums import Direction, TimeFrame


def serialize_param_value_for_parquet(value: object) -> object:
    """Scalar-friendly values for Parquet/BI; JSON-encode other structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return json.dumps(value, sort_keys=True, default=str)


def add_wide_param_columns(frame: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """Duplicate each param as ``param_<key>`` on every row (constant within a combo)."""
    out = frame.copy()
    for key in sorted(params.keys()):
        out[f"param_{key}"] = serialize_param_value_for_parquet(params[key])
    return out


def dedupe_combo_param_pairs(
    pairs: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    """Last-wins by ``param_combo_label``, sorted by label for deterministic exports."""
    merged: dict[str, dict[str, Any]] = {}
    for label, pmap in pairs:
        merged[label] = dict(pmap)
    return sorted(merged.items(), key=lambda x: x[0])


def flatten_params_for_combo_long_table(params: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested composite specs to scalar keys for visualization ``param_combo_long``.

    Matches the flat ``self.params`` layout from :class:`~nodes.composite.filter_gate.FilterGateNode`
    and     :class:`~nodes.composite.filter_and_signal.FilterAndSignalNode` (``f_*`` / ``s_*``) and
    :class:`~nodes.composite.dual_signal.DualSignalNode` (``a_*`` / ``b_*``), so slicers join the
    same way as for single-module bias nodes (no JSON blobs for inner dicts).
    """
    if not params:
        return {}
    p = dict(params)
    if (
        "filter_module" in p
        and "signal_module" in p
        and isinstance(p.get("filter_params"), dict)
        and isinstance(p.get("signal_params"), dict)
    ):
        fp = dict(p["filter_params"])
        sp = dict(p["signal_params"])
        out: dict[str, Any] = {
            "filter_module": p["filter_module"],
            "signal_module": p["signal_module"],
        }
        out.update({f"f_{k}": v for k, v in fp.items()})
        out.update({f"s_{k}": v for k, v in sp.items()})
        return out
    if (
        "moduleA" in p
        and "moduleB" in p
        and isinstance(p.get("paramsA"), dict)
        and isinstance(p.get("paramsB"), dict)
    ):
        pa = dict(p["paramsA"])
        pb = dict(p["paramsB"])
        out2: dict[str, Any] = {
            "moduleA": p["moduleA"],
            "moduleB": p["moduleB"],
        }
        out2.update({f"a_{k}": v for k, v in pa.items()})
        out2.update({f"b_{k}": v for k, v in pb.items()})
        return out2
    return dict(p)


def inflate_params_from_combo_long_table(params: Mapping[str, Any]) -> dict[str, Any]:
    """Restore nested composite specs from flat keys produced by ``flatten_params_for_combo_long_table``."""
    if not params:
        return {}
    p = dict(params)
    if isinstance(p.get("signal_params"), dict) or isinstance(p.get("filter_params"), dict):
        return p
    if "filter_module" in p and "signal_module" in p:
        filter_params = {key[2:]: value for key, value in p.items() if key.startswith("f_")}
        signal_params = {key[2:]: value for key, value in p.items() if key.startswith("s_")}
        if filter_params or signal_params:
            passthrough = {
                key: value
                for key, value in p.items()
                if not (key.startswith("f_") or key.startswith("s_"))
            }
            return {
                **passthrough,
                "filter_params": filter_params,
                "signal_params": signal_params,
            }
    if "moduleA" in p and "moduleB" in p:
        params_a = {key[2:]: value for key, value in p.items() if key.startswith("a_")}
        params_b = {key[2:]: value for key, value in p.items() if key.startswith("b_")}
        if params_a or params_b:
            passthrough = {
                key: value
                for key, value in p.items()
                if not (key.startswith("a_") or key.startswith("b_"))
            }
            return {
                **passthrough,
                "paramsA": params_a,
                "paramsB": params_b,
            }
    return p


def build_param_combo_long_table(
    pairs: Sequence[tuple[str, Mapping[str, Any]]],
) -> pd.DataFrame:
    """Long table for BI: one row per (combo, key); ``param_sort_order`` matches sorted keys."""
    records: list[dict[str, object]] = []
    for label, params in dedupe_combo_param_pairs(pairs):
        flat = flatten_params_for_combo_long_table(params)
        for order, (k, v) in enumerate(sorted(flat.items())):
            records.append(
                {
                    "param_combo_label": label,
                    "param_key": k,
                    "param_value": serialize_param_value_for_parquet(v),
                    "param_sort_order": order,
                }
            )
    cols = ["param_combo_label", "param_key", "param_value", "param_sort_order"]
    return pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=cols)


def internal_quantile_levels(*, n_bins: int) -> tuple[float, ...]:
    """Probability levels for (n_bins-1) interior edges, e.g. deciles → 0.1…0.9."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    return tuple(i / n_bins for i in range(1, n_bins))


def merge_feature_target_panel(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    *,
    feature_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Align feature and target on the shared row index (multi-ticker safe)."""
    need_f = [feature_col, "ticker"]
    missing_f = [c for c in need_f if c not in features_df.columns]
    if missing_f:
        raise ValueError(f"features_df missing columns: {missing_f}")
    if target_col not in targets_df.columns:
        raise ValueError(f"targets_df missing column: {target_col}")

    f = features_df[need_f]
    t = targets_df[[target_col]]
    merged = f.join(t, how="inner")
    merged = merged.rename(columns={feature_col: "feature", target_col: "target"})
    merged = merged.dropna(subset=["feature", "target", "ticker"])
    merged = merged.copy()
    merged["ticker"] = merged["ticker"].map(lambda x: x.name if hasattr(x, "name") else x)
    return merged


def quantile_bin_sets_for_strategy(direction: Direction, n_bins: int) -> tuple[frozenset[int], frozenset[int]]:
    """Long/short bin index sets for full-sample quantile binning (matches permutation / in-sample)."""
    if n_bins < 2:
        raise ValueError(f"quantile binning requires n_bins >= 2, got {n_bins}")
    top = n_bins - 1
    match direction:
        case Direction.LONG:
            return frozenset((0,)), frozenset()
        case Direction.SHORT:
            return frozenset(), frozenset((top,))
        case Direction.LONG_SHORT:
            return frozenset((0,)), frozenset((top,))


def discrete_signal_series_from_continuous_panel(
    aligned: pd.DataFrame,
    *,
    n_bins: int,
    strategy: Direction,
) -> pd.Series:
    """Full-sample ``pd.qcut`` per ``ticker``, then map bins to ±1/0 (same as permutation pipeline).

    Multi-ticker panels often repeat the same datetime in the index; ``reindex(aligned.index)``
    is invalid. We align via a stable integer row id, then rebuild the series with the original index.
    """
    need = {"feature", "ticker"}
    if not need.issubset(aligned.columns):
        raise ValueError(f"aligned must contain columns {sorted(need)}, got {list(aligned.columns)}")
    pos_col = "__fr_row_pos"
    work = aligned[["feature", "ticker"]].copy()
    work[pos_col] = np.arange(len(work), dtype=np.intp)
    long_bins, short_bins = quantile_bin_sets_for_strategy(strategy, n_bins)
    parts = [
        _discrete_signal_chunk_frame(
            grp,
            n_bins=n_bins,
            long_bins=long_bins,
            short_bins=short_bins,
            pos_col=pos_col,
        )
        for _, grp in work.groupby("ticker", sort=False)
    ]
    stacked = pd.concat(parts, axis=0).sort_values(pos_col, kind="mergesort")
    return pd.Series(
        stacked["_discrete_signal"].to_numpy(dtype=float),
        index=aligned.index,
        dtype=float,
    )


def _discrete_signal_chunk_frame(
    chunk: pd.DataFrame,
    *,
    n_bins: int,
    long_bins: frozenset[int],
    short_bins: frozenset[int],
    pos_col: str,
) -> pd.DataFrame:
    bpart = _assign_quantile_bins_one_group(
        chunk[["feature"]].astype(float),
        n_bins=n_bins,
    )
    sig = bpart["bin_index"].map(
        lambda bx: float(map_bin_selection_to_position(bx, long_bins=long_bins, short_bins=short_bins))
    )
    return pd.DataFrame(
        {
            pos_col: chunk[pos_col].to_numpy(dtype=np.intp),
            "_discrete_signal": sig.to_numpy(dtype=float),
        }
    )


def _assign_quantile_bins_one_group(g: pd.DataFrame, *, n_bins: int) -> pd.DataFrame:
    out = g.copy()
    feat = g["feature"].astype(float)
    try:
        cat = pd.qcut(feat, q=n_bins, duplicates="drop")
        out["bin_index"] = cat.cat.codes.replace(-1, np.nan).astype("Int64")
        out["bin_interval_label"] = cat.astype(str)
        clean = feat.dropna()
        edges = (
            np.quantile(clean.to_numpy(float), internal_quantile_levels(n_bins=n_bins))
            if len(clean) >= 2
            else np.array([])
        )
        out["bin_edges_json"] = json.dumps([float(x) for x in edges])
    except ValueError:
        out["bin_index"] = pd.Series(pd.NA, index=g.index, dtype="Int64")
        out["bin_interval_label"] = ""
        out["bin_edges_json"] = "[]"
    return out


def assign_quantile_bins(
    panel: pd.DataFrame,
    *,
    n_bins: int,
) -> pd.DataFrame:
    """Add bin_index and bin_edges_json using full-sample ``pd.qcut`` per (ticker, param_combo)."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    keys = [c for c in ("param_combo_label", "ticker") if c in panel.columns]
    if not keys:
        return _assign_quantile_bins_one_group(panel, n_bins=n_bins)
    parts = [_assign_quantile_bins_one_group(sub, n_bins=n_bins) for _, sub in panel.groupby(keys, sort=False)]
    return pd.concat(parts, axis=0).sort_index()


def _metric_row_for_targets(target: pd.Series, timeframe: TimeFrame) -> dict[str, float | int]:
    s = pd.to_numeric(target, errors="coerce").dropna()
    n_obs = int(len(s))
    if n_obs == 0:
        return {
            "n_obs": n_obs,
            "mean_return": float("nan"),
            "sortino": float("nan"),
            "sharpe": float("nan"),
            "t_stat": float("nan"),
        }
    if n_obs == 1:
        single = float(s.iloc[0])
        return {
            "n_obs": n_obs,
            "mean_return": single,
            "sortino": float("nan"),
            "sharpe": float("nan"),
            "t_stat": float("nan"),
        }
    return {
        "n_obs": n_obs,
        "mean_return": float(compute_param_sensitivity_metric(s, "mean", timeframe)),
        "sortino": float(compute_param_sensitivity_metric(s, "sortino", timeframe)),
        "sharpe": float(compute_param_sensitivity_metric(s, "sharpe", timeframe)),
        "t_stat": float(compute_param_sensitivity_metric(s, "t_stat", timeframe)),
    }


def summarize_bins_by_metrics(
    panel: pd.DataFrame,
    *,
    timeframe: TimeFrame,
    group_keys: tuple[str, ...] = ("param_combo_label", "ticker", "bin_index"),
    return_col: str = "target",
    active_signal_only: bool = False,
) -> pd.DataFrame:
    """One row per group with mean_return, sortino, sharpe, t_stat."""
    present = [k for k in group_keys if k in panel.columns]
    if "bin_index" not in present:
        raise ValueError("panel must contain bin_index")
    if return_col not in panel.columns:
        raise ValueError(f"panel must contain return column {return_col!r}")

    sliced = panel.dropna(subset=["bin_index"])
    records: list[dict[str, object]] = []
    for gkey, sub in sliced.groupby(list(present), observed=True):
        keys_tuple = gkey if isinstance(gkey, tuple) else (gkey,)
        key_map = dict(zip(present, keys_tuple))
        metric_slice = sub
        if active_signal_only and "strategy_signal" in metric_slice.columns:
            active = pd.to_numeric(metric_slice["strategy_signal"], errors="coerce").fillna(0.0) != 0.0
            metric_slice = metric_slice.loc[active]
        row = {**key_map, **_metric_row_for_targets(metric_slice[return_col], timeframe)}
        records.append(row)
    return pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=list(present) + [
        "n_obs",
        "mean_return",
        "sortino",
        "sharpe",
        "t_stat",
    ])


def rolling_quantile_edges_long(
    feature_series: pd.Series,
    *,
    window: int,
    min_periods: int,
    n_bins: int,
) -> pd.DataFrame:
    """Long-format rolling quantile thresholds at internal levels (stability over time)."""
    levels = internal_quantile_levels(n_bins=n_bins)
    s = feature_series.sort_index().astype(float)

    def level_frame(q: float, q_label: str) -> pd.DataFrame:
        rolled = s.rolling(window=window, min_periods=min_periods).quantile(q=q)
        df = rolled.rename("edge_value").reset_index()
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "datetime"})
        df["quantile_level"] = q_label
        return df

    frames = [level_frame(q, f"p{q:g}") for q in levels]
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def per_series_rolling_edges(
    panel: pd.DataFrame,
    *,
    window: int,
    min_periods: int,
    n_bins: int,
    group_keys: tuple[str, ...] = ("param_combo_label", "ticker"),
) -> pd.DataFrame:
    """Concat rolling edge rows for each feature series in panel."""
    keys = [k for k in group_keys if k in panel.columns]
    pieces: list[pd.DataFrame] = []

    def handle(sub: pd.DataFrame) -> None:
        feat = sub["feature"].sort_index()
        meta = {k: sub[k].iloc[0] for k in keys} if keys else {}
        edges = rolling_quantile_edges_long(
            feat,
            window=window,
            min_periods=min_periods,
            n_bins=n_bins,
        )
        for k, v in meta.items():
            edges[k] = v
        pieces.append(edges)

    if keys:
        for _, sub in panel.groupby(keys, sort=False):
            handle(sub)
    else:
        handle(panel)

    return pd.concat(pieces, axis=0, ignore_index=True) if pieces else pd.DataFrame()


def map_bin_selection_to_position(
    bin_index: object,
    *,
    long_bins: frozenset[int],
    short_bins: frozenset[int],
) -> int:
    """Pseudo signal: +1 long bin, -1 short, 0 neutral. For visualization export column."""
    if pd.isna(bin_index):
        return 0
    try:
        b = int(bin_index)
    except (TypeError, ValueError):
        return 0
    if b in long_bins:
        return 1
    if b in short_bins:
        return -1
    return 0


def add_research_position_column(
    panel: pd.DataFrame,
    *,
    long_bins: frozenset[int],
    short_bins: frozenset[int],
    position_col: str = "research_position",
) -> pd.DataFrame:
    """Optional column = sign(target weight) for cumulative return exploration in PBI."""
    out = panel.copy()
    out[position_col] = out["bin_index"].map(
        lambda x: map_bin_selection_to_position(x, long_bins=long_bins, short_bins=short_bins)
    )
    out["strategy_return"] = out[position_col].astype(float) * out["target"].astype(float)
    return out


def add_cumulative_return_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Time-ordered cumulative ``target`` / ``strategy_return`` per series for BI exports.

    Groups by ``ticker`` and ``param_combo_label`` when present; otherwise one global series.
    NaNs in the summed columns are treated as 0 for the running total.
    """
    out = panel.copy()
    if out.empty or "target" not in out.columns:
        return out

    idx_name = out.index.name
    work = out.reset_index()
    datetime_col = str(work.columns[0])

    group_keys = [k for k in ("ticker", "param_combo_label") if k in work.columns]
    sort_cols = group_keys + [datetime_col] if group_keys else [datetime_col]
    work = work.sort_values(sort_cols, kind="mergesort")

    def _cumsum_series(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").fillna(0.0).cumsum()

    if group_keys:
        gb = work.groupby(group_keys, sort=False)
        work["cumulative_target"] = gb["target"].transform(_cumsum_series)
        if "strategy_return" in work.columns:
            work["cumulative_strategy_return"] = gb["strategy_return"].transform(_cumsum_series)
    else:
        work["cumulative_target"] = _cumsum_series(work["target"])
        if "strategy_return" in work.columns:
            work["cumulative_strategy_return"] = _cumsum_series(work["strategy_return"])

    restored = work.set_index(datetime_col)
    restored.index.name = idx_name
    return restored.sort_index()
