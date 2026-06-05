"""Shared tabular exports for research phases (Matplotlib-ready CSV tables).

In-sample param dimensions are **long-only** (`param_combo_long.csv`): facts
(`param_sensitivity`, equity, permutation) carry ``param_combo_label`` only; join to
long param rows for slicers. Phase 0 binning writes long Parquet for combo metadata.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from quantfoundry_core.metrics import ReturnsValidationError, compute_rolling_sharpe

from research.feature.binning.transforms import build_param_combo_long_table
from research.feature.filter_research_labels import (
    build_filter_exploration_long_pairs,
    filter_exploration_semantics,
    research_display_label,
)
from research.feature._internal.core_helpers import combo_key, normalize_datetime_index
from research.feature.in_sample.data_loader import param_combo_label, permutation_combo_display_name
from research.feature.shared.visualization_paths import (
    VISUALIZATION_SUBDIR_NAME,
    canonical_in_sample_visualization_dir,
    walkforward_visualization_csv_dir,
)
from features.validation.stability_analysis import _param_combo_name
from lib.core.enums import TimeFrame


def return_kind_for_target(target_col: str) -> str:
    """Map a ``target_col`` name to the matching ``instrument_return_kind``.

    Overnight targets (``overnight_log_return*``) use ``'log_overnight'``;
    everything else defaults to ``'log_intraday'``.
    """
    return "log_overnight" if target_col.startswith("overnight_") else "log_intraday"
from lib.core.helpers import build_feature_column_name
from research.evaluation.walkforward.selected_params_codec import decode_selected_params_list

AGGREGATE_EQUITY_TICKER = "ALL"

_IDM_MAX: float = 2.5
# Half-year of daily bars; rolling Sharpe uses this window on per-bar strategy returns.
DEFAULT_ROLLING_SHARPE_WINDOW_BARS: int = 252

_WALKFORWARD_EQUITY_BASE_COLS: list[str] = [
    "datetime",
    "param_combo_label",
    "feature_name",
    "ticker",
    "strategy_return",
    "cumulative_strategy_return",
    "fold_id",
]
_WALKFORWARD_EQUITY_ROLLING_COLS: list[str] = [
    "rolling_sharpe_annualized",
    "rolling_sharpe_window_bars",
]

# Stable CSV schema for visualization consumers (avoids missing columns after refresh).
_PARAM_SENSITIVITY_POOL_COLS: list[str] = [
    "param_combo_label",
    "feature_name",
    "n_observations",
    "n_nonzero_signal",
    "sharpe",
    "t_stat",
    "sortino",
]
_PARAM_SENSITIVITY_BY_TICKER_COLS: list[str] = [
    "param_sensitivity_by_ticker_key",
    "param_combo_label",
    "feature_name",
    "ticker",
    "n_observations",
    "n_nonzero_signal",
    "sharpe",
    "t_stat",
    "sortino",
]


def _dataframe_with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reindex to exact column list; missing columns become NaN, extras are dropped."""
    return df.reindex(columns=columns)


def _assert_unique_param_sensitivity_by_ticker_keys(bt_df: pd.DataFrame) -> None:
    """``param_sensitivity_by_ticker`` is one row per (combo, ticker); keys must not collide."""
    col = bt_df["param_sensitivity_by_ticker_key"]
    if col.duplicated().any():
        dup_vals = col[col.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            "param_sensitivity_by_ticker_key must be unique per row (visualization row key). "
            f"Duplicate values: {dup_vals[:25]}"
            + (" …" if len(dup_vals) > 25 else "")
        )

if TYPE_CHECKING:
    from features.validation.reports import PermutationTestSuite


def _resolve_in_sample_visualization_dir(
    *,
    visualization_parent_dir: Path | None,
    powerbi_parent_dir: Path | None = None,
) -> Path:
    """Resolve the stable in-sample visualization directory.

    ``powerbi_parent_dir`` remains as a compatibility input while call sites migrate
    away from Power BI naming.
    """
    if visualization_parent_dir is not None and powerbi_parent_dir is not None:
        raise ValueError(
            "Pass only one of visualization_parent_dir or powerbi_parent_dir."
        )
    parent_dir = (
        visualization_parent_dir
        if visualization_parent_dir is not None
        else powerbi_parent_dir
    )
    if parent_dir is None:
        return canonical_in_sample_visualization_dir()
    return parent_dir / VISUALIZATION_SUBDIR_NAME


def objective_metric_display_label(spec: object | None) -> str:
    """Human-readable objective name from ``ObjectiveMetricSpec`` or similar."""
    if spec is None:
        return "unknown"
    builtin = getattr(spec, "builtin", None)
    if builtin is not None:
        return str(builtin)
    path = getattr(spec, "callable_path", None)
    if path is not None:
        return str(path)
    return "unknown"


def _write_frame_csv(df: pd.DataFrame, stem: Path) -> Path:
    csv_path = stem.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def _equity_curve_frame_for_combo(
    param_combo_label: str,
    signal: pd.Series,
    target: pd.Series,
    ticker: pd.Series,
    feature_name: str,
) -> pd.DataFrame:
    """Per-combo long table: cumsum of ``signal * target`` **per instrument** (multi-ticker safe)."""
    cols = [
        "datetime",
        "param_combo_label",
        "feature_name",
        "ticker",
        "strategy_return",
        "cumulative_strategy_return",
    ]
    work = pd.concat(
        [signal.rename("signal"), target.rename("target"), ticker.rename("ticker")],
        axis=1,
    ).dropna(how="any")
    if work.empty:
        return pd.DataFrame(columns=cols)
    work["ticker"] = work["ticker"].astype(str)

    def _block_for_instrument(inst: str) -> pd.DataFrame:
        sub = work.loc[work["ticker"].eq(inst), ["signal", "target"]].sort_index(kind="mergesort")
        strat = pd.to_numeric(sub["signal"] * sub["target"], errors="coerce").fillna(0.0)
        cumulative = strat.cumsum().rename("cumulative_strategy_return")
        block = (
            strat.rename("strategy_return")
            .to_frame()
            .assign(cumulative_strategy_return=cumulative, ticker=inst)
            .reset_index()
        )
        first = str(block.columns[0])
        block = block.rename(columns={first: "datetime"})
        block["param_combo_label"] = param_combo_label
        block["feature_name"] = feature_name
        return block[cols]

    instruments = sorted(work["ticker"].unique())
    pieces = [_block_for_instrument(inst) for inst in instruments]
    if len(instruments) > 1:
        pieces.append(
            _aggregate_equity_block_for_combo(
                work,
                param_combo_label=param_combo_label,
                feature_name=feature_name,
            )
        )
    return pd.concat(pieces, axis=0, ignore_index=True) if pieces else pd.DataFrame(columns=cols)


def _aggregate_equity_block_for_combo(
    work: pd.DataFrame,
    *,
    param_combo_label: str,
    feature_name: str,
) -> pd.DataFrame:
    """Equal-weight mean ``signal * target`` per datetime across instruments, then cumsum."""
    cols = [
        "datetime",
        "param_combo_label",
        "feature_name",
        "ticker",
        "strategy_return",
        "cumulative_strategy_return",
    ]
    strat = pd.to_numeric(work["signal"] * work["target"], errors="coerce").fillna(0.0)
    by_date = strat.groupby(strat.index).mean().sort_index(kind="mergesort")
    cumulative = by_date.cumsum()
    block = pd.DataFrame(
        {
            "datetime": by_date.index,
            "param_combo_label": param_combo_label,
            "feature_name": feature_name,
            "ticker": AGGREGATE_EQUITY_TICKER,
            "strategy_return": by_date.to_numpy(dtype=float),
            "cumulative_strategy_return": cumulative.to_numpy(dtype=float),
        }
    )
    block["datetime"] = pd.to_datetime(block["datetime"], errors="coerce")
    return block[cols]


def _vol_scaled_equity_frame_for_combo(
    param_combo_label_str: str,
    signal: pd.Series,
    ticker: pd.Series,
    feature_name: str,
    params: dict[str, Any],
    candles: pd.DataFrame,
    timeframe: TimeFrame,
    *,
    target_volatility: float,
    instrument_return_kind: str = "log_intraday",
) -> pd.DataFrame:
    """Per-combo equity frame using production vol-targeting: ``position_fraction × log_return``.

    Delegates to ``_build_portfolio_positions_df`` so the same forecast-scaling and IDM
    logic used in the validation equity curve is applied here, making both curves
    directly comparable.  Falls back to an empty DataFrame on any error.
    """
    from ensemble.ensemble_utils import normalize_ticker_key
    from ensemble.portfolio_impl.portfolio_tester import calculate_strategy_returns_from_positions

    cols = list(_WALKFORWARD_EQUITY_BASE_COLS) + ["fold_id"]
    try:
        positions_df = _build_portfolio_positions_df(
            signal,
            ticker,
            candles,
            candles,
            timeframe,
            target_volatility=target_volatility,
        )
        if positions_df.empty:
            return pd.DataFrame(columns=cols)

        candle_dt = pd.to_datetime(candles["datetime"]).dt.tz_localize(None)
        sig_idx = pd.to_datetime(signal.index)
        if sig_idx.tz is not None:
            sig_idx = sig_idx.tz_localize(None)
        sig_lo, sig_hi = sig_idx.min(), sig_idx.max()
        candles_slice = candles.loc[(candle_dt >= sig_lo) & (candle_dt <= sig_hi)].copy()

        frames: list[pd.DataFrame] = []
        for raw_t in sorted(positions_df["ticker"].astype(str).unique()):
            norm_t = normalize_ticker_key(raw_t)
            t_positions = positions_df[positions_df["ticker"].astype(str) == raw_t].copy()
            t_candles = candles_slice[
                candles_slice["ticker"].astype(str).map(normalize_ticker_key) == norm_t
            ].copy()
            if t_positions.empty or t_candles.empty:
                continue
            returns = calculate_strategy_returns_from_positions(
                t_positions, t_candles, instrument_return_kind=instrument_return_kind
            )
            if returns.empty:
                continue
            returns_idx = normalize_datetime_index(returns.index)
            block = pd.DataFrame(
                {
                    "datetime": returns_idx,
                    "param_combo_label": param_combo_label_str,
                    "feature_name": feature_name,
                    "ticker": raw_t,
                    "strategy_return": returns.to_numpy(dtype=float),
                    "cumulative_strategy_return": returns.cumsum().to_numpy(dtype=float),
                    "fold_id": 0,
                }
            )
            frames.append(block)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    except Exception:
        return pd.DataFrame(columns=cols)


def write_in_sample_equity_curve_csv(
    combo_store: Mapping[str, tuple[pd.Series, pd.Series, str, pd.Series, dict[str, Any]]],
    *,
    visualization_parent_dir: Path | None = None,
    portfolio_candles: pd.DataFrame | None = None,
    timeframe: TimeFrame = TimeFrame.D,
    target_volatility: float = 0.15,
    instrument_return_kind: str = "log_intraday",
    csv_stem: str = "equity_curve",
) -> Path:
    """Write a single long CSV of equity curves for all IS combos.

    When *portfolio_candles* is provided the equity curve uses production
    vol-targeting (``position_fraction × log_return``) — the same metric as the
    validation tearsheet — so both curves are directly comparable.

    When *portfolio_candles* is absent the legacy ``signal × log_return_ewsd``
    path is used as a fallback.

    Column names align with binning exports (``strategy_return``,
    ``cumulative_strategy_return``).  Writes next to other in-sample
    visualization CSVs under :func:`canonical_in_sample_visualization_dir`.
    """
    out_dir = _resolve_in_sample_visualization_dir(
        visualization_parent_dir=visualization_parent_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if portfolio_candles is not None and not portfolio_candles.empty:
        frames = [
            _vol_scaled_equity_frame_for_combo(
                label, sig, tkr, feat, params, portfolio_candles, timeframe,
                target_volatility=target_volatility,
                instrument_return_kind=instrument_return_kind,
            )
            for label, (sig, _tgt, feat, tkr, params) in sorted(combo_store.items())
        ]
    else:
        frames = [
            _equity_curve_frame_for_combo(label, sig, tgt, tkr, feat)
            for label, (sig, tgt, feat, tkr, _) in sorted(combo_store.items())
        ]

    non_empty = [f for f in frames if not f.empty]
    combined = (
        pd.concat(non_empty, axis=0, ignore_index=True)
        if non_empty
        else pd.DataFrame(
            columns=[
                "datetime",
                "param_combo_label",
                "feature_name",
                "ticker",
                "strategy_return",
                "cumulative_strategy_return",
            ]
        )
    )
    return _write_frame_csv(combined, out_dir / csv_stem)


def _normalize_ts(ts: pd.Timestamp | object) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is not None:
        out = out.tz_localize(None)
    return out


def _rolling_sharpe_annualized_series(
    per_bar_returns: pd.Series,
    *,
    window_bars: int,
    bars_per_year: float,
) -> pd.Series:
    """Rolling Sharpe from ``quantfoundry_core.metrics.compute_rolling_sharpe``."""
    x = pd.to_numeric(per_bar_returns, errors="coerce")
    win = max(2, int(window_bars))
    periods_per_year = max(1, int(round(float(bars_per_year))))
    try:
        snapshot = compute_rolling_sharpe(
            x,
            rolling_window=win,
            periods_per_year=periods_per_year,
        )
    except ReturnsValidationError:
        return pd.Series(float("nan"), index=x.index, name="rolling_sharpe_annualized")
    return snapshot.to_series().reindex(x.index).rename("rolling_sharpe_annualized")


def _rolling_sharpe_group(
    group: pd.DataFrame,
    *,
    window_bars: int,
    bars_per_year: float,
) -> pd.Series:
    indexed_returns = pd.Series(
        pd.to_numeric(group["strategy_return"], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(group["datetime"], utc=False)),
        dtype="float64",
    )
    rolled = _rolling_sharpe_annualized_series(
        indexed_returns,
        window_bars=window_bars,
        bars_per_year=bars_per_year,
    )
    return pd.Series(
        rolled.to_numpy(dtype=float),
        index=group.index,
        name="rolling_sharpe_annualized",
    )


def enrich_walkforward_equity_df_with_rolling_sharpe(
    df: pd.DataFrame,
    timeframe: TimeFrame,
    *,
    window_bars: int = DEFAULT_ROLLING_SHARPE_WINDOW_BARS,
) -> pd.DataFrame:
    """Add rolling annualized Sharpe columns for visualization consumers.

    ``strategy_return`` must be per-bar; rolling uses the same annualization as
    in-sample param sensitivity Sharpe (``sqrt(bars_per_year)`` × mean / std).
    """
    win = max(2, int(window_bars))
    bars_py = float(timeframe.bars_per_year)
    if df.empty:
        return pd.DataFrame(
            columns=[*_WALKFORWARD_EQUITY_BASE_COLS, *_WALKFORWARD_EQUITY_ROLLING_COLS]
        )
    keys = ["fold_id", "param_combo_label", "ticker"]
    ordered = df.sort_values([*keys, "datetime"], kind="mergesort")
    rolled_groups = [
        _rolling_sharpe_group(
            group,
            window_bars=win,
            bars_per_year=bars_py,
        )
        for _, group in ordered.groupby(keys, sort=False)
    ]
    rolled = pd.concat(rolled_groups, axis=0).sort_index() if rolled_groups else pd.Series(dtype=float)
    return ordered.assign(
        rolling_sharpe_annualized=rolled.reindex(ordered.index).to_numpy(dtype=float),
        rolling_sharpe_window_bars=win,
    )


def _slice_combo_panel(
    paired: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``signal``, ``target``, ``ticker`` series restricted to ``[start, end]`` inclusive."""
    if paired.empty:
        empty = pd.Series(dtype=float)
        tempty = pd.Series(dtype=str)
        return empty.rename("signal"), empty.rename("target"), tempty.rename("ticker")
    idx = paired.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(pd.to_datetime(idx, utc=False))
        paired = paired.copy()
        paired.index = idx
    idx = normalize_datetime_index(idx)
    paired = paired.copy()
    paired.index = idx
    lo, hi = _normalize_ts(start), _normalize_ts(end)
    mask = (idx >= lo) & (idx <= hi)
    sub = paired.loc[mask]
    signal = sub["signal"]
    target = sub["target"]
    ticker = sub["ticker"].astype(str)
    return signal, target, ticker


def _equity_frames_for_selection(
    combo_signal_target: Mapping[tuple[tuple[str, object], ...], pd.DataFrame],
    selected_params: list[dict[str, Any]],
    *,
    module_name: str,
    timeframe: TimeFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fold_id: int,
) -> list[pd.DataFrame]:
    """Legacy equity curve frames using ``signal × target`` (EWSD-normalised returns)."""
    frames: list[pd.DataFrame] = []
    for params in selected_params:
        key = combo_key(params)
        paired = combo_signal_target.get(key)
        if paired is None or paired.empty:
            continue
        sig, tgt, tkr = _slice_combo_panel(paired, start, end)
        label = param_combo_label(params)
        feat = build_feature_column_name(
            module=module_name,
            feature="signal",
            tf=timeframe,
            params=params,
        )
        block = _equity_curve_frame_for_combo(label, sig, tgt, tkr, feat)
        if block.empty:
            continue
        block = block.assign(fold_id=int(fold_id))
        frames.append(block)
    return frames


_FORECAST_CAP: float = 2.0


def _vol_scaled_forecast(
    signal: pd.Series,
    ticker: pd.Series,
    eval_candles: pd.DataFrame,
    *,
    target_volatility: float,
) -> pd.Series:
    """Return vol-targeted forecast: ``F = min(tau / EWSD[t], 2.0) * signal``.

    EWSD is computed causally from *eval_candles* (which should include training
    history so the EWMA is warmed up before the signal period begins).
    Falls back to *target_volatility* for any bar where EWSD cannot be resolved.
    """
    import numpy as np
    from lib.compute.daily_ewsd_volatility import DailyEWSDVolatilityService
    from ensemble.ensemble_utils import normalize_ticker_key

    svc = DailyEWSDVolatilityService()
    daily_vol_df = svc.compute_daily_series(eval_candles)

    # Build a minimal (datetime, ticker) frame for alignment — no close needed.
    signal_dt = pd.to_datetime(signal.index).tz_localize(None)
    signal_candles = pd.DataFrame(
        {"datetime": signal_dt, "ticker": ticker.astype(str).values}
    )
    try:
        aligned = svc.align_daily_volatility_to_candles(daily_vol_df, signal_candles)
    except ValueError:
        # Alignment failed (e.g. ticker not in eval_candles): use default vol.
        return pd.Series(
            (target_volatility / target_volatility) * signal.to_numpy(dtype=float),
            index=signal.index,
            name="forecast_score",
        )

    aligned["_dt_key"] = pd.to_datetime(aligned["datetime"]).dt.tz_localize(None).dt.floor("s")
    aligned["_tk_key"] = aligned["ticker"].map(normalize_ticker_key)
    vol_lookup = (
        aligned
        .drop_duplicates(subset=["_tk_key", "_dt_key"], keep="last")
        .set_index(["_tk_key", "_dt_key"])["ewsd_annual_vol"]
    )

    ticker_norm = ticker.astype(str).map(normalize_ticker_key)
    dt_keys = signal_dt.floor("s")
    mi = pd.MultiIndex.from_arrays([ticker_norm.values, dt_keys])
    ewsd = vol_lookup.reindex(mi).fillna(target_volatility).to_numpy(dtype=float)
    ewsd = np.maximum(ewsd, 1e-6)

    forecast = np.minimum(target_volatility / ewsd, _FORECAST_CAP) * signal.to_numpy(dtype=float)
    return pd.Series(forecast, index=signal.index, name="forecast_score")


def _build_portfolio_positions_df(
    signal: pd.Series,
    ticker: pd.Series,
    train_candles: pd.DataFrame,
    eval_candles: pd.DataFrame,
    timeframe: TimeFrame,
    *,
    target_volatility: float = 0.15,
) -> pd.DataFrame:
    """Fit TFPortfolio on training candles and return a positions_df for the signal period.

    Applies the full production vol-targeting formula before passing to TFPortfolio:

        forecast_score = min(tau / EWSD[t], 2.0) × signal

    where *tau* is *target_volatility* and EWSD[t] is the causal blended volatility
    from *eval_candles*.  TFPortfolio then applies instrument weights and IDM:

        position_fraction = forecast_score × instrument_weight × IDM
    """
    from ensemble.portfolio_impl.portfolio_returns import calculate_returns_from_candles
    from ensemble.portfolio_impl.tf_portfolio import TFPortfolio

    instrument_returns = calculate_returns_from_candles(train_candles)
    portfolio = TFPortfolio(ensembles=[], trading_timeframe=timeframe, idm_max=_IDM_MAX)
    portfolio.fit(instrument_returns)

    idx = normalize_datetime_index(signal.index)
    forecast_score = _vol_scaled_forecast(
        signal, ticker, eval_candles, target_volatility=target_volatility
    )
    combined_forecasts = pd.DataFrame(
        {
            "ticker": ticker.astype(str).to_numpy(),
            "forecast_score": forecast_score.to_numpy(dtype=float),
        },
        index=idx,
    )
    result = portfolio.predict(combined_forecasts)
    return pd.DataFrame(
        {
            "ticker": result["ticker"].to_numpy(),
            "datetime": idx,
            "position_fraction": result["position_fraction"].to_numpy(dtype=float),
        }
    )


def _portfolio_equity_frames_for_selection(
    combo_signal_target: Mapping[tuple[tuple[str, object], ...], pd.DataFrame],
    selected_params: list[dict[str, Any]],
    *,
    module_name: str,
    timeframe: TimeFrame,
    train_candles: pd.DataFrame,
    eval_candles: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fold_id: int,
    target_volatility: float = 0.15,
) -> list[pd.DataFrame]:
    """Equity curve frames using TFPortfolio simulation (actual price returns)."""
    from ensemble.ensemble_utils import normalize_ticker_key
    from ensemble.portfolio_impl.portfolio_tester import calculate_strategy_returns_from_positions

    cols = list(_WALKFORWARD_EQUITY_BASE_COLS)
    lo, hi = _normalize_ts(start), _normalize_ts(end)
    candle_dt = pd.to_datetime(eval_candles["datetime"])
    candles_slice = eval_candles.loc[(candle_dt >= lo) & (candle_dt <= hi)].copy()

    frames: list[pd.DataFrame] = []
    for params in selected_params:
        key = combo_key(params)
        paired = combo_signal_target.get(key)
        if paired is None or paired.empty:
            continue
        sig, _tgt, tkr = _slice_combo_panel(paired, start, end)
        if sig.empty:
            continue

        label = param_combo_label(params)
        feat = build_feature_column_name(
            module=module_name, feature="signal", tf=timeframe, params=params
        )

        try:
            positions_df = _build_portfolio_positions_df(
                sig, tkr, train_candles, eval_candles, timeframe,
                target_volatility=target_volatility,
            )
        except Exception:
            continue

        if positions_df.empty:
            continue

        raw_tickers = sorted(positions_df["ticker"].astype(str).unique())
        for raw_t in raw_tickers:
            norm_t = normalize_ticker_key(raw_t)
            t_positions = positions_df[positions_df["ticker"].astype(str) == raw_t].copy()
            t_candles = candles_slice[
                candles_slice["ticker"].astype(str).map(normalize_ticker_key) == norm_t
            ].copy()
            if t_positions.empty or t_candles.empty:
                continue

            returns = calculate_strategy_returns_from_positions(t_positions, t_candles)
            if returns.empty:
                continue

            returns_idx = normalize_datetime_index(returns.index)
            block = pd.DataFrame(
                {
                    "datetime": returns_idx,
                    "param_combo_label": label,
                    "feature_name": feat,
                    "ticker": raw_t,
                    "strategy_return": returns.to_numpy(dtype=float),
                    "cumulative_strategy_return": returns.cumsum().to_numpy(dtype=float),
                    "fold_id": int(fold_id),
                }
            )[cols]
            frames.append(block)
    return frames


def write_walkforward_equity_csvs(
    *,
    combo_signal_target: Mapping[tuple[tuple[str, object], ...], pd.DataFrame],
    selection_summary_df: pd.DataFrame,
    module_name: str,
    timeframe: TimeFrame,
    holdout_start: object,
    holdout_end: object,
    extended_start: object,
    extended_end: object,
    output_visualization_dir: Path,
    holdout_csv_stem: str,
    extended_csv_stem: str,
    portfolio_candles: pd.DataFrame | None = None,
    rolling_sharpe_window_bars: int = DEFAULT_ROLLING_SHARPE_WINDOW_BARS,
    target_volatility: float = 0.15,
) -> dict[str, Path]:
    """Write two equity-curve CSVs for validation or OOS (long format).

    Each CSV includes **rolling annualized Sharpe** on ``strategy_return`` (same
    annualization as in-sample param sensitivity): columns ``rolling_sharpe_annualized``
    and ``rolling_sharpe_window_bars`` (constant = ``rolling_sharpe_window_bars``).
    Rolling is computed **within** each ``(fold_id, param_combo_label, ticker)`` group
    after sorting by ``datetime`` (first ``window_bars - 1`` rows are NaN per group).

    When ``portfolio_candles`` is provided the equity curves are generated via
    the full production vol-targeting pipeline:

    * ``TFPortfolio`` is fit on the training candles (dates before the holdout
      window) to derive IDM and equal instrument weights.
    * EWSD is computed causally from ``portfolio_candles`` (includes train history
      for EWMA warm-up).
    * ``forecast_score = min(target_volatility / EWSD[t], 2.0) × signal``
    * ``position_fraction = forecast_score × instrument_weight × IDM``
    * ``calculate_strategy_returns_from_positions`` computes lookahead-free
      actual log P&L per ticker.

    When ``portfolio_candles`` is ``None`` the legacy ``signal × target`` path
    is used (EWSD-normalised returns).

    **Holdout** slice: validation phase → validation window only; OOS phase → OOS
    test window only.

    **Extended** slice: validation → train + validation; OOS → effective train
    start (train + val when ``research_window`` is set) through validation end.

    All rows carry a ``fold_id`` column so Matplotlib or other CSV consumers can
    slice per fold.
    """
    output_visualization_dir.mkdir(parents=True, exist_ok=True)
    hs, he = _normalize_ts(holdout_start), _normalize_ts(holdout_end)
    xs, xe = _normalize_ts(extended_start), _normalize_ts(extended_end)
    use_portfolio = portfolio_candles is not None and not portfolio_candles.empty

    holdout_parts: list[pd.DataFrame] = []
    extended_parts: list[pd.DataFrame] = []

    for _, summary_row in selection_summary_df.iterrows():
        fold_id = int(summary_row["fold_id"])
        selected = decode_selected_params_list(str(summary_row["selected_params_json"]))
        if not selected:
            continue

        if use_portfolio:
            assert portfolio_candles is not None
            # Training candles: everything before the holdout window.
            candle_dt = pd.to_datetime(portfolio_candles["datetime"])
            train_candles = portfolio_candles.loc[
                (candle_dt >= xs) & (candle_dt < hs)
            ].copy()
            holdout_parts.extend(
                _portfolio_equity_frames_for_selection(
                    combo_signal_target,
                    selected,
                    module_name=module_name,
                    timeframe=timeframe,
                    train_candles=train_candles,
                    eval_candles=portfolio_candles,
                    start=hs,
                    end=he,
                    fold_id=fold_id,
                    target_volatility=target_volatility,
                )
            )
            extended_parts.extend(
                _portfolio_equity_frames_for_selection(
                    combo_signal_target,
                    selected,
                    module_name=module_name,
                    timeframe=timeframe,
                    train_candles=train_candles,
                    eval_candles=portfolio_candles,
                    start=xs,
                    end=xe,
                    fold_id=fold_id,
                    target_volatility=target_volatility,
                )
            )
        else:
            holdout_parts.extend(
                _equity_frames_for_selection(
                    combo_signal_target,
                    selected,
                    module_name=module_name,
                    timeframe=timeframe,
                    start=hs,
                    end=he,
                    fold_id=fold_id,
                )
            )
            extended_parts.extend(
                _equity_frames_for_selection(
                    combo_signal_target,
                    selected,
                    module_name=module_name,
                    timeframe=timeframe,
                    start=xs,
                    end=xe,
                    fold_id=fold_id,
                )
            )

    def _concat_or_empty(pieces: list[pd.DataFrame]) -> pd.DataFrame:
        non_empty = [p for p in pieces if not p.empty]
        if not non_empty:
            return pd.DataFrame(columns=_WALKFORWARD_EQUITY_BASE_COLS)
        return pd.concat(non_empty, axis=0, ignore_index=True)

    holdout_df = enrich_walkforward_equity_df_with_rolling_sharpe(
        _concat_or_empty(holdout_parts),
        timeframe,
        window_bars=rolling_sharpe_window_bars,
    )
    extended_df = enrich_walkforward_equity_df_with_rolling_sharpe(
        _concat_or_empty(extended_parts),
        timeframe,
        window_bars=rolling_sharpe_window_bars,
    )
    return {
        "holdout": _write_frame_csv(holdout_df, output_visualization_dir / holdout_csv_stem),
        "extended": _write_frame_csv(extended_df, output_visualization_dir / extended_csv_stem),
    }


def write_param_sensitivity_tables(
    sensitivity_rows: Sequence[Mapping[str, object]],
    label_param_pairs: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    sensitivity_by_ticker_rows: Sequence[Mapping[str, object]] | None = None,
    visualization_parent_dir: Path | None = None,
) -> dict[str, Path]:
    """Write param-sensitivity metrics and long param dimension as CSV only.

    Fact tables omit wide ``param_<key>`` columns; join ``param_combo_label`` to
    ``param_combo_long.csv`` for parameter slicers.

    Writes under :func:`canonical_in_sample_visualization_dir` unless
    ``visualization_parent_dir`` is set (tests: files go to
    ``visualization_parent_dir / "visualization"``).

    Metrics (``sharpe`` annualized, ``t_stat``, ``sortino``) use
    ``feature_research.in_sample.metric_helpers.compute_param_sensitivity_metric`` on the
    full aligned sample of per-bar strategy returns ``signal * target`` (zeros on flat bars).

    When ``sensitivity_by_ticker_rows`` is non-empty, also writes ``param_sensitivity_by_ticker.csv``
    (same metrics computed **within** each instrument). Rows are one per
    ``(param_combo_label, ticker)``; ``param_combo_label`` repeats across tickers.
    Downstream plotting code should treat **only**
    ``param_sensitivity_by_ticker_key`` as the row identifier. Do **not** treat
    ``param_combo_label`` as a key; it repeats once per ticker. Relationships to
    ``param_combo_long`` / ``param_sensitivity`` use ``param_combo_label`` with
    *many* (by-ticker) → *one* (pooled / long) cardinality.

    Returns
    -------
    dict[str, Path]
        Paths keyed by artifact stem: ``param_sensitivity_csv``, ``param_sensitivity_by_ticker_csv``
        (only if by-ticker rows provided), ``param_combo_long_csv``.
    """
    out_dir = _resolve_in_sample_visualization_dir(
        visualization_parent_dir=visualization_parent_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = _dataframe_with_columns(
        pd.DataFrame(list(sensitivity_rows)),
        _PARAM_SENSITIVITY_POOL_COLS,
    )
    long_tbl = build_param_combo_long_table(label_param_pairs)

    paths: dict[str, Path] = {}
    paths["param_sensitivity_csv"] = _write_frame_csv(metrics_df, out_dir / "param_sensitivity")
    paths["param_combo_long_csv"] = _write_frame_csv(long_tbl, out_dir / "param_combo_long")

    if sensitivity_by_ticker_rows is not None and len(sensitivity_by_ticker_rows) > 0:
        bt_df = _dataframe_with_columns(
            pd.DataFrame(list(sensitivity_by_ticker_rows)),
            _PARAM_SENSITIVITY_BY_TICKER_COLS,
        )
        _assert_unique_param_sensitivity_by_ticker_keys(bt_df)
        paths["param_sensitivity_by_ticker_csv"] = _write_frame_csv(
            bt_df, out_dir / "param_sensitivity_by_ticker"
        )

    return paths


_FILTER_EXPLORATION_SUMMARY_COLS: list[str] = [
    "param_combo_label",
    "research_display_label",
    "gate_type",
    "filter_family",
    "gate_mode",
    "vol_max_rank",
    "filter_detail",
    "n_observations",
    "n_nonzero_signal",
    "trade_reduction_pct",
    "sharpe",
    "t_stat",
    "sortino",
]


def write_filter_exploration_tables(
    sensitivity_rows: Sequence[Mapping[str, object]],
    combo_store: Mapping[str, tuple[object, object, object, object, Mapping[str, Any]]],
    *,
    visualization_parent_dir: Path | None = None,
) -> dict[str, Path]:
    """Write filter-focused summary + long pivot CSVs (readable A/B/C exploration)."""
    out_dir = _resolve_in_sample_visualization_dir(
        visualization_parent_dir=visualization_parent_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_label = {str(row["param_combo_label"]): dict(row) for row in sensitivity_rows}
    baseline_n = None
    summary_rows: list[dict[str, object]] = []
    long_pairs: list[tuple[str, Mapping[str, Any]]] = []

    for label, (_, _, _, _, params) in sorted(combo_store.items()):
        metrics = metrics_by_label.get(label, {})
        display = research_display_label(params, label=label)
        n_nz = int(metrics.get("n_nonzero_signal", 0) or 0)
        if display.startswith("A:"):
            baseline_n = n_nz
        long_pairs.extend(build_filter_exploration_long_pairs(label, params))
        semantics = filter_exploration_semantics(params, label=label)
        trade_red = None
        if baseline_n and baseline_n > 0 and not display.startswith("A:"):
            trade_red = round((baseline_n - n_nz) / baseline_n * 100.0, 1)
        summary_rows.append(
            {
                "param_combo_label": label,
                "research_display_label": display,
                "gate_type": semantics["gate_type"],
                "filter_family": semantics["filter_family"],
                "gate_mode": semantics["gate_mode"],
                "vol_max_rank": semantics["vol_max_rank"],
                "filter_detail": semantics["filter_detail"],
                "n_observations": metrics.get("n_observations"),
                "n_nonzero_signal": n_nz,
                "trade_reduction_pct": trade_red,
                "sharpe": metrics.get("sharpe"),
                "t_stat": metrics.get("t_stat"),
                "sortino": metrics.get("sortino"),
            }
        )

    summary_df = _dataframe_with_columns(
        pd.DataFrame(summary_rows),
        _FILTER_EXPLORATION_SUMMARY_COLS,
    )
    long_tbl = build_param_combo_long_table(long_pairs)
    paths = {
        "filter_exploration_summary_csv": _write_frame_csv(
            summary_df, out_dir / "filter_exploration_summary"
        ),
        "filter_exploration_long_csv": _write_frame_csv(
            long_tbl, out_dir / "filter_exploration_long"
        ),
    }
    return paths


def permutation_vector_shuffle_records(
    suite: PermutationTestSuite,
    objective_metric_label: str,
    *,
    param_grid: list[dict[str, Any]] | None = None,
) -> list[dict[str, object]]:
    """One row per combo for vector-shuffle permutation results.

    ``param_combo`` is the stable internal key (``_param_combo_name``). When
    ``param_grid`` is provided, ``param_combo_label`` is a short human-readable
    description (for example cyclical RSI parent params from a nested research payload).
    """
    params_by_combo: dict[str, dict[str, Any]] | None = None
    if param_grid is not None:
        params_by_combo = {_param_combo_name(p): p for p in param_grid}

    def _label(combo_name: str) -> str:
        if params_by_combo is None:
            return combo_name
        params = params_by_combo.get(combo_name)
        return (
            permutation_combo_display_name(params)
            if params is not None
            else combo_name
        )

    rows: list[dict[str, object]] = []
    for combo_name, s1 in sorted(suite.stage1_reports.items()):
        null_ge_count = int((s1.null_distribution >= s1.original_metric).sum())
        p_denominator = int(s1.nreps) + 1
        p_numerator = null_ge_count + 1
        rows.append(
            {
                "feature_name": suite.feature_name,
                "feature_type": suite.feature_type,
                "objective_metric": objective_metric_label,
                "param_combo": combo_name,
                "param_combo_label": _label(combo_name),
                "observed_metric": s1.original_metric,
                "null_ge_count": null_ge_count,
                "p_value_numerator": p_numerator,
                "p_value_denominator": p_denominator,
                "p_value": s1.p_value,
                "passed": s1.passed,
                "alpha": s1.alpha,
                "n_reps": s1.nreps,
                "critical_value": s1.critical_value,
            }
        )
    return rows


def write_permutation_vector_shuffle_exports(
    suite: PermutationTestSuite,
    *,
    objective_metric_label: str,
    param_grid: list[dict[str, Any]] | None = None,
    visualization_parent_dir: Path | None = None,
    powerbi_parent_dir: Path | None = None,
) -> dict[str, Path]:
    """Write vector-shuffle permutation CSV next to other in-sample visualization CSVs."""
    out_dir = _resolve_in_sample_visualization_dir(
        visualization_parent_dir=visualization_parent_dir,
        powerbi_parent_dir=powerbi_parent_dir,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = permutation_vector_shuffle_records(
        suite, objective_metric_label, param_grid=param_grid
    )
    df = pd.DataFrame(rows)
    csv_path = _write_frame_csv(df, out_dir / "permutation_vector_shuffle")
    return {"permutation_vector_shuffle_csv": csv_path}
