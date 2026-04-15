"""Shared tabular exports for research phases (Power BI CSV tables).

In-sample param dimensions are **long-only** (`param_combo_long.csv`): facts
(`param_sensitivity`, equity, permutation) carry ``param_combo_label`` only; join to
long param rows for slicers. Phase 0 binning writes long Parquet for combo metadata.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from feature_research.binning.transforms import build_param_combo_long_table
from feature_research.core_helpers import combo_key, normalize_datetime_index
from feature_research.in_sample.data_loader import param_combo_label, permutation_combo_display_name
from feature_selection.validation.stability_analysis import _param_combo_name
from utils.core.enums import TimeFrame
from utils.core.helpers import build_feature_column_name
from utils.evaluation.walkforward.selected_params_codec import decode_selected_params_list

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

# Stable CSV schema for Power BI (avoids missing columns after refresh when row dicts differ).
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
            "param_sensitivity_by_ticker_key must be unique per row (Power BI row key). "
            f"Duplicate values: {dup_vals[:25]}"
            + (" …" if len(dup_vals) > 25 else "")
        )

if TYPE_CHECKING:
    from feature_selection.validation.reports import PermutationTestSuite

POWERBI_SUBDIR_NAME = "powerbi"


def walkforward_power_bi_dir(
    output_root: Path,
    phase: Literal["validation", "oos"],
) -> Path:
    """Stable folder for validation / OOS Power BI CSVs.

    Path is ``output_root / powerbi / <phase>`` — no ``feature_type`` or ``module_name``,
    so Power BI data source paths stay fixed when you change bias node or module.
    """
    return Path(output_root) / POWERBI_SUBDIR_NAME / phase


def canonical_in_sample_power_bi_dir() -> Path:
    """Stable in-repo folder for Power BI imports (does not vary by preset or ``reports_dir``)."""
    return Path(__file__).resolve().parent / "in_sample" / "results" / POWERBI_SUBDIR_NAME


def _resolve_in_sample_power_bi_dir(*, powerbi_parent_dir: Path | None) -> Path:
    """Default: ``canonical_in_sample_power_bi_dir()``; tests pass ``powerbi_parent_dir`` → ``parent/powerbi``."""
    if powerbi_parent_dir is None:
        return canonical_in_sample_power_bi_dir()
    return powerbi_parent_dir / POWERBI_SUBDIR_NAME


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
    return pd.concat(pieces, axis=0, ignore_index=True) if pieces else pd.DataFrame(columns=cols)


def write_in_sample_equity_curve_powerbi_csv(
    combo_store: Mapping[str, tuple[pd.Series, pd.Series, str, pd.Series, dict[str, Any]]],
    *,
    powerbi_parent_dir: Path | None = None,
) -> Path:
    """Write a single long CSV of per-bar ``signal * target`` and cumulative sum for all combos.

    For multiple instruments, cumulative returns are computed **separately per ticker**
    (same idea as binning ``add_cumulative_return_columns``), not mixed on duplicate dates.

    Column names align with binning exports (``strategy_return``, ``cumulative_strategy_return``).
    Writes next to other in-sample Power BI tables under :func:`canonical_in_sample_power_bi_dir`.
    """
    out_dir = _resolve_in_sample_power_bi_dir(powerbi_parent_dir=powerbi_parent_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    return _write_frame_csv(combined, out_dir / "equity_curve")


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
    """Rolling Sharpe on per-bar returns, annualized like ``metric_helpers.compute_param_sensitivity_metric``."""
    x = pd.to_numeric(per_bar_returns, errors="coerce")
    win = max(2, int(window_bars))
    m = x.rolling(window=win, min_periods=win).mean()
    st = x.rolling(window=win, min_periods=win).std(ddof=1)
    ratio = m / st.replace(0.0, float("nan"))
    return (ratio * math.sqrt(float(bars_per_year))).rename("rolling_sharpe_annualized")


def enrich_walkforward_equity_df_with_rolling_sharpe(
    df: pd.DataFrame,
    timeframe: TimeFrame,
    *,
    window_bars: int = DEFAULT_ROLLING_SHARPE_WINDOW_BARS,
) -> pd.DataFrame:
    """Add rolling annualized Sharpe columns for Power BI (per ticker / fold / combo).

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
    rolled = ordered.groupby(keys, sort=False, group_keys=False)["strategy_return"].transform(
        lambda ser: _rolling_sharpe_annualized_series(ser, window_bars=win, bars_per_year=bars_py)
    )
    return ordered.assign(
        rolling_sharpe_annualized=rolled.to_numpy(dtype=float),
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


def _build_portfolio_positions_df(
    signal: pd.Series,
    ticker: pd.Series,
    train_candles: pd.DataFrame,
    timeframe: TimeFrame,
) -> pd.DataFrame:
    """Fit TFPortfolio on training candles and return a positions_df for the signal period.

    Uses TFPortfolio purely for IDM + equal instrument weights — no base models required.
    ``position_fraction = signal * instrument_weight * IDM``.
    """
    from ensemble.portfolio_impl.portfolio_returns import calculate_returns_from_candles
    from ensemble.portfolio_impl.tf_portfolio import TFPortfolio

    instrument_returns = calculate_returns_from_candles(train_candles)
    portfolio = TFPortfolio(ensembles=[], trading_timeframe=timeframe, idm_max=_IDM_MAX)
    portfolio.fit(instrument_returns)

    idx = normalize_datetime_index(signal.index)
    combined_forecasts = pd.DataFrame(
        {
            "ticker": ticker.astype(str).to_numpy(),
            "forecast_score": signal.to_numpy(dtype=float),
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
            positions_df = _build_portfolio_positions_df(sig, tkr, train_candles, timeframe)
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


def write_walkforward_equity_powerbi_csvs(
    *,
    combo_signal_target: Mapping[tuple[tuple[str, object], ...], pd.DataFrame],
    selection_summary_df: pd.DataFrame,
    module_name: str,
    timeframe: TimeFrame,
    holdout_start: object,
    holdout_end: object,
    extended_start: object,
    extended_end: object,
    output_powerbi_dir: Path,
    holdout_csv_stem: str,
    extended_csv_stem: str,
    portfolio_candles: pd.DataFrame | None = None,
    rolling_sharpe_window_bars: int = DEFAULT_ROLLING_SHARPE_WINDOW_BARS,
) -> dict[str, Path]:
    """Write two equity-curve CSVs for validation or OOS (Power BI, long format).

    Each CSV includes **rolling annualized Sharpe** on ``strategy_return`` (same
    annualization as in-sample param sensitivity): columns ``rolling_sharpe_annualized``
    and ``rolling_sharpe_window_bars`` (constant = ``rolling_sharpe_window_bars``).
    Rolling is computed **within** each ``(fold_id, param_combo_label, ticker)`` group
    after sorting by ``datetime`` (first ``window_bars - 1`` rows are NaN per group).

    When ``portfolio_candles`` is provided the equity curves are generated via
    the Portfolio class for a realistic price-return simulation:

    * ``TFPortfolio`` is fit on the training candles (dates before the holdout
      window) to derive IDM and equal instrument weights.
    * ``position_fraction = signal × instrument_weight × IDM``
    * ``calculate_strategy_returns_from_positions`` computes lookahead-free
      actual log P&L per ticker.

    When ``portfolio_candles`` is ``None`` the legacy ``signal × target`` path
    is used (EWSD-normalised returns).

    **Holdout** slice: validation phase → validation window only; OOS phase → OOS
    test window only.

    **Extended** slice: validation → train + validation; OOS → effective train
    start (train + val when ``validation_window`` is set) through OOS test end.

    All rows carry a ``fold_id`` column so Power BI can filter per fold.
    """
    output_powerbi_dir.mkdir(parents=True, exist_ok=True)
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
        "holdout": _write_frame_csv(holdout_df, output_powerbi_dir / holdout_csv_stem),
        "extended": _write_frame_csv(extended_df, output_powerbi_dir / extended_csv_stem),
    }


def write_param_sensitivity_powerbi_tables(
    sensitivity_rows: Sequence[Mapping[str, object]],
    label_param_pairs: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    sensitivity_by_ticker_rows: Sequence[Mapping[str, object]] | None = None,
    powerbi_parent_dir: Path | None = None,
) -> dict[str, Path]:
    """Write param-sensitivity metrics and long param dimension for Power BI (CSV only).

    Fact tables omit wide ``param_<key>`` columns; join ``param_combo_label`` to
    ``param_combo_long.csv`` for parameter slicers.

    Writes under :func:`canonical_in_sample_power_bi_dir` unless ``powerbi_parent_dir``
    is set (tests: files go to ``powerbi_parent_dir / "powerbi"``).

    Metrics (``sharpe`` annualized, ``t_stat``, ``sortino``) use
    ``feature_research.in_sample.metric_helpers.compute_param_sensitivity_metric`` on the
    full aligned sample of per-bar strategy returns ``signal * target`` (zeros on flat bars).

    When ``sensitivity_by_ticker_rows`` is non-empty, also writes ``param_sensitivity_by_ticker.csv``
    (same metrics computed **within** each instrument). Rows are one per
    ``(param_combo_label, ticker)``; ``param_combo_label`` repeats across tickers.
    In Power BI, mark **only** ``param_sensitivity_by_ticker_key`` as this table’s primary
    key / row identifier. Do **not** mark ``param_combo_label`` as a key—it repeats (one
    row per ticker). Relationships to ``param_combo_long`` / ``param_sensitivity`` use
    ``param_combo_label`` with *many* (by-ticker) → *one* (pooled / long) cardinality.

    Returns
    -------
    dict[str, Path]
        Paths keyed by artifact stem: ``param_sensitivity_csv``, ``param_sensitivity_by_ticker_csv``
        (only if by-ticker rows provided), ``param_combo_long_csv``.
    """
    out_dir = _resolve_in_sample_power_bi_dir(powerbi_parent_dir=powerbi_parent_dir)
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

    return [
        {
            "feature_name": suite.feature_name,
            "feature_type": suite.feature_type,
            "objective_metric": objective_metric_label,
            "param_combo": combo_name,
            "param_combo_label": _label(combo_name),
            "observed_metric": s1.original_metric,
            "p_value": s1.p_value,
            "passed": s1.passed,
            "alpha": s1.alpha,
            "n_reps": s1.nreps,
            "critical_value": s1.critical_value,
        }
        for combo_name, s1 in sorted(suite.stage1_reports.items())
    ]


def write_permutation_vector_shuffle_exports(
    suite: PermutationTestSuite,
    *,
    objective_metric_label: str,
    param_grid: list[dict[str, Any]] | None = None,
    powerbi_parent_dir: Path | None = None,
) -> dict[str, Path]:
    """Write vector-shuffle permutation CSV next to other in-sample Power BI tables."""
    out_dir = _resolve_in_sample_power_bi_dir(powerbi_parent_dir=powerbi_parent_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = permutation_vector_shuffle_records(
        suite, objective_metric_label, param_grid=param_grid
    )
    df = pd.DataFrame(rows)
    csv_path = _write_frame_csv(df, out_dir / "permutation_vector_shuffle")
    return {"permutation_vector_shuffle_csv": csv_path}
