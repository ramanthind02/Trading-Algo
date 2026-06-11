from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import inspect
from pathlib import Path
import warnings
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence, cast

import pandas as pd

if TYPE_CHECKING:
    from research.portfolio.pnl.pnl_engine import PnLEngine

from ensemble.weight_layer import WeightLayerConfig
from research.feature.config import BinningAnalysisConfig, FeatureType
from research.feature._internal.core_helpers import normalize_timeframe_from_bias_spec
from features.validation.objective_metrics import (
    resolve_objective_metric_name as resolve_objective_metric,
)
from research.evaluation.walkforward.config import WalkforwardResearchConfig
from research.evaluation.walkforward.selected_params_codec import (
    decode_selected_params_list,
    serialize_selected_params,
)
from research.evaluation.walkforward.walkforward_labels import canonical_param_label
from lib.core.enums import Ticker, TimeFrame


@dataclass(frozen=True)
class FoldScoreRow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    param_label: str
    raw_objective: float
    oos_objective: float
    smoothed_objective: float
    rank: int
    trade_frequency: float = 0.0
    selected_long_bin: int | None = None  # 0-based bin index from continuous model; None for rule-based


@dataclass(frozen=True)
class WalkforwardRunReport:
    folds_df: pd.DataFrame
    fold_scores_df: pd.DataFrame
    selection_summary_df: pd.DataFrame
    portfolio_results_df: pd.DataFrame
    fold_signal_metrics_df: pd.DataFrame = dataclasses.field(
        default_factory=lambda: pd.DataFrame(columns=["fold_id", "signal_name", "oos_sharpe"])
    )
    aggregate_oos_returns: pd.Series | None = None
    objective_metric_name: str = "objective"
    timeframe: TimeFrame = TimeFrame.D


class _ResearchConfigLike(Protocol):
    """Research config passed into portfolio simulation; walkforward is no longer used."""
    bias_spec: Mapping[str, object]
    binning_params: object
    tickers: list[Ticker]
    generate_ticker_tearsheets: bool


def _empty_portfolio_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["fold_id", "oos_portfolio_sharpe", "n_params_selected", "error"]
    )


def _resolve_feature_type(research_config: object) -> FeatureType:
    """Resolve FeatureType from research config; accept enum or string."""
    raw = getattr(research_config, "feature_type", FeatureType.CONTINUOUS)
    if isinstance(raw, FeatureType):
        return raw
    if isinstance(raw, str) and raw.strip():
        return FeatureType(raw.strip().lower())
    return FeatureType.CONTINUOUS


def _resolve_weight_layer_config(wf_cfg: object | None) -> WeightLayerConfig | None:
    if wf_cfg is None:
        return None

    configured = getattr(wf_cfg, "weight_layer_config", None)
    raw_algorithm = getattr(wf_cfg, "weight_layer_algorithm", None)
    algorithm = getattr(raw_algorithm, "value", raw_algorithm)
    algorithm_name = str(algorithm) if isinstance(algorithm, str) else None

    if configured is None:
        return WeightLayerConfig(weighting_method=algorithm_name) if algorithm_name else None
    if not isinstance(configured, WeightLayerConfig):
        return None
    if algorithm_name is None or configured.weighting_method == algorithm_name:
        return configured

    return WeightLayerConfig(
        weighting_method=algorithm_name,
        fdm_max=configured.fdm_max,
        hierarchy_spec=configured.hierarchy_spec,
        hierarchy_path=configured.hierarchy_path,
    )


def _resolve_member_prediction_mode(wf_cfg: object | None) -> str | None:
    if wf_cfg is None:
        return None
    configured = getattr(wf_cfg, "member_prediction_mode", None)
    if configured is None:
        return None
    raw = getattr(configured, "value", configured)
    return str(raw) if isinstance(raw, str) else None


def _sanitize_tearsheet_name(name: str, *, max_component_len: int = 100) -> str:
    """Replace characters unsafe for filenames; shorten long stems for Windows path limits.

    A single path component should stay well below 255 characters (``tearsheet.html`` adds
    length; QuantStats may also reject ``\\\\?\\`` paths with extreme lengths). Long names
    get truncated with a stable SHA-256 suffix for uniqueness.
    """
    base = (
        str(name)
        .replace(" ", "_")
        .replace("::", "_")
        .replace("/", "_")
    )
    base = re.sub(r'[<>:"|?*\\]', "_", base).strip("_") or "signal"
    if len(base) <= max_component_len:
        return base
    digest = hashlib.sha256(str(name).encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    keep = max(16, max_component_len - len(digest) - 1)
    return f"{base[:keep]}_{digest}"


def _normalize_ticker_label(value: object) -> str:
    if isinstance(value, Ticker):
        return value.name
    raw_name = getattr(value, "name", None)
    if isinstance(raw_name, str) and raw_name:
        return raw_name
    text = str(value)
    return text.replace("Ticker.", "")


def run_portfolio_simulation(
    candles_df: pd.DataFrame,
    target: pd.Series,
    fold_rows: Sequence[Mapping[str, object]],
    selection_summary_df: pd.DataFrame,
    research_config: object,
    feature_data_by_combo: Mapping[tuple[tuple[str, object], ...], pd.DataFrame] | None = None,
    tearsheets_dir: Path | None = None,
    pnl_engine: "PnLEngine | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    from research.evaluation.walkforward.portfolio_evaluator import (
        ensure_portfolio_candle_columns,
        evaluate_fold_portfolio,
    )

    if "datetime" in candles_df.columns:
        all_datetimes = pd.to_datetime(candles_df["datetime"], utc=False)
    else:
        all_datetimes = pd.DatetimeIndex(candles_df.index)
    if getattr(all_datetimes, "tz", None) is not None:
        all_datetimes = all_datetimes.tz_localize(None)

    typed_research_config = cast(_ResearchConfigLike, research_config)
    feature_type = _resolve_feature_type(research_config)
    bias_spec_pf = getattr(typed_research_config, "eval_bias_spec", None)
    if not isinstance(bias_spec_pf, Mapping):
        bias_spec_pf = typed_research_config.bias_spec
    trading_timeframes = cast(
        Sequence[object],
        bias_spec_pf.get("timeframes", [None]),
    )
    trading_timeframe = trading_timeframes[0] if trading_timeframes else None
    tearsheet_timeframe = normalize_timeframe_from_bias_spec(
        bias_spec_pf,
        fallback=TimeFrame.D,
    )
    objective_metric_name = getattr(
        typed_research_config, "objective_metric_name", "t_stat"
    )
    generate_ticker_tearsheets = bool(
        getattr(typed_research_config, "generate_ticker_tearsheets", False)
    )
    tearsheet_vol_target: float | None = None
    _raw_tearsheet_vol = getattr(
        typed_research_config, "tearsheet_target_annual_volatility", None
    )
    if _raw_tearsheet_vol is not None:
        try:
            _tv = float(_raw_tearsheet_vol)
        except (TypeError, ValueError):
            _tv = 0.0
        tearsheet_vol_target = _tv if _tv > 0.0 else None

    from research.feature.research_table_exports import return_kind_for_target
    _target_col = getattr(typed_research_config, "target_col", "log_return_ewsd")
    _instrument_return_kind = return_kind_for_target(_target_col)

    rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    collected_oos_returns: list[pd.Series] = []
    collected_per_ticker_returns: dict[str, list[pd.Series]] = {}

    _tearsheet_available = False
    if tearsheets_dir is not None:
        try:
            from ensemble.portfolio_impl.portfolio_tester import calculate_baseline_returns
            from analysis.plotting.graphing.quantstats_reports import generate_tearsheet
            _tearsheet_available = True
        except ImportError as e:
            warnings.warn(
                f"Tearsheet generation skipped (missing dependency): {e}",
                UserWarning,
                stacklevel=2,
            )

    n_folds = len(fold_rows)
    single_fold = n_folds == 1

    # Use phase default strategy so portfolio build matches research_context (e.g. long_short for rule-based).
    raw_binning_config = getattr(
        typed_research_config,
        "binning_params",
        BinningAnalysisConfig(),
    )
    strategy_override = getattr(
        typed_research_config,
        "strategy",
        getattr(raw_binning_config, "strategy", None),
    )
    try:
        if strategy_override is None:
            raise TypeError("No strategy override available")
        binning_config = replace(raw_binning_config, strategy=strategy_override)
    except (TypeError, AttributeError):
        # Fallback when binning config is not a dataclass (e.g. lightweight test mocks).
        binning_config = raw_binning_config

    for fold_row in fold_rows:
        fold_id = int(cast(int, fold_row["fold_id"]))
        summary = selection_summary_df.loc[selection_summary_df["fold_id"] == fold_id]
        if summary.empty:
            rows.append(
                {
                    "fold_id": fold_id,
                    "oos_portfolio_sharpe": float("nan"),
                    "n_params_selected": 0,
                    "error": "missing_selection_summary",
                }
            )
            continue

        selected_params = decode_selected_params_list(
            str(summary.iloc[0]["selected_params_json"])
        )
        if not selected_params:
            rows.append(
                {
                    "fold_id": fold_id,
                    "oos_portfolio_sharpe": float("nan"),
                    "n_params_selected": 0,
                    "error": "no_selected_params",
                }
            )
            continue

        if all(key in fold_row for key in ("train_start", "train_end", "test_start", "test_end")):
            train_start = pd.Timestamp(fold_row["train_start"])
            train_end = pd.Timestamp(fold_row["train_end"])
            test_start = pd.Timestamp(fold_row["test_start"])
            test_end = pd.Timestamp(fold_row["test_end"])
            train_mask = (all_datetimes >= train_start) & (all_datetimes <= train_end)
            test_mask = (all_datetimes >= test_start) & (all_datetimes <= test_end)
        else:
            train_mask = cast(pd.Series, fold_row["_train_mask"])
            test_mask = cast(pd.Series, fold_row["_test_mask"])
        train_candles = candles_df.loc[train_mask].copy()
        test_candles = candles_df.loc[test_mask].copy()
        train_candles = ensure_portfolio_candle_columns(train_candles, trading_timeframe)
        test_candles = ensure_portfolio_candle_columns(test_candles, trading_timeframe)

        try:
            # Optional nested config no longer used by research pipeline; support if present.
            wf_cfg = getattr(research_config, "walkforward", None)
            weight_layer_config = _resolve_weight_layer_config(wf_cfg)
            member_prediction_mode = _resolve_member_prediction_mode(wf_cfg)
            result = evaluate_fold_portfolio(
                train_candles=train_candles,
                test_candles=test_candles,
                selected_params=selected_params,
                target_series=target,
                binning_config=binning_config,
                tickers=typed_research_config.tickers,
                trading_timeframe=trading_timeframe,
                target_volatility=tearsheet_vol_target or 0.15,
                module_name=str(bias_spec_pf.get("module_name", "rsi")),
                objective_metric_name=objective_metric_name,
                weight_layer_config=weight_layer_config,
                member_prediction_mode=member_prediction_mode,
                feature_data_by_combo=feature_data_by_combo,
                feature_type=feature_type,
                instrument_return_kind=_instrument_return_kind,
                pnl_engine=pnl_engine,
            )

            rows.append(
                {
                    "fold_id": fold_id,
                    "oos_portfolio_sharpe": result.oos_portfolio_sharpe,
                    "n_params_selected": result.n_params_selected,
                    "error": "",
                }
            )
            if result.per_signal_oos_sharpe:
                for sig_name, sharpe in result.per_signal_oos_sharpe.items():
                    signal_rows.append({"fold_id": fold_id, "signal_name": sig_name, "oos_sharpe": sharpe})
            if not result.oos_portfolio_returns.empty:
                collected_oos_returns.append(result.oos_portfolio_returns)
            if generate_ticker_tearsheets and result.per_ticker_oos_returns:
                for ticker_label, ticker_returns in sorted(result.per_ticker_oos_returns.items()):
                    if ticker_returns.empty:
                        continue
                    collected_per_ticker_returns.setdefault(ticker_label, []).append(
                        ticker_returns
                    )

            if (
                tearsheets_dir is not None
                and _tearsheet_available
                and not result.oos_portfolio_returns.empty
            ):
                try:
                    if single_fold:
                        fold_tearsheet_dir = tearsheets_dir
                    else:
                        fold_tearsheet_dir = tearsheets_dir / f"fold_{fold_id}"
                        fold_tearsheet_dir.mkdir(parents=True, exist_ok=True)
                    train_result = None
                    if fold_id == 0:
                        train_baseline = calculate_baseline_returns(train_candles)
                        train_result = evaluate_fold_portfolio(
                            train_candles=train_candles,
                            test_candles=train_candles,
                            selected_params=selected_params,
                            target_series=target,
                            binning_config=binning_config,
                            tickers=typed_research_config.tickers,
                            trading_timeframe=trading_timeframe,
                            target_volatility=tearsheet_vol_target or 0.15,
                            module_name=str(bias_spec_pf.get("module_name", "rsi")),
                            objective_metric_name=objective_metric_name,
                            weight_layer_config=weight_layer_config,
                            member_prediction_mode=member_prediction_mode,
                            feature_data_by_combo=feature_data_by_combo,
                            feature_type=feature_type,
                            instrument_return_kind=_instrument_return_kind,
                        )
                        train_file = (
                            fold_tearsheet_dir / "train_ensemble_tearsheet.html"
                            if single_fold
                            else fold_tearsheet_dir / f"fold_{fold_id}_train_ensemble_tearsheet.html"
                        )
                        from research.evaluation.walkforward.tearsheet_returns import (
                            WF_CANDIDATE_TRAIN_RETURNS_CSV,
                            write_wf_tearsheet_returns_csv,
                        )

                        train_strategy = train_result.oos_portfolio_returns
                        if single_fold and tearsheets_dir is not None:
                            write_wf_tearsheet_returns_csv(
                                strategy_returns=train_strategy,
                                baseline_returns=train_baseline,
                                output_path=tearsheets_dir / WF_CANDIDATE_TRAIN_RETURNS_CSV,
                            )
                        elif not single_fold:
                            generate_tearsheet(
                                strategy_returns=train_strategy,
                                baseline_returns=train_baseline,
                                feature_name=f"Candidate strategy — fold {fold_id} train",
                                output_file=str(train_file),
                                mode="html",
                                timeframe=tearsheet_timeframe,
                                target_annual_volatility=tearsheet_vol_target,
                            )
                    fold_baseline = calculate_baseline_returns(test_candles)
                    validation_file = (
                        fold_tearsheet_dir / "validation_ensemble_tearsheet.html"
                        if single_fold
                        else fold_tearsheet_dir / f"fold_{fold_id}_ensemble_tearsheet.html"
                    )
                    val_strategy = result.oos_portfolio_returns
                    if single_fold and tearsheets_dir is not None:
                        from research.evaluation.walkforward.tearsheet_returns import (
                            WF_CANDIDATE_VALIDATION_RETURNS_CSV,
                            write_wf_tearsheet_returns_csv,
                        )

                        write_wf_tearsheet_returns_csv(
                            strategy_returns=val_strategy,
                            baseline_returns=fold_baseline,
                            output_path=tearsheets_dir / WF_CANDIDATE_VALIDATION_RETURNS_CSV,
                        )
                    elif not single_fold:
                        generate_tearsheet(
                            strategy_returns=val_strategy,
                            baseline_returns=fold_baseline,
                            feature_name=f"Candidate strategy — fold {fold_id} validation",
                            output_file=str(validation_file),
                            mode="html",
                            timeframe=tearsheet_timeframe,
                            target_annual_volatility=tearsheet_vol_target,
                        )
                    if train_result is not None and not single_fold:
                        full_candles = pd.concat([train_candles, test_candles], axis=0)
                        combined_returns = pd.concat(
                            [train_result.oos_portfolio_returns, result.oos_portfolio_returns],
                            axis=0,
                        ).sort_index()
                        if combined_returns.index.duplicated().any():
                            combined_returns = combined_returns[~combined_returns.index.duplicated(keep="first")]
                        combined_baseline = calculate_baseline_returns(full_candles)
                        strategy_for_combined = combined_returns.reindex(
                            combined_baseline.index, fill_value=0.0
                        )
                        combined_file = (
                            fold_tearsheet_dir
                            / f"fold_{fold_id}_train_and_validation_ensemble_tearsheet.html"
                        )
                        generate_tearsheet(
                            strategy_returns=strategy_for_combined,
                            baseline_returns=combined_baseline,
                            feature_name=f"Candidate strategy — fold {fold_id} train+validation",
                            output_file=str(combined_file),
                            mode="html",
                            timeframe=tearsheet_timeframe,
                            target_annual_volatility=tearsheet_vol_target,
                        )
                    skip_per_signal = single_fold and (
                        result.per_signal_oos_returns is None
                        or len(result.per_signal_oos_returns) <= 1
                    )
                    if result.per_signal_oos_returns and not skip_per_signal:
                        for sig_name, ret_ser in result.per_signal_oos_returns.items():
                            safe_name = _sanitize_tearsheet_name(sig_name)
                            signal_file = (
                                fold_tearsheet_dir / f"{safe_name}_tearsheet.html"
                                if single_fold
                                else fold_tearsheet_dir / f"fold_{fold_id}_{safe_name}_tearsheet.html"
                            )
                            generate_tearsheet(
                                strategy_returns=ret_ser,
                                baseline_returns=fold_baseline,
                                feature_name=sig_name if single_fold else f"Fold {fold_id} {sig_name}",
                                output_file=str(signal_file),
                                mode="html",
                                timeframe=tearsheet_timeframe,
                                target_annual_volatility=tearsheet_vol_target,
                            )
                    if generate_ticker_tearsheets and result.per_ticker_oos_returns:
                        ticker_labels = test_candles["ticker"].map(_normalize_ticker_label)
                        for ticker_label, ret_ser in sorted(result.per_ticker_oos_returns.items()):
                            ticker_test_candles = test_candles.loc[ticker_labels == ticker_label]
                            if ticker_test_candles.empty:
                                continue
                            ticker_baseline = calculate_baseline_returns(ticker_test_candles)
                            ticker_strategy = ret_ser.reindex(
                                ticker_baseline.index, fill_value=0.0
                            )
                            safe_ticker = _sanitize_tearsheet_name(ticker_label)
                            ticker_file = (
                                fold_tearsheet_dir / f"{safe_ticker}_tearsheet.html"
                                if single_fold
                                else fold_tearsheet_dir / f"fold_{fold_id}_{safe_ticker}_tearsheet.html"
                            )
                            generate_tearsheet(
                                strategy_returns=ticker_strategy,
                                baseline_returns=ticker_baseline,
                                feature_name=ticker_label if single_fold else f"Fold {fold_id} {ticker_label}",
                                output_file=str(ticker_file),
                                mode="html",
                                timeframe=tearsheet_timeframe,
                                target_annual_volatility=tearsheet_vol_target,
                            )
                except ValueError as te:
                    if "linear regression" in str(te).lower() or "all x values are identical" in str(te).lower():
                        warnings.warn(
                            f"Fold {fold_id}: Skipping per-fold tearsheet (constant returns): {te}",
                            UserWarning,
                            stacklevel=2,
                        )
                    else:
                        raise
        except Exception as exc:  # pragma: no cover - defensive catch
            rows.append(
                {
                    "fold_id": fold_id,
                    "oos_portfolio_sharpe": float("nan"),
                    "n_params_selected": len(selected_params),
                    "error": str(exc),
                }
            )

    aggregate_oos_returns: pd.Series | None = None
    if collected_oos_returns:
        walkforward_portfolio_returns = pd.concat(collected_oos_returns, axis=0).sort_index()
        aggregate_oos_returns = walkforward_portfolio_returns
        if (
            not single_fold
            and tearsheets_dir is not None
            and _tearsheet_available
        ):
            try:
                tearsheets_dir.mkdir(parents=True, exist_ok=True)
                baseline_returns = calculate_baseline_returns(candles_df)
                # Trim baseline to OOS window only (no IS data); loc[start:end] gives the
                # dense daily calendar for just the test folds period.
                oos_start = walkforward_portfolio_returns.index.min()
                oos_end = walkforward_portfolio_returns.index.max()
                baseline_returns = baseline_returns.loc[oos_start:oos_end]
                # Expand sparse strategy to the dense OOS calendar; flat/inactive days → 0.0.
                strategy_for_tearsheet = walkforward_portfolio_returns.reindex(
                    baseline_returns.index, fill_value=0.0
                )
                generate_tearsheet(
                    strategy_returns=strategy_for_tearsheet,
                    baseline_returns=baseline_returns,
                    feature_name="Walkforward Ensemble",
                    output_file=str(tearsheets_dir / "walkforward_ensemble_tearsheet.html"),
                    mode="html",
                    timeframe=tearsheet_timeframe,
                    target_annual_volatility=tearsheet_vol_target,
                )
                if (
                    generate_ticker_tearsheets
                    and collected_per_ticker_returns
                    and "ticker" in candles_df.columns
                ):
                    candle_ticker_labels = candles_df["ticker"].map(_normalize_ticker_label)
                    for ticker_label, per_fold_series in sorted(collected_per_ticker_returns.items()):
                        if not per_fold_series:
                            continue
                        ticker_returns = pd.concat(per_fold_series, axis=0).sort_index()
                        ticker_candles = candles_df.loc[candle_ticker_labels == ticker_label]
                        if ticker_returns.empty or ticker_candles.empty:
                            continue
                        try:
                            ticker_baseline_returns = calculate_baseline_returns(ticker_candles)
                            ticker_baseline_returns = ticker_baseline_returns.loc[oos_start:oos_end]
                            ticker_strategy = ticker_returns.reindex(
                                ticker_baseline_returns.index, fill_value=0.0
                            )
                            safe_ticker = _sanitize_tearsheet_name(ticker_label)
                            generate_tearsheet(
                                strategy_returns=ticker_strategy,
                                baseline_returns=ticker_baseline_returns,
                                feature_name=f"Walkforward {ticker_label}",
                                output_file=str(
                                    tearsheets_dir / f"walkforward_{safe_ticker}_tearsheet.html"
                                ),
                                mode="html",
                                timeframe=tearsheet_timeframe,
                                target_annual_volatility=tearsheet_vol_target,
                            )
                        except ValueError as te:
                            if "linear regression" in str(te).lower() or "all x values are identical" in str(te).lower():
                                warnings.warn(
                                    f"Skipping walkforward ticker tearsheet for {ticker_label} (constant returns): {te}",
                                    UserWarning,
                                    stacklevel=2,
                                )
                            else:
                                raise
            except ValueError as te:
                if "linear regression" in str(te).lower() or "all x values are identical" in str(te).lower():
                    warnings.warn(
                        f"Skipping walkforward aggregate tearsheet (constant returns): {te}",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    raise

    portfolio_results_df = pd.DataFrame(
        rows, columns=["fold_id", "oos_portfolio_sharpe", "n_params_selected", "error"]
    )
    fold_signal_metrics_df = (
        pd.DataFrame(signal_rows, columns=["fold_id", "signal_name", "oos_sharpe"])
        if signal_rows
        else pd.DataFrame(columns=["fold_id", "signal_name", "oos_sharpe"])
    )
    return portfolio_results_df, fold_signal_metrics_df, aggregate_oos_returns


def _score_one_param_for_fold(
    params: dict[str, object],
    param_columns: Sequence[str],
    fold_candles: pd.DataFrame,
    fold_target: pd.Series,
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    train_end_ts: pd.Timestamp,
    evaluate_param_combo: Callable[..., pd.Series | tuple[pd.Series, dict[str, object]]],
    objective_metric: Callable[[pd.Series], float],
    accepts_train_end: bool,
) -> dict[str, object]:
    """Score a single param combo for one fold. Module-level for joblib pickling."""
    if accepts_train_end:
        result = evaluate_param_combo(
            fold_candles, fold_target, params, train_end=train_end_ts
        )
    else:
        result = evaluate_param_combo(fold_candles, fold_target, params)
    if isinstance(result, tuple):
        scored_returns, meta = result
        selected_long_bin = meta.get("selected_long_bin")
    else:
        scored_returns = result
        selected_long_bin = None
    if isinstance(scored_returns.index, pd.DatetimeIndex):
        train_returns = scored_returns.loc[scored_returns.index.isin(train_index)]
        test_returns = scored_returns.loc[scored_returns.index.isin(test_index)]
        # Sanity check: non-empty evaluator output but empty slices suggests index mismatch.
        if not scored_returns.empty and train_returns.empty and test_returns.empty:
            warnings.warn(
                "Walkforward scoring: scored_returns non-empty but train/test slices empty; "
                "check index alignment between evaluator output and fold train_index/test_index.",
                UserWarning,
                stacklevel=2,
            )
    else:
        train_len = len(train_index)
        test_len = len(test_index)
        train_returns = scored_returns.head(train_len)
        test_returns = scored_returns.tail(test_len)
    active_train_returns = train_returns[train_returns != 0]
    active_test_returns = test_returns[test_returns != 0]
    trade_frequency = (
        float((train_returns != 0).mean()) if len(train_returns) > 0 else 0.0
    )
    if len(active_train_returns) == 0:
        train_objective = float("-inf")
        oos_objective = float("-inf")
    else:
        train_objective = float(objective_metric(active_train_returns))
        oos_objective = (
            float("-inf")
            if len(active_test_returns) == 0
            else float(objective_metric(active_test_returns))
        )
    def _hashable(v: object) -> object:
        return tuple(v) if isinstance(v, list) else v

    return {
        **{column: _hashable(params.get(column)) for column in param_columns},
        "param_label": canonical_param_label(params),
        "raw_objective": train_objective,
        "oos_objective": oos_objective,
        "trade_frequency": trade_frequency,
        "selected_long_bin": selected_long_bin,
    }


def _build_fold_rows(
    datetime_index: pd.DatetimeIndex,
    config: WalkforwardResearchConfig,
) -> list[dict[str, object]]:
    fold_rows: list[dict[str, object]] = []

    for fold_id in range(config.num_steps):
        offset = timedelta(days=fold_id * config.test_step)
        train_start_boundary = pd.Timestamp(config.train_start + offset)
        train_end_boundary = pd.Timestamp(config.train_end + offset)
        test_start_boundary = train_end_boundary
        test_end_boundary = pd.Timestamp(test_start_boundary + timedelta(days=config.test_step))

        train_mask = (datetime_index >= train_start_boundary) & (datetime_index < train_end_boundary)
        test_mask = (datetime_index >= test_start_boundary) & (datetime_index < test_end_boundary)

        train_index = datetime_index[train_mask]
        test_index = datetime_index[test_mask]
        train_samples = int(train_index.size)
        test_samples = int(test_index.size)

        if train_samples < config.min_fold_samples or test_samples < config.min_fold_samples:
            continue

        train_start = pd.Timestamp(train_index.min())
        train_end = pd.Timestamp(train_index.max())
        test_start = pd.Timestamp(test_index.min())
        test_end = pd.Timestamp(test_index.max())
        if train_end >= test_start:
            raise ValueError("No-lookahead violation: train_end must be less than test_start")

        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_samples": train_samples,
                "test_samples": test_samples,
                "_train_mask": train_mask,
                "_test_mask": test_mask,
            }
        )

    return fold_rows


def build_fold_rows_from_explicit_specs(
    datetime_index: pd.DatetimeIndex,
    specs: list[tuple[datetime, datetime, datetime, datetime]],
    min_fold_samples: int = 10,
) -> list[dict[str, object]]:
    """Build fold rows from explicit (train_start, train_end, test_start, test_end) per fold.

    Same dict shape as _build_fold_rows so the rest of the runner can consume them.
    Used by OOS to pass a single precomputed fold.
    """
    rows: list[dict[str, object]] = []
    for fold_id, (train_start_dt, train_end_dt, test_start_dt, test_end_dt) in enumerate(specs):
        train_start_b = pd.Timestamp(train_start_dt)
        train_end_b = pd.Timestamp(train_end_dt)
        test_start_b = pd.Timestamp(test_start_dt)
        test_end_b = pd.Timestamp(test_end_dt)
        train_mask = (datetime_index >= train_start_b) & (datetime_index < train_end_b)
        test_mask = (datetime_index >= test_start_b) & (datetime_index < test_end_b)
        train_index = datetime_index[train_mask]
        test_index = datetime_index[test_mask]
        train_samples = int(train_index.size)
        test_samples = int(test_index.size)
        if train_samples < min_fold_samples or test_samples < min_fold_samples:
            continue
        train_start = pd.Timestamp(train_index.min())
        train_end = pd.Timestamp(train_index.max())
        test_start = pd.Timestamp(test_index.min())
        test_end = pd.Timestamp(test_index.max())
        if train_end >= test_start:
            raise ValueError("No-lookahead violation: train_end must be less than test_start")
        rows.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_samples": train_samples,
                "test_samples": test_samples,
                "_train_mask": train_mask,
                "_test_mask": test_mask,
            }
        )
    return rows


def _build_fold_scores(
    fold_row: dict[str, object],
    candles_df: pd.DataFrame,
    target: pd.Series,
    param_grid: list[dict[str, object]],
    evaluate_param_combo: Callable[..., pd.Series | tuple[pd.Series, dict[str, object]]],
    objective_metric: Callable[[pd.Series], float],
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train_mask = cast(pd.Series, fold_row["_train_mask"])
    test_mask = cast(pd.Series, fold_row["_test_mask"])

    combined_mask = train_mask | test_mask
    fold_candles = candles_df.loc[combined_mask]
    fold_target = target.loc[combined_mask]
    train_index = candles_df.index[train_mask]
    test_index = candles_df.index[test_mask]

    evaluator_signature = inspect.signature(evaluate_param_combo)
    accepts_train_end = "train_end" in evaluator_signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in evaluator_signature.parameters.values()
    )

    def _call_evaluator(
        fold_data: pd.DataFrame,
        fold_targets: pd.Series,
        params: dict[str, object],
        train_end: pd.Timestamp,
    ) -> pd.Series | tuple[pd.Series, dict[str, object]]:
        if accepts_train_end:
            return evaluate_param_combo(
                fold_data,
                fold_targets,
                params,
                train_end=train_end,
            )
        return evaluate_param_combo(fold_data, fold_targets, params)

    def score_param(params: dict[str, object]) -> tuple[float, float, float, object]:
        result = _call_evaluator(
            fold_data=fold_candles,
            fold_targets=fold_target,
            params=params,
            train_end=cast(pd.Timestamp, fold_row["train_end"]),
        )
        # Continuous evaluator returns (series, {"selected_long_bin": ...}); rule-based returns series only.
        if isinstance(result, tuple):
            scored_returns, meta = result
            selected_long_bin = meta.get("selected_long_bin")
        else:
            scored_returns = result
            selected_long_bin = None

        if isinstance(scored_returns.index, pd.DatetimeIndex):
            train_returns = scored_returns.loc[scored_returns.index.isin(train_index)]
            test_returns = scored_returns.loc[scored_returns.index.isin(test_index)]
            # Sanity check: non-empty evaluator output but empty slices suggests index mismatch.
            if not scored_returns.empty and train_returns.empty and test_returns.empty:
                warnings.warn(
                    "Walkforward scoring: scored_returns non-empty but train/test slices empty; "
                    "check index alignment between evaluator output and fold train_index/test_index.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            train_len = len(train_index)
            test_len = len(test_index)
            train_returns = scored_returns.head(train_len)
            test_returns = scored_returns.tail(test_len)

        active_train_returns = train_returns[train_returns != 0]
        active_test_returns = test_returns[test_returns != 0]
        trade_frequency = (
            float((train_returns != 0).mean())
            if len(train_returns) > 0
            else 0.0
        )
        # Sentinel when no long bin (fit failed or empty signal) so combo is excluded by positive filter.
        if len(active_train_returns) == 0:
            train_objective = float("-inf")
            oos_objective = float("-inf")
        else:
            train_objective = float(objective_metric(active_train_returns))
            oos_objective = (
                float("-inf")
                if len(active_test_returns) == 0
                else float(objective_metric(active_test_returns))
            )

        return (train_objective, oos_objective, trade_frequency, selected_long_bin)

    def _hashable_param(v: object) -> object:
        return tuple(v) if isinstance(v, list) else v

    param_columns = sorted({key for params in param_grid for key in params})
    train_end_ts = cast(pd.Timestamp, fold_row["train_end"])
    if n_jobs == 1:
        raw_rows = []
        for params in param_grid:
            train_objective, oos_objective, trade_frequency, selected_long_bin = score_param(params)
            raw_rows.append(
                {
                    **{column: _hashable_param(params.get(column)) for column in param_columns},
                    "param_label": canonical_param_label(params),
                    "raw_objective": train_objective,
                    "oos_objective": oos_objective,
                    "trade_frequency": trade_frequency,
                    "selected_long_bin": selected_long_bin,
                }
            )
    else:
        from multiprocessing import cpu_count
        from joblib import Parallel, delayed
        n_jobs_actual = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
        raw_rows = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_score_one_param_for_fold)(
                params,
                param_columns,
                fold_candles,
                fold_target,
                train_index,
                test_index,
                train_end_ts,
                evaluate_param_combo,
                objective_metric,
                accepts_train_end,
            )
            for params in param_grid
        )

    raw_df = pd.DataFrame(raw_rows)
    smoothed_df = raw_df.copy()
    smoothed_df["smoothed_objective"] = raw_df["raw_objective"].values

    ranked_df = (
        smoothed_df.sort_values(
            by=["smoothed_objective", "raw_objective", "param_label"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    ranked_df["rank"] = ranked_df.index + 1

    scored_rows = [
        FoldScoreRow(
            fold_id=int(cast(int, fold_row["fold_id"])),
            train_start=cast(pd.Timestamp, fold_row["train_start"]),
            train_end=cast(pd.Timestamp, fold_row["train_end"]),
            test_start=cast(pd.Timestamp, fold_row["test_start"]),
            test_end=cast(pd.Timestamp, fold_row["test_end"]),
            param_label=str(row.param_label),
            raw_objective=float(row.raw_objective),
            oos_objective=float(row.oos_objective),
            smoothed_objective=float(row.smoothed_objective),
            rank=int(row.rank),
            trade_frequency=float(getattr(row, "trade_frequency", 0.0)),
            selected_long_bin=getattr(row, "selected_long_bin", None),
        )
        for row in ranked_df.itertuples(index=False)
    ]

    selected_feature = scored_rows[0].param_label
    selected_row = scored_rows[0]
    fold_scores_df = pd.DataFrame(
        [
            {
                "fold_id": row.fold_id,
                "param_label": row.param_label,
                "raw_objective": row.raw_objective,
                "oos_objective": float(row.oos_objective),
                "smoothed_objective": row.smoothed_objective,
                "rank": row.rank,
                "selected_feature": row.param_label == selected_feature,
                "trade_frequency": row.trade_frequency,
                "selected_long_bin": row.selected_long_bin,
            }
            for row in scored_rows
        ]
    )

    win_row = ranked_df.iloc[0]
    winning_params = {col: win_row[col] for col in param_columns}
    selected_params_json = serialize_selected_params(winning_params)

    summary_row = {
        "fold_id": int(cast(int, fold_row["fold_id"])),
        "selected_feature": selected_feature,
        "selected_raw_objective": selected_row.raw_objective,
        "selected_smoothed_objective": selected_row.smoothed_objective,
        "selected_params_json": selected_params_json,
    }
    return fold_scores_df, summary_row


def run_walkforward_research(
    candles_df: pd.DataFrame,
    target: pd.Series,
    feature_type: str,
    module_name: str,
    config: WalkforwardResearchConfig,
    param_grid: list[dict[str, object]],
    evaluate_param_combo: Callable[..., pd.Series],
    research_config: object | None = None,
    portfolio_candles_df: pd.DataFrame | None = None,
    feature_data_by_combo: Mapping[tuple[tuple[str, object], ...], pd.DataFrame] | None = None,
    output_dir: Path | None = None,
    fold_rows_override: list[dict[str, object]] | None = None,
    phase_label: str | None = None,
    pnl_engine: "PnLEngine | None" = None,
) -> WalkforwardRunReport:
    if not isinstance(feature_type, str) or not feature_type.strip():
        raise ValueError("feature_type must be a non-empty string")
    if not isinstance(module_name, str) or not module_name.strip():
        raise ValueError("module_name must be a non-empty string")

    if not isinstance(candles_df.index, pd.DatetimeIndex):
        raise ValueError("candles_df must have a DatetimeIndex")
    if not isinstance(target.index, pd.DatetimeIndex):
        raise ValueError("target must have a DatetimeIndex")
    if not candles_df.index.equals(target.index):
        raise ValueError("candles_df and target must share the same index")
    if not param_grid:
        raise ValueError("param_grid must contain at least one parameter combination")

    objective_metric = resolve_objective_metric(config.objective_metric_name)
    fold_rows = (
        fold_rows_override
        if fold_rows_override is not None
        else _build_fold_rows(candles_df.index, config)
    )

    folds_df = pd.DataFrame(
        [
            {
                "fold_id": row["fold_id"],
                "train_start": row["train_start"],
                "train_end": row["train_end"],
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "train_samples": row["train_samples"],
                "test_samples": row["test_samples"],
            }
            for row in fold_rows
        ],
        columns=[
            "fold_id",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "train_samples",
            "test_samples",
        ],
    )

    fold_score_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []

    n_folds = len(fold_rows)
    n_params = len(param_grid)
    run_label = (phase_label.strip() if phase_label and phase_label.strip() else None) or "walkforward"
    print(f"Running {run_label}: {n_folds} folds, {n_params} param combos.", flush=True)
    for fold_idx, fold_row in enumerate(fold_rows):
        print(f"  Fold {fold_idx + 1}/{n_folds} ({n_params} params)...", flush=True)
        n_jobs = getattr(config, "n_jobs", 1)
        fold_scores_df, summary_row = _build_fold_scores(
            fold_row=fold_row,
            candles_df=candles_df,
            target=target,
            param_grid=param_grid,
            evaluate_param_combo=evaluate_param_combo,
            objective_metric=objective_metric,
            n_jobs=n_jobs,
        )
        fold_score_parts.append(fold_scores_df)
        selection_rows.append(summary_row)
        print(f"  Fold {fold_idx + 1}/{n_folds} done.", flush=True)

    fold_scores_df = (
        pd.concat(fold_score_parts, ignore_index=True)
        if fold_score_parts
        else pd.DataFrame(
            columns=[
                "fold_id",
                "param_label",
                "raw_objective",
                "oos_objective",
                "smoothed_objective",
                "rank",
                "selected_feature",
                "trade_frequency",
                "selected_long_bin",
            ]
        )
    )
    selection_summary_df = pd.DataFrame(
        selection_rows,
        columns=[
            "fold_id",
            "selected_feature",
            "selected_raw_objective",
            "selected_smoothed_objective",
            "selected_params_json",
        ],
    )

    aggregate_oos_returns: pd.Series | None = None
    if research_config is None:
        portfolio_results_df = _empty_portfolio_results_df()
        fold_signal_metrics_df = pd.DataFrame(columns=["fold_id", "signal_name", "oos_sharpe"])
    else:
        tearsheets_dir = (output_dir / "tearsheets") if output_dir is not None else None
        if tearsheets_dir is not None:
            tearsheets_dir.mkdir(parents=True, exist_ok=True)
        try:
            portfolio_results_df, fold_signal_metrics_df, aggregate_oos_returns = run_portfolio_simulation(
                candles_df=portfolio_candles_df if portfolio_candles_df is not None else candles_df,
                target=target,
                fold_rows=fold_rows,
                selection_summary_df=selection_summary_df,
                research_config=research_config,
                feature_data_by_combo=feature_data_by_combo,
                tearsheets_dir=tearsheets_dir,
                pnl_engine=pnl_engine,
            )
        except Exception as exc:  # pragma: no cover - defensive catch
            warnings.warn(
                f"Portfolio simulation (and tearsheets) failed: {exc}",
                UserWarning,
                stacklevel=2,
            )
            portfolio_results_df = pd.DataFrame(
                [
                    {
                        "fold_id": int(cast(int, row["fold_id"])),
                        "oos_portfolio_sharpe": float("nan"),
                        "n_params_selected": 0,
                        "error": str(exc),
                    }
                    for row in fold_rows
                ],
                columns=["fold_id", "oos_portfolio_sharpe", "n_params_selected", "error"],
            )
            fold_signal_metrics_df = pd.DataFrame(columns=["fold_id", "signal_name", "oos_sharpe"])

    report_timeframe = TimeFrame.D
    if research_config is not None:
        bias_spec_raw = getattr(research_config, "bias_spec", {})
        if isinstance(bias_spec_raw, Mapping):
            report_timeframe = normalize_timeframe_from_bias_spec(
                bias_spec_raw,
                fallback=TimeFrame.D,
            )

    return WalkforwardRunReport(
        folds_df=folds_df,
        fold_scores_df=fold_scores_df,
        selection_summary_df=selection_summary_df,
        portfolio_results_df=portfolio_results_df,
        fold_signal_metrics_df=fold_signal_metrics_df,
        aggregate_oos_returns=aggregate_oos_returns,
        objective_metric_name=config.objective_metric_name,
        timeframe=report_timeframe,
    )
