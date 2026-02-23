"""
Shared helpers for bias node research notebooks (single_node_test, bias_node_research).

Provides: build_feature_metadata, get_binning_model, get_best_feature_from_permutation,
test_bias_node, generate_node_tearsheet.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import utils.core.helpers as helpers
from eda.feature_explorer import FeatureExplorer
from feature_extraction.feature_extractor import extract_features_for_bias_node
from feature_selection.base_models import (
    BinningModelBase,
    QuantileBinningModel,
    TwoBinBinningModel,
)
from utils.core.enums import Ticker, TimeFrame


def build_feature_metadata(features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build feature_metadata dict from a features DataFrame for use with FeatureExplorer.

    Iterates over feature columns (excluding 'ticker') and uses
    utils.core.helpers.parse_feature_column_name to populate metadata.
    """
    metadata: Dict[str, Any] = {"feature_metadata": {}}
    for feature_col in features_df.columns:
        if feature_col == "ticker":
            continue
        parsed = helpers.parse_feature_column_name(feature_col)
        metadata["feature_metadata"][feature_col] = {
            "module": parsed.get("module"),
            "parameters": parsed.get("params", {}),
            "timeframes": [parsed.get("tf")] if parsed.get("tf") else [],
            "base_name": parsed.get("module"),
            "full_name": feature_col,
        }
    return metadata


def get_binning_model(
    is_continuous: bool,
    n_bins: int = 10,
    selection_metric: str = "sortino",
    strategy: str = "long",
) -> Union[QuantileBinningModel, TwoBinBinningModel]:
    """Return QuantileBinningModel or TwoBinBinningModel based on is_continuous."""
    if is_continuous:
        return QuantileBinningModel(
            n_bins=n_bins,
            selection_metric=selection_metric,
            strategy=strategy,
        )
    return TwoBinBinningModel(
        selection_metric=selection_metric,
        strategy=strategy,
    )


def get_best_feature_from_permutation(perm_df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Get the best feature name from permutation test results (lowest p-value).
    """
    if perm_df is None or len(perm_df) == 0:
        return None
    pval_col = "pval" if "pval" in perm_df.columns else "p_value"
    best_row = perm_df.loc[perm_df[pval_col].idxmin()]
    return best_row["feature"]


_VALID_STRATEGIES = ("long", "short", "long-short")


def test_bias_node(
    bias_spec: Dict[str, Any],
    node_name: str,
    is_continuous: bool,
    tickers: Union[Ticker, List[Ticker]],
    start_date: datetime,
    end_date: datetime,
    reports_dir: Path,
    permutation_reps: int = 100,
    show_plots: bool = False,
    export: bool = True,
    use_cache: bool = False,
    target_col: str = "log_return",
    binning_model: Optional[BinningModelBase] = None,
    strategy: str = "long",
) -> Optional[Dict[str, Any]]:
    """
    Test a bias node and generate summary report.

    Extracts features (cached or streaming), builds metadata, creates FeatureExplorer,
    fits binning model, runs generate_summary_report, and returns results dict
    with features_df, targets_df, binning_model, extract_time attached.

    Parameters
    ----------
    binning_model : BinningModelBase, optional
        If provided, this model is used for binning. If None, one is chosen from
        get_binning_model(is_continuous) (QuantileBinningModel or TwoBinBinningModel).
    strategy : str, default='long'
        Signal strategy: 'long', 'short', or 'long-short'.
    target_col : str, default='log_return'
        Target column for extraction and report (e.g. 'log_return', 'log_return_atr', 'log_return_ewsd').

    Returns
    -------
    results : dict or None
        Analysis results including permutation_test, features_df, targets_df,
        binning_model, extract_time, strategy; or None if extraction/report failed.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {_VALID_STRATEGIES}, got {strategy!r}")
    print(f"\n{'='*70}")
    print(f"Testing: {node_name} [{strategy.upper()}] {'[CACHED]' if use_cache else '[STREAMING]'}")
    print(f"{'='*70}")

    print("\n[1/3] Extracting features...")
    try:
        t0 = time.time()
        features_df, targets_df = extract_features_for_bias_node(
            bias_spec=bias_spec,
            ticker=tickers,
            start=start_date,
            end=end_date,
            use_millisecond_offset=True,
            target_col=target_col,
            use_cache=use_cache,
        )
        extract_time = time.time() - t0
        feature_cols = [col for col in features_df.columns if col != "ticker"]
        print(f"  [OK] Features shape: {features_df.shape}")
        print(f"  [OK] Targets shape: {targets_df.shape}")
        print(f"  [OK] Feature columns: {len(feature_cols)}")
        print(f"  [OK] Extraction time: {extract_time:.2f}s")
    except Exception as e:
        print(f"  [FAIL] Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("\n[2/3] Building metadata...")
    metadata = build_feature_metadata(features_df)
    explorer = FeatureExplorer(
        features_df=features_df,
        targets_df=targets_df,
        metadata=metadata,
    )
    print(f"  [OK] FeatureExplorer created with {explorer.n_features} features, {explorer.n_samples} samples")

    if binning_model is None:
        binning_model = get_binning_model(is_continuous)

    print("\n[3/3] Generating summary report...")
    export_path = str(reports_dir / node_name.lower().replace(" ", "_").replace("%", "pct"))

    try:
        results = explorer.generate_summary_report(
            binning_model=binning_model,
            target_col=target_col,
            strategy=strategy,
            show_plots=show_plots,
            verbose=False,
            export_report=export,
            export_path=export_path if export else None,
            permutation_test_nreps=permutation_reps,
            include_ticker_plots=False,
        )
        print(f"  [OK] Report generated successfully")
        if export:
            print(f"  [OK] Report exported to: {export_path}")

        if results and "permutation_test" in results and results["permutation_test"] is not None:
            perm_df = results["permutation_test"]
            if len(perm_df) > 0:
                print("\n  Permutation Test Results:")
                for _, row in perm_df.iterrows():
                    pval = row.get("pval", row.get("p_value", "N/A"))
                    sig = "[SIG]" if pval <= 0.1 else "[---]"
                    feature = row.get("feature", "unknown")[:50]
                    print(f"    {sig} {feature}: p={pval:.4f}")

        results["features_df"] = features_df
        results["targets_df"] = targets_df
        results["binning_model"] = binning_model
        results["extract_time"] = extract_time
        results["strategy"] = strategy
        return results

    except Exception as e:
        print(f"  [FAIL] Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _tickers_from_features_df(features_df: pd.DataFrame) -> List[Ticker]:
    """Infer list of Ticker enums from features_df['ticker'] column."""
    raw = features_df["ticker"].unique()
    return [
        t if isinstance(t, Ticker) else Ticker[str(t).strip()]
        for t in raw
    ]


def generate_node_tearsheet(
    node_name: str,
    is_continuous: bool,
    start_date: datetime,
    end_date: datetime,
    reports_dir: Path,
    perm_df: Optional[pd.DataFrame] = None,
    features_df: Optional[pd.DataFrame] = None,
    targets_df: Optional[pd.DataFrame] = None,
    bias_spec: Optional[Dict[str, Any]] = None,
    strategy: str = "long-short",
    target_col: str = "log_return",
    tickers: Optional[Union[Ticker, List[Ticker]]] = None,
    binning_model: Optional[BinningModelBase] = None,
) -> Optional[str]:
    """
    Generate a QuantStats HTML tearsheet for a bias node.

    Uses pre-extracted features from feature analysis. Fits binning on full ensemble
    (or uses provided binning_model), then generates predictions for all tickers
    and computes strategy returns. Baseline is equal-weight buy-and-hold.

    Parameters
    ----------
    node_name : str
        Name of the node for display
    is_continuous : bool
        Whether feature is continuous (used only if binning_model is None)
    start_date, end_date : datetime
        Date range for loading candles
    reports_dir : Path
        Directory under which to create node subdir and save tearsheet
    perm_df, features_df, targets_df : DataFrame, optional
        Permutation results and pre-extracted features/targets from test_bias_node
    bias_spec : dict, optional
        Kept for API compatibility; not used
    strategy : str, default='long-short'
        Passed to binning_model.predict() only (which exposure to emit). Does not affect
        return calculation: returns are always position_fraction * instrument_return
        (position is already signed -1/0/1 from node or binning model).
    target_col : str, default='log_return'
        Target column in targets_df for fitting (e.g. 'log_return', 'log_return_atr', 'log_return_ewsd').
    tickers : Ticker or List[Ticker], optional
        Tickers to include in the tearsheet. If None, inferred from features_df['ticker'].
    binning_model : BinningModelBase, optional
        If provided (e.g. results['binning_model'] from test_bias_node), use this model.
        Required for rule-based nodes so RuleBasedBinningModel pass-through is used.

    Returns
    -------
    tearsheet_path : str or None
    """
    from ensemble.portfolio_tester import (
        calculate_baseline_returns,
        calculate_strategy_returns_from_positions,
    )
    from metrics.plotting.graphing.quantstats_reports import generate_tearsheet

    if bias_spec is not None:
        pass  # unused, API compatibility
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {_VALID_STRATEGIES}, got {strategy!r}")

    print(f"\n  Generating QuantStats tearsheet for {node_name} [{strategy}]...")

    try:
        if features_df is None or targets_df is None:
            print("  [FAIL] No pre-extracted features/targets provided")
            return None

        if tickers is None:
            tickers = _tickers_from_features_df(features_df)
        if isinstance(tickers, Ticker):
            tickers = [tickers]
        tickers = list(tickers)
        if not tickers:
            print("  [FAIL] No tickers in features or provided")
            return None

        best_feature = get_best_feature_from_permutation(perm_df)
        if best_feature is None:
            feature_cols = [c for c in features_df.columns if c != "ticker"]
            best_feature = feature_cols[0] if feature_cols else None

        if best_feature is None or best_feature not in features_df.columns:
            print("  [FAIL] Could not find feature column")
            return None
        if target_col not in targets_df.columns:
            print(f"  [FAIL] Target column {target_col!r} not in targets_df (columns: {list(targets_df.columns)})")
            return None

        print(f"    Using feature: {best_feature}, target: {target_col}, tickers: {[t.name if hasattr(t, 'name') else str(t) for t in tickers]}")

        feature_series = features_df[best_feature].copy()
        feature_series.name = best_feature
        target_series = targets_df[target_col].copy()
        if binning_model is None:
            raise ValueError(
                "binning_model is required for tearsheet generation. "
                "Pass results['binning_model'] from test_bias_node (or the same binning_model you passed to test_bias_node). "
                "Do not rely on get_binning_model(); use the user-provided model so rule-based nodes (e.g. turtle) use the correct pass-through."
            )
        if not getattr(binning_model, "is_fitted_", False):
            binning_model.fit(feature_series, target_series)
        if getattr(binning_model, "best_long_bin_", None) is not None:
            print(f"    Model fitted on ensemble - best_long_bin: {binning_model.best_long_bin_}")

        # Build positions for all tickers
        position_dfs = []
        for ticker in tickers:
            ticker_str = ticker.name if hasattr(ticker, "name") else str(ticker)
            mask = features_df["ticker"].astype(str) == ticker_str
            ticker_features = features_df.loc[mask, best_feature].copy()
            ticker_features.name = best_feature
            if len(ticker_features) == 0:
                continue
            pred_series = binning_model.predict(ticker_features, strategy=strategy)
            if not isinstance(pred_series, pd.Series):
                pred_series = pd.Series(pred_series, index=ticker_features.index)
            position_fraction = pred_series.values.astype(float)
            positions_datetime = pred_series.index
            if hasattr(positions_datetime, "tz") and positions_datetime.tz is not None:
                positions_datetime = positions_datetime.tz_localize(None)
            position_dfs.append(pd.DataFrame({
                "datetime": positions_datetime,
                "ticker": ticker,
                "position_fraction": position_fraction,
            }))
        if not position_dfs:
            print("  [FAIL] No positions generated for any ticker")
            return None
        positions_df = pd.concat(position_dfs, ignore_index=True)

        # Load candles for all tickers (use same offset convention as multi-ticker features)
        if len(tickers) == 1:
            candles_df = helpers.load_data(
                ticker=tickers[0],
                timeframe=TimeFrame.D,
                start=start_date,
                end=end_date,
            )
            candles_df = candles_df.reset_index()
            candles_df["ticker"] = tickers[0]
            candles_df["timeframe"] = TimeFrame.D
        else:
            candles_df = helpers.load_data_multi_ticker(
                tickers=tickers,
                timeframe=TimeFrame.D,
                start=start_date,
                end=end_date,
                use_millisecond_offset=True,
            )
        if "datetime" not in candles_df.columns and candles_df.index.name == "timestamp":
            candles_df = candles_df.reset_index()
            if "timestamp" in candles_df.columns and "datetime" not in candles_df.columns:
                candles_df["datetime"] = pd.to_datetime(candles_df["timestamp"], unit="s")
        candles_df["datetime"] = pd.to_datetime(candles_df["datetime"])

        # Returns = position_fraction * instrument_return (strategy param unused)
        strategy_returns = calculate_strategy_returns_from_positions(
            positions_df, candles_df
        )
        baseline_returns = calculate_baseline_returns(candles_df, equal_weight=True)

        n_active = (np.abs(positions_df["position_fraction"]) > 0).sum()
        n_total = len(positions_df)
        signal_freq = n_active / n_total if n_total else 0.0
        print(f"    Tickers: {len(tickers)} | Signal frequency: {signal_freq:.1%} ({int(n_active)} position-days)")
        print(f"    Strategy return: {strategy_returns.sum():.2%}")
        print(f"    Baseline return (equal-weight): {baseline_returns.sum():.2%}")

        safe_name = node_name.lower().replace(" ", "_").replace("%", "pct")
        node_dir = Path(reports_dir) / safe_name
        node_dir.mkdir(parents=True, exist_ok=True)
        tearsheet_path = str(node_dir / f"{safe_name}_tearsheet.html")
        baseline_label = "equal-weight" if len(tickers) > 1 else tickers[0].name

        generate_tearsheet(
            strategy_returns=strategy_returns,
            baseline_returns=baseline_returns,
            feature_name=f"{node_name} ({strategy}) vs Buy & Hold ({baseline_label})",
            output_file=tearsheet_path,
            mode="html",
        )
        print(f"  [OK] Tearsheet saved to: {tearsheet_path}")
        return tearsheet_path

    except Exception as e:
        print(f"  [FAIL] Tearsheet generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
