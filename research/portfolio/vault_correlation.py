"""Vault vs research return correlation for OOS exports (cache-backed, DRY with data_loader)."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from ensemble.vault.discovery import iter_vault_feature_members
from ensemble.vault.feature_files import extract_feature_name
from features.extraction.feature_extractor import extract_features_for_bias_node
from research.feature._internal.core_helpers import combo_key, normalize_timeframe_from_bias_spec
from research.feature.in_sample.data_loader import param_combo_label, populate_cache_if_needed
from research.feature.research_table_exports import _normalize_ts, _slice_combo_panel
from lib.core.enums import Ticker, TimeFrame
from lib.core.helpers import build_feature_column_name
from lib.core.vault_paths import resolve_vault_prop
from research.evaluation.walkforward.selected_params_codec import decode_selected_params_list


def vault_bias_timeframe(bias_node_spec: Mapping[str, Any]) -> TimeFrame:
    """Primary timeframe from a vault ``bias_node_spec``."""
    return normalize_timeframe_from_bias_spec(bias_node_spec, fallback=TimeFrame.D)


def _tickers_from_vault_config(feature_config: Mapping[str, Any]) -> list[Ticker]:
    raw = feature_config.get("tickers", [])
    if not isinstance(raw, list):
        return []
    out: list[Ticker] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip().upper()
        if name in Ticker.__members__:
            out.append(Ticker[name])
    return out


def _ensemble_path_from_member_id(vault_member_id: str) -> str:
    parts = Path(vault_member_id).parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return vault_member_id


def candidate_returns_long_frame(
    combo_signal_target: Mapping[tuple[tuple[str, object], ...], pd.DataFrame],
    params: dict[str, object],
    *,
    extended_start: object,
    extended_end: object,
) -> pd.DataFrame:
    """Per-bar research returns ``signal * target`` on the extended window (long: datetime, ticker)."""
    key = combo_key(params)
    paired = combo_signal_target.get(key)
    if paired is None or paired.empty:
        return pd.DataFrame(columns=["datetime", "ticker", "research_return"])
    lo, hi = _normalize_ts(extended_start), _normalize_ts(extended_end)
    sig, tgt, tkr = _slice_combo_panel(paired, lo, hi)
    if sig.empty:
        return pd.DataFrame(columns=["datetime", "ticker", "research_return"])
    dt = pd.to_datetime(sig.index, utc=False)
    if getattr(dt, "tz", None) is not None:
        dt = dt.tz_localize(None)
    ret = (sig * tgt).astype(float)
    return pd.DataFrame(
        {
            "datetime": dt,
            "ticker": tkr.astype(str).to_numpy(),
            "research_return": ret.to_numpy(dtype=float),
        }
    )


def vault_returns_long_frame(
    feature_config: Mapping[str, Any],
    *,
    target_col: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Per-bar vault feature returns ``signal * target`` (aligned with research data_loader)."""
    bias = dict(feature_config["bias_node_spec"])
    tickers = _tickers_from_vault_config(feature_config)
    if not tickers:
        return pd.DataFrame(columns=["datetime", "ticker", "vault_return"])
    features_df, targets_df = extract_features_for_bias_node(
        bias_spec=bias,
        ticker=tickers,
        start=start,
        end=end,
        target_col=target_col,
        use_cache=True,
        populate_on_miss=True,
    )
    if features_df is None or features_df.empty or targets_df is None or targets_df.empty:
        return pd.DataFrame(columns=["datetime", "ticker", "vault_return"])
    feature_cols = [c for c in features_df.columns if c != "ticker"]
    if not feature_cols:
        return pd.DataFrame(columns=["datetime", "ticker", "vault_return"])
    feature_col = feature_cols[0]
    if target_col not in targets_df.columns:
        return pd.DataFrame(columns=["datetime", "ticker", "vault_return"])
    feat_s = features_df[feature_col].rename("feature")
    targ_s = targets_df[target_col].rename("target")
    if "ticker" in features_df.columns:
        tick_part = features_df["ticker"].astype(str).rename("ticker")
    else:
        first = tickers[0]
        tick_part = pd.Series(
            first.name if hasattr(first, "name") else str(first),
            index=feat_s.index,
            dtype=str,
            name="ticker",
        )
    aligned = pd.concat([feat_s, targ_s, tick_part], axis=1).dropna(how="any")
    if aligned.empty:
        return pd.DataFrame(columns=["datetime", "ticker", "vault_return"])
    vault_ret = (aligned["feature"] * aligned["target"]).astype(float)
    dt = pd.to_datetime(aligned.index, utc=False)
    if getattr(dt, "tz", None) is not None:
        dt = dt.tz_localize(None)
    return pd.DataFrame(
        {
            "datetime": dt,
            "ticker": aligned["ticker"].astype(str).to_numpy(),
            "vault_return": vault_ret.to_numpy(dtype=float),
        }
    )


def _drawdown_series(returns: pd.Series) -> pd.Series:
    """Underwater drawdown from per-bar returns: cum equity minus running max of cum equity."""
    r = returns.astype(float)
    equity = r.cumsum()
    return equity - equity.cummax()


def _corr_pearson_spearman(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    xv = x[mask].astype(float)
    yv = y[mask].astype(float)
    n = int(len(xv))
    if n < 2:
        return float("nan"), float("nan"), n
    if xv.std() == 0.0 or yv.std() == 0.0:
        return float("nan"), float("nan"), n
    return (
        float(xv.corr(yv)),
        float(xv.corr(yv, method="spearman")),
        n,
    )


def rows_for_vault_member_correlations(
    *,
    fold_id: int,
    window_kind: str,
    research_param_combo_label: str,
    research_feature_name_base: str,
    vault_member_id: str,
    vault_ensemble_path: str,
    vault_feature_name_base: str,
    candidate_df: pd.DataFrame,
    vault_df: pd.DataFrame,
) -> list[dict[str, object]]:
    """Emit long rows per ticker: return + drawdown correlations; omit skips entirely."""
    if candidate_df.empty or vault_df.empty:
        return []
    cand_tickers = set(candidate_df["ticker"].astype(str).unique())
    vault_tickers = set(vault_df["ticker"].astype(str).unique())
    common = sorted(cand_tickers & vault_tickers)
    rows: list[dict[str, object]] = []
    for tkr in common:
        research_feature_name = f"{research_feature_name_base}_{tkr}"
        vault_feature_name = f"{vault_feature_name_base}_{tkr}"
        base: dict[str, object] = {
            "fold_id": int(fold_id),
            "window_kind": window_kind,
            "research_param_combo_label": research_param_combo_label,
            "research_feature_name": research_feature_name,
            "vault_member_id": vault_member_id,
            "vault_ensemble_path": vault_ensemble_path,
            "vault_feature_name": vault_feature_name,
        }
        left = candidate_df[candidate_df["ticker"].astype(str) == tkr].sort_values("datetime")
        right = vault_df[vault_df["ticker"].astype(str) == tkr].sort_values("datetime")
        merged = pd.merge(
            left,
            right,
            on=["datetime", "ticker"],
            how="inner",
        )
        if merged.empty:
            continue
        rp = merged["research_return"].astype(float)
        vp = merged["vault_return"].astype(float)
        pearson, spearman, n_obs = _corr_pearson_spearman(rp, vp)
        dd_r = _drawdown_series(rp)
        dd_v = _drawdown_series(vp)
        dd_p, dd_s, n_dd = _corr_pearson_spearman(dd_r, dd_v)
        for metric_name, value, n in (
            ("pearson_return_corr", pearson, n_obs),
            ("spearman_return_corr", spearman, n_obs),
            ("pearson_drawdown_corr", dd_p, n_dd),
            ("spearman_drawdown_corr", dd_s, n_dd),
        ):
            if n < 2 or (isinstance(value, float) and value != value):
                continue
            rows.append(
                {
                    **base,
                    "ticker": tkr,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "n_obs": n,
                }
            )
    return rows


def ensure_cache_for_vault_feature(
    feature_config: Mapping[str, Any],
    *,
    start: datetime,
    end: datetime,
) -> None:
    """Bootstrap candles and bias artifacts for one vault feature (narrow ticker + spec)."""
    bias = dict(feature_config["bias_node_spec"])
    tickers = _tickers_from_vault_config(feature_config)
    if not tickers:
        return
    cfg = SimpleNamespace(
        tickers=tickers,
        bias_spec=bias,
        start=start,
        end=end,
    )
    populate_cache_if_needed(cfg, bias_spec=bias)


def build_vault_correlation_long_rows(
    *,
    combo_signal_target: Mapping[tuple[tuple[str, object], ...], pd.DataFrame],
    selection_summary_df: pd.DataFrame,
    module_name: str,
    research_timeframe: TimeFrame,
    research_eval_bias_spec: Mapping[str, Any],
    target_col: str,
    extended_start: datetime,
    extended_end: datetime,
    vault_root: Path,
) -> list[dict[str, object]]:
    """Cross research selections with all vault members (same timeframe only); cache per member."""
    research_tf = normalize_timeframe_from_bias_spec(
        dict(research_eval_bias_spec),
        fallback=research_timeframe,
    )
    window_kind = "extended_train_val_test"
    out: list[dict[str, object]] = []
    for _, summary_row in selection_summary_df.iterrows():
        fold_id = int(summary_row["fold_id"])
        selected_list = decode_selected_params_list(str(summary_row["selected_params_json"]))
        for params in selected_list:
            label = param_combo_label(params)
            research_feature_name_base = build_feature_column_name(
                module_name,
                "signal",
                research_timeframe,
                dict(params),
            )
            cand = candidate_returns_long_frame(
                combo_signal_target,
                dict(params),
                extended_start=extended_start,
                extended_end=extended_end,
            )
            for vault_member_id, feature_file, feature_config in iter_vault_feature_members(
                vault_root
            ):
                _ = feature_file
                v_tf = vault_bias_timeframe(dict(feature_config["bias_node_spec"]))
                ensemble_path = _ensemble_path_from_member_id(vault_member_id)
                vault_feature_name_base = extract_feature_name(
                    dict(feature_config),
                    Path(vault_member_id).name,
                )
                if v_tf != research_tf:
                    continue
                ensure_cache_for_vault_feature(
                    dict(feature_config),
                    start=extended_start,
                    end=extended_end,
                )
                vdf = vault_returns_long_frame(
                    dict(feature_config),
                    target_col=target_col,
                    start=extended_start,
                    end=extended_end,
                )
                if vdf.empty:
                    continue
                out.extend(
                    rows_for_vault_member_correlations(
                        fold_id=fold_id,
                        window_kind=window_kind,
                        research_param_combo_label=label,
                        research_feature_name_base=research_feature_name_base,
                        vault_member_id=vault_member_id,
                        vault_ensemble_path=ensemble_path,
                        vault_feature_name_base=vault_feature_name_base,
                        candidate_df=cand,
                        vault_df=vdf,
                    )
                )
    return out


def resolve_default_vault_root() -> Path:
    """Prop vault root (``TRADING_ALGO_VAULT_*`` / default ``<repo>/vault``)."""
    return resolve_vault_prop()
