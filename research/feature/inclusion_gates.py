"""Portfolio inclusion: validation forecast correlation vs peers and with/without-candidate tearsheets."""
from __future__ import annotations

import json
import logging
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ensemble.portfolio import Portfolio, PortfolioCacheQuery
from ensemble.portfolio_impl.portfolio_cache import (
    _query_candles_from_cache,
    _query_volatility_from_cache,
)
from ensemble.vault_manager import load_ensemble_from_vault
from research.feature.config import (
    PortfolioInclusionConfig,
    ResearchConfig,
    resolve_portfolio_benchmark_ticker,
)
from features.validation.objective_metrics import (
    metric_calmar,
    metric_sharpe,
    metric_sortino,
)
from metrics.plotting.graphing.quantstats_reports import generate_tearsheet
from research.portfolio.config import (
    PortfolioResearchConfig,
    scoped_tickers_for_ensemble_dirs,
)
from research.portfolio.pipelines.portfolio_test import (
    PhaseResult,
    _enable_cache,
    run_portfolio_research_cache_preflight,
    run_single_phase_for_prop_firm,
)
from lib.core.enums import TimeFrame

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DAILY_ANN = 252.0
_INCLUSION_WEIGHT_LAYER_METHOD = "ledoit_wolf_min_corr"
_INCLUSION_WEIGHT_LAYER_FDM_DEFAULT = 2.0


def _normalize_portfolio_config_for_inclusion(
    config: PortfolioResearchConfig,
) -> PortfolioResearchConfig:
    """Force Ledoit–Wolf / min-corr weighting so inclusion runs are comparable and stable.

    Ignores ``weight_layer_method`` and hierarchy-related kwargs from
    ``portfolio_research.config.load_config()``; preserves ``fdm_max`` when present.
    """
    raw_kw = dict(config.weight_layer_kwargs)
    fdm_raw = raw_kw.get("fdm_max", _INCLUSION_WEIGHT_LAYER_FDM_DEFAULT)
    try:
        fdm = float(fdm_raw)
    except (TypeError, ValueError):
        fdm = _INCLUSION_WEIGHT_LAYER_FDM_DEFAULT
    if fdm <= 0.0:
        fdm = _INCLUSION_WEIGHT_LAYER_FDM_DEFAULT
    return replace(
        config,
        weight_layer_method=_INCLUSION_WEIGHT_LAYER_METHOD,
        weight_layer_kwargs={"fdm_max": fdm},
    )


def _inject_weight_hierarchy_group_into_features(
    ensemble_dir: str,
    weight_hierarchy_group: str,
) -> None:
    """Set ``weight_hierarchy_group`` on each feature JSON (required for ``hierarchy_equal``).

    Ephemeral dirs are not under the repo ``vault/`` tree, so group cannot be inferred from path
    relative to project vault; the field must be present in the JSON.
    """
    from ensemble.vault.feature_files import validate_signed_signal_feature_config
    from lib.cache.runtime.cache_paths import win32_extended_path

    features_dir = Path(ensemble_dir) / "features"
    for fp in sorted(features_dir.glob("*.json")):
        with open(win32_extended_path(fp), "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        cfg["weight_hierarchy_group"] = weight_hierarchy_group
        validate_signed_signal_feature_config(cfg, feature_file=fp)
        with open(win32_extended_path(fp), "w", encoding="utf-8") as handle:
            json.dump(cfg, handle, indent=2)


def materialize_inclusion_candidate_from_eval_bias_spec(
    research: ResearchConfig,
    portfolio_config: PortfolioResearchConfig,
    *,
    ephemeral_ensemble_name: str = "inclusion_candidate",
    weight_hierarchy_group: str = "momentum",
    temp_parent: Path | None = None,
) -> tuple[str, Path]:
    """Create a one-feature ensemble directory from :attr:`ResearchConfig.eval_bias_spec` (frozen).

    Writes outside the repo ``vault/`` tree (under a temp directory). Returns ``(ensemble_dir,
    tmp_root)``; caller must ``shutil.rmtree(tmp_root)`` when finished.

    Tickers default from ``portfolio_config``; when ``research.portfolio_inclusion.candidate_tickers``
    is set, only those symbols are written on the candidate feature (e.g. GC-only IBS in a
    multi-instrument prop book). ``weight_hierarchy_group`` is written into the feature JSON and used when creating
    the ensemble directory so layout matches grouped vault ensembles (path layout only; inclusion
    still combines streams with ``ledoit_wolf_min_corr``).
    """
    from ensemble.vault.manager import create_ensemble_directory
    from research.feature.save_feature_to_vault import _normalize_bias_spec_for_model
    from features.models.feature_base_model import BaseModel

    bias_spec = _normalize_bias_spec_for_model(dict(research.eval_bias_spec))
    inclusion = research.portfolio_inclusion
    candidate_tickers = inclusion.candidate_tickers
    tickers = (
        list(candidate_tickers)
        if candidate_tickers is not None
        else list(research.tickers)
        if research.tickers
        else list(portfolio_config.tickers)
    )
    direction = research.strategy

    first_tf = bias_spec["timeframes"][0]
    timeframe = TimeFrame[first_tf] if isinstance(first_tf, str) else first_tf

    tmp_root = Path(
        tempfile.mkdtemp(
            prefix="inclusion_eval_bias_",
            dir=str(temp_parent) if temp_parent is not None else None,
        )
    )
    vault_root = tmp_root / "vault_root"
    ensemble_dir = create_ensemble_directory(
        timeframe,
        ephemeral_ensemble_name,
        direction,
        tickers=tickers,
        vault_root=str(vault_root),
        weight_hierarchy_group=weight_hierarchy_group,
    )
    feature_config: dict[str, Any] = {
        "bias_node_spec": bias_spec,
        "strategy": direction,
    }
    model = BaseModel(feature_config, tickers=tickers)
    model.save_to_vault(ensemble_dir, tickers=tickers)
    _inject_weight_hierarchy_group_into_features(ensemble_dir, weight_hierarchy_group)
    return ensemble_dir, tmp_root


def infer_candidate_key(repo_relative_path: str) -> str:
    """Default dict key for ``ensemble_dirs`` from a repo-relative vault path."""
    return Path(repo_relative_path).name


def _portfolio_cache_query(
    config: PortfolioResearchConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframes: tuple[TimeFrame, ...],
) -> PortfolioCacheQuery:
    return PortfolioCacheQuery(
        tickers=tuple(t.name if hasattr(t, "name") else str(t) for t in config.tickers),
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        timeframes=timeframes,
    )


def _rebuild_weight_layer_kwargs(
    config: PortfolioResearchConfig,
    ensemble_dirs: Mapping[str, str],
) -> dict[str, Any]:
    from ensemble.vault.hierarchy_spec import build_asset_first_hierarchy_spec_for_ensemble_dirs

    wl_kw = dict(config.weight_layer_kwargs)
    if config.weight_layer_method == "hierarchy_equal":
        wl_kw["hierarchy_spec"] = build_asset_first_hierarchy_spec_for_ensemble_dirs(
            _REPO_ROOT,
            ensemble_dirs,
            strict_group=True,
            portfolio_ticker_names=frozenset(t.name for t in config.tickers),
        )
    return wl_kw


def config_with_ensemble_dirs(
    config: PortfolioResearchConfig,
    ensemble_dirs: Mapping[str, str],
) -> PortfolioResearchConfig:
    """Clone config with new ``ensemble_dirs`` and rebuilt hierarchy when needed."""
    scoped_tickers = scoped_tickers_for_ensemble_dirs(config.tickers, ensemble_dirs)
    scoped_config = replace(
        config,
        ensemble_dirs=dict(ensemble_dirs),
        tickers=scoped_tickers,
        benchmark_ticker=resolve_portfolio_benchmark_ticker(
            baseline_mode=config.baseline_mode,
            portfolio_tickers=scoped_tickers,
            benchmark_ticker=config.benchmark_ticker,
        ),
    )
    return replace(
        scoped_config,
        weight_layer_kwargs=_rebuild_weight_layer_kwargs(scoped_config, ensemble_dirs),
    )


def _load_named_ensembles(
    ensemble_dirs: Mapping[str, str],
    config: PortfolioResearchConfig,
) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for name, path in ensemble_dirs.items():
        ens = load_ensemble_from_vault(
            path,
            refit=True,
            target_volatility=config.target_volatility,
            exclude_feature_stems_by_ensemble=getattr(
                config, "exclude_feature_stems_by_ensemble", None
            ),
        )
        out.append((name, _enable_cache(ens, config.use_cache)))
    return out


def _ensemble_timeframe(ensemble: object) -> TimeFrame:
    base_tf = getattr(ensemble, "base_tf", None)
    return base_tf if isinstance(base_tf, TimeFrame) else TimeFrame.D


def _norm_ticker_sym(x: object) -> str:
    return str(x).strip().upper()


def _intersect_tradable_tickers(
    cand_all: pd.DataFrame,
    peer_all: pd.DataFrame,
    portfolio_ticker_names: frozenset[str],
) -> list[str]:
    """Tickers that appear in both forecast frames and in the portfolio (fair comparison universe)."""
    port = {_norm_ticker_sym(t) for t in portfolio_ticker_names}
    c = {_norm_ticker_sym(x) for x in cand_all["ticker"].unique()}
    p = {_norm_ticker_sym(x) for x in peer_all["ticker"].unique()}
    return sorted((c & p) & port)


def _corr_forecast_streams(
    sub_c: pd.DataFrame,
    sub_p: pd.DataFrame,
) -> tuple[float, float, int]:
    """Pearson r, Spearman r, and inner-join row count for two single-ticker forecast slices."""
    sub_c = sub_c[["datetime", "forecast_score"]].copy()
    sub_p = sub_p[["datetime", "forecast_score"]].copy()
    sub_c["datetime"] = pd.to_datetime(sub_c["datetime"]).dt.floor("s")
    sub_p["datetime"] = pd.to_datetime(sub_p["datetime"]).dt.floor("s")
    merged = sub_c.merge(sub_p, on="datetime", suffixes=("_c", "_p"), how="inner")
    n = int(len(merged))
    if n < 3:
        return float("nan"), float("nan"), n
    rp = merged["forecast_score_c"].corr(merged["forecast_score_p"], method="pearson")
    rs = merged["forecast_score_c"].corr(merged["forecast_score_p"], method="spearman")
    return (
        float(rp) if rp == rp else float("nan"),
        float(rs) if rs == rs else float("nan"),
        n,
    )


def compute_peer_forecast_correlations(
    forecasts_by_name: Mapping[str, pd.DataFrame],
    *,
    candidate_key: str,
    name_to_tf: Mapping[str, TimeFrame],
    portfolio_ticker_names: frozenset[str],
) -> tuple[
    dict[str, float],
    dict[str, float],
    tuple[tuple[str, str, str, float, float, int], ...],
]:
    """Pearson and Spearman on validation **per (peer, ticker)** at the candidate's trading timeframe.

    Peers are restricted to the **same ``base_tf``** as the candidate. For each peer, only
    **tickers present in both** forecast tables and in ``portfolio_ticker_names`` are compared.

    Returns
    -------
    mean_pearson_by_peer, mean_spearman_by_peer
        Mean of finite per-ticker correlations over intersected tickers (one pair per peer).
    corr_detail_rows
        ``(peer_ensemble, ticker, timeframe_name, pearson, spearman, n_overlap_rows)``.
    """
    if candidate_key not in forecasts_by_name:
        raise KeyError(f"Missing candidate forecasts for {candidate_key!r}")
    cand_tf = name_to_tf[candidate_key]
    tf_name = cand_tf.name
    peer_keys = [
        n
        for n in forecasts_by_name
        if n != candidate_key and name_to_tf.get(n) == cand_tf
    ]
    cand_all = forecasts_by_name[candidate_key]
    mean_pearson: dict[str, float] = {}
    mean_spearman: dict[str, float] = {}
    detail: list[tuple[str, str, str, float, float, int]] = []

    for pk in peer_keys:
        peer_all = forecasts_by_name[pk]
        common = _intersect_tradable_tickers(cand_all, peer_all, portfolio_ticker_names)
        sym_col_c = cand_all["ticker"].map(_norm_ticker_sym)
        sym_col_p = peer_all["ticker"].map(_norm_ticker_sym)
        per_ticker_p: list[float] = []
        per_ticker_s: list[float] = []
        for tkr in common:
            sub_c = cand_all[sym_col_c == tkr]
            sub_p = peer_all[sym_col_p == tkr]
            rp, rs, n_ov = _corr_forecast_streams(sub_c, sub_p)
            detail.append((pk, tkr, tf_name, rp, rs, n_ov))
            if rp == rp:
                per_ticker_p.append(rp)
            if rs == rs:
                per_ticker_s.append(rs)
        mean_pearson[pk] = (
            float(np.mean(per_ticker_p)) if per_ticker_p else float("nan")
        )
        mean_spearman[pk] = (
            float(np.mean(per_ticker_s)) if per_ticker_s else float("nan")
        )

    return mean_pearson, mean_spearman, tuple(detail)


def pearson_corr_candidate_vs_each_peer(
    forecasts_by_name: Mapping[str, pd.DataFrame],
    *,
    candidate_key: str,
    name_to_tf: Mapping[str, TimeFrame],
    portfolio_ticker_names: frozenset[str],
) -> dict[str, float]:
    """Peer → mean Pearson over intersected tickers (Spearman available from :func:`compute_peer_forecast_correlations`)."""
    means_p, _, _ = compute_peer_forecast_correlations(
        forecasts_by_name,
        candidate_key=candidate_key,
        name_to_tf=name_to_tf,
        portfolio_ticker_names=portfolio_ticker_names,
    )
    return means_p


def collect_validation_ensemble_forecasts(
    config: PortfolioResearchConfig,
    ensemble_dirs: Mapping[str, str],
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, TimeFrame]]:
    """Fit each ensemble on train, return validation-window ensemble-level ``forecast_score`` frames."""
    named = _load_named_ensembles(ensemble_dirs, config)
    name_to_tf = {n: _ensemble_timeframe(e) for n, e in named}
    grouped: dict[TimeFrame, list[tuple[str, Any]]] = defaultdict(list)
    for name, ensemble in named:
        grouped[_ensemble_timeframe(ensemble)].append((name, ensemble))
    forecasts: dict[str, pd.DataFrame] = {}

    for tf, pairs in grouped.items():
        fit_q = _portfolio_cache_query(config, train_start, train_end, (tf,))
        val_q = _portfolio_cache_query(config, val_start, val_end, (tf,))
        vol_df = _query_volatility_from_cache(val_q)
        if vol_df.empty:
            raise ValueError(f"Missing EWSD volatility for validation query (timeframe={tf.name})")

        for name, ensemble in pairs:
            port = Portfolio(
                ensembles=[ensemble],
                trading_timeframe=tf,
                target_volatility=config.target_volatility,
                max_position_pct=config.max_position_pct,
                use_cache=True,
            )
            port.fit_from_cache(fit_q.for_timeframe(tf))
            tf_candles = _query_candles_from_cache(val_q, tf)
            if tf_candles.empty:
                logger.warning("No validation candles for TF %s ensemble %s", tf.name, name)
                forecasts[name] = pd.DataFrame(
                    columns=["ticker", "datetime", "forecast_score"]
                )
                continue
            raw = ensemble.predict_from_candles(
                tf_candles,
                daily_volatility_df=vol_df,
                return_base_model_predictions=True,
            )
            if not isinstance(raw, dict) or "ensemble" not in raw:
                raise TypeError(
                    f"Expected dict with 'ensemble' from {name!r}, got {type(raw).__name__}"
                )
            forecasts[name] = raw["ensemble"]

    return forecasts, name_to_tf


def _concat_returns(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.concat([a, b]).sort_index()
    return out[~out.index.duplicated(keep="first")]


def _metrics_bundle(returns: pd.Series, *, annualization: float = _DAILY_ANN) -> dict[str, float]:
    return {
        "sharpe": float(metric_sharpe(returns, annualization_factor=annualization)),
        "sortino": float(metric_sortino(returns, annualization_factor=annualization)),
        "calmar": float(metric_calmar(returns, annualization_factor=annualization)),
    }


def _standalone_metrics_row_for_single_ensemble(
    portfolio_config: PortfolioResearchConfig,
    ensemble_name: str,
    ensemble_path: str,
) -> tuple[str, float, float, float, float, float, float, float, float, float]:
    """One row: portfolio containing only ``ensemble_name`` — train / val / train+val Sharpe, Sortino, Calmar."""
    cfg_e = config_with_ensemble_dirs(portfolio_config, {ensemble_name: ensemble_path})
    ph_tr = run_single_phase_for_prop_firm(
        cfg_e,
        "train",
        emit_tearsheets=False,
        run_preflight=False,
        run_purpose="metrics_only",
    )
    ph_val = run_single_phase_for_prop_firm(
        cfg_e,
        "validation",
        emit_tearsheets=False,
        run_preflight=False,
        run_purpose="metrics_only",
    )
    m_tr = _metrics_bundle(ph_tr.combined_strategy_returns)
    m_val = _metrics_bundle(ph_val.combined_strategy_returns)
    m_tv = _metrics_bundle(
        _concat_returns(ph_tr.combined_strategy_returns, ph_val.combined_strategy_returns)
    )
    return (
        ensemble_name,
        m_tr["sharpe"],
        m_tr["sortino"],
        m_tr["calmar"],
        m_val["sharpe"],
        m_val["sortino"],
        m_val["calmar"],
        m_tv["sharpe"],
        m_tv["sortino"],
        m_tv["calmar"],
    )


def _returns_suitable_for_quantstats_tearsheet(returns: pd.Series) -> bool:
    """QuantStats R² / regression metrics require non-constant strategy returns."""
    clean = returns.astype(float).dropna()
    if len(clean) < 2:
        return False
    return float(clean.std(ddof=1)) > 1e-12


@dataclass(frozen=True)
class TearsheetComparisonLabels:
    """Human-readable scope for with/without candidate comparison tearsheets."""

    scope_title: str
    without_dir_name: str = "portfolio_without_candidate"
    with_dir_name: str = "portfolio_with_candidate"


def _write_portfolio_phase_tearsheet(
    phase_result: PhaseResult,
    *,
    output_root: Path,
    output_dir_name: str,
    phase_title: str,
    feature_name: str | None = None,
    output_filename: str = "tearsheet.html",
) -> Path | None:
    """Write one QuantStats HTML tearsheet for a portfolio phase result."""
    if not _returns_suitable_for_quantstats_tearsheet(phase_result.combined_strategy_returns):
        logger.info(
            "Skipping QuantStats tearsheet for %s: constant or insufficient strategy returns",
            feature_name or phase_title,
        )
        return None
    out_dir = Path(output_root) / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / output_filename
    generate_tearsheet(
        strategy_returns=phase_result.combined_strategy_returns,
        baseline_returns=phase_result.combined_baseline_returns,
        feature_name=feature_name or f"{phase_title} window",
        output_file=str(out),
        mode="html",
    )
    return out


def _append_tearsheet_path(paths: list[Path], path: Path | None) -> None:
    if path is not None:
        paths.append(path)


def _write_concat_portfolio_tearsheet(
    *,
    output_path: Path,
    strategy_returns: pd.Series,
    baseline_returns: pd.Series,
    feature_name: str,
) -> Path | None:
    if not _returns_suitable_for_quantstats_tearsheet(strategy_returns):
        logger.info(
            "Skipping QuantStats tearsheet for %s: constant or insufficient strategy returns",
            feature_name,
        )
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_tearsheet(
        strategy_returns=strategy_returns,
        baseline_returns=baseline_returns,
        feature_name=feature_name,
        output_file=str(output_path),
        mode="html",
    )
    return output_path


def _write_inclusion_tearsheets_from_phases(
    *,
    tearsheets_root: Path,
    phase_without_tr: PhaseResult,
    phase_without_val: PhaseResult,
    phase_with_tr: PhaseResult,
    phase_with_val: PhaseResult,
    skip_without_tearsheets: bool = False,
    labels: TearsheetComparisonLabels | None = None,
) -> tuple[Path, ...]:
    """Write with/without portfolio HTML for train, validation, and train+val (concat), no extra fits."""
    scope = labels or TearsheetComparisonLabels(scope_title="Portfolio comparison")
    root = Path(tearsheets_root)
    root.mkdir(parents=True, exist_ok=True)
    w_root = root / scope.without_dir_name
    i_root = root / scope.with_dir_name
    w_root.mkdir(parents=True, exist_ok=True)
    i_root.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    if not skip_without_tearsheets:
        _append_tearsheet_path(
            paths,
            _write_portfolio_phase_tearsheet(
                phase_without_tr,
                output_root=w_root,
                output_dir_name="train",
                phase_title="Train",
                feature_name=f"{scope.scope_title} — train (without candidate)",
            ),
        )
        _append_tearsheet_path(
            paths,
            _write_portfolio_phase_tearsheet(
                phase_without_val,
                output_root=w_root,
                output_dir_name="validation",
                phase_title="Validation",
                feature_name=f"{scope.scope_title} — validation (without candidate)",
            ),
        )
        _append_tearsheet_path(
            paths,
            _write_concat_portfolio_tearsheet(
                output_path=w_root / "train_plus_validation" / "tearsheet.html",
                strategy_returns=_concat_returns(
                    phase_without_tr.combined_strategy_returns,
                    phase_without_val.combined_strategy_returns,
                ),
                baseline_returns=_concat_returns(
                    phase_without_tr.combined_baseline_returns,
                    phase_without_val.combined_baseline_returns,
                ),
                feature_name=f"{scope.scope_title} — train+validation (without candidate)",
            ),
        )

    _append_tearsheet_path(
        paths,
        _write_portfolio_phase_tearsheet(
            phase_with_tr,
            output_root=i_root,
            output_dir_name="train",
            phase_title="Train",
            feature_name=f"{scope.scope_title} — train (with candidate)",
        ),
    )
    _append_tearsheet_path(
        paths,
        _write_portfolio_phase_tearsheet(
            phase_with_val,
            output_root=i_root,
            output_dir_name="validation",
            phase_title="Validation",
            feature_name=f"{scope.scope_title} — validation (with candidate)",
        ),
    )
    _append_tearsheet_path(
        paths,
        _write_concat_portfolio_tearsheet(
            output_path=i_root / "train_plus_validation" / "tearsheet.html",
            strategy_returns=_concat_returns(
                phase_with_tr.combined_strategy_returns,
                phase_with_val.combined_strategy_returns,
            ),
            baseline_returns=_concat_returns(
                phase_with_tr.combined_baseline_returns,
                phase_with_val.combined_baseline_returns,
            ),
            feature_name=f"{scope.scope_title} — train+validation (with candidate)",
        ),
    )

    return tuple(paths)


def write_candidate_strategy_tearsheets(
    *,
    output_root: Path,
    phase_candidate_tr: PhaseResult | None = None,
    phase_candidate_val: PhaseResult | None = None,
    candidate_key: str,
) -> tuple[Path, ...]:
    """Candidate-only tearsheets (train / validation / train+validation).

    Uses single-ensemble ``GlobalPortfolio`` phase results for the candidate
    in isolation (same return basis as sleeve/portfolio gate phases).
    """
    root = Path(output_root) / "candidate_strategy"

    if phase_candidate_tr is None or phase_candidate_val is None:
        return ()

    return _write_with_only_portfolio_tearsheets(
        tearsheets_root=root,
        phase_with_tr=phase_candidate_tr,
        phase_with_val=phase_candidate_val,
        feature_prefix="Candidate strategy",
    )


def write_full_portfolio_comparison_tearsheets(
    *,
    output_root: Path,
    phase_without_tr: PhaseResult,
    phase_without_val: PhaseResult,
    phase_with_tr: PhaseResult,
    phase_with_val: PhaseResult,
    portfolio_ticker_names: frozenset[str],
    skip_without_tearsheets: bool = False,
) -> tuple[Path, ...]:
    """Full prop-book global portfolio with vs without the candidate ensemble."""
    root = Path(output_root) / "full_portfolio"
    labels = TearsheetComparisonLabels(
        scope_title="Full portfolio",
        without_dir_name="without_candidate",
        with_dir_name="with_candidate",
    )
    return _write_inclusion_tearsheets_from_phases(
        tearsheets_root=root,
        phase_without_tr=phase_without_tr,
        phase_without_val=phase_without_val,
        phase_with_tr=phase_with_tr,
        phase_with_val=phase_with_val,
        skip_without_tearsheets=skip_without_tearsheets,
        labels=labels,
    )


write_candidate_standalone_tearsheets = write_candidate_strategy_tearsheets


def gate_tearsheet_panel_title(path: Path) -> str | None:
    """Map portfolio-gate tearsheet paths to the twelve canonical UI panel titles."""
    normalized = path.as_posix().lower()
    if path.suffix.lower() != ".html" or path.name.lower() != "tearsheet.html":
        return None
    if "portfolio_gate_tearsheets" not in normalized:
        return None

    phase_label = ""
    if "/train_plus_validation/" in normalized:
        phase_label = "train+validation"
    elif "/validation/" in normalized:
        phase_label = "validation"
    elif "/train/" in normalized:
        phase_label = "train"

    if "/candidate_strategy/" in normalized or "/candidate_standalone/" in normalized:
        return f"Candidate strategy — {phase_label}" if phase_label else "Candidate strategy"

    if "/full_portfolio/" in normalized:
        comparison = (
            "with candidate"
            if "/with_candidate/" in normalized
            else "without candidate"
        )
        return f"Full portfolio — {phase_label} ({comparison})"

    if "/sleeve_" in normalized:
        sleeve_dir = next((part for part in path.parts if part.startswith("sleeve_")), "")
        sleeve_slug = _display_sleeve_slug(sleeve_dir.removeprefix("sleeve_"))
        comparison = (
            "with candidate"
            if "/with_candidate/" in normalized
            else "without candidate"
        )
        return f"Sleeve ({sleeve_slug}) — {phase_label} ({comparison})"

    return None


def _sanitize_tearsheet_path_segment(label: str) -> str:
    """Filesystem-safe sleeve id; ``/`` in the label becomes ``__`` for round-trip display."""
    parts = [re.sub(r"[^\w.-]+", "_", part.strip()) for part in label.split("/") if part.strip()]
    return "__".join(parts) if parts else "sleeve"


def _display_sleeve_slug(slug: str) -> str:
    if "__" in slug:
        return " / ".join(part.replace("_", " ") for part in slug.split("__"))
    return slug.replace("_", " ")


def _write_with_only_portfolio_tearsheets(
    *,
    tearsheets_root: Path,
    phase_with_tr: PhaseResult,
    phase_with_val: PhaseResult,
    feature_prefix: str,
) -> tuple[Path, ...]:
    """Train / validation / train+val HTML for a single portfolio path."""
    root = Path(tearsheets_root)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    _append_tearsheet_path(
        paths,
        _write_portfolio_phase_tearsheet(
            phase_with_tr,
            output_root=root,
            output_dir_name="train",
            phase_title="Train",
            feature_name=f"{feature_prefix} — train",
        ),
    )
    _append_tearsheet_path(
        paths,
        _write_portfolio_phase_tearsheet(
            phase_with_val,
            output_root=root,
            output_dir_name="validation",
            phase_title="Validation",
            feature_name=f"{feature_prefix} — validation",
        ),
    )
    _append_tearsheet_path(
        paths,
        _write_concat_portfolio_tearsheet(
            output_path=root / "train_plus_validation" / "tearsheet.html",
            strategy_returns=_concat_returns(
                phase_with_tr.combined_strategy_returns,
                phase_with_val.combined_strategy_returns,
            ),
            baseline_returns=_concat_returns(
                phase_with_tr.combined_baseline_returns,
                phase_with_val.combined_baseline_returns,
            ),
            feature_name=f"{feature_prefix} — train+validation",
        ),
    )
    return tuple(paths)


def write_sleeve_level_tearsheets(
    *,
    output_dir: Path,
    sleeve_label: str,
    phase_without_tr: PhaseResult | None = None,
    phase_without_val: PhaseResult | None = None,
    phase_with_tr: PhaseResult | None = None,
    phase_with_val: PhaseResult | None = None,
) -> tuple[Path, ...]:
    """Write sleeve-scoped QuantStats HTML (with/without candidate when both sides exist).

    Output layout: ``<output_dir>/sleeve_tearsheets_<label>/...`` mirroring inclusion
    ``portfolio_without_candidate`` / ``portfolio_with_candidate`` folders. When only
    ``phase_with_*`` are provided (first strategy in sleeve), writes the with-candidate
    train, validation, and train+validation tearsheets only.
    """
    segment = _sanitize_tearsheet_path_segment(sleeve_label)
    root = Path(output_dir) / "portfolio_gate_tearsheets" / f"sleeve_{segment}"
    prefix = f"Sleeve ({sleeve_label})"

    has_without = phase_without_tr is not None and phase_without_val is not None
    has_with = phase_with_tr is not None and phase_with_val is not None
    if has_without and has_with:
        skip_without = not _returns_suitable_for_quantstats_tearsheet(
            phase_without_tr.combined_strategy_returns
        )
        if skip_without:
            logger.info(
                "Sleeve %s: empty without-candidate returns; writing with-candidate tearsheets only",
                sleeve_label,
            )
        sleeve_labels = TearsheetComparisonLabels(
            scope_title=prefix,
            without_dir_name="without_candidate",
            with_dir_name="with_candidate",
        )
        return _write_inclusion_tearsheets_from_phases(
            tearsheets_root=root,
            phase_without_tr=phase_without_tr,
            phase_without_val=phase_without_val,
            phase_with_tr=phase_with_tr,
            phase_with_val=phase_with_val,
            skip_without_tearsheets=skip_without,
            labels=sleeve_labels,
        )
    if has_with:
        with_root = root / "with_candidate"
        return _write_with_only_portfolio_tearsheets(
            tearsheets_root=with_root,
            phase_with_tr=phase_with_tr,
            phase_with_val=phase_with_val,
            feature_prefix=f"{prefix} (with candidate)",
        )
    return ()


@dataclass(frozen=True)
class PortfolioInclusionResult:
    """Correlation tables, per-ensemble standalone metrics, optional comparison tearsheets."""

    candidate_forecast_timeframe: str
    corr_peer_means: tuple[tuple[str, float, float], ...]
    corr_by_peer_ticker: tuple[tuple[str, str, str, float, float, int], ...]
    max_pearson_with_any_peer: float
    max_spearman_with_any_peer: float
    #: Sorted baseline ensembles, then **candidate** last — same layout as ``*_standalone_by_ensemble.csv``.
    standalone_metrics_by_ensemble: tuple[
        tuple[str, float, float, float, float, float, float, float, float, float],
        ...,
    ]
    tearsheet_paths: tuple[Path, ...]
    message: str


def run_portfolio_inclusion(
    portfolio_config: PortfolioResearchConfig,
    *,
    candidate_repo_relative_path: str | None = None,
    candidate_ensemble_dir: str | None = None,
    candidate_key: str | None = None,
    inclusion_config: PortfolioInclusionConfig | None = None,
    run_preflight: bool = True,
    tearsheets_output_dir: Path | None = None,
) -> PortfolioInclusionResult:
    """Validation-window forecast correlation; per-ensemble standalone metrics; with/without tearsheets.

    Standalone metrics use a **single-ensemble** portfolio (each baseline, then the candidate) on
    train, validation, and concatenated train+validation — same definition as the historical
    ``*_standalone_by_ensemble.csv``.

    Provide exactly one of ``candidate_repo_relative_path`` or ``candidate_ensemble_dir`` (e.g. temp
    dir from :func:`materialize_inclusion_candidate_from_eval_bias_spec`).

    **WeightLayer:** always ``ledoit_wolf_min_corr`` with ``fdm_max`` taken from
    ``portfolio_config.weight_layer_kwargs`` (default 2.0), independent of
    ``portfolio_config.weight_layer_method``.

    When ``inclusion_config.emit_tearsheets`` is True (default), writes six HTML tearsheets under
    ``tearsheets_output_dir`` or ``portfolio_config.output_root / inclusion_tearsheets_<candidate_key>``.
    """
    if (candidate_repo_relative_path is None) == (candidate_ensemble_dir is None):
        raise ValueError(
            "Provide exactly one of candidate_repo_relative_path or candidate_ensemble_dir."
        )
    cfg = inclusion_config or PortfolioInclusionConfig()
    if candidate_ensemble_dir is not None:
        candidate_path = str(Path(candidate_ensemble_dir).resolve())
    else:
        candidate_path = candidate_repo_relative_path  # type: ignore[assignment]
    ck = candidate_key or infer_candidate_key(candidate_path)

    baseline_dirs = dict(portfolio_config.ensemble_dirs)
    if ck in baseline_dirs:
        raise ValueError(
            f"Candidate key {ck!r} already exists in ensemble_dirs; use a distinct candidate_key."
        )
    cand_resolved = Path(candidate_path).resolve()
    if any(Path(p).resolve() == cand_resolved for p in baseline_dirs.values()):
        raise ValueError("Candidate ensemble path matches an existing ensemble_dirs path.")

    portfolio_config = _normalize_portfolio_config_for_inclusion(portfolio_config)

    merged_dirs = {**baseline_dirs, ck: candidate_path}

    if run_preflight:
        run_portfolio_research_cache_preflight(
            replace(portfolio_config, ensemble_dirs=merged_dirs),
        )

    train_start = pd.Timestamp(portfolio_config.train_window.start)
    train_end = pd.Timestamp(portfolio_config.train_window.end)
    val_start = pd.Timestamp(portfolio_config.validation_window.start)
    val_end = pd.Timestamp(portfolio_config.validation_window.end)

    forecasts, name_to_tf = collect_validation_ensemble_forecasts(
        portfolio_config,
        merged_dirs,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
    )

    tickers = frozenset(t.name for t in portfolio_config.tickers)
    cand_tf_name = name_to_tf[ck].name
    corr_p, corr_s, corr_detail = compute_peer_forecast_correlations(
        forecasts,
        candidate_key=ck,
        name_to_tf=name_to_tf,
        portfolio_ticker_names=tickers,
    )
    corr_peer_means = tuple(sorted(((pk, corr_p[pk], corr_s[pk]) for pk in corr_p), key=lambda x: x[0]))
    finite_p = [c for _, c, _ in corr_peer_means if c == c]
    finite_s = [s for _, _, s in corr_peer_means if s == s]
    max_p = max(finite_p) if finite_p else float("nan")
    max_s = max(finite_s) if finite_s else float("nan")

    standalone_rows: list[
        tuple[str, float, float, float, float, float, float, float, float, float]
    ] = []
    for ens_name, ens_path in sorted(baseline_dirs.items()):
        standalone_rows.append(
            _standalone_metrics_row_for_single_ensemble(
                portfolio_config, ens_name, ens_path
            )
        )
    standalone_rows.append(
        _standalone_metrics_row_for_single_ensemble(portfolio_config, ck, candidate_path)
    )
    standalone_metrics_by_ensemble = tuple(standalone_rows)

    cfg_base = portfolio_config
    cfg_with = config_with_ensemble_dirs(portfolio_config, merged_dirs)

    phase_without_tr = run_single_phase_for_prop_firm(
        cfg_base,
        "train",
        emit_tearsheets=False,
        run_preflight=False,
        run_purpose="metrics_only",
    )
    phase_with_tr = run_single_phase_for_prop_firm(
        cfg_with,
        "train",
        emit_tearsheets=False,
        run_preflight=False,
        run_purpose="metrics_only",
    )
    phase_without_val = run_single_phase_for_prop_firm(
        cfg_base,
        "validation",
        emit_tearsheets=False,
        run_preflight=False,
        run_purpose="metrics_only",
    )
    phase_with_val = run_single_phase_for_prop_firm(
        cfg_with,
        "validation",
        emit_tearsheets=False,
        run_preflight=False,
        run_purpose="metrics_only",
    )

    tearsheet_paths: tuple[Path, ...] = ()
    if cfg.emit_tearsheets:
        ts_root = tearsheets_output_dir or (
            Path(portfolio_config.output_root) / f"inclusion_tearsheets_{ck}"
        )
        tearsheet_paths = _write_inclusion_tearsheets_from_phases(
            tearsheets_root=ts_root,
            phase_without_tr=phase_without_tr,
            phase_without_val=phase_without_val,
            phase_with_tr=phase_with_tr,
            phase_with_val=phase_with_val,
        )

    peer_note = f"{len(corr_peer_means)} peer(s)"
    parts = [
        f"forecast_tf={cand_tf_name} peer_corr max_Pearson={max_p:.4f} max_Spearman={max_s:.4f} ({peer_note})",
    ]
    if tearsheet_paths:
        parts.append(f"tearsheets={len(tearsheet_paths)} files")
    parts.append(f"standalone_by_ensemble={len(standalone_metrics_by_ensemble)} rows")
    msg = "; ".join(parts)

    return PortfolioInclusionResult(
        candidate_forecast_timeframe=cand_tf_name,
        corr_peer_means=corr_peer_means,
        corr_by_peer_ticker=corr_detail,
        max_pearson_with_any_peer=max_p,
        max_spearman_with_any_peer=max_s,
        standalone_metrics_by_ensemble=standalone_metrics_by_ensemble,
        tearsheet_paths=tearsheet_paths,
        message=msg,
    )


def write_inclusion_artifacts(
    output_dir: Path, candidate_key: str, result: PortfolioInclusionResult
) -> list[Path]:
    """Write correlation CSVs and per-ensemble standalone performance (train / val / train+val)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"inclusion_{candidate_key}"
    paths: list[Path] = []

    corr_tkr = Path(output_dir) / f"{stem}_corr_by_peer_and_ticker.csv"
    pd.DataFrame(
        [
            {
                "peer_ensemble": row[0],
                "ticker": row[1],
                "timeframe": row[2],
                "pearson": row[3],
                "spearman": row[4],
                "n_overlap_rows": row[5],
            }
            for row in result.corr_by_peer_ticker
        ]
    ).to_csv(corr_tkr, index=False)
    paths.append(corr_tkr)

    corr_peer = Path(output_dir) / f"{stem}_corr_by_peer_summary.csv"
    pd.DataFrame(
        [
            {
                "peer_ensemble": name,
                "mean_pearson": mp,
                "mean_spearman": ms,
            }
            for name, mp, ms in result.corr_peer_means
        ]
    ).to_csv(corr_peer, index=False)
    paths.append(corr_peer)

    stand_path = Path(output_dir) / f"{stem}_standalone_by_ensemble.csv"
    pd.DataFrame(
        [
            {
                "ensemble": e,
                "sharpe_train": st,
                "sortino_train": sot,
                "calmar_train": ct,
                "sharpe_validation": sv,
                "sortino_validation": sov,
                "calmar_validation": cv,
                "sharpe_train_plus_validation": stv,
                "sortino_train_plus_validation": sotv,
                "calmar_train_plus_validation": ctv,
            }
            for (
                e,
                st,
                sot,
                ct,
                sv,
                sov,
                cv,
                stv,
                sotv,
                ctv,
            ) in result.standalone_metrics_by_ensemble
        ]
    ).to_csv(stand_path, index=False)
    paths.append(stand_path)

    return paths


def run_inclusion_decision(
    portfolio_config: PortfolioResearchConfig,
    *,
    candidate_repo_relative_path: str | None = None,
    candidate_ensemble_dir: str | None = None,
    candidate_key: str | None = None,
    inclusion_config: PortfolioInclusionConfig | None = None,
    thresholds: PortfolioInclusionConfig | None = None,
    confirm_test: bool = False,
    run_preflight: bool = True,
    tearsheets_output_dir: Path | None = None,
    uplift_tearsheets_output_dir: Path | None = None,
) -> PortfolioInclusionResult:
    """Back-compat alias for :func:`run_portfolio_inclusion`.

    Accepts legacy ``thresholds=`` / ``uplift_tearsheets_output_dir=``; ``confirm_test`` is ignored.
    """
    _ = confirm_test
    cfg = inclusion_config if inclusion_config is not None else thresholds
    out = (
        tearsheets_output_dir
        if tearsheets_output_dir is not None
        else uplift_tearsheets_output_dir
    )
    return run_portfolio_inclusion(
        portfolio_config,
        candidate_repo_relative_path=candidate_repo_relative_path,
        candidate_ensemble_dir=candidate_ensemble_dir,
        candidate_key=candidate_key,
        inclusion_config=cfg,
        run_preflight=run_preflight,
        tearsheets_output_dir=out,
    )


def write_inclusion_reports(
    output_dir: Path, candidate_key: str, result: PortfolioInclusionResult
) -> list[Path]:
    """Back-compat alias for :func:`write_inclusion_artifacts`."""
    return write_inclusion_artifacts(output_dir, candidate_key, result)
