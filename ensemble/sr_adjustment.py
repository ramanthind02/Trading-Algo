"""Carver handcrafting SR multipliers and hierarchical application for hierarchy_equal."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ensemble.ensemble_utils import normalize_ticker_key
from ensemble.portfolio_impl.global_weight_layer_adapter import decode_global_stream_id

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252.0


def _clip_nonneg_normalize_weights(weights: Mapping[str, float]) -> pd.Series:
    """Portfolio stream weights must be nonnegative and sum to one."""
    clipped = {str(key): max(0.0, float(value)) for key, value in weights.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        count = len(clipped)
        uniform = (1.0 / count) if count else 0.0
        return pd.Series({key: uniform for key in clipped}, dtype=float)
    return pd.Series({key: value / total for key, value in clipped.items()}, dtype=float)


@dataclass(frozen=True)
class SrAdjustmentParams:
    """Parameters for Carver mini-bootstrap SR tilt."""

    sr_avg: float = 0.5
    sr_p_step: float = 0.01
    sr_std: float = 0.15
    sr_min_years: float = 5.0


@dataclass(frozen=True)
class SrTiltDiagnostics:
    """Per-member diagnostics from one sibling-group SR tilt."""

    member_key: str
    sr_multiplier: float
    annualized_sr: float
    sr_diff: float
    subtree_mass_before: float
    subtree_mass_after: float


def omega_difference(std: float, years_of_data: float, avg_correlation: float) -> float:
    """Standard deviation of a difference in mean estimates (Carver blog)."""
    if years_of_data <= 0.0:
        return float("inf")
    omega_one = std / (years_of_data**0.5)
    omega_var = 2.0 * (omega_one**2) * (1.0 - avg_correlation)
    return float(max(omega_var, 0.0) ** 0.5)


def calculate_confident_mean_difference(
    std: float,
    years_of_data: float,
    mean_difference: float,
    confidence_interval: float,
    avg_correlation: float,
) -> float:
    omega = omega_difference(std, years_of_data, avg_correlation)
    return float(stats.norm(mean_difference, omega).ppf(confidence_interval))


def _boring_corr_matrix(size: int, offdiag: float = 0.99) -> np.ndarray:
    diag = np.ones(size, dtype=float)
    off = np.full((size, size), offdiag, dtype=float)
    np.fill_diagonal(off, 1.0)
    return off


def _sigma_from_corr_and_std(stdev_list: Sequence[float], corr_matrix: np.ndarray) -> np.ndarray:
    stdev = np.asarray(stdev_list, dtype=float).reshape(-1, 1)
    return stdev * corr_matrix * stdev.T


def _variance(weights: np.ndarray, sigma: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float).reshape(1, -1)
    return float((w @ sigma @ w.T)[0, 0])


def _neg_sharpe_ratio(weights: np.ndarray, sigma: np.ndarray, mus: np.ndarray) -> float:
    est_return = float(np.dot(weights, mus))
    vol = _variance(weights, sigma) ** 0.5
    if vol <= 0.0:
        return 0.0
    return -est_return / vol


def _optimise_two_asset_weights(
    mean_list: Sequence[float],
    avg_correlation: float,
    std: float,
) -> List[float]:
    n_assets = len(mean_list)
    corr = _boring_corr_matrix(n_assets, offdiag=avg_correlation)
    sigma = _sigma_from_corr_and_std([std] * n_assets, corr)
    mus = np.asarray(mean_list, dtype=float)
    start = np.full(n_assets, 1.0 / n_assets, dtype=float)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: 1.0 - float(np.sum(w))}]
    result = minimize(
        _neg_sharpe_ratio,
        start,
        args=(sigma, mus),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=1e-5,
    )
    return [float(x) for x in result.x]


def weights_given_sr_diff(
    sr_diff: float,
    avg_correlation: float,
    years_of_data: float,
    confidence_interval: float,
    *,
    avg_sr: float = 0.5,
    std: float = 0.15,
    how_many_assets: int = 2,
) -> List[float]:
    """Portfolio weights for one asset with SR offset vs peers at a CDF point."""
    average_mean = avg_sr * std
    asset1_mean = (sr_diff + avg_sr) * std
    mean_difference = asset1_mean - average_mean
    confident_diff = calculate_confident_mean_difference(
        std,
        years_of_data,
        mean_difference,
        confidence_interval,
        avg_correlation,
    )
    confident_asset1_mean = confident_diff + average_mean
    mean_list = [confident_asset1_mean] + [average_mean] * (how_many_assets - 1)
    return _optimise_two_asset_weights(mean_list, avg_correlation, std)


def _weight_ratio(weights: Sequence[float]) -> float:
    n = len(weights)
    if n == 0:
        return 1.0
    one_over_n = 1.0 / n
    return float(weights[0] / one_over_n)


def mini_bootstrap_weight_ratio(
    sr_diff: float,
    avg_correlation: float,
    years_of_data: float,
    *,
    avg_sr: float = 0.5,
    std: float = 0.15,
    how_many_assets: int = 2,
    p_step: float = 0.01,
) -> float:
    """Ratio of optimised weight to 1/N for an asset with SR difference ``sr_diff``."""
    if p_step <= 0.0 or p_step >= 1.0:
        raise ValueError(f"p_step must be in (0, 1), got {p_step}")
    dist_points = np.arange(p_step, 1.0 - p_step + 1e-9, p_step)
    ratios = [
        _weight_ratio(
            weights_given_sr_diff(
                sr_diff,
                avg_correlation,
                years_of_data,
                float(ci),
                avg_sr=avg_sr,
                std=std,
                how_many_assets=how_many_assets,
            )
        )
        for ci in dist_points
    ]
    ratio = float(np.nanmean(ratios))
    if np.sign(ratio - 1.0) != np.sign(sr_diff) and abs(sr_diff) > 1e-12:
        return 1.0
    return ratio


def annualized_sharpe(daily_returns: pd.Series) -> float:
    """Annualized Sharpe from daily PnL."""
    clean = daily_returns.astype(float).dropna()
    if len(clean) < 2:
        return 0.0
    std = float(clean.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(clean.mean() / std * (_TRADING_DAYS_PER_YEAR**0.5))


def mean_off_diagonal_correlation(frame: pd.DataFrame) -> float:
    """Mean pairwise correlation; clipped to [0, 1] for SR uncertainty."""
    if frame.shape[1] <= 1:
        return 0.0
    corr = frame.corr().clip(lower=0.0, upper=1.0)
    n = len(corr.columns)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    values = corr.to_numpy(dtype=float)[mask]
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def pivot_stream_signals(
    ticker_forecasts: pd.DataFrame,
    available_models: Sequence[str],
    *,
    value_column: str = "signal",
) -> pd.DataFrame:
    """Pivot a forecast column to date × model matrix."""
    if "datetime" not in ticker_forecasts.columns:
        return pd.DataFrame()

    col = value_column
    if col not in ticker_forecasts.columns:
        col = "forecast" if "forecast" in ticker_forecasts.columns else "signal"
    df = ticker_forecasts.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.normalize()
    pivot = df.pivot_table(
        index="date",
        columns="model_name",
        values=col,
        aggfunc="mean",
    )
    existing = [m for m in available_models if m in pivot.columns]
    if not existing:
        return pd.DataFrame()
    return pivot[existing].dropna(how="all")


def build_stream_pnl_series(
    position_pivot: pd.DataFrame,
    instrument_returns: pd.DataFrame,
    stream_ids: Sequence[str],
    *,
    lag_positions: bool = True,
) -> Dict[str, pd.Series]:
    """Daily PnL per global stream: lagged position proxy × instrument return.

    Uses the prior bar's forecast/signal (no same-day lookahead), matching
    ``portfolio_tester.calculate_strategy_returns_from_positions``.
    """
    if position_pivot.empty or instrument_returns.empty:
        return {}

    rets = instrument_returns.astype(float).copy()
    rets.index = pd.to_datetime(rets.index).normalize()
    rets.columns = [normalize_ticker_key(col) for col in rets.columns]

    pnls: Dict[str, pd.Series] = {}
    skipped: list[str] = []
    for stream_id in stream_ids:
        if stream_id not in position_pivot.columns:
            skipped.append(f"{stream_id} (missing from pivot)")
            continue
        try:
            ticker = normalize_ticker_key(decode_global_stream_id(stream_id)["ticker"])
        except ValueError:
            logger.warning("Skipping SR PnL for invalid stream_id %s", stream_id)
            continue
        if ticker not in rets.columns:
            skipped.append(f"{stream_id} (ticker {ticker!r} not in returns)")
            continue
        position = position_pivot[stream_id].astype(float).reindex(rets.index).fillna(0.0)
        if lag_positions:
            position = position.shift(1).fillna(0.0)
        ticker_ret = rets[ticker].reindex(rets.index).fillna(0.0)
        pnls[stream_id] = (position * ticker_ret).rename(stream_id)
    if skipped:
        logger.warning(
            "SR adjustment: skipped %d stream(s) when building PnL: %s",
            len(skipped),
            skipped[:8] if len(skipped) > 8 else skipped,
        )
    return pnls


def _member_prefix(parent_path: str, member_key: str) -> str:
    if not parent_path:
        return member_key
    return f"{parent_path}/{member_key}"


def _streams_in_subtree(
    assignments: Mapping[str, str],
    prefix: str,
) -> List[str]:
    return [
        sid
        for sid, path in assignments.items()
        if path == prefix or path.startswith(prefix + "/")
    ]


def _subtree_mass(
    leaf_weights: Mapping[str, float],
    assignments: Mapping[str, str],
    prefix: str,
) -> float:
    return sum(float(leaf_weights[sid]) for sid in _streams_in_subtree(assignments, prefix))


def _aggregate_member_pnl(
    member_prefix: str,
    leaf_weights: Mapping[str, float],
    stream_pnls: Mapping[str, pd.Series],
    assignments: Mapping[str, str],
) -> pd.Series:
    streams = _streams_in_subtree(assignments, member_prefix)
    if not streams:
        return pd.Series(dtype=float)
    if len(streams) == 1:
        return stream_pnls.get(streams[0], pd.Series(dtype=float))

    total_mass = sum(float(leaf_weights[sid]) for sid in streams)
    if total_mass <= 0.0:
        weights = {sid: 1.0 / len(streams) for sid in streams}
    else:
        weights = {sid: float(leaf_weights[sid]) / total_mass for sid in streams}

    combined: pd.Series | None = None
    for sid in streams:
        pnl = stream_pnls.get(sid)
        if pnl is None or pnl.empty:
            continue
        weighted = pnl.astype(float) * weights[sid]
        combined = weighted if combined is None else combined.add(weighted, fill_value=0.0)
    return combined if combined is not None else pd.Series(dtype=float)


def _sibling_groups(assignments: Mapping[str, str]) -> Dict[str, List[str]]:
    """Map parent path → direct child keys with at least two siblings."""
    children_by_parent: Dict[str, set[str]] = defaultdict(set)
    for path in assignments.values():
        parts = path.split("/")
        for depth in range(1, len(parts)):
            parent = "/".join(parts[:depth])
            child = parts[depth]
            children_by_parent[parent].add(child)
    return {
        parent: sorted(children)
        for parent, children in children_by_parent.items()
        if len(children) >= 2
    }


def _years_from_pnl(pnl: pd.Series) -> float:
    clean = pnl.astype(float).dropna()
    return float(len(clean)) / _TRADING_DAYS_PER_YEAR


def _apply_group_sr_tilt(
    leaf_weights: Dict[str, float],
    assignments: Mapping[str, str],
    stream_pnls: Mapping[str, pd.Series],
    parent_path: str,
    member_keys: Sequence[str],
    params: SrAdjustmentParams,
) -> List[SrTiltDiagnostics]:
    """Tilt subtree masses among siblings; preserve total mass in the group."""
    member_pnls: Dict[str, pd.Series] = {}
    masses_before: Dict[str, float] = {}
    for key in member_keys:
        prefix = _member_prefix(parent_path, key)
        pnl = _aggregate_member_pnl(prefix, leaf_weights, stream_pnls, assignments)
        if pnl.empty:
            return []
        member_pnls[key] = pnl
        masses_before[key] = _subtree_mass(leaf_weights, assignments, prefix)

    years = min(_years_from_pnl(pnl) for pnl in member_pnls.values())
    if years < params.sr_min_years:
        return []

    pnl_frame = pd.DataFrame(member_pnls).dropna(how="all")
    avg_corr = mean_off_diagonal_correlation(pnl_frame)
    srs = {key: annualized_sharpe(member_pnls[key]) for key in member_keys}
    avg_sr = float(np.mean(list(srs.values())))

    multipliers = {
        key: mini_bootstrap_weight_ratio(
            srs[key] - avg_sr,
            avg_corr,
            years,
            avg_sr=params.sr_avg,
            std=params.sr_std,
            p_step=params.sr_p_step,
        )
        for key in member_keys
    }

    total_before = sum(masses_before.values())
    adjusted = {key: masses_before[key] * multipliers[key] for key in member_keys}
    total_adj = sum(adjusted.values())
    if total_adj <= 0.0:
        return []

    scale = total_before / total_adj
    diagnostics: List[SrTiltDiagnostics] = []

    for key in member_keys:
        prefix = _member_prefix(parent_path, key)
        factor = (multipliers[key] * scale) if masses_before[key] > 0.0 else 1.0
        for sid in _streams_in_subtree(assignments, prefix):
            leaf_weights[sid] = float(leaf_weights[sid]) * factor
        mass_after = _subtree_mass(leaf_weights, assignments, prefix)
        diagnostics.append(
            SrTiltDiagnostics(
                member_key=f"{parent_path}/{key}" if parent_path else key,
                sr_multiplier=float(multipliers[key] * scale),
                annualized_sr=float(srs[key]),
                sr_diff=float(srs[key] - avg_sr),
                subtree_mass_before=float(masses_before[key]),
                subtree_mass_after=float(mass_after),
            )
        )
    return diagnostics


def apply_hierarchical_sr_tilt(
    leaf_weights: pd.Series,
    cluster_assignments: Mapping[str, str],
    stream_pnls: Mapping[str, pd.Series],
    params: SrAdjustmentParams,
    *,
    max_depth: int | None = None,
) -> tuple[pd.Series, List[SrTiltDiagnostics]]:
    """
    Apply Carver SR multipliers at every sibling level, deepest groups first.

    Preserves total weight sum and parent-group mass at each level.

    Parameters
    ----------
    max_depth:
        When set, SR tilt is only applied to sibling groups whose parent path
        depth is at most ``max_depth`` levels deep (counting from root = 1).
        For example ``max_depth=2`` applies tilt at the asset-class level (L1)
        and the style-group level (L2) but skips the within-group instrument
        level (L3).  ``None`` (default) applies tilt at all levels.
    """
    if leaf_weights.empty or not stream_pnls:
        return leaf_weights, []

    weights: Dict[str, float] = {str(k): float(v) for k, v in leaf_weights.items()}
    assignments = {str(k): str(v) for k, v in cluster_assignments.items()}
    pnls = {str(k): v for k, v in stream_pnls.items() if not v.empty}

    groups = _sibling_groups(assignments)
    sorted_parents = sorted(groups.keys(), key=lambda p: len(p.split("/")), reverse=True)

    all_diag: List[SrTiltDiagnostics] = []
    for parent_path in sorted_parents:
        # skip levels deeper than max_depth (e.g. within-group instrument level)
        if max_depth is not None and len(parent_path.split("/")) > max_depth:
            logger.debug(
                "SR tilt: skipping %r (depth %d > max_depth %d)",
                parent_path, len(parent_path.split("/")), max_depth,
            )
            continue
        diag = _apply_group_sr_tilt(
            weights,
            assignments,
            pnls,
            parent_path,
            groups[parent_path],
            params,
        )
        all_diag.extend(diag)

    return _clip_nonneg_normalize_weights(weights), all_diag


def apply_within_group_inv_corr(
    leaf_weights: pd.Series,
    cluster_assignments: Mapping[str, str],
    stream_pnls: Mapping[str, pd.Series],
    *,
    min_depth: int = 2,
) -> pd.Series:
    """Redistribute mass within sibling groups at ``parent_path`` depth >= ``min_depth``
    using inverse-average-pairwise-correlation weighting.

    Applied after ``apply_hierarchical_sr_tilt`` so that SR tilt handles the
    upper levels (e.g. L1 asset classes) while this function diversifies within the
    lower levels (e.g. L2 style groups and L3 instrument streams).

    Parameters
    ----------
    min_depth:
        Only apply to sibling groups whose parent path contains at least this many
        segments (depth 1 = root, 2 = first level below root, etc.).
        Default ``2`` skips the root-level split (L1) and handles L2+L3.
    """
    if leaf_weights.empty or not stream_pnls:
        return leaf_weights

    weights: Dict[str, float] = {str(k): float(v) for k, v in leaf_weights.items()}
    assignments = {str(k): str(v) for k, v in cluster_assignments.items()}
    pnls = {str(k): v for k, v in stream_pnls.items() if not v.empty}

    groups = _sibling_groups(assignments)
    for parent_path, member_keys in groups.items():
        if len(parent_path.split("/")) < min_depth:
            continue
        if len(member_keys) < 2:
            continue

        member_pnls: Dict[str, pd.Series] = {}
        masses_before: Dict[str, float] = {}
        for key in member_keys:
            prefix = _member_prefix(parent_path, key)
            pnl = _aggregate_member_pnl(prefix, weights, pnls, assignments)
            if pnl.empty:
                continue
            member_pnls[key] = pnl
            masses_before[key] = _subtree_mass(weights, assignments, prefix)

        if len(member_pnls) < 2:
            continue

        pnl_frame = pd.DataFrame(member_pnls).dropna(how="all")
        if len(pnl_frame) < 30:
            continue

        corr = pnl_frame.corr().clip(lower=0.0)
        n = len(corr.columns)
        avg_corr = (corr.sum(axis=1) - 1.0) / max(n - 1, 1)
        inv_scores = 1.0 / (avg_corr + 0.01)
        inv_weights = inv_scores / inv_scores.sum()

        total_before = sum(max(0.0, masses_before[key]) for key in member_pnls)
        if total_before <= 0.0:
            continue
        for key in member_pnls:
            target_mass = total_before * float(inv_weights[key])
            current_mass = max(0.0, masses_before[key])
            if current_mass <= 0.0:
                continue
            factor = target_mass / current_mass
            prefix = _member_prefix(parent_path, key)
            for sid in _streams_in_subtree(assignments, prefix):
                weights[sid] = float(weights[sid]) * factor

    return _clip_nonneg_normalize_weights(weights)


def enrich_cluster_metrics_with_sr(
    cluster_metrics: Dict[str, Dict[str, float | int | None]],
    cluster_assignments: Mapping[str, str],
    stream_pnls: Mapping[str, pd.Series],
    adjusted_weights: pd.Series,
    sr_diag: Sequence[SrTiltDiagnostics],
) -> Dict[str, Dict[str, float | int | None]]:
    """Attach stream-level SR fields to leaf path metrics."""
    del sr_diag
    enriched = {path: dict(metrics) for path, metrics in cluster_metrics.items()}
    for stream_id, path in cluster_assignments.items():
        pnl = stream_pnls.get(stream_id)
        if pnl is None or path not in enriched:
            continue
        enriched[path] = {
            **enriched.get(path, {}),
            "annualized_sr": annualized_sharpe(pnl),
            "stream_weight": float(adjusted_weights.get(stream_id, 0.0)),
        }
    return enriched


__all__ = [
    "SrAdjustmentParams",
    "SrTiltDiagnostics",
    "annualized_sharpe",
    "apply_hierarchical_sr_tilt",
    "apply_within_group_inv_corr",
    "build_stream_pnl_series",
    "enrich_cluster_metrics_with_sr",
    "mean_off_diagonal_correlation",
    "mini_bootstrap_weight_ratio",
    "omega_difference",
    "pivot_stream_signals",
    "weights_given_sr_diff",
]
