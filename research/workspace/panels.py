"""Panel builders for research workspace report sections."""
from __future__ import annotations

from pathlib import Path

from research.workspace.constants import PANEL_KIND_PRIMARY, PANEL_KIND_RAW
from research.workspace.models import PanelPolicy


def panel_view_type(kind: str) -> str:
    if kind in {"html", "image"}:
        return "embed"
    if kind == "table":
        return "table"
    if kind == "markdown":
        return "markdown"
    if kind == "json":
        return "json"
    return "text"


def panel_kind(path: Path, kind: str, policy: PanelPolicy) -> str:
    if kind in {"html", "image", "markdown"}:
        return PANEL_KIND_PRIMARY
    if kind == "json" and path.name.lower() in policy.summary_json_names:
        return PANEL_KIND_PRIMARY
    return PANEL_KIND_RAW


def is_featured_panel(path: Path, kind: str, policy: PanelPolicy) -> bool:
    if policy.is_featured_panel is not None:
        return policy.is_featured_panel(path, kind)
    return False


def default_friendly_panel_title(path: Path) -> str:
    stem = path.stem
    if "tearsheet" in stem.lower():
        return stem.replace("_", " ").replace("-", " ").title()
    if stem == "permutation_summary":
        return "Permutation summary"
    if stem == "permutation_vector_shuffle":
        return "Vector-shuffle results"
    if stem == "robustness_report":
        return "Robustness report"
    if stem == "validation_robustness_report":
        return "Validation robustness report"
    if stem == "holdout_robustness_report":
        return "Holdout robustness report"
    if stem == "portfolio_addition_report":
        return "Portfolio addition gate"
    if stem == "portfolio_holdout_report":
        return "Portfolio holdout report"
    if stem == "monitoring_rollup":
        return "Monitoring rollup"
    if stem == "monitoring_history":
        return "Monitoring history"
    if stem == "monitoring_status":
        return "Monitoring status"
    if stem == "fold_manifest":
        return "Fold manifest"
    if stem == "portfolio_addition_delta_sr_histogram":
        return "Portfolio addition ΔSR bootstrap"
    if stem == "portfolio_addition_correlation_heatmap":
        return "Portfolio addition correlation heatmap"
    if stem == "portfolio_addition_drawdown_correlation":
        return "Portfolio addition drawdown correlation"
    if stem == "validation_sharpe_comparison":
        return "IS vs validation Sharpe"
    if stem == "validation_z_cusum_rolling_sr":
        return "Validation z-score / CUSUM / rolling SR"
    if stem == "validation_equity_bands":
        return "Validation equity confidence bands"
    if stem == "validation_rank_scatter":
        return "Validation rank correlation scatter"
    if stem == "cusum_stability":
        return "6-month rolling CUSUM stability"
    if stem == "robustness_rolling_cusum":
        return "Rolling / CUSUM data"
    if stem.startswith("holdout_") and stem.endswith("_comparison"):
        return stem.replace("_", " ").title()
    if stem.startswith("equity_curve"):
        if "__ticker_" in stem:
            base, ticker = stem.split("__ticker_", 1)
            return f"{base.replace('_', ' ').title()} ({ticker})"
        return stem.replace("_", " ").title()
    return stem.replace("_", " ").replace("-", " ").title()


def friendly_panel_title(path: Path, policy: PanelPolicy) -> str:
    if policy.friendly_panel_title is not None:
        return policy.friendly_panel_title(path)
    return default_friendly_panel_title(path)


def artifact_panel(
    artifact: dict[str, object],
    *,
    index: int,
    policy: PanelPolicy,
) -> dict[str, object]:
    relative_path = str(artifact["relative_path"])
    path = Path(relative_path)
    kind = str(artifact["kind"])
    panel_id = (
        policy.panel_id_for_artifact(artifact, index)
        if policy.panel_id_for_artifact is not None
        else f"{artifact['category']}-{index}"
    )
    return {
        "panel_id": panel_id,
        "title": friendly_panel_title(path, policy),
        "view_type": panel_view_type(kind),
        "panel_kind": panel_kind(path, kind, policy),
        "kind": kind,
        "relative_path": relative_path,
        "category": str(artifact["category"]),
        "is_primary": is_featured_panel(path, kind, policy),
    }


def artifact_panel_sort_key(
    artifact: dict[str, object],
    policy: PanelPolicy,
) -> tuple[int, int, str]:
    path = Path(str(artifact["relative_path"]))
    kind = str(artifact["kind"])
    priority = {
        "html": 0,
        "image": 1,
        "table": 2,
        "markdown": 3,
        "json": 4,
        "text": 5,
    }.get(kind, 9)
    primary_bonus = 0 if is_featured_panel(path, kind, policy) else 1
    return (primary_bonus, priority, path.name.lower())
