"""Discover portfolio holdout workspace artifacts."""
from __future__ import annotations

from enum import Enum
from pathlib import Path

from research.portfolio.config import PortfolioResearchConfig
from research.portfolio.shared.phase import PortfolioResearchPhase
from research.portfolio.shared.visualization_paths import holdout_root
from lib.core.repo_bootstrap import require_repo_root, resolve_repo_path
from research.workspace.discovery import discover_artifact_records as _discover_records
from research.workspace.discovery import group_artifact_records as _group_records
from research.workspace.models import CategoryGroup, PanelPolicy
from research.workspace.panels import default_friendly_panel_title
from research.workspace.sections import build_phase_report_sections as _build_sections

_PORTFOLIO_REPO_ROOT = require_repo_root(Path(__file__))
_REPO_ROOT = Path(__file__).resolve().parents[3]

_SUMMARY_JSON_NAMES = frozenset(
    {
        "holdout_robustness_report.json",
        "portfolio_holdout_report.json",
        "fold_manifest.json",
        "monitoring_rollup.json",
        "monitoring_status.json",
        "weight_layer_report.json",
    }
)


class WorkspaceArtifactCategory(str, Enum):
    QUANTSTATS_TEARSHEET = "quantstats_tearsheet"
    MATPLOTLIB_REPORT = "matplotlib_report"
    ROBUSTNESS = "robustness"
    PORTFOLIO_ANALYTICS = "portfolio_analytics"
    WEIGHT_LAYER = "weight_layer"
    RETURNS = "returns"
    FOLD_MANIFEST = "fold_manifest"
    PROP_FIRM_REPORTS = "prop_firm_reports"
    OTHER = "other"


_CATEGORY_GROUPS: tuple[CategoryGroup, ...] = (
    CategoryGroup(
        WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET.value,
        "QuantStats Tearsheets",
        "HTML portfolio tearsheets from validation and test folds.",
        0,
    ),
    CategoryGroup(
        WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
        "Matplotlib Reports",
        "PNG monitoring charts for strategies and portfolio holdout.",
        1,
    ),
    CategoryGroup(
        WorkspaceArtifactCategory.ROBUSTNESS.value,
        "Robustness Summaries",
        "Per-strategy holdout monitoring JSON, markdown, and CSV exports.",
        2,
    ),
    CategoryGroup(
        WorkspaceArtifactCategory.PORTFOLIO_ANALYTICS.value,
        "Portfolio Analytics",
        "Portfolio-level holdout report, correlation, and contribution outputs.",
        3,
    ),
    CategoryGroup(
        WorkspaceArtifactCategory.WEIGHT_LAYER.value,
        "Weight Layer",
        "Cross-strategy weight layer weights from holdout evaluation.",
        4,
    ),
    CategoryGroup(
        WorkspaceArtifactCategory.RETURNS.value,
        "Returns",
        "Research and holdout return matrices exported as CSV.",
        5,
    ),
    CategoryGroup(
        WorkspaceArtifactCategory.FOLD_MANIFEST.value,
        "Fold Manifest",
        "Two-fold holdout fit/score window definitions.",
        6,
    ),
    CategoryGroup(
        WorkspaceArtifactCategory.PROP_FIRM_REPORTS.value,
        "Prop Firm Reports",
        "FundedNext CFD portfolio simulation HTML/MD reports and CSV exports per phase.",
        7,
    ),
    CategoryGroup(
        WorkspaceArtifactCategory.OTHER.value,
        "Other Artifacts",
        "Additional files discovered under holdout output roots.",
        8,
    ),
)

_GROUP_BY_ID = {group.category: group for group in _CATEGORY_GROUPS}


def _strategy_name_from_path(path: Path) -> str | None:
    posix = path.as_posix()
    marker = "/strategies/"
    if marker not in posix:
        return None
    tail = posix.split(marker, 1)[1]
    strategy = tail.split("/", 1)[0]
    return strategy or None


def _portfolio_friendly_panel_title(path: Path) -> str:
    normalized = path.as_posix().lower()
    if "/prop_firm/" in normalized:
        parts = [part for part in path.parts if part]
        firm = next(
            (parts[i + 1] for i, part in enumerate(parts) if part == "prop_firm"),
            "prop firm",
        )
        phase = next(
            (part for part in parts if part in {"train", "validation", "test"}),
            "",
        )
        label = f"{firm} — {phase} report" if phase else f"{firm} report"
        return label.replace("_", " ").title()
    strategy = _strategy_name_from_path(path)
    if "full_period_tearsheet" in path.name.lower():
        label = f"{strategy} — full period tearsheet" if strategy else "Full period tearsheet"
        return label.replace("_", " ")
    if path.name.lower() == "weight_layer_report.json":
        parts = path.parts
        phase = next(
            (part for part in parts if part in {"train", "validation", "test", "holdout"}),
            "portfolio",
        )
        return f"Weight layer — {phase.replace('_', ' ')}"
    chart_title = default_friendly_panel_title(path)
    if strategy is None:
        return chart_title
    return f"{strategy} — {chart_title}"


def _portfolio_panel_id(artifact: dict[str, object], index: int) -> str:
    _ = index
    relative = str(artifact["relative_path"]).replace("\\", "/")
    safe = relative.replace("/", "__").replace(".", "_")
    return safe


def _is_featured_panel(path: Path, kind: str) -> bool:
    name = path.name.lower()
    normalized = path.as_posix().lower()
    if kind == "html" and "tearsheet" in name:
        return True
    if kind == "html" and "/prop_firm/" in normalized:
        return True
    if kind == "json" and name in _SUMMARY_JSON_NAMES:
        return True
    if name in {"monitoring_rollup.csv", "monitoring_history.csv"}:
        return True
    if kind == "html" and "full_period_tearsheet" in name:
        return True
    if kind == "image" and (
        name.startswith("holdout_")
        or name.startswith("portfolio_")
        or name.startswith("correlation")
        or name.startswith("contribution")
    ):
        return True
    return False


_PORTFOLIO_PANEL_POLICY = PanelPolicy(
    summary_json_names=_SUMMARY_JSON_NAMES,
    always_visible_categories=frozenset(
        {
            WorkspaceArtifactCategory.ROBUSTNESS.value,
            WorkspaceArtifactCategory.PORTFOLIO_ANALYTICS.value,
            WorkspaceArtifactCategory.FOLD_MANIFEST.value,
        }
    ),
    is_featured_panel=_is_featured_panel,
    friendly_panel_title=_portfolio_friendly_panel_title,
    panel_id_for_artifact=_portfolio_panel_id,
)


def category_groups_for_phase(phase: PortfolioResearchPhase) -> tuple[CategoryGroup, ...]:
    if phase is PortfolioResearchPhase.FULL_PIPELINE:
        return _CATEGORY_GROUPS
    if phase is PortfolioResearchPhase.PROP_FIRM_REPORTS:
        allowed = {
            WorkspaceArtifactCategory.PROP_FIRM_REPORTS.value,
            WorkspaceArtifactCategory.OTHER.value,
        }
        return tuple(group for group in _CATEGORY_GROUPS if group.category in allowed)
    if phase is PortfolioResearchPhase.STRATEGY_HOLDOUT:
        excluded = {
            WorkspaceArtifactCategory.PORTFOLIO_ANALYTICS.value,
            WorkspaceArtifactCategory.WEIGHT_LAYER.value,
            WorkspaceArtifactCategory.RETURNS.value,
            WorkspaceArtifactCategory.FOLD_MANIFEST.value,
            WorkspaceArtifactCategory.PROP_FIRM_REPORTS.value,
        }
        return tuple(group for group in _CATEGORY_GROUPS if group.category not in excluded)
    if phase is PortfolioResearchPhase.PORTFOLIO_HOLDOUT:
        excluded = {
            WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET.value,
            WorkspaceArtifactCategory.WEIGHT_LAYER.value,
            WorkspaceArtifactCategory.PROP_FIRM_REPORTS.value,
        }
        return tuple(group for group in _CATEGORY_GROUPS if group.category not in excluded)
    # PORTFOLIO_TEST: exclude portfolio analytics and prop-firm (prop-firm has its own tab)
    excluded = {
        WorkspaceArtifactCategory.PORTFOLIO_ANALYTICS.value,
        WorkspaceArtifactCategory.PROP_FIRM_REPORTS.value,
    }
    return tuple(group for group in _CATEGORY_GROUPS if group.category not in excluded)


def phase_discovery_roots(
    config: PortfolioResearchConfig,
    phase: PortfolioResearchPhase,
) -> tuple[Path, ...]:
    output_root = resolve_repo_path(config.output_root, repo_root=_PORTFOLIO_REPO_ROOT)
    holdout = holdout_root(output_root)
    match phase:
        case PortfolioResearchPhase.FULL_PIPELINE:
            candidates = (
                output_root / "train",
                output_root / "validation",
                output_root / "test",
                output_root / "combined",
                output_root / "train" / "prop_firm",
                output_root / "validation" / "prop_firm",
                output_root / "test" / "prop_firm",
                holdout / "returns",
                holdout / "strategies",
                holdout / "visualization" / "portfolio",
                holdout / "fold_manifest.json",
            )
        case PortfolioResearchPhase.PORTFOLIO_TEST:
            candidates = (
                output_root / "train",
                output_root / "validation",
                output_root / "test",
                output_root / "combined",
                output_root / "train" / "prop_firm",
                output_root / "validation" / "prop_firm",
                output_root / "test" / "prop_firm",
                holdout / "returns",
            )
        case PortfolioResearchPhase.PROP_FIRM_REPORTS:
            candidates = (
                output_root / "train" / "prop_firm",
                output_root / "validation" / "prop_firm",
                output_root / "test" / "prop_firm",
            )
        case PortfolioResearchPhase.STRATEGY_HOLDOUT:
            candidates = (holdout / "strategies", holdout)
        case PortfolioResearchPhase.PORTFOLIO_HOLDOUT:
            candidates = (
                holdout / "visualization" / "portfolio",
                holdout / "fold_manifest.json",
            )
        case _:
            candidates = (holdout,)
    return tuple(path for path in candidates if path.exists())


def categorize_artifact_path(path: Path) -> WorkspaceArtifactCategory:
    name = path.name.lower()
    normalized = path.as_posix().lower()
    suffix = path.suffix.lower()

    if name == "fold_manifest.json":
        return WorkspaceArtifactCategory.FOLD_MANIFEST
    if "/prop_firm/" in normalized:
        return WorkspaceArtifactCategory.PROP_FIRM_REPORTS
    if suffix == ".html" and "tearsheet" in name:
        return WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET
    if suffix == ".png" and "/matplotlib/" in normalized:
        return WorkspaceArtifactCategory.MATPLOTLIB_REPORT
    if "weight_layer" in name:
        return WorkspaceArtifactCategory.WEIGHT_LAYER
    if "/returns/" in normalized or name.startswith("portfolio_returns") or name.startswith(
        "strategy_returns"
    ):
        return WorkspaceArtifactCategory.RETURNS
    if name == "portfolio_holdout_report.json" or (
        "correlation" in name or "contribution" in name or "idm" in name
    ) and "/visualization/portfolio/" in normalized:
        return WorkspaceArtifactCategory.PORTFOLIO_ANALYTICS
    if (
        "holdout_robustness" in name
        or "holdout_z_cusum" in name
        or "holdout_equity" in name
        or name.startswith("monitoring_")
    ):
        return WorkspaceArtifactCategory.ROBUSTNESS
    if suffix == ".png":
        return WorkspaceArtifactCategory.MATPLOTLIB_REPORT
    return WorkspaceArtifactCategory.OTHER


def _skip_legacy_portfolio_monitoring_duplicate(path: Path) -> bool:
    """Ignore nested holdout_* PNGs superseded by portfolio_* charts in matplotlib/."""

    normalized = path.as_posix().lower()
    return "/portfolio_monitoring/" in normalized and path.suffix.lower() == ".png"


def discover_artifact_records(
    config: PortfolioResearchConfig,
    phase: PortfolioResearchPhase,
) -> list[dict[str, object]]:
    records = _discover_records(
        phase_discovery_roots(config, phase),
        phase=phase.value,
        relative_to=_REPO_ROOT,
        categorize=lambda path: categorize_artifact_path(path).value,
        category_label=lambda category: _GROUP_BY_ID[category].label,
    )
    if phase is not PortfolioResearchPhase.PORTFOLIO_HOLDOUT:
        return records
    return [
        record
        for record in records
        if not _skip_legacy_portfolio_monitoring_duplicate(Path(str(record["relative_path"])))
    ]


def group_artifact_records(
    artifacts: list[dict[str, object]],
    phase: PortfolioResearchPhase,
) -> list[dict[str, object]]:
    return _group_records(artifacts, category_groups_for_phase(phase))


def build_phase_report_sections(
    grouped_artifacts: list[dict[str, object]],
    phase: PortfolioResearchPhase,
) -> tuple[list[dict[str, object]], str | None]:
    return _build_sections(
        grouped_artifacts,
        policy=_PORTFOLIO_PANEL_POLICY,
        default_section_priority=_phase_section_priority(phase),
    )


def _phase_section_priority(phase: PortfolioResearchPhase) -> tuple[str, ...]:
    if phase is PortfolioResearchPhase.PROP_FIRM_REPORTS:
        return (WorkspaceArtifactCategory.PROP_FIRM_REPORTS.value,)
    if phase is PortfolioResearchPhase.STRATEGY_HOLDOUT:
        return (
            "monitoring_status",
            WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET.value,
            WorkspaceArtifactCategory.ROBUSTNESS.value,
            WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
        )
    if phase is PortfolioResearchPhase.PORTFOLIO_HOLDOUT:
        return (
            WorkspaceArtifactCategory.PORTFOLIO_ANALYTICS.value,
            WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
            WorkspaceArtifactCategory.ROBUSTNESS.value,
        )
    return (
        WorkspaceArtifactCategory.WEIGHT_LAYER.value,
        WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET.value,
        WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
        WorkspaceArtifactCategory.RETURNS.value,
        WorkspaceArtifactCategory.FOLD_MANIFEST.value,
    )
