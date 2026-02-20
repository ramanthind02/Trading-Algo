"""Continuous binning diagnostics pipeline for research runs."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
import sys

import matplotlib

if TYPE_CHECKING:
    from feature_research.continuous_binning.config import ResearchConfig

from feature_research.continuous_binning.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    param_combo_label,
    populate_cache_if_needed,
)
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validators.binning import (
    BinningSuccessCriteria,
    generate_binning_report,
    save_report,
)


def _extract_binning_params(config: "ResearchConfig") -> dict[str, Any]:
    raw = getattr(config, "binning_params", None)
    if isinstance(raw, dict):
        return raw
    if raw is not None and not isinstance(raw, type) and is_dataclass(raw):
        return asdict(cast(Any, raw))
    return {}


def run_binning_analysis_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Path]:
    """Run binning diagnostics for each parameter combo in config.

    Parameters
    ----------
    config : ResearchConfig
        Researcher-defined settings (tickers, dates, bias_spec, cache flags).
    output_dir : Path
        Root directory for output reports. Created if it does not exist.
        Each param combo writes to ``output_dir / param_label /``.
    dry_run : bool
        When True, only creates output directories without extracting data.

    Returns
    -------
    dict[str, Path]
        Mapping of ``param_label`` → saved report path for each successful combo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    expanded = expand_bias_specs(config.bias_spec)
    for single_spec in expanded:
        combo_output_dir = output_dir / param_combo_label(single_spec["params"])
        combo_output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return {}

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")  # non-interactive backend (safe for scripts and tests)

    populate_cache_if_needed(config)

    params = _extract_binning_params(config)
    bin_counts = params.get("bin_counts", [10, 8, 5, 3])
    selection_metric = str(params.get("selection_metric", "sharpe"))
    strategy = str(params.get("strategy", "long"))
    metric_threshold = float(params.get("metric_threshold", 0.0))
    t_threshold = float(params.get("t_threshold", 2.0))
    min_region_width = int(params.get("min_region_width", 2))
    use_coverage_bonus = params.get("use_coverage_bonus", False)
    coverage_bonus_per_10pct = float(params.get("coverage_bonus_per_10pct", 0.02))
    max_coverage_bonus = float(params.get("max_coverage_bonus", 0.2))
    max_regions = int(params.get("max_regions", 1))
    direction_filter = str(params.get("direction_filter", "both"))

    results: dict[str, Path] = {}

    for single_spec in expanded:
        combo = single_spec["params"]
        label = param_combo_label(combo)

        data = load_features_for_combo(single_spec, config)
        if data is None:
            continue

        feature, target, feature_col = data
        if feature.name != feature_col:
            feature = feature.copy()
            feature.name = feature_col

        model = ContinuousBinningModel(
            bin_counts=bin_counts,
            selection_metric=selection_metric,
            strategy=strategy,
            metric_threshold=metric_threshold,
            t_threshold=t_threshold,
            min_region_width=min_region_width,
            use_coverage_bonus=use_coverage_bonus,
            coverage_bonus_per_10pct=coverage_bonus_per_10pct,
            max_coverage_bonus=max_coverage_bonus,
            shrinkage_k=float(params.get("shrinkage_k", 20.0)),
            long_clip_min=float(params.get("long_clip_min", 0.5)),
            long_clip_max=float(params.get("long_clip_max", 2.0)),
            short_clip_min=float(params.get("short_clip_min", 0.5)),
            short_clip_max=float(params.get("short_clip_max", 2.0)),
        )
        model.fit(feature, target)

        criteria = BinningSuccessCriteria(
            metric_threshold=metric_threshold,
            t_threshold=t_threshold,
            min_region_width=min_region_width,
        )
        report = generate_binning_report(
            model=model,
            feature_data=feature,
            criteria=criteria,
            strategy=strategy,
            max_regions=max_regions,
            direction_filter=direction_filter,
        )

        combo_output_dir = output_dir / label
        report_path = save_report(report, str(combo_output_dir))
        results[label] = Path(report_path)

    return results
