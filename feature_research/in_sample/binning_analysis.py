"""Continuous binning diagnostics pipeline for research runs.

Note: Phase 2 Binning Analysis is only for continuous features.
Rule-based features skip this phase (they use fixed 3 levels).
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
import sys

import matplotlib

if TYPE_CHECKING:
    from feature_research.config import BinningAnalysisConfig
    from feature_research.in_sample.config import ResearchConfig

from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    param_combo_label,
    populate_cache_if_needed,
)
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validation.binning import (
    BinningSuccessCriteria,
    generate_binning_report,
    save_report,
)


def binning_model_from_config(bp: "BinningAnalysisConfig", bin_count: int) -> ContinuousBinningModel:
    """Factory to build a ContinuousBinningModel from BinningAnalysisConfig + bin_count."""
    return ContinuousBinningModel(
        n_bins=bin_count,
        bin_counts=[bin_count],
        strategy=bp.strategy,
        bin_index_min=bp.bin_index_min,
        bin_index_max=bp.bin_index_max,
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
    strategy = str(params.get("strategy", "long"))
    bin_index_min = int(params.get("bin_index_min", 0))
    bin_index_max = params.get("bin_index_max")
    if bin_index_max is not None:
        bin_index_max = int(bin_index_max)

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
            strategy=strategy,
            bin_index_min=bin_index_min,
            bin_index_max=bin_index_max,
        )
        model.fit(feature, target)

        criteria = BinningSuccessCriteria(
            metric_threshold=0.0,
            t_threshold=2.0,
        )
        report, diagnostic_plots = generate_binning_report(
            model=model,
            feature_data=feature,
            criteria=criteria,
            strategy=strategy,
            max_regions=max_regions,
            direction_filter=direction_filter,
        )

        combo_output_dir = output_dir / label
        report_path = save_report(report, diagnostic_plots, str(combo_output_dir))
        results[label] = Path(report_path)

    return results
