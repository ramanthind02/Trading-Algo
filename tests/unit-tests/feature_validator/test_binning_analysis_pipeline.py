from dataclasses import asdict
from pathlib import Path

from feature_research.in_sample.continuous_binning.binning_analysis import (
    _extract_binning_params,
    run_binning_analysis_pipeline,
)
from feature_research.in_sample.config import BinningAnalysisConfig, load_config


def test_binning_analysis_dry_run(tmp_path: Path) -> None:
    config = load_config()
    output_dir = tmp_path / "results" / config.bias_spec["module_name"]
    results = run_binning_analysis_pipeline(config, output_dir, dry_run=True)
    assert output_dir.exists()
    assert results == {}


def test_extract_binning_params_accepts_dataclass() -> None:
    params = BinningAnalysisConfig(
        n_bins=7,
        selection_metric="sharpe",
        strategy="long",
        metric_threshold=0.2,
        t_threshold=1.5,
        min_region_width=2,
        max_regions=1,
        direction_filter="both",
    )
    config = load_config()
    config = config.__class__(
        tickers=config.tickers,
        start=config.start,
        end=config.end,
        bias_spec=config.bias_spec,
        target_col=config.target_col,
        strategy=config.strategy,
        binning_params=params,
        use_cache=config.use_cache,
        populate_cache=config.populate_cache,
        reports_dir=config.reports_dir,
    )
    assert _extract_binning_params(config) == asdict(params)
