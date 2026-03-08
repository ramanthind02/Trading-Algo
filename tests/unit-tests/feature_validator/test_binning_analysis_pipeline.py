from dataclasses import asdict, replace
from pathlib import Path

from feature_research.in_sample.binning_analysis import (
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
        bin_counts=[7],
        strategy="long",
        bin_index_min=0,
        bin_index_max=3,
    )
    config = load_config()
    config_with_params = replace(config, binning_params=params)
    assert _extract_binning_params(config_with_params) == asdict(params)
