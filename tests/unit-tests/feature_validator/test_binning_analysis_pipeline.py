from pathlib import Path

from feature_research.continuous_binning.binning_analysis import run_binning_analysis_pipeline
from feature_research.continuous_binning.config import load_config


def test_binning_analysis_dry_run(tmp_path: Path) -> None:
    config = load_config()
    output_dir = tmp_path / "results" / config.bias_spec["module_name"]
    results = run_binning_analysis_pipeline(config, output_dir, dry_run=True)
    assert output_dir.exists()
    assert results == {}
