from feature_research.continuous_binning.config import load_config


def test_binning_analysis_config_fields() -> None:
    config = load_config()
    assert isinstance(config.binning_params.bin_counts, list)
    assert all(isinstance(n, int) for n in config.binning_params.bin_counts)
    assert all(n > 0 for n in config.binning_params.bin_counts)
    assert config.binning_params.metric_threshold >= 0.0
    assert config.binning_params.use_coverage_bonus in [True, False]
    assert config.binning_params.coverage_bonus_per_10pct >= 0.0
    assert config.binning_params.max_coverage_bonus >= 0.0
