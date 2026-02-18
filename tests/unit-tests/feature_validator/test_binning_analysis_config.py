from feature_research.continuous_binning.config import load_config


def test_binning_analysis_config_fields() -> None:
    config = load_config()
    assert config.binning_params.n_bins > 0
    assert config.binning_params.metric_threshold >= 0.0
