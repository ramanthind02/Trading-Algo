from datetime import datetime

from feature_research.config import FeatureType, ResearchConfig, load_config


def test_load_config_returns_signed_signal_research_config() -> None:
    config = load_config()
    assert isinstance(config, ResearchConfig)
    assert config.in_sample_defaults.signed_signal is not None


def test_load_config_defaults() -> None:
    config = load_config()
    assert config.tickers
    assert config.start == datetime(2000, 1, 1)
    assert config.feature_type is FeatureType.SIGNED_SIGNAL


def test_reports_dir_includes_feature_type() -> None:
    config = load_config()
    assert "signed_signal" in str(config.reports_dir)
    assert "cyclical_rsi_signal" in str(config.reports_dir)
