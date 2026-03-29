from datetime import datetime

from feature_research.config import FeatureType
from feature_research.in_sample.config import ResearchConfig, load_config
from utils.core.enums import Ticker


def test_load_config_returns_signed_signal_research_config() -> None:
    config = load_config()
    assert isinstance(config, ResearchConfig)
    assert config.in_sample_defaults.signed_signal is not None


def test_load_config_defaults() -> None:
    config = load_config()
    assert Ticker.ES in config.tickers
    assert config.start == datetime(2000, 1, 1)
    assert config.feature_type is FeatureType.CONTINUOUS


def test_reports_dir_includes_feature_type() -> None:
    config = load_config()
    assert "continuous" in str(config.reports_dir)
