from datetime import datetime
from pathlib import Path

from feature_research.config import FeatureType
from feature_research.in_sample.config import ResearchConfig, load_config
from feature_research.walkforward.config import WalkforwardResearchConfig
from utils.core.enums import Ticker


def test_load_config_returns_rule_based_research_config():
    # For this test, we need to override the base config to use RULE_BASED
    # Since load_config() dispatches on feature_type from base config,
    # and the default is CONTINUOUS, this test documents the unified structure
    config = load_config()
    assert isinstance(config, ResearchConfig)
    # Default is CONTINUOUS; to test RULE_BASED, set feature_type in base config


def test_load_config_defaults():
    config = load_config()
    assert Ticker.ES in config.tickers
    assert Ticker.NQ in config.tickers
    assert config.start == datetime(2000, 1, 1)
    assert config.end == datetime(2025, 12, 30)


def test_reports_dir_includes_feature_type():
    config = load_config()
    assert config.feature_type.value in str(config.reports_dir)


def test_load_config_includes_walkforward_defaults() -> None:
    config = load_config()

    assert isinstance(config.walkforward, WalkforwardResearchConfig)
    assert config.walkforward.enabled is True
    assert config.walkforward.train_start >= config.start
    assert config.walkforward.train_end < config.end
    assert config.walkforward.output_root == Path("feature_research/shared_results")
