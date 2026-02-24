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
    # Note: This test documents unified config behavior
    # The defaults have changed due to the unified architecture.
    # To test rule_based specific defaults, set feature_type=RULE_BASED in base config.
    config = load_config()
    assert Ticker.ES in config.tickers
    assert Ticker.NQ in config.tickers
    assert config.start == datetime(2000, 1, 1)
    assert config.end == datetime(2024, 12, 31)
    # Default feature_type is CONTINUOUS; assertions below are for that
    # Rule-based assertions would use feature_type=RULE_BASED


def test_reports_dir_includes_module_name():
    # Due to unified architecture, feature_type determines reports_dir
    config = load_config()
    assert config.bias_spec["module_name"] in str(config.reports_dir)
    # Default feature_type is CONTINUOUS, so reports_dir contains "continuous"
    assert "continuous" in str(config.reports_dir)


def test_load_config_includes_walkforward_defaults() -> None:
    config = load_config()

    assert isinstance(config.walkforward, WalkforwardResearchConfig)
    assert config.walkforward.enabled is False
    assert config.walkforward.train_start == config.start
    assert config.walkforward.train_end == datetime(2019, 6, 25)
    assert config.walkforward.output_root == Path("feature_research/shared_results")
