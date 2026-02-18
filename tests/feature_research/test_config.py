from datetime import datetime
from pathlib import Path

import pytest

from feature_research.continuous_binning.config import ResearchConfig, load_config
from utils.enums import Ticker, TimeFrame


def test_load_config_returns_research_config():
    config = load_config()
    assert isinstance(config, ResearchConfig)


def test_load_config_defaults():
    config = load_config()
    assert Ticker.ES in config.tickers
    assert config.start == datetime(2000, 1, 1)
    assert config.end == datetime(2024, 12, 31)
    assert config.bias_spec["module_name"] == "rsi"
    assert isinstance(config.bias_spec["params"]["lookback"], list)
    assert config.target_col == "log_return"
    assert config.use_cache is True
    assert config.populate_cache is True


def test_reports_dir_includes_module_name():
    config = load_config()
    assert config.bias_spec["module_name"] in str(config.reports_dir)
    assert "continuous_binning" in str(config.reports_dir)
