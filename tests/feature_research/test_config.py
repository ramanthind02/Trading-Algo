from datetime import datetime
from pathlib import Path

import pytest

from feature_research.continuous_binning.config import ResearchConfig, load_config
from feature_research.walkforward.config import WalkforwardResearchConfig
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
    assert config.target_col == "log_return_atr"
    assert config.use_cache is True
    assert config.populate_cache is True


def test_reports_dir_includes_module_name():
    config = load_config()
    assert config.bias_spec["module_name"] in str(config.reports_dir)
    assert "continuous_binning" in str(config.reports_dir)


def test_load_config_includes_walkforward_defaults() -> None:
    config = load_config()

    assert isinstance(config.walkforward, WalkforwardResearchConfig)
    assert config.walkforward.enabled is False
    assert config.walkforward.train_start == config.start
    assert config.walkforward.train_end < config.end
    assert config.walkforward.output_root == Path("feature_research/shared_results")


def _make_research_config(*, tickers: list[Ticker], target_col: str) -> ResearchConfig:
    return ResearchConfig(
        tickers=tickers,
        start=datetime(2000, 1, 1),
        end=datetime(2024, 12, 31),
        bias_spec={
            "module_name": "rsi",
            "timeframes": [TimeFrame.D],
            "params": {"lookback": 5},
        },
        target_col=target_col,
        strategy="long",
        use_cache=True,
        populate_cache=False,
        reports_dir=Path("/tmp/test_reports"),
    )


def test_research_config_rejects_log_return_with_multiple_tickers() -> None:
    with pytest.raises(ValueError, match="log_return.*multiple tickers"):
        _make_research_config(tickers=[Ticker.ES, Ticker.NQ], target_col="log_return")


def test_research_config_rejects_raw_return_with_multiple_tickers() -> None:
    with pytest.raises(ValueError, match="raw_return.*multiple tickers"):
        _make_research_config(
            tickers=[Ticker.ES, Ticker.NQ, Ticker.YM],
            target_col="raw_return",
        )


def test_research_config_allows_log_return_with_single_ticker() -> None:
    config = _make_research_config(tickers=[Ticker.ES], target_col="log_return")
    assert config.target_col == "log_return"


def test_research_config_allows_normalized_targets_with_multiple_tickers() -> None:
    ewsd_config = _make_research_config(
        tickers=[Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY],
        target_col="log_return_ewsd",
    )
    atr_config = _make_research_config(
        tickers=[Ticker.ES, Ticker.NQ],
        target_col="log_return_atr",
    )

    assert ewsd_config.target_col == "log_return_ewsd"
    assert atr_config.target_col == "log_return_atr"
