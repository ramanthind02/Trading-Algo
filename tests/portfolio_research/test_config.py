from datetime import datetime
from pathlib import Path

import pytest

from research.portfolio.config import (
    EnsembleDirsPolicy,
    PortfolioResearchConfig,
    ResearchWindow,
    filter_ensemble_dirs_for_portfolio_tickers,
    load_prop_firm_portfolio_research_config,
    scoped_tickers_for_ensemble_dirs,
)
from lib.core.enums import Ticker, TimeFrame


def test_research_window_order_validation() -> None:
    train = ResearchWindow(start=datetime(2000, 1, 1), end=datetime(2005, 12, 31))
    validation = ResearchWindow(start=datetime(2006, 1, 1), end=datetime(2010, 12, 31))
    test = ResearchWindow(start=datetime(2011, 1, 1), end=datetime(2015, 12, 31))

    config = PortfolioResearchConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2015, 12, 31),
        use_cache=True,
        train_window=train,
        validation_window=validation,
        test_window=test,
        ensemble_dirs={"dummy": "vault/D/dummy"},
    )

    assert config.train_window is train
    assert config.validation_window is validation
    assert config.test_window is test


def test_invalid_window_order_raises() -> None:
    train = ResearchWindow(start=datetime(2000, 1, 1), end=datetime(2010, 12, 31))
    validation = ResearchWindow(start=datetime(2005, 1, 1), end=datetime(2012, 12, 31))
    test = ResearchWindow(start=datetime(2013, 1, 1), end=datetime(2015, 12, 31))

    with pytest.raises(ValueError, match="train_window.end must be before"):
        PortfolioResearchConfig(
            tickers=[Ticker.ES],
            timeframe=TimeFrame.D,
            start=datetime(2000, 1, 1),
            end=datetime(2015, 12, 31),
            use_cache=True,
            train_window=train,
            validation_window=validation,
            test_window=test,
            ensemble_dirs={"dummy": "vault/D/dummy"},
        )


def test_discover_ensemble_dirs_uses_vault_relative_paths(tmp_path: Path, monkeypatch) -> None:
    vault_root = tmp_path / "vault"
    d_dir = vault_root / "D"
    ensemble_dir = d_dir / "example_ensemble_long"
    features_dir = ensemble_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "feature.json").write_text("{}", encoding="utf-8")

    # Point the config module's _PORTFOLIO_RESEARCH_DIR parent to our temp root
    # and import the helper directly to avoid interacting with real vault/.
    import research.portfolio.config as cfg

    monkeypatch.setattr(cfg, "_PORTFOLIO_RESEARCH_DIR", tmp_path / "research.portfolio")

    discovered = cfg._discover_ensemble_dirs()

    assert "example_ensemble_long" in discovered
    expected = str(Path("vault/D/example_ensemble_long"))
    assert discovered["example_ensemble_long"].startswith(expected)


def test_discover_ensemble_dirs_supports_nested_weight_group_folders(tmp_path: Path, monkeypatch) -> None:
    vault_root = tmp_path / "vault"
    nested = vault_root / "D" / "momentum" / "my_strat_long" / "features"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "feature.json").write_text("{}", encoding="utf-8")

    import research.portfolio.config as cfg

    monkeypatch.setattr(cfg, "_PORTFOLIO_RESEARCH_DIR", tmp_path / "research.portfolio")

    discovered = cfg._discover_ensemble_dirs()
    assert Path(discovered["my_strat_long"]) == Path("vault/D/momentum/my_strat_long")


def _patch_feature_iter(
    monkeypatch: pytest.MonkeyPatch,
    handler: object,
) -> None:
    import research.portfolio.config as cfg

    monkeypatch.setattr(
        cfg._vault_feature_files,
        "iter_validated_feature_configs",
        handler,
    )


def test_filter_keeps_rebalancing_es_when_only_cross_ticker_is_outside_book(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ES-traded rebalancing_flow with TLT cross peer stays on ES-only portfolios."""

    def _iter(features_dir: Path):
        path_s = str(features_dir).replace("\\", "/")
        if "rebalancing_es_tlt_long" in path_s:
            yield Path("x.json"), {
                "bias_node_spec": {
                    "module_name": "rebalancing_flow",
                    "params": {"flow": "both", "cross_tickers": ["TLT"]},
                },
                "tickers": ["ES"],
            }
        else:
            yield Path("y.json"), {
                "bias_node_spec": {"module_name": "turnaround", "params": {}},
                "tickers": ["ES"],
            }

    _patch_feature_iter(monkeypatch, _iter)
    out = filter_ensemble_dirs_for_portfolio_tickers(
        {
            "tlt_cross": "vault/D/es_tlt/rebalancing_es_tlt_long",
            "es_local": "vault/D/mean_reversion_indices/mr_indices_long",
        },
        [Ticker.ES],
    )
    assert set(out.keys()) == {"tlt_cross", "es_local"}


def test_filter_drops_rebalancing_when_primary_ticker_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rebalancing_flow TLT leg is in feature tickers, not only cross_tickers."""
    import research.portfolio.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_PORTFOLIO_RESEARCH_DIR", tmp_path / "research.portfolio")
    for rel in (
        "vault/D/es_tlt/rebalancing_tlt_es_long/features",
        "vault/D/mean_reversion_indices/mr_indices_long/features",
    ):
        (tmp_path / Path(rel)).mkdir(parents=True, exist_ok=True)

    def _iter(features_dir: Path):
        path_s = str(features_dir).replace("\\", "/")
        if "rebalancing_tlt_es_long" in path_s:
            yield Path("x.json"), {
                "bias_node_spec": {
                    "module_name": "rebalancing_flow",
                    "params": {"flow": "reversal", "cross_tickers": ["ES"]},
                },
                "tickers": ["TLT"],
            }
        else:
            yield Path("y.json"), {
                "bias_node_spec": {"module_name": "turnaround", "params": {}},
                "tickers": ["ES", "NQ"],
            }

    _patch_feature_iter(monkeypatch, _iter)
    out = filter_ensemble_dirs_for_portfolio_tickers(
        {
            "tlt_primary": "vault/D/es_tlt/rebalancing_tlt_es_long",
            "es_only": "vault/D/mean_reversion_indices/mr_indices_long",
        },
        [Ticker.ES, Ticker.NQ],
    )
    assert list(out.keys()) == ["es_only"]


def test_filter_drops_mr_indices_when_portfolio_is_gc_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MR indices ensembles trade ES+NQ per ensemble_config; must not load on GC-only books."""

    def _iter(_features_dir: Path):
        yield Path("turnaround.json"), {
            "bias_node_spec": {"module_name": "turnaround_tuesday", "params": {}},
            "tickers": ["ES", "NQ"],
        }

    _patch_feature_iter(monkeypatch, _iter)

    def _fake_ensemble_symbols(path: str) -> frozenset[str]:
        if "mr_indices_long" in path:
            return frozenset({"ES", "NQ"})
        return frozenset()

    monkeypatch.setattr(
        "research.portfolio.config._ensemble_config_ticker_symbols",
        _fake_ensemble_symbols,
    )
    out = filter_ensemble_dirs_for_portfolio_tickers(
        {
            "mr_indices_long": "vault/D/mean_reversion_indices/mr_indices_long",
            "sma_regime": "vault/D/momentum/sma_regime_long_short_long_short",
        },
        [Ticker.GC],
    )
    assert "mr_indices_long" not in out


def test_filter_keeps_ensemble_when_extra_tickers_only_in_config_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """buy_hold lists many tickers; portfolio subset is still allowed."""

    def _iter(_features_dir: Path):
        yield Path("x.json"), {
            "bias_node_spec": {"module_name": "buy_hold", "params": {}},
            "tickers": ["ES", "GC", "NQ", "RTY", "TLT"],
        }

    _patch_feature_iter(monkeypatch, _iter)
    out = filter_ensemble_dirs_for_portfolio_tickers(
        {
            "wide_universe": "vault/M/buy_hold/buy_hold_long",
        },
        [Ticker.ES, Ticker.NQ],
    )
    assert list(out.keys()) == ["wide_universe"]


def test_filter_drops_single_instrument_feature_without_that_ticker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bonds/TLT-only features must not pass when TLT is absent (no cross_tickers)."""

    import research.portfolio.config as cfg

    monkeypatch.setattr(cfg, "_PORTFOLIO_RESEARCH_DIR", tmp_path / "research.portfolio")
    for rel in (
        "vault/D/seasonal/seasonal_bonds_long_short/features",
        "vault/D/mean_reversion_indices/mr_indices_long/features",
    ):
        (tmp_path / Path(rel)).mkdir(parents=True, exist_ok=True)

    def _iter(features_dir: Path):
        path_s = str(features_dir).replace("\\", "/")
        if "seasonal_bonds" in path_s:
            yield Path("x.json"), {
                "bias_node_spec": {"module_name": "seasonal_bonds_month", "params": {}},
                "tickers": ["TLT"],
            }
        else:
            yield Path("y.json"), {
                "bias_node_spec": {"module_name": "turnaround", "params": {}},
                "tickers": ["ES"],
            }

    _patch_feature_iter(monkeypatch, _iter)
    out = filter_ensemble_dirs_for_portfolio_tickers(
        {
            "bonds": "vault/D/seasonal/seasonal_bonds_long_short",
            "es_feat": "vault/D/mean_reversion_indices/mr_indices_long",
        },
        [Ticker.ES],
    )
    assert list(out.keys()) == ["es_feat"]


def test_load_prop_firm_portfolio_research_config_disables_vault_refit() -> None:
    cfg = load_prop_firm_portfolio_research_config()
    assert cfg.ensemble_vault_refit is False


def test_portfolio_research_config_tearsheet_export_defaults_enabled() -> None:
    train = ResearchWindow(start=datetime(2000, 1, 1), end=datetime(2005, 12, 31))
    validation = ResearchWindow(start=datetime(2006, 1, 1), end=datetime(2010, 12, 31))
    test = ResearchWindow(start=datetime(2011, 1, 1), end=datetime(2015, 12, 31))
    cfg = PortfolioResearchConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2015, 12, 31),
        use_cache=True,
        train_window=train,
        validation_window=validation,
        test_window=test,
        ensemble_dirs={"dummy": "vault/D/dummy"},
    )
    assert cfg.export_per_timeframe_tearsheets is True
    assert cfg.export_per_ensemble_tearsheets is True


def test_portfolio_research_config_empty_ensemble_dirs_allowed_when_policy_set() -> None:
    cfg = PortfolioResearchConfig(
        tickers=[Ticker.GC],
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2023, 12, 31),
        use_cache=True,
        train_window=ResearchWindow(
            start=datetime(2000, 1, 1),
            end=datetime(2005, 12, 31),
        ),
        validation_window=ResearchWindow(
            start=datetime(2006, 1, 1),
            end=datetime(2010, 12, 31),
        ),
        test_window=ResearchWindow(
            start=datetime(2011, 1, 1),
            end=datetime(2023, 12, 31),
        ),
        ensemble_dirs={},
        ensemble_dirs_policy=EnsembleDirsPolicy.ALLOW_EMPTY,
    )
    assert cfg.ensemble_dirs == {}


def test_portfolio_research_config_empty_ensemble_dirs_raises() -> None:
    with pytest.raises(ValueError, match="ensemble_dirs must be non-empty"):
        PortfolioResearchConfig(
            tickers=[Ticker.ES],
            timeframe=TimeFrame.D,
            start=datetime(2000, 1, 1),
            end=datetime(2023, 12, 31),
            use_cache=True,
            train_window=ResearchWindow(
                start=datetime(2000, 1, 1),
                end=datetime(2005, 12, 31),
            ),
            validation_window=ResearchWindow(
                start=datetime(2006, 1, 1),
                end=datetime(2010, 12, 31),
            ),
            test_window=ResearchWindow(
                start=datetime(2011, 1, 1),
                end=datetime(2015, 12, 31),
            ),
            ensemble_dirs={},
        )


def test_portfolio_research_config_invalid_baseline_mode_raises() -> None:
    with pytest.raises(ValueError, match="baseline_mode must be"):
        PortfolioResearchConfig(
            tickers=[Ticker.ES],
            timeframe=TimeFrame.D,
            start=datetime(2000, 1, 1),
            end=datetime(2023, 12, 31),
            use_cache=True,
            train_window=ResearchWindow(
                start=datetime(2000, 1, 1),
                end=datetime(2005, 12, 31),
            ),
            validation_window=ResearchWindow(
                start=datetime(2006, 1, 1),
                end=datetime(2010, 12, 31),
            ),
            test_window=ResearchWindow(
                start=datetime(2011, 1, 1),
                end=datetime(2015, 12, 31),
            ),
            ensemble_dirs={"a": "vault/D/some_ensemble"},
            baseline_mode="invalid",
        )


def test_portfolio_research_config_rejects_removed_sector_surface() -> None:
    with pytest.raises(TypeError):
        PortfolioResearchConfig(
            tickers=[Ticker.ES],
            timeframe=TimeFrame.D,
            start=datetime(2000, 1, 1),
            end=datetime(2023, 12, 31),
            use_cache=True,
            train_window=ResearchWindow(
                start=datetime(2000, 1, 1),
                end=datetime(2005, 12, 31),
            ),
            validation_window=ResearchWindow(
                start=datetime(2006, 1, 1),
                end=datetime(2010, 12, 31),
            ),
            test_window=ResearchWindow(
                start=datetime(2011, 1, 1),
                end=datetime(2015, 12, 31),
            ),
            ensemble_dirs={"a": "vault/D/some_ensemble"},
            sector_allocation_config_path="feature_research/config/sector.json",  # type: ignore[call-arg]
        )


def test_scoped_tickers_for_gc_only_ensemble(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import research.portfolio.config as cfg_mod

    repo = tmp_path
    (repo / "portfolio_research").mkdir()
    monkeypatch.setattr(cfg_mod, "_PORTFOLIO_RESEARCH_DIR", repo / "research.portfolio")

    vault = repo / "vault" / "D" / "crude_oil_mr" / "mr_cl_long"
    (vault / "features").mkdir(parents=True)
    (vault / "ensemble_config.json").write_text(
        json.dumps({"tickers": ["GC"]}),
        encoding="utf-8",
    )
    feature_payload = {
        "tickers": ["GC"],
        "bias_node_spec": {"module_name": "ibs", "timeframes": ["D"], "params": {}},
    }
    (vault / "features" / "ibs_gc.json").write_text(
        json.dumps(feature_payload),
        encoding="utf-8",
    )

    rel = "vault/D/crude_oil_mr/mr_cl_long"

    def _iter(features_dir: Path):
        yield vault / "features" / "ibs_gc.json", feature_payload

    monkeypatch.setattr(cfg_mod._vault_feature_files, "iter_validated_feature_configs", _iter)

    scoped = scoped_tickers_for_ensemble_dirs(
        [Ticker.ES, Ticker.GC, Ticker.NQ],
        {"mr_gc": rel},
    )
    assert scoped == (Ticker.GC,)


def test_scoped_tickers_excludes_cross_ticker_peers_for_rebalancing_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross peers are not portfolio legs; EWSD/candles for them come from preflight bootstrap."""

    def _iter(features_dir: Path):
        path_s = str(features_dir).replace("\\", "/")
        if "rebalancing_es_tlt_long_short" in path_s:
            yield Path("x.json"), {
                "bias_node_spec": {
                    "module_name": "rebalancing_flow",
                    "params": {"flow": "both", "cross_tickers": ["TLT"]},
                },
                "tickers": ["ES"],
            }

    _patch_feature_iter(monkeypatch, _iter)
    scoped = scoped_tickers_for_ensemble_dirs(
        [Ticker.ES, Ticker.NQ, Ticker.GC],
        {"rebalancing_es_tlt_long_short": "vault/D/es_tlt/rebalancing_es_tlt_long_short"},
    )
    assert scoped == (Ticker.ES,)
