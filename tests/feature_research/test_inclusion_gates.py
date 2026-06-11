"""Unit tests for portfolio inclusion gate helpers (no vault I/O)."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.feature.config import (
    PortfolioInclusionConfig,
    VaultSaveConfig,
    inferred_inclusion_candidate_path,
    load_config,
)
from research.feature.inclusion_gates import (
    _write_inclusion_tearsheets_from_phases,
    compute_peer_forecast_correlations,
    config_with_ensemble_dirs,
    infer_candidate_key,
    materialize_inclusion_candidate_from_eval_bias_spec,
    pearson_corr_candidate_vs_each_peer,
)
from research.portfolio.pipelines.portfolio_test import PhaseResult
from research.portfolio.config import EnsembleDirsPolicy, PortfolioResearchConfig, ResearchWindow
from cache.runtime.cache_paths import win32_extended_path
from lib.core.enums import Ticker, TimeFrame


def test_pearson_corr_vs_each_peer_perfect() -> None:
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    rows = []
    for t in ("ES", "NQ"):
        for dt in dates:
            rows.append({"ticker": t, "datetime": dt, "forecast_score": float(np.sin(dt.day))})
    df = pd.DataFrame(rows)
    forecasts = {"cand": df.copy(), "peer": df.copy()}
    name_to_tf = {"cand": TimeFrame.D, "peer": TimeFrame.D}
    by_peer = pearson_corr_candidate_vs_each_peer(
        forecasts,
        candidate_key="cand",
        name_to_tf=name_to_tf,
        portfolio_ticker_names=frozenset(["ES", "NQ"]),
    )
    assert by_peer["peer"] == pytest.approx(1.0)


def test_pearson_corr_vs_each_peer_uncorrelated() -> None:
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    x = np.linspace(0.0, 10.0, len(dates))
    rows_c = []
    rows_p = []
    for t in ("ES",):
        for i, dt in enumerate(dates):
            rows_c.append({"ticker": t, "datetime": dt, "forecast_score": float(np.sin(x[i]))})
            rows_p.append({"ticker": t, "datetime": dt, "forecast_score": float(np.cos(x[i]))})
    forecasts = {"cand": pd.DataFrame(rows_c), "peer": pd.DataFrame(rows_p)}
    name_to_tf = {"cand": TimeFrame.D, "peer": TimeFrame.D}
    by_peer = pearson_corr_candidate_vs_each_peer(
        forecasts,
        candidate_key="cand",
        name_to_tf=name_to_tf,
        portfolio_ticker_names=frozenset(["ES"]),
    )
    assert abs(by_peer["peer"]) < 0.3


def test_pearson_corr_candidate_vs_each_peer_two_peers() -> None:
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    rows_c = []
    rows_p1 = []
    rows_p2 = []
    for t in ("ES",):
        for i, dt in enumerate(dates):
            rows_c.append({"ticker": t, "datetime": dt, "forecast_score": float(i)})
            rows_p1.append({"ticker": t, "datetime": dt, "forecast_score": float(i)})
            rows_p2.append({"ticker": t, "datetime": dt, "forecast_score": float(i * 2)})
    forecasts = {
        "cand": pd.DataFrame(rows_c),
        "p1": pd.DataFrame(rows_p1),
        "p2": pd.DataFrame(rows_p2),
    }
    name_to_tf = {k: TimeFrame.D for k in forecasts}
    by_peer = pearson_corr_candidate_vs_each_peer(
        forecasts,
        candidate_key="cand",
        name_to_tf=name_to_tf,
        portfolio_ticker_names=frozenset(["ES"]),
    )
    assert by_peer["p1"] == pytest.approx(1.0)
    assert by_peer["p2"] == pytest.approx(1.0)


def test_corr_uses_intersection_of_tickers_not_full_portfolio() -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    rows_c = []
    rows_p = []
    for dt in dates:
        rows_c.append({"ticker": "ES", "datetime": dt, "forecast_score": 1.0})
        rows_c.append({"ticker": "NQ", "datetime": dt, "forecast_score": 2.0})
        rows_p.append({"ticker": "ES", "datetime": dt, "forecast_score": 1.0})
    forecasts = {"cand": pd.DataFrame(rows_c), "peer": pd.DataFrame(rows_p)}
    name_to_tf = {"cand": TimeFrame.D, "peer": TimeFrame.D}
    _, _, detail = compute_peer_forecast_correlations(
        forecasts,
        candidate_key="cand",
        name_to_tf=name_to_tf,
        portfolio_ticker_names=frozenset(["ES", "NQ"]),
    )
    peers_tickers = {(row[0], row[1]) for row in detail}
    assert ("peer", "ES") in peers_tickers
    assert ("peer", "NQ") not in peers_tickers


def test_pearson_corr_skips_different_timeframe_peer() -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        [{"ticker": "ES", "datetime": dt, "forecast_score": 1.0} for dt in dates]
    )
    forecasts = {"cand": df.copy(), "other_tf": df.copy()}
    name_to_tf = {"cand": TimeFrame.D, "other_tf": TimeFrame.M}
    by_peer = pearson_corr_candidate_vs_each_peer(
        forecasts,
        candidate_key="cand",
        name_to_tf=name_to_tf,
        portfolio_ticker_names=frozenset(["ES"]),
    )
    assert by_peer == {}


def test_portfolio_inclusion_config_validation() -> None:
    with pytest.raises(ValueError, match="candidate_mode"):
        PortfolioInclusionConfig(candidate_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ephemeral_ensemble_name"):
        PortfolioInclusionConfig(ephemeral_ensemble_name="  ")
    with pytest.raises(ValueError, match="ephemeral_weight_hierarchy_group"):
        PortfolioInclusionConfig(ephemeral_weight_hierarchy_group="not_a_bucket")
    c = PortfolioInclusionConfig()
    assert c.emit_tearsheets is True
    assert c.candidate_mode == "eval_bias_spec"


def test_materialize_inclusion_candidate_from_eval_bias_spec_writes_feature(
    tmp_path: Path,
) -> None:
    import shutil

    from ensemble.vault.hierarchy_spec import global_stream_ids_for_signed_signal_feature
    from research.feature.config import TREND_FOLLOWING_UNIVERSE

    fr = load_config()
    train = ResearchWindow(start=datetime(2000, 1, 1), end=datetime(2005, 12, 31))
    validation = ResearchWindow(start=datetime(2006, 1, 1), end=datetime(2010, 12, 31))
    test = ResearchWindow(start=datetime(2011, 1, 1), end=datetime(2015, 12, 31))
    pr = PortfolioResearchConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2015, 12, 31),
        use_cache=True,
        train_window=train,
        validation_window=validation,
        test_window=test,
        ensemble_dirs={"a": "vault/D/a"},
        weight_layer_method="equal_signal",
        weight_layer_kwargs={"fdm_max": 2.0},
    )
    ens, root = materialize_inclusion_candidate_from_eval_bias_spec(
        fr,
        pr,
        ephemeral_ensemble_name="test_inc",
        weight_hierarchy_group="trend_following",
        temp_parent=tmp_path,
    )
    try:
        p = Path(ens)
        assert p.is_dir()
        feats = list((p / "features").glob("*.json"))
        assert len(feats) == 1
        with open(win32_extended_path(feats[0]), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload.get("weight_hierarchy_group") == "trend_following"
        assert frozenset(payload.get("tickers", [])) == frozenset(t.name for t in TREND_FOLLOWING_UNIVERSE)
        stream_ids = global_stream_ids_for_signed_signal_feature(
            payload,
            trading_timeframe=TimeFrame.D,
            ensemble_idx=0,
            portfolio_ticker_names=frozenset(t.name for t in TREND_FOLLOWING_UNIVERSE),
        )
        assert len(stream_ids) == len(TREND_FOLLOWING_UNIVERSE)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_infer_candidate_key() -> None:
    assert infer_candidate_key("vault/D/momentum/foo_long") == "foo_long"


def test_inferred_inclusion_candidate_path_requires_existing_directory() -> None:
    cfg = load_config()
    vs = cfg.vault_save
    assert vs is not None
    p = inferred_inclusion_candidate_path(cfg)
    assert p is None or (isinstance(p, str) and p.startswith("vault/"))

    existing = "vault/D/momentum/algomatic_momentum_signal_long_defaults_long"
    cfg2 = replace(
        cfg,
        vault_save=VaultSaveConfig(
            direction=vs.direction,
            existing_ensemble_dir=existing,
        ),
    )
    assert inferred_inclusion_candidate_path(cfg2) == existing


def test_config_with_ensemble_dirs_rebuilds_asset_first_hierarchy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.feature.inclusion_gates as inclusion_gates_mod
    from ensemble.weight_hierarchy import parse_hierarchy_spec

    monkeypatch.setattr(inclusion_gates_mod, "_REPO_ROOT", tmp_path)

    vault = tmp_path / "vault"
    ens = vault / "D" / "momentum" / "sma_regime"
    (ens / "features").mkdir(parents=True)
    (ens / "ensemble_config.json").write_text(
        json.dumps(
            {
                "timeframe": "D",
                "ensemble_name": "t",
                "direction": "long",
                "tickers": ["ES"],
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "feature_name": "f",
        "weight_hierarchy_group": "momentum",
        "bias_node_spec": {"module_name": "m", "timeframes": ["D"], "params": {}},
        "tickers": ["ES", "GC"],
        "base_models": [
            {
                "model_id": "rule_based_3",
                "model_name": "f::rule_based_3",
                "strategy": "long",
                "model_type": "signed_signal",
                "feature_column": "f",
                "bias_node_spec": {"module_name": "m", "timeframes": ["D"], "params": {}},
            }
        ],
    }
    (ens / "features" / "f.json").write_text(json.dumps(payload), encoding="utf-8")

    train = ResearchWindow(start=datetime(2000, 1, 1), end=datetime(2005, 12, 31))
    validation = ResearchWindow(start=datetime(2006, 1, 1), end=datetime(2010, 12, 31))
    test = ResearchWindow(start=datetime(2011, 1, 1), end=datetime(2015, 12, 31))
    base = PortfolioResearchConfig(
        tickers=[Ticker.ES, Ticker.GC],
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2015, 12, 31),
        use_cache=True,
        train_window=train,
        validation_window=validation,
        test_window=test,
        ensemble_dirs={"sma_regime": "vault/D/momentum/sma_regime"},
        weight_layer_method="hierarchy_equal",
        weight_layer_kwargs={"fdm_max": 2.0},
    )
    out = config_with_ensemble_dirs(base, {"sma_regime": "vault/D/momentum/sma_regime"})
    spec = out.weight_layer_kwargs.get("hierarchy_spec")
    assert isinstance(spec, dict)
    root = parse_hierarchy_spec(spec)
    assert root.group_id == "root"
    asset_ids = {child.group_id for child in root.children}
    assert asset_ids == {"equity_indices"}


def test_config_with_ensemble_dirs_scopes_tickers_to_active_ensemble(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import research.feature.inclusion_gates as inclusion_gates_mod
    import research.portfolio.config as cfg_mod

    repo = tmp_path
    monkeypatch.setattr(inclusion_gates_mod, "_REPO_ROOT", repo)
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
        "weight_hierarchy_group": "crude_oil_mr",
        "bias_node_spec": {"module_name": "ibs", "timeframes": ["D"], "params": {}},
    }
    (vault / "features" / "ibs_gc.json").write_text(
        json.dumps(feature_payload),
        encoding="utf-8",
    )
    rel = "vault/D/crude_oil_mr/mr_cl_long"

    def _iter(features_dir: Path):
        if features_dir.name == "features":
            yield vault / "features" / "ibs_gc.json", feature_payload

    monkeypatch.setattr(cfg_mod._vault_feature_files, "iter_validated_feature_configs", _iter)

    train = ResearchWindow(start=datetime(2000, 1, 1), end=datetime(2005, 12, 31))
    validation = ResearchWindow(start=datetime(2006, 1, 1), end=datetime(2010, 12, 31))
    test = ResearchWindow(start=datetime(2011, 1, 1), end=datetime(2015, 12, 31))
    base = PortfolioResearchConfig(
        tickers=[Ticker.ES, Ticker.GC, Ticker.NQ],
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2015, 12, 31),
        use_cache=True,
        train_window=train,
        validation_window=validation,
        test_window=test,
        ensemble_dirs={},
        ensemble_dirs_policy=EnsembleDirsPolicy.ALLOW_EMPTY,
        baseline_mode="buy_hold",
        benchmark_ticker=Ticker.ES,
        weight_layer_method="equal_signal",
        weight_layer_kwargs={"fdm_max": 2.0},
    )
    out = config_with_ensemble_dirs(base, {"mr_gc": rel})
    assert out.tickers == (Ticker.GC,)
    assert out.benchmark_ticker == Ticker.GC


def test_config_with_ensemble_dirs_merges_and_preserves_equal_signal() -> None:
    train = ResearchWindow(start=datetime(2000, 1, 1), end=datetime(2005, 12, 31))
    validation = ResearchWindow(start=datetime(2006, 1, 1), end=datetime(2010, 12, 31))
    test = ResearchWindow(start=datetime(2011, 1, 1), end=datetime(2015, 12, 31))
    base = PortfolioResearchConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2015, 12, 31),
        use_cache=True,
        train_window=train,
        validation_window=validation,
        test_window=test,
        ensemble_dirs={"a": "vault/D/a"},
        weight_layer_method="equal_signal",
        weight_layer_kwargs={"fdm_max": 2.0},
    )
    out = config_with_ensemble_dirs(base, {"a": "vault/D/a", "b": "vault/D/b"})
    assert out.ensemble_dirs == {"a": "vault/D/a", "b": "vault/D/b"}
    assert out.weight_layer_kwargs == {"fdm_max": 2.0}


def test_write_inclusion_tearsheets_from_phases_writes_six_html(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.001, 0.002, size=len(dates)), index=dates)
    b = pd.Series(rng.normal(0.0005, 0.001, size=len(dates)), index=dates)
    pr = PhaseResult(
        name="Train",
        output_dir=tmp_path,
        combined_strategy_returns=s,
        combined_baseline_returns=b,
    )
    paths = _write_inclusion_tearsheets_from_phases(
        tearsheets_root=tmp_path / "ts",
        phase_without_tr=pr,
        phase_without_val=pr,
        phase_with_tr=pr,
        phase_with_val=pr,
    )
    assert len(paths) == 6
    assert all(p.suffix == ".html" for p in paths)
    assert all(p.exists() for p in paths)
