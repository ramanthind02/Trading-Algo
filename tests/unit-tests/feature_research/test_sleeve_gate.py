"""Unit tests for sleeve-scoped portfolio addition helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.feature.portfolio_addition.gate_runner import (
    _candidate_weight_assigned,
    _normalize_gate_weight_assigned,
    _sleeve_weight_assigned_fraction,
)
from research.feature.portfolio_addition.sleeve_gate import (
    candidate_sleeve_identity,
    candidate_stream_ids_for_ensemble,
    candidate_ticker_names_from_ensemble,
    ensemble_dirs_in_sleeve,
    sleeve_label,
)


def _minimal_ensemble(tmp_path: Path, *, group: str, tickers: list[str], fname: str) -> str:
    ens = tmp_path / "vault" / "D" / group / fname
    (ens / "features").mkdir(parents=True)
    (ens / "ensemble_config.json").write_text(
        json.dumps(
            {
                "timeframe": "D",
                "ensemble_name": fname,
                "direction": "long",
                "tickers": ["ES"],
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "feature_name": "f",
        "weight_hierarchy_group": group,
        "bias_node_spec": {"module_name": "m", "timeframes": ["D"], "params": {}},
        "tickers": tickers,
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
    return f"vault/D/{group}/{fname}"


def test_candidate_sleeve_gc_mean_reversion_group() -> None:
    asset, style = candidate_sleeve_identity(
        weight_hierarchy_group="crude_oil_mr",
        candidate_ticker_names=frozenset({"GC"}),
    )
    assert asset == "commodities"
    assert style == "crude_oil_mr"
    assert sleeve_label(asset, style) == "commodities/crude_oil_mr"


def test_candidate_sleeve_identity_from_tickers() -> None:
    asset, style = candidate_sleeve_identity(
        weight_hierarchy_group="momentum",
        candidate_ticker_names=frozenset({"ES", "NQ"}),
    )
    assert asset == "equity_indices"
    assert style == "momentum"
    assert sleeve_label(asset, style) == "equity_indices/momentum"


def test_candidate_sleeve_es_tlt_routes_to_equity_indices() -> None:
    """ES rebalancing-flow vs TLT peer → equity_indices/es_tlt (not diversified)."""
    asset, style = candidate_sleeve_identity(
        weight_hierarchy_group="es_tlt",
        candidate_ticker_names=frozenset({"ES"}),
    )
    assert asset == "equity_indices"
    assert style == "es_tlt"
    assert sleeve_label(asset, style) == "equity_indices/es_tlt"


def test_candidate_sleeve_seasonal_routes_to_equity_indices() -> None:
    """EOY / indices seasonal on ES/NQ must not land in diversified/seasonal."""
    asset, style = candidate_sleeve_identity(
        weight_hierarchy_group="seasonal",
        candidate_ticker_names=frozenset({"ES", "NQ"}),
    )
    assert asset == "equity_indices"
    assert style == "seasonal"
    assert sleeve_label(asset, style) == "equity_indices/seasonal"


def test_candidate_sleeve_buy_hold_uses_ticker_asset_class() -> None:
    asset_es, style = candidate_sleeve_identity(
        weight_hierarchy_group="buy_hold",
        candidate_ticker_names=frozenset({"ES"}),
    )
    assert asset_es == "equity_indices"
    assert style == "buy_hold"

    asset_mixed, _ = candidate_sleeve_identity(
        weight_hierarchy_group="buy_hold",
        candidate_ticker_names=frozenset({"ES", "GC"}),
    )
    assert asset_mixed == "diversified"


def test_ensemble_dirs_in_sleeve_filters_mixed_momentum(tmp_path: Path) -> None:
    mixed = _minimal_ensemble(tmp_path, group="momentum", tickers=["ES", "GC"], fname="mixed")
    equity_only = _minimal_ensemble(tmp_path, group="momentum", tickers=["ES", "NQ"], fname="eq")
    dirs = {
        "mixed": mixed,
        "eq": equity_only,
    }
    in_sleeve = ensemble_dirs_in_sleeve(
        tmp_path,
        dirs,
        asset_class="equity_indices",
        style_group="momentum",
        portfolio_ticker_names=frozenset({"ES", "NQ", "GC"}),
    )
    assert "eq" in in_sleeve
    assert "mixed" in in_sleeve


def test_candidate_sleeve_uses_ensemble_tickers_not_portfolio_mix(tmp_path: Path) -> None:
    """Portfolio GC must not force diversified sleeve when the candidate is ES/NQ only."""
    mr = _minimal_ensemble(
        tmp_path, group="mean_reversion_indices", tickers=["ES", "NQ"], fname="mr_peer"
    )
    portfolio_tickers = frozenset({"ES", "NQ", "GC"})
    candidate_tickers = candidate_ticker_names_from_ensemble(
        tmp_path, mr, portfolio_ticker_names=portfolio_tickers
    )
    assert candidate_tickers == frozenset({"ES", "NQ"})
    asset, style = candidate_sleeve_identity(
        weight_hierarchy_group="mean_reversion_indices",
        candidate_ticker_names=candidate_tickers,
    )
    assert asset == "equity_indices"
    assert style == "mean_reversion_indices"
    mixed_portfolio_asset, _ = candidate_sleeve_identity(
        weight_hierarchy_group="mean_reversion_indices",
        candidate_ticker_names=portfolio_tickers,
    )
    assert mixed_portfolio_asset == "diversified"
    in_sleeve = ensemble_dirs_in_sleeve(
        tmp_path,
        {"mr_peer": mr},
        asset_class=asset,
        style_group=style,
        portfolio_ticker_names=portfolio_tickers,
    )
    assert "mr_peer" in in_sleeve


def test_multi_asset_trend_following_candidate_uses_diversified_sleeve() -> None:
    from research.feature.config import TREND_FOLLOWING_UNIVERSE

    names = frozenset(t.name for t in TREND_FOLLOWING_UNIVERSE)
    asset, style = candidate_sleeve_identity(
        weight_hierarchy_group="trend_following",
        candidate_ticker_names=names,
    )
    assert asset == "diversified"
    assert style == "trend_following"


def test_candidate_stream_ids_for_ensemble(tmp_path: Path) -> None:
    rel = _minimal_ensemble(tmp_path, group="momentum", tickers=["ES"], fname="cand")
    sids = candidate_stream_ids_for_ensemble(
        tmp_path,
        rel,
        portfolio_ticker_names=frozenset({"ES"}),
    )
    assert len(sids) == 1
    assert next(iter(sids)).startswith("ES::")


def test_candidate_stream_ids_match_portfolio_ensemble_index(tmp_path: Path) -> None:
    e0 = _minimal_ensemble(tmp_path, group="momentum", tickers=["ES"], fname="peer_a")
    e1 = _minimal_ensemble(tmp_path, group="buy_hold", tickers=["ES"], fname="peer_b")
    cand = _minimal_ensemble(tmp_path, group="mean_reversion_indices", tickers=["ES"], fname="cand")
    portfolio = {"peer_a": e0, "peer_b": e1, "cand": cand}
    solo_sid = next(
        iter(
            candidate_stream_ids_for_ensemble(
                tmp_path,
                cand,
                portfolio_ticker_names=frozenset({"ES"}),
            )
        )
    )
    portfolio_sid = next(
        iter(
            candidate_stream_ids_for_ensemble(
                tmp_path,
                cand,
                portfolio_ticker_names=frozenset({"ES"}),
                portfolio_ensemble_dirs=portfolio,
            )
        )
    )
    assert "ensemble_0" in solo_sid
    assert "ensemble_2" in portfolio_sid
    assert "ensemble_0" not in portfolio_sid


def test_sleeve_weight_assigned_fraction_matches_portfolio_stream_ids() -> None:
    candidate_sid = "ES::D::f__D::ensemble_2"
    peer_sid = "NQ::D::g__D::ensemble_0"
    row_defaults = {
        "stream_source_model_name": "model",
        "weight_method": "hierarchy_equal",
    }
    diagnostics = pd.DataFrame(
        [
            {
                **row_defaults,
                "weight_layer_ticker": "__GLOBAL__",
                "stream_or_model_id": candidate_sid,
                "stream_weight": 0.033,
                "cluster_id": "root/equity_indices/mean_reversion_indices",
            },
            {
                **row_defaults,
                "weight_layer_ticker": "__GLOBAL__",
                "stream_or_model_id": peer_sid,
                "stream_weight": 0.033,
                "cluster_id": "root/equity_indices/mean_reversion_indices",
            },
            {
                **row_defaults,
                "weight_layer_ticker": "__GLOBAL__",
                "stream_or_model_id": "GC::D::h__D::ensemble_1",
                "stream_weight": 0.34,
                "cluster_id": "root/commodities/momentum",
            },
        ]
    )
    wrong_ids = frozenset({"ES::D::f__D::ensemble_0"})
    correct_ids = frozenset({candidate_sid})
    assert _sleeve_weight_assigned_fraction(
        diagnostics,
        asset_class="equity_indices",
        style_group="mean_reversion_indices",
        candidate_stream_ids=wrong_ids,
    ) == 0.0
    assert _sleeve_weight_assigned_fraction(
        diagnostics,
        asset_class="equity_indices",
        style_group="mean_reversion_indices",
        candidate_stream_ids=correct_ids,
    ) == 0.5


def test_normalize_gate_weight_assigned_clamps_invalid_values() -> None:
    assert _normalize_gate_weight_assigned(1.5) == 1.0
    assert _normalize_gate_weight_assigned(-0.1) == 0.0
    assert _normalize_gate_weight_assigned(float("nan")) == 0.0


def test_candidate_weight_assigned_ignores_total_book_when_no_new_rows(tmp_path: Path) -> None:
    from dataclasses import replace

    from research.portfolio.pipelines.portfolio_test import PhaseResult

    index = pd.date_range("2020-01-01", periods=5, freq="B")
    diagnostics = pd.DataFrame(
        [
            {
                "weight_layer_ticker": "__GLOBAL__",
                "stream_or_model_id": "ES::D::f__D::ensemble_0",
                "stream_weight": 0.5,
            },
            {
                "weight_layer_ticker": "__GLOBAL__",
                "stream_or_model_id": "NQ::D::g__D::ensemble_0",
                "stream_weight": 0.5,
            },
        ]
    )
    with_train = PhaseResult(
        name="with",
        output_dir=tmp_path,
        combined_strategy_returns=pd.Series(0.0, index=index),
        combined_baseline_returns=pd.Series(0.0, index=index),
        weight_layer_diagnostics_df=diagnostics,
    )
    without_train = replace(with_train, name="without")
    assert _candidate_weight_assigned(with_train, without_train) == 0.0
