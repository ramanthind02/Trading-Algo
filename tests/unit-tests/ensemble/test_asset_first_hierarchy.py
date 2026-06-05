"""Unit tests for asset-first hierarchy_equal spec generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ensemble.portfolio_impl.global_weight_layer_adapter import (
    build_global_model_name,
    build_global_stream_id,
)
from ensemble.vault.constants import TICKER_ASSET_CLASS
from ensemble.vault.hierarchy_spec import (
    build_asset_first_hierarchy_spec,
    build_asset_first_hierarchy_spec_for_ensemble_dirs,
    collect_streams_by_asset_and_style,
    collect_streams_by_asset_and_style_for_ensemble_dirs,
)
from ensemble.weight_hierarchy import (
    compute_equal_split_weights,
    parse_hierarchy_spec,
    validate_strict_stream_coverage,
)
from lib.core.enums import TimeFrame


def _minimal_ensemble_config(ensemble_dir: Path, *, timeframe: str = "D") -> None:
    (ensemble_dir / "ensemble_config.json").write_text(
        json.dumps(
            {
                "timeframe": timeframe,
                "ensemble_name": "t",
                "direction": "long",
                "tickers": ["ES"],
            }
        ),
        encoding="utf-8",
    )


def _signed_signal_payload(
    *,
    feature_name: str,
    group: str,
    tickers: list[str],
    model_name: str,
    timeframe: str = "D",
) -> dict[str, object]:
    return {
        "feature_name": feature_name,
        "weight_hierarchy_group": group,
        "bias_node_spec": {"module_name": "m", "timeframes": [timeframe], "params": {}},
        "tickers": tickers,
        "base_models": [
            {
                "model_id": "rule_based_3",
                "model_name": model_name,
                "strategy": "long",
                "model_type": "signed_signal",
                "feature_column": feature_name,
                "bias_node_spec": {"module_name": "m", "timeframes": [timeframe], "params": {}},
            }
        ],
    }


def _write_feature(
    ens: Path,
    *,
    fname: str,
    group: str,
    tickers: list[str],
    model_name: str,
    timeframe: str = "D",
) -> None:
    feat_dir = ens / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    (feat_dir / f"{fname}.json").write_text(
        json.dumps(
            _signed_signal_payload(
                feature_name=fname,
                group=group,
                tickers=tickers,
                model_name=model_name,
                timeframe=timeframe,
            )
        ),
        encoding="utf-8",
    )


def test_ticker_asset_class_covers_fixture_tickers() -> None:
    for ticker in ("ES", "NQ", "GC"):
        assert ticker in TICKER_ASSET_CLASS


def test_collect_splits_mixed_momentum_by_asset(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    ens = vault / "D" / "momentum" / "sma_regime"
    ens.mkdir(parents=True)
    _minimal_ensemble_config(ens, timeframe="D")
    _write_feature(
        ens,
        fname="regime",
        group="momentum",
        tickers=["ES", "NQ", "GC"],
        model_name="regime::rule_based_3",
    )

    by_asset = collect_streams_by_asset_and_style(vault, strict_group=True)
    inner = build_global_model_name(TimeFrame.D, 0, "regime::rule_based_3")
    equity = by_asset["equity_indices"]["momentum"]
    commodity = by_asset["commodities"]["momentum"]
    assert build_global_stream_id("ES", "D", inner) in equity
    assert build_global_stream_id("NQ", "D", inner) in equity
    assert build_global_stream_id("GC", "D", inner) in commodity
    assert build_global_stream_id("GC", "D", inner) not in equity


def test_buy_hold_splits_by_ticker_asset_class(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    ens = vault / "M" / "buy_hold" / "buy_hold_long"
    ens.mkdir(parents=True)
    _minimal_ensemble_config(ens, timeframe="M")
    _write_feature(
        ens,
        fname="bh",
        group="buy_hold",
        tickers=["ES", "GC"],
        model_name="bh::rule_based_3",
        timeframe="M",
    )

    by_asset = collect_streams_by_asset_and_style(vault, strict_group=True)
    inner = build_global_model_name(TimeFrame.M, 0, "bh::rule_based_3")
    equity_bh = by_asset["equity_indices"]["buy_hold"]
    commodity_bh = by_asset["commodities"]["buy_hold"]
    assert build_global_stream_id("ES", "M", inner) in equity_bh
    assert build_global_stream_id("GC", "M", inner) in commodity_bh
    assert "diversified" not in by_asset or "buy_hold" not in by_asset.get("diversified", {})


def test_build_asset_first_three_levels(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    ens = vault / "D" / "momentum" / "sma_regime"
    ens.mkdir(parents=True)
    _minimal_ensemble_config(ens, timeframe="D")
    _write_feature(
        ens,
        fname="regime",
        group="momentum",
        tickers=["ES", "GC"],
        model_name="regime::rule_based_3",
    )

    spec = build_asset_first_hierarchy_spec_for_ensemble_dirs(
        tmp_path,
        {"sma_regime": "vault/D/momentum/sma_regime"},
        strict_group=True,
    )
    root = parse_hierarchy_spec(spec)
    assert root.group_id == "root"
    assert len(root.children) == 2
    asset_ids = {c.group_id for c in root.children}
    assert asset_ids == {"equity_indices", "commodities"}


def test_strict_coverage_and_equal_split_budgets() -> None:
    """Three asset classes with one style each; leaf counts differ by design."""
    streams = {
        "equity_indices": {
            "momentum": [
                "ES::D::m1",
                "NQ::D::m2",
            ],
        },
        "commodities": {
            "momentum": ["GC::D::m3"],
        },
        "diversified": {
            "buy_hold": [
                "ES::M::m4",
                "GC::M::m5",
            ],
        },
    }
    spec = build_asset_first_hierarchy_spec(streams)
    root = parse_hierarchy_spec(spec)
    all_streams = sorted(
        sid
        for styles in streams.values()
        for sids in styles.values()
        for sid in sids
    )
    validate_strict_stream_coverage(
        frozenset(all_streams),
        available_models=frozenset(all_streams),
    )
    weights, assignments, _, _ = compute_equal_split_weights(root, all_streams)
    by_asset: dict[str, float] = {}
    for sid, w in weights.items():
        parts = assignments[sid].split("/")
        if parts and parts[0] == "root":
            parts = parts[1:]
        asset = parts[0] if parts else "unknown"
        by_asset[asset] = by_asset.get(asset, 0.0) + float(w)
    assert by_asset["equity_indices"] == pytest.approx(1.0 / 3.0)
    assert by_asset["commodities"] == pytest.approx(1.0 / 3.0)
    assert by_asset["diversified"] == pytest.approx(1.0 / 3.0)


def test_collect_for_ensemble_dirs_asset_first(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    ens = vault / "M" / "buy_hold" / "buy_hold_long"
    ens.mkdir(parents=True)
    _minimal_ensemble_config(ens, timeframe="M")
    _write_feature(
        ens,
        fname="f",
        group="buy_hold",
        tickers=["ES"],
        model_name="f::rule_based_3",
        timeframe="M",
    )
    by_asset = collect_streams_by_asset_and_style_for_ensemble_dirs(
        tmp_path,
        {"buy_hold_long": "vault/M/buy_hold/buy_hold_long"},
        strict_group=True,
    )
    assert "equity_indices" in by_asset
    assert "buy_hold" in by_asset["equity_indices"]
    spec = build_asset_first_hierarchy_spec_for_ensemble_dirs(
        tmp_path,
        {"buy_hold_long": "vault/M/buy_hold/buy_hold_long"},
        strict_group=True,
    )
    root = parse_hierarchy_spec(spec)
    asset_ids = {child.group_id for child in root.children}
    assert "equity_indices" in asset_ids
    assert "diversified" not in asset_ids
