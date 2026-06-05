"""Unit tests for vault-driven hierarchy_equal JSON generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ensemble.portfolio_impl.global_weight_layer_adapter import (
    build_global_model_name,
    build_global_stream_id,
)
from ensemble.vault.hierarchy_spec import (
    build_hierarchy_equal_spec,
    build_hierarchy_spec_for_ensemble_dirs,
    build_hierarchy_spec_from_vault,
    global_stream_ids_for_signed_signal_feature,
    global_stream_ids_for_vault_feature_member,
    infer_weight_hierarchy_group_from_feature_path,
)
from ensemble.weight_hierarchy import parse_hierarchy_spec
from lib.core.enums import TimeFrame


def _minimal_ensemble_config(ensemble_dir: Path, *, timeframe: str = "M") -> None:
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


def test_global_stream_ids_filters_to_portfolio_tickers() -> None:
    cfg = {
        "bias_node_spec": {"module_name": "m", "timeframes": ["D"], "params": {}},
        "tickers": ["ES", "RTY", "NQ"],
        "base_models": [
            {
                "model_name": "feat::rule_based_3",
                "model_type": "signed_signal",
                "feature_column": "x",
                "bias_node_spec": {"module_name": "m", "timeframes": ["D"], "params": {}},
            }
        ],
    }
    inner = build_global_model_name(TimeFrame.D, 0, "feat::rule_based_3")
    sids = global_stream_ids_for_signed_signal_feature(
        cfg,
        trading_timeframe=TimeFrame.D,
        ensemble_idx=0,
        portfolio_ticker_names=frozenset({"ES", "NQ"}),
    )
    assert set(sids) == {
        build_global_stream_id("ES", "D", inner),
        build_global_stream_id("NQ", "D", inner),
    }


def test_global_stream_ids_match_tf_portfolio_naming() -> None:
    cfg = {
        "bias_node_spec": {"module_name": "m", "timeframes": ["D"], "params": {}},
        "tickers": ["ES", "NQ"],
        "base_models": [
            {
                "model_name": "feat::rule_based_3",
                "model_type": "signed_signal",
                "feature_column": "x",
                "bias_node_spec": {"module_name": "m", "timeframes": ["D"], "params": {}},
            }
        ],
    }
    inner = build_global_model_name(TimeFrame.D, 2, "feat::rule_based_3")
    sids = global_stream_ids_for_signed_signal_feature(
        cfg,
        trading_timeframe=TimeFrame.D,
        ensemble_idx=2,
    )
    assert set(sids) == {
        build_global_stream_id("ES", "D", inner),
        build_global_stream_id("NQ", "D", inner),
    }


def test_infer_group_from_nested_path(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    feat = vault / "D" / "es_tlt" / "ens_long" / "features" / "a.json"
    feat.parent.mkdir(parents=True)
    feat.write_text("{}", encoding="utf-8")
    assert infer_weight_hierarchy_group_from_feature_path(feat, vault) == "es_tlt"


def test_infer_group_flat_returns_none(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    feat = vault / "D" / "flat_ensemble" / "features" / "a.json"
    feat.parent.mkdir(parents=True)
    feat.write_text("{}", encoding="utf-8")
    assert infer_weight_hierarchy_group_from_feature_path(feat, vault) is None


def test_build_hierarchy_equal_spec_unknown_bucket() -> None:
    with pytest.raises(ValueError, match="unknown bucket"):
        build_hierarchy_equal_spec({"not_a_bucket": ["a::b::c"]})


def test_build_and_parse_round_trip() -> None:
    spec = build_hierarchy_equal_spec(
        {
            "buy_hold": ["ES::M::m1"],
            "momentum": ["NQ::D::m2", "ES::D::m3"],
        },
        root_id="root",
    )
    root = parse_hierarchy_spec(spec)
    assert root.group_id == "root"
    assert len(root.children) == 2


def test_collect_raises_on_stream_id_bucket_conflict(tmp_path: Path) -> None:
    """Two features in one ensemble cannot claim the same stream for different hierarchy groups."""
    from ensemble.vault.hierarchy_spec import collect_streams_by_group_from_vault

    vault = tmp_path / "vault"
    ens = vault / "M" / "buy_hold" / "shared"
    feat_dir = ens / "features"
    feat_dir.mkdir(parents=True)
    _minimal_ensemble_config(ens, timeframe="M")

    def _payload(fname: str, group: str) -> dict[str, object]:
        return {
            "feature_name": fname,
            "weight_hierarchy_group": group,
            "bias_node_spec": {"module_name": "m", "timeframes": ["M"], "params": {}},
            "tickers": ["ES"],
            "base_models": [
                {
                    "model_id": "rule_based_3",
                    "model_name": "dup::rule_based_3",
                    "strategy": "long",
                    "model_type": "signed_signal",
                    "feature_column": fname,
                    "bias_node_spec": {"module_name": "m", "timeframes": ["M"], "params": {}},
                }
            ],
        }

    (feat_dir / "one.json").write_text(json.dumps(_payload("one", "buy_hold")), encoding="utf-8")
    (feat_dir / "two.json").write_text(json.dumps(_payload("two", "momentum")), encoding="utf-8")

    with pytest.raises(ValueError, match="stream_id .* mapped to both"):
        collect_streams_by_group_from_vault(vault)


def test_build_hierarchy_spec_for_ensemble_dirs_non_vault_prefix(tmp_path: Path) -> None:
    vault = tmp_path / "vault_personal"
    ens = vault / "M" / "buy_hold" / "buy_hold_long"
    (ens / "features").mkdir(parents=True)
    _minimal_ensemble_config(ens, timeframe="M")
    payload = {
        "feature_name": "f",
        "weight_hierarchy_group": "buy_hold",
        "bias_node_spec": {"module_name": "buy_hold", "timeframes": ["M"], "params": {}},
        "tickers": ["ES"],
        "base_models": [
            {
                "model_id": "rule_based_3",
                "model_name": "f::rule_based_3",
                "strategy": "long",
                "model_type": "signed_signal",
                "feature_column": "f",
                "bias_node_spec": {"module_name": "buy_hold", "timeframes": ["M"], "params": {}},
            }
        ],
    }
    (ens / "features" / "f.json").write_text(json.dumps(payload), encoding="utf-8")

    spec = build_hierarchy_spec_for_ensemble_dirs(
        tmp_path,
        {"buy_hold_long": "vault_personal/M/buy_hold/buy_hold_long"},
        strict_group=True,
    )
    parse_hierarchy_spec(spec)


def test_global_stream_ids_for_vault_feature_member_personal_vault(tmp_path: Path) -> None:
    vault = tmp_path / "vault_personal"
    ens = vault / "M" / "buy_hold" / "buy_hold_long"
    (ens / "features").mkdir(parents=True)
    _minimal_ensemble_config(ens, timeframe="M")
    payload = {
        "feature_name": "f",
        "weight_hierarchy_group": "buy_hold",
        "bias_node_spec": {"module_name": "buy_hold", "timeframes": ["M"], "params": {}},
        "tickers": ["ES"],
        "base_models": [
            {
                "model_id": "rule_based_3",
                "model_name": "f::rule_based_3",
                "strategy": "long",
                "model_type": "signed_signal",
                "feature_column": "f",
                "bias_node_spec": {"module_name": "buy_hold", "timeframes": ["M"], "params": {}},
            }
        ],
    }
    (ens / "features" / "f.json").write_text(json.dumps(payload), encoding="utf-8")
    ensemble_dirs = {"buy_hold_long": "vault_personal/M/buy_hold/buy_hold_long"}
    member_id = "M/buy_hold/buy_hold_long/features/f"
    sids = global_stream_ids_for_vault_feature_member(
        tmp_path,
        ensemble_dirs,
        member_id,
    )
    inner = build_global_model_name(TimeFrame.M, 0, "f::rule_based_3")
    assert build_global_stream_id("ES", "M", inner) in sids


def test_build_hierarchy_spec_for_ensemble_dirs_scoped(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    ens = vault / "M" / "buy_hold" / "buy_hold_long"
    (ens / "features").mkdir(parents=True)
    _minimal_ensemble_config(ens, timeframe="M")
    payload = {
        "feature_name": "f",
        "weight_hierarchy_group": "buy_hold",
        "bias_node_spec": {"module_name": "buy_hold", "timeframes": ["M"], "params": {}},
        "tickers": ["ES"],
        "base_models": [
            {
                "model_id": "rule_based_3",
                "model_name": "f::rule_based_3",
                "strategy": "long",
                "model_type": "signed_signal",
                "feature_column": "f",
                "bias_node_spec": {"module_name": "buy_hold", "timeframes": ["M"], "params": {}},
            }
        ],
    }
    (ens / "features" / "f.json").write_text(json.dumps(payload), encoding="utf-8")

    spec = build_hierarchy_spec_for_ensemble_dirs(
        tmp_path,
        {"buy_hold_long": "vault/M/buy_hold/buy_hold_long"},
        strict_group=True,
    )
    parse_hierarchy_spec(spec)


def test_build_hierarchy_spec_from_vault_minimal(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    ens = vault / "M" / "buy_hold" / "buy_hold_long"
    feat_dir = ens / "features"
    feat_dir.mkdir(parents=True)
    _minimal_ensemble_config(ens, timeframe="M")
    payload = {
        "feature_name": "f",
        "weight_hierarchy_group": "buy_hold",
        "bias_node_spec": {"module_name": "buy_hold", "timeframes": ["M"], "params": {}},
        "tickers": ["ES"],
        "base_models": [
            {
                "model_id": "rule_based_3",
                "model_name": "f::rule_based_3",
                "strategy": "long",
                "model_type": "signed_signal",
                "feature_column": "f",
                "bias_node_spec": {"module_name": "buy_hold", "timeframes": ["M"], "params": {}},
            }
        ],
    }
    (feat_dir / "f.json").write_text(json.dumps(payload), encoding="utf-8")

    spec, by_group = build_hierarchy_spec_from_vault(vault)
    parse_hierarchy_spec(spec)
    inner = build_global_model_name(TimeFrame.M, 0, "f::rule_based_3")
    expected = build_global_stream_id("ES", "M", inner)
    assert expected in by_group["buy_hold"]
