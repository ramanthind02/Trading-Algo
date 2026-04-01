"""Unit tests for vault-manager schema-v2 behavior."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ensemble.vault_manager import (
    create_ensemble_directory,
    generate_model_id,
    ensure_vault_cache_coverage,
    get_all_base_model_names,
    get_bias_node_specs,
    list_ensembles,
    list_features,
    load_feature_base_models,
    migrate_legacy_feature_members_schema,
    update_base_model_fitted_params,
    validate_ensemble_directory,
)
from ensemble.ensemble_utils import create_base_model_from_config
from feature_selection.domain_discrete import (
    DomainDiscreteMigrationError,
    build_domain_discrete_bias_node_spec,
)
from utils.core.enums import Direction, TimeFrame, Ticker


def _make_base_model(lookback: int = 2):
    config = {
        "name": "rsi_signal_D_domain_discrete",
        "model_type": "domain_discrete",
        "feature_column": f"rsi_signal_D_lookback_{lookback}",
        "strategy": "long",
        "bias_node_spec": {
            "module_name": "domain_discrete",
            "timeframes": ["D"],
            "params": {
                "source_bias_node_spec": {
                    "module_name": "rsi",
                    "timeframes": ["D"],
                    "params": {"lookback": lookback},
                },
                "ticker_scope": {"tickers": ["ES"], "scope_name": "ES"},
                "edges": [-0.5, 0.5],
                "n_bins": 3,
                "long_bins": [2],
                "short_bins": [],
                "direction": "long",
                "spec_version": "v1",
            },
        },
        "tickers": ["ES"],
    }
    return create_base_model_from_config(config, use_cache=False)


def _write_domain_discrete_feature_file(
    ensemble_dir: str,
    feature_name: str = "rsi_signal_D",
    lookback: int = 2,
    *,
    members: list[dict[str, object]] | None = None,
) -> str:
    model = _make_base_model(lookback=lookback)
    model_id = generate_model_id(
        "domain_discrete",
        {},
        bias_node_params={"params": model.domain_discrete_spec.to_mapping()},
    )
    serializable_bias_node_spec = build_domain_discrete_bias_node_spec(
        model.domain_discrete_spec
    )
    payload = {
        "feature_name": feature_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "bias_node_spec": serializable_bias_node_spec,
        "tickers": [Ticker.ES.name],
        "base_models": [
            {
                "model_id": model_id,
                "model_name": f"{feature_name}::{model_id}",
                "model_type": "domain_discrete",
                "feature_column": feature_name,
                "strategy": model.strategy.value if hasattr(model.strategy, "value") else model.strategy,
                "bias_node_spec": serializable_bias_node_spec,
            }
        ],
    }
    if members is not None:
        payload["base_models"][0]["members"] = members

    feature_file = Path(ensemble_dir) / "features" / f"{feature_name}.json"
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return model_id


def test_generate_model_id_appends_bias_node_params() -> None:
    model_id = generate_model_id(
        "domain_discrete",
        {
            "source_bias_node_spec": {
                "module_name": "rsi",
                "timeframes": ["D"],
                "params": {"lookback": 2},
            },
            "ticker_scope": {"tickers": ["ES"], "scope_name": "ES"},
            "edges": [-0.5, 0.5],
            "n_bins": 3,
            "long_bins": [2],
            "short_bins": [],
            "direction": "long",
            "spec_version": "v1",
        },
    )
    assert model_id == "domain_discrete_rsi_long_es_v1"


def test_add_feature_writes_feature_name_only(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="schema_v2",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    model_id = _write_domain_discrete_feature_file(ensemble_dir)

    feature_file = Path(ensemble_dir) / "features" / "rsi_signal_D.json"
    assert feature_file.exists()
    with open(feature_file, "r") as handle:
        payload = json.load(handle)
    assert payload["feature_name"] == "rsi_signal_D"
    assert "feature_column" not in payload
    assert payload["base_models"][0]["model_id"] == model_id
    assert payload["base_models"][0]["model_type"] == "domain_discrete"
    assert payload["base_models"][0]["bias_node_spec"]["module_name"] == "domain_discrete"
    assert payload["base_models"][0]["bias_node_spec"]["params"]["spec_version"] == "v1"
    assert "requires_fit" not in payload["base_models"][0]
    assert "fitted_params" not in payload["base_models"][0]


def test_load_models_merges_bias_node_params_single_model(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="load_members",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    _write_domain_discrete_feature_file(ensemble_dir, lookback=5)

    loaded = load_feature_base_models(
        feature_name="rsi_signal_D",
        ensemble_dir=ensemble_dir,
    )
    assert len(loaded) == 1
    loaded_model = next(iter(loaded.values()))
    assert loaded_model.model_type == "domain_discrete"
    assert loaded_model.bias_node_spec["params"]["source_bias_node_spec"]["params"]["lookback"] == 5


def test_update_fitted_params_path(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="member_update",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    _write_domain_discrete_feature_file(ensemble_dir)
    feature_file = Path(ensemble_dir).joinpath("features", "rsi_signal_D.json")
    with open(feature_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    base_model_id = payload["base_models"][0]["model_id"]

    with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
        update_base_model_fitted_params(
            ensemble_dir=ensemble_dir,
            feature_name="rsi_signal_D",
            model_id=base_model_id,
            fitted_params={"model_version": "binning_v2"},
            train_start="2020-01-01",
            train_end="2020-12-31",
    )


def test_list_specs_names_and_validate(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="specs_names",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    _write_domain_discrete_feature_file(ensemble_dir)

    features_df = list_features(ensemble_dir)
    assert "feature_name" in features_df.columns
    assert features_df.iloc[0]["feature_name"] == "rsi_signal_D"

    specs = get_bias_node_specs(ensemble_dir)
    assert len(specs) == 1
    assert specs[0]["module_name"] == "domain_discrete"
    assert specs[0]["timeframes"] == ["D"]

    names = get_all_base_model_names(ensemble_dir)
    assert len(names) == 1
    assert names[0].startswith("rsi_signal_D::")

    validate_ensemble_directory(ensemble_dir)


def test_list_features_autodetects_ensemble_dir_when_default_missing(tmp_path, monkeypatch) -> None:
    from ensemble import vault_manager

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vault_manager, "_DEFAULT_ENSEMBLE_DIR", None)

    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="autodetect",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    _write_domain_discrete_feature_file(ensemble_dir)
    monkeypatch.setattr(vault_manager, "_DEFAULT_ENSEMBLE_DIR", None)

    features_df = list_features()

    assert features_df.iloc[0]["feature_name"] == "rsi_signal_D"
    assert vault_manager._DEFAULT_ENSEMBLE_DIR == ensemble_dir


def test_validate_rejects_legacy_continuous_artifact(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="reject_continuous",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    _write_domain_discrete_feature_file(ensemble_dir)
    feature_file = Path(ensemble_dir) / "features" / "rsi_signal_D.json"
    with open(feature_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["base_models"][0]["model_type"] = "continuous_binning"
    with open(feature_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
        load_feature_base_models(
            feature_name="rsi_signal_D",
            ensemble_dir=ensemble_dir,
        )


def test_migrate_legacy_members_strips_empty_keys_and_updates_timestamp(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="migrate_members",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    _write_domain_discrete_feature_file(ensemble_dir)
    feature_file = Path(ensemble_dir) / "features" / "rsi_signal_D.json"
    with open(feature_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    original_updated_at = payload["updated_at"]
    payload["base_models"][0]["members"] = []
    with open(feature_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    updated_files = migrate_legacy_feature_members_schema(ensemble_dir)
    assert str(feature_file) in updated_files

    with open(feature_file, "r", encoding="utf-8") as handle:
        migrated = json.load(handle)
    assert "members" not in migrated["base_models"][0]
    assert migrated["updated_at"] != original_updated_at

    validate_ensemble_directory(ensemble_dir)


def test_migrate_legacy_members_rejects_non_empty_payload(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="reject_non_empty_members",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    _write_domain_discrete_feature_file(ensemble_dir)
    feature_file = Path(ensemble_dir) / "features" / "rsi_signal_D.json"
    with open(feature_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["base_models"][0]["members"] = [
        {"member_name": "legacy_member", "params": {"lookback": 2}}
    ]
    with open(feature_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with pytest.raises(ValueError, match="legacy members schema"):
        migrate_legacy_feature_members_schema(ensemble_dir)


def test_ensure_vault_cache_coverage_dedupes_specs_and_tickers(tmp_path, monkeypatch) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="cache_preflight",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    _write_domain_discrete_feature_file(ensemble_dir)
    feature_file = Path(ensemble_dir) / "features" / "rsi_signal_D.json"
    with open(feature_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["base_models"][0]["members"] = []
    with open(feature_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    calls: dict[str, object] = {}

    class _DummyCacheManager:
        def __init__(self) -> None:
            calls["initialized_at"] = datetime.now(timezone.utc)

        def ensure_vault_cache_coverage(self, **kwargs):  # noqa: ANN003
            calls["ensure_vault_cache_coverage"] = kwargs
            return {"total_tasks": 1, "rebuilt": 1, "validated": 0, "failed": 0}

    monkeypatch.setattr("utils.cache.cache_manager.CacheManager", _DummyCacheManager)

    summary = ensure_vault_cache_coverage(
        [ensemble_dir, ensemble_dir],
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )

    assert summary["rebuilt"] == 1
    refresh_kwargs = calls["ensure_vault_cache_coverage"]
    assert refresh_kwargs["refresh_mode"] == "missing_stale_only"
    assert refresh_kwargs["vault_ensemble_dirs"] == [ensemble_dir]
    assert "members" not in json.loads(feature_file.read_text(encoding="utf-8"))["base_models"][0]


def test_validate_rejects_multiple_base_models_in_feature_file(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="reject_multi_models",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    _write_domain_discrete_feature_file(ensemble_dir)
    feature_file = Path(ensemble_dir) / "features" / "rsi_signal_D.json"
    with open(feature_file, "r") as handle:
        payload = json.load(handle)
    payload["base_models"].append(dict(payload["base_models"][0], model_id="dup", model_name="rsi_signal_D::dup"))
    with open(feature_file, "w") as handle:
        json.dump(payload, handle, indent=2)

    with pytest.raises(ValueError, match="exactly one base model"):
        validate_ensemble_directory(ensemble_dir)


def test_create_ensemble_directory_reuses_existing_dir_with_matching_tickers(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="existing_match",
        direction=Direction.LONG,
        tickers=[Ticker.ES, Ticker.NQ],
        vault_root=str(tmp_path / "vault"),
    )

    repeated = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="existing_match",
        direction=Direction.LONG,
        tickers=[Ticker.NQ, Ticker.ES],
        vault_root=str(tmp_path / "vault"),
    )

    assert repeated == ensemble_dir
    config = json.loads((Path(ensemble_dir) / "ensemble_config.json").read_text())
    assert config["tickers"] == ["ES", "NQ"]


def test_create_ensemble_directory_rejects_existing_dir_with_different_tickers(tmp_path) -> None:
    create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="existing_mismatch",
        direction=Direction.LONG,
        tickers=[Ticker.ES],
        vault_root=str(tmp_path / "vault"),
    )

    with pytest.raises(ValueError, match="different tickers"):
        create_ensemble_directory(
            timeframe=TimeFrame.D,
            ensemble_name="existing_mismatch",
            direction=Direction.LONG,
            tickers=[Ticker.NQ],
            vault_root=str(tmp_path / "vault"),
        )


def test_list_ensembles_parses_named_dirs_and_counts_features(tmp_path) -> None:
    vault_root = tmp_path / "vault"
    ensemble_a = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="enum_a",
        direction=Direction.LONG,
        vault_root=str(vault_root),
    )
    ensemble_b = create_ensemble_directory(
        timeframe=TimeFrame.W,
        ensemble_name="enum_b",
        direction=Direction.SHORT,
        vault_root=str(vault_root),
    )
    _write_domain_discrete_feature_file(ensemble_a, feature_name="rsi_signal_D")
    _write_domain_discrete_feature_file(ensemble_b, feature_name="rsi_signal_W")
    (vault_root / "D" / "invalid_name").mkdir(parents=True)

    ensembles = list_ensembles(str(vault_root)).sort_values(
        ["timeframe", "ensemble_name"]
    ).reset_index(drop=True)

    assert list(ensembles["ensemble_name"]) == ["enum_a", "enum_b"]
    assert list(ensembles["direction"]) == ["long", "short"]
    assert list(ensembles["n_features"]) == [1, 1]
