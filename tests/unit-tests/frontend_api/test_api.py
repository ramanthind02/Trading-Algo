"""Unit tests for the frontend research API service layer.

These exercise the pure service functions (schema, spec CRUD/validation, artifact-path safety,
run-manager guards) without running the real pipeline or needing an HTTP server. A thin
TestClient smoke (skipped when httpx is absent) covers the HTTP wiring.
"""

from __future__ import annotations

import json

import pytest

from frontend.api import artifacts, schema, spec_store, vault_browser
from frontend.api.runs import SpecRunManager


# ── schema ───────────────────────────────────────────────────────────────────


def test_form_schema_has_all_field_options():
    s = schema.form_schema()
    for key in (
        "tickers",
        "timeframes",
        "modes",
        "directions",
        "vol_scalings",
        "order_policies",
        "vault_sleeves",
        "max_grid_combos",
    ):
        assert key in s, key
    # The feed dropdown is removed: signals are always additive futures and both result lanes
    # (ratio-futures + CFD) are produced every run, so the schema no longer offers a feed select.
    assert "data_feeds" not in s
    # The schema exposes the built-in sleeves unioned with any user-defined ones (an untracked
    # custom_sleeves.json adds to them at runtime), so assert the built-ins are all present and the
    # catalog is at least that size rather than hard-coding a count that drifts per environment.
    from ensemble.vault.constants import _BASE_WEIGHT_HIERARCHY_GROUP_DIR_NAMES

    assert _BASE_WEIGHT_HIERARCHY_GROUP_DIR_NAMES <= set(s["vault_sleeves"])
    assert len(s["vault_sleeves"]) >= len(_BASE_WEIGHT_HIERARCHY_GROUP_DIR_NAMES) == 13
    assert s["timeframes"]["daily"] == ["D", "W", "M"]
    assert {o["value"] for o in s["directions"]} == {"long", "short", "long_short"}
    assert s["max_grid_combos"] == 300


def test_module_catalog_dedupes_and_categorizes():
    catalog = schema.module_catalog()
    names = {m["name"] for m in catalog}
    classes = [m["class_name"] for m in catalog]
    assert "double7s" in names
    assert len(classes) == len(set(classes))  # deduped by class
    double7s = next(m for m in catalog if m["name"] == "double7s")
    assert double7s["category"] == "mean_reversion"


def test_module_detail_introspects_params():
    detail = schema.module_detail("double7s")
    param_names = {p["name"] for p in detail["params"]}
    assert {"short_period", "ma_period"} <= param_names


def test_module_detail_unknown_raises():
    with pytest.raises(KeyError):
        schema.module_detail("not_a_real_module")


# ── spec CRUD / validation (tmp specs dir) ───────────────────────────────────


@pytest.fixture()
def tmp_specs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(spec_store, "SPECS_DIR", tmp_path)
    return tmp_path


def _payload(**overrides) -> dict:
    base = {
        "name": "unit_rsi_mr",
        "hypothesis": "test",
        "author": "tester",
        "created": "2026-01-01T00:00:00",
        "tickers": ["ES"],
        "data_feed": "darwinex_cfd",
        "mode": "daily",
        "timeframe": "D",
        "signal": {"module_name": "rsi_signal", "param_grid": {"rsi_period": [2, 3]}},
        "direction": "long_short",
        "vault": {"weight_hierarchy_group": "mean_reversion_indices", "ensemble_name": "x"},
    }
    base.update(overrides)
    return base


def test_validate_payload_valid_has_derived():
    result = spec_store.validate_payload(_payload())
    assert result["valid"] is True
    assert result["derived"]["num_combos"] == 2
    assert result["derived"]["resolved_fill_feed"] == "signal_bar"
    # ``feed_literal`` is now a fixed contract label (no per-spec feed selection): signals are
    # always additive futures and both result lanes are produced. The CFD ``data_feed`` in the
    # payload no longer drives it.
    assert result["derived"]["feed_literal"] == "futures (signals) · both (results)"


def test_validate_payload_invalid_grid():
    bad = _payload(signal={"module_name": "x", "param_grid": {"a": list(range(21)), "b": list(range(21))}})
    result = spec_store.validate_payload(bad)
    assert result["valid"] is False
    assert "exceeding the cap" in result["error"]


def test_spec_crud_round_trip(tmp_specs_dir):
    saved = spec_store.save_spec("unit_rsi_mr", _payload())
    assert saved["id"] == "unit_rsi_mr"
    assert (tmp_specs_dir / "unit_rsi_mr.json").exists()

    listed = spec_store.list_specs()
    assert [s["id"] for s in listed] == ["unit_rsi_mr"]
    assert listed[0]["valid"] is True
    assert listed[0]["num_combos"] == 2

    fetched = spec_store.get_spec("unit_rsi_mr")
    assert fetched["spec"]["name"] == "unit_rsi_mr"
    assert fetched["validation"]["valid"] is True

    spec_store.delete_spec("unit_rsi_mr")
    assert spec_store.list_specs() == []


def test_save_invalid_spec_raises(tmp_specs_dir):
    with pytest.raises(ValueError):
        spec_store.save_spec("bad", _payload(mode="intraday"))  # intraday + D timeframe


def test_list_specs_flags_broken_file(tmp_specs_dir):
    (tmp_specs_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
    listed = spec_store.list_specs()
    broken = next(s for s in listed if s["id"] == "broken")
    assert broken["valid"] is False
    assert broken["error"]


def test_get_missing_spec_raises(tmp_specs_dir):
    with pytest.raises(FileNotFoundError):
        spec_store.get_spec("does_not_exist")


# ── artifact path safety ─────────────────────────────────────────────────────


def test_artifact_path_traversal_rejected():
    with pytest.raises(ValueError):
        artifacts.preview("../../etc/passwd")


def test_artifact_missing_in_allowed_root_raises():
    from frontend.api.paths import resolve_artifact_path

    with pytest.raises(FileNotFoundError):
        resolve_artifact_path("feature_research/shared_results/signed_signal/__nope__/x.csv")


# ── run manager guards (no real pipeline run) ────────────────────────────────


def test_run_manager_rejects_invalid_spec():
    manager = SpecRunManager()
    with pytest.raises(ValueError):
        manager.start("bad", _payload(mode="intraday"))


def test_run_manager_rejects_unknown_phase():
    manager = SpecRunManager()
    with pytest.raises(ValueError, match="Unknown phase"):
        manager.start("x", _payload(), phase="nonsense")


# ── vault browser (synthetic vault) ──────────────────────────────────────────


@pytest.fixture()
def fake_vault(tmp_path, monkeypatch):
    ens = tmp_path / "D" / "mean_reversion_indices" / "double7s_long"
    (ens / "features").mkdir(parents=True)
    import json

    (ens / "ensemble_config.json").write_text(
        json.dumps(
            {
                "timeframe": "D",
                "ensemble_name": "double7s",
                "direction": "long",
                "tickers": ["ES", "NQ"],
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (ens / "features" / "double7s_signal_D_maPeriod_200_shortPeriod_10.json").write_text(
        json.dumps(
            {
                "feature_name": "double7s_signal_D_maPeriod_200_shortPeriod_10",
                "bias_node_spec": {
                    "module_name": "double7s",
                    "timeframes": ["D"],
                    "params": {"short_period": 10, "ma_period": 200},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(vault_browser._VAULT_ROOTS, "prop", tmp_path)
    return tmp_path


def test_list_vault_catalogs_features(fake_vault):
    listing = vault_browser.list_vault("prop")
    assert len(listing["features"]) == 1
    f = listing["features"][0]
    assert f["ensemble_name"] == "double7s"
    assert f["sleeve"] == "mean_reversion_indices"
    assert f["timeframe"] == "D"
    assert f["tickers"] == ["ES", "NQ"]


def test_spec_from_feature_round_trips_to_valid_spec(fake_vault):
    listing = vault_browser.list_vault("prop")
    spec_dict = vault_browser.spec_from_feature("prop", listing["features"][0]["path"])
    # The reconstructed spec must validate (single-value grid from the stored params).
    result = spec_store.validate_payload(spec_dict)
    assert result["valid"] is True
    assert spec_dict["signal"]["module_name"] == "double7s"
    assert spec_dict["signal"]["param_grid"] == {"short_period": [10], "ma_period": [200]}
    assert spec_dict["vault"]["weight_hierarchy_group"] == "mean_reversion_indices"
    assert spec_dict["direction"] == "long"


# ── HTTP wiring smoke (skipped without httpx) ────────────────────────────────


def test_http_health_and_schema():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from frontend.api.server import app

    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    assert "vault_sleeves" in client.get("/api/schema").json()
