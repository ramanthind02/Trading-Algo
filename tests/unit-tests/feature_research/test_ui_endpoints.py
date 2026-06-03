from __future__ import annotations

from types import SimpleNamespace

import frontend.app as frontend_app


def test_feature_research_defaults_endpoint_returns_config_summary(monkeypatch) -> None:
    monkeypatch.setattr(frontend_app, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(
        frontend_app,
        "build_ui_defaults",
        lambda _config: SimpleNamespace(
            default_phase=SimpleNamespace(value="exploration"),
            default_tickers=(),
        ),
    )
    monkeypatch.setattr(frontend_app, "build_ui_request", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(frontend_app, "build_phase_plan", lambda _config, _request: SimpleNamespace())
    monkeypatch.setattr(
        frontend_app,
        "defaults_to_dict",
        lambda _defaults: {
            "default_phase": "exploration",
            "default_tickers": "ES",
            "ticker_options": ["ES", "NQ"],
            "phase_options": [],
            "workflow_steps": [],
            "configured_module": "stacked_sma_long_only",
            "configured_module_description": "Stacked SMA node.",
            "configured_search_space": "495 ordered combos from config.",
            "config_source_path": "feature_research/config.py",
            "research_window": None,
        },
    )
    monkeypatch.setattr(frontend_app, "plan_to_dict", lambda _plan: {"phase": "exploration"})
    monkeypatch.setattr(
        frontend_app,
        "build_workspace_view",
        lambda _config, _request, current_job: {"job": current_job, "phases": []},
    )
    monkeypatch.setattr(frontend_app, "JOB_MANAGER", SimpleNamespace(latest_job=lambda: None))

    with frontend_app.app.test_client() as client:
        response = client.get("/api/feature-research/defaults")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["defaults"]["configured_module"] == "stacked_sma_long_only"
    assert payload["defaults"]["configured_search_space"] == "495 ordered combos from config."


def test_feature_research_pivot_data_endpoint_returns_payload(monkeypatch) -> None:
    monkeypatch.setattr(frontend_app, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(
        frontend_app,
        "load_parameter_sensitivity_pivot_payload_for_phase",
        lambda _config, _phase: {
            "default_metric": "t_stat",
            "metrics": ["t_stat", "sharpe"],
            "datasets": [],
            "scale": {"vmin": 0.0, "vmax": 1.0, "observed_min": 0.0, "observed_max": 1.0},
        },
    )

    with frontend_app.app.test_client() as client:
        response = client.get("/api/feature-research/pivot-data?phase=exploration")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["available"] is True
    assert payload["default_metric"] == "t_stat"


def test_feature_research_vault_save_preview_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(frontend_app, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(
        frontend_app,
        "_build_ui_request_from_payload",
        lambda _config, _payload: SimpleNamespace(),
    )
    monkeypatch.setattr(
        frontend_app,
        "build_vault_commit_view",
        lambda _config, _request: {
            "configured": True,
            "ready": True,
            "gate_status": "passed",
            "blockers": [],
            "preview": {"ensemble_dir_repo_relative": "vault/D/momentum/test_long"},
        },
    )

    with frontend_app.app.test_client() as client:
        response = client.post(
            "/api/feature-research/vault-save/preview",
            json={"phase": "validation", "tickers": "ES"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["vault_commit"]["ready"] is True


def test_feature_research_vault_save_endpoint_writes(monkeypatch) -> None:
    monkeypatch.setattr(frontend_app, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(
        frontend_app,
        "_build_ui_request_from_payload",
        lambda _config, _payload: SimpleNamespace(),
    )
    monkeypatch.setattr(
        frontend_app,
        "resolve_ui_config",
        lambda _config, _request: (SimpleNamespace(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        frontend_app,
        "execute_vault_save",
        lambda _config, dry_run=False: SimpleNamespace(
            dry_run=False,
            vault_root="vault",
            ensemble_dir="vault/D/momentum/test_long",
            ensemble_dir_repo_relative="vault/D/momentum/test_long",
            tickers=("ES",),
            direction="long",
            feature_column="sma_signal_D_period_200",
            model_id="signed_signal_sma_signal_D_period_200",
            bias_spec={"module_name": "sma_above_filter"},
            vault_profile="prop",
            weight_hierarchy_group="momentum",
        ),
    )
    monkeypatch.setattr(
        frontend_app,
        "vault_save_execution_to_dict",
        lambda result: {
            "ensemble_dir_repo_relative": result.ensemble_dir_repo_relative,
            "model_id": result.model_id,
        },
    )

    with frontend_app.app.test_client() as client:
        response = client.post(
            "/api/feature-research/vault-save",
            json={"phase": "validation", "tickers": "ES"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["saved"] is True
    assert payload["result"]["model_id"]


def test_feature_research_pivot_data_endpoint_reports_missing_inputs(monkeypatch) -> None:
    monkeypatch.setattr(frontend_app, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(
        frontend_app,
        "load_parameter_sensitivity_pivot_payload_for_phase",
        lambda _config, _phase: None,
    )

    with frontend_app.app.test_client() as client:
        response = client.get("/api/feature-research/pivot-data?phase=exploration")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["available"] is False


def test_feature_research_plan_endpoint_accepts_payload_without_module_fields(monkeypatch) -> None:
    captured_payload: dict[str, object] = {}

    monkeypatch.setattr(frontend_app, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(frontend_app, "build_ui_defaults", lambda _config: SimpleNamespace(default_phase=SimpleNamespace(value="exploration"), default_tickers=()))

    def _build_ui_request_from_payload(_config, payload):
        captured_payload.update(payload)
        return SimpleNamespace()

    monkeypatch.setattr(frontend_app, "_build_ui_request_from_payload", _build_ui_request_from_payload)
    monkeypatch.setattr(frontend_app, "build_phase_plan", lambda _config, _request: SimpleNamespace())
    monkeypatch.setattr(frontend_app, "plan_to_dict", lambda _plan: {"phase": "validation"})
    monkeypatch.setattr(
        frontend_app,
        "build_workspace_view",
        lambda _config, _request, current_job: {"job": current_job, "phases": []},
    )
    monkeypatch.setattr(frontend_app, "JOB_MANAGER", SimpleNamespace(latest_job=lambda: None))

    with frontend_app.app.test_client() as client:
        response = client.post(
            "/api/feature-research/plan",
            json={"phase": "validation", "tickers": "ES,NQ"},
        )

    assert response.status_code == 200
    assert "module" not in captured_payload
    assert "module_params" not in captured_payload
