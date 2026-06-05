from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from flask import Flask, jsonify, redirect, request, send_file, send_from_directory
from flask_cors import CORS

from research.feature.config import load_config
from research.feature.ui.job_manager import WorkspaceJobManager
from research.feature.ui.planner import (
    build_phase_plan,
    build_ui_defaults,
    build_ui_request,
    defaults_to_dict,
    plan_to_dict,
    resolve_ui_config,
)
from research.feature.shared import FeatureResearchPhase
from research.feature.ui.pivot_data import load_parameter_sensitivity_pivot_payload_for_phase
from research.feature.ui.workspace import (
    build_workspace_view,
    load_artifact_preview,
    resolve_workspace_artifact_path,
)
from research.feature.ui.vault_save import (
    build_vault_commit_view,
    execute_vault_save,
    vault_save_execution_to_dict,
)
from research.portfolio.config import load_config as load_portfolio_config
from research.portfolio.ui.job_manager import PortfolioWorkspaceJobManager
from research.portfolio.ui.planner import (
    build_phase_plan as build_portfolio_phase_plan,
    build_ui_defaults as build_portfolio_ui_defaults,
    build_ui_request as build_portfolio_ui_request,
    defaults_to_dict as portfolio_defaults_to_dict,
    plan_to_dict as portfolio_plan_to_dict,
)
from research.portfolio.ui.workspace import (
    build_workspace_view as build_portfolio_workspace_view,
    load_artifact_preview as load_portfolio_artifact_preview,
    resolve_workspace_artifact_path as resolve_portfolio_artifact_path,
)

JOB_MANAGER = WorkspaceJobManager()
PORTFOLIO_JOB_MANAGER = PortfolioWorkspaceJobManager()

app = Flask(__name__, static_folder=".")
CORS(app)


@app.route("/")
def index():
    return redirect("/feature-research", code=302)


@app.route("/feature-research")
def feature_research_ui():
    return send_from_directory(".", "feature_research.html")


@app.route("/portfolio-research")
def portfolio_research_ui():
    return send_from_directory(".", "portfolio_research.html")


@app.route("/<path:filename>")
def static_files(filename: str):
    return send_from_directory(".", filename)


@app.route("/api/feature-research/defaults")
def feature_research_defaults():
    config = load_config()
    defaults = build_ui_defaults(config)
    initial_request = build_ui_request(
        phase_name=defaults.default_phase.value,
        tickers_text=",".join(ticker.name for ticker in defaults.default_tickers),
        fallback_tickers=defaults.default_tickers,
    )
    plan = build_phase_plan(config, initial_request)
    workspace = build_workspace_view(
        config,
        initial_request,
        current_job=JOB_MANAGER.latest_job(),
    )
    return jsonify(
        {
            "defaults": defaults_to_dict(defaults),
            "plan": plan_to_dict(plan),
            "workspace": workspace,
            "job": JOB_MANAGER.latest_job(),
        }
    )


@app.route("/api/feature-research/plan", methods=["POST"])
def feature_research_plan():
    raw_payload = request.get_json(silent=True)
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    config = load_config()
    try:
        ui_request = _build_ui_request_from_payload(config, payload)
        plan = build_phase_plan(config, ui_request)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    workspace = build_workspace_view(
        config,
        ui_request,
        current_job=JOB_MANAGER.latest_job(),
    )
    return jsonify(
        {
            "plan": plan_to_dict(plan),
            "workspace": workspace,
            "job": JOB_MANAGER.latest_job(),
        }
    )


@app.route("/api/feature-research/run", methods=["POST"])
def feature_research_run():
    raw_payload = request.get_json(silent=True)
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    config = load_config()
    try:
        ui_request = _build_ui_request_from_payload(config, payload)
        job = JOB_MANAGER.start_job(config, ui_request)
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"job": job})


@app.route("/api/feature-research/jobs/latest")
def feature_research_latest_job():
    return jsonify({"job": JOB_MANAGER.latest_job()})


@app.route("/api/feature-research/jobs/<job_id>")
def feature_research_job(job_id: str):
    job = JOB_MANAGER.job_snapshot(job_id)
    if job is None:
        return jsonify({"error": "Job not found."}), 404
    return jsonify({"job": job})


@app.route("/api/feature-research/pivot-data")
def feature_research_pivot_data():
    phase_name = request.args.get("phase", FeatureResearchPhase.EXPLORATION.value).strip()
    try:
        phase = FeatureResearchPhase(phase_name)
    except ValueError:
        return jsonify({"error": f"Unknown phase: {phase_name}"}), 400
    config = load_config()
    payload = load_parameter_sensitivity_pivot_payload_for_phase(config, phase)
    if payload is None:
        return jsonify(
            {
                "available": False,
                "phase": phase.value,
                "message": "Parameter sensitivity CSV exports are not available for this phase.",
            }
        )
    return jsonify({"available": True, "phase": phase.value, **payload})


@app.route("/api/feature-research/artifact-preview")
def feature_research_artifact_preview():
    relative_path = request.args.get("path", "").strip()
    if not relative_path:
        return jsonify({"error": "Artifact path is required."}), 400
    try:
        preview = load_artifact_preview(relative_path)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(preview)


@app.route("/api/feature-research/vault-save/preview", methods=["POST"])
def feature_research_vault_save_preview():
    raw_payload = request.get_json(silent=True)
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    config = load_config()
    try:
        ui_request = _build_ui_request_from_payload(config, payload)
        commit_view = build_vault_commit_view(config, ui_request)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"vault_commit": commit_view})


@app.route("/api/feature-research/vault-save", methods=["POST"])
def feature_research_vault_save():
    raw_payload = request.get_json(silent=True)
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    config = load_config()
    try:
        ui_request = _build_ui_request_from_payload(config, payload)
        configured, _selection = resolve_ui_config(config, ui_request)
        result = execute_vault_save(configured, dry_run=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(
        {
            "saved": True,
            "result": vault_save_execution_to_dict(result),
        }
    )


@app.route("/api/feature-research/artifact-raw")
def feature_research_artifact_raw():
    relative_path = request.args.get("path", "").strip()
    if not relative_path:
        return jsonify({"error": "Artifact path is required."}), 400
    try:
        artifact_path = resolve_workspace_artifact_path(relative_path)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 404
    return send_file(artifact_path)


def _payload_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    return None if value is None else str(value)


def _build_ui_request_from_payload(
    config,
    payload: Mapping[str, object],
):
    defaults = build_ui_defaults(config)
    return build_ui_request(
        phase_name=_payload_text(payload, "phase") or defaults.default_phase.value,
        tickers_text=_payload_text(payload, "tickers"),
        fallback_tickers=defaults.default_tickers,
        portfolio_tickers_text=_payload_text(payload, "portfolio_tickers"),
        train_start=_payload_text(payload, "train_start"),
        train_end=_payload_text(payload, "train_end"),
        val_start=_payload_text(payload, "val_start"),
        val_end=_payload_text(payload, "val_end"),
        test_start=_payload_text(payload, "test_start"),
        test_end=_payload_text(payload, "test_end"),
    )


def _build_portfolio_ui_request_from_payload(payload: Mapping[str, object]):
    config = load_portfolio_config()
    defaults = build_portfolio_ui_defaults(config)
    return build_portfolio_ui_request(
        phase_name=_payload_text(payload, "phase") or defaults.default_phase.value,
        tickers_text=_payload_text(payload, "tickers"),
        fallback_tickers=defaults.default_tickers,
        fit_mode_name=_payload_text(payload, "fit_mode"),
    )


@app.route("/api/portfolio-research/defaults")
def portfolio_research_defaults():
    config = load_portfolio_config()
    defaults = build_portfolio_ui_defaults(config)
    initial_request = build_portfolio_ui_request(
        phase_name=defaults.default_phase.value,
        tickers_text=",".join(ticker.name for ticker in defaults.default_tickers),
        fallback_tickers=defaults.default_tickers,
    )
    plan = build_portfolio_phase_plan(config, initial_request)
    workspace = build_portfolio_workspace_view(
        config,
        initial_request,
        current_job=PORTFOLIO_JOB_MANAGER.latest_job(),
    )
    return jsonify(
        {
            "defaults": portfolio_defaults_to_dict(defaults),
            "plan": portfolio_plan_to_dict(plan),
            "workspace": workspace,
            "job": PORTFOLIO_JOB_MANAGER.latest_job(),
        }
    )


@app.route("/api/portfolio-research/plan", methods=["POST"])
def portfolio_research_plan():
    payload = request.get_json(silent=True) or {}
    config = load_portfolio_config()
    try:
        ui_request = _build_portfolio_ui_request_from_payload(payload)
        plan = build_portfolio_phase_plan(config, ui_request)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    workspace = build_portfolio_workspace_view(
        config,
        ui_request,
        current_job=PORTFOLIO_JOB_MANAGER.latest_job(),
    )
    return jsonify(
        {
            "plan": portfolio_plan_to_dict(plan),
            "workspace": workspace,
            "job": PORTFOLIO_JOB_MANAGER.latest_job(),
        }
    )


@app.route("/api/portfolio-research/run", methods=["POST"])
def portfolio_research_run():
    payload = request.get_json(silent=True) or {}
    config = load_portfolio_config()
    try:
        ui_request = _build_portfolio_ui_request_from_payload(payload)
        job = PORTFOLIO_JOB_MANAGER.start_job(config, ui_request)
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"job": job})


@app.route("/api/portfolio-research/jobs/<job_id>")
def portfolio_research_job(job_id: str):
    job = PORTFOLIO_JOB_MANAGER.job_snapshot(job_id)
    if job is None:
        return jsonify({"error": "Job not found."}), 404
    return jsonify({"job": job})


@app.route("/api/portfolio-research/artifact-preview")
def portfolio_research_artifact_preview():
    relative_path = request.args.get("path", "").strip()
    if not relative_path:
        return jsonify({"error": "Artifact path is required."}), 400
    try:
        preview = load_portfolio_artifact_preview(relative_path)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(preview)


@app.route("/api/portfolio-research/artifact-raw")
def portfolio_research_artifact_raw():
    relative_path = request.args.get("path", "").strip()
    if not relative_path:
        return jsonify({"error": "Artifact path is required."}), 400
    try:
        artifact_path = resolve_portfolio_artifact_path(relative_path)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 404
    return send_file(artifact_path)


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5001)
