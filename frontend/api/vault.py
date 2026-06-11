"""Vault promotion for a spec — preview (dry-run) and the human-gated commit.

Reuses the existing :mod:`research.feature.ui.vault_save` logic, but drives it from a
``StrategySpec`` (via the adapter) instead of the canonical config. The real write goes through
``BaseModel.save_to_vault`` (plain Python I/O), which the Edit/Write guard hook does not block.

Eligibility (a passing portfolio-addition gate from a validation run) is surfaced as-is — it
becomes satisfiable once portfolio research is wired (Phase 5). Preview always works.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.feature.ui.vault_save import (
    assess_vault_save_eligibility,
    execute_vault_save,
    vault_save_execution_to_dict,
)

_log = logging.getLogger(__name__)

# Registry DB path override (None → canonical data/registry.db).  Tests monkeypatch this.
_REGISTRY_DB_PATH: Path | None = None


def _compute_spec_hash(payload: dict[str, Any]) -> str:
    """sha256 over canonical json.dumps(sort_keys=True, separators=(',',':'))."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _resolve_producing_run(spec_hash: str) -> tuple[str | None, str | None]:
    """Look up the newest completed validation run for spec_hash.

    Returns ``(run_id, gate_report_path)`` where gate_report_path is the
    absolute path to the portfolio_addition_report.json when the file exists.
    Both values are None when no matching run is found or the DB is absent.
    """
    try:
        from data_platform.registry import db as _db, reader as _reader
        from frontend.api.paths import REPO_ROOT

        db_path = _REGISTRY_DB_PATH or _db.registry_path()
        conn = _db.connect_readonly(db_path)
        try:
            rows = _reader.runs_for_spec_hash(conn, spec_hash, kind="validation")
            completed = [r for r in rows if r["status"] == "completed"]
        finally:
            conn.close()

        for row in completed:
            run_id = str(row["run_id"])
            viz_dir = row["viz_dir"]
            gate_report: str | None = None
            if viz_dir:
                report_path = REPO_ROOT / viz_dir / "portfolio_addition_report.json"
                if report_path.is_file():
                    gate_report = str(report_path)
            return run_id, gate_report

    except Exception as exc:
        _log.debug("_resolve_producing_run: registry lookup failed (%s)", exc)

    return None, None


def _upsert_vault_entry(
    result_dict: dict[str, Any],
    *,
    producing_run_id: str | None,
    gate_report_path: str | None,
    bias_spec: dict[str, Any],
    feature_file_path: str,
    config_json: str,
) -> None:
    """Upsert a vault_entries registry row after a successful commit."""
    try:
        from data_platform.registry import db as _db, writer as _writer

        db_path = _REGISTRY_DB_PATH or _db.registry_path()
        now = datetime.now(timezone.utc).isoformat()
        timeframe = str(bias_spec.get("timeframes", ["D"])[0])
        entry = _writer.VaultEntry(
            vault_profile=str(result_dict.get("vault_profile") or "prop"),
            timeframe=timeframe,
            ensemble_leaf=Path(str(result_dict["ensemble_dir"])).name,
            feature_name=str(result_dict["feature_column"]),
            feature_column=str(result_dict["feature_column"]),
            module_name=str(bias_spec.get("module_name", "")),
            direction=str(result_dict.get("direction") or ""),
            weight_hierarchy_group=result_dict.get("weight_hierarchy_group"),
            producing_run_id=producing_run_id,
            gate_report_path=gate_report_path,
            promoted_by="api",
            config_json=config_json,
            file_path=feature_file_path,
            created_at=now,
            updated_at=now,
        )
        conn = _db.connect(db_path)
        try:
            with _db.transaction(conn):
                # Ensure the sleeve FK exists before the vault_entries insert.
                if entry.weight_hierarchy_group is not None:
                    conn.execute(
                        "INSERT OR IGNORE INTO sleeves (name) VALUES (?)",
                        (entry.weight_hierarchy_group,),
                    )
                _writer.upsert_vault_entry(conn, entry)
        finally:
            conn.close()
    except Exception as exc:
        _log.warning("_upsert_vault_entry: registry write failed (%s)", exc)


def _config(spec_dict: dict[str, Any]):
    # Lazy imports: the adapter pulls the heavy research configs.
    from research.spec import to_feature_config
    from research.spec.serialization import spec_from_dict

    return to_feature_config(spec_from_dict(spec_dict))


def preview(spec_dict: dict[str, Any]) -> dict[str, Any]:
    """Eligibility + a dry-run of exactly what would be written to the vault."""

    config = _config(spec_dict)
    spec_hash = _compute_spec_hash(spec_dict)
    eligibility = assess_vault_save_eligibility(config, spec_hash=spec_hash)
    result = None
    error: str | None = None
    try:
        result = vault_save_execution_to_dict(execute_vault_save(config, dry_run=True))
    except (ValueError, FileNotFoundError) as exc:
        error = str(exc)
    return {"eligibility": eligibility, "preview": result, "preview_error": error}


def commit(spec_dict: dict[str, Any]) -> dict[str, Any]:
    """Write the spec's eval combo to the vault. Raises ValueError if not eligible."""

    spec_hash = _compute_spec_hash(spec_dict)
    producing_run_id, gate_report_path = _resolve_producing_run(spec_hash)
    if producing_run_id is None:
        _log.info(
            "commit: no completed validation run found for spec_hash=%s; "
            "vault_entries.producing_run_id will be NULL",
            spec_hash[:16],
        )

    config = _config(spec_dict)
    result = execute_vault_save(
        config,
        dry_run=False,
        producing_run_id=producing_run_id,
        spec_hash=spec_hash,
    )
    result_dict = vault_save_execution_to_dict(result)

    # Derive feature file path and read its JSON for the registry row.
    try:
        from ensemble.vault.feature_files import feature_json_stem

        feature_file = (
            Path(result.ensemble_dir)
            / "features"
            / f"{feature_json_stem(result.feature_column)}.json"
        )
        config_json = (
            feature_file.read_text(encoding="utf-8")
            if feature_file.is_file()
            else json.dumps(result.bias_spec)
        )
        _upsert_vault_entry(
            result_dict,
            producing_run_id=producing_run_id,
            gate_report_path=gate_report_path,
            bias_spec=result.bias_spec,
            feature_file_path=str(feature_file),
            config_json=config_json,
        )
    except Exception as exc:
        _log.warning("commit: vault_entries upsert skipped (%s)", exc)

    return result_dict
