"""Filesystem-backed CRUD + validation for ``research/specs/*.json``.

A spec's ``id`` is its filename stem. Reads are tolerant (a malformed/invalid file still lists,
flagged ``valid: false`` with its error) so the Library can show — and let you fix — a broken
spec instead of hiding it.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from frontend.api.paths import SPECS_DIR, repo_relative
from research.spec.serialization import spec_from_dict, spec_to_dict
from research.spec.strategy_spec import StrategySpec

# Registry DB path override (None → canonical data/registry.db).  Tests monkeypatch this.
_REGISTRY_DB_PATH: Path | None = None

_ID_RE = re.compile(r"[^a-z0-9_-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_upsert_spec(safe_id: str, spec: StrategySpec, normalized: dict[str, Any], path: Path) -> None:
    """Best-effort upsert of a spec into the registry DB.  Never raises."""
    try:
        from data_platform.registry import db as _db, writer as _writer  # lazy heavy import

        db_path = _REGISTRY_DB_PATH or _db.registry_path()
        spec_json_str = json.dumps(normalized)
        content_hash = hashlib.sha256(spec_json_str.encode()).hexdigest()
        now = _utc_now()
        record = _writer.SpecRecord(
            id=safe_id,
            name=spec.name,
            name_slug=safe_id,
            content_hash=content_hash,
            spec_json=spec_json_str,
            file_path=repo_relative(path),
            created_at=now,
            updated_at=now,
            spec_version="1.0",
            valid=1,
            error=None,
        )
        conn = _db.connect(db_path)
        try:
            with _db.transaction(conn):
                _writer.upsert_spec(conn, record)
        finally:
            conn.close()
    except Exception:
        pass


def _registry_mark_spec_deleted(safe_id: str) -> None:
    """Best-effort: mark a spec valid=0 in the registry (don't delete — runs reference it).  Never raises."""
    try:
        from data_platform.registry import db as _db  # lazy heavy import

        db_path = _REGISTRY_DB_PATH or _db.registry_path()
        conn = _db.connect(db_path)
        try:
            conn.execute(
                "UPDATE specs SET valid = 0, updated_at = ? WHERE id = ?",
                (_utc_now(), safe_id),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def _slugify(name: str) -> str:
    slug = _ID_RE.sub("_", name.strip().lower()).strip("_")
    return slug or "spec"


def _spec_path(spec_id: str):
    safe = _slugify(spec_id)
    return SPECS_DIR / f"{safe}.json", safe


def _summary(spec_id: str, data: dict[str, Any], spec: StrategySpec | None, error: str | None,
             mtime: float | None) -> dict[str, Any]:
    """Compact card for the Library list (works for valid and invalid specs)."""

    signal = data.get("signal") or {}
    return {
        "id": spec_id,
        "name": data.get("name", spec_id),
        "hypothesis": data.get("hypothesis", ""),
        "module": (signal.get("module_name") if isinstance(signal, dict) else None),
        "tickers": data.get("tickers", []),
        "timeframe": data.get("timeframe"),
        "mode": data.get("mode"),
        "direction": data.get("direction"),
        "data_feed": data.get("data_feed"),
        "vault_sleeve": (data.get("vault") or {}).get("weight_hierarchy_group")
        if isinstance(data.get("vault"), dict)
        else None,
        "num_combos": spec.signal.num_combos if spec is not None else None,
        "valid": spec is not None,
        "error": error,
        "updated_at": (
            datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat() if mtime else None
        ),
    }


def list_specs() -> list[dict[str, Any]]:
    """Summaries for every spec JSON in ``research/specs/`` (newest first)."""

    if not SPECS_DIR.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for path in SPECS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            summaries.append(_summary(path.stem, {}, None, f"Unreadable JSON: {exc}", None))
            continue
        spec, error = _try_build(data)
        summaries.append(_summary(path.stem, data, spec, error, path.stat().st_mtime))
    return sorted(summaries, key=lambda item: item.get("updated_at") or "", reverse=True)


def _try_build(data: dict[str, Any]) -> tuple[StrategySpec | None, str | None]:
    try:
        return spec_from_dict(data), None
    except (ValueError, KeyError, TypeError) as exc:
        return None, str(exc)


def get_spec(spec_id: str) -> dict[str, Any]:
    """Raw stored spec dict + its validation result. Raises FileNotFoundError if absent."""

    path, safe = _spec_path(spec_id)
    if not path.exists():
        raise FileNotFoundError(f"Spec not found: {spec_id}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    spec, error = _try_build(data)
    return {
        "id": safe,
        "spec": data,
        "validation": validation_result(data, spec, error),
    }


def save_spec(spec_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate then persist a spec. Raises ValueError if the payload is not a valid spec."""

    spec = spec_from_dict(payload)  # raises on invalid
    path, safe = _spec_path(spec_id)
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    normalized = spec_to_dict(spec)
    path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    # Registry upsert (best-effort; file stays canonical).
    _registry_upsert_spec(safe, spec, normalized, path)
    return {"id": safe, "spec": normalized, "validation": validation_result(normalized, spec, None)}


def delete_spec(spec_id: str) -> None:
    path, safe = _spec_path(spec_id)
    if not path.exists():
        raise FileNotFoundError(f"Spec not found: {spec_id}")
    path.unlink()
    # Mark deleted in registry (don't remove row — runs reference it).
    _registry_mark_spec_deleted(safe)


def derive_id(payload: dict[str, Any]) -> str:
    """The id to store a new spec under (slug of its name)."""

    return _slugify(str(payload.get("name", "spec")))


def validation_result(
    data: dict[str, Any],
    spec: StrategySpec | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Validate a payload and compute the derived fields the form shows live."""

    if spec is None and error is None:
        spec, error = _try_build(data)
    if spec is None:
        return {"valid": False, "error": error, "derived": None}

    windows = spec.windows
    derived = {
        "num_combos": spec.signal.num_combos,
        "resolved_fill_feed": spec.execution.resolved_fill_feed().value,
        "uses_limit": spec.execution.uses_limit(),
        # Feed selection is gone: signals are always additive futures and both result lanes
        # (ratio-futures + CFD) are produced every run. Kept in the shape (the frontend stops
        # rendering the feed badge) but now a fixed contract label, not a per-spec selection.
        "feed_literal": "futures (signals) · both (results)",
        "windows_explicit": windows is not None,
        "windows": (
            None
            if windows is None
            else {
                "train": [windows.train[0].isoformat(), windows.train[1].isoformat()],
                "validation": [
                    windows.validation[0].isoformat(),
                    windows.validation[1].isoformat(),
                ],
                "test": [windows.test[0].isoformat(), windows.test[1].isoformat()],
            }
        ),
    }
    return {"valid": True, "error": None, "derived": derived}


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Public validate entrypoint for the form (debounced live validation)."""

    return validation_result(payload)
