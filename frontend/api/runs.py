"""In-process run manager: launch a spec's exploration phase and capture its log.

One run at a time (like the legacy ``WorkspaceJobManager``). Each run's artifacts are isolated
under a per-run subfolder keyed by ``run_id`` (reports + visualization CSVs), so results read
back exactly that run's outputs and a run can be deleted independently. Heavy pipeline imports
(the adapter, the exploration orchestrator, the EWSD/feed setters) are deferred to the worker
thread so importing this module — and starting the API — stays fast.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from frontend.api.paths import (
    IN_SAMPLE_RESULTS_ROOT,
    REPO_ROOT,
    SHARED_RESULTS_ROOT,
    repo_relative,
)
from research.spec.serialization import spec_from_dict

# Where the adapter writes a spec's reports (see research/spec/adapter.py _FEATURE_RESULTS_ROOT).
_REPORTS_PREFIX = "feature_research/shared_results/signed_signal"
# Per-run visualization root per phase. The pipeline writers append the ``visualization`` subdir
# (validation/OOS also append the phase), so we point ``visualization_parent_dir`` at the per-run
# root below and the CSVs land at ``<root>/visualization[/<phase>]``.
_VIZ_ROOT = IN_SAMPLE_RESULTS_ROOT
_VALIDATION_VIZ_ROOT = SHARED_RESULTS_ROOT

# Subfolder the pipeline visualization writers append under ``visualization_parent_dir``.
_VIZ_SUBDIR = "visualization"

# Phases the run manager can execute for a spec.
_PHASES = ("exploration", "validation")

# Frozen disk index (schema-v0 rebuild source for ``registry rebuild --domains runs``).
# Never written by this manager after the M4.1 registry cutover — the registry DB is now
# the sole runs store.  Keep the path constant so tests can monkeypatch it to a no-op location.
_RUNS_INDEX = SHARED_RESULTS_ROOT / "_runs_index.json"

# Registry DB path override (None → use canonical data/registry.db).  Tests monkeypatch this.
_REGISTRY_DB_PATH: Path | None = None

# Per-run stdout/stderr log files (written on completion/failure; avoids holding full text in DB).
_LOG_DIR = SHARED_RESULTS_ROOT / "logs"

# Exploration (vectorized) runs on ONE research feed per run — no dual-lane, no 2x wall-time. The
# feed is chosen per spec by research.spec.adapter.exploration_feed_for: DAILY futures-backed
# tickers (ES/NQ/GC/CL/SI/…, which have a ratio parquet) use the faithful RATIO futures feed over
# the pre-2018 train window; pure-CFD instruments (forex — no ratio) and intraday use CFD.
# Validation (realistic sim) runs on post-2018 CFD via config.data_feed (research/spec/adapter.py).


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Registry helpers ──────────────────────────────────────────────────────────


def _compute_spec_hash(payload: dict[str, Any]) -> str:
    """sha256 over canonical json.dumps(sort_keys=True, separators=(',',':'))."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _git_sha() -> str | None:
    """Short git commit hash at HEAD, or None on any failure."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def _derive_viz_parent_dir(viz_dir: str | None, phase: str) -> str:
    """Reverse-derive viz_parent_dir from viz_dir (the registry doesn't store it separately)."""
    if not viz_dir:
        return ""
    suffix = "/visualization/validation" if phase == "validation" else "/visualization"
    if viz_dir.endswith(suffix):
        return viz_dir[: -len(suffix)]
    return viz_dir


def _to_run_record(run: "SpecRun") -> "Any":
    """Convert a SpecRun to a RunRecord for registry persistence."""
    from data_platform.registry import writer as _writer  # lazy: keeps startup fast

    return _writer.RunRecord(
        run_id=run.run_id,
        kind=run.phase,  # phase (exploration|validation) maps 1:1 to kind
        status=run.status,
        spec_id=run.spec_id or None,
        spec_hash=run.spec_hash,
        spec_snapshot_json=run.spec_snapshot_json,
        research_feed_used=run.research_feed_used,
        git_sha=run.git_sha,
        num_combos=run.num_combos if run.num_combos else None,
        reports_dir=run.reports_dir or None,
        viz_dir=run.viz_dir or None,
        log_path=run.log_path,
        headline_metrics_json=run.headline_metrics_json,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error_text=run.error_text,
    )


def _extract_headline_metrics(run: "SpecRun") -> str | None:
    """Read the run's output CSVs and return a JSON-encoded headline dict, or None."""
    try:
        from frontend.api.results import _headline, _validation_headline  # type: ignore[attr-defined]

        if run.phase == "validation":
            viz = (REPO_ROOT / run.viz_dir).resolve() if run.viz_dir else None
            headline = _validation_headline(viz)
        else:
            reports = (REPO_ROOT / run.reports_dir).resolve() if run.reports_dir else None
            headline = _headline(reports)
        if headline:
            return json.dumps(headline)
    except Exception:
        pass
    return None


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class SpecRun:
    """Mutable status record for one exploration run."""

    run_id: str
    spec_id: str
    spec_name: str
    phase: str  # exploration | validation
    status: str  # queued | running | completed | failed
    created_at: str
    updated_at: str
    num_combos: int
    reports_dir: str
    viz_dir: str
    #: Per-run ``visualization_parent_dir`` (the pipeline appends ``visualization[/<phase>]``);
    #: ``viz_dir`` above is the resolved CSV folder build_results reads.
    viz_parent_dir: str
    started_at: str | None = None
    finished_at: str | None = None
    log_text: str = ""
    error_text: str | None = None
    exit_code: int | None = None
    # Registry-persisted lineage fields (captured at start time):
    spec_hash: str | None = None
    spec_snapshot_json: str | None = None
    research_feed_used: str | None = None
    git_sha: str | None = None
    log_path: str | None = None          # repo-relative path to the on-disk log file
    headline_metrics_json: str | None = None  # JSON dict extracted at completion


# ── Run manager ───────────────────────────────────────────────────────────────


class SpecRunManager:
    """Single-run-at-a-time launcher for spec exploration."""

    def __init__(self, db_path: Path | None = None) -> None:
        # Resolve DB path: explicit arg > module-level override > canonical registry path.
        # Tests inject a tmp path either via the constructor or by monkeypatching
        # ``_REGISTRY_DB_PATH`` before creating this instance.
        from data_platform.registry import db as _db  # lazy import

        self._db_path: Path = db_path or _REGISTRY_DB_PATH or _db.registry_path()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spec-run")
        self._lock = threading.Lock()
        self._runs: dict[str, SpecRun] = {}
        self._active_id: str | None = None
        self._load()

    # -- persistence -----------------------------------------------------------

    def _persist_one(self, run: SpecRun) -> None:
        """Upsert a single run to the registry DB. Best-effort: never raises. Callers may hold
        ``self._lock``.

        Assumes the matching ``specs`` row already exists (inserted via ``_persist_start_run``).
        Status-update calls (running, completed, failed) hit this path; the FK is already satisfied.
        """
        try:
            from data_platform.registry import db as _db, writer as _writer  # lazy

            conn = _db.connect(self._db_path)
            try:
                with _db.transaction(conn):
                    _writer.upsert_run(conn, _to_run_record(run))
            finally:
                conn.close()
        except Exception:
            pass

    def _persist_start_run(self, run: SpecRun, spec: Any, payload: dict[str, Any]) -> None:
        """Upsert spec + run together in a single transaction.  Used only from ``start()``.

        ``runs.spec_id`` has a FK → ``specs(id)``, so the spec row must exist before or alongside
        the run row.  Doing both in one transaction avoids the FK constraint violation that would
        occur if no ``save_spec`` call had been made before launching the run (e.g. in tests or
        when the user starts a run directly from the Spec Builder without first saving).
        """
        try:
            from data_platform.registry import db as _db, writer as _writer  # lazy
            from research.spec.serialization import spec_to_dict

            normalized = spec_to_dict(spec)
            spec_json_str = json.dumps(normalized)
            content_hash = _compute_spec_hash(normalized)
            now = _utc_now()
            spec_id = run.spec_id or run.run_id  # FK must be satisfiable
            spec_record = _writer.SpecRecord(
                id=spec_id,
                name=spec.name,
                name_slug=spec_id,
                content_hash=content_hash,
                spec_json=spec_json_str,
                # File path is cosmetic here; the canonical source is the JSON on disk.
                # spec_store.save_spec will overwrite this on the next explicit save.
                file_path=str(
                    SHARED_RESULTS_ROOT.parent.parent / "research" / "specs" / f"{spec_id}.json"
                ),
                created_at=now,
                updated_at=now,
            )
            conn = _db.connect(self._db_path)
            try:
                with _db.transaction(conn):
                    _writer.upsert_spec(conn, spec_record)
                    _writer.upsert_run(conn, _to_run_record(run))
            finally:
                conn.close()
        except Exception:
            pass

    def _delete_from_registry(self, run_id: str) -> None:
        """Delete a run row from the registry DB. Best-effort: never raises."""
        try:
            from data_platform.registry import db as _db  # lazy

            conn = _db.connect(self._db_path)
            try:
                conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def _load(self) -> None:
        """Restore runs from the registry DB so they survive ``--reload`` restarts.

        A run that was still queued/running when the process died is marked failed — its worker
        thread is gone — rather than left as a phantom 'running' that never finishes.  This
        matches the semantics of the legacy ``_runs_index.json`` loader.
        """
        try:
            from data_platform.registry import db as _db  # lazy

            conn = _db.connect(self._db_path)
            try:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY created_at"
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return

        interrupted: list[SpecRun] = []
        for row in rows:
            try:
                run_id = str(row["run_id"])
                phase = str(row["kind"] or "exploration")
                status = str(row["status"] or "failed")
                spec_id = str(row["spec_id"] or "")
                viz_dir = str(row["viz_dir"] or "")
                viz_parent_dir = _derive_viz_parent_dir(viz_dir, phase)

                # Derive spec_name from the stored snapshot JSON (preferred) or spec_id
                spec_name = spec_id
                snap = row["spec_snapshot_json"]
                if snap:
                    try:
                        spec_name = json.loads(snap).get("name") or spec_id
                    except Exception:
                        pass

                # Derive exit_code from terminal status
                if status == "completed":
                    exit_code: int | None = 0
                elif status == "failed":
                    exit_code = 1
                else:
                    exit_code = None

                # Read log from file into in-memory log_text (so the API still works)
                log_text = ""
                log_path_str = row["log_path"]
                if log_path_str:
                    try:
                        log_text = Path(log_path_str).read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        pass

                # updated_at: not stored in the registry; use finished_at > started_at > created_at
                updated_at = (
                    str(row["finished_at"])
                    if row["finished_at"]
                    else (
                        str(row["started_at"])
                        if row["started_at"]
                        else str(row["created_at"] or _utc_now())
                    )
                )

                run = SpecRun(
                    run_id=run_id,
                    spec_id=spec_id,
                    spec_name=spec_name,
                    phase=phase,
                    status=status,
                    created_at=str(row["created_at"] or _utc_now()),
                    updated_at=updated_at,
                    num_combos=int(row["num_combos"] or 0),
                    reports_dir=str(row["reports_dir"] or ""),
                    viz_dir=viz_dir,
                    viz_parent_dir=viz_parent_dir,
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                    log_text=log_text,
                    log_path=log_path_str,
                    error_text=row["error_text"],
                    exit_code=exit_code,
                    spec_hash=row["spec_hash"],
                    spec_snapshot_json=row["spec_snapshot_json"],
                    research_feed_used=row["research_feed_used"],
                    git_sha=row["git_sha"],
                    headline_metrics_json=row["headline_metrics_json"],
                )

                if run.status in {"queued", "running"}:
                    run.status = "failed"
                    run.error_text = "Interrupted by a server restart."
                    run.exit_code = 1
                    run.finished_at = run.finished_at or _utc_now()
                    interrupted.append(run)

                self._runs[run.run_id] = run
            except Exception:
                continue

        # No run survives as active: the worker threads died with the process.
        self._active_id = None

        # Persist the interrupted-run updates in a separate pass (connection already closed)
        for run in interrupted:
            self._persist_one(run)

    # -- queries ---------------------------------------------------------------

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            return None if run is None else asdict(run)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._runs:
                return None
            return asdict(max(self._runs.values(), key=lambda r: r.created_at))

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
            return [asdict(r) for r in runs]

    # -- command ---------------------------------------------------------------

    def start(self, spec_id: str, payload: dict[str, Any], phase: str = "exploration") -> dict[str, Any]:
        """Validate the payload and launch the given phase. Raises on invalid spec or busy manager."""

        if phase not in _PHASES:
            raise ValueError(f"Unknown phase '{phase}'. Expected one of {_PHASES}.")
        spec = spec_from_dict(payload)  # raises ValueError on invalid

        # Capture lineage fields at start time (cheap; errors are tolerated)
        spec_hash = _compute_spec_hash(payload)
        git_sha = _git_sha()
        spec_snapshot_json = json.dumps(payload)
        try:
            from research.spec.adapter import exploration_feed_for as _eff  # lazy heavy import
            research_feed_used: str | None = _eff(spec) if phase == "exploration" else "cfd"
        except Exception:
            research_feed_used = "cfd" if phase == "validation" else None

        with self._lock:
            if self._active() is not None:
                raise RuntimeError("A research run is already in progress.")
            run_id = str(uuid4())

            # Pre-compute the log file path so it's in the registry record from the start.
            log_file = _LOG_DIR / f"{run_id}.log"
            log_path = repo_relative(log_file)

            # Per-run isolation: reports + visualization CSVs land under a run-id subfolder so
            # each run reads back its own outputs and can be deleted independently.
            if phase == "exploration":
                viz_parent = _VIZ_ROOT / run_id
                run = SpecRun(
                    run_id=run_id,
                    spec_id=spec_id,
                    spec_name=spec.name,
                    phase=phase,
                    status="queued",
                    created_at=_utc_now(),
                    updated_at=_utc_now(),
                    num_combos=spec.signal.num_combos,
                    reports_dir=f"{_REPORTS_PREFIX}/{spec.name}/{run_id}",
                    viz_dir=repo_relative(viz_parent / _VIZ_SUBDIR),
                    viz_parent_dir=repo_relative(viz_parent),
                    spec_hash=spec_hash,
                    spec_snapshot_json=spec_snapshot_json,
                    research_feed_used=research_feed_used,
                    git_sha=git_sha,
                    log_path=log_path,
                )
            else:
                # Validation: single-lane, unchanged. CSVs land under <viz_parent>/visualization/
                # validation; reports under the per-run reports folder.
                viz_parent = _VALIDATION_VIZ_ROOT / run_id
                viz_csv_dir = viz_parent / _VIZ_SUBDIR / "validation"
                run = SpecRun(
                    run_id=run_id,
                    spec_id=spec_id,
                    spec_name=spec.name,
                    phase=phase,
                    status="queued",
                    created_at=_utc_now(),
                    updated_at=_utc_now(),
                    num_combos=spec.signal.num_combos,
                    reports_dir=f"{_REPORTS_PREFIX}/{spec.name}/{run_id}",
                    viz_dir=repo_relative(viz_csv_dir),
                    viz_parent_dir=repo_relative(viz_parent),
                    spec_hash=spec_hash,
                    spec_snapshot_json=spec_snapshot_json,
                    research_feed_used=research_feed_used,
                    git_sha=git_sha,
                    log_path=log_path,
                )
            self._runs[run.run_id] = run
            self._active_id = run.run_id
            # Use _persist_start_run (not _persist_one) so the spec row is upserted in the same
            # transaction as the run row, satisfying the runs.spec_id FK → specs(id) constraint.
            self._persist_start_run(run, spec, payload)
            self._executor.submit(self._run, run.run_id, payload, phase)
            return asdict(run)

    def delete(self, run_id: str) -> None:
        """Drop a finished run from the registry and remove its per-run output folders.

        Raises ``KeyError`` if the run is unknown (→ 404) and ``RuntimeError`` if it is still
        in progress (→ 409); an in-progress run is writing to those folders, so deleting it
        would corrupt the live output.
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run not found: {run_id}")
            if run.status in {"queued", "running"} or run_id == self._active_id:
                raise RuntimeError("Cannot delete a run in progress.")
            self._runs.pop(run_id)
            # The per-run output ROOTS — removing them cleans up the run's whole output tree.
            targets = (run.reports_dir, run.viz_parent_dir)
        # Registry delete and filesystem cleanup outside the lock.
        self._delete_from_registry(run_id)
        for rel in targets:
            shutil.rmtree((REPO_ROOT / rel).resolve(), ignore_errors=True)

    # -- internals -------------------------------------------------------------

    def _update(self, run_id: str, **changes: Any) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            for key, value in changes.items():
                setattr(run, key, value)
            run.updated_at = _utc_now()
            self._persist_one(run)

    def _active(self) -> SpecRun | None:
        if self._active_id is None:
            return None
        run = self._runs.get(self._active_id)
        if run is None or run.status in {"completed", "failed"}:
            self._active_id = None
            return None
        return run

    def _run(self, run_id: str, payload: dict[str, Any], phase: str) -> None:
        buffer = io.StringIO()
        with self._lock:
            run = self._runs[run_id]
            viz_parent_dir = run.viz_parent_dir
            reports_dir = run.reports_dir
            log_path_str = run.log_path
        # Resolve log file to an absolute path for writing
        log_file: Path | None = (REPO_ROOT / log_path_str).resolve() if log_path_str else None

        self._update(run_id, status="running", started_at=_utc_now())
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                self._execute(payload, phase, viz_parent_dir, reports_dir)

            log_content = buffer.getvalue()
            _write_log(log_file, log_content)

            # Extract headline metrics from the just-written output CSVs
            with self._lock:
                current_run = self._runs.get(run_id)
            headline_json = _extract_headline_metrics(current_run) if current_run else None

            self._update(
                run_id,
                status="completed",
                finished_at=_utc_now(),
                exit_code=0,
                log_text=log_content,
                headline_metrics_json=headline_json,
            )
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the UI
            log_content = buffer.getvalue() + "\n" + traceback.format_exc()
            _write_log(log_file, log_content)
            self._update(
                run_id,
                status="failed",
                finished_at=_utc_now(),
                exit_code=1,
                log_text=log_content,
                error_text=str(exc),
            )
        finally:
            with self._lock:
                if self._active_id == run_id:
                    self._active_id = None

    @staticmethod
    def _execute(
        payload: dict[str, Any],
        phase: str,
        viz_parent_dir: str,
        reports_dir: str,
    ) -> None:
        # Deferred heavy imports (nautilus-free; the adapter is lazy about the engine).
        from dataclasses import replace

        from lib.core.research_feed import set_research_feed
        from research.spec import apply_vol_scaling, to_feature_config
        from research.spec.adapter import exploration_feed_for

        spec = spec_from_dict(payload)
        base_config = to_feature_config(spec)
        try:
            apply_vol_scaling(spec)
        except NotImplementedError as exc:
            print(f"[vol_scaling] {exc} Proceeding with the pipeline default blend.")

        if phase == "validation":
            # Realistic sim on post-2018 CFD: config.data_feed = "cfd" (set by the adapter); the
            # validation pipeline re-sets the process feed from it. One config, one viz folder.
            set_research_feed(
                base_config.data_feed,
                futures_tickers=getattr(base_config, "exploration_futures_index_tickers", None),
            )
            config = replace(
                base_config,
                visualization_parent_dir=(REPO_ROOT / viz_parent_dir).resolve(),
            )
            SpecRunManager._run_validation(config)
            return

        # Exploration: a SINGLE vectorized run on ONE research feed (no dual-lane, no 2x). The feed
        # is per asset class: DAILY futures-backed tickers → faithful ratio futures; forex/intraday
        # → CFD. Both signal and target come from that one feed.
        from research.feature.exploration import execute_exploration_phase

        exploration_feed = exploration_feed_for(spec)
        set_research_feed(exploration_feed)
        config = replace(
            base_config,
            visualization_parent_dir=(REPO_ROOT / viz_parent_dir).resolve(),
        )
        run_reports_dir = (REPO_ROOT / reports_dir).resolve()
        result = execute_exploration_phase(config, run_reports_dir)
        print(
            f"Exploration complete ({exploration_feed}): {len(result.eda_results)} "
            f"combo(s) -> {run_reports_dir}"
        )

    @staticmethod
    def _run_validation(config: Any) -> None:
        """Validation walkforward + the portfolio-addition gate (the vault-save gate).

        The gate scores the candidate against the portfolio baseline, so we enrich the
        spec-built config with the portfolio settings from the canonical feature config
        (portfolio_source / portfolio_addition_gate / portfolio_inclusion) — the feature↔
        portfolio dependency. The adapter stays pure; the composition happens here.
        """

        from research.feature.ui.workspace_manifest import write_validation_manifest
        from research.feature.validation import run_validation_pipeline

        config = SpecRunManager._with_portfolio_gate(config)
        report = run_validation_pipeline(config, output_dir=None)
        # The manifest lets the vault-save eligibility check recognise these validation artifacts
        # (incl. the portfolio-addition gate report) as current for this spec.
        write_validation_manifest(config)
        print(f"Validation complete: {len(report.folds_df)} fold(s) -> {config.output_root}")

    @staticmethod
    def _with_portfolio_gate(config: Any) -> Any:
        from dataclasses import replace

        from research.feature.config import (
            ephemeral_weight_hierarchy_group_for_tickers,
            load_config as load_feature_config,
        )

        canonical = load_feature_config()
        tickers = tuple(config.tickers)
        inclusion = canonical.portfolio_inclusion
        if inclusion is not None:
            inclusion = replace(
                inclusion,
                candidate_tickers=tickers,
                ephemeral_weight_hierarchy_group=ephemeral_weight_hierarchy_group_for_tickers(
                    tickers,
                    configured_group=inclusion.ephemeral_weight_hierarchy_group,
                ),
            )
        # Force the gate serial (n_jobs=1): the runtime-set research feed + EWSD blend are
        # process-global and do NOT propagate to loky worker processes, so a parallel gate fits
        # baseline ensembles (e.g. calendar_ensemble) under the wrong feed and fails. Serial is
        # correct here and runs are already one-at-a-time. (Verified: serial gate passes.)
        gate = replace(canonical.portfolio_addition_gate, n_jobs=1)
        return replace(
            config,
            portfolio_source=canonical.portfolio_source,
            portfolio_addition_gate=gate,
            portfolio_inclusion=inclusion,
        )


# ── Module-level helpers ──────────────────────────────────────────────────────


def _write_log(log_file: Path | None, content: str) -> None:
    """Write log content to file; silently ignore I/O errors."""
    if log_file is None:
        return
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(content, encoding="utf-8")
    except OSError:
        pass


RUN_MANAGER = SpecRunManager()
