"""Background job manager for portfolio research UI."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import io
import json
import threading
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

from research.portfolio.config import PortfolioResearchConfig
from research.portfolio.ui.planner import build_phase_plan
from research.portfolio.ui.runner import execute_phase_request

# Registry DB path override (None → canonical data/registry.db).  Tests monkeypatch this.
_REGISTRY_DB_PATH: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PortfolioWorkspaceJob:
    job_id: str
    phase: str
    status: str
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    plan_title: str = ""
    output_path: str = ""
    log_text: str = ""
    error_text: str | None = None
    exit_code: int | None = None


def _to_run_record(job: PortfolioWorkspaceJob, spec_snapshot_json: str | None) -> Any:
    """Convert a PortfolioWorkspaceJob to a RunRecord for registry persistence."""
    from data_platform.registry import writer as _writer  # lazy

    return _writer.RunRecord(
        run_id=job.job_id,
        kind="portfolio",
        status=job.status,
        spec_id=None,
        spec_snapshot_json=spec_snapshot_json,
        reports_dir=job.output_path or None,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error_text=job.error_text,
    )


@dataclass
class PortfolioWorkspaceJobManager:
    max_workers: int = 1
    db_path: Path | None = field(default=None)
    _executor: ThreadPoolExecutor = field(init=False)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _jobs: dict[str, PortfolioWorkspaceJob] = field(init=False, default_factory=dict)
    _job_snapshots: dict[str, str | None] = field(init=False, default_factory=dict)
    _active_job_id: str | None = field(init=False, default=None)
    _db_path: Path = field(init=False)

    def __post_init__(self) -> None:
        from data_platform.registry import db as _db  # lazy

        self._db_path = self.db_path or _REGISTRY_DB_PATH or _db.registry_path()
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="portfolio-research-ui",
        )
        self._load()

    # -- persistence -----------------------------------------------------------

    def _persist_one(self, job: PortfolioWorkspaceJob) -> None:
        """Upsert a single job to the registry DB. Best-effort: never raises."""
        try:
            from data_platform.registry import db as _db, writer as _writer  # lazy

            snap = self._job_snapshots.get(job.job_id)
            conn = _db.connect(self._db_path)
            try:
                with _db.transaction(conn):
                    _writer.upsert_run(conn, _to_run_record(job, snap))
            finally:
                conn.close()
        except Exception:
            pass

    def _load(self) -> None:
        """Restore jobs from the registry DB; mark interrupted jobs as failed."""
        try:
            from data_platform.registry import db as _db  # lazy

            conn = _db.connect(self._db_path)
            try:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE kind = 'portfolio' ORDER BY created_at"
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return

        interrupted: list[PortfolioWorkspaceJob] = []
        for row in rows:
            try:
                job_id = str(row["run_id"])
                status = str(row["status"] or "failed")
                updated_at = (
                    str(row["finished_at"])
                    if row["finished_at"]
                    else (
                        str(row["started_at"])
                        if row["started_at"]
                        else str(row["created_at"] or _utc_now())
                    )
                )
                job = PortfolioWorkspaceJob(
                    job_id=job_id,
                    phase="portfolio",
                    status=status,
                    created_at=str(row["created_at"] or _utc_now()),
                    updated_at=updated_at,
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                    output_path=str(row["reports_dir"] or ""),
                    error_text=row["error_text"],
                    exit_code=0 if status == "completed" else (1 if status == "failed" else None),
                )
                self._job_snapshots[job_id] = row["spec_snapshot_json"]

                if job.status in {"queued", "running"}:
                    job.status = "failed"
                    job.error_text = "Interrupted by a server restart."
                    job.exit_code = 1
                    job.finished_at = job.finished_at or _utc_now()
                    interrupted.append(job)

                self._jobs[job_id] = job
            except Exception:
                continue

        self._active_job_id = None
        for job in interrupted:
            self._persist_one(job)

    # -- queries ---------------------------------------------------------------

    def start_job(self, config: PortfolioResearchConfig, request) -> dict[str, Any]:
        with self._lock:
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active is not None and active.status in {"queued", "running"}:
                    raise RuntimeError("A portfolio research run is already in progress.")
            plan = build_phase_plan(config, request)
            job = PortfolioWorkspaceJob(
                job_id=str(uuid4()),
                phase=request.phase.value,
                status="queued",
                created_at=_utc_now(),
                updated_at=_utc_now(),
                plan_title=plan.title,
                output_path=plan.output_path.as_posix(),
            )
            # Snapshot the config for the registry record (best-effort serialisation).
            try:
                from dataclasses import asdict as _asdict
                snap = json.dumps(_asdict(config))
            except Exception:
                snap = None
            self._job_snapshots[job.job_id] = snap
            self._jobs[job.job_id] = job
            self._active_job_id = job.job_id
            self._persist_one(job)
            self._executor.submit(self._run_job, job.job_id, config, request)
            return asdict(job)

    def latest_job(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._jobs:
                return None
            job = max(self._jobs.values(), key=lambda item: item.created_at)
            return asdict(job)

    def job_snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else asdict(job)

    def _run_job(self, job_id: str, config: PortfolioResearchConfig, request) -> None:
        buffer = io.StringIO()
        self._update_job(job_id, status="running", started_at=_utc_now())
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                exit_code = execute_phase_request(config, request)
            self._update_job(
                job_id,
                status="completed",
                finished_at=_utc_now(),
                exit_code=exit_code,
                log_text=buffer.getvalue(),
            )
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self._update_job(
                job_id,
                status="failed",
                finished_at=_utc_now(),
                exit_code=1,
                log_text=buffer.getvalue(),
                error_text="".join(traceback.format_exception(exc)),
            )
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _update_job(self, job_id: str, **kwargs: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in kwargs.items():
                setattr(job, key, value)
            job.updated_at = _utc_now()
        self._persist_one(job)
