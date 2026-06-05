"""Background job manager for portfolio research UI."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import io
import threading
import traceback
from typing import Any
from uuid import uuid4

from research.portfolio.config import PortfolioResearchConfig
from research.portfolio.ui.planner import build_phase_plan
from research.portfolio.ui.runner import execute_phase_request


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


@dataclass
class PortfolioWorkspaceJobManager:
    max_workers: int = 1
    _executor: ThreadPoolExecutor = field(init=False)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _jobs: dict[str, PortfolioWorkspaceJob] = field(init=False, default_factory=dict)
    _active_job_id: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="portfolio-research-ui",
        )

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
            self._jobs[job.job_id] = job
            self._active_job_id = job.job_id
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
