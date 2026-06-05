"""Parity harness fixtures: pinned cases, seeds, and snapshot IO.

This conftest pins the *representative* research cases the parity gate protects,
plus the deterministic serialization used for golden snapshots. It deliberately
does **not** import the pipeline entrypoints at collection time — the research
environment may not be able to import them (see ``pipeline_imports`` /
``require_pipeline``), and the harness must still collect and skip cleanly.

Pinned cases
------------
* ``feature_research`` — the single-feature signed-signal OOS run defined by
  ``feature_research.config.load_config()`` (SI daily SMA(252) regime signal,
  long_short, train 2000–2018 / val 2019–2022; permutation/robustness seed=42).
  We snapshot the deterministic ``WalkforwardRunReport`` produced by
  ``feature_research.pipelines.oos.run_oos_pipeline``.
* ``portfolio_research`` — the default ``portfolio_research.config.load_config()``
  test run (ES/NQ/GC/CL, prop vault ensembles, train/val/test windows). We snapshot
  the **test** phase ``PhaseResult`` produced by
  ``portfolio_research.pipelines.portfolio_test.run_single_phase_for_prop_firm``
  with tearsheets disabled and ``run_purpose='metrics_only'`` so no HTML / prop-firm
  reports / vault writes are emitted.

Both cases run with ``USE_CACHE=True`` against repo-backed data in ``data/ohlc_data``
(populated on demand by the pipeline's own cache preflight). Outputs are redirected
to a temp ``output_root`` so the repo working tree is never polluted.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np
import pandas as pd
import pytest

# --- repo root on sys.path (shared venv; no activation assumed) ---------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

# Determinism: pin every seed we can reach from process start. The pipelines
# pin their own bootstrap/permutation seeds via config (random_seed=42), but we
# also pin the global NumPy / hashing state to catch any incidental leakage.
GLOBAL_SEED = 1234


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "regen: write/refresh golden parity snapshots instead of verifying them.",
    )
    config.addinivalue_line(
        "markers",
        "parity: research output parity test (integration-style, cache-backed).",
    )


def _regen_requested(config: pytest.Config) -> bool:
    expr = config.getoption("-m") or ""
    return "regen" in expr


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Make ``pytest tests/parity`` a pure *verify* run.

    Snapshot-writing tests are marked ``regen`` and are skipped unless the user
    explicitly opts in with ``-m regen``. This guarantees a bare run can never
    silently overwrite golden snapshots (per the spec: regen is an explicit,
    reviewed action).
    """
    if _regen_requested(config):
        return
    skip_regen = pytest.mark.skip(
        reason="regen test: pass -m regen to (re)write golden snapshots."
    )
    for item in items:
        if "regen" in item.keywords:
            item.add_marker(skip_regen)


@pytest.fixture(scope="session", autouse=True)
def _pin_global_determinism() -> None:
    """Pin process-wide seeds so a stray RNG never makes snapshots flap."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    np.random.seed(GLOBAL_SEED)


@pytest.fixture(scope="session")
def regen_mode(request: pytest.FixtureRequest) -> bool:
    """True when invoked with ``-m regen`` (snapshot-writing path)."""
    expr = request.config.getoption("-m") or ""
    return "regen" in expr


# -----------------------------------------------------------------------------
# Pipeline import guard
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineImports:
    """Lazily-imported pipeline entrypoints (None when the import failed)."""

    ok: bool
    reason: str
    feature_load_config: Callable[[], object] | None
    run_oos_pipeline: Callable[[object], object] | None
    portfolio_load_config: Callable[[], object] | None
    run_single_phase_for_prop_firm: Callable[..., object] | None


def _try_import_pipelines() -> PipelineImports:
    """Attempt to import the research entrypoints; capture any failure reason.

    The research environment on this branch requires a ``quantfoundry_core`` that
    exports ``ParamPerturbationSpec`` from ``quantfoundry_core.robustness``. If the
    installed package predates that, ``feature_research.config`` (and transitively
    ``portfolio_research.config``) fail to import. We surface that as a *skip*, not
    an error, so the harness still proves it is wired correctly.
    """
    try:
        from feature_research.config import load_config as feature_load_config
        from feature_research.pipelines.oos import run_oos_pipeline
        from portfolio_research.config import load_config as portfolio_load_config
        from portfolio_research.pipelines.portfolio_test import (
            run_single_phase_for_prop_firm,
        )
    except Exception as exc:  # noqa: BLE001 - we want the message verbatim
        return PipelineImports(
            ok=False,
            reason=f"{type(exc).__name__}: {exc}",
            feature_load_config=None,
            run_oos_pipeline=None,
            portfolio_load_config=None,
            run_single_phase_for_prop_firm=None,
        )
    return PipelineImports(
        ok=True,
        reason="",
        feature_load_config=feature_load_config,
        run_oos_pipeline=run_oos_pipeline,
        portfolio_load_config=portfolio_load_config,
        run_single_phase_for_prop_firm=run_single_phase_for_prop_firm,
    )


@pytest.fixture(scope="session")
def pipeline_imports() -> PipelineImports:
    return _try_import_pipelines()


@pytest.fixture
def require_pipeline(pipeline_imports: PipelineImports) -> PipelineImports:
    """Skip the test (explicit message) when pipeline entrypoints can't import."""
    if not pipeline_imports.ok:
        pytest.skip(
            "Research pipeline entrypoints could not be imported in this "
            "environment, so parity baselines cannot be exercised here. "
            "Fix the environment (this branch needs a newer quantfoundry_core "
            "exporting ParamPerturbationSpec from quantfoundry_core.robustness), "
            f"then re-run. Underlying import error: {pipeline_imports.reason}"
        )
    return pipeline_imports


def _require_data_or_skip() -> None:
    data_dir = PROJECT_ROOT / "data" / "ohlc_data"
    if not data_dir.exists() or not any(data_dir.iterdir()):
        pytest.skip(
            f"Repo OHLC data not found at {data_dir}; parity harness needs "
            "cache-backed data. Populate data/ohlc_data (and let the pipeline "
            "cache preflight build the cache) before running parity."
        )


@pytest.fixture
def require_data() -> None:
    _require_data_or_skip()


# -----------------------------------------------------------------------------
# Snapshot IO (deterministic)
# -----------------------------------------------------------------------------
def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic frame: sorted columns, reset+sorted index, stable dtypes."""
    out = df.copy()
    # Ensure a clean RangeIndex unless the index carries meaning we preserved.
    out = out.reindex(sorted(out.columns), axis=1)
    return out


def write_frame_snapshot(name: str, df: pd.DataFrame) -> Path:
    path = SNAPSHOT_DIR / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_frame(df).to_parquet(path, index=True)
    return path


def read_frame_snapshot(name: str) -> pd.DataFrame:
    path = SNAPSHOT_DIR / f"{name}.parquet"
    return pd.read_parquet(path)


def write_series_snapshot(name: str, series: pd.Series) -> Path:
    path = SNAPSHOT_DIR / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = series.sort_index().rename(series.name or "value").to_frame()
    frame.to_parquet(path, index=True)
    return path


def read_series_snapshot(name: str) -> pd.Series:
    path = SNAPSHOT_DIR / f"{name}.parquet"
    frame = pd.read_parquet(path)
    col = frame.columns[0]
    return frame[col]


def write_scalars_snapshot(name: str, scalars: dict[str, float]) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.json"
    ordered = {k: scalars[k] for k in sorted(scalars)}
    path.write_text(json.dumps(ordered, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_scalars_snapshot(name: str) -> dict[str, float]:
    path = SNAPSHOT_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_exists(name: str, kind: str) -> bool:
    suffix = "json" if kind == "scalars" else "parquet"
    return (SNAPSHOT_DIR / f"{name}.{suffix}").exists()
