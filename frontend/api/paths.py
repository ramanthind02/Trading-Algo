"""Filesystem locations the API reads/writes, plus safe artifact-path resolution."""

from __future__ import annotations

from pathlib import Path

# frontend/api/paths.py -> parents[2] is the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Round-trippable StrategySpec JSON files (the shared agent/UI contract).
SPECS_DIR: Path = REPO_ROOT / "research" / "specs"

#: Per-spec exploration outputs (EDA per-combo, robustness, permutation, PNGs).
SHARED_RESULTS_ROOT: Path = REPO_ROOT / "feature_research" / "shared_results"

#: In-sample visualization CSVs/PNGs (param sensitivity, equity curve, perturbation).
#: The spec-driven run manager isolates each exploration run under a per-run subfolder
#: (``<root>/<run_id>/visualization``); the canonical run_is.py / ui/runner.py path still
#: writes the shared ``<root>/visualization`` folder.
IN_SAMPLE_RESULTS_ROOT: Path = REPO_ROOT / "research" / "feature" / "in_sample" / "results"

#: Portfolio research outputs (train/val/test tearsheets, holdout returns, prop-firm reports).
PORTFOLIO_RESULTS_ROOT: Path = REPO_ROOT / "research" / "portfolio" / "results"

#: Roots an artifact path is allowed to resolve under (prevents path traversal).
ARTIFACT_ROOTS: tuple[Path, ...] = (
    SHARED_RESULTS_ROOT,
    IN_SAMPLE_RESULTS_ROOT,
    PORTFOLIO_RESULTS_ROOT,
)


def repo_relative(path: Path) -> str:
    """POSIX path relative to the repo root (the form artifact references use).

    Falls back to an absolute POSIX path for locations outside the repo (e.g. a temp dir in
    tests); production roots are always under the repo.
    """

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_artifact_path(relative: str) -> Path:
    """Resolve a repo-relative artifact path, rejecting anything outside the allowed roots.

    Raises ``ValueError`` for traversal / out-of-root paths and ``FileNotFoundError`` if the
    file does not exist.
    """

    candidate = (REPO_ROOT / relative).resolve()
    for root in ARTIFACT_ROOTS:
        root_resolved = root.resolve()
        if candidate == root_resolved or root_resolved in candidate.parents:
            if not candidate.exists():
                raise FileNotFoundError(f"Artifact not found: {relative}")
            return candidate
    raise ValueError(f"Artifact path is outside the allowed results roots: {relative}")
