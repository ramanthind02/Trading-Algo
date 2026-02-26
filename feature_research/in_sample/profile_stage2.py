"""Profile Stage 2 (candle-shuffle) permutation to find bottlenecks.

Runs the same permutation pipeline as run_is.py with a reduced number of
reps under cProfile, then prints a cumulative-time summary and optionally
saves a .prof file for snakeviz.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/in_sample/profile_stage2.py

    # Save profile for snakeviz (interactive flame graph)
    python feature_research/in_sample/profile_stage2.py --out profile.stats

    # View with snakeviz (install: pip install snakeviz)
    snakeviz profile.stats

Options
-------
    --reps N       Number of Stage 2 reps to run (default: 3). More reps = more
                   representative but slower.
    --jobs-reps N  Stage 2 rep-level parallel workers (default: 1 for profiling).
                   Use 4+ to benchmark multiprocessing speedup.
    --benchmark    Run twice (jobs-reps=1 then jobs-reps=4) and print wall time
                   to compare; no cProfile output.
    --out PATH     Write cProfile stats to PATH for snakeviz.
    --top N        Show top N functions by cumulative time (default: 60).

Interpreting results
--------------------
Look for: (1) feature_extractor lambdas / _normalize_ticker_str (millions of
calls → consider vectorizing or caching); (2) continuous_binning fit/predict
(inherent per-combo cost); (3) pandas map_array / parquet reads (I/O or
per-row Python). Use --out and snakeviz to drill into call trees.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from dataclasses import replace
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    search_root = start if start.is_dir() else start.parent
    for parent in (search_root, *search_root.parents):
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return None


_repo_root = _find_repo_root(Path(__file__).resolve())
if _repo_root is not None and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from feature_research.config import PermutationSuiteConfig
from feature_research.in_sample.config import load_config
from feature_research.pipeline import run_permutation_pipeline


def _cython_nodes_status() -> str:
    """Return one-line status of Cython bias-node optimizations (for profiling/benchmark)."""
    try:
        from utils.compute.fast_nodes import CYTHON_NODES_AVAILABLE  # noqa: PLC0415
        return "available" if CYTHON_NODES_AVAILABLE else "not built (build: python utils/compute/cython/setup_cython.py build_ext --inplace)"
    except Exception:
        return "unknown"


def _run_with_reps(
    config_reps: int,
    output_dir: Path,
    n_jobs_stage2_reps: int = 1,
) -> None:
    """Run permutation pipeline with config using config_reps for Stage 2."""
    config = load_config()
    if not config.in_sample_permutation.enabled:
        raise ValueError(
            "Permutation suite is disabled. Set in_sample_permutation.enabled=True in config."
        )
    perm_suite = replace(
        config.in_sample_permutation,
        nreps=config_reps,
        n_jobs_stage2_reps=n_jobs_stage2_reps,
    )
    config = replace(config, in_sample_permutation=perm_suite)
    run_permutation_pipeline(config, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile Stage 2 permutation pipeline (cProfile)."
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of Stage 2 reps to run (default: 3)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write cProfile stats to this path for snakeviz",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=60,
        help="Show top N functions by cumulative time (default: 60)",
    )
    parser.add_argument(
        "--jobs-reps",
        type=int,
        default=1,
        help="Stage 2 rep-level parallel workers (default: 1 for profiling)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run jobs-reps=1 then jobs-reps=4 and print wall time comparison",
    )
    args = parser.parse_args()

    config = load_config()
    output_dir = config.reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark:
        nreps = args.reps
        print(f"Benchmark: Stage 2 with {nreps} reps (wall time comparison).")
        print(f"Cython bias nodes: {_cython_nodes_status()}")
        print(f"Output dir: {output_dir.resolve()}\n")
        for n_jobs in (1, 4):
            t0 = time.perf_counter()
            try:
                _run_with_reps(nreps, output_dir, n_jobs_stage2_reps=n_jobs)
            except Exception as e:
                print(f"  n_jobs_stage2_reps={n_jobs} (Stage 3 or later error): {e}")
            elapsed = time.perf_counter() - t0
            print(f"  n_jobs_stage2_reps={n_jobs}: {elapsed:.1f}s wall\n")
        return 0

    print(f"Profiling Stage 2 with {args.reps} reps (n_jobs_stage2_reps={args.jobs_reps}).")
    print(f"Cython bias nodes: {_cython_nodes_status()}")
    print(f"Output dir: {output_dir.resolve()}\n")

    profile = cProfile.Profile()
    profile.enable()
    try:
        _run_with_reps(args.reps, output_dir, n_jobs_stage2_reps=args.jobs_reps)
    except Exception as e:
        print(f"\nPipeline error (Stage 3 may fail with few reps): {e}")
        print("Profile below still reflects Stage 1 + Stage 2.\n")
    finally:
        profile.disable()

    if args.out is not None:
        profile.dump_stats(str(args.out))
        print(f"\nProfile saved to: {args.out.resolve()}")
        print("View with: snakeviz", args.out)

    stats = pstats.Stats(profile)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    print("\n" + "=" * 80)
    print("Top functions by cumulative time (Stage 2 hot path)")
    print("=" * 80)
    stats.print_stats(args.top)

    print("\n" + "=" * 80)
    print("Top functions by total time (self + callees)")
    print("=" * 80)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(args.top)

    return 0


if __name__ == "__main__":
    sys.exit(main())
