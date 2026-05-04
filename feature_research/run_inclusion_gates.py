"""CLI for portfolio inclusion: peer forecast correlations and with/without-candidate tearsheets.

Loads ``feature_research.config.load_config()`` and the baseline portfolio from
``portfolio_research.config.load_config()``.

Usage (repo root, venv Python)::

    .\\.venv\\Scripts\\python.exe -m feature_research.run_inclusion_gates

Default: ``portfolio_inclusion.candidate_mode = "eval_bias_spec"`` — the candidate is the frozen
``ResearchConfig.eval_bias_spec`` (same as evaluation). Use ``--candidate-mode vault_path`` and
``--candidate-path`` (or set ``candidate_repo_relative_path`` / ``vault_save``) to test a saved
ensemble instead. By default, six HTML portfolio tearsheets are written; use ``--no-emit-tearsheets``
to skip.

Baseline + candidate portfolio phases always use WeightLayer ``ledoit_wolf_min_corr`` (Ledoit–Wolf
shrinkage, inverse column-sum weights); ``fdm_max`` is taken from ``portfolio_research`` config
(see :func:`feature_research.inclusion_gates.run_portfolio_inclusion`).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path


def _prepend_repo_root_to_syspath() -> None:
    start = Path(__file__).resolve()
    for parent in (start.parent, *start.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise RuntimeError(
        "Could not locate repository root (no pyproject.toml or .git above this file)."
    )


_prepend_repo_root_to_syspath()

from utils.repo_bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from feature_research.config import (
    PortfolioInclusionConfig,
    ResearchConfig,
    inferred_inclusion_candidate_path,
    load_config as load_feature_config,
)
from feature_research.inclusion_gates import (
    infer_candidate_key,
    materialize_inclusion_candidate_from_eval_bias_spec,
    run_portfolio_inclusion,
    write_inclusion_artifacts,
)
from portfolio_research.config import load_config as load_portfolio_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Portfolio inclusion: validation forecast correlation vs peers and portfolio comparison "
            "tearsheets (baseline vs with candidate). Uses feature_research config for output paths "
            "and portfolio_research config for the baseline portfolio."
        )
    )
    p.add_argument(
        "--candidate-path",
        type=str,
        default=None,
        help=(
            "Repo-relative path to candidate ensemble (e.g. vault/... or vault_personal/...). "
            "Default: portfolio_inclusion.candidate_repo_relative_path, else vault_save-derived path."
        ),
    )
    p.add_argument(
        "--candidate-key",
        type=str,
        default=None,
        help=(
            "Key for ensemble_dirs. Default: portfolio_inclusion.candidate_key, else leaf directory name."
        ),
    )
    p.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip run_portfolio_research_cache_preflight (overrides portfolio_inclusion.preflight).",
    )
    p.add_argument(
        "--candidate-mode",
        choices=("vault_path", "eval_bias_spec"),
        default=None,
        help=(
            "eval_bias_spec: materialize candidate from ResearchConfig.eval_bias_spec (default). "
            "vault_path: use candidate_repo_relative_path / inferred vault dir. "
            "Default: portfolio_inclusion.candidate_mode."
        ),
    )
    p.add_argument(
        "--emit-tearsheets",
        action="store_true",
        help=(
            "Force on: six HTML portfolio tearsheets (overrides portfolio_inclusion.emit_tearsheets=False)."
        ),
    )
    p.add_argument(
        "--no-emit-tearsheets",
        action="store_true",
        help=(
            "Skip writing HTML tearsheets (overrides portfolio_inclusion.emit_tearsheets=True)."
        ),
    )
    return p.parse_args(argv)


def _resolve_candidate(
    args: argparse.Namespace,
    fr: ResearchConfig,
    th: PortfolioInclusionConfig,
) -> tuple[str, str]:
    """Return (candidate_repo_relative_path, candidate_key) from CLI with config fallback."""
    path = (
        (args.candidate_path or "").strip()
        or (th.candidate_repo_relative_path or "").strip()
        or (inferred_inclusion_candidate_path(fr) or "").strip()
    )
    if not path:
        raise SystemExit(
            "Set portfolio_inclusion.candidate_repo_relative_path to an existing ensemble directory, "
            "or fix vault_save so inferred_inclusion_candidate_path finds a real directory "
            "(ensemble_name must match an on-disk folder under your vault root), "
            "or pass --candidate-path with a repo-relative path such as vault/... or vault_personal/..."
        )
    key_raw = args.candidate_key if args.candidate_key is not None else th.candidate_key
    ck = (key_raw.strip() if isinstance(key_raw, str) and key_raw.strip() else infer_candidate_key(path))
    return path, ck


def main(argv: list[str] | None = None) -> list[Path]:
    args = _parse_args(argv)
    fr = load_feature_config()
    th = fr.portfolio_inclusion
    mode = args.candidate_mode or th.candidate_mode
    pr = load_portfolio_config()

    if mode == "eval_bias_spec":
        candidate_path, tmp_root = materialize_inclusion_candidate_from_eval_bias_spec(
            fr,
            pr,
            ephemeral_ensemble_name=th.ephemeral_ensemble_name,
            weight_hierarchy_group=th.ephemeral_weight_hierarchy_group,
        )
        key_raw = args.candidate_key if args.candidate_key is not None else th.candidate_key
        ck = (
            key_raw.strip()
            if isinstance(key_raw, str) and key_raw.strip()
            else infer_candidate_key(candidate_path)
        )
    else:
        candidate_path, ck = _resolve_candidate(args, fr, th)
        tmp_root = None

    run_preflight = bool(th.preflight) and not args.no_preflight

    if args.no_emit_tearsheets:
        emit_ts = False
    elif args.emit_tearsheets:
        emit_ts = True
    else:
        emit_ts = bool(th.emit_tearsheets)
    th_run = replace(th, emit_tearsheets=emit_ts)
    tearsheets_dir: Path | None = None
    if emit_ts:
        tearsheets_dir = fr.output_root / th.output_subdir / "tearsheets" / ck

    try:
        if mode == "eval_bias_spec":
            result = run_portfolio_inclusion(
                pr,
                candidate_ensemble_dir=candidate_path,
                candidate_key=ck,
                inclusion_config=th_run,
                run_preflight=run_preflight,
                tearsheets_output_dir=tearsheets_dir,
            )
        else:
            result = run_portfolio_inclusion(
                pr,
                candidate_repo_relative_path=candidate_path,
                candidate_key=ck,
                inclusion_config=th_run,
                run_preflight=run_preflight,
                tearsheets_output_dir=tearsheets_dir,
            )
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)

    out_dir = fr.output_root / th.output_subdir
    written = write_inclusion_artifacts(out_dir, ck, result)

    print(result.message)
    for p in written:
        print(f"Wrote {p}")
    if result.tearsheet_paths:
        print("Tearsheets:")
        for p in result.tearsheet_paths:
            print(f"  {p}")
    return written


if __name__ == "__main__":
    main()
