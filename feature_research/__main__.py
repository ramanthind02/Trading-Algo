from __future__ import annotations

import sys
from collections.abc import Callable, Sequence


def _call_main_with_argv(main_fn: Callable[[], int | None], argv: Sequence[str]) -> int:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *argv]
        result = main_fn()
    finally:
        sys.argv = original_argv
    return 0 if result is None else int(result)


def _run_in_sample(_argv: Sequence[str]) -> int:
    from feature_research.in_sample.run_is import main as run_main

    return _call_main_with_argv(run_main, [])


def _run_validation(_argv: Sequence[str]) -> int:
    from feature_research.validation.run_validation import main as run_main

    return _call_main_with_argv(run_main, [])


def _run_oos(_argv: Sequence[str]) -> int:
    from feature_research.oos.run_oos import main as run_main

    return _call_main_with_argv(run_main, [])


def _run_binning(_argv: Sequence[str]) -> int:
    from feature_research.binning.run_phase import main as run_main

    return _call_main_with_argv(run_main, [])


def _run_oos_permutation(argv: Sequence[str]) -> int:
    from feature_research.oos.run_oos_permutation import main as run_main

    return _call_main_with_argv(run_main, argv)


def _run_validation_permutation(argv: Sequence[str]) -> int:
    from feature_research.validation.run_validation_permutation import main as run_main

    return _call_main_with_argv(run_main, argv)


def _build_usage() -> str:
    return (
        "Usage: python -m feature_research <phase> [args]\n\n"
        "Phases:\n"
        "  in-sample | in_sample | eda\n"
        "  validation\n"
        "  oos\n"
        "  binning\n"
        "  oos-permutation | oos_permutation [--nreps N --seed S --n-jobs N --output-dir PATH]\n"
        "  validation-permutation | validation_permutation [--nreps N --seed S --n-jobs N --output-dir PATH]\n"
    )


def _phase_help(phase: str) -> str:
    if phase in {"in-sample", "in_sample"}:
        return (
            "Usage: python -m feature_research in-sample\n\n"
            "Runs in-sample EDA and optional permutation work from feature_research.config."
        )
    if phase == "validation":
        return "Usage: python -m feature_research validation\n\nRuns the validation pipeline."
    if phase == "oos":
        return "Usage: python -m feature_research oos\n\nRuns the out-of-sample pipeline."
    if phase == "binning":
        return "Usage: python -m feature_research binning\n\nRuns the research-only continuous binning exports."
    if phase in {"oos-permutation", "oos_permutation"}:
        return (
            "Usage: python -m feature_research oos-permutation [--nreps N --seed S --n-jobs N --output-dir PATH]\n\n"
            "Runs the OOS vector-shuffle permutation pipeline."
        )
    if phase in {"validation-permutation", "validation_permutation"}:
        return (
            "Usage: python -m feature_research validation-permutation [--nreps N --seed S --n-jobs N --output-dir PATH]\n\n"
            "Runs the validation vector-shuffle permutation pipeline."
        )
    return _build_usage()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_build_usage())
        return 0 if args else 1

    phase, phase_args = args[0], args[1:]
    if phase_args and phase_args[0] in {"-h", "--help"}:
        print(_phase_help(phase))
        return 0

    dispatch: dict[str, Callable[[Sequence[str]], int]] = {
        "in-sample": _run_in_sample,
        "in_sample": _run_in_sample,
        "eda": _run_in_sample,
        "validation": _run_validation,
        "oos": _run_oos,
        "binning": _run_binning,
        "oos-permutation": _run_oos_permutation,
        "oos_permutation": _run_oos_permutation,
        "validation-permutation": _run_validation_permutation,
        "validation_permutation": _run_validation_permutation,
    }
    runner = dispatch.get(phase)
    if runner is None:
        print(f"Unknown phase: {phase}")
        print(_build_usage())
        return 1
    return runner(phase_args)


if __name__ == "__main__":
    raise SystemExit(main())
