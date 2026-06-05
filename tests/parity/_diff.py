"""Numeric diff utilities for the research parity harness.

The parity gate compares *current* pipeline outputs against committed golden
snapshots. These helpers do the comparison with explicit, tight tolerances and
produce a human-readable report of the **worst offenders** so a failure is
immediately actionable.

Design notes
------------
- Comparisons are *value* comparisons, not formatting comparisons. Inputs are
  normalised (sorted index/columns, coerced dtypes) before diffing so that an
  incidental reordering never trips the gate while a genuine numeric drift always
  does.
- Default tolerances are tight (``rtol=1e-8, atol=1e-10``) because the data-layer
  migration (WP-2) is expected to be *exactly* identical — only the candle source
  changes, not any math. A looser profile may be passed explicitly for a
  deliberate, reviewed behaviour change.
- NaNs compare equal to NaNs (both-NaN is not a diff). A NaN appearing/vanishing
  on only one side *is* a diff.

This module has no dependency on the pipelines under test, so it stays importable
even when the research environment cannot import the pipeline entrypoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10


@dataclass(frozen=True)
class Tolerance:
    """Relative/absolute tolerance pair for numeric comparison."""

    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL


@dataclass(frozen=True)
class Offender:
    """A single worst-offender cell in a numeric diff."""

    label: str
    expected: float
    actual: float
    abs_diff: float
    rel_diff: float

    def format(self) -> str:
        return (
            f"  {self.label}: expected={self.expected!r} actual={self.actual!r} "
            f"abs={self.abs_diff:.3e} rel={self.rel_diff:.3e}"
        )


@dataclass(frozen=True)
class DiffResult:
    """Outcome of comparing one artifact against its golden snapshot."""

    name: str
    passed: bool
    messages: tuple[str, ...] = field(default_factory=tuple)
    offenders: tuple[Offender, ...] = field(default_factory=tuple)

    def report(self, max_offenders: int = 10) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.name}"]
        lines.extend(f"  {m}" for m in self.messages)
        if self.offenders:
            lines.append(f"  worst offenders (up to {max_offenders}):")
            lines.extend(o.format() for o in self.offenders[:max_offenders])
        return "\n".join(lines)


def _abs_rel(expected: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    abs_diff = np.abs(actual - expected)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(
            expected != 0.0,
            abs_diff / np.abs(expected),
            np.where(abs_diff == 0.0, 0.0, np.inf),
        )
    return abs_diff, rel_diff


def _both_nan_mask(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.isnan(expected) & np.isnan(actual)


def _close_mask(
    expected: np.ndarray,
    actual: np.ndarray,
    tol: Tolerance,
) -> np.ndarray:
    """Element-wise closeness honouring NaN==NaN."""
    finite_close = np.isclose(
        actual, expected, rtol=tol.rtol, atol=tol.atol, equal_nan=False
    )
    return finite_close | _both_nan_mask(expected, actual)


def diff_scalar(
    name: str,
    expected: float,
    actual: float,
    tol: Tolerance = Tolerance(),
) -> DiffResult:
    """Compare two scalars with tolerance."""
    exp_arr = np.asarray([float(expected)], dtype=float)
    act_arr = np.asarray([float(actual)], dtype=float)
    close = _close_mask(exp_arr, act_arr, tol)[0]
    if close:
        return DiffResult(name=name, passed=True)
    abs_diff, rel_diff = _abs_rel(exp_arr, act_arr)
    offender = Offender(
        label="value",
        expected=float(expected),
        actual=float(actual),
        abs_diff=float(abs_diff[0]),
        rel_diff=float(rel_diff[0]),
    )
    return DiffResult(
        name=name,
        passed=False,
        messages=(f"scalar mismatch (rtol={tol.rtol:g}, atol={tol.atol:g})",),
        offenders=(offender,),
    )


def diff_scalars(
    name: str,
    expected: Mapping[str, float],
    actual: Mapping[str, float],
    tol: Tolerance = Tolerance(),
) -> DiffResult:
    """Compare two scalar dictionaries (e.g. headline metrics) with tolerance."""
    exp_keys = set(expected)
    act_keys = set(actual)
    messages: list[str] = []
    if exp_keys != act_keys:
        missing = sorted(exp_keys - act_keys)
        extra = sorted(act_keys - exp_keys)
        if missing:
            messages.append(f"missing keys: {missing}")
        if extra:
            messages.append(f"unexpected keys: {extra}")

    offenders: list[Offender] = []
    for key in sorted(exp_keys & act_keys):
        exp_v = np.asarray([float(expected[key])], dtype=float)
        act_v = np.asarray([float(actual[key])], dtype=float)
        if _close_mask(exp_v, act_v, tol)[0]:
            continue
        abs_diff, rel_diff = _abs_rel(exp_v, act_v)
        offenders.append(
            Offender(
                label=key,
                expected=float(expected[key]),
                actual=float(actual[key]),
                abs_diff=float(abs_diff[0]),
                rel_diff=float(rel_diff[0]),
            )
        )

    offenders.sort(key=lambda o: o.abs_diff, reverse=True)
    passed = not messages and not offenders
    if not passed and not messages:
        messages.append(
            f"{len(offenders)} metric(s) out of tolerance "
            f"(rtol={tol.rtol:g}, atol={tol.atol:g})"
        )
    return DiffResult(
        name=name,
        passed=passed,
        messages=tuple(messages),
        offenders=tuple(offenders),
    )


def diff_series(
    name: str,
    expected: pd.Series,
    actual: pd.Series,
    tol: Tolerance = Tolerance(),
) -> DiffResult:
    """Compare two numeric Series (e.g. a returns series) with tolerance.

    Both series are sorted by index before comparison. Index mismatches are
    reported as structural failures; numeric drift inside the common index is
    reported as worst offenders.
    """
    exp = expected.sort_index()
    act = actual.sort_index()
    messages: list[str] = []

    exp_idx = exp.index
    act_idx = act.index
    if not exp_idx.equals(act_idx):
        only_exp = exp_idx.difference(act_idx)
        only_act = act_idx.difference(exp_idx)
        if len(only_exp):
            messages.append(
                f"{len(only_exp)} index label(s) missing from actual "
                f"(e.g. {list(only_exp[:3])})"
            )
        if len(only_act):
            messages.append(
                f"{len(only_act)} extra index label(s) in actual "
                f"(e.g. {list(only_act[:3])})"
            )
        common = exp_idx.intersection(act_idx)
        exp = exp.loc[common]
        act = act.loc[common]

    exp_vals = exp.to_numpy(dtype=float)
    act_vals = act.to_numpy(dtype=float)
    close = _close_mask(exp_vals, act_vals, tol)
    bad = ~close
    offenders: list[Offender] = []
    if bad.any():
        abs_diff, rel_diff = _abs_rel(exp_vals, act_vals)
        bad_idx = np.flatnonzero(bad)
        order = bad_idx[np.argsort(abs_diff[bad_idx])[::-1]]
        labels = exp.index
        offenders = [
            Offender(
                label=str(labels[i]),
                expected=float(exp_vals[i]),
                actual=float(act_vals[i]),
                abs_diff=float(abs_diff[i]),
                rel_diff=float(rel_diff[i]),
            )
            for i in order
        ]

    passed = not messages and not offenders
    if offenders:
        messages.append(
            f"{len(offenders)}/{len(exp_vals)} value(s) out of tolerance "
            f"(rtol={tol.rtol:g}, atol={tol.atol:g})"
        )
    return DiffResult(
        name=name,
        passed=passed,
        messages=tuple(messages),
        offenders=tuple(offenders),
    )


def diff_dataframe(
    name: str,
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    tol: Tolerance = Tolerance(),
) -> DiffResult:
    """Compare two DataFrames with tolerance.

    Numeric columns are diffed with ``rtol``/``atol``; non-numeric (object/bool)
    columns are compared for exact equality (NaN==NaN). Both frames are reindexed
    to a sorted, aligned shape first so column/row order is never load-bearing.
    """
    messages: list[str] = []

    exp = expected.copy()
    act = actual.copy()

    # Align columns.
    exp_cols = list(exp.columns)
    act_cols = list(act.columns)
    if set(exp_cols) != set(act_cols):
        missing = sorted(set(map(str, exp_cols)) - set(map(str, act_cols)))
        extra = sorted(set(map(str, act_cols)) - set(map(str, exp_cols)))
        if missing:
            messages.append(f"missing column(s): {missing}")
        if extra:
            messages.append(f"unexpected column(s): {extra}")
    common_cols = [c for c in exp_cols if c in set(act_cols)]
    exp = exp.loc[:, common_cols].sort_index(axis=0)
    act = act.loc[:, common_cols].sort_index(axis=0)

    # Align rows.
    if not exp.index.equals(act.index):
        only_exp = exp.index.difference(act.index)
        only_act = act.index.difference(exp.index)
        if len(only_exp):
            messages.append(
                f"{len(only_exp)} row label(s) missing from actual "
                f"(e.g. {list(only_exp[:3])})"
            )
        if len(only_act):
            messages.append(
                f"{len(only_act)} extra row label(s) in actual "
                f"(e.g. {list(only_act[:3])})"
            )
        common_idx = exp.index.intersection(act.index)
        exp = exp.loc[common_idx]
        act = act.loc[common_idx]

    offenders: list[Offender] = []
    n_cells = 0
    for col in common_cols:
        e_col = exp[col]
        a_col = act[col]
        is_numeric = pd.api.types.is_numeric_dtype(e_col) and not pd.api.types.is_bool_dtype(
            e_col
        )
        if is_numeric:
            e_vals = e_col.to_numpy(dtype=float)
            a_vals = a_col.to_numpy(dtype=float)
            n_cells += len(e_vals)
            close = _close_mask(e_vals, a_vals, tol)
            bad = ~close
            if bad.any():
                abs_diff, rel_diff = _abs_rel(e_vals, a_vals)
                for i in np.flatnonzero(bad):
                    offenders.append(
                        Offender(
                            label=f"{col}[{exp.index[i]}]",
                            expected=float(e_vals[i]),
                            actual=float(a_vals[i]),
                            abs_diff=float(abs_diff[i]),
                            rel_diff=float(rel_diff[i]),
                        )
                    )
        else:
            # Exact comparison for non-numeric columns; NaN==NaN.
            e_obj = e_col.to_numpy(dtype=object)
            a_obj = a_col.to_numpy(dtype=object)
            n_cells += len(e_obj)
            for i in range(len(e_obj)):
                ev, av = e_obj[i], a_obj[i]
                both_nan = _is_nan(ev) and _is_nan(av)
                if both_nan or ev == av:
                    continue
                offenders.append(
                    Offender(
                        label=f"{col}[{exp.index[i]}]",
                        expected=float("nan"),
                        actual=float("nan"),
                        abs_diff=float("nan"),
                        rel_diff=float("nan"),
                    )
                )
                messages.append(f"non-numeric mismatch {col}[{exp.index[i]}]: {ev!r} != {av!r}")

    offenders.sort(
        key=lambda o: (-(o.abs_diff if np.isfinite(o.abs_diff) else np.inf)),
    )
    passed = not messages and not offenders
    if offenders and not any("out of tolerance" in m for m in messages):
        messages.append(
            f"{len(offenders)}/{n_cells} cell(s) out of tolerance "
            f"(rtol={tol.rtol:g}, atol={tol.atol:g})"
        )
    return DiffResult(
        name=name,
        passed=passed,
        messages=tuple(messages),
        offenders=tuple(offenders),
    )


def _is_nan(value: object) -> bool:
    try:
        return bool(np.isnan(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def assert_all_pass(results: list[DiffResult], max_offenders: int = 10) -> None:
    """Raise AssertionError with a combined report if any diff failed."""
    failed = [r for r in results if not r.passed]
    report = "\n".join(r.report(max_offenders=max_offenders) for r in results)
    if failed:
        raise AssertionError(
            f"\nPARITY FAILURE: {len(failed)} of {len(results)} artifact(s) drifted.\n"
            f"A parity failure BLOCKS the change — investigate before regenerating "
            f"snapshots.\n\n{report}"
        )
