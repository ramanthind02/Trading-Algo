"""Unit tests for the pure interval-coverage math in tick_cache.

These cover `merge_intervals` and `missing_ranges` — the logic that decides
which sub-ranges of a requested window must be fetched from MT5. No MT5
connection or filesystem is touched, so they run anywhere.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data_platform.providers.mt5.tick_cache import merge_intervals, missing_ranges


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# ── merge_intervals ──────────────────────────────────────────────────────────

def test_merge_empty():
    assert merge_intervals([]) == []


def test_merge_disjoint_sorted():
    ivs = [(_t("2024-01-01"), _t("2024-01-02")), (_t("2024-01-05"), _t("2024-01-06"))]
    assert merge_intervals(ivs) == ivs


def test_merge_unsorted_overlapping():
    ivs = [
        (_t("2024-01-05"), _t("2024-01-07")),
        (_t("2024-01-01"), _t("2024-01-03")),
        (_t("2024-01-02"), _t("2024-01-06")),  # bridges the two
    ]
    assert merge_intervals(ivs) == [(_t("2024-01-01"), _t("2024-01-07"))]


def test_merge_touching_intervals_join():
    # end == next start → adjacent, should merge into one
    ivs = [(_t("2024-01-01"), _t("2024-01-02")), (_t("2024-01-02"), _t("2024-01-03"))]
    assert merge_intervals(ivs) == [(_t("2024-01-01"), _t("2024-01-03"))]


# ── missing_ranges ───────────────────────────────────────────────────────────

def test_missing_nothing_covered():
    req_s, req_e = _t("2024-01-01"), _t("2024-01-02")
    assert missing_ranges(req_s, req_e, []) == [(req_s, req_e)]


def test_missing_fully_covered():
    covered = [(_t("2024-01-01"), _t("2024-01-10"))]
    assert missing_ranges(_t("2024-01-02"), _t("2024-01-03"), covered) == []


def test_missing_leading_gap():
    covered = [(_t("2024-01-05"), _t("2024-01-10"))]
    gaps = missing_ranges(_t("2024-01-01"), _t("2024-01-07"), covered)
    assert gaps == [(_t("2024-01-01"), _t("2024-01-05"))]


def test_missing_trailing_gap():
    covered = [(_t("2024-01-01"), _t("2024-01-05"))]
    gaps = missing_ranges(_t("2024-01-03"), _t("2024-01-08"), covered)
    assert gaps == [(_t("2024-01-05"), _t("2024-01-08"))]


def test_missing_interior_hole():
    covered = [
        (_t("2024-01-01"), _t("2024-01-03")),
        (_t("2024-01-05"), _t("2024-01-07")),
    ]
    gaps = missing_ranges(_t("2024-01-01"), _t("2024-01-07"), covered)
    assert gaps == [(_t("2024-01-03"), _t("2024-01-05"))]


def test_missing_multiple_holes():
    covered = [
        (_t("2024-01-02"), _t("2024-01-03")),
        (_t("2024-01-04"), _t("2024-01-05")),
    ]
    gaps = missing_ranges(_t("2024-01-01"), _t("2024-01-06"), covered)
    assert gaps == [
        (_t("2024-01-01"), _t("2024-01-02")),
        (_t("2024-01-03"), _t("2024-01-04")),
        (_t("2024-01-05"), _t("2024-01-06")),
    ]


def test_missing_empty_request():
    # start >= end → nothing to fetch
    assert missing_ranges(_t("2024-01-02"), _t("2024-01-02"), []) == []
    assert missing_ranges(_t("2024-01-03"), _t("2024-01-02"), []) == []


def test_missing_partial_overlap_both_sides():
    # request straddles a single covered block → leading + trailing gaps
    covered = [(_t("2024-01-03"), _t("2024-01-05"))]
    gaps = missing_ranges(_t("2024-01-01"), _t("2024-01-08"), covered)
    assert gaps == [
        (_t("2024-01-01"), _t("2024-01-03")),
        (_t("2024-01-05"), _t("2024-01-08")),
    ]


def test_roundtrip_merge_then_missing_is_empty():
    # After covering exactly the gaps, a re-request has no missing ranges.
    req_s, req_e = _t("2024-01-01"), _t("2024-01-10")
    covered: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    gaps = missing_ranges(req_s, req_e, covered)
    covered = merge_intervals(covered + gaps)
    assert missing_ranges(req_s, req_e, covered) == []
