"""Unit tests for lib.core.atomic_io (atomic writes + corrupt-read quarantine)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lib.core.atomic_io import atomic_write, atomic_write_text, read_or_quarantine


def test_atomic_write_text_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "state.json"
    atomic_write_text(p, json.dumps({"a": 1}))
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}
    # No temp files left behind.
    assert list(p.parent.glob(".*tmp*")) == []


def test_atomic_write_replaces_existing(tmp_path: Path) -> None:
    p = tmp_path / "f.parquet"
    atomic_write(p, lambda tmp: pd.DataFrame({"x": [1, 2]}).to_parquet(tmp))
    atomic_write(p, lambda tmp: pd.DataFrame({"x": [9]}).to_parquet(tmp))
    assert pd.read_parquet(p)["x"].tolist() == [9]


def test_atomic_write_leaves_original_on_failure(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    atomic_write_text(p, "good")

    def _boom(_tmp: Path) -> None:
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError):
        atomic_write(p, _boom)
    assert p.read_text(encoding="utf-8") == "good"  # original intact
    assert list(p.parent.glob(".*tmp*")) == []      # temp cleaned up


def test_read_or_quarantine_missing_returns_none(tmp_path: Path) -> None:
    assert read_or_quarantine(tmp_path / "nope.parquet", pd.read_parquet) is None


def test_read_or_quarantine_corrupt_quarantines(tmp_path: Path) -> None:
    p = tmp_path / "f.parquet"
    p.write_bytes(b"not a parquet file")  # corrupt
    out = read_or_quarantine(p, pd.read_parquet)
    assert out is None
    assert not p.exists()                                  # moved aside
    assert (tmp_path / "f.parquet.corrupt").exists()       # quarantined


def test_read_or_quarantine_valid(tmp_path: Path) -> None:
    p = tmp_path / "f.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(p)
    out = read_or_quarantine(p, pd.read_parquet)
    assert out is not None and out["x"].tolist() == [1]
