from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Win32 extended paths only apply on Windows")


def test_win32_extended_path_prefixes_drive_path(tmp_path: Path) -> None:
    from lib.cache.runtime.cache_paths import win32_extended_path

    f = tmp_path / "sample.txt"
    f.write_text("x", encoding="utf-8")
    out = win32_extended_path(f)
    assert out.startswith("\\\\?\\")
    assert out.endswith("sample.txt")


def test_win32_extended_path_idempotent(tmp_path: Path) -> None:
    from lib.cache.runtime.cache_paths import win32_extended_path

    f = tmp_path / "a.txt"
    f.write_text("a", encoding="utf-8")
    once = win32_extended_path(f)
    assert win32_extended_path(Path(once)) == once


def test_feature_file_exists_round_trip(tmp_path: Path) -> None:
    """Vault feature JSON existence check uses extended paths on Windows."""
    from ensemble.vault.feature_files import feature_file_exists

    f = tmp_path / "short_name.json"
    f.write_text("{}", encoding="utf-8")
    assert feature_file_exists(f)
    assert not feature_file_exists(tmp_path / "missing.json")


def test_read_utf8_text_materialization_helper(tmp_path: Path) -> None:
    """portfolio_materialization reads vault JSON via extended path on Windows."""
    from lib.cache.runtime.portfolio_materialization import _read_utf8_text

    f = tmp_path / "feature.json"
    f.write_text('{"base_models": []}', encoding="utf-8")
    assert _read_utf8_text(f).strip() == '{"base_models": []}'
