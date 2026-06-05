"""Tests for OOS pipeline public API: report-only vs report+bundle."""
from __future__ import annotations

import pytest

from research.feature.config import load_config
from research.feature.pipeline import run_oos_pipeline, run_oos_pipeline_with_bundle


def test_run_oos_pipeline_returns_only_report(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_report = object()
    sentinel_bundle = object()

    def _fake_run(*_args: object, **_kwargs: object) -> tuple[object, object]:
        return sentinel_report, sentinel_bundle

    monkeypatch.setattr(
        "research.feature.pipelines.oos._run_evaluation_pipeline",
        _fake_run,
    )
    cfg = load_config()
    out = run_oos_pipeline(cfg)
    assert out is sentinel_report


def test_run_oos_pipeline_with_bundle_returns_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_report = object()
    sentinel_bundle = object()

    def _fake_run(*_args: object, **_kwargs: object) -> tuple[object, object]:
        return sentinel_report, sentinel_bundle

    monkeypatch.setattr(
        "research.feature.pipelines.oos._run_evaluation_pipeline",
        _fake_run,
    )
    cfg = load_config()
    report, bundle = run_oos_pipeline_with_bundle(cfg)
    assert report is sentinel_report
    assert bundle is sentinel_bundle
