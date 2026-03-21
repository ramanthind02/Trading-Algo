"""Tests for control-file selection metadata without member schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ensemble.ensemble_utils import parse_control_file, save_control_file


def _base_models() -> list[dict]:
    return [
        {
            "name": "test_model",
            "model_type": "continuous_binning",
            "feature_column": "test_feature",
            "strategy": "long",
            "constructor_params": {"n_bins": 3},
        }
    ]


def test_save_control_file_includes_selection_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        control_file_path = f"{tmpdir}/control.json"
        metadata = {
            "is_fit": False,
            "version": "2.0.0",
            "selection_method": "walkforward_stability",
            "selection_hyperparams": {"min_stability_score": 0.7},
        }
        save_control_file(
            filepath=control_file_path,
            base_models=_base_models(),
            metadata=metadata,
        )
        with open(control_file_path, "r") as handle:
            saved = json.load(handle)
        assert saved["metadata"]["selection_method"] == "walkforward_stability"
        assert saved["metadata"]["selection_hyperparams"]["min_stability_score"] == 0.7


def test_parse_control_file_with_selection_metadata() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        payload = {
            "metadata": {
                "is_fit": False,
                "version": "2.0.0",
                "selection_method": "walkforward_stability",
                "selection_hyperparams": {"n_folds": 5},
            },
            "base_models": _base_models(),
            "tickers": ["ES"],
        }
        json.dump(payload, handle)
        path = handle.name
    try:
        parsed = parse_control_file(path)
        assert parsed["metadata"]["selection_hyperparams"]["n_folds"] == 5
    finally:
        Path(path).unlink(missing_ok=True)
