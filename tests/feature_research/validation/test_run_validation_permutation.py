"""Tests for validation permutation script (single fold)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from feature_research.config import OOSWindowConfig
from feature_research.validation.run_validation_permutation import main


def test_validation_permutation_main_returns_1_when_validation_window_none() -> None:
    with (
        patch("sys.argv", ["run_validation_permutation.py"]),
        patch("feature_research.validation.run_validation_permutation.load_config") as m_load,
    ):
        config = m_load.return_value
        config.validation_window = None
        exit_code = main()
    assert exit_code == 1


def test_validation_permutation_produces_report_and_null_distribution(tmp_path: Path) -> None:
    from dataclasses import replace

    from feature_research.in_sample.config import load_config

    index = pd.date_range("2020-01-01", periods=1100, freq="D")
    reference_target = pd.Series(0.01, index=index, name="walkforward_target")
    reference_candles = pd.DataFrame({"close": reference_target}, index=index)
    validation = OOSWindowConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2021, 12, 31),
        test_start=datetime(2022, 1, 1),
        test_end=datetime(2022, 12, 31),
    )

    def fake_load_research_data(cfg: object) -> tuple:
        return (
            reference_candles,
            reference_target,
            [{"x": 1}],
            lambda *args, **kwargs: pd.Series(0.1, index=reference_target.index),
            cfg,
            None,
            None,
        )

    minimal_report = type("Report", (), {})()
    minimal_report.selection_summary_df = pd.DataFrame(
        [{"fold_id": 0, "top_k_features": "[]"}],
    )

    with (
        patch("sys.argv", ["run_validation_permutation.py", "--nreps", "3"]),
        patch("feature_research.validation.run_validation_permutation.load_config") as m_load,
        patch(
            "feature_research.validation.permutation_helpers.load_research_data",
            side_effect=fake_load_research_data,
        ),
        patch(
            "utils.evaluation.walkforward.runner.run_walkforward_research",
            return_value=minimal_report,
        ),
        patch(
            "utils.evaluation.walkforward.permutation_core.run_vector_shuffle_null",
            return_value=np.array([0.0, 0.0, 0.0]),
        ),
    ):
        base_config = load_config()
        config_with_validation = replace(base_config, validation_window=validation)
        m_load.return_value = config_with_validation
        with patch(
            "utils.evaluation.walkforward.io.resolve_walkforward_output_dir",
            return_value=tmp_path / "continuous" / "rsi" / "validation",
        ):
            exit_code = main()
    assert exit_code == 0
    perm_dir = tmp_path / "continuous" / "rsi" / "validation" / "permutation"
    assert perm_dir.is_dir()
    report_path = perm_dir / "validation_permutation_report.json"
    null_path = perm_dir / "null_distribution.npy"
    assert report_path.exists()
    assert null_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "p_value" in report
    assert 0 <= report["p_value"] <= 1
    null_arr = np.load(null_path)
    assert null_arr.shape == (3,)
