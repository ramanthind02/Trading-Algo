"""Tests for OOS permutation script (vector-shuffle only, single fold)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from feature_research.config import OOSWindowConfig
from feature_research.oos.run_oos_permutation import main


def test_oos_permutation_main_returns_1_when_oos_window_none() -> None:
    """When config.oos_window is None, main() exits with code 1."""
    with (
        patch("sys.argv", ["run_oos_permutation.py"]),
        patch("feature_research.oos.run_oos_permutation.load_config") as m_load,
    ):
        config = m_load.return_value
        config.oos_window = None
        exit_code = main()
    assert exit_code == 1


def test_oos_permutation_produces_report_and_null_distribution(tmp_path: Path) -> None:
    """With mocked data and pipeline, main() writes report and null dist; p_value in [0, 1]."""
    from dataclasses import replace

    from feature_research.in_sample.config import load_config

    index = pd.date_range("2020-01-01", periods=1100, freq="D")
    reference_target = pd.Series(0.01, index=index, name="walkforward_target")
    reference_candles = pd.DataFrame({"close": reference_target}, index=index)
    oos = OOSWindowConfig(
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
    minimal_report.aggregate_oos_returns = pd.Series(
        [0.01, -0.005, 0.003], index=pd.date_range("2022-01-01", periods=3, freq="D")
    )

    with (
        patch("sys.argv", ["run_oos_permutation.py"]),
        patch("feature_research.oos.run_oos_permutation.load_config") as m_load,
        patch(
            "feature_research.oos.run_oos_permutation._load_research_data",
            side_effect=fake_load_research_data,
        ),
        patch(
            "feature_research.oos.run_oos_permutation.run_walkforward_research",
            return_value=minimal_report,
        ),
        patch(
            "feature_research.oos.run_oos_permutation.run_return_shuffle_null",
            return_value=np.array([0.0, 0.0, 0.0]),
        ),
    ):
        base_config = load_config()
        config_with_oos = replace(base_config, oos_window=oos)
        m_load.return_value = config_with_oos
        # Force output to tmp_path
        with patch(
            "feature_research.oos.run_oos_permutation.resolve_walkforward_output_dir",
            return_value=tmp_path / "continuous" / "rsi" / "oos",
        ):
            exit_code = main()
    assert exit_code == 0
    perm_dir = tmp_path / "continuous" / "rsi" / "oos" / "permutation"
    assert perm_dir.is_dir()
    report_path = perm_dir / "oos_permutation_report.json"
    null_path = perm_dir / "null_distribution.npy"
    assert report_path.exists()
    assert null_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "p_value" in report
    assert 0 <= report["p_value"] <= 1
    null_arr = np.load(null_path)
    assert null_arr.shape == (3,)
