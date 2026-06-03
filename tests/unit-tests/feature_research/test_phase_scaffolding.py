from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from feature_research.exploration import (
    PHASE as EXPLORATION_PHASE,
    run_exploration_permutation_pipeline,
    run_exploration_pipeline,
    run_exploration_robustness_pipeline,
    write_exploration_permutation_summary,
    write_exploration_robustness_summary,
)
from feature_research.portfolio_addition import (
    PHASE as PORTFOLIO_ADDITION_PHASE,
    PortfolioAdditionBundle,
    run_portfolio_addition_pipeline,
    run_portfolio_addition_pipeline_with_bundle,
)
from feature_research.shared import FeatureResearchPhase, OosCorrelationBundle
from feature_research.validation import (
    PHASE as VALIDATION_PHASE,
    run_validation_pipeline,
)
from utils.core.enums import TimeFrame


def test_shared_contracts_preserve_legacy_oos_bundle_alias() -> None:
    combo_signal_target = {(("lookback", 10),): pd.DataFrame({"signal": [1.0]})}
    bundle = PortfolioAdditionBundle(
        combo_signal_target=combo_signal_target,
        selection_summary_df=pd.DataFrame({"fold_id": [1]}),
        eval_tf=TimeFrame.D,
        research_eval_bias_spec={"module_name": "rsisignal"},
        target_col="target",
        extended_start=datetime(2020, 1, 1),
        extended_end=datetime(2020, 1, 31),
        module_name="rsisignal",
    )

    assert OosCorrelationBundle is PortfolioAdditionBundle
    assert bundle.eval_tf == TimeFrame.D


def test_exploration_public_api_forwards_to_current_pipelines(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = object()
    exploration_result = {"combo_a": tmp_path / "combo_a"}
    robustness_report = object()
    permutation_suite = object()
    raw_parameter_grid = [{"lookback": 2, "enabled": True}]
    artifact_paths = {"summary_csv": tmp_path / "robustness_summary.csv"}
    permutation_paths = (tmp_path / "permutation_summary.csv", tmp_path / "permutation_summary.md")

    monkeypatch.setattr(
        "feature_research.exploration._run_eda_pipeline",
        lambda received_config, received_output_dir: (
            exploration_result
            if received_config is config and received_output_dir == tmp_path
            else None
        ),
    )
    monkeypatch.setattr(
        "feature_research.exploration._run_robustness_pipeline",
        lambda received_config, received_output_dir: (
            robustness_report,
            raw_parameter_grid,
        )
        if received_config is config and received_output_dir == tmp_path
        else (None, []),
    )
    monkeypatch.setattr(
        "feature_research.exploration._write_robustness_summary",
        lambda received_report, received_output_dir: (
            artifact_paths
            if received_report is robustness_report and received_output_dir == tmp_path
            else {}
        ),
    )
    monkeypatch.setattr(
        "feature_research.exploration._run_permutation_pipeline",
        lambda received_config, received_output_dir: (
            permutation_suite,
            raw_parameter_grid,
        )
        if received_config is config and received_output_dir == tmp_path
        else (None, []),
    )
    monkeypatch.setattr(
        "feature_research.exploration._write_permutation_summary",
        lambda received_suite, received_output_dir, *, objective_metric=None, param_grid=None: (
            permutation_paths
            if received_suite is permutation_suite
            and received_output_dir == tmp_path
            and objective_metric is None
            and param_grid == raw_parameter_grid
            else (tmp_path / "unexpected.csv", tmp_path / "unexpected.md")
        ),
    )

    assert EXPLORATION_PHASE == FeatureResearchPhase.EXPLORATION
    assert run_exploration_pipeline(config, tmp_path) == exploration_result
    assert run_exploration_robustness_pipeline(config, tmp_path) == (
        robustness_report,
        raw_parameter_grid,
    )
    assert run_exploration_permutation_pipeline(config, tmp_path) == (
        permutation_suite,
        raw_parameter_grid,
    )
    assert (
        write_exploration_robustness_summary(robustness_report, tmp_path)
        == artifact_paths
    )
    assert write_exploration_permutation_summary(
        permutation_suite,
        tmp_path,
        param_grid=raw_parameter_grid,
    ) == permutation_paths


def test_validation_public_api_forwards_to_current_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = object()
    validation_report = object()

    monkeypatch.setattr(
        "feature_research.validation._run_validation_pipeline",
        lambda received_config, received_output_dir: (
            validation_report
            if received_config is config and received_output_dir == tmp_path
            else None
        ),
    )

    assert VALIDATION_PHASE == FeatureResearchPhase.VALIDATION
    assert run_validation_pipeline(config, tmp_path) is validation_report


def test_portfolio_addition_public_api_forwards_to_oos_pipeline(
    monkeypatch,
) -> None:
    config = object()
    report = object()
    bundle = PortfolioAdditionBundle(
        combo_signal_target={},
        selection_summary_df=pd.DataFrame(),
        eval_tf=TimeFrame.D,
        research_eval_bias_spec={"module_name": "carry"},
        target_col="target",
        extended_start=datetime(2021, 1, 1),
        extended_end=datetime(2021, 1, 31),
        module_name="carry",
    )

    monkeypatch.setattr(
        "feature_research.portfolio_addition._run_oos_pipeline",
        lambda received_config: report if received_config is config else None,
    )
    monkeypatch.setattr(
        "feature_research.portfolio_addition._run_oos_pipeline_with_bundle",
        lambda received_config: (report, bundle) if received_config is config else (None, None),
    )

    assert PORTFOLIO_ADDITION_PHASE == FeatureResearchPhase.PORTFOLIO_ADDITION
    assert run_portfolio_addition_pipeline(config) is report
    assert run_portfolio_addition_pipeline_with_bundle(config) == (report, bundle)
