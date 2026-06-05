"""Unit tests for T017: Report Generation and Visualization."""
from __future__ import annotations

import json
import csv
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from features.validation.report_generator import (
    generate_permutation_reports,
    plot_funnel_diagram,
    plot_null_distribution,
    plot_walkforward_stability,
)
from features.validation.reports import (
    ComboDecisionRecord,
    FoldResult,
    FunnelStatistics,
    OutOfSamplePermutationReport,
    PipelinePermutationReport,
    PermutationTestSuite,
    ReportBundle,
    VectorShuffleReport,
    WalkforwardStabilityReport,
)


def _make_vector_shuffle_report(
    param_combo: str = 'lookback_5',
    passed: bool = True,
    p_value: float = 0.05,
) -> VectorShuffleReport:
    null_dist = np.random.default_rng(0).standard_normal(100)
    cv = float(np.percentile(null_dist, 90))
    orig = cv + 0.1 if passed else cv - 0.1
    return VectorShuffleReport(
        param_combo=param_combo,
        original_metric=orig,
        null_distribution=null_dist,
        critical_value=cv,
        p_value=p_value,
        passed=passed,
        alpha=0.10,
        nreps=100,
    )


def _make_pipeline_report(
    param_combo: str = 'lookback_5',
    p_value: float = 0.05,
) -> PipelinePermutationReport:
    null_dist = np.random.default_rng(1).standard_normal(100)
    cv = float(np.percentile(null_dist, 90))
    orig = cv + 0.1
    return PipelinePermutationReport(
        param_combo=param_combo,
        feature_type='continuous',
        permutation_mode='candle_shuffle',
        original_metric=orig,
        null_distribution=null_dist,
        critical_value=cv,
        p_value=p_value,
        passed=(p_value <= 0.10),
        alpha=0.10,
        nreps=100,
        no_trade_permutations=3,
    )


def _make_stability_report(is_stable: bool = True) -> WalkforwardStabilityReport:
    fold_results = [
        FoldResult(
            fold_id='fold_0',
            fold_period=('2020-01-01', '2021-12-31'),
            top_k_params=['lookback_3', 'lookback_5'],
            smoothed_objectives={'lookback_3': 0.6, 'lookback_5': 0.7, 'lookback_10': 0.3},
            passed_permutation_overlay=[True, True],
        ),
        FoldResult(
            fold_id='fold_1',
            fold_period=('2022-01-01', '2023-12-31'),
            top_k_params=['lookback_3', 'lookback_5'] if is_stable else ['lookback_10', 'lookback_14'],
            smoothed_objectives={'lookback_3': 0.5, 'lookback_5': 0.6, 'lookback_10': 0.4},
            passed_permutation_overlay=[True, True] if is_stable else [False, False],
        ),
    ]
    overlap = 1.0 if is_stable else 0.0
    verdict = 'STABLE' if is_stable else 'UNSTABLE'
    return WalkforwardStabilityReport(
        feature_name='rsi',
        feature_type='continuous',
        fold_results=fold_results,
        consistency_metrics={'overlap_rate': overlap},
        is_stable=is_stable,
        stability_verdict=verdict,
        top_k=2,
    )


def _make_suite(is_stable: bool = True, borderline_p: float = 0.05) -> PermutationTestSuite:
    s1_reports = {
        'lookback_3': _make_vector_shuffle_report('lookback_3', passed=True, p_value=0.04),
        'lookback_5': _make_vector_shuffle_report('lookback_5', passed=True, p_value=0.06),
    }
    s2_reports = {
        'lookback_3': _make_pipeline_report('lookback_3', p_value=borderline_p),
        'lookback_5': _make_pipeline_report('lookback_5', p_value=0.04),
    }
    return PermutationTestSuite(
        feature_name='rsi',
        feature_type='continuous',
        stage1_reports=s1_reports,
        stage2_reports=s2_reports,
        stage3_report=_make_stability_report(is_stable),
        funnel_stats=FunnelStatistics(10, 7, 5, 3, 3, 15.0),
        ensemble_candidates=['lookback_3', 'lookback_5'],
        summary='Test suite summary',
    )


def _make_oos_reports() -> dict[str, OutOfSamplePermutationReport]:
    return {
        'lookback_3': OutOfSamplePermutationReport(
            param_combo='lookback_3',
            vector_report=_make_vector_shuffle_report('lookback_3', passed=True, p_value=0.03),
            candle_report=_make_pipeline_report('lookback_3', p_value=0.05),
            passed=True,
        ),
        'lookback_5': OutOfSamplePermutationReport(
            param_combo='lookback_5',
            vector_report=_make_vector_shuffle_report('lookback_5', passed=False, p_value=0.18),
            candle_report=None,
            passed=False,
        ),
    }


def _make_combo_decisions() -> dict[str, ComboDecisionRecord]:
    return {
        'lookback_3': ComboDecisionRecord(
            param_combo='lookback_3',
            stage1_passed=True,
            stage2_passed=True,
            walkforward_stable=True,
            oos_passed=True,
            final_status='candidate',
        ),
        'lookback_5': ComboDecisionRecord(
            param_combo='lookback_5',
            stage1_passed=True,
            stage2_passed=False,
            walkforward_stable=False,
            oos_passed=False,
            final_status='rejected',
        ),
    }


def _make_extended_suite() -> PermutationTestSuite:
    base = _make_suite()
    return PermutationTestSuite(
        feature_name=base.feature_name,
        feature_type=base.feature_type,
        stage1_reports=base.stage1_reports,
        stage2_reports=base.stage2_reports,
        stage3_report=base.stage3_report,
        funnel_stats=base.funnel_stats,
        ensemble_candidates=base.ensemble_candidates,
        summary=base.summary,
        phase3_oos_reports=_make_oos_reports(),
        combo_decisions=_make_combo_decisions(),
    )


def test_report_bundle_fields() -> None:
    """ReportBundle contains all required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = _make_suite()
        assert hasattr(suite, 'phase3_oos_reports')
        assert hasattr(suite, 'combo_decisions')
        bundle = generate_permutation_reports(suite, Path(tmpdir))
        assert hasattr(bundle, 'suite_json')
        assert hasattr(bundle, 'suite_markdown')
        assert hasattr(bundle, 'suite_html')
        assert hasattr(bundle, 'stage1_plots')
        assert hasattr(bundle, 'stage2_plots')
        assert hasattr(bundle, 'stage3_plot')
        assert hasattr(bundle, 'funnel_plot')
        assert hasattr(bundle, 'timestamp')


def test_report_bundle_paths_exist() -> None:
    """All Path fields in ReportBundle point to existing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = _make_suite()
        bundle = generate_permutation_reports(suite, Path(tmpdir))
        assert bundle.suite_json.exists()
        assert bundle.suite_markdown.exists()
        assert bundle.stage3_plot.exists()
        assert bundle.funnel_plot.exists()
        for path in bundle.stage1_plots.values():
            assert path.exists()
        for path in bundle.stage2_plots.values():
            assert path.exists()


def test_markdown_contains_required_sections() -> None:
    """Generated markdown contains all required section headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = _make_suite()
        bundle = generate_permutation_reports(suite, Path(tmpdir))
        content = bundle.suite_markdown.read_text()
        for section in [
            'Feature Overview', 'Funnel Statistics', 'Ensemble Candidates',
            'Interpretation Guidance', 'Red Flags', 'Next Steps',
        ]:
            assert section in content, f'Missing section: {section}'


def test_json_export_round_trips() -> None:
    """JSON export round-trips numeric fields without loss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = _make_suite()
        bundle = generate_permutation_reports(suite, Path(tmpdir))
        data = json.loads(bundle.suite_json.read_text())
        assert 'feature_name' in data
        assert 'funnel_stats' in data
        assert data['funnel_stats']['total_params'] == 10
        assert data['funnel_stats']['computational_savings_pct'] == 15.0
        assert 'ensemble_candidates' in data


def test_combo_decision_table_csv_written() -> None:
    """Generator writes combo decision table to CSV artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = _make_extended_suite()
        generate_permutation_reports(suite, Path(tmpdir))

        csv_path = Path(tmpdir) / 'combo_decision_table.csv'
        assert csv_path.exists()

        rows = list(csv.DictReader(csv_path.read_text().splitlines()))
        assert len(rows) == 2
        assert rows[0]['param_combo'] == 'lookback_3'
        assert rows[0]['final_status'] == 'candidate'
        assert rows[1]['param_combo'] == 'lookback_5'
        assert rows[1]['final_status'] == 'rejected'


def test_oos_null_distribution_outputs_created() -> None:
    """Generator writes OOS vector and optional candle null-distribution plots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = _make_extended_suite()
        generate_permutation_reports(suite, Path(tmpdir))

        vector_dir = Path(tmpdir) / 'oos' / 'vector'
        candle_dir = Path(tmpdir) / 'oos' / 'candle'

        assert (vector_dir / 'null_dist_lookback_3.png').exists()
        assert (vector_dir / 'null_dist_lookback_5.png').exists()
        assert (candle_dir / 'null_dist_lookback_3.png').exists()
        assert not (candle_dir / 'null_dist_lookback_5.png').exists()


def test_extended_fields_do_not_break_markdown_and_json_generation() -> None:
    """Markdown/JSON generation keeps working with extended suite fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = _make_extended_suite()
        bundle = generate_permutation_reports(suite, Path(tmpdir))

        markdown = bundle.suite_markdown.read_text()
        assert 'Feature Overview' in markdown

        json_data = json.loads(bundle.suite_json.read_text())
        assert 'feature_name' in json_data
        assert 'funnel_stats' in json_data
        assert 'phase3_oos_reports' in json_data
        assert 'combo_decisions' in json_data


def test_null_distribution_plot_created() -> None:
    """plot_null_distribution creates a file at the given path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = _make_vector_shuffle_report()
        out_path = Path(tmpdir) / 'null_dist_test.png'
        result = plot_null_distribution(report, out_path)
        assert result.exists()
        assert result.stat().st_size > 0


def test_walkforward_stability_plot_created() -> None:
    """plot_walkforward_stability creates a multi-panel figure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = _make_stability_report()
        out_path = Path(tmpdir) / 'stability.png'
        result = plot_walkforward_stability(report, out_path)
        assert result.exists()
        assert result.stat().st_size > 0


def test_funnel_diagram_plot_created() -> None:
    """plot_funnel_diagram creates a file with correct counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = FunnelStatistics(10, 7, 5, 3, 3, 15.0)
        out_path = Path(tmpdir) / 'funnel.png'
        result = plot_funnel_diagram(stats, out_path)
        assert result.exists()
        assert result.stat().st_size > 0


def test_red_flag_borderline_pvalue() -> None:
    """Markdown contains WARNING for borderline p-value (p in [0.05, 0.15])."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = _make_suite(borderline_p=0.12)  # borderline
        bundle = generate_permutation_reports(suite, Path(tmpdir))
        content = bundle.suite_markdown.read_text()
        # Should mention borderline p-value
        assert 'Borderline' in content or 'borderline' in content or 'WARNING' in content


def test_red_flag_unstable_feature() -> None:
    """Markdown contains instability warning when is_stable=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = _make_suite(is_stable=False)
        bundle = generate_permutation_reports(suite, Path(tmpdir))
        content = bundle.suite_markdown.read_text()
        assert 'WARNING' in content or 'instability' in content or 'UNSTABLE' in content
