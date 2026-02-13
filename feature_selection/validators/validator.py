"""Main FeatureValidator class."""
from pathlib import Path
from typing import Generator, Any
import pandas as pd
import numpy as np

from datetime import datetime

from feature_selection.validators.config import ValidationConfig
from feature_selection.validators.reports.eda import EDAReport, ContinuousEDAReport, RuleEDAReport
from feature_selection.validators.reports.permutation import PermutationReport
from feature_selection.validators.reports.validation import ValidationReport
from feature_selection.validators.eda.common import (
    compute_distribution_stats,
    compute_correlations,
    run_stationarity_tests,
    compute_lagged_correlations,
    compute_rolling_correlation,
)
from feature_selection.validators.eda.continuous import (
    compute_decile_stats,
    check_monotonicity,
    detect_outliers,
    fit_polynomial_regression,
)
from feature_selection.validators.eda.rule_based import (
    compute_level_distribution,
    compute_level_stats,
    compute_transition_matrix,
    compute_average_duration,
    compute_level_confidence_intervals,
)
from feature_selection.validators.permutation import run_vector_shuffle_test


class FeatureValidator:
    """
    Unified feature validation interface for continuous and rule-based features.

    Orchestrates end-to-end validation workflow:
    1. EDA (distributions, correlations, stationarity)
    2. Stage 1: Vector Shuffle Permutation
    3. Stage 2: Pipeline Permutation (feature/candle shuffle)
    4. Stage 3: Walkforward Stability Analysis
    5. Researcher Ensemble Formation (guided by reports)
    """

    def __init__(
        self,
        config: ValidationConfig,
        permutation_engine: Any,  # PermutationEngine from utils
        parameter_analyzer: Any,  # ParameterAnalyzer from eda
        output_dir: Path,
    ):
        """
        Initialize validator with configuration and dependencies.

        Args:
            config: Validation configuration
            permutation_engine: Existing permutation testing engine
            parameter_analyzer: Existing parameter sensitivity analyzer
            output_dir: Base directory for report outputs
        """
        self.config = config
        self.permutation_engine = permutation_engine
        self.parameter_analyzer = parameter_analyzer
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_eda(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
    ) -> EDAReport:
        """
        Run exploratory data analysis.

        Args:
            feature_data: Feature DataFrame (single column for now)
            target: Target series

        Returns:
            EDAReport with all EDA results
        """
        # Extract single feature column (for now, assume single feature)
        if len(feature_data.columns) > 1:
            raise NotImplementedError("Multi-feature EDA not yet implemented")

        feature = feature_data.iloc[:, 0]

        # Common EDA
        feature_stats = compute_distribution_stats(feature)
        target_stats = compute_distribution_stats(target)

        correlations = compute_correlations(feature, target)
        lagged_correlations = compute_lagged_correlations(feature, target, max_lag=10)

        adf_test, kpss_test = run_stationarity_tests(feature)

        # Rolling correlation (use smaller window if insufficient data)
        window = min(252, len(feature) // 4)
        if window >= 20:
            rolling_correlation = compute_rolling_correlation(feature, target, window=window)
        else:
            rolling_correlation = pd.Series([correlations['pearson']])

        # Regime stats (placeholder - could add regime detection later)
        regime_stats = {}

        # Feature-type-specific EDA
        if self.config.feature_type == 'continuous':
            continuous_report = self._run_continuous_eda(feature, target)
            rule_report = None
        else:
            continuous_report = None
            rule_report = self._run_rule_eda(feature, target)

        # Diagnostic flags (basic checks)
        warnings = []
        red_flags = []

        if abs(correlations['pearson']) < 0.05:
            warnings.append("Very low feature-target correlation")

        if not adf_test.is_stationary:
            warnings.append("Feature may be non-stationary")

        return EDAReport(
            feature_stats=feature_stats,
            target_stats=target_stats,
            correlations=correlations,
            lagged_correlations=lagged_correlations,
            adf_test=adf_test,
            kpss_test=kpss_test,
            rolling_correlation=rolling_correlation,
            regime_stats=regime_stats,
            distribution_plot=None,  # Plotting handled separately
            correlation_plot=None,
            time_series_plot=None,
            stationarity_plot=None,
            continuous_report=continuous_report,
            rule_report=rule_report,
            warnings=warnings,
            red_flags=red_flags,
        )

    def _run_continuous_eda(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> ContinuousEDAReport:
        """Run continuous-specific EDA."""
        # Decile analysis
        decile_stats = compute_decile_stats(feature, target, n_deciles=10)
        monotonicity_test = check_monotonicity(decile_stats['mean_target'])

        # Outlier analysis
        outlier_mask, outlier_fraction = detect_outliers(feature, method='iqr')

        # Estimate outlier impact (simplified)
        if outlier_fraction > 0:
            clean_target = target[~outlier_mask]
            clean_sharpe = clean_target.mean() / clean_target.std() if clean_target.std() > 0 else 0
            full_sharpe = target.mean() / target.std() if target.std() > 0 else 0
            outlier_impact = clean_sharpe - full_sharpe
        else:
            outlier_impact = 0.0

        # Non-linearity tests
        polynomial_r2 = fit_polynomial_regression(feature, target, max_degree=3)
        linear_r2 = polynomial_r2[1]

        return ContinuousEDAReport(
            decile_stats=decile_stats,
            decile_plot=None,
            monotonicity_test=monotonicity_test,
            outlier_fraction=outlier_fraction,
            outlier_impact=outlier_impact,
            outlier_plot=None,
            linear_r2=linear_r2,
            polynomial_r2=polynomial_r2,
            non_linearity_plot=None,
        )

    def _run_rule_eda(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> RuleEDAReport:
        """Run rule-based-specific EDA."""
        # Level distribution
        level_counts, level_fractions, imbalance_flag = compute_level_distribution(feature)

        # Per-level statistics
        level_stats = compute_level_stats(feature, target)
        level_confidence_intervals = compute_level_confidence_intervals(
            feature, target, confidence_level=0.95, n_bootstrap=1000
        )

        # Regime transitions
        transition_matrix = compute_transition_matrix(feature)
        average_duration = compute_average_duration(feature)

        return RuleEDAReport(
            level_counts=level_counts,
            level_fractions=level_fractions,
            imbalance_flag=imbalance_flag,
            level_stats=level_stats,
            level_confidence_intervals=level_confidence_intervals,
            level_plot=None,
            transition_matrix=transition_matrix,
            average_duration=average_duration,
            transition_plot=None,
        )

    def run_stage1_permutation(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
    ) -> PermutationReport:
        """
        Run Stage 1: Vector Shuffle Permutation Test.

        Args:
            feature_data: Feature DataFrame (single column for now)
            target: Target series

        Returns:
            PermutationReport with test results
        """
        # Extract single feature column
        if len(feature_data.columns) > 1:
            raise NotImplementedError("Multi-feature permutation not yet implemented")

        feature = feature_data.iloc[:, 0]

        # Delegate to permutation testing function
        return run_vector_shuffle_test(
            feature=feature,
            target=target,
            n_permutations=self.config.n_permutations,
            confidence_level=self.config.confidence_level,
            random_seed=self.config.random_seed,
        )

    def run_full_validation(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
        feature_name: str,
        params_grid: dict[str, list] | None = None,
    ) -> ValidationReport:
        """
        Execute complete validation pipeline: EDA → Stage 1 (→ Stage 2 → Stage 3 when implemented).

        Returns accumulated ValidationReport. Stops early if permutation tests fail.

        Args:
            feature_data: Feature DataFrame
            target: Target series
            feature_name: Name of the feature being validated
            params_grid: Parameter grid for stage 3 stability analysis (not yet implemented)

        Returns:
            ValidationReport with all completed stages
        """
        # Initialize report
        timestamp = datetime.now()

        # Stage 1: EDA
        eda_report = self.run_eda(feature_data=feature_data, target=target)

        # Stage 2: Stage 1 Permutation Test
        stage1_report = self.run_stage1_permutation(feature_data=feature_data, target=target)

        # Determine validation status based on stage 1
        if not stage1_report.passed:
            validation_status = 'failed'
            failure_stage = 'Stage 1: Vector Shuffle'
        else:
            # For now, mark as incomplete since we haven't implemented Stages 2 & 3
            validation_status = 'incomplete'
            failure_stage = None

        return ValidationReport(
            feature_name=feature_name,
            feature_type=self.config.feature_type,
            timestamp=timestamp,
            eda_report=eda_report,
            stage1_report=stage1_report,
            stage2_report=None,  # Not yet implemented
            stage3_report=None,  # Not yet implemented
            parameter_report=None,  # Not yet implemented
            validation_status=validation_status,
            failure_stage=failure_stage,
            researcher_notes="",
            ensemble_decision=None,
        )

    def run_progressive_validation(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
        params_grid: dict[str, list] | None = None,
    ) -> Generator[tuple[str, EDAReport | PermutationReport], None, None]:
        """
        Yield stage-by-stage results for interactive workflow.

        Args:
            feature_data: Feature DataFrame
            target: Target series
            params_grid: Parameter grid for stage 3 stability analysis (not yet implemented)

        Yields:
            (stage_name, stage_report) tuples as each stage completes
        """
        # Stage 1: EDA
        eda_report = self.run_eda(feature_data=feature_data, target=target)
        yield ("EDA", eda_report)

        # Stage 2: Stage 1 Permutation Test
        stage1_report = self.run_stage1_permutation(feature_data=feature_data, target=target)
        yield ("Stage 1: Vector Shuffle", stage1_report)

        # Early exit if Stage 1 failed
        if not stage1_report.passed:
            return

        # Stage 3 (Stage 2 permutation) - not yet implemented
        # Stage 4 (Stage 3 stability) - not yet implemented
