"""Permutation testing integration."""
import time
import pandas as pd
import numpy as np

from feature_selection.validators.reports.permutation import PermutationReport


def _compute_sharpe(returns: np.ndarray) -> float:
    """Compute Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std())


def _compute_t_stat(returns: np.ndarray) -> float:
    """Compute t-statistic."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / (returns.std() / np.sqrt(len(returns))))


def run_vector_shuffle_test(
    feature: pd.Series,
    target: pd.Series,
    n_permutations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> PermutationReport:
    """
    Run Stage 1: Vector Shuffle Permutation Test.

    Shuffles feature vector, recomputes Sharpe, tests significance.

    Args:
        feature: Feature series
        target: Target series
        n_permutations: Number of permutations
        confidence_level: Confidence level (e.g., 0.95)
        random_seed: Random seed for reproducibility

    Returns:
        PermutationReport
    """
    start_time = time.time()

    # Align data
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    # Compute observed statistics
    # For vector shuffle, we're testing if feature values predict target
    # Simple approach: bin feature into quantiles, compute Sharpe of target in top quantile
    q_high = aligned['feature'].quantile(0.75)
    high_feature_mask = aligned['feature'] >= q_high
    selected_returns = aligned.loc[high_feature_mask, 'target'].values

    observed_sharpe = _compute_sharpe(selected_returns)
    observed_t_stat = _compute_t_stat(selected_returns)
    observed_returns_mean = float(selected_returns.mean())

    # Run permutations
    permuted_sharpes = []
    permuted_t_stats = []

    rng = np.random.RandomState(random_seed)

    for _ in range(n_permutations):
        # Shuffle feature
        shuffled_feature = aligned['feature'].values.copy()
        rng.shuffle(shuffled_feature)

        # Recompute statistics
        q_high_perm = np.quantile(shuffled_feature, 0.75)
        high_mask_perm = shuffled_feature >= q_high_perm
        selected_returns_perm = aligned['target'].values[high_mask_perm]

        permuted_sharpes.append(_compute_sharpe(selected_returns_perm))
        permuted_t_stats.append(_compute_t_stat(selected_returns_perm))

    permuted_sharpes_array = np.array(permuted_sharpes)
    permuted_t_stats_array = np.array(permuted_t_stats)

    # Compute p-value (one-sided: observed > permuted)
    p_value = float((permuted_sharpes_array >= observed_sharpe).sum() / n_permutations)

    # Compute critical value
    critical_value = float(np.percentile(permuted_sharpes_array, confidence_level * 100))

    # Verdict
    passed = bool(observed_sharpe > critical_value)
    margin = float(observed_sharpe - critical_value)

    execution_time = time.time() - start_time

    return PermutationReport(
        stage='stage1_vector_shuffle',
        observed_sharpe=observed_sharpe,
        observed_t_stat=observed_t_stat,
        observed_returns_mean=observed_returns_mean,
        permuted_sharpes=permuted_sharpes_array,
        permuted_t_stats=permuted_t_stats_array,
        p_value=p_value,
        confidence_level=confidence_level,
        critical_value=critical_value,
        passed=passed,
        margin=margin,
        permutation_histogram=None,  # Plotting handled separately
        qq_plot=None,
        n_permutations=n_permutations,
        random_seed=random_seed,
        execution_time=execution_time,
    )
