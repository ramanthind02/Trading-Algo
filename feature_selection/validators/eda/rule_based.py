"""Rule-based feature EDA methods."""
import pandas as pd
import numpy as np
from scipy import stats


def compute_level_distribution(
    feature: pd.Series,
    min_level_fraction: float = 0.10,
) -> tuple[dict[int, int], dict[int, float], bool]:
    """
    Compute level distribution for rule-based features.

    Args:
        feature: Rule-based feature (values should be -1, 0, 1)
        min_level_fraction: Minimum fraction per level (default 10%)

    Returns:
        Tuple of (level_counts, level_fractions, imbalance_flag)
    """
    # Count samples per level
    level_counts = feature.value_counts().to_dict()

    # Compute fractions
    total_samples = len(feature)
    level_fractions = {level: count / total_samples for level, count in level_counts.items()}

    # Check for imbalance
    min_fraction = min(level_fractions.values())
    imbalance_flag = min_fraction < min_level_fraction

    return level_counts, level_fractions, imbalance_flag


def compute_level_stats(
    feature: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Compute per-level target statistics.

    Args:
        feature: Rule-based feature
        target: Target series

    Returns:
        DataFrame with level statistics (mean, std, Sharpe, t-stat, adjusted_Sharpe)
    """
    # Align and drop NaN
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    level_stats = []

    for level in sorted(aligned['feature'].unique()):
        level_data = aligned[aligned['feature'] == level]['target']

        mean_target = level_data.mean()
        std_target = level_data.std()
        n_samples = len(level_data)

        # Sharpe ratio
        sharpe = mean_target / std_target if std_target > 0 else 0.0

        # t-statistic
        t_stat = mean_target / (std_target / np.sqrt(n_samples)) if std_target > 0 else 0.0

        # Adjusted Sharpe (from feature selection spec)
        k = 1.0  # penalty parameter
        adjusted_sharpe = sharpe * np.sqrt(n_samples) / (np.sqrt(n_samples) + k)

        level_stats.append({
            'level': level,
            'mean_target': mean_target,
            'std_target': std_target,
            'sharpe': sharpe,
            't_stat': t_stat,
            'adjusted_sharpe': adjusted_sharpe,
            'n_samples': n_samples,
        })

    return pd.DataFrame(level_stats)


def compute_transition_matrix(feature: pd.Series) -> pd.DataFrame:
    """
    Compute transition matrix for rule-based feature levels.

    P(level_t+1 | level_t)

    Args:
        feature: Rule-based feature

    Returns:
        Transition matrix (DataFrame with rows=from_level, cols=to_level)
    """
    # Create shifted series
    from_level = feature[:-1].reset_index(drop=True)
    to_level = feature[1:].reset_index(drop=True)

    # Count transitions
    transitions = pd.crosstab(from_level, to_level)

    # Normalize to probabilities (row-wise)
    transition_matrix = transitions.div(transitions.sum(axis=1), axis=0)

    # Ensure all levels are present (even if zero transitions)
    for level in [-1, 0, 1]:
        if level not in transition_matrix.index:
            transition_matrix.loc[level] = 0.0
        if level not in transition_matrix.columns:
            transition_matrix[level] = 0.0

    transition_matrix = transition_matrix.sort_index().sort_index(axis=1)

    return transition_matrix


def compute_average_duration(feature: pd.Series) -> dict[int, float]:
    """
    Compute average duration (consecutive periods) in each level.

    Args:
        feature: Rule-based feature

    Returns:
        Dictionary mapping level → average duration
    """
    # Identify regime switches
    switches = feature != feature.shift(1)
    regime_ids = switches.cumsum()

    # Group by regime and compute durations
    durations = feature.groupby(regime_ids).size()
    levels = feature.groupby(regime_ids).first()

    # Compute average duration per level
    average_durations = {}

    for level in feature.unique():
        level_durations = durations[levels == level]
        average_durations[int(level)] = float(level_durations.mean()) if len(level_durations) > 0 else 0.0

    return average_durations


def compute_level_confidence_intervals(
    feature: pd.Series,
    target: pd.Series,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> dict[int, tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for mean target per level.

    Args:
        feature: Rule-based feature
        target: Target series
        confidence_level: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary mapping level → (lower_ci, upper_ci)
    """
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    confidence_intervals = {}
    alpha = 1 - confidence_level

    for level in sorted(aligned['feature'].unique()):
        level_data = aligned[aligned['feature'] == level]['target'].values

        if len(level_data) < 10:
            # Insufficient data for bootstrap
            confidence_intervals[int(level)] = (np.nan, np.nan)
            continue

        # Bootstrap resampling
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(level_data, size=len(level_data), replace=True)
            bootstrap_means.append(sample.mean())

        # Compute confidence interval
        lower_ci = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper_ci = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        confidence_intervals[int(level)] = (float(lower_ci), float(upper_ci))

    return confidence_intervals
