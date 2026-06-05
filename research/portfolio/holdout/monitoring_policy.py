"""Pure helpers for holdout monitoring windows and traffic-light policy."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd
from quantfoundry_core.robustness.validation import ValidationRobustnessReport


class ReferenceSigmaMethod(str, Enum):
    """How reference return volatility was estimated for monitoring tests."""

    VALIDATION_ONLY = "validation_only"
    POOLED_TRAIN_VALIDATION = "pooled_train_validation"
    WEIGHTED_TOWARD_VALIDATION = "weighted_toward_validation"


class MonitoringTrafficLight(Enum):
    """Aggregate monitoring state from failed robustness test count."""

    GREEN = auto()
    YELLOW = auto()
    RED = auto()


@dataclass(frozen=True)
class MonitoringWindow:
    """Inclusive date bounds for a returns slice."""

    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class ReferenceCalibration:
    """Split reference parameters: mu from validation, sigma from train+validation policy."""

    mu: float
    sigma: float
    sigma_method: ReferenceSigmaMethod
    sigma_train: float
    sigma_validation: float
    sigma_relative_shift: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "sigma_method": self.sigma_method.value,
            "sigma_train": self.sigma_train,
            "sigma_validation": self.sigma_validation,
            "sigma_relative_shift": self.sigma_relative_shift,
        }


@dataclass(frozen=True)
class StrategyReferenceSlices:
    """Return slices used for holdout monitoring on one strategy."""

    validation_returns: pd.Series
    train_returns: pd.Series
    full_holdout_returns: pd.Series
    validation_window: MonitoringWindow
    train_window: MonitoringWindow
    reference_volatility_returns: pd.Series
    calibration: ReferenceCalibration


@dataclass(frozen=True)
class MonitoringStatus:
    """Traffic-light summary for one evaluation snapshot."""

    evaluation_window: MonitoringWindow
    reference_window: MonitoringWindow
    tests_failed: int
    traffic_light: MonitoringTrafficLight
    advisory_weight_fraction: float
    override_weight_fraction: float | None
    effective_weight_fraction: float
    sharpe_passed: bool
    cusum_passed: bool
    rolling_passed: bool
    equity_bands_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "evaluation_window": {
                "start": self.evaluation_window.start.date().isoformat(),
                "end": self.evaluation_window.end.date().isoformat(),
            },
            "reference_window": {
                "start": self.reference_window.start.date().isoformat(),
                "end": self.reference_window.end.date().isoformat(),
            },
            "tests_failed": self.tests_failed,
            "traffic_light": self.traffic_light.name,
            "advisory_weight_fraction": self.advisory_weight_fraction,
            "override_weight_fraction": self.override_weight_fraction,
            "effective_weight_fraction": self.effective_weight_fraction,
            "sharpe_passed": self.sharpe_passed,
            "cusum_passed": self.cusum_passed,
            "rolling_passed": self.rolling_passed,
            "equity_bands_passed": self.equity_bands_passed,
        }
        return payload

    def to_dict_with_calibration(
        self,
        calibration: ReferenceCalibration,
        *,
        reference_volatility_window: MonitoringWindow,
    ) -> dict[str, object]:
        payload = self.to_dict()
        payload["reference_calibration"] = calibration.to_dict()
        payload["reference_volatility_window"] = {
            "start": reference_volatility_window.start.date().isoformat(),
            "end": reference_volatility_window.end.date().isoformat(),
        }
        return payload


def _period_std(returns: pd.Series) -> float:
    values = returns.dropna().to_numpy(dtype=float)
    if values.shape[0] < 2:
        return 0.0
    std = float(values.std(ddof=1))
    return 1e-12 if not math.isfinite(std) or std <= 0.0 else std


def resolve_reference_sigma(
    train_returns: pd.Series,
    validation_returns: pd.Series,
    *,
    regime_shift_threshold: float,
    recent_weight_on_shift: float,
) -> tuple[float, ReferenceSigmaMethod, float, float, float]:
    """Estimate reference sigma; weight toward validation when train/val vol diverge."""

    sigma_train = _period_std(train_returns)
    sigma_validation = _period_std(validation_returns)
    denominator = max(sigma_validation, 1e-12)
    relative_shift = abs(sigma_train - sigma_validation) / denominator

    if relative_shift > regime_shift_threshold:
        weight = min(max(recent_weight_on_shift, 0.0), 1.0)
        sigma = weight * sigma_validation + (1.0 - weight) * sigma_train
        return (
            sigma,
            ReferenceSigmaMethod.WEIGHTED_TOWARD_VALIDATION,
            sigma_train,
            sigma_validation,
            relative_shift,
        )

    pooled = pd.concat([train_returns, validation_returns]).sort_index()
    return (
        _period_std(pooled),
        ReferenceSigmaMethod.POOLED_TRAIN_VALIDATION,
        sigma_train,
        sigma_validation,
        relative_shift,
    )


def build_reference_calibration(
    train_returns: pd.Series,
    validation_returns: pd.Series,
    *,
    regime_shift_threshold: float,
    recent_weight_on_shift: float,
    split_calibration: bool,
) -> ReferenceCalibration:
    """Build mu from validation and sigma per portfolio monitoring policy."""

    val_clean = validation_returns.dropna().sort_index()
    mu = float(val_clean.mean()) if val_clean.shape[0] >= 1 else 0.0
    if not split_calibration:
        sigma = _period_std(val_clean)
        return ReferenceCalibration(
            mu=mu,
            sigma=sigma,
            sigma_method=ReferenceSigmaMethod.VALIDATION_ONLY,
            sigma_train=_period_std(train_returns),
            sigma_validation=sigma,
            sigma_relative_shift=0.0,
        )

    sigma, method, sigma_train, sigma_validation, relative_shift = resolve_reference_sigma(
        train_returns,
        validation_returns,
        regime_shift_threshold=regime_shift_threshold,
        recent_weight_on_shift=recent_weight_on_shift,
    )
    return ReferenceCalibration(
        mu=mu,
        sigma=sigma,
        sigma_method=method,
        sigma_train=sigma_train,
        sigma_validation=sigma_validation,
        sigma_relative_shift=relative_shift,
    )


def slice_trailing_window(
    series: pd.Series,
    *,
    end: pd.Timestamp,
    months: int,
) -> pd.Series:
    """Return returns on ``(end - months, end]`` using calendar-month offset."""

    end_ts = pd.Timestamp(end).normalize()
    start_ts = end_ts - pd.DateOffset(months=int(months))
    clipped = series.sort_index()
    return clipped.loc[(clipped.index >= start_ts) & (clipped.index <= end_ts)]


def month_end_dates(
    index: pd.DatetimeIndex,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Last index timestamp in each calendar month between start and end."""

    clipped = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    if clipped.empty:
        return []
    monthly_last = pd.Series(1, index=clipped).resample("ME").last().dropna()
    return [pd.Timestamp(ts) for ts in monthly_last.index]


def count_failed_tests(report: ValidationRobustnessReport) -> int:
    """Count how many of the four robustness tests failed."""

    flags = (
        report.sharpe_comparison.passed,
        report.cusum.passed,
        report.rolling_sharpe_zscore.passed,
        report.equity_curve_bands.passed,
    )
    return sum(1 for passed in flags if not passed)


def traffic_light_from_fail_count(n_failed: int) -> MonitoringTrafficLight:
    """Map failed-test count to Green / Yellow / Red."""

    if n_failed <= 1:
        return MonitoringTrafficLight.GREEN
    if n_failed == 2:
        return MonitoringTrafficLight.YELLOW
    return MonitoringTrafficLight.RED


def advisory_weight_fraction(light: MonitoringTrafficLight) -> float:
    """Recommended position multiplier for a traffic-light state."""

    match light:
        case MonitoringTrafficLight.GREEN:
            return 1.0
        case MonitoringTrafficLight.YELLOW:
            return 0.5
        case MonitoringTrafficLight.RED:
            return 0.0


def effective_weight(
    advisory: float,
    override: float | None,
) -> float:
    """Researcher override wins when set; otherwise use advisory weight."""

    return advisory if override is None else override


def build_monitoring_status(
    report: ValidationRobustnessReport,
    *,
    evaluation_window: MonitoringWindow,
    reference_window: MonitoringWindow,
    override_weight_fraction: float | None,
) -> MonitoringStatus:
    """Derive traffic-light status from a robustness report."""

    tests_failed = count_failed_tests(report)
    light = traffic_light_from_fail_count(tests_failed)
    advisory = advisory_weight_fraction(light)
    return MonitoringStatus(
        evaluation_window=evaluation_window,
        reference_window=reference_window,
        tests_failed=tests_failed,
        traffic_light=light,
        advisory_weight_fraction=advisory,
        override_weight_fraction=override_weight_fraction,
        effective_weight_fraction=effective_weight(advisory, override_weight_fraction),
        sharpe_passed=report.sharpe_comparison.passed,
        cusum_passed=report.cusum.passed,
        rolling_passed=report.rolling_sharpe_zscore.passed,
        equity_bands_passed=report.equity_curve_bands.passed,
    )


def history_row_from_status(
    status: MonitoringStatus,
    *,
    as_of_date: pd.Timestamp,
) -> dict[str, object]:
    """One CSV row for monthly monitoring history."""

    return {
        "as_of_date": pd.Timestamp(as_of_date).date().isoformat(),
        "eval_start": status.evaluation_window.start.date().isoformat(),
        "eval_end": status.evaluation_window.end.date().isoformat(),
        "sharpe_passed": status.sharpe_passed,
        "cusum_passed": status.cusum_passed,
        "rolling_passed": status.rolling_passed,
        "equity_bands_passed": status.equity_bands_passed,
        "tests_failed": status.tests_failed,
        "traffic_light": status.traffic_light.name,
        "advisory_weight_fraction": status.advisory_weight_fraction,
    }


def trend_traffic_lights(
    history: pd.DataFrame,
    *,
    current_as_of: str,
) -> tuple[str | None, str | None]:
    """Return traffic lights for the prior two month-ends before current."""

    if history.empty or "as_of_date" not in history.columns:
        return None, None
    prior = history.loc[history["as_of_date"] < current_as_of].sort_values("as_of_date")
    if prior.empty:
        return None, None
    lights = [str(value) for value in prior["traffic_light"].tolist()]
    prev = lights[-1] if len(lights) >= 1 else None
    two_ago = lights[-2] if len(lights) >= 2 else None
    return prev, two_ago
