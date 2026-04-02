"""Data-only feature exploration helpers.

This module keeps the exploratory analysis and export contracts that are still
useful in research workflows, while retiring in-repo chart generation in favor
of downstream BI tooling.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import utils.core.helpers as helpers
from utils.core.enums import DirectionInput, coerce_direction
from utils.evaluation.permutation_test.permutation_engine import (
    FeaturePermutationStrategy,
    PermutationEngine,
)


_PLOTTING_REMOVED_MESSAGE = (
    "FeatureExplorer plotting was removed for the Power BI migration. "
    "Use the tabular exports from generate_summary_report() or "
    "generate_parameter_sensitivity_report() instead."
)


@dataclass(frozen=True)
class SignalCumsumSummary:
    feature: str
    final_cumsum: float
    n_samples: int
    metric: float | None


class FeatureExplorer:
    """Explore extracted features without generating Python-side plots."""

    @staticmethod
    def _permutation_criterion(
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        base_model: Any,
        metric: Any,
        verbose: bool,
    ) -> float:
        X = df[feature_col]
        y = df[target_col]

        valid_mask = ~(X.isna() | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        if len(X_clean) < 10:
            return 0.0

        try:
            base_model.fit(X_clean, y_clean)
            signals = base_model.predict(X_clean, strategy="long")
            selected_returns = y_clean[signals == 1]
            if len(selected_returns) < 5:
                return 0.0
            return float(metric.compute(selected_returns))
        except Exception as exc:  # pragma: no cover - defensive logging path
            if verbose:
                print(f"  [WARNING] Failed to compute criterion for '{feature_col}': {exc}")
            return 0.0

    def __init__(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not isinstance(features_df, pd.DataFrame):
            raise TypeError("features_df must be a pandas DataFrame")
        if not isinstance(targets_df, pd.DataFrame):
            raise TypeError("targets_df must be a pandas DataFrame")
        if not features_df.index.equals(targets_df.index):
            raise ValueError(
                "features_df and targets_df must have matching indices. "
                "Use FeatureExtractor to ensure proper alignment."
            )

        self.features_df = features_df
        self.targets_df = targets_df
        self.metadata = metadata or {}
        self.results: dict[str, Any] = {}
        self.feature_metadata = self.metadata.get("feature_metadata", {})

        def is_normalization_column(col: str) -> bool:
            col_lower = col.lower()
            if col_lower.startswith("atr_") or col_lower.startswith("ewsd_"):
                return True
            if col_lower.startswith("atr") and ("_252" in col_lower or "_atr" in col_lower):
                return True
            if col_lower.startswith("ewsd") and ("_252" in col_lower or "_ewsd" in col_lower):
                return True
            return False

        self.feature_names = [
            col
            for col in features_df.columns
            if col != "ticker" and not is_normalization_column(col)
        ]
        self.n_features = len(self.feature_names)
        self.n_samples = len(features_df)
        self.date_range = (features_df.index.min(), features_df.index.max())
        self.has_ticker = "ticker" in features_df.columns
        self.tickers = features_df["ticker"].unique().tolist() if self.has_ticker else None
        self.n_tickers = len(self.tickers) if self.tickers is not None else 1
        self._param_mapping: dict[str, dict[str, Any]] = {}
        self._feature_groups: dict[tuple[str, str], list[tuple[Any, str]]] = {}
        self._initialize_parameter_mapping()

    def _initialize_parameter_mapping(self) -> None:
        mapping: dict[str, dict[str, Any]] = {}

        for feature_name in self.feature_names:
            meta = self.feature_metadata.get(feature_name)
            if meta is not None:
                mapping[feature_name] = {
                    "module": meta.get("module") or meta.get("base_name"),
                    "timeframe": (meta.get("timeframes") or [None])[0],
                    "parameters": dict(meta.get("parameters", {})),
                }
                continue

            parsed = helpers.parse_feature_column_name(feature_name)
            mapping[feature_name] = {
                "module": parsed.get("module"),
                "timeframe": parsed.get("tf"),
                "parameters": dict(parsed.get("params", {})),
            }

        self._param_mapping = mapping
        feature_groups: dict[tuple[str, str], list[tuple[Any, str]]] = {}
        for feature_name, info in mapping.items():
            module = str(info.get("module") or "")
            timeframe = str(info.get("timeframe") or "")
            params = info.get("parameters", {})
            if not module or not params:
                continue
            feature_groups.setdefault((module, timeframe), [])
            feature_groups[(module, timeframe)].append((params, feature_name))
        self._feature_groups = feature_groups

    def get_parameterized_features(self) -> dict[tuple[str, str], list[tuple[Any, str]]]:
        return copy.deepcopy(self._feature_groups)

    def _has_parameterized_features(self) -> bool:
        return any(info.get("parameters") for info in self._param_mapping.values())

    def _count_parameters_per_module(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for info in self._param_mapping.values():
            module = str(info.get("module") or "")
            if not module:
                continue
            counts[module] = max(counts.get(module, 0), len(info.get("parameters", {})))
        return counts

    @staticmethod
    def _canonicalize_param_name(name: str) -> str:
        return "".join(ch.lower() for ch in str(name) if ch.isalnum())

    def _extract_parameter_grid(
        self,
        module_name: str,
        param_names: list[str],
    ) -> dict[tuple[Any, ...], list[str]]:
        canonical_params = [self._canonicalize_param_name(name) for name in param_names]
        grouped: dict[tuple[Any, ...], list[str]] = {}

        for feature_name in self.feature_names:
            info = self._param_mapping.get(feature_name, {})
            module = str(info.get("module") or "")
            if module != module_name:
                continue

            parameters = info.get("parameters", {})
            canonical_lookup = {
                self._canonicalize_param_name(param_name): value
                for param_name, value in parameters.items()
            }

            if not all(name in canonical_lookup for name in canonical_params):
                continue

            key = tuple(canonical_lookup[name] for name in canonical_params)
            grouped.setdefault(key, [])
            grouped[key].append(feature_name)

        return grouped

    def plot_parameter_sensitivity(
        self,
        module_name: str,
        param_name: str,
        target_col: str = "log_return",
        metric: Any | None = None,
        title: str | None = None,
        show_plot: bool = False,
    ) -> tuple[pd.DataFrame, None]:
        from eda.parameter_analysis import ParameterAnalyzer

        del title, show_plot
        analyzer = ParameterAnalyzer(self.features_df, self.targets_df)
        grid = self._extract_parameter_grid(module_name, [param_name])
        results_df = analyzer.analyze_nd_parameters(
            feature_grid=grid,
            param_names=[param_name],
            target_col=target_col,
            metric=metric,
        )
        return analyzer.plot_parameter_sensitivity(results_df, param_name, metric="sortino")

    def plot_2d_parameter_surface(
        self,
        module_name: str,
        param1: str,
        param2: str,
        target_col: str = "log_return",
        metric: Any | None = None,
        title: str | None = None,
        show_plot: bool = False,
        plot_type: str = "surface",
    ) -> tuple[pd.DataFrame, None]:
        from eda.parameter_analysis import ParameterAnalyzer

        del title, show_plot
        analyzer = ParameterAnalyzer(self.features_df, self.targets_df)
        grid = self._extract_parameter_grid(module_name, [param1, param2])
        results_df = analyzer.analyze_nd_parameters(
            feature_grid=grid,
            param_names=[param1, param2],
            target_col=target_col,
            metric=metric,
        )
        return analyzer.plot_2d_parameter_surface(
            results_df,
            param1,
            param2,
            metric="sortino",
            plot_type=plot_type,
        )

    def plot_nd_parameter_analysis(
        self,
        module_name: str,
        param_names: list[str],
        target_col: str = "log_return",
        metric: Any | None = None,
        show_plot: bool = False,
        plot_3d_mode: str = "surface_slices",
    ) -> tuple[pd.DataFrame, None]:
        from eda.parameter_analysis import ParameterAnalyzer

        del show_plot
        if plot_3d_mode not in {"surface_slices", "heatmap_slices"}:
            raise ValueError(f"Unsupported plot_3d_mode: {plot_3d_mode}")

        analyzer = ParameterAnalyzer(self.features_df, self.targets_df)
        grid = self._extract_parameter_grid(module_name, param_names)
        results_df = analyzer.analyze_nd_parameters(
            feature_grid=grid,
            param_names=param_names,
            target_col=target_col,
            metric=metric,
        )
        if len(param_names) == 1:
            return analyzer.plot_parameter_sensitivity(results_df, param_names[0], metric="sortino")
        if len(param_names) == 2:
            return analyzer.plot_2d_parameter_surface(
                results_df,
                param_names[0],
                param_names[1],
                metric="sortino",
            )
        return results_df, None

    def _format_parameter_sensitivity_report(
        self,
        module_name: str,
        param_names: list[str],
        results_df: pd.DataFrame,
        robustness: dict[str, Any],
    ) -> str:
        lines = [
            "=" * 70,
            "PARAMETER SENSITIVITY REPORT",
            "=" * 70,
            f"Module: {module_name}",
            f"Parameters: {', '.join(param_names)}",
            f"Configurations: {len(results_df)}",
            "",
            "ROBUSTNESS",
            "-" * 70,
            f"Overall Score: {robustness['overall_score']:.2f}",
            f"Rating: {robustness['rating']}",
            f"Variance Score: {robustness['variance_score']:.2f}",
            f"Consistency Score: {robustness['consistency_score']:.2f}",
            f"Risk Score: {robustness['risk_score']:.2f}",
            "",
            "PARAMETER SENSITIVITY",
            "-" * 70,
        ]

        sensitivity = robustness.get("parameter_sensitivity", {})
        if sensitivity:
            for param_name, weight in sensitivity.items():
                lines.append(f"{param_name}: {weight:.4f}")
        else:
            lines.append("No sensitivity decomposition available.")

        return "\n".join(lines)

    def generate_parameter_sensitivity_report(
        self,
        module_name: str,
        param_names: list[str],
        target_col: str = "log_return",
        metric: Any | None = None,
        export_path: str | None = None,
        verbose: bool = True,
        plot_3d_mode: str = "surface_slices",
    ) -> dict[str, Any]:
        from eda.parameter_analysis import ParameterAnalyzer, _get_metric_name_from_object

        analyzer = ParameterAnalyzer(self.features_df, self.targets_df)
        results_df, fig = self.plot_nd_parameter_analysis(
            module_name=module_name,
            param_names=param_names,
            target_col=target_col,
            metric=metric,
            show_plot=False,
            plot_3d_mode=plot_3d_mode,
        )
        metric_name = _get_metric_name_from_object(metric)
        robustness = analyzer.compute_robustness_metrics(results_df, metric_name)
        report_text = self._format_parameter_sensitivity_report(
            module_name=module_name,
            param_names=param_names,
            results_df=results_df,
            robustness=robustness,
        )

        report_path = None
        if export_path is not None:
            report_path = str(export_path)
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text(report_text, encoding="utf-8")
            if verbose:
                print(f"  [OK] Exported parameter sensitivity report: {report_path}")

        return {
            "results_df": results_df,
            "summary_stats": robustness["statistics"],
            "robustness_scores": {
                "variance_score": robustness["variance_score"],
                "consistency_score": robustness["consistency_score"],
                "risk_score": robustness["risk_score"],
                "overall_score": robustness["overall_score"],
                "rating": robustness["rating"],
            },
            "figures": [fig],
            "parameter_sensitivity": robustness.get("parameter_sensitivity", {}),
            "report_text": report_text,
            "report_path": report_path,
        }

    def _plotting_removed(self) -> None:
        raise RuntimeError(_PLOTTING_REMOVED_MESSAGE)

    def plot_all_deciles(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        self._plotting_removed()

    def plot_2bin(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._plotting_removed()

    def plot_deciles(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._plotting_removed()

    def plot_uniform_bins(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._plotting_removed()

    def plot_all_uniform_bins(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        self._plotting_removed()

    def get_summary(self) -> pd.DataFrame:
        summaries = [
            {
                "feature_name": feature_name,
                "n_samples": len(self.features_df[feature_name]),
                "n_nan": int(self.features_df[feature_name].isna().sum()),
                "mean": self.features_df[feature_name].mean(),
                "std": self.features_df[feature_name].std(),
                "min": self.features_df[feature_name].min(),
                "max": self.features_df[feature_name].max(),
                "dtype": str(self.features_df[feature_name].dtype),
            }
            for feature_name in self.feature_names
        ]
        return pd.DataFrame(summaries)

    def get_correlations(
        self,
        target_col: str = "log_return",
        method: str = "spearman",
    ) -> pd.Series:
        if target_col not in self.targets_df.columns:
            raise ValueError(f"Target '{target_col}' not found")
        target_data = self.targets_df[target_col]
        correlations = {}
        for feature_name in self.feature_names:
            feature_data = self.features_df[feature_name]
            if not pd.api.types.is_numeric_dtype(feature_data):
                correlations[feature_name] = np.nan
                continue
            correlations[feature_name] = feature_data.corr(target_data, method=method)
        return pd.Series(correlations, name=f"{method}_correlation")

    def _compute_signal_cumsum_summary(
        self,
        target_col: str,
        binning_model: Any | None,
        metric: Any | None,
        strategy: DirectionInput,
    ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        feature_summaries: list[SignalCumsumSummary] = []
        gated_returns: dict[str, pd.Series] = {}
        direction = coerce_direction(strategy, field_name="strategy").value

        for feature_name in self.feature_names:
            feature_series = self.features_df[feature_name]
            target_series = self.targets_df[target_col]
            valid_mask = ~(feature_series.isna() | target_series.isna())
            feature_clean = feature_series[valid_mask]
            target_clean = target_series[valid_mask]
            if len(feature_clean) < 5:
                continue

            if binning_model is None:
                signals = feature_clean
            else:
                try:
                    model = binning_model.clone() if hasattr(binning_model, "clone") else None
                except Exception:
                    model = None
                model = model or copy.deepcopy(binning_model)
                model.fit(feature_clean, target_clean)
                signals = model.predict(feature_clean, strategy=direction)

            signal_series = pd.Series(signals, index=feature_clean.index)
            gated = target_clean * signal_series
            gated_returns[feature_name] = gated
            metric_value = None if metric is None else float(metric.compute(gated))
            feature_summaries.append(
                SignalCumsumSummary(
                    feature=feature_name,
                    final_cumsum=float(gated.cumsum().iloc[-1]),
                    n_samples=int(len(gated)),
                    metric=metric_value,
                )
            )

        summary_df = pd.DataFrame(
            [
                {
                    "feature": item.feature,
                    "final_cumsum": item.final_cumsum,
                    "n_samples": item.n_samples,
                    "metric": item.metric,
                }
                for item in feature_summaries
            ]
        )
        if not summary_df.empty:
            summary_df = summary_df.sort_values("final_cumsum", ascending=False).reset_index(drop=True)
        return summary_df, gated_returns

    def _export_summary_report(
        self,
        results: dict[str, Any],
        export_dir: Path,
    ) -> None:
        export_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(results.get("summary_stats"), pd.DataFrame):
            results["summary_stats"].to_csv(export_dir / "summary_stats.csv", index=False)
        if isinstance(results.get("correlations"), pd.Series):
            results["correlations"].rename("correlation").to_csv(export_dir / "feature_target_correlations.csv")
        if isinstance(results.get("feature_correlations"), pd.DataFrame):
            results["feature_correlations"].to_csv(export_dir / "feature_correlations.csv")
        if isinstance(results.get("signal_cumsum_summary"), pd.DataFrame):
            results["signal_cumsum_summary"].to_csv(export_dir / "signal_cumsum_summary.csv", index=False)
        if isinstance(results.get("permutation_test"), pd.DataFrame):
            results["permutation_test"].to_csv(export_dir / "permutation_test.csv", index=False)

        parameter_sensitivity = results.get("parameter_sensitivity") or {}
        if parameter_sensitivity:
            param_dir = export_dir / "parameter_sensitivity"
            param_dir.mkdir(parents=True, exist_ok=True)
            for name, payload in parameter_sensitivity.items():
                payload["results_df"].to_csv(param_dir / f"{name}_grid.csv", index=False)
                (param_dir / f"{name}_report.txt").write_text(payload["report_text"], encoding="utf-8")

        summary_lines = [
            "=" * 70,
            "Feature Analysis Summary Report",
            "=" * 70,
            f"Date Range: {self.date_range[0].date()} to {self.date_range[1].date()}",
            f"Number of Features: {self.n_features}",
            f"Number of Samples: {self.n_samples}",
            "",
            "Plotting has been removed from FeatureExplorer. Use the exported CSV and TXT",
            "artifacts in this directory to build Power BI visuals.",
        ]
        (export_dir / "summary_report.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    def generate_summary_report(
        self,
        target_col: str = "log_return",
        binning_model: Any | None = None,
        metric: Any | None = None,
        strategy: DirectionInput = "long",
        show_plots: bool = False,
        verbose: bool = True,
        export_report: bool = False,
        export_path: str | None = None,
        permutation_test_nreps: int = 100,
        include_ticker_plots: bool = False,
    ) -> dict[str, Any]:
        del show_plots, include_ticker_plots
        if target_col not in self.targets_df.columns:
            raise ValueError(f"Target '{target_col}' not found")

        summary_stats = self.get_summary()
        correlations = self.get_correlations(target_col=target_col, method="spearman")
        numeric_features = [
            feature_name
            for feature_name in self.feature_names
            if pd.api.types.is_numeric_dtype(self.features_df[feature_name])
        ]
        feature_correlations = (
            self.features_df[numeric_features].corr(method="pearson")
            if len(numeric_features) >= 2
            else None
        )
        signal_cumsum_summary, signal_gated_returns = self._compute_signal_cumsum_summary(
            target_col=target_col,
            binning_model=binning_model,
            metric=metric,
            strategy=strategy,
        )

        parameter_sensitivity: dict[str, Any] = {}
        if self._has_parameterized_features():
            module_param_counts = self._count_parameters_per_module()
            for module_name, param_count in module_param_counts.items():
                if param_count <= 0:
                    continue
                sample_feature = next(
                    (
                        feature_name
                        for feature_name, info in self._param_mapping.items()
                        if info.get("module") == module_name and info.get("parameters")
                    ),
                    None,
                )
                if sample_feature is None:
                    continue
                raw_param_names = list(self._param_mapping[sample_feature]["parameters"].keys())
                selected_param_names = raw_param_names[: min(2, len(raw_param_names))]
                if not selected_param_names:
                    continue
                report = self.generate_parameter_sensitivity_report(
                    module_name=module_name,
                    param_names=selected_param_names,
                    target_col=target_col,
                    metric=metric,
                    export_path=None,
                    verbose=False,
                )
                parameter_sensitivity[f"{module_name}_{'_'.join(selected_param_names)}"] = report

        permutation_results = None
        if binning_model is not None and metric is not None:
            permutation_results = self.run_permutation_test(
                target_col=target_col,
                base_model=binning_model,
                metric=metric,
                nreps=permutation_test_nreps,
                verbose=verbose,
            )

        results = {
            "summary_stats": summary_stats,
            "correlations": correlations,
            "feature_correlations": feature_correlations,
            "feature_correlations_figure": None,
            "decile_figures": {},
            "signal_cumsum_figures": {},
            "signal_cumsum_summary": signal_cumsum_summary,
            "signal_cumsum_gated_returns": signal_gated_returns,
            "parameter_sensitivity": parameter_sensitivity,
            "parameter_2d_surface": {},
            "distribution_figures": {},
            "timeseries_figures": {},
            "decile_figures_by_ticker": {} if self.has_ticker else None,
            "signal_cumsum_figures_by_ticker": {} if self.has_ticker else None,
            "signal_cumsum_summary_by_ticker": None,
            "permutation_test": permutation_results,
        }

        if export_report:
            export_dir = Path(export_path) if export_path is not None else Path.cwd() / "feature_explorer_report"
            self._export_summary_report(results, export_dir)
            if verbose:
                print(f"  [OK] Exported data-only feature summary: {export_dir}")

        self.results["summary_report"] = results
        return results

    def run_permutation_test(
        self,
        target_col: str = "log_return",
        base_model: Any | None = None,
        metric: Any | None = None,
        nreps: int = 100,
        n_jobs: int = -1,
        random_seed: int | None = 42,
        alpha: float = 0.1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        if target_col not in self.targets_df.columns:
            raise ValueError(f"Target '{target_col}' not found. Available targets: {list(self.targets_df.columns)}")
        if base_model is None:
            raise ValueError("You must provide a base_model instance (fit/predict methods) for permutation test.")
        if metric is None:
            raise ValueError("You must provide a metric object (with .compute()) for permutation test.")

        from functools import partial

        data = self.features_df.copy()
        data[target_col] = self.targets_df[target_col]
        criterion_func = partial(
            self._permutation_criterion,
            target_col=target_col,
            base_model=base_model,
            metric=metric,
            verbose=verbose,
        )
        strategy = FeaturePermutationStrategy()
        engine = PermutationEngine(strategy, n_jobs=n_jobs, verbose=verbose)
        results = engine.run_permutation_test(
            data=data,
            feature_cols=self.feature_names,
            criterion_func=criterion_func,
            nreps=nreps,
            random_seed=random_seed,
            alpha=alpha,
            target_col=target_col,
        )
        self.results["permutation_test"] = {
            "target_col": target_col,
            "model": str(base_model),
            "metric": str(metric),
            "nreps": nreps,
            "alpha": alpha,
            "results": results,
        }
        return results

    def get_feature(self, feature_name: str) -> tuple[pd.Series, pd.DataFrame]:
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found. Available: {self.feature_names}")
        return self.features_df[feature_name], self.targets_df

    def sensitivity_analysis(
        self,
        target_col: str = "log_return",
        base_model: Any | None = None,
        metric: Any | None = None,
        nreps: int = 100,
        n_jobs: int = -1,
        random_seed: int | None = 42,
        alpha: float = 0.1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        return self.run_permutation_test(
            target_col=target_col,
            base_model=base_model,
            metric=metric,
            nreps=nreps,
            n_jobs=n_jobs,
            random_seed=random_seed,
            alpha=alpha,
            verbose=verbose,
        )

    def __str__(self) -> str:
        lines = [f"FeatureExplorer with {self.n_features} features:"]
        for feature_name in self.feature_names[:5]:
            lines.append(f"  - {feature_name}")
        if self.n_features > 5:
            lines.append(f"  ... and {self.n_features - 5} more")
        lines.append(f"\nSamples: {self.n_samples}")
        lines.append(f"Date range: {self.date_range[0].date()} to {self.date_range[1].date()}")
        if self.has_ticker:
            lines.append(f"Tickers: {self.tickers}")
        return "\n".join(lines)

    def __getitem__(self, feature_name: str) -> tuple[pd.Series, pd.DataFrame]:
        return self.get_feature(feature_name)

    def __iter__(self):
        return iter(self.feature_names)

    def __len__(self) -> int:
        return self.n_features
