"""
GlobalWeightLayer — Cross-Timeframe Forecast Combination

Combines per-timeframe forecast streams using HRP weights derived from
downside semi-covariance and a cross-TF Forecast Diversification Multiplier.

Pipeline position:
    WeightLayer (per-TF) → GlobalWeightLayer → Portfolio

Usage::

    layer = GlobalWeightLayer()
    layer.fit(tf_forecast_streams, instrument_returns)
    combined = layer.combine(tf_forecast_streams)

Reference: Robert Carver's "Systematic Trading"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass as _dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from utils.core.enums import TimeFrame
from ensemble.weight_layer import (
    _compute_downside_semi_covariance,
    _hrp_weights_from_semi_cov,
    _compute_fdm_from_corr_matrix,
)

logger = logging.getLogger(__name__)

_FORECAST_CLIP = 2.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@_dataclass(frozen=True)
class GlobalWeightLayerConfig:
    """Configuration for GlobalWeightLayer.

    Attributes
    ----------
    shrinkage : str
        ``"ledoit_wolf"`` or ``"none"``.  Passed to the downside semi-cov
        estimator.
    linkage : str
        Linkage method for HRP clustering: ``"ward"``, ``"complete"``,
        ``"single"``.
    fdm_max : float
        Upper cap on the cross-TF FDM (default 2.0).
    resample_method : str
        Strategy used to project each TF stream onto the daily grid.
        Only ``"forward_fill"`` is supported.
    """

    shrinkage: str = "ledoit_wolf"
    linkage: str = "ward"
    fdm_max: float = 2.0
    resample_method: str = "forward_fill"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_daily_grid(
    tf_forecast_streams: Dict[TimeFrame, pd.DataFrame],
) -> pd.DatetimeIndex:
    """Union of all datetime values across all TF streams."""
    all_dates = pd.Index([], dtype="datetime64[ns]")
    for df in tf_forecast_streams.values():
        if "datetime" in df.columns:
            all_dates = all_dates.union(pd.DatetimeIndex(df["datetime"]))
    return pd.DatetimeIndex(sorted(all_dates.unique()))


def _resample_stream_to_grid(
    df: pd.DataFrame,
    daily_grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forward-fill a single TF forecast stream onto *daily_grid*.

    Parameters
    ----------
    df : DataFrame with columns ``['ticker', 'datetime', 'forecast_score']``
    daily_grid : target DatetimeIndex

    Returns
    -------
    DataFrame with columns ``['ticker', 'datetime', 'forecast_score']``
    pivoted then stacked back, with ffill applied per ticker.
    """
    required = {"ticker", "datetime", "forecast_score"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"forecast stream must contain {required}, got {set(df.columns)}"
        )

    # Pivot to wide: index=datetime, columns=ticker
    wide = df.pivot_table(
        index="datetime",
        columns="ticker",
        values="forecast_score",
        aggfunc="last",
    )
    wide.index = pd.DatetimeIndex(wide.index)
    # Reindex to full daily grid, forward-fill, fill remaining NaN with 0.0
    wide = wide.reindex(daily_grid).ffill().fillna(0.0)

    # Melt back to long form
    long = (
        wide.reset_index()
        .rename(columns={"index": "datetime"})
        .melt(id_vars="datetime", var_name="ticker", value_name="forecast_score")
    )
    return long


# ---------------------------------------------------------------------------
# GlobalWeightLayer
# ---------------------------------------------------------------------------

class GlobalWeightLayer:
    """Combine per-timeframe forecast streams into a single daily forecast.

    Parameters
    ----------
    config : GlobalWeightLayerConfig, optional
        If omitted, defaults are used.

    Fitted attributes (set after :meth:`fit`)
    ------------------------------------------
    tf_weights_ : Dict[TimeFrame, float]
    fdm_ : float
    mean_cross_tf_correlation_ : float
    is_fitted_ : bool
    """

    def __init__(self, config: Optional[GlobalWeightLayerConfig] = None) -> None:
        self.config: GlobalWeightLayerConfig = config or GlobalWeightLayerConfig()
        self.tf_weights_: Dict[TimeFrame, float] = {}
        self.fdm_: float = 1.0
        self.mean_cross_tf_correlation_: float = float("nan")
        self._daily_grid: pd.DatetimeIndex = pd.DatetimeIndex([])
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        tf_forecast_streams: Dict[TimeFrame, pd.DataFrame],
        instrument_returns: pd.DataFrame,
    ) -> "GlobalWeightLayer":
        """Fit TF weights from historical forecast streams and returns.

        Parameters
        ----------
        tf_forecast_streams : dict mapping TimeFrame → DataFrame
            Each DataFrame must have columns
            ``['ticker', 'datetime', 'forecast_score']``.
        instrument_returns : pd.DataFrame
            Daily returns; ``index`` = datetime, ``columns`` = tickers.

        Returns
        -------
        self
        """
        tfs = list(tf_forecast_streams.keys())
        K = len(tfs)

        # ----------------------------------------------------------------
        # Step 1 — single-TF fallback
        # ----------------------------------------------------------------
        if K == 1:
            tf = tfs[0]
            self.tf_weights_ = {tf: 1.0}
            self.fdm_ = 1.0
            self.mean_cross_tf_correlation_ = 1.0
            daily_grid = _build_daily_grid(tf_forecast_streams)
            self._daily_grid = daily_grid
            self.is_fitted_ = True
            return self

        # ----------------------------------------------------------------
        # Step 2 — build daily grid
        # ----------------------------------------------------------------
        daily_grid = _build_daily_grid(tf_forecast_streams)
        self._daily_grid = daily_grid

        # ----------------------------------------------------------------
        # Step 3 — resample each stream to daily grid
        # ----------------------------------------------------------------
        resampled: Dict[TimeFrame, pd.DataFrame] = {
            tf: _resample_stream_to_grid(df, daily_grid)
            for tf, df in tf_forecast_streams.items()
        }

        # ----------------------------------------------------------------
        # Step 4 — TF return proxies
        # ----------------------------------------------------------------
        # Align instrument_returns to daily_grid
        instr_ret = instrument_returns.reindex(daily_grid).fillna(0.0)

        tf_return_streams: Dict[TimeFrame, pd.Series] = {}
        for tf, long_df in resampled.items():
            pivot = long_df.pivot_table(
                index="datetime",
                columns="ticker",
                values="forecast_score",
                aggfunc="last",
            )
            pivot.index = pd.DatetimeIndex(pivot.index)
            pivot = pivot.reindex(daily_grid).fillna(0.0)

            # Align columns to shared tickers
            common_tickers = pivot.columns.intersection(instr_ret.columns)
            if common_tickers.empty:
                # No overlap — return stream is zero
                tf_return_streams[tf] = pd.Series(0.0, index=daily_grid)
                continue

            aligned_pivot = pivot[common_tickers]
            aligned_ret = instr_ret[common_tickers]

            proxy: pd.Series = (
                aligned_pivot.multiply(aligned_ret, axis=0)
            ).mean(axis=1)
            proxy.index = pd.DatetimeIndex(proxy.index)
            tf_return_streams[tf] = proxy

        # ----------------------------------------------------------------
        # Step 5 — (T, K) return matrix
        # ----------------------------------------------------------------
        tfs_sorted = sorted(tfs, key=lambda t: t.name)
        return_matrix = pd.DataFrame(
            {tf: tf_return_streams[tf] for tf in tfs_sorted},
            index=daily_grid,
        ).fillna(0.0)

        # ----------------------------------------------------------------
        # Step 6 — downside semi-covariance
        # ----------------------------------------------------------------
        semi_cov = _compute_downside_semi_covariance(
            return_matrix, shrinkage=self.config.shrinkage
        )

        # ----------------------------------------------------------------
        # Fallback: < 2 rows or singular/degenerate semi-cov
        # ----------------------------------------------------------------
        if (
            return_matrix.shape[0] < 2
            or not np.all(np.isfinite(semi_cov))
            or np.allclose(semi_cov, 0.0)
        ):
            logger.warning(
                "GlobalWeightLayer.fit: insufficient data or degenerate "
                "semi-covariance — falling back to equal TF weights."
            )
            equal_w = 1.0 / K
            self.tf_weights_ = {tf: equal_w for tf in tfs}
            self.fdm_ = 1.0
            self.mean_cross_tf_correlation_ = float("nan")
            self.is_fitted_ = True
            return self

        # ----------------------------------------------------------------
        # Step 7 — HRP weights
        # ----------------------------------------------------------------
        raw_weights = _hrp_weights_from_semi_cov(
            semi_cov, linkage_method=self.config.linkage
        )  # length K, sorted by tfs_sorted

        self.tf_weights_ = {
            tf: float(raw_weights[i]) for i, tf in enumerate(tfs_sorted)
        }

        # ----------------------------------------------------------------
        # Step 8 — cross-TF FDM
        # ----------------------------------------------------------------
        diag = np.diag(semi_cov)
        denom = np.sqrt(np.outer(diag, diag))
        denom[denom == 0] = 1.0
        corr_matrix = np.clip(semi_cov / denom, -1.0, 1.0)
        np.fill_diagonal(corr_matrix, 1.0)

        K_mat = corr_matrix.shape[0]
        mask = np.triu(np.ones((K_mat, K_mat), dtype=bool), k=1)
        off_diag = corr_matrix[mask]
        self.mean_cross_tf_correlation_ = float(np.clip(off_diag, 0.0, None).mean())

        self.fdm_ = _compute_fdm_from_corr_matrix(
            corr_matrix, fdm_max=self.config.fdm_max
        )

        # ----------------------------------------------------------------
        # Step 9 — mark fitted
        # ----------------------------------------------------------------
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # combine
    # ------------------------------------------------------------------

    def combine(
        self,
        tf_forecast_streams: Dict[TimeFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Apply fitted TF weights to produce a combined daily forecast.

        Parameters
        ----------
        tf_forecast_streams : dict mapping TimeFrame → DataFrame
            Same schema as in :meth:`fit`.

        Returns
        -------
        pd.DataFrame with columns ``['ticker', 'datetime', 'forecast_score']``
            forecast_score clipped to [-2.0, 2.0].
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "GlobalWeightLayer must be fitted before calling combine()."
            )

        # Build grid from the *input* streams (not the training grid)
        # so that predict-time dates (validation/test) are covered correctly.
        predict_grid = _build_daily_grid(
            {tf: df for tf, df in tf_forecast_streams.items() if tf in self.tf_weights_}
        )
        if len(predict_grid) == 0:
            return pd.DataFrame(columns=["ticker", "datetime", "forecast_score"])

        # Resample each TF stream onto the predict-time grid
        resampled: Dict[TimeFrame, pd.DataFrame] = {
            tf: _resample_stream_to_grid(df, predict_grid)
            for tf, df in tf_forecast_streams.items()
            if tf in self.tf_weights_
        }

        # Collect all tickers
        all_tickers: set = set()
        for long_df in resampled.values():
            all_tickers.update(long_df["ticker"].unique())

        if not all_tickers:
            return pd.DataFrame(columns=["ticker", "datetime", "forecast_score"])

        # Build combined scores
        combined_wide = pd.DataFrame(
            0.0,
            index=predict_grid,
            columns=sorted(all_tickers),
        )

        for tf, long_df in resampled.items():
            w = self.tf_weights_.get(tf, 0.0)
            if w == 0.0:
                continue
            pivot = long_df.pivot_table(
                index="datetime",
                columns="ticker",
                values="forecast_score",
                aggfunc="last",
            )
            pivot.index = pd.DatetimeIndex(pivot.index)
            pivot = pivot.reindex(predict_grid).fillna(0.0)
            # Add weighted contribution (only for tickers present in this TF)
            for ticker in pivot.columns.intersection(combined_wide.columns):
                combined_wide[ticker] += w * pivot[ticker]

        # Apply FDM
        combined_wide = combined_wide * self.fdm_

        # Clip
        combined_wide = combined_wide.clip(-_FORECAST_CLIP, _FORECAST_CLIP)

        # Melt back to long form
        result = (
            combined_wide.reset_index()
            .rename(columns={"index": "datetime"})
            .melt(id_vars="datetime", var_name="ticker", value_name="forecast_score")
        )
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return a snapshot of fitted state for inspection/logging.

        Returns
        -------
        dict with keys:
            ``is_fitted``, ``tf_weights``, ``fdm``,
            ``mean_cross_tf_correlation``, ``daily_grid_len``.
        """
        return {
            "is_fitted": self.is_fitted_,
            "tf_weights": {tf.name: w for tf, w in self.tf_weights_.items()},
            "fdm": self.fdm_,
            "mean_cross_tf_correlation": self.mean_cross_tf_correlation_,
            "daily_grid_len": len(self._daily_grid),
        }
