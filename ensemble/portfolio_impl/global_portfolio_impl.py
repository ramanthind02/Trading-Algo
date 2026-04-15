"""Global multi-timeframe portfolio (``GlobalPortfolio`` and snapshot helpers)."""
from __future__ import annotations

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import pandas as pd

from .portfolio_allocation import (
    effective_instrument_weights,
    load_sector_allocation_config,
    resolve_sector_allocation,
)
from .portfolio_postprocessing import (
    aggregate_forecast_vectors_fallback,
    apply_forecast_risk_management,
)
from .portfolio_result_formatting import format_portfolio_result
from .portfolio_returns import calculate_idm_from_returns, calculate_returns_from_candles
from .portfolio_global_streams import (
    align_forecast_vectors_to_daily_grid,
    build_daily_grid,
    build_global_signals_df,
    normalize_global_signals_by_downside_vol,
)
from utils.cache.runtime.central_cache_errors import ArtifactMissingError
from utils.cache.runtime.central_cache_models import ArtifactScope
from utils.core.enums import TimeFrame
from .global_weight_layer_adapter import (
    _GLOBAL_WEIGHT_LAYER_TICKER,
    build_global_adapter_rollups,
    build_global_model_name,
    decode_global_weight_layer_output,
    encode_forecast_vectors_for_global_weight_layer,
)
from .global_portfolio_runtime import (
    apply_global_position_constraints,
    build_global_returns_proxy,
    build_reference_grid_from_daily_candles,
    collect_tf_forecast_streams,
)
from .global_portfolio_diagnostics import collect_global_strategy_health_diagnostics
from .portfolio_cache import (
    PortfolioCacheQuery,
    _query_candles_from_cache,
    _query_volatility_from_cache,
)
from utils.vault_paths import resolve_vault_root
from ensemble.weight_layer import (
    BaseWeightLayer,
    WeightLayer,
    _correlation_multiplier_from_corr_matrix,
)
from .tf_portfolio import TFPortfolio, PortfolioWorld

logger = logging.getLogger(__name__)


class GlobalPortfolio:
    """Global multi-timeframe portfolio combining TFPortfolio streams via WeightLayer.

    This is the top-level portfolio object in the multi-TF pipeline:

        TFPortfolio (per-TF) → WeightLayer → GlobalPortfolio → PositionSizer

    It:
    1. Fits and runs each TFPortfolio to obtain per-TF forecast streams.
    2. Fits / calls WeightLayer to combine them into a single daily forecast.
    3. Applies global instrument weights and a global IDM.
    4. Clips to ``[-max_position_pct, +max_position_pct]``.

    Parameters
    ----------
    tf_portfolios : list of TFPortfolio
        One per trading timeframe.
    weight_layer : BaseWeightLayer, optional
        Cross-TF weight layer. Defaults to ``equal_signal``.
    instrument_weights : dict mapping ticker → weight, optional
        Global instrument weights. If None, equal weight is applied.
    idm_max : float, default 2.5
        Maximum global IDM (Carver's cap).
    max_position_pct : float, default 2.0
        Position-fraction clip bound.

    Fitted attributes
    -----------------
    global_idm_ : float
    mean_instrument_return_correlation_ : float
    instruments_ : list of str
    is_fitted_ : bool
    """

    def __init__(
        self,
        tf_portfolios: List["TFPortfolio"],
        weight_layer: Optional[BaseWeightLayer] = None,
        instrument_weights: Optional[Dict[str, float]] = None,
        idm_max: float = 2.5,
        max_position_pct: float = 2.0,
    ) -> None:
        self.tf_portfolios = tf_portfolios
        self.weight_layer: BaseWeightLayer = (
            weight_layer
            if weight_layer is not None
            else WeightLayer(weight_method="equal_signal", fdm_max=2.0)
        )
        self.idm_max = idm_max
        self.max_position_pct = max_position_pct
        self.instrument_weights = instrument_weights

        # Fitted attributes
        self.global_idm_: Optional[float] = None
        self.mean_instrument_return_correlation_: Optional[float] = None
        self.instruments_: Optional[List[str]] = None
        self.global_tf_weights_compat_: Dict[str, float] = {}
        self.weight_layer_diagnostics_: Dict[str, Any] = {}
        self.global_adapter_diagnostics_: Dict[str, Any] = {}
        self.global_eligible_models_by_ticker_: Dict[str, Set[str]] = {}
        self.global_eligibility_diagnostics_: Dict[str, Any] = {}
        self.is_fitted_: bool = False
        self.portfolio_id_: Optional[str] = None

    def _load_candles_per_timeframe_from_cache(
        self,
        query: PortfolioCacheQuery,
    ) -> Dict[TimeFrame, pd.DataFrame]:
        candles_per_tf: Dict[TimeFrame, pd.DataFrame] = {}
        for timeframe in query.timeframes:
            candles_df = _query_candles_from_cache(query, timeframe)
            if candles_df.empty:
                continue
            candles_per_tf[timeframe] = candles_df
        return candles_per_tf

    def _collect_global_strategy_health_diagnostics(
        self,
        forecast_vectors: List[pd.DataFrame],
        global_returns: Optional[pd.Series],
    ) -> tuple[List[pd.DataFrame], Dict[str, Any]]:
        """Collect low-information diagnostics without dropping any strategy."""
        eligible_vectors, eligible_by_ticker, diagnostics = (
            collect_global_strategy_health_diagnostics(
                forecast_vectors,
                global_returns,
            )
        )
        self.global_eligible_models_by_ticker_ = eligible_by_ticker
        self.global_eligibility_diagnostics_ = diagnostics
        return eligible_vectors, diagnostics

    # ------------------------------------------------------------------
    # Instrument weight helper
    # ------------------------------------------------------------------

    def _get_effective_instrument_weights(self, tickers: List[str]) -> Dict[str, float]:
        """Return global instrument weights for the given tickers (equal-weight fallback)."""
        return effective_instrument_weights(tickers, self.instrument_weights)

    # ------------------------------------------------------------------
    # IDM calculation
    # ------------------------------------------------------------------

    def _calculate_global_idm(self, instrument_returns: pd.DataFrame) -> None:
        """Compute and store global IDM from instrument return correlations.

        Formula: ``IDM = min(sqrt(1 / (mean_corr + 0.01)), idm_max)``
        where ``mean_corr`` is the mean of off-diagonal return correlations,
        floored at 0.

        Parameters
        ----------
        instrument_returns : pd.DataFrame
            Daily returns; columns = tickers.
        """
        self.instruments_ = list(instrument_returns.columns)

        mean_corr, idm = calculate_idm_from_returns(
            instrument_returns,
            idm_max=self.idm_max,
        )
        self.mean_instrument_return_correlation_ = mean_corr
        self.global_idm_ = idm

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        candles_per_tf: Dict[TimeFrame, pd.DataFrame],
        instrument_returns: pd.DataFrame,
        daily_volatility_df: pd.DataFrame,
    ) -> "GlobalPortfolio":
        """Fit all TFPortfolios, WeightLayer, and global IDM.

        Parameters
        ----------
        candles_per_tf : dict mapping TimeFrame → DataFrame
            Candles for each TF's portfolio.  Each DataFrame must contain at
            least the columns expected by ``TFPortfolio.fit_from_candles``.
        instrument_returns : pd.DataFrame
            Daily instrument returns; ``columns`` = tickers.
        daily_volatility_df : pd.DataFrame
            Daily EWSD volatility DataFrame with columns:
            ['datetime', 'ticker', 'ewsd_annual_vol'].

        Returns
        -------
        self
        """
        if daily_volatility_df is None:
            raise ValueError(
                "daily_volatility_df is required for GlobalPortfolio.fit(). "
                "Expected columns: ['datetime', 'ticker', 'ewsd_annual_vol']."
            )

        # Step 1 — fit each TFPortfolio
        for tf_p in self.tf_portfolios:
            tf_candles = candles_per_tf.get(tf_p.trading_timeframe)
            if tf_candles is None:
                raise ValueError(
                    f"No candles provided for timeframe {tf_p.trading_timeframe.name}"
                )
            tf_p.fit_from_candles(tf_candles)

        # Step 2 — collect per-TF forecast streams
        tf_forecast_streams = collect_tf_forecast_streams(
            self.tf_portfolios,
            candles_per_tf,
            daily_volatility_df,
        )

        forecast_vectors = list(tf_forecast_streams.values())
        if not forecast_vectors:
            raise ValueError("No forecast vectors available for global strategy weighting")

        # Use aggregate daily return proxy for eligibility diagnostics and
        # downside-vol normalization.
        global_returns = build_global_returns_proxy(instrument_returns)

        # Align all strategy streams to a shared daily grid first.
        daily_grid = build_daily_grid(
            forecast_vectors=forecast_vectors,
            reference_index=global_returns.index if global_returns is not None else None,
        )
        forecast_vectors = align_forecast_vectors_to_daily_grid(
            forecast_vectors=forecast_vectors,
            daily_grid=daily_grid,
        )

        # Step 3 — diagnostics + downside-vol normalization (fit inputs only)
        eligible_vectors, _elig_diag = self._collect_global_strategy_health_diagnostics(
            forecast_vectors=forecast_vectors,
            global_returns=global_returns,
        )
        normalized_vectors = normalize_global_signals_by_downside_vol(
            forecast_vectors=eligible_vectors,
            global_returns=global_returns,
        )
        encoded_fit_vectors, stream_decode_map = (
            encode_forecast_vectors_for_global_weight_layer(normalized_vectors)
        )
        signals_df = build_global_signals_df(encoded_fit_vectors)
        if signals_df.empty:
            raise ValueError("Global strategy signals are empty after normalization")

        # Step 4 — fit WeightLayer (optional returns kept for API compatibility)
        self.weight_layer.fit(
            forecast_vectors=encoded_fit_vectors,
            signals=signals_df,
            returns=global_returns,
        )
        self.weight_layer_diagnostics_ = self.weight_layer.get_diagnostics()
        self.global_adapter_diagnostics_ = build_global_adapter_rollups(
            stream_decode_map=stream_decode_map,
            global_weights=self.weight_layer.weights_.get(_GLOBAL_WEIGHT_LAYER_TICKER),
        )
        self.global_tf_weights_compat_ = dict(
            self.global_adapter_diagnostics_.get("timeframe_rollups", {})
        )

        # Step 5 — compute global IDM
        self._calculate_global_idm(instrument_returns)

        # Step 6 — mark fitted
        self.is_fitted_ = True
        return self

    def fit_from_cache(
        self,
        query: PortfolioCacheQuery,
        instrument_returns: pd.DataFrame,
    ) -> "GlobalPortfolio":
        """Fit using central-cache-backed candle and volatility reads."""
        candles_per_tf = self._load_candles_per_timeframe_from_cache(query)
        if not candles_per_tf:
            raise ValueError("No candles available in the central cache for the requested query")
        self.fit(
            candles_per_tf=candles_per_tf,
            instrument_returns=instrument_returns,
            daily_volatility_df=_query_volatility_from_cache(query),
        )
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        candles_per_tf: Dict[TimeFrame, pd.DataFrame],
        daily_volatility_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate global position fractions for all instruments.

        Parameters
        ----------
        candles_per_tf : dict mapping TimeFrame → DataFrame
            Current candles for each TF.
        daily_volatility_df : pd.DataFrame
            Daily EWSD volatility DataFrame with columns:
            ['datetime', 'ticker', 'ewsd_annual_vol'].

        Returns
        -------
        pd.DataFrame with columns
            ``['ticker', 'datetime', 'forecast_score', 'position_fraction']``
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "GlobalPortfolio must be fitted before calling predict(). "
                "Call fit() first."
            )
        if daily_volatility_df is None:
            raise ValueError(
                "daily_volatility_df is required for GlobalPortfolio.predict(). "
                "Expected columns: ['datetime', 'ticker', 'ewsd_annual_vol']."
            )

        # Step 1 — collect per-TF forecast streams (pre-IDM, instrument-weighted)
        tf_forecast_streams = collect_tf_forecast_streams(
            self.tf_portfolios,
            candles_per_tf,
            daily_volatility_df,
        )

        forecast_vectors = list(tf_forecast_streams.values())
        if not forecast_vectors:
            return pd.DataFrame(
                columns=['ticker', 'datetime', 'forecast_score', 'position_fraction']
            )

        reference_grid = build_reference_grid_from_daily_candles(candles_per_tf)
        daily_grid = build_daily_grid(
            forecast_vectors=forecast_vectors,
            reference_index=reference_grid,
        )
        forecast_vectors = align_forecast_vectors_to_daily_grid(
            forecast_vectors=forecast_vectors,
            daily_grid=daily_grid,
        )

        encoded_predict_vectors, stream_decode_map = (
            encode_forecast_vectors_for_global_weight_layer(forecast_vectors)
        )
        if not encoded_predict_vectors:
            return pd.DataFrame(
                columns=['ticker', 'datetime', 'forecast_score', 'position_fraction']
            )

        # Step 2 — combine via WeightLayer + adapter decode.
        # Keep the raw synthetic combine call for diagnostics parity.
        _ = self.weight_layer.combine(encoded_predict_vectors)
        combined = decode_global_weight_layer_output(
            encoded_vectors=encoded_predict_vectors,
            stream_decode_map=stream_decode_map,
            global_weights=self.weight_layer.weights_.get(_GLOBAL_WEIGHT_LAYER_TICKER),
            global_fdm=float(self.weight_layer.fdm_.get(_GLOBAL_WEIGHT_LAYER_TICKER, 1.0)),
        )

        if combined.empty:
            return pd.DataFrame(
                columns=['ticker', 'datetime', 'forecast_score', 'position_fraction']
            )

        # Step 3 — apply global instrument weights, IDM, and cap
        return apply_global_position_constraints(
            combined,
            instrument_weights=self.instrument_weights,
            global_idm=self.global_idm_,
            max_position_pct=self.max_position_pct,
        )

    def predict_from_cache(
        self,
        query: PortfolioCacheQuery,
        return_ensemble_predictions: bool = False,
        return_base_model_predictions: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Predict using central cache reads for candles and volatility."""
        candles_per_tf = self._load_candles_per_timeframe_from_cache(query)
        if not candles_per_tf:
            raise ValueError("No candles available in the central cache for the requested query")
        result = self.predict(
            candles_per_tf=candles_per_tf,
            daily_volatility_df=_query_volatility_from_cache(query),
        )
        if return_ensemble_predictions or return_base_model_predictions:
            payload: Dict[str, Any] = {"portfolio": result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)}
            if return_ensemble_predictions:
                payload["ensembles"] = {}
            if return_base_model_predictions:
                payload["base_models"] = {}
            return payload
        return result

    def save_to_vault(
        self,
        fit_start: datetime,
        fit_end: datetime,
        vault_root: str = "vault",
    ) -> str:
        """Persist an immutable portfolio snapshot and return its ``portfolio_id``."""
        from .portfolio_vault import save_global_portfolio_snapshot

        return save_global_portfolio_snapshot(
            self,
            fit_start=fit_start,
            fit_end=fit_end,
            vault_root=vault_root,
        )

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return a snapshot of fitted state for inspection / logging."""
        return {
            "is_fitted": self.is_fitted_,
            "global_signal_eligibility": self.global_eligibility_diagnostics_,
            "global_idm": self.global_idm_,
            "mean_instrument_return_correlation": self.mean_instrument_return_correlation_,
            "instruments": self.instruments_,
            "idm_max": self.idm_max,
            "max_position_pct": self.max_position_pct,
            "n_tf_portfolios": len(self.tf_portfolios),
            "weight_layer": {
                "is_fitted": bool(
                    self.weight_layer_diagnostics_.get("is_fitted", False)
                ),
                "tf_weights": self.global_tf_weights_compat_,
                "fdm": float(
                    self.weight_layer_diagnostics_
                    .get("summary", {})
                    .get("mean_fdm", 1.0)
                ),
                "daily_grid_len": int(
                    len(
                        self.global_adapter_diagnostics_.get(
                            "stream_decode_map",
                            {},
                        )
                    )
                ),
                "diagnostics": self.weight_layer_diagnostics_,
                "adapter_diagnostics": self.global_adapter_diagnostics_,
            },
        }

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        n = len(self.tf_portfolios)
        return f"GlobalPortfolio({fitted_str}, n_tf_portfolios={n}, idm={self.global_idm_})"


# ``Portfolio`` remains a backward-compatible alias for TFPortfolio so that
# existing single-timeframe call sites continue to work without changes.
# GlobalPortfolio is the top-level multi-TF class and is exported separately
# from ensemble/__init__.py.
Portfolio = TFPortfolio


def load_global_portfolio_snapshot(
    portfolio_id: str,
    vault_root: str = "vault",
) -> GlobalPortfolio:
    """Load a previously snapshotted GlobalPortfolio by ``portfolio_id``."""
    from .portfolio_vault import load_global_portfolio_snapshot as _load_global_portfolio_snapshot

    return _load_global_portfolio_snapshot(portfolio_id=portfolio_id, vault_root=vault_root)


def materialize_global_portfolio_predictions(
    portfolio: GlobalPortfolio,
    query: PortfolioCacheQuery,
    portfolio_id: str,
    world: PortfolioWorld | str,
    research_run_id: Optional[str] = None,
    scope: ArtifactScope = ArtifactScope.LIVE,
):
    """Materialize portfolio/base-model predictions into the dedicated cache tree."""
    from utils.cache.runtime.portfolio_materialization import (
        materialize_global_portfolio_predictions as _materialize_global_portfolio_predictions,
    )

    return _materialize_global_portfolio_predictions(
        portfolio=portfolio,
        query=query,
        portfolio_id=portfolio_id,
        world=world,
        research_run_id=research_run_id,
        scope=scope,
    )


def prune_inactive_base_model_materializations(
    vault_root: str = "vault",
    scope: ArtifactScope = ArtifactScope.LIVE,
    ensemble_dirs: Optional[Iterable[str]] = None,
):
    """Delete base-model materializations whose identities are no longer active in the live vault."""
    from utils.cache.runtime.portfolio_materialization import (
        prune_inactive_base_model_materializations as _prune_inactive_base_model_materializations,
    )

    return _prune_inactive_base_model_materializations(
        vault_root=vault_root,
        scope=scope,
        ensemble_dirs=ensemble_dirs,
    )
