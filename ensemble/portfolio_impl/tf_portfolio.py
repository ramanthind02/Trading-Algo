"""
Portfolio Class for Position Sizing and Instrument Allocation

This module provides timeframe and global portfolio layers.

- ``TFPortfolio`` fits ensembles and produces per-timeframe forecast streams.
- ``GlobalPortfolio`` combines those streams with ``WeightLayer``, then applies
  instrument weighting, IDM, and optional position capping.

Key responsibilities:
1. Build timeframe-level forecasts from ensembles
2. Combine those forecasts globally
3. Apply instrument weights, IDM, and optional position capping

Reference: Robert Carver's "Systematic Trading" and "Leveraged Trading"
"""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from enum import Enum
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
from lib.cache.runtime.central_cache_errors import ArtifactMissingError
from lib.cache.runtime.central_cache_models import ArtifactScope
from lib.core.enums import TimeFrame
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
from ensemble.ensemble_utils import normalize_candles_datetime_column
from .portfolio_cache import (
    PortfolioCacheQuery,
    _query_candles_from_cache,
    _query_volatility_from_cache,
)
from lib.core.vault_paths import resolve_vault_root
from ensemble.weight_layer import (
    BaseWeightLayer,
    WeightLayer,
    _correlation_multiplier_from_corr_matrix,
)

logger = logging.getLogger(__name__)


def _vault_ensemble_label(ensemble: object, idx: int) -> str:
    """Human-readable label for errors (vault folder name when available)."""
    name = getattr(ensemble, "vault_ensemble_name", None)
    if isinstance(name, str) and name.strip():
        return f"{name} (index {idx})"
    return f"index {idx}"


def ensemble_prediction_dict_key(ensemble: object, ensemble_idx: int) -> str:
    """Stable key for per-ensemble / per-base-model prediction dicts.

    Prefer the vault ensemble directory leaf name (``vault_ensemble_name``) so
    exports (e.g. tearsheets) match folder names under ``vault/``. Fall back to
    ``ensemble_{idx}`` when the attribute is missing (non-vault ensembles).
    """
    name = getattr(ensemble, "vault_ensemble_name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"ensemble_{ensemble_idx}"


def _normalize_candles_datetime_column(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime is only a column; delegate to shared helper."""
    return normalize_candles_datetime_column(candles_df)


def _forecast_to_activity_signal(forecast: pd.Series) -> pd.Series:
    """Convert signed forecast magnitudes to binary activity flags.

    Any non-zero forecast means the model is active (long or short).
    """
    return forecast.ne(0.0).astype(int)


class PortfolioWorld(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    LIVE = "live"


class TFPortfolio:
    """
    Timeframe portfolio for fitting ensembles and producing forecast streams.

    ``TFPortfolio`` does not own the cross-timeframe ``WeightLayer``. That
    combiner now sits one level up inside ``GlobalPortfolio``.

    ``TFPortfolio`` applies:
    1. Instrument weights (default: equal weight per instrument)
    2. Instrument Diversification Multiplier (IDM)
    3. Optional position capping

    Parameters
    ----------
    ensembles : List[DiversifiedEnsemble], optional
        List of ensembles for this portfolio (new API)
    trading_timeframe : TimeFrame, default=TimeFrame.D
        The timeframe this portfolio trades on
    target_volatility : float, optional
        Target volatility for position sizing (new API)
    max_position_pct : float, default=2.0
        Maximum position size per instrument (e.g., 2.0 = 200%)
    instrument_weights : Dict[str, float], optional
        Custom weights per instrument. Ignored when sector_allocation_config_path
        is provided.
    sector_allocation_config_path : str, optional
        JSON file path for a nested sector allocation tree. When provided, this
        config is resolved to ticker-level instrument_weights and takes precedence
        over instrument_weights.
    idm_max : float, default=2.5
        Maximum IDM value (Carver's recommendation)

    Attributes
    ----------
    ensembles : List[DiversifiedEnsemble]
        List of ensembles for this portfolio
    trading_timeframe : TimeFrame
        The trading timeframe
    max_position_pct : float or None
        Maximum position size
    instrument_weights : Dict[str, float] or None
        Custom instrument weights
    sector_allocation_config_path : str or None
        Optional path to sector allocation config used to resolve instrument weights
    sector_allocation_config_ : Dict[str, Any] or None
        Loaded sector allocation config object when sector allocation is enabled
    idm_max : float
        Maximum IDM value
    idm_ : float
        Fitted Instrument Diversification Multiplier
    mean_return_correlation_ : float
        Mean correlation between instrument returns (for diagnostics)
    instruments_ : List[str]
        List of instruments seen during fit
    is_fitted_ : bool
        Whether the Portfolio has been fitted

    Examples
    --------
    >>> portfolio = TFPortfolio(
    ...     ensembles=[ensemble1, ensemble2],
    ...     trading_timeframe=TimeFrame.D,
    ...     max_position_pct=2.0
    ... )
    >>> portfolio.fit_from_candles(candles_df, target_data)
    >>> positions = portfolio.predict_from_candles(
    ...     test_candles,
    ...     daily_volatility_df=daily_volatility_df,
    ... )
    >>>
    """
    
    def __init__(
        self,
        ensembles: Optional[List] = None,
        ensemble_names: Optional[List[str]] = None,
        vault_root: str = "vault",
        trading_timeframe: TimeFrame = TimeFrame.D,
        target_volatility: Optional[float] = None,
        max_position_pct: float = 2.0,
        instrument_weights: Optional[Dict[str, float]] = None,
        idm_max: float = 2.5,
        dm: Optional[float] = None,
        use_cache: bool = True,
        sector_allocation_config_path: Optional[str] = None
    ):
        """
        Initialize Portfolio.

        Parameters
        ----------
        ensembles : List[DiversifiedEnsemble], optional
            List of ensembles for this portfolio. When None, ensembles are loaded
            from vault via `_load_ensembles_from_vault`.
        ensemble_names : List[str], optional
            Optional filter on vault ensemble directory names (full names like
            `mean-reversion_indices_long`). Used only when `ensembles is None`.
        vault_root : str, default='vault'
            Root vault directory used for auto-loading when `ensembles is None`.
        trading_timeframe : TimeFrame, default=TimeFrame.D
            The timeframe this portfolio trades on
        target_volatility : float, optional
            Target volatility for position sizing (new API)
        max_position_pct : float, default=2.0
            Maximum position size per instrument (e.g., 2.0 = 200%)
        instrument_weights : Dict[str, float], optional
            Custom weights per instrument. If None, equal weight.
            Ignored when sector_allocation_config_path is provided.
        sector_allocation_config_path : str, optional
            Path to JSON sector allocation tree. If provided, resolved weights
            override instrument_weights.
        idm_max : float, default=2.5
            Maximum IDM value (Carver's recommendation)
        dm : float, optional
            Backward-compatible alias for `idm_max`.
        """
        legacy_dm = dm
        if dm is not None:
            idm_max = dm

        # New API: explicit ensembles bypasses vault auto-load (including empty list).
        if ensembles is None:
            self.ensembles = self._load_ensembles_from_vault(vault_root, ensemble_names)
        else:
            self.ensembles = ensembles
        self.trading_timeframe = trading_timeframe
        self.target_volatility = target_volatility
        self.use_cache = use_cache
        
        # Common attributes
        self.max_position_pct = max_position_pct
        self.sector_allocation_config_path = sector_allocation_config_path
        self.sector_allocation_config_: Optional[Dict[str, Any]] = None
        if self.sector_allocation_config_path is not None:
            self.sector_allocation_config_ = load_sector_allocation_config(
                self.sector_allocation_config_path
            )
            self.instrument_weights = resolve_sector_allocation(
                self.sector_allocation_config_
            )
        else:
            self.instrument_weights = instrument_weights
        self.idm_max = idm_max

        # Fitted attributes
        self.idm_: Optional[float] = None
        self.mean_return_correlation_: Optional[float] = None
        self.instruments_: Optional[List[str]] = None
        self.is_fitted_: bool = False

    def _load_ensembles_from_vault(
        self,
        vault_root: str,
        names: Optional[List[str]] = None,
    ) -> List[Any]:
        """Auto-load ensemble directories from vault/{D,W,M}."""
        from ensemble.vault_manager import load_ensemble_from_vault

        vault_path = resolve_vault_root(vault_root)
        if not vault_path.exists():
            logger.warning("Vault root not found for portfolio auto-load: %s", vault_root)
            return []

        name_filter = set(names or [])
        loaded: List[Any] = []
        for tf_name in ["D", "W", "M"]:
            tf_dir = vault_path / tf_name
            if not tf_dir.exists() or not tf_dir.is_dir():
                continue
            for ensemble_dir in sorted(tf_dir.iterdir()):
                if not ensemble_dir.is_dir():
                    continue
                if name_filter and ensemble_dir.name not in name_filter:
                    continue
                try:
                    loaded.append(load_ensemble_from_vault(str(ensemble_dir)))
                except Exception as exc:
                    logger.warning(
                        "Skipping vault ensemble '%s' during auto-load: %s",
                        ensemble_dir,
                        exc,
                    )
        return loaded

    def _get_effective_instrument_weights(self, tickers: List[str]) -> Dict[str, float]:
        """Return instrument weights for the provided tickers, including fallback handling."""
        return effective_instrument_weights(tickers, self.instrument_weights)

    def fit(
        self,
        instrument_returns: pd.DataFrame,
        idm_override: Optional[float] = None
    ) -> 'TFPortfolio':
        """
        Fit IDM from historical instrument returns.

        The IDM (Instrument Diversification Multiplier) accounts for portfolio-level
        diversification benefits. Lower correlation between instruments = higher IDM.

        Formula:
            IDM = sqrt(1 / (mean_corr + epsilon))
            Capped at idm_max (typically 2.5)

        Parameters
        ----------
        instrument_returns : pd.DataFrame
            Historical returns for all instruments.
            Columns: instrument tickers, rows: time periods
        idm_override : float, optional
            If provided, use this IDM value instead of calculating from returns.
            Useful for testing or when using pre-calculated IDM.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If instrument_returns is empty or has insufficient data
        """
        if idm_override is not None:
            self.idm_ = min(idm_override, self.idm_max)
            self.mean_return_correlation_ = None
            self.instruments_ = list(instrument_returns.columns) if not instrument_returns.empty else []
            self.is_fitted_ = True
            return self

        if instrument_returns.empty:
            raise ValueError("instrument_returns DataFrame cannot be empty")

        if len(instrument_returns) < 2:
            raise ValueError("Need at least 2 time periods to calculate IDM")

        self.instruments_ = list(instrument_returns.columns)

        # Calculate IDM from return correlations
        self.idm_ = self._calculate_idm(instrument_returns)
        self.is_fitted_ = True

        return self

    def _calculate_idm(self, instrument_returns: pd.DataFrame) -> float:
        """
        Calculate Instrument Diversification Multiplier from instrument return correlations.
        
        Formula: IDM = sqrt(1 / (mean_correlation + epsilon))
        Where mean_correlation is calculated from the correlation matrix of instrument returns.
        
        Steps:
        1. Build correlation matrix of instrument returns (tickers as columns)
        2. Calculate mean absolute correlation: mean(|rho_ij|) for i != j
        3. Floor negative correlations at zero (Carver's recommendation)
        4. Calculate IDM: sqrt(1 / (mean_correlation + epsilon))
        5. Cap at idm_max (default 2.5)
        
        Parameters
        ----------
        instrument_returns : pd.DataFrame
            Historical returns for all instruments.
            Columns: instrument tickers, rows: time periods

        Returns
        -------
        float
            IDM value (capped at idm_max)
        """
        mean_correlation, idm = calculate_idm_from_returns(
            instrument_returns,
            idm_max=self.idm_max,
        )
        self.mean_return_correlation_ = mean_correlation
        return idm

    def predict(
        self,
        combined_forecasts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply instrument weighting, IDM, and optional capping to combined forecasts.

        Parameters
        ----------
        combined_forecasts : pd.DataFrame
            Combined forecasts from WeightLayer.
            Required columns: ['ticker', 'forecast_score']
            forecast_score should already be FDM-scaled
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - ticker: Instrument identifier
            - forecast_score: Original forecast from WeightLayer (passed through)
            - position_fraction: Position fraction after instrument weighting, IDM, and capping

        Raises
        ------
        ValueError
            If Portfolio has not been fitted or input is invalid
        """
        if not self.is_fitted_:
            raise ValueError(
                "Portfolio must be fitted before calling predict(). "
                "Call fit() with instrument returns first."
            )

        # Validate input
        if not isinstance(combined_forecasts, pd.DataFrame):
            raise ValueError(
                f"combined_forecasts must be pd.DataFrame, got {type(combined_forecasts)}"
            )

        required_cols = ['ticker', 'forecast_score']
        missing_cols = set(required_cols) - set(combined_forecasts.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if combined_forecasts.empty:
            return pd.DataFrame(columns=['ticker', 'forecast_score', 'position_fraction'])

        # Step 1: Apply instrument weights
        weighted = self._apply_instrument_weights(combined_forecasts)

        # Step 2: Apply IDM
        idm_scaled = self._apply_idm(weighted)

        # Step 3: Apply position cap (optional)
        result = self._apply_position_cap(idm_scaled)

        return result

    def predict_raw(
        self,
        combined_forecasts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply instrument weights to combined forecasts WITHOUT IDM and WITHOUT position cap.

        This is the pre-IDM counterpart of ``predict()``.  It is useful for
        multi-timeframe aggregation layers that want to combine TF-level signals
        before applying a global IDM.

        Parameters
        ----------
        combined_forecasts : pd.DataFrame
            Combined forecasts from WeightLayer.
            Required columns: ['ticker', 'forecast_score']
            forecast_score should already be FDM-scaled.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - ticker: Instrument identifier
            - forecast_score: Original forecast from WeightLayer (passed through)
            - position_weighted: forecast_score * instrument_weight (no IDM, no cap)

        Raises
        ------
        ValueError
            If Portfolio has not been fitted or input is invalid
        """
        if not self.is_fitted_:
            raise ValueError(
                "Portfolio must be fitted before calling predict_raw(). "
                "Call fit() with instrument returns first."
            )

        if not isinstance(combined_forecasts, pd.DataFrame):
            raise ValueError(
                f"combined_forecasts must be pd.DataFrame, got {type(combined_forecasts)}"
            )

        required_cols = ['ticker', 'forecast_score']
        missing_cols = set(required_cols) - set(combined_forecasts.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if combined_forecasts.empty:
            return pd.DataFrame(columns=['ticker', 'forecast_score', 'position_weighted'])

        # Apply instrument weights only — skip IDM and position cap
        weighted = self._apply_instrument_weights(combined_forecasts)

        return weighted[['ticker', 'forecast_score', 'position_weighted']]

    def predict_from_candles_raw(
        self,
        candles_df: pd.DataFrame,
        daily_volatility_df: pd.DataFrame,
        return_ensemble_predictions: bool = False,
        return_base_model_predictions: bool = False,
        start_date=None,
        end_date=None
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Generate pre-IDM position fractions using candles DataFrame.

        Mirrors ``predict_from_candles`` but calls ``predict_raw`` instead of
        ``predict`` so the returned ``position_weighted`` column reflects
        instrument-weighted forecasts WITHOUT IDM scaling or position capping.

        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
            Should contain candles for the trading_timeframe of this portfolio
        daily_volatility_df : pd.DataFrame
            Daily EWSD volatility DataFrame with columns:
            ['datetime', 'ticker', 'ewsd_annual_vol'].
        return_ensemble_predictions : bool, default=False
            If True, return ensemble-level predictions in result dict
        return_base_model_predictions : bool, default=False
            If True, return base model-level predictions in result dict
        start_date : datetime, optional
            Start date for cached data. If None, inferred from candles_df.
        end_date : datetime, optional
            End date for cached data. If None, inferred from candles_df.

        Returns
        -------
        pd.DataFrame or Dict[str, Any]
            If both flags are False: DataFrame with columns
            ['ticker', 'datetime', 'forecast_score', 'position_weighted']
            If either flag is True: Dict with structure:
            {
                'portfolio': pd.DataFrame,
                'ensembles': Dict[str, pd.DataFrame],  # only if requested
                'base_models': Dict[str, pd.DataFrame]  # only if requested
            }
        """
        if not self.ensembles:
            raise ValueError("No ensembles provided. Cannot predict without ensembles.")

        # Filter candles for this portfolio's trading timeframe
        tf_candles = candles_df[candles_df['timeframe'] == self.trading_timeframe].copy()

        if tf_candles.empty:
            empty_df = pd.DataFrame(
                columns=['ticker', 'datetime', 'forecast_score', 'position_weighted']
            )
            if return_ensemble_predictions or return_base_model_predictions:
                result: Dict[str, Any] = {'portfolio': empty_df}
                if return_ensemble_predictions:
                    result['ensembles'] = {}
                if return_base_model_predictions:
                    result['base_models'] = {}
                return result
            return empty_df

        # Delegate to predict_from_candles to collect the intra-TF pipeline result
        full_result = self.predict_from_candles(
            candles_df,
            daily_volatility_df=daily_volatility_df,
            return_ensemble_predictions=True,
            return_base_model_predictions=return_base_model_predictions,
            start_date=start_date,
            end_date=end_date,
        )

        # full_result is always a dict here because we requested ensemble predictions
        portfolio_df: pd.DataFrame = full_result['portfolio'] if isinstance(full_result, dict) else full_result

        # Re-derive pre-IDM positions from the combined forecast_score column
        # portfolio_df has columns: ticker, datetime, forecast_score, position_fraction
        if not portfolio_df.empty and 'forecast_score' in portfolio_df.columns:
            weights_by_ticker = self._get_effective_instrument_weights(
                portfolio_df['ticker'].tolist()
            )
            raw_df = portfolio_df[['ticker', 'datetime', 'forecast_score']].copy()
            raw_df['position_weighted'] = (
                raw_df['forecast_score'] * raw_df['ticker'].map(weights_by_ticker)
            )
        else:
            raw_df = pd.DataFrame(
                columns=['ticker', 'datetime', 'forecast_score', 'position_weighted']
            )

        if return_ensemble_predictions or return_base_model_predictions:
            out: Dict[str, Any] = {'portfolio': raw_df}
            if return_ensemble_predictions:
                out['ensembles'] = full_result.get('ensembles', {}) if isinstance(full_result, dict) else {}
            if return_base_model_predictions:
                out['base_models'] = full_result.get('base_models', {}) if isinstance(full_result, dict) else {}
            return out

        return raw_df

    def predict_base_model_vectors_from_candles(
        self,
        candles_df: pd.DataFrame,
        daily_volatility_df: pd.DataFrame,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """Emit raw base-model forecast vectors for global strategy-level weighting.

        Returns a DataFrame with columns:
        ``['ticker', 'datetime', 'model_name', 'forecast', 'signal', 'timeframe']``.
        ``signal`` is the signed, volatility-scaled forecast stream. This allows
        global weighting methods that use ``signal * returns`` to operate on
        scaled signals rather than binary activity flags.
        """
        if not self.ensembles:
            raise ValueError("No ensembles provided. Cannot predict without ensembles.")
        if daily_volatility_df is None:
            raise ValueError(
                "daily_volatility_df is required for predict_base_model_vectors_from_candles(). "
                "Expected columns: ['datetime', 'ticker', 'ewsd_annual_vol']."
            )

        tf_candles = candles_df[candles_df["timeframe"] == self.trading_timeframe].copy()
        if tf_candles.empty:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "datetime",
                    "model_name",
                    "forecast",
                    "signal",
                    "timeframe",
                ]
            )

        forecast_vectors: List[pd.DataFrame] = []

        for ensemble_idx, ensemble in enumerate(self.ensembles):
            if not hasattr(ensemble, "predict_from_candles"):
                raise TypeError(
                    f"Global base-model vector extraction: ensemble {_vault_ensemble_label(ensemble, ensemble_idx)} "
                    "has no predict_from_candles() (not a DiversifiedEnsemble?)."
                )

            try:
                ensemble_result = ensemble.predict_from_candles(
                    tf_candles,
                    daily_volatility_df=daily_volatility_df,
                    return_base_model_predictions=True,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Global base-model vector extraction failed for ensemble "
                    f"{_vault_ensemble_label(ensemble, ensemble_idx)} "
                    "(each vault ensemble must predict successfully for global WeightLayer fitting)."
                ) from exc

            if not isinstance(ensemble_result, dict):
                raise TypeError(
                    f"Global base-model vector extraction: ensemble {_vault_ensemble_label(ensemble, ensemble_idx)} "
                    f"returned {type(ensemble_result).__name__}, expected dict when "
                    "return_base_model_predictions=True."
                )

            base_models = ensemble_result.get("base_models", {})
            for model_name, model_pred in base_models.items():
                if (
                    not isinstance(model_pred, pd.DataFrame)
                    or "forecast_score" not in model_pred.columns
                ):
                    continue

                model_df = model_pred.copy()
                model_df["datetime"] = pd.to_datetime(model_df["datetime"]).dt.floor("s")
                model_df["forecast"] = model_df["forecast_score"].astype(float)
                model_df["signal"] = model_df["forecast"].astype(float)
                model_df["model_name"] = build_global_model_name(
                    timeframe=self.trading_timeframe,
                    ensemble_idx=ensemble_idx,
                    model_name=model_name,
                )
                model_df["timeframe"] = self.trading_timeframe.name
                forecast_vectors.append(
                    model_df[
                        [
                            "ticker",
                            "datetime",
                            "model_name",
                            "forecast",
                            "signal",
                            "timeframe",
                        ]
                    ]
                )

        if not forecast_vectors:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "datetime",
                    "model_name",
                    "forecast",
                    "signal",
                    "timeframe",
                ]
            )

        return pd.concat(forecast_vectors, ignore_index=True)

    def _apply_instrument_weights(
        self,
        combined_forecasts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply instrument weights to combined forecasts.

        Parameters
        ----------
        combined_forecasts : pd.DataFrame
            Columns: ['ticker', 'forecast_score']

        Returns
        -------
        pd.DataFrame
            Columns: ['ticker', 'forecast_score', 'instrument_weight', 'position_weighted']
        """
        df = combined_forecasts.copy()
        weights_by_ticker = self._get_effective_instrument_weights(df['ticker'].tolist())
        df['instrument_weight'] = df['ticker'].map(weights_by_ticker)

        # Calculate weighted position
        df['position_weighted'] = df['forecast_score'] * df['instrument_weight']

        return df

    def _apply_idm(
        self,
        positions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply IDM to account for portfolio-level diversification.

        Parameters
        ----------
        positions : pd.DataFrame
            Columns: ['ticker', 'forecast_score', 'instrument_weight', 'position_weighted']

        Returns
        -------
        pd.DataFrame
            Same columns plus 'position_idm'
        """
        df = positions.copy()
        df['position_idm'] = df['position_weighted'] * self.idm_
        return df

    def _apply_position_cap(
        self,
        positions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply position cap if specified.

        Parameters
        ----------
        positions : pd.DataFrame
            Columns include 'position_idm'

        Returns
        -------
        pd.DataFrame
            Columns: ['ticker', 'forecast_score', 'position_fraction']
        """
        df = positions.copy()

        if self.max_position_pct is not None:
            df['position_fraction'] = df['position_idm'].clip(
                lower=-self.max_position_pct,
                upper=self.max_position_pct
            )
        else:
            df['position_fraction'] = df['position_idm']

        return df[['ticker', 'forecast_score', 'position_fraction']]

    def fit_from_cache(
        self,
        query: PortfolioCacheQuery,
        target_data: Optional[pd.Series] = None,
    ) -> "TFPortfolio":
        """Fit from the central cache using a range/grid query."""
        candles_df = _query_candles_from_cache(query, self.trading_timeframe)
        return self.fit_from_candles(
            candles_df,
            target_data=target_data,
            start_date=query.start,
            end_date=query.end,
        )

    def predict_from_cache(
        self,
        query: PortfolioCacheQuery,
        return_ensemble_predictions: bool = False,
        return_base_model_predictions: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Predict from the central cache without requiring daily_volatility_df."""
        candles_df = _query_candles_from_cache(query, self.trading_timeframe)
        volatility_df = _query_volatility_from_cache(query)
        if volatility_df.empty:
            raise ArtifactMissingError(
                module_name="ewsd",
                ticker=None,
                timeframe=query.volatility_timeframe,
                requested_at=query.end,
                reason="EWSD volatility is missing from the central cache",
            )
        return self.predict_from_candles(
            candles_df,
            daily_volatility_df=volatility_df,
            return_ensemble_predictions=return_ensemble_predictions,
            return_base_model_predictions=return_base_model_predictions,
            start_date=query.start,
            end_date=query.end,
        )

    def predict_base_model_vectors_from_cache(
        self,
        query: PortfolioCacheQuery,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Emit base-model vectors using cache-native inputs."""
        candles_df = _query_candles_from_cache(query, self.trading_timeframe)
        volatility_df = _query_volatility_from_cache(query)
        if volatility_df.empty:
            raise ArtifactMissingError(
                module_name="ewsd",
                ticker=None,
                timeframe=query.volatility_timeframe,
                requested_at=query.end,
                reason="EWSD volatility is missing from the central cache",
            )
        return self.predict_base_model_vectors_from_candles(
            candles_df,
            daily_volatility_df=volatility_df,
            start_date=start_date or query.start,
            end_date=end_date or query.end,
        )
    
    def fit_from_candles(
        self,
        candles_df: pd.DataFrame,
        target_data: Optional[pd.Series] = None,
        start_date=None,
        end_date=None
    ) -> 'TFPortfolio':
        """
        Fit all ensembles using candles DataFrame.
        
        This is the new DataFrame-based API for fitting portfolios.
        Routes candles to each ensemble, which routes to base models.
        
        Uses date_range-based caching to avoid refitting for the same date range.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
            Should contain candles for the trading_timeframe of this portfolio
        target_data : pd.Series, optional
            Optional target/return series used while fitting the underlying ensembles.
            If None, ensembles must already be fitted.
        start_date : datetime, optional
            Start date for cached data. If None, inferred from candles_df.
        end_date : datetime, optional
            End date for cached data. If None, inferred from candles_df.

        Returns
        -------
        self
            Fitted portfolio
        """
        if not self.ensembles:
            raise ValueError("No ensembles provided. Cannot fit portfolio without ensembles.")
        
        # Filter candles for this portfolio's trading timeframe
        tf_candles = candles_df[candles_df['timeframe'] == self.trading_timeframe].copy()
        
        if tf_candles.empty:
            raise ValueError(
                f"No candles found for trading timeframe {self.trading_timeframe.name}. "
                f"Available timeframes: {candles_df['timeframe'].unique()}"
            )
        
        # Fit all ensembles sequentially (Cython-friendly, no GIL contention).
        # DiversifiedEnsemble.fit_from_candles currently ignores target_data and
        # computes aligned returns from candles, but it still requires the arg.
        fit_target = target_data if target_data is not None else pd.Series(dtype=float)
        logger.debug("Fitting %d ensemble(s) sequentially", len(self.ensembles))
        for idx, ensemble in enumerate(self.ensembles):
            logger.debug("Fitting ensemble %s...", idx)
            try:
                ensemble.fit_from_candles(tf_candles, fit_target, start_date, end_date)
            except Exception as exc:
                raise RuntimeError(
                    "TFPortfolio: ensemble fit failed for "
                    f"{_vault_ensemble_label(ensemble, idx)}. "
                    "Downstream global WeightLayer requires every ensemble to be fitted."
                ) from exc

        # Fit IDM if we have return data
        if target_data is not None:
            # Calculate returns from candles for IDM calculation
            returns_df = self._calculate_returns_from_candles(tf_candles)
            
            # Debug logging
            logger.debug(
                f"IDM calculation: returns_df shape={returns_df.shape}, "
                f"columns={list(returns_df.columns) if not returns_df.empty else []}, "
                f"unique tickers in candles={sorted(tf_candles['ticker'].unique().tolist())}"
            )
            
            if not returns_df.empty and len(returns_df.columns) >= 2:
                # Need at least 2 instruments to calculate IDM
                # Calculate IDM from correlations
                try:
                    self.fit(returns_df)
                    logger.debug(
                        f"IDM calculated successfully: IDM={self.idm_}, "
                        f"mean_correlation={self.mean_return_correlation_}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error calculating IDM: {e}. Using default IDM=1.0",
                        exc_info=True
                    )
                    self.idm_ = 1.0
                    self.mean_return_correlation_ = 1.0
                
                # If IDM wasn't calculated (e.g., insufficient data), default to 1.0
                if self.idm_ is None:
                    logger.warning("IDM is None after fit(). Using default IDM=1.0")
                    self.idm_ = 1.0
                    self.mean_return_correlation_ = 1.0
            else:
                # Not enough instruments or empty returns - set default IDM
                if returns_df.empty:
                    logger.warning(
                        f"Returns DataFrame is empty after processing. "
                        f"Input candles: {len(tf_candles)} rows, "
                        f"unique tickers: {sorted(tf_candles['ticker'].unique().tolist())}. "
                        f"Cannot calculate IDM. Using default IDM=1.0"
                    )
                elif len(returns_df.columns) < 2:
                    logger.warning(
                        f"Only {len(returns_df.columns)} instrument(s) in returns DataFrame. "
                        f"Need at least 2 instruments to calculate IDM. "
                        f"Available tickers in candles: {sorted(tf_candles['ticker'].unique().tolist())}. "
                        f"Using default IDM=1.0"
                    )
                self.idm_ = 1.0
                self.mean_return_correlation_ = 1.0
                # Set instruments_ from candles if not already set
                if self.instruments_ is None:
                    self.instruments_ = sorted(tf_candles['ticker'].unique().tolist())
            
            # Hard cutover: TF-level weight layer is no longer part of the
            # execution path; global strategy-level weighting owns combination.
        
        self.is_fitted_ = True
        
        return self
    
    def predict_from_candles(
        self,
        candles_df: pd.DataFrame,
        daily_volatility_df: pd.DataFrame,
        return_ensemble_predictions: bool = False,
        return_base_model_predictions: bool = False,
        start_date=None,
        end_date=None
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Generate position fractions using candles DataFrame.
        
        This is the new DataFrame-based API for prediction.
        Routes candles to ensembles, aggregates predictions, and applies risk management.
        
        Uses date_range-based caching to avoid recomputation for the same date range.
        Core predictions are cached regardless of return flags, enabling cache reuse
        between basic and granular prediction calls.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
            Should contain candles for the trading_timeframe of this portfolio
        daily_volatility_df : pd.DataFrame
            Daily EWSD volatility DataFrame with columns:
            ['datetime', 'ticker', 'ewsd_annual_vol'].
        return_ensemble_predictions : bool, default=False
            If True, return ensemble-level predictions in result dict
        return_base_model_predictions : bool, default=False
            If True, return base model-level predictions in result dict
        start_date : datetime, optional
            Start date for cached data. If None, inferred from candles_df.
        end_date : datetime, optional
            End date for cached data. If None, inferred from candles_df.

        Returns
        -------
        pd.DataFrame or Dict[str, Any]
            If both flags are False: DataFrame with portfolio positions
            If either flag is True: Dict with structure:
            {
                'portfolio': pd.DataFrame,  # Portfolio-level positions
                'ensembles': Dict[str, pd.DataFrame],  # Only if return_ensemble_predictions=True
                'base_models': Dict[str, pd.DataFrame]  # Only if return_base_model_predictions=True
            }
        """
        if not self.ensembles:
            raise ValueError("No ensembles provided. Cannot predict without ensembles.")

        if daily_volatility_df is None:
            raise ValueError(
                "daily_volatility_df is required for predict_from_candles(). "
                "Expected columns: ['datetime', 'ticker', 'ewsd_annual_vol']."
            )
        
        # Filter candles for this portfolio's trading timeframe
        tf_candles = candles_df[candles_df['timeframe'] == self.trading_timeframe].copy()
        
        if tf_candles.empty:
            empty_df = pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score', 'position_fraction'])
            if return_ensemble_predictions or return_base_model_predictions:
                result = {'portfolio': empty_df}
                if return_ensemble_predictions:
                    result['ensembles'] = {}
                if return_base_model_predictions:
                    result['base_models'] = {}
                return result
            return empty_df
        
        # Storage for all predictions (always compute everything for caching)
        forecast_vectors = []  # For WeightLayer.combine()
        ensemble_predictions_dict = {}
        base_model_predictions_dict = {}
        
        # Get predictions from all ensembles (with parallelization if available)
        def get_ensemble_predictions(ensemble_idx, ensemble):
            """Helper function for parallel prediction."""
            if not hasattr(ensemble, 'predict_from_candles'):
                return None, None
            
            try:
                ensemble_result = ensemble.predict_from_candles(
                    tf_candles,
                    daily_volatility_df=daily_volatility_df,
                    return_base_model_predictions=True,
                    start_date=start_date,
                    end_date=end_date
                )
                return ensemble_idx, ensemble_result
            except Exception as e:
                logger.error(
                    f"Error getting predictions from ensemble {ensemble_idx}: {e}",
                    exc_info=True
                )
                return ensemble_idx, None
        
        # Get predictions from all ensembles sequentially (Cython-friendly)
        logger.debug(f"Getting predictions from {len(self.ensembles)} ensemble(s) sequentially")
        ensemble_results = [
            get_ensemble_predictions(idx, ensemble)
            for idx, ensemble in enumerate(self.ensembles)
        ]
        
        # Process results - ALWAYS compute all predictions for caching
        for ensemble_idx, ensemble_result in ensemble_results:
            if ensemble_result is None:
                continue

            ensemble_obj = self.ensembles[ensemble_idx]
            ensemble_key = ensemble_prediction_dict_key(ensemble_obj, ensemble_idx)

            if isinstance(ensemble_result, dict):
                ensemble_pred = ensemble_result.get('ensemble')
                base_models = ensemble_result.get('base_models', {})
                
                # Build forecast vectors for WeightLayer (one per ensemble)
                # WeightLayer expects: ['ticker', 'model_name', 'forecast', 'signal']
                ensemble_forecast_vector = []
                for model_name, model_pred in base_models.items():
                    if isinstance(model_pred, pd.DataFrame) and 'forecast_score' in model_pred.columns:# Convert to WeightLayer format
                        forecast_df = model_pred.copy()
                        forecast_df['model_name'] = model_name
                        forecast_df['forecast'] = forecast_df['forecast_score']
                        # Binary activity flag: any non-zero forecast is active.
                        forecast_df['signal'] = _forecast_to_activity_signal(
                            forecast_df['forecast']
                        )
                        # Keep only required columns
                        forecast_df = forecast_df[['ticker', 'datetime', 'model_name', 'forecast', 'signal']]
                        ensemble_forecast_vector.append(forecast_df)
                
                if ensemble_forecast_vector:
                    # Combine all base models from this ensemble into one forecast vector
                    ensemble_vector_df = pd.concat(ensemble_forecast_vector, ignore_index=True)
                    forecast_vectors.append(ensemble_vector_df)
                
                # ALWAYS compute base model predictions (now fast with vectorization)
                for model_name, model_pred in base_models.items():
                    full_model_name = f"{ensemble_key}::{model_name}"
                    # Convert to position fractions (vectorized - O(n+m) complexity)
                    base_model_positions = apply_forecast_risk_management(
                        model_pred,
                        tf_candles,
                        instrument_weights=self.instrument_weights,
                        idm=self.idm_,
                        max_position_pct=self.max_position_pct,
                    )
                    base_model_predictions_dict[full_model_name] = base_model_positions
                
                # ALWAYS compute ensemble-level predictions (now fast with vectorization)
                if ensemble_pred is not None:
                    # Convert to position fractions (vectorized - O(n+m) complexity)
                    ensemble_positions = apply_forecast_risk_management(
                        ensemble_pred,
                        tf_candles,
                        instrument_weights=self.instrument_weights,
                        idm=self.idm_,
                        max_position_pct=self.max_position_pct,
                    )
                    
                    ensemble_predictions_dict[ensemble_key] = ensemble_positions
            else:
                # If ensemble doesn't return dict, it's already aggregated
                ensemble_pred = ensemble_result
                if ensemble_pred is not None:
                    ensemble_positions = apply_forecast_risk_management(
                        ensemble_pred,
                        tf_candles,
                        instrument_weights=self.instrument_weights,
                        idm=self.idm_,
                        max_position_pct=self.max_position_pct,
                    )
                    ensemble_predictions_dict[ensemble_key] = ensemble_positions
        
        if not forecast_vectors:
            empty_df = pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score', 'position_fraction'])
            if return_ensemble_predictions or return_base_model_predictions:
                result = {'portfolio': empty_df}
                if return_ensemble_predictions:
                    result['ensembles'] = {}
                if return_base_model_predictions:
                    result['base_models'] = {}
                return result
            return empty_df
        
        # Hard cutover: bypass TF-level WeightLayer and use unweighted model aggregation.
        forecast_scores_df = aggregate_forecast_vectors_fallback(forecast_vectors)
        
        # Apply risk management for portfolio-level
        positions_df = apply_forecast_risk_management(
            forecast_scores_df,
            tf_candles,
            instrument_weights=self.instrument_weights,
            idm=self.idm_,
            max_position_pct=self.max_position_pct,
        )
        
        full_result = {
            'portfolio': positions_df,
            'ensembles': ensemble_predictions_dict,
            'base_models': base_model_predictions_dict
        }
        return format_portfolio_result(
            full_result,
            return_ensemble_predictions=return_ensemble_predictions,
            return_base_model_predictions=return_base_model_predictions,
        )
    
    def _calculate_returns_from_candles(
        self,
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate returns DataFrame from candles for IDM calculation."""
        return calculate_returns_from_candles(candles_df)
    
    def get_diagnostics(self) -> Dict:
        """Return a snapshot of fitted state for inspection."""
        return {
            'is_fitted': self.is_fitted_,
            'idm': self.idm_,
            'mean_return_correlation': self.mean_return_correlation_,
            'n_instruments': len(self.instruments_) if self.instruments_ else 0,
            'instruments': self.instruments_,
            'idm_max': self.idm_max,
            'max_position_pct': self.max_position_pct,
            'trading_timeframe': self.trading_timeframe.name if self.trading_timeframe else None,
        }
    
    def print_diagnostics(self) -> None:
        """
        Print comprehensive diagnostic information in a readable format.
        
        This method prints IDM, FDM, weight layer information, and other
        portfolio diagnostics in a formatted way for easy inspection.
        """
        print("=" * 60)
        print("PORTFOLIO DIAGNOSTICS")
        print("=" * 60)
        
        print(f"\n📊 Portfolio Status:")
        print(f"  Fitted: {self.is_fitted_}")
        print(f"  Trading Timeframe: {self.trading_timeframe.name if self.trading_timeframe else 'N/A'}")
        print(f"  Number of Instruments: {len(self.instruments_) if self.instruments_ else 0}")
        print(f"  Instruments: {self.instruments_ if self.instruments_ else 'N/A'}")
        
        print(f"\n🎯 Instrument Diversification Multiplier (IDM):")
        print(f"  IDM: {self.idm_ if self.idm_ is not None else 'Not calculated'}")
        print(f"  IDM Max: {self.idm_max}")
        print(f"  Mean Return Correlation: {self.mean_return_correlation_ if self.mean_return_correlation_ is not None else 'N/A'}")
        
        print(f"\n📏 Position Sizing:")
        print(f"  Max Position %: {self.max_position_pct}")
        print(f"  Instrument Weights: {'Custom' if self.instrument_weights else 'Equal weight'}")
        
        # Print base model binning details for all ensembles
        if self.ensembles:
            print(f"\n🔬 Base Model Binning Details:")
            for ensemble_idx, ensemble in enumerate(self.ensembles):
                ensemble_name = f"Ensemble {ensemble_idx + 1}"
                if hasattr(ensemble, 'base_models') and ensemble.base_models:
                    print(f"\n  {ensemble_name}:")
                    for model_name, base_model in ensemble.base_models.items():
                        binning_model = getattr(base_model, 'binning_model', None)
                        if binning_model is None:
                            print(f"    {model_name}: No binning model")
                            continue
                        
                        # Get binning information
                        n_bins = getattr(binning_model, 'n_bins', 'N/A')
                        is_fitted = getattr(binning_model, 'is_fitted_', False)
                        strategy = getattr(binning_model, 'strategy', 'long')
                        best_long_bin = getattr(binning_model, 'best_long_bin_', None)
                        best_short_bin = getattr(binning_model, 'best_short_bin_', None)
                        thresholds = getattr(binning_model, 'thresholds_', None)
                        bin_stats = getattr(binning_model, 'bin_stats_', None)
                        
                        # Get feature column name if available
                        feature_column = getattr(base_model, 'feature_column', None)
                        
                        print(f"    {model_name}:")
                        if feature_column:
                            print(f"      Feature Column: {feature_column}")
                        print(f"      Fitted: {is_fitted}")
                        print(f"      Strategy: {strategy}")
                        print(f"      Number of Bins: {n_bins}")
                        
                        if is_fitted:
                            # Determine selected bin and strategy for display
                            selected_bin = None
                            selected_strategy = None
                            if strategy == 'long' and best_long_bin is not None:
                                selected_bin = best_long_bin
                                selected_strategy = 'LONG'
                            elif strategy == 'short' and best_short_bin is not None:
                                selected_bin = best_short_bin
                                selected_strategy = 'SHORT'
                            
                            # Show thresholds if available
                            if thresholds is not None and len(thresholds) > 0:
                                print(f"      Thresholds: {thresholds.tolist()}")
                            elif thresholds is not None and len(thresholds) == 0:
                                print(f"      Thresholds: [Constant feature - single bin]")
                            
                            # Show bin statistics with selected bin highlighted
                            if bin_stats is not None and len(bin_stats) > 0:
                                print(f"      Bin Statistics:")
                                # Sort bins by index (handle both string and int keys)
                                def get_bin_key(bin_item):
                                    key = bin_item[0]
                                    try:
                                        return int(key)
                                    except (ValueError, TypeError):
                                        return 0
                                
                                sorted_bins = sorted(bin_stats.items(), key=get_bin_key)
                                for bin_idx, stats in sorted_bins:
                                    # Convert bin_idx to int for comparison
                                    try:
                                        bin_idx_int = int(bin_idx)
                                    except (ValueError, TypeError):
                                        bin_idx_int = None
                                    
                                    # Highlight selected bin similar to decile plots
                                    is_selected = (
                                        (selected_bin is not None and bin_idx_int == selected_bin) or
                                        (selected_bin is None and len(sorted_bins) == 1)  # Single bin case
                                    )
                                    
                                    if is_selected and selected_strategy:
                                        bin_label = f"        Bin {bin_idx} ⭐ SELECTED [{selected_strategy}]"
                                    elif is_selected:
                                        bin_label = f"        Bin {bin_idx} ⭐ SELECTED"
                                    else:
                                        bin_label = f"        Bin {bin_idx}"
                                    
                                    print(bin_label)
                                    
                                    mean_ret = stats.get('mean_return', 'N/A')
                                    sortino = stats.get('sortino_metric', 'N/A')
                                    count = stats.get('count', 'N/A')
                                    feat_min = stats.get('feature_min', 'N/A')
                                    feat_max = stats.get('feature_max', 'N/A')
                                    
                                    print(f"          Mean Return: {mean_ret:.6f}" if isinstance(mean_ret, (int, float)) else f"          Mean Return: {mean_ret}")
                                    print(f"          Sortino Metric: {sortino:.4f}" if isinstance(sortino, (int, float)) else f"          Sortino Metric: {sortino}")
                                    print(f"          Sample Count: {count}" if isinstance(count, (int, float)) else f"          Sample Count: {count}")
                                    if isinstance(feat_min, (int, float)) and isinstance(feat_max, (int, float)):
                                        print(f"          Feature Range: [{feat_min:.4f}, {feat_max:.4f}]")
                                    else:
                                        print(f"          Feature Range: [{feat_min}, {feat_max}]")
                            else:
                                print(f"      Bin Statistics: Not available")
                        else:
                            print(f"      ⚠ Model not fitted yet")
        
        print("=" * 60)

    def __repr__(self) -> str:
        """String representation of the portfolio."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return f"Portfolio({fitted_str}, timeframe={self.trading_timeframe.name}, idm={self.idm_})"

    def __str__(self) -> str:
        """Detailed string description of the portfolio."""
        lines = [
            "Portfolio",
            f"  Trading Timeframe: {self.trading_timeframe.name}",
            f"  Is Fitted: {self.is_fitted_}",
            f"  IDM: {self.idm_}",
            f"  IDM Max: {self.idm_max}",
            f"  Max Position: {self.max_position_pct}",
            f"  Instruments: {self.instruments_}"
        ]
        return "\n".join(lines)


