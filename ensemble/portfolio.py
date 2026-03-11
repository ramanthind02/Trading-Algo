"""
Portfolio Class for Position Sizing and Instrument Allocation

This module provides a Portfolio class that applies instrument weighting,
Instrument Diversification Multiplier (IDM), and optional position capping
to combined forecasts from the WeightLayer.

The Portfolio is the final layer before Execution:
    WeightLayer.combine() -> Portfolio.predict() -> PositionSizer.calculate_positions()

Key responsibilities:
1. Apply instrument weights (equal weight or custom allocation)
2. Calculate and apply IDM from instrument return correlations
3. Apply optional position capping

Reference: Robert Carver's "Systematic Trading" and "Leveraged Trading"
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from utils.core.enums import TimeFrame
from utils.compute.fast_volatility import compute_ewsd_annualized_from_closes
from .ensemble_utils import normalize_candles_datetime_column, normalize_ticker_key
from .weight_layer import BaseWeightLayer, WeightLayer

logger = logging.getLogger(__name__)


def _normalize_candles_datetime_column(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime is only a column; delegate to shared helper."""
    return normalize_candles_datetime_column(candles_df)


def _forecast_to_activity_signal(forecast: pd.Series) -> pd.Series:
    """Convert signed forecast magnitudes to binary activity flags.

    Any non-zero forecast means the model is active (long or short).
    """
    return forecast.ne(0.0).astype(int)


def _build_global_model_name(
    timeframe: TimeFrame,
    ensemble_idx: int,
    model_name: str,
) -> str:
    """Stable global model namespace with explicit strategy + timeframe tags.

    Format: ``{model_name}__{TF}::ensemble_{idx}``.
    """
    return f"{model_name}__{timeframe.name}::ensemble_{ensemble_idx}"


def _parse_timeframe_from_global_model_name(model_name: str) -> str:
    """Extract timeframe tag from global names produced by _build_global_model_name."""
    left = str(model_name).split("::", 1)[0]
    if "__" in left:
        _, tf = left.rsplit("__", 1)
        return tf
    return left


class TFPortfolio:
    """
    Portfolio class for applying instrument weighting and IDM to combined forecasts.

    The Portfolio uses WeightLayer to combine forecasts from all base models across all ensembles.
    WeightLayer applies FDM (Forecast Diversification Multiplier) during combination.
    Portfolio then applies:
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
    >>> # Using default WeightLayer
    >>> portfolio = TFPortfolio(
    ...     ensembles=[ensemble1, ensemble2],
    ...     trading_timeframe=TimeFrame.D,
    ...     max_position_pct=2.0
    ... )
    >>> portfolio.fit_from_candles(candles_df, target_data)
    >>> positions = portfolio.predict_from_candles(test_candles)
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
            self.sector_allocation_config_ = self._load_sector_allocation_config(
                self.sector_allocation_config_path
            )
            self.instrument_weights = self._resolve_sector_allocation(
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

        vault_path = Path(vault_root)
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

    def _load_sector_allocation_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate a sector allocation configuration file."""
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except FileNotFoundError:
            raise FileNotFoundError(f"Sector allocation configuration file not found: {config_path}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in sector allocation configuration file: {exc}") from exc

        if not isinstance(config, dict):
            raise ValueError("Sector allocation root must be a JSON object")

        self._validate_sector_allocation_node(config, seen_tickers=set(), node_path="root")
        return config

    def _validate_sector_allocation_node(
        self,
        node: Dict[str, Any],
        seen_tickers: Set[str],
        node_path: str,
    ) -> None:
        """Validate node schema recursively before resolution."""
        weight = node.get("weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
            raise ValueError(f"Node '{node_path}' weight must be > 0")

        has_children = "children" in node
        has_tickers = "tickers" in node
        if has_children == has_tickers:
            raise ValueError(
                f"Node '{node_path}' must define exactly one of 'children' or 'tickers'"
            )

        if has_children:
            children = node["children"]
            if not isinstance(children, list) or len(children) == 0:
                raise ValueError(f"Node '{node_path}' children must be a non-empty list")
            for idx, child in enumerate(children):
                if not isinstance(child, dict):
                    raise ValueError(f"Node '{node_path}.children[{idx}]' must be an object")
                self._validate_sector_allocation_node(
                    child,
                    seen_tickers=seen_tickers,
                    node_path=f"{node_path}.children[{idx}]",
                )
            return

        tickers = node["tickers"]
        if not isinstance(tickers, list) or len(tickers) == 0:
            raise ValueError(f"Node '{node_path}' must define at least one ticker")
        if not all(isinstance(ticker, str) and ticker for ticker in tickers):
            raise ValueError(f"Node '{node_path}' tickers must be non-empty strings")
        if len(set(tickers)) != len(tickers):
            raise ValueError(f"Node '{node_path}' contains duplicate tickers within a leaf")

        duplicates = [ticker for ticker in tickers if ticker in seen_tickers]
        if duplicates:
            raise ValueError(f"Duplicate ticker in sector allocation config: {duplicates[0]}")
        seen_tickers.update(tickers)

        ticker_weights = node.get("ticker_weights")
        if ticker_weights is None:
            return

        if not isinstance(ticker_weights, dict):
            raise ValueError(f"Node '{node_path}' ticker_weights must be an object")
        if set(ticker_weights.keys()) != set(tickers):
            raise ValueError(
                f"Node '{node_path}' ticker_weights keys must match tickers exactly"
            )
        for ticker, ticker_weight in ticker_weights.items():
            if (
                not isinstance(ticker_weight, (int, float))
                or isinstance(ticker_weight, bool)
                or ticker_weight <= 0
            ):
                raise ValueError(
                    f"Node '{node_path}' ticker_weights values must be > 0 (ticker={ticker})"
                )

    def _resolve_sector_allocation(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Resolve sector tree into normalized ticker->weight mapping."""
        resolved: Dict[str, float] = {}

        if "children" in config:
            children = config["children"]
            total_weight = sum(child["weight"] for child in children)
            for child in children:
                contribution = child["weight"] / total_weight
                self._resolve_sector_allocation_node(child, contribution, resolved)
        elif "tickers" in config:
            self._resolve_sector_allocation_node(config, 1.0, resolved)
        else:
            raise ValueError("Sector allocation root must define exactly one of 'children' or 'tickers'")

        total_resolved = sum(resolved.values())
        if total_resolved <= 0:
            raise ValueError("Resolved sector allocation produced zero total weight")
        return {
            ticker: weight / total_resolved
            for ticker, weight in resolved.items()
        }

    def _resolve_sector_allocation_node(
        self,
        node: Dict[str, Any],
        parent_contribution: float,
        resolved: Dict[str, float],
    ) -> None:
        """Recursively accumulate ticker contributions from a validated node tree."""
        if "children" in node:
            children = node["children"]
            total_weight = sum(child["weight"] for child in children)
            for child in children:
                contribution = parent_contribution * (child["weight"] / total_weight)
                self._resolve_sector_allocation_node(child, contribution, resolved)
            return

        tickers = node["tickers"]
        ticker_weights = node.get("ticker_weights")
        if ticker_weights is None:
            equal_share = parent_contribution / len(tickers)
            for ticker in tickers:
                resolved[ticker] = resolved.get(ticker, 0.0) + equal_share
            return

        total_ticker_weight = sum(ticker_weights[ticker] for ticker in tickers)
        for ticker in tickers:
            ticker_share = parent_contribution * (ticker_weights[ticker] / total_ticker_weight)
            resolved[ticker] = resolved.get(ticker, 0.0) + ticker_share

    def _get_effective_instrument_weights(self, tickers: List[str]) -> Dict[str, float]:
        """Return instrument weights for the provided tickers, including fallback handling."""
        unique_tickers = list(dict.fromkeys(tickers))
        if not unique_tickers:
            return {}

        if self.instrument_weights is None:
            equal_weight = 1.0 / len(unique_tickers)
            return {ticker: equal_weight for ticker in unique_tickers}

        configured_weights = self.instrument_weights
        missing_tickers = [ticker for ticker in unique_tickers if ticker not in configured_weights]
        used_weight = sum(
            configured_weights[ticker] for ticker in unique_tickers if ticker in configured_weights
        )
        remaining_weight = max(1.0 - used_weight, 0.0)
        fallback_weight = (
            remaining_weight / len(missing_tickers)
            if missing_tickers
            else 0.0
        )
        return {
            ticker: configured_weights.get(ticker, fallback_weight)
            for ticker in unique_tickers
        }

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
        if instrument_returns.empty or len(instrument_returns.columns) < 2:
            # Single instrument or no data: IDM = 1.0
            self.mean_return_correlation_ = 1.0
            return 1.0
        
        # Build correlation matrix of instrument returns
        # Columns are tickers, rows are time periods
        corr_matrix = instrument_returns.corr()
        
        # Floor negative correlations at zero (Carver's recommendation)
        # This treats negative correlations as zero (no diversification benefit from negative correlation)
        corr_matrix = corr_matrix.clip(lower=0.0)
        
        # Calculate mean correlation (excluding diagonal)
        # Get upper triangle (excluding diagonal) and calculate mean
        # This gives us mean(|rho_ij|) for i != j as specified
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack()
        
        if len(correlations) == 0:
            # No correlations to calculate (shouldn't happen with 2+ instruments)
            self.mean_return_correlation_ = 1.0
            return 1.0
        
        # Mean correlation: mean(|rho_ij|) for i != j
        # Since we've already floored at zero, this is effectively mean(|rho_ij|)
        mean_correlation = correlations.mean()
        self.mean_return_correlation_ = mean_correlation
        
        # Calculate IDM: sqrt(1 / (mean_correlation + epsilon))
        # Lower correlation = higher IDM (more diversification benefit)
        epsilon = 0.01  # Small epsilon to avoid division by zero
        idm = np.sqrt(1.0 / (mean_correlation + epsilon))
        
        # Cap at idm_max (Carver's recommendation: 2.5)
        idm = min(idm, self.idm_max)
        
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

        volatility = self._calculate_volatility_from_candles(tf_candles)
        forecast_vectors: List[pd.DataFrame] = []

        for ensemble_idx, ensemble in enumerate(self.ensembles):
            if not hasattr(ensemble, "predict_from_candles"):
                continue

            try:
                ensemble_volatility = (
                    None
                    if getattr(ensemble, "fitted_ticker_volatility_", None)
                    else volatility
                )
                ensemble_result = ensemble.predict_from_candles(
                    tf_candles,
                    volatility=ensemble_volatility,
                    return_base_model_predictions=True,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as exc:
                logger.warning(
                    "Skipping ensemble %s during global vector extraction: %s",
                    ensemble_idx,
                    exc,
                )
                continue

            if not isinstance(ensemble_result, dict):
                continue

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
                model_df["model_name"] = _build_global_model_name(
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
            Optional return series passed to WeightLayer.fit(returns=...) for
            downside-risk weighting methods. If None, ensembles must be pre-fitted.
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
        fitted_ensembles = 0
        logger.debug(f"Fitting {len(self.ensembles)} ensemble(s) sequentially")
        for idx, ensemble in enumerate(self.ensembles):
            try:
                logger.debug(f"Fitting ensemble {idx}...")
                ensemble.fit_from_candles(tf_candles, fit_target, start_date, end_date)
                fitted_ensembles += 1
            except Exception as e:
                logger.error(f"Error fitting ensemble {idx}: {e}", exc_info=True)

        if fitted_ensembles == 0:
            raise RuntimeError(
                "TFPortfolio fitting failed: all ensemble fits failed. "
                "Check per-ensemble logs above."
            )

        
        # Fit IDM if we have return data
        if target_data is not None:
            # #region agent log
            _tickers = tf_candles["ticker"].unique().tolist()
            try:
                import json
                _log = {"sessionId": "1a52b7", "hypothesisId": "H1", "location": "portfolio.py:fit_from_candles", "message": "before _calculate_returns_from_candles", "data": {"n_tickers": len(_tickers), "tickers": [str(t) for t in _tickers]}, "timestamp": int(__import__("time").time() * 1000)}
                open("/home/raman/repos/Trading-Algo/.cursor/debug-1a52b7.log", "a").write(json.dumps(_log) + "\n")
            except Exception: pass
            # #endregion
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
        
        # Calculate volatility (from candles) - needed for all predictions
        volatility = self._calculate_volatility_from_candles(tf_candles)
        
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
                ensemble_volatility = (
                    None
                    if getattr(ensemble, "fitted_ticker_volatility_", None)
                    else volatility
                )
                ensemble_result = ensemble.predict_from_candles(
                    tf_candles,
                    volatility=ensemble_volatility,
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
                    ensemble_name = f"ensemble_{ensemble_idx}"
                    full_model_name = f"{ensemble_name}::{model_name}"
                    # Convert to position fractions (vectorized - O(n+m) complexity)
                    base_model_positions = self._apply_risk_management_to_forecasts(
                        model_pred, volatility, tf_candles
                    )
                    base_model_predictions_dict[full_model_name] = base_model_positions
                
                # ALWAYS compute ensemble-level predictions (now fast with vectorization)
                if ensemble_pred is not None:
                    ensemble_name = f"ensemble_{ensemble_idx}"
                    
                    # Convert to position fractions (vectorized - O(n+m) complexity)
                    ensemble_positions = self._apply_risk_management_to_forecasts(
                        ensemble_pred, volatility, tf_candles
                    )
                    
                    ensemble_predictions_dict[ensemble_name] = ensemble_positions
            else:
                # If ensemble doesn't return dict, it's already aggregated
                ensemble_pred = ensemble_result
                if ensemble_pred is not None:
                    ensemble_name = f"ensemble_{ensemble_idx}"
                    ensemble_positions = self._apply_risk_management_to_forecasts(
                        ensemble_pred, volatility, tf_candles
                    )
                    ensemble_predictions_dict[ensemble_name] = ensemble_positions
        
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
        forecast_scores_df = self._aggregate_ensembles_fallback(forecast_vectors, tf_candles)
        
        # Apply risk management for portfolio-level
        positions_df = self._apply_risk_management(
            forecast_scores_df, volatility, tf_candles
        )
        
        full_result = {
            'portfolio': positions_df,
            'ensembles': ensemble_predictions_dict,
            'base_models': base_model_predictions_dict
        }
        return self._format_cached_result(full_result, return_ensemble_predictions, return_base_model_predictions)
    
    def _format_cached_result(
        self,
        cached_result: Dict[str, Any],
        return_ensemble_predictions: bool,
        return_base_model_predictions: bool
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Format cached result based on return flags.
        
        Returns a deep copy of relevant portions to avoid cache mutation.
        
        Parameters
        ----------
        cached_result : Dict[str, Any]
            Full cached result with 'portfolio', 'ensembles', 'base_models' keys
        return_ensemble_predictions : bool
            Whether to include ensemble predictions
        return_base_model_predictions : bool
            Whether to include base model predictions
            
        Returns
        -------
        pd.DataFrame or Dict[str, Any]
            Formatted result based on flags
        """
        if return_ensemble_predictions or return_base_model_predictions:
            result = {'portfolio': self._deep_copy_result(cached_result['portfolio'])}
            if return_ensemble_predictions:
                result['ensembles'] = self._deep_copy_result(cached_result.get('ensembles', {}))
            if return_base_model_predictions:
                result['base_models'] = self._deep_copy_result(cached_result.get('base_models', {}))
            return result
        else:
            # Just return portfolio DataFrame
            return self._deep_copy_result(cached_result['portfolio'])
    
    def _deep_copy_result(self, obj: Any) -> Any:
        """
        Recursively deep copy DataFrames in nested structures.
        
        Handles:
        - pd.DataFrame: returns .copy()
        - dict: recursively copies values
        - list: recursively copies elements
        - other: returns as-is (immutable or primitive types)
        
        Parameters
        ----------
        obj : Any
            Object to deep copy (DataFrame, dict, list, or primitive)
            
        Returns
        -------
        Any
            Deep copied object with all DataFrames copied
        """
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        elif isinstance(obj, dict):
            return {k: self._deep_copy_result(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy_result(item) for item in obj]
        else:
            return obj
    
    def _aggregate_ensembles(
        self,
        ensemble_predictions: List[pd.DataFrame],
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate predictions from multiple ensembles.
        
        Aggregates across ensembles (if multiple), but preserves ticker-level information.
        Each ticker keeps its own forecast_score.
        
        Parameters
        ----------
        ensemble_predictions : List[pd.DataFrame]
            List of prediction DataFrames from each ensemble (columns: ticker, datetime, forecast_score)
        candles_df : pd.DataFrame
            Candles DataFrame for alignment
            
        Returns
        -------
        pd.DataFrame
            Aggregated forecast scores with columns: ticker, datetime, forecast_score
            (aggregated across ensembles, but preserving ticker-level granularity)
        """
        if not ensemble_predictions:
            return pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score'])
        
        # Combine all predictions (DataFrames)
        combined_df = pd.concat(ensemble_predictions, ignore_index=True)
        
        # Group by (datetime, ticker) and take mean of forecast_score across ensembles
        # This preserves ticker-level information while aggregating across ensembles
        if 'ticker' in combined_df.columns and 'datetime' in combined_df.columns:
            # Group by (datetime, ticker) and take mean across ensembles
            aggregated = combined_df.groupby(['datetime', 'ticker'])['forecast_score'].mean().reset_index()
        elif 'datetime' in combined_df.columns:
            # If no ticker column, just group by datetime
            aggregated = combined_df.groupby('datetime')['forecast_score'].mean().reset_index()
            # Add dummy ticker column if needed (shouldn't happen in normal flow)
            if 'ticker' not in aggregated.columns:
                aggregated['ticker'] = candles_df['ticker'].iloc[0] if len(candles_df) > 0 else None
        else:
            # Fallback: use index if datetime is index
            if combined_df.index.name == 'datetime' or isinstance(combined_df.index, pd.DatetimeIndex):
                aggregated = combined_df.groupby(combined_df.index)['forecast_score'].mean().reset_index()
                aggregated.columns = ['datetime', 'forecast_score']
            else:
                # Last resort: try to use index as-is
                aggregated = combined_df.set_index('datetime')['forecast_score'].groupby(level=0).mean().reset_index()
                aggregated.columns = ['datetime', 'forecast_score']
            
            # Add ticker column if missing
            if 'ticker' not in aggregated.columns:
                aggregated['ticker'] = candles_df['ticker'].iloc[0] if len(candles_df) > 0 else None
        
        # Ensure columns are in correct order
        if 'ticker' in aggregated.columns and 'datetime' in aggregated.columns:
            aggregated = aggregated[['ticker', 'datetime', 'forecast_score']]
        
        # Note: FDM is now applied by WeightLayer.combine(), not here
        # This method is only used as fallback when WeightLayer is not fitted
        
        return aggregated
    
    def _calculate_volatility_from_candles(
        self,
        candles_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate blended volatility from close prices using a fast, array-based EWSD approximation.

        This replaces the earlier per-candle EWSDNode loop with a vectorized
        implementation that operates directly on NumPy arrays. Conceptually it
        matches Carver's approach:

        - 70% short-run EWMA-32 of squared returns
        - 30% long-run historical standard deviation (10-year window)
        - Annualized by multiplying daily sigma by 16

        Parameters
        ----------
        candles_df : pd.DataFrame
            Candles DataFrame with columns: datetime, ticker, close

        Returns
        -------
        Dict[str, float]
            Mapping from ticker to annualized blended volatility (as decimal, not percentage)
        """
        volatility_dict: Dict[str, float] = {}

        if candles_df.empty or 'ticker' not in candles_df.columns or 'close' not in candles_df.columns:
            return volatility_dict

        df = _normalize_candles_datetime_column(candles_df)
        df['datetime'] = pd.to_datetime(df['datetime'])

        for ticker_name, ticker_candles in df.groupby('ticker'):
            ticker_candles = ticker_candles.sort_values('datetime')

            closes = ticker_candles['close'].to_numpy(dtype=np.float64)
            if closes.size < 2:
                volatility_dict[ticker_name] = 0.20
                continue

            try:
                vol = compute_ewsd_annualized_from_closes(closes)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Error computing EWSD volatility for ticker '%s': %s. "
                    "Falling back to simple annualized std.",
                    ticker_name,
                    exc,
                )
                vol = np.nan

            if not np.isfinite(vol) or vol <= 0.0:
                ticker_candles = ticker_candles.copy()
                ticker_candles['returns'] = ticker_candles['close'].pct_change()
                daily_vol = float(ticker_candles['returns'].std())
                annual_vol = daily_vol * np.sqrt(252.0)
                vol = annual_vol if np.isfinite(annual_vol) and annual_vol > 0.0 else 0.20
                logger.warning(
                    "EWSD calculation produced invalid value for ticker '%s'. "
                    "Using simple volatility calculation as fallback.",
                    ticker_name,
                )

            volatility_dict[ticker_name] = float(vol)

        return volatility_dict
    
    def _calculate_returns_from_candles(
        self,
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate returns DataFrame from candles for IDM calculation.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            Candles DataFrame
            
        Returns
        -------
        pd.DataFrame
            Returns DataFrame with tickers as columns, datetime as index
        """
        returns_dict: Dict[str, pd.Series] = {}
        date_ranges_dict = {}  # Store date ranges for logging

        candles_df = _normalize_candles_datetime_column(candles_df)
        
        for ticker in candles_df['ticker'].unique():
            ticker_candles = candles_df[candles_df['ticker'] == ticker].copy()
            
            # Ensure datetime column is properly formatted as datetime
            ticker_candles['datetime'] = pd.to_datetime(ticker_candles['datetime'])
            ticker_candles = ticker_candles.sort_values('datetime')
            
            # Calculate returns
            ticker_candles['returns'] = ticker_candles['close'].pct_change()
            
            # Set datetime as index (already datetime type)
            ticker_candles = ticker_candles.set_index('datetime')
            # Drop NaN from individual ticker (first row will be NaN from pct_change)
            ticker_returns = ticker_candles['returns'].dropna()
            
            ticker_key = self._normalize_ticker_key(ticker)
            
            if len(ticker_returns) > 0:
                # Ensure datetime index is properly formatted as DatetimeIndex
                if not isinstance(ticker_returns.index, pd.DatetimeIndex):
                    ticker_returns.index = pd.to_datetime(ticker_returns.index)
                
                # Normalize datetime index to remove timezone and time components for alignment
                # This ensures all tickers align on the same dates
                ticker_returns.index = ticker_returns.index.normalize()
                
                returns_dict[ticker_key] = ticker_returns
                date_ranges_dict[ticker_key] = (ticker_returns.index.min(), ticker_returns.index.max())
                
                logger.debug(
                    f"Added returns for ticker {ticker_key}: {len(ticker_returns)} rows, "
                    f"date range: {ticker_returns.index.min()} to {ticker_returns.index.max()}"
                )
            else:
                logger.warning(f"No valid returns for ticker {ticker_key} after dropna()")
                date_ranges_dict[ticker_key] = (None, None)
        
        if not returns_dict:
            logger.warning("No returns calculated for any ticker")
            return pd.DataFrame()
        
        # Create DataFrame with all ticker returns
        # This will align by datetime index (union of all datetimes)
        # Ensure all indices are normalized and the same type before creating DataFrame
        normalized_returns_dict: Dict[str, pd.Series] = {}
        for ticker_key, ticker_returns in returns_dict.items():
            # Create a copy to avoid modifying original
            normalized_returns = ticker_returns.copy()
            
            # Ensure index is DatetimeIndex and normalized (date-only, no time)
            if not isinstance(normalized_returns.index, pd.DatetimeIndex):
                normalized_returns.index = pd.to_datetime(normalized_returns.index)
            
            # Normalize to remove time components (ensures alignment)
            normalized_returns.index = normalized_returns.index.normalize()
            
            normalized_returns_dict[ticker_key] = normalized_returns
        
        returns_df = pd.DataFrame(normalized_returns_dict)
        
        logger.debug(
            f"Returns DataFrame created: shape={returns_df.shape}, "
            f"columns={list(returns_df.columns)}, "
            f"index type: {type(returns_df.index)}, "
            f"NaN count per column: {returns_df.isna().sum().to_dict()}, "
            f"Sample index values: {returns_df.index[:5].tolist() if len(returns_df) > 0 else 'empty'}"
        )
        
        # Only drop rows where we have fewer than 2 tickers with valid returns
        # We need at least 2 tickers for correlation calculation
        # Use dropna with thresh=2 to keep rows with at least 2 non-NaN values
        if len(returns_df.columns) >= 2:
            rows_before = len(returns_df)
            # Count non-NaN values per row
            non_nan_per_row = returns_df.notna().sum(axis=1)
            rows_with_2plus = (non_nan_per_row >= 2).sum()
            
            logger.debug(
                f"Before dropna(thresh=2): {rows_before} rows, "
                f"{rows_with_2plus} rows have 2+ non-NaN values"
            )
            
            returns_df = returns_df.dropna(thresh=2)
            rows_after = len(returns_df)
            
            if rows_after == 0:
                # Get date ranges from stored dict (before DataFrame creation)
                date_ranges = [(col, date_ranges_dict.get(col, (None, None))[0], date_ranges_dict.get(col, (None, None))[1]) 
                              for col in returns_df.columns]
                
                logger.warning(
                    f"Returns DataFrame is empty after dropna(thresh=2). "
                    f"Input: {rows_before} rows, {len(returns_df.columns)} columns. "
                    f"Only {rows_with_2plus} rows had 2+ non-NaN values. "
                    f"This suggests tickers have no overlapping datetime indices. "
                    f"Date ranges from individual tickers: {date_ranges}"
                )
            else:
                logger.debug(
                    f"After dropna(thresh=2): {rows_before} -> {rows_after} rows, "
                    f"columns={list(returns_df.columns)}"
                )
        else:
            # Fewer than 2 tickers: return 1-column returns (non-empty); fit_from_candles
            # will set IDM=1.0 and log the "Only N instrument(s)" warning.
            # #region agent log
            try:
                import json
                _log = {"sessionId": "1a52b7", "hypothesisId": "H1", "location": "portfolio.py:_calculate_returns_from_candles", "message": "returning 1-column returns (columns < 2)", "data": {"n_columns": len(returns_df.columns), "columns": list(returns_df.columns)}, "timestamp": int(__import__("time").time() * 1000)}
                open("/home/raman/repos/Trading-Algo/.cursor/debug-1a52b7.log", "a").write(json.dumps(_log) + "\n")
            except Exception: pass
            # #endregion
            return returns_df

        return returns_df

    @staticmethod
    def _normalize_ticker_key(ticker: object) -> str:
        """Normalize ticker to string key; delegate to shared helper."""
        return normalize_ticker_key(ticker)
    
    def _align_forecasts_with_candles(
        self,
        combined_forecasts: pd.DataFrame,
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align combined forecasts (from WeightLayer) with candles DataFrame.
        
        WeightLayer.combine() returns ['ticker', 'datetime', 'forecast_score'] if datetime is available.
        If datetime is missing, we merge with candles to add it.
        
        Parameters
        ----------
        combined_forecasts : pd.DataFrame
            Combined forecasts from WeightLayer with columns: ['ticker', 'forecast_score'] or ['ticker', 'datetime', 'forecast_score']
        candles_df : pd.DataFrame
            Candles DataFrame for datetime alignment
            
        Returns
        -------
        pd.DataFrame
            Forecast scores with columns: ['ticker', 'datetime', 'forecast_score']
        """
        # If datetime is already in combined_forecasts, use it directly
        if 'datetime' in combined_forecasts.columns:
            # Ensure datetime is datetime type
            combined_forecasts = combined_forecasts.copy()
            combined_forecasts['datetime'] = pd.to_datetime(combined_forecasts['datetime'])
            return combined_forecasts[['ticker', 'datetime', 'forecast_score']]
        
        # Fallback: datetime not in combined_forecasts; broadcast forecast_score per ticker over candle datetimes
        ticker_scores = combined_forecasts.groupby('ticker', as_index=False)['forecast_score'].first()
        candles_subset = candles_df[['ticker', 'datetime']].copy()
        candles_subset['datetime'] = pd.to_datetime(candles_subset['datetime'])
        merged = candles_subset.merge(ticker_scores, on='ticker', how='left')
        merged['forecast_score'] = merged['forecast_score'].fillna(0.0)
        return merged[['ticker', 'datetime', 'forecast_score']]
    
    def _aggregate_ensembles_fallback(
        self,
        forecast_vectors: List[pd.DataFrame],
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fallback aggregation when WeightLayer is not fitted.
        
        Simple averaging across all base models.
        
        Parameters
        ----------
        forecast_vectors : List[pd.DataFrame]
            List of forecast vectors from all ensembles
        candles_df : pd.DataFrame
            Candles DataFrame for alignment
            
        Returns
        -------
        pd.DataFrame
            Aggregated forecast scores with columns: ['ticker', 'datetime', 'forecast_score']
        """
        if not forecast_vectors:
            return pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score'])
        
        # Combine all forecast vectors
        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        
        # Group by (datetime, ticker) and take mean of forecast
        if 'ticker' in all_forecasts.columns and 'datetime' in all_forecasts.columns:
            aggregated = all_forecasts.groupby(['datetime', 'ticker'])['forecast'].mean().reset_index()
            aggregated.columns = ['datetime', 'ticker', 'forecast_score']
            aggregated = aggregated[['ticker', 'datetime', 'forecast_score']]
        else:
            # Fallback
            aggregated = pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score'])
        
        return aggregated
    
    def _apply_risk_management_to_forecasts(
        self,
        forecasts_df: pd.DataFrame,
        volatility: Dict[str, float],
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply risk management to forecast DataFrame (for ensemble/base model level).
        
        Uses vectorized pandas operations for O(n+m) complexity instead of O(n*m).
        Note: Forecasts are already volatility-adjusted from Ensemble layer.
        
        Parameters
        ----------
        forecasts_df : pd.DataFrame
            Forecast scores with columns: ticker, datetime, forecast_score
            (already volatility-adjusted from Ensemble)
        volatility : Dict[str, float]
            Volatility per ticker (used for alignment, not scaling)
        candles_df : pd.DataFrame
            Candles DataFrame for alignment
            
        Returns
        -------
        pd.DataFrame
            Position fractions with columns: ticker, datetime, forecast_score, position_fraction
        """
        # Normalize to bar granularity for (datetime, ticker) merge
        forecasts_clean = forecasts_df[['ticker', 'datetime', 'forecast_score']].copy()
        forecasts_clean['datetime'] = pd.to_datetime(forecasts_clean['datetime']).dt.floor('s')
        candles_subset = candles_df[['ticker', 'datetime']].copy()
        candles_subset['datetime'] = pd.to_datetime(candles_subset['datetime']).dt.floor('s')
        
        # Vectorized merge on (ticker, datetime) - O(n+m) complexity
        result = candles_subset.merge(
            forecasts_clean,
            on=['ticker', 'datetime'],
            how='left'
        )
        
        # Fill missing forecast scores with 0.0
        result['forecast_score'] = result['forecast_score'].fillna(0.0)
        
        # Vectorized IDM application
        idm_value = self.idm_ if self.idm_ is not None else 1.0
        result['position_fraction'] = result['forecast_score'] * idm_value
        
        # Vectorized instrument weight application
        weights_by_ticker = self._get_effective_instrument_weights(result['ticker'].tolist())
        result['position_fraction'] *= result['ticker'].map(weights_by_ticker)
        
        # Vectorized position cap
        if self.max_position_pct is not None:
            result['position_fraction'] = result['position_fraction'].clip(
                lower=-self.max_position_pct,
                upper=self.max_position_pct
            )
        
        return result[['ticker', 'datetime', 'forecast_score', 'position_fraction']]
    
    def _apply_risk_management(
        self,
        forecast_scores_df: pd.DataFrame,
        volatility: Dict[str, float],
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply risk management to forecast scores.
        
        Uses vectorized pandas operations for O(n+m) complexity instead of O(n*m).
        Note: Forecasts are already volatility-adjusted from Ensemble layer.
        
        Parameters
        ----------
        forecast_scores_df : pd.DataFrame
            Forecast scores with columns: ticker, datetime, forecast_score
            (already volatility-adjusted from Ensemble)
        volatility : Dict[str, float]
            Volatility per ticker (used for alignment, not scaling)
        candles_df : pd.DataFrame
            Candles DataFrame for alignment
            
        Returns
        -------
        pd.DataFrame
            Position fractions with columns: ticker, datetime, forecast_score, position_fraction
        """
        # Delegate to vectorized implementation (same logic)
        return self._apply_risk_management_to_forecasts(forecast_scores_df, volatility, candles_df)

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
        Cross-TF weight layer. Defaults to ``inv_avg_pairwise_corr_grouped``.
    instrument_weights : dict mapping ticker → weight, optional
        Global instrument weights. If None, equal weight is applied.
    sector_allocation_config_path : str, optional
        Path to a JSON sector allocation tree. When provided, the resolved
        ticker-level weights supersede ``instrument_weights``.
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
        buy_hold_target_weight: float = 0.20,
        instrument_weights: Optional[Dict[str, float]] = None,
        sector_allocation_config_path: Optional[str] = None,
        idm_max: float = 2.5,
        max_position_pct: float = 2.0,
    ) -> None:
        self.tf_portfolios = tf_portfolios
        self.weight_layer: BaseWeightLayer = (
            weight_layer
            if weight_layer is not None
            else WeightLayer(weight_method="inv_avg_pairwise_corr_grouped", fdm_max=2.0)
        )
        self.buy_hold_target_weight = float(np.clip(buy_hold_target_weight, 0.0, 1.0))
        self.idm_max = idm_max
        self.max_position_pct = max_position_pct

        # Sector-allocation config resolution (same pattern as TFPortfolio)
        self.sector_allocation_config_path = sector_allocation_config_path
        self.sector_allocation_config_: Optional[Dict[str, Any]] = None
        if sector_allocation_config_path is not None:
            self.sector_allocation_config_ = self._load_sector_allocation_config(
                sector_allocation_config_path
            )
            self.instrument_weights: Optional[Dict[str, float]] = (
                self._resolve_sector_allocation(self.sector_allocation_config_)
            )
        else:
            self.instrument_weights = instrument_weights

        # Fitted attributes
        self.global_idm_: Optional[float] = None
        self.mean_instrument_return_correlation_: Optional[float] = None
        self.instruments_: Optional[List[str]] = None
        self.global_tf_weights_compat_: Dict[str, float] = {}
        self.weight_layer_diagnostics_: Dict[str, Any] = {}
        self.global_eligible_models_by_ticker_: Dict[str, Set[str]] = {}
        self.global_eligibility_diagnostics_: Dict[str, Any] = {}
        self.is_fitted_: bool = False

    def _enforce_buy_hold_target_weight(self) -> None:
        """Force a configured buy/hold share per ticker on fitted strategy weights."""
        weights_map = getattr(self.weight_layer, "weights_", None)
        if not isinstance(weights_map, dict):
            return

        target = self.buy_hold_target_weight
        for ticker, ticker_weights in list(weights_map.items()):
            if not isinstance(ticker_weights, pd.Series) or ticker_weights.empty:
                continue

            names = [str(n) for n in ticker_weights.index]
            buy_hold_mask = np.array(["buy_hold" in n.lower() for n in names], dtype=bool)
            if not bool(np.any(buy_hold_mask)):
                continue

            bh_names = [names[i] for i, is_bh in enumerate(buy_hold_mask) if is_bh]
            other_names = [names[i] for i, is_bh in enumerate(buy_hold_mask) if not is_bh]
            current = ticker_weights.astype(float).copy()

            if not other_names:
                out = pd.Series({k: 0.0 for k in names}, dtype=float)
                out.loc[bh_names] = 1.0 / len(bh_names)
                weights_map[ticker] = out
                continue

            bh_sum = float(current.loc[bh_names].sum())
            other_sum = float(current.loc[other_names].sum())

            if bh_sum > 0:
                bh_scaled = current.loc[bh_names] / bh_sum * target
            else:
                bh_scaled = pd.Series(1.0 / len(bh_names), index=bh_names, dtype=float) * target

            residual = 1.0 - target
            if other_sum > 0:
                other_scaled = current.loc[other_names] / other_sum * residual
            else:
                other_scaled = (
                    pd.Series(1.0 / len(other_names), index=other_names, dtype=float)
                    * residual
                )

            out = pd.concat([bh_scaled, other_scaled]).reindex(names).fillna(0.0)
            total = float(out.sum())
            if total > 0:
                out = out / total
            weights_map[ticker] = out

    @staticmethod
    def _build_global_signals_df(
        forecast_vectors: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """Build global (date × model) scaled-signal matrix for strategy-level weighting."""
        if not forecast_vectors:
            return pd.DataFrame()

        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        if all_forecasts.empty:
            return pd.DataFrame()

        if "datetime" not in all_forecasts.columns or "signal" not in all_forecasts.columns:
            return pd.DataFrame()

        df = all_forecasts.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.normalize()
        return (
            df.pivot_table(
                index="date",
                columns="model_name",
                values="signal",
                aggfunc="mean",
            )
            .fillna(0.0)
            .sort_index()
        )

    @staticmethod
    def _build_daily_grid(
        forecast_vectors: List[pd.DataFrame],
        reference_index: Optional[pd.Index] = None,
    ) -> pd.DatetimeIndex:
        """Build a daily date grid used to align all strategy streams."""
        if reference_index is not None and len(reference_index) > 0:
            idx = pd.to_datetime(reference_index)
            if isinstance(idx, pd.Series):
                idx = idx.dt.normalize()
            else:
                idx = pd.DatetimeIndex(idx).normalize()
            idx = idx[~idx.isna()]
            if len(idx) > 0:
                return pd.DatetimeIndex(sorted(set(idx)))

        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        if all_forecasts.empty or "datetime" not in all_forecasts.columns:
            return pd.DatetimeIndex([])

        dates = pd.to_datetime(all_forecasts["datetime"]).dt.normalize().dropna()
        if dates.empty:
            return pd.DatetimeIndex([])
        return pd.date_range(dates.min(), dates.max(), freq="D")

    @staticmethod
    def _align_forecast_vectors_to_daily_grid(
        forecast_vectors: List[pd.DataFrame],
        daily_grid: pd.DatetimeIndex,
    ) -> List[pd.DataFrame]:
        """Resample each ticker/model stream to a common daily grid via forward-fill."""
        if not forecast_vectors or len(daily_grid) == 0:
            return forecast_vectors

        combined = pd.concat(forecast_vectors, ignore_index=True)
        if combined.empty:
            return forecast_vectors

        required_cols = {"ticker", "datetime", "model_name", "forecast", "signal", "timeframe"}
        if not required_cols.issubset(set(combined.columns)):
            return forecast_vectors

        aligned_parts: List[pd.DataFrame] = []
        for (ticker, model_name, timeframe), grp in combined.groupby(
            ["ticker", "model_name", "timeframe"], sort=False
        ):
            base = grp.copy()
            base["datetime"] = pd.to_datetime(base["datetime"]).dt.normalize()
            base = (
                base.sort_values("datetime")
                .drop_duplicates(subset=["datetime"], keep="last")
                .set_index("datetime")
            )
            stream = base[["forecast", "signal"]].astype(float)
            aligned = stream.reindex(daily_grid).ffill().fillna(0.0).reset_index()
            aligned = aligned.rename(columns={"index": "datetime"})
            aligned["ticker"] = ticker
            aligned["model_name"] = model_name
            aligned["timeframe"] = timeframe
            aligned_parts.append(
                aligned[
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

        if not aligned_parts:
            return forecast_vectors

        return [pd.concat(aligned_parts, ignore_index=True)]

    @staticmethod
    def _derive_tf_weights_from_strategy_diagnostics(
        diagnostics: Dict[str, Any],
    ) -> Dict[str, float]:
        """Derive compatibility TF-level weights from strategy-level weights."""
        tickers = diagnostics.get("tickers", {}) if isinstance(diagnostics, dict) else {}
        tf_sums: Dict[str, float] = {}
        n_tickers = 0

        for ticker_info in tickers.values():
            weights = ticker_info.get("weights") or {}
            if not isinstance(weights, dict) or not weights:
                continue
            n_tickers += 1
            per_ticker_tf: Dict[str, float] = {}
            for model_name, weight in weights.items():
                tf_name = _parse_timeframe_from_global_model_name(str(model_name))
                per_ticker_tf[tf_name] = per_ticker_tf.get(tf_name, 0.0) + float(weight)
            for tf_name, tf_weight in per_ticker_tf.items():
                tf_sums[tf_name] = tf_sums.get(tf_name, 0.0) + tf_weight

        if n_tickers == 0 or not tf_sums:
            return {}

        averaged = {tf: w / n_tickers for tf, w in tf_sums.items()}
        total = sum(averaged.values())
        if total <= 0:
            return {}
        return {tf: w / total for tf, w in averaged.items()}

    @staticmethod
    def _count_sign_changes(signal: pd.Series, eps: float = 1e-12) -> int:
        """Count sign flips across non-zero points in a signal stream."""
        signs = np.sign(signal.to_numpy(dtype=float))
        non_zero = signs[np.abs(signs) > eps]
        if len(non_zero) < 2:
            return 0
        return int(np.sum(non_zero[1:] != non_zero[:-1]))

    def _collect_global_strategy_health_diagnostics(
        self,
        forecast_vectors: List[pd.DataFrame],
        global_returns: Optional[pd.Series],
    ) -> tuple[List[pd.DataFrame], Dict[str, Any]]:
        """Collect low-information diagnostics without dropping any strategy.

        Rules are diagnostic-only:
        - min observations
        - min non-zero activity ratio
        - min effective observations where signal*return is non-zero
        """
        min_obs = 252
        min_activity_ratio = 0.02
        min_effective_obs = 21
        returns_norm: Optional[pd.Series] = None
        if global_returns is not None and not global_returns.empty:
            returns_norm = global_returns.astype(float).copy()
            returns_norm.index = pd.to_datetime(returns_norm.index).normalize()

        combined = pd.concat(forecast_vectors, ignore_index=True)
        if combined.empty:
            return forecast_vectors, {
                "min_obs": min_obs,
                "min_activity_ratio": min_activity_ratio,
                "min_effective_obs": min_effective_obs,
                "tickers": {},
            }

        tickers_diag: Dict[str, Any] = {}
        eligible_by_ticker: Dict[str, Set[str]] = {}

        for ticker, ticker_df in combined.groupby("ticker", sort=False):
            ticker_key = str(ticker)
            model_stats = []
            for model_name, model_df in ticker_df.groupby("model_name", sort=False):
                signal = model_df["signal"].astype(float)
                obs = int(len(signal))
                activity_ratio = float((signal.abs() > 1e-12).mean()) if obs else 0.0
                effective_obs = 0
                if returns_norm is not None:
                    model_dates = pd.to_datetime(model_df["datetime"]).dt.normalize()
                    model_rets = returns_norm.reindex(model_dates).fillna(0.0).to_numpy(dtype=float)
                    effective_obs = int(np.sum(np.abs(signal.to_numpy(dtype=float) * model_rets) > 1e-12))
                else:
                    effective_obs = int(np.sum(np.abs(signal.to_numpy(dtype=float)) > 1e-12))
                is_eligible = (
                    obs >= min_obs
                    and activity_ratio >= min_activity_ratio
                    and effective_obs >= min_effective_obs
                )
                model_stats.append(
                    {
                        "model_name": str(model_name),
                        "obs": obs,
                        "activity_ratio": activity_ratio,
                        "effective_obs": effective_obs,
                        "eligible": is_eligible,
                    }
                )

            eligible_models = {str(s["model_name"]) for s in model_stats}

            eligible_by_ticker[ticker_key] = eligible_models
            tickers_diag[ticker_key] = {
                "eligible_models": sorted(eligible_models),
                "n_models_before": len(model_stats),
                "n_models_after": len(eligible_models),
                "model_stats": model_stats,
            }

        self.global_eligible_models_by_ticker_ = eligible_by_ticker
        diag = {
            "min_obs": min_obs,
            "min_activity_ratio": min_activity_ratio,
            "min_effective_obs": min_effective_obs,
            "mode": "diagnostics_only_no_filtering",
            "tickers": tickers_diag,
        }
        self.global_eligibility_diagnostics_ = diag
        return forecast_vectors, diag

    @staticmethod
    def _normalize_global_signals_by_downside_vol(
        forecast_vectors: List[pd.DataFrame],
        global_returns: Optional[pd.Series],
    ) -> List[pd.DataFrame]:
        """Normalize per-(ticker, model) signal streams by downside vol of signal*returns."""
        if global_returns is None or global_returns.empty:
            return forecast_vectors

        clean_returns = global_returns.astype(float).copy()
        clean_returns.index = pd.to_datetime(clean_returns.index).normalize()

        combined = pd.concat(forecast_vectors, ignore_index=True).copy()
        combined["date"] = pd.to_datetime(combined["datetime"]).dt.normalize()

        scales: Dict[tuple[str, str], float] = {}
        for (ticker, model_name), grp in combined.groupby(["ticker", "model_name"], sort=False):
            merged = (
                grp[["date", "signal"]]
                .merge(
                    clean_returns.rename("ret").to_frame(),
                    left_on="date",
                    right_index=True,
                    how="left",
                )
                .fillna({"ret": 0.0})
            )
            signal_ret = merged["signal"].astype(float) * merged["ret"].astype(float)
            downside = np.minimum(signal_ret.to_numpy(dtype=float), 0.0)
            downside_vol = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
            scales[(str(ticker), str(model_name))] = max(downside_vol, 1e-8)

        normalized: List[pd.DataFrame] = []
        for vec in forecast_vectors:
            out = vec.copy()
            out["signal"] = out.apply(
                lambda r: float(r["signal"])
                / scales.get((str(r["ticker"]), str(r["model_name"])), 1.0),
                axis=1,
            )
            normalized.append(out)
        return normalized

    # ------------------------------------------------------------------
    # Sector allocation helpers (delegated from TFPortfolio logic)
    # ------------------------------------------------------------------

    def _load_sector_allocation_config(self, config_path: str) -> Dict[str, Any]:
        """Load a sector allocation JSON config (identical logic to TFPortfolio)."""
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Sector allocation configuration file not found: {config_path}"
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in sector allocation configuration file: {exc}"
            ) from exc

        if not isinstance(config, dict):
            raise ValueError("Sector allocation root must be a JSON object")

        self._validate_sector_allocation_node(config, seen_tickers=set(), node_path="root")
        return config

    def _validate_sector_allocation_node(
        self,
        node: Dict[str, Any],
        seen_tickers: Set[str],
        node_path: str,
    ) -> None:
        """Recursively validate a sector allocation node."""
        weight = node.get("weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
            raise ValueError(f"Node '{node_path}' weight must be > 0")

        has_children = "children" in node
        has_tickers = "tickers" in node
        if has_children == has_tickers:
            raise ValueError(
                f"Node '{node_path}' must define exactly one of 'children' or 'tickers'"
            )

        if has_children:
            children = node["children"]
            if not isinstance(children, list) or len(children) == 0:
                raise ValueError(f"Node '{node_path}' children must be a non-empty list")
            for idx, child in enumerate(children):
                if not isinstance(child, dict):
                    raise ValueError(f"Node '{node_path}.children[{idx}]' must be an object")
                self._validate_sector_allocation_node(
                    child,
                    seen_tickers=seen_tickers,
                    node_path=f"{node_path}.children[{idx}]",
                )
            return

        tickers = node["tickers"]
        if not isinstance(tickers, list) or len(tickers) == 0:
            raise ValueError(f"Node '{node_path}' must define at least one ticker")
        if not all(isinstance(t, str) and t for t in tickers):
            raise ValueError(f"Node '{node_path}' tickers must be non-empty strings")
        if len(set(tickers)) != len(tickers):
            raise ValueError(f"Node '{node_path}' contains duplicate tickers within a leaf")

        duplicates = [t for t in tickers if t in seen_tickers]
        if duplicates:
            raise ValueError(
                f"Duplicate ticker in sector allocation config: {duplicates[0]}"
            )
        seen_tickers.update(tickers)

        ticker_weights = node.get("ticker_weights")
        if ticker_weights is None:
            return

        if not isinstance(ticker_weights, dict):
            raise ValueError(f"Node '{node_path}' ticker_weights must be an object")
        if set(ticker_weights.keys()) != set(tickers):
            raise ValueError(
                f"Node '{node_path}' ticker_weights keys must match tickers exactly"
            )
        for ticker, tw in ticker_weights.items():
            if (
                not isinstance(tw, (int, float))
                or isinstance(tw, bool)
                or tw <= 0
            ):
                raise ValueError(
                    f"Node '{node_path}' ticker_weights values must be > 0 (ticker={ticker})"
                )

    def _resolve_sector_allocation(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Resolve sector tree into a normalised ticker → weight mapping."""
        resolved: Dict[str, float] = {}
        if "children" in config:
            children = config["children"]
            total_weight = sum(child["weight"] for child in children)
            for child in children:
                contribution = child["weight"] / total_weight
                self._resolve_sector_allocation_node(child, contribution, resolved)
        elif "tickers" in config:
            self._resolve_sector_allocation_node(config, 1.0, resolved)
        else:
            raise ValueError(
                "Sector allocation root must define exactly one of 'children' or 'tickers'"
            )

        total_resolved = sum(resolved.values())
        if total_resolved <= 0:
            raise ValueError("Resolved sector allocation produced zero total weight")
        return {t: w / total_resolved for t, w in resolved.items()}

    def _resolve_sector_allocation_node(
        self,
        node: Dict[str, Any],
        parent_contribution: float,
        resolved: Dict[str, float],
    ) -> None:
        """Recursively accumulate ticker contributions from a validated node tree."""
        if "children" in node:
            children = node["children"]
            total_weight = sum(child["weight"] for child in children)
            for child in children:
                contribution = parent_contribution * (child["weight"] / total_weight)
                self._resolve_sector_allocation_node(child, contribution, resolved)
            return

        tickers = node["tickers"]
        ticker_weights = node.get("ticker_weights")
        if ticker_weights is None:
            equal_share = parent_contribution / len(tickers)
            for ticker in tickers:
                resolved[ticker] = resolved.get(ticker, 0.0) + equal_share
            return

        total_ticker_weight = sum(ticker_weights[t] for t in tickers)
        for ticker in tickers:
            ticker_share = parent_contribution * (ticker_weights[ticker] / total_ticker_weight)
            resolved[ticker] = resolved.get(ticker, 0.0) + ticker_share

    # ------------------------------------------------------------------
    # Instrument weight helper
    # ------------------------------------------------------------------

    def _get_effective_instrument_weights(self, tickers: List[str]) -> Dict[str, float]:
        """Return global instrument weights for the given tickers (equal-weight fallback)."""
        unique_tickers = list(dict.fromkeys(tickers))
        if not unique_tickers:
            return {}

        if self.instrument_weights is None:
            eq = 1.0 / len(unique_tickers)
            return {t: eq for t in unique_tickers}

        configured = self.instrument_weights
        missing = [t for t in unique_tickers if t not in configured]
        used_w = sum(configured[t] for t in unique_tickers if t in configured)
        remaining = max(1.0 - used_w, 0.0)
        fallback = remaining / len(missing) if missing else 0.0
        return {t: configured.get(t, fallback) for t in unique_tickers}

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

        if instrument_returns.empty or len(instrument_returns.columns) < 2:
            self.mean_instrument_return_correlation_ = 1.0
            self.global_idm_ = 1.0
            return

        corr_matrix = instrument_returns.corr().clip(lower=0.0)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        off_diag = corr_matrix.where(mask).stack()

        if len(off_diag) == 0:
            self.mean_instrument_return_correlation_ = 1.0
            self.global_idm_ = 1.0
            return

        mean_corr = float(off_diag.mean())
        self.mean_instrument_return_correlation_ = mean_corr

        epsilon = 0.01
        idm = float(np.sqrt(1.0 / (mean_corr + epsilon)))
        self.global_idm_ = min(idm, self.idm_max)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        candles_per_tf: Dict[TimeFrame, pd.DataFrame],
        instrument_returns: pd.DataFrame,
    ) -> "GlobalPortfolio":
        """Fit all TFPortfolios, WeightLayer, and global IDM.

        Parameters
        ----------
        candles_per_tf : dict mapping TimeFrame → DataFrame
            Candles for each TF's portfolio.  Each DataFrame must contain at
            least the columns expected by ``TFPortfolio.fit_from_candles``.
        instrument_returns : pd.DataFrame
            Daily instrument returns; ``columns`` = tickers.

        Returns
        -------
        self
        """
        # Step 1 — fit each TFPortfolio
        for tf_p in self.tf_portfolios:
            tf_candles = candles_per_tf.get(tf_p.trading_timeframe)
            if tf_candles is None:
                raise ValueError(
                    f"No candles provided for timeframe {tf_p.trading_timeframe.name}"
                )
            tf_p.fit_from_candles(tf_candles)

        # Step 2 — collect per-TF forecast streams
        tf_forecast_streams: Dict[TimeFrame, pd.DataFrame] = {}
        for tf_p in self.tf_portfolios:
            tf_candles = candles_per_tf[tf_p.trading_timeframe]
            vectors = tf_p.predict_base_model_vectors_from_candles(tf_candles)
            if vectors.empty:
                continue
            tf_forecast_streams[tf_p.trading_timeframe] = vectors

        forecast_vectors = list(tf_forecast_streams.values())
        if not forecast_vectors:
            raise ValueError("No forecast vectors available for global strategy weighting")

        # Use aggregate daily return proxy for downside-risk grouped global weighting.
        global_returns: Optional[pd.Series] = None
        if isinstance(instrument_returns, pd.DataFrame) and not instrument_returns.empty:
            global_returns = instrument_returns.mean(axis=1).astype(float)
        elif isinstance(instrument_returns, pd.Series) and not instrument_returns.empty:
            global_returns = instrument_returns.astype(float)
        if global_returns is not None:
            global_returns.index = pd.to_datetime(global_returns.index).normalize()

        # Align all strategy streams to a shared daily grid first.
        daily_grid = self._build_daily_grid(
            forecast_vectors=forecast_vectors,
            reference_index=global_returns.index if global_returns is not None else None,
        )
        forecast_vectors = self._align_forecast_vectors_to_daily_grid(
            forecast_vectors=forecast_vectors,
            daily_grid=daily_grid,
        )

        # Step 3 — diagnostics + downside-vol normalization (fit inputs only)
        eligible_vectors, _elig_diag = self._collect_global_strategy_health_diagnostics(
            forecast_vectors=forecast_vectors,
            global_returns=global_returns,
        )
        normalized_vectors = self._normalize_global_signals_by_downside_vol(
            forecast_vectors=eligible_vectors,
            global_returns=global_returns,
        )
        signals_df = self._build_global_signals_df(normalized_vectors)
        if signals_df.empty:
            raise ValueError("Global strategy signals are empty after normalization")

        # Step 4 — fit WeightLayer
        self.weight_layer.fit(
            forecast_vectors=normalized_vectors,
            signals=signals_df,
            returns=global_returns,
        )
        self._enforce_buy_hold_target_weight()
        self.weight_layer_diagnostics_ = self.weight_layer.get_diagnostics()
        self.global_tf_weights_compat_ = self._derive_tf_weights_from_strategy_diagnostics(
            self.weight_layer_diagnostics_
        )

        # Step 5 — compute global IDM
        self._calculate_global_idm(instrument_returns)

        # Step 6 — mark fitted
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        candles_per_tf: Dict[TimeFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Generate global position fractions for all instruments.

        Parameters
        ----------
        candles_per_tf : dict mapping TimeFrame → DataFrame
            Current candles for each TF.

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

        # Step 1 — collect per-TF forecast streams (pre-IDM, instrument-weighted)
        tf_forecast_streams: Dict[TimeFrame, pd.DataFrame] = {}
        for tf_p in self.tf_portfolios:
            tf_candles = candles_per_tf.get(tf_p.trading_timeframe)
            if tf_candles is None:
                raise ValueError(
                    f"No candles provided for timeframe {tf_p.trading_timeframe.name}"
                )
            vectors = tf_p.predict_base_model_vectors_from_candles(tf_candles)
            if vectors.empty:
                continue
            tf_forecast_streams[tf_p.trading_timeframe] = vectors

        forecast_vectors = list(tf_forecast_streams.values())
        if not forecast_vectors:
            return pd.DataFrame(
                columns=['ticker', 'datetime', 'forecast_score', 'position_fraction']
            )

        reference_grid: Optional[pd.Index] = None
        daily_candles = candles_per_tf.get(TimeFrame.D)
        if isinstance(daily_candles, pd.DataFrame) and "datetime" in daily_candles.columns:
            reference_grid = pd.to_datetime(daily_candles["datetime"]).dt.normalize().dropna()
        daily_grid = self._build_daily_grid(
            forecast_vectors=forecast_vectors,
            reference_index=reference_grid,
        )
        forecast_vectors = self._align_forecast_vectors_to_daily_grid(
            forecast_vectors=forecast_vectors,
            daily_grid=daily_grid,
        )

        # Step 2 — combine via WeightLayer
        combined = self.weight_layer.combine(forecast_vectors)

        if combined.empty:
            return pd.DataFrame(
                columns=['ticker', 'datetime', 'forecast_score', 'position_fraction']
            )

        # Step 3 — apply global instrument weights
        tickers = combined['ticker'].tolist()
        weights_by_ticker = self._get_effective_instrument_weights(tickers)
        result = combined.copy()
        result['position_weighted'] = (
            result['forecast_score'] * result['ticker'].map(weights_by_ticker)
        )

        # Step 4 — apply global IDM
        result['idm_scaled'] = result['position_weighted'] * self.global_idm_

        # Step 5 — clip to [-max_position_pct, +max_position_pct]
        result['position_fraction'] = result['idm_scaled'].clip(
            lower=-self.max_position_pct,
            upper=self.max_position_pct,
        )

        return result[['ticker', 'datetime', 'forecast_score', 'position_fraction']].reset_index(
            drop=True
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
                    self.weight_layer_diagnostics_
                    .get("summary", {})
                    .get("n_tickers", 0)
                ),
                "diagnostics": self.weight_layer_diagnostics_,
            },
        }

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        n = len(self.tf_portfolios)
        return f"GlobalPortfolio({fitted_str}, n_tf_portfolios={n}, idm={self.global_idm_})"


# ``Portfolio`` remains a backward-compatible alias for TFPortfolio so that
# existing call sites (sector allocation tests, vault auto-load, etc.) continue
# to work without changes.  GlobalPortfolio is the new top-level multi-TF class
# and is exported separately from ensemble/__init__.py.
Portfolio = TFPortfolio
