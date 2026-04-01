from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from utils.core.enums import Ticker, TimeFrame


def normalize_forecast_value(forecast: float) -> float:
    """Project raw ensemble forecast magnitudes onto a 0-1 display scale."""
    return float(1.0 / (1.0 + np.exp(-forecast * 2)))


@dataclass
class ForecastPredictionRuntime:
    volatility_service: Any
    ensembles: Dict[tuple[Ticker, TimeFrame], Any]
    ml_managers: Dict[tuple[Ticker, TimeFrame], Any]
    candle_buffers: Dict[tuple[Ticker, TimeFrame], list[Any]]
    tickers: list[Ticker]
    logger: logging.Logger

    def prepare_prediction_data(
        self,
        ticker: Ticker,
        n_samples: int,
    ) -> tuple[pd.Series, pd.Series]:
        """Build ticker and daily-volatility vectors required by ensemble prediction."""
        ticker_series = pd.Series([ticker.value] * n_samples)

        latest_map = self.volatility_service.latest_volatility_map()
        ticker_key = ticker.name
        if ticker_key not in latest_map:
            raise ValueError(
                f"Missing daily EWSD volatility for ticker {ticker_key}. "
                "Daily volatility must be available before forecasting."
            )
        volatility = float(latest_map[ticker_key])
        volatility_series = pd.Series([volatility] * n_samples)
        return ticker_series, volatility_series

    def generate_portfolio_forecasts(self, timeframe: TimeFrame) -> Dict[str, float]:
        """Generate normalized forecasts for all configured tickers in one timeframe."""
        forecasts: Dict[str, float] = {}

        for ticker in self.tickers:
            try:
                ensemble_key = (ticker, timeframe)
                if ensemble_key not in self.ensembles:
                    self.logger.warning("No ensemble for %s %s", ticker.name, timeframe.name)
                    continue

                ensemble = self.ensembles[ensemble_key]
                ml_manager = self.ml_managers.get(ensemble_key)
                if ml_manager is None:
                    continue

                if len(self.candle_buffers.get(ensemble_key, [])) < 20:
                    self.logger.warning(
                        "Insufficient candles for %s: %s/20",
                        ticker.name,
                        len(self.candle_buffers.get(ensemble_key, [])),
                    )
                    continue

                features_df = ml_manager.matrix_df
                if features_df is None or features_df.empty:
                    self.logger.warning("No features for %s", ticker.name)
                    continue

                latest_features = features_df.iloc[[-1]]
                ticker_series, volatility_series = self.prepare_prediction_data(
                    ticker,
                    len(latest_features),
                )
                predictions = ensemble.predict(
                    X=latest_features,
                    ticker=ticker_series,
                    volatility=volatility_series,
                )

                if not isinstance(predictions, np.ndarray):
                    self.logger.error(
                        "Unexpected predictions type for %s: %s",
                        ticker.name,
                        type(predictions),
                    )
                    continue
                if len(predictions) == 0:
                    self.logger.warning("No predictions for %s", ticker.name)
                    continue

                normalized = normalize_forecast_value(float(predictions[0]))
                forecasts[ticker.name] = normalized
                self.logger.info("✅ %s: %.4f", ticker.name, normalized)
            except Exception as exc:
                self.logger.error("❌ Error forecasting %s: %s", ticker.name, exc)
                self.logger.exception("Full traceback:")

        return forecasts
