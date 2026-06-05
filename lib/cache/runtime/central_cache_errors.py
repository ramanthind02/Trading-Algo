from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from lib.core.enums import Ticker, TimeFrame


class CentralCacheError(Exception):
    """Base class for central-cache failures."""


class ArtifactMissingError(CentralCacheError):
    """Raised when a requested artifact or candle is absent."""

    def __init__(
        self,
        message: str = "Artifact is missing",
        *,
        module_name: Optional[str] = None,
        ticker: Optional[Ticker] = None,
        timeframe: Optional[TimeFrame] = None,
        requested_at: Optional[datetime] = None,
        requested_range: Any = None,
        available_range: Any = None,
        path: Optional[str] = None,
        key: Any = None,
        reason: Optional[str] = None,
    ) -> None:
        self.module_name = module_name
        self.ticker = ticker
        self.timeframe = timeframe
        self.requested_at = requested_at
        self.requested_range = requested_range
        self.available_range = available_range
        self.path = path
        self.key = key
        self.reason = reason or message
        self.message = self._build_message(message)
        super().__init__(self.message)

    def _build_message(self, message: str) -> str:
        ticker = self.ticker.name if self.ticker is not None else "*"
        timeframe = self.timeframe.name if self.timeframe is not None else "*"
        parts = [message]
        if self.module_name is not None:
            parts.append(f"module={self.module_name}")
        if self.key is not None:
            parts.append(f"key={self.key}")
        parts.append(f"ticker={ticker}")
        parts.append(f"timeframe={timeframe}")
        if self.requested_at is not None:
            parts.append(f"requested_at={self.requested_at}")
        if self.requested_range is not None:
            parts.append(f"requested_range={self.requested_range}")
        if self.available_range is not None:
            parts.append(f"available_range={self.available_range}")
        if self.path is not None:
            parts.append(f"path={self.path}")
        if self.reason and self.reason != message:
            parts.append(f"reason={self.reason}")
        return " | ".join(parts)


class CacheCoverageError(CentralCacheError):
    """Raised when stored data does not cover a requested range."""

    def __init__(
        self,
        message: str = "Cache coverage error",
        *,
        module_name: Optional[str] = None,
        ticker: Optional[Ticker] = None,
        timeframe: Optional[TimeFrame] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        coverage_start: Optional[datetime] = None,
        coverage_end: Optional[datetime] = None,
        requested_range: Any = None,
        available_range: Any = None,
        path: Optional[str] = None,
        key: Any = None,
    ) -> None:
        self.module_name = module_name
        self.ticker = ticker
        self.timeframe = timeframe
        self.start = start
        self.end = end
        self.coverage_start = coverage_start
        self.coverage_end = coverage_end
        self.requested_range = requested_range
        self.available_range = available_range
        self.path = path
        self.key = key
        self.message = self._build_message(message)
        super().__init__(self.message)

    def _build_message(self, message: str) -> str:
        ticker = self.ticker.name if self.ticker is not None else "*"
        timeframe = self.timeframe.name if self.timeframe is not None else "*"
        parts = [message]
        if self.module_name is not None:
            parts.append(f"module={self.module_name}")
        if self.key is not None:
            parts.append(f"key={self.key}")
        parts.append(f"ticker={ticker}")
        parts.append(f"timeframe={timeframe}")
        if self.start is not None or self.end is not None:
            parts.append(f"range=({self.start}, {self.end})")
        if self.requested_range is not None:
            parts.append(f"requested_range={self.requested_range}")
        if self.available_range is not None:
            parts.append(f"available_range={self.available_range}")
        if self.coverage_start is not None or self.coverage_end is not None:
            parts.append(f"coverage=({self.coverage_start}, {self.coverage_end})")
        if self.path is not None:
            parts.append(f"path={self.path}")
        return " | ".join(parts)
