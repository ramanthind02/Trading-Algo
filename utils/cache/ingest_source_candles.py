"""Re-export for CLI and imports; implementation: ``utils.cache.runtime.ingest_source_candles``."""

import utils.cache.runtime.ingest_source_candles as _impl

from .runtime.ingest_source_candles import *  # noqa: F403

if __name__ == "__main__":
    _impl.main()
