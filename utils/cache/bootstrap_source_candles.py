"""Re-export for CLI and imports; implementation: ``utils.cache.runtime.bootstrap_source_candles``."""

import utils.cache.runtime.bootstrap_source_candles as _impl

from .runtime.bootstrap_source_candles import *  # noqa: F403

if __name__ == "__main__":
    _impl.main()
