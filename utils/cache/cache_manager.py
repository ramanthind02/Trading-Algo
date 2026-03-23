"""Re-export for CLI and imports; implementation: ``utils.cache.runtime.cache_manager``."""

import utils.cache.runtime.cache_manager as _impl

from .runtime.cache_manager import *  # noqa: F403

if __name__ == "__main__":
    _impl.main()
