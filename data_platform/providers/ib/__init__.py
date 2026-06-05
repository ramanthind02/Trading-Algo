"""IB (Interactive Brokers) data adapter.

Currently implements the individual-contract archive (futures expiries via the
TWS API). Live trading + the daily CONTFUT append remain in
scripts/enigma_live_forecast.py and utils/cache/runtime/ (the runtime/engine
layer); see README.md.
"""
from ._client import IbDataClient, IbExpiry, ib_available
from .contracts import archive as archive_ib_contracts
from .contracts import ib_contracts_dir, ib_ticker_dir

__all__ = [
    "IbDataClient",
    "IbExpiry",
    "ib_available",
    "archive_ib_contracts",
    "ib_contracts_dir",
    "ib_ticker_dir",
]
