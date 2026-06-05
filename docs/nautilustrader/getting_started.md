<!-- source: https://nautilustrader.io/docs/latest/getting_started -->

[Getting Started](#getting-started)

To get started with NautilusTrader you need:

- Python 3.12–3.14 with the
`nautilus_trader`

package installed - A way to run Python scripts or Jupyter notebooks

[Examples](#examples)

The docs cover a subset of examples. For the full set, see the
[nautilus_trader repository](https://github.com/nautechsystems/nautilus_trader).

| Directory | Description |
|---|---|
|

[docs/tutorials/](../tutorials/)[docs/concepts/](../concepts/)[nautilus_trader/examples/](https://github.com/nautechsystems/nautilus_trader/tree/master/nautilus_trader/examples)[tests/unit_tests/](https://github.com/nautechsystems/nautilus_trader/tree/master/tests/unit_tests)[Backtesting API levels](#backtesting-api-levels)

NautilusTrader provides two API levels for backtesting:

| API Level | Description | Characteristics |
|---|---|---|
| High-Level | `BacktestNode` / `TradingNode` | Recommended for production, easier transition to live trading; requires a Parquet data catalog |
| Low-Level | `BacktestEngine` | For library development, direct component access; no live-trading path |

One node per process

Running multiple `BacktestNode`

or `TradingNode`

instances in the same process is not supported
due to global singleton state. Sequential execution with proper disposal between runs is supported.
See [Processes and threads](../concepts/architecture#processes-and-threads).

See the [Backtesting](../concepts/backtesting) guide for help choosing an API level.


### Backtest (low-level API)

Load raw data with loaders and wranglers, then run a backtest with BacktestEngine.


### Backtest (high-level API)

Load raw data into the data catalog, then run a backtest with BacktestNode.

[Running in Docker](#running-in-docker)

A self-contained Jupyter notebook server is available as a Docker image, no local setup required.

Then open `http://localhost:8888`

in your browser.

Container data is ephemeral, deleting the container removes all data.

NautilusTrader log output can exceed Jupyter's default rate limit, causing notebooks to hang.
Set `log_level`

to `"ERROR"`

to avoid this.
