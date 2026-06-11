"""Runtime building blocks for the Nautilus + vault live trading node.

Pure, broker-aware helpers (clock, sizing, demand-driven tick control, node
assembly) consumed by :mod:`deployment.live.vault_strategy` and
:mod:`deployment.live.run_vault_sandbox`. Nothing here connects to a terminal on
import.
"""
