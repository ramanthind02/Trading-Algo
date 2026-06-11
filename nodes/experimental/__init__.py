"""Experimental bias nodes invented during agent-driven research runs.

A strategy designed via the ``/research`` workflow may introduce a new single-timeframe,
signal-only ``{-1, 0, +1}`` :class:`~nodes.BiasNode` here so the pipeline can import it before the
strategy is promoted to the vault. These are research artifacts, not production nodes — promote a
proven one into the appropriate ``nodes/<category>/`` folder deliberately.
"""
