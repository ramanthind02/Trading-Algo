"""Per-broker MT5 credentials + execution-tier model for the live runtime.

Resolves a broker's MT5 credentials from the environment (`.env`), keyed by the
TARGET broker — never a hardcoded FTMO default. This closes the multi-broker
foot-gun where the entrypoint loaded FTMO creds regardless of `--broker`.

``darwinex`` may be armed for DEMO/SANDBOX only (a demo account now exists on the
Darwinex terminal; the demo-server check in the entrypoint still applies). Its
LIVE (funded) tier is refused — there is no Darwinex funded account here, only the
do-not-disturb tick-scraper, so live arming would risk touching it.

Reading env vars has no side effects; nothing here connects to a terminal.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class ExecTier(str, Enum):
    """Where orders go. ``sandbox`` is the only zero-broker-risk tier."""

    SANDBOX = "sandbox"  # local simulated fills (orders never leave the machine)
    DEMO = "demo"        # REAL orders to the broker's DEMO account
    LIVE = "live"        # REAL orders to a FUNDED account


# broker -> env-var prefix for that broker's DEMO account (creds live in .env).
_DEMO_ENV_PREFIX: dict[str, str] = {
    "ftmo": "FTMO_DEMO",
    "fundednext": "FUNDEDNEXT_DEMO",
    "darwinex": "DARWINEX_DEMO",  # demo account on the Darwinex terminal (demo/sandbox only)
}

# Brokers with no funded account on this machine — never arm a LIVE (funded) tier.
# darwinex's only non-demo terminal is the do-not-disturb tick scraper.
_NOT_LIVE_ARMABLE: frozenset[str] = frozenset({"darwinex"})


@dataclass(frozen=True)
class BrokerCredentials:
    """MT5 account creds for one broker/tier. The password is never echoed."""

    broker: str
    server: str
    login: int
    password: str = field(repr=False)

    @property
    def is_demo_server(self) -> bool:
        return "demo" in self.server.lower()

    @staticmethod
    def env_prefix(broker: str, tier: ExecTier) -> str:
        """Env-var prefix for ``broker`` at ``tier``. Raises for non-armable combos."""
        if tier is ExecTier.LIVE:
            if broker in _NOT_LIVE_ARMABLE:
                raise RuntimeError(
                    f"Broker {broker!r} has no funded (live) account on this runtime — "
                    "demo only. Its only non-demo terminal is the do-not-disturb scraper."
                )
            return f"{broker.upper()}_LIVE"  # funded account (none configured yet)
        prefix = _DEMO_ENV_PREFIX.get(broker)
        if prefix is None:
            raise RuntimeError(
                f"No demo credential mapping for broker {broker!r}. "
                f"Known: {sorted(_DEMO_ENV_PREFIX)}."
            )
        return prefix

    @staticmethod
    def from_env(broker: str, tier: ExecTier = ExecTier.DEMO) -> "BrokerCredentials":
        """Read ``{PREFIX}_SERVER/LOGIN/PASSWORD`` for ``broker`` at ``tier``.

        ``sandbox`` and ``demo`` both use the broker's DEMO account (the sandbox
        still connects to the broker for live DATA); ``live`` uses the funded
        account prefix (not configured until a funded account is added).
        """
        prefix = BrokerCredentials.env_prefix(broker, tier)
        try:
            server = os.environ[f"{prefix}_SERVER"]
            login = int(os.environ[f"{prefix}_LOGIN"])
            password = os.environ[f"{prefix}_PASSWORD"]
        except KeyError as exc:
            raise RuntimeError(
                f"Missing credentials for broker={broker} tier={tier.value}: expected "
                f"{prefix}_SERVER, {prefix}_LOGIN, {prefix}_PASSWORD in the environment/.env."
            ) from exc
        except ValueError as exc:
            raise RuntimeError(f"{prefix}_LOGIN must be an integer.") from exc
        return BrokerCredentials(broker=broker, server=server, login=login, password=password)


__all__ = ["BrokerCredentials", "ExecTier"]
