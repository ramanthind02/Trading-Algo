"""Unit tests for deployment.live.credentials (per-broker creds + exec tiers)."""
from __future__ import annotations

import pytest

from deployment.live.credentials import BrokerCredentials, ExecTier


def test_env_prefix_per_broker_and_tier() -> None:
    assert BrokerCredentials.env_prefix("ftmo", ExecTier.DEMO) == "FTMO_DEMO"
    assert BrokerCredentials.env_prefix("ftmo", ExecTier.SANDBOX) == "FTMO_DEMO"  # sandbox uses demo data
    assert BrokerCredentials.env_prefix("ftmo", ExecTier.LIVE) == "FTMO_LIVE"
    assert BrokerCredentials.env_prefix("fundednext", ExecTier.DEMO) == "FUNDEDNEXT_DEMO"


def test_darwinex_demo_armable_live_blocked() -> None:
    # A demo account now exists on the Darwinex terminal: demo/sandbox resolve.
    assert BrokerCredentials.env_prefix("darwinex", ExecTier.DEMO) == "DARWINEX_DEMO"
    assert BrokerCredentials.env_prefix("darwinex", ExecTier.SANDBOX) == "DARWINEX_DEMO"
    # But there is no funded Darwinex account — LIVE (funded) is refused.
    with pytest.raises(RuntimeError, match="no funded"):
        BrokerCredentials.env_prefix("darwinex", ExecTier.LIVE)


def test_unknown_broker_raises() -> None:
    with pytest.raises(RuntimeError, match="No demo credential mapping"):
        BrokerCredentials.env_prefix("nonsense", ExecTier.DEMO)


def test_from_env_reads_target_broker(monkeypatch) -> None:
    monkeypatch.setenv("FTMO_DEMO_SERVER", "FTMO-Demo")
    monkeypatch.setenv("FTMO_DEMO_LOGIN", "1513568029")
    monkeypatch.setenv("FTMO_DEMO_PASSWORD", "secret")
    creds = BrokerCredentials.from_env("ftmo", ExecTier.DEMO)
    assert creds.broker == "ftmo"
    assert creds.login == 1513568029
    assert creds.server == "FTMO-Demo"
    assert creds.is_demo_server
    assert "secret" not in repr(creds)  # password never echoed


def test_from_env_missing_creds_raises(monkeypatch) -> None:
    for k in ("FUNDEDNEXT_DEMO_SERVER", "FUNDEDNEXT_DEMO_LOGIN", "FUNDEDNEXT_DEMO_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="Missing credentials"):
        BrokerCredentials.from_env("fundednext", ExecTier.DEMO)


def test_from_env_non_integer_login_raises(monkeypatch) -> None:
    monkeypatch.setenv("FTMO_DEMO_SERVER", "FTMO-Demo")
    monkeypatch.setenv("FTMO_DEMO_LOGIN", "not-an-int")
    monkeypatch.setenv("FTMO_DEMO_PASSWORD", "secret")
    with pytest.raises(RuntimeError, match="must be an integer"):
        BrokerCredentials.from_env("ftmo", ExecTier.DEMO)


def test_is_demo_server_false_for_live() -> None:
    live = BrokerCredentials(broker="ftmo", server="FTMO-Server", login=1, password="x")
    assert not live.is_demo_server
