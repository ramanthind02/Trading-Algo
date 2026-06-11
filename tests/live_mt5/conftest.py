r"""Safety harness + fixtures for the live FTMO-demo adapter tests.

╔══════════════════════════════════════════════════════════════════════════╗
║  SAFETY MODEL — read before touching anything in this directory           ║
╠══════════════════════════════════════════════════════════════════════════╣
║ 1. ONE TERMINAL PER PROCESS. The MetaTrader5 bridge drives a single        ║
║    terminal per OS process. Every fixture here binds via                   ║
║    mt5.initialize(path=<FTMO terminal64.exe>, login, server, password), so ║
║    THIS pytest process talks only to the FTMO terminal. The Darwinex       ║
║    tick-scraper runs in its own process and is never touched.              ║
║ 2. HARD GUARD. Before yielding, ftmo_session asserts the bound account is  ║
║    the FTMO demo login, on the FTMO server, with trade_mode == DEMO.       ║
║    Any mismatch → pytest.exit() (whole session aborts) — nothing           ║
║    downstream can place an order.                                          ║
║ 3. DEDICATED MAGIC. Order tiers tag everything with TEST_MAGIC (≠ the      ║
║    adapter's production magic 510, ≠ manual trades). Cleanup flattens      ║
║    ONLY that magic.                                                        ║
║ 4. OPT-IN GATES. Skipped unless MT5_LIVE_TESTS=1; order tiers also need    ║
║    MT5_LIVE_ORDERS=1. So `pytest tests/` stays green (skips) in CI.        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ── Make the vendored adapter importable (it is sys.path-based, not installed) ──
_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER = _ROOT / "deployment" / "nautilus_mt5" / "vendor" / "mt5-connect"
if str(_ADAPTER) not in sys.path:
    sys.path.insert(0, str(_ADAPTER))

# ── Load gitignored .env (creds) without clobbering already-set env ────────────
_ENV = _ROOT / ".env"
if _ENV.exists():
    for _line in _ENV.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_FTMO_TERMINAL_PATH = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
FTMO_TERMINAL_PATH = os.environ.get("FTMO_DEMO_TERMINAL_PATH", DEFAULT_FTMO_TERMINAL_PATH)

# Distinct from the adapter's production MT5_MAGIC_NUMBER (510) and from any
# manual trade — so flatten_test_magic() can only ever touch our own orders.
TEST_MAGIC = 990510

# Symbols used by the live tests (FTMO native names — confirmed in
# configs/mt5_brokers.yaml). EURUSD = FX (CurrencyPair), US100.cash = index CFD.
ORDER_SYMBOL = "EURUSD"
INDEX_SYMBOL = "US100.cash"


# ── Gate helpers ───────────────────────────────────────────────────────────────

def _creds() -> tuple[int, str, str] | None:
    """Return (login, server, password) from .env, or None if incomplete."""
    try:
        login = int(os.environ["FTMO_DEMO_LOGIN"])
        server = os.environ["FTMO_DEMO_SERVER"]
        password = os.environ["FTMO_DEMO_PASSWORD"]
    except (KeyError, ValueError):
        return None
    return login, server, password


def _live_disabled_reason() -> str | None:
    """Return a skip reason if the live suite cannot/should-not run, else None."""
    if os.environ.get("MT5_LIVE_TESTS") != "1":
        return "live MT5 tests disabled (set MT5_LIVE_TESTS=1 to enable)"
    if sys.platform != "win32":
        return "MetaTrader5 is Windows-only"
    try:
        import MetaTrader5  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guard
        return f"MetaTrader5 package not importable: {exc}"
    if _creds() is None:
        return "FTMO_DEMO_{LOGIN,SERVER,PASSWORD} missing/invalid in .env"
    if not Path(FTMO_TERMINAL_PATH).exists():
        return (
            f"FTMO terminal not found at {FTMO_TERMINAL_PATH!r} "
            "(set FTMO_DEMO_TERMINAL_PATH)"
        )
    return None


@pytest.fixture(scope="session", autouse=True)
def live_gate():
    """Autouse session gate — skips the entire directory unless live is enabled."""
    reason = _live_disabled_reason()
    if reason:
        pytest.skip(reason, allow_module_level=False)
    yield


@pytest.fixture
def orders_enabled():
    """Skip an order-placing test unless MT5_LIVE_ORDERS=1 is also set."""
    if os.environ.get("MT5_LIVE_ORDERS") != "1":
        pytest.skip("order-placing tests disabled (set MT5_LIVE_ORDERS=1 to enable)")


# ── Filling-mode + cleanup helpers (module-level so tests can reuse) ────────────

def supported_filling(mt5, symbol: str) -> int:
    """Pick a filling mode the symbol actually supports (avoids retcode 10030)."""
    info = mt5.symbol_info(symbol)
    mask = getattr(info, "filling_mode", 0) if info else 0
    if mask & 2:
        return mt5.ORDER_FILLING_IOC
    if mask & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def flatten_test_magic(mt5) -> int:
    """Close every position and cancel every pending order tagged TEST_MAGIC.

    Returns the number of close/cancel requests sent. Safe to call repeatedly;
    only ever touches TEST_MAGIC, never manual or production (510) orders.
    """
    sent = 0
    for p in (mt5.positions_get() or ()):
        if p.magic != TEST_MAGIC:
            continue
        tick = mt5.symbol_info_tick(p.symbol)
        if tick is None:
            continue
        if p.type == mt5.ORDER_TYPE_BUY:
            close_type, price = mt5.ORDER_TYPE_SELL, tick.bid
        else:
            close_type, price = mt5.ORDER_TYPE_BUY, tick.ask
        mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": close_type,
            "position": p.ticket,
            "price": price,
            "deviation": 50,
            "magic": TEST_MAGIC,
            "comment": "live-test-cleanup",
            "type_filling": supported_filling(mt5, p.symbol),
        })
        sent += 1
    for o in (mt5.orders_get() or ()):
        if o.magic != TEST_MAGIC:
            continue
        mt5.order_send({
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": o.ticket,
            "comment": "live-test-cleanup",
        })
        sent += 1
    return sent


# ── Session fixture: bind FTMO terminal + hard guard ───────────────────────────

class FtmoSession:
    """Lightweight handle yielded to tests — identity of the bound account."""

    def __init__(self, login: int, server: str, password: str, path: str):
        self.login = login
        self.server = server
        self.password = password
        self.path = path


@pytest.fixture(scope="session")
def ftmo_session(live_gate):
    """Bind this process to the FTMO terminal, assert it's the demo, then yield.

    Teardown flattens any TEST_MAGIC residue and shuts the IPC handle (the
    terminal process itself keeps running).
    """
    import MetaTrader5 as mt5

    login, server, password = _creds()  # gate guarantees not None

    ok = mt5.initialize(path=FTMO_TERMINAL_PATH, login=login, server=server,
                        password=password, timeout=20_000)
    if not ok:
        code, msg = mt5.last_error()
        pytest.exit(f"mt5.initialize(path=FTMO) failed — error {code}: {msg}", returncode=2)

    # ── HARD GUARD: refuse to proceed unless this is EXACTLY the FTMO demo ──────
    info = mt5.account_info()
    if info is None:
        mt5.shutdown()
        pytest.exit("account_info() is None after initialize — aborting", returncode=2)

    problems = []
    if int(info.login) != login:
        problems.append(f"login {info.login} != expected FTMO {login}")
    if str(info.server) != server:
        problems.append(f"server {info.server!r} != expected {server!r}")
    if int(info.trade_mode) != int(mt5.ACCOUNT_TRADE_MODE_DEMO):
        problems.append(f"trade_mode {info.trade_mode} != DEMO")
    if problems:
        mt5.shutdown()
        pytest.exit(
            "FTMO DEMO GUARD FAILED — refusing to run live tests: "
            + "; ".join(problems),
            returncode=2,
        )

    # Bound, verified demo. Clean any leftover test residue before starting.
    flatten_test_magic(mt5)

    yield FtmoSession(login, server, password, FTMO_TERMINAL_PATH)

    # Teardown: never leave the account holding a test position.
    flatten_test_magic(mt5)
    mt5.shutdown()


# ── Config + connection helpers ────────────────────────────────────────────────

def make_ftmo_config(ftmo_session: FtmoSession, symbols, magic: int = TEST_MAGIC):
    """Build a real MT5Config pinned to the FTMO terminal + TEST_MAGIC."""
    from mt5connect.config import MT5Config
    return MT5Config(
        account=ftmo_session.login,
        password=ftmo_session.password,
        server=ftmo_session.server,
        symbols=list(symbols),
        path=ftmo_session.path,
        magic_number=magic,
        poll_interval_ms=100,
        exec_poll_interval_ms=100,
        reconnect_initial_delay_s=0.5,
        reconnect_max_delay_s=2.0,
        reconnect_max_attempts=3,
        timeout_s=20.0,
    )


class LiveKit:
    """Everything a live test needs, in one object — avoids importing helpers
    out of conftest. Access via the ``live`` fixture."""

    TEST_MAGIC = TEST_MAGIC
    ORDER_SYMBOL = ORDER_SYMBOL
    INDEX_SYMBOL = INDEX_SYMBOL

    def __init__(self, mt5, session: FtmoSession):
        self.mt5 = mt5
        self.session = session

    def make_config(self, symbols, magic: int = TEST_MAGIC):
        return make_ftmo_config(self.session, symbols, magic=magic)

    def supported_filling(self, symbol: str) -> int:
        return supported_filling(self.mt5, symbol)

    def flatten(self) -> int:
        return flatten_test_magic(self.mt5)

    def market_open(self, symbol: str, max_age_s: float = 120.0) -> bool:
        """True if ``symbol`` has a LIVE feed (a fresh tick).

        This is the only reliable open-market signal: ``order_check`` returns
        "Done" even on a closed market, and ``trade_mode`` ignores sessions —
        but on a closed market the last tick is hours stale. During open hours
        a liquid symbol like EURUSD ticks sub-second, so a 120s threshold
        cleanly separates open from closed.
        """
        import datetime as _dt
        self.mt5.symbol_select(symbol, True)
        t = self.mt5.symbol_info_tick(symbol)
        if t is None or not t.bid:
            return False
        age = _dt.datetime.now(_dt.timezone.utc).timestamp() - int(t.time)
        return age <= max_age_s

    def require_market_open(self, symbol: str) -> None:
        """Skip the current test unless ``symbol`` has a live feed."""
        if not self.market_open(symbol):
            pytest.skip(f"{symbol} market closed (weekend/holiday) — needs a live feed")


@pytest.fixture
def live(ftmo_session):
    """Bundle: the bound mt5 module, session identity, config factory, cleanup."""
    import MetaTrader5 as mt5
    return LiveKit(mt5, ftmo_session)


@pytest.fixture
def ftmo_conn(ftmo_session):
    """A connected real MT5Connection (adapter's own connect path, path-bound)."""
    from mt5connect.connection import MT5Connection
    conn = MT5Connection(make_ftmo_config(ftmo_session, [ORDER_SYMBOL]))
    conn.connect()
    yield conn
    conn.disconnect()


@pytest.fixture
def nt_components():
    """Real NautilusTrader (msgbus, cache, clock) — MagicMock fails PyConditions."""
    from nautilus_trader.common.component import LiveClock
    from nautilus_trader.test_kit.stubs.component import TestComponentStubs
    return TestComponentStubs.msgbus(), TestComponentStubs.cache(), LiveClock()
