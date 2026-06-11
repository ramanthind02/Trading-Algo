"""Tests for the M3 daily data-platform maintenance chain.

Covers:
  1. run_daily_chain — --only / --skip step ordering, exit-code aggregation,
     failure-continues-chain, notifier-called-on-failure
  2. lib.core.notify CLI entrypoint — exits 0 with logs-only when env unset
  3. register_tasks.ps1 -WhatIf — prints all 5 task rows, registers nothing
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists()
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Fixture: chain module (imported once; bootstrap_runtime runs at import time)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chain_mod():
    import deployment.ops.run_daily_chain as m
    return m


# ---------------------------------------------------------------------------
# 1. run_chain tests
# ---------------------------------------------------------------------------

class TestRunChain:
    """Unit tests for run_chain() + _summarize()."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, chain_mod):
        self._chain = chain_mod
        self._exit_codes: dict[str, int] = {}
        self._subprocess_calls: list[str] = []
        self._alerts: list[str] = []

        # Suppress registry round-trips (no DB in unit tests).
        monkeypatch.setattr(self._chain, "_record_start", lambda *a, **kw: (None, None))
        monkeypatch.setattr(self._chain, "_record_finish", lambda *a, **kw: None)

        # Replace subprocess step runner with a fast stub.
        def _fake_run(name: str, cmd: list[str], **kwargs) -> chain_mod.StepResult:
            self._subprocess_calls.append(name)
            code = self._exit_codes.get(name, 0)
            detail = "mocked" if code == 0 else f"mocked exit {code}"
            return self._chain.StepResult(name, code, 0.01, detail)

        monkeypatch.setattr(self._chain, "_run_step_subprocess", _fake_run)

        # Replace MT5-dependent clock drift step with a no-op.
        monkeypatch.setattr(
            self._chain,
            "step_clock_drift",
            lambda: self._chain.StepResult("clock_drift", 0, 0.0, "mocked"),
        )

        # Capture Telegram alert calls.
        monkeypatch.setattr(self._chain, "_send_alert", self._alerts.append)

    # ── step ordering ──────────────────────────────────────────────────────

    def test_all_steps_run_in_declared_order_by_default(self):
        results = self._chain.run_chain(frozenset(self._chain.ALL_STEPS))
        assert [r.name for r in results] == list(self._chain.ALL_STEPS)

    def test_only_single_step_runs(self):
        results = self._chain.run_chain(frozenset(["scrape"]))
        ran = [r.name for r in results if r.exit_code != -1]
        skipped = [r.name for r in results if r.exit_code == -1]
        assert ran == ["scrape"]
        assert set(skipped) == set(self._chain.ALL_STEPS) - {"scrape"}

    def test_skip_excludes_requested_steps(self):
        active = frozenset(self._chain.ALL_STEPS) - {"scrape", "freshness"}
        results = self._chain.run_chain(active)
        ran = {r.name for r in results if r.exit_code != -1}
        skipped = {r.name for r in results if r.exit_code == -1}
        assert ran == active
        assert skipped == {"scrape", "freshness"}

    # ── exit-code aggregation ──────────────────────────────────────────────

    def test_all_ok_returns_zero(self):
        results = self._chain.run_chain(frozenset(self._chain.ALL_STEPS))
        assert self._chain._summarize(results) == 0

    def test_one_failure_returns_one(self):
        self._exit_codes = {"scrape": 1}
        results = self._chain.run_chain(frozenset(self._chain.ALL_STEPS))
        assert self._chain._summarize(results) == 1

    def test_multiple_failures_returns_one(self):
        self._exit_codes = {"scrape": 1, "registry_backup": 2}
        results = self._chain.run_chain(frozenset(self._chain.ALL_STEPS))
        assert self._chain._summarize(results) == 1

    # ── failure continues chain ────────────────────────────────────────────

    def test_failure_does_not_abort_remaining_steps(self):
        """A failing step b (scrape) must not prevent steps c-e from running."""
        self._exit_codes = {"scrape": 1}
        results = self._chain.run_chain(frozenset(self._chain.ALL_STEPS))
        # Every step should have been attempted, not skipped.
        non_skipped = {r.name for r in results if r.exit_code != -1}
        assert non_skipped == set(self._chain.ALL_STEPS)

    def test_middle_failure_does_not_abort_later_steps(self):
        self._exit_codes = {"signal_refresh": 99}
        results = self._chain.run_chain(frozenset(self._chain.ALL_STEPS))
        non_skipped = {r.name for r in results if r.exit_code != -1}
        assert non_skipped == set(self._chain.ALL_STEPS)

    # ── notifier on failure ────────────────────────────────────────────────

    def test_notifier_called_when_step_fails(self):
        self._exit_codes = {"signal_refresh": 2}
        results = self._chain.run_chain(frozenset(self._chain.ALL_STEPS))
        self._chain._summarize(results)
        assert len(self._alerts) == 1
        assert "signal_refresh" in self._alerts[0]

    def test_notifier_not_called_on_all_ok(self):
        results = self._chain.run_chain(frozenset(self._chain.ALL_STEPS))
        self._chain._summarize(results)
        assert self._alerts == []

    def test_notifier_mentions_all_failed_steps(self):
        self._exit_codes = {"scrape": 1, "registry_backup": 2}
        results = self._chain.run_chain(frozenset(self._chain.ALL_STEPS))
        self._chain._summarize(results)
        assert len(self._alerts) == 1
        assert "scrape" in self._alerts[0]
        assert "registry_backup" in self._alerts[0]


# ---------------------------------------------------------------------------
# 2. CLI --only / --skip parsing
# ---------------------------------------------------------------------------

class TestCliParsing:
    """Tests for the argparse front-door of main()."""

    def _run_main(self, argv: list[str], chain_mod) -> int:
        """Invoke main() with patched argv and a stubbed run_chain."""
        all_ok = [chain_mod.StepResult(s, 0, 0.0, "mocked") for s in chain_mod.ALL_STEPS]
        with patch.object(sys, "argv", ["run_daily_chain"] + argv):
            with patch.object(chain_mod, "run_chain", return_value=all_ok):
                with patch.object(chain_mod, "_send_alert", lambda *a: None):
                    return chain_mod.main()

    def test_no_args_runs_all_steps(self, chain_mod):
        assert self._run_main([], chain_mod) == 0

    def test_only_valid_step(self, chain_mod):
        assert self._run_main(["--only", "scrape"], chain_mod) == 0

    def test_only_invalid_step_returns_2(self, chain_mod):
        with patch.object(sys, "argv", ["run_daily_chain", "--only", "nonexistent"]):
            code = chain_mod.main()
        assert code == 2

    def test_skip_valid_steps(self, chain_mod):
        assert self._run_main(["--skip", "scrape,freshness"], chain_mod) == 0

    def test_skip_invalid_step_returns_2(self, chain_mod):
        with patch.object(sys, "argv", ["run_daily_chain", "--skip", "nonexistent"]):
            code = chain_mod.main()
        assert code == 2

    def test_only_and_skip_are_mutually_exclusive(self, chain_mod):
        """argparse should reject --only and --skip together."""
        proc = subprocess.run(
            [sys.executable, "-m", "deployment.ops.run_daily_chain",
             "--only", "scrape", "--skip", "freshness"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        )
        assert proc.returncode != 0


# ---------------------------------------------------------------------------
# 3. notify CLI entrypoint — log-only when env unset → exit 0
# ---------------------------------------------------------------------------

class TestNotifyEntrypoint:
    def test_exits_zero_with_unset_telegram_env(self):
        """python -m lib.core.notify --channel generic --message test → exit 0."""
        # Strip all known Telegram env vars to guarantee logs-only mode.
        _telegram_vars = {
            "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
            "TELEGRAM_PROP_BOT_TOKEN", "TELEGRAM_PROP_CHAT_ID",
            "TELEGRAM_PERSONAL_BOT_TOKEN", "TELEGRAM_PERSONAL_CHAT_ID",
            "TELEGRAM_CFD_PROP_BOT_TOKEN", "TELEGRAM_CFD_PROP_CHAT_ID",
        }
        clean_env = {k: v for k, v in os.environ.items() if k not in _telegram_vars}
        result = subprocess.run(
            [sys.executable, "-m", "lib.core.notify",
             "--channel", "generic", "--message", "test-chain-notification"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env={**clean_env, "PYTHONPATH": str(_REPO_ROOT)},
        )
        assert result.returncode == 0, (
            f"Expected exit 0 but got {result.returncode}.\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )

    @pytest.mark.parametrize("channel", ["generic", "prop", "personal", "cfd_prop"])
    def test_all_channels_accepted(self, channel):
        """All four channel names are valid CLI arguments."""
        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith("TELEGRAM_")}
        result = subprocess.run(
            [sys.executable, "-m", "lib.core.notify",
             "--channel", channel, "--message", f"test-{channel}"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env={**clean_env, "PYTHONPATH": str(_REPO_ROOT)},
        )
        assert result.returncode == 0, f"channel={channel!r}: {result.stderr!r}"

    def test_missing_message_exits_nonzero(self):
        """--message is required; omitting it should fail with argparse."""
        result = subprocess.run(
            [sys.executable, "-m", "lib.core.notify", "--channel", "generic"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# 4. register_tasks.ps1 -WhatIf — safe preview, prints 5 task rows
# ---------------------------------------------------------------------------

class TestRegisterTasksWhatIf:
    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="PowerShell task scheduler script is Windows-only",
    )
    def test_whatif_exits_zero_and_prints_five_tasks(self):
        """powershell -File register_tasks.ps1 -WhatIf must exit 0 and list all 5 tasks."""
        ps_script = str(_REPO_ROOT / "deployment" / "ops" / "register_tasks.ps1")
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", ps_script,
                "-WhatIf",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(_REPO_ROOT),
            env=dict(os.environ),
            timeout=60,
        )
        assert result.returncode == 0, (
            f"register_tasks.ps1 -WhatIf failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr!r}"
        )
        expected_task_names = [
            "DailyChainAtLogon",
            "DailyChainCatchup",
            "MT5RolloverTickScrape",
            "DataFreshnessCheck",
            "RegistryBackup",
        ]
        for name in expected_task_names:
            assert name in result.stdout, (
                f"Expected task {name!r} in -WhatIf output.\nOutput:\n{result.stdout}"
            )
