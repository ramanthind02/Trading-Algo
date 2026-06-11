#!/usr/bin/env python3
"""PreToolUse guardrail for the agent-driven research pipeline.

Denies ``Edit``/``Write`` to the two canonical research configs and to every vault tree.
This turns two of the README's *hard constraints* from hope into mechanical enforcement
(see ``docs/library/Strategy_research/agent_harness.md`` § Hooks):

* ``research/feature/config.py`` / ``research/portfolio/config.py`` — the adapter must build
  **fresh** config objects; the canonical configs are never mutated.
* repo-root ``vault/`` / ``vault_personal/`` / ``vault_cfd_prop/`` — vault promotion is
  **human-only**; the agent can never auto-promote. Only the explicit ``/research-promote``
  step (a human action) may write there.

It does NOT gate on metrics (no accept/reject) — purely structural.

Mechanism: reads the PreToolUse JSON on stdin, and if the target path is forbidden, prints a
reason to stderr and exits ``2`` (the documented PreToolUse "block" exit code). Any other path
exits ``0`` (allow). The guard fails *open* on malformed input so it never wedges normal work.
"""

from __future__ import annotations

import json
import os
import sys

# Canonical config files the adapter must never edit (matched by path suffix).
FORBIDDEN_FILES = (
    "research/feature/config.py",
    "research/portfolio/config.py",
)
# Repo-root vault data trees (matched as the first path segment under the project root).
FORBIDDEN_ROOT_DIRS = ("vault", "vault_personal", "vault_cfd_prop")


def _block(reason: str) -> None:
    sys.stderr.write(reason + "\n")
    sys.exit(2)


def _project_relative(path: str) -> str:
    """Path relative to the project root (lower-cased, forward slashes)."""

    norm = path.replace("\\", "/").lower()
    root = (os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()).replace("\\", "/").rstrip("/").lower()
    if root and norm.startswith(root + "/"):
        return norm[len(root) + 1 :]
    return norm


def main() -> None:
    try:
        # Decode with utf-8-sig so a stray BOM (some shells add one) is stripped.
        payload = sys.stdin.buffer.read().decode("utf-8-sig")
        data = json.loads(payload)
    except Exception:
        return  # malformed payload → fail open
    tool_input = data.get("tool_input", data) or {}
    raw_path = str(tool_input.get("file_path") or "")
    if not raw_path:
        return

    rel = _project_relative(raw_path)

    for forbidden in FORBIDDEN_FILES:
        if rel.endswith(forbidden):
            _block(
                f"BLOCKED by research guardrail: '{forbidden}' is a canonical research config. "
                "The adapter must build FRESH config objects (research/spec/adapter.py); never "
                "edit the canonical configs. See docs/library/Strategy_research/agent_harness.md."
            )

    first_segment = rel.split("/")[0]
    if first_segment in FORBIDDEN_ROOT_DIRS:
        _block(
            f"BLOCKED by research guardrail: writing under '{first_segment}/' (a vault tree). "
            "Vault promotion is human-only — use the explicit /research-promote command. "
            "See docs/library/Strategy_research/agent_harness.md."
        )


if __name__ == "__main__":
    main()
