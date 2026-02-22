# OpenCode Launcher Scripts (OpenAI vs Zen) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a repo-local launcher `scripts/oc` so users can start OpenCode with `--provider openai|zen` and `--preset execute|research|dynamic` instead of manually setting env vars.

**Architecture:** A small POSIX shell wrapper that maps flags to `OH_MY_OPENCODE_SLIM_PRESET` (`execute` vs `execute_zen`, etc), optionally sets `RAG_MODE`, and then `exec`s `opencode` with remaining arguments unchanged.

**Tech Stack:** POSIX `sh`, OpenCode CLI, omos presets configured in `~/.config/opencode/oh-my-opencode-slim.json`.

---

### Task 1: Create launcher script

**Files:**
- Create: `scripts/oc`

**Step 1: Add `scripts/oc` with flag parsing and help text**

Implement flags:

- `--provider openai|zen` (default: `openai`)
- `--preset execute|research|dynamic` (default: `execute`)
- `--rag off|execute|research` (optional; sets `RAG_MODE`)
- `--help` prints usage
- `--` ends wrapper parsing; everything after goes to `opencode`

Mapping:

- provider=openai -> `OH_MY_OPENCODE_SLIM_PRESET={dynamic|research|execute}`
- provider=zen -> `OH_MY_OPENCODE_SLIM_PRESET={dynamic_zen|research_zen|execute_zen}`

**Step 2: Make it executable**

Run: `chmod +x scripts/oc`
Expected: no output

### Task 2: Update workflow docs to recommend the launcher

**Files:**
- Modify: `docs/library/opencode_trading_repo_workflow_guide.md`

**Step 1: Add a short “Launcher script” section near presets**

Include examples:

```bash
./scripts/oc --provider openai --preset execute
./scripts/oc --provider zen --preset execute
./scripts/oc --provider zen --preset research
```

Mention that it just selects `OH_MY_OPENCODE_SLIM_PRESET` for you.

### Task 3: Verification

**Step 1: Smoke-test help output**

Run: `./scripts/oc --help`
Expected: usage text prints and exit code 0

**Step 2: Smoke-test env mapping without launching chat**

Run:

```bash
OH_MY_OPENCODE_SLIM_PRESET= ./scripts/oc --provider zen --preset execute -- --version
```

Expected: `opencode --version` prints version; wrapper does not error.

**Step 3: Optional: ensure `opencode agent list` still works**

Run: `opencode agent list`
Expected: prints available primary agents.
