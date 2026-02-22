#+#+#+#+markdown
# OSOS Model Mapping Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan.

**Goal:** Update repo documentation to reflect the agreed OSOS agent→model mapping with explicit backup models for rate-limit fallback.

**Architecture:** Treat primary models as the default mapping in the docs; treat backup models as operational fallbacks used only when the OpenAI subscription hits rate limits, switching over to Zen API models.

**Tech Stack:** OpenCode + omos (oh-my-opencode-slim) + repo docs under `docs/library/`.

---

### Task 1: Update the explicit model mapping block

**Files:**
- Modify: `docs/library/opencode_multi_agent_workflow_deep.md`

**Step 1: Edit the mapping block**

Replace the current “This setup currently uses:” mapping with:

```text
- orchestrator: openai/gpt-5.2 (backup: opencode/kimi-2.5)
- oracle: openai/gpt-5.2-codex (backup: opencode/kimi-k2-thinking)
- explorer: openai/gpt-5.1-codex-mini (backup: opencode/minimax-2.5)
- librarian: openai/gpt-5.1-codex-mini (backup: opencode/gemini-3-flash)
- fixer: openai/gpt-5.1-codex-max (backup: opencode/qwen3-coder-480b)
```

**Step 2: Sanity-check wording**

Ensure the paragraph immediately around the mapping still matches reality:
- “primary” models are OpenAI
- “backup” models are Zen API models used when rate-limited

**Step 3: Optional verify**

Run: `python -m compileall -q .`
Expected: no errors (docs changes only; this is a cheap sanity check).

### Task 2: Update canonical workflow guide with the same mapping

**Files:**
- Modify: `docs/library/opencode_trading_repo_workflow_guide.md`

**Step 1: Add a short “Current OSOS model mapping” section**

Add a small section near the “Two Layers” / presets discussion that states the same mapping and explicitly notes:
- backups are only used if OpenAI rate limits are hit
- switching is done by updating the preset mapping in `~/.config/opencode/oh-my-opencode-slim.json(.jsonc)`

**Step 2: Optional verify**

Run: `python -m compileall -q .`
Expected: no errors.

### Task 3: (Optional, out-of-repo) Update your local omos preset config

**Files:**
- Modify (user machine): `~/.config/opencode/oh-my-opencode-slim.json(.jsonc)`

**Step 1: Backup the config**

Run:

```bash
mkdir -p "$HOME/.config/opencode/backups"
cp -a "$HOME/.config/opencode/oh-my-opencode-slim.json" \
  "$HOME/.config/opencode/backups/oh-my-opencode-slim.json.$(date +%Y%m%d_%H%M%S)"
```

**Step 2: Update models for each agent in the active preset**

Set each agent’s `model` string to the primary model.

**Step 3: Define a fallback preset (recommended)**

Create a second preset (e.g. `research_zen` / `execute_zen`) that uses the backup models.
Operationally: if you hit rate limits, restart OpenCode with `OH_MY_OPENCODE_SLIM_PRESET=<..._zen>`.

**Step 4: Verify**

Run: `opencode agent list`
Expected: agents still enumerate and respond.
