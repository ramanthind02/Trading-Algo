# OpenCode Lean Workflow Config Implementation Plan (No Slim)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `oh-my-opencode-slim` completely and keep OpenCode in a lean, direct (Cursor-like) mode.

**Architecture:** Use the native OpenCode config (`~/.config/opencode/opencode.jsonc`) to control plugins + UI diff style + disable extra agents, and use the repo hook (`.opencode/hooks/rag_retriever.py`) with `RAG_MODE` only (default `off`).

**Tech Stack:** OpenCode global config (JSON/JSONC), repo-local OpenCode hook (Python), shared `venv`.

---

### Task 1: Remove Slim plugin

**Files:**
- Modify: `/home/raman/.config/opencode/opencode.jsonc`

**Step 1: Remove `oh-my-opencode-slim` from plugins**

Ensure `plugin` does not include `oh-my-opencode-slim`.

**Step 2: Set unified diff rendering**

Set:

```json
"tui": { "diff_style": "stacked" }
```

**Step 3: Disable extra agents**

Set:

```json
"agent": { "explore": { "disable": true }, "general": { "disable": true } }
```

---

### Task 2: Make RAG manual-only

**Files:**
- Modify: `.opencode/hooks/rag_retriever.py`

**Step 1: Remove preset-based mode detection**

Make `_detect_mode()` rely on `RAG_MODE` only.

**Step 2: Default to `RAG_MODE=off`**

If `RAG_MODE` is unset or unknown, return `off`.

---

### Task 3: Simplify launcher

**Files:**
- Modify: `scripts/oc`

**Step 1: Stop exporting `OH_MY_OPENCODE_SLIM_PRESET`**

Remove provider/preset flags and only support `--rag off|execute|research`.

**Step 2: Default `RAG_MODE=off`**

Make `./scripts/oc` run lean by default.

---

### Task 4: Update docs

**Files:**
- Modify: `docs/library/workflow.md`
- Modify: `docs/library/RAG_memory_system.md`

**Step 1: Remove Slim references**

Replace preset language with `RAG_MODE` / `--rag` guidance.

---

### Task 5: Verify

Run:

```bash
python3 -c 'import json; json.load(open("/home/raman/.config/opencode/opencode.jsonc"))'
source venv/bin/activate
python -m py_compile .opencode/hooks/rag_retriever.py
./scripts/oc -- --version
RAG_MODE=execute python .opencode/hooks/rag_retriever.py "hello" >/dev/null
```
