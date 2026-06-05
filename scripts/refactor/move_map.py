"""WP-8 restructure move-map, encoded as data.

Single source of truth for the approved "consolidate flat top-level" restructure.
Transcribed verbatim from ``docs/refactor/nautilus/wp8_move_map.md`` (§1, §2, §6).

This module performs **no** I/O and mutates nothing. It is imported by
``rewrite_imports.py`` (for the import-prefix rename rules) and ``run_moves.py``
(for the ``git mv`` path moves, in batch order).

Conventions
-----------
* ``IMPORT_RENAMES`` — ordered list of ``ImportRename`` (longest-prefix-first).
  Applied as dotted-prefix renames by the codemod. The optional ``nodes.`` ->
  ``signals.`` rule (R15) is the only ``optional=True`` entry and is gated OFF by
  default behind ``--rename-nodes`` in the codemod.
* ``PATH_MOVES`` — flat list of ``PathMove`` (old_path -> new_path), each tagged
  with the batch it belongs to. ``run_moves.py`` filters by batch.
* ``BATCH_ORDER`` — leaf-first batch order from §6. Optional batches (B11
  prop_firms, B12 nodes->signals) are present but flagged.

All paths are repo-relative, POSIX-style (forward slashes); ``run_moves.py``
resolves them against the repo root.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Import-prefix rename rules (§1, "the codemod's rename rules")
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ImportRename:
    """A single dotted-prefix rename rule for the import codemod."""

    rule_id: str
    old_prefix: str
    new_prefix: str
    optional: bool = False


# Ordered LONGEST-PREFIX-FIRST so sub-splits (utils.evaluation, the
# feature_selection.* splits) land before any shorter parent rule. R4
# (prop_firms) is omitted by default — the plan recommends keeping prop_firms/
# top-level (see §2.4 / Open decision #1). It is listed here commented for the
# record; flip OPTIONAL_PROP_FIRMS to include it.
IMPORT_RENAMES: list[ImportRename] = [
    # --- feature_selection.* sub-splits (R6, R7, R8) before any utils rule ---
    ImportRename("R6", "feature_selection.base_models.", "features.models."),
    ImportRename("R7", "feature_selection.eda.", "features.eda."),
    ImportRename("R8", "feature_selection.validation.", "features.validation."),
    # --- feature_extraction (R5) ---
    ImportRename("R5", "feature_extraction.", "features.extraction."),
    # --- utils.evaluation diverges to research/, NOT lib/ (R3) — before R9-R14 ---
    ImportRename("R3", "utils.evaluation.", "research.evaluation."),
    # --- utils.* subtree splits (R9, R10, R11, R12, R13) ---
    ImportRename("R9", "utils.core.", "lib.core."),
    ImportRename("R10", "utils.cache.", "lib.cache."),
    ImportRename("R11", "utils.compute.", "lib.compute."),
    ImportRename("R12", "utils.dev.", "tools.dev."),
    ImportRename("R13", "utils.research_workspace.", "tools.research_workspace."),
    # --- loose utils/*.py modules -> lib.core (R14). These are full-module
    #     renames (no trailing dot) because they are leaf modules, not packages.
    #     They MUST precede any bare "utils." parent rule (there is none here). ---
    ImportRename("R14a", "utils.vault_paths", "lib.core.vault_paths"),
    ImportRename("R14b", "utils.repo_bootstrap", "lib.core.repo_bootstrap"),
    ImportRename("R14c", "utils.futures_micro_specs", "lib.core.futures_micro_specs"),
    # --- research packages (R1, R2). Depend on R3 already being relocated. ---
    ImportRename("R2", "portfolio_research.", "research.portfolio."),
    ImportRename("R1", "feature_research.", "research.feature."),
    # --- OPTIONAL: nodes -> signals (R15). Gated OFF by default. ---
    ImportRename("R15", "nodes.", "signals.", optional=True),
]

# Optional R4 (prop_firms -> research.prop_firms). Default: NOT applied (keep
# prop_firms/ top-level). Exposed so callers can opt in explicitly.
OPTIONAL_PROP_FIRMS: ImportRename = ImportRename(
    "R4", "prop_firms.", "research.prop_firms.", optional=True
)


def active_import_renames(
    *, rename_nodes: bool = False, rename_prop_firms: bool = False
) -> list[ImportRename]:
    """Return the rename rules to apply, honoring the optional flags.

    ``nodes.`` -> ``signals.`` (R15) is included only when ``rename_nodes`` is
    True. ``prop_firms.`` -> ``research.prop_firms.`` (R4) only when
    ``rename_prop_firms`` is True. Order is preserved (longest-prefix-first);
    R4, when enabled, is inserted ahead of the research package rules.
    """
    rules: list[ImportRename] = []
    for rule in IMPORT_RENAMES:
        if rule.rule_id == "R15" and not rename_nodes:
            continue
        # Insert R4 just before R2/R1 (the research.* rules) when enabled.
        if rule.rule_id == "R2" and rename_prop_firms:
            rules.append(OPTIONAL_PROP_FIRMS)
        rules.append(rule)
    return rules


# --------------------------------------------------------------------------- #
# Path moves (§2). batch == one of the BATCH_ORDER names.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PathMove:
    """A single ``git mv`` of a file or directory subtree."""

    old_path: str
    new_path: str
    batch: str
    is_dir: bool = False
    optional: bool = False
    note: str = ""


# Whole-subtree moves are expressed as directory moves (is_dir=True): run_moves
# performs one ``git mv <old_dir> <new_dir>`` which carries every file beneath.
# This matches the doc's "git mv feature_research research/feature" style and
# preserves history per-file.
PATH_MOVES: list[PathMove] = [
    # --- B1: scripts/_*.py -> scripts/dev/ (kept probes only) (§2.8) ---
    # Per Phase 1, each _*.py is keep-or-cull; only KEPT probes move. The doc
    # lists 6 candidates; resolution against the working tree is done by
    # run_moves (missing ones are reported, not moved).
    PathMove("scripts/_bootstrap.py", "scripts/dev/_bootstrap.py", "B1",
             note="referenced by tests as scripts._bootstrap — verify before moving"),
    PathMove("scripts/_check_catalog.py", "scripts/dev/_check_catalog.py", "B1"),
    PathMove("scripts/_check_ndx_data.py", "scripts/dev/_check_ndx_data.py", "B1"),
    PathMove("scripts/_etf_position_sizing.py", "scripts/dev/_etf_position_sizing.py", "B1"),
    PathMove("scripts/_mt5_m1_sizing.py", "scripts/dev/_mt5_m1_sizing.py", "B1"),
    PathMove("scripts/_tick_download_schedule.py", "scripts/dev/_tick_download_schedule.py", "B1"),

    # --- B2: feature_extraction/ -> features/extraction/ (R5) (§2.5) ---
    # feature_extraction has a single live module; move the file and create the
    # new package __init__ as a fixup step (see README / §5 step 7).
    PathMove("feature_extraction/feature_extractor.py",
             "features/extraction/feature_extractor.py", "B2",
             note="add features/extraction/__init__.py as fixup"),

    # --- B3: feature_selection/{base_models,eda,validation} -> features/* (R6-R8) (§2.5) ---
    # base_models: only the live surface moves; base_model.py stubs are CULLED
    # in Phase 1 (B0) and must NOT be moved.
    PathMove("feature_selection/base_models/__init__.py", "features/models/__init__.py", "B3",
             note="confirm __getattr__ lazy-alias shim for culled stubs is removed, not moved"),
    PathMove("feature_selection/base_models/feature_base_model.py",
             "features/models/feature_base_model.py", "B3"),
    PathMove("feature_selection/eda", "features/eda", "B3", is_dir=True),
    PathMove("feature_selection/validation", "features/validation", "B3", is_dir=True),

    # --- B4: utils/research_workspace -> tools/; utils/dev -> tools/ (R12, R13) (§2.6) ---
    PathMove("utils/research_workspace", "tools/research_workspace", "B4", is_dir=True),
    PathMove("utils/dev", "tools/dev", "B4", is_dir=True),

    # --- B5: utils/evaluation/ -> research/evaluation/ (R3) (§2.3) ---
    PathMove("utils/evaluation", "research/evaluation", "B5", is_dir=True),

    # --- B6: utils/compute/ -> lib/compute/ (R11, L6 cython) (§2.6) ---
    PathMove("utils/compute", "lib/compute", "B6", is_dir=True,
             note="L6: fix setup_cython.py Extension name=/sources=; recompile in place"),

    # --- B7: utils/cache/ -> lib/cache/ (R10) (§2.6) ---
    PathMove("utils/cache", "lib/cache", "B7", is_dir=True),

    # --- B8: utils/core/ -> lib/core/ + loose utils/*.py -> lib/core/ (R9, R14) (§2.6) ---
    PathMove("utils/core", "lib/core", "B8", is_dir=True),
    PathMove("utils/vault_paths.py", "lib/core/vault_paths.py", "B8"),
    PathMove("utils/repo_bootstrap.py", "lib/core/repo_bootstrap.py", "B8"),
    PathMove("utils/futures_micro_specs.py", "lib/core/futures_micro_specs.py", "B8"),
    # utils/__init__.py: drop after the split (no utils namespace remains). Left
    # out of PATH_MOVES intentionally — it is a deletion decision (Open #4), not
    # a move. utils/simulation/* is CULLED in Phase 1 (not moved).

    # --- B9: portfolio_research/ -> research/portfolio/ (R2) (§2.2) ---
    PathMove("portfolio_research", "research/portfolio", "B9", is_dir=True),

    # --- B10: feature_research/ -> research/feature/ (R1, L4, L5, L8) (§2.1) ---
    PathMove("feature_research", "research/feature", "B10", is_dir=True,
             note="L4 _LEGACY_MODULE_ALIASES string values; L5 UI-emitted -m strings"),

    # --- B11: OPTIONAL prop_firms/ -> research/prop_firms/ (R4) (§2.4) ---
    PathMove("prop_firms", "research/prop_firms", "B11", is_dir=True, optional=True,
             note="DEFAULT SKIP — keep prop_firms/ top-level"),

    # --- B12: OPTIONAL FINAL nodes/ -> signals/ (R15, L1, L2) (§2.7) ---
    PathMove("nodes", "signals", "B12", is_dir=True, optional=True,
             note="270 _taxonomy.py CANONICAL_MODULE_IMPORTS string values + nodes.* patch strings"),
]


# --------------------------------------------------------------------------- #
# Batch order (§6, leaf-first). B0 (cull) and B13 (docs/tests mirror) are not
# move batches handled by this tooling; included for completeness/ordering.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Batch:
    name: str
    summary: str
    optional: bool = False
    has_moves: bool = True


BATCH_ORDER: list[Batch] = [
    Batch("B0", "Phase 1 cull (prereq) — not a move batch", has_moves=False),
    Batch("B1", "scripts/_*.py -> scripts/dev/ (kept probes only)"),
    Batch("B2", "feature_extraction/ -> features/extraction/ (R5)"),
    Batch("B3", "feature_selection/{base_models,eda,validation} -> features/* (R6-R8)"),
    Batch("B4", "utils/research_workspace + utils/dev -> tools/ (R12,R13)"),
    Batch("B5", "utils/evaluation/ -> research/evaluation/ (R3)"),
    Batch("B6", "utils/compute/ -> lib/compute/ (R11, cython L6)"),
    Batch("B7", "utils/cache/ -> lib/cache/ (R10)"),
    Batch("B8", "utils/core/ + loose utils/*.py -> lib/core/ (R9,R14)"),
    Batch("B9", "portfolio_research/ -> research/portfolio/ (R2)"),
    Batch("B10", "feature_research/ -> research/feature/ (R1,L4,L5,L8)"),
    Batch("B11", "OPTIONAL prop_firms/ -> research/prop_firms/ (R4)", optional=True),
    Batch("B12", "OPTIONAL FINAL nodes/ -> signals/ (R15,L1,L2)", optional=True),
    Batch("B13", "Mirror tests/ + Phase-3 docs repoint — not a move batch", has_moves=False),
]

BATCH_NAMES: list[str] = [b.name for b in BATCH_ORDER]


# --------------------------------------------------------------------------- #
# Dynamic-import landmines (§4). Used by rewrite_imports.py Pass B to scope
# string-literal rewrites to call arguments only.
# --------------------------------------------------------------------------- #
# Function names whose STRING arguments may contain dotted module paths that the
# codemod must rewrite (L1, L4, L8). Matched on the call's attribute/name tail,
# so both ``patch(...)`` and ``mock.patch(...)`` / ``mocker.patch(...)`` hit.
DYNAMIC_IMPORT_CALL_NAMES: frozenset[str] = frozenset(
    {
        "import_module",   # importlib.import_module
        "__import__",
        "find_spec",       # importlib.util.find_spec
        "patch",           # mock.patch / unittest.mock.patch / mocker.patch
        "object",          # patch.object  (call tail is "object")
        "setattr",         # monkeypatch.setattr / setattr
    }
)

# The _taxonomy.py CANONICAL_MODULE_IMPORTS dict — its VALUES (dotted "nodes...."
# strings) get rewritten only on the R15 (nodes->signals) batch (L2). KEYS are
# never touched (vault JSON contract). Both old and new file locations listed so
# Pass C resolves whichever exists at run time.
TAXONOMY_FILES: tuple[str, ...] = ("nodes/_taxonomy.py", "signals/_taxonomy.py")
TAXONOMY_DICT_NAME: str = "CANONICAL_MODULE_IMPORTS"


def all_old_paths(*, include_optional: bool = True) -> list[str]:
    """Return every old_path in PATH_MOVES (for existence validation)."""
    return [
        m.old_path
        for m in PATH_MOVES
        if include_optional or not m.optional
    ]
