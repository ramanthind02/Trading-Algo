"""WP-8 import-rewrite codemod (libcst-based, move-map-driven).

Applies the dotted-prefix renames from ``move_map.IMPORT_RENAMES`` across every
``.py`` file in the repo. Implements the four passes from
``wp8_move_map.md`` §5:

* Pass A — ``Import`` / ``ImportFrom`` statement module names.
* Pass B — dynamic STRING literals (L1/L4/L8): string args of
  ``import_module`` / ``__import__`` / ``find_spec`` / ``patch`` /
  ``patch.object`` / ``monkeypatch.setattr`` / ``setattr`` whose value is, or
  starts with, ``<moved-prefix>.``.
* Pass C — ``nodes/_taxonomy.py`` ``CANONICAL_MODULE_IMPORTS`` dict VALUES
  (``nodes.`` -> ``signals.``), keys left untouched. Only on the nodes->signals
  batch (``--rename-nodes``).

Modes
-----
``--dry-run``  report per-file / per-prefix counts, change NOTHING (default).
``--apply``    rewrite files in place.

Flags
-----
``--rename-nodes``       include the optional ``nodes.`` -> ``signals.`` rule
                         (R15) and enable Pass C. Default OFF.
``--rename-prop-firms``  include the optional ``prop_firms.`` rule (R4). OFF.
``--root PATH``          repo root (default: auto-detected from this file).
``--path PATH``          restrict to a sub-path (repeatable) — useful for
                         per-batch application.

libcst
------
The transform requires ``libcst``. If it is not installed, ``--apply`` refuses
to run. ``--dry-run`` still produces accurate counts via a libcst-free
tokenless regex scanner that mirrors the same prefix rules (Pass A/B), so the
operator can plan even before installing libcst. The dry-run scanner is a
COUNTER ONLY; it never writes.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Make ``move_map`` importable whether run as a module or a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import move_map as mm  # noqa: E402

try:  # libcst is required only for --apply (and the precise dry-run path).
    import libcst as cst  # type: ignore

    _HAS_LIBCST = True
except ImportError:  # pragma: no cover - environment dependent
    cst = None  # type: ignore
    _HAS_LIBCST = False


# --------------------------------------------------------------------------- #
# Shared prefix-matching helpers (used by both libcst and regex paths)
# --------------------------------------------------------------------------- #
def _rename_dotted(dotted: str, renames: list[mm.ImportRename]) -> str | None:
    """Return rewritten dotted path if any rule matches, else None.

    A rule matches when the dotted string equals the prefix (sans trailing dot)
    or starts with the prefix. Longest-prefix-first ordering of ``renames``
    guarantees the most specific rule wins. The first match short-circuits.
    """
    for rule in renames:
        pref = rule.old_prefix
        if pref.endswith("."):
            base = pref[:-1]
            if dotted == base:
                return rule.new_prefix[:-1]
            if dotted.startswith(pref):
                return rule.new_prefix + dotted[len(pref):]
        else:  # full-module rename (R14 loose modules)
            if dotted == pref:
                return rule.new_prefix
            if dotted.startswith(pref + "."):
                return rule.new_prefix + dotted[len(pref):]
    return None


@dataclass
class FileReport:
    path: str
    import_rewrites: int = 0
    string_rewrites: int = 0
    taxonomy_rewrites: int = 0

    @property
    def total(self) -> int:
        return self.import_rewrites + self.string_rewrites + self.taxonomy_rewrites


# --------------------------------------------------------------------------- #
# libcst transformer (Pass A + B + C)
# --------------------------------------------------------------------------- #
if _HAS_LIBCST:

    class _Codemod(cst.CSTTransformer):  # type: ignore[misc]
        def __init__(
            self,
            renames: list[mm.ImportRename],
            *,
            is_taxonomy: bool,
            rename_nodes: bool,
        ) -> None:
            self.renames = renames
            self.is_taxonomy = is_taxonomy
            self.rename_nodes = rename_nodes
            self.report = FileReport(path="")
            self._call_depth_ok: list[bool] = []
            self._in_taxonomy_value = 0

        # ---- Pass A: import statements ----
        def _attr_to_dotted(self, node: cst.BaseExpression) -> str | None:
            if isinstance(node, cst.Name):
                return node.value
            if isinstance(node, cst.Attribute):
                base = self._attr_to_dotted(node.value)
                return f"{base}.{node.attr.value}" if base else None
            return None

        def _dotted_to_attr(self, dotted: str) -> cst.BaseExpression:
            parts = dotted.split(".")
            node: cst.BaseExpression = cst.Name(parts[0])
            for part in parts[1:]:
                node = cst.Attribute(value=node, attr=cst.Name(part))
            return node

        def leave_ImportFrom(self, original, updated):  # type: ignore[no-untyped-def]
            if updated.module is None:  # relative-only import
                return updated
            dotted = self._attr_to_dotted(updated.module)
            if dotted is None:
                return updated
            new = _rename_dotted(dotted, self.renames)
            if new is None:
                return updated
            self.report.import_rewrites += 1
            return updated.with_changes(module=self._dotted_to_attr(new))

        def leave_Import(self, original, updated):  # type: ignore[no-untyped-def]
            changed = False
            new_names = []
            for alias in updated.names:
                dotted = self._attr_to_dotted(alias.name)
                if dotted is not None:
                    new = _rename_dotted(dotted, self.renames)
                    if new is not None:
                        changed = True
                        self.report.import_rewrites += 1
                        alias = alias.with_changes(name=self._dotted_to_attr(new))
                new_names.append(alias)
            return updated.with_changes(names=new_names) if changed else updated

        # ---- Pass B: dynamic string args, scoped to known call names ----
        def visit_Call(self, node: cst.Call) -> None:  # type: ignore[no-untyped-def]
            func = node.func
            tail: str | None = None
            if isinstance(func, cst.Name):
                tail = func.value
            elif isinstance(func, cst.Attribute):
                tail = func.attr.value
            self._call_depth_ok.append(tail in mm.DYNAMIC_IMPORT_CALL_NAMES)

        def leave_Call(self, original, updated):  # type: ignore[no-untyped-def]
            self._call_depth_ok.pop()
            return updated

        # ---- Pass C: taxonomy dict value detection ----
        def visit_DictElement(self, node) -> None:  # type: ignore[no-untyped-def]
            # We treat any SimpleString value inside the taxonomy file as a
            # candidate; the nodes->signals rule only fires when rename_nodes.
            if self.is_taxonomy:
                self._in_taxonomy_value += 1

        def leave_DictElement(self, original, updated):  # type: ignore[no-untyped-def]
            if self.is_taxonomy and self._in_taxonomy_value > 0:
                self._in_taxonomy_value -= 1
            return updated

        def leave_SimpleString(self, original, updated):  # type: ignore[no-untyped-def]
            raw = updated.value
            quote = raw[-1]
            inner = updated.evaluated_value
            if not isinstance(inner, str) or not inner:
                return updated

            in_dyn_call = bool(self._call_depth_ok) and self._call_depth_ok[-1]
            in_taxonomy_value = self.is_taxonomy and self._in_taxonomy_value > 0

            if not (in_dyn_call or in_taxonomy_value):
                return updated

            new_inner = _rename_dotted(inner, self.renames)
            if new_inner is None:
                return updated
            if in_taxonomy_value:
                self.report.taxonomy_rewrites += 1
            else:
                self.report.string_rewrites += 1
            return updated.with_changes(value=f"{quote}{new_inner}{quote}")


def _apply_libcst(
    text: str,
    renames: list[mm.ImportRename],
    *,
    is_taxonomy: bool,
    rename_nodes: bool,
    path: str,
) -> tuple[str, FileReport]:
    module = cst.parse_module(text)
    tx = _Codemod(renames, is_taxonomy=is_taxonomy, rename_nodes=rename_nodes)
    tx.report.path = path
    new_module = module.visit(tx)
    return new_module.code, tx.report


# --------------------------------------------------------------------------- #
# libcst-free dry-run scanner (COUNTER ONLY — never writes)
# --------------------------------------------------------------------------- #
_IMPORT_FROM_RE = re.compile(r"^\s*from\s+([\w.]+)\s+import\b")
_IMPORT_RE = re.compile(r"^\s*import\s+([\w.]+(?:\s*,\s*[\w.]+)*)")
_DYN_CALL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(n) for n in sorted(mm.DYNAMIC_IMPORT_CALL_NAMES))
    + r")\s*\(\s*(['\"])([\w.]+)\1"
)
_STRING_LITERAL_RE = re.compile(r"(['\"])([\w.]+)\1")


def _scan_regex(
    text: str,
    renames: list[mm.ImportRename],
    *,
    is_taxonomy: bool,
    path: str,
) -> FileReport:
    rep = FileReport(path=path)
    for line in text.splitlines():
        m = _IMPORT_FROM_RE.match(line)
        if m and _rename_dotted(m.group(1), renames) is not None:
            rep.import_rewrites += 1
            continue
        m = _IMPORT_RE.match(line)
        if m:
            for mod in re.split(r"\s*,\s*", m.group(1)):
                mod = mod.split(" as ")[0].strip()
                if _rename_dotted(mod, renames) is not None:
                    rep.import_rewrites += 1
            continue
        for _, dotted in _DYN_CALL_RE.findall(line):
            if _rename_dotted(dotted, renames) is not None:
                rep.string_rewrites += 1
    if is_taxonomy:
        # Count dotted string literals whose value renames — approximates the
        # CANONICAL_MODULE_IMPORTS values (keys are short names that never match).
        for _, dotted in _STRING_LITERAL_RE.findall(text):
            if _rename_dotted(dotted, renames) is not None:
                rep.taxonomy_rewrites += 1
    return rep


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
_SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "build", "dist", ".codegraph"}


def _iter_py_files(root: Path, sub_paths: list[Path]) -> list[Path]:
    roots = sub_paths or [root]
    seen: set[Path] = set()
    out: list[Path] = []
    for base in roots:
        if base.is_file() and base.suffix == ".py":
            if base not in seen:
                seen.add(base)
                out.append(base)
            continue
        for p in base.rglob("*.py"):
            if any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
                continue
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _is_taxonomy(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return rel in mm.TAXONOMY_FILES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WP-8 import-rewrite codemod")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True,
                      help="report counts, change nothing (default)")
    mode.add_argument("--apply", action="store_true",
                      help="rewrite files in place (requires libcst)")
    parser.add_argument("--rename-nodes", action="store_true",
                        help="include optional nodes.->signals. (R15) + Pass C")
    parser.add_argument("--rename-prop-firms", action="store_true",
                        help="include optional prop_firms. (R4)")
    parser.add_argument("--root", type=Path, default=None,
                        help="repo root (default: auto-detect)")
    parser.add_argument("--path", type=Path, action="append", default=[],
                        help="restrict to sub-path (repeatable)")
    args = parser.parse_args(argv)

    apply = bool(args.apply)
    root = (args.root or Path(__file__).resolve().parents[2]).resolve()
    sub_paths = [(root / p).resolve() if not p.is_absolute() else p for p in args.path]

    renames = mm.active_import_renames(
        rename_nodes=args.rename_nodes, rename_prop_firms=args.rename_prop_firms
    )

    if apply and not _HAS_LIBCST:
        print("ERROR: --apply requires libcst, which is not installed.", file=sys.stderr)
        print("       Install it (pip install libcst) and retry.", file=sys.stderr)
        return 2

    files = _iter_py_files(root, sub_paths)
    reports: list[FileReport] = []
    per_prefix: Counter[str] = Counter()

    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = f.relative_to(root).as_posix()
        is_tax = _is_taxonomy(f, root)

        if _HAS_LIBCST:
            try:
                new_text, rep = _apply_libcst(
                    text, renames, is_taxonomy=is_tax,
                    rename_nodes=args.rename_nodes, path=rel,
                )
            except Exception as exc:  # parse error -> report, skip
                print(f"  SKIP (parse error): {rel}: {exc}", file=sys.stderr)
                continue
            if apply and rep.total and new_text != text:
                f.write_text(new_text, encoding="utf-8")
        else:
            rep = _scan_regex(text, renames, is_taxonomy=is_tax, path=rel)

        if rep.total:
            reports.append(rep)
            # Attribute import rewrites to their winning prefix for reporting.
            _tally_prefixes(text, renames, per_prefix)

    _print_summary(reports, per_prefix, apply=apply, has_libcst=_HAS_LIBCST,
                   n_files=len(files))
    return 0


def _tally_prefixes(
    text: str, renames: list[mm.ImportRename], per_prefix: Counter[str]
) -> None:
    """Best-effort per-prefix tally for the summary (regex-level)."""
    for line in text.splitlines():
        m = _IMPORT_FROM_RE.match(line) or _IMPORT_RE.match(line)
        if not m:
            continue
        for mod in re.split(r"\s*,\s*", m.group(1)):
            mod = mod.split(" as ")[0].strip()
            for rule in renames:
                pref = rule.old_prefix
                base = pref[:-1] if pref.endswith(".") else pref
                if mod == base or mod.startswith(pref if pref.endswith(".") else pref + "."):
                    per_prefix[rule.rule_id] += 1
                    break


def _print_summary(
    reports: list[FileReport],
    per_prefix: Counter[str],
    *,
    apply: bool,
    has_libcst: bool,
    n_files: int,
) -> None:
    mode = "APPLIED" if apply else "DRY-RUN"
    engine = "libcst" if has_libcst else "regex-fallback (counts only; --apply blocked)"
    imports = sum(r.import_rewrites for r in reports)
    strings = sum(r.string_rewrites for r in reports)
    tax = sum(r.taxonomy_rewrites for r in reports)
    print(f"=== rewrite_imports [{mode}] engine={engine} ===")
    print(f"Scanned {n_files} .py files; {len(reports)} files would change.")
    print(f"Import-statement rewrites : {imports}")
    print(f"Dynamic-string rewrites   : {strings}")
    print(f"Taxonomy-value rewrites   : {tax}")
    print(f"TOTAL rewrites            : {imports + strings + tax}")
    if per_prefix:
        print("\nPer-rule (import statements):")
        order = {r.rule_id: i for i, r in enumerate(mm.IMPORT_RENAMES)}
        for rid, cnt in sorted(per_prefix.items(), key=lambda kv: order.get(kv[0], 99)):
            print(f"  {rid:5s} {cnt}")
    print("\nTop files by rewrite count:")
    for r in sorted(reports, key=lambda x: -x.total)[:20]:
        print(f"  {r.total:4d}  {r.path}")


if __name__ == "__main__":
    raise SystemExit(main())
