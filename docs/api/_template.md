# <Module / Package Name>

> **Path:** `<repo-relative path>`  
> **Status:** Draft | Stable | Deprecated  
> **Last updated:** YYYY-MM-DD

## Purpose
Short description of what this module/package is responsible for.

## Public API policy (what we document)
This document covers **public API** only:

Public items include:
- Top-level functions/classes/constants **not** prefixed with `_`
- Anything exported via `__init__.py` (re-exports)
- Anything imported/used outside this module/package (cross-module dependency surface)
- Entry points: CLIs, scripts, main functions, factory builders

Not public (skip unless required to use a public API):
- Helpers prefixed with `_`
- Internal modules under `internal/`, `_internal/`, `utils/` *unless* referenced by public API
- Test-only code

If a “private” symbol is required to use a public API, document it under **Internal but required**.

## Quickstart (minimal)
Minimal example showing how a typical user would use this module.

```python
# short example

Data contracts
Describe any important schemas (DataFrame columns, dataclasses, pydantic models, dict shapes).

Input(s):

...

Output(s):

...

Public API reference
<SymbolName>
Type: function | class | constant
Signature:

<copy the signature>
Description: What it does.

Parameters

name (type): description

Returns

type: description

Raises

ExceptionType: when it happens

Notes / Constraints

Determinism, time alignment, no-lookahead rules, units, expected sorting, etc.

Examples

# tiny usage example
Internal but required
List any _private or internal utilities that must be called/understood to use the public API.

Errors & logging
Exceptions raised commonly

Important log events, warnings, or metrics emitted

Open questions
Use this when code behavior is ambiguous or inconsistent.

Q1:

Q2: