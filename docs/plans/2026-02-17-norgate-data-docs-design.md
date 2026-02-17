# Norgate Data Docs Update Design

## Goal
Align data migration kanban tasks with the Norgate data source and add a comprehensive library reference that documents data structure, formats, identifiers, and operational constraints for future context.

## Scope
- Update `docs/kanban/to-do/data_migration/*.md` to reference Norgate as the migration target and link to the new library spec.
- Add a single authoritative library doc under `docs/library/Data/` that captures Norgate data structure, formats, and constraints.
- No code changes, no tests required (docs-only).

## Context
- Current migration tasks focus on fixed-date rolling and back-adjustment derived from legacy data.
- Norgate uses volume-based rolling for continuous futures, so comparison tooling must document expected deltas.
- Library currently lacks a data source spec for Norgate, and `docs/library/Data/` is empty.

## Design

### Library doc
Create `docs/library/Data/Norgate.md` as the canonical spec. It will include:
- Operational requirements (Windows-only NDU, local `.norgatedata` cache, active subscription).
- Time series options (pandas/ndarray/recarray), datetime formats/timezones, and interval/padding/adjustment settings.
- Price/volume schema and futures-specific fields (open interest, delivery month).
- Identifier strategy (symbol vs assetid) and storage guidance.
- Futures metadata fields (tick size, point value, margin, sessions/markets) and date-related constraints.
- Error handling and missing data behavior (ValueError, None returns).
- Migration implications: roll methodology differences, expected comparison deltas, and resampling notes for 1-minute vs daily data.

### Kanban updates
For each task in `docs/kanban/to-do/data_migration/`:
- Add a short “Data Source” paragraph referencing `docs/library/Data/Norgate.md`.
- Update Context/References to include Norgate as the migration source and note roll methodology differences.
- Clarify data contracts where Norgate inputs/outputs are expected (especially T006).
- Provide a concrete placeholder for Norgate data paths used in comparisons (with a note to update when ingestion path is finalized).

## Success Criteria
- Library has a complete, future-proof Norgate data reference.
- Migration tasks explicitly reference the new source, expected roll differences, and comparison inputs.
- No changes outside the agreed docs locations.

## Risks / Notes
- Norgate’s continuous futures roll methodology may not match fixed-date roll rules; documentation must make this explicit.
- Comparisons may require resampling intraday data to daily to match Norgate daily series.
