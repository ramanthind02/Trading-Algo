"""FastAPI JSON backend for the research frontend.

Endpoints (all under ``/api``) let the React Spec Builder + results app:

* read the form schema (enum option lists, vault sleeves, node module catalog),
* CRUD + validate ``StrategySpec`` JSON files in ``research/specs/``,
* launch the exploration phase for a spec (via :mod:`research.spec.adapter`) and stream its log,
* browse + preview the run artifacts.

Nothing here mutates the canonical configs or the vault. The spec → pipeline translation is the
existing :mod:`research.spec.adapter`; this package only wires it to HTTP.
"""
