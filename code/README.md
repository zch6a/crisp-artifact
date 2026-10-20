# jevschema

Core library for CRISP. Start from `src/jevschema/`:

- `schema_graph.py` — typed schema / foreign keys.
- `selection.py` — monotone closure (keys, foreign keys, padding, small tables).
- `certify.py` — budget families, split-conformal quantile, coverage.
- `gold_labels.py` — requirement extraction from reference SQL.
- `pipeline.py`, `risk.py`, `router.py`, `refine.py` — readers / selection / routing.
- `exec_eval.py` — execution accuracy.
- `paths.py` — configurable roots.

Experiment drivers live in `scripts/`; each docstring states its arguments and output files.
