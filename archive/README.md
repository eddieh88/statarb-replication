# Superseded

Kept because the corrections are part of the record, not because they are useful.

- `dlsa.py`, `dlsa3.py`, `run_dlsa.py`, `run_full.py`, `run_linear.py`
  — first constructions on our own yfinance panel, before switching to the
  authors' published residuals. `dlsa.py` had a beta window that included day *t*
  (look-ahead); fixing it moved a null from 2.06 to 1.01. The corrected version
  is `src/dlsa2.py`, which is still used by `src/extend.py`.
- `add_cells.py`, `build_notebook.py` — notebook builders that write to the
  current directory and would overwrite the executed
  `rankspace/RankSpace_Replication.ipynb` with an unexecuted one.
- `phase1_costs.py` — superseded by `netcost.py`.
- `causal_test.py`, `causal2.py`, `crux.py`, `lagcheck.py`, `parametric.py`
  — exploratory; conclusions folded into the surviving scripts.

**These do not run.** `dlsa3.py` imports `dlsa2` and `build_notebook.py` imports
`replicate`, both of which now live elsewhere (`src/` and `rankspace/`). They are
kept for the record, not for execution.
