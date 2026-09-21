# Rank-space statistical arbitrage — a separate replication

Li & Papanicolaou, *Statistical Arbitrage in Rank Space*
([arXiv:2410.06568](https://arxiv.org/abs/2410.06568)).

Kept alongside the DLSA work because the method is the same — fix the test,
simulate the null, then look — and because one finding carries over: a
construction can manufacture apparent mean reversion with no cross-sectional
signal present at all.

**Result.** Their parametric benchmark scores a gross Sharpe of **13.86 on sorted
pure noise**, against **9.63** on real data and the **6.14** they report in
Table 1. Rank-space residual AR(1) sits at the **50th percentile** of a null with
no mean reversion built in. Name space, run through identical code, beats its own
null at every lag.

Order statistics mean-revert by construction, so the question is never *"does
rank space mean-revert?"* — it must — but *"more than sorting alone would
produce?"*

The authors are **not** naive about this: Appendix B derives a hybrid-Atlas model
with local times. An earlier version of this work claimed otherwise and was
wrong; the retraction is in `RankSpace_Replication.ipynb` cell 18, alongside
three other corrections, all of which strengthened the conclusion.

`RankSpace_Replication.ipynb` is the readable artifact — executed, with outputs,
and every claim about the paper tied to a line number reproducible via
`pdftotext -layout`.

**Run from the repository root**, not from this directory — paths to
`cache/prices_big.parquet` are relative:

```bash
python3 fetch_big.py                    # build the price panel (repo root)
python3 rankspace/run_replication.py
```

`PREREGISTRATION.md` was committed before any estimation; `FINDINGS.md` is the
writeup.

### Exit codes

`ranktest.py` returns **1** when the pre-registered Phase 0 test *fails* — which
on this data it does, by design. That is the scientific verdict, not a crash:
the script runs to completion and writes `results/phase0_results.json` either
way. Do not treat a non-zero exit here as a broken script.

It also reads prices from `cache/`. With a cold cache it hits yfinance, which
rate-limits, so the first run can fail transiently. Re-run it.
