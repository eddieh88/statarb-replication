# Exploration — uncharted, nothing here is concluded

**This is deliberately separated from the findings in the repository root.**

Everything in `README.md` was tested against a known answer: the paper's
published figures, the authors' own residuals, a permuted-return null, a
stock-level match to CRSP. When a result was wrong, something external said so.

Nothing in this directory has that. There is no published benchmark for
"residual reversal in the 6th liquidity decile", no prior Sharpe to reproduce,
no second dataset to cross-check. **That is precisely the condition under which
this project generated five separate artifacts** — order statistics, a
normalisation bug, a leaky null, survivorship bias, and a hindsight parameter
choice, the last of which was written into the README as a finding before it was
caught.

## Rules for anything in here

1. **Pre-register first.** A `PREREG_*.md` committed before the data is touched,
   naming the statistic, the threshold, and what would falsify the idea.
2. **Build the null before the result.** Specifically designed to break the
   specific claim, not a generic shuffle. A null that leaves the exploitable
   quantity intact (as the sign-flip null did) proves nothing.
3. **Charge costs before concluding anything.** Every net comparison in Part 2
   was decided by turnover, and sign flipped between 1bp and 2bp. Down-cap work
   needs 10–50bp and volume-based position caps, not the large-cap numbers.
4. **Choose parameters walk-forward.** The +0.37 that had to be retracted was the
   maximum across five values of K that were within one standard error of each
   other. Vary what convention holds fixed.
5. **Nothing merges to `main` until it reaches a conclusion** — including a
   negative one, which is a perfectly good result and the more likely one.

## Status

| experiment | pre-registered | run | conclusion |
|---|---|---|---|
| Step 1 — liquidity deciles | `PREREG_liquidity.md` | yes | **FALSIFIED** — [`FINDINGS_liquidity.md`](FINDINGS_liquidity.md) |
| Step 2 — longer holding periods | — | — | **dropped** — ACF shows no reversion beyond lag 10 |
| Step 3 — OSAP characteristics | `PREREG_characteristics.md` | yes | **DEAD** — [`FINDINGS_characteristics.md`](FINDINGS_characteristics.md) |

| Step 4 — delayed overshoot | `PREREG_delayed.md` | yes | **DEAD** — [`FINDINGS_delayed.md`](FINDINGS_delayed.md), beats its null, loses to costs |

| Step 5 — intraday | `PREREG_intraday.md` | yes | **NOT INTRADAY** — [`FINDINGS_intraday.md`](FINDINGS_intraday.md); timing decay ~2%, a prior claim retracted |

| Step 6 — continuation | `PREREG_continuation.md` | yes | **DEAD** — [`FINDINGS_continuation.md`](FINDINGS_continuation.md); gross zero both directions |

| Step 7 — index reversion | `PREREG_market.md` | yes | **DEAD** — [`FINDINGS_market.md`](FINDINGS_market.md); measurement stands, strategy unstable |

Reasoning for the ordering: [`../NEXT_STEPS.md`](../NEXT_STEPS.md).

## The intraday series (E1-E13)

Separate from the nine numbered exploration steps above. Tested on 10.7 GB of
5-minute data, 2021-2026, and documented in `INTRADAY_LOG.md` — which carries
the claims, how each was measured, the limitations, and two corrections
(a friction artifact and a clustered-standard-error mistake).

| file | role |
|---|---|
| `INTRADAY_LOG.md` | the record: claims, measurements, limitations, corrections |
| `PREREG_orb.md`, `PREREG_bos_fvg.md` | the two pre-registered ones |
| `intraday_build.py` | caches one daily summary table so every test shares definitions |
| `intraday_levels.py` | prior-day / pre-market level detection, with the advance-and-separation rules |
| `e2..e13_*.py` | one experiment per file |
| `render_experiments.py` | regenerates every figure in `figures/` |
| `diagnostics/` | one-off scripts behind quoted numbers |

**Read `figures/` before the numbers.** Rendering caught three errors that
every statistical check passed: flat volume in simulated images, a missing
moving-average line, and 73% of "retests" being the bar after the breakout.
