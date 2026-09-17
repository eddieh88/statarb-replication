# Phase 0: FAIL. Rank-space mean reversion is an order-statistic artifact.

Run 2026-09-17. 170 US large caps, 4,653 daily observations, 2008-03 to 2026-09.
Statistics and threshold fixed in `PREREGISTRATION.md`, committed (fc3efcf)
before any data was fetched. Raw numbers in `phase0_results.json`.

## The headline looks like a strong confirmation

| | name space | rank space |
|---|---|---|
| PC1 variance explained | 43.2% | 54.7% |
| residual AR(1) | −0.0134 | **−0.1319** |
| implied OU half-life | 51.4 days | **4.9 days** |

Rank-space residuals mean-revert roughly ten times faster. Taken at face value
this reproduces Li & Papanicolaou's central representation claim emphatically.

## It is almost entirely mechanical

Independent geometric Brownian motions — matched per-name volatility, matched
starting capitalisations, **zero cross-sectional structure and zero mean
reversion by construction** — put through identical rank construction:

| | value |
|---|---|
| null rank-space AR(1), median | **−0.1293** |
| observed rank-space AR(1) | −0.1319 |
| null ADVANTAGE, median | +0.1291 |
| null ADVANTAGE, 95th pct | +0.1341 |
| **observed ADVANTAGE** | **+0.1185** |
| fraction of null draws ≥ observed | **100%** |

Pure noise produces *more* rank-space advantage than the real data does. Every
one of 200 null draws exceeded the observed value.

This is the collision/leakage effect from Stochastic Portfolio Theory, recorded
in the pre-registration before estimation: rank-slot returns are returns on
**order statistics**, which are less volatile and more mean-reverting than the
underlying by arithmetic. No party is obligated to close it, and there is
nothing to trade.

## Robustness: it survives a fairer null, narrowly

The pre-registered null uses independent GBMs, which churn ranks more than real
correlated equities do — arguably too high a bar. Re-running with a common
factor (`robustness.py`, post-hoc, not pre-registered):

| factor share | null PC1 | null ADV p95 | observed clears? |
|---|---|---|---|
| 0.00 (pre-registered) | 2.0% | +0.1341 | no |
| 0.20 | 20.6% | +0.1279 | no |
| **0.432 (calibrated to real PC1)** | 43.6% | **+0.1215** | **no** |
| 0.60 | 60.3% | +0.1133 | YES |

At the correctly-calibrated factor strength it still fails, by 0.0030 in AR(1)
units. It only clears at a factor share well above what the data exhibits. So
the conclusion holds, but the margin is narrow rather than overwhelming —
roughly 97.5% of the effect is mechanical, and the residual sliver is inside
the noise.

## And the panel was rigged in the effect's favour

Per the pre-registration: the universe is currently-listed names only, with
static current share counts. Names that fell in capitalisation and delisted are
absent, so the losing tail is truncated and each slot is backfilled from below —
manufacturing mean reversion exactly where the strategy claims to find it.

**The test was biased toward H1 and H1 still failed.** That is what makes a null
result here sufficient to stop, and it was recorded in advance as the reason for
running a deliberately one-way test.

## Limitations

- 170 names; the paper uses the top 500. A wider universe churns ranks more,
  which raises the mechanical baseline rather than lowering it.
- Daily frequency. The paper's effect is strongest intraday (they optimise to a
  225-minute rebalance), and this test cannot speak to intraday dynamics. That
  is also where their costs are worst — the strategy ceases to profit at 5bp by
  their own account.
- PC1 is removed on the full sample, which is in-sample. It is applied
  identically to both spaces and to every null draw, so it cancels in the
  comparison, which is the only quantity under test.
- This tests the *representation* claim only. It says nothing about whether a
  DNN can extract something the pooled AR(1) misses.

## Verdict

The pre-registered threshold was: proceed only if observed ADVANTAGE exceeds the
null 95th percentile. It does not — at any plausible null calibration.

**Stop.** No further work, no data purchase, no Phase 1. The 4.9-day half-life
is real and is worth nothing: it is what sorting does to random numbers.

The one thing still worth doing is unaffected by this result, because it does not
depend on rank-space portfolio construction at all: adding rank-percentile and
distance-to-rank-median **features** to an existing name-space daily pipeline.
That tests whether rank is a useful predictor, which is a different claim from
whether rank-space residuals mean-revert.
