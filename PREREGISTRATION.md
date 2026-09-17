# Pre-registration: does rank-space indexing improve residual mean reversion?

Written 2026-09-17, **before any estimation**. Nothing below is revised after
seeing results.

Tests the central *representation* claim of Li & Papanicolaou, "Statistical
Arbitrage in Rank Space" (arXiv 2410.06568): that indexing equities by
capitalization rank rather than by name yields residual returns with stronger
mean-reverting properties.

This tests **only** that claim. It does not test whether the resulting signal is
tradeable; the paper itself reports the strategy ceases to profit at 5bp costs.

## The mechanical effect that must be controlled for

Rank-slot returns are returns on **order statistics**, not on names. If names A
and B swap adjacent ranks, the cap at rank k moves less than either name did,
because the ordering re-sorts. Example: A=100, B=99 at t; A falls to 95, B rises
to 98 at t+1. Name returns are -5% and -1%, but rank k's cap goes 100 -> 98
(-2%) and rank k+1 goes 99 -> 95 (-4%).

Order statistics are less volatile and more mean-reverting than the underlying
series **by construction**. This is the collision/leakage effect from Stochastic
Portfolio Theory. It is arithmetic, not an inefficiency, and no party is
obligated to close it.

**Therefore a rank-space advantage measured against zero proves nothing.** It
must be measured against a synthetic null in which no cross-sectional signal
exists at all.

## Hypotheses

- **H0:** rank-space residual mean reversion is fully explained by the
  order-statistic effect, i.e. it does not exceed what independent random walks
  produce under identical construction.
- **H1 (paper's claim):** rank-space residuals mean-revert materially more than
  name-space residuals, beyond the mechanical effect.

## Construction (fixed in advance)

1. Universe: N largest US names by capitalization with continuous history over
   the sample. Cap approximated as price x shares outstanding.
2. **Name panel:** daily log returns per name.
3. **Rank panel:** each day, sort by cap; the return of rank slot k on day t is
   the log change in the capitalization occupying rank k.
4. Residuals in each space: regress each column's daily return on the first
   principal component of that space's return panel, estimated on an expanding
   window with a 250-day minimum. Residual = return minus fitted.
5. Statistics per space:
   - `pc1_var`: fraction of variance explained by PC1
   - `ar1`: pooled AR(1) coefficient of residuals (more negative = stronger
     mean reversion)
   - `halflife`: implied OU half-life in days, `-ln(2)/ln(1+ar1)`

## The test statistic

    ADVANTAGE = ar1(name) - ar1(rank)

Positive under H1 (rank residuals more negatively autocorrelated, i.e. revert
faster). Reported in AR(1) units.

## The null

Simulate independent geometric Brownian motions, one per name, with each name's
**realised volatility and starting capitalization taken from the real data**,
and **zero cross-sectional structure and zero mean reversion by construction**.
Rebuild the rank panel from the simulated caps, recompute ADVANTAGE. 200 draws.

This null contains the order-statistic effect and nothing else.

## Decision threshold, set now

**Proceed past Phase 0 only if** observed ADVANTAGE exceeds the **95th
percentile** of the null distribution of ADVANTAGE.

If it does not, the rank-space effect is an artifact of order statistics on this
data, and the project stops.

## Known bias, recorded in advance

The panel is **survivorship-biased**: only currently-listed names, with static
current share counts. Names that fell in capitalization and delisted are absent,
so the losing tail is truncated and the surviving slot is backfilled from below.

**This biases the test toward H1.** Accordingly:

- A **null result is informative** and sufficient to stop the project.
- A **positive result is uninterpretable** and requires re-running on
  survivorship-free data (CRSP, Sharadar, or Norgate) before it means anything.

This is a deliberate one-way test, chosen because the negative branch is free.
