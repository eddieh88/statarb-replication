# Rank-space stat arb: the mean-reversion claim is an order-statistic artifact;
# the structural claim is roughly half real.

Two runs. The first (170 names, K=1 factor in both spaces) was under-specified.
This file reports the second, at the paper's scale and construction.

**Setup:** 985 US names with ≥98% history 2006-05 → 2026-09 (5,109 days), top 500
by capitalisation selected every 21 days, 252-day rolling PCA, **K=5 factors in
name space and K=1 in rank space** per App. A.2 of arXiv 2410.06568, 60-day
factor loadings, rank return per eq (3.5).

**Null:** identical pipeline driven by simulated capitalisations — common factor
(share 0.40) plus idiosyncratic noise, matched per-name volatility and starting
caps, **no mean reversion and no cross-sectional predictability by construction.**
The order-statistic (local-time) effect is therefore present in both arms, and
only genuine structure can separate them.

## Result 1 — mean reversion: rank space *is* the null

| | observed | null median | null 5th pct | |
|---|---|---|---|---|
| name-space AR(1), K=5 | **−0.0113** | −0.0013 | −0.0026 | **beats null** |
| rank-space AR(1), K=1 | **−0.1587** | **−0.1588** | −0.1688 | within null |

Rank-space excess over the null median: **+0.0001**. The observed value sits at
the **50th percentile** of the null distribution.

It holds across the whole autocorrelation function, not just lag 1:

| lag | rank obs | rank null | diff | name obs | name null | diff |
|---|---|---|---|---|---|---|
| 1 | −0.1587 | −0.1566 | −0.0021 | −0.0113 | −0.0008 | −0.0105 |
| 2 | −0.0666 | −0.0683 | +0.0017 | −0.0071 | −0.0034 | −0.0037 |
| 3 | −0.0360 | −0.0321 | −0.0039 | −0.0108 | −0.0033 | −0.0075 |
| 5 | −0.0201 | −0.0159 | −0.0042 | −0.0069 | −0.0032 | −0.0037 |
| 10 | −0.0046 | −0.0072 | +0.0027 | −0.0038 | −0.0022 | −0.0016 |

Rank space oscillates around the null at ±0.004. Name space is more negative
than its null at **every** lag.

And on the paper's own statistic — OU half-life of 60-day **cumulative**
residuals, the τ of their Figure 4:

| | observed | null |
|---|---|---|
| rank space | **10.86 d** | **10.61 d** |
| name space | 35.21 d | 38.85 d |

Rank space reproduces the null. Name space beats it.

**The genuine mean reversion is in name space.** That is the opposite of the
paper's framing, and it is the documented short-term reversal effect.

## Result 2 — market structure: roughly half real

Their other claim is that rank space has a larger leading eigenvalue and cleaner
Marchenko–Pastur bulk-edge separation. It replicates, and the null reproduces
much of it:

| | PC1 share | eigenvalues > M-P |
|---|---|---|
| observed, name | 43.2% | 12 |
| observed, rank | **54.7%** | **4** |
| null, name | 40.3% | 7 |
| null, rank | **46.9%** | **2** |

Observed rank-over-name gap **+11.5pp** against a mechanical **+6.6pp**. So
about **57% artifact, 43% genuine**. Sorting concentrates variance into the
first eigenvalue and cleans the spectrum on its own — but not entirely.

## What this implies about the paper

The authors are **not** naive about the mechanism. Appendix B derives a
hybrid-Atlas model with local times Λ(k,k+1), citing Banner–Ghomrasni and
Banner–Fernholz–Karatzas, and their intraday section is explicitly about trading
through collision versus idle regimes.

But if rank-space residual dynamics are statistically indistinguishable from
sorted noise, the DNN cannot be exploiting superior residual mean reversion —
there is none to exploit. The coherent reading is that it harvests the
**local-time / collision term itself**, i.e. the Stochastic Portfolio Theory
rebalancing premium. Which explains their cost profile exactly: gross annual
return 206.49%, net 35.68% at 2bp, zero at 5bp. **The premium and the cost are
the same trades.**

## Corrections to the first run

- Factor count: they use K=5 in name space, K=1 in rank space. Using K=1 in both
  left four factors of common variation in the name-space residuals and inflated
  the apparent rank advantage.
- Universe: 170 → 985 names with the top-500 rule applied.
- Statistic: the first run's `ADVANTAGE = ar1(name) − ar1(rank)` penalised the
  real data for having genuine name-space reversal, which the null lacks by
  construction. Comparing each space to its own null is the correct framing.

Fixing all three made the conclusion **stronger**, not weaker.

## Limitations

- Survivorship-biased: currently-listed names only, static current share counts.
  This biases **toward** the paper's hypothesis, and the hypothesis still fails.
- Daily frequency. The paper's effect is strongest intraday (they optimise to a
  225-minute rebalance); this says nothing about intraday dynamics, which is also
  where their costs are worst.
- PCA recalibrated every 21 days rather than daily; applied identically to both
  arms.
- Tests the representation claim only. A DNN may extract structure that pooled
  autocorrelation misses — though it would have to be structure absent from the
  entire ACF and from the cumulative-residual OU fit.

## Verdict

**Do not build the rank-space portfolio.** The mean-reversion premise is an
artifact of sorting. The structural premise is about half real but is not itself
tradeable.

The residual finding worth keeping is the opposite of the paper's: **name-space
residuals carry genuine mean reversion that beats its null at every lag**, with
K=5 factors removed. That is a conventional short-term-reversal signal and it
lives in the pipeline you already have.
