# Pre-registration: is the overshoot delayed rather than gone?

Written before any conditional strategy is estimated. Nothing below is revised
after seeing results.

## What prompted this, stated honestly

Step 1 found residual reversal flat across all five liquidity bands, and AR(1)
collapsed 93% while dispersion held steady. Diagnosing *what the residual does
instead* produced this, conditional on the day-t move being in the top quartile
by magnitude:

| lag | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| 2002–2008 | **−0.0262** | −0.0146 | −0.0108 | 0.0000 | −0.0032 | −0.0041 |
| 2017–2026 | **+0.0036** | **−0.0093** | −0.0014 | **−0.0070** | −0.0001 | +0.0004 |

Large moves used to be overshoots already complete at the close. Now they
**continue through day 1** and correct over days 2–5. If that is real, L=1
reversal's collapse from +2.81 to −0.03 is a timing failure, not a dead signal:
the strategy buys at close(t) and the move keeps going at t+1.

**This is a post-hoc observation.** We examined roughly six conditioning
dimensions — magnitude, volume, lag, liquidity band, factor count, horizon —
before it surfaced. That is enough specification search that something had to
look interesting, and two of ten null draws beat the Step 3 composite. The whole
point of this document is to test the idea on terms fixed before it is run.

## Construction

Universe: top 900 by trailing 21-day dollar volume, re-selected every 21 days,
point-in-time, from the survivorship-free panel. Residuals as in
`src/mp_residuals.py`; K chosen **walk-forward** over {1,3,5,10,15}.

Signal, formed at close(t):

- **Trigger.** |residual| on day **t−1** in the top quartile of that day's
  cross-section. Names outside it are not traded.
- **Direction.** Short the move: weight ∝ −sign(residual at t−1).
- **Delay.** One day. Day t−1's move is skipped, which is the point.
- **Hold.** H days, H ∈ {3, 5}, overlapping positions averaged.

Weights L1-normalised to gross 1 across triggered names only.

## The statistic, fixed now

Net Sharpe, 2017-01-01 → 2026-09-18, after 2bp one-way cost and 35bp/yr borrow.
Report gross, turnover and breakeven alongside. **No conclusion from a gross
number.**

Comparators, all already measured on identical residuals:

| arm | 2017–2026 net |
|---|---|
| reversal L=30, fixed K=5 (hindsight) | +0.35 |
| reversal L=30, walk-forward K | **−0.02** |
| unconditional skip-1, lags 2–5 | +0.10 |

The bar is **walk-forward −0.02**, not the hindsight figure.

## The null, fixed now

**Trigger-preserving permutation.** Within each day, permute which of the
triggered names receives which forward return, among triggered names only. This
holds the trigger rule, the number of positions, the cross-sectional return
distribution and volatility clustering exactly fixed, and destroys only the link
between a name's own move and its own subsequent return.

A generic shuffle will not do. The specific claim is that *this name's* large
move predicts *this name's* correction; the null must break precisely that. Ten
draws.

## Decision thresholds

**WORKS** — net ≥ **+0.50**, above every null draw, breakeven ≥ 6bp, and positive
in at least 7 of the 10 calendar years.

**DEAD** — net ≤ **+0.15**, or inside the null distribution.

**AMBIGUOUS** — anything else. Reported as ambiguous. **Not** re-cut by quartile
cutoff, delay length, holding period, universe size, or factor count until
something clears. H ∈ {3,5} is the only free choice and both are reported.

## Known hazards, recorded in advance

- **Post-hoc origin.** The strongest reason to expect failure. A −0.0093
  correlation at lag 2 is ~6 SE on 550k observations, but it was found by looking,
  and the same dataset produced the lag-1 flip we are now conditioning on.
- **Trigger selects volatility.** Top-quartile |residual| names are the most
  volatile. Their forward returns have higher variance, which inflates gross
  Sharpe denominators unpredictably and raises real trading costs above the 2bp
  assumed. Report realised volatility of the traded book.
- **Concentration.** Trading a quartile of 900 means ~225 names, and the
  overlapping hold means fewer new positions daily. Turnover should be modest —
  that is the attraction — but check it is not so concentrated that a handful of
  names drive the result. Report the share of P&L from the top 10 names.
- **Earnings clustering.** Large residual moves cluster on earnings dates. If the
  effect is entirely post-earnings drift and its reversal, that is a known
  published anomaly (PEAD) and therefore subject to the same publication decay
  Step 3 measured. Report results excluding the 3 days around large-move dates
  that recur quarterly per name.
- **This is the fourth strategy tested on one dataset.** Reversal, liquidity
  bands, characteristics, now this. A fourth test on the same 2,441 days is not
  independent evidence, and a marginal pass should be read accordingly.
