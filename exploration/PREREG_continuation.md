# Pre-registration: if reversal became continuation, does continuation pay?

Written before any continuation strategy is estimated. Nothing below is revised
after seeing results.

## Why this must be tested

The central finding of this project is that reversal became continuation, measured
three independent ways:

| measurement | early era | modern era |
|---|---|---|
| intraday, morning → afternoon | −0.0051 / −0.0772 | **+0.0219** |
| daily lag 1, large moves | −0.0262 | **+0.0036** |
| daily lag 1, all moves | −0.0269 | −0.0018 |

**Every backtest in this project shorts the move.** We have never tested the
flipped sign. A repository whose headline is "reversal became continuation" and
which never tests continuation has an obvious hole in it.

## Prior, recorded now

**I expect this to fail**, for two reasons stated before the run:

1. **Turnover has killed every result in this project without exception** — the
   CNN, one-day reversal, the delayed-overshoot fade, the skip-lag variants. A
   continuation signal at daily or intraday frequency implies the same or worse.
2. **It is in the published catalogue.** Cross-sectional and market intraday
   momentum are documented (Gao, Han, Li & Zhou, JFE 2018; Jegadeesh 1990 for the
   reversal side). Step 3 showed that 209 published characteristics collectively
   do not pay out of sample. This is subject to the same decay.

A positive result would therefore be surprising, and should be treated with more
suspicion than a negative one.

## Construction

Survivorship-free panel, top 900 by trailing 21-day dollar volume, re-selected
every 21 days, point-in-time. Residuals as in `src/mp_residuals.py`, K chosen
**walk-forward** over {1,3,5,10,15}.

Two arms, both the sign-flip of strategies already measured:

- **Daily continuation.** Weight ∝ +residual(t−1), the exact negation of
  reversal L=1. Held one day.
- **Conditional daily continuation.** Same, restricted to names whose |residual|
  on day t−1 is in the top cross-sectional quartile — the cell where the flip was
  measured (+0.0036 vs −0.0262).

Weights L1-normalised to gross 1.

**Intraday continuation is deliberately excluded.** We hold 5-minute bars for 500
sampled days only, not a continuous series, so an intraday strategy cannot be
backtested honestly on what we have. Testing it would require the full 95GB
series and its own pre-registration.

## The statistic, fixed now

Net Sharpe, 2017-01-01 → 2026-09-18, after **2bp** one-way cost and 35bp/yr
borrow, walk-forward K. Report gross, turnover and breakeven alongside.

Comparators, already measured on identical residuals:

| arm | 2017–2026 net |
|---|---|
| reversal L=30, walk-forward K | −0.02 |
| reversal L=1 (what this negates) | −0.62 |
| delayed-overshoot fade, H=3 | −0.30 |

## The null, fixed now

Permute forward returns across names within each day, among names actually held.
Holds the position count, cross-sectional return distribution, volatility
clustering and turnover exactly fixed; destroys only the link between a name's
own move and its own next return. Ten draws.

## Decision thresholds

**WORKS** — net ≥ **+0.50**, above every null draw, breakeven ≥ 6bp, positive in
at least 7 of 10 calendar years.

**DEAD** — net ≤ **+0.15**, or inside the null distribution.

**AMBIGUOUS** — anything else, reported as such. **Not** re-cut by quartile
cutoff, holding period, universe, or factor count.

## Known hazards

- **This is the sixth test on the same 2,441 days.** Reversal, liquidity bands,
  characteristics, delayed overshoot, intraday, now continuation. Not independent
  evidence, and a marginal pass should be read as noise until replicated out of
  sample.
- **The flip was measured on this data.** The +0.0036 that motivates this test
  came from the same 2017–2026 window the strategy will be evaluated on. That is
  in-sample by construction, and is the strongest reason to distrust a positive.
- **Sign-flipping a dead strategy is not a new idea.** If reversal nets −0.62 and
  continuation nets +0.62, that is a cost artifact, not alpha: both pay the same
  turnover. Check that gross and net move together, not opposite.
