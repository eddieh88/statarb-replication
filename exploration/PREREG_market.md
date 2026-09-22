# Pre-registration: did daily mean reversion move from stocks to the index?

Written before any index-level strategy is estimated. Nothing below is revised
after seeing results.

## What prompted this

Six exploration steps asked why reversal died in the residual. A diagnostic on
factor-removal method found that it may never have died — we were subtracting it:

| era | PC1 | equal-weight market | residual |
|---|---|---|---|
| 2002–2008 | −0.0193 | −0.0293 | −0.0269 |
| 2009–2016 | −0.0038 | −0.0356 | −0.0110 |
| 2017–2026 | −0.0349 | **−0.0750** | **−0.0018** |

Raw returns mean-revert *more* now (−0.0285) than in 2002–2008 (−0.0195). Linear
factor removal strips 87% of that in the modern era versus 3% in the early era.
Market-level reversion more than doubled over precisely the window the residual's
went to zero.

**Hypothesis.** Liquidity provision migrated from single names to baskets. ETF
creation/redemption, index rebalancing and basket trading put the urgent,
uninformed flow at the index level, so the overshoot forms there. Single-stock
flow is increasingly news-driven, which is why those moves now continue.

## Prior, recorded now

**Weakly negative, for one reason that dominates.** Market-level short-term
reversal is heavily documented. The through-line of eleven negatives in this
project is that documented edges are gone, and Step 3 showed 209 published
characteristics collectively do not pay out of sample.

Against that: this strategy is **one instrument** with ~0.5bp spreads and no
borrow, where every prior failure was killed by turnover on 900 positions. The
cost structure is qualitatively different, so a small edge survives here that
would not survive there. That is the only reason the prior is not strongly
negative.

## Construction

**Instrument.** Equal-weight daily return of the top 900 by trailing 21-day
dollar volume, re-selected every 21 days, point-in-time, from the
survivorship-free panel. This is the series in which the effect was measured.

**Also reported, and the one that matters for tradeability:** the cap-weight
proxy, since SPY and ES are cap-weighted and the equal-weight basket is not
directly tradeable. If the effect exists only equal-weight, it is a small-cap
and bid-ask-bounce finding, not an index-reversion finding.

**Signal.** Position at close(t) = −z(r_t), where z is the return standardised by
its trailing 63-day volatility, clipped to ±2. Held one day.

**No parameter search.** Lookback for volatility fixed at 63 days, clip at ±2,
holding period one day. These are stated here and not varied.

## The statistic, fixed now

Net Sharpe, 2017-01-01 → 2026-09-18, after **1bp** round-trip per unit traded
(generous for SPY/ES but not free) and no borrow — a short index position via
futures or an ETF short has financing embedded in the price, not a borrow fee.

Report gross, turnover, breakeven, and the equal-weight/cap-weight gap.

## The null, fixed now

**Block bootstrap of the return series**, 21-day blocks, 200 draws. This
preserves volatility clustering and the marginal distribution while destroying
serial dependence at the daily horizon — which is precisely the claim. A
sign-flip or iid shuffle would not do: iid shuffling destroys the volatility
clustering that a vol-scaled signal partly trades.

## Decision thresholds

**WORKS** — net Sharpe ≥ **+0.60**, above the 95th percentile of the null,
breakeven ≥ 3bp, positive in at least 7 of 10 calendar years, **and** the
cap-weight version is at least half the equal-weight version.

**DEAD** — net ≤ **+0.20**, or inside the null's 95% interval, or the cap-weight
version is under a quarter of equal-weight.

**AMBIGUOUS** — anything else. Reported as such. Not re-cut by volatility
lookback, clip level, holding period, or universe size.

## Known hazards, recorded in advance

- **Bid-ask bounce.** The single most likely artifact. Closing prices alternate
  between bid and ask, manufacturing negative autocorrelation. In a 900-name
  equal-weight basket independent bounce diversifies as 1/√900, which argues
  against it — bounce should have inflated the *residual* AR(1), which instead
  went to zero. But equal-weighting overweights small caps where spreads are
  widest. **The cap-weight comparison is the test**, which is why it is a
  threshold rather than a footnote.
- **Non-synchronous closing prices.** Thinly traded names print stale closes,
  which induces spurious reversion in an equal-weight basket. Restricting to the
  top 900 by dollar volume should limit this; the cap-weight check bounds it.
- **Documented effect.** Short-horizon index reversal is in the literature. A
  positive result here is a positive result for something already known, and
  should be read as "still works" rather than "discovered".
- **Seventh test on overlapping data.** Not independent evidence.
- **One instrument means low breadth.** Even at AR(1) = −0.075 the achievable
  Sharpe depends entirely on how much of that is harvestable after costs and
  how stable it is year to year. Ten calendar years is 10 observations of annual
  performance, which is why 7 of 10 positive is a threshold.
