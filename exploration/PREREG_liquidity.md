# Pre-registration: does residual reversal survive where crowding cannot reach?

Written before any decile is estimated. Nothing below is revised after seeing
results.

## The claim being tested

Part 2 established that residual reversal in liquid US large caps is gone out of
sample (walk-forward K: +1.01 net in 2009–2016, **−0.02** in 2017–2026), and that
the decay was horizon-specific — one-day reversal died hardest (+2.81 → −0.03),
thirty-day decayed far more slowly.

The proposed explanation is **crowding**: the strategy is paid liquidity
provision, that is a service fee rather than a mispricing, and fees compete away
where capacity is largest and holding periods shortest.

That explanation predicts:

> The signal should persist in names too small for the capital that competed it
> away, and the persistence should increase monotonically as liquidity falls.

## What would falsify it

A flat profile across deciles — every decile within noise of zero in 2017–2026 —
means the signal died uniformly across the liquidity spectrum. Crowding would not
explain that, and the explanation in `README.md` would be wrong.

**This is the outcome I consider more likely.** Recording that here so a flat
result is not later reframed as expected.

## Construction

Universe by trailing 21-day dollar volume, re-selected every 21 days, in bands of
900: ranks 1–900, 901–1800, 1801–2700, 2701–3600, and the remainder above a
minimum-price floor. Residuals per band exactly as in `src/mp_residuals.py`:
252-day rolling PCA, 60-day betas estimated strictly before day *t*, K chosen
**walk-forward** from the preceding 1000 days over {1, 3, 5, 10, 15}.

Bands are estimated independently. A stock contributes only to the band it
occupied on that date.

## The statistic, fixed now

Walk-forward-K **net** Sharpe of reversal L=30, 2017-01-01 → 2026-09-18, by band,
after trading cost and borrow.

## Costs, fixed now — these are not the large-cap numbers

| band | one-way cost charged | position cap |
|---|---|---|
| 1–900 | 2bp | none |
| 901–1800 | 5bp | 1% of 21-day median dollar volume |
| 1801–2700 | 15bp | 1% |
| 2701–3600 | 30bp | 1% |
| beyond | 50bp | 1% |

Borrow: 35bp/yr on short notional for bands 1–2, **150bp/yr** below that, and any
name whose 21-day median dollar volume is under $1M is excluded as unborrowable.

Gross Sharpe by band is reported for diagnosis only. **No conclusion is drawn
from a gross number.**

## Decision thresholds

**Crowding confirmed** — net Sharpe rises monotonically across at least three
consecutive bands, and the lowest-liquidity band clears **+0.50 net** after its
own cost schedule.

**Falsified** — no band exceeds +0.20 net, or the profile is non-monotone with no
band clearing +0.50.

**Ambiguous** — anything else. Reported as ambiguous. Not re-cut by decile
boundary, cost assumption, or lookback until something clears.

## Known hazards, recorded in advance

- **Cost assumptions are doing the work.** A gross Sharpe of 2 in the smallest
  band is worth nothing if realistic cost is 50bp. If the result turns on the
  cost schedule, say so rather than picking the favourable column.
- **Capacity.** A strategy that only works on $2M of capital is a finding about
  capacity, not about alpha. Report implied capacity at each band from the volume
  caps.
- **Microcap data quality.** Part 2 measured 448 zero/negative prices and 0.458%
  of returns beyond ±35% in the full 16,003-symbol panel, concentrated entirely
  in sub-dollar names — versus **zero** in the top 500. Bad ticks manufacture
  reversal. Every band needs the quality diagnostics run before the strategy.
- **Delisting returns.** Small caps delist far more often. Prices stop; they do
  not go to zero in the data. A reversal strategy that buys a falling name which
  then stops trading records no loss. This must be handled explicitly and is the
  single most likely source of a spurious positive.
