# Step 1 result: FALSIFIED — and the mechanism is not what we proposed

Pre-registered in [`PREREG_liquidity.md`](PREREG_liquidity.md) before any decile
was estimated. The pre-registration recorded that a flat profile was the more
likely outcome. It was.

## The hypothesis

Part 2 showed residual reversal in liquid US large caps is gone out of sample
(walk-forward K: **+1.01** net in 2009–2016, **−0.02** in 2017–2026), and that the
decay was horizon-specific. We proposed **crowding**: the strategy sells liquidity
provision, that is a service fee, and fees compete away where capacity is largest.

That predicts the signal should persist in names too small for the competing
capital, rising monotonically as liquidity falls.

## Result

Walk-forward-K net Sharpe, reversal L=30, 2017–2026, at the **optimistic** 0%
delisting assumption:

| band | median $vol | cost charged | gross | **net** |
|---|---|---|---|---|
| 1–900 | $160M | 2bp | +0.04 | **−0.17** |
| 901–1800 | $33M | 5bp | +0.29 | −0.20 |
| 1801–2700 | $9.8M | 15bp | +0.15 | −1.28 |
| 2701–3600 | $3.0M | 30bp | +0.08 | −2.76 |
| 3601–4500 | $1.0M | 50bp | −0.52 | **−4.10** |

**Gross is within one standard error of zero in every band** (max +0.29, SE ≈ 0.32).
There is no liquidity gradient. Net falls as liquidity drops only because costs
rise against nothing.

Pre-registered verdict: **FALSIFIED** — no band exceeds +0.20 net and the profile
is not monotone rising.

Quality diagnostics were run per band first, as required. Bad ticks stay modest
(|r|>35% peaks at 0.235%; 32 zero/negative prices across the two smallest bands),
but delisting exposure more than triples — 240 → 809 name-stops/year — which is
why the test was specified at both 0% and −30% delisting returns. Being falsified
at the optimistic end, the conservative end cannot rescue it.

## What actually happened

The raw material is intact; the reversion is not.

| era | residual vol/day | cross-sectional dispersion | **AR(1)** |
|---|---|---|---|
| 2002–2008 | 2.060% | 3.910% | **−0.0269** |
| 2009–2016 | 1.740% | 3.252% | −0.0110 |
| 2017–2026 | **2.078%** | **3.713%** | **−0.0018** |

Idiosyncratic volatility and dispersion in 2017–2026 **exceed** 2009–2016 and
nearly match 2002–2008. Stocks move on their own news as much as they ever did.
What collapsed by 93% is the one quantity reversal trades.

So neither available explanation survives:

- **Not crowding.** That predicts overshoots still form and someone faster
  captures them, leaving a gradient by liquidity. There is no gradient.
- **Not shrinking dispersion.** There is as much idiosyncratic movement as ever.

**The overshoot itself stopped forming.** Prices still move on stock-specific
news and flow; those moves now stick. Either urgency is absorbed efficiently
enough that no overshoot develops, or the moves are more genuinely informative.
Operationally identical: the service this strategy sold is no longer needed, at
any size, anywhere on the liquidity spectrum.

## Reversion compressed rather than vanished — and it is still not tradeable

Autocorrelation of the residual by lag, median across names:

| era | lag 1 | lag 2 | lag 3 | lag 5 | lag 10 | lag 20 | lag 40 | cumulative ≤60 |
|---|---|---|---|---|---|---|---|---|
| 2002–2008 | −0.0269 | −0.0089 | −0.0088 | −0.0009 | −0.0023 | +0.0005 | −0.0054 | **0.180** |
| 2009–2016 | −0.0111 | −0.0042 | −0.0058 | −0.0011 | −0.0037 | −0.0007 | −0.0015 | 0.111 |
| 2017–2026 | **−0.0017** | −0.0072 | −0.0047 | **−0.0059** | 0.0000 | +0.0002 | +0.0002 | **0.033** |

Lag 1 is dead, but lags 2–5 are alive — at lag 5 the modern era shows *stronger*
reversion than either earlier period. Beyond lag 10 there is nothing, where the
earlier eras kept accumulating. Total available reversion is **18%** of 2002–2008,
all of it inside a 2–5 day window.

That implies a signal excluding the dead lag should do better. It does, on gross,
and it does not survive costs:

| signal, 2017–2026 | gross | turnover | net |
|---|---|---|---|
| L=1 (the dead lag) | −0.03 | 1.45 | −0.62 |
| lags 2–5 (skip 1) | +0.42 | 0.71 | +0.10 |
| lags 2–10 (skip 1) | +0.36 | 0.46 | +0.15 |
| L=30 | +0.47 | 0.25 | +0.35 |

Skipping the dead lag recovers nearly all the gross — but harvesting a 2–5 day
window costs 3× the turnover, and the surviving edge cannot fund it. (L=30's
+0.35 is the hindsight-K figure; walk-forward gives −0.02.)

## Consequences for the plan

**Step 2 — longer holding periods — is dead, and this kills it.** It rested on the
horizon gradient. The ACF shows nothing beyond lag 10: longer horizons have
*less* reversion, not more. That would have been days of work; the measurement
took two minutes.

**Step 3 — OSAP characteristics — is untouched.** Cross-sectional fundamentals at
a monthly horizon are compensation for information processing and risk transfer,
not for absorbing urgency. Nothing established here applies to them.

**Residual reversal in US equities is finished in every variant we can
construct** — every liquidity band, every lag range, both delisting assumptions.
Not decayed. The overshoot no longer forms.
