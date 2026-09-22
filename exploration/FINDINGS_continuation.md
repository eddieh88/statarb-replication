# Step 6 result: DEAD — neither direction pays

Pre-registered in [`PREREG_continuation.md`](PREREG_continuation.md), which
recorded failure as the expected outcome and named the check that distinguishes a
genuine flip from a cost artifact.

## Why it had to be run

Every backtest in this project shorts the move, while its central finding is that
reversal became continuation. The direct implication had never been tested.

## Result

| arm | gross | turnover | net @2bp+borrow | null max | positive years |
|---|---|---|---|---|---|
| daily continuation | **+0.01** | 1.45 | −0.88 | −3.53 | 1/10 |
| conditional, top-quartile move | **−0.00** | 1.70 | −0.75 | −1.53 | 3/10 |

Pre-registered verdict: **DEAD** (net ≤ +0.15).

## The hazard check fired as designed

The pre-registration warned: *"if reversal nets −0.62 and continuation nets
+0.62, that is a cost artifact, not alpha: both pay the same turnover. Check that
gross and net move together."*

| | gross | net |
|---|---|---|
| reversal L=1 | −0.03 | −0.62 |
| continuation (its exact negation) | **+0.01** | **−0.88** |

Gross flipped sign, as arithmetic requires. Both are indistinguishable from zero.
Both lose to identical turnover. There is no alpha in either direction, and the
negative net figures are entirely the cost of trading.

## What this establishes

The +0.0036 lag-1 correlation that motivated the test is real — it survived a
trigger-preserving null in Step 4 — but it is far too small to survive portfolio
construction. Weighting by residual magnitude concentrates positions into
high-volatility names, and the variance that introduces swamps a 0.4%
correlation.

So the finding is stronger than "reversal died":

**For portfolio purposes the modern residual is unforecastable from its own past
in either direction.** Reversal does not pay. Continuation does not pay. It has
become a martingale difference in practice — which is what an efficient price
looks like.

The transition measured across three timescales (intraday −0.0051/−0.0772 →
+0.0219; daily large-move −0.0262 → +0.0036) is a real change in market
microstructure. It is not a tradeable one.

## Closing the exploration

| step | verdict |
|---|---|
| 1 — liquidity deciles | FALSIFIED — flat across all five bands |
| 2 — longer holding periods | dropped — no reversion beyond lag 10 |
| 3 — characteristics composite | DEAD — inside its own null |
| 4 — delayed overshoot | DEAD — beats its null, loses to costs |
| 5 — intraday relocation | NOT INTRADAY — and timing decay is ~2% |
| 6 — continuation | **DEAD — gross zero in both directions** |

Six pre-registered tests, six negatives, one consistent mechanism. Nothing here
indicates the measurements are at fault: the same harness produced +1.01 net on
reversal in 2009–2016, the crosswalk validates at r = 0.9998 on a held-out
signal, and the panel reproduces CRSP residuals at r = 0.881 per stock.

The measurements work. What they measure is gone.
