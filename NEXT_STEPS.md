# Next steps, and why

## What the evidence actually says

Three findings constrain where to look next, and they point the same way.

**1. The decay is horizon-specific, not general.**

| arm | 2002–2008 | 2009–2016 | 2017–2026 |
|---|---|---|---|
| reversal L=1 | +2.81 | +0.85 | **−0.03** |
| reversal L=5 | +1.87 | +0.87 | +0.36 |
| reversal L=30 | +0.86 | +0.74 | +0.47 |

Fastest died first and hardest. Slowest lasted longest. That gradient is
information, not noise.

**2. The economics say why.** The strategy is paid liquidity provision — someone
sells urgently, pushes a price below where its peers say it belongs, you
warehouse the risk. That is a service fee. Fees compete away as capital arrives,
and capital arrives fastest where capacity is largest and holding periods are
shortest. Which is exactly the pattern in the table.

**3. Out of sample, in liquid US large caps, it is gone.** Walk-forward K
selection nets **+1.01** in 2009–2016 and **−0.02** in 2017–2026. Not decayed —
gone.

## The hypothesis, and that it is falsifiable

> Residual reversal decayed because **competing capital arrived**, and competing
> capital concentrates in the most liquid, highest-capacity, shortest-horizon
> corner of the market.

This is not just a story. It makes a prediction we can test **with data already
on disk**:

> If crowding killed it, the same signal should survive where crowding cannot
> reach — in names too small for the capital that competed it away. If it died
> uniformly across the liquidity spectrum, the crowding explanation is wrong and
> something else happened.

We have never tested this. We have used the **top 900 by dollar volume** of a
panel containing **16,003 symbols**, survivorship-free, 2000–2026. The
capacity-constrained corner is sitting in data we already own.

---

## Step 1 — Does it survive down-cap? (days, no new data, no GPU)

Run the identical harness on dollar-volume deciles: names 1–900, 900–1800,
1800–2700, and so on, down to the smallest tradeable band. Same residual
construction, same walk-forward K, same cost and borrow accounting.

**Why it comes first**
- It tests our own causal explanation rather than assuming it.
- Every input exists; it is a universe-selection change, not a rewrite.
- It is a genuine fork. A clear liquidity gradient means we know where to work.
  A flat profile falsifies the crowding story and saves months.

**Costs must be re-modelled, not reused.** Small caps have wider spreads, thinner
auctions, real borrow constraints, and hard capacity limits. A gross Sharpe
down-cap means nothing until it is charged 10–50bp rather than 1–2bp, and until
position sizes are capped at a fraction of daily volume. That analysis is the
deliverable, not the gross number.

**Pre-register before running.** The statistic is walk-forward-K net Sharpe by
decile, 2017–2026. Decayed-everywhere means every decile within noise of zero.
Crowding-confirmed means a monotone rise as liquidity falls, with the smallest
decile clearing its own (much higher) cost hurdle.

## Step 2 — Longer horizons (weeks to months)

The horizon gradient points this way, and the economics agree: capital is scarcer
where you must carry risk for a month, because that consumes balance sheet and
risk budget rather than turnover.

Concretely: hold 5, 21, 63 days instead of 1. Turnover falls by the holding
period, so the cost hurdle collapses — and **turnover decided every net
comparison in this project.** The paper's own Section III reports Sharpe ~1.5 at
a one-month hold and we never tested it.

The statistical price is steep and must be stated up front: quarterly horizons
give ~4 independent observations per name per year. Over 25 years that is ~100
non-overlapping periods and a standard error on Sharpe near 1.0. Cross-sectional
breadth helps; the time-series dimension does not. Any long-horizon result needs
that stated beside it, not buried.

## Step 3 — A different signal on the same machinery

Only after 1 and 2. We hold OSAP characteristics linked to our panel by a
crosswalk validated at median r = 1.000 on a held-out signal, and nothing has
used them. Cross-sectional fundamentals at a monthly horizon are a different
economic bet — compensation for information processing and risk transfer, not for
absorbing urgency — so they should not share the decay we measured.

## What we are not doing, and why

**Not more architecture.** Three variants of the network landed within noise of
one another (4.92 / 4.77, SE 0.26) on data where the signal was strong, and the
network earns +0.05 where a zero-parameter rule earns +0.47. Architecture is a
refinement on a signal that exists.

**Not more of daily US large-cap reversal.** It is the most studied, most crowded
cell in the space, run by better-resourced desks with better execution and
borrow. We entered it because a paper claimed Sharpe 4, not because it was a
sensible place to look.

**Not a live allocation** until an execution measurement exists. Net results in
this project flipped sign between 1bp and 2bp. That parameter cannot be
established from a backtest — it takes small real orders measured against the
official close.

---

## The discipline that carries forward

Five times a clean result dissolved against a control, and **not once was it
visible by inspecting the number**:

| apparent result | what it was |
|---|---|
| large rank-space Sharpe | order statistics |
| Sharpe ≈ 3 on noise | residual-space weight normalisation |
| null at z = −7.8 | a sign-flip null leaving \|residual\| intact |
| Sharpe 4.2 past 2016 | survivorship bias |
| **+0.37 net, 2017–2026** | **hindsight choice of K** |

The last one was found after it had been written into this repository as a
finding, and after a retail position had been sized on it. The rule that caught
each: **a result counts only when identical code, run against a control, fails to
produce it.** Vary what convention holds fixed — universe, parameters,
normalisation, the null itself.

Whatever we test next gets the same treatment: pre-registered statistic and
threshold, a null built to break the specific claim, costs charged before any
conclusion, and parameters chosen walk-forward rather than by hindsight.
