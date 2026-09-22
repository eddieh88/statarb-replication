# Where this stands, and what is left

## What was tested, and what happened

| study | question | verdict |
|---|---|---|
| Part 1 | Does the paper replicate? | **Yes** — OU 0.70 vs published 0.73 |
| Part 1 | Does its Sharpe survive a null? | **Yes** — 4.92 vs 0.46, 28 of 30 blocks |
| Part 1 | Is profitability non-declining, as claimed? | **No** — t = −8.92 inside their own sample |
| Part 2 | Does it survive past 2016, survivorship-free? | **No** — walk-forward K nets −0.02 |
| Part 2 | Does the network beat a 30-day reversal rule? | **No** — +0.05 gross, −0.85 net |
| Part 3 | Does it survive where crowding cannot reach? | **No** — flat across all five liquidity bands |
| Part 3 | Did reversion move to longer horizons? | **No** — nothing beyond lag 10 |
| Part 3 | Do 209 published characteristics pay? | **No** — inside their own null |

Eight questions, one yes that matters: the paper is real and replicates. Every
question about whether any of it still works came back negative.

## Why the negatives are consistent

Residual reversal was payment for absorbing urgency. Published characteristics
were payment for information processing. **Both were documented, and being
documented is what ended them.**

The reversal case is unusually clean. Dispersion is unchanged — stocks move on
their own news as much as ever — while AR(1) fell 93%. Nobody outcompeted us to
the overshoot; the overshoot stopped forming.

## What is *not* in doubt

The machinery works. The same harness measured **+1.01 net** on reversal in
2009–2016 with walk-forward parameter selection. The crosswalk validates at
r = 0.9998 on a signal never used to build it. The panel reproduces CRSP
residuals at r = 0.881 per stock and the exact NYSE trading calendar. When
something is there, these measurements find it.

## Remaining directions, in order

**1. Measure execution before anything else is believed.**
Every net comparison in this project flipped sign between 1bp and 2bp. That
parameter cannot be established from a backtest. A few hundred dollars of small
closing-auction orders, measured against the official close, resolves it — and it
is a prerequisite for taking *any* future result seriously, not just these.

**2. Something not in a public catalogue.**
The through-line of eight negatives is that documented edges are gone. OSAP
exists precisely because those 209 signals were published. The reversal
literature is thirty years deep. Anything tested next should be a source of
return that has not been written down, which in practice means non-price data,
unusual horizons, or structural flows rather than a better model of past prices.

**3. If revisiting the network, do it at the paper's own configuration.**
Our CNN result used 30 epochs against their 100 and retrained every 250 days
against their 125, on PCA-5 rather than their best IPCA arm. The simple
benchmarks have no hyperparameters to underfit, so the comparison is asymmetric.
We showed it earns nothing at *our* configuration while a zero-parameter rule
earns more; we did not show it fails at theirs. Roughly 6.6× the compute.

## What is deliberately not next

**Not more residual reversal.** Every liquidity band, every lag range, both
delisting assumptions, daily through monthly horizons. It is finished.

**Not fitted combinations of published anomalies.** 209 signals over 96 months
will always produce a beautiful backtest. The equal-weight test is the honest
one, and it failed.

**Not more architecture.** Three variants within noise of each other on data
where the signal was strong; the network earns +0.05 where a zero-parameter rule
earns +0.47. Architecture refines a signal that exists.

## The method, which is the part that transfers

Five times in this project a clean-looking result dissolved against a control,
and **not once was it visible by inspecting the number**:

| apparent result | what it was |
|---|---|
| large rank-space Sharpe | order statistics |
| Sharpe ≈ 3 on noise | residual-space weight normalisation |
| null at z = −7.8 | a sign-flip null leaving \|residual\| intact |
| Sharpe 4.2 past 2016 | survivorship bias |
| +0.37 net, 2017–2026 | hindsight choice of K |

The last reached this README as a finding, with a retail position sized on it,
before it was caught. The rule that caught each: **a result counts only when
identical code, run against a control, fails to produce it** — and the control
must break the specific claim, not merely shuffle something.

Two cheap diagnostics killed two expensive plans here: varying K invalidated a
headline number, and one autocorrelation function killed a fortnight of
longer-horizon work. Test the reasoning before running the experiment that
assumes it.
