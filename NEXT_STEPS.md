# Where this ended, and what would come next

## The scorecard

| study | question | answer |
|---|---|---|
| Part 1 | Does the paper replicate? | **Yes** — OU 0.70 vs published 0.73 |
| Part 1 | Does its Sharpe survive a null? | **Yes** — 4.92 vs 0.46, 28 of 30 blocks |
| Part 1 | Is profitability non-declining, as claimed? | **No** — t = −8.92 inside their own sample |
| Part 2 | Does it survive past 2016, survivorship-free? | **No** — walk-forward K nets −0.02 |
| Part 2 | Does the network beat a 30-day reversal rule? | **No** — +0.05 gross, −0.85 net |
| Part 3 | Does it survive where crowding cannot reach? | **No** — flat across five liquidity bands |
| Part 3 | Did reversion move to longer horizons? | **No** — nothing beyond lag 10 |
| Part 3 | Do 209 published characteristics pay? | **No** — inside their own null |
| Part 3 | Is it a timing failure, fadeable a day late? | **No** — beats its null, loses to costs |
| Part 3 | Did the overshoot move intraday? | **No** — it flipped sign instead |
| Part 3 | Does continuation pay, then? | **No** — gross zero in both directions |

One yes that matters: the paper is real and replicates. Every question about
whether any of it still works came back negative.

## The finding

**Reversal became continuation, and neither pays.**

Idiosyncratic dispersion is unchanged — stocks move on their own news as much as
ever. The bounce is what vanished. Measured at three timescales, the sign flipped
from negative to positive; measured as a portfolio, both directions have a gross
Sharpe indistinguishable from zero and lose identically to turnover.

The modern residual is a martingale difference in practice. That is what an
efficient price looks like.

## What is not in doubt

The machinery works. The same harness produced **+1.01 net** on reversal in
2009–2016 under walk-forward parameter selection. The crosswalk validates at
r = 0.9998 on a signal never used to build it, against 0.2094 shuffled. The panel
reproduces CRSP residuals at r = 0.881 per stock and the exact NYSE calendar.
When something is there, these measurements find it.

## What would come next — a different project

**Not more of this.** Six pre-registered tests, every liquidity band, every lag
range, both signs, daily through intraday, both delisting assumptions. US
equity residual reversal is finished, and the reason is understood.

**Not execution measurement either — yet.** It was the highest-leverage unknown
while a candidate strategy existed. With the honest walk-forward answer at −0.02,
measuring fill quality answers a question about nothing. It becomes the first
priority again the moment there is something worth executing.

**Something not in a public catalogue.** The through-line of eleven negatives is
that documented edges are gone. OSAP exists *because* those 209 signals were
published; the reversal literature is thirty years deep; intraday momentum is in
the JFE. In practice that means non-price data, horizons where capital is scarce
because it must carry risk, or asset classes this project never touched.

**If revisiting the network, at the paper's own configuration.** Our CNN ran at 30
epochs against their 100, retrained every 250 days against their 125, on PCA-5
rather than their best IPCA arm. We showed it earns nothing at *our* settings
while a zero-parameter rule earns more; we did not show it fails at theirs.
Roughly 6.6× the compute.

## The part that transfers

Six times in this project a clean-looking result dissolved against a control, and
**not once was it visible by inspecting the number**:

| apparent result | what it was |
|---|---|
| large rank-space Sharpe | order statistics |
| Sharpe ≈ 3 on noise | residual-space weight normalisation |
| null at z = −7.8 | a sign-flip null leaving \|residual\| intact |
| Sharpe 4.2 past 2016 | survivorship bias |
| +0.37 net, 2017–2026 | hindsight choice of K |
| "L=1 is structurally unexecutable" | a 2% signal degradation, asserted and never measured |

The fifth reached this README as a finding, with a retail position sized on it.
The sixth was my own reasoning, repeated confidently for hours, wrong.

The rule that caught each: **a result counts only when identical code, run
against a control, fails to produce it** — and the control must break the
specific claim, not merely shuffle something.

Two cheap diagnostics killed two expensive plans: varying K invalidated a
headline number, and one autocorrelation function killed a fortnight of planned
longer-horizon work. **Test the reasoning before running the experiment that
assumes it.**
