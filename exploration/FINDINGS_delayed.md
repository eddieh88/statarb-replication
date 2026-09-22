# Step 4 result: DEAD — the structure is real and too small to trade

Pre-registered in [`PREREG_delayed.md`](PREREG_delayed.md), which recorded the
post-hoc origin of the idea as the strongest reason to expect failure.

## The hypothesis

Diagnosing what the residual does *instead* of mean-reverting produced this,
conditional on a top-quartile move:

| lag | 1 | 2 | 3 | 5 |
|---|---|---|---|---|
| 2002–2008 | **−0.0262** | −0.0146 | −0.0108 | 0.0000 |
| 2017–2026 | **+0.0036** | **−0.0093** | −0.0014 | **−0.0070** |

Large moves used to be overshoots complete at the close. Now they continue
through day 1 and correct over days 2–5. If real, L=1 reversal's collapse from
+2.81 to −0.03 is a **timing failure**, not a dead signal — the strategy buys at
close(t) while the move is still going.

## Result

Trigger on |residual(t−1)| in the top cross-sectional quartile, short the move,
one-day delay, hold H days overlapping, walk-forward K, 2017–2026:

| | H=3 | H=5 |
|---|---|---|
| gross | +0.22 | +0.14 |
| turnover | 0.79 | 0.58 |
| **net @2bp + borrow** | **−0.30** | **−0.34** |
| breakeven | 1bp | 1bp |
| null mean / max | −1.87 / −1.43 | −1.29 / −1.01 |
| positive years | 3/10 | 3/10 |

Pre-registered verdict: **DEAD** (net ≤ +0.15).

## But the structure is real

The strategy beats its trigger-preserving null by **~1.5 Sharpe** and clears all
ten draws. The null holds the trigger rule, position count, cross-sectional
return distribution and turnover fixed, permuting only which triggered name
receives which forward return. Beating it means the link between *this name's*
large move and *this name's* subsequent correction genuinely exists.

It cannot fund its own trading. Gross +0.22 on turnover 0.79 is a **1bp
breakeven** against 2bp assumed.

## What this adds

The lag-2 correlation is not an artifact of specification search — it survived a
null built specifically to break it. So the diagnosis stands:

**The overshoot still forms. It takes a day longer than it used to, and it is
about a fifth the size.**

That is a more precise account of what happened than "reversion died". Whoever
now absorbs same-day urgency has compressed the profitable window out of
existence: what remains is a correction that arrives too late and too small to
cover the cost of waiting for it.

It also explains the single most dramatic number in this project. L=1 reversal's
fall from +2.81 to −0.03 is not the disappearance of mean reversion — it is a
strategy that became systematically one day early.

## Honest accounting

This is the fourth strategy tested on the same 2,441 days, after reversal,
liquidity bands, and characteristics. A fourth test on one dataset is not
independent evidence. The pre-registration named the post-hoc origin as the
strongest reason to expect failure, and it failed on the criterion that mattered
— net of costs — while passing the one it was designed around.

Both outcomes are informative and they are different claims:

- **market structure:** a measurable, null-surviving delayed correction exists
- **tradeability:** it is worth less than the turnover required to collect it
