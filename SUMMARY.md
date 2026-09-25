# Summary — what was tested, what was found, what went wrong

Two investigations, both closed negative. This document is the scoreboard and
the methodology record. Detail lives in `README.md` (the replication),
`exploration/` (the pre-registered tests), and `exploration/INTRADAY_LOG.md`.

---

## 1. Characteristic factor models (Step 9)

**Claim.** Firm characteristics driving factor loadings — IPCA and the
Gu-Kelly-Xiu autoencoder — should beat PCA-on-prices, which is all this project
had used.

**Result: AMBIGUOUS by the pre-registered rule, dead in substance.**

| statistic | threshold | measured |
|---|---|---|
| net Sharpe 2017-2024 | WORKS ≥ +0.50 | **+0.31** (5 seeds, sd 0.05) |
| returns capped | — | −0.06 |
| market-hedged | — | +0.28 |
| rank IC | — | +0.006, t=0.60 |
| permutation null | DEAD if inside | +0.35 vs max +0.23 — clears it |

Gross profitability alone returns **+0.58 hedged at a quarter the turnover**.

**A first round reported +1.95 and passed every check** — a 10-draw permutation
null, a K sweep, 7/7 positive years, a publication-vintage test, a 6-month
accounting-lag test. All void: the liquidity screen selected on dollar volume
measured *through* the month whose return it was predicting, so names entered
the universe **because they had already spiked**.

Cost of that bug, measured by just holding the universe equal-weighted:

| screen | ann. return | Sharpe |
|---|---|---|
| volume through month *m* (as built) | **+27.5%** | **+1.26** |
| volume through *m−1* (correct) | +8.3% | +0.40 |

**19 points a year.** No model-side test could see it: the permutation null
shuffles characteristics *within the same contaminated universe*, and the lag
test lags the features, not which rows exist.
`exploration/universe_leak_check.py` now catches it without fitting anything.

---

## 2. Intraday — 13 experiments on 10.7 GB of 5-minute data

2021-2026, 1,462 sessions, top 100 US stocks by dollar volume.

| # | claim | result |
|---|---|---|
| E1 | the open differs from the rest of the day | **confirmed** — 2.94× midday range, 35% of volume in 23% of hours |
| E2 | opening gaps fill | size-dependent — 69% for <0.2%, **27% for >2%** |
| E3 | the opening move reverses (PO3 / judas swing) | null — 50.5% |
| E4 | the opening range is swept then reverses | **refuted** — breaks continue; fading loses |
| E5 | pre-market sets the day's direction | **refuted** — 49.2%; the sweep rule is *backwards* |
| E6 | prior-day S/R break + retest | negative under two stop definitions |
| E7 | ORB on high-relative-volume names | dead by prereg; the relvol filter does sort monotonically |
| E8 | the exit rule decides it | **refuted** — seven policies within 0.04R, all negative |
| E9 | (descriptive) what trades actually do | losers peak at bar 1-7, winners at bar 40 |
| E10 | entries, incl. a random control | **no entry beats random** |
| E12 | FVGs fill at published rates | matches once the displacement filter is applied |
| E13 | break-of-structure → FVG → tap | −0.109R, same as random |

**The definitive version is E10c/E10d:** every entry given a long *and* short
form so drift cancels in the spread. Excess over random: **−0.001 to +0.007R**,
every p > 0.42. The chart-pattern entry is *worse* than random.

One confirmed finding — the open is volatile — which describes where the
competition is, not an edge.

---

## 3. Four measurement errors, and how each was caught

None of these were caught by the tests. All four were caught by looking at
something, or by a question that forced a check.

**A look-ahead in the universe screen.** Nine model-side tests passed while it
was live. The tell was visible in one glance at the holdings: NCTY, CAPR, CBAT,
POLA, SPRT — nano-caps in a "top-1500 by dollar volume" universe.

**73% of "retests" were the next bar.** The daily break-and-retest detector
took the first bar whose low touched the level, which is usually the bar
immediately after the breakout — not price advancing, stalling, and returning
days later. Caught by rendering six real detections and looking at them.

**Stops too tight, making friction look like signal failure.** Every
trade-simulating experiment returned −0.07 to −0.14R. Thirteen strategies
converging on one number is evidence about the measurement. Decomposition:
gross is +0.013 to +0.018R at every stop width, slippage is 0.002R, and the
same 2bp round trip is −0.135R at a half-bar stop but −0.034R at a two-bar stop.

**Unclustered standard errors.** A random long-short spread was reported as
"+0.0195R, t=3.63" as though it measured drift. That t-stat treated 243,026
trades as independent when ~174 share each session's market move. A dated
re-run gives **−0.0140R** — opposite sign — with a year-to-year sd of 0.070R.
At a within-day correlation of only 0.05 the true t is 1.17.

**Every single-series t-stat in the intraday work is inflated for the same
reason.** The point estimates stand; the significance claims need clustering
before anyone quotes them. The *comparisons between arms* are safe, since both
arms trade the same days.

---

## 4. What this adds up to

Twenty-two pre-registered or semi-pre-registered tests across the project, all
using public information — published characteristics, chart shapes, session
times, price levels — in the most heavily traded market in the world. One
confirmed finding, and it is a description of competition rather than an edge.

That convergence is the result. It is not bad luck across twenty-two
independent attempts; it is the same answer restated.

**What is genuinely untested:** ES/NQ futures (511 MB, downloaded and verified
back-adjusted) with real Asia and London sessions, which is the instrument
every session-based claim is actually taught on. And the 2016-2020 window for
the ORB paper, whose 2021-2023 overlap with our data gives exactly zero.

**What has value regardless:** the apparatus. A pre-registration discipline, a
null that must break the specific claim, a universe leak check, a friction
decomposition, and a habit of rendering the thing and looking at it. Four real
errors surfaced in one session, three of them from the user's questions rather
than from the test suite.
