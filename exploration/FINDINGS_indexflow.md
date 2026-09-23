# Step 8 result: the constraint decayed too — on the most competed venue

Pre-registered in [`PREREG_indexflow.md`](PREREG_indexflow.md). The verdict is
**DECAYED LIKE EVERYTHING ELSE**, but the pre-registration contained a
mis-specification that limits what the result licenses.

## Why this test was different

Steps 1–7 all sought a **statistical pattern**. This sought a **constraint**:
when a stock enters the S&P 500, index funds tracking it must buy, at whatever
price, by a known date. That is the standard the brief opening this project
demanded and nothing since had met —

> *"Who is contractually obligated to close this gap, in both directions, at what
> bounded cost, by what date?"*

Patterns decay when people learn about them. A mandate is not a mistake, so the
prior was that it might decay differently.

## Result — 333 additions, survivorship-free prices, matched controls

Abnormal return versus five nearest neighbours matched on 60-day volatility and
12-month momentum:

| window | 2000–2008 | 2009–2016 | 2017–2026 |
|---|---|---|---|
| ED−10..ED−5 | +0.01% (t 0.0) | +0.37% (t 0.8) | −2.06% (t −0.8) |
| **ED−5..ED−1** | **+2.62% (t 3.2)** | +0.50% (t 1.0) | **+0.20% (t 0.3)** |
| ED−1..ED | −0.17% | −0.75% | −0.03% |
| ED..ED+5 | **−1.98% (t −2.9)** | −0.11% | −0.70% |
| ED..ED+20 | −1.99% | −0.01% | −0.47% |

Placebo null on ED−5..ED−1, 2017–2026: real **+0.20%**, placebo mean +0.00%,
95th percentile **+1.36%**. Well inside.

**The 2000–2008 row is the validation.** A +2.62% run-up into inclusion followed
by a −1.98% reversal, both significant, is the classic index effect at textbook
magnitude (Shleifer 1986; Harris & Gurel 1986). The event-study machinery
recovers a known effect at the right size in the right era — and then finds it
gone by 2017–2026.

## Corroborated in the wild, days before this was run

SpaceX entered the Nasdaq-100 on 7 July 2026 via a 15-day fast entry, forcing
~$27bn of buying from $800bn of tracking product. **The stock fell 6% on
inclusion day** as desks that had front-run it for weeks sold into the passive
demand.

The obligation is entirely real and unchanged. What has gone is any capturable
return at the moment the flow occurs — it has been competed forward and
arbitraged flat.

## The mis-specification, recorded

The pre-registration described the S&P 500 as *"the venue where it should be
strongest"*. That is wrong. It has the **largest flow** but the **most
competition**, and those pull opposite ways. It was the *hardest* venue, not the
easiest.

So this result does not distinguish two live hypotheses:

1. **Constraints decay like patterns** — obligations get front-run to zero
   wherever they exist.
2. **Constraints survive where competition is thin**, and the S&P 500 is where it
   is thickest.

The pre-registration said a decay verdict closes this direction. On the stated
reasoning that is too strong: what closed is the *S&P 500 index effect*, which
was already documented as attenuated.

Distinguishing the two needs a venue nobody builds a desk around — GICS
reclassifications, float-adjustment changes, small-index reconstitutions,
spin-off forced selling. Those are untested here.

## What is established

- The methodology recovers a known effect at correct magnitude and timing.
- The most visible forced-flow trade in equities no longer pays by the time it is
  observable.
- **Eight pre-registered tests, eight negatives**, now including one whose
  economics were structurally different from the other seven.
