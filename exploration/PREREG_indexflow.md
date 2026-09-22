# Pre-registration: does forced index flow move prices, and has it decayed?

Written before any event window is examined. Nothing below is revised after
seeing results.

## Why this is a different question

Everything tested so far in this project sought a **statistical pattern** —
residual reversal, characteristics, continuation, index reversion. All decayed,
and the project's own conclusion is that documented patterns are arbitraged away.

This tests a **constraint** instead. When a stock enters the S&P 500, index funds
tracking it are obligated to buy, at whatever price, by a known date. That is the
test the gas-market brief that opened this work demanded and that nothing since
has satisfied:

> *"Who is contractually obligated to close this gap, in both directions, at what
> bounded cost, by what date?"*

A constraint has a different decay profile from a pattern. Publishing "index
funds must buy on inclusion day" does not relieve them of the obligation. What
can decay is *who captures it and how early*, not whether the flow occurs.

The July 2026 SpaceX Nasdaq-100 entry is the visible case: ~$27bn of forced
buying from $800bn of tracking product, front-run for weeks, and the stock fell
6% on inclusion day as the front-runners sold into the passive demand.

## Prior, recorded now

**The S&P 500 index effect is documented to have attenuated**, and this is the
most-watched index in the world. I expect the pre-effective-date run-up to be
materially smaller in 2017–2026 than in 2000–2008, and possibly absent.

The point is not to discover the index effect. It is to measure **whether a
constraint-based effect decays the same way a pattern-based one does** — because
if it decays more slowly, or survives in less-watched venues, that justifies
building the full mechanical-flow calendar. If it is as dead as everything else,
the constraint framing is wrong and this direction closes cheaply.

## Data and its limits, stated first

Event dates from the Wikipedia S&P 500 constituents table: **333 additions,
2000–2026**, all matched to the survivorship-free price panel.

Three limitations that cannot be fixed with free data:

1. **The event sample is survivorship-biased.** Only *current* members are
   listed, so additions later removed are missing — and those are likely the
   underperformers. This biases post-addition returns upward. It should matter
   less for the pre-effective window, since deletions typically occur years
   later, but it is not zero.
2. **Effective dates only, no announcement dates.** The classic effect runs from
   announcement (≈5 trading days prior) to effective. We measure a window that
   contains the announcement rather than separating it.
3. **S&P 500 only** — the single most-competed index event.

All three push toward understating any effect except the post-event drift, which
is overstated. Recorded now so neither can be invoked selectively afterwards.

## Construction

For each addition with effective date ED, compute returns in:
`[ED−10, ED−5]`, `[ED−5, ED−1]`, `[ED−1, ED]`, `[ED, ED+5]`, `[ED, ED+20]`.

**Abnormal return** = event stock minus a matched control: same GICS-free
proxy — nearest neighbour by trailing 60-day volatility and trailing 12-month
return, drawn from the top-1500 dollar-volume universe on ED, never itself an
index event within ±60 days. Five controls per event, averaged.

## The statistic, fixed now

Mean abnormal return in `[ED−5, ED−1]` — the forced-buying run-up — by era, with
a t-statistic across events.

Secondary, reported but not decisive: `[ED, ED+20]` reversal, which the
survivorship bias inflates.

## The null, fixed now

**Placebo events.** For each real event, draw a pseudo-event: same calendar date,
a stock matched on volatility and momentum that was *not* added to the index.
Run the identical pipeline. 200 draws of the full event set.

This holds the calendar, the matching procedure and the universe fixed, and
destroys only the presence of forced flow — which is the specific claim.

## Decision thresholds

**CONSTRAINT HOLDS** — `[ED−5, ED−1]` abnormal return ≥ **+1.0%** with t ≥ 3 in
2017–2026, and outside the placebo distribution.

**DECAYED LIKE EVERYTHING ELSE** — 2017–2026 abnormal return ≤ **+0.25%** or
inside the placebo distribution, *and* materially smaller than 2000–2008.

**AMBIGUOUS** — anything else. Reported as such. Not re-cut by window, matching
variable, or universe.

## What each outcome licenses

- **Holds** → build the mechanical-flow calendar across index families, and the
  less-watched events (GICS reclassification, float adjustments, spin-offs) where
  competition is thinner.
- **Decayed** → the constraint framing fails its first test on the venue where it
  should be strongest, and this direction closes for the cost of one afternoon.
