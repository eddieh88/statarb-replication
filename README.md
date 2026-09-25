# Does deep-learning stat arb still work?

**Statistical arbitrage** bets that when a stock moves away from similar
stocks for no lasting reason, it will drift back. You buy the ones that fell
too far and short the ones that rose too far.

In 2021, *Deep Learning Statistical Arbitrage* (Guijarro-Ordonez, Pelger &
Zanotti, [arXiv:2106.04028](https://arxiv.org/abs/2106.04028)) reported that a
neural network doing this earned a **Sharpe ratio above 4**, and that its
profitability was **not declining over time**. For scale, a Sharpe above 1 is
good for a real strategy. Above 4 is extraordinary.

This repository checks that claim, then asks whether any of it still works
today.

## The short version

- **The paper replicates.** Run on the authors' own data with their own code,
  the network scores **4.92**, and a fake-data control scores about 0.46. The
  signal is real.
- **But it was dying inside their own sample.** It averaged 7.4 in 2002–2008
  and 2.5 in 2009–2016. That contradicts the paper's "non-declining"
  conclusion.
- **After trading costs, it barely survived even then.** By 2009–2016 the
  network netted **+0.06**. A plain 30-day reversal rule did better at +0.41.
- **After 2016 it is gone.** On clean data that includes delisted stocks, the
  network nets **−0.85** and the best honest simple rule nets **−0.02**.
- **The reason is not competition.** The price overshoot that the strategy
  trades has **stopped happening**. Stocks move as much as they ever did, but
  the moves no longer bounce back.

## The story

### 1. Replicating the paper

We ran the authors' published model on their own residuals — the part of each
stock's return left over after removing market-wide moves. Our benchmark
landed at **0.70** against their published 0.73, so the setup was right.

The paper never tested its result against a control. We built one: shuffle
which stock gets which next-day return, retrain the model on that, and see what
it scores. A real signal should beat the shuffle by a wide margin.

| | Sharpe, 2002–2016 |
|---|---|
| the paper's network | **+4.92** |
| same network, shuffled data | +0.43, +0.49 |

The real model beat the shuffle in **28 of 30** retraining periods.

### 2. It was already fading

The paper pools 2002–2016 into one number. Splitting it by period shows a
different picture:

| period | average Sharpe |
|---|---|
| 2002–2008 | **7.38** |
| 2009–2016 | **2.46** |

It fell about 0.6 Sharpe per year, steadily (t = −8.9). The shuffled control
stayed flat near zero throughout, so this is not the model decaying. The
opportunity was shrinking.

### 3. Trading costs erase the rest

The network turns over about half its book every day, so every basis point of
cost comes straight out of its returns. Costs here are 1bp to trade plus the
fee for borrowing shares to short.

| 2009–2016, after costs | Sharpe |
|---|---|
| 30-day reversal, simple rule | **+0.41** |
| the paper's network | **+0.06** |

The network made *less money* than one-day reversal. Its Sharpe looked higher
only because its returns were unusually smooth, and smoothness does not pay for
trading costs.

### 4. After 2016: gone

The authors' data stops at 2016. To go further we bought survivorship-free
prices: 16,003 stocks, about 60% of which no longer trade. That matters —
testing only on stocks that survived inflated one result from its true 0.63 to
**4.22**.

| 2017–2026, after costs | Sharpe |
|---|---|
| the paper's network, their code | **−0.85** |
| simple reversal, settings chosen honestly | **−0.02** |

We first reported this era as surviving at +0.37. **That was wrong, and it was
retracted.** The +0.37 came from picking the best of five settings after seeing
the results. Choosing the setting using only past data gives −0.02.

### 5. Why it died

The obvious explanation is competition: other funds crowded in. If that were
true, the strategy should still work in small, illiquid stocks where big
capital can't go. **It doesn't** — it is flat at zero across every liquidity
band, from $160M of daily volume down to $1M.

What actually changed is the thing being traded:

| era | how much stocks move on their own | how much those moves bounce back |
|---|---|---|
| 2002–2008 | 3.91% | **−0.027** |
| 2009–2016 | 3.25% | −0.011 |
| 2017–2026 | 3.71% | **−0.002** |

Stocks move on their own news almost as much as they did in 2002. But the bounce-back —
the overshoot this whole strategy is built on — has collapsed by **93%**. The
moves now stick. The service is not being provided more cheaply by someone
else. **It is no longer needed.**

We also tested whether the effect had moved somewhere else — longer holding
periods, intraday, a one-day delay, the opposite direction, the index level.
Every follow-up came back negative. Details are in [`exploration/`](exploration/).

## What went wrong along the way

Four results in this project looked large and clean and were artifacts. **None
of them was caught by looking at the number itself** — each was caught by a
control that should have failed and didn't:

- a large Sharpe that turned out to come from **sorting noise**
- a Sharpe of ~3 on pure noise, caused by **how the weights were normalised**
- a null test that let **volatility timing** leak straight through
- the **4.22** on post-2016 data, which was **survivorship bias**

Plus the retracted +0.37, and a series of smaller bugs: a look-ahead in a beta
window, an off-by-one date pairing, a sign error, a date index eight months
short. All are listed in [RESULTS.md](RESULTS.md#method).

## Where to go next

| | |
|---|---|
| **[RESULTS.md](RESULTS.md)** | The full technical write-up: every table, cost model, and correction |
| [exploration/](exploration/) | Six follow-up tests on why it died, each pre-registered |
| [NEXT_STEPS.md](NEXT_STEPS.md) | Where this ended, and what a different project would look like |
| [PREREG_DLSA.md](PREREG_DLSA.md) | The pre-registration, committed before any model was trained |
| [rankspace/](rankspace/) | A separate replication using the same method — its benchmark scores 13.86 on sorted pure noise |

## Words used here

| term | meaning |
|---|---|
| **Sharpe ratio** | Return divided by volatility, annualised. Above 1 is good; the paper's 4+ is extraordinary. |
| **residual** | What's left of a stock's return after removing the moves it shares with the market and a few other common factors. |
| **null / control** | The same test run on data where the effect has been deliberately destroyed. A real result must beat it. |
| **survivorship bias** | Testing only on stocks that still exist today, which quietly removes every company that failed. |
| **bp** | Basis point — one hundredth of a percent. |
| **reversal L=30** | Bet against each stock's last 30 days of residual movement. |

## Running it

```bash
pip install -r requirements.txt
python3 setup_data.py          # fetches and masks the authors' residuals (~5 min)
python3 src/real_ladder.py     # sanity check: the benchmark should land near 0.70
```

The full list of scripts is in [RESULTS.md](RESULTS.md#reproducing). A
MarketParquet key is expected at `~/.market_parquest/api_key.txt`, never in
the repo; the pre-commit hook in `hooks/` scans for it.

## Related repositories

This work started as one repository and split into three when the questions
stopped being the same question.

| repository | question | answer |
|---|---|---|
| **this one** | Does deep-learning stat arb replicate, and does it still work? | Replicates. Does not survive past 2016. |
| [**characteristic-factors**](https://github.com/eddieh88/characteristic-factors) | If prices alone stopped working, do models built on company fundamentals do better? | No. What they found was market exposure. |
| [**intraday-patterns**](https://github.com/eddieh88/intraday-patterns) | Do the chart setups taught in trading education work? | No. Every edge they have is plain momentum. |
