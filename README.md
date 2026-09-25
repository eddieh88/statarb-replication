# Deep learning statistical arbitrage: a replication, and a test of whether it still works

Two studies. **Part 1** replicates the paper on the authors' own residuals
(1998–2016). **Part 2** buys survivorship-free data and asks whether any of it
survives into 2017–2026. The answer to Part 2 is no, and getting there required
correcting a result this README previously reported.

A line-by-line replication of **"Deep Learning Statistical Arbitrage"**
(Guijarro-Ordonez, Pelger & Zanotti, [arXiv:2106.04028](https://arxiv.org/abs/2106.04028)),
run on **the authors' own published CRSP residuals**, using their model code and
their preprocessing.

Two questions the paper does not answer, both of which turned out to matter:

1. **Does the headline Sharpe survive a null?** The paper contains no permutation
   test, no placebo, no synthetic control. (Verified: zero occurrences of
   *permutation, placebo, randomization, shuffle, surrogate, Monte Carlo,
   null hypothesis, data snooping, multiple testing* across all 68 pages.)
2. **Is profitability actually non-declining over time?** The paper's conclusion
   states *"We document non-declining profitability of arbitrage trading over
   time."* Every performance table pools 2002–2016; there is no per-period split.

**Answers: yes, and no.**

---

## Results

### The signal is real

Their CNN+Transformer, retrained every 250 days on a rolling 1000-day window,
PCA-5 residuals, residual-space weights:

| arm | pooled Sharpe (2002–2016) |
|---|---|
| CNN+Transformer | **+4.92** |
| permuted-return null, draw 1 | +0.43 |
| permuted-return null, draw 2 | +0.49 |

The null permutes **which stock receives which next-day return**, each day, among
that day's active names. Every input window, the daily cross-sectional return
distribution, and volatility clustering are preserved exactly; the only thing
destroyed is the link between a stock's own history and its own next return —
precisely what the strategy claims to exploit. The model is trained on the
permuted data too, so it gets every chance to find something that isn't there.

Real beat null in **28 of 30 block comparisons** (sign test p ≈ 3×10⁻⁵).

Harness validation: our OU+Threshold benchmark scores **0.70** on PCA-5 over
2002–2016, against the paper's published **0.73**. `src/real_ladder.py` prints
this check on startup.

### Profitability declined sharply — inside their own sample

Sharpe of each 250-day retrain block, in order:

| period | per-block Sharpe | mean |
|---|---|---|
| 2002–2008 | 9.94 · 8.74 · 8.43 · 6.09 · 7.18 · 4.35 · 7.14 | **+7.38** |
| 2009–2016 | 4.54 · 3.91 · 2.43 · 0.97 · 2.03 · 2.43 · 1.83 · 1.92 | **+2.46** |

Slope **−0.614 Sharpe/year**, t = **−8.92**, R² = 0.87.

It is not the artifact decaying. The null runs +0.47 in the first half and +0.42
in the second — flat, and indistinguishable from zero throughout.

### After realistic costs, nothing in this family works by 2009–2016

Costs are charged as `c × Σ|wₜ − wₜ₋₁|` with weights L1-normalised to gross 1,
plus general-collateral borrow at 35bp/yr on short notional. Breakeven is
verified leverage-invariant.

| arm (2009–2016) | gross | turnover | breakeven | net@1bp | **+ borrow** |
|---|---|---|---|---|---|
| reversal L=30 + no-trade band | 0.67 | 0.21 | 3.4bp | +0.47 | **+0.41** |
| ensemble 5/10/20/30 + band | 0.67 | 0.35 | 2.1bp | +0.36 | +0.30 |
| **CNN+Transformer** | **2.23** | **0.96** | **1.1bp** | +0.21 | **+0.06** |
| OU+Threshold (their benchmark) | 0.00 | 0.30 | 0.0bp | −0.48 | −0.59 |

A 30-day reversal rule with a no-trade band beats the transformer after costs.
The standard error on a Sharpe over 1,875 days is 0.37, so none of these are
distinguishable from zero — or from each other.

### Why the network loses despite 3× the gross Sharpe

Decomposing Sharpe into its parts, annualised, per unit of gross exposure:

| arm | Sharpe | **mean %/yr** | **vol %/yr** | turnover |
|---|---|---|---|---|
| reversal L=1 | 3.14 | **10.90** | 3.47 | 1.46 |
| ensemble | 1.52 | 5.06 | 3.32 | 0.38 |
| reversal L=30 | 1.07 | 3.37 | 3.15 | 0.25 |
| **CNN+Transformer** | **4.92** | **8.06** | **1.64** | 1.00 |

The network earns *less money* than one-day reversal. Its Sharpe is higher only
because its volatility is half everything else's — a Sharpe objective with no
cost term minimises variance by spreading weight evenly, and costs are paid out
of the mean, not out of the Sharpe. A low-mean, low-vol book is structurally
fragile to transaction costs, needs ~6× gross leverage to reach a 10% vol target,
and takes the largest risk-adjusted hit from a fixed borrow drag.

### Architecture was never the constraint

| variant | params | gross | turnover | breakeven |
|---|---|---|---|---|
| their CNN+Transformer | 769 | 4.92 | 1.00 | 3.2bp |
| + cross-sectional attention (ISAB) | 1,785 | 4.77 | 1.00 | 3.1bp |

Three variants within noise of one another (SE 0.26). A plausible reason the
cross-sectional block does nothing: residualisation has already removed the
cross-sectional structure it was looking for. That predicts it *should* help on
K=0 residuals (raw returns) — a clean, untested follow-up.

Consistent with this, the successor paper
([Attention Factors, arXiv:2510.11616](https://arxiv.org/abs/2510.11616), same
senior author) puts attention at the *factor-construction* stage rather than the
trading-policy stage, and reports 2.3 net of costs. Attention and factor
models appear to be substitutes; it belongs where the factors are made.

### The optimal linear filter is one-day reversal

Fitting all 30 lag weights by ridge, refit per block, the learned coefficients
concentrate on the last two lags (+0.77, −1.00) — which *is* one-day reversal.
The fit rediscovers it from 30 free parameters, then underperforms it
(2.70 vs 3.14), because the other 28 coefficients fit noise.

Shorter is monotonically better, gross: L=1 > L=2 > L=3 > L=5 > L=10 > L=30, and
exponential half-life 2d > 3d > 5d > 10d. Under costs the ordering **inverts
completely**, because L=1 turns over 1.46 of a maximum 2.0 every day.

L=1 also has an execution problem no cost assumption fixes: it requires being
positioned at close(t−1) on a signal determined *by* close(t−1).

---

## Supporting results

**Market neutrality.** Betas of 0.004–0.029 against Mkt-RF, R² of 0.1–3%. Longer
lookbacks carry more market exposure (L=1: 0.042, L=30: 0.177) — rolling-window
beta estimation error accumulates the longer you cumulate.

**Survivorship bias, measured on clean data.** Restricting the authors'
survivorship-free universe to survivors doubles reversal Sharpe (0.63 → 1.25,
**1.98×**); adding a large-cap restriction roughly doubles it again (→ 2.32).

**A permno↔ticker crosswalk without WRDS.** [Open Source Asset
Pricing](https://www.openassetpricing.com/) publishes 200+ characteristics keyed
by CRSP permno; our price panel is keyed by ticker, and every published crosswalk
sits behind WRDS. Solved by **fingerprint matching**: recompute a price-derived
characteristic (`MaxRet`) from our own data and match on correlation.

| check | result |
|---|---|
| coverage | 1,443 of 1,506 tickers (95.8%) |
| match quality | median r = 1.0000, min 0.9977 |
| **held-out signal** (`Mom12m`, unused in matching) | median r = **1.0000** |
| **shuffled-pairing control** | median r = **0.27** |
| known anchors (AAPL, MSFT, IBM, XOM, GE, JNJ, KO) | 7/7 exact |

It also recovers era-specific identities where a ticker changed entity (DD, DXC,
FTI), and pinned OSAP's `Mom12m` definition as the 11-month cumulative return
through t−1 (r = 1.000, vs 0.91 and 0.85 for the alternatives).

---

## Method

Every number above was produced under a rule fixed before estimation: **a result
counts only if identical code, run against a control, fails to produce it.**

That rule earned its keep. Four large, clean-looking results in this project
turned out to be artifacts, and **not one was caught by inspecting the number**:

| apparent result | what it actually was |
|---|---|
| a large rank-space Sharpe | order-statistics / collision effects |
| Sharpe ≈ 3 on pure noise | residual-space weight normalisation |
| null test at z = −7.8 | a sign-flip null leaving \|residual\| intact, so volatility timing passed straight through |
| Sharpe 4.2 on an extension past 2016 | survivorship bias |

Errors found and corrected along the way, listed because the corrections are the
content:

- a beta window that included day *t* — look-ahead; moved a null from 2.06 to 1.01
- an off-by-one pairing weights with the wrong day's returns
- a sign error in the OU signal, plus a filter carried over from an unrelated
  paper — 0.09 → 0.70
- a claim that the rank-space authors ignored the collision effect, fully
  retracted on finding their Appendix B derives it
- a positional encoding whose base-10000 schedule left 3 of 8 dimensions inert
  over a 30-step sequence
- a survivorship test built on a wrong model of the vendor's tagging, which
  returned a false negative until corrected
- a date index built from business days rather than trading days, landing 8
  months short and silently misaligning the Fama-French join

Pre-registrations are committed *before* the corresponding results
(`PREREG_DLSA.md`, and `rankspace/PREREGISTRATION.md` for that project), and
result files are tracked in git — a pre-registration is worthless if the result
can be quietly regenerated after the fact.

---

## Part 2 — does it still work? (survivorship-free, 2000–2026)

The authors' residuals stop at 2016-12. Answering the obvious next question
needed survivorship-free prices, since a currently-listed universe produced a
Sharpe of 4.22 where the truth was 0.63.

**Data.** 6,833 daily files from MarketParquet ($79, one-time), 2000-01-03 →
2026-09-18, 16,003 symbols of which ~60% no longer trade. Validated before use:

| check | result |
|---|---|
| delisted names in the 2010 file | 3,527 (vendor claims 3,516) |
| trading days per year after dropping 115 holiday files | 252/252/251/251/252 — exact NYSE |
| **residuals vs CRSP, same stock** | **median r = 0.881**, 354 confident matches |
| same 331 stocks, same days, same code | ours +1.33 vs theirs +1.17 (L=5) |

The last two matter most. An earlier version of this comparison was **compound** —
strategy Sharpe on our residuals against theirs — which conflates universe,
construction, and data. Matching stocks first showed our construction reproduces
theirs; the apparent gap was universe composition (we traded 500 names, their
file carries 890 active).

### The decay is horizon-specific

Reversal Sharpe, gross, on one consistent survivorship-free dataset:

| arm | 2002–2008 | 2009–2016 | 2017–2026 |
|---|---|---|---|
| reversal L=1 | +2.81 | +0.85 | **−0.03** |
| reversal L=5 | +1.87 | +0.87 | +0.36 |
| reversal L=30 | +0.86 | +0.74 | +0.47 |

One-day reversal — the strongest gross signal in the early era — is **dead**.
Longer lookbacks decayed more slowly. Short-horizon liquidity provision is where
competing capital concentrated.

### But choosing K honestly removes what looked like a survivor

The number of PCA factors was fixed at K=5 throughout, because the paper uses it.
Varying it changes the conclusion:

| K | 2009–2016 net | 2017–2026 net |
|---|---|---|
| 1 | +0.38 | +0.19 |
| 3 | +0.37 | +0.11 |
| **5** | +0.54 | **+0.37** |
| 10 | +0.92 | +0.08 |
| 15 | **+1.09** | +0.09 |

The 2017–2026 values sit within one standard error of each other (SE ≈ 0.32), and
**+0.37 is the maximum of the set**. Re-selecting K at each retrain from prior
data only:

| | 2009–2016 net | 2017–2026 net |
|---|---|---|
| **walk-forward K** | **+1.01** | **−0.02** |
| fixed K=5 (hindsight) | +0.54 | +0.37 |

The selector is not broken — in 2009–2016 it picks K=15 consistently and beats
every fixed choice. In 2017–2026 it churns between values that are
indistinguishable, because there is nothing stable to select.

**Correction.** This README previously reported that the signal survived at 0.47
gross / 0.35 net and had stopped decaying, and sized a retail book on it. That
was hindsight on an untested dimension. **Out of sample, the modern era nets zero.**

### The network does worse than the simple rule

Their CNN+Transformer, their code, trained on the same modern residuals:

| | CNN+Transformer | reversal L=30 |
|---|---|---|
| gross 2017–2026 | **+0.05** | +0.47 (K=5) |
| turnover | 0.80 | 0.25 |
| net @1bp | **−0.85** | +0.37 |
| positive periods | 5 of 10 blocks | 9 of 10 years |

Each model is trained from scratch on a **rolling four-year window**, which is the
paper's spec. By 2020 there is no strong-reversal era left inside that window to
learn from — so the architecture that scored 4.92 on 1998–2016 residuals has
nothing to extract. That also implies a real share of the original 4.92 came from
models trained on 1998–2001.

**Caveat, stated plainly:** this ran at a reduced configuration (30 epochs vs
their 100, retrain every 250 days vs 125) and on PCA-5 residuals rather than
their best IPCA arm. The simple benchmarks have no hyperparameters to underfit,
so the comparison is not symmetric. We have not shown the network fails at the
paper's own configuration — only that it earns nothing at ours, while a
zero-parameter rule on identical inputs earns more.

### What the whole thing amounts to — and a correction

The obvious reading is that the strategy sold liquidity provision — somebody must
sell urgently, pushes a price below where its peers say it belongs, you warehouse
the risk — and that this service fee was competed away.

**That explanation was tested in Part 3 and falsified.** Crowding predicts the
signal should persist where competing capital cannot reach. It does not: gross
Sharpe is within one standard error of zero in all five liquidity bands, from
$160M median daily volume down to $1M, with no gradient.

What actually changed:

| era | residual vol/day | cross-sectional dispersion | **AR(1)** |
|---|---|---|---|
| 2002–2008 | 2.060% | 3.910% | **−0.0269** |
| 2009–2016 | 1.740% | 3.252% | −0.0110 |
| 2017–2026 | **2.078%** | **3.713%** | **−0.0018** |

Idiosyncratic volatility and dispersion in 2017–2026 **exceed** 2009–2016 and
nearly match 2002–2008. Stocks move on their own news as much as they ever did.
What collapsed by 93% is the single quantity reversal trades.

**The overshoot stopped forming.** Prices still move on stock-specific news and
flow; those moves now stick. The service is not being provided by someone else
more cheaply — it is no longer required.

## Part 3 — what else was tested (`exploration/`)

Six pre-registered experiments, each with its statistic, null and thresholds
fixed before the data was touched. Two of the six grew into projects of their
own and now live in separate repositories — see
[Related repositories](#related-repositories).

| experiment | verdict |
|---|---|
| liquidity deciles — does it survive where crowding cannot reach? | **FALSIFIED** — flat across all five bands |
| longer holding periods | **dropped** — no reversion beyond lag 10 |
| 209 published characteristics, monthly | **DEAD** — inside its own null ([moved →](https://github.com/eddieh88/characteristic-factors)) |
| delayed overshoot — is it a timing failure? | **DEAD** — beats its null, loses to costs |
| did the overshoot move intraday? | **NOT INTRADAY** — and it flipped sign |
| continuation — does the flipped sign pay? | **DEAD** — gross zero in both directions |

### The mechanism

Reversal did not decay. **It became continuation**, measured three independent
ways at three timescales:

| measurement | early era | modern era |
|---|---|---|
| intraday, morning → afternoon | −0.0051 / −0.0772 | **+0.0219** |
| daily lag 1, large moves | −0.0262 | **+0.0036** |
| daily lag 1, all moves | −0.0269 | −0.0018 |

Idiosyncratic dispersion is **unchanged** — 2.078%/day in 2017–2026, above
2009–2016 and near 2002–2008. Stocks move on their own news as much as ever.
What vanished is the bounce.

And neither direction is tradeable. Continuation's gross Sharpe is **+0.01**;
reversal's is −0.03. Both lose the same amount to identical turnover. **For
portfolio purposes the modern residual is unforecastable from its own past in
either direction** — a martingale difference in practice, which is what an
efficient price looks like.

### Two corrections this produced

**The crowding explanation was wrong.** Competing capital predicts the signal
should survive where that capital cannot go. It does not: gross is within one
standard error of zero in every liquidity band from $160M median daily volume
down to $1M, with no gradient.

**And one of my own claims was wrong.** This project repeatedly argued that L=1
reversal is structurally unexecutable because its signal is determined by the
price it must trade at. Measured against 5-minute data, the 15:45→close move
carries 4% of a day's cross-sectional variance: a 15:45 signal is ~98% correlated
with a close signal. It is a 2% degradation, not a structural flaw. L=1 died
because large moves now continue at lag 1.

Details in [`exploration/`](exploration/), one pre-registration and one findings
document per step.

## Related repositories

This work started as one repository and split into three when the questions
stopped being the same question.

| repository | question | answer |
|---|---|---|
| **this one** | Does *Deep Learning Statistical Arbitrage* replicate, and does it still work? | Replicates; does not survive past 2016 |
| [**characteristic-factors**](https://github.com/eddieh88/characteristic-factors) | Do IPCA and the conditional autoencoder pay out of sample? | Ambiguous by the pre-registered rule, dead in substance — the tilt is beta +0.33 |
| [**intraday-patterns**](https://github.com/eddieh88/intraday-patterns) | Do the chart setups taught in trading education work? | No entry beats a naive momentum rule at the same bar |

The shared thread across all three is the error log. Every repository records
what went wrong and what caught it, because in this kind of work that is the
part that transfers.

## Reproducing

```bash
pip install -r requirements.txt
python3 setup_data.py            # downloads the authors' residuals, applies their
                                 # superMask, writes dlsa_real/*_masked.npy (~5 min)

# run from the repository root; results are written to results/
python3 src/real_ladder.py       # harness check: OU should land near 0.70
python3 src/per_block_bench.py   # per-retrain-block Sharpe, every benchmark arm
python3 src/filters.py           # lookback sweep, exponential filters, fitted 30-lag ridge
python3 src/netcost.py           # turnover, net Sharpe by cost, breakeven
python3 src/secondhalf.py        # the same, restricted to 2009–2016
python3 src/borrow.py            # adds stock-borrow costs
python3 src/market_beta.py       # Fama-French market exposure
python3 src/audit.py             # cost-model and leverage-invariance checks
python3 src/crosswalk.py         # permno<->ticker without WRDS
```

GPU arms (Colab, ~40 min each on a T4): `notebooks/DLSA_null_test_colab.ipynb`
(null test, per-block, turnover), `notebooks/DLSA_xsection_colab.ipynb`
(architecture ablation). Both open with a summary of their measured results, so
the findings are readable without running anything.

## References

- Guijarro-Ordonez, Pelger & Zanotti, *Deep Learning Statistical Arbitrage*, [arXiv:2106.04028](https://arxiv.org/abs/2106.04028) · [code](https://github.com/gregzanotti/dlsa-public)
- Epstein, Wang, Choi & Pelger, *Attention Factors for Statistical Arbitrage*, [arXiv:2510.11616](https://arxiv.org/abs/2510.11616) (ICAIF '25)
- Siranosian, [independent DLSA replication](https://bsiranosian.com/blog/deep-learning-statistical-arbitrage-part-1/) (2026) — reaches parity from scratch; reports OU at 1.16 against the paper's 0.73, which is roughly the survivorship factor measured here
- Chen & Zimmermann, [Open Source Asset Pricing](https://www.openassetpricing.com/)
- Lee et al., *Set Transformer* (ISAB), [arXiv:1810.00825](https://arxiv.org/abs/1810.00825)


## Repository map

```
setup_data.py     run first: fetches and masks the authors' residuals
src/              analysis code -- run from the repo root, e.g. python3 src/netcost.py
notebooks/        the two GPU arms, each carrying its measured results
results/          tracked JSON output
rankspace/        a separate replication, self-contained with its own README
archive/          superseded code, kept because the corrections are the record
hooks/            pre-commit secret scanner
```

Everything in `src/` imports `src/real_ladder.py`, which loads the masked residual
arrays and implements the authors' `preprocess_cumsum` and `preprocess_ou`.

**Core harness**

| file | what it does |
|---|---|
| `setup_data.py` | downloads + masks the authors' residuals. **Run first.** |
| `src/real_ladder.py` | shared harness: windows, validity mask, portfolio returns, OU rule |
| `src/per_block_bench.py` | per-retrain-block Sharpe for every non-neural arm |
| `src/filters.py` | lookback sweep, exponential filters, fitted 30-lag ridge, inverse-vol |
| `src/netcost.py` | turnover, net Sharpe by cost level, breakeven |
| `src/secondhalf.py` | the same, restricted to 2009–2016 |
| `src/borrow.py` | adds stock-borrow cost on the short leg |
| `src/sparsity.py` | performance vs number of positions held |
| `src/market_beta.py` | Fama-French market exposure of each arm |
| `src/audit.py` | checks the cost model: leverage-invariance, mean/vol decomposition |

**Neural arms** (GPU; Colab notebooks carry their measured results in a summary
cell so you can read the findings without running anything)

| file | what it does |
|---|---|
| `notebooks/DLSA_null_test_colab.ipynb` | null test, per-block Sharpe, turnover — imports **their** model and preprocessing |
| `notebooks/DLSA_xsection_colab.ipynb` | architecture ablation |
| `src/xsection_model.py` | faithful reimplementation (769 params, matching theirs) + positional encoding + ISAB cross-sectional block |
| `src/run_xsec_local.py` | CPU/MPS runner for the same, ~8.5h |

**Data and bias**

| file | what it does |
|---|---|
| `src/survivorship.py` | isolates survivorship bias on the authors' own clean universe |
| `src/extend.py` | the failed extension past 2016 — kept because the failure is the result |
| `src/crosswalk.py` | permno↔ticker without WRDS, by characteristic fingerprint matching |
| `src/mp_validate.py`, `src/mp_gateA.py` | pre-purchase gates for a survivorship-free price vendor |
| `src/fetch_big.py`, `src/fetch_shares.py` | build the yfinance price panel (used by `src/crosswalk.py` and `rankspace/`) |

**`rankspace/`** — a separate replication of Li & Papanicolaou
([arXiv:2410.06568](https://arxiv.org/abs/2410.06568)), self-contained with its
own README and pre-registration. Included because the method is the same and one
finding carries over: their parametric benchmark scores a gross Sharpe of
**13.86 on sorted pure noise** against 9.63 on real data. A construction can
manufacture apparent mean reversion with no cross-sectional signal present.

**Pre-registration:** `PREREG_DLSA.md`, committed before any model was trained.
Results files are tracked in git.

`archive/` holds superseded code, kept because the corrections are part of the
record — see `archive/README.md`.
