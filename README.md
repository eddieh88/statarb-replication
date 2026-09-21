# Deep Learning Statistical Arbitrage — an independent replication, and what survives costs

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

## What is still open

**Did any of this survive past 2016?** The authors' residuals end 2016-12. Our own
extension using a currently-listed universe read 4.22 where the truth was 0.63 —
survivorship manufacturing the entire result. Answering it needs
survivorship-free prices. A validation harness for one such source is included
(`src/mp_validate.py`, `src/mp_gateA.py`); the correct first test is whether that
data reproduces the 2002–2016 numbers above, before anything it says about 2017+
is believed.

**Does putting costs in the training objective change the picture?** This is the
successor paper's central claim, and the one modification aimed at the constraint
that actually binds. A `COST_BP` flag is wired into the notebook, unrun.

**Do multi-horizon holding periods help?** Every net comparison here was decided
by turnover; three architecture variants were not. The paper's own Section III
reports Sharpe ~1.5 at a one-month hold.

**Does the successor's 2.3 net survive a per-period split?** It reports no
subperiod analysis — the same omission this project found decisive in its
predecessor.

---

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
