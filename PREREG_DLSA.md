# Pre-registration: does the DLSA pipeline produce Sharpe on data with no signal?

Written 2026-09-18, **before any model was trained**. Nothing below is revised
after seeing results.

Tests the machinery of Guijarro-Ordonez, Pelger & Zanotti, "Deep Learning
Statistical Arbitrage" (arXiv:2106.04028v2). Their reported out-of-sample
Sharpe ratios reach 4.16 (IPCA-5 residuals, gross of costs; Table I).

## What is and is not being tested

This does **not** test whether their result is correct. Their data (CRSP,
survivorship-free, point-in-time shares, 46 Compustat characteristics for IPCA)
is not available here, so their headline IPCA number cannot be reproduced at all.

It tests one thing their paper does not: **what does this pipeline output when
trained and evaluated on data containing no directional signal?**

Their controls are (i) alphas against the Fama-French 8-factor model, (ii)
ablations across signal/allocation choices, (iii) hyperparameter grids and a
frozen-model test. None of these answer the question above. A flexible model
optimised directly on an in-sample Sharpe objective, with a rolling retrain, is
exactly the setting where a leak would be hardest to see and most damaging.

## Procedure

Identical code for both arms. The only difference is the input returns.

- **Universe:** fixed set of the largest N names by median capitalisation.
- **Residuals:** PCA with **K = 5**, fitted on a rolling 252-day window,
  loadings on a rolling 60-day window, recomputed every 250 days.
- **Input:** L = 30 days of cumulative residuals per name.
- **Model:** their architecture (Table A.II) — 2-layer CNN (D = 8 filters,
  size 2, instance norm, residual connection) -> 1-layer transformer
  (H = 4 heads, dropout 0.25) -> FFN (16, 8, 4). ~1k parameters.
- **Objective:** maximise the Sharpe ratio of the portfolio formed from the
  model's weights, normalised so that the absolute weights sum to one.
- **Protocol:** train on 1,000 days, evaluate out-of-sample on the following
  250 days, roll forward. Report the concatenated out-of-sample series.

## The null

Each trading day's **entire cross-sectional return vector has its sign flipped**
with probability 1/2, independently across days.

This preserves, exactly:
- the cross-sectional correlation structure (the whole vector flips together)
- volatility clustering (magnitudes keep their time ordering)
- the marginal distribution of every name

It destroys only the sign of each day's move, and therefore any directional
time-series predictability. It is the same null that resolved the rank-space
analysis in this repository on 2026-09-17, after three weaker nulls gave three
different answers.

## Decision threshold, set now

Let SR_null be the out-of-sample Sharpe from the sign-flipped arm, over R = 5
independent null draws.

- **PASS (pipeline is clean):** mean SR_null is within +/- 0.3 of zero, and the
  real arm exceeds the maximum SR_null across draws.
- **FAIL (pipeline leaks):** mean SR_null > 0.5.

If it fails, no comparison of real-data Sharpe ratios from this pipeline means
anything, and Phase 1 (costs) is abandoned.

## Known limitations, recorded in advance

- Survivorship-biased universe with static current share counts. This inflates
  the **real** arm, not the null arm, so it cannot manufacture a null failure.
- No IPCA residuals; PCA-5 only. Their PCA-5 result is 3.36, not 4.16.
- Weights are applied to residuals directly rather than mapped back to stock
  space through Phi'. Applied identically to both arms.
- Reduced compute: retrain every 250 days rather than 125, fewer epochs.
  Applied identically to both arms.

None of these affect the comparison under test, which is real versus null
through the same code.
