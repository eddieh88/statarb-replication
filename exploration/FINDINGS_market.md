# Step 7 result: the measurement stands, the strategy does not

Pre-registered in [`PREREG_market.md`](PREREG_market.md). **The null in the
original pre-registration was wrong**; the amendment and the reason are recorded
there rather than the error being quietly replaced.

## The measurement

Mean reversion did not disappear from US equities. It moved:

| era | PC1 | equal-weight market | **cap-weight proxy** | residual |
|---|---|---|---|---|
| 2002–2008 | −0.0193 | −0.0293 | — | −0.0269 |
| 2009–2016 | −0.0038 | −0.0356 | — | −0.0110 |
| 2017–2026 | −0.0349 | −0.0692 | **−0.0838** | **−0.0018** |

Raw returns mean-revert *more* now (−0.0285) than in 2002–2008 (−0.0195). Linear
factor removal strips 87% of it in the modern era against 3% in the early era.
**Six exploration steps asked why reversal died in the residual. It did not die —
the methodology was subtracting it.**

The effect is **stronger cap-weighted than equal-weighted** (−0.0838 vs −0.0692,
net +0.37 vs +0.12). That is the decisive check against bid-ask bounce and stale
closes, both of which concentrate in small caps and would have shown the opposite
pattern.

## The interpretation

Indexing split order flow into two kinds that behave oppositely.

| | index/basket flow | single-stock flow |
|---|---|---|
| what it is | allocation, rebalancing, ETF create/redeem | a view on *that company* |
| informed | no | increasingly yes |
| price impact | temporary — overshoot | permanent — repricing |
| what follows | **reverts** | **continues** |

Buying SPY makes an authorised participant buy 500 stocks at once, carrying no
view about any of them. As indexing grew, the uninformed urgent flow migrated to
baskets, and the overshoot followed it. What remains in single-name trading is
disproportionately informed — which is why those moves now continue (+0.0036 at
lag 1) rather than bounce.

One story accounts for every measurement in this project: residual reversion
going to zero, single-stock moves flipping to continuation, market-level
reversion doubling, and the effect being strongest in exactly the cap-weighted
basket ETFs track.

**This is an interpretation consistent with the measurements, not a causal
finding.** Establishing it would need flow data we do not have.

## The strategy

Fade yesterday's index move, standardised by trailing 63-day volatility, clipped
at ±2, held one day, 2017–2026:

| | net | null median | null 95th | breakeven | positive years |
|---|---|---|---|---|---|
| equal-weight | +0.12 | −0.10 | +0.40 | 2bp | 6/10 |
| **cap-weight** | **+0.37** | −0.12 | +0.40 | 5bp | 6/10 |

Pre-registered verdict: **DEAD** — it required ≥ +0.60 and to clear the 95th
percentile, and does neither.

It sits at roughly the 93rd–94th percentile of the null (p ≈ 0.06 one-sided),
t ≈ 1.2 on its own SE of 0.32. **Costs are not the constraint** — breakeven 5bp
against 1bp assumed, because this is one instrument with no borrow, unlike every
prior failure which died of turnover across 900 positions.

What kills it is instability: **2022 −0.42, 2023 −1.24, 2024 −0.13**, three
consecutive losing years, then 2025 +1.29 and 2026 +0.68.

## A different kind of failure

The previous six steps failed because nothing was there. This one has something
there that a simple strategy cannot reliably extract — the profile of an effect
that is real, well documented, and already traded by desks with better execution
than ours.

## The null error, recorded

The original null was a 21-day block bootstrap, justified as destroying "serial
dependence at the daily horizon". A 21-day block keeps 20 of its 21 consecutive
pairs, so ~95% of the daily autocorrelation survives. The run made it obvious:
null median +0.12 against a real +0.12, and +0.38 against +0.37 — the null
reproduced the result because it essentially was the data.

Replaced with: decompose `r_t = sigma_t * z_t`, shuffle the z's, re-impose the
sigma path. Preserves volatility clustering, destroys daily serial dependence
completely. Null median moved to −0.11.

This is the same mistake as the sign-flip null in Part 1, which left every
|residual| intact and passed a strategy timing volatility. **A null must break
the specific claim; "roughly similar data" is not a null.** Twice now in this
project, and both times the broken null was generous rather than harsh.
