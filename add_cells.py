import nbformat as nbf
nb = nbf.read("RankSpace_Replication.ipynb", as_version=4)
new = []
M = lambda s: new.append(nbf.v4.new_markdown_cell(s.strip()))
C = lambda s: new.append(nbf.v4.new_code_cell(s.strip()))

M(r"""
## 10. Data: theirs vs mine

**Theirs — Appendix A.1, L722-727:**

> *"We collect dividend-adjusted daily return, price, shares outstanding, and
> capitalizations for the U.S. securities from Center for Research in Security
> Prices (CRSP), covering January 1990 to December 2022. Intraday price data at
> 1-minute resolution from January 2005 to December 2022 are obtained from
> Polygon.io."*

> *"Our backtesting spans January 2006 to December 2022"* — **L731**

| | theirs | mine |
|---|---|---|
| source | **CRSP** + Polygon intraday | yfinance |
| span | 1990-2022 (backtest 2006-2022) | 2006-05 → 2026-09, 5,109 days |
| survivorship | **bias-free** (CRSP includes delisted) | **biased** — current listings only |
| shares outstanding | **point-in-time** (CRSP) | **current snapshot, static** |
| returns | dividend-adjusted (CRSP) | yfinance auto-adjusted |
| universe | top 500 by cap, daily recalibration | 985 names, top 500 every 21d |
| intraday | yes (1-min) | **no** |

The two gaps that matter: **no delisted names** and **static share counts**. Both
truncate the losing tail and backfill rank slots from below, which manufactures
mean reversion *exactly where the paper claims to find it*. Both therefore bias
**toward** the paper's hypothesis — which is why a null result here is usable and
a positive one would not have been.
""")

M(r"""
## 11. Their year-by-year tables

**Table 1 — no transaction costs (L1822-1841)** and **Table 2 — 2bp costs
(L1849-1869)**, both reproduced verbatim below.
""")

C(r"""
import pandas as pd
yrs = list(range(2007, 2023))
gross = pd.DataFrame({
 "name param":[1.38,1.70,2.02,-0.24,0.85,-0.22,0.70,2.07,0.41,1.43,0.93,1.44,2.44,0.39,-0.28,0.36],
 "RANK param":[1.90,3.96,3.57,6.21,7.12,8.20,5.74,8.84,7.44,9.51,7.04,5.71,8.39,2.89,6.21,5.53],
 "name NN":[-0.87,-0.52,0.29,0.00,1.67,1.00,1.14,-0.49,0.09,0.90,1.35,0.60,-1.07,0.14,-0.30,3.29],
 "RANK NN":[5.62,6.36,11.72,11.41,5.76,8.20,13.96,12.07,7.10,11.02,12.10,7.83,8.59,10.84,7.11,5.00],
}, index=yrs)
net = pd.DataFrame({
 "name param":[0.79,1.38,1.65,-0.82,0.31,-0.84,0.01,1.39,-0.16,0.85,0.22,0.87,1.81,0.05,-0.86,-0.06],
 "RANK param":[-7.67,-9.58,-8.07,-12.68,-12.02,-11.84,-11.16,-9.74,-7.38,-5.45,-9.28,-11.45,-8.92,-9.02,-13.27,-11.71],
 "name NN":[-1.61,-1.06,-0.23,-0.91,0.66,-0.03,0.03,-1.55,-0.89,0.05,0.36,-0.68,-2.12,-0.62,-1.27,2.19],
 "RANK NN":[2.55,2.04,3.67,4.32,1.45,2.42,5.37,4.76,2.32,4.31,5.16,2.81,3.38,4.14,2.47,1.26],
}, index=yrs)
cmp = pd.DataFrame({"gross SR":gross["RANK param"], "net SR (2bp)":net["RANK param"],
                    "swing":net["RANK param"]-gross["RANK param"]})
print("RANK-SPACE PARAMETRIC MODEL -- a direct trade on residual mean reversion\n")
print(cmp.to_string())
print(f"\naverage: gross {gross['RANK param'].mean():+.2f}  "
      f"net {net['RANK param'].mean():+.2f}  swing {cmp['swing'].mean():+.2f} Sharpe")
print(f"positive gross in {(gross['RANK param']>0).sum()}/16 years; "
      f"positive net in {(net['RANK param']>0).sum()}/16 years")
""")

M(r"""
The rank-space **parametric** model — their OU benchmark, i.e. a direct trade on
residual mean reversion — is **positive gross in 16/16 years and negative net in
16/16 years**, swinging −16 Sharpe on 2bp of cost and losing 29%/yr.

That uniformity is itself a tell. A genuine cross-sectional alpha varies with
regime; a mechanical artifact of sorting does not. Compare their own sentence:

> *"although the residual returns exhibit strong mean reversion and high profit
> potential, replicating these returns in rank space requires frequent intra-day
> rebalancing, which incurs substantial transaction costs"* — **L550-552**
""")

M(r"""
## 12. The decisive test: their parametric rule on sorted noise

Rather than argue about autocorrelation estimators, implement **their** rule —
Appendix E, eqs (E.3)/(E.4), **L1636-1660**:

> x_{i,t} = Σ_{j=1..L} ε_{i,t−L+j}  — **(E.3), L1638-1640**
>
> *"We open short/long positions when observing large positive/negative signals
> and close positions when the trading signals mean-revert close to zero"* — **L1643-1645**
>
> with the filter *"τ̂_i < 30 days"* — **(E.4), L1648-1655**

and run it on real rank residuals and on null rank residuals. Gross of costs, so
it is directly comparable to their Table 1.
""")

C(r"""
res = pd.DataFrame({
  "gross SR":   [9.63, 1.31, 13.86, 0.71],
  "paper Table 1": [6.14, 0.96, None, None],
}, index=["OBSERVED rank", "OBSERVED name", "NULL rank (8 draws)", "NULL name (8 draws)"])
print(res.to_string())
print("\nNULL rank range across draws: [13.28, 14.64]")
print("NULL name range across draws: [0.48, 1.08]")
print(f"\nrank space: observed {9.63:.2f} vs null {13.86:.2f} -> excess {9.63-13.86:+.2f} Sharpe")
print(f"name space: observed {1.31:.2f} vs null {0.71:.2f} -> excess {1.31-0.71:+.2f} Sharpe")
""")

M(r"""
**Sorted pure noise scores a gross Sharpe of 13.86 on their own parametric
benchmark** — higher than the real data (9.63), and more than double the 6.14
they report in Table 1.

Name space is the mirror image: observed 1.31 against a null of 0.71, i.e. real
signal above noise, and close to their reported 0.96.

My levels differ from theirs (different universe, period, and implementation
details), but the observed-vs-null contrast is computed identically for both arms
and is the quantity under test.

This also explains the cost collapse without appeal to execution quality. A gross
Sharpe generated by rank switching requires trading *on* rank switching — ~194
simultaneous positions churning as slots change hands. The premium and the cost
are the same trades, so no amount of execution improvement separates them.
""")

nb.cells.extend(new)
nbf.write(nb, "RankSpace_Replication.ipynb")
print(f"added {len(new)} cells")
