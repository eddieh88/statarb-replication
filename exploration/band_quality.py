"""Data quality by liquidity band, BEFORE any strategy is run.

The pre-registration names two hazards that would manufacture the result we are
looking for, in exactly the bands we are looking in:
  * bad ticks -- a spurious down-move followed by a correction IS reversal
  * delisting -- a falling name whose price simply stops records no loss
"""
import numpy as np, pandas as pd
BANDS = [(1,900),(901,1800),(1801,2700),(2701,3600),(3601,4500)]
LO, HI = "2017-01-01", "2026-09-18"

px = pd.read_parquet("cache/mp_close.parquet").loc[LO:HI]
dv = pd.read_parquet("cache/mp_dv.parquet").loc[LO:HI]
dvr = dv.rolling(21, min_periods=10).mean()
ret = px.pct_change(fill_method=None)

print(f"panel {px.shape}  {px.index[0].date()} .. {px.index[-1].date()}")
print(f"median names/day {px.notna().sum(1).median():.0f}\n")
print("{:>12s} {:>9s} {:>11s} {:>9s} {:>9s} {:>10s} {:>11s}".format(
    "band","names/day","med $vol","zero/neg","|r|>35%","sub-$1","stops(/yr)"))
print("-"*78)
rows = {}
for a, b in BANDS:
    memb = dvr.rank(axis=1, ascending=False, method="first")
    m = (memb >= a) & (memb <= b)
    n = m.sum(1)
    if n.median() < 50:
        print(f"{f'{a}-{b}':>12s}  band mostly empty -- panel does not reach this depth"); continue
    sub = ret.where(m)
    px_b = px.where(m)
    obs = sub.notna().sum().sum()
    zero = (px_b <= 0).sum().sum()
    big  = (sub.abs() > 0.35).sum().sum()
    mdv  = float(np.nanmedian(dvr.where(m).values))
    cheap = float((px_b < 1).sum().sum() / max(px_b.notna().sum().sum(), 1))
    # "stops": last observation before the end of sample, per name-year
    lastday = px_b.apply(lambda c: c.last_valid_index())
    stops = (lastday < px.index[-5]).sum()
    yrs = (px.index[-1]-px.index[0]).days/365.25
    rows[(a,b)] = dict(n=n.median(), mdv=mdv, zero=zero, big=big/max(obs,1), cheap=cheap,
                       stops=stops/yrs)
    print("{:>12s} {:9.0f} {:>11s} {:9d} {:8.3%} {:9.2%} {:11.0f}".format(
        f"{a}-{b}", n.median(), f"${mdv/1e6:.1f}M", int(zero), big/max(obs,1), cheap, stops/yrs))
print("\nzero/neg prices and |r|>35% manufacture reversal.")
print("stops/yr = names whose price series ends mid-sample: delisting exposure.")
