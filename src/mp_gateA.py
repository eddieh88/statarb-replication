"""Corrected Gate A.  The -DELISTED suffix is applied RETROACTIVELY to every
historical file, so a name that died in 2026 is already tagged in 2025 files.
The right test: take names tagged -DELISTED, confirm they have real continuous
prices while they were trading, and that they STOP appearing when they die.
Also re-run the split/quality gate restricted to the tradeable universe."""
import glob, pandas as pd, numpy as np
F = sorted(glob.glob("cache/mp_free/*.parquet"))
fr = {}
for p in F:
    d = p.split("/")[-1][:-8]
    f = pd.read_parquet(p)
    f["sym"] = f["symbol"].astype(str)
    f["base"] = f["sym"].str.replace("-DELISTED","",regex=False).str.upper()
    f["dead"] = f["sym"].str.contains("DELISTED")
    fr[d] = f
days = sorted(fr)
print(f"{len(days)} days {days[0]}..{days[-1]}")

first = fr[days[0]]
dead = first.loc[first.dead, "base"].tolist()
print(f"\n=== GATE A (corrected): {len(dead)} names tagged delisted on {days[0]} ===")
rows=[]
for t in dead:
    present=[d for d in days if (fr[d].base==t).any()]
    if not present: continue
    px=[float(fr[d].loc[fr[d].base==t,"close"].iloc[0]) for d in present]
    vol=[float(fr[d].loc[fr[d].base==t,"volume"].iloc[0]) for d in present]
    rows.append((t,len(present),present[-1],px[0],px[-1],np.mean(vol)))
r=pd.DataFrame(rows,columns=["sym","days_present","last_seen","first_px","last_px","avg_vol"])
print(f"  {len(r)} have price history in this window")
print(f"  days present: median {r.days_present.median():.0f} of {len(days)}")
print(f"  {(r.last_seen < days[-1]).sum()} stop appearing before {days[-1]}  <-- they die mid-window")
print("\n  sample (name, days present, last seen, first->last price):")
print(r.sort_values("avg_vol",ascending=False).head(8).to_string(index=False))

print("\n=== GATE D (corrected): quality INSIDE the tradeable universe ===")
px=(pd.concat([f.assign(d=d)[["d","base","close"]] for d,f in fr.items()])
      .pivot_table(index="d",columns="base",values="close",aggfunc="last").sort_index())
vol=(pd.concat([f.assign(d=d)[["d","base","close","volume"]] for d,f in fr.items()])
      .assign(dv=lambda x:x.close*x.volume)
      .pivot_table(index="d",columns="base",values="dv",aggfunc="last"))
for N in (500, 1000, 3000, px.shape[1]):
    top=vol.mean().nlargest(N).index
    sub=px[top]
    ret=sub.pct_change(fill_method=None)
    zero=(sub<=0).sum().sum()
    big=(ret.abs()>0.35).sum().sum()
    inf=np.isinf(ret.values).sum()
    print(f"  top {N:5d}: zero/neg prices {zero:5d}   |r|>35% {big:5d} "
          f"({big/max(ret.notna().sum().sum(),1):.3%} of obs)   inf {inf}")
