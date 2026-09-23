"""Index-flow event study, exactly as pre-registered in PREREG_indexflow.md."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

px = pd.read_parquet("cache/mp_close.parquet")
dv = pd.read_parquet("cache/mp_dv.parquet")
ret = px.pct_change(fill_method=None).mask(lambda x: x.abs() > 0.5)
dvr = dv.rolling(21, min_periods=10).mean()
vol60 = ret.rolling(60, min_periods=30).std()
mom12 = px.pct_change(252, fill_method=None)

ev = pd.read_csv("cache/sp500_adds.csv")
ev["added"] = pd.to_datetime(ev.added)
ev = ev[ev.added.dt.year >= 2000]
cols = {c.upper(): c for c in px.columns}
ev["col"] = ev.clean.str.upper().map(cols).fillna(ev.symbol.str.upper().map(cols))
ev = ev.dropna(subset=["col"])
idx = px.index
WINDOWS = [("ED-10..ED-5",-10,-5), ("ED-5..ED-1",-5,-1), ("ED-1..ED",-1,0),
           ("ED..ED+5",0,5), ("ED..ED+20",0,20)]

def cum(col, i0, i1):
    if i0 < 0 or i1 >= len(idx): return np.nan
    a, b = px[col].iloc[i0], px[col].iloc[i1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0: return np.nan
    return b/a - 1

def controls(col, t, n=5, exclude=set()):
    """nearest neighbours by 60d vol and 12m momentum, from the top-1500 by $vol"""
    u = dvr.iloc[t].dropna().nlargest(1500).index
    u = [c for c in u if c != col and c not in exclude]
    v0, m0 = vol60[col].iloc[t], mom12[col].iloc[t]
    if not np.isfinite(v0) or not np.isfinite(m0): return []
    vv = vol60.iloc[t].reindex(u); mm = mom12.iloc[t].reindex(u)
    d = ((vv-v0)/max(v0,1e-9))**2 + ((mm-m0)/0.5)**2
    return list(d.dropna().nsmallest(n).index)

def run(events, placebo_seed=None):
    rng = np.random.default_rng(placebo_seed) if placebo_seed is not None else None
    ex = set(events.col)
    out = {w[0]: [] for w in WINDOWS}
    out["_era"] = []
    for _, r in events.iterrows():
        t = idx.searchsorted(r.added)
        if t < 300 or t > len(idx)-30: continue
        col = r.col
        if rng is not None:                       # placebo: same date, matched stock
            cc = controls(col, t, n=20, exclude=ex)
            if not cc: continue
            col = cc[rng.integers(0, len(cc))]
        ctrl = controls(col, t, n=5, exclude=ex)
        if len(ctrl) < 3: continue
        ok = True; vals = {}
        for nm, a, b in WINDOWS:
            e = cum(col, t+a, t+b)
            c = np.nanmean([cum(x, t+a, t+b) for x in ctrl])
            if not np.isfinite(e) or not np.isfinite(c): ok = False; break
            vals[nm] = e - c
        if ok:
            for k, v in vals.items(): out[k].append(v)
            out["_era"].append(r.added.year)
    return out

print(f"{len(ev)} S&P 500 additions matched to the panel, 2000-2026\n")
res = run(ev)
era = np.array(res["_era"])
print("Abnormal return vs matched controls (event minus 5 nearest neighbours)\n")
print("{:>14s}".format("window") + "".join(f"{e:>22s}" for e in ("2000-2008","2009-2016","2017-2026")))
print("{:>14s}".format("") + "".join(f"{'mean':>10s}{'t':>6s}{'n':>6s}" for _ in range(3)))
print("-"*80)
for nm, _, _ in WINDOWS:
    v = np.array(res[nm]); row = ""
    for lo, hi in ((2000,2008),(2009,2016),(2017,2026)):
        m = (era>=lo)&(era<=hi)
        x = v[m]
        t = x.mean()/x.std()*np.sqrt(len(x)) if len(x) > 3 else np.nan
        row += f"{x.mean()*100:+9.2f}%{t:6.1f}{len(x):6d}"
    print("{:>14s}".format(nm)+row)

KEY = "ED-5..ED-1"
print(f"\n=== placebo null on {KEY} (200 draws, 2017-2026) ===")
nulls = []
for i in range(200):
    p = run(ev, placebo_seed=1000+i)
    e2 = np.array(p["_era"]); v2 = np.array(p[KEY])
    m = (e2>=2017)
    if m.sum() > 10: nulls.append(v2[m].mean())
nulls = np.array(nulls)
v = np.array(res[KEY]); m = era>=2017
real = v[m].mean(); tt = v[m].mean()/v[m].std()*np.sqrt(m.sum())
print(f"  real {real*100:+.2f}%  t={tt:.1f}   placebo mean {nulls.mean()*100:+.2f}%  "
      f"95th pct {np.percentile(nulls,95)*100:+.2f}%")
early = np.array(res[KEY])[(era>=2000)&(era<=2008)].mean()
if real >= 0.010 and tt >= 3 and real > np.percentile(nulls,95): v_ = "CONSTRAINT HOLDS"
elif real <= 0.0025 or real <= np.percentile(nulls,95): v_ = "DECAYED LIKE EVERYTHING ELSE"
else: v_ = "AMBIGUOUS"
print(f"  2000-2008 was {early*100:+.2f}%")
print(f"\n  VERDICT: {v_}")
