"""Index-reversion test, exactly as pre-registered in PREREG_market.md."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
COST_BP, VOL_WIN, CLIP = 1.0, 63, 2.0

px = pd.read_parquet("cache/mp_close.parquet")
dv = pd.read_parquet("cache/mp_dv.parquet")
ret = px.pct_change(fill_method=None).mask(lambda x: x.abs() > 0.5)
dvr = dv.rolling(21, min_periods=10).mean()

def series(weight):
    """daily return of the top-900 basket, re-selected every 21d, point-in-time"""
    out, univ = [], None
    idx = ret.index
    for t in range(252, len(ret)):
        if (t-252) % 21 == 0 or univ is None:
            c = dvr.iloc[t-1].dropna()
            univ = c.nlargest(900).index
            w = None if weight == "eq" else dvr.iloc[t-1].reindex(univ)
        r = ret.iloc[t][univ]
        ok = r.notna()
        if ok.sum() < 200: out.append((idx[t], np.nan)); continue
        if weight == "eq":
            out.append((idx[t], float(r[ok].mean())))
        else:                                  # dollar-volume weight ~ cap proxy
            ww = w[ok]; ww = ww/ww.sum()
            out.append((idx[t], float((r[ok]*ww).sum())))
    s = pd.Series(dict(out)).dropna()
    return s

def strat(s):
    v = s.rolling(VOL_WIN, min_periods=20).std().shift(1)
    z = (s/v).clip(-CLIP, CLIP)
    pos = (-z).shift(1)                        # position from yesterday's move
    r = (pos*s).dropna()
    to = pos.diff().abs().reindex(r.index).fillna(0.0)
    return r, to

def sharpe(x): return float(x.mean()/x.std()*np.sqrt(252)) if len(x) > 20 else float("nan")

def evaluate(s, lab):
    r, to = strat(s)
    m = r.index >= "2017-01-01"
    g, t = r[m].values, to[m].values
    net = g - COST_BP*1e-4*t
    lo, hi = 0.0, 500.0
    for _ in range(60):
        mm = (lo+hi)/2
        if sharpe(g - mm*1e-4*t) > 0: lo = mm
        else: hi = mm
    yrs = pd.Series(net, index=r.index[m]).groupby(r.index[m].year).apply(lambda x: sharpe(x.values))
    # NULL (amended): r_t = sigma_t * z_t.  Shuffle z, keep the sigma path.
    # Preserves volatility clustering, destroys daily serial dependence fully.
    # The original 21-day block bootstrap preserved ~95% of the autocorrelation
    # it was meant to destroy -- see PREREG_market.md.
    rng = np.random.default_rng(0)
    sm = s[s.index >= "2016-01-01"]
    sig = sm.rolling(VOL_WIN, min_periods=20).std().shift(1)
    z = (sm/sig).dropna()
    sig = sig.reindex(z.index)
    nulls = []
    for _ in range(200):
        zs = z.values.copy(); rng.shuffle(zs)
        bss = pd.Series(zs*sig.values, index=z.index)
        rn, tn = strat(bss)
        nulls.append(sharpe(rn.values - COST_BP*1e-4*tn.values))
    nulls = np.array(nulls)
    print(f"\n=== {lab} ===")
    print(f"  gross {sharpe(g):+.2f}   turnover {t.mean():.2f}   breakeven {lo:.0f}bp")
    print(f"  NET   {sharpe(net):+.2f}   SE ~ {np.sqrt(252/m.sum()):.2f}")
    print(f"  null  median {np.median(nulls):+.2f}  95th pct {np.percentile(nulls,95):+.2f}")
    print(f"  positive years {int((yrs>0).sum())}/{len(yrs)}   {dict(yrs.round(2))}")
    return sharpe(net), np.percentile(nulls,95), lo, int((yrs>0).sum()), len(yrs)

eq = series("eq"); cw = series("cw")
print(f"series built: {len(eq)} days, {eq.index[0].date()} .. {eq.index[-1].date()}")
print(f"AR(1) 2017-2026  eq {eq[eq.index>='2017'].autocorr(1):+.4f}   "
      f"cap-proxy {cw[cw.index>='2017'].autocorr(1):+.4f}")
ne, p95e, bee, pye, ny = evaluate(eq, "equal-weight (where the effect was measured)")
nc, p95c, bec, pyc, _  = evaluate(cw, "cap-weight proxy (what you could trade)")

print("\n" + "="*58)
ratio = nc/ne if ne != 0 else 0
print(f"cap-weight / equal-weight net = {ratio:.2f}")
if ne >= 0.60 and ne > p95e and bee >= 3 and pye >= 7 and ratio >= 0.5: v = "WORKS"
elif ne <= 0.20 or ne <= p95e or ratio < 0.25: v = "DEAD"
else: v = "AMBIGUOUS"
print(f"VERDICT: {v}")
print("="*58)
