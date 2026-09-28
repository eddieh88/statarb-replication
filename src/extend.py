"""Extend the decay analysis past 2016 using our own panel.

Validate on the 2006-2016 overlap against the authors' residuals; only then
read the 2017-2026 extension.
"""
import warnings
import numpy as np, pandas as pd, dlsa2 as d, real_ladder as rl

def rev_by_year(E, dates, L):
    """Trivial reversal, L1-normalised, Sharpe by calendar year."""
    E = np.nan_to_num(E); cs = np.cumsum(E, axis=0); T, N = E.shape
    ti, rr = [], []
    for t in range(L+1, T):
        v = ~np.any(E[t-L:t] == 0, axis=0) & np.isfinite(E[t])
        w = np.where(v, -(cs[t-1]-cs[t-L-1]), 0.0)
        s = np.abs(w).sum()
        if s > 0: ti.append(t); rr.append(float(w @ E[t])/s)
    s = pd.Series(rr, index=dates[np.array(ti)])
    return s.groupby(s.index.year).apply(lambda x: x.mean()/x.std()*np.sqrt(252))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # ---- theirs ----
    dt = pd.to_datetime(np.load("dlsa_real/dates.npy"))
    Et = rl.load("PCA-5")
    theirs = {L: rev_by_year(Et, dt, L) for L in (5, 30)}

    # ---- ours ----
    R = d.load_returns(500)
    print(f"our panel: {R.shape[1]} names, {len(R):,} days, "
          f"{R.index.min():%Y-%m} to {R.index.max():%Y-%m}", flush=True)
    Eo, _, _ = d.build(R, K=5)
    ours = {L: rev_by_year(Eo, R.index, L) for L in (5, 30)}

    yrs = sorted(set(theirs[5].index) | set(ours[5].index))
    print(f"\n{'year':>6}{'THEIRS L=5':>12}{'OURS L=5':>10}{'THEIRS L=30':>13}{'OURS L=30':>11}")
    print("-"*54)
    for y in yrs:
        f = lambda s: f"{s[y]:>10.2f}" if y in s.index else f"{'-':>10}"
        g = lambda s: f"{s[y]:>11.2f}" if y in s.index else f"{'-':>11}"
        print(f"{y:>6}{f(theirs[5])+'  '}{f(ours[5])}{g(theirs[30])}{g(ours[30])[:11]}")
    ov = [y for y in yrs if y in theirs[5].index and y in ours[5].index]
    print(f"\noverlap {min(ov)}-{max(ov)}:")
    for L in (5, 30):
        t = theirs[L][ov].mean(); o = ours[L][ov].mean()
        c = np.corrcoef(theirs[L][ov], ours[L][ov])[0,1]
        print(f"  L={L:<3} theirs {t:+.2f}   ours {o:+.2f}   year-by-year corr {c:+.2f}")
    post = [y for y in ours[5].index if y >= 2017]
    print(f"\nEXTENSION {min(post)}-{max(post)} (ours only):")
    for L in (5, 30):
        print(f"  L={L:<3} mean Sharpe {ours[L][post].mean():+.2f}   "
              f"by year {np.round(ours[L][post].values,2).tolist()}")
