"""2017-2026 on survivorship-free data -- the question this project has been
blocked on since the authors' residuals ended in 2016.

PRE-REGISTERED before running (reversal arms only; OU is excluded because its
Sharpe is ~half drift, which our data measures differently -- see README):

  L=30 gross Sharpe, 2017-2026
    > 0.50   signal persisted
    < 0.20   dead
    between  ambiguous, report as such

  net of 1bp trading cost + 35bp/yr borrow on the short leg
    > 0.30   potentially tradeable at retail execution
    < 0.10   not

  The 2009-2016 figure on THIS data is the like-for-like baseline, not the
  CRSP figure -- same universe rule, same construction, same code.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np, pandas as pd
from real_ladder import windows_and_mask, sharpe, LOOKBACK

BORROW_BP_YR = 35.0

def arms(d):
    W, sel = windows_and_mask(d); last = W[:, :, -1]
    def rev(L): return -(last - (W[:, :, -L-1] if L < LOOKBACK else 0.0))
    a = {"L=1": rev(1), "L=5": rev(5), "L=10": rev(10), "L=30": rev(30)}
    a["ensemble"] = sum(x/np.abs(x).sum(1, keepdims=True).clip(1e-12)
                        for x in (rev(5), rev(10), rev(20), rev(30)))
    return a, sel, W

def band(w, b=0.5):
    out = np.zeros_like(w); prev = np.zeros(w.shape[1])
    for t in range(len(w)):
        thr = b*np.abs(w[t]).mean()
        cur = np.where(np.abs(w[t]-prev) > thr, w[t], prev)
        s = np.abs(cur).sum(); cur = cur/s if s > 0 else cur
        out[t] = cur; prev = cur
    return out

def evaluate(d, label, use_band=False):
    a, sel, W = arms(d)
    print(f"\n=== {label}  ({d.shape[0]} days, {int((d!=0).sum(1).mean())} names/day) ===")
    hdr = "{:10s} {:>7s} {:>7s} {:>8s} {:>9s} {:>10s}".format(
        "arm","gross","turn","b/e bp","net@1bp","+borrow")
    print(hdr); print("-"*56)
    res = {}
    for nm, Wt in a.items():
        w = np.where(sel, Wt, 0.0)
        w = w/np.abs(w).sum(1, keepdims=True).clip(1e-12)
        if use_band: w = band(w)
        g = (w[:-1]*d[LOOKBACK+1:LOOKBACK+len(w)]).sum(1) if False else \
            np.array([float(w[i] @ d[LOOKBACK+i]) for i in range(len(w))])
        to = np.abs(np.diff(w, axis=0)).sum(1); to = np.r_[to[0], to]
        short = np.abs(np.clip(w, None, 0)).sum(1)
        bor = short*BORROW_BP_YR*1e-4/252
        lo, hi = 0.0, 500.0
        for _ in range(60):
            m = (lo+hi)/2
            if sharpe(g - m*1e-4*to) > 0: lo = m
            else: hi = m
        res[nm] = (sharpe(g), to.mean(), lo, sharpe(g-1e-4*to), sharpe(g-1e-4*to-bor))
        print("{:10s} {:+7.2f} {:7.2f} {:8.1f} {:+9.2f} {:+10.2f}".format(nm, *res[nm]))
    return res

def main(eps="cache/mp_eps_K5_N900.parquet"):
    e = pd.read_parquet(eps)
    for lab, lo, hi in (("2002-2008", "2002-01-01", "2008-12-31"),
                        ("2009-2016", "2009-01-01", "2016-12-31"),
                        ("2017-2026", "2017-01-01", "2026-09-18")):
        sub = e.loc[lo:hi]; sub = sub.loc[:, sub.notna().any()]
        evaluate(np.nan_to_num(sub.values.astype(np.float32)), lab)
    print("\n--- with a no-trade band, 2017-2026 ---")
    sub = e.loc["2017-01-01":"2026-09-18"]; sub = sub.loc[:, sub.notna().any()]
    evaluate(np.nan_to_num(sub.values.astype(np.float32)), "2017-2026 + band", use_band=True)

if __name__ == "__main__":
    main(*sys.argv[1:])
