"""Continuation test, exactly as pre-registered in PREREG_continuation.md.

Two arms, both the sign-flip of strategies already measured:
  daily        w ∝ +resid(t-1)                       (negation of reversal L=1)
  conditional  same, top-quartile |resid(t-1)| only  (where the flip was measured)
"""
import numpy as np, pandas as pd
KS, LOOK, STEP = [1,3,5,10,15], 1000, 250
COST_BP, BORROW = 2.0, 35.0

def sharpe(x): return float(x.mean()/x.std()*np.sqrt(252)) if len(x) > 20 else float("nan")

def arm(v, conditional, seed=None):
    """v: days x names, nan = absent.  Returns gross, turnover, borrow."""
    T, N = v.shape
    rng = np.random.default_rng(seed) if seed is not None else None
    W = np.zeros((T, N))
    for t in range(1, T):
        x = v[t-1]; ok = np.isfinite(x)
        if ok.sum() < 100: continue
        w = np.zeros(N)
        if conditional:
            q = np.nanpercentile(np.abs(x[ok]), 75)
            sel = ok & (np.abs(x) >= q)
        else:
            sel = ok
        if sel.sum() < 20: continue
        w[sel] = x[sel]                      # + sign: CONTINUATION
        w -= w[sel].mean() * sel             # cross-sectionally demeaned
        a = np.abs(w).sum()
        if a > 0: W[t] = w/a
    r = np.zeros(T-1)
    for t in range(T-1):
        w = W[t]; f = v[t+1]
        m = (w != 0) & np.isfinite(f)
        if m.sum() < 5: continue
        ff = f[m]
        if rng is not None: ff = ff[rng.permutation(len(ff))]
        r[t] = float(w[m] @ ff)
    to = np.abs(np.diff(W, axis=0)).sum(1); to = np.r_[to[0], to][:T-1]
    bo = np.abs(np.clip(W, None, 0)).sum(1)[:T-1]*BORROW*1e-4/252
    return r, to, bo

def main():
    EK = {}
    for K in KS:
        e = pd.read_parquet(f"cache/mp_eps_K{K}_N900.parquet")
        EK[K] = e.loc[:, e.notna().any()]
    idx = EK[5].index
    for cond in (False, True):
        lab = "conditional (top-quartile move)" if cond else "daily (all names)"
        per = {}
        for K in KS:
            v = EK[K].values.astype(np.float64); v[v == 0] = np.nan
            r, to, bo = arm(v, cond)
            per[K] = (r - COST_BP*1e-4*to - bo, r, to)
        T = len(per[5][0]); starts = list(range(LOOK, T-STEP, STEP))
        wn, wg, wt, wd = [], [], [], []
        for s0 in starts:
            tr, te = slice(s0-LOOK, s0), slice(s0, min(s0+STEP, T))
            best = max(KS, key=lambda K: sharpe(per[K][0][tr]))
            wn.append(per[best][0][te]); wg.append(per[best][1][te])
            wt.append(per[best][2][te]); wd.append(idx[1:][te])
        wn, wg, wt = np.concatenate(wn), np.concatenate(wg), np.concatenate(wt)
        wd = pd.DatetimeIndex(np.concatenate(wd)); m = wd >= "2017-01-01"
        lo_, hi_ = 0.0, 500.0
        for _ in range(60):
            mm = (lo_+hi_)/2
            if sharpe(wg[m] - mm*1e-4*wt[m]) > 0: lo_ = mm
            else: hi_ = mm
        v5 = EK[5].values.astype(np.float64); v5[v5 == 0] = np.nan
        nulls = []
        for i in range(10):
            rn, tn, bn = arm(v5, cond, seed=3000+i)
            nn = (rn - COST_BP*1e-4*tn - bn)[-int(m.sum()):]
            nulls.append(sharpe(nn))
        nulls = np.array(nulls)
        yrs = pd.Series(wn[m], index=wd[m]).groupby(wd[m].year).apply(lambda s: sharpe(s.values))
        net, pos = sharpe(wn[m]), int((yrs > 0).sum())
        print(f"\n=== {lab} ===")
        print(f"  gross {sharpe(wg[m]):+.2f}   turnover {wt[m].mean():.2f}   breakeven {lo_:.0f}bp")
        print(f"  NET   {net:+.2f}    SE ~ {np.sqrt(252/m.sum()):.2f}")
        print(f"  null  mean {nulls.mean():+.2f}  max {nulls.max():+.2f}")
        print(f"  positive years {pos}/{len(yrs)}")
        if net >= 0.50 and net > nulls.max() and lo_ >= 6 and pos >= 7: vd = "WORKS"
        elif net <= 0.15 or net <= nulls.max(): vd = "DEAD"
        else: vd = "AMBIGUOUS"
        print(f"  VERDICT: {vd}")
    print("\ncomparators: reversal L=30 wf-K -0.02 | reversal L=1 -0.62 | fade H=3 -0.30")

if __name__ == "__main__":
    main()
