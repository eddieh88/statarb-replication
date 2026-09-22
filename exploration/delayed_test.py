"""Delayed-overshoot test, exactly as pre-registered in PREREG_delayed.md.

Trigger |resid(t-1)| in the top cross-sectional quartile; short the move; one-day
delay; hold H days overlapping. Null permutes forward returns AMONG TRIGGERED
NAMES ONLY, holding the trigger rule and position count fixed.
"""
import sys, numpy as np, pandas as pd
KS, LOOK, STEP = [1,3,5,10,15], 1000, 250
COST_BP, BORROW = 2.0, 35.0

def load_K():
    d = {}
    for K in KS:
        e = pd.read_parquet(f"cache/mp_eps_K{K}_N900.parquet")
        d[K] = e.loc[:, e.notna().any()]
    return d

def weights(v, H, seed=None):
    """v: days x names residual array (nan = absent). Returns weights, triggered mask."""
    T, N = v.shape
    rng = np.random.default_rng(seed) if seed is not None else None
    W = np.zeros((T, N)); 
    raw = np.zeros((T, N))
    for t in range(1, T):
        x = v[t-1]
        ok = np.isfinite(x)
        if ok.sum() < 100: continue
        q = np.nanpercentile(np.abs(x[ok]), 75)
        trig = ok & (np.abs(x) >= q)
        if trig.sum() < 20: continue
        s = -np.sign(x)                       # short the move
        w = np.zeros(N); w[trig] = s[trig]
        raw[t] = w
    # overlapping hold: average the last H signals
    for t in range(T):
        lo = max(0, t-H+1)
        w = raw[lo:t+1].mean(0)
        a = np.abs(w).sum()
        if a > 0: W[t] = w/a
    return W

def pnl(W, v, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    T = len(W); r = np.zeros(T-1); 
    for t in range(T-1):
        w = W[t]; fwd = v[t+1]
        m = (w != 0) & np.isfinite(fwd)
        if m.sum() < 5: continue
        f = fwd[m]
        if rng is not None: f = f[rng.permutation(len(f))]   # permute AMONG TRIGGERED
        r[t] = float(w[m] @ f)
    to = np.abs(np.diff(W, axis=0)).sum(1); to = np.r_[to[0], to][:T-1]
    short = np.abs(np.clip(W, None, 0)).sum(1)[:T-1]
    return r, to, short*BORROW*1e-4/252

def sharpe(x): return float(x.mean()/x.std()*np.sqrt(252)) if len(x) > 20 else float("nan")

def main():
    EK = load_K()
    idx = EK[5].index
    print(f"residuals {EK[5].shape}  {idx[0].date()} .. {idx[-1].date()}")
    for H in (3, 5):
        # walk-forward K on net of the delayed strategy itself
        per = {}
        for K in KS:
            v = EK[K].values.astype(np.float64); v[v == 0] = np.nan
            W = weights(v, H)
            r, to, bo = pnl(W, v)
            per[K] = (r - COST_BP*1e-4*to - bo, r, to)
        T = len(per[5][0]); starts = list(range(LOOK, T-STEP, STEP))
        wf, wg, wt, wd = [], [], [], []
        for s0 in starts:
            tr, te = slice(s0-LOOK, s0), slice(s0, min(s0+STEP, T))
            best = max(KS, key=lambda K: sharpe(per[K][0][tr]))
            wf.append(per[best][0][te]); wg.append(per[best][1][te]); wt.append(per[best][2][te])
            wd.append(idx[1:][te])
        wf, wg, wt = np.concatenate(wf), np.concatenate(wg), np.concatenate(wt)
        wd = pd.DatetimeIndex(np.concatenate(wd))
        m = wd >= "2017-01-01"
        lo_, hi_ = 0.0, 500.0
        for _ in range(60):
            mm = (lo_+hi_)/2
            if sharpe(wg[m] - mm*1e-4*wt[m]) > 0: lo_ = mm
            else: hi_ = mm
        # null, trigger-preserving, on the modal K
        from collections import Counter
        v5 = EK[5].values.astype(np.float64); v5[v5 == 0] = np.nan
        W5 = weights(v5, H)
        nulls = []
        for i in range(10):
            rn, tn, bn = pnl(W5, v5, seed=2000+i)
            nn = rn - COST_BP*1e-4*tn - bn
            nulls.append(sharpe(nn[-int(m.sum()):]))
        nulls = np.array(nulls)
        yrs = pd.Series(wf[m], index=wd[m]).groupby(wd[m].year).apply(lambda s: sharpe(s.values))
        pos = int((yrs > 0).sum())
        net = sharpe(wf[m])
        print(f"\n=== H={H} days ===")
        print(f"  gross {sharpe(wg[m]):+.2f}   turnover {wt[m].mean():.2f}   breakeven {lo_:.0f}bp")
        print(f"  NET   {net:+.2f}   (walk-forward K)      SE ~ {np.sqrt(252/m.sum()):.2f}")
        print(f"  null  mean {nulls.mean():+.2f}  max {nulls.max():+.2f}  {np.round(nulls,2).tolist()}")
        print(f"  positive years: {pos}/{len(yrs)}   {dict(yrs.round(2))}")
        if net >= 0.50 and net > nulls.max() and lo_ >= 6 and pos >= 7: v = "WORKS"
        elif net <= 0.15 or net <= nulls.max(): v = "DEAD"
        else: v = "AMBIGUOUS"
        print(f"  VERDICT: {v}")

if __name__ == "__main__":
    main()
