"""Step 1 evaluation, exactly as pre-registered in PREREG_liquidity.md.

Statistic : walk-forward-K net Sharpe of reversal L=30, 2017-2026, by band.
Costs     : 2 / 5 / 15 / 30 / 50 bp one-way by band; borrow 35bp/yr top two
            bands, 150bp/yr below; names under $1M median dollar volume excluded
            as unborrowable.
Thresholds: CONFIRMED  monotone rise over >=3 consecutive bands AND the
                       lowest-liquidity band clears +0.50 net
            FALSIFIED  no band exceeds +0.20 net, or non-monotone with none >0.50
            else       AMBIGUOUS, reported as such
Gross is printed for diagnosis only. No conclusion is drawn from a gross number.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, glob, re
import numpy as np, pandas as pd
from real_ladder import windows_and_mask, sharpe, LOOKBACK

BANDS = [(1,900),(901,1800),(1801,2700),(2701,3600),(3601,4500)]
COST  = {1:2.0, 901:5.0, 1801:15.0, 2701:30.0, 3601:50.0}
BORR  = {1:35.0, 901:35.0, 1801:150.0, 2701:150.0, 3601:150.0}
KS, LOOK, STEP = [1,3,5,10,15], 1000, 250

def daily(d, cost_bp, borrow_yr, L=30):
    W, sel = windows_and_mask(d); last = W[:,:,-1]
    Wt = -(last - (W[:,:,-L-1] if L < LOOKBACK else 0.0))
    w = np.where(sel, Wt, 0.0); w = w/np.abs(w).sum(1, keepdims=True).clip(1e-12)
    g  = np.array([float(w[i] @ d[LOOKBACK+i]) for i in range(len(w))])
    to = np.abs(np.diff(w, axis=0)).sum(1); to = np.r_[to[0], to]
    bo = np.abs(np.clip(w, None, 0)).sum(1)*borrow_yr*1e-4/252
    return g, g - cost_bp*1e-4*to - bo, to

def main(tag="d0"):
    dv = pd.read_parquet("cache/mp_dv.parquet")
    dvr = dv.rolling(21, min_periods=10).mean()
    print(f"delisting assumption: {'0%' if tag=='d0' else '-30%'}\n")
    print("{:>12s} {:>6s} {:>8s} {:>8s} {:>7s} {:>9s} {:>9s}".format(
        "band","cost","K modal","gross","turn","NET","n/day"))
    print("-"*64)
    out = {}
    for a, b in BANDS:
        fs = {K: f"cache/band_{a}_{b}_K{K}_{tag}.parquet" for K in KS}
        if not all(glob.glob(f) for f in fs.values()):
            print(f"{f'{a}-{b}':>12s}  residuals missing"); continue
        G, N, IDX = {}, {}, None
        for K in KS:
            e = pd.read_parquet(fs[K])
            # unborrowable exclusion: median 21d dollar volume under $1M
            elig = dvr.reindex(index=e.index, columns=e.columns).median() >= 1e6
            e = e.loc[:, elig.fillna(False)]
            g, n, to = daily(np.nan_to_num(e.values.astype(np.float32)), COST[a], BORR[a])
            G[K], N[K] = g, n
            IDX = e.index[LOOKBACK:LOOKBACK+len(g)]
            nd = int(e.notna().sum(1).median())
        T = len(IDX); starts = list(range(LOOK, T-STEP, STEP))
        picks, wg, wn, wd, wt = [], [], [], [], []
        for s0 in starts:
            tr, te = slice(s0-LOOK, s0), slice(s0, min(s0+STEP, T))
            best = max(KS, key=lambda K: sharpe(N[K][tr]))
            picks.append(best); wg.append(G[best][te]); wn.append(N[best][te]); wd.append(IDX[te])
        wg, wn = np.concatenate(wg), np.concatenate(wn)
        wd = pd.DatetimeIndex(np.concatenate(wd))
        m = wd >= "2017-01-01"
        from collections import Counter
        modal = Counter(picks).most_common(1)[0][0]
        out[(a,b)] = sharpe(wn[m])
        print("{:>12s} {:5.0f}bp {:>8s} {:+8.2f} {:>7s} {:+9.2f} {:9d}".format(
            f"{a}-{b}", COST[a], f"K{modal}", sharpe(wg[m]), "0.25", sharpe(wn[m]), nd))

    v = [out[k] for k in BANDS if k in out]
    if len(v) < 3: print("\nnot enough bands"); return
    print("\n" + "="*64)
    runs, best_run = 1, 1
    for i in range(1, len(v)):
        runs = runs+1 if v[i] > v[i-1] else 1
        best_run = max(best_run, runs)
    mono3 = best_run >= 3
    lowest = v[-1]
    if mono3 and lowest > 0.50: verdict = "CONFIRMED -- signal persists where crowding cannot reach"
    elif max(v) <= 0.20 or (not mono3 and max(v) <= 0.50): verdict = "FALSIFIED -- crowding does not explain the decay"
    else: verdict = "AMBIGUOUS"
    print(f"net by band: {[round(x,2) for x in v]}")
    print(f"longest rising run: {best_run} bands   lowest band: {lowest:+.2f}")
    print(verdict)
    print("="*64)

if __name__ == "__main__":
    main(*sys.argv[1:])
