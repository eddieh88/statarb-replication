"""Residuals per liquidity band, with an explicit delisting return.

A reversal strategy buys names that have fallen. Small caps delist constantly.
If a falling name's price simply STOPS, the backtest records no loss -- which
manufactures exactly the result we are testing for, in exactly the bands we are
testing. So a delisting return is assigned on the day after a name's last
observation, and the whole test is run at both 0% (optimistic) and -30%
(CRSP's convention for performance-related delistings).
"""
import sys, numpy as np, pandas as pd
PCA_WIN, BETA_WIN, REFRESH = 252, 60, 21
KS = [1, 3, 5, 10, 15]

def build(a, b, delist_ret, lo="2015-01-01", hi="2026-09-18"):
    px = pd.read_parquet("cache/mp_close.parquet").loc[lo:hi]
    dv = pd.read_parquet("cache/mp_dv.parquet").loc[lo:hi]
    ret = px.pct_change(fill_method=None).mask(lambda x: x.abs() > 0.5)

    # delisting: last valid observation before the end of sample -> assign return
    if delist_ret != 0.0:
        last = px.apply(lambda c: c.last_valid_index())
        cut = px.index[-5]
        dead = last[(last.notna()) & (last < cut)]
        pos = {d: i for i, d in enumerate(px.index)}
        n = 0
        for c, d in dead.items():
            i = pos[d] + 1
            if i < len(ret): ret.iat[i, ret.columns.get_loc(c)] = delist_ret; n += 1
        print(f"  assigned {delist_ret:+.0%} to {n} delistings", flush=True)

    dvr = dv.rolling(21, min_periods=10).mean()
    rank = dvr.rank(axis=1, ascending=False, method="first")
    T = len(ret); warm = PCA_WIN + BETA_WIN
    out = {K: pd.DataFrame(np.nan, index=ret.index, columns=ret.columns, dtype="float32")
           for K in KS}
    load, univ = {}, None
    for t in range(warm, T):
        if (t-warm) % REFRESH == 0 or univ is None:
            r_ = rank.iloc[t-1]
            cand = r_[(r_ >= a) & (r_ <= b)].index
            hist = ret.iloc[t-PCA_WIN:t]
            ok = hist[cand].notna().sum() >= int(0.9*PCA_WIN)
            univ = cand[ok.values]
            if len(univ) < 100: continue
            X = hist[univ].fillna(0.0).values; X = X - X.mean(0)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            for K in KS: load[K] = Vt[:K]
        if univ is None or len(univ) < 100: continue
        H = ret.iloc[t-BETA_WIN:t][univ].fillna(0.0).values
        r_t = ret.iloc[t][univ].values
        cols = [ret.columns.get_loc(c) for c in univ]
        for K in KS:
            F = H @ load[K].T
            A = np.column_stack([np.ones(len(F)), F])
            beta = np.linalg.lstsq(A, H, rcond=None)[0][1:].T
            out[K].iloc[t, cols] = (r_t - beta @ (np.nan_to_num(r_t) @ load[K].T)).astype("float32")
        if (t-warm) % 500 == 0: print(f"    {ret.index[t].date()}  n={len(univ)}", flush=True)
    tag = "d0" if delist_ret == 0 else "d30"
    for K in KS:
        e = out[K].loc[ret.index[warm]:]
        e = e.loc[:, e.notna().any()]
        e.to_parquet(f"cache/band_{a}_{b}_K{K}_{tag}.parquet")
    print(f"  band {a}-{b} {tag}: wrote K={KS}", flush=True)

if __name__ == "__main__":
    a, b, dr = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    build(a, b, dr)
