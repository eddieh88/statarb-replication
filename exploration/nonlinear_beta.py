"""Diagnostic: does the residual's AR(1) depend on how the factors were removed?

All six exploration steps took the residuals as given.  If linear betas leave
nonlinear factor exposure behind, that exposure is persistent and would dilute
reversion in the true idiosyncratic part -- making the martingale-difference
finding an artifact of construction rather than a fact about the market.

Four constructions, identical universe / factors / windows, only the beta step
differs.  This is a diagnostic, not a strategy test: no thresholds, no null.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
PCA_WIN, BETA_WIN, REFRESH, K, N = 252, 60, 21, 5, 900

px = pd.read_parquet("cache/mp_close.parquet")
dv = pd.read_parquet("cache/mp_dv.parquet")
ret = px.pct_change(fill_method=None).mask(lambda x: x.abs() > 0.5)
dvr = dv.rolling(21, min_periods=10).mean()

def build(kind, lo, hi):
    r = ret.loc[lo:hi]; d = dvr.loc[lo:hi]
    T = len(r); warm = PCA_WIN + BETA_WIN
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns, dtype="float32")
    load = univ = None
    for t in range(warm, T):
        if (t-warm) % REFRESH == 0 or load is None:
            cand = d.iloc[t-1].dropna()
            hist = r.iloc[t-PCA_WIN:t]
            ok = hist.notna().sum() >= int(0.9*PCA_WIN)
            cand = cand[cand.index.map(lambda c: bool(ok.get(c, False)))]
            univ = cand.nlargest(N).index
            X = hist[univ].fillna(0.0).values; X = X - X.mean(0)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            load = Vt[:K]
        H = r.iloc[t-BETA_WIN:t][univ].fillna(0.0).values
        F = H @ load.T
        r_t = r.iloc[t][univ].values
        f_t = np.nan_to_num(r_t) @ load.T
        if kind == "linear":
            A = np.column_stack([np.ones(len(F)), F]); ft = f_t
        elif kind == "quadratic":
            A = np.column_stack([np.ones(len(F)), F, F**2]); ft = np.r_[f_t, f_t**2]
        elif kind == "asymmetric":                       # separate up/down betas
            A = np.column_stack([np.ones(len(F)), np.clip(F,0,None), np.clip(F,None,0)])
            ft = np.r_[np.clip(f_t,0,None), np.clip(f_t,None,0)]
        elif kind == "none":                             # raw returns, no factors removed
            out.iloc[t, [r.columns.get_loc(c) for c in univ]] = r_t.astype("float32"); continue
        b = np.linalg.lstsq(A, H, rcond=None)[0][1:].T
        out.iloc[t, [r.columns.get_loc(c) for c in univ]] = (r_t - b @ ft).astype("float32")
    return out.loc[r.index[warm]:]

def ar1(e):
    v = e.values.astype(np.float64); ac = []
    for j in range(v.shape[1]):
        c = v[:, j]; c = c[np.isfinite(c)]
        if len(c) > 400: ac.append(np.corrcoef(c[:-1], c[1:])[0,1])
    return np.median(ac), np.nanmedian(np.nanstd(v, axis=1))

print("Residual lag-1 autocorrelation by factor-removal method\n")
print("{:>12s}".format("era") + "".join(f"{k:>14s}" for k in ("none (raw)","linear","quadratic","asymmetric")))
print("-"*(12+14*4))
for lab, lo, hi in (("2002-2008","2002-01-01","2008-12-31"),
                    ("2017-2026","2017-01-01","2026-09-18")):
    row = []
    for kind in ("none","linear","quadratic","asymmetric"):
        a, v = build(kind, lo, hi).pipe(ar1)
        row.append(a)
    print("{:>12s}".format(lab) + "".join(f"{x:+14.4f}" for x in row))
print("\nif construction were the issue, a richer beta would restore reversion.")
