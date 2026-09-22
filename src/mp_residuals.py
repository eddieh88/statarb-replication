"""PCA residuals from the survivorship-free MarketParquet panel.

Follows the paper's construction as closely as the data allows:
  universe  top N by trailing 21-day dollar volume, re-selected every 21 days
            (the paper uses market cap > 0.01% of total; MarketParquet ships no
            shares outstanding, and dollar volume is the better proxy anyway for
            a strategy whose binding constraint turned out to be tradability)
  factors   K principal components of the trailing 252 days of returns,
            re-estimated every 21 days
  betas     OLS on the trailing 60 days, re-estimated daily, strictly EXCLUDING
            day t (Carhart 1997; including t is look-ahead and was a real bug
            earlier in this project)
  residual  eps_t = r_t - beta_{t-1} @ F_t

Everything is point-in-time: a name enters the universe on a date only if it was
trading on that date, delisted names included.

  python3 src/mp_residuals.py 2000-01-03 2026-09-18 500 5
"""
import sys
import numpy as np, pandas as pd

PCA_WIN, BETA_WIN, REFRESH = 252, 60, 21

def main(start="2000-01-03", end="2026-09-18", N=500, K=5, out=None):
    N, K = int(N), int(K)
    px = pd.read_parquet("cache/mp_close.parquet").loc[start:end]
    dv = pd.read_parquet("cache/mp_dv.parquet").loc[start:end]
    ret = px.pct_change(fill_method=None)
    ret = ret.mask(ret.abs() > 0.5)                 # guard bad ticks, not real moves
    print(f"panel {px.shape}  {px.index[0].date()} .. {px.index[-1].date()}")

    dvr = dv.rolling(21, min_periods=10).mean()
    T = len(ret)
    eps = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns, dtype="float32")
    warm = PCA_WIN + BETA_WIN
    load, univ = None, None

    for t in range(warm, T):
        if (t - warm) % REFRESH == 0 or load is None:
            # universe and factors, both estimated strictly before t
            cand = dvr.iloc[t-1].dropna()
            hist = ret.iloc[t-PCA_WIN:t]
            ok = hist.notna().sum() >= int(0.9 * PCA_WIN)
            cand = cand[cand.index.map(lambda c: bool(ok.get(c, False)))]
            univ = cand.nlargest(N).index
            X = hist[univ].fillna(0.0).values
            X = X - X.mean(0)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            load = Vt[:K]                            # K x N
        H = ret.iloc[t-BETA_WIN:t][univ].fillna(0.0).values     # 60 x N, excludes t
        F = H @ load.T                                          # 60 x K factor returns
        A = np.column_stack([np.ones(len(F)), F])
        beta = np.linalg.lstsq(A, H, rcond=None)[0][1:].T       # N x K
        r_t = ret.iloc[t][univ].values
        f_t = np.nan_to_num(r_t) @ load.T                       # K
        eps.iloc[t, [ret.columns.get_loc(c) for c in univ]] = (r_t - beta @ f_t).astype("float32")
        if (t - warm) % 500 == 0:
            print(f"  {t:5d}/{T}  {ret.index[t].date()}  univ {len(univ)}", flush=True)

    eps = eps.loc[ret.index[warm]:]
    out = out or f"cache/mp_eps_K{K}_N{N}.parquet"
    eps.to_parquet(out)
    cov = eps.notna().sum(1)
    print(f"\nresiduals {eps.shape}  {eps.index[0].date()} .. {eps.index[-1].date()}")
    print(f"  names/day {cov.median():.0f}   wrote {out}")

if __name__ == "__main__":
    main(*sys.argv[1:])
