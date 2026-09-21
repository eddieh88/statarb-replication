"""
Build a permno <-> ticker crosswalk WITHOUT WRDS/CRSP.

Method: fingerprint matching. OSAP publishes price-derived characteristics keyed
by permno. We recompute the same characteristic from our own ticker panel and
match on correlation. A correct pairing gives r ~ 1.0; a wrong one ~ 0.2.
Primary fingerprint: MaxRet (max daily simple return within the calendar month).
Held-out check: Mom12m (11-month cumulative return through t-1), never used to
build the map.
"""
import numpy as np, pandas as pd, polars as pl, json, sys, time

PANEL   = 'cache/prices_big.parquet'
OSAP_FP = 'cache/osap_fingerprint.parquet'
WINDOWS = [(200602,200912),(201001,201312),(201401,201712),(201801,202112),(202201,202412)]
MIN_OVERLAP, MIN_CORR, MIN_MARGIN = 30, 0.98, 0.05


def our_maxret(px):
    r = px.pct_change(fill_method=None)
    r = r.mask(r.abs() > 1.0)
    ym = r.index.year*100 + r.index.month
    mx, cnt = r.groupby(ym).max(), r.groupby(ym).count()
    return mx.mask(cnt < 15)


def corr_block(A, B):
    """Pairwise correlation between every column of A and every column of B,
    over their overlapping non-NaN months."""
    def z(df):
        X = df.to_numpy(float)
        Z = (X - np.nanmean(X,0)) / np.where(np.nanstd(X,0) > 0, np.nanstd(X,0), np.nan)
        return np.nan_to_num(Z, nan=0.0), (~np.isnan(X)).astype(float)
    ZA, mA = z(A); ZB, mB = z(B)
    den = mA.T @ mB
    C = (ZA.T @ ZB) / np.maximum(den, 1)
    C[den < MIN_OVERLAP] = np.nan
    return C


def main():
    t0 = time.time()
    px = pd.read_parquet(PANEL)
    ours = our_maxret(px)
    o = pl.read_parquet(OSAP_FP).select('permno','yyyymm','MaxRet').drop_nulls()
    theirs = o.to_pandas().pivot(index='yyyymm', columns='permno', values='MaxRet')
    print(f'[{time.time()-t0:.0f}s] panel {ours.shape}, osap {theirs.shape}', flush=True)

    hits = {}   # ticker -> list of (corr, margin, permno, window)
    for lo, hi in WINDOWS:
        W = [m for m in ours.index if lo <= m <= hi and m in theirs.index]
        if len(W) < MIN_OVERLAP:
            continue
        A = ours.loc[W]; B = theirs.loc[W]
        A = A.loc[:, A.notna().sum() >= MIN_OVERLAP]
        B = B.loc[:, B.notna().sum() >= MIN_OVERLAP]
        C = corr_block(A, B)
        idx = np.argsort(-np.nan_to_num(C, nan=-9), axis=1)
        c1 = C[np.arange(len(idx)), idx[:,0]]
        c2 = C[np.arange(len(idx)), idx[:,1]]
        for i, tk in enumerate(A.columns):
            if np.isfinite(c1[i]) and c1[i] >= MIN_CORR and (c1[i]-c2[i]) >= MIN_MARGIN:
                hits.setdefault(tk, []).append((c1[i], c1[i]-c2[i], int(B.columns[idx[i,0]]), f'{lo}-{hi}'))
        print(f'[{time.time()-t0:.0f}s] {lo}-{hi}: {A.shape[1]} tickers x {B.shape[1]} permnos '
              f'-> {sum(1 for i in range(len(idx)) if np.isfinite(c1[i]) and c1[i]>=MIN_CORR and c1[i]-c2[i]>=MIN_MARGIN)} accepted', flush=True)

    rows, conflicts = [], []
    for tk, hs in hits.items():
        pns = {h[2] for h in hs}
        best = max(hs)
        rows.append({'ticker': tk, 'permno': best[2], 'corr': round(best[0],5),
                     'margin': round(best[1],4), 'n_windows': len(hs), 'window': best[3],
                     'ambiguous': len(pns) > 1})
        if len(pns) > 1:
            conflicts.append((tk, sorted({(h[2], h[3]) for h in hs})))

    cw = pd.DataFrame(rows).sort_values('ticker')
    cw.to_csv('cache/crosswalk.csv', index=False)
    print(f'\n{len(cw)} of {px.shape[1]} tickers mapped '
          f'({len(cw)/px.shape[1]:.1%}); {cw.ambiguous.sum()} ambiguous across windows')
    print(f'corr distribution: min {cw["corr"].min():.4f}, p5 {cw["corr"].quantile(.05):.4f}, median {cw["corr"].median():.4f}')
    if conflicts:
        print('\nticker reuse (same ticker, different permno by era) -- first 10:')
        for tk, v in conflicts[:10]:
            print(f'  {tk}: {v}')
    print('\nwrote cache/crosswalk.csv')

if __name__ == '__main__':
    main()
