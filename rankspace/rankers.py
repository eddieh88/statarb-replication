"""Does the CHOICE of ranker matter, or only the churn it induces?

If mean reversion in rank space comes from collisions (Thm B.3), then ANY
ranking variable should produce it, and the magnitude should track churn rate
rather than the economic content of the ranker.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

px = pd.read_parquet("cache/prices_big.parquet")
px = px[px.index >= "2010-01-01"]
px = px.dropna(axis=1, thresh=int(len(px)*0.98)).ffill().dropna()
sh = pd.read_parquet("cache/shares_big.parquet")["shares"]
cols = [c for c in px.columns if c in sh.index][:400]
px = px[cols]; cap = px * sh.reindex(cols)
ret = px.pct_change().iloc[1:]
N = len(cols)

def slot_resid_ar1(score: pd.DataFrame):
    """Sort each day by `score`, build slot-return panel from the sorted CAPS,
    remove PC1, return pooled residual AR(1) and the daily churn rate."""
    order = np.argsort(-score.to_numpy(), axis=1)
    capv = cap.to_numpy()
    slot = np.take_along_axis(capv, order, axis=1)
    r = slot[1:] / slot[:-1] - 1.0
    r = np.nan_to_num(r)
    Xc = r - r.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    f = U[:, :1] * S[0]
    e = Xc - f @ ((f.T @ Xc) / (f.T @ f))
    a, b = e[1:], e[:-1]
    ar1 = float((a*b).sum() / (b*b).sum())
    churn = float((order[1:] != order[:-1]).mean())
    return ar1, churn

rng = np.random.default_rng(0)
# a persistent random score: random walk per name, no economic content at all
rw = pd.DataFrame(np.cumsum(rng.standard_normal((len(px), N))*0.02, axis=0),
                  index=px.index, columns=cols)
vol = ret.rolling(60).std().reindex(px.index).bfill().ffill()
mom = px.pct_change(126).reindex(px.index).bfill().ffill()

tests = {
    "market cap (theirs)":      cap,
    "share price":              px,
    "60d volatility":           vol,
    "6m momentum":              mom,
    "PERSISTENT RANDOM WALK":   rw,
    "pure noise (max churn)":   pd.DataFrame(rng.standard_normal((len(px), N)),
                                             index=px.index, columns=cols),
}
print(f"{N} names, {len(px):,} days, slots built from sorted CAPS in every case;")
print("only the SORTING VARIABLE differs.\n")
print(f"{'ranking variable':26}{'residual AR(1)':>16}{'churn/day':>12}")
print("-"*54)
for k, v in tests.items():
    ar1, ch = slot_resid_ar1(v)
    print(f"{k:26}{ar1:>16.4f}{ch:>12.1%}")
