"""
Do alternative rankers beat their OWN matched-churn null?

For each ranking scheme: build the slot panel from sorted caps, strip PC1, and
measure pooled residual AR(1).  Then repeat the ENTIRE construction on simulated
capitalisations with no signal (common factor + idiosyncratic, matched vols).
The quantity of interest is the EXCESS of observed over null -- an artifact
cancels, genuine structure does not.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

px = pd.read_parquet("cache/prices_big.parquet"); px = px[px.index >= "2010-01-01"]
px = px.dropna(axis=1, thresh=int(len(px)*0.98)).ffill().dropna()
sh = pd.read_parquet("cache/shares_big.parquet")["shares"]
sec = pd.read_parquet("cache/sectors.parquet")["sector"]
cols = [c for c in px.columns if c in sh.index and c in sec.index]
px = px[cols]; cap = px * sh.reindex(cols); sect = sec.reindex(cols)
ret = px.pct_change().iloc[1:]
SEC = sorted(sect.unique()); idx = {s: np.where(sect.values == s)[0] for s in SEC}
MINSEC = min(len(v) for v in idx.values())
print(f"{len(cols)} names, {len(px):,} days, {len(SEC)} sectors "
      f"(smallest has {MINSEC} names)\n")

def resid_ar1(slot: np.ndarray) -> float:
    r = np.nan_to_num(slot[1:] / slot[:-1] - 1.0)
    Xc = r - r.mean(0)
    U_, S_, _ = np.linalg.svd(Xc, full_matrices=False)
    f = U_[:, :1] * S_[0]
    e = Xc - f @ ((f.T @ Xc) / (f.T @ f))
    a, b = e[1:], e[:-1]
    return float((a*b).sum() / (b*b).sum())

def slots_global(C, K=400):
    return -np.sort(-C, axis=1)[:, :K]

def slots_by_sector(C, per=None):
    per = per or MINSEC
    out = []
    for s in SEC:
        sub = C[:, idx[s]]
        out.append(-np.sort(-sub, axis=1)[:, :per])
    return np.hstack(out)

def slots_sector_pct(C, G=20):
    """Within each sector, bucket into G equal percentile groups; slot = mean cap
    of the bucket.  Coarser than integer ranks, so far less churn."""
    out = []
    for s in SEC:
        sub = -np.sort(-C[:, idx[s]], axis=1)
        edges = np.linspace(0, sub.shape[1], G+1).astype(int)
        out.append(np.stack([sub[:, edges[g]:edges[g+1]].mean(1) for g in range(G)], 1))
    return np.hstack(out)

RANKERS = {
    "global cap rank (theirs)": slots_global,
    "within-sector cap rank":   slots_by_sector,
    "sector x 20 cap buckets":  slots_sector_pct,
}

def sim_caps(rng, share=0.40):
    r = (cap / cap.shift(1) - 1).iloc[1:]
    sig = r.std().to_numpy(); T, N = len(cap)-1, cap.shape[1]
    f = rng.standard_normal((T,1)); e = rng.standard_normal((T,N))
    shocks = (np.sqrt(share)*f + np.sqrt(1-share)*e) * sig
    lc = np.log(cap.iloc[0].to_numpy()) + np.cumsum(shocks, axis=0)
    return np.vstack([cap.iloc[0].to_numpy(), np.exp(lc)])

capv = cap.to_numpy()
rng = np.random.default_rng(0)
NULLS = 10
print(f"{'ranker':28}{'slots':>7}{'observed':>11}{'null mean':>11}{'null sd':>9}"
      f"{'excess':>9}{'z':>7}")
print("-"*82)
for name, fn in RANKERS.items():
    obs = resid_ar1(fn(capv))
    nl = [resid_ar1(fn(sim_caps(rng))) for _ in range(NULLS)]
    nl = np.array(nl); ex = obs - nl.mean()
    z = ex / nl.std() if nl.std() > 0 else np.nan
    print(f"{name:28}{fn(capv).shape[1]:>7}{obs:>11.4f}{nl.mean():>11.4f}"
          f"{nl.std():>9.4f}{ex:>9.4f}{z:>7.2f}")
print("\n(negative excess = observed reverts LESS than its own no-signal null)")
