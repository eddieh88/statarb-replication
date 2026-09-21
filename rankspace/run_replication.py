import warnings; warnings.filterwarnings("ignore")
import json, time
import numpy as np, pandas as pd
import replicate as rp

px = pd.read_parquet("cache/prices_big.parquet")
px = px[px.index >= "2006-01-01"].dropna(axis=1, thresh=int(len(px)*0.98)).ffill().dropna()
sh = pd.read_parquet("cache/shares_big.parquet")["shares"]
cols = [c for c in px.columns if c in sh.index]
cap = px[cols] * sh.reindex(cols)
print(f"{len(cols)} names, {len(cap):,} days, {cap.index.min():%Y-%m} to {cap.index.max():%Y-%m}")
print(f"top-{rp.TOP_N} selected every {rp.REBAL}d | PCA {rp.PCA_WIN}d, "
      f"K_name={rp.K_NAME} K_rank={rp.K_RANK} | beta {rp.BETA_WIN}d\n")

t0=time.time()
en = rp.residuals_from_caps(cap, "name")
er = rp.residuals_from_caps(cap, "rank")
an, ar = rp.pooled_ar1(en), rp.pooled_ar1(er)
print(f"=== OBSERVED ({time.time()-t0:.0f}s) ===")
print(f"{'':18}{'name (K=5)':>13}{'rank (K=1)':>13}")
print(f"{'residual AR(1)':18}{an:>13.4f}{ar:>13.4f}")
print(f"{'OU half-life (d)':18}{rp.halflife(an):>13.1f}{rp.halflife(ar):>13.1f}\n")

rng = np.random.default_rng(0)
N_NULL = 15
share = 0.40
nar_r, nar_n = [], []
for i in range(N_NULL):
    sc = rp.simulate_caps(cap, rng, share)
    nar_n.append(rp.pooled_ar1(rp.residuals_from_caps(sc, "name")))
    nar_r.append(rp.pooled_ar1(rp.residuals_from_caps(sc, "rank")))
    if (i+1) % 10 == 0: print(f"  null {i+1}/{N_NULL} ({time.time()-t0:.0f}s)", flush=True)
nar_r, nar_n = np.array(nar_r), np.array(nar_n)

print(f"\n=== NULL (factor share {share}, no signal, same pipeline) ===")
print(f"{'':22}{'median':>10}{'5th pct':>10}{'observed':>11}{'verdict':>10}")
for lab, nv, ov in (("name-space AR(1)", nar_n, an), ("rank-space AR(1)", nar_r, ar)):
    p5 = np.percentile(nv, 5)
    print(f"{lab:22}{np.median(nv):>10.4f}{p5:>10.4f}{ov:>11.4f}"
          f"{('BEATS' if ov < p5 else 'within'):>10}")
print(f"\nrank-space excess over null median: {ar-np.median(nar_r):+.4f}")
print(f"fraction of null rank AR(1) <= observed: {np.mean(nar_r<=ar):.1%}")
json.dump(dict(ar1_name=an, ar1_rank=ar, null_rank_median=float(np.median(nar_r)),
               null_rank_p5=float(np.percentile(nar_r,5)),
               null_name_median=float(np.median(nar_n)),
               frac_null_le_obs=float(np.mean(nar_r<=ar)), n_names=len(cols),
               n_days=len(cap), n_null=N_NULL, factor_share=share),
          open("results/replication_results.json", "w"), indent=2)
print("\nwrote replication_results.json")
