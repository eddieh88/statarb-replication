"""
Phase 0: does rank-space indexing improve residual mean reversion, beyond the
order-statistic effect?

    python ranktest.py            (add --refresh to re-pull prices)

Thresholds and statistics are fixed in PREREGISTRATION.md, committed before any
data was fetched.  STOP if the observed advantage does not clear the null.
"""
from __future__ import annotations
import json, sys
import numpy as np, pandas as pd
import rankdata as rd

N_NULL, SEED, MIN_OBS = 200, 0, 250


def residuals(R: pd.DataFrame) -> pd.DataFrame:
    """Remove PC1, estimated on the full sample (same treatment both spaces, and
    identically applied to the null -- so any look-ahead cancels in the
    comparison, which is the only quantity under test)."""
    X = R.to_numpy()
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1_var = float(S[0] ** 2 / (S ** 2).sum())
    f = U[:, 0:1] * S[0]                      # first PC scores
    beta = (f.T @ Xc) / (f.T @ f)             # loading per column
    resid = Xc - f @ beta
    return pd.DataFrame(resid, index=R.index, columns=R.columns), pc1_var


def pooled_ar1(E: pd.DataFrame) -> float:
    """Pooled AR(1) across columns: sum(e_t * e_{t-1}) / sum(e_{t-1}^2)."""
    X = E.to_numpy()
    a, b = X[1:], X[:-1]
    return float((a * b).sum() / (b * b).sum())


def halflife(ar1: float) -> float:
    return float(-np.log(2) / np.log(1 + ar1)) if -1 < ar1 < 0 else np.inf


def measure(name_ret, rank_ret):
    en, vn = residuals(name_ret)
    er, vr = residuals(rank_ret)
    an, ar = pooled_ar1(en), pooled_ar1(er)
    return dict(pc1_name=vn, pc1_rank=vr, ar1_name=an, ar1_rank=ar,
                hl_name=halflife(an), hl_rank=halflife(ar), advantage=an - ar)


def main() -> int:
    refresh = "--refresh" in sys.argv
    px = rd.load_prices(refresh=refresh)
    sh = rd.load_shares(list(px.columns), refresh=refresh)
    name_ret, rank_ret, cap = rd.build_panels(px, sh)
    print(f"{name_ret.shape[1]} names, {len(name_ret):,} daily obs, "
          f"{name_ret.index.min():%Y-%m} to {name_ret.index.max():%Y-%m}")
    print("SURVIVORSHIP-BIASED by construction (see prereg): biased toward H1.\n")

    obs = measure(name_ret, rank_ret)
    print("=== observed ===")
    print(f"{'':16}{'name space':>14}{'rank space':>14}")
    print(f"{'PC1 var explained':16}{obs['pc1_name']:>13.1%}{obs['pc1_rank']:>14.1%}")
    print(f"{'residual AR(1)':16}{obs['ar1_name']:>14.4f}{obs['ar1_rank']:>14.4f}")
    print(f"{'OU half-life (d)':16}{obs['hl_name']:>14.1f}{obs['hl_rank']:>14.1f}")
    print(f"\nADVANTAGE = ar1(name) - ar1(rank) = {obs['advantage']:+.4f}\n")

    rng = np.random.default_rng(SEED)
    null = []
    for i in range(N_NULL):
        sn, sr = rd.simulate_null(cap, name_ret, rng)
        null.append(measure(sn, sr))
        if (i + 1) % 50 == 0:
            print(f"  null {i+1}/{N_NULL} ...")
    nadv = np.array([x["advantage"] for x in null])
    nar_rank = np.array([x["ar1_rank"] for x in null])
    p95 = float(np.percentile(nadv, 95))

    print("\n=== null: independent GBMs, matched vol, NO cross-sectional signal ===")
    print(f"null rank-space AR(1): median {np.median(nar_rank):+.4f}   "
          f"(observed {obs['ar1_rank']:+.4f})")
    print(f"null ADVANTAGE: median {np.median(nadv):+.4f}, "
          f"95th pct {p95:+.4f}, max {nadv.max():+.4f}")
    print(f"observed ADVANTAGE {obs['advantage']:+.4f}")
    frac = float((nadv >= obs["advantage"]).mean())
    print(f"fraction of null draws >= observed: {frac:.1%}")

    ok = bool(obs["advantage"] > p95)
    print("\n" + "=" * 64)
    print(f"PHASE 0: {'PASS -- advantage exceeds the null' if ok else 'FAIL -- within the order-statistic null'}")
    print("=" * 64)
    if not ok:
        print("The rank-space mean reversion on this panel is what independent")
        print("random walks produce under the same construction. Order statistics,")
        print("not an inefficiency. Per the pre-registration: STOP.")
    else:
        print("Positive, but UNINTERPRETABLE on a survivorship-biased panel")
        print("(prereg, 'Known bias'). Re-run on CRSP/Sharadar/Norgate before")
        print("this means anything.")

    out = dict(observed=obs, null_p95=p95, null_median=float(np.median(nadv)),
               frac_null_ge_obs=frac, n_names=int(name_ret.shape[1]),
               n_obs=int(len(name_ret)), phase0_pass=ok)
    with open("results/phase0_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nwrote results/phase0_results.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
