"""Null test on the AUTHORS' OWN residuals.

Null: flip the sign of each day's entire cross-sectional residual vector.
Preserves cross-sectional correlation and volatility clustering exactly;
destroys directional time-series predictability. Identical code both arms.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, real_ladder as rl, json

SKIP = 4*252
TAG = "PCA-5"
d = rl.load(TAG)
print(f"{TAG} residuals {d.shape};  trading window = their 2002-2016\n")

def arm(data, label):
    W, sel = rl.windows_and_mask(data)
    out = {}
    for name, fn in (("reversal (0 params)", rl.w_reversal), ("OU+Threshold", rl.w_ou)):
        w = fn(W, sel)
        out[name] = rl.sharpe(rl.port_returns(w[SKIP:], data[SKIP:], sel[SKIP:]))
    return out

real = arm(d, "REAL")
rng = np.random.default_rng(0)
nulls = []
for i in range(5):
    dn = d * rng.choice([-1.0, 1.0], size=(len(d), 1)).astype(np.float32)
    nulls.append(arm(dn, f"NULL{i}"))
    print(f"  null draw {i+1}/5 done", flush=True)

print(f"\n{'model':24}{'REAL':>8}{'NULL mean':>11}{'NULL sd':>9}{'excess':>9}")
print("-"*61)
res={}
for k in real:
    nv = np.array([n[k] for n in nulls])
    print(f"{k:24}{real[k]:>8.2f}{nv.mean():>11.2f}{nv.std():>9.2f}{real[k]-nv.mean():>9.2f}")
    res[k]=dict(real=real[k], null_mean=float(nv.mean()), null_sd=float(nv.std()),
                excess=float(real[k]-nv.mean()), nulls=nv.tolist())
json.dump(res, open("results/real_null_results.json", "w"), indent=2)
print("\npaper Table I, PCA-5 gross: OU+Threshold 0.73")
