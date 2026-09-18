"""Is the null Sharpe caused by look-ahead in MY residual construction,
or by the method itself?  Compare three constructions on sign-flipped noise."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, dlsa

def residuals_causal(R, strict_beta=True, strict_demean=True):
    X = R.to_numpy(); T, N = X.shape
    out = np.full((T, N), np.nan)
    for a in range(dlsa.PCA_WIN, T, dlsa.REFIT):
        b = min(a + dlsa.REFIT, T)
        tr = X[a-dlsa.PCA_WIN:a]
        Z = tr - tr.mean(0)
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        W = Vt[:dlsa.K_PCA].T
        mu_tr = tr.mean(0)                       # trailing mean only
        for t in range(a, b):
            lo = t - dlsa.BETA_WIN
            if lo < 0: continue
            end = t if strict_beta else t+1      # exclude or include day t
            seg = X[lo:end]
            m = mu_tr if strict_demean else X[a-dlsa.PCA_WIN:b].mean(0)
            F = (seg - m) @ W
            A = F.T @ F + np.eye(dlsa.K_PCA)*1e-10
            B = F.T @ (seg - m)
            beta = np.linalg.solve(A, B)
            ft = (X[t] - m) @ W
            out[t] = (X[t] - m) - ft @ beta
    return out

R = dlsa.load_returns(n_names=300)
rng = np.random.default_rng(0)
Rn = dlsa.signflip(R, rng)                        # SAME first null draw
print("Linear model (31 params) on the SAME sign-flipped noise,")
print("under three residual constructions:\n")
print(f"{'construction':44}{'null Sharpe':>13}")
print("-"*58)
cfgs = [("as-run (beta includes t, block demean)", lambda: dlsa.residuals(Rn)),
        ("strict beta (t excluded), block demean",
         lambda: residuals_causal(Rn, True, False)),
        ("strict beta AND trailing-only demean",
         lambda: residuals_causal(Rn, True, True))]
for lbl, fn in cfgs:
    E = fn()
    s,_ = dlsa.fit_eval(E, epochs=30, seed=100, arch=dlsa.LinearSig, dev="cpu")
    print(f"{lbl:44}{s:>13.3f}")
print("\nIf the artifact survives the strictest construction, it is the method,")
print("not my implementation.")
