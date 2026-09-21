"""Strictly causal residuals, Carhart (1997) style: PCA weights from days
before the block, loadings from the PRIOR 60 days with an intercept, applied
out-of-sample to day t.  No information from day t or later is used."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, dlsa

def residuals_strict(R):
    X = R.to_numpy(); T, N = X.shape
    out = np.full((T, N), np.nan)
    for a in range(dlsa.PCA_WIN, T, dlsa.REFIT):
        b = min(a + dlsa.REFIT, T)
        tr = X[a-dlsa.PCA_WIN:a]
        _, _, Vt = np.linalg.svd(tr - tr.mean(0), full_matrices=False)
        W = Vt[:dlsa.K_PCA].T
        F_all = X @ W                                   # factor returns, all days
        for t in range(a, b):
            lo = t - dlsa.BETA_WIN
            if lo < 0: continue
            Fw = F_all[lo:t]                            # STRICTLY prior 60 days
            Yw = X[lo:t]
            A = np.column_stack([np.ones(len(Fw)), Fw]) # intercept, as Carhart
            coef, *_ = np.linalg.lstsq(A, Yw, rcond=None)
            pred = coef[0] + F_all[t] @ coef[1:]
            out[t] = X[t] - pred
    return out

if __name__ == "__main__":
    R = dlsa.load_returns(n_names=300)
    rng = np.random.default_rng(0)
    print("Linear model (31 params), STRICTLY CAUSAL residuals\n")
    Er = residuals_strict(R)
    sr,_ = dlsa.fit_eval(Er, epochs=30, seed=0, arch=dlsa.LinearSig, dev="cpu")
    print(f"REAL   {sr:+.3f}")
    nl = []
    for d in range(5):
        Rn = dlsa.signflip(R, rng)
        s,_ = dlsa.fit_eval(residuals_strict(Rn), epochs=30, seed=100+d,
                            arch=dlsa.LinearSig, dev="cpu")
        nl.append(s); print(f"null {d+1}/5 {s:+.3f}")
    nl = np.array(nl)
    print(f"\nSTRICT: real {sr:+.3f} | null mean {nl.mean():+.3f} sd {nl.std():.3f}"
          f" | excess {sr-nl.mean():+.3f}")
    print(f"CONTAMINATED (earlier): real +2.241 | null +1.699 | excess +0.542")
