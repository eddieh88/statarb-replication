"""
Phase 1: the comparison the DLSA paper never makes.

Their Table IX reports trading frictions for the CNN+Transformer ONLY -- there
is no OU-with-costs row anywhere in the paper.  So whether the deep model still
beats the cheap parametric benchmark AFTER costs is unknown.

This applies their exact cost model (Section III.J)

    cost(w_t, w_{t-1}) = 0.0005 * ||w_t - w_{t-1}||_1        (5bp turnover)
                       + 0.0001 * ||min(w_t, 0)||_1          (1bp short holding)

identically to any weight matrix, so OU and CNN can be compared on equal terms.
Weights are applied to residuals directly (as in Phase 0), not mapped back
through Phi' -- a deviation from the paper, applied identically to both arms.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ETA_TURNOVER, ETA_SHORT = 0.0005, 0.0001


def normalise(W: np.ndarray) -> np.ndarray:
    """Scale each day so absolute weights sum to one (their leverage constraint)."""
    s = np.nansum(np.abs(W), axis=1, keepdims=True)
    return np.nan_to_num(W) / np.where(s > 0, s, 1.0)


def evaluate(W: np.ndarray, E: np.ndarray, label: str = "") -> dict:
    """W, E are (T, N). Returns gross/net Sharpe, mean, vol, turnover."""
    W = normalise(W)
    r_gross = np.nansum(W[:-1] * np.nan_to_num(E[1:]), axis=1)
    turn = np.abs(W[1:] - W[:-1]).sum(1)
    shorts = np.abs(np.minimum(W[:-1], 0)).sum(1)
    cost = ETA_TURNOVER * turn + ETA_SHORT * shorts
    r_net = r_gross - cost
    ok = np.isfinite(r_gross)
    def stats(r):
        r = r[ok]
        return (float(r.mean()/r.std()*np.sqrt(252)), float(r.mean()*252),
                float(r.std()*np.sqrt(252)))
    sg, mg, vg = stats(r_gross); sn, mn, vn = stats(r_net)
    return dict(label=label, sr_gross=sg, mu_gross=mg, sr_net=sn, mu_net=mn,
                vol=vg, turnover=float(turn[ok].mean()),
                cost_ann=float(cost[ok].mean()*252))


def ou_weights(E: np.ndarray, L=60, win=252, c_open=1.25, c_close=0.5,
               tau_max=30) -> np.ndarray:
    """Avellaneda-Lee / Yeo-Papanicolaou s-score rule on cumulative residuals,
    with the tau<30d filter, as in their Appendix B.B."""
    X = pd.DataFrame(np.nan_to_num(E)).rolling(L).sum()
    s = ((X - X.rolling(win).mean()) / X.rolling(win).std()).to_numpy()
    Xv = X.to_numpy(); a, b = Xv[1:], Xv[:-1]
    num = pd.DataFrame(a*b).rolling(win).sum().to_numpy()
    den = pd.DataFrame(b*b).rolling(win).sum().to_numpy()
    with np.errstate(all="ignore"):
        tau = -1.0/np.log(np.clip(num/den, 1e-6, 1-1e-9))
    tau = np.vstack([np.full((1, E.shape[1]), np.nan), tau])
    T, N = s.shape
    W = np.zeros((T, N)); prev = np.zeros(N)
    ok = np.isfinite(s) & np.isfinite(tau) & (tau < tau_max)
    for t in range(1, T):
        cur = prev.copy(); st, okt = s[t], ok[t]
        cur[(prev == 0) & okt & (st >  c_open)] = -1
        cur[(prev == 0) & okt & (st < -c_open)] =  1
        cur[(prev != 0) & (np.abs(st) < c_close)] = 0
        cur[~np.isfinite(st)] = 0
        W[t] = cur; prev = cur
    return W


if __name__ == "__main__":
    import dlsa
    R = dlsa.load_returns(n_names=300)
    E = dlsa.residuals(R)
    print(f"PCA-{dlsa.K_PCA} residuals: {E.shape[0]:,} days x {E.shape[1]} names\n")
    res = evaluate(ou_weights(E), E, "OU+Threshold")
    print(f"{'model':16}{'SR gross':>10}{'SR net':>9}{'mu gross':>10}{'mu net':>9}"
          f"{'turnover':>10}{'cost/yr':>9}")
    print("-"*73)
    r = res
    print(f"{r['label']:16}{r['sr_gross']:>10.2f}{r['sr_net']:>9.2f}"
          f"{r['mu_gross']:>9.1%}{r['mu_net']:>9.1%}"
          f"{r['turnover']:>10.2f}{r['cost_ann']:>9.1%}")
    print("\nPaper Table I (PCA-5, gross): OU+Thresh SR 0.73 | CNN+Trans SR 3.36")
    print("Paper Table IX: CNN+Trans with costs SR 0.94-1.24 (IPCA). No OU row exists.")
