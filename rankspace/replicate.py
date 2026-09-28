"""
Replication at the paper's scale and construction (arXiv 2410.06568, App. A.2):

  * universe: top 500 by capitalisation, recalibrated (here every 21 days)
  * PCA on a 252-day lookback; K=5 factors in NAME space, K=1 in RANK space
  * factor loadings on a 60-day lookback; residual = return - beta @ factor
  * rank return per eq (3.5): simple return on the capitalisation at rank k

The null is identical in construction, driven by simulated capitalisations with
no cross-sectional signal, so the order-statistic (local-time) effect is present
in both and only genuine structure can separate them.
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd

TOP_N, PCA_WIN, BETA_WIN, REBAL = 500, 252, 60, 21
K_NAME, K_RANK = 5, 1


def _roll_beta_resid(R: np.ndarray, F: np.ndarray, win: int, only_from: int) -> np.ndarray:
    """Rolling-window OLS of each column of R on factors F, vectorised.
    beta_t from the trailing `win` observations; residual_t = R_t - F_t @ beta_t.
    Only days >= only_from are computed -- earlier ones are discarded by the
    caller, and computing them dominated the runtime."""
    T, N = R.shape
    K = F.shape[1]
    FF = np.einsum("ti,tj->tij", F, F)
    FR = np.einsum("ti,tn->tin", F, R)
    cFF = np.cumsum(FF, 0); cFR = np.cumsum(FR, 0)
    resid = np.full((T, N), np.nan)
    for t in range(max(win, only_from), T):
        A = cFF[t] - cFF[t - win]
        B = cFR[t] - cFR[t - win]
        try:
            beta = np.linalg.solve(A + np.eye(K) * 1e-12, B)
        except np.linalg.LinAlgError:
            continue
        resid[t] = R[t] - F[t] @ beta
    return resid


def residuals_from_caps(cap: pd.DataFrame, space: str) -> np.ndarray:
    """space in {'name','rank'}.  Returns a T x N residual matrix (NaN-padded)."""
    K = K_NAME if space == "name" else K_RANK
    capv = cap.to_numpy()
    T = len(cap)
    out = []
    for start in range(PCA_WIN, T, REBAL):
        stop = min(start + REBAL, T)
        lo = start - PCA_WIN
        # universe = top TOP_N by cap at the rebalance date
        sel = np.argsort(-capv[start - 1])[:TOP_N]
        blk = capv[lo:stop][:, sel]
        if space == "rank":
            blk = -np.sort(-capv[lo:stop], axis=1)[:, :TOP_N]   # order statistics
        r = blk[1:] / blk[:-1] - 1.0                            # eq (3.5)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        train = r[:PCA_WIN - 1]
        Xc = train - train.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        W = Vt[:K].T                                            # N,K loadings
        F = (r - r.mean(0)) @ W                                 # T,K factor returns
        live = PCA_WIN - 2
        res = _roll_beta_resid(r - r.mean(0), F, BETA_WIN, live)
        out.append(res[live:])                       # keep the live block
    return np.vstack([o for o in out if len(o)])


def pooled_ar1(E: np.ndarray) -> float:
    E = np.where(np.isfinite(E), E, np.nan)
    a, b = E[1:], E[:-1]
    m = np.isfinite(a) & np.isfinite(b)
    return float((a[m] * b[m]).sum() / (b[m] * b[m]).sum())


def halflife(ar1: float) -> float:
    return float(-np.log(2) / np.log(1 + ar1)) if -1 < ar1 < 0 else np.inf


def simulate_caps(cap: pd.DataFrame, rng, factor_share: float) -> pd.DataFrame:
    """Simulated caps: common factor + idiosyncratic, matched per-name vol and
    starting caps, NO mean reversion and NO cross-sectional predictability."""
    r = (cap / cap.shift(1) - 1).iloc[1:]
    sig = r.std().to_numpy()
    T, N = len(cap) - 1, cap.shape[1]
    f = rng.standard_normal((T, 1))
    e = rng.standard_normal((T, N))
    sh = (np.sqrt(factor_share) * f + np.sqrt(1 - factor_share) * e) * sig
    logc = np.log(cap.iloc[0].to_numpy()) + np.cumsum(sh, axis=0)
    return pd.DataFrame(np.vstack([cap.iloc[0].to_numpy(), np.exp(logc)]),
                        index=cap.index, columns=cap.columns)
