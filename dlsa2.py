"""
Corrected DLSA implementation.

Fixes relative to the first attempt:
  * strictly causal betas -- estimated on days [t-60, t), EXCLUDING t
    (Carhart 1997, which the paper follows). The first version included t.
  * portfolio weights mapped to stock space via Phi' and normalised THERE,
    as in their eqs (3) and (5). Gross returns are unchanged by the mapping,
    but the normalisation is not: ||w_stock||_1 / ||w_eps||_1 moves daily, so
    normalising in residual space is an accidental time-varying leverage bet.
  * transaction costs measured on STOCK weight turnover, which is non-zero
    even when residual weights are static, because Phi changes daily.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn

L, PCA_WIN, BETA_WIN, REFIT = 30, 252, 60, 250
ETA_TURN, ETA_SHORT = 0.0005, 0.0001
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_returns(n_names=300, start="2006-01-01"):
    px = pd.read_parquet("cache/prices_big.parquet")
    px = px[px.index >= start]
    px = px.dropna(axis=1, thresh=int(len(px)*0.98)).ffill().dropna()
    sh = pd.read_parquet("cache/shares_big.parquet")["shares"]
    cols = [c for c in px.columns if c in sh.index]
    cap = px[cols] * sh.reindex(cols)
    keep = cap.median().nlargest(n_names).index
    return px[keep].pct_change().iloc[1:]


def signflip(R, rng):
    return pd.DataFrame(R.to_numpy()*rng.choice([-1.,1.], size=(len(R),1)),
                        index=R.index, columns=R.columns)


def build(R: pd.DataFrame, K=5):
    """Strictly causal PCA factor model.
    Returns E (T,N) residuals, Wt (T,N,K) factor weights, Bt (T,N,K) loadings."""
    X = R.to_numpy(); T, N = X.shape
    E = np.full((T, N), np.nan)
    Wt = np.zeros((T, N, K), dtype=np.float32)
    Bt = np.zeros((T, N, K), dtype=np.float32)
    for a in range(PCA_WIN, T, REFIT):
        b = min(a + REFIT, T)
        tr = X[a-PCA_WIN:a]
        _, _, Vt = np.linalg.svd(tr - tr.mean(0), full_matrices=False)
        W = Vt[:K].T                                   # N x K, F = W' R
        F_all = X @ W
        for t in range(a, b):
            lo = t - BETA_WIN
            if lo < 0: continue
            A = np.column_stack([np.ones(BETA_WIN), F_all[lo:t]])   # prior 60d
            coef, *_ = np.linalg.lstsq(A, X[lo:t], rcond=None)
            beta = coef[1:].T                           # N x K
            E[t] = X[t] - (coef[0] + F_all[t] @ coef[1:])
            Wt[t] = W; Bt[t] = beta
    return E, Wt, Bt


def to_stock(w_eps, W, B):
    """w_stock = Phi' w_eps = w_eps - W (B' w_eps).  Normalised L1 = 1."""
    ws = w_eps - W @ (B.T @ w_eps)
    s = np.abs(ws).sum()
    return ws / s if s > 0 else ws


def backtest(w_eps_mat, R, E, Wt, Bt, norm="stock", costs=True):
    """w_eps_mat: (T,N) residual-space weights. Returns dict of stats."""
    X = R.to_numpy(); T, N = X.shape
    prev = np.zeros(N); rets, costs_ = [], []
    for t in range(T-1):
        w = np.nan_to_num(w_eps_mat[t])
        if not np.any(w): 
            prev = np.zeros(N); continue
        if norm == "stock":                      # their eq (5): normalise in
            ws = to_stock(w, Wt[t], Bt[t])       # STOCK space, trade stocks
            r = float(ws @ np.nan_to_num(X[t+1]))
        else:                                    # normalise in RESIDUAL space
            ws = w/np.abs(w).sum()               # and trade residuals
            r = float(ws @ np.nan_to_num(E[t+1]))
        c = ETA_TURN*np.abs(ws-prev).sum() + ETA_SHORT*np.abs(np.minimum(ws,0)).sum()
        rets.append(r); costs_.append(c); prev = ws
    r = np.array(rets); c = np.array(costs_)
    g, n = r, r - c
    f = lambda v: (float(v.mean()/v.std()*np.sqrt(252)), float(v.mean()*252))
    sg, mg = f(g); sn, mn = f(n)
    return dict(sr_gross=sg, mu_gross=mg, sr_net=sn, mu_net=mn,
                vol=float(g.std()*np.sqrt(252)), cost_ann=float(c.mean()*252),
                turnover=float(np.mean([np.abs(x).sum() for x in [c]])*0+
                               np.mean(c)/ETA_TURN))
