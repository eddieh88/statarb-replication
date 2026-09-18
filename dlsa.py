"""
DLSA pipeline (arXiv:2106.04028) -- residual construction, CNN+Transformer,
rolling train/eval.  Used to test the pipeline against a no-signal null; see
PREREG_DLSA.md.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn

L, D, DSIZE, H = 30, 8, 2, 4          # their Table A.II
K_PCA, PCA_WIN, BETA_WIN, REFIT = 5, 252, 60, 250
TRAIN, TEST = 1000, 250
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_returns(n_names=400, start="2006-01-01"):
    px = pd.read_parquet("cache/prices_big.parquet")
    px = px[px.index >= start]
    px = px.dropna(axis=1, thresh=int(len(px)*0.98)).ffill().dropna()
    sh = pd.read_parquet("cache/shares_big.parquet")["shares"]
    cols = [c for c in px.columns if c in sh.index]
    cap = px[cols] * sh.reindex(cols)
    keep = cap.median().nlargest(n_names).index          # fixed universe
    return px[keep].pct_change().iloc[1:]


def signflip(R: pd.DataFrame, rng) -> pd.DataFrame:
    """Flip each DAY's whole cross-sectional vector. Preserves cross-sectional
    correlation and volatility clustering exactly; destroys direction only."""
    s = rng.choice([-1.0, 1.0], size=(len(R), 1))
    return pd.DataFrame(R.to_numpy()*s, index=R.index, columns=R.columns)


def residuals(R: pd.DataFrame) -> np.ndarray:
    """Rolling PCA(K=5) factors, rolling 60d betas, daily residuals."""
    X = R.to_numpy(); T, N = X.shape
    out = np.full((T, N), np.nan)
    for a in range(PCA_WIN, T, REFIT):
        b = min(a + REFIT, T)
        tr = X[a-PCA_WIN:a]
        Z = tr - tr.mean(0)
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        W = Vt[:K_PCA].T
        lo = max(a - BETA_WIN, 0)
        seg = X[lo:b]
        F = (seg - seg.mean(0)) @ W
        FF = np.einsum("ti,tj->tij", F, F).cumsum(0)
        FR = np.einsum("ti,tn->tin", F, seg - seg.mean(0)).cumsum(0)
        for t in range(a - lo, b - lo):
            if t < BETA_WIN: continue
            A = FF[t] - FF[t-BETA_WIN]; B = FR[t] - FR[t-BETA_WIN]
            beta = np.linalg.solve(A + np.eye(K_PCA)*1e-10, B)
            out[lo+t] = (seg[t] - seg.mean(0)) - F[t] @ beta
    return out


def windows(E: np.ndarray):
    """(samples, L) cumulative-residual inputs and the next-day residual."""
    T, N = E.shape
    cum = np.nan_to_num(E).cumsum(0)
    xs, ys, ts = [], [], []
    for t in range(L, T-1):
        w = cum[t-L+1:t+1] - cum[t-L]          # L x N cumulative over window
        nxt = E[t+1]
        m = np.isfinite(w).all(0) & np.isfinite(nxt)
        if m.sum() < 20: continue
        xs.append(w[:, m].T); ys.append(nxt[m]); ts.append(np.full(m.sum(), t))
    return (np.concatenate(xs).astype(np.float32),
            np.concatenate(ys).astype(np.float32),
            np.concatenate(ts))


class CNNTrans(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(1, D, DSIZE, padding=DSIZE-1)
        self.c2 = nn.Conv1d(D, D, DSIZE, padding=DSIZE-1)
        self.n1, self.n2 = nn.InstanceNorm1d(D), nn.InstanceNorm1d(D)
        self.att = nn.MultiheadAttention(D, H, dropout=0.25, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D, 2*D), nn.ReLU(), nn.Linear(2*D, D))
        self.out = nn.Sequential(nn.Linear(D,16), nn.ReLU(), nn.Linear(16,8),
                                 nn.ReLU(), nn.Linear(8,4), nn.ReLU(), nn.Linear(4,1))
    def forward(self, x):
        z = x.unsqueeze(1)
        z = torch.relu(self.n1(self.c1(z)[:, :, :L]))
        z = torch.relu(self.n2(self.c2(z)[:, :, :L]))
        z = z.transpose(1,2) + x.unsqueeze(-1)
        a,_ = self.att(z,z,z); z = z + a; z = z + self.ff(z)
        return torch.tanh(self.out(z[:, -1, :]).squeeze(-1))


def port_sharpe(w, y, t_idx, n_t):
    """Daily portfolio return with |w| summing to 1 each day, then Sharpe."""
    num = torch.zeros(n_t, device=w.device).index_add_(0, t_idx, w*y)
    den = torch.zeros(n_t, device=w.device).index_add_(0, t_idx, w.abs())
    r = num / den.clamp(min=1e-8)
    r = r[den > 0]
    return r.mean()/r.std().clamp(min=1e-8)*np.sqrt(252)


BATCH_DAYS = 100          # temporal batching, as in their Appendix B.D


def _chunks(t_arr, size):
    """Contiguous groups of `size` distinct day-indices."""
    days = np.unique(t_arr)
    for i in range(0, len(days), size):
        blk = days[i:i+size]
        yield np.isin(t_arr, blk)


def fit_eval(E: np.ndarray, epochs=30, seed=0, log=None):
    torch.manual_seed(seed)
    x, y, t = windows(E)
    oos, starts = [], list(range(TRAIN, int(t.max())-TEST, TEST))
    for si, s in enumerate(starts):
        trm = (t >= s-TRAIN) & (t < s)
        tem = (t >= s) & (t < s+TEST)
        if trm.sum() < 1000 or tem.sum() < 100: continue
        m = CNNTrans().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        xtr, ytr, ttr = x[trm], y[trm], t[trm]
        batches = [(torch.tensor(xtr[b], device=DEV), torch.tensor(ytr[b], device=DEV),
                    torch.tensor(pd.factorize(ttr[b])[0], device=DEV))
                   for b in _chunks(ttr, BATCH_DAYS)]
        for _ in range(epochs):
            for xb, yb, tb in batches:
                opt.zero_grad()
                (-port_sharpe(m(xb), yb, tb, int(tb.max())+1)).backward()
                opt.step()
        m.eval()
        xte, yte, tte = x[tem], y[tem], t[tem]
        with torch.no_grad():
            for b in _chunks(tte, BATCH_DAYS):
                xb = torch.tensor(xte[b], device=DEV)
                yb = torch.tensor(yte[b], device=DEV)
                tb = torch.tensor(pd.factorize(tte[b])[0], device=DEV)
                nt = int(tb.max())+1
                w = m(xb)
                num = torch.zeros(nt, device=DEV).index_add_(0, tb, w*yb)
                den = torch.zeros(nt, device=DEV).index_add_(0, tb, w.abs())
                r = (num/den.clamp(min=1e-8)).cpu().numpy()
                oos.append(r[np.isfinite(r)])
        del batches
        if DEV == "mps": torch.mps.empty_cache()
        if log: log(f"    block {si+1}/{len(starts)}")
    r = np.concatenate(oos)
    return float(r.mean()/r.std()*np.sqrt(252)), r
