"""
Full model ladder on the corrected pipeline.

Neural models are trained on the STOCK-SPACE Sharpe objective (their eq 4):
weights come out in residual space, are mapped through Phi', normalised to
L1 = 1 in stock space, and scored against stock returns.  The mapping is inside
the differentiable graph, so the model optimises what is actually traded.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn
import dlsa2 as d

L, TRAIN, TEST, BATCH = d.L, 1000, 250, 100
DEV = "cpu"


def panel(R, E, Wt, Bt):
    """Aligned arrays: cumulative-residual windows, factor matrices, next return."""
    X = R.to_numpy(); T, N = E.shape
    C = np.nan_to_num(E).cumsum(0)
    win = np.zeros((T, N, L), np.float32)
    for l in range(L):
        win[:, :, l] = np.roll(C, -(l-L+1), axis=0) - np.roll(C, -(-L), axis=0)
    # simpler and correct: window l covers days t-L+1+l
    for t in range(L, T):
        win[t] = (C[t-L+1:t+1] - C[t-L]).T
    ok = np.isfinite(E).all(1)
    valid = np.zeros(T, bool); valid[L:T-1] = ok[L:T-1] & ok[L+1:T]
    return win, np.nan_to_num(X), valid


def sharpe_loss(w_eps, W, B, Rn):
    """w_eps (D,N); W,B (D,N,K); Rn (D,N). Stock-space normalised Sharpe."""
    proj = torch.einsum("dnk,dk->dn", W, torch.einsum("dnk,dn->dk", B, w_eps))
    ws = w_eps - proj
    ws = ws / ws.abs().sum(1, keepdim=True).clamp(min=1e-9)
    r = (ws * Rn).sum(1)
    return r.mean()/r.std().clamp(min=1e-9)*np.sqrt(252), ws


class Linear(nn.Module):
    def __init__(s): super().__init__(); s.f = nn.Linear(L, 1)
    def forward(s, x): return torch.tanh(s.f(x).squeeze(-1))

class FFN(nn.Module):
    def __init__(s, din=L):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(din,16), nn.ReLU(), nn.Dropout(0.25),
                            nn.Linear(16,8), nn.ReLU(), nn.Linear(8,4),
                            nn.ReLU(), nn.Linear(4,1))
    def forward(s, x): return torch.tanh(s.f(x).squeeze(-1))

class FourierFFN(FFN):
    def __init__(s): super().__init__(din=L)
    def forward(s, x):
        f = torch.fft.rfft(x, dim=-1)
        z = torch.cat([f.real, f.imag[:, 1:-1]], -1)[:, :L]
        return torch.tanh(s.f(z).squeeze(-1))

class CNNTrans(nn.Module):
    def __init__(s, D=8, H=4):
        super().__init__()
        s.c1, s.c2 = nn.Conv1d(1,D,2,padding=1), nn.Conv1d(D,D,2,padding=1)
        s.n1, s.n2 = nn.InstanceNorm1d(D), nn.InstanceNorm1d(D)
        s.att = nn.MultiheadAttention(D,H,dropout=0.25,batch_first=True)
        s.ff = nn.Sequential(nn.Linear(D,2*D), nn.ReLU(), nn.Linear(2*D,D))
        s.out = nn.Sequential(nn.Linear(D,16), nn.ReLU(), nn.Linear(16,8),
                              nn.ReLU(), nn.Linear(8,4), nn.ReLU(), nn.Linear(4,1))
    def forward(s, x):
        z = x.unsqueeze(1)
        z = torch.relu(s.n1(s.c1(z)[:,:,:L])); z = torch.relu(s.n2(s.c2(z)[:,:,:L]))
        z = z.transpose(1,2) + x.unsqueeze(-1)
        a,_ = s.att(z,z,z); z = z + a; z = z + s.ff(z)
        return torch.tanh(s.out(z[:,-1,:]).squeeze(-1))


def run_nn(arch, win, Xn, Wt, Bt, valid, epochs=30, seed=0):
    torch.manual_seed(seed)
    T = len(valid); days = np.where(valid)[0]
    W_out = np.zeros((T, win.shape[1]))
    for s0 in range(TRAIN, len(days)-TEST, TEST):
        tr = days[s0-TRAIN:s0]; te = days[s0:s0+TEST]
        m = arch().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        chunks = [tr[i:i+BATCH] for i in range(0, len(tr), BATCH)]
        tens = [(torch.tensor(win[c]), torch.tensor(Wt[c]), torch.tensor(Bt[c]),
                 torch.tensor(Xn[c+1], dtype=torch.float32)) for c in chunks]
        for _ in range(epochs):
            for xb, Wb, Bb, Rb in tens:
                opt.zero_grad()
                D_, N_, _ = xb.shape
                w = m(xb.reshape(-1, L)).reshape(D_, N_)
                sr, _ = sharpe_loss(w, Wb, Bb, Rb)
                (-sr).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            for i in range(0, len(te), BATCH):
                c = te[i:i+BATCH]
                xb = torch.tensor(win[c]); D_, N_, _ = xb.shape
                W_out[c] = m(xb.reshape(-1, L)).reshape(D_, N_).numpy()
    return W_out


def reversal(E):
    T,N = E.shape; C = np.nan_to_num(E).cumsum(0); W = np.zeros((T,N))
    for t in range(L,T): W[t] = -(C[t]-C[t-L])
    return W


def ou_threshold(E, win_=252, c_open=1.25, c_close=0.5, tau_max=30):
    X = pd.DataFrame(np.nan_to_num(E)).rolling(60).sum()
    s = ((X - X.rolling(win_).mean())/X.rolling(win_).std()).to_numpy()
    Xv = X.to_numpy(); a,b = Xv[1:], Xv[:-1]
    num = pd.DataFrame(a*b).rolling(win_).sum().to_numpy()
    den = pd.DataFrame(b*b).rolling(win_).sum().to_numpy()
    with np.errstate(all="ignore"):
        tau = -1.0/np.log(np.clip(num/den,1e-6,1-1e-9))
    tau = np.vstack([np.full((1,E.shape[1]),np.nan), tau])
    T,N = s.shape; W = np.zeros((T,N)); prev = np.zeros(N)
    ok = np.isfinite(s) & np.isfinite(tau) & (tau < tau_max)
    for t in range(1,T):
        cur = prev.copy(); st, okt = s[t], ok[t]
        cur[(prev==0)&okt&(st> c_open)] = -1
        cur[(prev==0)&okt&(st<-c_open)] =  1
        cur[(prev!=0)&(np.abs(st)<c_close)] = 0
        cur[~np.isfinite(st)] = 0
        W[t] = cur; prev = cur
    return W
