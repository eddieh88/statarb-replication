"""
Simple, non-neural variants of residual reversal, on the authors' PCA-5
residuals. Reported gross, net of 5bp turnover cost, and with turnover, since
cost is what determines whether any of this is tradeable.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, real_ladder as rl

ETA = 0.0005
d = rl.load("PCA-5"); dates = pd.to_datetime(np.load("dlsa_real/dates.npy"))
T, N = d.shape
cs = np.cumsum(d, axis=0)
alive = (d != 0)

def cum(L, skip=0):
    """-(cumulative residual over L days ending `skip` days before today)."""
    out = np.full((T, N), np.nan)
    for t in range(L+skip+1, T):
        out[t] = -(cs[t-1-skip] - cs[t-L-1-skip])
    return out

def valid(L, skip=0):
    v = np.zeros((T, N), bool)
    for t in range(L+skip+1, T):
        v[t] = ~np.any(d[t-L-skip:t] == 0, axis=0)
    return v

vol = pd.DataFrame(d).replace(0, np.nan).rolling(60, min_periods=20).std().to_numpy()

def zscore(x):
    m = np.nanmean(x, 1, keepdims=True); s = np.nanstd(x, 1, keepdims=True)
    return (x - m)/np.where(s > 0, s, 1)

def xrank(x):
    r = pd.DataFrame(x).rank(axis=1, pct=True).to_numpy()
    return r - 0.5

def evaluate(W, V, label):
    prev = np.zeros(N); rets, turns = [], []
    for t in range(T):        # W[t] uses data through t-1, so it trades day t
        w = np.where(V[t] & np.isfinite(W[t]), np.nan_to_num(W[t]), 0.0)
        s = np.abs(w).sum()
        if s == 0: continue
        w = w/s
        rets.append(float(w @ d[t])); turns.append(float(np.abs(w-prev).sum())); prev = w
    r = np.array(rets); tn = np.array(turns)
    net = r - ETA*tn
    f = lambda v: v.mean()/v.std()*np.sqrt(252)
    return dict(label=label, gross=f(r), net=f(net), ret=r.mean()*252,
                turn=tn.mean(), cost=ETA*tn.mean()*252)

rows = []
L0 = 30
rows.append(evaluate(cum(L0), valid(L0), "base: -cum 30d (theirs)"))
rows.append(evaluate(cum(5),  valid(5),  "shorter: -cum 5d"))
# multi-horizon ensemble, equal weight on standardised signals
ens = np.nanmean(np.stack([zscore(cum(L)) for L in (5,10,20,30)]), 0)
rows.append(evaluate(ens, valid(30), "ensemble of 5/10/20/30d"))
rows.append(evaluate(cum(5)/np.where(vol>0,vol,np.nan), valid(5), "5d / trailing vol"))
rows.append(evaluate(zscore(cum(5)), valid(5), "5d, cross-sec z-score"))
rows.append(evaluate(xrank(cum(5)), valid(5), "5d, cross-sec rank"))
q = zscore(cum(5)); qq = np.where(np.abs(q) > 1.0, q, 0.0)
rows.append(evaluate(qq, valid(5), "5d z, trade |z|>1 only"))
rows.append(evaluate(cum(5, skip=1), valid(5,1), "5d, skip most recent day"))

print(f"{'variant':30}{'gross':>8}{'net 5bp':>9}{'ann ret':>9}{'turn/day':>10}{'cost/yr':>9}")
print("-"*76)
for r in rows:
    print(f"{r['label']:30}{r['gross']:>8.2f}{r['net']:>9.2f}{r['ret']:>8.1%}"
          f"{r['turn']:>10.2f}{r['cost']:>9.1%}")


# ---- turnover-reduction variants (run as: python3 tweaks.py --turnover) ----
def run_buffered(W, V, label, buffer=0.0, smooth=0):
    """No-trade band vs EWMA smoothing, both aimed at the binding constraint."""
    if smooth:
        W = pd.DataFrame(W).ewm(span=smooth).mean().to_numpy()
    prev = np.zeros(N); rets, turns = [], []
    for t in range(T):
        w = np.where(V[t] & np.isfinite(W[t]), np.nan_to_num(W[t]), 0.0)
        s = np.abs(w).sum()
        if s == 0: continue
        w = w/s
        if buffer > 0:
            keep = np.abs(w - prev) < buffer*np.abs(prev).mean()
            w = np.where(keep, prev, w)
            s2 = np.abs(w).sum(); w = w/s2 if s2 > 0 else w
        rets.append(float(w @ d[t])); turns.append(float(np.abs(w-prev).sum())); prev = w
    r, tn = np.array(rets), np.array(turns); net = r - ETA*tn
    f = lambda v: v.mean()/v.std()*np.sqrt(252)
    print(f"{label:34}{f(r):>8.2f}{f(net):>9.2f}{r.mean()*252:>9.1%}{tn.mean():>10.2f}")
