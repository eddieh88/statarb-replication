"""Choose the factor count K walk-forward, using only prior data.

The K-sweep showed our headline 0.47 was the maximum across K = {1,3,5,10,15},
and that K=5 was picked only because the paper uses it.  That is hindsight.  Here
K is re-selected at every retrain boundary from performance on the PRECEDING
window, and applied to the next block out of sample -- the same discipline the
rest of the project uses.

Reported alongside: fixed K=5 (what we quoted), the mean across K (a robust
estimate), and best-in-hindsight (an upper bound nobody could have traded).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys, re, glob
from real_ladder import windows_and_mask, sharpe, LOOKBACK

KS = [1, 3, 5, 10, 15]
LOOK, STEP, BORROW = 1000, 250, 35.0

def daily(d, L=30):
    W, sel = windows_and_mask(d); last = W[:, :, -1]
    Wt = -(last - (W[:, :, -L-1] if L < LOOKBACK else 0.0))
    w = np.where(sel, Wt, 0.0); w = w/np.abs(w).sum(1, keepdims=True).clip(1e-12)
    g  = np.array([float(w[i] @ d[LOOKBACK+i]) for i in range(len(w))])
    to = np.abs(np.diff(w, axis=0)).sum(1); to = np.r_[to[0], to]
    bo = np.abs(np.clip(w, None, 0)).sum(1)*BORROW*1e-4/252
    return g, to, bo

print("loading residuals for K =", KS, flush=True)
G, TO, BO, IDX = {}, {}, {}, None
for K in KS:
    e = pd.read_parquet(f"cache/mp_eps_K{K}_N900.parquet")
    e = e.loc[:, e.notna().any()]
    g, to, bo = daily(np.nan_to_num(e.values.astype(np.float32)))
    G[K], TO[K], BO[K] = g, to, bo
    IDX = e.index[LOOKBACK:LOOKBACK+len(g)]
    print(f"  K={K:2d}  {len(g)} days", flush=True)

net = {K: G[K] - 1e-4*TO[K] - BO[K] for K in KS}
T = len(IDX)
starts = list(range(LOOK, T - STEP, STEP))
picks, wf_g, wf_n, wf_d = [], [], [], []
for s0 in starts:
    tr = slice(s0-LOOK, s0); te = slice(s0, min(s0+STEP, T))
    best = max(KS, key=lambda K: sharpe(net[K][tr]))      # prior data only
    picks.append((str(IDX[s0].date()), best))
    wf_g.append(G[best][te]); wf_n.append(net[best][te]); wf_d.append(IDX[te])
wf_g = np.concatenate(wf_g); wf_n = np.concatenate(wf_n)
wf_d = pd.DatetimeIndex(np.concatenate(wf_d))

def era(d, g, n, lo, hi):
    m = (d >= lo) & (d <= hi)
    return sharpe(g[m]), sharpe(n[m])

print("\nK chosen walk-forward at each retrain:")
print("  " + "  ".join(f"{d[:4]}:K{k}" for d, k in picks))
from collections import Counter
print("  distribution:", dict(Counter(k for _, k in picks)))

print(f"\n{'method':26s} {'2009-2016':>18s} {'2017-2026':>18s}")
print(f"{'':26s} {'gross':>9s}{'net':>9s} {'gross':>9s}{'net':>9s}")
print("-"*64)
rows = [("walk-forward K", wf_d, wf_g, wf_n)]
for K in KS:
    rows.append((f"fixed K={K}", IDX, G[K], net[K]))
for nm, d, g, n in rows:
    a = era(d, g, n, "2009-01-01", "2016-12-31")
    b = era(d, g, n, "2017-01-01", "2026-12-31")
    star = "  <-- honest number" if nm == "walk-forward K" else ""
    print(f"{nm:26s} {a[0]:+9.2f}{a[1]:+9.2f} {b[0]:+9.2f}{b[1]:+9.2f}{star}")
m = (IDX >= "2017-01-01")
mean_net = np.mean([sharpe(net[K][m]) for K in KS])
print(f"{'mean across K (robust)':26s} {'':>18s} {'':>9s}{mean_net:+9.2f}")
