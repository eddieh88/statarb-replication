"""How much performance survives if you only hold the top-N conviction names?
Each day: keep the N largest |weight|, renormalise to gross 1, trade those."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from real_ladder import load, windows_and_mask, sharpe, LOOKBACK

S0 = 1000
data = load("PCA-5"); W, sel = windows_and_mask(data)
last = W[:, :, -1]
def rev(L): return -(last - (W[:, :, -L-1] if L < LOOKBACK else 0.0))
ARMS = {"reversal L=1": rev(1),
        "ensemble 5/10/20/30": sum(a/np.abs(a).sum(1,keepdims=True).clip(1e-12)
                                   for a in (rev(5),rev(10),rev(20),rev(30)))}

def run(Wt, N):
    w = np.where(sel, Wt, 0.0)
    if N is not None:
        k = w.shape[1]-N
        idx = np.argpartition(np.abs(w), k, axis=1)[:, :k]
        np.put_along_axis(w, idx, 0.0, axis=1)
    w = w/np.abs(w).sum(1, keepdims=True).clip(1e-12)
    g  = (w[S0:]*data[LOOKBACK+S0:LOOKBACK+W.shape[0]]).sum(1)
    to = np.abs(np.diff(w[S0-1:], axis=0)).sum(1)
    h = len(g)//2
    be = 0.0; lo, hi = 0.0, 500.0
    for _ in range(60):
        m=(lo+hi)/2
        if sharpe(g-m*1e-4*to) > 0: lo=m
        else: hi=m
    return sharpe(g), sharpe(g[h:]), to.mean(), lo, sharpe(g-2e-4*to)

print(f"{'arm':22s} {'N':>5s} {'gross':>7s} {'2nd half':>9s} {'turn':>6s} {'b/e bp':>7s} {'net@2bp':>8s} {'$ @5k/pos':>11s}")
print("-"*82)
for nm, Wt in ARMS.items():
    for N in (10, 20, 50, 100, 200, 400, None):
        g, g2, to, be, n2 = run(Wt.copy(), N)
        lab = "all" if N is None else str(N)
        cap = "" if N is None else f"${N*5_000/1e6:.2f}M"
        if N is None: cap = "$4.45M"
        print(f"{nm:22s} {lab:>5s} {g:+7.2f} {g2:+9.2f} {to:6.2f} {be:7.1f} {n2:+8.2f} {cap:>11s}")
    print()
