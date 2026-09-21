"""Turnover and NET Sharpe for every arm, on the authors' PCA-5 residuals.

Cost model: weights are L1-normalised to gross 1 each day, so
    turnover_t = sum_i |w_{i,t} - w_{i,t-1}|      (0..2 for a full flip)
    net_t      = gross_t - c * turnover_t
c is round-trip cost per unit traded.  Reported for 1/3/5/10 bp.
Also reports the breakeven c at which net Sharpe hits zero.
"""
import numpy as np, json, warnings; warnings.filterwarnings("ignore")
from real_ladder import load, windows_and_mask, sharpe, w_ou, LOOKBACK

TRAIN_LEN, RETRAIN_FREQ, LAM = 1000, 250, 1e2
data = load("PCA-5"); W, sel = windows_and_mask(data)
Tn = W.shape[0]; S0 = TRAIN_LEN
last = W[:, :, -1]
def rev(L): return -(last - (W[:, :, -L-1] if L < LOOKBACK else 0.0))
def norm(Wt):
    w = np.where(sel, Wt, 0.0)
    return w / np.abs(w).sum(1, keepdims=True).clip(1e-12)

def band(w, b=0.5):
    """No-trade band: hold the old weight unless the target moved more than
    b * (mean |target|).  Buffering, not smoothing."""
    out = np.zeros_like(w); prev = np.zeros(w.shape[1])
    for t in range(len(w)):
        thr = b*np.abs(w[t]).mean()
        cur = np.where(np.abs(w[t]-prev) > thr, w[t], prev)
        s = np.abs(cur).sum(); cur = cur/s if s > 0 else cur
        out[t] = cur; prev = cur
    return out

ARMS = {
 "reversal L=1": rev(1), "reversal L=2": rev(2), "reversal L=5": rev(5),
 "reversal L=30": rev(30), "OU+Threshold": w_ou(W, sel),
 "ensemble 5/10/20/30": sum(a/np.abs(a).sum(1,keepdims=True).clip(1e-12)
                            for a in (rev(5),rev(10),rev(20),rev(30))),
}
ARMS["exp decay hl=2d"] = -(last - (W*(lambda w: w/np.abs(w).sum())(
    np.exp(-np.log(2)*np.arange(LOOKBACK)[::-1]/2))).sum(2))

rows = []
for nm, Wt in list(ARMS.items()):
    for tag, w in ((nm, norm(Wt)), (nm+" + band", band(norm(Wt)))):
        g = (w[S0:]*data[LOOKBACK+S0:LOOKBACK+Tn]).sum(1)
        to = np.abs(np.diff(w[S0-1:], axis=0)).sum(1)
        nets = {c: sharpe(g - c*1e-4*to) for c in (0,1,3,5,10)}
        lo, hi = 0.0, 200.0
        for _ in range(60):
            mid = (lo+hi)/2
            if sharpe(g - mid*1e-4*to) > 0: lo = mid
            else: hi = mid
        rows.append((tag, to.mean(), nets, lo))

print(f"{'arm':30s} {'turn':>6s} {'gross':>7s} {'1bp':>7s} {'3bp':>7s} {'5bp':>7s} {'10bp':>7s} {'b/e bp':>7s}")
print("-"*82)
for tag, to, n, be in sorted(rows, key=lambda r: -r[2][5]):
    print(f"{tag:30s} {to:6.2f} {n[0]:+7.2f} {n[1]:+7.2f} {n[3-0]:+7.2f} {n[5]:+7.2f} {n[10]:+7.2f} {be:7.1f}")
print("\nCNN+Transformer gross 4.96 (turnover not measured -- needs the Colab run)")
print("null (permute) gross 0.46")
