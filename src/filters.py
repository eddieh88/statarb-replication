"""Weighted averages of the cumulative-residual path, on the authors' PCA-5
residuals.  Same blocks / normalisation as the CNN run, so everything is
directly comparable to CNN 4.96 and null 0.46.

The signal for stock i on day t is  s = sum_k beta_k * W[t,i,k],  where W is the
30-day cumulative residual path.  Every arm below is a choice of beta:
  reversal L      : beta = -1 on the last point, +1 at lag L   (a difference)
  exponential     : geometric decay
  fitted (ridge)  : beta learned per block from the previous 1000 days
"""
import numpy as np, json, warnings; warnings.filterwarnings("ignore")
from real_ladder import load, windows_and_mask, port_returns, sharpe, LOOKBACK

TRAIN_LEN, RETRAIN_FREQ, LAM = 1000, 250, 1e2
data = load("PCA-5"); W, sel = windows_and_mask(data)
Tn = W.shape[0]; starts = list(range(TRAIN_LEN, Tn - RETRAIN_FREQ, RETRAIN_FREQ))

def blockwise(r):
    b = [sharpe(r[s0:min(s0+RETRAIN_FREQ, len(r))]) for s0 in starts]
    b = np.array(b); h = len(b)//2
    return b, b[:h].mean(), b[h:].mean()

def report(nm, Wt, store):
    r = port_returns(Wt, data, sel)
    b, f, s = blockwise(r)
    store[nm] = b.round(3).tolist()
    print(f"{nm:26s} pooled {sharpe(r[starts[0]:]):+5.2f}   first {f:+5.2f}  "
          f"second {s:+5.2f}   net-of-null {sharpe(r[starts[0]:])-0.46:+5.2f}")

out = {}
last = W[:, :, -1]
def rev(L): return -(last - (W[:, :, -L-1] if L < LOOKBACK else 0.0))

for L in (1,2,3,5):
    report(f"reversal L={L}", rev(L), out)

# --- 1. exponential-decay weights: the optimal shape if residuals are OU ---
for hl in (2, 3, 5, 10):
    w = np.exp(-np.log(2)*np.arange(LOOKBACK)[::-1]/hl)      # weight by recency
    w = w/np.abs(w).sum()
    # signal = -(weighted average of path deviations from its own weighted mean)
    sig = -(last - (W*w).sum(2))
    report(f"exp decay hl={hl}d", sig, out)

# --- 2. inverse-volatility scaled reversal ---
vol = W[:, :, 1:].std(2) + 1e-8
report("reversal L=5 / vol", rev(5)/vol, out)
report("ensemble / vol", sum(a/np.abs(a).sum(1,keepdims=True).clip(1e-12)
                             for a in (rev(5),rev(10),rev(20),rev(30)))/vol, out)

# --- 3. fitted 30-lag filter, ridge, refit every block on prior 1000 days ---
sig = np.zeros((Tn, W.shape[1]), np.float32); betas = []
for bi, s0 in enumerate(starts):
    tr = np.arange(s0-TRAIN_LEN, s0)
    m = sel[tr]
    X = W[tr][m].astype(np.float64); y = data[LOOKBACK+tr][m].astype(np.float64)
    sd = X.std(0) + 1e-12; Xs = X/sd
    beta = np.linalg.solve(Xs.T@Xs + LAM*np.eye(LOOKBACK), Xs.T@y)/sd
    betas.append(beta)
    lo = 0 if bi == 0 else s0            # warm-up days use the first beta so
    hi = Tn if s0 == starts[-1] else s0+RETRAIN_FREQ   # r stays full length
    sig[lo:hi] = (W[lo:hi]*beta).sum(2)
report("fitted 30-lag (ridge)", sig, out)

B = np.array(betas); bm = B.mean(0); bm = bm/np.abs(bm).max()
print("\nfitted beta by lag (1=oldest, 30=newest), scaled to max |1|:")
print("  " + "  ".join(f"{v:+.2f}" for v in bm[:10]))
print("  " + "  ".join(f"{v:+.2f}" for v in bm[10:20]))
print("  " + "  ".join(f"{v:+.2f}" for v in bm[20:]))
json.dump(out, open("results/filters_blocks.json", "w"), indent=1)
