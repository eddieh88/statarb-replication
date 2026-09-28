"""Per-retrain-block Sharpe for the non-NN benchmarks, on the SAME block
boundaries the CNN+Transformer used, so the comparison is apples-to-apples:
  their PCA-5 residuals, residual-space L1 normalisation, TRAIN_LEN=1000,
  RETRAIN_FREQ=250, LOOKBACK=30, blocks start at day 1000 of the window index.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from real_ladder import load, windows_and_mask, port_returns, sharpe, w_ou, LOOKBACK

TRAIN_LEN, RETRAIN_FREQ = 1000, 250
CNN = [9.94,8.743,8.433,6.086,7.182,4.353,7.142,4.54,
       3.907,2.425,0.966,2.029,2.426,1.829,1.915]          # measured on Colab

data = load("PCA-5")
W, sel = windows_and_mask(data)
Tn = W.shape[0]
starts = list(range(TRAIN_LEN, Tn - RETRAIN_FREQ, RETRAIN_FREQ))
print(f"residuals {data.shape}  windows {W.shape}  {len(starts)} blocks")

def rev(L):                       # cumulative residual over the last L days
    return -(W[:, :, -1] - (W[:, :, -L-1] if L < LOOKBACK else 0.0))

ARMS = {
    "reversal L=30": rev(30),
    "reversal L=10": rev(10),
    "reversal L=5":  rev(5),
    "OU+Threshold":  w_ou(W, sel),
}
ARMS["ensemble 5/10/20/30"] = sum(
    a / np.abs(a).sum(1, keepdims=True).clip(1e-12) for a in
    (rev(5), rev(10), rev(20), rev(30)))

out = {"CNN+Transformer": CNN}
for nm, Wt in ARMS.items():
    r = port_returns(Wt, data, sel)
    blocks = []
    for s0 in starts:
        seg = r[s0:min(s0+RETRAIN_FREQ, len(r))]
        blocks.append(round(sharpe(seg), 3) if len(seg) > 20 else None)
    out[nm] = blocks
    b = np.array([x for x in blocks if x is not None])
    h = len(b)//2
    print(f"{nm:22s} pooled(blocks) {sharpe(r[starts[0]:]):+.2f}   "
          f"first {b[:h].mean():+.2f}  second {b[h:].mean():+.2f}  ratio {b[h:].mean()/b[:h].mean():.2f}")

c = np.array(CNN); h = len(c)//2
print(f"{'CNN+Transformer':22s} pooled          +4.96   first {c[:h].mean():+.2f}  second {c[h:].mean():+.2f}  ratio {c[h:].mean()/c[:h].mean():.2f}")
print(f"{'NULL (permute, x2)':22s} pooled          +0.46   first  +0.47  second  +0.42")

json.dump(out, open("results/per_block_bench.json", "w"), indent=1)
print("\nper-block series -> per_block_bench.json")
