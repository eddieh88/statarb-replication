import nbformat as nbf
nb = nbf.v4.new_notebook()
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip()))

M(r"""
# Null test on Deep Learning Statistical Arbitrage

Runs the CNN+Transformer from **Guijarro-Ordonez, Pelger & Zanotti** (arXiv:2106.04028)
against a no-signal null, on **the authors' own published residuals**.

Uses their model code and preprocessing directly from
[gregzanotti/dlsa-public](https://github.com/gregzanotti/dlsa-public); the training
loop is a reimplementation validated to reproduce their published OU+Threshold
benchmark (0.70 measured vs 0.73 published, PCA-5, 2002-2016).

**The question:** their Table I reports Sharpe 3.36 on PCA-5 residuals, gross of costs.
Does that survive a null in which the residuals carry no directional signal?

**Runtime:** set `EPOCHS` and `RETRAIN_FREQ` below to fit your budget. On a T4,
their full config (100 epochs / retrain 125d) is roughly 3-6h per arm; the reduced
default here is well under an hour per arm. Results are written to disk after every
arm, so a disconnect loses at most one arm.
""")

C(r"""
#@title Setup and GPU check
import torch, subprocess, os
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - set Runtime > Change runtime type > GPU")
print("torch", torch.__version__)
!git clone -q https://github.com/gregzanotti/dlsa-public.git 2>/dev/null || echo "already cloned"
os.chdir("dlsa-public")
print("files:", sorted(os.listdir())[:12])
""")

C(r"""
#@title Parameters
FACTOR_MODEL  = "PCA"    #@param ["PCA", "FamaFrench"]
K             = 5        #@param {type:"integer"}
LOOKBACK      = 30       #@param {type:"integer"}
EPOCHS        = 30       #@param {type:"integer"}   # their config: 100
RETRAIN_FREQ  = 250      #@param {type:"integer"}   # their config: 125
TRAIN_LEN     = 1000     #@param {type:"integer"}   # their config: 1000
BATCH_DAYS    = 125      #@param {type:"integer"}   # their config: 125
LR            = 0.001
N_NULL        = 3        #@param {type:"integer"}
SKIP_YEARS    = 4        # their Table I trades 2002-2016, using 1998-2001 to train
""")

C(r"""
#@title Load the authors' residuals
import gzip, shutil, numpy as np

PATHS = {
 "PCA":        f"residuals/pca/AvPCA_OOSresiduals_{K}_factors_1998_initialOOSYear_60_rollingWindow_252_covWindow_0.01_Cap.npy",
 "FamaFrench": f"residuals/famafrench/DailyFamaFrench_OOSresiduals_{K}_factors_1998_initialOOSYear_60_rollingWindow_0.01_Cap.npy",
}
p = PATHS[FACTOR_MODEL]
if not os.path.exists(p):
    with gzip.open(p + ".gz", "rb") as fi, open(p, "wb") as fo:
        shutil.copyfileobj(fi, fo)
resid = np.load(p)
mask  = np.load("residuals/superMask.npy")
data  = np.asarray(resid[:, mask], dtype=np.float32)
del resid
print(f"{FACTOR_MODEL}-{K} residuals {data.shape}   (0 codes missing data, as in their preprocess.py)")
print(f"active stocks/day: median {np.median((data!=0).sum(1)):.0f}")
""")

C(r"""
#@title Their preprocessing and their model, imported directly from the repo
import sys; sys.path.insert(0, ".")
from preprocess import preprocess_cumsum          # THEIR code
from models.CNNTransformer import CNNTransformer  # THEIR code
import numpy as np, torch

windows, idxs = preprocess_cumsum(data, LOOKBACK)
idxs = idxs.numpy() if hasattr(idxs, "numpy") else idxs
print("windows", windows.shape, " valid (day,stock) pairs", idxs.sum())
""")

C(r"""
#@title Benchmarks with no training -- these validate the harness
def port_rets(W, d, sel, lb=LOOKBACK):
    out = []
    for i in range(W.shape[0]):
        w = np.where(sel[i], W[i], 0.0); s = np.abs(w).sum()
        if s > 0: out.append(float(w @ d[lb+i]) / s)
    return np.array(out)

def sharpe(r): return float(r.mean()/r.std()*np.sqrt(252))

def w_reversal(W, sel): return -W[:, :, -1]

def w_ou(W, sel, c_thresh=1.25, c_crit=0.25, c_close=0.50):
    \"\"\"Matches preprocess_ou() in their repo exactly.\"\"\"
    Tn, N, L = W.shape
    out = np.zeros((Tn, N)); prev = np.zeros(N)
    for i in range(Tn):
        x = W[i]; Ys, Xs = x[:,1:], x[:,:-1]
        mX, mY = Xs.mean(1), Ys.mean(1); vX, vY = Xs.var(1), Ys.var(1)
        cov = ((Xs-mX[:,None])*(Ys-mY[:,None])).mean(1)
        with np.errstate(all="ignore"):
            R2 = cov**2/(vX*vY); b = cov/vX; c = mY - b*mX
            mu = c/(1-b+1e-6)
            res = Ys - b[:,None]*Xs - c[:,None]
            sig = np.sqrt(res.var(1)/np.abs(1-b**2+1e-6))
            s = np.where(sig>0, (mu-Ys[:,-1])/sig, 0.0)
        ok = sel[i] & (b>0) & (b<1) & np.isfinite(s) & (R2>c_crit)
        cur = prev.copy()
        cur[(prev==0)&ok&(s> c_thresh)] =  1
        cur[(prev==0)&ok&(s<-c_thresh)] = -1
        cur[(prev!=0)&(np.abs(s)<c_close)] = 0
        cur[~sel[i]] = 0
        out[i] = cur; prev = cur
    return out

SKIP = SKIP_YEARS*252
for nm, fn in (("reversal (0 params)", w_reversal), ("OU+Threshold", w_ou)):
    W = fn(windows, idxs)
    print(f"{nm:22} full {sharpe(port_rets(W, data, idxs)):+.2f}   "
          f"2002-2016 {sharpe(port_rets(W[SKIP:], data[SKIP:], idxs[SKIP:])):+.2f}")
print("\nPaper Table I, PCA-5 gross: OU+Threshold 0.73")
print("If OU lands near 0.73 on the 2002-2016 column, the harness is faithful.")
""")

C(r"""
#@title Training loop (their model, their objective)
import time, json

def train_eval(d, sel_all, win_all, seed=0, log=True):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    Tn = win_all.shape[0]
    all_r = []
    starts = list(range(TRAIN_LEN, Tn - RETRAIN_FREQ, RETRAIN_FREQ))
    for bi, s0 in enumerate(starts):
        model = CNNTransformer(logdir=None, random_seed=seed, lookback=LOOKBACK,
                               device=dev, dropout=0.25, filter_numbers=[1,8],
                               filter_size=2, attention_heads=4,
                               hidden_units_factor=2, normalization_conv=True,
                               use_transformer=True, use_convolution=True).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        tr = np.arange(s0-TRAIN_LEN, s0)
        chunks = [tr[i:i+BATCH_DAYS] for i in range(0, len(tr), BATCH_DAYS)]
        for ep in range(EPOCHS):
            for ch in chunks:
                sub = sel_all[ch]
                x = torch.tensor(win_all[ch][sub], device=dev)
                if len(x) < 10: continue
                w_flat = model(x).squeeze(-1)
                W = torch.zeros(len(ch), sel_all.shape[1], device=dev)
                W[torch.tensor(sub, device=dev)] = w_flat
                W = W / W.abs().sum(1, keepdim=True).clamp(min=1e-9)
                rets = (W * torch.tensor(d[LOOKBACK+ch], device=dev)).sum(1)
                loss = -rets.mean()/rets.std().clamp(min=1e-9)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        te = np.arange(s0, min(s0+RETRAIN_FREQ, Tn))
        with torch.no_grad():
            for ch in [te[i:i+BATCH_DAYS] for i in range(0, len(te), BATCH_DAYS)]:
                sub = sel_all[ch]
                x = torch.tensor(win_all[ch][sub], device=dev)
                if len(x) < 10: continue
                W = torch.zeros(len(ch), sel_all.shape[1], device=dev)
                W[torch.tensor(sub, device=dev)] = model(x).squeeze(-1)
                W = W / W.abs().sum(1, keepdim=True).clamp(min=1e-9)
                r = (W * torch.tensor(d[LOOKBACK+ch], device=dev)).sum(1)
                all_r.append(r.cpu().numpy())
        if log: print(f"    block {bi+1}/{len(starts)}  ({time.time()-T0:.0f}s)", flush=True)
    r = np.concatenate(all_r)
    return float(r.mean()/r.std()*np.sqrt(252)), r
""")

C(r"""
#@title Run: real arm, then nulls
T0 = time.time()
results = {}

print("REAL arm ...")
sr_real, _ = train_eval(data, idxs, windows, seed=0)
results["real"] = sr_real
print(f"REAL Sharpe = {sr_real:+.3f}")
json.dump(results, open("/content/null_results.json","w"), indent=2)

rng = np.random.default_rng(0)
results["nulls"] = []
for i in range(N_NULL):
    print(f"\nNULL {i+1}/{N_NULL} (sign-flip each day's cross-sectional vector) ...")
    dn = data * rng.choice([-1.0, 1.0], size=(len(data),1)).astype(np.float32)
    wn, sn = preprocess_cumsum(dn, LOOKBACK)
    sn = sn.numpy() if hasattr(sn, "numpy") else sn
    s, _ = train_eval(dn, sn, wn, seed=100+i)
    results["nulls"].append(s); print(f"  null Sharpe = {s:+.3f}")
    json.dump(results, open("/content/null_results.json","w"), indent=2)

nl = np.array(results["nulls"])
print("\n" + "="*58)
print(f"REAL          {sr_real:+.3f}")
print(f"NULL mean     {nl.mean():+.3f}   sd {nl.std():.3f}   draws {np.round(nl,3).tolist()}")
print(f"EXCESS        {sr_real-nl.mean():+.3f}")
print("="*58)
print("Paper Table I, PCA-5 CNN+Transformer, gross: 3.36")
""")

M(r"""
## Reading the result

- **Null near zero, real well above it** -> the deep model finds genuine signal, and
  the paper's headline stands on a clean foundation.
- **Null materially positive** -> the training protocol manufactures performance, and
  every Sharpe in this literature needs re-reading against its own null.
- **Real not clearly above null** -> inconclusive on this configuration; try raising
  `EPOCHS` toward their 100 and lowering `RETRAIN_FREQ` toward their 125.

## Caveats

- The Phi transition matrices for their stock-space normalisation (`use_residual_weights:
  True`) are gitignored in the repo and 404, so this runs their residual-space path.
  Applied identically to both arms, so the real-vs-null comparison holds.
- Reduced `EPOCHS`/`RETRAIN_FREQ` relative to their config, for runtime. Also identical
  across arms.
- The `ipca_normalized` residuals appear to be scaled differently -- our OU benchmark
  gives 0.52 there against their published 0.97 -- so this notebook defaults to PCA,
  where the benchmark reproduces at 0.70 vs 0.73.
""")

nbf.write(nb, "DLSA_null_test_colab.ipynb")
print("wrote DLSA_null_test_colab.ipynb")
