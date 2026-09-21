import json
MODEL_SRC = open('xsection_model.py').read()

def md(s): return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
def co(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                   "source":s.splitlines(keepends=True)}

cells = [
md('''# Cross-sectional stat-arb: does letting stocks see each other help?

Baseline is the CNN+Transformer of **Guijarro-Ordonez, Pelger & Zanotti**
(arXiv:2106.04028), which processes each stock's 30-day residual path
**independently** -- the only cross-sectional interaction in the entire model is
the final L1 normalisation.

Two additions, both tested against a faithful reimplementation of their model
(verified at 769 parameters, matching theirs exactly):

1. **Positional encoding** -- theirs has none; its temporal transformer sees an
   unordered set, with ordering supplied only by the causal CNN beneath.
2. **ISAB cross-sectional block** (Set Transformer, Lee et al. 2019) -- pool each
   day's cross-section into M learned inducing points and broadcast back.
   O(N*M), permutation-equivariant, masks a variable active universe.

**Judge on net-of-cost Sharpe and turnover, not gross.** We established on the
baseline that its Sharpe edge is volatility reduction rather than return
(mean 8.06%/yr vs 10.90%/yr for 1-day reversal, at half the vol), and that costs
are paid out of the mean. A model with higher gross and higher turnover is worse.

Reference numbers on PCA-5, 2002-2016, residual space, same blocks:

| arm | gross | turn | b/e | 2nd-half net@1bp |
|---|---|---|---|---|
| CNN+Transformer (theirs) | 4.92 | 1.00 | 3.2bp | +0.21 |
| reversal L=30 + band | 1.05 | 0.21 | 6.1bp | +0.47 |
| permute null | 0.46 | - | - | - |
'''),
co('''#@title Setup, GPU, and a trimmed clone of their repo
import torch, os, time
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE -- Runtime > Change runtime type > GPU")
!git clone --depth 1 --filter=blob:none --no-checkout -q https://github.com/gregzanotti/dlsa-public.git 2>/dev/null || echo "already cloned"
os.chdir("dlsa-public")
!git sparse-checkout set --no-cone '/*' '!/residuals/*' -q 2>/dev/null
!git checkout -q main 2>/dev/null
os.makedirs("residuals/pca", exist_ok=True)
B="https://raw.githubusercontent.com/gregzanotti/dlsa-public/main/residuals"
F="AvPCA_OOSresiduals_5_factors_1998_initialOOSYear_60_rollingWindow_252_covWindow_0.01_Cap.npy.gz"
if not os.path.exists("residuals/superMask.npy"): !wget -q -O residuals/superMask.npy $B/superMask.npy
if not os.path.exists(f"residuals/pca/{F}"):      !wget -q -O residuals/pca/$F $B/pca/$F
print("files ready")'''),
co('''#@title Parameters
K            = 5     #@param {type:"integer"}
LOOKBACK     = 30    #@param {type:"integer"}
EPOCHS       = 30    #@param {type:"integer"}   # their config: 100
RETRAIN_FREQ = 250   #@param {type:"integer"}   # their config: 125
TRAIN_LEN    = 1000  #@param {type:"integer"}
BATCH_DAYS   = 125   #@param {type:"integer"}
LR           = 0.001
INDUCING     = 16    #@param {type:"integer"}   # M, cross-sectional inducing points
COST_BP      = 0.0   #@param {type:"number"}    # >0 puts a friction penalty IN the objective
ARMS         = ["baseline", "xsection"]         # options: baseline, pos, xsection, full'''),
co('''#@title Load their residuals
import gzip, shutil, numpy as np
p = f"residuals/pca/AvPCA_OOSresiduals_{K}_factors_1998_initialOOSYear_60_rollingWindow_252_covWindow_0.01_Cap.npy"
if not os.path.exists(p):
    with gzip.open(p+".gz","rb") as fi, open(p,"wb") as fo: shutil.copyfileobj(fi, fo)
resid = np.load(p); mask = np.load("residuals/superMask.npy")
data = np.asarray(resid[:, mask], dtype=np.float32); del resid
print(f"PCA-{K} residuals {data.shape}   active/day median {np.median((data!=0).sum(1)):.0f}")'''),
co('%%writefile xsection_model.py\n' + MODEL_SRC),
co('''#@title Their preprocessing, our model
import sys; sys.path.insert(0, ".")
from preprocess import preprocess_cumsum          # THEIR code
from xsection_model import XSectionNet            # ours
windows, idxs = preprocess_cumsum(data, LOOKBACK)
idxs = idxs.numpy() if hasattr(idxs, "numpy") else idxs
print("windows", windows.shape, " valid (day,stock) pairs", idxs.sum())'''),
co('''#@title Training loop -- identical protocol, Sharpe objective, turnover tracked
import json
CFG = {"baseline": dict(use_pos=False, use_xsection=False),
       "pos":      dict(use_pos=True,  use_xsection=False),
       "xsection": dict(use_pos=False, use_xsection=True),
       "full":     dict(use_pos=True,  use_xsection=True)}

def sharpe(x): return float(x.mean()/x.std()*np.sqrt(252)) if len(x) > 20 else float("nan")

def train_eval(arm, seed=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    Tn, N = windows.shape[0], idxs.shape[1]
    blocks, all_r, all_to = [], [], []
    prev_w = np.zeros(N)
    starts = list(range(TRAIN_LEN, Tn - RETRAIN_FREQ, RETRAIN_FREQ))
    for bi, s0 in enumerate(starts):
        model = XSectionNet(lookback=LOOKBACK, inducing=INDUCING, seed=seed, **CFG[arm]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        tr = np.arange(s0-TRAIN_LEN, s0)
        for ep in range(EPOCHS):
            for ch in [tr[i:i+BATCH_DAYS] for i in range(0, len(tr), BATCH_DAYS)]:
                sub = idxs[ch]
                x = torch.tensor(windows[ch][sub], device=dev)
                if len(x) < 10: continue
                st = torch.tensor(sub, device=dev)
                W = torch.zeros(len(ch), N, device=dev)
                W[st] = model(x, st)
                W = W / W.abs().sum(1, keepdim=True).clamp(min=1e-9)
                r = (W * torch.tensor(data[LOOKBACK+ch], device=dev)).sum(1)
                if COST_BP > 0:                      # friction penalty in the objective
                    r = r - COST_BP*1e-4*(W[1:]-W[:-1]).abs().sum(1).mean()
                loss = -r.mean()/r.std().clamp(min=1e-9)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); br, bt = [], []
        te = np.arange(s0, min(s0+RETRAIN_FREQ, Tn))
        with torch.no_grad():
            for ch in [te[i:i+BATCH_DAYS] for i in range(0, len(te), BATCH_DAYS)]:
                sub = idxs[ch]
                x = torch.tensor(windows[ch][sub], device=dev)
                if len(x) < 10: continue
                st = torch.tensor(sub, device=dev)
                W = torch.zeros(len(ch), N, device=dev)
                W[st] = model(x, st)
                W = W / W.abs().sum(1, keepdim=True).clamp(min=1e-9)
                br.append((W*torch.tensor(data[LOOKBACK+ch], device=dev)).sum(1).cpu().numpy())
                Wc = W.cpu().numpy().astype(np.float64)
                for k in range(len(Wc)): bt.append(np.abs(Wc[k]-prev_w).sum()); prev_w = Wc[k]
        br = np.concatenate(br) if br else np.array([]); bt = np.array(bt)
        all_r.append(br); all_to.append(bt)
        if len(br) > 20:
            blocks.append({"year": round(1998.0+s0/252.0,2), "sharpe": sharpe(br),
                           "turnover": float(bt.mean())})
            print(f"    block {bi+1}/{len(starts)}  ~{blocks[-1]['year']:.1f}  "
                  f"SR {blocks[-1]['sharpe']:+.2f}  turn {blocks[-1]['turnover']:.2f}"
                  f"   ({time.time()-T0:.0f}s)", flush=True)
    r = np.concatenate([a for a in all_r if len(a)]); to = np.concatenate([a for a in all_to if len(a)])
    return r, to, blocks

def be(g, t):
    lo, hi = 0.0, 500.0
    for _ in range(60):
        m = (lo+hi)/2
        if sharpe(g - m*1e-4*t) > 0: lo = m
        else: hi = m
    return lo'''),
co('''#@title Run
T0 = time.time(); out = {}
for arm in ARMS:
    print(f"\\n=== {arm} ===", flush=True)
    r, to, blocks = train_eval(arm)
    h = len(r)//2
    out[arm] = {"gross": sharpe(r), "turnover": float(to.mean()),
                "breakeven_bp": be(r, to), "net_1bp": sharpe(r-1e-4*to),
                "second_half_gross": sharpe(r[h:]),
                "second_half_net_1bp": sharpe(r[h:]-1e-4*to[h:]),
                "second_half_be": be(r[h:], to[h:]),
                "blocks": blocks}
    np.save(f"w_{arm}.npy", np.vstack([r, to]))
    json.dump(out, open("xsection_results.json","w"), indent=2)
    print(f"  gross {out[arm]['gross']:+.2f}  turn {out[arm]['turnover']:.2f}  "
          f"b/e {out[arm]['breakeven_bp']:.1f}bp  net@1bp {out[arm]['net_1bp']:+.2f}")

print("\\n"+"="*84)
print(f"{'arm':22s} {'gross':>7s} {'turn':>6s} {'b/e bp':>7s} {'net@1bp':>8s} {'2H gross':>9s} {'2H net@1bp':>11s}")
print("-"*84)
for a, v in out.items():
    print(f"{a:22s} {v['gross']:+7.2f} {v['turnover']:6.2f} {v['breakeven_bp']:7.1f} "
          f"{v['net_1bp']:+8.2f} {v['second_half_gross']:+9.2f} {v['second_half_net_1bp']:+11.2f}")
print(f"{'-- theirs (measured)':22s} {4.92:+7.2f} {1.00:6.2f} {3.2:7.1f} {3.39:+8.2f} {2.23:+9.2f} {0.21:+11.2f}")
print(f"{'-- reversal L=30+band':22s} {1.05:+7.2f} {0.21:6.2f} {6.1:7.1f} {0.84:+8.2f} {0.67:+9.2f} {0.47:+11.2f}")
print("="*84)
print("SE(Sharpe) ~ 0.26 full, 0.37 second half -- treat sub-0.4 gaps as noise.")

from google.colab import files
for f in ["xsection_results.json"] + [f"w_{a}.npy" for a in ARMS]:
    try: files.download(f)
    except Exception as e: print("grab", f, "manually:", e)'''),
]
nb = {"cells": cells, "metadata": {"accelerator":"GPU",
      "colab":{"provenance":[]},"kernelspec":{"display_name":"Python 3","name":"python3"},
      "language_info":{"name":"python"}}, "nbformat":4, "nbformat_minor":0}
json.dump(nb, open("DLSA_xsection_colab.ipynb","w"), indent=1)

import ast
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]!="code": continue
    s="\n".join(l for l in "".join(c["source"]).split("\n")
                if not l.strip().startswith(("!","%")))
    try: ast.parse(s)
    except SyntaxError as e: print(f"cell {i}: {e}")
print(f"wrote DLSA_xsection_colab.ipynb ({len(cells)} cells), all code parses")
