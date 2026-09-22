"""Generate DLSA_modern_colab.ipynb -- their CNN+Transformer, our survivorship-free
residuals, 2012-2026.  Run: python3 notebooks/build_modern_nb.py"""
import json, os
def md(s): return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
def co(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                   "source":s.splitlines(keepends=True)}

cells = [
md('''# Did the deep model survive past 2016?

The CNN+Transformer of Guijarro-Ordonez, Pelger & Zanotti
([arXiv:2106.04028](https://arxiv.org/abs/2106.04028)) — **their model code,
their preprocessing** — trained on **our survivorship-free residuals** running to
September 2026. Their own published residuals stop at 2016-12, which is why this
question has been unanswerable until now.

The non-neural arms, measured on the same residuals:

| arm | 2002–2008 | 2009–2016 | 2017–2026 |
|---|---|---|---|
| reversal L=1 | +2.81 | +0.85 | **−0.03** |
| reversal L=5 | +1.87 | +0.87 | +0.36 |
| **reversal L=30** | +0.86 | +0.74 | **+0.47** |

Fast reversal is dead; slow reversal survives at about half strength and has
stopped decaying (2017–2021: 0.48, 2022–2026: 0.45).

**The question:** does the network still beat those, and does it still beat a
null? On their 1998–2016 residuals it scored 4.92 gross against a permuted-return
null of 0.46 — real, but earning almost everything before 2009 and losing to a
30-day reversal rule once costs were paid.'''),

co('''#@title 1. Upload the residuals
# From the repo:  python3 src/mp_export_cnn.py   ->  cache/mp_eps_cnn.npy (51 MB)
# Drag that file (and mp_eps_cnn_dates.npy) into Colab's file browser, or mount Drive.
import os, numpy as np
for f in ("mp_eps_cnn.npy", "mp_eps_cnn_dates.npy"):
    print(f, "FOUND" if os.path.exists(f) else "MISSING -- upload it")
data = np.load("mp_eps_cnn.npy").astype(np.float32)
dates = np.load("mp_eps_cnn_dates.npy")
import pandas as pd; dates = pd.DatetimeIndex(dates)
print(f"\\nresiduals {data.shape}  {dates[0].date()} .. {dates[-1].date()}")
print(f"active names/day: median {np.median((data != 0).sum(1)):.0f}")'''),

co('''#@title 2. Their model and preprocessing (trimmed clone, ~2 MB of code)
import torch, time, json, subprocess
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE -- Runtime > Change runtime type > GPU")
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).returncode
if not os.path.isdir("dlsa-public"):
    sh("git clone --depth 1 --filter=blob:none --no-checkout -q "
       "https://github.com/gregzanotti/dlsa-public.git")
    sh("cd dlsa-public && git sparse-checkout set --no-cone '/*' '!/residuals/*' "
       "&& git checkout -q main")
import sys; sys.path.insert(0, "dlsa-public")
from preprocess import preprocess_cumsum          # THEIR code
from models.CNNTransformer import CNNTransformer  # THEIR code
print("imported their preprocess_cumsum and CNNTransformer")'''),

co('''#@title 3. Parameters
LOOKBACK     = 30    #@param {type:"integer"}
EPOCHS       = 30    #@param {type:"integer"}   # their config: 100
RETRAIN_FREQ = 250   #@param {type:"integer"}   # their config: 125
TRAIN_LEN    = 1000  #@param {type:"integer"}
BATCH_DAYS   = 125   #@param {type:"integer"}
LR           = 0.001
N_NULL       = 1     #@param {type:"integer"}
BORROW_BP_YR = 35.0

windows, idxs = preprocess_cumsum(data, LOOKBACK)
idxs = idxs.numpy() if hasattr(idxs, "numpy") else idxs
wdates = dates[LOOKBACK:]
print("windows", windows.shape, " valid (day,stock) pairs", idxs.sum())'''),

co('''#@title 4. Benchmarks on the SAME residuals -- the bar the network must clear
def sharpe(r): return float(r.mean()/r.std()*np.sqrt(252)) if len(r) > 20 else float("nan")

def bench(Wt, sel, d, lb=LOOKBACK):
    w = np.where(sel, Wt, 0.0)
    w = w / np.abs(w).sum(1, keepdims=True).clip(1e-12)
    g  = np.array([float(w[i] @ d[lb+i]) for i in range(len(w))])
    to = np.abs(np.diff(w, axis=0)).sum(1); to = np.r_[to[0], to]
    bo = np.abs(np.clip(w, None, 0)).sum(1) * BORROW_BP_YR*1e-4/252
    return g, to, bo

last = windows[:, :, -1]
def rev(L): return -(last - (windows[:, :, -L-1] if L < LOOKBACK else 0.0))
m17 = wdates >= "2017-01-01"
print(f"{'arm':10s} {'gross':>7s} {'turn':>6s} {'net@1bp+borrow':>16s}   (2017-2026)")
print("-"*48)
for nm, Wt in (("L=1", rev(1)), ("L=5", rev(5)), ("L=30", rev(30))):
    g, to, bo = bench(Wt, idxs, data)
    print(f"{nm:10s} {sharpe(g[m17]):+7.2f} {to[m17].mean():6.2f} "
          f"{sharpe(g[m17]-1e-4*to[m17]-bo[m17]):+16.2f}")'''),

co('''#@title 5. Training loop -- their model, per-block, turnover tracked
MAX_ROWS = 32768
def fwd(model, x):
    if x.shape[0] <= MAX_ROWS: return model(x).squeeze(-1)
    return torch.cat([model(x[i:i+MAX_ROWS]).squeeze(-1)
                      for i in range(0, x.shape[0], MAX_ROWS)])

def train_eval(d, sel_all, win_all, seed=0, null=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    rp = np.random.default_rng(seed + 7919)
    Tn, N = win_all.shape[0], sel_all.shape[1]
    dd = d
    if null == "permute":                 # destroy own-history -> own-future only
        dd = d.copy()
        for t in range(LOOKBACK, len(dd)):
            act = np.nonzero(dd[t])[0]
            if len(act) > 1: dd[t, act] = dd[t, act][rp.permutation(len(act))]
    blocks, all_r, all_to, all_d = [], [], [], []
    prev_w = np.zeros(N)
    starts = list(range(TRAIN_LEN, Tn - RETRAIN_FREQ, RETRAIN_FREQ))
    for bi, s0 in enumerate(starts):
        model = CNNTransformer(logdir=None, random_seed=seed, lookback=LOOKBACK,
                               device=dev, dropout=0.25, filter_numbers=[1,8],
                               filter_size=2, attention_heads=4,
                               hidden_units_factor=2, normalization_conv=True,
                               use_transformer=True, use_convolution=True).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        tr = np.arange(s0-TRAIN_LEN, s0)
        for ep in range(EPOCHS):
            for ch in [tr[i:i+BATCH_DAYS] for i in range(0, len(tr), BATCH_DAYS)]:
                sub = sel_all[ch]
                x = torch.tensor(win_all[ch][sub], device=dev)
                if len(x) < 10: continue
                W = torch.zeros(len(ch), N, device=dev)
                W[torch.tensor(sub, device=dev)] = fwd(model, x)
                W = W / W.abs().sum(1, keepdim=True).clamp(min=1e-9)
                r = (W * torch.tensor(dd[LOOKBACK+ch], device=dev)).sum(1)
                loss = -r.mean()/r.std().clamp(min=1e-9)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); br, bt = [], []
        te = np.arange(s0, min(s0+RETRAIN_FREQ, Tn))
        with torch.no_grad():
            for ch in [te[i:i+BATCH_DAYS] for i in range(0, len(te), BATCH_DAYS)]:
                sub = sel_all[ch]
                x = torch.tensor(win_all[ch][sub], device=dev)
                if len(x) < 10: continue
                W = torch.zeros(len(ch), N, device=dev)
                W[torch.tensor(sub, device=dev)] = fwd(model, x)
                W = W / W.abs().sum(1, keepdim=True).clamp(min=1e-9)
                br.append((W*torch.tensor(dd[LOOKBACK+ch], device=dev)).sum(1).cpu().numpy())
                Wc = W.cpu().numpy().astype(np.float64)
                for k in range(len(Wc)):
                    bt.append(np.abs(Wc[k]-prev_w).sum()); prev_w = Wc[k]
        br = np.concatenate(br); bt = np.array(bt)
        all_r.append(br); all_to.append(bt); all_d.append(wdates[te][:len(br)])
        blocks.append({"year": str(wdates[s0].date()), "sharpe": sharpe(br),
                       "turnover": float(bt.mean())})
        print(f"    block {bi+1}/{len(starts)}  {blocks[-1]['year']}  "
              f"SR {blocks[-1]['sharpe']:+.2f}  turn {blocks[-1]['turnover']:.2f}"
              f"   ({time.time()-T0:.0f}s)", flush=True)
    return (np.concatenate(all_r), np.concatenate(all_to),
            pd.DatetimeIndex(np.concatenate(all_d)), blocks)'''),

co('''#@title 6. Run
T0 = time.time()
print("REAL arm ...", flush=True)
r, to, dd_, blocks = train_eval(data, idxs, windows, seed=0)
bo = 0.0  # borrow applied below via the benchmark helper convention
m = dd_ >= "2017-01-01"
res = {"blocks": blocks,
       "gross_all": sharpe(r), "gross_2017": sharpe(r[m]),
       "turnover_2017": float(to[m].mean()),
       "net1bp_2017": sharpe(r[m] - 1e-4*to[m])}
print(f"\\nCNN  2017-2026: gross {res['gross_2017']:+.2f}  "
      f"turn {res['turnover_2017']:.2f}  net@1bp {res['net1bp_2017']:+.2f}")

nulls = []
for i in range(N_NULL):
    print(f"\\nNULL [permute] {i+1}/{N_NULL} ...", flush=True)
    rn, tn, dn, bn = train_eval(data, idxs, windows, seed=100+i, null="permute")
    mn = dn >= "2017-01-01"
    nulls.append(sharpe(rn[mn])); print(f"  null 2017-2026 = {nulls[-1]:+.2f}")
res["null_2017"] = nulls
json.dump(res, open("modern_results.json","w"), indent=2)
np.save("cnn_modern_daily.npy", np.vstack([r, to]))

print("\\n" + "="*58)
print(f"{'arm':22s} {'2017-2026 gross':>16s}")
print("-"*58)
print(f"{'CNN+Transformer':22s} {res['gross_2017']:+16.2f}")
print(f"{'  permute null':22s} {np.mean(nulls):+16.2f}")
print(f"{'reversal L=30 (ours)':22s} {0.47:+16.2f}")
print(f"{'reversal L=1  (ours)':22s} {-0.03:+16.2f}")
print("="*58)
try:
    from google.colab import files
    for f in ("modern_results.json","cnn_modern_daily.npy"): files.download(f)
except Exception as e: print("download skipped:", e)'''),
]
nb = {"cells": cells, "metadata": {"accelerator":"GPU","colab":{"provenance":[]},
      "kernelspec":{"display_name":"Python 3","name":"python3"},
      "language_info":{"name":"python"}}, "nbformat":4,"nbformat_minor":0}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DLSA_modern_colab.ipynb")
json.dump(nb, open(out,"w"), indent=1)
import ast
bad=0
for i,c in enumerate(cells):
    if c["cell_type"]!="code": continue
    t="\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith(("!","%")))
    try: ast.parse(t)
    except SyntaxError as e: print(f"cell {i}: {e}"); bad+=1
print(f"wrote {out}: {len(cells)} cells, {bad} syntax problems")
