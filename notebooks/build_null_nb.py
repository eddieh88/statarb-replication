"""Generate DLSA_null_test_colab.ipynb.

Kept as a builder rather than a hand-edited notebook so the code is reviewable
as plain Python and regenerable.  Run:  python3 notebooks/build_null_nb.py
"""
import json, os

def md(s): return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
def co(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                   "source":s.splitlines(keepends=True)}

cells = [
md('''# Null test on Deep Learning Statistical Arbitrage

Runs the CNN+Transformer of **Guijarro-Ordonez, Pelger & Zanotti**
([arXiv:2106.04028](https://arxiv.org/abs/2106.04028)) against a no-signal null,
on **the authors' own published residuals**, using **their** model code and
**their** preprocessing imported directly from
[gregzanotti/dlsa-public](https://github.com/gregzanotti/dlsa-public).

The training loop is a reimplementation, validated against their published
OU+Threshold benchmark (0.70 measured vs 0.73 published, PCA-5, 2002-2016).

**The questions.** Their Table I reports Sharpe 3.36 on PCA-5 residuals gross of
costs. Does that survive a null in which the residuals carry no directional
signal? And is profitability really "non-declining over time", as their
conclusion states?'''),

md('''## Measured results (T4, ~40 min per arm)

Reduced config (`EPOCHS=30, RETRAIN_FREQ=250` vs their `100 / 125`), applied
identically to every arm so the comparisons hold.

### Real arm, per 250-day retrain block

| period | Sharpe | turnover |
|---|---|---|
| 2002–2008 | 9.87 · 8.61 · 8.44 · 6.08 · 7.21 · 4.36 · 7.13 | 0.91 → 1.20 |
| 2009–2016 | 4.54 · 3.90 · 2.45 · 0.82 · 1.98 · 2.52 · 1.72 · 1.74 | 1.18 → 0.90 |

Pooled gross **+4.92**, turnover **1.00**, first half **+7.38**, second half
**+2.46**, slope **−0.614 Sharpe/year**, **t = −8.92**.

### Null — forward returns permuted across stocks, two draws

| | first half | second half | pooled |
|---|---|---|---|
| draw 1 | +0.54 | +0.29 | **+0.43** |
| draw 2 | +0.39 | +0.55 | **+0.49** |

Real beat null in **28 of 30** block comparisons. The null is flat over time, so
the real arm's decline is not a decaying artifact.

### Net of cost

| window | gross | 1bp | 3bp | 5bp | breakeven |
|---|---|---|---|---|---|
| full | +4.92 | +3.39 | +0.31 | −2.77 | 3.2bp |
| first half | +6.96 | +5.56 | +2.78 | +0.01 | 5.0bp |
| second half | **+2.23** | **+0.21** | −3.84 | −7.87 | **1.1bp** |

For reference, reversal L=30 with a no-trade band nets **+0.47** in the second
half at 1bp, on turnover of 0.21 — the 30-day rule beats the transformer once
trading is paid for.'''),

co('''#@title Setup, GPU, and a trimmed clone of their repo (~40MB, not 840MB)
import torch, os, time, subprocess, urllib.request, json
import numpy as np
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE -- Runtime > Change runtime type > GPU")

def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).returncode

if not os.path.isdir("dlsa-public"):
    sh("git clone --depth 1 --filter=blob:none --no-checkout -q "
       "https://github.com/gregzanotti/dlsa-public.git")
    sh("cd dlsa-public && git sparse-checkout set --no-cone '/*' '!/residuals/*' "
       "&& git checkout -q main")
if os.path.basename(os.getcwd()) != "dlsa-public":
    os.chdir("dlsa-public")

os.makedirs("residuals/pca", exist_ok=True)
B = "https://raw.githubusercontent.com/gregzanotti/dlsa-public/main/residuals"
F = ("AvPCA_OOSresiduals_5_factors_1998_initialOOSYear_60_rollingWindow_252_"
     "covWindow_0.01_Cap.npy.gz")
for src, dst in ((f"{B}/superMask.npy", "residuals/superMask.npy"),
                 (f"{B}/pca/{F}",       f"residuals/pca/{F}")):
    if not os.path.exists(dst):
        print("downloading", os.path.basename(dst), flush=True)
        urllib.request.urlretrieve(src, dst)
print("files ready")'''),

co('''#@title Parameters
FACTOR_MODEL = "PCA" #@param ["PCA", "FamaFrench"]
K            = 5     #@param {type:"integer"}
LOOKBACK     = 30    #@param {type:"integer"}
EPOCHS       = 30    #@param {type:"integer"}   # their config: 100
RETRAIN_FREQ = 250   #@param {type:"integer"}   # their config: 125
TRAIN_LEN    = 1000  #@param {type:"integer"}
BATCH_DAYS   = 125   #@param {type:"integer"}
LR           = 0.001
N_NULL       = 2     #@param {type:"integer"}
COST_BP      = 0.0   #@param {type:"number"}    # >0 puts a friction penalty IN the objective'''),

co('''#@title Load the authors' residuals
import gzip, shutil
p = (f"residuals/pca/AvPCA_OOSresiduals_{K}_factors_1998_initialOOSYear_60_"
     f"rollingWindow_252_covWindow_0.01_Cap.npy")
if not os.path.exists(p):
    with gzip.open(p + ".gz", "rb") as fi, open(p, "wb") as fo:
        shutil.copyfileobj(fi, fo)
resid = np.load(p); mask = np.load("residuals/superMask.npy")
data = np.asarray(resid[:, mask], dtype=np.float32); del resid
print(f"{FACTOR_MODEL}-{K} residuals {data.shape}   "
      f"(0 codes missing data, as in their preprocess.py)")
print(f"active stocks/day: median {np.median((data != 0).sum(1)):.0f}")'''),

co('''#@title Their preprocessing and their model, imported directly from the repo
import sys; sys.path.insert(0, ".")
from preprocess import preprocess_cumsum          # THEIR code
from models.CNNTransformer import CNNTransformer  # THEIR code

windows, idxs = preprocess_cumsum(data, LOOKBACK)
idxs = idxs.numpy() if hasattr(idxs, "numpy") else idxs
print("windows", windows.shape, " valid (day,stock) pairs", idxs.sum())'''),

co('''#@title Benchmarks with no training -- these validate the harness
def port_rets(W, d, sel, lb=LOOKBACK):
    out = []
    for i in range(W.shape[0]):
        w = np.where(sel[i], W[i], 0.0); s = np.abs(w).sum()
        if s > 0: out.append(float(w @ d[lb + i]) / s)
    return np.array(out)

def sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 20 else float("nan")

def w_reversal(W, sel):
    return -W[:, :, -1]

def w_ou(W, sel, c_thresh=1.25, c_crit=0.25, c_close=0.50):
    # Matches preprocess_ou() in their repo: AR(1) on the cumulative residual
    # path, mu = c/(1-b), sigma = sqrt(var(resid)/|1-b^2|), signal (mu-Y)/sigma,
    # gated on R2 and 0 < b < 1.
    Tn, N, L = W.shape
    out = np.zeros((Tn, N)); prev = np.zeros(N)
    for i in range(Tn):
        x = W[i]; Ys, Xs = x[:, 1:], x[:, :-1]
        mX, mY = Xs.mean(1), Ys.mean(1); vX, vY = Xs.var(1), Ys.var(1)
        cov = ((Xs - mX[:, None]) * (Ys - mY[:, None])).mean(1)
        with np.errstate(all="ignore"):
            R2 = cov ** 2 / (vX * vY); b = cov / vX; c = mY - b * mX
            mu = c / (1 - b + 1e-6)
            res = Ys - b[:, None] * Xs - c[:, None]
            sig = np.sqrt(res.var(1) / np.abs(1 - b ** 2 + 1e-6))
            s = np.where(sig > 0, (mu - Ys[:, -1]) / sig, 0.0)
        ok = sel[i] & (b > 0) & (b < 1) & np.isfinite(s) & (R2 > c_crit)
        cur = prev.copy()
        cur[(prev == 0) & ok & (s >  c_thresh)] =  1
        cur[(prev == 0) & ok & (s < -c_thresh)] = -1
        cur[(prev != 0) & (np.abs(s) < c_close)] = 0
        cur[~sel[i]] = 0
        out[i] = cur; prev = cur
    return out

SKIP = 4 * 252   # their Table I trades 2002-2016, using 1998-2001 to warm up
for nm, fn in (("reversal (0 params)", w_reversal), ("OU+Threshold", w_ou)):
    W = fn(windows, idxs)
    print(f"{nm:22} full {sharpe(port_rets(W, data, idxs)):+.2f}   "
          f"2002-2016 {sharpe(port_rets(W[SKIP:], data[SKIP:], idxs[SKIP:])):+.2f}")
print("\\nPaper Table I, PCA-5 gross: OU+Threshold 0.73")
print("If OU lands near 0.73 on the 2002-2016 column, the harness is faithful.")'''),
]

cells += [
co('''#@title Training loop -- per retrain block, turnover tracked
# Their attention kernel caps batch at 65535 rows; 125 days x ~890 active names
# is ~111k.  Chunk the forward instead of shrinking BATCH_DAYS, which would
# change the sample the Sharpe objective is computed over.  Gradients flow
# through the cat, so this is numerically identical to one pass.
MAX_ROWS = 32768
def fwd(model, x):
    if x.shape[0] <= MAX_ROWS:
        return model(x).squeeze(-1)
    return torch.cat([model(x[i:i+MAX_ROWS]).squeeze(-1)
                      for i in range(0, x.shape[0], MAX_ROWS)])


def train_eval(d, sel_all, win_all, seed=0, log=True, null=None):
    """Returns (pooled_sharpe, blocks, daily_gross, daily_turnover).

    null=None      : real arm.
    null="permute" : each day, permute stock identities in the FORWARD return
                     vector among that day's active names.  Windows, the daily
                     return cross-section and volatility clustering all survive
                     exactly; only own-history -> own-future-return is destroyed,
                     which is precisely what the strategy claims to exploit.
    null="signflip": flips each day's whole cross-sectional vector, leaving every
                     |residual| intact, so a strategy timing volatility rather
                     than direction survives it.  Leaky; secondary arm only.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    rp = np.random.default_rng(seed + 7919)
    Tn, N = win_all.shape[0], sel_all.shape[1]

    dd = d
    if null == "permute":
        dd = d.copy()
        for t in range(LOOKBACK, len(dd)):
            act = np.nonzero(dd[t])[0]
            if len(act) > 1:
                dd[t, act] = dd[t, act][rp.permutation(len(act))]
    elif null == "signflip":
        dd = d * rp.choice([-1.0, 1.0], size=(len(d), 1)).astype(np.float32)

    blocks, all_r, all_to = [], [], []
    prev_w = np.zeros(N)
    starts = list(range(TRAIN_LEN, Tn - RETRAIN_FREQ, RETRAIN_FREQ))
    for bi, s0 in enumerate(starts):
        model = CNNTransformer(logdir=None, random_seed=seed, lookback=LOOKBACK,
                               device=dev, dropout=0.25, filter_numbers=[1, 8],
                               filter_size=2, attention_heads=4,
                               hidden_units_factor=2, normalization_conv=True,
                               use_transformer=True, use_convolution=True).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        tr = np.arange(s0 - TRAIN_LEN, s0)
        for ep in range(EPOCHS):
            for ch in [tr[i:i+BATCH_DAYS] for i in range(0, len(tr), BATCH_DAYS)]:
                sub = sel_all[ch]
                x = torch.tensor(win_all[ch][sub], device=dev)
                if len(x) < 10: continue
                W = torch.zeros(len(ch), N, device=dev)
                W[torch.tensor(sub, device=dev)] = fwd(model, x)
                W = W / W.abs().sum(1, keepdim=True).clamp(min=1e-9)
                r = (W * torch.tensor(dd[LOOKBACK + ch], device=dev)).sum(1)
                if COST_BP > 0:
                    r = r - COST_BP * 1e-4 * (W[1:] - W[:-1]).abs().sum(1).mean()
                loss = -r.mean() / r.std().clamp(min=1e-9)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); block_r, block_to = [], []
        te = np.arange(s0, min(s0 + RETRAIN_FREQ, Tn))
        with torch.no_grad():
            for ch in [te[i:i+BATCH_DAYS] for i in range(0, len(te), BATCH_DAYS)]:
                sub = sel_all[ch]
                x = torch.tensor(win_all[ch][sub], device=dev)
                if len(x) < 10: continue
                W = torch.zeros(len(ch), N, device=dev)
                W[torch.tensor(sub, device=dev)] = fwd(model, x)
                W = W / W.abs().sum(1, keepdim=True).clamp(min=1e-9)
                block_r.append((W * torch.tensor(dd[LOOKBACK + ch],
                                                 device=dev)).sum(1).cpu().numpy())
                Wc = W.cpu().numpy().astype(np.float64)
                for k in range(len(Wc)):
                    block_to.append(np.abs(Wc[k] - prev_w).sum()); prev_w = Wc[k]

        br = np.concatenate(block_r) if block_r else np.array([])
        bt = np.array(block_to)
        all_r.append(br); all_to.append(bt)
        if len(br) > 20:
            blocks.append({"block": bi, "year": round(1998.0 + s0 / 252.0, 2),
                           "sharpe": sharpe(br), "turnover": float(bt.mean())})
            if log:
                print(f"    block {bi+1}/{len(starts)}  ~{blocks[-1]['year']:.1f}  "
                      f"SR {blocks[-1]['sharpe']:+.2f}  "
                      f"turn {blocks[-1]['turnover']:.2f}   "
                      f"({time.time() - T0:.0f}s)", flush=True)
    r  = np.concatenate([x for x in all_r  if len(x)])
    to = np.concatenate([x for x in all_to if len(x)])
    return sharpe(r), blocks, r, to


def decay_report(blocks, label):
    """OLS slope of per-block Sharpe on block index -- the pre-registered statistic."""
    s = np.array([b["sharpe"] for b in blocks]); x = np.arange(len(s), dtype=float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    resid = s - A @ coef
    se = np.sqrt((resid @ resid) / max(len(s) - 2, 1) * np.linalg.inv(A.T @ A)[1, 1])
    h = len(s) // 2
    print(f"\\n{label}")
    print(f"  per-block SR   {s.round(3).tolist()}")
    print(f"  per-block turn {[round(b['turnover'], 3) for b in blocks]}")
    print(f"  first half {s[:h].mean():+.2f}   second half {s[h:].mean():+.2f}")
    print(f"  slope {coef[1] * 252.0 / RETRAIN_FREQ:+.3f} SR/year   "
          f"t = {coef[1] / se if se > 0 else float('nan'):+.2f}")
    return {"per_block": s.round(3).tolist(), "first_half": float(s[:h].mean()),
            "second_half": float(s[h:].mean()),
            "slope_per_year": float(coef[1] * 252.0 / RETRAIN_FREQ),
            "t_slope": float(coef[1] / se) if se > 0 else None}


def net_report(r, to, label):
    """Net Sharpe after cost c (bp per unit traded) and the breakeven c."""
    def be(g, t):
        lo, hi = 0.0, 500.0
        for _ in range(60):
            m = (lo + hi) / 2
            if sharpe(g - m * 1e-4 * t) > 0: lo = m
            else: hi = m
        return lo
    h = len(r) // 2
    print(f"\\n{label}   mean turnover {to.mean():.2f}  (0..2 scale)")
    print(f"{'window':12s} {'gross':>7s} {'1bp':>7s} {'3bp':>7s} {'5bp':>7s} "
          f"{'10bp':>7s} {'b/e bp':>8s}")
    for nm, g, t in (("full", r, to), ("first half", r[:h], to[:h]),
                     ("second half", r[h:], to[h:])):
        row = "  ".join(f"{sharpe(g - c * 1e-4 * t):+5.2f}" for c in (0, 1, 3, 5, 10))
        print(f"{nm:12s}  {row}   {be(g, t):7.1f}")
    return {"turnover": float(to.mean()), "breakeven_bp": be(r, to),
            "breakeven_bp_second_half": be(r[h:], to[h:]),
            "net": {c: sharpe(r - c * 1e-4 * to) for c in (0, 1, 3, 5, 10)}}'''),

co('''#@title Run: real arm, then the permuted-return null
T0 = time.time()
OUT = "dlsa_null_results.json"
res = {"config": {"K": K, "LOOKBACK": LOOKBACK, "EPOCHS": EPOCHS,
                  "RETRAIN_FREQ": RETRAIN_FREQ, "TRAIN_LEN": TRAIN_LEN,
                  "COST_BP": COST_BP}}

print("REAL arm ...", flush=True)
sr, blk, r, to = train_eval(data, idxs, windows, seed=0, null=None)
print(f"REAL pooled gross Sharpe = {sr:+.3f}")
res["real"] = {"gross": sr, "blocks": blk,
               "decay": decay_report(blk, "REAL: decay over retrain blocks"),
               "cost":  net_report(r, to, "REAL: net of cost")}
np.save("cnn_daily.npy", np.vstack([r, to]))     # for offline cost/beta analysis
json.dump(res, open(OUT, "w"), indent=2)

res["null"] = []
for i in range(N_NULL):
    print(f"\\nNULL [permute] {i+1}/{N_NULL} ...", flush=True)
    s, b, rn, tn = train_eval(data, idxs, windows, seed=100 + i, null="permute")
    h = len(b) // 2
    sh_ = np.array([x["sharpe"] for x in b])
    res["null"].append({"gross": s, "first_half": float(sh_[:h].mean()),
                        "second_half": float(sh_[h:].mean())})
    print(f"  null pooled = {s:+.3f}")
    json.dump(res, open(OUT, "w"), indent=2)

nl = np.array([x["gross"] for x in res["null"]])
print("\\n" + "=" * 62)
print(f"REAL           {sr:+.3f}")
print(f"NULL (permute) mean {nl.mean():+.3f}  draws {np.round(nl, 3).tolist()}")
print(f"EXCESS         {sr - nl.mean():+.3f}")
print("=" * 62)
print("Paper Table I, PCA-5 CNN+Transformer, gross: 3.36")
print(f"\\nsaved -> {OUT}, cnn_daily.npy")
try:
    from google.colab import files
    for f in (OUT, "cnn_daily.npy"): files.download(f)
except Exception as e:
    print("download skipped:", e)'''),

md('''## Reading the result

- **Null near zero, real well above it** — the model finds genuine signal and the
  paper's headline stands on a clean foundation. *This is what we measured:
  4.92 against 0.43 / 0.49, with real beating null in 28 of 30 blocks.*
- **Null materially positive** — the training protocol manufactures performance,
  and every Sharpe in this literature needs re-reading against its own null.
- **Real not clearly above null** — inconclusive at this configuration; raise
  `EPOCHS` toward their 100 and lower `RETRAIN_FREQ` toward their 125.

Then read `decay_report`: a strongly negative slope means the pooled figure is an
average over a period that ended, not a forecast.

### Caveats

- The Φ transition matrices for their stock-space normalisation
  (`use_residual_weights: True`) are gitignored in their repo and 404, so this
  runs the residual-space path. Applied identically to every arm, so the
  real-vs-null comparison holds — but a residual-space Sharpe is not a tradable
  Sharpe.
- `EPOCHS` and `RETRAIN_FREQ` are reduced relative to their config for runtime,
  again identically across arms.
- `COST_BP > 0` puts a turnover penalty in the training objective. Left at 0 for
  the clean comparison; it is the modification most likely to matter, since
  turnover decided every net-of-cost result in this project.'''),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "DLSA_null_test_colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)

import ast
bad = 0
for i, c in enumerate(cells):
    if c["cell_type"] != "code": continue
    src = "".join(c["source"])
    t = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("!", "%")))
    try: ast.parse(t)
    except SyntaxError as e: print(f"cell {i}: {e}"); bad += 1
print(f"wrote {out}: {len(cells)} cells, {bad} syntax problems")
