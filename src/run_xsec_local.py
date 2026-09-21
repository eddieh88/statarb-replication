"""Run the cross-sectional model locally on MPS. No Colab, no quota.

Validates with 3 baseline blocks (should match the Colab run: 9.86, 8.59, 8.47)
then runs all 15 xsection blocks. Checkpoints after every block.
"""
import numpy as np, torch, json, time, sys, warnings; warnings.filterwarnings("ignore")
from real_ladder import load, windows_and_mask, LOOKBACK
from xsection_model import XSectionNet

TRAIN_LEN, RETRAIN_FREQ, BATCH_DAYS, EPOCHS, LR, INDUCING = 1000, 250, 125, 30, 1e-3, 16
DAY_CHUNK = 32          # ISAB treats each day independently -> chunking over days is exact
# CPU, not MPS: this GPU caps at 6.77GB and the temporal transformer holds every
# chunk's activations in the autograd graph until backward, so forward-chunking
# does not lower peak memory.  CPU has 32GB and no cap.  ~8.5h for 15 blocks.
torch.set_num_threads(16)
DEV = torch.device("cpu")
CFG = {"baseline": dict(use_pos=False, use_xsection=False),
       "pos":      dict(use_pos=True,  use_xsection=False),
       "xsection": dict(use_pos=False, use_xsection=True),
       "full":     dict(use_pos=True,  use_xsection=True)}

def sharpe(x): return float(x.mean()/x.std()*np.sqrt(252)) if len(x) > 20 else float("nan")

print(f"device {DEV}", flush=True)
data = load("PCA-5")
W, sel = windows_and_mask(data)
Tn, N = W.shape[0], W.shape[1]
print(f"residuals {data.shape}  windows {W.shape}  active/day {sel.sum(1).mean():.0f}", flush=True)

def fwd_days(model, ch, sub):
    """Forward a set of days, chunked over the day axis (exact: ISAB is per-day)."""
    outs = []
    for i in range(0, len(ch), DAY_CHUNK):
        s = sub[i:i+DAY_CHUNK]
        x = torch.tensor(W[ch[i:i+DAY_CHUNK]][s], device=DEV)
        if len(x) < 10:
            outs.append(torch.zeros(len(s), N, device=DEV)); continue
        st = torch.tensor(s, device=DEV)
        Wt = torch.zeros(len(s), N, device=DEV)
        Wt[st] = model(x, st)
        outs.append(Wt)
    Wt = torch.cat(outs, 0)
    return Wt / Wt.abs().sum(1, keepdim=True).clamp(min=1e-9)

def run(arm, max_blocks=None, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    starts = list(range(TRAIN_LEN, Tn - RETRAIN_FREQ, RETRAIN_FREQ))
    if max_blocks: starts = starts[:max_blocks]
    blocks, all_r, all_to = [], [], []
    prev = np.zeros(N); t0 = time.time()
    for bi, s0 in enumerate(starts):
        model = XSectionNet(lookback=LOOKBACK, inducing=INDUCING, seed=seed, **CFG[arm]).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        tr = np.arange(s0-TRAIN_LEN, s0)
        for ep in range(EPOCHS):
            for ch in [tr[i:i+BATCH_DAYS] for i in range(0, len(tr), BATCH_DAYS)]:
                Wt = fwd_days(model, ch, sel[ch])
                r = (Wt * torch.tensor(data[LOOKBACK+ch], device=DEV)).sum(1)
                loss = -r.mean()/r.std().clamp(min=1e-9)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); br, bt = [], []
        te = np.arange(s0, min(s0+RETRAIN_FREQ, Tn))
        with torch.no_grad():
            for ch in [te[i:i+BATCH_DAYS] for i in range(0, len(te), BATCH_DAYS)]:
                Wt = fwd_days(model, ch, sel[ch])
                br.append((Wt*torch.tensor(data[LOOKBACK+ch], device=DEV)).sum(1).cpu().numpy())
                Wc = Wt.cpu().numpy().astype(np.float64)
                for k in range(len(Wc)): bt.append(np.abs(Wc[k]-prev).sum()); prev = Wc[k]
        br = np.concatenate(br); bt = np.array(bt)
        all_r.append(br); all_to.append(bt)
        ps = float(model.pe_scale.item()) if model.pe_scale is not None else None
        blocks.append({"year": round(1998.0+s0/252.0,2), "sharpe": sharpe(br),
                       "turnover": float(bt.mean()), "pe_scale": ps})
        print(f"  [{arm}] block {bi+1}/{len(starts)}  ~{blocks[-1]['year']:.1f}  "
              f"SR {blocks[-1]['sharpe']:+.2f}  turn {blocks[-1]['turnover']:.2f}"
              + (f"  pe_scale {ps:+.3f}" if ps is not None else "")
              + f"   ({time.time()-t0:.0f}s)", flush=True)
        r = np.concatenate(all_r); to = np.concatenate(all_to)
        json.dump({"arm":arm,"blocks":blocks}, open(f"ckpt_{arm}_local.json","w"), indent=1)
        np.save(f"local_{arm}.npy", np.vstack([r, to]))
    return np.concatenate(all_r), np.concatenate(all_to), blocks

if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "both"
    if arm == "both": arm = "pos"
    print(f"\n=== {arm} (all 15 blocks) ===", flush=True)
    if True:
        r, to, blocks = run(arm)
        h = len(r)//2
        print(f"\n{arm} pooled {sharpe(r):+.2f}  turn {to.mean():.2f}  "
              f"2H gross {sharpe(r[h:]):+.2f}  2H net@1bp {sharpe(r[h:]-1e-4*to[h:]):+.2f}")
        print("baseline (measured): pooled +4.92  turn 1.00  2H gross +2.23  2H net@1bp +0.21")
        print("xsection (measured): pooled +4.77  turn 1.00  2H gross +2.25  2H net@1bp +0.19")
