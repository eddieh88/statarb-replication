"""Minimal-capacity control: linear signal, same protocol, real vs null."""
import warnings; warnings.filterwarnings("ignore")
import json, time, numpy as np, torch, dlsa
t0=time.time()
def log(s): print(f"[{time.time()-t0:6.0f}s] {s}", flush=True)
R = dlsa.load_returns(n_names=300)
n = sum(p.numel() for p in dlsa.LinearSig().parameters())
log(f"LinearSig parameters: {n}  (CNN+Transformer: 1,049)")
log("REAL arm ...")
sr_real,_ = dlsa.fit_eval(dlsa.residuals(R), epochs=30, seed=0, arch=dlsa.LinearSig, dev="cpu")
log(f"REAL Sharpe = {sr_real:+.3f}")
rng = np.random.default_rng(0); nulls=[]
for d in range(5):
    Rn = dlsa.signflip(R, rng)     # same seed sequence as the CNN run
    s,_ = dlsa.fit_eval(dlsa.residuals(Rn), epochs=30, seed=100+d,
                        arch=dlsa.LinearSig, dev="cpu")
    nulls.append(s); log(f"null {d+1}/5 = {s:+.3f}")
nl=np.array(nulls)
print(f"\nLINEAR  real {sr_real:+.3f} | null mean {nl.mean():+.3f} sd {nl.std():.3f}"
      f" | excess {sr_real-nl.mean():+.3f}")
json.dump(dict(sr_real=sr_real,nulls=nl.tolist(),null_mean=float(nl.mean()),
               excess=float(sr_real-nl.mean()),params=n),
          open("linear_results.json","w"),indent=2)
