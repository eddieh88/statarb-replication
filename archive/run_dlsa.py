"""Phase 0: real vs sign-flipped null through identical code. See PREREG_DLSA.md."""
import warnings; warnings.filterwarnings("ignore")
import json, time, sys
import numpy as np
import dlsa

t0 = time.time()
def log(s): print(f"[{time.time()-t0:6.0f}s] {s}", flush=True)

R = dlsa.load_returns(n_names=300)
log(f"{R.shape[1]} names, {len(R):,} days, {R.index.min():%Y-%m} to {R.index.max():%Y-%m}")
log(f"device={dlsa.DEV}  K_PCA={dlsa.K_PCA}  L={dlsa.L}  train={dlsa.TRAIN} test={dlsa.TEST}")

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
NDRAWS = int(sys.argv[2]) if len(sys.argv) > 2 else 5

log("REAL arm ...")
sr_real, r_real = dlsa.fit_eval(dlsa.residuals(R), epochs=EPOCHS, seed=0, log=log)
log(f"REAL out-of-sample Sharpe = {sr_real:.3f}  ({len(r_real)} days)")

rng = np.random.default_rng(0)
nulls = []
for d in range(NDRAWS):
    log(f"NULL draw {d+1}/{NDRAWS} ...")
    Rn = dlsa.signflip(R, rng)
    s, _ = dlsa.fit_eval(dlsa.residuals(Rn), epochs=EPOCHS, seed=100+d)
    nulls.append(s); log(f"  null Sharpe = {s:+.3f}")
nl = np.array(nulls)

print("\n" + "="*62)
print(f"REAL out-of-sample Sharpe   {sr_real:+.3f}")
print(f"NULL mean                   {nl.mean():+.3f}   sd {nl.std():.3f}")
print(f"NULL draws                  {np.round(nl,3).tolist()}")
print(f"NULL max                    {nl.max():+.3f}")
clean = abs(nl.mean()) <= 0.3 and sr_real > nl.max()
leak  = nl.mean() > 0.5
print("="*62)
print("VERDICT:", "PASS - pipeline clean" if clean else
      ("FAIL - pipeline leaks" if leak else "INCONCLUSIVE vs pre-registered thresholds"))
json.dump(dict(sr_real=sr_real, nulls=nl.tolist(), null_mean=float(nl.mean()),
               null_sd=float(nl.std()), epochs=EPOCHS, n_names=int(R.shape[1]),
               clean=bool(clean), leak=bool(leak)),
          open("dlsa_phase0_results.json","w"), indent=2)
print("wrote dlsa_phase0_results.json")
