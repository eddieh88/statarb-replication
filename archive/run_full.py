"""Full ladder: real vs null, gross and net, all on the corrected pipeline."""
import warnings; warnings.filterwarnings("ignore")
import json, time, sys, numpy as np, dlsa2 as d, dlsa3 as m

t0=time.time()
def log(s): print(f"[{time.time()-t0:6.0f}s] {s}", flush=True)
EPOCHS = int(sys.argv[1]) if len(sys.argv)>1 else 20
NDRAW  = int(sys.argv[2]) if len(sys.argv)>2 else 3
R = d.load_returns(300)
log(f"{R.shape[1]} names, {len(R):,} days  epochs={EPOCHS} nulls={NDRAW}")

MODELS = [("reversal (no model)", None), ("OU+Threshold", None),
          ("Linear (31p)", m.Linear), ("FFN", m.FFN),
          ("Fourier+FFN", m.FourierFFN), ("CNN+Transformer", m.CNNTrans)]

def one_arm(RR, seed):
    E, Wt, Bt = d.build(RR, K=5)
    win, Xn, valid = m.panel(RR, E, Wt, Bt)
    out = {}
    for name, arch in MODELS:
        if name.startswith("reversal"): W = m.reversal(E)
        elif name.startswith("OU"):     W = m.ou_threshold(E)
        else: W = m.run_nn(arch, win, Xn, Wt, Bt, valid, epochs=EPOCHS, seed=seed)
        r = d.backtest(W, RR, E, Wt, Bt, norm="stock")
        out[name] = r; log(f"    {name:20} gross {r['sr_gross']:+.2f}  net {r['sr_net']:+.2f}")
    return out

log("REAL arm ..."); real = one_arm(R, 0)
rng = np.random.default_rng(0); nulls = []
for i in range(NDRAW):
    log(f"NULL {i+1}/{NDRAW} ...")
    nulls.append(one_arm(d.signflip(R, rng), 100+i))

print("\n" + "="*84)
print(f"{'model':22}{'REAL gross':>12}{'NULL gross':>12}{'excess':>9}"
      f"{'REAL net':>11}{'NULL net':>11}{'excess':>9}")
print("-"*84)
res={}
for name,_ in MODELS:
    ng = np.mean([n[name]['sr_gross'] for n in nulls])
    nn_ = np.mean([n[name]['sr_net'] for n in nulls])
    rg, rn = real[name]['sr_gross'], real[name]['sr_net']
    print(f"{name:22}{rg:>12.2f}{ng:>12.2f}{rg-ng:>9.2f}{rn:>11.2f}{nn_:>11.2f}{rn-nn_:>9.2f}")
    res[name]=dict(real_gross=rg,null_gross=float(ng),excess_gross=float(rg-ng),
                   real_net=rn,null_net=float(nn_),excess_net=float(rn-nn_))
print("="*84)
print("Paper Table I, PCA-5 gross: OU+Thresh 0.73 | Fourier+FFN 1.98 | CNN+Trans 3.36")
json.dump(res, open("full_ladder_results.json","w"), indent=2)
