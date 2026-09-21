"""THE CRUX: does the Sharpe-3 artifact on no-signal data survive correct
normalisation in stock space?  Trivial reversal rule, no model, no training."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, dlsa2 as d

R = d.load_returns(300)
rng = np.random.default_rng(0)

def reversal_weights(E):
    """w = -(cumulative residual over last L days). No fitting whatsoever."""
    T, N = E.shape
    C = np.nan_to_num(E).cumsum(0)
    W = np.zeros((T, N))
    for t in range(d.L, T):
        W[t] = -(C[t] - C[t-d.L])
    return W

print(f"{'data':14}{'normalisation':16}{'SR gross':>10}{'SR net':>9}"
      f"{'ann mean':>10}{'cost/yr':>9}")
print("-"*70)
for lbl, RR in (("REAL", R), ("NULL", d.signflip(R, rng)), ("NULL", d.signflip(R, rng))):
    E, Wt, Bt = d.build(RR, K=5)
    Wm = reversal_weights(E)
    for norm in ("residual", "stock"):
        r = d.backtest(Wm, RR, E, Wt, Bt, norm=norm)
        print(f"{lbl:14}{norm:16}{r['sr_gross']:>10.2f}{r['sr_net']:>9.2f}"
              f"{r['mu_gross']:>9.1%}{r['cost_ann']:>9.1%}")
print("\nIf 'stock' normalisation kills the NULL Sharpe, the artifact was my")
print("residual-space normalisation (time-varying leverage). If it survives,")
print("residual reversal genuinely scores on noise.")
