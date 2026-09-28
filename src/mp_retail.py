"""Is the surviving signal tradeable, and at what account size?
All on survivorship-free residuals, 2017-01-01 .. 2026-09-18."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys
from real_ladder import windows_and_mask, sharpe, LOOKBACK
BORROW = 35.0

e = pd.read_parquet("cache/mp_eps_K5_N900.parquet").loc["2017-01-01":"2026-09-18"]
e = e.loc[:, e.notna().any()]
d = np.nan_to_num(e.values.astype(np.float32))
W, sel = windows_and_mask(d); last = W[:,:,-1]
def rev(L): return -(last-(W[:,:,-L-1] if L<LOOKBACK else 0.0))

def band(w, b=0.5):
    out=np.zeros_like(w); prev=np.zeros(w.shape[1])
    for t in range(len(w)):
        thr=b*np.abs(w[t]).mean()
        cur=np.where(np.abs(w[t]-prev)>thr, w[t], prev)
        s=np.abs(cur).sum(); cur=cur/s if s>0 else cur
        out[t]=cur; prev=cur
    return out

def topn(w,N):
    w=w.copy(); k=w.shape[1]-N
    idx=np.argpartition(np.abs(w),k,axis=1)[:,:k]
    np.put_along_axis(w,idx,0.0,axis=1)
    return w/np.abs(w).sum(1,keepdims=True).clip(1e-12)

def stats(w):
    g=np.array([float(w[i] @ d[LOOKBACK+i]) for i in range(len(w))])
    to=np.abs(np.diff(w,axis=0)).sum(1); to=np.r_[to[0],to]
    bo=np.abs(np.clip(w,None,0)).sum(1)*BORROW*1e-4/252
    lo,hi=0.,500.
    for _ in range(60):
        m=(lo+hi)/2
        if sharpe(g-m*1e-4*to)>0: lo=m
        else: hi=m
    return dict(gross=sharpe(g), turn=to.mean(), be=lo,
                net1=sharpe(g-1e-4*to-bo), net2=sharpe(g-2e-4*to-bo),
                mean1=(g-1e-4*to-bo).mean()*252*100,
                vol=g.std()*np.sqrt(252)*100)

base = np.where(sel, rev(30), 0.0)
base = base/np.abs(base).sum(1,keepdims=True).clip(1e-12)
print("reversal L=30 + no-trade band, 2017-2026, survivorship-free\n")
print("{:>6s} {:>10s} {:>7s} {:>6s} {:>7s} {:>8s} {:>8s} {:>9s}".format(
      "names","capital","gross","turn","b/e bp","net@1bp","net@2bp","%/yr @1bp"))
print("-"*70)
for N in (10,20,50,100,200,500,base.shape[1]):
    w = band(topn(base,N)) if N < base.shape[1] else band(base)
    s = stats(w)
    cap = f"${N*5_000/1e6:.2f}M" if N < base.shape[1] else "$4.5M+"
    print("{:>6d} {:>10s} {:+7.2f} {:6.2f} {:7.1f} {:+8.2f} {:+8.2f} {:+9.2f}".format(
          N, cap, s["gross"], s["turn"], s["be"], s["net1"], s["net2"], s["mean1"]))
w = band(topn(base,20)); s = stats(w)
print(f"\nN=20 detail:  vol {s['vol']:.1f}%/yr   net mean {s['mean1']:+.2f}%/yr")
print(f"  on $100k gross exposure: {s['mean1']*1000:+,.0f}/yr, vol ${s['vol']*1000:,.0f}")
print(f"  orders/day ~ {s['turn']*20/2:.0f}   (turnover {s['turn']:.2f} on 20 names)")
