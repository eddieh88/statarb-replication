"""Add stock-borrow costs. General-collateral large-cap borrow ~25-50bp/yr on
SHORT notional.  With a balanced book at gross 1 the short leg is ~0.5, so the
drag is ~12-25bp/yr on gross.  It is a fixed subtraction from the mean, so it
bites low-mean / low-vol strategies hardest -- exactly the CNN's profile."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from real_ladder import load, windows_and_mask, sharpe, w_ou, LOOKBACK
S0=1000; BORROW_BP_YR=35.0
data=load("PCA-5"); W,sel=windows_and_mask(data); Tn=W.shape[0]
last=W[:,:,-1]
def rev(L): return -(last-(W[:,:,-L-1] if L<LOOKBACK else 0.0))
def norm(Wt):
    w=np.where(sel,Wt,0.0); return w/np.abs(w).sum(1,keepdims=True).clip(1e-12)
def band(w,b=0.5):
    out=np.zeros_like(w); prev=np.zeros(w.shape[1])
    for t in range(len(w)):
        thr=b*np.abs(w[t]).mean()
        cur=np.where(np.abs(w[t]-prev)>thr,w[t],prev)
        s=np.abs(cur).sum(); cur=cur/s if s>0 else cur
        out[t]=cur; prev=cur
    return out
ens=sum(a/np.abs(a).sum(1,keepdims=True).clip(1e-12) for a in (rev(5),rev(10),rev(20),rev(30)))
ARMS={"reversal L=1":norm(rev(1)),"reversal L=30":norm(rev(30)),
      "reversal L=30 + band":band(norm(rev(30))),"ensemble":norm(ens),
      "ensemble + band":band(norm(ens)),"OU+Threshold":norm(w_ou(W,sel))}

print(f"borrow {BORROW_BP_YR:.0f}bp/yr on short notional\n")
print(f"{'arm':24s} {'short%':>7s} {'borrow%/yr':>11s} {'2H net@1bp':>11s} {'+borrow':>9s} {'delta':>7s}")
print("-"*76)
for nm,w in ARMS.items():
    g=(w[S0:]*data[LOOKBACK+S0:LOOKBACK+Tn]).sum(1)
    t=np.abs(np.diff(w[S0-1:],axis=0)).sum(1)
    shortfrac=np.abs(np.clip(w[S0:],None,0)).sum(1)          # short notional per day
    bday=shortfrac*BORROW_BP_YR*1e-4/252
    h=len(g)//2
    a=sharpe(g[h:]-1e-4*t[h:]); b=sharpe(g[h:]-1e-4*t[h:]-bday[h:])
    print(f"{nm:24s} {shortfrac.mean()*100:6.1f}% {shortfrac.mean()*BORROW_BP_YR/100:10.2f}% "
          f"{a:+11.2f} {b:+9.2f} {b-a:+7.2f}")
# CNN: mean 2.66%/yr gross second half, vol 1.19%, turnover 0.96, short ~0.5
m=2.66-0.96*1e-4*252*100; v=1.19
print(f"\n{'CNN+Transformer':24s}  ~50.0%      0.18%  {m/v:+11.2f} {(m-0.175)/v:+9.2f} {((m-0.175)-m)/v:+7.2f}")
