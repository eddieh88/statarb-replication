import numpy as np, warnings; warnings.filterwarnings("ignore")
from real_ladder import load, windows_and_mask, sharpe, w_ou, LOOKBACK
S0=1000
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
def topn(w,N):
    w=w.copy(); k=w.shape[1]-N
    idx=np.argpartition(np.abs(w),k,axis=1)[:,:k]
    np.put_along_axis(w,idx,0.0,axis=1)
    return w/np.abs(w).sum(1,keepdims=True).clip(1e-12)
def be(g,t):
    lo,hi=0.0,500.0
    for _ in range(60):
        m=(lo+hi)/2
        if sharpe(g-m*1e-4*t)>0: lo=m
        else: hi=m
    return lo

ens=sum(a/np.abs(a).sum(1,keepdims=True).clip(1e-12) for a in (rev(5),rev(10),rev(20),rev(30)))
ARMS={"reversal L=1":norm(rev(1)),"reversal L=5":norm(rev(5)),
      "reversal L=30":norm(rev(30)),"reversal L=30 + band":band(norm(rev(30))),
      "ensemble":norm(ens),"ensemble + band":band(norm(ens)),
      "ensemble + band, N=50":topn(band(norm(ens)),50),
      "ensemble + band, N=20":topn(band(norm(ens)),20),
      "OU+Threshold":norm(w_ou(W,sel))}

print(f"{'arm':26s} {'gross':>7s} {'turn':>6s} {'b/e bp':>7s} {'net@1bp':>8s} {'net@2bp':>8s}   [SECOND HALF 2009-16]")
print("-"*90)
rows=[]
for nm,w in ARMS.items():
    g=(w[S0:]*data[LOOKBACK+S0:LOOKBACK+Tn]).sum(1)
    t=np.abs(np.diff(w[S0-1:],axis=0)).sum(1)
    h=len(g)//2; g2,t2=g[h:],t[h:]
    rows.append((nm,sharpe(g2),t2.mean(),be(g2,t2),sharpe(g2-1e-4*t2),sharpe(g2-2e-4*t2)))
rows.append(("CNN+Transformer",2.23,0.96,1.1,0.21,-1.8))
for nm,gr,tu,b,n1,n2 in sorted(rows,key=lambda r:-r[4]):
    print(f"{nm:26s} {gr:+7.2f} {tu:6.2f} {b:7.1f} {n1:+8.2f} {n2:+8.2f}")
