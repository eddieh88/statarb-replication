import numpy as np, warnings; warnings.filterwarnings("ignore")
from real_ladder import load, windows_and_mask, sharpe, w_ou, LOOKBACK
S0=1000
data=load("PCA-5"); W,sel=windows_and_mask(data); Tn=W.shape[0]
last=W[:,:,-1]
def rev(L): return -(last-(W[:,:,-L-1] if L<LOOKBACK else 0.0))
def norm(Wt):
    w=np.where(sel,Wt,0.0); return w/np.abs(w).sum(1,keepdims=True).clip(1e-12)
ens=sum(a/np.abs(a).sum(1,keepdims=True).clip(1e-12) for a in (rev(5),rev(10),rev(20),rev(30)))

print("=== 1. mean / vol decomposition (annualised, per unit GROSS exposure) ===")
print(f"{'arm':20s} {'SR':>6s} {'mean%':>7s} {'vol%':>6s} {'turn':>6s} {'cost%@1bp':>10s} {'net mean%':>10s}")
for nm,Wt in (("reversal L=1",rev(1)),("reversal L=5",rev(5)),("reversal L=30",rev(30)),
              ("ensemble",ens),("OU+Threshold",w_ou(W,sel))):
    w=norm(Wt)
    g=(w[S0:]*data[LOOKBACK+S0:LOOKBACK+Tn]).sum(1)
    t=np.abs(np.diff(w[S0-1:],axis=0)).sum(1)
    m,v,tu = g.mean()*252*100, g.std()*np.sqrt(252)*100, t.mean()
    c = tu*1e-4*252*100
    print(f"{nm:20s} {sharpe(g):+6.2f} {m:+7.2f} {v:6.2f} {tu:6.2f} {c:10.2f} {m-c:+10.2f}")
print("\nCNN (from Colab): SR +4.92, turnover 1.00, breakeven 3.2bp")
print("  => mean = 3.2e-4*1.00*252*100 =", round(3.2e-4*1.00*252*100,2), "%/yr")
print("  => vol  = mean/SR             =", round(3.2e-4*1.00*252*100/4.92,2), "%/yr")

print("\n=== 2. is breakeven leverage-invariant? (it must be) ===")
w=norm(rev(30)); g=(w[S0:]*data[LOOKBACK+S0:LOOKBACK+Tn]).sum(1)
t=np.abs(np.diff(w[S0-1:],axis=0)).sum(1)
def be(g,t):
    lo,hi=0.,500.
    for _ in range(60):
        m=(lo+hi)/2
        if sharpe(g-m*1e-4*t)>0: lo=m
        else: hi=m
    return lo
for L in (1,2,5):
    print(f"  leverage {L}x -> breakeven {be(L*g, L*t):.3f} bp   (should be identical)")

print("\n=== 3. does the no-trade band survive top-N selection? ===")
def band(w,b=0.5):
    out=np.zeros_like(w); prev=np.zeros(w.shape[1])
    for tt in range(len(w)):
        thr=b*np.abs(w[tt]).mean()
        cur=np.where(np.abs(w[tt]-prev)>thr,w[tt],prev)
        s=np.abs(cur).sum(); cur=cur/s if s>0 else cur
        out[tt]=cur; prev=cur
    return out
def topn(w,N):
    w=w.copy(); k=w.shape[1]-N
    idx=np.argpartition(np.abs(w),k,axis=1)[:,:k]
    np.put_along_axis(w,idx,0.0,axis=1)
    return w/np.abs(w).sum(1,keepdims=True).clip(1e-12)
base=norm(ens)
for lab,w in (("band only", band(base)),
              ("band THEN top50 (what I ran)", topn(band(base),50)),
              ("top50 THEN band (correct)",    band(topn(base,50)))):
    t2=np.abs(np.diff(w[S0-1:],axis=0)).sum(1)
    g2=(w[S0:]*data[LOOKBACK+S0:LOOKBACK+Tn]).sum(1)
    h=len(g2)//2
    print(f"  {lab:32s} turn {t2.mean():.2f}  2ndhalf SR {sharpe(g2[h:]):+.2f}  "
          f"b/e {be(g2[h:],t2[h:]):.1f}bp  net@1bp {sharpe(g2[h:]-1e-4*t2[h:]):+.2f}")

print("\n=== 4. standard error of an annualised Sharpe ===")
for N,lab in ((3751,"full 2002-16"),(1875,"second half 2009-16")):
    print(f"  {lab:22s} N={N}  SE(SR) ~ {np.sqrt(252/N):.3f}")
