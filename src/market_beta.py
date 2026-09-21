"""Market exposure of each arm, on the authors' PCA-5 residuals.
Regress daily strategy returns on Ken French's daily Mkt-RF (and SMB/HML)."""
import numpy as np, pandas as pd, io, zipfile, urllib.request, warnings
warnings.filterwarnings("ignore")
from real_ladder import load, windows_and_mask, port_returns, sharpe, w_ou, LOOKBACK

U="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(U, timeout=60).read()))
raw=z.read(z.namelist()[0]).decode("latin-1").splitlines()
rows=[l for l in raw if len(l.split(','))==5 and l.split(',')[0].strip().isdigit()]
ff=pd.DataFrame([l.split(',') for l in rows], columns=["date","MktRF","SMB","HML","RF"])
ff["date"]=pd.to_datetime(ff.date.str.strip(), format="%Y%m%d")
for c in ["MktRF","SMB","HML","RF"]: ff[c]=ff[c].astype(float)/100.0
ff=ff.set_index("date")
print(f"Fama-French daily {ff.index.min().date()} to {ff.index.max().date()}")

data=load("PCA-5"); W,sel=windows_and_mask(data)
dates=pd.DatetimeIndex(np.load("dlsa_real/dates.npy", allow_pickle=True))
S0=1000
last=W[:,:,-1]
def rev(L): return -(last-(W[:,:,-L-1] if L<LOOKBACK else 0.0))
ARMS={"reversal L=1":rev(1),"reversal L=5":rev(5),"reversal L=30":rev(30),
      "OU+Threshold":w_ou(W,sel),
      "ensemble 5/10/20/30":sum(a/np.abs(a).sum(1,keepdims=True).clip(1e-12)
                                for a in (rev(5),rev(10),rev(20),rev(30)))}

print(f"\n{'arm':22s} {'corr(Mkt)':>10s} {'beta':>8s} {'t(beta)':>8s} {'alpha %/yr':>11s} {'R2':>6s}")
print("-"*70)
for nm,Wt in ARMS.items():
    keep=[]; r=[]
    for i in range(Wt.shape[0]):
        w=np.where(sel[i],Wt[i],0.0); s=np.abs(w).sum()
        if s>0: r.append(float(w@data[LOOKBACK+i])/s); keep.append(LOOKBACK+i)
    r=np.array(r); kd=dates[np.array(keep)]
    df=pd.DataFrame({"r":r},index=kd).join(ff[["MktRF","SMB","HML"]],how="inner")
    df=df.iloc[S0:]
    x=df.MktRF.values; y=df.r.values
    b=np.cov(x,y)[0,1]/x.var(); a=y.mean()-b*x.mean()
    res=y-a-b*x; se=res.std()/ (x.std()*np.sqrt(len(x)))
    c=np.corrcoef(x,y)[0,1]
    print(f"{nm:22s} {c:+10.3f} {b:+8.3f} {b/se:+8.2f} {a*252*100:+11.2f} {c**2:6.3f}")
print("\nCNN+Transformer: needs its daily return series exported from Colab.")
