"""Does the match to the null hold across the whole autocorrelation function,
not just lag 1?  If it does, any statistic derived from these residuals --
including the paper's OU fit to CUMULATIVE residuals -- must match too."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, replicate as rp

px=pd.read_parquet("cache/prices_big.parquet")
px=px[px.index>="2006-01-01"].dropna(axis=1,thresh=int(len(px)*0.98)).ffill().dropna()
sh=pd.read_parquet("cache/shares_big.parquet")["shares"]
cols=[c for c in px.columns if c in sh.index]; cap=px[cols]*sh.reindex(cols)

def acf(E,L=10):
    E=np.where(np.isfinite(E),E,np.nan); out=[]
    for k in range(1,L+1):
        a,b=E[k:],E[:-k]; m=np.isfinite(a)&np.isfinite(b)
        out.append((a[m]*b[m]).sum()/(b[m]*b[m]).sum())
    return np.array(out)

def cumOU(E,L=60):
    """OU half-life of the rolling L-day cumulative residual -- the paper's tau."""
    E=np.nan_to_num(E); X=pd.DataFrame(E).rolling(L).sum().to_numpy()[L:]
    a,b=X[1:],X[:-1]; m=np.isfinite(a)&np.isfinite(b)
    ar=(a[m]*b[m]).sum()/(b[m]*b[m]).sum()
    return -np.log(2)/np.log(ar) if 0<ar<1 else np.inf

obs_r=rp.residuals_from_caps(cap,"rank"); obs_n=rp.residuals_from_caps(cap,"name")
a_or,a_on=acf(obs_r),acf(obs_n)
rng=np.random.default_rng(3); NR,NN=[],[]
for _ in range(8):
    sc=rp.simulate_caps(cap,rng,0.40)
    NR.append(acf(rp.residuals_from_caps(sc,"rank")))
    NN.append(acf(rp.residuals_from_caps(sc,"name")))
NR,NN=np.array(NR),np.array(NN)
print("AUTOCORRELATION BY LAG -- observed vs null (8 draws)\n")
print(f"{'lag':>4}{'rank obs':>11}{'rank null':>11}{'diff':>9}   |"
      f"{'name obs':>11}{'name null':>11}{'diff':>9}")
for k in range(10):
    print(f"{k+1:>4}{a_or[k]:>11.4f}{NR[:,k].mean():>11.4f}{a_or[k]-NR[:,k].mean():>9.4f}   |"
          f"{a_on[k]:>11.4f}{NN[:,k].mean():>11.4f}{a_on[k]-NN[:,k].mean():>9.4f}")
print(f"\nPAPER'S STATISTIC -- OU half-life of 60d CUMULATIVE residuals:")
print(f"{'':22}{'observed':>11}{'null mean':>12}")
print(f"{'rank space':22}{cumOU(obs_r):>11.2f}{np.mean([cumOU(rp.residuals_from_caps(rp.simulate_caps(cap,rng,0.40),'rank')) for _ in range(4)]):>12.2f}")
print(f"{'name space':22}{cumOU(obs_n):>11.2f}{np.mean([cumOU(rp.residuals_from_caps(rp.simulate_caps(cap,rng,0.40),'name')) for _ in range(4)]):>12.2f}")
