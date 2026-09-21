import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, yfinance as yf
C=Path("cache"); out=C/"shares_big.parquet"
px=pd.read_parquet(C/"prices_big.parquet")
px=px[px.index>="2006-01-01"].dropna(axis=1,thresh=int(len(px)*0.98))
names=list(px.columns); print(f"{len(names)} names to look up",flush=True)
def one(t):
    for _ in range(2):
        try:
            v=yf.Ticker(t).get_fast_info().get("shares")
            if v: return t,float(v)
            v=yf.Ticker(t).info.get("sharesOutstanding")
            if v: return t,float(v)
        except Exception: pass
    return t,None
res={}
with ThreadPoolExecutor(max_workers=12) as ex:
    for i,(t,v) in enumerate(ex.map(one,names)):
        if v: res[t]=v
        if (i+1)%200==0: print(f"  {i+1}/{len(names)}  ok={len(res)}",flush=True)
s=pd.Series(res,name="shares")
s.to_frame().to_parquet(out)
print(f"got shares for {len(s)}/{len(names)}")
