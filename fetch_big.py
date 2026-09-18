"""Pull a ~1500-name universe so the rank panel approaches the paper's top-500."""
import warnings; warnings.filterwarnings("ignore")
import json, time
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
C=Path("cache"); C.mkdir(exist_ok=True)
tick=json.load(open(C/"universe.json"))
out=C/"prices_big.parquet"
if out.exists():
    px=pd.read_parquet(out); print("cached:",px.shape)
else:
    frames=[]
    for i in range(0,len(tick),150):
        ch=tick[i:i+150]
        d=yf.download(ch,start="2006-01-01",auto_adjust=True,progress=False,threads=True)
        if isinstance(d.columns,pd.MultiIndex): d=d["Close"]
        frames.append(d); print(f"  {i+len(ch)}/{len(tick)} -> {d.shape[1]} cols",flush=True)
        time.sleep(1)
    px=pd.concat(frames,axis=1)
    px=px.loc[:,~px.columns.duplicated()]
    px.to_parquet(out); print("saved:",px.shape)
