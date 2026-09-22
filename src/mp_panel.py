"""Build a survivorship-free price/return panel from the MarketParquet archive.

The -DELISTED suffix is applied RETROACTIVELY across the whole archive, so a name
that died in 2015 already carries it in 2005 files.  It therefore encodes the
future and must never be used as a filter: we strip it and keep every symbol
present on each date.  That is what makes the panel point-in-time.

Writes:
  cache/mp_close.parquet   days x symbols, split-adjusted close (NOT dividend-adjusted)
  cache/mp_dv.parquet      days x symbols, dollar volume (close * volume)
"""
import glob, os, sys
import numpy as np, pandas as pd

SRC, OUT = "cache/mp", "cache"

def main(limit=None):
    files = sorted(glob.glob(f"{SRC}/stock_daily_*.parquet"))
    if limit: files = files[:int(limit)]
    print(f"{len(files)} daily files, {files[0][-18:-8]} .. {files[-1][-18:-8]}")
    close, dv = {}, {}
    for i, p in enumerate(files):
        # archive files use `timestamp`; the free sample used `date`
        f = pd.read_parquet(p)
        dcol = "timestamp" if "timestamp" in f.columns else "date"
        # strip the retroactive tag; it is future information
        s = f["symbol"].astype(str).str.replace("-DELISTED", "", regex=False).str.upper()
        d = str(f[dcol].iloc[0])[:10]
        c = f["close"].astype("float32").values
        close[d] = pd.Series(c, index=s).groupby(level=0).last()
        dv[d]    = pd.Series((c * f["volume"].astype("float64").values),
                             index=s).groupby(level=0).last()
        if i % 500 == 0:
            print(f"  {i:5d}/{len(files)}  {d}  {len(s):5d} symbols", flush=True)
    px = pd.DataFrame(close).T.sort_index(); px.index = pd.to_datetime(px.index)
    vv = pd.DataFrame(dv).T.sort_index();    vv.index = pd.to_datetime(vv.index)
    # The vendor emits a file for US market holidays containing ~1 symbol.
    # 115 such days in 2000-2026, mostly Mondays (MLK/Presidents/Memorial/Labor)
    # and Thursdays (Thanksgiving, July 4).  Drop any day with fewer than 20% of
    # the trailing-median name count -- real sessions never come close.
    cnt = px.notna().sum(1)
    thr = cnt.rolling(250, min_periods=20).median() * 0.20
    keep = cnt >= thr.fillna(cnt.median() * 0.20)
    if (~keep).sum():
        print(f"  dropping {(~keep).sum()} non-session days "
              f"(holiday files with ~1 symbol), e.g. "
              f"{[str(d.date()) for d in px.index[~keep][:3]]}")
    px, vv = px[keep], vv[keep]
    px = px.astype("float32"); vv = vv.astype("float32")
    px.to_parquet(f"{OUT}/mp_close.parquet")
    vv.to_parquet(f"{OUT}/mp_dv.parquet")
    print(f"\npanel {px.shape}  {px.index[0].date()} .. {px.index[-1].date()}")
    print(f"  symbols ever seen : {px.shape[1]:,}")
    print(f"  median names/day  : {px.notna().sum(1).median():.0f}")
    print(f"  NaN fraction      : {px.isna().mean().mean():.1%}  "
          f"(expected: most names do not exist for most of the sample)")
    r = px.pct_change(fill_method=None)
    print(f"  |ret|>35% share   : {(r.abs()>0.35).sum().sum()/r.notna().sum().sum():.3%}")
    print(f"  wrote {OUT}/mp_close.parquet, {OUT}/mp_dv.parquet")

if __name__ == "__main__":
    main(*sys.argv[1:])
