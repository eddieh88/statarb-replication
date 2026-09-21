"""Pre-purchase gate for MarketParquet.  Sales are final, so this has to be
decisive on the free tier (API key, ~250 trading days from 2025-09-20).

We cannot test history DEPTH for free.  We CAN test the survivorship MECHANISM,
which is the thing that actually matters:

  GATE A  do names that delisted during the free year still appear in files
          dated BEFORE their delisting, with normal prices?
          -> if no, the archive is useless to us at any depth.
  GATE B  is the delisted count monotonically higher the further back you go?
  GATE C  does the whole pipeline run -- panel -> returns -> PCA -> residuals
          -> reversal -> costs -- on real files, with no schema surprises?
  GATE D  do splits in the past year look correctly back-adjusted?

Key: ~/.market_parquest/api_key.txt (never in the repo; pre-commit hook scans it).
"""
import os, sys, io, time
import requests, pandas as pd, numpy as np

KEY = os.path.expanduser("~/.market_parquest/api_key.txt")
URL = "https://marketparquet.com/api/data/download/stock_daily/{}.parquet"
CACHE = "cache/mp_free"

def hdrs():
    if os.path.exists(KEY):
        return {"Authorization": f"Bearer {open(KEY).read().strip()}"}
    print(f"! no key at {KEY} -- only the last ~2 days are open without one")
    return {}

def day(d, h):
    os.makedirs(CACHE, exist_ok=True)
    p = f"{CACHE}/{d}.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    r = requests.get(URL.format(d), headers=h, timeout=90)
    if r.status_code != 200:
        return None
    open(p, "wb").write(r.content)
    return pd.read_parquet(io.BytesIO(r.content))

def main():
    h = hdrs()
    days = [d.strftime("%Y-%m-%d") for d in
            pd.bdate_range("2025-09-22", pd.Timestamp.today().normalize())]
    frames, got = {}, []
    for d in days:
        f = day(d, h)
        if f is not None:
            frames[d] = f; got.append(d)
    print(f"downloaded {len(got)} trading days: {got[0] if got else '-'} .. {got[-1] if got else '-'}")
    if len(got) < 20:
        sys.exit("too few days -- add the API key before judging anything")

    for f in frames.values():
        f["sym"] = f["symbol"].astype(str)
        f["base"] = f["sym"].str.replace("-DELISTED", "", regex=False).str.upper()
        f["dead"] = f["sym"].str.contains("DELISTED")

    print("\n=== GATE B: delisted count by date (should rise going back) ===")
    ser = {d: int(frames[d]["dead"].sum()) for d in got}
    for d in got[::max(1, len(got)//12)]:
        print(f"  {d}  {ser[d]:5d} delisted of {len(frames[d]):5d}")

    print("\n=== GATE A: do delisted names have history BEFORE they died? ===")
    last = frames[got[-1]]
    dead_now = set(last.loc[last.dead, "base"])
    alive_first = set(frames[got[0]].loc[~frames[got[0]].dead, "base"])
    transitioned = sorted(dead_now & alive_first)
    print(f"  {len(transitioned)} names are alive on {got[0]} and delisted by {got[-1]}")
    for t in transitioned[:5]:
        hist = [(d, float(frames[d].loc[frames[d].base == t, "close"].iloc[0]))
                for d in got if (frames[d].base == t).any()]
        print(f"    {t:8s} {len(hist)} days, first {hist[0][1]:.2f} @ {hist[0][0]}, "
              f"last {hist[-1][1]:.2f} @ {hist[-1][0]}")
    if not transitioned:
        print("    NONE -- cannot confirm the mechanism.  DO NOT BUY on this evidence.")

    print("\n=== GATE C: full pipeline on real files ===")
    px = (pd.concat([f.assign(d=d)[["d", "base", "close"]] for d, f in frames.items()])
            .pivot_table(index="d", columns="base", values="close", aggfunc="last")
            .sort_index())
    print(f"  panel {px.shape}  NaN {px.isna().mean().mean():.1%}")
    dv = (pd.concat([f.assign(d=d)[["d","base","close","volume"]] for d,f in frames.items()])
            .assign(dv=lambda x: x.close*x.volume)
            .pivot_table(index="d", columns="base", values="dv", aggfunc="last"))
    top = dv.mean().nlargest(500).index
    r = px[top].pct_change(fill_method=None)
    r = r.mask(r.abs() > 0.5).dropna(axis=1, thresh=int(0.9*len(r))).fillna(0.0)
    X = r.values - r.values.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    F = U[:, :5]*S[:5]
    B = np.linalg.lstsq(F, X, rcond=None)[0]
    eps = X - F @ B
    print(f"  top-500 by dollar volume -> {r.shape[1]} usable names")
    print(f"  PCA-5 explains {1-eps.var()/X.var():.1%} of return variance")
    w = -eps[:-1]
    w = w/np.abs(w).sum(1, keepdims=True).clip(1e-12)
    ret = (w*eps[1:]).sum(1)
    to = np.abs(np.diff(w, axis=0)).sum(1)
    sr = ret.mean()/ret.std()*np.sqrt(252)
    print(f"  1-day residual reversal: Sharpe {sr:+.2f}, turnover {to.mean():.2f}  "
          f"(n={len(ret)} days -- NOT a result, an integration test)")

    print("\n=== GATE D: splits in the window ===")
    big = (px.pct_change(fill_method=None).abs() > 0.35).sum()
    print(f"  {int((big>0).sum())} names with a >35% one-day move (splits should be ABSENT if back-adjusted)")
    print("  worst:", px.pct_change(fill_method=None).abs().max().nlargest(5).round(2).to_dict())

    print("\nBUY only if: GATE A found transitioned names WITH prior history,")
    print("GATE B rises going back, GATE C runs clean, GATE D shows no split artefacts.")

if __name__ == "__main__":
    main()
