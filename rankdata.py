"""
Build name-space and rank-space daily return panels.

Rank-slot returns are returns on ORDER STATISTICS of capitalisation, not on
names.  See PREREGISTRATION.md -- this is the whole reason a synthetic null is
required.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

CACHE = Path(__file__).with_name("cache"); CACHE.mkdir(exist_ok=True)
START = "2008-01-01"

TICKERS = ("AAPL MSFT GOOGL AMZN META NVDA TSLA JPM JNJ V PG UNH HD MA DIS BAC ADBE CRM "
 "NFLX XOM CVX PFE KO PEP TMO ABT CSCO ACN AVGO WMT MRK COST LLY DHR NKE TXN NEE ORCL "
 "PM INTC WFC UPS MS RTX HON QCOM LOW UNP AMD BA CAT GS BLK SBUX MMM AXP DE LMT SPGI "
 "GILD ISRG TJX MDLZ ADI SYK VRTX ZTS CI CB MO SO DUK PLD CL BDX ITW EOG APD SHW AON "
 "ICE CME NSC EMR FDX PSA ECL ROP CTAS MCD IBM GE F GM DAL LUV CCL T VZ CMCSA CVS HUM "
 "KR TGT DG DLTR ROST ORLY AZO YUM CMG DPZ MAR HLT AIG MET PRU ALL TRV PGR AFL STT BK "
 "SCHW USB PNC TFC COF DFS SYF FITB KEY RF CFG HBAN MTB NTRS ZION CMA WY NUE STLD X "
 "FCX NEM MOS CF DOW DD LYB PPG IP PKG BLL AMCR SEE AVY EXPD CHRW JBHT ODFL LSTR R "
 "CSX KSU GWW FAST SWK SNA PH ETN ROK DOV XYL FTV AME TT JCI CARR OTIS IR CMI PCAR "
 "WAB URI PWR MAS MHK LEN DHI PHM NVR TOL KBH").split()


def load_prices(refresh: bool = False) -> pd.DataFrame:
    p = CACHE / "prices.parquet"
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    px = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]
    px = px.dropna(axis=1, thresh=int(len(px) * 0.98)).ffill().dropna()
    px.to_parquet(p)
    return px


def load_shares(tickers, refresh: bool = False) -> pd.Series:
    """CURRENT shares outstanding -- static, a known bias.  See prereg."""
    p = CACHE / "shares.parquet"
    if p.exists() and not refresh:
        s = pd.read_parquet(p)["shares"]
        if set(tickers) <= set(s.index):
            return s.reindex(tickers)
    out = {}
    for t in tickers:
        try:
            out[t] = yf.Ticker(t).info.get("sharesOutstanding")
        except Exception:
            out[t] = None
    s = pd.Series({k: v for k, v in out.items() if v}, name="shares")
    s.to_frame().to_parquet(p)
    return s


def build_panels(px: pd.DataFrame, shares: pd.Series):
    """Returns (name_returns, rank_returns, caps).  Both panels are daily log
    returns; rank column k is the log change in the cap occupying rank k."""
    cols = [c for c in px.columns if c in shares.index]
    cap = px[cols] * shares.reindex(cols)
    name_ret = np.log(px[cols]).diff().iloc[1:]
    sorted_cap = pd.DataFrame(
        np.sort(cap.to_numpy(), axis=1)[:, ::-1],       # descending: rank 1 = largest
        index=cap.index, columns=[f"r{i+1}" for i in range(len(cols))])
    rank_ret = np.log(sorted_cap).diff().iloc[1:]
    return name_ret, rank_ret, cap


def simulate_null(cap: pd.DataFrame, name_ret: pd.DataFrame, rng):
    """Independent GBMs, per-name realised vol, matched starting caps.
    ZERO cross-sectional structure and ZERO mean reversion by construction."""
    T, N = name_ret.shape
    sig = name_ret.std().to_numpy()
    shocks = rng.standard_normal((T, N)) * sig                 # independent
    logcap = np.log(cap.iloc[0].to_numpy()) + np.cumsum(shocks, axis=0)
    sim_cap = pd.DataFrame(np.exp(logcap), index=name_ret.index, columns=cap.columns)
    sim_name = pd.DataFrame(shocks, index=name_ret.index, columns=cap.columns)
    sorted_cap = pd.DataFrame(np.sort(sim_cap.to_numpy(), axis=1)[:, ::-1],
                              index=sim_cap.index, columns=[f"r{i+1}" for i in range(N)])
    sim_rank = np.log(sorted_cap).diff().iloc[1:]
    return sim_name.iloc[1:], sim_rank
