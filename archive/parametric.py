"""
Their parametric benchmark (App. E, eqs E.3/E.4, L1636-1660) run on real vs null
rank residuals.

  x_{i,t} = sum_{j=1..L} eps_{i,t-L+j}          (E.3, L1638-1640)
  s-score = (x - mean)/sd on a trailing window
  open  short if s > +c_open, long if s < -c_open      (E.4)
  close when |s| < c_close
  trade only if estimated mean-reversion time tau < 30 days   (L1647-1652)

If sorted NOISE reproduces their Table 1 gross Sharpe (rank parametric SR 6.14,
L1841), the parametric edge is the order-statistic effect and nothing else.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, replicate as rp

L, WIN, C_OPEN, C_CLOSE, TAU_MAX = 60, 252, 1.25, 0.50, 30


def s_score_and_tau(E, L=L, win=WIN):
    X = pd.DataFrame(np.nan_to_num(E)).rolling(L).sum()
    mu = X.rolling(win).mean(); sd = X.rolling(win).std()
    s = ((X - mu) / sd).to_numpy()
    Xv = X.to_numpy()
    a, b = Xv[1:], Xv[:-1]
    num = pd.DataFrame(a * b).rolling(win).sum().to_numpy()
    den = pd.DataFrame(b * b).rolling(win).sum().to_numpy()
    with np.errstate(all="ignore"):
        rho = np.clip(num / den, 1e-6, 0.999999)
        tau = -1.0 / np.log(rho)
    tau = np.vstack([np.full((1, E.shape[1]), np.nan), tau])
    return s, tau


def run_parametric(E):
    """E = T x N residual matrix. Returns annualised gross Sharpe, no costs."""
    s, tau = s_score_and_tau(E)
    T, N = s.shape
    pos = np.zeros((T, N))
    ok = np.isfinite(s) & np.isfinite(tau) & (tau < TAU_MAX)
    prev = np.zeros(N)
    for t in range(1, T):
        cur = prev.copy()
        st, okt = s[t], ok[t]
        cur[(prev == 0) & okt & (st > C_OPEN)] = -1
        cur[(prev == 0) & okt & (st < -C_OPEN)] = 1
        cur[(prev != 0) & (np.abs(st) < C_CLOSE)] = 0
        cur[~np.isfinite(st)] = 0
        pos[t] = cur; prev = cur
    r = np.nan_to_num(E)
    pnl = (pos[:-1] * r[1:])
    n = np.maximum(np.abs(pos[:-1]).sum(1, keepdims=True), 1.0)
    daily = (pnl / n).sum(1)
    daily = daily[np.isfinite(daily)]
    live = daily[WIN + L:]
    return float(live.mean() / live.std() * np.sqrt(252)), float(np.abs(pos).sum(1).mean())


if __name__ == "__main__":
    px = pd.read_parquet("cache/prices_big.parquet")
    px = px[px.index >= "2006-01-01"].dropna(axis=1, thresh=int(len(px)*0.98)).ffill().dropna()
    sh = pd.read_parquet("cache/shares_big.parquet")["shares"]
    cols = [c for c in px.columns if c in sh.index]
    cap = px[cols] * sh.reindex(cols)

    print("Their parametric benchmark (App. E), GROSS of costs.")
    print("Paper Table 1 (L1841): rank parametric SR 6.14, name parametric SR 0.96\n")
    er = rp.residuals_from_caps(cap, "rank"); en = rp.residuals_from_caps(cap, "name")
    sr_r, np_r = run_parametric(er); sr_n, np_n = run_parametric(en)
    print(f"{'':28}{'gross SR':>10}{'avg positions':>15}")
    print(f"{'OBSERVED rank space':28}{sr_r:>10.2f}{np_r:>15.0f}")
    print(f"{'OBSERVED name space':28}{sr_n:>10.2f}{np_n:>15.0f}")

    rng = np.random.default_rng(11)
    nr, nn = [], []
    for i in range(8):
        sc = rp.simulate_caps(cap, rng, 0.40)
        nr.append(run_parametric(rp.residuals_from_caps(sc, "rank"))[0])
        nn.append(run_parametric(rp.residuals_from_caps(sc, "name"))[0])
    print(f"{'NULL rank space (8 draws)':28}{np.mean(nr):>10.2f}"
          f"   [{min(nr):.2f}, {max(nr):.2f}]")
    print(f"{'NULL name space (8 draws)':28}{np.mean(nn):>10.2f}"
          f"   [{min(nn):.2f}, {max(nn):.2f}]")
    print(f"\nrank-space excess over null: {sr_r-np.mean(nr):+.2f} Sharpe")
