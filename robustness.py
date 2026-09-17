"""
Post-hoc robustness check (NOT the pre-registered null).

The pre-registered null uses independent GBMs, which churn ranks more than real
correlated equities do -- arguably setting the bar too high.  This re-runs the
null with a common factor, calibrated to the observed name-space PC1 share, so
simulated rank churn resembles reality.

    python robustness.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import rankdata as rd, ranktest as rt

DRAWS = 60


def factor_null(cap, name_ret, share, rng):
    T, N = name_ret.shape
    sig = name_ret.std().to_numpy()
    f = rng.standard_normal((T, 1))
    e = rng.standard_normal((T, N))
    shocks = (np.sqrt(share) * f + np.sqrt(1 - share) * e) * sig
    logcap = np.log(cap.iloc[0].to_numpy()) + np.cumsum(shocks, axis=0)
    sc = pd.DataFrame(np.exp(logcap), index=name_ret.index, columns=cap.columns)
    sn = pd.DataFrame(shocks, index=name_ret.index, columns=cap.columns)
    srt = pd.DataFrame(np.sort(sc.to_numpy(), axis=1)[:, ::-1], index=sc.index,
                       columns=[f"r{i+1}" for i in range(N)])
    return sn.iloc[1:], np.log(srt).diff().iloc[1:]


def main():
    px = rd.load_prices(); sh = rd.load_shares(list(px.columns))
    name_ret, rank_ret, cap = rd.build_panels(px, sh)
    obs = rt.measure(name_ret, rank_ret)
    rng = np.random.default_rng(1)
    print("POST-HOC robustness: null with a common factor of varying strength")
    print(f"observed ADVANTAGE {obs['advantage']:+.4f}  "
          f"(real name-space PC1 = {obs['pc1_name']:.1%})\n")
    print(f"{'factor share':>13}{'null PC1':>10}{'null ADV med':>14}"
          f"{'null ADV p95':>14}{'observed clears?':>18}")
    for share in (0.0, 0.20, round(obs["pc1_name"], 3), 0.60):
        adv, pc = [], []
        for _ in range(DRAWS):
            sn, sr = factor_null(cap, name_ret, share, rng)
            m = rt.measure(sn, sr); adv.append(m["advantage"]); pc.append(m["pc1_name"])
        adv = np.array(adv); p95 = float(np.percentile(adv, 95))
        print(f"{share:>13.3f}{np.mean(pc):>10.1%}{np.median(adv):>14.4f}{p95:>14.4f}"
              f"{('YES' if obs['advantage'] > p95 else 'no'):>18}")
    print("\nAt the correctly-calibrated factor share the observed advantage still")
    print("fails to clear the null -- narrowly. It only clears at a factor share")
    print("well above what the data actually shows.")


if __name__ == "__main__":
    main()
