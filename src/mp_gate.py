"""THE GATE.  Does the MarketParquet-built panel reproduce 2002-2016?

We know the answers from the authors' published CRSP residuals.  Bands below were
fixed BEFORE this data existed -- they are wide because the universe rule differs
(dollar volume vs market cap) and MarketParquet ships no dividends.

If this fails, nothing the data says about 2017-2026 is worth anything.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np, pandas as pd
from real_ladder import windows_and_mask, port_returns, sharpe, w_ou, LOOKBACK

BANDS = {          # (known CRSP value, low, high)
    "OU+Threshold":  (0.70, 0.40, 1.10),
    "reversal L=30": (1.07, 0.70, 1.50),
    "reversal L=5":  (1.59, 1.10, 2.20),
}

def main(eps_file="cache/mp_eps_K5_N500.parquet", lo="2002-01-01", hi="2016-12-31"):
    eps = pd.read_parquet(eps_file).loc[lo:hi]
    eps = eps.loc[:, eps.notna().any()]
    d = np.nan_to_num(eps.values.astype(np.float32))      # 0 codes missing, as theirs does
    print(f"residuals {d.shape}  {eps.index[0].date()} .. {eps.index[-1].date()}")
    W, sel = windows_and_mask(d)
    last = W[:, :, -1]
    def rev(L): return -(last - (W[:, :, -L-1] if L < LOOKBACK else 0.0))
    arms = {"reversal L=1": rev(1), "reversal L=5": rev(5), "reversal L=10": rev(10),
            "reversal L=30": rev(30), "OU+Threshold": w_ou(W, sel)}
    arms["ensemble 5/10/20/30"] = sum(
        a / np.abs(a).sum(1, keepdims=True).clip(1e-12)
        for a in (rev(5), rev(10), rev(20), rev(30)))

    res, fails = {}, []
    print(f"\n{'arm':22s} {'ours':>7s} {'CRSP':>7s} {'band':>14s}  verdict")
    print("-" * 64)
    for nm, Wt in arms.items():
        r = port_returns(Wt, d, sel)
        s = sharpe(r); res[nm] = (s, r)
        if nm in BANDS:
            k, a, b = BANDS[nm]
            ok = a <= s <= b
            if not ok: fails.append(nm)
            print(f"{nm:22s} {s:+7.2f} {k:+7.2f} {f'[{a},{b}]':>14s}  {'PASS' if ok else 'FAIL'}")
        else:
            print(f"{nm:22s} {s:+7.2f} {'-':>7s} {'-':>14s}")

    order = [res[f"reversal L={L}"][0] for L in (5, 10, 30)]
    ok_order = order[0] > order[1] > order[2]
    print(f"\nordering L=5 > L=10 > L=30: {[round(x,2) for x in order]}  "
          f"{'PASS' if ok_order else 'FAIL'}")
    if not ok_order: fails.append("ordering")

    r = res["reversal L=30"][1]; h = len(r) // 2
    ratio = sharpe(r[h:]) / sharpe(r[:h]) if sharpe(r[:h]) else float("nan")
    ok_dec = 0.15 <= ratio <= 0.60
    print(f"decay second/first half (L=30): {ratio:.2f}  [0.15,0.60]  "
          f"{'PASS' if ok_dec else 'FAIL'}   (CRSP: 0.34)")
    if not ok_dec: fails.append("decay")

    print("\n" + "=" * 64)
    print("GATE PASSED -- proceed to 2017-2026" if not fails
          else f"GATE FAILED on {fails} -- diagnose before trusting 2017+")
    print("=" * 64)
    return 0 if not fails else 1

if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
