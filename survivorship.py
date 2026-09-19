"""
Isolate the survivorship bias, measured on the authors' own survivorship-free data.

Their panel holds 2,434 stocks; only ~32% still trade at the end of the sample.
Restricting to end-of-sample survivors reproduces the bias in a universe built
from today's index membership -- which is exactly how our own panel was built.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, real_ladder as rl
from extend import rev_by_year

OURS = {5: 1.45, 30: 4.22}     # our panel, same years, same code

if __name__ == "__main__":
    d = rl.load("PCA-5")
    dates = pd.to_datetime(np.load("dlsa_real/dates.npy"))
    alive = (d != 0)
    survivors = alive[-252:].sum(0) > 200
    print(f"their universe: {alive.shape[1]:,} stocks")
    print(f"still active in the final year: {survivors.sum():,} ({survivors.mean():.0%})")
    print(f"died or left during the sample: {(~survivors).sum():,}")
    print("a universe built from TODAY's index membership contains none of these\n")
    ov = list(range(2007, 2017))
    rows = {}
    for lbl, sub in (("FULL (survivorship-free)", np.ones(d.shape[1], bool)),
                     ("SURVIVORS ONLY (our bias)", survivors)):
        rows[lbl] = {}
        for L in (5, 30):
            s = rev_by_year(d[:, sub], dates, L)
            rows[lbl][L] = s[[y for y in ov if y in s.index]].mean()
    print(f"{'universe (2007-2016)':34}{'L=5':>9}{'L=30':>9}")
    print("-"*54)
    for lbl, v in rows.items():
        print(f"{lbl:34}{v[5]:>9.2f}{v[30]:>9.2f}")
    print(f"{'ours, same years, same code':34}{OURS[5]:>9.2f}{OURS[30]:>9.2f}")
    f = rows["FULL (survivorship-free)"]; s = rows["SURVIVORS ONLY (our bias)"]
    print(f"\nbias multiplier from survivorship alone: "
          f"L=5 {s[5]/f[5]:.2f}x, L=30 {s[30]/f[30]:.2f}x")
    print(f"our L=5 ({OURS[5]:.2f}) sits on survivors-only ({s[5]:.2f}) -- fully explained")
    print(f"our L=30 ({OURS[30]:.2f}) is {OURS[30]/s[30]:.1f}x survivors-only "
          f"({s[30]:.2f}) -- additional contamination, unidentified")
