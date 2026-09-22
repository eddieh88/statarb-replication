"""Export our survivorship-free residuals in the authors' array format so the
GPU notebook can train on them unchanged.

Theirs: dense float32 (days x stocks), 0 codes missing, masked to columns that
are ever active.  We match that exactly.

Default window starts 2012 so there are >=1000 training days before the first
2017 test block, while keeping the file small enough to upload to Colab.
"""
import sys
import numpy as np, pandas as pd

def main(lo="2012-01-01", hi="2026-09-18", src="cache/mp_eps_K5_N900.parquet",
         out="cache/mp_eps_cnn"):
    e = pd.read_parquet(src).loc[lo:hi]
    e = e.loc[:, e.notna().any()]
    d = np.nan_to_num(e.values.astype(np.float32))
    np.save(f"{out}.npy", d)
    np.save(f"{out}_dates.npy", e.index.values)
    # single compressed file for the Colab upload: ~13 MB vs 51, lossless
    np.savez_compressed(f"{out}.npz", eps=d, dates=e.index.values)
    mb = d.nbytes/1e6
    print(f"{out}.npy  {d.shape}  {mb:.0f} MB")
    print(f"  {e.index[0].date()} .. {e.index[-1].date()}")
    print(f"  active names/day: median {int(np.median((d!=0).sum(1)))}")
    print(f"  columns ever active: {d.shape[1]}")
    tr = (e.index < '2017-01-01').sum()
    print(f"  training days before 2017-01-01: {tr}  "
          f"({'enough' if tr >= 1000 else 'NOT ENOUGH -- start earlier'})")

if __name__ == "__main__":
    main(*sys.argv[1:])
