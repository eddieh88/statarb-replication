"""One-shot data setup.  Run this first; everything else depends on it.

Downloads the authors' published out-of-sample residuals from
github.com/gregzanotti/dlsa-public, applies their superMask, and writes the
compact float32 arrays the harness reads.

  dlsa_real/PCA-5_masked.npy    (4781 x 2434)  ~44 MB
  dlsa_real/IPCA-5_masked.npy                  ~44 MB
  dlsa_real/superMask.npy, dates.npy

Downloads ~40 MB gzipped per factor model; the intermediate uncompressed array
is ~346 MB and is deleted afterwards.  Total runtime a few minutes.
"""
import os, gzip, shutil, urllib.request, sys
import numpy as np

RAW = "https://raw.githubusercontent.com/gregzanotti/dlsa-public/main/residuals"
OUT = "dlsa_real"
FILES = {
    "PCA-5":  ("pca",  "AvPCA_OOSresiduals_5_factors_1998_initialOOSYear_60_"
                       "rollingWindow_252_covWindow_0.01_Cap.npy"),
    "IPCA-5": ("ipca_normalized", "IPCA_DailyOOSresiduals_5_factors_420_"
                       "initialMonths_240_window_12_reestimationFreq_0.01_cap.npy"),
}

def fetch(url, dst):
    if os.path.exists(dst):
        print(f"  have {os.path.basename(dst)}"); return
    print(f"  downloading {os.path.basename(dst)} ...", flush=True)
    urllib.request.urlretrieve(url, dst)

def main(which=("PCA-5",)):
    os.makedirs(OUT, exist_ok=True)
    fetch(f"{RAW}/superMask.npy", f"{OUT}/superMask.npy")
    mask = np.load(f"{OUT}/superMask.npy")
    print(f"superMask: {mask.sum()} of {len(mask)} stocks retained")

    for tag in which:
        sub, name = FILES[tag]
        out = f"{OUT}/{tag}_masked.npy"
        if os.path.exists(out):
            print(f"{tag}: already built"); continue
        gz, raw = f"{OUT}/{name}.gz", f"{OUT}/{name}"
        fetch(f"{RAW}/{sub}/{name}.gz", gz)
        print(f"  decompressing (~346 MB) ...", flush=True)
        with gzip.open(gz, "rb") as fi, open(raw, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        r = np.load(raw)
        np.save(out, np.asarray(r[:, mask], dtype=np.float32))
        print(f"{tag}: {r.shape} -> {r[:, mask].shape}  saved {out}")
        del r
        os.remove(raw); os.remove(gz)          # keep only the compact array

    # Trading-day index.  Business days are NOT trading days -- ~9 market
    # holidays a year means 4781 trading days span more calendar time than 4781
    # business days, and a bdate_range approximation lands 8 months short.  The
    # real NYSE calendar gives exactly 4781 days for 1998-01-02..2016-12-30,
    # matching the residual rows, which also confirms the file's date span.
    dp = f"{OUT}/dates.npy"
    if not os.path.exists(dp):
        import pandas as pd
        n = np.load(f"{OUT}/PCA-5_masked.npy", mmap_mode="r").shape[0]
        try:
            import pandas_market_calendars as mcal
            d = mcal.get_calendar("NYSE").valid_days("1998-01-02", "2016-12-30").tz_localize(None)
            assert len(d) == n, f"calendar {len(d)} != residual rows {n}"
            print(f"dates: {n} NYSE trading days, {d[0].date()} .. {d[-1].date()}  [exact]")
        except ImportError:
            bd = pd.bdate_range("1997-06-01", "2016-12-30")
            d = bd[-n:]
            print(f"dates: {n} days, {d[0].date()} .. {d[-1].date()}")
            print("       APPROXIMATE -- pip install pandas_market_calendars for exact "
                  "NYSE dates (market_beta.py joins on these)")
        np.save(dp, np.asarray(d.values))

    print("\nready.  next:  python3 src/real_ladder.py   "
          "# OU on PCA-5 should land near 0.68")
    print("        (this script builds PCA-5 by default; "
          "pass IPCA-5 as an argument for the other arm)")

if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("PCA-5",))
