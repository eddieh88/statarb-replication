"""Checks on the authors' residuals; skipped until setup_data.py has been run."""
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PCA = ROOT / "dlsa_real" / "PCA-5_masked.npy"


@pytest.mark.skipif(not PCA.exists(), reason="run setup_data.py first")
def test_residuals_have_one_row_per_trading_day():
    arr = np.load(PCA, mmap_mode="r")
    dates = np.load(ROOT / "dlsa_real" / "dates.npy")
    assert arr.shape[0] == len(dates) == 4781
