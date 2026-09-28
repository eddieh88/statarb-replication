"""Unit tests for the harness in src/real_ladder.py. No data cache needed."""
import numpy as np
import pytest

import real_ladder as rl


def test_windows_are_cumulative_returns_within_each_window():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.01, (8, 3)).astype(np.float32)
    W, sel = rl.windows_and_mask(data, lookback=3)
    assert W.shape == (5, 3, 3) and sel.shape == (5, 3)
    for t in range(3, 8):
        np.testing.assert_allclose(W[t - 3], np.cumsum(data[t - 3:t], axis=0).T, rtol=1e-5, atol=1e-6)


def test_zero_return_marks_stock_invalid_while_in_window():
    data = np.full((7, 2), 0.01, np.float32)
    data[2, 1] = 0.0                      # missing data is encoded as 0
    _, sel = rl.windows_and_mask(data, lookback=3)
    # windows start at rows 0..3; stock 1 is invalid while row 2 is inside the window
    assert sel[:, 0].all()
    assert sel[:, 1].tolist() == [False, False, False, True]


def test_port_returns_is_l1_normalised_and_ignores_invalid_stocks():
    data = np.array([[0, 0], [0, 0], [0.02, -0.01], [0.01, 0.03]])
    weights = np.array([[2.0, 1.0], [1.0, -1.0]])
    sel = np.array([[True, False], [True, True]])
    r = rl.port_returns(weights, data, sel, lookback=2)
    np.testing.assert_allclose(r, [0.02, (0.01 - 0.03) / 2])


def test_reversal_bets_against_the_last_cumulative_move():
    W = np.array([[[0.0, 0.01, 0.03], [0.0, -0.02, -0.05]]])
    np.testing.assert_allclose(rl.w_reversal(W, None), [[-0.03, 0.05]])


def test_sharpe_is_annualised_mean_over_std():
    r = np.array([0.01, 0.03, 0.02, 0.00])
    assert rl.sharpe(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))


def test_ou_weights_are_zero_for_invalid_stocks():
    rng = np.random.default_rng(1)
    W = np.cumsum(rng.normal(0, 0.01, (4, 5, 30)), axis=2)
    sel = np.ones((4, 5), bool)
    sel[:, 0] = False
    out = rl.w_ou(W, sel)
    assert out.shape == (4, 5)
    assert np.all(out[:, 0] == 0)
