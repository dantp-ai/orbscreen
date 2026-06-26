import numpy as np

from orbscreen.eval.metrics import expected_calibration_error


def test_ece_perfect_calibration_is_low():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)  # calibrated by construction
    assert expected_calibration_error(y, p, n_bins=10) < 0.05


def test_ece_miscalibration_is_higher():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])  # confidently wrong
    assert expected_calibration_error(y, p, n_bins=5) > 0.5
