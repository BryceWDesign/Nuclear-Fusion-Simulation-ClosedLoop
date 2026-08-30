import numpy as np
import pytest

from closedloop.losses import bremsstrahlung_W_m3, electron_ion_exchange_W_m3, impurity_line_radiation_W_m3


def test_bremsstrahlung_scales_with_density_squared():
    p1 = bremsstrahlung_W_m3(np.array([1e20]), np.array([10.0]), 1.2)[0]
    p2 = bremsstrahlung_W_m3(np.array([2e20]), np.array([10.0]), 1.2)[0]
    assert p2 == pytest.approx(4 * p1)


def test_bremsstrahlung_scales_with_sqrt_temperature():
    p1 = bremsstrahlung_W_m3(np.array([1e20]), np.array([4.0]), 1.0)[0]
    p2 = bremsstrahlung_W_m3(np.array([1e20]), np.array([16.0]), 1.0)[0]
    assert p2 == pytest.approx(2 * p1)


def test_exchange_positive_when_ions_hotter():
    q = electron_ion_exchange_W_m3(np.array([1e20]), np.array([10.0]), np.array([5.0]), 1.0)[0]
    assert q > 0


def test_exchange_negative_when_electrons_hotter():
    q = electron_ion_exchange_W_m3(np.array([1e20]), np.array([5.0]), np.array([10.0]), 1.0)[0]
    assert q < 0


def test_exchange_zero_at_equal_temperature():
    q = electron_ion_exchange_W_m3(np.array([1e20]), np.array([5.0]), np.array([5.0]), 1.0)[0]
    assert q == 0


def test_impurity_radiation_zero_when_no_impurity():
    p = impurity_line_radiation_W_m3(np.array([1e20]), 0.0, 1e-31)
    assert p[0] == 0
