import pytest

from closedloop.fast_particles import alpha_orbit_screen
from closedloop.models import TokamakGeometryConfig
from closedloop.stability import operating_envelope


def test_operating_envelope_computes_finite_tokamak_screens():
    g = TokamakGeometryConfig()
    result = operating_envelope(geometry=g, volume_average_pressure_Pa=3e5, volume_average_electron_density_m3=8e19)
    assert result.beta_fraction > 0
    assert result.greenwald_density_limit_m3 > 0
    assert result.cylindrical_edge_q > 0


def test_greenwald_fraction_rises_with_density():
    g = TokamakGeometryConfig()
    low = operating_envelope(geometry=g, volume_average_pressure_Pa=1e5, volume_average_electron_density_m3=4e19)
    high = operating_envelope(geometry=g, volume_average_pressure_Pa=1e5, volume_average_electron_density_m3=8e19)
    assert high.greenwald_fraction == pytest.approx(2 * low.greenwald_fraction)


def test_alpha_orbit_screen_is_deterministic_and_energy_preserving():
    g = TokamakGeometryConfig()
    a = alpha_orbit_screen(g, markers=32, steps=100, dt_s=2e-10, seed=3)
    b = alpha_orbit_screen(g, markers=32, steps=100, dt_s=2e-10, seed=3)
    assert a == b
    assert 0 <= a.retained_fraction <= 1
    assert a.relative_kinetic_energy_drift_max < 1e-12


def test_alpha_orbit_screen_rejects_bad_inputs():
    with pytest.raises(ValueError):
        alpha_orbit_screen(TokamakGeometryConfig(), markers=0)
