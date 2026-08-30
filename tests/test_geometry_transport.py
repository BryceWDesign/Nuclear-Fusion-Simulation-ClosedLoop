from math import pi

import numpy as np
import pytest

from closedloop.geometry import build_radial_geometry
from closedloop.models import TokamakGeometryConfig
from closedloop.transport import (
    TransportStabilityError,
    conductive_heat_flux,
    conservative_density_step,
    conservative_energy_step,
    diffusive_particle_flux,
    explicit_diffusion_dt_limit,
)


def geom(n=16):
    return build_radial_geometry(TokamakGeometryConfig(6.2, 2.0, 5.3), n)


def test_circular_torus_volume_is_exact_for_declared_geometry():
    g = geom()
    assert g.total_volume_m3 == pytest.approx(2 * pi**2 * 6.2 * 2.0**2)


def test_shell_volumes_sum_to_total():
    g = geom(32)
    assert np.sum(g.shell_volumes_m3) == pytest.approx(g.total_volume_m3)


def test_axis_face_area_is_zero():
    assert geom().face_areas_m2[0] == 0


def test_uniform_density_has_only_edge_outflow_for_vacuum_boundary():
    g = geom()
    n = np.full(16, 1e19)
    flux = diffusive_particle_flux(n, 0.05, g, 0.0)
    assert np.allclose(flux[1:-1], 0.0)
    assert flux[-1] > 0


def test_no_diffusion_means_no_flux():
    g = geom()
    flux = diffusive_particle_flux(np.linspace(2e19, 1e19, 16), 0.0, g, 0.0)
    assert np.all(flux == 0)


def test_density_step_conserves_internal_flux_when_edge_closed():
    g = geom()
    n = np.linspace(2e19, 1e19, 16)
    flux = diffusive_particle_flux(n, 0.05, g, n[-1])
    before = np.sum(n * g.shell_volumes_m3)
    out = conservative_density_step(n, flux, g, 1e-4)
    after = np.sum(out.values * g.shell_volumes_m3)
    assert after == pytest.approx(before, rel=1e-14)


def test_density_change_matches_edge_outflow():
    g = geom()
    n = np.full(16, 1e19)
    flux = diffusive_particle_flux(n, 0.01, g, 0.0)
    dt = 1e-5
    before = np.sum(n * g.shell_volumes_m3)
    out = conservative_density_step(n, flux, g, dt)
    after = np.sum(out.values * g.shell_volumes_m3)
    assert before - after == pytest.approx(flux[-1] * dt, rel=2e-10)


def test_heat_flux_zero_for_uniform_temperature_and_matching_edge():
    g = geom()
    n = np.full(16, 1e19)
    T = np.full(16, 5.0)
    q = conductive_heat_flux(n, T, 0.2, g, edge_temperature_keV=5.0)
    assert np.allclose(q, 0.0)


def test_heat_flux_is_outward_for_hot_plasma_cold_edge():
    g = geom()
    n = np.full(16, 1e19)
    T = np.full(16, 5.0)
    q = conductive_heat_flux(n, T, 0.2, g, edge_temperature_keV=0.2)
    assert q[-1] > 0


def test_energy_step_conserves_closed_boundary():
    g = geom()
    u = np.linspace(10.0, 5.0, 16)
    power = np.zeros(17)
    power[1:-1] = np.linspace(2, 1, 15)
    before = np.sum(u * g.shell_volumes_m3)
    out = conservative_energy_step(u, power, g, 1e-4)
    after = np.sum(out.values * g.shell_volumes_m3)
    assert after == pytest.approx(before)


def test_explicit_dt_limit_is_infinite_for_zero_diffusion():
    assert explicit_diffusion_dt_limit(0.0, 0.1) == float("inf")


def test_unstable_density_step_raises_instead_of_clamping():
    g = geom(8)
    n = np.full(8, 1.0)
    flux = np.zeros(9)
    flux[-1] = 1e20
    with pytest.raises(TransportStabilityError):
        conservative_density_step(n, flux, g, 1.0)
