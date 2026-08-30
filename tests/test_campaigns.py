import math

from closedloop.convergence import run_convergence_campaign
from closedloop.models import SimulationConfig
from closedloop.sensitivity import sensitivity_screen


def tiny_config():
    return SimulationConfig(n_cells=12, dt_s=5e-4, duration_s=0.005)


def test_convergence_campaign_executes_three_real_runs():
    report = run_convergence_campaign(tiny_config())
    assert report.coarse["fusion_power_MW"] > 0
    assert report.temporal_refined["fusion_power_MW"] > 0
    assert report.spatial_refined["fusion_power_MW"] > 0
    assert report.relative_differences["fusion_power_MW"]["temporal"] is not None


def test_density_sensitivity_is_finite():
    result = sensitivity_screen(tiny_config(), "core_fuel_density_m3", "fusion_power_MW", 0.01)
    assert math.isfinite(result.derivative_per_unit)
    assert result.normalized_elasticity > 0


def test_temperature_sensitivity_is_positive_in_baseline_regime():
    result = sensitivity_screen(tiny_config(), "core_ion_temperature_keV", "fusion_power_MW", 0.01)
    assert result.derivative_per_unit > 0
