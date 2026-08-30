from closedloop.models import SimulationConfig
from closedloop.optimizer import search_operating_points
from closedloop.uncertainty import run_uncertainty_campaign


def tiny():
    return SimulationConfig(n_cells=12, dt_s=5e-4, duration_s=0.005)


def test_uncertainty_campaign_is_seeded_and_real():
    a = run_uncertainty_campaign(tiny(), samples=4, seed=8)
    b = run_uncertainty_campaign(tiny(), samples=4, seed=8)
    assert a == b
    assert a.q_p05 <= a.q_median <= a.q_p95
    assert 0 <= a.q_above_one_fraction <= 1
    assert 0 <= a.gate_pass_fraction <= 1


def test_optimizer_evaluates_declared_grid_and_returns_feasible_or_explicit_none():
    report = search_operating_points(
        tiny(),
        density_multipliers=(0.95, 1.0),
        ion_temperature_multipliers=(0.95, 1.0),
        external_heating_MW=(45.0, 50.0),
    )
    assert report.evaluated == 8
    assert 0 <= report.feasible <= report.evaluated
    if report.best is not None:
        assert report.best.causal_gate_pass
        assert report.best.combined_operating_envelope_pass
