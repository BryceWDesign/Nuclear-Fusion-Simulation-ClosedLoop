from dataclasses import replace

import pytest

from closedloop.models import SimulationConfig
from closedloop.simulation import ClosedLoopSimulator


def short_config(**kwargs):
    base = SimulationConfig(n_cells=24, dt_s=5e-4, duration_s=0.03)
    return replace(base, **kwargs)


def test_closed_loop_run_passes_conservation_gates():
    result = ClosedLoopSimulator(short_config()).run()
    assert abs(result.summary["energy_balance_relative_residual"]) < 1e-10
    assert result.summary["particle_balance_relative_residual"] < 1e-10
    assert result.gates.causal_fusion_result_allowed


def test_fusion_power_is_generated_from_dt_state():
    result = ClosedLoopSimulator(short_config()).run()
    assert result.summary["fusion_power_MW"] > 0
    assert result.summary["cumulative_DT_reactions"] > 0


def test_q_is_fusion_power_over_external_heating():
    cfg = short_config()
    result = ClosedLoopSimulator(cfg).run()
    assert result.summary["Q_plasma"] == pytest.approx(result.summary["fusion_power_MW"] / cfg.heating.external_heating_MW)


def test_no_external_heating_does_not_invent_q():
    cfg = short_config(heating=replace(short_config().heating, external_heating_MW=0.0))
    result = ClosedLoopSimulator(cfg).run()
    assert result.summary["Q_plasma"] is None
    assert not result.gates.causal_fusion_result_allowed


def test_alpha_population_is_born_from_fusion():
    result = ClosedLoopSimulator(short_config()).run()
    assert result.summary["cumulative_alpha_birth_energy_MJ"] > 0
    assert result.summary["fast_alpha_energy_MJ"] > 0


def test_neutron_power_has_dt_branching_ratio():
    result = ClosedLoopSimulator(short_config()).run()
    assert result.summary["neutron_power_MW"] / result.summary["fusion_power_MW"] == pytest.approx(14.1 / 17.6)


def test_cold_control_has_many_orders_less_fusion_than_hot_case():
    hot = ClosedLoopSimulator(short_config()).run().summary["fusion_power_MW"]
    base = short_config()
    cold_initial = replace(
        base.initial,
        core_ion_temperature_keV=0.3,
        edge_ion_temperature_keV=0.2,
        core_electron_temperature_keV=0.3,
        edge_electron_temperature_keV=0.2,
    )
    cold = ClosedLoopSimulator(replace(base, initial=cold_initial, heating=replace(base.heating, external_heating_MW=0.0))).run().summary["fusion_power_MW"]
    assert hot > cold * 1e6


def test_plant_net_power_stays_unknown_without_load_inventory():
    result = ClosedLoopSimulator(short_config()).run()
    assert result.summary["plant_closure"]["net_electric_MW"] is None


def test_plant_net_power_is_emitted_only_when_loads_declared():
    base = short_config()
    plant = replace(base.plant, recirculating_components_MW={"heating": 50.0, "cryo_assumption": 25.0})
    result = ClosedLoopSimulator(replace(base, plant=plant)).run()
    assert result.summary["plant_closure"]["closure_complete"]
    assert result.summary["plant_closure"]["net_electric_MW"] is not None


def test_timeseries_ends_at_requested_duration():
    cfg = short_config()
    result = ClosedLoopSimulator(cfg).run()
    assert result.timeseries[-1]["time_s"] == pytest.approx(cfg.duration_s)


def test_experimental_validation_is_always_false():
    result = ClosedLoopSimulator(short_config()).run()
    assert result.summary["experimental_validation"] is False
    assert result.gates.experimental_validation_claim_allowed is False
