import pytest

from closedloop.gates import evaluate_claim_gates
from closedloop.neutronics import breeding_constraint
from closedloop.plant import close_plant_power


def test_breeding_coverage_penalty():
    b = breeding_constraint(1e20, 1.1, 0.85)
    assert b.required_local_tbr_if_uncovered_regions_breed_zero == pytest.approx(1.1 / 0.85)
    assert b.required_local_tbr_if_uncovered_regions_breed_zero > b.global_tbr_target


def test_breeding_source_is_positive():
    b = breeding_constraint(1e20, 1.1, 1.0)
    assert b.neutron_power_MW > 0
    assert b.tritium_burn_kg_day > 0


def test_plant_refuses_to_invent_missing_recirculating_loads():
    p = close_plant_power(500, 1.0, 0.4, None)
    assert not p.closure_complete
    assert p.net_electric_MW is None


def test_plant_closes_when_all_declared_loads_supplied():
    p = close_plant_power(500, 1.0, 0.4, {"heating": 50, "cryo": 30})
    assert p.closure_complete
    assert p.net_electric_MW == pytest.approx(120.0)


def test_claim_gates_never_allow_experimental_validation():
    g = evaluate_claim_gates(
        energy_residual=0,
        particle_residual=0,
        energy_limit=1e-5,
        particle_limit=1e-5,
        q_plasma=5.0,
        alpha_heating_fraction=0.8,
        external_heating_MW=50,
        plant_closure_complete=True,
    )
    assert g.causal_fusion_result_allowed
    assert g.burning_plasma_screen_allowed
    assert g.net_electric_statement_allowed
    assert not g.experimental_validation_claim_allowed
