import numpy as np
import pytest

from closedloop.constants import DT_ALPHA_ENERGY_J, DT_NEUTRON_ENERGY_J, DT_TOTAL_ENERGY_J
from closedloop.fusion import bosch_hale_dt_reactivity, reaction_ledger_from_rate, reaction_rate_density


def test_bosch_hale_reference_values_are_in_expected_range():
    assert bosch_hale_dt_reactivity(10.0) == pytest.approx(1.13616547e-22, rel=2e-7)
    assert bosch_hale_dt_reactivity(20.0) == pytest.approx(4.33020190e-22, rel=2e-7)


def test_bosch_hale_rises_across_common_burn_range():
    values = [bosch_hale_dt_reactivity(t) for t in (5.0, 10.0, 15.0, 20.0)]
    assert values == sorted(values)


@pytest.mark.parametrize("temperature", [0.19, 100.01, -1.0, 1000.0])
def test_bosch_hale_refuses_extrapolation(temperature):
    with pytest.raises(ValueError):
        bosch_hale_dt_reactivity(temperature)


def test_vector_reactivity_matches_scalar():
    temps = np.array([5.0, 10.0, 20.0])
    vec = bosch_hale_dt_reactivity(temps)
    assert np.allclose(vec, [bosch_hale_dt_reactivity(float(t)) for t in temps])


def test_reaction_rate_is_zero_below_fit_boundary_not_extrapolated():
    rate = reaction_rate_density(np.array([1e20]), np.array([1e20]), np.array([0.1]))
    assert rate[0] == 0.0


def test_reaction_rate_is_nd_nt_sigma_v():
    nd = np.array([4e19])
    nt = np.array([5e19])
    t = np.array([10.0])
    rate = reaction_rate_density(nd, nt, t)[0]
    assert rate == pytest.approx(nd[0] * nt[0] * bosch_hale_dt_reactivity(10.0))


def test_reaction_ledger_energy_channels_sum():
    ledger = reaction_ledger_from_rate(1e20)
    assert ledger.fusion_power_MW == pytest.approx(1e20 * DT_TOTAL_ENERGY_J / 1e6)
    assert ledger.alpha_power_MW + ledger.neutron_power_MW == pytest.approx(ledger.fusion_power_MW)
    assert DT_ALPHA_ENERGY_J + DT_NEUTRON_ENERGY_J == pytest.approx(DT_TOTAL_ENERGY_J)


def test_reaction_ledger_burns_more_tritium_mass_than_deuterium():
    ledger = reaction_ledger_from_rate(1e20)
    assert ledger.tritium_burn_kg_day > ledger.deuterium_burn_kg_day > 0


def test_negative_reaction_rate_rejected():
    with pytest.raises(ValueError):
        reaction_ledger_from_rate(-1.0)
