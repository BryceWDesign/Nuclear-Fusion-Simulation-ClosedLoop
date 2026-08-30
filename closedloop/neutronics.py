"""Exact D-T source accounting and geometric tritium-breeding constraints."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .fusion import reaction_ledger_from_rate


@dataclass(frozen=True)
class BreedingConstraint:
    reaction_rate_per_s: float
    global_tbr_target: float
    breeding_coverage_fraction: float
    required_local_tbr_if_uncovered_regions_breed_zero: float
    neutron_power_MW: float
    tritium_burn_kg_day: float
    required_bred_tritium_kg_day: float
    authority: str = "exact_source_and_coverage_bound_not_neutron_transport"


def breeding_constraint(reaction_rate_per_s: float, global_tbr_target: float, coverage_fraction: float) -> BreedingConstraint:
    if reaction_rate_per_s < 0 or global_tbr_target <= 0 or not 0 < coverage_fraction <= 1:
        raise ValueError("invalid breeding inputs")
    ledger = reaction_ledger_from_rate(reaction_rate_per_s)
    return BreedingConstraint(
        reaction_rate_per_s=reaction_rate_per_s,
        global_tbr_target=global_tbr_target,
        breeding_coverage_fraction=coverage_fraction,
        required_local_tbr_if_uncovered_regions_breed_zero=global_tbr_target / coverage_fraction,
        neutron_power_MW=ledger.neutron_power_MW,
        tritium_burn_kg_day=ledger.tritium_burn_kg_day,
        required_bred_tritium_kg_day=ledger.tritium_burn_kg_day * global_tbr_target,
    )


def as_jsonable(value: BreedingConstraint) -> dict[str, object]:
    return asdict(value)
