"""Conditional plant-power closure. Unknown recirculating loads remain unknown."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PlantClosure:
    fusion_power_MW: float
    blanket_energy_multiplier: float
    thermal_power_MW: float
    gross_thermal_efficiency: float
    gross_electric_MW: float
    recirculating_power_MW: float | None
    net_electric_MW: float | None
    closure_complete: bool
    authority: str = "algebraic_conditional_plant_closure_not_integrated_engineering_prediction"


def close_plant_power(
    fusion_power_MW: float,
    blanket_energy_multiplier: float,
    gross_thermal_efficiency: float,
    recirculating_components_MW: dict[str, float] | None,
) -> PlantClosure:
    if fusion_power_MW < 0 or blanket_energy_multiplier < 1 or not 0 < gross_thermal_efficiency < 1:
        raise ValueError("invalid plant inputs")
    thermal = fusion_power_MW * blanket_energy_multiplier
    gross = thermal * gross_thermal_efficiency
    if recirculating_components_MW is None:
        return PlantClosure(fusion_power_MW, blanket_energy_multiplier, thermal, gross_thermal_efficiency, gross, None, None, False)
    if not recirculating_components_MW or min(recirculating_components_MW.values()) < 0:
        raise ValueError("recirculating components must be a nonempty map of nonnegative loads")
    recirc = float(sum(recirculating_components_MW.values()))
    return PlantClosure(fusion_power_MW, blanket_energy_multiplier, thermal, gross_thermal_efficiency, gross, recirc, gross - recirc, True)


def as_jsonable(value: PlantClosure) -> dict[str, object]:
    return asdict(value)
