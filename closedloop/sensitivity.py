"""Finite-difference sensitivity screen for selected physically declared inputs."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .models import SimulationConfig
from .simulation import ClosedLoopSimulator


@dataclass(frozen=True)
class Sensitivity:
    parameter: str
    baseline_value: float
    perturbation_fraction: float
    metric: str
    baseline_metric: float
    derivative_per_unit: float
    normalized_elasticity: float


def _run_metric(config: SimulationConfig, metric: str) -> float:
    value = ClosedLoopSimulator(config).run().summary[metric]
    if value is None:
        raise ValueError(f"metric {metric} is undefined")
    return float(value)


def sensitivity_screen(config: SimulationConfig, parameter: str, metric: str = "Q_plasma", fraction: float = 0.02) -> Sensitivity:
    if not 0 < fraction < 0.2:
        raise ValueError("fraction must be in (0,0.2)")
    if parameter == "external_heating_MW":
        base = config.heating.external_heating_MW
        low = replace(config, heating=replace(config.heating, external_heating_MW=base * (1 - fraction)))
        high = replace(config, heating=replace(config.heating, external_heating_MW=base * (1 + fraction)))
    elif parameter == "core_fuel_density_m3":
        base = config.initial.core_total_fuel_density_m3
        low = replace(config, initial=replace(config.initial, core_total_fuel_density_m3=base * (1 - fraction)))
        high = replace(config, initial=replace(config.initial, core_total_fuel_density_m3=base * (1 + fraction)))
    elif parameter == "core_ion_temperature_keV":
        base = config.initial.core_ion_temperature_keV
        low = replace(config, initial=replace(config.initial, core_ion_temperature_keV=base * (1 - fraction)))
        high = replace(config, initial=replace(config.initial, core_ion_temperature_keV=base * (1 + fraction)))
    elif parameter == "ion_thermal_diffusivity_m2_s":
        base = config.transport.ion_thermal_diffusivity_m2_s
        low = replace(config, transport=replace(config.transport, ion_thermal_diffusivity_m2_s=base * (1 - fraction)))
        high = replace(config, transport=replace(config.transport, ion_thermal_diffusivity_m2_s=base * (1 + fraction)))
    else:
        raise ValueError("unsupported parameter")
    if base <= 0:
        raise ValueError("baseline parameter must be positive")
    baseline_metric = _run_metric(config, metric)
    low_metric = _run_metric(low, metric)
    high_metric = _run_metric(high, metric)
    derivative = (high_metric - low_metric) / (2 * base * fraction)
    elasticity = derivative * base / baseline_metric if baseline_metric != 0 else float("inf")
    return Sensitivity(parameter, base, fraction, metric, baseline_metric, derivative, elasticity)
