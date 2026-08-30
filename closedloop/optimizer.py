"""Small transparent constrained operating-point grid search."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product

from .models import SimulationConfig
from .simulation import ClosedLoopSimulator


@dataclass(frozen=True)
class Candidate:
    density_multiplier: float
    ion_temperature_multiplier: float
    external_heating_MW: float
    fusion_power_MW: float
    Q_plasma: float
    combined_operating_envelope_pass: bool
    causal_gate_pass: bool


@dataclass(frozen=True)
class OptimizationReport:
    evaluated: int
    feasible: int
    best: Candidate | None
    authority: str = "finite_declared_grid_search_not_global_reactor_optimization"

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluated": self.evaluated,
            "feasible": self.feasible,
            "best": asdict(self.best) if self.best is not None else None,
            "authority": self.authority,
        }


def search_operating_points(
    config: SimulationConfig,
    *,
    density_multipliers: tuple[float, ...] = (0.9, 1.0, 1.1),
    ion_temperature_multipliers: tuple[float, ...] = (0.9, 1.0, 1.1),
    external_heating_MW: tuple[float, ...] = (40.0, 50.0, 60.0),
) -> OptimizationReport:
    if not density_multipliers or not ion_temperature_multipliers or not external_heating_MW:
        raise ValueError("search grids cannot be empty")
    best: Candidate | None = None
    feasible = 0
    evaluated = 0
    for dmul, tmul, heat in product(density_multipliers, ion_temperature_multipliers, external_heating_MW):
        if dmul <= 0 or tmul <= 0 or heat <= 0:
            raise ValueError("search grid values must be positive")
        temp = config.initial.core_ion_temperature_keV * tmul
        if temp > 100.0:
            continue
        varied = replace(
            config,
            initial=replace(
                config.initial,
                core_total_fuel_density_m3=config.initial.core_total_fuel_density_m3 * dmul,
                core_ion_temperature_keV=temp,
            ),
            heating=replace(config.heating, external_heating_MW=heat),
        )
        result = ClosedLoopSimulator(varied).run()
        evaluated += 1
        q = result.summary["Q_plasma"]
        envelope_ok = bool(result.summary["operating_envelope"]["combined_screen_pass"])
        gate_ok = bool(result.gates.causal_fusion_result_allowed)
        if q is None:
            continue
        candidate = Candidate(
            density_multiplier=dmul,
            ion_temperature_multiplier=tmul,
            external_heating_MW=heat,
            fusion_power_MW=float(result.summary["fusion_power_MW"]),
            Q_plasma=float(q),
            combined_operating_envelope_pass=envelope_ok,
            causal_gate_pass=gate_ok,
        )
        if envelope_ok and gate_ok:
            feasible += 1
            if best is None or candidate.Q_plasma > best.Q_plasma:
                best = candidate
    return OptimizationReport(evaluated, feasible, best)
