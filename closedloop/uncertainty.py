"""Seeded model-input uncertainty campaign for reduced-order robustness screening."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .models import SimulationConfig
from .simulation import ClosedLoopSimulator


@dataclass(frozen=True)
class UncertaintyReport:
    samples: int
    seed: int
    q_median: float
    q_p05: float
    q_p95: float
    q_above_one_fraction: float
    fusion_power_median_MW: float
    gate_pass_fraction: float
    authority: str = "declared_model_input_uncertainty_not_experimental_probability"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_uncertainty_campaign(
    config: SimulationConfig,
    *,
    samples: int = 32,
    seed: int = 1234,
    density_sigma_fraction: float = 0.05,
    ion_temperature_sigma_fraction: float = 0.05,
    transport_sigma_fraction: float = 0.10,
    heating_sigma_fraction: float = 0.03,
) -> UncertaintyReport:
    if samples < 2:
        raise ValueError("samples must be >= 2")
    sigmas = (density_sigma_fraction, ion_temperature_sigma_fraction, transport_sigma_fraction, heating_sigma_fraction)
    if min(sigmas) < 0 or max(sigmas) > 0.5:
        raise ValueError("uncertainty fractions must be in [0,0.5]")
    rng = np.random.default_rng(seed)
    q_values: list[float] = []
    fusion_values: list[float] = []
    gate_passes = 0
    for _ in range(samples):
        density_factor = max(0.1, float(rng.normal(1.0, density_sigma_fraction)))
        temp_factor = max(0.1, float(rng.normal(1.0, ion_temperature_sigma_fraction)))
        transport_factor = max(0.1, float(rng.normal(1.0, transport_sigma_fraction)))
        heating_factor = max(0.1, float(rng.normal(1.0, heating_sigma_fraction)))
        varied = replace(
            config,
            initial=replace(
                config.initial,
                core_total_fuel_density_m3=config.initial.core_total_fuel_density_m3 * density_factor,
                core_ion_temperature_keV=min(99.0, config.initial.core_ion_temperature_keV * temp_factor),
            ),
            transport=replace(
                config.transport,
                ion_thermal_diffusivity_m2_s=config.transport.ion_thermal_diffusivity_m2_s * transport_factor,
                electron_thermal_diffusivity_m2_s=config.transport.electron_thermal_diffusivity_m2_s * transport_factor,
            ),
            heating=replace(config.heating, external_heating_MW=config.heating.external_heating_MW * heating_factor),
        )
        result = ClosedLoopSimulator(varied).run()
        q = result.summary["Q_plasma"]
        if q is None:
            raise ValueError("uncertainty campaign requires nonzero external heating")
        q_values.append(float(q))
        fusion_values.append(float(result.summary["fusion_power_MW"]))
        gate_passes += int(result.gates.causal_fusion_result_allowed)
    q_arr = np.asarray(q_values)
    fusion_arr = np.asarray(fusion_values)
    return UncertaintyReport(
        samples=samples,
        seed=seed,
        q_median=float(np.median(q_arr)),
        q_p05=float(np.quantile(q_arr, 0.05)),
        q_p95=float(np.quantile(q_arr, 0.95)),
        q_above_one_fraction=float(np.mean(q_arr > 1.0)),
        fusion_power_median_MW=float(np.median(fusion_arr)),
        gate_pass_fraction=gate_passes / samples,
    )
