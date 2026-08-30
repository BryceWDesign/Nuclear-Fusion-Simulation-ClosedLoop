"""Typed configuration and state models for the ClosedLoop engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TokamakGeometryConfig:
    major_radius_m: float = 6.2
    minor_radius_m: float = 2.0
    toroidal_field_T: float = 5.3
    plasma_current_MA: float = 15.0
    elongation: float = 1.7

    def validate(self) -> None:
        if self.major_radius_m <= self.minor_radius_m:
            raise ValueError("major_radius_m must exceed minor_radius_m")
        if min(self.minor_radius_m, self.toroidal_field_T, self.plasma_current_MA, self.elongation) <= 0:
            raise ValueError("geometry, field, plasma current and elongation must be positive")


@dataclass(frozen=True)
class InitialPlasmaConfig:
    core_total_fuel_density_m3: float = 9.0e19
    edge_total_fuel_density_m3: float = 1.5e19
    tritium_fraction: float = 0.5
    core_ion_temperature_keV: float = 12.0
    edge_ion_temperature_keV: float = 1.0
    core_electron_temperature_keV: float = 10.0
    edge_electron_temperature_keV: float = 0.8
    profile_exponent: float = 1.5

    def validate(self) -> None:
        if min(self.core_total_fuel_density_m3, self.edge_total_fuel_density_m3) <= 0:
            raise ValueError("fuel density must be positive")
        if self.core_total_fuel_density_m3 < self.edge_total_fuel_density_m3:
            raise ValueError("core fuel density must be >= edge fuel density")
        if not 0.0 < self.tritium_fraction < 1.0:
            raise ValueError("tritium_fraction must be between zero and one")
        temps = (
            self.core_ion_temperature_keV,
            self.edge_ion_temperature_keV,
            self.core_electron_temperature_keV,
            self.edge_electron_temperature_keV,
        )
        if min(temps) < 0.2 or max(temps) > 100.0:
            raise ValueError("initial temperatures must stay within Bosch-Hale model range 0.2-100 keV")
        if self.profile_exponent <= 0:
            raise ValueError("profile_exponent must be positive")


@dataclass(frozen=True)
class TransportConfig:
    particle_diffusivity_D_m2_s: float = 0.05
    particle_diffusivity_T_m2_s: float = 0.05
    particle_diffusivity_He_m2_s: float = 0.03
    ion_thermal_diffusivity_m2_s: float = 0.25
    electron_thermal_diffusivity_m2_s: float = 0.35
    edge_density_m3: float = 0.0
    edge_ion_temperature_keV: float = 0.2
    edge_electron_temperature_keV: float = 0.2

    def validate(self) -> None:
        vals = (
            self.particle_diffusivity_D_m2_s,
            self.particle_diffusivity_T_m2_s,
            self.particle_diffusivity_He_m2_s,
            self.ion_thermal_diffusivity_m2_s,
            self.electron_thermal_diffusivity_m2_s,
        )
        if min(vals) < 0:
            raise ValueError("transport coefficients cannot be negative")
        if self.edge_density_m3 < 0:
            raise ValueError("edge density cannot be negative")
        if min(self.edge_ion_temperature_keV, self.edge_electron_temperature_keV) < 0:
            raise ValueError("edge temperatures cannot be negative")


@dataclass(frozen=True)
class AlphaConfig:
    slowing_time_s: float = 0.35
    orbit_loss_time_s: float = 2.0
    electron_heating_fraction: float = 0.80

    def validate(self) -> None:
        if min(self.slowing_time_s, self.orbit_loss_time_s) <= 0:
            raise ValueError("alpha timescales must be positive")
        if not 0 <= self.electron_heating_fraction <= 1:
            raise ValueError("electron_heating_fraction must be in [0,1]")


@dataclass(frozen=True)
class HeatingConfig:
    external_heating_MW: float = 50.0
    electron_fraction: float = 0.45
    deposition_width_rho: float = 0.55

    def validate(self) -> None:
        if self.external_heating_MW < 0:
            raise ValueError("external heating cannot be negative")
        if not 0 <= self.electron_fraction <= 1:
            raise ValueError("heating electron fraction must be in [0,1]")
        if self.deposition_width_rho <= 0:
            raise ValueError("deposition width must be positive")


@dataclass(frozen=True)
class RadiationConfig:
    zeff: float = 1.4
    impurity_fraction: float = 0.0
    impurity_cooling_coefficient_W_m3: float = 0.0

    def validate(self) -> None:
        if self.zeff < 1:
            raise ValueError("zeff must be >= 1")
        if self.impurity_fraction < 0:
            raise ValueError("impurity_fraction cannot be negative")
        if self.impurity_cooling_coefficient_W_m3 < 0:
            raise ValueError("impurity cooling coefficient cannot be negative")


@dataclass(frozen=True)
class CouplingConfig:
    electron_ion_equilibration_time_s: float = 0.8

    def validate(self) -> None:
        if self.electron_ion_equilibration_time_s <= 0:
            raise ValueError("electron-ion equilibration time must be positive")


@dataclass(frozen=True)
class PlantConfig:
    blanket_energy_multiplier: float = 1.0
    gross_thermal_efficiency: float = 0.40
    blanket_coverage_fraction: float = 0.85
    global_tbr_target: float = 1.10
    recirculating_components_MW: dict[str, float] | None = None

    def validate(self) -> None:
        if self.blanket_energy_multiplier < 1:
            raise ValueError("blanket multiplier must be >= 1")
        if not 0 < self.gross_thermal_efficiency < 1:
            raise ValueError("gross efficiency must be between zero and one")
        if not 0 < self.blanket_coverage_fraction <= 1:
            raise ValueError("blanket coverage must be in (0,1]")
        if self.global_tbr_target <= 0:
            raise ValueError("global TBR target must be positive")
        if self.recirculating_components_MW is not None:
            if not self.recirculating_components_MW:
                raise ValueError("recirculating_components_MW cannot be empty when supplied")
            if min(self.recirculating_components_MW.values()) < 0:
                raise ValueError("recirculating loads cannot be negative")


@dataclass(frozen=True)
class VerificationConfig:
    max_energy_balance_relative_residual: float = 2.0e-6
    max_particle_balance_relative_residual: float = 2.0e-10
    max_fractional_fuel_burn_per_step: float = 0.02

    def validate(self) -> None:
        if min(
            self.max_energy_balance_relative_residual,
            self.max_particle_balance_relative_residual,
            self.max_fractional_fuel_burn_per_step,
        ) <= 0:
            raise ValueError("verification tolerances must be positive")
        if self.max_fractional_fuel_burn_per_step >= 1:
            raise ValueError("max_fractional_fuel_burn_per_step must be < 1")


@dataclass(frozen=True)
class SimulationConfig:
    n_cells: int = 48
    dt_s: float = 1.0e-3
    duration_s: float = 1.0
    random_seed: int = 7
    geometry: TokamakGeometryConfig = field(default_factory=TokamakGeometryConfig)
    initial: InitialPlasmaConfig = field(default_factory=InitialPlasmaConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    heating: HeatingConfig = field(default_factory=HeatingConfig)
    radiation: RadiationConfig = field(default_factory=RadiationConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

    def validate(self) -> None:
        if self.n_cells < 8:
            raise ValueError("n_cells must be >= 8")
        if self.dt_s <= 0 or self.duration_s <= 0:
            raise ValueError("dt_s and duration_s must be positive")
        if self.duration_s < self.dt_s:
            raise ValueError("duration_s must be >= dt_s")
        self.geometry.validate()
        self.initial.validate()
        self.transport.validate()
        self.alpha.validate()
        self.heating.validate()
        self.radiation.validate()
        self.coupling.validate()
        self.plant.validate()
        self.verification.validate()

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_s / self.dt_s))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SimulationConfig":
        return cls(
            n_cells=int(raw.get("n_cells", cls.n_cells)),
            dt_s=float(raw.get("dt_s", cls.dt_s)),
            duration_s=float(raw.get("duration_s", cls.duration_s)),
            random_seed=int(raw.get("random_seed", cls.random_seed)),
            geometry=TokamakGeometryConfig(**raw.get("geometry", {})),
            initial=InitialPlasmaConfig(**raw.get("initial", {})),
            transport=TransportConfig(**raw.get("transport", {})),
            alpha=AlphaConfig(**raw.get("alpha", {})),
            heating=HeatingConfig(**raw.get("heating", {})),
            radiation=RadiationConfig(**raw.get("radiation", {})),
            coupling=CouplingConfig(**raw.get("coupling", {})),
            plant=PlantConfig(**raw.get("plant", {})),
            verification=VerificationConfig(**raw.get("verification", {})),
        )


@dataclass
class PlasmaState:
    n_D_m3: np.ndarray
    n_T_m3: np.ndarray
    n_He_m3: np.ndarray
    n_alpha_fast_m3: np.ndarray
    ion_energy_J_m3: np.ndarray
    electron_energy_J_m3: np.ndarray
    alpha_fast_energy_J_m3: np.ndarray

    def copy(self) -> "PlasmaState":
        return PlasmaState(*(np.array(v, copy=True) for v in self.__dict__.values()))
