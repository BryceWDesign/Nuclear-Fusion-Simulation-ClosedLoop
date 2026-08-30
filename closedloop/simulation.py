"""Causally closed reduced-order D-T burn + radial transport engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
from typing import Any

import numpy as np

from .constants import DT_ALPHA_ENERGY_J, DT_TOTAL_ENERGY_J, KEV_TO_J
from .fusion import reaction_ledger_from_rate, reaction_rate_density
from .gates import ClaimGateReport, evaluate_claim_gates
from .geometry import build_radial_geometry
from .losses import bremsstrahlung_W_m3, electron_ion_exchange_W_m3, impurity_line_radiation_W_m3
from .models import PlasmaState, SimulationConfig
from .neutronics import BreedingConstraint, breeding_constraint
from .plant import PlantClosure, close_plant_power
from .stability import operating_envelope
from .transport import (
    conductive_heat_flux,
    conservative_density_step,
    conservative_energy_step,
    convective_energy_flux_from_particle_flux,
    diffusive_particle_flux,
    explicit_diffusion_dt_limit,
    temperature_keV_from_energy,
)


class SimulationValidityError(RuntimeError):
    """Raised when a requested run leaves the declared numerical/model authority."""


@dataclass(frozen=True)
class SimulationResult:
    summary: dict[str, Any]
    timeseries: list[dict[str, Any]]
    gates: ClaimGateReport
    breeding: BreedingConstraint
    plant: PlantClosure
    final_state: PlasmaState


class ClosedLoopSimulator:
    """Finite-volume reduced-order tokamak burn model.

    Bulk D, T and thermal He are radial fluid populations. Fusion-born alphas are a
    fast-particle energy/number reservoir with explicit slowing and escape timescales.
    The model is deterministic, conservative within declared numerical tolerances, and
    intentionally refuses to call itself experimentally validated.
    """

    def __init__(self, config: SimulationConfig):
        config.validate()
        self.config = config
        self.geometry = build_radial_geometry(config.geometry, config.n_cells)
        self._check_explicit_stability()
        self.state = self._initial_state()
        self._external_profile_W_m3 = self._build_external_heating_profile()

    def _check_explicit_stability(self) -> None:
        c = self.config.transport
        coeffs = {
            "D particle": c.particle_diffusivity_D_m2_s,
            "T particle": c.particle_diffusivity_T_m2_s,
            "He particle": c.particle_diffusivity_He_m2_s,
            "ion heat": c.ion_thermal_diffusivity_m2_s,
            "electron heat": c.electron_thermal_diffusivity_m2_s,
        }
        for name, value in coeffs.items():
            limit = explicit_diffusion_dt_limit(value, self.geometry.dr_m)
            if self.config.dt_s > limit:
                raise SimulationValidityError(
                    f"dt={self.config.dt_s:g}s exceeds explicit {name} stability screen {limit:g}s"
                )

    def _profile(self, core: float, edge: float) -> np.ndarray:
        rho = self.geometry.rho_centers
        shape = np.maximum(1.0 - rho**2, 0.0) ** self.config.initial.profile_exponent
        return edge + (core - edge) * shape

    def _initial_state(self) -> PlasmaState:
        init = self.config.initial
        fuel = self._profile(init.core_total_fuel_density_m3, init.edge_total_fuel_density_m3)
        nT = init.tritium_fraction * fuel
        nD = (1.0 - init.tritium_fraction) * fuel
        nHe = np.zeros_like(fuel)
        nAlpha = np.zeros_like(fuel)
        ti = self._profile(init.core_ion_temperature_keV, init.edge_ion_temperature_keV)
        te = self._profile(init.core_electron_temperature_keV, init.edge_electron_temperature_keV)
        ni = nD + nT
        ne = nD + nT
        ui = 1.5 * ni * ti * KEV_TO_J
        ue = 1.5 * ne * te * KEV_TO_J
        return PlasmaState(nD, nT, nHe, nAlpha, ui, ue, np.zeros_like(fuel))

    def _build_external_heating_profile(self) -> np.ndarray:
        heat = self.config.heating
        shape = np.exp(-((self.geometry.rho_centers / heat.deposition_width_rho) ** 2))
        weighted = float(np.sum(shape * self.geometry.shell_volumes_m3))
        if heat.external_heating_MW == 0:
            return np.zeros_like(shape)
        return shape * heat.external_heating_MW * 1.0e6 / weighted

    def _densities_temperatures(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = self.state
        ni = s.n_D_m3 + s.n_T_m3 + s.n_He_m3
        # Quasineutral electron fluid. Fast and thermal He both carry +2e.
        ne = s.n_D_m3 + s.n_T_m3 + 2.0 * (s.n_He_m3 + s.n_alpha_fast_m3)
        ti = temperature_keV_from_energy(s.ion_energy_J_m3, ni)
        te = temperature_keV_from_energy(s.electron_energy_J_m3, ne)
        return ni, ne, ti, te

    def _integral(self, density: np.ndarray) -> float:
        return float(np.sum(np.asarray(density) * self.geometry.shell_volumes_m3))

    def _volume_average(self, value: np.ndarray) -> float:
        return self._integral(value) / self.geometry.total_volume_m3

    def run(self) -> SimulationResult:
        cfg = self.config
        s = self.state
        dt = cfg.dt_s

        initial_D = self._integral(s.n_D_m3)
        initial_T = self._integral(s.n_T_m3)
        initial_energy = self._integral(s.ion_energy_J_m3 + s.electron_energy_J_m3 + s.alpha_fast_energy_J_m3)

        cumulative_reactions = 0.0
        cumulative_D_edge = 0.0
        cumulative_T_edge = 0.0
        cumulative_He_edge = 0.0
        cumulative_alpha_escape = 0.0
        cumulative_external_J = 0.0
        cumulative_alpha_birth_J = 0.0
        cumulative_radiation_J = 0.0
        cumulative_transport_J = 0.0
        cumulative_alpha_escape_J = 0.0
        cumulative_fusion_thermal_J = 0.0
        cumulative_ambipolar_J = 0.0

        timeseries: list[dict[str, Any]] = []
        record_stride = max(1, cfg.n_steps // 200)
        final_metrics: dict[str, Any] = {}

        for step in range(cfg.n_steps):
            t = (step + 1) * dt
            ni, ne, ti, te = self._densities_temperatures()

            # --- Causal D-T reaction source ---
            rate_density = reaction_rate_density(s.n_D_m3, s.n_T_m3, ti)
            reacted_density = rate_density * dt
            min_fuel = np.minimum(s.n_D_m3, s.n_T_m3)
            active = min_fuel > 0
            max_fraction = float(np.max(np.divide(reacted_density, min_fuel, out=np.zeros_like(min_fuel), where=active)))
            if max_fraction > cfg.verification.max_fractional_fuel_burn_per_step:
                raise SimulationValidityError(
                    f"fusion burn fraction per step {max_fraction:.3e} exceeds declared limit "
                    f"{cfg.verification.max_fractional_fuel_burn_per_step:.3e}; reduce dt"
                )
            if np.any(reacted_density > min_fuel * (1.0 + 1e-12)):
                raise SimulationValidityError("fusion step consumed more fuel than available")

            # Reactant thermal kinetic energy leaves the thermal ion population with the products.
            fusion_thermal_sink_J_m3 = reacted_density * (3.0 * ti * KEV_TO_J)
            if np.any(fusion_thermal_sink_J_m3 > s.ion_energy_J_m3 * (1.0 + 1e-12)):
                raise SimulationValidityError("fusion thermal sink exceeded ion thermal energy")
            s.ion_energy_J_m3 -= fusion_thermal_sink_J_m3
            s.n_D_m3 -= reacted_density
            s.n_T_m3 -= reacted_density
            s.n_alpha_fast_m3 += reacted_density
            alpha_birth_J_m3 = reacted_density * DT_ALPHA_ENERGY_J
            s.alpha_fast_energy_J_m3 += alpha_birth_J_m3

            reactions_this_step = self._integral(reacted_density)
            cumulative_reactions += reactions_this_step
            cumulative_alpha_birth_J += self._integral(alpha_birth_J_m3)
            cumulative_fusion_thermal_J += self._integral(fusion_thermal_sink_J_m3)

            # --- Fast-alpha slowing and orbit loss, exact exponential reservoir decay ---
            alpha_cfg = cfg.alpha
            lambda_s = 1.0 / alpha_cfg.slowing_time_s
            lambda_l = 1.0 / alpha_cfg.orbit_loss_time_s
            lambda_total = lambda_s + lambda_l
            decay_fraction = 1.0 - exp(-lambda_total * dt)
            slow_branch = lambda_s / lambda_total
            loss_branch = lambda_l / lambda_total
            alpha_n_before = np.array(s.n_alpha_fast_m3, copy=True)
            alpha_e_before = np.array(s.alpha_fast_energy_J_m3, copy=True)
            thermalized_density = alpha_n_before * decay_fraction * slow_branch
            escaped_alpha_density = alpha_n_before * decay_fraction * loss_branch
            deposited_alpha_J_m3 = alpha_e_before * decay_fraction * slow_branch
            escaped_alpha_J_m3 = alpha_e_before * decay_fraction * loss_branch
            survival = 1.0 - decay_fraction
            s.n_alpha_fast_m3 = alpha_n_before * survival
            s.alpha_fast_energy_J_m3 = alpha_e_before * survival
            s.n_He_m3 += thermalized_density
            alpha_escape_count = self._integral(escaped_alpha_density)
            cumulative_alpha_escape += alpha_escape_count
            cumulative_alpha_escape_J += self._integral(escaped_alpha_J_m3)

            # Ambipolar electron energy carried with the +2 alpha charge that escapes.
            _, ne_pretransport, _, te_pretransport = self._densities_temperatures()
            ambipolar_electron_loss_J_m3 = 2.0 * escaped_alpha_density * (1.5 * te_pretransport * KEV_TO_J)
            if np.any(ambipolar_electron_loss_J_m3 > s.electron_energy_J_m3 * (1.0 + 1e-12)):
                raise SimulationValidityError("ambipolar alpha-loss electron energy exceeded available electron energy")
            s.electron_energy_J_m3 -= ambipolar_electron_loss_J_m3
            cumulative_ambipolar_J += self._integral(ambipolar_electron_loss_J_m3)

            # --- Conservative particle transport ---
            ni, ne, ti, te = self._densities_temperatures()
            tr = cfg.transport
            flux_D = diffusive_particle_flux(s.n_D_m3, tr.particle_diffusivity_D_m2_s, self.geometry, tr.edge_density_m3)
            flux_T = diffusive_particle_flux(s.n_T_m3, tr.particle_diffusivity_T_m2_s, self.geometry, tr.edge_density_m3)
            flux_He = diffusive_particle_flux(s.n_He_m3, tr.particle_diffusivity_He_m2_s, self.geometry, 0.0)

            # Energy carried by particle diffusion plus separate conductive heat transport.
            ion_particle_flux = flux_D + flux_T + flux_He
            electron_particle_flux = flux_D + flux_T + 2.0 * flux_He
            ion_convective_W = convective_energy_flux_from_particle_flux(ion_particle_flux, ti, self.geometry, tr.edge_ion_temperature_keV)
            electron_convective_W = convective_energy_flux_from_particle_flux(electron_particle_flux, te, self.geometry, tr.edge_electron_temperature_keV)
            ion_conductive_W = conductive_heat_flux(ni, ti, tr.ion_thermal_diffusivity_m2_s, self.geometry, tr.edge_ion_temperature_keV)
            electron_conductive_W = conductive_heat_flux(ne, te, tr.electron_thermal_diffusivity_m2_s, self.geometry, tr.edge_electron_temperature_keV)
            ion_face_W = ion_convective_W + ion_conductive_W
            electron_face_W = electron_convective_W + electron_conductive_W

            d_step = conservative_density_step(s.n_D_m3, flux_D, self.geometry, dt)
            t_step = conservative_density_step(s.n_T_m3, flux_T, self.geometry, dt)
            he_step = conservative_density_step(s.n_He_m3, flux_He, self.geometry, dt)
            ui_step = conservative_energy_step(s.ion_energy_J_m3, ion_face_W, self.geometry, dt)
            ue_step = conservative_energy_step(s.electron_energy_J_m3, electron_face_W, self.geometry, dt)
            s.n_D_m3, s.n_T_m3, s.n_He_m3 = d_step.values, t_step.values, he_step.values
            s.ion_energy_J_m3, s.electron_energy_J_m3 = ui_step.values, ue_step.values

            D_edge = float(flux_D[-1]) * dt
            T_edge = float(flux_T[-1]) * dt
            He_edge = float(flux_He[-1]) * dt
            cumulative_D_edge += D_edge
            cumulative_T_edge += T_edge
            cumulative_He_edge += He_edge
            edge_energy_J = float(ion_face_W[-1] + electron_face_W[-1]) * dt
            cumulative_transport_J += edge_energy_J

            # --- Heating, alpha deposition, electron-ion exchange, radiation ---
            ni, ne, ti, te = self._densities_temperatures()
            exchange_W_m3 = electron_ion_exchange_W_m3(ne, ti, te, cfg.coupling.electron_ion_equilibration_time_s)
            brem_W_m3 = bremsstrahlung_W_m3(ne, te, cfg.radiation.zeff)
            impurity_W_m3 = impurity_line_radiation_W_m3(
                ne, cfg.radiation.impurity_fraction, cfg.radiation.impurity_cooling_coefficient_W_m3
            )
            radiation_W_m3 = brem_W_m3 + impurity_W_m3
            alpha_deposit_W_m3 = deposited_alpha_J_m3 / dt
            ext_e = cfg.heating.electron_fraction * self._external_profile_W_m3
            ext_i = (1.0 - cfg.heating.electron_fraction) * self._external_profile_W_m3
            alpha_e = cfg.alpha.electron_heating_fraction * alpha_deposit_W_m3
            alpha_i = (1.0 - cfg.alpha.electron_heating_fraction) * alpha_deposit_W_m3

            next_ue = s.electron_energy_J_m3 + dt * (ext_e + alpha_e + exchange_W_m3 - radiation_W_m3)
            next_ui = s.ion_energy_J_m3 + dt * (ext_i + alpha_i - exchange_W_m3)
            if np.any(next_ue < -1e-12) or np.any(next_ui < -1e-12):
                raise SimulationValidityError("thermal source/loss update produced negative energy; reduce dt or loss strength")
            s.electron_energy_J_m3 = np.maximum(next_ue, 0.0)
            s.ion_energy_J_m3 = np.maximum(next_ui, 0.0)

            external_J = cfg.heating.external_heating_MW * 1.0e6 * dt
            radiation_J = self._integral(radiation_W_m3) * dt
            cumulative_external_J += external_J
            cumulative_radiation_J += radiation_J

            # --- Instantaneous observables derive from this same causal state ---
            reaction_rate_s = self._integral(rate_density)
            ledger = reaction_ledger_from_rate(reaction_rate_s)
            alpha_deposit_MW = self._integral(alpha_deposit_W_m3) / 1.0e6
            alpha_escape_MW = self._integral(escaped_alpha_J_m3) / dt / 1.0e6
            radiation_MW = self._integral(radiation_W_m3) / 1.0e6
            transport_MW = float(ion_face_W[-1] + electron_face_W[-1]) / 1.0e6
            thermal_energy_J = self._integral(s.ion_energy_J_m3 + s.electron_energy_J_m3)
            thermal_loss_MW = radiation_MW + max(transport_MW, 0.0)
            tau_E_s = thermal_energy_J / (thermal_loss_MW * 1e6) if thermal_loss_MW > 0 else None
            q_plasma = (
                ledger.fusion_power_MW / cfg.heating.external_heating_MW
                if cfg.heating.external_heating_MW > 0
                else None
            )
            alpha_heating_fraction = (
                alpha_deposit_MW / (alpha_deposit_MW + cfg.heating.external_heating_MW)
                if alpha_deposit_MW + cfg.heating.external_heating_MW > 0
                else 0.0
            )
            ni_now, ne_now, ti_now, te_now = self._densities_temperatures()
            lawson = self._volume_average(ne_now * ti_now) * tau_E_s if tau_E_s is not None else None
            final_metrics = {
                "time_s": t,
                "fusion_power_MW": ledger.fusion_power_MW,
                "alpha_birth_power_MW": ledger.alpha_power_MW,
                "neutron_power_MW": ledger.neutron_power_MW,
                "alpha_deposition_power_MW": alpha_deposit_MW,
                "alpha_escape_power_MW": alpha_escape_MW,
                "external_heating_MW": cfg.heating.external_heating_MW,
                "radiation_loss_MW": radiation_MW,
                "transport_edge_loss_MW": transport_MW,
                "Q_plasma": q_plasma,
                "alpha_heating_fraction": alpha_heating_fraction,
                "tau_E_s": tau_E_s,
                "lawson_nTtau_keV_s_m3": lawson,
                "volume_avg_Ti_keV": self._volume_average(ti_now),
                "volume_avg_Te_keV": self._volume_average(te_now),
                "volume_avg_ne_m3": self._volume_average(ne_now),
                "fast_alpha_energy_MJ": self._integral(s.alpha_fast_energy_J_m3) / 1e6,
                "helium_ash_inventory": self._integral(s.n_He_m3),
            }
            if step % record_stride == 0 or step == cfg.n_steps - 1:
                timeseries.append(dict(final_metrics))

        final_energy = self._integral(s.ion_energy_J_m3 + s.electron_energy_J_m3 + s.alpha_fast_energy_J_m3)
        delta_energy = final_energy - initial_energy
        expected_delta = (
            cumulative_external_J
            + cumulative_alpha_birth_J
            - cumulative_radiation_J
            - cumulative_transport_J
            - cumulative_alpha_escape_J
            - cumulative_fusion_thermal_J
            - cumulative_ambipolar_J
        )
        energy_error_J = delta_energy - expected_delta
        energy_scale = max(
            abs(delta_energy),
            abs(cumulative_external_J) + abs(cumulative_alpha_birth_J) + abs(cumulative_radiation_J)
            + abs(cumulative_transport_J) + abs(cumulative_alpha_escape_J) + abs(cumulative_fusion_thermal_J)
            + abs(cumulative_ambipolar_J),
            1.0,
        )
        energy_relative_residual = energy_error_J / energy_scale

        final_D = self._integral(s.n_D_m3)
        final_T = self._integral(s.n_T_m3)
        final_He = self._integral(s.n_He_m3)
        final_fast = self._integral(s.n_alpha_fast_m3)
        D_error = final_D - (initial_D - cumulative_reactions - cumulative_D_edge)
        T_error = final_T - (initial_T - cumulative_reactions - cumulative_T_edge)
        He_error = (final_He + final_fast + cumulative_alpha_escape + cumulative_He_edge) - cumulative_reactions
        particle_scale = max(initial_D, initial_T, cumulative_reactions, 1.0)
        particle_relative_residual = max(abs(D_error), abs(T_error), abs(He_error)) / particle_scale

        # Use final instantaneous source for reactor-level source-term constraints.
        final_rate = final_metrics["fusion_power_MW"] * 1.0e6 / DT_TOTAL_ENERGY_J
        breeding = breeding_constraint(final_rate, cfg.plant.global_tbr_target, cfg.plant.blanket_coverage_fraction)
        plant = close_plant_power(
            final_metrics["fusion_power_MW"],
            cfg.plant.blanket_energy_multiplier,
            cfg.plant.gross_thermal_efficiency,
            cfg.plant.recirculating_components_MW,
        )
        final_pressure_Pa = (2.0 / 3.0) * self._volume_average(s.ion_energy_J_m3 + s.electron_energy_J_m3)
        envelope = operating_envelope(
            geometry=cfg.geometry,
            volume_average_pressure_Pa=final_pressure_Pa,
            volume_average_electron_density_m3=final_metrics["volume_avg_ne_m3"],
        )
        gates = evaluate_claim_gates(
            energy_residual=energy_relative_residual,
            particle_residual=particle_relative_residual,
            energy_limit=cfg.verification.max_energy_balance_relative_residual,
            particle_limit=cfg.verification.max_particle_balance_relative_residual,
            q_plasma=final_metrics["Q_plasma"],
            alpha_heating_fraction=final_metrics["alpha_heating_fraction"],
            external_heating_MW=cfg.heating.external_heating_MW,
            plant_closure_complete=plant.closure_complete,
        )
        summary = {
            **final_metrics,
            "duration_s": cfg.duration_s,
            "steps": cfg.n_steps,
            "plasma_volume_m3": self.geometry.total_volume_m3,
            "energy_balance_relative_residual": energy_relative_residual,
            "energy_balance_error_J": energy_error_J,
            "particle_balance_relative_residual": particle_relative_residual,
            "cumulative_DT_reactions": cumulative_reactions,
            "cumulative_external_energy_MJ": cumulative_external_J / 1e6,
            "cumulative_alpha_birth_energy_MJ": cumulative_alpha_birth_J / 1e6,
            "cumulative_radiation_loss_MJ": cumulative_radiation_J / 1e6,
            "cumulative_transport_net_out_MJ": cumulative_transport_J / 1e6,
            "cumulative_alpha_escape_MJ": cumulative_alpha_escape_J / 1e6,
            "deuterium_edge_net_out": cumulative_D_edge,
            "tritium_edge_net_out": cumulative_T_edge,
            "helium_edge_net_out": cumulative_He_edge,
            "authority": gates.highest_authority,
            "experimental_validation": False,
            "model_boundary": "1.5D circular-torus reduced transport + Maxwellian Bosch-Hale D-T burn; not predictive tokamak validation",
            "breeding_constraint": asdict(breeding),
            "plant_closure": asdict(plant),
            "operating_envelope": asdict(envelope),
            "claim_gates": gates.to_dict(),
        }
        return SimulationResult(summary, timeseries, gates, breeding, plant, s.copy())
