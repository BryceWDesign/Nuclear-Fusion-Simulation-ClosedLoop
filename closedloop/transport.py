"""Conservative finite-volume particle and heat transport operators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import KEV_TO_J
from .geometry import RadialGeometry


class TransportStabilityError(RuntimeError):
    """Raised when an explicit transport step would create a negative conserved state."""


@dataclass(frozen=True)
class TransportStep:
    values: np.ndarray
    outward_face_flux: np.ndarray
    edge_outflow_per_s: float


def explicit_diffusion_dt_limit(diffusivity_m2_s: float, dr_m: float) -> float:
    if diffusivity_m2_s <= 0:
        return float("inf")
    return 0.45 * dr_m**2 / diffusivity_m2_s


def diffusive_particle_flux(
    density_m3: np.ndarray,
    diffusivity_m2_s: float,
    geometry: RadialGeometry,
    edge_density_m3: float = 0.0,
) -> np.ndarray:
    """Outward particle flux across each radial face [particles/s]."""
    n = np.asarray(density_m3, dtype=float)
    if len(n) != len(geometry.r_centers_m):
        raise ValueError("density length does not match radial geometry")
    if np.any(n < 0) or diffusivity_m2_s < 0 or edge_density_m3 < 0:
        raise ValueError("invalid density/diffusivity inputs")
    flux = np.zeros(len(n) + 1, dtype=float)
    if diffusivity_m2_s == 0:
        return flux
    dr = geometry.dr_m
    # Internal faces. Positive means outward from cell i-1 to i.
    gradients = (n[1:] - n[:-1]) / dr
    flux[1:-1] = -diffusivity_m2_s * geometry.face_areas_m2[1:-1] * gradients
    # Symmetry at magnetic axis, Dirichlet edge reservoir at r=a.
    edge_gradient = (edge_density_m3 - n[-1]) / (0.5 * dr)
    flux[-1] = -diffusivity_m2_s * geometry.face_areas_m2[-1] * edge_gradient
    return flux


def conservative_density_step(
    density_m3: np.ndarray,
    outward_face_flux: np.ndarray,
    geometry: RadialGeometry,
    dt_s: float,
) -> TransportStep:
    n = np.asarray(density_m3, dtype=float)
    flux = np.asarray(outward_face_flux, dtype=float)
    if flux.shape != (len(n) + 1,):
        raise ValueError("face flux has wrong shape")
    particle_number = n * geometry.shell_volumes_m3
    updated = particle_number + dt_s * (flux[:-1] - flux[1:])
    tol = max(float(np.max(particle_number)), 1.0) * 1e-13
    if float(np.min(updated)) < -tol:
        raise TransportStabilityError("particle transport step became negative; reduce dt or transport coefficient")
    updated = np.maximum(updated, 0.0)
    return TransportStep(
        values=updated / geometry.shell_volumes_m3,
        outward_face_flux=flux,
        edge_outflow_per_s=max(float(flux[-1]), 0.0),
    )


def temperature_keV_from_energy(energy_J_m3: np.ndarray, number_density_m3: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy_J_m3, dtype=float)
    density = np.asarray(number_density_m3, dtype=float)
    denom = 1.5 * density * KEV_TO_J
    result = np.zeros_like(energy)
    mask = denom > 0
    result[mask] = energy[mask] / denom[mask]
    return result


def convective_energy_flux_from_particle_flux(
    particle_flux_per_s: np.ndarray,
    temperature_keV: np.ndarray,
    geometry: RadialGeometry,
    edge_temperature_keV: float,
) -> np.ndarray:
    """Thermal energy carried by diffusing particles across faces [W]."""
    particle_flux = np.asarray(particle_flux_per_s, dtype=float)
    T = np.asarray(temperature_keV, dtype=float)
    face_T = np.zeros(len(T) + 1, dtype=float)
    face_T[0] = T[0]
    face_T[1:-1] = 0.5 * (T[:-1] + T[1:])
    face_T[-1] = 0.5 * (T[-1] + edge_temperature_keV)
    return particle_flux * (1.5 * face_T * KEV_TO_J)


def conductive_heat_flux(
    number_density_m3: np.ndarray,
    temperature_keV: np.ndarray,
    thermal_diffusivity_m2_s: float,
    geometry: RadialGeometry,
    edge_temperature_keV: float,
) -> np.ndarray:
    """Outward reduced-order conductive heat flux across faces [W].

    Uses q = -(3/2) n chi d(kT)/dr. This is a declared diffusive heat model,
    not a gyrokinetic/turbulent transport closure.
    """
    n = np.asarray(number_density_m3, dtype=float)
    T = np.asarray(temperature_keV, dtype=float)
    if thermal_diffusivity_m2_s < 0:
        raise ValueError("thermal diffusivity cannot be negative")
    flux = np.zeros(len(T) + 1, dtype=float)
    if thermal_diffusivity_m2_s == 0:
        return flux
    n_face = np.zeros(len(T) + 1, dtype=float)
    n_face[0] = n[0]
    n_face[1:-1] = 0.5 * (n[:-1] + n[1:])
    n_face[-1] = n[-1]
    dT = (T[1:] - T[:-1]) / geometry.dr_m
    flux[1:-1] = -1.5 * n_face[1:-1] * thermal_diffusivity_m2_s * KEV_TO_J * geometry.face_areas_m2[1:-1] * dT
    edge_grad = (edge_temperature_keV - T[-1]) / (0.5 * geometry.dr_m)
    flux[-1] = -1.5 * n_face[-1] * thermal_diffusivity_m2_s * KEV_TO_J * geometry.face_areas_m2[-1] * edge_grad
    return flux


def conservative_energy_step(
    energy_J_m3: np.ndarray,
    outward_face_power_W: np.ndarray,
    geometry: RadialGeometry,
    dt_s: float,
) -> TransportStep:
    u = np.asarray(energy_J_m3, dtype=float)
    power = np.asarray(outward_face_power_W, dtype=float)
    total_cell_energy = u * geometry.shell_volumes_m3
    updated = total_cell_energy + dt_s * (power[:-1] - power[1:])
    tol = max(float(np.max(total_cell_energy)), 1.0) * 1e-13
    if float(np.min(updated)) < -tol:
        raise TransportStabilityError("heat transport step became negative; reduce dt or thermal diffusivity")
    updated = np.maximum(updated, 0.0)
    return TransportStep(
        values=updated / geometry.shell_volumes_m3,
        outward_face_flux=power,
        edge_outflow_per_s=max(float(power[-1]), 0.0),
    )
