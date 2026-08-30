"""Vectorized vacuum-orbit alpha screen using the Boris magnetic pusher.

This is deliberately a vacuum-orbit screen: no collisions, electric fields, ripple,
MHD perturbations, or realistic shaped equilibrium are claimed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, sqrt

import numpy as np

from .constants import DT_ALPHA_ENERGY_J, E_CHARGE_C, MASS_ALPHA_KG, MU0_H_M
from .models import TokamakGeometryConfig


@dataclass(frozen=True)
class AlphaOrbitScreen:
    markers: int
    steps: int
    dt_s: float
    simulated_time_s: float
    retained_fraction: float
    lost_markers: int
    mean_loss_time_s: float | None
    relative_kinetic_energy_drift_max: float
    authority: str = "analytic_field_vacuum_alpha_orbit_screen_not_fast_ion_transport_prediction"


def _magnetic_field_xyz(position_m: np.ndarray, geometry: TokamakGeometryConfig) -> np.ndarray:
    x, y, z = position_m[:, 0], position_m[:, 1], position_m[:, 2]
    R = np.sqrt(x * x + y * y)
    R_safe = np.maximum(R, 1e-9)
    eR = np.column_stack((x / R_safe, y / R_safe, np.zeros_like(R)))
    ephi = np.column_stack((-y / R_safe, x / R_safe, np.zeros_like(R)))
    dR = R - geometry.major_radius_m
    rminor = np.sqrt(dR * dR + z * z)
    r_safe = np.maximum(rminor, 1e-12)
    # Poloidal unit vector tangent to circles around the magnetic axis in R-Z plane.
    etheta = (-z / r_safe)[:, None] * eR
    etheta[:, 2] += dR / r_safe
    Btor = geometry.toroidal_field_T * geometry.major_radius_m / R_safe
    Bp_edge = MU0_H_M * geometry.plasma_current_MA * 1e6 / (2.0 * pi * geometry.minor_radius_m)
    Bpol = Bp_edge * np.minimum(rminor / geometry.minor_radius_m, 1.5)
    return Btor[:, None] * ephi + Bpol[:, None] * etheta


def _boris_magnetic_step(position: np.ndarray, velocity: np.ndarray, dt_s: float, geometry: TokamakGeometryConfig) -> tuple[np.ndarray, np.ndarray]:
    B = _magnetic_field_xyz(position, geometry)
    q_over_m = 2.0 * E_CHARGE_C / MASS_ALPHA_KG
    t = q_over_m * B * (0.5 * dt_s)
    t2 = np.sum(t * t, axis=1)[:, None]
    s = 2.0 * t / (1.0 + t2)
    v_prime = velocity + np.cross(velocity, t)
    v_plus = velocity + np.cross(v_prime, s)
    return position + v_plus * dt_s, v_plus


def alpha_orbit_screen(
    geometry: TokamakGeometryConfig,
    *,
    markers: int = 256,
    steps: int = 1200,
    dt_s: float = 2.0e-10,
    seed: int = 17,
    birth_rho_max: float = 0.85,
) -> AlphaOrbitScreen:
    if markers < 1 or steps < 1 or dt_s <= 0 or not 0 < birth_rho_max < 1:
        raise ValueError("invalid alpha orbit screen inputs")
    geometry.validate()
    rng = np.random.default_rng(seed)
    rho = birth_rho_max * np.sqrt(rng.random(markers))
    theta = rng.uniform(0.0, 2.0 * pi, markers)
    phi = rng.uniform(0.0, 2.0 * pi, markers)
    rminor = rho * geometry.minor_radius_m
    R = geometry.major_radius_m + rminor * np.cos(theta)
    z = rminor * np.sin(theta)
    pos = np.column_stack((R * np.cos(phi), R * np.sin(phi), z))

    speed = sqrt(2.0 * DT_ALPHA_ENERGY_J / MASS_ALPHA_KG)
    direction = rng.normal(size=(markers, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    vel = speed * direction
    initial_ke = 0.5 * MASS_ALPHA_KG * np.sum(vel * vel, axis=1)
    alive = np.ones(markers, dtype=bool)
    loss_time = np.full(markers, np.nan)
    max_drift = 0.0

    for step in range(steps):
        if not np.any(alive):
            break
        idx = np.where(alive)[0]
        p_new, v_new = _boris_magnetic_step(pos[idx], vel[idx], dt_s, geometry)
        pos[idx], vel[idx] = p_new, v_new
        R_now = np.sqrt(pos[idx, 0] ** 2 + pos[idx, 1] ** 2)
        minor_now = np.sqrt((R_now - geometry.major_radius_m) ** 2 + pos[idx, 2] ** 2)
        lost_local = minor_now > geometry.minor_radius_m
        if np.any(lost_local):
            lost_idx = idx[lost_local]
            alive[lost_idx] = False
            loss_time[lost_idx] = (step + 1) * dt_s
        current_ke = 0.5 * MASS_ALPHA_KG * np.sum(vel[idx] * vel[idx], axis=1)
        drift = np.max(np.abs(current_ke - initial_ke[idx]) / initial_ke[idx])
        max_drift = max(max_drift, float(drift))

    lost = int(np.sum(~alive))
    mean_loss = float(np.nanmean(loss_time)) if lost else None
    return AlphaOrbitScreen(
        markers=markers,
        steps=steps,
        dt_s=dt_s,
        simulated_time_s=steps * dt_s,
        retained_fraction=float(np.mean(alive)),
        lost_markers=lost,
        mean_loss_time_s=mean_loss,
        relative_kinetic_energy_drift_max=max_drift,
    )


def as_jsonable(value: AlphaOrbitScreen) -> dict[str, object]:
    return asdict(value)
