"""Small Grad-Shafranov finite-difference solver with manufactured verification.

The physical mode uses constant p'(psi) and F F'(psi), i.e. a Solovev-class source.
It is a reduced equilibrium model, not EFIT/FreeGSNKE replacement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MU0_H_M


@dataclass(frozen=True)
class GradShafranovResult:
    R_m: np.ndarray
    Z_m: np.ndarray
    psi_Wb_rad: np.ndarray
    source: np.ndarray
    iterations: int
    max_update: float
    residual_linf: float


def grad_shafranov_operator(psi: np.ndarray, R: np.ndarray, dR: float, dZ: float) -> np.ndarray:
    op = np.zeros_like(psi)
    d2R = (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dR**2
    dR1 = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dR)
    d2Z = (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dZ**2
    op[1:-1, 1:-1] = d2R - dR1 / R[1:-1, 1:-1] + d2Z
    return op


def solve_grad_shafranov_source(
    R_min_m: float,
    R_max_m: float,
    Z_min_m: float,
    Z_max_m: float,
    nR: int,
    nZ: int,
    source: np.ndarray,
    omega: float = 1.6,
    tolerance: float = 1e-9,
    max_iterations: int = 50_000,
) -> GradShafranovResult:
    if nR < 8 or nZ < 8 or R_min_m <= 0 or R_max_m <= R_min_m or Z_max_m <= Z_min_m:
        raise ValueError("invalid Grad-Shafranov grid")
    r = np.linspace(R_min_m, R_max_m, nR)
    z = np.linspace(Z_min_m, Z_max_m, nZ)
    R, Z = np.meshgrid(r, z, indexing="ij")
    if source.shape != (nR, nZ):
        raise ValueError("source shape does not match grid")
    dR = r[1] - r[0]
    dZ = z[1] - z[0]
    psi = np.zeros((nR, nZ), dtype=float)
    inv = 1.0 / (2.0 / dR**2 + 2.0 / dZ**2)
    max_update = float("inf")
    for iteration in range(1, max_iterations + 1):
        max_update = 0.0
        for i in range(1, nR - 1):
            Ri = r[i]
            c_plus = 1.0 / dR**2 - 1.0 / (2.0 * Ri * dR)
            c_minus = 1.0 / dR**2 + 1.0 / (2.0 * Ri * dR)
            for j in range(1, nZ - 1):
                candidate = inv * (
                    c_plus * psi[i + 1, j]
                    + c_minus * psi[i - 1, j]
                    + (psi[i, j + 1] + psi[i, j - 1]) / dZ**2
                    - source[i, j]
                )
                delta = omega * (candidate - psi[i, j])
                psi[i, j] += delta
                max_update = max(max_update, abs(delta))
        if max_update < tolerance:
            break
    residual = grad_shafranov_operator(psi, R, dR, dZ) - source
    residual_linf = float(np.max(np.abs(residual[1:-1, 1:-1])))
    return GradShafranovResult(R, Z, psi, np.array(source, copy=True), iteration, max_update, residual_linf)


def solve_solovev_class_equilibrium(
    R0_m: float = 6.2,
    minor_radius_m: float = 2.0,
    nR: int = 65,
    nZ: int = 65,
    pressure_derivative_Pa_per_Wb: float = -1.0e6,
    ff_derivative_T2m2_per_Wb: float = 0.0,
) -> GradShafranovResult:
    r = np.linspace(R0_m - minor_radius_m, R0_m + minor_radius_m, nR)
    z = np.linspace(-minor_radius_m, minor_radius_m, nZ)
    R, _ = np.meshgrid(r, z, indexing="ij")
    source = -MU0_H_M * R**2 * pressure_derivative_Pa_per_Wb - ff_derivative_T2m2_per_Wb
    return solve_grad_shafranov_source(r[0], r[-1], z[0], z[-1], nR, nZ, source)


def manufactured_solution_error(nR: int = 41, nZ: int = 41) -> tuple[float, GradShafranovResult]:
    R_min, R_max, Z_min, Z_max = 4.0, 8.0, -2.0, 2.0
    r = np.linspace(R_min, R_max, nR)
    z = np.linspace(Z_min, Z_max, nZ)
    R, Z = np.meshgrid(r, z, indexing="ij")
    f = (R - R_min) * (R_max - R)
    g = (Z - Z_min) * (Z_max - Z)
    fprime = (R_min + R_max) - 2.0 * R
    exact = f * g
    source = (-2.0 * g) - (fprime * g / R) + (-2.0 * f)
    result = solve_grad_shafranov_source(R_min, R_max, Z_min, Z_max, nR, nZ, source, tolerance=1e-10)
    interior = (slice(1, -1), slice(1, -1))
    denom = float(np.linalg.norm(exact[interior]))
    error = float(np.linalg.norm((result.psi_Wb_rad - exact)[interior]) / denom)
    return error, result
