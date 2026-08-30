"""Axisymmetric circular-torus finite-volume geometry."""
from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from .models import TokamakGeometryConfig


@dataclass(frozen=True)
class RadialGeometry:
    r_edges_m: np.ndarray
    r_centers_m: np.ndarray
    rho_centers: np.ndarray
    shell_volumes_m3: np.ndarray
    face_areas_m2: np.ndarray
    dr_m: float
    total_volume_m3: float
    plasma_surface_area_m2: float


def build_radial_geometry(config: TokamakGeometryConfig, n_cells: int) -> RadialGeometry:
    config.validate()
    edges = np.linspace(0.0, config.minor_radius_m, n_cells + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr = config.minor_radius_m / n_cells
    # Circular tokamak approximation: V(r)=2*pi^2*R*r^2 and A_flux(r)=dV/dr.
    volumes = 2.0 * pi**2 * config.major_radius_m * (edges[1:] ** 2 - edges[:-1] ** 2)
    areas = 4.0 * pi**2 * config.major_radius_m * edges
    total = float(np.sum(volumes))
    surface = 4.0 * pi**2 * config.major_radius_m * config.minor_radius_m
    return RadialGeometry(
        r_edges_m=edges,
        r_centers_m=centers,
        rho_centers=centers / config.minor_radius_m,
        shell_volumes_m3=volumes,
        face_areas_m2=areas,
        dr_m=dr,
        total_volume_m3=total,
        plasma_surface_area_m2=surface,
    )
