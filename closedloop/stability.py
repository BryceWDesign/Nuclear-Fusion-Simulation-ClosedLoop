"""Tokamak operating-envelope screens, not nonlinear MHD predictions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi

from .constants import MU0_H_M
from .models import TokamakGeometryConfig


@dataclass(frozen=True)
class OperatingEnvelope:
    volume_average_pressure_Pa: float
    beta_fraction: float
    beta_normalized: float
    greenwald_density_limit_m3: float
    greenwald_fraction: float
    cylindrical_edge_q: float
    betaN_screen_pass: bool
    greenwald_screen_pass: bool
    q_screen_pass: bool
    combined_screen_pass: bool
    authority: str = "reduced_operating_envelope_screen_not_mhd_stability_proof"


def operating_envelope(
    *,
    geometry: TokamakGeometryConfig,
    volume_average_pressure_Pa: float,
    volume_average_electron_density_m3: float,
    betaN_limit: float = 3.5,
    greenwald_fraction_limit: float = 0.9,
    q_edge_minimum: float = 2.0,
) -> OperatingEnvelope:
    if volume_average_pressure_Pa < 0 or volume_average_electron_density_m3 < 0:
        raise ValueError("pressure and density cannot be negative")
    geometry.validate()
    magnetic_pressure = geometry.toroidal_field_T**2 / (2.0 * MU0_H_M)
    beta = volume_average_pressure_Pa / magnetic_pressure
    betaN = (100.0 * beta) * geometry.minor_radius_m * geometry.toroidal_field_T / geometry.plasma_current_MA
    n_greenwald = geometry.plasma_current_MA / (pi * geometry.minor_radius_m**2) * 1.0e20
    f_greenwald = volume_average_electron_density_m3 / n_greenwald
    current_A = geometry.plasma_current_MA * 1.0e6
    q_circular = 2.0 * pi * geometry.minor_radius_m**2 * geometry.toroidal_field_T / (
        MU0_H_M * geometry.major_radius_m * current_A
    )
    # First-order elongation correction for a shaped cross-section. This remains a
    # screening estimate, not a solved q-profile.
    q_edge = q_circular * 0.5 * (1.0 + geometry.elongation**2)
    beta_ok = betaN <= betaN_limit
    greenwald_ok = f_greenwald <= greenwald_fraction_limit
    q_ok = q_edge >= q_edge_minimum
    return OperatingEnvelope(
        volume_average_pressure_Pa=volume_average_pressure_Pa,
        beta_fraction=beta,
        beta_normalized=betaN,
        greenwald_density_limit_m3=n_greenwald,
        greenwald_fraction=f_greenwald,
        cylindrical_edge_q=q_edge,
        betaN_screen_pass=beta_ok,
        greenwald_screen_pass=greenwald_ok,
        q_screen_pass=q_ok,
        combined_screen_pass=beta_ok and greenwald_ok and q_ok,
    )


def as_jsonable(value: OperatingEnvelope) -> dict[str, object]:
    return asdict(value)
