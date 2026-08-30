"""D-T Maxwellian reactivity and exact reaction-source bookkeeping."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, sqrt

import numpy as np

from .constants import (
    DT_ALPHA_ENERGY_J,
    DT_NEUTRON_ENERGY_J,
    DT_TOTAL_ENERGY_J,
    MASS_DEUTERIUM_KG,
    MASS_TRITIUM_KG,
    SECONDS_PER_DAY,
)


@dataclass(frozen=True)
class DTReactionLedger:
    reaction_rate_per_s: float
    fusion_power_MW: float
    alpha_power_MW: float
    neutron_power_MW: float
    deuterium_burn_kg_day: float
    tritium_burn_kg_day: float


def _bosch_hale_scalar(temperature_keV: float) -> float:
    if not 0.2 <= temperature_keV <= 100.0:
        raise ValueError("Bosch-Hale D-T fit is restricted to 0.2-100 keV")
    c1, c2, c3, c4 = 1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3
    c5, c6, c7 = 1.35000e-2, -1.06750e-4, 1.36600e-5
    bg, mrc2 = 34.3827, 1_124_656.0
    t = temperature_keV
    theta = t / (1.0 - t * (c2 + t * (c4 + t * c6)) / (1.0 + t * (c3 + t * (c5 + t * c7))))
    xi = (bg * bg / (4.0 * theta)) ** (1.0 / 3.0)
    return c1 * theta * sqrt(xi / (mrc2 * t**3)) * exp(-3.0 * xi) * 1.0e-6


def bosch_hale_dt_reactivity(temperature_keV: float | np.ndarray) -> float | np.ndarray:
    """Return Maxwellian D-T <sigma v> [m^3/s], Bosch & Hale 1992.

    The implementation refuses extrapolation outside the fit interval.
    """
    if np.isscalar(temperature_keV):
        return _bosch_hale_scalar(float(temperature_keV))
    arr = np.asarray(temperature_keV, dtype=float)
    if np.any((arr < 0.2) | (arr > 100.0)):
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        raise ValueError(f"Bosch-Hale D-T fit requires 0.2-100 keV; observed {lo:g}-{hi:g} keV")
    out = np.empty_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for value in it:
        out[it.multi_index] = _bosch_hale_scalar(float(value))
    return out


def reaction_rate_density(n_D_m3: np.ndarray, n_T_m3: np.ndarray, ion_temperature_keV: np.ndarray) -> np.ndarray:
    if np.any(n_D_m3 < 0) or np.any(n_T_m3 < 0):
        raise ValueError("fuel densities cannot be negative")
    temperature = np.asarray(ion_temperature_keV, dtype=float)
    if np.any(temperature > 100.0):
        raise ValueError("ion temperature exceeded the 100 keV Bosch-Hale authority boundary")
    # Cells colder than the published 0.2 keV fit boundary are assigned zero D-T
    # burn rather than extrapolating the fit into an unvalidated regime.
    active = temperature >= 0.2
    reactivity = np.zeros_like(temperature)
    if np.any(active):
        reactivity[active] = np.asarray(bosch_hale_dt_reactivity(temperature[active]))
    return n_D_m3 * n_T_m3 * reactivity


def reaction_ledger_from_rate(reaction_rate_per_s: float) -> DTReactionLedger:
    if reaction_rate_per_s < 0:
        raise ValueError("reaction rate cannot be negative")
    return DTReactionLedger(
        reaction_rate_per_s=reaction_rate_per_s,
        fusion_power_MW=reaction_rate_per_s * DT_TOTAL_ENERGY_J / 1.0e6,
        alpha_power_MW=reaction_rate_per_s * DT_ALPHA_ENERGY_J / 1.0e6,
        neutron_power_MW=reaction_rate_per_s * DT_NEUTRON_ENERGY_J / 1.0e6,
        deuterium_burn_kg_day=reaction_rate_per_s * MASS_DEUTERIUM_KG * SECONDS_PER_DAY,
        tritium_burn_kg_day=reaction_rate_per_s * MASS_TRITIUM_KG * SECONDS_PER_DAY,
    )


def as_jsonable(ledger: DTReactionLedger) -> dict[str, float]:
    return asdict(ledger)
