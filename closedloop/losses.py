"""Reduced-order radiation and collisional coupling models with declared authority."""
from __future__ import annotations

import numpy as np

from .constants import KEV_TO_J

# Common nonrelativistic fusion-plasma bremsstrahlung screening coefficient.
_BREM_COEFF = 5.35e-37  # W m^3 / sqrt(keV)


def bremsstrahlung_W_m3(electron_density_m3: np.ndarray, electron_temperature_keV: np.ndarray, zeff: float) -> np.ndarray:
    ne = np.asarray(electron_density_m3, dtype=float)
    te = np.asarray(electron_temperature_keV, dtype=float)
    if zeff < 1 or np.any(ne < 0) or np.any(te < 0):
        raise ValueError("invalid bremsstrahlung inputs")
    return _BREM_COEFF * zeff * ne**2 * np.sqrt(te)


def impurity_line_radiation_W_m3(
    electron_density_m3: np.ndarray,
    impurity_fraction: float,
    cooling_coefficient_W_m3: float,
) -> np.ndarray:
    """Simple n_e*n_z*Lz loss channel for user-supplied effective Lz."""
    ne = np.asarray(electron_density_m3, dtype=float)
    if impurity_fraction < 0 or cooling_coefficient_W_m3 < 0:
        raise ValueError("invalid impurity radiation inputs")
    nz = impurity_fraction * ne
    return ne * nz * cooling_coefficient_W_m3


def electron_ion_exchange_W_m3(
    electron_density_m3: np.ndarray,
    ion_temperature_keV: np.ndarray,
    electron_temperature_keV: np.ndarray,
    equilibration_time_s: float,
) -> np.ndarray:
    """Conservative relaxation closure. Positive power flows ions -> electrons."""
    if equilibration_time_s <= 0:
        raise ValueError("equilibration_time_s must be positive")
    ne = np.asarray(electron_density_m3, dtype=float)
    ti = np.asarray(ion_temperature_keV, dtype=float)
    te = np.asarray(electron_temperature_keV, dtype=float)
    return 1.5 * ne * KEV_TO_J * (ti - te) / equilibration_time_s
