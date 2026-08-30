"""ClosedLoop: verification-first reduced-order D-T tokamak burn simulator.

This package is intentionally separate from the repository's legacy particle demo.
It provides a causally closed reduced-order burn/transport model with explicit
particle and energy accounting. It is not experimentally validated reactor software.
"""
from .fusion import bosch_hale_dt_reactivity
from .models import SimulationConfig
from .simulation import ClosedLoopSimulator, SimulationResult

__all__ = [
    "ClosedLoopSimulator",
    "SimulationConfig",
    "SimulationResult",
    "bosch_hale_dt_reactivity",
]
