"""Executable resolution/time-step convergence campaign."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .models import SimulationConfig
from .simulation import ClosedLoopSimulator


@dataclass(frozen=True)
class ConvergenceComparison:
    coarse: dict[str, float | None]
    temporal_refined: dict[str, float | None]
    spatial_refined: dict[str, float | None]
    relative_differences: dict[str, dict[str, float | None]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _metrics(config: SimulationConfig) -> dict[str, float | None]:
    summary = ClosedLoopSimulator(config).run().summary
    keys = ("fusion_power_MW", "Q_plasma", "tau_E_s", "volume_avg_Ti_keV", "volume_avg_Te_keV")
    return {k: summary[k] for k in keys}


def _rel(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    scale = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / scale


def run_convergence_campaign(config: SimulationConfig) -> ConvergenceComparison:
    coarse = _metrics(config)
    temporal_cfg = replace(config, dt_s=config.dt_s / 2.0)
    spatial_cfg = replace(config, n_cells=config.n_cells * 2, dt_s=config.dt_s / 2.0)
    temporal = _metrics(temporal_cfg)
    spatial = _metrics(spatial_cfg)
    diffs = {
        key: {"temporal": _rel(coarse[key], temporal[key]), "spatial": _rel(temporal[key], spatial[key])}
        for key in coarse
    }
    return ConvergenceComparison(coarse, temporal, spatial, diffs)
