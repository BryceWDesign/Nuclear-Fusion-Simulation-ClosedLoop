from dataclasses import replace

import pytest

from closedloop.models import SimulationConfig
from closedloop.simulation import ClosedLoopSimulator, SimulationValidityError


def test_config_round_trip_dict():
    cfg = SimulationConfig(n_cells=24, duration_s=0.02)
    restored = SimulationConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_invalid_geometry_rejected():
    cfg = SimulationConfig()
    bad = replace(cfg, geometry=replace(cfg.geometry, major_radius_m=1.0, minor_radius_m=2.0))
    with pytest.raises(ValueError):
        bad.validate()


def test_unstable_transport_timestep_rejected_before_run():
    cfg = SimulationConfig(n_cells=16, dt_s=1.0, duration_s=1.0)
    with pytest.raises(SimulationValidityError):
        ClosedLoopSimulator(cfg)
