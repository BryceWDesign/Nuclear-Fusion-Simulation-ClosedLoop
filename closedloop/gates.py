"""Claim gates that prevent reduced-order results from being promoted above evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ClaimGateReport:
    gates: tuple[Gate, ...]
    causal_fusion_result_allowed: bool
    burning_plasma_screen_allowed: bool
    net_electric_statement_allowed: bool
    experimental_validation_claim_allowed: bool
    highest_authority: str

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        raw["gates"] = [asdict(g) for g in self.gates]
        return raw


def evaluate_claim_gates(
    *,
    energy_residual: float,
    particle_residual: float,
    energy_limit: float,
    particle_limit: float,
    q_plasma: float | None,
    alpha_heating_fraction: float,
    external_heating_MW: float,
    plant_closure_complete: bool,
) -> ClaimGateReport:
    energy_ok = abs(energy_residual) <= energy_limit
    particle_ok = abs(particle_residual) <= particle_limit
    q_defined = q_plasma is not None
    causal = energy_ok and particle_ok and q_defined
    # "burning plasma screen" is deliberately not called ignition. It requires alpha
    # heating to dominate all externally deposited heating at the evaluated instant.
    burning = causal and alpha_heating_fraction >= 0.5
    plant_ok = causal and plant_closure_complete
    gates = (
        Gate("energy_conservation", energy_ok, f"relative residual={energy_residual:.3e}, limit={energy_limit:.3e}"),
        Gate("particle_conservation", particle_ok, f"relative residual={particle_residual:.3e}, limit={particle_limit:.3e}"),
        Gate("q_causal_chain", q_defined, "Q uses D-T reaction power divided by declared external plasma heating"),
        Gate("alpha_self_heating_screen", burning, f"alpha heating fraction={alpha_heating_fraction:.4f}"),
        Gate("plant_closure", plant_ok, "net-electric is only emitted when all declared recirculating loads are supplied"),
        Gate("experimental_validation", False, "no experimental tokamak validation is bundled with this repository"),
    )
    if plant_ok:
        authority = "REDUCED_WHOLE_DEVICE_SCREEN"
    elif causal:
        authority = "VERIFIED_REDUCED_BURN_TRANSPORT_SCREEN"
    else:
        authority = "NUMERICAL_RESULT_BLOCKED_BY_VERIFICATION"
    return ClaimGateReport(gates, causal, burning, plant_ok, False, authority)
