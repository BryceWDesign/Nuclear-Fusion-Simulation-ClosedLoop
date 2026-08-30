"""Command-line interface for ClosedLoop runs and verification campaigns."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .convergence import run_convergence_campaign
from .equilibrium import manufactured_solution_error, solve_solovev_class_equilibrium
from .evidence import verify_evidence_bundle, write_evidence_bundle
from .fast_particles import alpha_orbit_screen
from .models import SimulationConfig
from .optimizer import search_operating_points
from .sensitivity import sensitivity_screen
from .simulation import ClosedLoopSimulator
from .uncertainty import run_uncertainty_campaign


def load_config(path: str | Path) -> SimulationConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = SimulationConfig.from_dict(raw)
    cfg.validate()
    return cfg


def _print_summary(summary: dict[str, object]) -> None:
    fields = (
        "fusion_power_MW",
        "Q_plasma",
        "alpha_deposition_power_MW",
        "neutron_power_MW",
        "radiation_loss_MW",
        "transport_edge_loss_MW",
        "tau_E_s",
        "lawson_nTtau_keV_s_m3",
        "energy_balance_relative_residual",
        "particle_balance_relative_residual",
        "authority",
    )
    print("CLOSEDLOOP RESULT")
    print("=" * 72)
    for key in fields:
        print(f"{key:38s} {summary.get(key)}")
    print("experimental_validation                 False")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    result = ClosedLoopSimulator(cfg).run()
    _print_summary(result.summary)
    if args.output:
        bundle = write_evidence_bundle(
            args.output,
            config=cfg.to_dict(),
            summary=result.summary,
            timeseries=result.timeseries,
            gates=result.gates.to_dict(),
            repo_root=Path(__file__).resolve().parents[1],
        )
        ok, errors = verify_evidence_bundle(bundle)
        print(f"evidence_bundle                         {bundle}")
        print(f"evidence_hash_verification              {'PASS' if ok else 'FAIL'}")
        if errors:
            for error in errors:
                print(f"  {error}")
            return 2
    return 0 if result.gates.causal_fusion_result_allowed else 3


def cmd_convergence(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = run_convergence_campaign(cfg)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    result = sensitivity_screen(cfg, args.parameter, args.metric, args.fraction)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


def cmd_equilibrium_verify(args: argparse.Namespace) -> int:
    error, result = manufactured_solution_error(args.nr, args.nz)
    payload = {
        "relative_l2_error": error,
        "iterations": result.iterations,
        "residual_linf": result.residual_linf,
        "authority": "manufactured_solution_numerical_verification",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if error < args.max_error else 4


def cmd_equilibrium_solve(args: argparse.Namespace) -> int:
    result = solve_solovev_class_equilibrium(
        R0_m=args.R0,
        minor_radius_m=args.a,
        nR=args.nr,
        nZ=args.nz,
        pressure_derivative_Pa_per_Wb=args.pprime,
        ff_derivative_T2m2_per_Wb=args.ffprime,
    )
    payload = {
        "psi_min": float(result.psi_Wb_rad.min()),
        "psi_max": float(result.psi_Wb_rad.max()),
        "iterations": result.iterations,
        "residual_linf": result.residual_linf,
        "authority": "reduced_solovev_class_constant_source_grad_shafranov",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_alpha_orbit(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    result = alpha_orbit_screen(
        cfg.geometry, markers=args.markers, steps=args.steps, dt_s=args.dt, seed=args.seed
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


def cmd_uncertainty(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = run_uncertainty_campaign(cfg, samples=args.samples, seed=args.seed)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = search_operating_points(cfg)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.best is not None else 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="closedloop", description="Verification-first reduced-order tokamak D-T burn simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one declared scenario")
    p_run.add_argument("config")
    p_run.add_argument("--output", help="write a hashed evidence bundle")
    p_run.set_defaults(func=cmd_run)

    p_conv = sub.add_parser("convergence", help="run time/space refinement campaign")
    p_conv.add_argument("config")
    p_conv.set_defaults(func=cmd_convergence)

    p_sens = sub.add_parser("sensitivity", help="finite-difference sensitivity screen")
    p_sens.add_argument("config")
    p_sens.add_argument("parameter", choices=["external_heating_MW", "core_fuel_density_m3", "core_ion_temperature_keV", "ion_thermal_diffusivity_m2_s"])
    p_sens.add_argument("--metric", default="Q_plasma")
    p_sens.add_argument("--fraction", type=float, default=0.02)
    p_sens.set_defaults(func=cmd_sensitivity)

    p_eqv = sub.add_parser("equilibrium-verify", help="verify Grad-Shafranov discretization against a manufactured solution")
    p_eqv.add_argument("--nr", type=int, default=41)
    p_eqv.add_argument("--nz", type=int, default=41)
    p_eqv.add_argument("--max-error", type=float, default=0.02)
    p_eqv.set_defaults(func=cmd_equilibrium_verify)

    p_eq = sub.add_parser("equilibrium-solve", help="solve reduced constant-source Solovev-class equilibrium")
    p_eq.add_argument("--R0", type=float, default=6.2)
    p_eq.add_argument("--a", type=float, default=2.0)
    p_eq.add_argument("--nr", type=int, default=65)
    p_eq.add_argument("--nz", type=int, default=65)
    p_eq.add_argument("--pprime", type=float, default=-1e6)
    p_eq.add_argument("--ffprime", type=float, default=0.0)
    p_eq.set_defaults(func=cmd_equilibrium_solve)

    p_alpha = sub.add_parser("alpha-orbit", help="run analytic-field 3-D Boris alpha vacuum-orbit screen")
    p_alpha.add_argument("config")
    p_alpha.add_argument("--markers", type=int, default=256)
    p_alpha.add_argument("--steps", type=int, default=1200)
    p_alpha.add_argument("--dt", type=float, default=2e-10)
    p_alpha.add_argument("--seed", type=int, default=17)
    p_alpha.set_defaults(func=cmd_alpha_orbit)

    p_uq = sub.add_parser("uncertainty", help="run seeded declared-input uncertainty campaign")
    p_uq.add_argument("config")
    p_uq.add_argument("--samples", type=int, default=32)
    p_uq.add_argument("--seed", type=int, default=1234)
    p_uq.set_defaults(func=cmd_uncertainty)

    p_opt = sub.add_parser("optimize", help="run transparent constrained operating-point grid search")
    p_opt.add_argument("config")
    p_opt.set_defaults(func=cmd_optimize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
