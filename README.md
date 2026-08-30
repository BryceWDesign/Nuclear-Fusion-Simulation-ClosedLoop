# Nuclear-Fusion-Simulation-ClosedLoop

**Verification-first reduced-order D-T tokamak burn, transport, and reactor-accounting fork.**

Canonical fork: `https://github.com/BryceWDesign/Nuclear-Fusion-Simulation-ClosedLoop`

This repository preserves the original educational particle/PIC simulator as legacy code and adds a separate `closedloop` engine whose fusion power, alpha source, neutron source, fuel consumption, plasma heating, losses, Q, and confinement metrics are derived from one causal state.

> **Claim boundary:** this is not an experimentally validated tokamak predictor, not an ITER-class whole-device code, and not proof of ignition or net-electric fusion. It is a reproducible reduced-order research simulator with explicit numerical and claim gates.

## What changed

The original reactor path injected 3.5 MeV alpha particles on a schedule and later inferred fusion power from alpha power. ClosedLoop removes that causal shortcut in its new engine:

```text
D + T state
  -> Bosch-Hale <sigma v>
  -> local D-T reaction rate
  -> D/T fuel depletion
  -> 3.5 MeV alpha birth + 14.1 MeV neutron source
  -> fast-alpha slowing / escape reservoir
  -> alpha deposition into electron / ion thermal energy
  -> changed Te, Ti
  -> changed D-T reactivity
```

No scheduled alpha population is used by `closedloop`.

## Implemented, executable subsystems

- Separate radial D and T fuel populations
- Maxwellian Bosch-Hale D-T reactivity with hard 0.2-100 keV authority boundary
- Causal D-T fuel burn and helium production
- Exact 17.6 MeV D-T energy branching into 3.5 MeV alpha and 14.1 MeV neutron channels
- Fast-alpha number and energy reservoir with explicit slowing and orbit-loss timescales
- Quasineutral electron-fluid charge accounting
- Separate ion and electron thermal-energy states
- Conservative finite-volume radial particle transport
- Conservative reduced heat transport
- Electron-ion energy exchange
- Bremsstrahlung plus optional declared impurity cooling coefficient
- Real thermal energy confinement time, `tau_E = W_thermal / P_thermal_loss`
- Scientific plasma gain, `Q = P_fusion / P_external_plasma_heating`
- Lawson `n T tau_E` diagnostic from the same simulated state
- D, T, He/alpha particle-balance audit
- Plasma + fast-alpha energy-balance audit
- Reduced Grad-Shafranov solver with manufactured-solution verification
- Reduced tokamak operating-envelope screens: Greenwald fraction, normalized beta, shaping-adjusted cylindrical edge-q
- Vectorized 3-D Boris 3.5 MeV alpha vacuum-orbit screen in an analytic toroidal+poloidal field
- Exact neutron-source and tritium-consumption ledger
- Geometric tritium-breeding coverage constraint
- Conditional plant power closure that refuses to invent missing recirculating loads
- Time-step and spatial-resolution convergence campaign
- Finite-difference parameter sensitivity screen
- Seeded declared-input uncertainty campaign
- Transparent constrained operating-point grid search
- SHA-256 reproducible evidence bundles
- Claim gates that permanently block experimental-validation claims
- GitHub Actions on Python 3.11, 3.12, and 3.13

## Baseline result included in this ZIP

`scenarios/baseline_dt.json` was executed and persisted under `results/baseline_dt/`.

At the declared final time of 1.0 s, the current reduced model produced:

| Quantity | Result |
|---|---:|
| Fusion power | 163.565 MW |
| Q plasma | 3.2713 |
| Alpha deposition | 21.824 MW |
| Neutron power | 131.038 MW |
| Radiation loss | 2.883 MW |
| Edge transport loss | 15.161 MW |
| tau_E | 6.4178 s |
| Lawson nTtau | 3.171e21 keV s m^-3 |
| Energy-balance relative residual | 5.88e-16 |
| Particle-balance relative residual | 1.90e-16 |

**This is a reduced-model result, not a prediction that a physical tokamak with these inputs will achieve Q=3.27.**

The alpha deposition fraction of total alpha+external heating is only about 30%, so the `alpha_self_heating_screen` remains **FAIL**. The code therefore does not promote this baseline to a burning-plasma or ignition claim.

### Numerical refinement result

`results/baseline_convergence.json` compares the declared baseline, half time step, and doubled radial resolution. Fusion power changes by about 0.035% under time refinement and 0.855% from the temporally refined to spatially refined run. `tau_E` changes by about 7.65% under the spatial refinement, so confinement remains a materially weaker numerical result than fusion source power in this model.

### Model-input uncertainty result

`results/uncertainty_16.json` contains a seeded 16-sample campaign over declared density, ion-temperature, transport, and heating uncertainty. It reports a Q distribution of approximately 2.82 / 3.24 / 3.66 at p05 / median / p95. This is explicitly **model-input uncertainty**, not an experimental probability distribution.

### Alpha vacuum-orbit screen

`results/alpha_orbit_screen.json` contains a 256-marker, 1200-step Boris screen. The current analytic-field run retained all markers over 0.24 microseconds and showed maximum relative kinetic-energy drift below 8e-15. This is a short collisionless vacuum-orbit numerical screen, **not** evidence of reactor-scale alpha confinement.

## Run it

Minimal ClosedLoop dependency:

```bash
python -m pip install -r requirements-closedloop.txt
```

Run the baseline and write a hashed evidence bundle:

```bash
python -m closedloop run scenarios/baseline_dt.json --output results/my_run
```

Run the numerical equilibrium verification:

```bash
python -m closedloop equilibrium-verify
```

Run convergence:

```bash
python -m closedloop convergence scenarios/baseline_dt.json
```

Run a parameter sensitivity:

```bash
python -m closedloop sensitivity scenarios/baseline_dt.json core_ion_temperature_keV --metric fusion_power_MW
```

Run the alpha orbit screen:

```bash
python -m closedloop alpha-orbit scenarios/baseline_dt.json
```

Run declared-input uncertainty:

```bash
python -m closedloop uncertainty scenarios/baseline_dt.json --samples 32 --seed 1234
```

Run the finite grid search:

```bash
python -m closedloop optimize scenarios/baseline_dt.json
```

Local deterministic gate:

```bash
python scripts/check_closedloop_green.py
```

## Evidence bundle

A run with `--output` writes:

```text
config.json
summary.json
timeseries.json
claim_gates.json
environment.json
SHA256SUMS
```

The manifest is verified immediately by the CLI. A result with a failed conservation or causal-Q gate returns a nonzero status.

## Model authority

ClosedLoop currently earns the authority label:

`VERIFIED_REDUCED_BURN_TRANSPORT_SCREEN`

when conservation and causal-Q gates pass.

It does **not** earn any of the following:

- experimental tokamak validation
- predictive nonlinear MHD
- gyrokinetic turbulence validation
- production free-boundary equilibrium validation
- 3-D neutron transport / TBR validation
- blanket thermo-hydraulics qualification
- divertor qualification
- magnet qualification
- net-electric reactor validation
- ignition demonstration

See `docs/closedloop/CLAIM_BOUNDARY.md` and `FINAL_STATUS.md`.

## Legacy simulator

The upstream educational simulator remains at repository root (`main.py`, `physics_engine.py`, `mhd_equilibrium.py`, etc.). Its original README is preserved as `docs/legacy/UPSTREAM_README.md`.

ClosedLoop does not claim that its verification gates retroactively validate the legacy PIC/GPU path.

## Licensing status

The upstream repository did not include a software license when this fork was created. This fork therefore intentionally does **not** add a new blanket license over the inherited code. See `LICENSE_STATUS.md` before redistributing or relicensing this derivative repository.

## Technical references

- Bosch, H.-S. and Hale, G. M., *Improved formulas for fusion cross-sections and thermal reactivities*, Nuclear Fusion 32 (1992) 611.
- Standard finite-volume conservation methods for radial diffusion.
- Standard Boris magnetic particle pusher.
- Grad-Shafranov equation with a constant-source Solovev-class reduced mode.

The equations are implemented locally and regression-tested. External experimental validation is not bundled.
