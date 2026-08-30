# ClosedLoop Validation Report, v1.0.0

## Automated verification

Local final gate at packaging time:

- Python compileall: PASS
- pytest: 65 PASS
- Grad-Shafranov manufactured solution: PASS
- baseline evidence SHA-256 verification: PASS
- legacy `test_maxwellian.py` smoke test: PASS in packaging environment
- baseline energy conservation: PASS, relative residual 5.88e-16
- baseline particle conservation: PASS, relative residual 1.90e-16
- baseline causal fusion/Q chain: PASS
- baseline alpha self-heating promotion: FAIL
- experimental validation: NOT PRESENT / hard gate FALSE

## Manufactured equilibrium verification

The Grad-Shafranov finite-difference operator is verified against a polynomial manufactured solution with Dirichlet boundaries. At 25 x 25 the observed relative L2 error is approximately 7.13e-11 and residual L-infinity approximately 1.76e-9 for the packaged solver/tolerance.

## Resolution sensitivity

See `results/baseline_convergence.json`.

The final fusion-power result is substantially less sensitive to time-step refinement than to radial-grid refinement. `tau_E` is more spatially sensitive than fusion power and must not be presented at stronger numerical authority than the convergence data justify.

## Validation not performed

No experimental discharge reconstruction, EFIT/FreeGSNKE comparison, gyrokinetic validation, OpenMC neutron transport, disruption validation, blanket thermal-hydraulics, structural qualification, magnet qualification, or net-electric plant validation is included.
