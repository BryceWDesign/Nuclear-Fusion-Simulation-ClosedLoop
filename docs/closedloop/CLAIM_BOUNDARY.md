# ClosedLoop Claim Boundary

## Allowed claims

The code may claim that it:

1. numerically evaluates a published-range Bosch-Hale Maxwellian D-T reactivity fit;
2. derives local D-T reaction rates from simulated D and T density and ion temperature;
3. consumes D and T according to that rate;
4. creates alpha and neutron source terms from the same reaction ledger;
5. couples alpha energy deposition back into thermal plasma energy;
6. conserves its declared particle ledgers and plasma/fast-alpha energy ledger within configured numerical tolerances;
7. evaluates reduced finite-volume transport, reduced operating-envelope screens, and conditional plant arithmetic;
8. can reproduce stored evidence from declared configuration and software state.

## Prohibited promotions

A green ClosedLoop run is not evidence that a physical machine will reproduce the calculated Q, confinement time, TBR, net power, alpha retention, stability, or burn state.

The project must not describe a result as experimentally validated unless external experimental evidence and a documented validation campaign are added later. The `experimental_validation` gate is hard-coded false in this release.

## Specific reduced-model limitations

- Circular-torus 1.5D radial geometry, not shaped flux-surface transport.
- Diffusive particle/thermal transport coefficients are declared inputs, not first-principles turbulent transport.
- Electron-ion equilibration uses a declared relaxation closure.
- Alpha slowing/loss in the burn engine uses declared timescales; the separate Boris orbit screen is not yet coupled back into those timescales.
- Bremsstrahlung is reduced; detailed line radiation requires a user-supplied effective cooling coefficient.
- Grad-Shafranov physical mode uses constant source functions, a Solovev-class reduction.
- q, Greenwald, and beta_N are screening metrics only.
- Tritium breeding is an exact source/coverage bound, not neutron transport.
- Plant closure is algebraic and remains unknown if recirculating loads are not explicitly supplied.
