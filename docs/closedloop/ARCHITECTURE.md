# ClosedLoop Architecture

```text
SimulationConfig
    |
    +--> radial circular-torus finite volumes
    +--> D / T / He thermal populations
    +--> electron + ion thermal-energy states
    |
    +--> Bosch-Hale D-T source
            |
            +--> D/T depletion
            +--> neutron source ledger
            +--> fast-alpha number/energy reservoir
                       |
                       +--> alpha escape
                       +--> alpha slowing/deposition
                                  |
                                  +--> electron heating
                                  +--> ion heating

particle diffusion ---------> conservative particle ledger
heat transport -------------> conservative energy ledger
radiation ------------------> energy loss ledger
e-i exchange ---------------> internal conservative transfer
external heating -----------> energy input ledger

same state --> Q, tau_E, Lawson, TBR constraint, plant arithmetic,
               operating-envelope screens, evidence + claim gates
```

The `closedloop` package is independent from the legacy `main.py` reactor loop. This prevents new evidence labels from silently certifying old calculations.
