# Changelog

## ClosedLoop v1.0.0

- preserved upstream simulator as legacy/reference path
- added independent `closedloop` package
- replaced scheduled-alpha causal architecture in the new engine with D-T-rate-driven alpha/neutron production
- added separate D/T/He/fast-alpha state
- added dual electron/ion thermal energy state
- added conservative radial particle and heat transport
- added alpha slowing/escape reservoir and thermal feedback
- added radiation and electron-ion exchange
- added real thermal `tau_E`, Q, Lawson diagnostics
- added particle and energy ledgers with hard verification gates
- added reduced Grad-Shafranov solver plus manufactured verification
- added reduced q/Greenwald/beta_N operating-envelope screens
- added vectorized 3-D Boris alpha vacuum-orbit screen
- added neutron/tritium source accounting and breeding-coverage constraint
- added conditional plant closure
- added convergence, sensitivity, uncertainty and constrained grid-search tools
- added SHA-256 evidence bundles
- added Python 3.11/3.12/3.13 CI
- added 65 automated tests
- added explicit claim and license boundaries
