# Transient Pipe State Finder

Physics-constrained reconstruction of initial pressure and mass-flow fields in an isothermal gas pipeline using boundary observations.

## Research plan

1. Reproduce the validated staggered-grid, theta-method forward model from `isothermal-pipe-sim` (including boundary half-cells and global mass conservation).
2. Generate synthetic observations from a known initial state, prescribing inlet pressure and outlet mass flow; fit measured inlet mass flow and outlet pressure.
3. Minimize uncertainty-scaled boundary least-squares misfit with L-BFGS-B and a discrete adjoint generated using automatic differentiation / implicit differentiation.
4. Verify adjoint gradients against finite differences and assess state reconstruction, linepack, and observability.
5. Compare steady-state, evolution-operator, and boundary-history continuation methods.

The repository is an independent inverse-problem project; do not assume the existing forward solver can be differentiated without adaptation.

## License

GNU Affero General Public License v3.0 **only** (`AGPL-3.0-only`). No additional linking exception.
