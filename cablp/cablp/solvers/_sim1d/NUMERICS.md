# sim1d Numerical Methods

Brief outline of the numerical schemes used by `LAPDSim1D`
(`cablp/solvers/_sim1d/`). This is a conservative, axially-resolved 1D LAPD
transport model (1D interior with 0D boundary cells). For the continuous model
equations these schemes discretize, see [`MODEL.md`](MODEL.md).

## State and discretization

- **Conservative variables** (`core/state.py`, `ConservativeState1D`): electron
  density `n`, neutral density `nn`, parallel momentum density `M`, and electron
  and ion energy densities `Ee`, `Ei`. Primitive quantities (`Te`, `Ti`, `v`)
  are recovered by `derive_state`.
- **Grid** (`core/geometry.py`): finite-volume cells along the axial (`z`)
  coordinate with cell-centered states and face-based fluxes. Plasma and neutral
  fields carry separate face areas and cell volumes so inventory
  (`area × flux`, `volume × density`) is tracked consistently.
- **Floors**: density and temperature floors are enforced after every stage
  (`floor_state_vector` / `apply_state_floors`) to keep the state physical.

## Spatial discretization (finite volume)

- **Plasma advection** (`physics/flux.py`): a **Rusanov / local Lax–Friedrichs**
  flux at each interior face,
  `F = ½(F_L + F_R) − ½·a_max·(U_R − U_L)`,
  with `a_max` from the local ion sound speed. External faces are
  **closed/reflecting** (no particle or thermal-energy flux) — the 0D boundary
  behavior.
- **Front filling** (optional): a sonic-relaxation "front-filling" face flux
  (`front_filling_fluxes`, `alpha_front`) models plasma filling into unfilled
  cells, added alongside the Rusanov flux.
- **Flux divergence**: RHS terms are formed as
  `−Δ(area·F)/volume` per cell (`_flux_divergence`).
- **Heat conduction** (`physics/conduction.py`): conductive face fluxes
  `q = −κ ∇T` differenced to a conservative flux divergence, with
  frozen-conductivity coefficients for the implicit path.

## Source / reaction terms

Local (cell-wise) RHS contributions, all in `physics/`:

- `reactions.py` — bulk ionization/recombination reaction rates.
- `neutrals.py` — neutral exchange (diffusive coupling between cells),
  gas-puff sources, and pumping sinks.
- `energy.py` — electron cooling, electron–ion energy exchange, ion
  charge-exchange energy loss.
- `sources.py` — surface neutralization.
- `cathode.py` — cathode/sheath boundary physics.

## Time integration

- **Explicit path** (default, `core/integrator.py`): a **two-stage strong
  stability preserving Runge–Kutta (SSPRK2 / Heun)** step,
  `y¹ = floor(y⁰ + Δt·L(tⁿ, y⁰))`,
  `yⁿ⁺¹ = floor(½y⁰ + ½(y¹ + Δt·L(tⁿ+Δt, y¹)))`,
  with floors applied at each stage. The stages are evaluated at `tⁿ` and
  `tⁿ+Δt`, which preserves second-order accuracy for explicitly time-dependent
  forcing such as the gas-puff schedule. `ssprk2_step` freezes the forcing at
  the step start when its `time` argument is omitted, which is only
  first-order accurate in that forcing.
- **Operator-split path** (optional, `implicit_heat_conduction` flag /
  `--operator-split`): Lie splitting — one explicit SSPRK2 step over all
  **non-heat** terms, followed by an implicit heat-conduction substep
  (`implicit_heat_conduction_step`) solved per species as a tridiagonal system
  via `scipy.linalg.solve_banded`. This removes the stiff parabolic
  heat-conduction stability limit from the explicit timestep.

  The substep is a **theta method**,
  `(C + θ·Δt·K)·Tⁿ⁺¹ = C·Tⁿ − (1−θ)·Δt·K·Tⁿ`, selected by the
  `implicit_heat_scheme` parameter:

  | `implicit_heat_scheme` | θ | `R(−∞)` | substep order |
  |------------------------|-----|--------|---------------|
  | `backward_euler` (default) | 1   | 0    | 1 |
  | `shifted`                  | 0.6 | −2/3 | 1 |
  | `crank_nicolson`           | 0.5 | −1   | 2 |

  The explicit half is assembled from `conductive_face_flux` /
  `flux_divergence_rhs`, which is exactly `−K·Tⁿ` built from the same face
  coefficients as the implicit operator, so both halves stay consistent by
  construction. θ=1 keeps the right-hand side at the raw conservative energy
  and reproduces the original backward-Euler solve bit-for-bit.

  Only θ=1 is **L-stable**. There `C + Δt·K` is an M-matrix whose rows sum to
  `C` (since `K·1 = 0`), giving a discrete maximum principle `Tⁿ⁺¹ ≥ min(Tⁿ)`:
  backward Euler is unconditionally monotone and *cannot* undershoot the
  temperature floors. For θ<1 the amplification factor tends to `−(1−θ)/θ` as
  `Δt·λ → −∞`, so stiff modes ring — undamped at θ=1/2 — and can be clipped by
  the floor, which silently injects energy. See
  `scripts/audit_sim1d_floor_activation.py` for measuring whether that actually
  happens for a given configuration.

  Conductivity is frozen at the incoming state for every θ, and the splitting
  is Lie rather than Strang, so the split step stays first-order overall no
  matter which scheme is selected; θ<1 reduces the substep's error constant
  rather than raising the global order.
- **Neutral-only / prebreakdown path**: when plasma is disabled (or during
  neutral prebreakdown/equilibration), neutrals are advanced with an implicit
  (backward-Euler) linear solve over exchange, pumping, and gas-puff terms
  (`_implicit_neutral_step`).

## Adaptive timestep control

`core/timestep.py` (`suggest_timestep`) picks Δt as the minimum over several
physically-motivated candidate bounds, then clamps to `[dt_min, dt_max]`:

- **Plasma CFL** (`cfl`, default 0.4) from the local sound speed and cell size.
- **Front-filling** density-change bound.
- **Fractional-change** limits (`density_dt_fraction`, `neutral_dt_fraction`,
  `heat_dt_fraction`, default 0.25) so no single explicit source/sink changes a
  field by more than a set fraction per step: surface loss, neutral exchange,
  neutral sources, reactions, energy exchange, electron cooling, ion
  charge-exchange, and (explicit path only) heat conduction.

`TimestepDiagnostics` records every candidate, the `active_constraint` that set
Δt, and per-step accept/reject bookkeeping.

## Step acceptance and rejection

The solver attempts a candidate step without committing state
(`_attempt_step`), then validates it. On a **non-finite** or otherwise invalid
state the step is **rejected and retried with a reduced Δt** (`retry_count`,
`rejection_reason`, `TimestepRejectionError`). `BreakdownError` is raised for
unrecoverable breakdown conditions. Rejection events and constraint histories
are stored for post-run diagnostics.

## Output

Results are written to HDF5 (`results/io.py`, format `sim1d-hdf5-v1`) including
time series, axial profiles, and per-step diagnostics. `results/compat.py` adds
`sim3`-compatible aliases; `results/health.py` reports finiteness and
conservation drift (particle inventory, thermal energy). See
`scripts/run_sim1d.py` (drive/save) and `scripts/plot_sim1d_run.py` (contour and
time-slice plots).
