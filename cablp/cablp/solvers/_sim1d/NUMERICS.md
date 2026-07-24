# sim1d Numerical Methods

Brief outline of the numerical schemes used by `LAPDSim1D`
(`cablp/solvers/_sim1d/`). This is a conservative, axially-resolved 1D LAPD
transport model (1D interior with 0D boundary cells). For the continuous model
equations these schemes discretize, see [`MODEL.md`](MODEL.md).

## State and discretization

- **Conservative variables** (`core/state.py`, `ConservativeState1D`): electron
  density `n`, neutral density `nn`, parallel momentum density `M`, and electron
  and ion energy densities `Ee`, `Ei`. Primitive quantities (`Te`, `Ti`, `v`)
  are recovered by `derive_state`. Default-off neutral reductions append
  optional rows in introduction order: column/chamber neutral momentum `M_n`,
  annulus density `nn_a`, and (only for `kinetic_two_moment`) annulus momentum
  `M_n_a`. Existing 5-, 6-, and 7-row layouts are unchanged.
- **Grid** (`core/geometry.py`): finite-volume cells along the axial (`z`)
  coordinate with cell-centered states and face-based fluxes. Plasma and neutral
  fields carry separate face areas and cell volumes so inventory
  (`area × flux`, `volume × density`) is tracked consistently.
- **Floors**: density and temperature floors are enforced after every stage
  (`floor_state_vector` / `apply_state_floors`) to keep the state physical.
  They are numerical admissibility limits, not initial conditions: the live
  repaired seeds are `Te0=0.21 eV` and `Ti0=0.125 eV`, strictly above the
  unchanged `0.1 eV` floors. A raw-validation config rejects floor-bound
  initial temperatures at construction.

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
- **Optional flux-tube expansion**: the default-off expanded-end geometry
  supplies resolved plasma face areas and cell volumes. The momentum ledger
  gains `flux_tube_geometry = p·ΔA/V`, which exactly cancels the area change
  in the pressure flux for a uniform stationary state. No mirror-force or
  pressure-anisotropy closure is implied.
- **Optional thin annular baffles**: requested axial positions map to the
  nearest existing interior face. Knudsen transport combines the adjacent
  tube conductance and the zero-thickness aperture conductance in series. In
  two-zone mode only the annulus coefficient changes; plasma and column-neutral
  transport remain identical to the unbaffled geometry. The selector is
  default off and incomplete or flag-off parameter sets fail at construction.
- **Kinetic-derived two-momentum neutral advection**: column density/momentum
  use plasma face areas and volumes; annulus density/momentum use
  `neutral_face_area - plasma_face_area` and annulus volumes. Each zone uses
  first-order donor-cell upwinding with its own drift. Interior face
  inventories cancel pairwise. Radial column/annulus momentum transfer is a
  local conservative RHS; annulus wall and baffle accommodation are
  sign-safe sinks. The thermal two-zone Knudsen operator is retained and no
  pressure flux is duplicated.
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
  `--operator-split`): an explicit SSPRK2 step over all **non-heat** terms
  (operator `A`) composed with an implicit heat-conduction substep
  (`implicit_heat_conduction_step`, operator `B`) solved per species as a
  tridiagonal system via `scipy.linalg.solve_banded`. This removes the stiff
  parabolic heat-conduction stability limit from the explicit timestep.

  `operator_splitting` selects the composition: `"lie"` (default) does
  `A(Δt)` then `B(Δt)` and is O(Δt), since the splitting error goes as
  `Δt·[A,B]`; `"strang"` does `B(Δt/2) → A(Δt) → B(Δt/2)`, whose symmetry
  cancels that leading commutator term and leaves O(Δt²). `B` is the halved
  operator because it is the cheap one — banded solves against a tridiagonal
  matrix, versus `A`'s reaction-rate evaluations — so Strang costs one extra
  heat substep, not one extra explicit step.

  The substep discretization is selected by the `implicit_heat_scheme`
  parameter. Three of the four are **theta methods**, solving
  `(C + θ·Δt·K)·Tⁿ⁺¹ = C·Tⁿ − (1−θ)·Δt·K·Tⁿ`:

  | `implicit_heat_scheme` | θ | `R(−∞)` | L-stable | solves | substep order |
  |------------------------|-----|--------|----------|--------|---------------|
  | `backward_euler` (default) | 1   | 0    | yes | 1 | 1 |
  | `shifted`                  | 0.6 | −2/3 | no  | 1 | 1 |
  | `crank_nicolson`           | 0.5 | −1   | no  | 1 | 2 |
  | `tr_bdf2`                  | —   | 0    | yes | 2 | 2 |

  The explicit half is assembled from `conductive_face_flux` /
  `flux_divergence_rhs`, which is exactly `−K·Tⁿ` built from the same face
  coefficients as the implicit operator, so both halves stay consistent by
  construction. θ=1 keeps the right-hand side at the raw conservative energy
  and reproduces the original backward-Euler solve bit-for-bit.

  Among the theta methods only θ=1 is **L-stable**. There `C + Δt·K` is an
  M-matrix whose rows sum to `C` (since `K·1 = 0`), giving a discrete maximum
  principle `Tⁿ⁺¹ ≥ min(Tⁿ)`: backward Euler is unconditionally monotone and
  *cannot* undershoot the temperature floors. For θ<1 the amplification factor
  tends to `−(1−θ)/θ` as `Δt·λ → −∞`, so stiff modes ring — undamped at θ=1/2 —
  and can be clipped by the floor, which silently injects energy. See
  `scripts/audit_sim1d_floor_activation.py` for measuring whether that actually
  happens for a given configuration.

  `tr_bdf2` (Bank et al. 1985) is second-order *and* L-stable: a trapezoidal
  stage out to `tⁿ + γΔt` followed by a BDF2 stage through `(Tⁿ, T_γ, Tⁿ⁺¹)`,
  with `γ = 2 − √2`. That γ makes the two stages share an implicit coefficient
  (`γ/2 = (1−γ)/(2−γ) ≈ 0.2929`) and hence one banded operator, so the cost is
  two `solve_banded` calls against a single matrix. The trapezoidal stage rings
  exactly as Crank–Nicolson does; the BDF2 stage annihilates what it leaves
  behind. It is *not* monotone the way backward Euler is — it damps undershoot
  rather than preventing it.

  **Conductivity is frozen at the incoming state for every scheme, and this —
  not the scheme — is what caps the substep at first order.** Measured
  self-convergence of the substep, on a fixed conductivity versus the live
  Braginskii `κ ∝ T^{5/2}`:

  | scheme | order, κ fixed (linear) | order, κ frozen at `Tⁿ` |
  |--------|------------------------|--------------------------|
  | `backward_euler` | 1.02 | 1.03 |
  | `shifted`        | 1.01 | 1.03 |
  | `crank_nicolson` | 2.00 | 1.05 |
  | `tr_bdf2`        | 2.02 | 1.06 |

  So a second-order scheme buys a smaller error constant (~6× versus backward
  Euler) but not a higher order until `κ` is evaluated at the scheme's own flux
  point, which `heat_picard_iterations` does.

## Measured order of the whole split step

`scripts/verify_sim1d_order.py` measures the observed temporal order of the
split step by fixed-Δt Richardson refinement, in a deliberately clean regime
(floors inert and watched, single phase, autonomous RHS, no cathode). At 62
cells with `t_end = 1e-6 s`:

| `heat_picard_iterations` | `operator_splitting` | `backward_euler` | `shifted` | `crank_nicolson` | `tr_bdf2` |
|---|---|---|---|---|---|
| 0 | `lie`    | 0.97 | 1.01 | 1.01 | 1.02 |
| 4 | `lie`    | 0.97 | 1.00 | 1.01 | 1.02 |
| 0 | `strang` | 0.98 | 0.98 | 1.04 | 0.98 |
| 4 | `strang` | 0.99 | 0.96 | **1.99** | **2.00** |

Second order requires **all three** of a second-order `implicit_heat_scheme`, a
positive `heat_picard_iterations`, and `operator_splitting = "strang"`. The
frozen conductivity and the Lie splitting are independent first-order terms, so
each caps the step on its own and removing only one changes nothing.
`backward_euler` and `shifted` staying at ~1.0 throughout is the negative
control: neither can be second-order at any Δt.

Note that a **production discharge will not show this**. Floors bind on ~42% of
cell-visits there and phase transitions are threshold-triggered, so the step
degrades to first order wherever those engage. The table above verifies the
scheme, not the production path.
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
- **Resolved electrode/source margin** (`surface_loss` diagnostic key): with
  raw-stage validation active, the combined cathode/sheath, anode collection,
  and boundary absorption bundle is bounded against `n-n_floor` and the exact
  conservative temperature margins
  `Ee-3/2 n Te_floor` / `Ei-3/2 n Ti_floor`. The corresponding rates include
  the change in floor energy when density changes.

`TimestepDiagnostics` records every candidate, the `active_constraint` that set
Δt, and per-step accept/reject bookkeeping.

## Step acceptance and rejection

The solver attempts a candidate step without committing state
(`_attempt_step`), then validates it. On a **non-finite** or otherwise invalid
state the step is **rejected and retried with a reduced Δt** (`retry_count`,
`rejection_reason`, `TimestepRejectionError`). `BreakdownError` is raised for
unrecoverable breakdown conditions. Rejection events and constraint histories
are stored for post-run diagnostics.

The R1 `raw_stage_validation` repair additionally inspects both
SSPRK candidates and implicit heat/neutral candidates *before* floors are
applied. It covers `n`, `nn`, optional `nn_a`, `Ee`, and `Ei`; a failed
candidate carries its raw rejection evidence but cannot mutate accepted state,
circuit/surface caches, kinetic targets, time, or the cumulative floor ledger.
Each accepted repair records the exact extensive debit in
`floor_ledger`: plasma/neutral particles added and electron/ion energy added,
using `V_p`, `V_col=V_p`, and `V_ann=V_m-V_p` as appropriate. Healthy focused
five-/six-/seven-/eight-row trajectories have an exactly zero ledger.

The repaired live stance selects raw-stage validation. Its resolved-source
timestep candidate prevents the audited launch candidate from crossing a
floor, while raw rejection remains the backstop. The unchanged checkpoint
golden explicitly pins the historical selector-off path and remains bit-exact;
no baseline was captured or updated.

## Output

Results are written to HDF5 (`results/io.py`, format `sim1d-hdf5-v1`) including
time series, axial profiles, and per-step diagnostics. `results/compat.py` adds
`sim3`-compatible aliases; `results/health.py` reports finiteness and
conservation drift (particle inventory, thermal energy). See
`scripts/run_sim1d.py` (drive/save) and `scripts/plot_sim1d_run.py` (contour and
time-slice plots).

R1 makes `rhs_terms`, `total_rhs`, finiteness, and inventory output follow the
actual packed five-/six-/seven-/eight-row state while retaining absent-dataset
compatibility for older H5 files. Every run result also carries the exact
constructed `params` and `flags`. `save_result_hdf5` writes those resolved
values to `params_json`/`flags_json` and rejects caller metadata that differs
from the constructed solver. `scripts/audit_sim1d_configs.py` verifies
reviewed SHA-256 snapshots for the production golden and every config-complete
campaign driver without running a campaign point.

The optional HDF5 `atomic_rate_domain` group is written on new results and
read as an empty mapping from older files. It contains the exact bundled He
ADF11 `Te` bounds, active-cell/volume below-grid fractions versus time, active
minimum `Te`, and first whole-run/afterglow crossings. Plotting tools use the
claim-port crossing for a vertical dashed afterglow-validity marker rather
than stopping on an unrelated cold cell.

## R2 conservative hyperbolic core (2026-07-24)

Two default-off selectors make the plasma hyperbolic update discretely
total-energy conservative; the checkpoint golden pins the old stance and stays
bit-exact.

- `hyperbolic_wave_speed`: `"isothermal"` (default, `sqrt(Te/m_i)`) or
  `"adiabatic"` (`sqrt((5/3)(Te+Ti)/m_i)`, the exact spectral radius of the
  γ=5/3 two-species system). It sets the Rusanov `a_max` and the plasma CFL —
  the dissipation strength and stability bound, not the physical wave speed,
  which the pressure flux already sets.
- `hyperbolic_energy_consistent`: replaces the convective momentum flux with
  the kinetic-energy-preserving `{u}{M}` form (Jameson 2008), and adds the
  `hyperbolic_energy_correction` RHS term that deposits the Rusanov `(n,M)`
  numerical kinetic-energy dissipation into `Ei` and applies a KEP pressure-work
  discretization. With it on, `Σ V (K+Ee+Ei)` is conserved by the semi-discrete
  flux + pressure-work operator to machine precision; explicit SSPRK2 leaves an
  `O(Δt²)` time-integration drift of the nonlinear kinetic energy (verified by
  Δt refinement).

The sonic `front_flux` is retired from the repaired stance: its L1 transport
activity vanishes under mesh refinement (a density diffusion with `D ~ c_s·dz`
layered on top of Rusanov's own). Rusanov/LLF is retained; a contact-restoring
(HLLC) or higher-order (MUSCL) scheme is a deferred follow-up gated on the G7
numerical-diffusion evidence. The pre-registered gate suite G1–G7 lives in
`scripts/verify_sim1d_r2_hyperbolic.py`.
