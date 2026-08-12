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
  `−Δ(area·F)/volume` per cell (`_flux_divergence`). The `u·∇` convective
  derivatives of the material-derivative model are never discretized separately:
  each is fused with its compression partner inside one Rusanov face flux via
  `∇·(Uu) = u·∇U + U∇·u`. See [`MODEL.md`](MODEL.md) "Conservative form: where the
  convective derivatives go" for the term-by-term map.
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
- `sources.py` — surface neutralization; ion-neutral drag / frictional heating /
  thermalization, and (default-off `ion_neutral_moment_closure`, audit A7/A8) the
  moment-closed reduced ion-neutral collision operator that replaces that quartet
  with one Phelps-rate equal-mass Braginskii momentum-transfer term (MODEL.md
  R4.3). The Phelps He⁺/He rate coefficients are built once at import in
  `funcs/_cross.py` (Maxwellian averages of the analytic isotropic + backscatter
  cross sections vs `T_eff`), analogous to the IAEA `charge_ex_react` table.
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
  raw-stage validation active — or unconditionally once a kinetic neutral arm
  is engaged — the combined cathode/sheath, anode collection, and
  plasma-terminating boundary bundle is bounded against `n-n_floor` and the
  exact conservative temperature margins
  `Ee-3/2 n Te_floor` / `Ei-3/2 n Ti_floor`. The corresponding rates include
  the change in floor energy when density changes. The boundary half is the
  operator the *stance* runs (the characteristic ghost-cell flux, or the
  legacy volumetric absorber when `characteristic_boundary` is off), and an
  engaged `kinetic_dvm` arm adds its plasma-side coupling term, which is
  otherwise a volumetric ion momentum/energy source no bound could see.

**A bound must describe a row the step applies.** A kinetic neutral arm
supersedes whole rows of the fluid terms — it zeroes them and carries them in
its own coupling term — and a bound still computed from the unstripped form is
a phantom that can set Δt and name itself `active_constraint` while the row it
describes is identically zero (measured, 2026-08-05: the fatal step's
constraint named a term whose applied `Ei` row was `0.0`). While the arm is
engaged, the `ion_charge_exchange`, `ion_neutral_drag`, `neutral_exchange` and
`neutral_sources` candidates are therefore withdrawn to `inf` and the reaction
bound keeps only its plasma channel; the replacements are bounded through the
`surface_loss` bundle above. The withdrawal is presence-gated on engagement,
so the moment path is untouched.

`TimestepDiagnostics` records every candidate, the `active_constraint` that set
Δt, and per-step accept/reject bookkeeping.

### Growth ramp and its accelerated re-approach (default off)

Between accepted steps Δt may grow by at most `dt_growth_factor` (1.25). The
ramp is applied *after* `suggest_timestep`, as one more `cap_step`, so it can
only shrink the step and never widens any physical bound.

Geometric re-approach is the cost. Recovering from a factor *F* below the
binding bound takes `log F / log 1.25` steps — ~26 steps from 364× below. In
knife-edge `surface_loss` regimes the collapse-and-recover episodes recur
often enough to dominate the step count: one probe measured **80.6% of steps
capped by `dt_growth`, at a median 364× below the binding physics bound**, with
~40-step recovery episodes recurring every ~15 steps. Those steps are not
resolving anything — no physical bound bound during any of them.

`dt_growth_recovery_patience` (**default 0 = off**) and
`dt_growth_recovery_factor` (4.0, consulted only when patience > 0) are an
opt-in accelerated re-approach. After `patience` CONSECUTIVE accepted steps
capped by `dt_growth`, the ramp's factor becomes the recovery factor; one step
capped by anything else — a physics bound, an output cadence, or a retry after
a rejection — resets the streak and the base factor returns immediately.

That asymmetry is the hysteresis, and it is the whole safety argument:

- **Engaging needs evidence.** Being growth-capped for many steps running means
  no physical candidate has bound in all that time, so the ramp is
  re-approaching rather than tracking. The key is a PATIENCE, not a threshold
  on Δt: nothing inspects how far below the bound the step is, so a genuinely
  small physics bound can never be mistaken for a ramp.
- **Releasing needs nothing.** A single non-growth-capped step ends
  acceleration, and the streak must be re-earned from zero.
- **No bound is weakened.** Every step remains the minimum over all candidates;
  this only widens the ceiling the ramp itself imposes. The reject/retry path
  is unchanged and remains the backstop.

Honest limits of the design, as built and unmeasured here:

- It is a **heuristic on the controller, not a physical improvement**. A larger
  jump toward the bound raises the chance of overshooting a bound that is
  moving, which costs a rejection and a retry — trading many cheap ramp steps
  for occasional expensive rejections. Whether that trade pays is regime
  dependent and is **not** established by anything in this repository.
- Accepted steps therefore change wherever it engages, so it is **trajectory
  changing** and stays default off. No default flip is proposed here.
- With patience 0 the branch is never evaluated, the ramp is uniformly
  `dt_growth_factor`, and the step sequence is bit-identical to a run predating
  the keys. The production golden is unaffected and was not recaptured.

**The clamp is recorded separately from the constraint name** (2026-08-05).
`active_constraint` always names the bound that actually minimized; when that
bound asked for less than `dt_min`, `clamped_to_dt_min` is set and `dt_raw`
keeps the unclamped request. It previously overwrote the name with `"dt_min"`,
which hid the true bound exactly when it mattered most. `dt_raw == 0.0` is the
drained floor-pinned signature: `_negative_margin_timestep` returns zero for a
cell sitting ON a floor while a term still drains it, which is a modelling
breakdown rather than a timestep request, and the clamp then keeps such a run
alive at `dt_min` indefinitely. `dt_min_lock_max_steps` (default 250000)
bounds the number of CONSECUTIVE clamped adaptive steps and raises
`RuntimeError` past it; consecutiveness is the discriminator, because clamp
episodes that release on their own are a normal, known-good family. Saved
files from before this date carry the old overwriting semantics and are not
migrated (see `results/io.py`).

## Step acceptance and rejection

The solver attempts a candidate step without committing state
(`_attempt_step`), then validates it. On a **non-finite** or otherwise invalid
state the step is **rejected and retried with a reduced Δt** (`retry_count`,
`rejection_reason`, `TimestepRejectionError`). Rejection events and constraint
histories are stored for post-run diagnostics.

A run that fails to IGNITE is not a step rejection but a phase transition. Two
guards, both armed only while the cathode drive is on and the run is still
pre-ignition, open the cathode switch and route the run through the ordinary
afterglow to a finite end time (`core/ignition.py`):

- the **stall detector** — `gamma_N <= 0` AND `d(Ee_total)/dt <= 0` for every
  saved sample across a 2.5 ms sustained window (a structural joint condition
  with no tuned rates and no clock in the verdict), phase event reason
  `ignition_stalled`;
- the **`tau_prebreakdown` timeout** — the machine's own 50 ms hardware guard,
  phase event reason `prebreakdown_timeout`. `prebreakdown_timeout_action`
  selects between this (`"switch_open"`, default) and the historical
  `BreakdownError` (`"raise"`, which loses the in-progress trajectory).

Both leave the run without a `main_discharge` phase, and the ES scorers refuse
to score such a run.

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

## R3 characteristic material boundaries (2026-07-24)

The default-off `characteristic_boundary` selector changes how the plasma-
terminating faces are discretized; default-off is golden bit-exact.

- **Boundary flux.** At each absorbing face a ghost state is set to the Bohm
  outflow (`n_se = n·presheath_alpha`, `u = c_s` into the wall, `Te`, `Ti`) and
  the same R2 KEP/Rusanov single-face flux (`flux.kep_rusanov_face_scalar`,
  following the interior's `hyperbolic_energy_consistent` / `hyperbolic_wave_speed`
  choice) is evaluated between the interior cell and the ghost. It is applied as a
  **one-sided divergence on the live cell only** (`± area·F / V`), because the
  shared telescoping face array would hand the removed plasma to the plasma-dead
  plenum; correspondingly the advective-flux path carries zero at these faces when
  the selector is on (no double-count of the reflecting wall pressure).
- **Sheath-edge sampling.** `sources.electrode_sheath_alpha` is the single source
  of the mesh-independent factor `n_se/n = presheath_alpha` (`τ`-independent; a
  cell-length/presheath-depth exponent), called by both the fluid boundary and the
  circuit (cathode) so they cannot disagree. In `funcs._cathode_solver_idriven`
  the flat `exp(-1/2)` becomes a passed `alpha_sheath`, with the electron lift
  `Λ → Λ − ln(α)` (`= Λ + 1/2` at `α = exp(-1/2)`, kept exact via a sentinel so the
  golden and the M2 equivalence gate are bit-exact). The cathode and anode carry
  **independent** presheath factors (`Λ` vs `Λ_anode`) — the anode mesh's short
  geometric presheath keeps flat `exp(-1/2)`.
- **Energy routing.** The circuit power split (`P_*_e/i_thermal + _phi`) derives
  the `phi` part as the remainder, so the historical `P_*` expressions that feed
  the golden are byte-for-byte unchanged; the fluid electrode energy is booked
  once (cathode electron via the circuit thermal part, collector via a floating
  `2 Te` sheath in the boundary term). Gate: `scripts/verify_sim1d_r3_routing.py`
  (split exactness, boxed γ, load-power closure to machine zero + cathode
  Kirchhoff). Boundary gates: `..._boundary.py` (unit) and `..._boundary_startup.py`
  (run: `u → c_s`, net sink). A11 coupling gate: `..._a11.py` (fixed-`dt`
  refinement at the current-gated knee; `--picard`/`--picard-tol` toggle the R5.1
  fix).
- **A13 controls.** The 0D surface-loss area scales / per-face enables are
  deprecated (never consumed by the resolved boundary); non-default use warns.

## R5.1 gated fluid<->circuit Picard (default off, audit A11)

The accepted step is sequential: the fluid stages run at a loop current frozen
over the step, then `T_s` and the circuit advance from the accepted plasma
(`_accept_step_attempt`). The default-off `coupled_circuit_picard` flag wraps both
step entry points (`_accept_step_with_picard`) and, in a driven phase, re-runs the
accepted step (`<= circuit_picard_max_iter`) with the frozen loop current updated
to the previous iteration's result until the current a step produces matches the
one it ran at (relative `circuit_picard_tol_rel`). `_picard_snapshot`/
`_picard_restore` capture and exactly restore every step-mutated attribute (the
persistent step cache is the four cathode fields), so a rejected iteration leaves
no trace — validated bit-exactly by `verify_sim1d_r5_picard.py` (H1 round-trip, N1
no-op, P1 knee perturbation, G1 default-off + K4a guard). It is a strict no-op
where the trigger does not fire (golden bit-exact). **R5.1 finding:** at production
`dt` the per-step loop-current change is below the trigger (Picard-1% ≈
sequential); the coupling sensitivity is confined to the internal sheath potential
`V_b`/`φ_c` (the SCL-corner regime), while `I_tot` (~3%) and `T_s` are robust.
Retained as a **default-off diagnostic**; sequential stays production.

## R5.2 electron heat-flux limiter (default off, audit A9)

The default electron conduction is classical Spitzer–Härm (`q = -κ_e ∇Te`), a
local law valid only where `λ_mfp ≪ L_T`. A9 measured `q_SH` reaching 1.7–3.3×
(static probe: ~4× median) the free-streaming ceiling `n·Te·v_the` at the resolved
gap faces — the constitutive law leaving its validity domain. The default-off
`electron_heat_flux_limit` flag scales `κ_e` per cell by the harmonic
(Cowie–McKee) limiter `λ = q_sat/(q_sat + q_SH)`, `q_sat = f·n·Te·v_the`
(`f = heat_flux_limiter_f`), so the flux caps at free-streaming where gradients are
steep and recovers Spitzer where they are shallow (`flux_limited_electron_conductivity`,
applied in both the explicit and implicit paths at the frozen incoming `Te`, so the
operator stays a conservative flux divergence). Identities
(`verify_sim1d_r5_heatflux.py`): Spitzer limit at large `f`, saturation cap
`κ_eff|∇Te| ≤ q_sat`, closed-domain energy conservation. Default off (golden
bit-exact); a declared A9 closure-family bracket — `f=1` targets only the ~gap
cells (flux → ~42%), `f=0.1` suppresses conduction globally. The static
engagement bracket is `probe_sim1d_r5_heatflux_bracket.py`; the dynamic
scored-observable bracket (runs at each `f`) is deferred.

## Regime-R2 pre-breakdown passive-tracer bridge (default off)

*Not to be confused with the "R2 conservative hyperbolic core" section above —
that R2 is an audit number from the 2026-07-24 hyperbolic repair. This one is
the REGIME programme's step R2 and touches no hyperbolic operator.*

During the conducting/pre-breakdown leg the plasma is **passive**: it conducts
a negligible share of the loop current, it takes a negligible share of the
beam's single-pass energy, and it burns a negligible share of the local
neutrals. Nothing it does feeds back on the circuit, the beam or the neutral
background, so the background (circuit ramp, cathode thermal state, coverage,
neutrals) owns the clock and the plasma is a *tracer* riding on it. Integrating
that leg with the full fluid solver is wasteful and, worse, inaccurate: the
density sits within a small factor of `ne_floor`, so the floor clip binds, the
`_negative_margin_timestep` bound collapses, and the run crawls at `dt_min`
through a regime whose physics is a single scalar ODE per cell.

`regime_tracer` (flag, default OFF, golden bit-exact off) replaces the fluid
update on **passive** cells with the exact integral of that ODE, and leaves the
fluid in charge of everything else. The two descriptions coexist on one grid,
so the object that has to be defined carefully is not the ODE — it is the
**interface between the passive and active regions**, which moves.

### The tracer equation

On a passive cell the plasma density obeys an **affine** scalar ODE

```
dn/dt = γ(z; background) · n + S(z, t)
```

- `γ` is a functional of the SLOW variables only — bulk ionization by thermal
  electrons, minus recombination, minus the surface/end absorption frequency.
  It is a Picard iterate: refreshed when the background it is built from has
  moved by more than `tracer_refresh_tol`, frozen in between.
- `S` is the **beam-impact ionization** birth rate: the mesh-transmitted,
  circuit-voltage-bounded beam current times the local neutral density times
  the He EII cross-section. It is independent of `n` by construction (the beam
  is launched and marched by the cathode solve, which the tracer consumes
  rather than reimplements). The beam current and birth energy come from the
  R1 objects — `cathode.circuit_available_voltage_V` composed into the sheath
  ceiling — never from the raw `cathode_phi_c_cap_V` atomic-data cap.

Neither `γ` nor `S` is re-derived here. `physics.tracer` builds both by
evaluating the solver's OWN term functions (`reactions.reaction_rates`,
`sources.boundary_absorption_rhs`, `cathode.beam_ionization_rhs_terms`) on a
probe state and dividing out the known homogeneity degree of each channel in
`n` — degree 1 for bulk ionization and surface loss, 2 for radiative
recombination, 3 for three-body, 0 for the beam birth. That is exact, and it
means the tracer automatically consumes whatever closure the run configured
(ADAS vs Janev, coverage split, CSDA vs Beer–Lambert deposition) with no
duplicated physics. `smoke_sim1d.py` asserts the identity `γ·n + S ==` the
fluid's own summed `n` row at a non-vacuum state.

The probe state carries `n = n_ref = max(n, ne_floor)` so that the ADAS
coefficients are looked up at exactly the density the fluid itself uses
(`reaction_rates` clamps its `n_safe` the same way), and each channel is then
rescaled to the true `n` by its own degree. That is why the identity above
holds bit-for-bit above the floor and remains *correct* (rather than merely
close) below it.

### Exact affine update

The step is the closed-form integral, written in the `γ → 0`-regular form so
that a vanishing growth rate is not a special case:

```
x     = γ Δt
φ₁(x) = expm1(x)/x,   φ₁(0) = 1
n⁺    = n + n·expm1(x) + S·Δt·φ₁(x)
```

`n + n·expm1(x)` rather than `n·exp(x)` because it avoids the cancellation at
small `|x|`. This update has

- **no stability limit** — `Δt` is chosen by the background, not by the plasma;
- **no floor race** — `n = 0` is a regular state (`n⁺ = S·Δt`), and a decaying
  cell relaxes onto the exact equilibrium `−S/γ` instead of oscillating about
  the density floor. The density floor is therefore NOT applied on passive
  cells; `floor_state_vector` skips them while the tracer is engaged. This is
  what lets `ne0 = 0` be a legitimate initial condition rather than a crash.
- **exact growth** — over a window in which `γ` and `S` are constant the tracer
  is not an approximation of the ODE, it is its solution.

The initial condition is a config choice, not a baked-in convention: `ne0 = 0`
runs a true-vacuum start and any `ne0 > 0` runs the existing seed convention.

The **time integral** of `n` over the step (needed by the depletion accumulator
and the conservation ledger) is likewise closed-form,
`∫n dt = Δt·(n·φ₁(x) + S·Δt·φ₂(x))` with `φ₂(x) = (φ₁(x) − 1)/x`, `φ₂(0) = ½`.
`φ₂` cancels catastrophically as `x → 0`, so it switches to its Taylor series
below `|x| = eps^(1/3)` — the standard optimum for a second-difference
cancellation, taken from `numpy.finfo(float).eps` rather than written as a
literal. `φ₂` never touches the state update; only the accumulators read it,
where a relative error of `10⁻¹⁰` in a ratio compared against a `10⁻²` criterion
is immaterial.

### Electron temperature: quasi-static local balance

`γ` needs `Te`, and the tracer does not integrate the electron energy equation.
Instead `Te` is the root of the **per-cell quasi-static electron energy
balance**, refreshed on the same cadence as `γ`:

```
1.5 · Te · S  =  P_dep(z)  −  n·nn·L₁(Te)  −  n²·L₂(Te)
```

- The left side is the **dilution cost**: the model's beam ionization births its
  electron at `Ee = 0` (the standing convention in
  `cathode.beam_ionization_rhs_terms`), so every beam-born electron has to be
  raised to the bulk temperature out of the deposited power. Bulk-ionization
  births carry the local `Te` (`Te_birth_ionization = "local"`) and therefore do
  not appear.
- `P_dep` is the beam's deposited power density plus the ohmic gap booking,
  minus the ionization cost and the excitation radiation — the same rows the
  fluid books. It is independent of `n`.
- `L₁` is the per-`n·nn` electron loss (ionization cost + electron–neutral line
  power), `L₂` the per-`n²` loss (electron–ion line power + electron–ion
  thermal exchange). Both come from `energy.electron_cooling_rhs_terms` and
  `energy.electron_ion_exchange_rhs` on the same probe state, divided by their
  own degree.

**Why this is well-posed at `n = 0`.** As `n → 0` the balance does not
degenerate: it becomes `1.5·Te·S = P_dep`, i.e. `Te → (2/3)·(deposited energy
per beam-born electron)` — a finite, positive, `n`-independent number, the
W-value of the beam in the gas. This is the reason the dilution term is the
term that must not be dropped: without it the vacuum limit has no root and `Te`
runs away. With it, `F(Te) = 1.5·Te·S + n·nn·L₁ + n²·L₂ − P_dep` satisfies
`F(0) = −P_dep < 0` and `F → +∞` (the dilution term is strictly linear in `Te`
and the loss coefficients are non-negative), so a root always **exists** in
`[Te_floor, Te_max]`.

**Uniqueness is not guaranteed a priori** and is not claimed. `L₁` rises
steeply with `Te` over the 2–15 eV window (the SCD ionization coefficient
climbs four decades), but the `Q_ei` part of `L₂` falls like `Te^(-1/2)` at
`Te ≫ Ti`, so monotonicity of `F` is a property of the operating point rather
than a theorem. The solve is therefore a **bracketed bisection** — which cannot
diverge and cannot leave the bracket — and the tracer additionally **counts
sign changes of `F` on the bracket** on every refresh. More than one sign
change means the balance is multi-valued at that cell and the description is
not usable there; that raises rather than silently picking a branch. Any
occurrence is a reportable finding, not something to be smoothed over.

Ions are not given a balance: on a passive cell `Ti = Ti_floor` (the beam-born
ion is born cold and the passive leg has no ion heating channel worth
resolving) and `M = 0` (see the interface, below — a passive cell exchanges no
momentum). Both are stated conventions, not derivations.

Parallel electron heat conduction is **not** in the balance: it is local by
construction. That is a real omission — `κ_e ∝ Te^2.5` is fast — and its effect
is to let neighbouring passive cells hold `Te` values that conduction would
flatten. The balance is a leading-order local closure and the passive leg's
`Te(z)` profile should be read as such.

### Seed transport: the quantified neglect

A passive cell's plasma does not advect (see the interface). The neglected term
is the parallel divergence `∇·(n u)`, whose size relative to the retained
growth term is

```
|∇·(n u)| / (γ n)  ≈  c_s / (L_n · γ)
```

with `L_n` the axial density scale length, at most the plasma half-length
(1000 cm on the shipped 67-cell grid, which carries 2000 cm of plasma-active
column). At the shipped `nn = 2×10¹³ cm⁻³` fill, with `c_s = sqrt(Te/m_i)` and
`γ = nn·SCD(Te)`:

| `Te` [eV] | `c_s` [cm/s] | `γ` [1/s] | `1/γ` [ms] | half-transit [ms] | `c_s/(L_n γ)` |
|---|---|---|---|---|---|
| 3  | 8.5e5  | 96     | 10.4  | 1.18 | 8.8   |
| 4  | 9.8e5  | 548    | 1.83  | 1.02 | 1.8   |
| 5  | 1.10e6 | 2.1e3  | 0.47  | 0.91 | 0.52  |
| 7  | 1.30e6 | 8.2e3  | 0.12  | 0.77 | 0.16  |
| 10 | 1.55e6 | 2.4e4  | 0.042 | 0.64 | 0.06  |

So the neglect is **not uniformly small**: growth dominates transport by ≥6×
only above ~7 eV, is a ~50% correction at 5 eV, and *loses* below ~4 eV. The
tracer is a valid description of the leg only where the quasi-static `Te` sits
in the upper part of that range, and the honest statement of its accuracy is
the last column evaluated at the run's own `Te`, not a single number. The
`active_criterion` census reports `transport_ratio = c_s/(L_n γ)` per cell
alongside the three passivity criteria for exactly this reason: it is the term
the description drops, and a run in which it is not small is a run whose tracer
leg should not be trusted.

(For context, the R2 design sketch quoted "transit ~3.6 ms vs e-folds 0.1–0.7
ms, growth dominates 5–10×". That ratio uses the *cold-seed* sound speed
(`Te = Te0 = 0.21 eV`, half-transit 4.4 ms) against the *hot* quasi-static `γ`.
Evaluating both at the same `Te` — which is what the tracer actually does —
gives the table above, and the margin is smaller. The correction is recorded
here because the code is what the reader can check.)

### The passive/active interface

Cells cross out of passivity at different times, so the active region is a
**set** with a moving boundary, not a front index.

**Passivity criteria.** A cell is passive when ALL THREE hold:

| # | criterion | measured quantity | constant |
|---|---|---|---|
| a | the plasma conducts a negligible share of the loop current | `I_cond(z) / I_loop`, with `I_cond = σ_∥(n,Te)·A_plasma(z)·(V_dev/L_plasma)` — the current the cell **actually conducts** under the applied device drop, Spitzer `σ_∥ = n e² τ_e/m_e` | `tracer_passivity_current_ratio` |
| b | the beam is optically thin to the plasma | cumulative single-pass fractional beam-energy loss to plasma electrons from the launch end to this cell, `Σ (dE/dx)_plasma Δz / E_beam` with `(dE/dx)_plasma = 2π e⁴ n_e lnΛ / E_beam`; max over cathode ends | `tracer_passivity_thinness` |
| c | the plasma has not eaten the neutrals | `D(z)/nn(z)`, `D` the running accumulator of plasma-driven (bulk, NOT beam) neutral burn since the tracer engaged, advanced with the exact `∫n dt` above | `tracer_passivity_depletion` |

Criterion (a) is the **conducted** current, not an emission capability. The
earlier sketch conflated the two; the cathode's Richardson emission capability
says nothing about whether the plasma column is shunting the loop, and the
conflation is on record as wrong. `V_dev` is the R1-bounded device voltage
(`min(cathode_phi_c_cap_V, V_avail(I))`), consumed from the cathode
diagnostics; the atomic-data cap alone would inflate `I_cond` by the same ~5×
the R1 pass removed from `V_b`.

**Hysteresis.** A passive cell becomes active when its worst criterion ratio
exceeds 1. It becomes passive again only when that ratio falls below
`1/tracer_passivity_hysteresis`. Physically all three ratios are monotone
increasing while the discharge builds, so re-entry is not an expected event —
the hysteresis exists so that a cell sitting on a criterion cannot chatter
between descriptions on background noise, which would make the run's step
sequence depend on round-off.

**Flux at the interface.** A face between an active and a passive cell is
**closed**: zero particle, momentum-advective and thermal-energy flux, with the
active cell's pressure acting on it. This is exactly the existing
`flux._apply_plasma_walls` closed-face condition, reached by composing the
tracer's passive mask into the geometry's `plasma_open`/`plasma_face_live_cell`
view, so the interface reuses the operator that is already known to keep a
uniform stationary state at zero divergence.

The alternative — a one-sided Rusanov flux against a ghost built from the
tracer's `(n, Te_qs, u = 0)` — was rejected: it would advect fluid mass into a
region whose density is owned by an ODE that does not know about it, so the
same particles would be booked twice, once by the flux divergence and once by
the tracer's next exact update. The closed face is the only treatment under
which each cell has exactly one owner.

**Conservation across the interface.** The closed face means the tracer region
and the fluid region exchange no particles, so the run's inventory is the sum
of two separately closed ledgers. That is a *choice with a cost*, and the cost
is exactly the neglected `∇·(n u)` of the previous section: the plasma that
would have crossed the interface is the same plasma the seed-transport neglect
drops. The ledger books it as an explicit zero row
(`tracer_interface_particle_flux`) rather than dropping it silently, so a
reader can see that the term exists and is zero by construction rather than
absent by oversight. `regime_r2_handoff_check.py` closes the two-part ledger to
a stated tolerance.

**Activation handoff.** When a cell leaves passivity AND its tracer density has
reached `tracer_activation_ne`, its packed rows are written from the tracer
(`n` from the exact update, `Ee = 1.5·n·Te_qs·ev_to_erg`,
`Ei = 1.5·n·Ti_floor·ev_to_erg`, `M = 0`) and the fluid owns it from the next
step. Both conditions are required: passivity failing at a density the fluid
cannot represent (near `ne_floor`, where the clip binds) would hand the fluid
exactly the floor-poisoned state the tracer exists to avoid.

When the LAST passive cell activates the tracer disengages for the rest of the
run. Handing the whole column to a *separately configured* main arm is a
**state transfer**, and the instrument for it is the restart machinery
(`results/restart.py`, `RESTART.md`): stage 1 runs the conducting leg with
`regime_tracer` on and exports at the handoff instant; stage 2 resumes with the
flag off. The restart's structural-key check is what guarantees the two stages
agree about what the stored fields mean.

**DVM is refused.** `results/restart.py` refuses `kinetic`/`kinetic_dvm`
neutral models because it does not serialise a distribution function, and the
tracer refuses them at construction for the same reason plus a second one: the
tracer's `γ` is built from moment-model neutral densities. R2 is fluid-arms
only and does not extend DVM support.

### Registered criterion constants

No anonymous threshold appears in the tracer. Every description-selecting
number is a config key, sweepable by config alone (`ε/3`, `ε`, `3ε`) with no
code edit, and carries a classed provenance entry in
`core/config_defaults_provenance.md`.

| key | symbol | shipped | class |
|---|---|---|---|
| `tracer_passivity_current_ratio` | ε_I | 0.01 | DERIVED |
| `tracer_passivity_thinness` | ε_thin | 0.01 | ASSUMED |
| `tracer_passivity_depletion` | ε_dep | 0.01 | ASSUMED |
| `tracer_passivity_hysteresis` | h | 3.0 | ASSUMED |
| `tracer_refresh_tol` | — | 0.01 | NUMERICS |
| `tracer_activation_ne` | n_act | 1.0e10 | DERIVED |
| `tracer_overlap_band_ne` | — | (1.0e10, 1.0e11) | DERIVED |
| `tracer_overlap_rtol` | — | 0.05 | ASSUMED |

### The census

Every tracer run reports, from day one, **which criterion binds where and
when** — the `active_constraint` idiom the timestep limiter already uses. Per
save frame the result carries `tracer_criterion` (the binding criterion name
per cell), the four ratios, the passive mask, `Te_qs`, `γ`, and the refresh
count; at the end of a tracer run `run()` prints a one-line summary naming the
criterion that bound most often, the cell and time of first activation, and
whether the transport ratio stayed small. It is presence-gated on the flag, so
a non-tracer run prints and records nothing.

### Gates

- `regime_r2_overlap_gate.py` — the **two-sided** gate. Over the registered
  overlap band, where both descriptions are valid, run both and compare; PASS
  iff the densities agree within `tracer_overlap_rtol`. The band and the
  tolerance are registered in the script header (and in config) before the
  comparison is implemented.
- `regime_r2_handoff_check.py` — conducting leg via tracer → restart export →
  full solver resume; asserts the restart config-identity checks, state
  finiteness, and closure of the two-part conservation ledger.
- `smoke_sim1d.py` — affine-update exactness against a closed-form two-cell
  case (including `γ → 0` and `n = 0`), presence-gating (byte-identical
  trajectory with the flag off), each construction-time `ValueError` class,
  and an anti-vacuity variant per guard that must fail if the guard is removed.

### Explicitly out of scope at stage 1

The `V_cm` vessel/common-mode node (a one-ODE floating closure) is **not**
built: stage 1 is wall-referenced, `V_cm ≡ 0`. The documented seam is the
`V_dev` read in criterion (a), which is where a common-mode offset would enter.
The φ_a-aware `V_b`-object bound is an R1 follow-up in the cathode solver; the
tracer consumes the R1 bound's objects as they are and does not touch its
contract.
