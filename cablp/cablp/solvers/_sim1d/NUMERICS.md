# sim1d Numerical Methods

Brief outline of the numerical schemes used by `LAPDSim1D`
(`cablp/solvers/_sim1d/`). This is a conservative, axially-resolved 1D LAPD
transport model (1D interior with 0D boundary cells). For the continuous model
equations these schemes discretize, see [`MODEL.md`](MODEL.md).

## State and discretization

- **Conservative variables** (`core/state.py`, `ConservativeState1D`): electron
  density `n`, neutral density `nn`, parallel momentum density `M`, and electron
  and ion energy densities `Ee`, `Ei`. Primitive quantities (`Te`, `Ti`, `v`)
  are recovered by `derive_state`. The optional neutral reductions append
  rows in introduction order: column/chamber neutral momentum `M_n`,
  annulus density `nn_a`, (only for `kinetic_two_moment`) annulus momentum
  `M_n_a`, and (only for `neutral_energy`) neutral thermal energy `En`, which
  carries the neutral temperature as `Tn = (2/3) En / (nn k)` on the volume
  `nn` itself lives on. Existing 5-, 6-, 7- and 8-row layouts are unchanged.
  In the current package `neutral_momentum`, `neutral_two_zone` and
  `neutral_energy` are all **on by default**, so `M_n`, `nn_a` and `En` are
  present by default; `neutral_momentum_radial` defaults to `"uniform"`, so the
  `"kinetic_two_moment"`-only row `M_n_a` is not.
- **Grid** (`core/geometry.py`): finite-volume cells along the axial (`z`)
  coordinate with cell-centered states and face-based fluxes. Plasma and neutral
  fields carry separate face areas and cell volumes so inventory
  (`area × flux`, `volume × density`) is tracked consistently.
- **Floors**: density and temperature floors are enforced after every stage
  (`floor_state_vector` / `apply_state_floors`) to keep the state physical.
  They are numerical admissibility limits, not initial conditions: the live
  repaired seeds are `Te0=0.21 eV` and `Ti0=0.026 eV`, each strictly above its
  own floor (`Te_floor=0.1 eV`, `Ti_floor=0.02585 eV`). A raw-validation config
  rejects floor-bound
  initial temperatures at construction.

## Spatial discretization (finite volume)

- **Plasma advection** (`physics/flux.py`): a **Rusanov / local Lax–Friedrichs**
  flux at each interior face,
  `F = ½(F_L + F_R) − ½·a_max·(U_R − U_L)`,
  with `a_max` from the local ion sound speed. Every face with `plasma_open`
  False is **closed** (`flux._apply_plasma_walls`): it carries no particle or
  thermal-energy flux and keeps the live cell's pressure as its momentum flux
  — the 0D closed-wall behavior. Under `resolved_boundaries` that closed set
  is not the two domain ends but every face bounding the plasma inside the
  neutral domain. At the plasma-terminating (absorbing) subset the treatment
  depends on `characteristic_boundary` (**on by default**): with it on the
  advective flux carries NOTHING there, momentum included, because the
  one-sided ghost-cell Bohm outflow (`sources.characteristic_boundary_rhs`)
  supplies the particle, momentum and energy flux together with its own
  pressure term and the wall pressure on top would double-count; with it off
  the closed-wall form above stands, which is the 0D legacy behaviour.
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
- **Cold neutral fluid mini-flux** (`neutral_energy` only,
  `physics/neutrals.py:neutral_fluid_flux_rhs`): with an evolved `En` the
  neutral gas is transported as a fluid rather than a drifting density. A
  Rusanov flux on `(nn, M_n, En)` with `a_max = |u_n| + c_n(Tn)` carries the
  COLD channel's own pressure `p_n = (2/3) En` in the momentum flux, and the
  energy row splits exactly as the plasma's does: pure `En u_n` advection in
  the flux plus a separate `−p_n ∇·u_n` pressure work. The quasi-1D wall
  reaction `p_n·ΔA/V` accompanies the area-weighted pressure flux (the same
  pair `flux_tube_geometry` supplies for the plasma), so a uniform gas at rest
  is bit-exactly stationary. It **supersedes** the donor-cell wind advection
  below rather than composing with it — the solver runs exactly one neutral
  advection operator and the ledger name `neutral_wind_advection` is unchanged.
  Rows are divided by the volume of the field each transports: `nn` and `En` on
  the column under `neutral_two_zone` and the chamber otherwise, `M_n` always
  on the chamber (it is a chamber-mean momentum), and the pressure force
  crosses the area the pressure acts on and lands on the momentum's volume.
  The hot, CX-born neutral population is collisionally decoupled and its
  pressure is deliberately absent from `p_n`.
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
  thermalization, and (`ion_neutral_moment_closure`, **on by default**, audit
  A7/A8) the
  moment-closed reduced ion-neutral collision operator that replaces that quartet
  with one Phelps-rate equal-mass Braginskii momentum-transfer term (MODEL.md
  R4.3). The Phelps He⁺/He rate coefficients are built once at import in
  `funcs/_cross.py` (Maxwellian averages of the analytic isotropic + backscatter
  cross sections vs `T_eff`), analogous to the IAEA `charge_ex_react` table.
- `cathode.py` — cathode/sheath boundary physics.

## Time integration

- **Explicit path** (`core/integrator.py`) — the whole step only when
  `implicit_heat_conduction` is off; otherwise it is operator `A` of the
  default operator-split path below. A **two-stage strong
  stability preserving Runge–Kutta (SSPRK2 / Heun)** step,
  `y¹ = floor(y⁰ + Δt·L(tⁿ, y⁰))`,
  `yⁿ⁺¹ = floor(½y⁰ + ½(y¹ + Δt·L(tⁿ+Δt, y¹)))`,
  with floors applied at each stage. The stages are evaluated at `tⁿ` and
  `tⁿ+Δt`, which preserves second-order accuracy for explicitly time-dependent
  forcing such as the gas-puff schedule. `ssprk2_step` freezes the forcing at
  the step start when its `time` argument is omitted, which is only
  first-order accurate in that forcing.
- **Operator-split path** (**the default**: `implicit_heat_conduction` is on
  by default and the live golden-at-stance fixture runs it; also reachable
  from the CLI as `--operator-split`): an explicit SSPRK2 step over all
  **non-heat** terms
  (operator `A`) composed with an implicit heat-conduction substep
  (`implicit_heat_conduction_step`, operator `B`) solved per species as a
  tridiagonal system via `scipy.linalg.solve_banded`. This removes the stiff
  parabolic heat-conduction stability limit from the explicit timestep.

  `operator_splitting` selects the composition: `"lie"` does
  `A(Δt)` then `B(Δt)` and is O(Δt), since the splitting error goes as
  `Δt·[A,B]`; `"strang"` (**the current package default**) does
  `B(Δt/2) → A(Δt) → B(Δt/2)`, whose symmetry
  cancels that leading commutator term and leaves O(Δt²). `B` is the halved
  operator because it is the cheap one — banded solves against a tridiagonal
  matrix, versus `A`'s reaction-rate evaluations — so Strang costs one extra
  heat substep, not one extra explicit step.

  The substep discretization is selected by the `implicit_heat_scheme`
  parameter. Three of the four are **theta methods**, solving
  `(C + θ·Δt·K)·Tⁿ⁺¹ = C·Tⁿ − (1−θ)·Δt·K·Tⁿ`:

  | `implicit_heat_scheme` | θ | `R(−∞)` | L-stable | solves | substep order |
  |------------------------|-----|--------|----------|--------|---------------|
  | `backward_euler`           | 1   | 0    | yes | 1 | 1 |
  | `shifted`                  | 0.6 | −2/3 | no  | 1 | 1 |
  | `crank_nicolson`           | 0.5 | −1   | no  | 1 | 2 |
  | `tr_bdf2` (**default**)    | —   | 0    | yes | 2 | 2 |

  **Naming the object.** The current package defaults are
  `implicit_heat_scheme="tr_bdf2"`, `operator_splitting="strang"` and
  `heat_picard_iterations=2` — the second-order production package, shipped at
  the R5 flip (2026-07-25). `backward_euler` + `"lie"` +
  `heat_picard_iterations=0` is the historical FIRST-order package; the three
  are independent first-order error terms, so falling back on any ONE of them
  caps the whole step at first order. The live golden-at-stance fixture is
  captured at the **stance of record** — `default_config()` plus the committed
  `scripts/stances/g1atrim.toml` (minus that stance's mesh-sized package) plus
  `nx = 60` — not at the package defaults; the three time-integration keys its
  JSON sidecar (`scripts/baselines/production_discharge.json`) records,
  `tr_bdf2`, `"strang"` and `heat_picard_iterations=2`, ARE the package
  defaults, and the sidecar is the authority for what the fixture ran.

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
  default 0.25, and the fixed `conduction.HEAT_DT_FRACTION` = 0.25) so no
  single explicit source/sink changes a
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

- **Current-driven loop relaxation** (`circuit`, `circuit_dt_fraction`,
  default 0.25), **presence-gated on `cathode_circuit_voltage_bound`** and on
  a live loop, so an unarmed run neither evaluates it nor moves. The loop
  equation's local time constant is `tau_circuit = L / (R_comp + R_mesh +
  dV_dis/dI)`, with the device slope read by a one-sided finite difference of
  the same `vdis_of_I` the TR-BDF2 advance integrates (two extra sheath
  solves, which also yield `f(I)` free). It exists because neither `L/R_comp`
  (1.12 ms) nor the bank `RC` (68.6 ms) is the stiff mode: the **sheath
  capability wall** is, at a measured device slope up to ~2 kOhm — `tau` ~ 4
  ns, crossed by the sub-wall slew in ~45 ns — and the controller previously
  carried no circuit term at all, so the adaptive path stepped `dt_max` (1e-4
  s) straight across it. TR-BDF2 is L-stable, so this is an **accuracy**
  bound, not a stability one.

  It is **withdrawn once the loop reaches its local equilibrium**, tested as
  `|f(I)| * tau_circuit < dI_probe`: the distance to that equilibrium, in
  amperes, below the resolution of the finite difference the bound is built
  from. This is the phantom rule below applied to the circuit rather than an
  optimization. The relaxation rate depends only on the device slope, which
  stays steep at a stiff FIXED POINT, so a slope-only bound pins the step
  there forever — measured on the conducting-phase stance: `dt` held at
  1.64e-10 s while `I_loop` stood at 0.894481 A to six figures, ~122000 steps
  to cross a 20 us window containing no circuit transient. With the
  withdrawal the same window costs 53 steps, 7 of them `circuit`-bound
  through the actual transient.

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

### The DVM transfer hold (`neutral_kinetic_dvm_transfer_hold`)

The transient DVM books a plasma-side momentum/energy transfer **once per
neutral clock tick** and the plasma steps many times inside that tick. How the
plasma applies it between ticks is a time-discretization choice, and it is the
one this selector makes. Default `"exponential"`; `"zoh"` is the pre-2026-08-24
behaviour, kept as the negative control.

**The CX/elastic pair is not a source, it is a relaxation.** The booked
transfer splits exactly into the ionization/recombination rows — genuine
sources, which stay constant within the tick under either selector — and the
charge-exchange/elastic pair, which is

$$\frac{dE_i}{dt}=-\nu\,(E_i-E_i^{\rm eq}),\qquad
  \frac{dM}{dt}=-\nu\,(M-M^{\rm eq}),$$

with **one** `nu` per cell for the pair (one physical exchange, one rate),
`nu = N_loss / (V dt n_i)` the collision rate per ion, and the targets set by
the measured moments of the lost neutral population:
`M_eq = m n_i u_n,eff` with `u_n,eff = P_loss/(m N_loss)`, and
`Ei_eq = (3/2) n_i k T_eff` with

$$\tfrac32 k T_{\rm eff}
  = \frac{E_{\rm loss}-u_i P_{\rm loss}+\tfrac12 m u_i^2 N_{\rm loss}}
         {N_{\rm loss}} .$$

`T_eff` is the second moment taken about the **ion** drift, so it carries the
frictional term `(m/3k)|u_n − u_i|²` by construction. That term is *not*
small — ~0.3 eV at the collector end cell of the g1atrim arm — so `Ei_eq` must
never be built from a Maxwellian at `T_n`. Everything above is published by
the engine at booking (`nu_pair`, `M_transfer_pair`, `Ei_transfer_pair`,
`u_n_eff`, `T_eff_eV`); `T_eff` is linearized at the tick's `u_i`, which is the
same freeze the tick makes everywhere else.

**Why the zero-order hold fails.** Freezing the booked *rate* across the tick
advances the pair by `X_{k+1} = X_k − νΔt(X_k − X_eq)`, i.e. multiplies the
distance to the target by `1 − νΔt` every tick. For `νΔt > 2` that is an
amplification with a sign flip: the booked `Ei` alternates sign every one or
two ticks with growing amplitude, the heating half-cycles are unbounded, the
drain half-cycles drive the cell onto the `Ti` floor, and the K2d relax
limiter and the `surface_loss` bound then crawl at `dt_min`. Measured on the
2026-08-24 g1atrim `kinetic_dvm` arm: `νΔt` 1.4 at 11.5 ms rising to 3.8 at
12.02 ms (`nn` 6.5e13 against `n` 1.2e12 at the collector end), 184,101 of
199,999 steps limited, and the run spending 184,475 of its 200,000 steps in
the 0.24 ms after t = 12.0 ms.

**The exponential hold.** Per cell, per plasma step of length `dt`, at the
tick's frozen `(nu, target)`:

$$E_i \leftarrow E_i^{\rm eq}+(E_i-E_i^{\rm eq})e^{-\nu\,dt},$$

and the momentum row at the same `nu`. Applied as a constant rate over the
step, so the SSPRK2 stages integrate it exactly. It is unconditionally stable,
exact for the linearized system, cannot carry a row past its target at any
`dt`, and reduces to the zero-order hold to `O(ν dt)`.

**Hold debt.** What the plasma applies then differs from what the tick booked,
because the neutrals already received their births at `M_i(T_i^{\rm tick})`.
That difference is carried as a per-cell **hold debt**, kept SEPARATE from the
floor debt of `neutral_kinetic_dvm_transfer_relax_fraction`: floor debt says
*the plasma could not absorb it*, hold debt says *the tick froze a rate at a
state that then moved*. Hold debt is therefore the **cadence meter** — it is
first-order in `νΔt` and vanishes as the neutral clock refines. The ledger
identity is

    applied_cum + debt + hold_debt == booked_cum

per cell at every accepted step, and is checked to roundoff by
`scripts/verify_sim1d_dvm_hold.py` (case `ledger-closure`) and reported in the
saved `dvm_transfer_ledger` group.

**Repaying it is where the subtlety is.** Spreading the outstanding debt `D`
over the following tick as a flat `D/Δt_tick` source re-injects exactly the
zero-order increment the hold removed, and the coupled `(gap, debt)` map goes
unstable again at the same `νΔt ≈ 2` — with a margin that depends on how many
plasma steps happen to fall inside a tick, which is not a property anything
should rest on (measured at four steps per tick: growth at `νΔt` = 4 and 20).
The repayment is therefore offered **through the relaxation**, as
`D φ(ν dt)/Δt_tick` with `φ(x) = (1−e^{−x})/x`. That delivers exactly
`D/Δt_tick` in the resolved limit (`φ → 1`) and damps it by the same
exponential when the tick is coarse. The per-tick map is then

$$\begin{pmatrix}g\\ D\end{pmatrix}_{k+1}=
  \begin{pmatrix} e^{-X} & a\\ -(X-1+e^{-X}) & 1-a\end{pmatrix}
  \begin{pmatrix}g\\ D\end{pmatrix}_k ,\qquad
  a=\frac{1-e^{-X}}{X},\quad X=\nu\,\Delta t_{\rm tick},$$

whose determinant is exactly `1 − a` and whose trace is `e^{−X} + 1 − a`, so
both eigenvalues lie strictly inside the unit circle for every `X > 0`,
independently of how the tick is subdivided, and the only fixed point is
`g = D = 0` — the debt is driven to zero, not merely bounded. Spectral radius:
0.899 at `X` = 0.1, 0.607 at 1, 0.869 at 4, 0.975 at 20. The battery asserts
the shipped arithmetic against that closed form to `1e-9` relative on every
row that is genuinely linear.

**Accuracy is a separate, conditional statement.** A tick spanning twenty
e-folds cannot be integrated *accurately* by any scheme frozen at its start;
what the hold guarantees there is stability and a bounded, self-retiring debt
that says the cadence is too coarse. Measured on the synthetic cell against a
finely integrated reference, the error is within `2 νΔt` for `νΔt ≤ 1`.

**The golden is unaffected by construction**: the moment neutral path never
builds a DVM and never enters this code.

The `"zoh"` branch takes the pre-hold expressions unchanged, books no hold
debt, and exists so a pre-fix artifact can be reproduced and so the
instability above can be exhibited on demand rather than remembered.

### The counted ionization debit, and why it is taken last

The counted-particle handshake makes the plasma and the neutral arm destroy
the *same* atoms: whatever count the plasma books as ionization over a tick is
exactly what leaves the column, with the march's own frequency tally
reconciled to it (`_debit_booked_ionization`). The reconciliation is a
renormalization — the march already removed `sum(L_ion)`, and the remainder is
taken from the cell in proportion to what the cell holds.

**Which population "what the cell holds" means is the whole of the ordering
question.** Substep A's march removes ionization, charge exchange and elastic
scattering together, but only ionization is a real loss: the CX/elastic pair is
re-born in the same cell, at the same count, inside the same tick. The debit is
therefore applied **after** those re-births, against

$$f_c^{\rm marched} + \text{birth}_{cx} + \text{birth}_{el},$$

which is the inventory the cell genuinely carries once the tick ends, net of
the non-conserving losses only. The alternative — capping against the pre-march
inventory less the marched ionization — is *not* equivalent and is not used:
the drop has to be drawn from an array that actually exists, and a cap computed
against one population while the drop is taken from a smaller one can drive a
bin negative. Taking both from the same array makes positivity structural
rather than checked.

Two identities pin the choice, and both are exact at every tick:

- the per-cell particle handshake `ion_removed_cum + ion_debt ==
  ion_booked_cum`, and the ledger's own
  `inventory_after − inventory_before == births − losses`;
- the energy ledger. The re-birth counts `N_cx`, `N_el` are still tallied from
  the **marched** state, before anything is debited, so the CX/elastic channel
  amounts and the energy booked for them — `N × E[M_i(T_i^{\rm tick})]`, at the
  birth spectrum — are untouched by the ordering. The ionization energy row is
  the moment of the bins the channel actually removed, so it closes whichever
  population those bins came from.

**This changes which atoms the debit takes, not only how many, and it does so
on every counted tick — not merely where the positivity cap binds.** The
marched remnant and the re-births are different spectra (the latter sits at the
ion Maxwellian, hotter and drifting with the ions), so the reconciliation now
draws from the mixture. That is the *velocity-blind* reading the channel-1
convention asks for: `nu_ion` is velocity-blind, and the population it must be
blind about is the one that is there. Drawing only from the marched remnant,
while the re-births sit untouched in the same cell, is the velocity-biased
choice.

**What the pre-2026-08-24 ordering did.** The debit ran before the re-births,
so `held` was the marched state stripped of CX and elastic as well. A cell
whose booking was smaller than its true post-tick inventory but larger than
that stripped remnant was told it could not pay: the positivity limiter fired
against atoms that had never left, the shortfall went to `ion_debt`, and
because the same thing happened on the following tick the debt was monotone and
never retired. On the B0c fixture this failed R13 at every cadence at or
coarser than 2.5e-5 (42 shortfall cell-ticks at 2.5e-5 over cells {1,5,6}) —
an artifact of the ordering, not a statement about the cadence. With the debit
taken last the shipped 2.5e-5 arm carries **zero** shortfall updates and an ion
debt ratio of 1.2e-19, and R13 discriminates cadence as intended: 5.0e-5 still
fails, and the probe reason is genuine exhaustion — at its first binding tick
the plasma books 1.086× the cell's *entire* pre-tick inventory, which no
ordering can satisfy (`max nu_ion*dt_n` = 2.8 against the R14 bar of 0.1).

Registered as gate `D5` in `scripts/verify_sim1d_k2_dvm.py`, which pays every
booking up to a synthetic cell's whole inventory with zero shortfall and runs
the pre-fix ordering beside it as a negative control — the control fires
exactly above the closed-form threshold `1 − N_{cx,el}/I_0` and not below it.
**The golden is unaffected by construction**: the moment neutral path never
builds a DVM and never enters this code.

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

**The lock counts the ACCEPTED step, not only the raw candidate minimum**
(2026-08-24). `clamped_to_dt_min` is computed inside `suggest_timestep`, i.e.
*before* the run loop's caps (`dt_growth`, `t_end`, `phase_boundary`,
`save_time`) and before the retry ladder, every one of which can only shrink
the step. So an accepted step can land at or below `dt_min` while no candidate
ever asked for less than `dt_min` — the raw flag stays clear, the consecutive
counter RESETS, and that accepted sub-`dt_min` step becomes
`previous_accepted_dt`, which anchors the ×`dt_growth_factor` ramp. Every step
of the grind that follows is then set by the ramp re-approaching from below,
not by a physics bound, and `dt_min_lock_max_steps` never fires. (Measured:
the g1atrim `kinetic_dvm` arm of 2026-08-24 spent 184,475 of its 200,000 steps
after t = 12.0 ms, 162,055 of them capped by `dt_growth`, with an accepted dt
as low as 3.05e-12 against `dt_min` = 1e-10 — and no lock.) The counter is
therefore driven by `clamped_to_dt_min OR (accepted_dt <= dt_min AND dt_raw >
dt_min)`, evaluated after acceptance; the second conjunct is what keeps it a
clamp rather than a step count — the step must have been pushed under the
floor by a *cap*, so a run configured with `dt_max` at `dt_min` (which accepts
`dt_min` every step by construction) is not counted, and a step whose raw
minimum is itself below `dt_min` is already owned by the raw flag. Both facts
are recorded, the raw one in `clamped_to_dt_min` and the new one in
`clamped_to_dt_min_accepted`, and the lock's error message names which signal
fired. The step SEQUENCE is unchanged — the snap still
lands on the save time, and the trajectory of any run that does not trip the
lock is bit-identical.

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
floor, while raw rejection remains the backstop. The R1-era checkpoint golden
pinned `raw_stage_validation` off explicitly through the baseline driver's
override table so the checkpoint stayed reproducible; that override was
dropped at the R2b re-anchor, and the live golden-at-stance fixture runs the
selector on, as `default_config()` always has. The retired checkpoint is
reproducible only at the `pre-refactor-2026-08-20` anchor.

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

Two selectors make the plasma hyperbolic update discretely total-energy
conservative. Both are shipped defaults now, and the golden fixture runs them
so since the R2b re-anchor captured it at the stance of record.

- `hyperbolic_wave_speed`: `"isothermal"` (`sqrt(Te/m_i)`) or
  `"adiabatic"` (default, `sqrt((5/3)(Te+Ti)/m_i)`, the exact spectral radius of
  the γ=5/3 two-species system). It sets the Rusanov `a_max` and the plasma CFL —
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

As introduced at R3 the `characteristic_boundary` selector was default-off, and
default-off was bit-exact against the R3-era checkpoint golden; in the current
package it is **on by default**. The selector changes how the plasma-
terminating faces are discretized.

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

## R5.2 electron heat-flux limiter (default on, audit A9)

The UNLIMITED electron conduction is classical Spitzer–Härm (`q = -κ_e ∇Te`),
a local law valid only where `λ_mfp ≪ L_T`. That is the arm
`electron_heat_flux_limit=False` selects; it is not what the package ships.
A9 measured `q_SH` reaching 1.7–3.3×
(static probe: ~4× median) the free-streaming ceiling `n·Te·v_the` at the resolved
gap faces — the constitutive law leaving its validity domain. The
`electron_heat_flux_limit` flag — default-off as introduced at R5.2, **on by
default** in the current package — scales `κ_e` per cell by the harmonic
limiter `λ = q_sat/(q_sat + q_SH)` (Malone, McCrory & Morse, PRL 34 (1975) 721;
equivalently Fundamenski, PPCF 47 (2005) R163, eq. 10a), riding on the Cowie &
McKee, ApJ 211 (1977) 135, eq. (7) free-streaming ceiling `q_sat = f·n·Te·v_the`
(`f = heat_flux_limiter_f`), so the flux caps at free-streaming where gradients are
steep and recovers Spitzer where they are shallow (`flux_limited_electron_conductivity`,
applied in both the explicit and implicit paths at the frozen incoming `Te`, so the
operator stays a conservative flux divergence). Identities
(`verify_sim1d_r5_heatflux.py`): Spitzer limit at large `f`, saturation cap
`κ_eff|∇Te| ≤ q_sat`, closed-domain energy conservation. Default off as
introduced at R5.2 and bit-exact there against the R5-era checkpoint golden;
**on by default in the current package**, so the live golden-at-stance fixture
runs the limited form. The shipped `f` is **BOXED (literature), not fitted**,
and carries a bracket of record; its value, class and bracket live in
`core/config_defaults_provenance.md` and
`scripts/production_stance_provenance.md`, which are the authority — this
document names the flag, not the number. The two `f` values that appear in the
static A9 engagement probe are declared **arms of the closure-family bracket**,
never the stance: `f=1` targets only the ~gap
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
  ceiling — never from the raw `cathode_phi_c_cap_V` atomic-data cap. (The
  loop equation's own integrand is the one thing that does NOT read the
  bounded objects; see MODEL.md, "What the bound does not constrain".)

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
- `L₁` is the degree-1 electron loss (ionization cost, electron–neutral line
  power, and the `1.5·Te` the surface absorption carries out with each lost
  particle), `L₂` the degree-2 loss (electron–ion line power + electron–ion
  thermal exchange). All come from `energy.electron_cooling_rhs_terms`,
  `energy.electron_ion_exchange_rhs` and `sources.boundary_absorption_rhs` on
  the same probe state, divided by their own degree. The surface term is in the
  balance because it is genuinely per-cell and because `γ` already consumes the
  same term function for the particle channel; taking one row without the other
  would be an inconsistency in the tracer.

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

Parallel electron heat conduction is **not** in the balance: the balance is
local by construction and conduction is not a local term.

#### MEASURED: the local balance has no root at the production stance

This is the flagged design risk, and it fired. It is recorded here rather than
worked around, and **the tracer raises `TracerBalanceError` rather than
producing a number** wherever it happens.

Measured on the production-stance ES1 arm (`nx = 20`, current-driven cathode,
CSDA + quasilinear deposition, circuit voltage bound on) at
`t = 1.0423e-05 s`, the first instant the beam is live:

| `Ee` row at cell 2 (cathode) | erg cm⁻³ s⁻¹ | at cell 7 (column) |
|---|---|---|
| `beam_power_deposition` | **+8.851e5** | +5.533e5 |
| `heat_conduction` | **−3.633e5** | **−8.635e5** |
| `plasma_advective_flux` | +1.279e5 | +4.095e4 |
| `electrode_e_sheath_loss` | −6.910e4 | **−9.974e5** |
| `ionization_energy_cost` | −5.511e4 | −6.082e4 |
| `electron_neutral_cooling` | −2.152e4 | −2.438e4 |
| `anode_collection` | 0 | −5.454e4 |

The two dominant sinks are **parallel heat conduction** and the **boundary
losses** — each an order of magnitude larger than every local radiative
channel. A purely per-cell balance cannot see conduction at all, so at the
actual pre-breakdown density it has **no root**: the bisection bracket's top
end still has `G < 0`. The fluid itself sits at `Te ≈ 49–61 eV` in those cells
at that instant, so the model is not producing runaway electrons; the local
closure simply omits what is holding them down.

The arithmetic behind it is worth stating, because it also shows why the
vacuum-limit argument above, while correct as an argument, does not rescue the
production point. Over that solve the beam launches 4182 W and the deposition
module books **4161 W (99.5%) as plasma heating** — Coulomb drag plus the
terminal residual — while creating `3.14e18` ionization events/s. That is
**8.3 keV of deposited energy per beam-born electron** (only 1.7% of the beam
electrons ionize, so `177 eV / 0.017`), against an atomic W-value of tens of
eV. The dilution term would therefore have to carry `Te ≈ 5.4 keV`, far above
both the ADF11 grid and the `(2/3)·E_beam = 118 eV` hard ceiling. The beam's
"W-value in the gas" **in this model** is a transport quantity, not an atomic
one.

Scanning the density at that same background, holding everything else fixed:

| density | outcome |
|---|---|
| ×1 (actual, `n ≈ 5e9`) | **no root** |
| ×10 | root, `Te` = 22.9 eV (cell 2), 49.4 eV max, 1 sign change |
| ×100 | root, `Te` = 5.9 eV (cell 2), 9.6 eV max, 1 sign change |
| ×1000 | root, `Te` = 1.2 eV (cell 2), **2 sign changes** — MULTI-VALUED |
| ×10⁴ and above | root at the floor |

So both flagged failure modes are real: non-existence at the density the tracer
is meant to run at, and multi-valuedness three decades above it. Adding the
surface-loss row to `L₁` (which the balance now carries, on consistency
grounds) moves `Te` at ×10 from 75.5 eV to 22.9 eV but does **not** create a
root at ×1 — it is a completion of the local object, not a repair of the
non-local omission, and must not be read as one.

**Consequence.** The affine density core, the passive/active interface, the
criteria and the census are all independent of this and stand. The
`Te`-closure, as a per-cell object, does not describe the production
pre-breakdown leg. Resolving it is a design decision — a `Te(z)` two-point
boundary-value closure that carries conduction, or an evolved (rather than
quasi-static) electron energy on the passive set, or a beam-booking change —
and none of those is a fix to be improvised inside the tracer. Until it is
resolved the tracer cannot run at the production stance, and the two run-level
gates below report that rather than a number.

#### Corrected beam power booking on passive cells (AMENDMENT)

The section above is the record of a measurement and stands as written. This
block is what came of it: the balance's failure was traced to the **booking**
the balance was fed, not to the balance, and the correction restores a root at
the stance. Both are kept, because the second only makes sense against the
first.

**The diagnosis.** Of the beam power the deposition module books into plasma
electrons at the pre-breakdown stance, **91.8%** is the quasilinear/anomalous
channel — 3978 W of the 4334 W it banks, against 0.36 W of collisional Coulomb
drag. Per cell it is 95.3% of the `beam_power_deposition` row at the cathode
cell and 91.6% at the column cell. But quasilinear absorption is a
beam-**plasma** instability: the beam's energy goes into a Langmuir wave that
the plasma then damps. At vacuum-class density there is no wave medium, so the
channel does not exist and booking its power was describing an interaction with
a plasma that is not there. That single channel is the whole of the 8.3 keV per
beam-born electron the section above reports: with it refused the figure is
**607 eV**, and the local channels can absorb what remains.

**The correction.** On a cell the tracer owns, the anomalous share of the
deposited power is subtracted from the beam power the quasi-static balance
absorbs. What survives on a passive cell is what does not need a wave medium:
collisional beam drag on plasma electrons (degree 1 in `n`, and accordingly
tiny at vacuum), the end-of-range terminal dump, the secondary-electron
residue, and the ionization-birth bookkeeping — all unchanged.

**The gate is the passive mask and nothing else.** No density threshold is
introduced and no constant is registered: the tracer→fluid handoff and the
onset of quasilinear absorption are made the *same event* by construction. An
ACTIVE cell books the anomalous channel in full, exactly as before, and the
fluid path — every cell active, tracer off or absent — is untouched and
bit-exact. `physics.tracer.passive_anomalous_leak` is the auditable invariant
(zero on every passive cell); it recomputes the anomalous share through its own
reference rather than through the subtraction it audits, so a removed refusal
is still caught, and `smoke_sim1d.py` removes the refusal and checks that it is.

`regime_tracer` refuses `beam_anomalous_model != "none"` with a non-CSDA
deposition model at construction: the anomalous channel exists only on the CSDA
rays, so such a run reads as though the correction is doing work when neither
the channel nor its refusal is live.

**Where the refused power goes (convention).** The quasilinear power refused on
a passive cell is neither deposited nor destroyed: **the beam keeps it.** With
no wave medium the primary is not slowed by that channel, so the energy stays
in the beam and is carried along the ray, out of the tracer's domain, to
whatever terminating surface the ray reaches — the far end or the wall. The
tracer phase carries **no wall-load ledger**, so that arriving power is not
booked anywhere today and no row reports it; this paragraph is what records
that it exists and where it goes, so its absence from the ledger is a known gap
rather than a silent one. The convention is stated in terms of the terminating
surface, not of the tracer, precisely so that a build which adds a vessel /
common-mode node can attach the far-end power to that node as-is: the
destination does not change, only whether something is listening at it.

**The balance is solved on the PASSIVE SET and nowhere else.** A cell the fluid
owns has its own electron energy equation, integrated with conduction and the
boundary terms in it, so the local quasi-static closure — which cannot see
either — was never a description of it. Two things follow, and both are the
rule rather than a consequence of it:

- the balance is never consulted for an active cell, so it can never refuse on
  one (an earlier build did, at `t = 7e-5`; see Gates);
- everything that reads a temperature on an active cell — criterion (a)'s
  Spitzer conductivity, criterion (b)'s stopping power, the re-entry side of
  the hysteresis, and the census `Te_qs` field — reads **the fluid's own `Te`**
  for that cell. Off the passive set the balance's output is a
  floor-by-convention filler and means nothing, so publishing or acting on it
  would have described a cold cell wherever the fluid was in fact running hot.

The per-cell values below are unaffected by this: the bisection is independent
per cell, so restricting which cells are solved changes which cells can raise,
not what any solved cell returns.

**RE-MEASURED**, same stance, same instant (`nx = 20`, `t = 1.0423e-05 s`,
fluid-arm background), by `scripts/regime_pb_balance_table.py`:

| `Ee` row at cell 2 (cathode) | erg cm⁻³ s⁻¹ | at cell 7 (column) |
|---|---|---|
| `beam_power_deposition` as booked | +8.852e5 | +5.533e5 |
| — of which QL/anomalous | +8.439e5 | +5.071e5 |
| — **remainder, what a passive cell keeps** | **+4.132e4** | **+4.626e4** |
| `heat_conduction` | −4.336e5 | −8.259e5 |
| `plasma_advective_flux` | +1.278e5 | +4.123e4 |
| `electrode_e_sheath_loss` | −6.912e4 | −9.973e5 |
| `ionization_energy_cost` | −5.512e4 | −6.080e4 |
| `electron_neutral_cooling` | −2.153e4 | −2.438e4 |
| `anode_collection` | 0 | −5.450e4 |

The kept remainder is now *smaller* than the local sinks rather than an order
of magnitude above them, which is the whole of the repair. Conduction and the
boundary losses are still the dominant sinks and are still not local — that
statement from the section above is unchanged and is not what was wrong.

Density scan at that same background, `Te` reported at cells 2/7:

| density | as booked (the original scan) | with QL refused |
|---|---|---|
| ×1 (actual, `n ≈ 4.5e9`) | **no root** | **root, `Te` = 23.9 / 24.5 eV, 1 sign change** |
| ×10 | root, 75.5 / 31.6 eV, 1 sign change | root, 7.63 / 7.61 eV, 1 sign change |
| ×100 | root, 10.8 / 8.31 eV, 1 sign change | root, 4.23 / 4.20 eV, 1 sign change |
| ×1000 | root, 4.86 / 0.1 eV, **2 sign changes** | root at the floor, 0 sign changes |
| ×10⁴ | root at the floor | root at the floor |

**OUTCOME: the quasi-static closure SURVIVES.** The balance has a root at the
stance the tracer is meant to run at, and the multi-valuedness three decades
above it is gone as well. End to end, the tracer arm at the production stance
runs the full window (`t_end = 3e-5 s`) instead of raising, so the tracer arms.

Two honest caveats on the comparison, neither of which changes the outcome:

- The original scan's `Te` values at ×10 and ×100 (22.9 eV at cell 2, 49.4 eV
  max) do **not** reproduce against the shipped code at the shipped stance; the
  as-booked column above (75.5 eV at cell 2) is what the current code gives,
  and 49.4 eV is its cell-3 value. The original text attributes the difference
  to the surface-loss row, and at the shipped `characteristic_boundary` stance
  that row's `Ee` is routed to the cathode term and is zero at the cathode
  cell. The ×1 refusal, the ×1000 multi-valuedness and the ×10⁴ floor all
  reproduce exactly, as does every non-conduction row of the channel table to
  four figures.
- The `heat_conduction` row re-measures at −4.336e5 / −8.259e5 against the
  original's −3.633e5 / −8.635e5. It is a second difference of `Te` and so the
  most reproduction-path-sensitive row in the table; every other row agrees to
  four figures.

What this does **not** settle: the tracer's density growth over that window
stays near its seed (`n` max 2.9e9 at `t = 3e-5 s`, no cell activating), so the
overlap gate's registered band is not reached from the tracer side. See Gates.

#### The anomalous closure bracket: the two legs and the middle one

The amendment above is one arm of a declared closure family, not a verdict on
the beam-plasma channel. `beam_anomalous_model` now carries three arms and a
result must state which one produced it:

| arm | what it books | passive-cell policy |
|---|---|---|
| `"none"` | nothing | nothing to book |
| `"quasilinear"` (shipped default) | near-total absorption by FIAT: `dE/dx = E/l_QL` with `l_QL = (n_e/n_b)(v_b/ω_pe)ln(n_e/n_b)`, weak-beam domain only | REFUSED (the amendment above) |
| `"ql_relaxation"` | the RELAXATION physics, coefficients boxed | **BOOKED in full** |

**Why the refusal is model-keyed.** The passive-cell refusal is an answer to a
closure that asserts a total, not to the beam-plasma channel as such. The fiat
arm books ~92% of the deposited power through a wave the cell has no medium to
carry, so on a passive cell the correct answer is zero. `ql_relaxation` reaches
the same question from the other end: it books what a cell of that density can
actually take, through an extracted fraction that goes as `(n_b/2n_e)^(1/3)` and
a gate it must first pass. Refusing that wholesale would delete the physics the
arm exists to supply. So the two legs bracket the truth and the middle leg sits
between them by construction, and `physics.tracer.passive_anomalous_leak`
re-reads the model key itself so the keying is audited, not assumed.

**What the memo settled about onset** (`QL_ONSET_MEMO_2026-08-12.md`). The
LINEAR beam-plasma onset is **always on** in the working range
`n_e = 1e8 – 1e11 cm⁻³`, by a margin of ×400–2500 against He collisional
damping. Onset is therefore **not the gating physics**, and the ASSUMED
QL-onset role that `tracer_activation_ne` was carrying is RESOLVED BELOW RANGE
rather than pinned — see the provenance note. What does gate is
RELAXATION/SATURATION, which is what the middle leg is built on:

* reactive trapping extracts `f_ext = C_trap·min(n_b/2n_e, 1)^(1/3)` of the
  beam energy, `C_trap = 1` [O'Neil, Winfrey & Malmberg, Phys. Fluids 14, 1204
  (1971)];
* the plateau forms over `τ_QL = c·(n_e/n_b)/ω_pe` (Vedenov-era scaling as
  restated in Krall & Trivelpiece §10, order-of-magnitude class), so the
  extracted power is spread over `L_rel = τ_QL·v_b`. `c` is the closure's ONE
  registered bracket constant, `ql_relaxation_coeff`, and every headline is
  quoted at 10, 30 and 100;
* the wave hands its energy to BULK electrons by collisional damping at
  `ν_en/2` [Ginzburg 1970; Alexandrov–Bogdankevich–Rukhadze 1984], which is why
  the deposition is bulk heating in the cell where the waves damp.

The onset inequality is still evaluated, per cell, and still gates the booking —
`0.687·ω_pe·min(n_b/n_e,1)^(1/3) > ν_en/2` **and** `ω_pe > ν_en`, with
`ν_en = nn·K_m(Te)` on the boxed He e-n momentum-transfer table. That keeps
"the gate is open here" a computed property of the run rather than a claim made
once about a range; `smoke_sim1d.py` checks it is open across the working range
AND that it closes (one case per conjunct) with exactly zero booked when it
does. The `min(·,1)` caps that carry the `n_b ≳ n_e` corner are a FLAGGED
INFERENCE, not part of the cited results.

`ql_relaxation` is never offered to the compiled CSDA march: that kernel takes
the anomalous channel as a boolean and applies the fiat drag, so the closure
takes the Python march. The smoke pins that precondition and shows the same
harness DOES reach the kernel for the other two arms.

##### MEASURED: the third balance column

`scripts/regime_pb_balance_table.py --nx 20`, section H — the same stance and
the same instant as the table above, one fluid arm per bracket arm, fed
`P_full` (the middle leg is booked, not refused). Registered before running:
root at the ×1 row → the middle leg is viable; no root → report and stop, no
fallback built.

| `ql_relaxation_coeff` | anomalous share of the row (cells 2/7) | ×1 (actual, `n ≈ 1e9`) | bin |
|---|---|---|---|
| 10 | 95.1% / 94.7% | **no root** | **NO ROOT** |
| 30 (default) | 74.7% / 72.7% | root, `Te` = 18.9 / 14.8 eV, 1 sign change | **ROOT AT STANCE** |
| 100 | 34.8% / 32.8% | root, `Te` = 7.95 / 6.99 eV, 1 sign change | **ROOT AT STANCE** |

**The bin is SPLIT across the registered bracket, and the split IS the result.**
At the short-relaxation endpoint the middle leg concentrates enough power to
reproduce the fiat arm's failure (95% of the row, no root); over the rest of the
bracket it does not. Nothing was moved to remove the split: `c = 10` is a
registered endpoint and the balance's refusal there is a finding about the
closure's short-length limit, not about the balance. The honest claim is
therefore "viable over `c` ≳ 30", with the lower endpoint disclosed.

##### MEASURED: the overlap gate under MATCHED closures

`scripts/regime_r2_overlap_gate.py --nx 20 --t-end 3e-5 --anomalous-model
ql_relaxation --ql-relaxation-coeff 30` — **BLOCKED**, which the gate defines as
not a pass. The tracer arm refuses at cell 2 (deposited beam power
`115237 erg cm⁻³ s⁻¹` against a bracket top of 118.456 eV). Registered reading:
with the closure matched across the passive/active interface the gate stops
measuring the closure gap and starts measuring tracer-vs-fluid NUMERICS — and
under matched closures the tracer cannot produce a number at all at this stance,
so the numerics question is not reached. Reported as-is; nothing tuned.

Two facts travel with it. The fluid arm under `ql_relaxation` reaches
`n` max `4.472e9 cm⁻³` over the window, against `1.325e11` under the fiat arm,
so it never enters the registered band `[1e10, 1e11]` either — a matched-closure
comparison at this window would have had an empty overlap sample regardless of
the refusal. And the refusal is the same *shape* as the one the amendment above
removed, arriving from the other side: the middle leg still concentrates enough
power on the cathode cell for the local balance to want an electron hotter than
the beam heating it.

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
(the composed ceiling `min(cathode_phi_c_cap_V, <circuit member>)`, whose
circuit member is chosen by `cathode_circuit_bound_object`), consumed from the
cathode
diagnostics; the atomic-data cap alone would inflate `I_cond` by the same ~5×
the R1 pass removed from `V_b`.

Criterion (a) is nevertheless an **upper bound**, and the census must be read
that way: putting the whole device drop across the column overstates the axial
field, because most of that drop is the cathode sheath fall. Passing (a) proves
passivity; failing it may only mean the bound is loose. The error is in the
safe direction — the criterion gives cells to the fluid early rather than
holding them in the cheap description too long — and the refinement (the column
drop `V_b − φ_c − φ_a`, or the solver's own `R_p` network) is deliberately not
taken at stage 1: under the capability-limited branch `φ_c → V_b`, so that
difference collapses toward zero and the criterion would fail the other way,
and the sheath partition it rests on is what the R1 follow-up is still moving.
In practice this means the **density gate `tracer_activation_ne` is the binding
condition at low density** and the criteria only start to discriminate above
it, which is what the census shows.

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
drops.

There is **no named RHS row** for that dropped transport, and deliberately so:
a row that is identically zero at every face it is defined on would still be
summed, saved and plotted on every run, for no information. What exists instead
is an **assertion** — invariant I4 of `regime_r2_handoff_check.py`. It evaluates
the Rusanov face fluxes on the tracer's geometry view and requires the `n`,
`Ee` and `Ei` flux at every passive/active face to be exactly zero, then
evaluates the *same state* on the base geometry, where those faces are open,
and requires a **nonzero** particle flux there. The pair is what makes the zero
demonstrably deliberate rather than an artefact of nothing flowing: the term
exists, it is zero because the face is closed, and removing the closure would
change the answer.

The assertion is at the face, not on a cell row, because `_mask_inactive_rhs`
writes literal zeros onto every cell the tracer owns — a cell-row check would
be true whatever the flux did, and would say nothing about whether the ACTIVE
neighbour lost plasma. Invariant I3 of the same script closes the two-part
inventory across the handoff to a stated tolerance.

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

Every tracer run reports **which criterion binds and where** — the
`active_constraint` idiom the timestep limiter already uses. What ships is
exactly two things, and no more:

1. **A printed one-line summary at the end of `run()`**, naming the criterion
   that bound on the most cells, how many cells are still passive, the time and
   cell of the first activation and which criterion caused it, the refresh
   count, and the worst value of the dropped-transport ratio.
2. **One in-memory attribute on the result**, `result.tracer_criterion_census`:
   a mapping carrying the per-cell binding criterion index, the worst ratio,
   the three criterion ratios, the transport ratio, the passive mask, `Te_qs`,
   `γ`, `S`, and the refresh count.

**That attribute is a SINGLE INSTANT, not a time series.** `_tracer_census` is
overwritten on every accepted step and only the last accepted step's value is
attached, so it describes the end of the run and nothing before it. It is also
**not serialized**: `results/io.py` does not write it, so it does not survive a
save/load round-trip and is absent from every HDF5 result. A per-save-frame,
serialized census is a reasonable thing to want and is deliberately not in this
pass — anything needing the history has to read the printed line or keep the
live solver.

Both are presence-gated on the flag: a non-tracer run prints nothing and its
result carries no such attribute at all (asserted in `smoke_sim1d.py`).

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
  the passive-cell QL leak invariant (including the smoothed booking and the
  phase-gated read-back), and an anti-vacuity variant per guard that must fail
  if the guard is removed.
- `regime_pb_balance_table.py` — reproduces the measured balance table above
  under both bookings, plus the QL-onset gap below, plus the third
  (`ql_relaxation`) column at all three registered bracket arms.
- `regime_r2_overlap_gate.py --anomalous-model ql_relaxation` — the same gate
  with BOTH arms on the middle leg, i.e. matched closures. Its registered
  reading and its measured verdict are in "The anomalous closure bracket".

**Measured under the corrected booking** (`nx = 20`, production stance). Both
run-level gates now RUN rather than being blocked by a refusal, and both report
the same thing from different directions: the tracer's plasma grows far more
slowly than the fluid's, because the fluid is being heated by power the tracer
refuses.

- `regime_r2_overlap_gate.py --t-end 3e-5` — **FAIL**, worst relative
  disagreement `0.978` against `rtol = 0.05`, over 198 in-band samples. The
  fluid arm reaches `n` max `1.325e11`; the tracer arm reaches `2.91e9`. The
  gate is not tuned to pass and `tracer_activation_ne` is not moved: the
  failure is the measurement.
- **The gap it measures.** `quasilinear_relaxation_length_cm` returns `inf` —
  no anomalous drag at all — unless `n_b < 0.1 n_e`, so the module's OWN
  weak-beam gate puts QL onset at `n_e = 10 n_b`. At the stance
  (`E_0 = 177.6 eV`, `Γ_0 = 1.647e20 s⁻¹`, `n_b = 2.95e8 cm⁻³`) that is
  `n_QL,onset = 2.95e9 cm⁻³`, against `tracer_activation_ne = 1e10 cm⁻³`:

  | quantity | value |
  |---|---|
  | QL onset, `10 n_b` | 2.95e9 cm⁻³ |
  | `tracer_activation_ne` | 1.0e10 cm⁻³ |
  | **gap** | **×3.39, i.e. 0.53 decades** |

  Over that window quasilinear absorption is live by the code's own criterion
  while the cell is still passive and the refusal is suppressing it. The
  pre-breakdown background sits inside it (`n = 4.5e9`). The two criteria are
  therefore NOT the same event yet — making them one is what the correction
  intends, and `n_act` is the criterion that does not match. Reconciling them
  is a follow-up against a settled criterion, not a number to move here.
- `regime_r2_handoff_check.py --t-handoff 2e-5` — **FAIL at I4 only**, and
  structurally: I1 (config identity), I2 (finiteness) and I3 (the two-part
  ledger, relative change exactly 0) all pass, but no cell reaches
  `tracer_activation_ne` inside that window, so no passive/active interface
  exists for I4 to check. The R2 PASS at the same window was disclosed as
  cadence-conditional; under the corrected booking the leg is simply colder
  and slower, and does not hand off that early.
- Run out to `t = 1e-4` and the tracer leg hands **nine** cells over (2–10,
  at `n = 2.19–4.32e10`) and then raises at `t = 7.476e-5` on **cell 32,
  which it still owns**. An earlier build solved the balance on every cell
  carrying plasma, including the ones the fluid had just taken, and raised at
  `t = 7.0e-5` on cell 2 — an active cell — for exactly the reason this
  amendment removes on passive cells. That was a solve-domain defect and is
  fixed, not documented: the run now passes both the instant and the cell that
  used to stop it.

  The refusal that remains is the OPPOSITE limit from the one at the top of
  this block. Past the beam's IONIZING range `S` collapses to a denormal while
  `P_net` does not, so the **dilution denominator goes to zero** and the
  balance demands an unbounded `Te` — too few beam-born electrons to dilute
  into, rather than too much power.

#### MEASURED: what the residual power on the far cells is

`scripts/regime_pb_pnet_decomposition.py`, at the refusal state
(`t = 7.476e-5 s`, `nx = 20`). Stated as measurement; no remedy is proposed
here.

Off the cathode–anode gap, `P_full` is the smoothed plasma-heating bank and
nothing else — worst relative mismatch **1.12e-15** against
`smoothed(plasma_heating)/Vp` across the grid, because the smoothed radiated
and ionization-cost banks cancel the excitation and cost rows exactly. So the
residual is not a second channel: the anomalous channel is `1e-289` there and
the ohmic booking is 0.49 W confined to cells 2–6.

What it is, is **one cell's terminal dump spread by the kernel**. The CSDA
primary's end-of-range residual is banked whole in the single cell where `E`
crosses `E_stop` — cell 39, `2.2603e10 erg/s`, **9.35% of all banked plasma
heating in one 10 cm cell**. The 50 cm conservative kernel then redistributes
it:

| cell | z [cm] | dz | `Vp` [cm³] | raw bank [erg/s] | raw density | smoothed density | from OTHER cells |
|---|---|---|---|---|---|---|---|
| 13 | 235 | 90 | 63617 | 5.124e7 | 805 | 1383 | 58.5% |
| 22 | 1045 | 90 | 63617 | 3.760e7 | 591 | 591 | 28.5% |
| 31 | 1855 | 90 | 63617 | 2.592e7 | 407 | 21932 | **98.7%** |
| 32 | 1905 | 10 | 7069 | 2.824e6 | 400 | 1.442e5 | **99.98%** |
| 40 | 1985 | 10 | 7069 | **0** | 0 | 3.759e5 | **100%** |

Cells 31, 32 and 40 draw 98.1%, 99.7% and 100% of their smoothed power from
cell 39 alone. Cell 40 banks *nothing* of its own and still carries the largest
power density in the column.

Two mechanisms compound, and both are confirmed by the numbers rather than
inferred. The kernel conserves the **extensive** power (to `1.1e-16`) while the
balance consumes a **density**, and the far cells are short: `dz = 10 cm` and
`Vp = 7069 cm³` against `90 cm` and `63617 cm³` mid-column, a factor 9. So the
same erg/s deposited there is a 9× larger source term. And the kernel's reach
in *cells* is set by the local mesh spacing, not by a fixed stencil, so a 50 cm
width spans many cells wherever the mesh is fine.

The consequence for the tracer is the refusal above: the column's largest
`P_net` density lands exactly where `S` has gone to zero. The consequence for
the deposition model generally is that at this mesh the smoothing kernel, not
the stopping calculation, sets the applied deposition geometry in the
end-of-range region — the raw stopping there is sub-cell.

### Both stage-1 exclusions are now built

The `V_cm` vessel/common-mode node is the next section; the φ_a-aware
`V_b`-object bound is `cathode_circuit_bound_object` (MODEL.md, "The bound's
object"). The tracer still consumes the R1 bound's objects as they are and
still does not touch its contract — what changed is which quantity that bound
holds. One tracer read moved with the node: criterion (b)'s beam energy is the
CHOKED launch energy, because that is the beam the plasma is thin or thick to.
Criterion (a)'s `V_dev` deliberately did **not** move — a common-mode offset
translates the whole cathode/anode system against the wall and cannot change
the anode-to-cathode differential the column conducts under.

## Vessel common-mode node (`regime_vessel_node`, default off)

The LAPD cathode/anode system **floats** with respect to the machine wall. The
whole electrically connected stainless vessel — some 20 m of it — is ONE wall
conductor, and the anode is referenced to it only through four feedthrough
capacitors bridging the ceramic gap insulators. Their **type is visually
unresolved**: axial polypropylene film on the second look, aluminium
electrolytic on the first, with a black band on one side of the cylinder that
does not settle it and mildly favours film (on a film part a plain band marks
the OUTER FOIL — a shielding convention; electrolytics mark polarity with
explicit −/+). So `vessel_leak_resistance_ohm` is finite and ESTIMATED over a
bracket spanning both readings, 2.5e7–1e11 Ω, defaulting to the film reading.
`None` is accepted and means the idealized hard float, an A/B arm rather than
the hardware. A bench measurement resolves the type and the value together.

**THE STRUCTURAL FACT THE MODEL RESTS ON IS TYPE-INSENSITIVE, and that is the
statement to quote.** Over the joint `R_leak × C_total` bracket

```
tau_leak = R_leak · C_total ≈ 10 s … 4e5 s
```

against a ~25 ms discharge — at least ~400× at the most pessimistic corner
(the aged-electrolytic edge) and vastly more at the film edge. So **within a
shot the node is hard-float in kind at both bracket edges, whichever type
these turn out to be**: the leak drains a negligible fraction of the node's
charge and the phase sequence is unchanged by it.
`scripts/regime_vcm_r0b_check.py` sweeps the `R_leak` endpoints alongside the
`C_total` endpoints and reports the in-window sensitivity as a NUMBER — the
worst shift anywhere in the joint bracket is `1.25e-3` relative, exactly the
closed form's `dt/(2·tau_leak)` at that corner, two decades below the
factor-of-ten `C_total` bracket the same result already carries. A shift
reaching the percent level would be a finding. The separation is a claim about
the discharge window only and fails on any question posed over seconds.

**Two documented deviations, neither modelled.** *Polarity, conditional on the
unresolved type*: if the parts are electrolytic they are polarized and conduct
asymmetrically under reverse bias (diode-like above ~1–2 V), and the machine's
plateau common-mode bias is observed at either sign, so the reverse branch is
reachable — the shipped leak is a **symmetric** linear resistor, which is the
deviation; if they are film there is no polarity nuance and the black band is
the outer-foil marking. Either way a nonlinear asymmetric model would buy
nothing on the discharge timescale, where the leak moves nothing in either
direction, and would matter only if `V_cm` scoring came to care about the
negative-plateau branch. *Inter-shot memory*: `tau_leak` far exceeds the ~3 s
shot period under both readings, so the capacitors cannot reset the node
between shots — the physical reset path is the afterglow plasma conductance.
Runs here are single-shot and start from `V_cm = 0`.

### The node

ONE new state variable: `V_cm`, the anode-to-wall (common-mode) potential,
obeying

```
C_total dV_cm/dt = I_e_wall − I_i_wall − V_cm/R_leak
```

with `C_total` the four capacitors' parallel sum. The sign convention is the
physical one and is the load-bearing part: electrons landing on the wall
charge it negative and so **raise** `V_cm`; ions landing on it **lower**
`V_cm`; the steady state of the pair is the floating condition, zero net
system-to-wall current.

Neither current is re-derived here.

- `I_e_wall` is the CSDA rays' **transmitted primary flux** times `e`. The far
  end IS the vessel, so the flux that leaves the domain there is exactly the
  electron current the wall conductor collects. Flux the anode mesh intercepts
  is system-side, and flux that stops in the column is plasma-side; neither is
  booked. This is why the node **refuses** any deposition model but `csda`:
  no other model carries a flux at a terminating surface, so the wall electron
  channel would be identically zero and the node would charge on the ion flux
  alone — a run that reads as though the bootstrap is live when half of it is
  missing.
- `I_i_wall` is the LIVE plasma-terminating boundary term — whichever of the
  characteristic ghost-cell outflow and the volumetric absorption the run
  configured — evaluated on the accepted state and integrated over the
  collector cells' plasma volume. It is the same term the fluid itself
  subtracts, so the node cannot book an ion flux the column did not lose.

### The choke, and the bootstrap it closes

`V_cm` is the potential a transmitted beam electron must **climb** going from
the mesh (at system potential) into the column (referenced to the wall), so
the energy that reaches column physics is

```
E_launch = max(φ_c − max(V_cm, 0), 0)
```

with `φ_c` the R1 **circuit-bounded** sheath drop, never the raw
`cathode_phi_c_cap_V` atomic-data cap — which is why the node requires
`cathode_circuit_voltage_bound`. Only a positive `V_cm` decelerates: a
common-mode offset cannot *accelerate* the beam into the column, and the
floor at zero is the fully choked limit (no beam at all), not an error. The
**flux is untouched** — the same electrons arrive, decelerated — so the
climb moves `E_launch` and nothing else.

One launch energy per ray serves the Beer-Lambert beam arrays, the CSDA
deposition ray, the gap-transmission probe, the tail birth keying, the
reflection threshold and the `sigma_eff` inversion, so the item-35 tripwire
keeps comparing three views of one number. Modelling choice to be aware of:
the step is applied at LAUNCH rather than at the mesh face, which is exact for
the column leg and slightly over-applies the climb over the ~`L_cath` gap.

That closes the loop the node exists to describe. A rising `V_cm` chokes the
beam; a choked beam deposits less and transmits less, so `I_e_wall` falls; the
column's ionization feeds `I_i_wall`, which pulls `V_cm` back down. The
floating constraint therefore **permits beam leakage into the column in
proportion to the ion wall flux**, and column seeding becomes ion-loss
throttled rather than emission throttled.

### The step

Advanced once per **accepted** step, after the circuit, with both wall
currents frozen at their accepted-state values — the same explicit coupling
the loop current and the cathode thermal state already use, so the choke a
step produces reaches the beam at the next solve. A rejected attempt moves
nothing.

The step is the **closed-form** solution rather than an Euler step, because
the leak is linear in `V_cm` and `R_leak·C_total` can be arbitrarily short
against `Δt`:

```
A      = I_e_wall − I_i_wall
hard:    ΔV = A Δt / C_total,                    Q_leak = 0
soft:    ΔV = (A R_leak − V_cm)·(−expm1(−Δt/(R_leak C_total)))
         Q_leak = A Δt − C_total ΔV
```

`Q_leak` is the exact `∫ V_cm/R_leak dt` the same closed form implies (it
follows from integrating the ODE itself), which makes the node's conservation
statement

```
C_total·Σ ΔV  ==  Q_electron − Q_ion − Q_leak
```

close to round-off by construction rather than to a tolerance.
`LAPDSim1D.vessel_charge_residual()` is the auditable form, returning the
absolute residual and its ratio to the total charge MOVED (not to a cancelling
net, which can pass through zero). The smoke suite drives a scripted
seed→engagement→bootstrap→tail scenario at three leak settings and requires
the relative residual below `1e-12`, and asserts the same on a real
trajectory.

### The prediction channel

`V_cm(t)` and its three current channels ride the cathode diagnostics
(`vessel_V_cm_V`, `vessel_beam_climb_V`, `vessel_I_e_wall_A`,
`vessel_I_i_wall_A`, `vessel_I_leak_A`, `vessel_I_wall_net_A`,
`vessel_Q_node_C`, `vessel_charge_residual_C`), present only when the node is
armed. **Nothing here is scored.** The channel exists so that the qualitative
shape observed on the machine — high early positive bias, decaying as the
bootstrap relaxes it, plateauing at either sign at the main discharge — has
something to be compared against later. There are **zero tuned constants**:
the two config values are hardware quantities, and `C_total` is ESTIMATED with
the bracket as the claim.

### Phase sequence (reported, not gated)

`scripts/regime_vcm_r0b_check.py` reports the sequence across the whole
`C_total` bracket, because no single capacitance is a claim: early build
wall-referenced (at mA-scale seed current the charging time is tens to
hundreds of ms, far longer than the cycle, so the float cannot engage);
engagement mid-build in the sub-amp decade; and the bootstrap's sign. The
sequence is stable in kind across the bracket — only the currents at which the
phases occur move, by the 10× span of `C_total` itself. The same script sweeps
the `R_leak` bracket endpoints and the hard float and reports that the
in-window numbers do not move with them, which is the executable form of the
timescale argument above.

### Restart

`V_cm` and its charge ledger ride the payload's `circuit` group, and only when
the node is armed, so a payload from a run without it is structurally what it
always was. `regime_vessel_node` is a **structural flag key**, so a resume
that changes the arming refuses rather than reading half a node.
