# sim1d Numerical Methods

The schemes `LAPDSim1D` uses to discretize [`MODEL.md`](MODEL.md): a
conservative finite-volume plasma on an axial grid, coupled to a
discrete-velocity kinetic neutral gas.

## State and grid

The plasma carries $(n,M,E_e,E_i)$ on finite-volume cells along $z$, with
cell-centred states and face-based fluxes (`core/state.py`,
`core/geometry.py`); $T_e$, $T_i$, $u$ are recovered by `derive_state`. Plasma
and neutral fields carry separate face areas and cell volumes, so inventory
(area $\times$ flux, volume $\times$ density) is tracked consistently on each.
The neutral gas is carried as bin masses on the velocity grid below; the packed
$n_n$ field rides alongside as $\max(\text{moment},\,\text{floor})$, republished
from the distribution each tick rather than independently evolved.

## Spatial discretization

**Rusanov / local Lax–Friedrichs flux** at each interior face
(`physics/flux.py`):

$$F=\tfrac12\left(F_L+F_R\right)-\tfrac12\,a_\text{max}\left(U_R-U_L\right),\qquad a_\text{max}=\max\left(\lvert u_L\rvert+c_L,\ \lvert u_R\rvert+c_R\right)$$

`hyperbolic_wave_speed` selects the sound speed in that pair: `"adiabatic"`
uses $\sqrt{\tfrac53(T_e+T_i)/m_i}$, the exact spectral radius of the
$\gamma=5/3$ two-species system, `"isothermal"` uses $\sqrt{T_e/m_i}$, which
under-bounds it. It sets the dissipation strength and the CFL, not the physical
wave speed, which the pressure flux carries. RHS terms are formed as
$-\Delta(\text{area}\cdot F)/\text{volume}$ per cell, each
$u\,\partial_z$ derivative fused with its compression partner inside one face
flux rather than discretized separately. As in [`MODEL.md`](MODEL.md),
$\partial_z$ is the only spatial derivative and
$\nabla_\parallel\!\cdot F\equiv A^{-1}\partial_z(AF)$.

**`front_flux` selects a second face-flux operator** beside the Rusanov one: a
sonic-relaxation front flux that fills unfilled cells, capped by `alpha_front`
and carrying donor-cell energy. It is the operator the `front_density` timestep
bound below describes.

**Walls.** Every face bounding the plasma carries no particle or
thermal-energy flux and keeps the live cell's pressure as its momentum flux
(zero where there is no live cell at all). At the plasma-terminating subset the
advective flux carries nothing: the ghost-cell Bohm flux
(`sources.characteristic_boundary_rhs`) supplies particle, momentum and energy
flux with its own pressure term, using the same face kernel as the interior
(`flux.kep_rusanov_face_scalar`) between the interior cell and the ghost state,
applied as a one-sided divergence $\pm\,\text{area}\cdot F/V$ on the live cell.

**Energy-consistent hyperbolic core** (`hyperbolic_energy_consistent`). The
convective momentum flux becomes the kinetic-energy-preserving $\{u\}\{M\}$
form, the pressure work a kinetic-energy-preserving
$-u\,\partial_zp_s$ per species folded into the `pressure_work`
term, and the Rusanov $(n,M)$ numerical kinetic-energy dissipation is deposited
into $E_i$ as `hyperbolic_dissipation_heating` — a flux divergence contracted
with the local velocity, so non-negative only in the volume-weighted total. The
operator is CONSTRUCTED so that the semi-discrete flux and pressure-work pair
conserves $\sum V(K+E_e+E_i)$ on a closed domain, the explicit integration
leaving a time-integration drift of the nonlinear kinetic energy; the size of
that drift is a property of the step, not of this operator, and the suite that
exercises the claim is `scripts/verify/verify_sim1d_r2_hyperbolic.py`.

**Geometry source and conduction.** With a varying area the momentum ledger
carries

$$\left.\frac{\partial M_i}{\partial t}\right|_\text{geom}=\frac{p\,A_{i+1/2}-p\,A_{i-1/2}}{V_{\text{col},i}},\qquad p=p_e+p_i$$

written with the same multiply-then-subtract ordering as the area-weighted
pressure flux it pairs with, so for a stationary uniform-pressure plasma the two
cancel bit for bit — the well-balanced property, excepting the
plasma-terminating cells where the ghost-cell outflow supplies the face
momentum. Conductive face fluxes $q=-\kappa_\text{face}\,\partial_zT$ are differenced
to a conservative flux divergence (`physics/conduction.py`) with
$\kappa_\text{face}$ the arithmetic mean of the two cells, each face scaled by
its transmission factor (zero at a plasma wall, $1-\eta$ across the anode mesh,
one otherwise).

## Time integration

### Explicit stage

A two-stage strong-stability-preserving Runge–Kutta step (SSPRK2 / Heun),
`core/integrator.py`:

$$y^{(1)}=\Pi\!\left[y^n+\Delta t\,L(t^n,y^n)\right],\qquad y^{n+1}=\Pi\!\left[\tfrac12y^n+\tfrac12\left(y^{(1)}+\Delta t\,L(t^n+\Delta t,y^{(1)})\right)\right]$$

$\Pi$ the floor projection below, applied at each stage. The stages are
evaluated at $t^n$ and $t^n+\Delta t$, preserving second-order accuracy for
explicitly time-dependent forcing; omitting the `time` argument freezes the
forcing at the step start and is first-order in it.

### The neutral-only step

**The plasma is not always integrated by the stage above.** While the plasma is
off, and through the pre-breakdown phase, the step is a backward-Euler solve of
the neutral density alone:

$$\left(\mathbb I+\Delta t\,K_n\right)n_n^{\,\text{next}}=n_n+\Delta t\,S_n$$

$K_n$ the tridiagonal operator assembled from the face exchange conductances
(each face writing $(i,i)$, $(i,i{+}1)$, $(i{+}1,i{+}1)$ and $(i{+}1,i)$) plus
the pump on the diagonal, and $S_n$ the fueling source. It is held in LAPACK
banded storage with bandwidths $(1,1)$, so assembly and solve are $O(N)$; a
two-zone variant solves the column/annulus pair together. This is the update
formula the pre-breakdown phase runs, and the density floor is applied to its
result unless the caller asks otherwise.

### Operator split

`implicit_heat_conduction` composes an explicit SSPRK2 step over all non-heat
terms (operator $A$) with an implicit heat substep (operator $B$), removing the
stiff parabolic stability limit from the explicit step. `operator_splitting`
selects the composition: `"lie"` applies $A(\Delta t)$ then $B(\Delta t)$ and is
$O(\Delta t)$, the splitting error going as $\Delta t\,[A,B]$; `"strang"`
applies $B(\Delta t/2)\to A(\Delta t)\to B(\Delta t/2)$, whose symmetry cancels
that leading commutator and leaves $O(\Delta t^2)$.
`beam_deposition_in_heat_substep` moves the beam's electron-energy term from
$A$'s explicit sum into $B$, applied as a source held constant over each substep
on the same tridiagonal operator; the beam's particle births, ionization cost
and excitation radiation stay in $A$.

### Implicit heat substep

Solved per species as a tridiagonal system via `scipy.linalg.solve_banded`.
Three of the four schemes are theta methods,

$$\left(C+\theta\,\Delta t\,K\right)T^{n+1}=C\,T^n-(1-\theta)\,\Delta t\,K\,T^n$$

$C$ the heat capacity and $K$ the conduction operator built from the same face
coefficients as the explicit half.

| `implicit_heat_scheme` | $\theta$ | $R(-\infty)$ | L-stable | banded solves | substep order |
|---|---|---|---|---|---|
| `backward_euler` | 1 | 0 | yes | 1 | 1 |
| `shifted` | 0.6 | $-2/3$ | no | 1 | 1 |
| `crank_nicolson` | 0.5 | $-1$ | no | 1 | 2 |
| `tr_bdf2` | — | 0 | yes | 2 | 2 |

At $\theta=1$, $C+\Delta t\,K$ is an M-matrix satisfying
$(C+\Delta t\,K)\mathbf 1=C\mathbf 1$ (as $K\mathbf 1=0$), giving the discrete
maximum principle $T^{n+1}\ge\min(T^n)$:
backward Euler is unconditionally monotone and cannot undershoot the temperature
floors. For $\theta<1$ the amplification factor tends to $-(1-\theta)/\theta$ as
$\Delta t\,\lambda\to-\infty$, so stiff modes ring — undamped at
$\theta=\tfrac12$ — and can be clipped by a floor, which injects energy.
`tr_bdf2` is second-order *and* L-stable: a trapezoidal stage out to
$t^n+\gamma\Delta t$ then a BDF2 stage through $(T^n,T_\gamma,T^{n+1})$, with
$\gamma=2-\sqrt2$ making both stages share the implicit coefficient
$\gamma/2=(1-\gamma)/(2-\gamma)$ and hence one banded operator — two
`solve_banded` calls against a single matrix. It damps undershoot rather than
preventing it. A third floor clip sits inside the substep itself, on its
returned temperatures.

**Picard iterations on $\kappa$.** The conductivity is frozen at the incoming
state, and that — not the scheme — caps the substep at first order.
`heat_picard_iterations` re-evaluates $\kappa$ at the scheme's own flux point,
costing one more SUBSTEP solve per species per iteration: one banded solve for a
theta method, TWO for `tr_bdf2`. The iteration stops early once
$\max\lvert\Delta T\rvert\le$ `heat_picard_tol` $\cdot\max\lvert T\rvert$ on
both species, and the temperature at which $\kappa$ is evaluated is clamped at
$10^4$ eV as an overflow guard. **The flux limiter follows the same iterate:**
in the explicit path it is applied at the incoming $T_e$, but in the implicit
substep it is re-evaluated at the blended temperature each Picard pass uses —
the incoming $T_e$ only when `heat_picard_iterations = 0`.

Second order in the whole split step requires **all three** of a second-order
`implicit_heat_scheme`, a positive `heat_picard_iterations`, and
`operator_splitting = "strang"`: the frozen conductivity and the Lie splitting
are independent first-order error terms, so each caps the step on its own.
`scripts/gates/verify_sim1d_order.py` is the harness that measures the observed
order by fixed-$\Delta t$ Richardson refinement, in a regime with floors inert
and watched, a single phase, an autonomous RHS and no cathode. A discharge does
not show that order: floors bind and phase transitions are threshold-triggered,
so the step degrades to first order wherever those engage.

## Adaptive timestep control

`core/timestep.py:suggest_timestep` takes $\Delta t$ as the minimum over
candidate bounds, then clamps to $[\Delta t_\text{min},\Delta t_\text{max}]$;
`dt_global_scale` multiplies the result AFTER that clamp and is an instrument,
not a bound. Three inequality forms recur:

$$\text{distance: }\Delta t\le\varepsilon\min\frac{d}{s},\qquad \text{fractional: }\Delta t\le\varepsilon\min\frac{X-X_\text{floor}}{|\dot X|},\qquad \text{rate: }\Delta t\le\frac{\varepsilon}{\max\nu}$$

The fractional form skips any cell with a zero rate or a non-positive margin.
**The plasma bounds run over the plasma-active cells; the neutral bounds
(`neutral_exchange`, `neutral_sources`, `neutral_wind`, `neutral_energy`) run
over every cell.**

| candidate | inequality |
|---|---|
| `plasma_cfl` | distance, $d$ the centre distance and $s=\tfrac12(\lvert u_L\rvert+\lvert u_R\rvert+c_L+c_R)$ per face, $\varepsilon$ = `cfl`; a face counts only where both cells are active and the face is open |
| `front_density` | fractional on $n$ against the front-filling flux term, $\varepsilon$ = `density_dt_fraction` |
| `reactions` | fractional on $n$ (floor $n_\text{floor}$) AND on $n_n$ (floor 0) against the bulk reaction term |
| `surface_loss` | negative-margin — $\Delta t\le\varepsilon\min(\text{margin}/\lvert\dot X\rvert)$ over DRAINING cells only ($\varepsilon$ = `density_dt_fraction`), margins $n-n_\text{floor}$ and the exact conservative $E_s-\tfrac32nT_{s,\text{floor}}$ whose rates include the change in floor energy when $n$ changes, $d(E-\tfrac32nT_\text{floor})/dt=\dot E-\tfrac32T_\text{floor}\dot n$; a non-positive margin returns 0. Bundles the cathode/sheath, anode-collection and plasma-terminating boundary terms plus an engaged kinetic arm's coupling term, and is assembled only under `raw_stage_validation` or an engaged kinetic arm |
| `energy_exchange` | fractional on $E_e$, $E_i$ against $Q_{ie}$ (floor 0) |
| `electron_cooling` | fractional on $E_e$ against the inelastic and radiative terms |
| `ion_charge_exchange` | fractional on $E_i$ against the charge-exchange term |
| `ion_neutral_drag` | rate, $\Delta t\max\nu_{in}\le$ `DRAG_DT_FRACTION`, $\nu$ scaled by $\lvert b_\text{ion\_neutral\_drag}\rvert$ |
| `heat_conduction` | explicit parabolic bound $\displaystyle\Delta t\le\varepsilon\min_i\frac{V_iC_i}{\sum_{\text{faces of }i}A_f\kappa_fh_f/d_f}$ at `conduction.HEAT_DT_FRACTION`, $h_f$ the face transmission — on a uniform grid $\varepsilon\Delta z^2C/(2\kappa)$, the 2 being the two faces; withdrawn on the implicit path |
| `neutral_exchange` | fractional on $n_n$ against the pair-exchange term, $\varepsilon$ = `neutral_dt_fraction` |
| `neutral_sources` | fractional on $n_n$ against the fueling and pumping terms |
| `neutral_wind` | distance, $\Delta t\le\varepsilon\min(\Delta z/\lvert u_n\rvert)$ at $\varepsilon$ = `cfl`, folding in the annulus drift where the state carries one |
| `neutral_energy` | rate — $\Delta t$ times the summed neutral-energy relaxation rates below `neutral_dt_fraction`, folding in the neutral signal speed $(\lvert u_n\rvert+c_n)/\Delta z$ |
| `circuit` | $\Delta t\le\varepsilon\,\tau_\text{circuit}$, $\tau_\text{circuit}=L/(xR_\text{comp}+dV_\text{dis}/dI)$ — the same external series share the loop advance integrates against — with the device slope read by a one-sided finite difference of that same evaluator. Withdrawn at local equilibrium ($\lvert f(I)\rvert\,\tau_\text{circuit}<dI_\text{probe}$), for $L\le0$ or a non-positive slope, and unless `cathode_circuit_voltage_bound` is armed. An accuracy bound, the loop advance being L-stable |
| `dt_max` | the configured ceiling |

**Floor-aware drain exemption** (under `surface_loss_floor_exempt`). A cell
whose energy margin above its floor energy is within
`SURFACE_LOSS_FLOOR_EXEMPT_RTOL` of it is dropped from the `surface_loss` bound
alone, re-admitted at the wider `surface_loss_floor_exempt_exit_rtol` so a
hovering cell holds its previous verdict inside the band; an exit threshold at
or below the entry one, or at or above 1, is refused. The accept-time clip
resets a floor-pinned cell's margin to float residue every step, so without the
exemption a persistent drain re-trips the bound forever and pins $\Delta t$ at
$\Delta t_\text{min}$ while the floor holds the cell. The density channel is
never exempted, every other bound still governs an exempted cell, and an
exempted cell is never the reported active constraint.

**A bound must describe something the step applies.** The kinetic neutral arm
zeroes whole contributions of the fluid terms and carries them in its own
coupling term, so
while engaged the `ion_charge_exchange`, `ion_neutral_drag`, `neutral_exchange`
and `neutral_sources` candidates are withdrawn to infinity and the reaction
bound keeps only its plasma channel, their replacements bounded through the
`surface_loss` bundle.

**Growth ramp and clamp.** Between accepted steps $\Delta t$ may grow by at most
`dt_growth_factor`, applied after `suggest_timestep` as one more cap, so it can
only shrink the step; after `dt_growth_recovery_patience` CONSECUTIVE
ramp-capped accepted steps the factor becomes `dt_growth_recovery_factor`, and
one step capped by anything else — a physics bound, an output cadence, or a
retry — resets the streak. The active constraint names the bound that actually
minimized; when it asked for less than $\Delta t_\text{min}$,
`clamped_to_dt_min` is set and `dt_raw` keeps the unclamped request,
$\texttt{dt\_raw}=0$ being the drained floor-pinned signature.
`dt_min_lock_max_steps` bounds CONSECUTIVE clamped ADAPTIVE steps and raises
past it; its counter is driven by

```
clamped_to_dt_min  OR  (accepted_dt <= dt_min(1 + 1e-9)  AND  dt_raw > dt_min(1 + 1e-9))
```

evaluated after acceptance, the relative slack keeping a step that merely lands
on the floor from counting, and the run loop's own caps and the retry ladder
being able to push an accepted step under the floor while no candidate asked for
less.

## Step acceptance and rejection

A candidate step is attempted without committing state (`_attempt_step`) and
then validated. **The retry ladder** halves the step on rejection
(`DT_REJECT_FACTOR` $=0.5$), retries at most `max_step_retries` times under
`adaptive_retries_enabled`, and raises `TimestepRejectionError` once the next
$\Delta t$ would fall below $\Delta t_\text{min}$. A candidate is rejected on a
non-finite value, a negative density or energy, or a fractional change past
`max_density_step_fraction`, `max_neutral_step_fraction` or
`max_energy_step_fraction` where those are set; rejection events and constraint
histories are stored for post-run diagnostics.

`raw_stage_validation` additionally inspects both SSPRK candidates, the implicit
heat candidate and the neutral-only candidate *before* floors are applied,
covering $n$, $n_n$, $n_{n,a}$, $E_e$ and $E_i$, with non-finiteness scanned
over the whole packed vector; a failed candidate carries its raw rejection
evidence but cannot mutate accepted state, the circuit or surface caches,
kinetic targets, time, or the cumulative floor ledger.

A run that fails to ignite is a phase transition, not a step rejection. **Four
routes open the cathode switch** and send the run through the ordinary afterglow
to a finite end time: the **stall detector** — $\gamma_N\le0$ AND
$d(E_{e,\text{total}})/dt\le0$ for every saved sample across a sustained window,
a structural joint condition with no tuned rates, evaluated over a fixed window
and rate sub-window with a minimum sample count; the **pre-breakdown timeout**,
the machine's own hardware guard, whose `prebreakdown_timeout_action` selects
between opening the switch and raising; and the **wall-clock and accepted-step
budgets**, `ignition_wall_clock_cap_s` and `ignition_accepted_step_cap`, which
route to the same wind-down so a tripped run leaves the same kind of artifact.

## Floors

Enforced by `apply_state_floors` / `floor_state_vector` at construction and at
the end of every stage, with `derive_state` additionally applying the
temperature floors on every read and the implicit heat substep clipping its own
result. Densities are clipped directly on the packed fields; the temperatures are
clipped on the DERIVED quantities, and the packed $E_e$, $E_i$ change only where
`conservative_from_primitives` rebuilds them from the floored primitives. Order
within a call: $n$, then $n_n$, then the temperatures, then the rebuild, then
the optional fields, the neutral energy floor taken last against the
already-floored $n_n$. Momenta are not clipped — $u$ is recovered with the
floored density and $M$ rebuilt from it, which leaves $M$ unchanged to roundoff
and bit-identical on every state probed, though $(m n)(M/(m n))$ carries no IEEE
guarantee of exactness. Each accepted repair books its exact extensive debit in
`floor_ledger`; `scripts/gates/audit_sim1d_floor_activation.py` instruments the
clip sites at run time, which cannot be done post-hoc.

## The cathode solve

The sheath root is bracketed on a monotone residual and closed by `brentq`. The
voltage-driven form roots the coupled residual in $\psi_+$ at `xtol` $10^{-8}$
and `rtol` $10^{-6}$, and its bracket search runs three stages IN THIS ORDER:
where a previous root is available, two WARM WINDOWS about it first
($\times[0.5,2]$, then $\times[0.125,8]$, each allowed two doublings); failing
those, the full range, allowed fifteen; and only on a convergence failure there,
an unconstrained re-bracket — the one path that can return a root the ceiling
does not bound. The current-driven form roots $J_\text{tot}(\psi_+)=J_\text{imposed}$ at `xtol`
$10^{-12}$ and `rtol` $10^{-14}$, doubling $\psi_\text{top}$ up to two hundred
times and carrying a plateau tolerance of $64\epsilon$. The prescribed-drive
form roots $\phi_c+V_p-\phi_a(\phi_c)-V_b$ on
$[10^{-8}\ \mathrm{V},\ \phi_{c,\text{cap}}]$ at the same tight tolerances; a
residual already non-negative at the bottom returns $10^{-8}$ V UNTAGGED.
Uniqueness rests on each residual's monotonicity.

**A demand past the composed ceiling is CLAMPED, not raised — in the
current-driven and prescribed forms.** There the solve returns the ceiling value
and TAGS itself `capability_limited`, `bound_active` recording which member of
the ceiling bound; no error is raised and the run continues. The voltage-driven
form does not share that contract: on a convergence failure it re-brackets
UNCONSTRAINED, so a root it returns is not bounded by the ceiling.

The loop current is advanced by an L-stable TR-BDF2 stage split over
$L\,dI/dt=V_\text{src}-I\,xR_\text{comp}-V_\text{dis}(I)$ — the external share
of the compliance resistance only, the rest being inside $V_\text{dis}$ — with
$V_\text{dis}$ evaluated at the frozen plasma state as a function of trial
current and a modelled bank capacitor discharging trapezoidally alongside. Each
stage is itself a `brentq` root at `xtol` $10^{-10}$ and `rtol` $10^{-12}$,
bracketed by up to two hundred doublings, with the current clamped at $I\ge0$.
The fluid stages run at a loop current frozen over the step;
`coupled_circuit_picard` re-runs the accepted step, at most
`circuit_picard_max_iter` times in a driven phase, until the current a step
produces matches the one it ran at to `circuit_picard_tol_rel` relative to
$\max(\lvert I\rvert,1\ \mathrm{A})$, a snapshot/restore pair restoring every
step-mutated attribute exactly so a rejected iteration leaves no trace.

**The beam march.** The CSDA ray is integrated over adaptive substeps
$dz_\text{sub}=\min(\text{remaining},\,f_\text{sub}E/L_\text{tot})$,
$f_\text{sub}$ = `max_energy_fraction_per_substep`, so each substep resolves a
fixed fraction of the primary's remaining energy; the substep is additionally
clamped so the ray lands exactly on the stopping energy
$E_\text{stop}=20.6158$ eV rather than stepping past it, and the sub-threshold
residual is banked there. The flux is carried unattenuated, so the same
floating-point products feed the energy decrement and every per-channel bank and
the per-ray power identity closes to accumulated roundoff by construction.

$l_b$ itself is CLOSED FORM, the harmonic sum of a Coulomb range and a neutral
range,

$$\frac{1}{l_b}=\frac{1}{l_{bi}}+\frac{1}{l_{bn}},\qquad l_{bi}=v_b\,\tau_{ei}(T_e,n_e),\qquad l_{bn}=\frac{1}{\sigma_b n_n},$$

$v_b=\sqrt{2e\phi_c/m_e}$ the launch speed, reducing to $l_{bi}$ where there is
no neutral term and returning zero for $\phi_c\le0$. **Where it is evaluated
differs by form.** The voltage-driven solve holds the bypass fraction FROZEN as
a parameter inside its residual and recomputes $l_b$ BETWEEN `brentq` solves,
cycling the $(\psi_+,\,l_b,\,\beta_\text{bypass})$ triple to $10^{-4}$ in the
bypass fraction over at most four passes. The current-driven and prescribed
forms evaluate $l_b$ INSIDE the residual instead, so their single root already
carries it and no outer cycle runs. The one BISECTED
quantity in the deposition module is the plateau-edge energy $E_1$: a fixed
bisection budget on a monotone residual between $E_\text{stop}$ and the beam
energy, exiting early when the midpoint reaches a bracket endpoint, and clamped
to the floor and COUNTED when the edge the equation asks for falls inside the
bulk.

## The kinetic neutral solver

**Velocity grid** (`physics/kinetic_neutrals.py:VGrid`). Both axes are
sinh-stretched about one fine scale $v_\text{fine}=\tfrac14\sqrt{kT_\text{wall}/m}$
out to a common half-extent, $v_\parallel$ spanning
$(-v_\text{max},+v_\text{max})$ and the perpendicular SPEED axis
$(0,v_\text{max})$:

$$v_k=v_\text{fine}\sinh\!\left(a\,u_k\right),\qquad a=\operatorname{arcsinh}\!\left(v_\text{max}/v_\text{fine}\right)$$

$u_k$ the half-offset normalized index, so resolution is $\sim v_\text{fine}$
near zero (the wall gas) and coarsens toward $v_\text{max}$ (the
charge-exchange tail). No bin sits at exactly zero — centres are at
half-offsets — so the upwind march never divides by zero and every bin
transports; an even axial bin count is refused.

**Extent.** `neutral_kinetic_dvm_vmax_cm_s` pins the half-extent. Unset, it is
sized to the launch band the armed surface jets can produce,
$1.25\sqrt{2\varepsilon_\text{max}/m}$ with $\varepsilon_\text{max}$ the largest
per-atom launch energy over those jets, the factor a sizing margin whose
sufficiency is checked rather than assumed; with no jet armed it falls back to a
thermal/sonic sizing built from an ion-temperature cap and a drift cap. The
whole band is checked at CONSTRUCTION, sampled energies spanning each armed
jet's band going through the same launch-spectrum builder the tick uses. A drift
past the LAST BIN CENTRE cannot be represented — no non-negative weighting of
bins has a mean beyond it — and a finer axial resolution LOWERS that ceiling,
the grid-tied smear being the local bin width. **The refusal is the launch
path's, not the projection's:** `VGrid.maxwellian` itself silently CLAMPS an
off-grid drift to the boundary half-space for every other caller (the ion
Maxwellian the charge-exchange and elastic births are placed on, among them);
what turns that clamp into a raise on the launch path is the moment check below.

**Moment-preserving projection** (`VGrid.maxwellian`). A drifting Maxwellian is
placed as analytic per-bin masses — erf differences along $v_\parallel$,
Rayleigh CDF differences along $c_\perp$ (the 2D perpendicular speed measure) —
so the density is exact by construction and the masses sum to 1. A two-basis
compensation,

$$f'=f+a\left(v_\parallel-\langle v_\parallel\rangle\right)f+b\left(V^2-\langle V^2\rangle\right)f$$

is then solved so the DISCRETE first and second moments, evaluated at bin centres
the way the transport uses them, hit their analytic targets
$\langle v_\parallel\rangle=u$ and $\langle V^2\rangle=u^2+3s^2$. At most four
passes run, stopping once both moments are within $10^{-10}$ relative, with a
positivity clip-and-retry at $-10^{-12}\max f$. Both basis functions are
moment-free about the current state analytically, so the density is preserved;
numerically that cancellation leaves a residue, which two guards hold. The
$2\times2$ is NON-DIMENSIONALIZED before it is judged or solved — one of its two
equations is a $v_\parallel$ moment and the other a $V^2$ moment, while the two
unknown amplitudes carry the reciprocal dimensions, so the raw condition number
is dimensional and says nothing about the solve. A scaled condition number strictly ABOVE $1/\epsilon$ is
rejected, and the sum is restored at the end if it has drifted past
$10^{-12}$; both guards are inert while the invariant holds.

**Launch spectra.** A monoenergetic surface jet at energy per atom
$\varepsilon$, equivalently a speed $v_\text{back}=\sqrt{2\varepsilon/m}$, is
placed through that projection with the drift SOLVED FROM THE ENERGY,

$$u=\pm\sqrt{v_\text{back}^2-3kT_\text{launch}/m}$$

which makes $\langle V^2\rangle=v_\text{back}^2$ and the discrete mean energy
exactly $\varepsilon$; a spectrum drifting at $v_\text{back}$ would carry
$\varepsilon+\tfrac32kT_\text{launch}$ per atom, the smear's own thermal content
on top of the beam, which the surface book was not debited for. The SIGN follows
the launch side: $v_\parallel$ is a signed coordinate, so a launch into the
half-space $v_\parallel<0$ is placed at drift $u<0$ and the projection carries a
negative drift exactly as it carries a positive one. $T_\text{launch}$ is grid-tied,
$m\,\Delta v_\parallel(v_\text{back})^2/k_B$ — the narrowest spectrum the grid
resolves there, narrower leaving the compensation nothing to redistribute —
unless a named launch temperature overrides it. The accepted spectrum's density,
drift and mean energy are compared against their targets at a relative bar and a
miss RAISES; the energy ledger books the birth as the count times the moment of
the array actually placed, closing by construction rather than by the
projection's accuracy. A launch energy at or below $\tfrac32kT_\text{launch}$
has no drift and raises.

**Energy-matched wall return.** Under
`neutral_kinetic_dvm_wall_reflection = "diffuse_elastic"` the non-accommodated
share is re-emitted on the cosine shape at a temperature that must be SOLVED, so
the spectrum carries the retained share's own incident mean energy per atom
$\bar e=E_\text{incident}/N_\text{incident}$; the `"specular"` alternative
instead returns the incident array scaled by $1-\alpha_\text{acc}$ and solves
nothing. The discrete mean energy

$$E(s)=\sum_{jk}f_{jk}(s)\,\tfrac12m\left(v_{\parallel,j}^2+c_{\perp,k}^2\right),\qquad s=\sqrt{kT/m}$$

rises monotonically with $s$, so the inverse is a one-parameter root find in two
tiers. First a **secant iteration in $\ln s$** on
$F(\ln s)=\ln E(s)-\ln\bar e$, evaluated on the SEPARABLE contraction of that sum
so no per-cell $(\text{cells},n_{v_\parallel},n_{c_\perp})$ array is built per
evaluation: in logs the continuum relation $E=2ms^2$ is an exactly linear
residual of slope 2, so the seed $s_0=\sqrt{\bar e/2m}$ plus one Newton step at
that slope lands close and the secant closes the rest, its step clipped to
$\pm1$ in $\ln s$. Every cell steps on every sweep and the stopping test is a
single whole-call reduction on $\max\lvert F\rvert$. The secant hands the WHOLE
call to the second tier when it misses its bar, when a residual or step is not
finite, or when its denominator vanishes on a flat residual — what a target
above the grid's saturation energy produces. The second tier is a **bracketed
bisection**: seeded at the same continuum relation, bracketed outward by halving
and doubling, then bisected until the bracket reaches the floating-point
resolution of its endpoints. $E(s)$ SATURATES once the spectrum outruns the
outermost bins, so a target above that has no solution and both the bracket
search and the final agreement check raise rather than returning a spectrum at
an energy the caller did not ask for.

**The transfer relaxation.** The plasma-side momentum/energy transfer is booked
once per neutral clock tick (`neutral_kinetic_dvm_cadence_s`) while the plasma
steps many times inside it. The charge-exchange/elastic pair is a relaxation,
$dE_i/dt=-\nu(E_i-E_i^\text{eq})$ and $dM/dt=-\nu(M-M^\text{eq})$, with one
$\nu=N_\text{loss}/(V\Delta t\,n_i)$ per cell.
`neutral_kinetic_dvm_transfer_hold` selects how the plasma applies it between
ticks: `"zoh"` freezes the booked rate, which over a TICK advances

$$X_{k+1}=X_k-\nu\,\Delta t_\text{tick}\left(X_k-X_\text{eq}\right)$$

— multiplying the distance to the target by $1-\nu\Delta t_\text{tick}$ each
tick, an amplification with a sign flip once $\nu\Delta t_\text{tick}>2$ (within
a tick it is simply a constant rate). `"exponential"` applies, per cell and per
plasma step, at the tick's frozen rate and target,

$$E_i\leftarrow E_i^\text{eq}+\left(E_i-E_i^\text{eq}\right)e^{-\nu\,dt}$$

and the momentum term at the same $\nu$. Applied as a constant rate over the step,
so the SSPRK2 stages integrate it exactly: unconditionally stable, exact for the
linearized system, unable to carry either field past its target at any
$\Delta t$, and
reducing to the zero-order hold to $O(\nu\,dt)$.

Two ledgers separate two different shortfalls. **Floor debt** comes from a cap:
the applied drain is limited to
`relax_fraction` $\cdot(E_i-E_i^\text{floor})/dt$, with the same factor on the
momentum term, and whatever the plasma could not absorb is withheld as `debt`.
**Hold debt** is the difference between what the plasma applied and what the
tick booked — the tick froze a rate at a state that then moved. It is
first-order in $\nu\Delta t$ and vanishes as the neutral clock refines, making it
the cadence meter, and the ledger identity
`applied_cum + debt + hold_debt == booked_cum` holds per cell at every accepted
step. Repayment goes THROUGH the relaxation as
$D\,\varphi(\nu\,dt)/\Delta t_\text{tick}$ with $\varphi(x)=(1-e^{-x})/x$,
delivering exactly $D/\Delta t_\text{tick}$ in the resolved limit and damping it
when the tick is coarse; a flat $D/\Delta t_\text{tick}$ would re-inject the
zero-order increment the hold removed. The per-tick map is then

$$\begin{pmatrix}g\\D\end{pmatrix}_{k+1}=\begin{pmatrix}e^{-X}&a\\-(X-1+e^{-X})&1-a\end{pmatrix}\begin{pmatrix}g\\D\end{pmatrix}_k,\qquad a=\frac{1-e^{-X}}{X},\quad X=\nu\,\Delta t_\text{tick}$$

with determinant $1-a$ and trace $e^{-X}+1-a$, so both eigenvalues lie strictly
inside the unit circle for every $X>0$ independently of how the tick is
subdivided, and the only fixed point is $g=D=0$: the debt is driven to zero, not
merely bounded. Stability is unconditional; accuracy is not — a tick spanning
many e-folds cannot be integrated accurately by any scheme frozen at its start,
and what the hold guarantees there is a bounded, self-retiring debt saying the
cadence is too coarse.

**The counted ionization debit** is taken LAST AMONG THE LOSS RECONCILIATIONS —
after the charge-exchange and elastic re-births, against
$f_c^\text{marched}+\text{birth}_{cx}+\text{birth}_{el}$, the inventory the cell
carries once those returns are in — so the drop is drawn from an array that
actually exists and positivity is structural rather than checked. It is NOT the
last operation of the tick: the anode-mesh re-emission, the recombination births
and the surface-jet births are all added to $f_c$ AFTER it, and so sit outside
the inventory the debit draws from. (Fueling is not among them — the puff is
born in the ANNULUS, into $f_a$, and never enters the column array the debit
reads.) The per-cell
handshake `ion_removed_cum + ion_debt == ion_booked_cum` and the ledger's
`inventory_after − inventory_before == births − losses` are exact at every tick.

## Compiled kernels

The Cython module transcribes **the cathode scalar kernels, the current-driven
sheath root find, the beam coupling length, the CSDA beam march and the fused
lerp**. It is a **faithful transcription, not a reimplementation**, and is
bit-exact against the pure path by CONSTRUCTION of the source: both write the
interpolation fusion out explicitly — `math.fma` in `cablp/numerics/interp.py`,
C's `fma()` in the kernel module — so neither depends on whether the compiler
that built numpy emitted a fused multiply-add. `numpy.interp` computes its
interior lerp as `slope*(x - xp[j]) + fp[j]`, and whether that rounds once or
twice is fixed by that compiler rather than by numpy's source; the two forms
disagree by 1 ULP on a few parts in ten thousand of queries, enough to separate
two implementations that are supposed to be bit-identical. `-ffp-contract=off`
is used everywhere else.

Selection is by the environment variable `CABLP_COMPILED_KERNELS`, read once at
module import so the hot path is a plain function object with no per-call branch,
and therefore resolved before any solver is constructed. Three states, none
silent: unset or a falsy value runs pure Python; a truthy value runs compiled and
raises at import if the extension is not importable, so requesting it cannot
quietly yield the pure path; anything else raises. The kernel provenance is
recorded in every artifact's metadata, and a gate depending on the compiled path
having loaded should probe `KERNEL_ID`.

## Restart

A solver can write its complete evolving state to a restart payload and construct
a later solver whose initial condition is that state (`results/restart.py`:
`save_restart_state`, and the `restart_from` parameter). The payload is a
self-describing HDF5 file with its own format string, `sim1d-restart-v1`,
independent of the trajectory format; it carries one instant, not a history.

**The kinetic neutral closure cannot be restarted, and says so.** The payload
serialises fluid fields, not a distribution function, so combining `restart_from`
with `neutral_model` in `{"kinetic", "kinetic_dvm"}` RAISES at construction
rather than resuming: reseeding the kinetic half from a Maxwellian would not be
a continuation. The model [`MODEL.md`](MODEL.md) presents uses that closure, so
what follows covers the fluid neutral closure and every plasma-side member and
does NOT cover a run of the presented model. `neutral_equilibration` is refused
alongside it, for the different reason that it would overwrite the restored
state.

**The contract is continuation bit-identity**: running $0\to t_\text{end}$ in one
call and running $0\to t_\text{mid}$, exporting, restarting, then
$t_\text{mid}\to t_\text{end}$ produce raw-byte-identical saved frames after
$t_\text{mid}$. `scripts/gates/restart_bitidentity.py` is the gate, comparing
every frame at raw uint64 across four scenarios. Meeting it requires carrying
more than the conserved fields: the continuation caches that seed the next
nonlinear solve, latched phase triggers, accumulators, and the run-loop
controller state living in local variables rather than on the instance. Two are
order-unity rather than last-bit — the previous solve's beam attenuation
cross-section, which seeds the next one, and the save-cadence bookkeeping, a save
not being passive: taking one issues a cathode solve that rewrites the
continuation cache, so save times are part of the trajectory.

The identity reproduces the continuation of the step sequence the first stage
actually took, and an unsplit run's frames only when the handoff instant is one
the unsplit run also steps to exactly: every save instant is a step boundary, but
the save lattice is ACCUMULATED and carries float drift, so stopping at a nominal
save time can stop one ulp before any instant the unsplit run visits. NOT
carried: the `surface_loss` floor-exempt latch, so a handoff taken while a cell
holds an exemption inside the re-admission band resumes with that verdict
re-derived; and the wall-clock budget, wall clock being a property of the process
rather than of the trajectory, so a two-stage run gets two budgets while the
accepted-STEP cap IS carried as an offset.

On load, the cell count, the packed state-vector length and field names, and the
STRUCTURAL configuration keys — those deciding what the payload's members mean
rather than how big a number is — must match the constructed solver exactly, or
the load raises. Every other key may differ, which is what makes a two-stage
hybrid possible, and the producing run's full resolved configuration is retained
in the payload so any difference is auditable.

## Output

Results are written to HDF5 (`results/io.py`, format `sim1d-hdf5-v1`): time
series, axial profiles, per-step diagnostics, the named RHS terms, and the exact
resolved configuration the solver was constructed with, against which the writer
rejects differing caller metadata. `results/health.py` reports finiteness and
conservation drift over a saved trajectory. `TimestepDiagnostics` records every
candidate, the active constraint that set $\Delta t$, and accept/reject
bookkeeping.

## Where each method is implemented

| method | implementation |
|---|---|
| Rusanov / LLF face flux | `physics/flux.py:_rusanov_raw_faces`, `_rusanov_face` |
| Cell-centred physical fluxes; wall closure | `physics/flux.py:physical_fluxes`, `_apply_plasma_walls` |
| Front-filling flux | `physics/flux.py:front_filling_fluxes` |
| KEP single-face flux (boundary) | `physics/flux.py:kep_rusanov_face_scalar` |
| Flux divergence | `physics/flux.py:_flux_divergence` |
| Ghost-cell Bohm outflow | `physics/sources.py:characteristic_boundary_rhs` |
| Sheath-edge sampling | `physics/sources.py:presheath_alpha`, `electrode_sheath_alpha` |
| Geometric momentum source | `physics/sources.py:flux_tube_geometry_rhs` |
| Pressure work, velocity divergence | `physics/sources.py:pressure_work_rhs`, `velocity_divergence`, `hyperbolic_energy_correction_rhs` |
| SSPRK2 | `core/integrator.py:ssprk2_step` |
| Operator split | `solver.py:operator_split_step` |
| Neutral-only backward-Euler step | `solver.py:_implicit_neutral_step`, `_implicit_neutral_step_two_zone` |
| Theta / TR-BDF2 heat substep | `physics/conduction.py:implicit_heat_conduction_step`, `_theta_temperature`, `_tr_bdf2_temperature`, `_banded_heat_operator` |
| Flux-limited conductivity | `physics/conduction.py:flux_limited_electron_conductivity` |
| Adaptive timestep | `core/timestep.py:suggest_timestep` and the per-candidate functions |
| Floors and the floor ledger | `core/state.py:apply_state_floors`, `derive_state`; `solver.py:_floor_additions` |
| Step attempt, validation, retry ladder | `solver.py:_attempt_step`, `_accept_step_attempt`, `_attempt_step_with_retries` |
| Ignition guards | `core/ignition.py`; `solver.py:_open_ignition_switch` |
| Sheath root find, loop advance | `cablp/cathode/circuit.py:solve`, `circuit_idriven.py:solve_idriven`, `circuit_prescribed.py:solve_prescribed`; `physics/cathode.py:advance_circuit_current_driven` |
| Fluid↔circuit Picard | `solver.py:_accept_step_with_picard`, `_picard_snapshot`, `_picard_restore` |
| CSDA beam march; plateau edge | `cablp/cathode/beam_deposition.py:deposit_beam`, `plateau_edge_energy_eV` |
| Velocity grid, moment projection | `physics/kinetic_neutrals.py:stretched_axis`, `stretched_positive_axis`, `VGrid.maxwellian` |
| Launch spectra; extent guard | `physics/kinetic_dvm.py:_cathode_jet_launch_spectrum`, `_anode_jet_launch_spectrum`, `_collector_jet_launch_spectrum`, `_refuse_unreachable_launch_band` |
| Energy-matched wall return | `physics/kinetic_dvm.py:_solve_wall_return_spectra`, `_secant_wall_return_speeds`, `_bisect_wall_return_speeds` |
| Tick booking and ionization debit | `physics/kinetic_dvm.py:_book_transfer`, `_debit_booked_ionization` |
| Transfer hold, scoping and debt ledger | `solver.py:_dvm_scope_step_transfer`, `_dvm_arm_transfer_hold`, `_dvm_transfer_hold_offer`, `_dvm_book_step_transfer` |
| Compiled-kernel selection | `cablp/cathode/kernels.py`; fused lerp `cablp/numerics/interp.py` |
| Restart export, load and compatibility | `results/restart.py:save_restart_state`, `load_restart_state`, `check_restart_compatibility` |
| HDF5 output, health | `results/io.py`, `results/health.py` |

Gate scripts live under `scripts/gates/` (smoke suite, golden baseline and its
digest, order verification, floor-activation audit, interpolation bit-exactness,
restart bit-identity) and `scripts/verify/` (the per-subsystem identity gates);
`scripts/README.md` maps the directory.
