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
The neutral gas is carried as bin masses on the velocity grid below.

## Spatial discretization

**Rusanov / local Lax–Friedrichs flux** at each interior face
(`physics/flux.py`):

$$F=\tfrac12\left(F_L+F_R\right)-\tfrac12\,a_\text{max}\left(U_R-U_L\right)$$

`hyperbolic_wave_speed` selects $a_\text{max}$: `"adiabatic"` uses
$\sqrt{\tfrac53(T_e+T_i)/m_i}$, the exact spectral radius of the $\gamma=5/3$
two-species system, `"isothermal"` uses $\sqrt{T_e/m_i}$, which under-bounds
it. It sets the dissipation strength and the CFL, not the physical wave speed,
which the pressure flux carries. RHS rows are formed as
$-\Delta(\text{area}\cdot F)/\text{volume}$ per cell, each
$\mathbf u\!\cdot\!\nabla$ derivative fused with its compression partner inside
one face flux rather than discretized separately.

**Walls.** Every face bounding the plasma carries no particle or
thermal-energy flux and keeps the live cell's pressure as its momentum flux. At
the plasma-terminating subset the advective flux carries nothing at all: the
ghost-cell Bohm flux (`sources.characteristic_boundary_rhs`) supplies particle,
momentum and energy flux with its own pressure term, using the same face kernel
as the interior (`flux.kep_rusanov_face_scalar`) between the interior cell and
the ghost state, applied as a one-sided divergence
$\pm\,\text{area}\cdot F/V$ on the live cell.

**Energy-consistent hyperbolic core** (`hyperbolic_energy_consistent`). The
convective momentum flux becomes the kinetic-energy-preserving $\{u\}\{M\}$
form, the pressure work a kinetic-energy-preserving $-u\,dM_\text{press}$ per
species folded into the `pressure_work` row, and the Rusanov $(n,M)$ numerical
kinetic-energy dissipation is deposited into $E_i$ as
`hyperbolic_dissipation_heating` — a flux divergence contracted with the local
velocity, so non-negative only in the volume-weighted total. $\sum V(K+E_e+E_i)$
is then conserved by the semi-discrete flux and pressure-work operator to
machine precision, the explicit integration leaving an $O(\Delta t^2)$ drift of
the nonlinear kinetic energy.

**Geometry source and conduction.** With a varying area the momentum ledger
carries

$$\left.\frac{\partial M_i}{\partial t}\right|_\text{geom}=\frac{p_iA_{i+1/2}-p_iA_{i-1/2}}{V_{\mathrm{p},i}}$$

written with the same multiply-then-subtract ordering as the area-weighted
pressure flux it pairs with, so for a stationary uniform-pressure plasma the two
cancel bit for bit — the well-balanced property, excepting the
plasma-terminating cells where the ghost-cell outflow supplies the face
momentum. Conductive face fluxes $q=-\kappa\nabla T$ are differenced to a
conservative flux divergence (`physics/conduction.py`), the flux limiter applied
at the frozen incoming $T_e$ in both paths.

## Time integration

**Explicit stage** — a two-stage strong-stability-preserving Runge–Kutta step
(SSPRK2 / Heun), `core/integrator.py`:

$$y^{(1)}=\Pi\!\left[y^n+\Delta t\,L(t^n,y^n)\right],\qquad y^{n+1}=\Pi\!\left[\tfrac12y^n+\tfrac12\left(y^{(1)}+\Delta t\,L(t^n+\Delta t,y^{(1)})\right)\right]$$

$\Pi$ the floor projection below, applied at each stage. The stages are
evaluated at $t^n$ and $t^n+\Delta t$, preserving second-order accuracy for
explicitly time-dependent forcing; omitting the `time` argument freezes the
forcing at the step start and is first-order in it.

**Operator split.** `implicit_heat_conduction` composes an explicit SSPRK2 step
over all non-heat terms (operator $A$) with an implicit heat substep (operator
$B$), removing the stiff parabolic stability limit from the explicit step.
`operator_splitting` selects the composition: `"lie"` applies $A(\Delta t)$ then
$B(\Delta t)$ and is $O(\Delta t)$, the splitting error going as
$\Delta t\,[A,B]$; `"strang"` applies
$B(\Delta t/2)\to A(\Delta t)\to B(\Delta t/2)$, whose symmetry cancels that
leading commutator and leaves $O(\Delta t^2)$.
`beam_deposition_in_heat_substep` moves the beam's electron-energy row from
$A$'s explicit sum into $B$, applied as a source held constant over each substep
on the same tridiagonal operator; the beam's particle births, ionization cost
and excitation radiation stay in $A$.

**Implicit heat substep**, solved per species as a tridiagonal system via
`scipy.linalg.solve_banded`. Three of the four schemes are theta methods,

$$\left(C+\theta\,\Delta t\,K\right)T^{n+1}=C\,T^n-(1-\theta)\,\Delta t\,K\,T^n$$

$C$ the heat capacity and $K$ the conduction operator built from the same face
coefficients as the explicit half.

| `implicit_heat_scheme` | $\theta$ | $R(-\infty)$ | L-stable | banded solves | substep order |
|---|---|---|---|---|---|
| `backward_euler` | 1 | 0 | yes | 1 | 1 |
| `shifted` | 0.6 | $-2/3$ | no | 1 | 1 |
| `crank_nicolson` | 0.5 | $-1$ | no | 1 | 2 |
| `tr_bdf2` | — | 0 | yes | 2 | 2 |

At $\theta=1$, $C+\Delta t\,K$ is an M-matrix whose rows sum to $C$ (with
$K\mathbf 1=0$), giving the discrete maximum principle $T^{n+1}\ge\min(T^n)$:
backward Euler is unconditionally monotone and cannot undershoot the temperature
floors. For $\theta<1$ the amplification factor tends to $-(1-\theta)/\theta$ as
$\Delta t\,\lambda\to-\infty$, so stiff modes ring — undamped at
$\theta=\tfrac12$ — and can be clipped by a floor, which injects energy.
`tr_bdf2` is second-order *and* L-stable: a trapezoidal stage out to
$t^n+\gamma\Delta t$ then a BDF2 stage through $(T^n,T_\gamma,T^{n+1})$, with
$\gamma=2-\sqrt2$ making both stages share the implicit coefficient
$\gamma/2=(1-\gamma)/(2-\gamma)$ and hence one banded operator — two
`solve_banded` calls against a single matrix. It damps undershoot rather than
preventing it.

**Picard iterations on $\kappa$.** The conductivity is frozen at the incoming
state, and that — not the scheme — caps the substep at first order: against the
live $\kappa\propto T^{5/2}$ the measured substep order is $\approx1.03$ for
`backward_euler` and $\approx1.06$ for `tr_bdf2`, while at fixed linear $\kappa$
the same schemes measure $1.02$ and $2.02$. `heat_picard_iterations`
re-evaluates $\kappa$ at the scheme's own flux point, one more banded solve per
species per iteration. Second order in the whole split step requires **all
three** of a second-order `implicit_heat_scheme`, a positive
`heat_picard_iterations`, and `operator_splitting = "strang"`, the frozen
conductivity and the Lie splitting being independent first-order error terms.
`scripts/gates/verify_sim1d_order.py` measures the observed order by
fixed-$\Delta t$ Richardson refinement, in a regime with floors inert and
watched, a single phase, an autonomous RHS and no cathode; a discharge does not
show it, floors binding and phase transitions being threshold-triggered.

## Adaptive timestep control

`core/timestep.py:suggest_timestep` takes $\Delta t$ as the minimum over
candidate bounds, then clamps to $[\Delta t_\text{min},\Delta t_\text{max}]$.
Three inequality forms recur, over the plasma-active cells:

$$\text{distance: }\Delta t\le\varepsilon\min\frac{d}{s},\qquad \text{fractional: }\Delta t\le\varepsilon\min\frac{X-X_\text{floor}}{|\dot X|},\qquad \text{rate: }\Delta t\le\frac{\varepsilon}{\max\nu}$$

| candidate | inequality |
|---|---|
| `plasma_cfl` | distance, $d$ the centre distance and $s=\tfrac12(|u_L|+|u_R|+c_{s,L}+c_{s,R})$ per face, $\varepsilon$ = `cfl` |
| `front_density` | fractional on $n$ against the front-filling flux row, $\varepsilon$ = `density_dt_fraction` |
| `reactions` | fractional on $n$ against the bulk reaction row |
| `surface_loss` | negative-margin — $\Delta t\le\varepsilon\min(\text{margin}/\lvert\dot X\rvert)$ over DRAINING cells only, margins $n-n_\text{floor}$ and the exact conservative $E_s-\tfrac32nT_{s,\text{floor}}$ whose rates include the change in floor energy when $n$ changes, $d(E-\tfrac32nT_\text{floor})/dt=\dot E-\tfrac32T_\text{floor}\dot n$. Bundles the cathode/sheath, anode-collection and plasma-terminating boundary rows, plus an engaged kinetic arm's plasma-side coupling term |
| `energy_exchange` | fractional on $E_e$, $E_i$ against $Q_{ie}$ |
| `electron_cooling` | fractional on $E_e$ against the inelastic and radiative rows |
| `ion_charge_exchange` | fractional on $E_i$ against the charge-exchange row |
| `ion_neutral_drag` | rate, $\Delta t\max\nu_{in}\le$ `DRAG_DT_FRACTION` |
| `heat_conduction` | explicit parabolic diffusion bound $\Delta t\le\varepsilon\min(\Delta z^2C/\kappa)$ at `conduction.HEAT_DT_FRACTION`; withdrawn on the implicit path |
| `neutral_exchange` | fractional on $n_n$ against the pair-exchange row, $\varepsilon$ = `neutral_dt_fraction` |
| `neutral_sources` | fractional on $n_n$ against the fueling and pumping rows |
| `neutral_wind` | distance, $\Delta t\le\varepsilon\min(\Delta z/\lvert u_n\rvert)$, the neutral advective CFL |
| `neutral_energy` | rate — $\Delta t$ times the summed neutral-energy relaxation rates below `neutral_dt_fraction`, folding in the neutral signal speed $\lvert u_n\rvert+c_n$ |
| `circuit` | $\Delta t\le\varepsilon\,\tau_\text{circuit}$, $\tau_\text{circuit}=L/(R_\text{comp}+R_\text{mesh}+dV_\text{dis}/dI)$, the device slope read by a one-sided finite difference of the same evaluator the loop advance integrates. Withdrawn once the loop reaches local equilibrium, tested as $\lvert f(I)\rvert\,\tau_\text{circuit}<dI_\text{probe}$. An accuracy bound, the loop advance being L-stable |
| `dt_max` | the configured ceiling |

**Floor-aware drain exemption.** A cell whose energy margin above its floor
energy is within `SURFACE_LOSS_FLOOR_EXEMPT_RTOL` of it is dropped from the
`surface_loss` bound alone, re-admitted at the wider
`surface_loss_floor_exempt_exit_rtol` so a hovering cell holds its previous
verdict inside the band (values at or above 1 are refused). The accept-time clip
resets a floor-pinned cell's margin to float residue every step, so without the
exemption a persistent drain re-trips the bound forever and pins $\Delta t$ at
$\Delta t_\text{min}$ while the floor holds the cell. The density channel is
never exempted, every other bound still governs an exempted cell, and an
exempted cell is never the reported active constraint.

**A bound must describe a row the step applies.** The kinetic neutral arm zeroes
whole rows of the fluid terms and carries them in its own coupling term, so
while engaged the `ion_charge_exchange`, `ion_neutral_drag`, `neutral_exchange`
and `neutral_sources` candidates are withdrawn to infinity and the reaction
bound keeps only its plasma channel, their replacements bounded through the
`surface_loss` bundle.

**Growth ramp and clamp.** Between accepted steps $\Delta t$ may grow by at most
`dt_growth_factor`, applied after `suggest_timestep` as one more cap, so it can
only shrink the step; after `dt_growth_recovery_patience` CONSECUTIVE
ramp-capped accepted steps the factor becomes `dt_growth_recovery_factor`, and
one step capped by anything else resets the streak. The active constraint names
the bound that actually minimized; when it asked for less than
$\Delta t_\text{min}$, `clamped_to_dt_min` is set and `dt_raw` keeps the
unclamped request, $\texttt{dt\_raw}=0$ being the drained floor-pinned signature.
`dt_min_lock_max_steps` bounds CONSECUTIVE clamped adaptive steps and raises past
it, its counter driven by
`clamped_to_dt_min OR (accepted_dt <= dt_min AND dt_raw > dt_min)` evaluated
after acceptance.

## Step acceptance and rejection

A candidate step is attempted without committing state (`_attempt_step`) and
then validated; a non-finite or otherwise invalid state is rejected and retried
at a reduced $\Delta t$, with rejection events and constraint histories stored
for post-run diagnostics. `raw_stage_validation` additionally inspects both
SSPRK candidates and the implicit candidates *before* floors are applied,
covering $n$, $n_n$, $E_e$ and $E_i$; a failed candidate carries its raw
rejection evidence but cannot mutate accepted state, the circuit or surface
caches, kinetic targets, time, or the cumulative floor ledger.

A run that fails to ignite is a phase transition, not a step rejection. Two
guards open the cathode switch and route the run through the ordinary afterglow
to a finite end time (`core/ignition.py`): a **stall detector**,
$\gamma_N\le0$ AND $d(E_{e,\text{total}})/dt\le0$ for every saved sample across a
sustained window — a structural joint condition with no tuned rates — and the
**pre-breakdown timeout**, the machine's own hardware guard.
`prebreakdown_timeout_action` selects between opening the switch and raising.

## Floors

Enforced by `apply_state_floors` / `floor_state_vector` at construction and at
the end of every stage, with `derive_state` additionally applying the
temperature floors on every read. Densities are clipped directly on the packed
rows; the temperatures are clipped on the DERIVED quantities, and the packed
$E_e$, $E_i$ change only where `conservative_from_primitives` rebuilds them from
the floored primitives. Order within a call: $n$, then $n_n$, then the
temperatures, then the rebuild, then the optional rows, the neutral energy floor
taken last against the already-floored $n_n$. Momenta are not clipped — $u$ is
recovered with the floored density and $M$ rebuilt from it, leaving $M$
numerically unchanged. Each accepted repair books its exact extensive debit in
`floor_ledger`; `scripts/gates/audit_sim1d_floor_activation.py` instruments the
clip sites at run time, which cannot be done post-hoc.

## The cathode solve

The sheath root is bracketed on a monotone residual and solved to the bracket's
floating-point resolution: the voltage-driven form roots the coupled residual in
$\psi_+$, the current-driven form roots
$J_\text{tot}(\psi_+)=J_\text{imposed}$, and the prescribed-drive form roots
$\phi_c+V_p-\phi_a(\phi_c)-V_b$ on $(0,\phi_{c,\text{cap}}]$. Uniqueness rests on
each residual's monotonicity, and a root above the composed ceiling raises rather
than being accepted. The loop current is advanced by an L-stable TR-BDF2 stage
split over $L\,dI/dt=V_\text{src}-IR-V_\text{dis}(I)$, with $V_\text{dis}$
evaluated at the frozen plasma state as a function of trial current and a
modelled bank capacitor discharging trapezoidally alongside. The fluid stages run
at a loop current frozen over the step; `coupled_circuit_picard` re-runs the
accepted step, at most `circuit_picard_max_iter` times, with that current updated
to the previous iteration's result until the current a step produces matches the
one it ran at to `circuit_picard_tol_rel`, a snapshot/restore pair restoring
every step-mutated attribute exactly so a rejected iteration leaves no trace.

**The beam march.** The CSDA ray is integrated over adaptive substeps
$dz_\text{sub}=\min(\text{remaining},\,f_\text{sub}E/L_\text{tot})$, so each
substep resolves a fixed fraction of the primary's remaining energy. The same
floating-point decrements feed the energy update and the per-channel banks, so
the per-ray energy identity closes to accumulated roundoff by construction
rather than by a tolerance. The beam coupling length is solved by bisection on a
monotone relation at every extraction solve.

## The kinetic neutral solver

**Velocity grid** (`physics/kinetic_neutrals.py:VGrid`). Both axes are
sinh-stretched about one fine scale $v_\text{fine}$ out to a common half-extent,
$v_\parallel$ spanning $(-v_\text{max},+v_\text{max})$ and the perpendicular
SPEED axis $(0,v_\text{max})$:

$$v_k=v_\text{fine}\sinh\!\left(a\,u_k\right),\qquad a=\operatorname{arcsinh}\!\left(v_\text{max}/v_\text{fine}\right)$$

$u_k$ the half-offset normalized index, so resolution is $\sim v_\text{fine}$
near zero (the wall gas) and coarsens toward $v_\text{max}$ (the
charge-exchange tail). No bin sits at exactly zero — centres are at
half-offsets — so the upwind march never divides by zero and every bin
transports.

**Extent.** `neutral_kinetic_dvm_vmax_cm_s` pins the half-extent. Unset, it is
sized to the launch band the armed surface jets can produce,
$1.25\sqrt{2\varepsilon_\text{max}/m}$ with
$\varepsilon_\text{max}=\max(R_E/R_N)(\phi_{c,\text{cap}}+10\ \mathrm{eV})$ over
those jets, the factor a sizing margin whose sufficiency is checked rather than
assumed; with no jet armed it falls back to a thermal/sonic sizing from the ion
thermal speed and sonic drift. The whole band is checked at CONSTRUCTION,
energies spanning each armed jet's band going through the same launch-spectrum
builder the tick uses. A drift past the LAST BIN CENTRE is not approximated — no
non-negative weighting of bins has a mean beyond it — so a spectrum asked for
there raises, and a finer $n_{v_\parallel}$ LOWERS that ceiling, the grid-tied
smear being the local bin width.

**Moment-preserving projection** (`VGrid.maxwellian`). A drifting Maxwellian is
placed as analytic per-bin masses — erf differences along $v_\parallel$,
Rayleigh CDF differences along $v_\perp$ (the 2D perpendicular speed measure) —
so the density is exact by construction and the masses sum to 1. A two-basis
compensation,

$$f'=f+a\left(v_\parallel-\langle v_\parallel\rangle\right)f+b\left(V^2-\langle V^2\rangle\right)f$$

is then solved so the DISCRETE first and second moments, evaluated at bin centres
the way the transport uses them, hit their analytic targets
$\langle v_\parallel\rangle=u$ and $\langle V^2\rangle=u^2+3s^2$. Both basis
functions are moment-free about the current state analytically, so the density is
preserved; numerically that cancellation leaves a residue, which two guards hold.
The $2\times2$ is NON-DIMENSIONALIZED before it is judged or solved — its rows
are moments of different order and its columns carry the reciprocal dimensions of
the two amplitudes, so the raw condition number is dimensional and says nothing
about the solve. A scaled condition number at or above $1/\epsilon$ is REJECTED,
and the sum is restored at the end if it has drifted past tolerance; both guards
are inert while the invariant holds.

**Launch spectra.** A monoenergetic surface jet at energy per atom
$\varepsilon$, equivalently a speed $v_\text{back}=\sqrt{2\varepsilon/m}$, is
placed through that projection with the drift SOLVED FROM THE ENERGY,

$$u=\pm\sqrt{v_\text{back}^2-3kT_\text{launch}/m}$$

which makes $\langle V^2\rangle=v_\text{back}^2$ and the discrete mean energy
exactly $\varepsilon$; a spectrum drifting at $v_\text{back}$ would carry
$\varepsilon+\tfrac32kT_\text{launch}$ per atom, the smear's own thermal content
on top of the beam, which the surface book was not debited for. The sign is taken
from the launch side. $T_\text{launch}$ is grid-tied,
$m\,\Delta v_\parallel(v_\text{back})^2/k_B$ — the narrowest spectrum the grid
resolves there, narrower leaving the compensation nothing to redistribute. The
accepted spectrum's density, drift and mean energy are compared against their
targets at a relative bar and a miss RAISES; the energy ledger books the birth as
the count times the moment of the array actually placed, closing by construction
rather than by the projection's accuracy. A launch energy below
$\tfrac32kT_\text{launch}$ has no drift and raises.

**Energy-matched wall return.** The non-accommodated share is re-emitted on the
cosine shape at a temperature that must be SOLVED, so the spectrum carries the
retained share's own incident mean energy per atom
$\bar e=E_\text{incident}/N_\text{incident}$. The discrete mean energy

$$E(s)=\sum_{jk}f_{jk}(s)\,\tfrac12m\left(v_{\parallel,j}^2+v_{\perp,k}^2\right),\qquad s=\sqrt{kT/m}$$

rises monotonically with $s$, so the inverse is a one-parameter root find in two
tiers. First a **secant iteration in $\ln s$** on
$F(\ln s)=\ln E(s)-\ln\bar e$, evaluated on the SEPARABLE contraction of that sum
so no per-cell $(\text{cells},n_{v_\parallel},n_{v_\perp})$ array is built per
evaluation: in logs the continuum relation $E=2ms^2$ is an exactly linear
residual of slope 2, so the seed $s_0=\sqrt{\bar e/2m}$ plus one Newton step at
that slope lands close and the secant closes the rest. Every cell steps on every
sweep and the stopping test is a single whole-call reduction. The secant hands
the WHOLE call to the second tier when it misses its bar, when a residual or step
is not finite, or when its denominator vanishes on a flat residual — what a
target above the grid's saturation energy produces. The second tier is a
**bracketed bisection**: seeded at the same continuum relation, bracketed outward
by halving and doubling, then bisected to the floating-point resolution of its
endpoints. $E(s)$ SATURATES once the spectrum outruns the outermost bins, so a
target above that has no solution and both the bracket search and the final
agreement check raise rather than returning a spectrum at an energy the caller
did not ask for.

**The transfer relaxation.** The plasma-side momentum/energy transfer is booked
once per neutral clock tick while the plasma steps many times inside it. The
charge-exchange/elastic pair is a relaxation,
$dE_i/dt=-\nu(E_i-E_i^\text{eq})$ and $dM/dt=-\nu(M-M^\text{eq})$, with one
$\nu=N_\text{loss}/(V\Delta t\,n_i)$ per cell for the pair.
`neutral_kinetic_dvm_transfer_hold` selects how the plasma applies it between
ticks: `"zoh"` freezes the booked rate, advancing
$X_{k+1}=X_k-\nu\Delta t(X_k-X_\text{eq})$, which for $\nu\Delta t>2$ is an
amplification with a sign flip; `"exponential"` applies, per cell and per plasma
step, at the tick's frozen rate and target,

$$E_i\leftarrow E_i^\text{eq}+\left(E_i-E_i^\text{eq}\right)e^{-\nu\,dt}$$

and the momentum row at the same $\nu$. Applied as a constant rate over the step,
so the SSPRK2 stages integrate it exactly: unconditionally stable, exact for the
linearized system, unable to carry a row past its target at any $\Delta t$, and
reducing to the zero-order hold to $O(\nu\,dt)$.

What the plasma applies differs from what the tick booked, and the difference is
carried as a per-cell **hold debt** — distinct from the floor debt, which says
the plasma could not absorb it, where hold debt says the tick froze a rate at a
state that then moved. It is first-order in $\nu\Delta t$ and vanishes as the
neutral clock refines, making it the cadence meter, and the ledger identity
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

**The counted ionization debit** is taken LAST in the tick, against
$f_c^\text{marched}+\text{birth}_{cx}+\text{birth}_{el}$ — the inventory the cell
carries once the tick ends, net of the non-conserving losses only — so the drop
is drawn from an array that actually exists and positivity is structural rather
than checked. The per-cell handshake
`ion_removed_cum + ion_debt == ion_booked_cum` and the ledger's
`inventory_after − inventory_before == births − losses` are exact at every tick.

## Compiled kernels

The cathode scalar kernels and the sheath root find built on them have a Cython
transcription. It is a **faithful transcription, not a reimplementation**, and is
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

**The contract is continuation bit-identity**: running $0\to t_\text{end}$ in one
call and running $0\to t_\text{mid}$, exporting, restarting, then
$t_\text{mid}\to t_\text{end}$ produce raw-byte-identical saved frames after
$t_\text{mid}$. Meeting it requires carrying more than the conserved rows: the
continuation caches that seed the next nonlinear solve, latched phase triggers,
accumulators, and the run-loop controller state living in local variables rather
than on the instance. Two are order-unity rather than last-bit — the previous
solve's beam attenuation cross-section, which seeds the next one, and the
save-cadence bookkeeping, a save not being passive: taking one issues a cathode
solve that rewrites the continuation cache, so save times are part of the
trajectory.

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
series, axial profiles, per-step diagnostics, the named RHS rows, and the exact
resolved configuration the solver was constructed with, against which the writer
rejects differing caller metadata. `results/health.py` reports finiteness and
conservation drift over a saved trajectory. `TimestepDiagnostics` records every
candidate, the active constraint that set $\Delta t$, and accept/reject
bookkeeping.

## Where each method is implemented

| method | implementation |
|---|---|
| Rusanov / LLF face flux, wall closure | `physics/flux.py:physical_fluxes`, `_apply_plasma_walls` |
| KEP single-face flux (boundary) | `physics/flux.py:kep_rusanov_face_scalar` |
| Flux divergence | `physics/flux.py:_flux_divergence` |
| Ghost-cell Bohm outflow | `physics/sources.py:characteristic_boundary_rhs` |
| Sheath-edge sampling | `physics/sources.py:presheath_alpha`, `electrode_sheath_alpha` |
| Geometric momentum source | `physics/sources.py:flux_tube_geometry_rhs` |
| Pressure work, velocity divergence | `physics/sources.py:pressure_work_rhs`, `velocity_divergence` |
| SSPRK2, operator split | `core/integrator.py:ssprk2_step`, `operator_split_step` |
| Theta / TR-BDF2 heat substep | `physics/conduction.py:implicit_heat_conduction_step`, `_banded_heat_operator` |
| Flux-limited conductivity | `physics/conduction.py:flux_limited_electron_conductivity` |
| Adaptive timestep | `core/timestep.py:suggest_timestep` and the per-candidate functions |
| Floors and the floor ledger | `core/state.py:apply_state_floors`, `derive_state`; `solver.py:_floor_additions` |
| Step attempt, validation, retry | `solver.py:_attempt_step`, `_accept_step_attempt` |
| Ignition guards | `core/ignition.py` |
| Sheath root find, loop advance | `cablp/cathode/circuit.py:solve`, `circuit_idriven.py:solve_idriven`, `circuit_prescribed.py:solve_prescribed`; `physics/cathode.py:advance_circuit_current_driven` |
| Fluid↔circuit Picard | `solver.py:_accept_step_with_picard`, `_picard_snapshot`, `_picard_restore` |
| CSDA beam march | `cablp/cathode/beam_deposition.py:deposit_beam` |
| Velocity grid, moment projection | `physics/kinetic_neutrals.py:stretched_axis`, `stretched_positive_axis`, `VGrid.maxwellian` |
| Launch spectra, extent guard | `physics/kinetic_dvm.py:TransientDVM._refuse_unreachable_launch_band` |
| Energy-matched wall return | `physics/kinetic_dvm.py:_secant_wall_return_speeds`, `_bisect_wall_return_speeds` |
| Transfer hold, debt ledger, ionization debit | `physics/kinetic_dvm.py` (tick booking; `_debit_booked_ionization`) |
| Compiled-kernel selection | `cablp/cathode/kernels.py`; fused lerp `cablp/numerics/interp.py` |
| Restart export and load | `results/restart.py:save_restart_state` |
| HDF5 output, health | `results/io.py`, `results/health.py` |

Gate scripts live under `scripts/gates/` (smoke suite, golden baseline and its
digest, order verification, floor-activation audit, interpolation bit-exactness,
restart bit-identity) and `scripts/verify/` (the per-subsystem identity gates);
`scripts/README.md` maps the directory.
