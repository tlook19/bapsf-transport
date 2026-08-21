# sim1d Model Equations

The five fields evolved by `LAPDSim1D` written in **non-conservative,
convective-derivative (Braginskii material-derivative) form** — the continuum
equations from which the code's conservative finite-volume scheme (see
[`NUMERICS.md`](NUMERICS.md)) is derived. Source/sink terms and signs mirror the
implementation in `physics/`.

## Notation

- Material derivative: $\dfrac{D}{Dt} \equiv \dfrac{\partial}{\partial t} + \mathbf{u}\cdot\nabla$
- Quasineutral, singly charged: $n_e = n_i \equiv n$; one bulk flow $\mathbf{u}$
  (ions carry the inertia); neutrals $n_n$ with their own flow $\mathbf{u}_n$.
- Total pressure $p = n(T_e + T_i)$; species pressures $p_e = nT_e$,
  $p_i = nT_i$ (energies stored as $\tfrac32 nT$).
- Rates: ionization $S_{iz} = n\,n_n\langle\sigma v\rangle_{iz}(T_e)$; radiative
  recombination $S_{rr} = \alpha_r(T_e)\,n^2$; three-body recombination
  $S_{3b} = \alpha_3(T_e)\,n^3$.

## Equations

> **Production stance (2026-07-27).** The equations below are the model's
> base (legacy-default) forms. The production configuration replaces three
> of them with later closures documented in their own sections:
> (i) the ion-neutral quartet in eqs. 3 and 5 — the $\nu_{in}(T_i)$ drag and
> the $Q_{cx}/Q_{\text{fric}}/Q_{\text{eq,el}}$ energy terms — is replaced by
> the **R4.3 moment-closed operator** (`ion_neutral_moment_closure=True`):
> drag at the Phelps momentum-transfer rate $\nu_{mt}$, with
> $Q_{in} = \tfrac12 m_i \nu_{mt} n u^2 + \tfrac32 \nu_{mt} n (T_n - T_i)$;
> (ii) the neutral channel (eq. 2) runs with **no dynamically evolved bulk
> neutral flow** — transport is the Knudsen/Clausing conductance closure, so
> the production form is $\partial_t n_n + \nabla_\parallel\!\cdot\Gamma_n =
> -S_{iz} + (S_{rr}+S_{3b}) + S_{gp} - S_{\text{pump}}$ with $\Gamma_n$ the
> conductance flux (neutrals ionize at rest; the `neutral_momentum` /
> two-zone selectors are default-off closure instruments);
> (iii) the electron energy (eq. 4) additionally carries $Q_{\text{beam}}$,
> the CSDA primary-beam deposition (R4.1/R4.2 sections), so $S_{iz}$ there
> is thermal + beam ionization. `core/config.py` is the authoritative flag
> list; this note only marks where the headline forms and the production
> defaults differ.

**1. Plasma continuity** (`physics/reactions.py`, `physics/flux.py`)

$$\frac{Dn}{Dt} = -\,n\,\nabla\!\cdot\mathbf{u} \;+\; S_{iz} \;-\; \big(S_{rr} + S_{3b}\big)$$

**2. Neutral continuity** (`physics/neutrals.py`, `physics/reactions.py`)

$$\frac{D_n n_n}{Dt} = -\,n_n\,\nabla\!\cdot\mathbf{u}_n \;-\; S_{iz} \;+\; \big(S_{rr} + S_{3b}\big)$$

By default the directed neutral flow is not solved dynamically: thermal
transport uses the **Knudsen/Clausing conductance closure**. The default-off
`neutral_momentum` selector additionally evolves directed axial momentum on
top of that thermal transport.

Optional thin annular baffles are neutral-transport surfaces, not plasma
boundaries. A baffle with clear radius $R_b\ge R_p$ leaves the plasma channel
and its fluxes unchanged while adding the free-molecular series-orifice
conductance $(\bar v_n/4)\,\pi R_b^2$ to the single neutral channel. In the
two-zone closure it leaves the column conductance unchanged and adds only the
annulus orifice $(\bar v_n/4)\,\pi(R_b^2-R_p^2)$. Positions and clear radii are
presence-gated by the default-off `neutral_baffles` selector.

The default-off
`neutral_momentum_radial="kinetic_two_moment"` reduction requires both
`neutral_momentum` and `neutral_two_zone`. It evolves column and annulus
momenta $M_c=m_i n_c u_c$ and $M_a=m_i n_a u_a$. Ion-neutral drag transfers
momentum to $M_c$ exactly. Radial transfer is

$$
\begin{aligned}
\dot M_c|_r&=-\nu_{ca}M_c+\frac{V_a}{V_c}\nu_{ac}M_a,\\
\dot M_a|_r&=\frac{V_c}{V_a}\nu_{ca}M_c-\nu_{ac}M_a-\nu_wM_a,
\end{aligned}
$$

with $\nu_{ca}=\bar v(T_i)/(2R_p)$,
$\nu_{ac}=\bar v(300\,\mathrm{K})R_p/[2(R_m^2-R_p^2)]$, and
$\nu_w=\bar v(300\,\mathrm{K})R_m/[2(R_m^2-R_p^2)]$. Thus the radial exchange
conserves $V_cM_c+V_aM_a$ and only the annulus deposits directed momentum on
the vessel. Baffles restrict annulus advection to their open annular aperture
and add the diffuse blocked-area loss $(\bar v/4)A_{\rm blocked}M_a/V_a$ on
their adjacent cells. There is no fitted coefficient and no added neutral
pressure term; the existing Clausing operator continues to carry thermal
transport.

**3. Momentum** (ion inertia, total-pressure gradient; `physics/flux.py`,
`physics/reactions.py`, `physics/sources.py`)

$$m_i\,n\,\frac{D\mathbf{u}}{Dt} = -\,\nabla p \;-\; m_i\,\mathbf{u}\,S_{iz} \;-\; m_i\,\nu_{in}(T_i)\,n\,\mathbf{u}$$

For the optional variable-area flux-tube geometry, the implemented
quasi-1D conservative form is

$$\partial_t(A\rho u)+\partial_z[A(\rho u^2+p)] = p\,\partial_z A+A S_M.$$

The geometric pressure source is required to preserve a uniform stationary
plasma exactly. This closure represents area divergence only; it does not
include pressure anisotropy or an explicit magnetic-mirror force, so parallel
acceleration through an end-solenoid flare is an observable rather than an
imposed result.

- $-m_i\mathbf{u}\,S_{iz}$ — **ion-loading drag**: neutrals ionize at rest, so
  newly created cold ions mass-load and slow the flow. (The recombination and
  wall-loss momentum sinks cancel identically against their continuity
  contributions when moved to convective form.)
- $-m_i\,\nu_{in}\,n\,\mathbf{u}$ — **ion-neutral collisional drag** (friction on
  the flow from the neutral background), with momentum-transfer collision
  frequency
  $$\nu_{in}(T_i) = n_n\,\bigl(k_b + \tfrac12 k_\text{iso}\bigr)(T_\text{eff}), \qquad T_\text{eff} = \tfrac12 (T_i + T_n),$$
  the Phelps He$^+$/He isotropic + backscatter momentum-transfer rate
  (`sigma_in_model = "phelps"`, the only accepted value since D3).
  Toggled by the `ion_neutral_drag` flag and scaled by `b_ion_neutral_drag`.
  Setting the
  `ion_neutral_drag_cx_only` flag drives the drag with the resonant
  charge-exchange rate $\nu_{cx} = n_n\langle\sigma v\rangle_{cx}$ instead of
  $\nu_{in}$; in that mode the elastic fraction $\nu_{el}\to0$, so the
  $Q_{\text{fric}}$ and $Q_{\text{eq,el}}$ ion-energy terms below vanish.

**4. Electron energy** (`physics/sources.py`, `physics/conduction.py`,
`physics/energy.py`)

$$\frac{3}{2}\,n\,\frac{DT_e}{Dt} = -\,p_e\,\nabla\!\cdot\mathbf{u} \;+\; \nabla\!\cdot\!\big(\kappa_{\parallel e}\nabla T_e\big) \;-\; Q_{ie} \;-\; C_e \;+\; \tfrac{3}{2}\big(T_{e,\text{birth}} - T_e\big)S_{iz}$$

**5. Ion energy** (`physics/sources.py`, `physics/conduction.py`,
`physics/energy.py`)

$$\frac{3}{2}\,n\,\frac{DT_i}{Dt} = -\,p_i\,\nabla\!\cdot\mathbf{u} \;+\; \nabla\!\cdot\!\big(\kappa_{\parallel i}\nabla T_i\big) \;+\; Q_{ie} \;-\; Q_{cx} \;+\; Q_{\text{fric}} \;+\; Q_{\text{eq,el}} \;+\; \tfrac{3}{2}\big(T_{i,\text{birth}} - T_i\big)S_{iz}$$

## Energy source glossary

- $-p_{e,i}\,\nabla\!\cdot\mathbf{u}$ — compressional (pressure) work.
- $\nabla\!\cdot(\kappa_\parallel\nabla T)$ — parallel Braginskii heat conduction
  ($\kappa_{\parallel e}$, $\kappa_{\parallel i}$).
- $Q_{ie}$ — electron→ion collisional energy exchange (sink for electrons,
  source for ions).
- $C_e$ — electron inelastic/radiative cooling: line radiation
  ($\propto n\,n_n$), ionization cost, and recombination radiation
  ($\propto n^2$).
- $Q_{cx}$ — ion charge-exchange energy loss to neutrals (relaxes
  $T_i \to T_n$).
- $Q_{\text{fric}} = \tfrac12 m_i\,\nu_{el}\,n\,u^2$ — **elastic ion-neutral
  frictional heating**: for equal masses, half the drift energy dissipated by the
  elastic collision fraction heats the ions (the charge-exchange fraction instead
  carries its energy off with the fast neutral). Uses the elastic rate
  $\nu_{el} = \max(\nu_{in} - \nu_{cx},\,0)$ with $\nu_{cx} = n_n\langle\sigma
  v\rangle_{cx}$; shares the `ion_neutral_drag` flag and `b_ion_neutral_drag`
  scale.
- $Q_{\text{eq,el}} = \tfrac32\,\nu_{el}\,n\,(T_n - T_i)$ — **elastic thermal
  equilibration** (the elastic companion to $Q_{cx}$), gated separately by the
  `ion_neutral_thermalization` flag (default off). The drag, $Q_{cx}$,
  $Q_{\text{fric}}$, and $Q_{\text{eq,el}}$ quartet is **replaced together** by
  one moment-closed operator under the default-off `ion_neutral_moment_closure`
  flag (audit A7/A8; see the R4.3 section below).
- $\tfrac32(T_\text{birth} - T)\,S_{iz}$ — thermal cost of injecting
  freshly-ionized particles at the birth temperature (vanishes for the default
  `birth="local"` electron choice).

## Conservative form: where the convective derivatives go

The equations above are written in **material-derivative** form
$D/Dt = \partial_t + \mathbf{u}\cdot\nabla$ for physical readability, but
`LAPDSim1D` never discretizes a $\mathbf{u}\cdot\nabla$ term. It integrates the
algebraically-equivalent **conservative** form

$$\partial_t U + \nabla\!\cdot\mathbf{F}(U) = S,\qquad U\in\{n,\,M,\,E_e,\,E_i\},$$

in which each convective derivative is fused with its compression partner inside
a single flux divergence and evaluated as one Rusanov/LLF face flux (see
[`NUMERICS.md`](NUMERICS.md)). The bridge between the two forms is the
product-rule identity

$$\nabla\!\cdot(U\mathbf{u}) = \underbrace{\mathbf{u}\cdot\nabla U}_{\text{convective}} \;+\; \underbrace{U\,\nabla\!\cdot\mathbf{u}}_{\text{compression}},$$

applied once per advected field. **Continuity** is the archetype: the code's
$\partial_t n + \nabla\!\cdot(n\mathbf{u}) = S$ carries the whole divergence in
$F_n = n u$; moving the convective half onto the LHS to build $Dn/Dt$ leaves the
compression half $-n\,\nabla\!\cdot\mathbf{u}$ on the RHS, which is the term
printed in Eq. 1. The two are the same object split down the middle, so there is
nothing to grep for — $\mathbf{u}\cdot\nabla n$ exists only welded to
$n\,\nabla\!\cdot\mathbf{u}$ inside the flux.

The mapping for every advected field (`physics/flux.py`, `physical_fluxes`):

| material-derivative piece | fused into flux |
|---|---|
| $\mathbf{u}\cdot\nabla n$ | $F_n = n u$ |
| $\mathbf{u}\cdot\nabla\mathbf{u}$ (in $D\mathbf{u}/Dt$) | $F_M = M u + p$ (convective part) |
| $\mathbf{u}\cdot\nabla E_e$ | $F_{Ee} = E_e u$ |
| $\mathbf{u}\cdot\nabla E_i$ | $F_{Ei} = E_i u$ |

Conversely, three RHS terms in Eqs. 1–5 are **residues of the change of form**,
not independent physics — they are what does *not* cancel when the conservative
densities are expanded back into material-derivative form:

- $-n\,\nabla\!\cdot\mathbf{u}$ and $-p_{s}\,\nabla\!\cdot\mathbf{u}$ — the
  compression partners just described. The plasma one rides inside $F_n$; the
  **pressure work** $-p_s\,\nabla\!\cdot\mathbf{u}$ is instead a *separate* local
  source (`sources.pressure_work_rhs`), because the advected energy flux is the
  internal-energy flux $E_s u$, not the enthalpy flux $(E_s+p_s)u$ — the missing
  $p_s u$ is added back explicitly. (R2's `hyperbolic_energy_consistent` folds it
  into a KEP $-u\,dM_{\text{press}}$ form instead.)
- $-m_i\mathbf{u}\,S_{iz}$ (momentum, ion loading) — because $M = m_i n u$,
  expanding $\partial_t M + \nabla\!\cdot(Mu)$ produces
  $m_i\mathbf{u}\,[\partial_t n + \nabla\!\cdot(n u)] = m_i\mathbf{u}\,S_n$;
  moving it to the RHS gives the mass-loading drag. It is a bookkeeping
  consequence of evolving a conservative momentum density over a source-carrying
  continuity, not a distinct force.
- $-\nabla p$ (momentum) — stays inside the flux as the $+p$ of $M u + p$; it is
  not discretized as a separate gradient stencil.

Two numerical consequences follow directly. First, convective transport and
compression cannot be tuned apart: they share one flux and therefore the same
Rusanov numerical diffusion $\sim\tfrac12 a_{\max}\Delta z$. Second, a reflecting
wall zeroes the convective half only — the closed face carries
$F_n = F_{Ee} = F_{Ei} = 0$ but keeps $F_M = p_{\text{live}}$ — the discrete
statement of "no flux through the wall, but the wall still pushes back"
(`flux._apply_plasma_walls`).

## Reductions relative to full 3D Braginskii

Dropped in this model: ion viscosity/stress $\nabla\!\cdot\pi$, the $\mathbf{E}$
and $\mathbf{u}\times\mathbf{B}$ forces and diamagnetic/drift heat fluxes, and
perpendicular conduction — only the parallel (axial) dynamics are retained. Wall
and end losses (surface neutralization, gas puff, pumping) are folded in as 0D
boundary-cell source terms rather than bulk 3D terms.

### Radial recycling proxy (REMOVED R3b, 2026-08-20) — the closure rationale of record

For a period the model carried `radial_recycling_rhs`, a deliberate stand-in
for the radial physics the reduction above discards. Because no radial
coordinate is resolved, radial confinement and the wall recycling it drives
cannot emerge; the term imposed them through a single knob $\tau_s$. Plasma
was lost at $-n/\tau_s$, the wall neutralized it, and the neutral returned
*locally* as cold gas: per cell, with $S = n/\tau_s$, the plasma channel gave
up particles, momentum **and** thermal energy — the wall kept all three, so
this was a radial energy loss channel too — while the neutral inventory
gained $S\,V_p/V_m$. Total particle inventory was conserved exactly, and
because the returned gas was cold no energy came back.

Its motivation was the mid-column neutral burnout canyon, which has no refill
channel in this model: the physical refill — wall recycling of radially-lost
plasma, a *distributed* neutral source — is radial. $\tau_s$ was therefore a
**calibrated** quantity, not one derived from anything else in the model, and
any result using it had to say so. Its honesty test: LAPD radial confinement
is of order 5–25 ms, so a fitted $\tau_s$ in the low-ms range is plausible
compensation and anything far outside it is a documented failure.

The term was default-off ($\tau_s =$ `None`) and the golden baseline never
exercised it; the implicit neutral-only step omitted it deliberately, since
that step runs only before plasma launch where $n$ sits on its floor. It was
deleted as unreachable code — no caller, no gating flag, no config key — at
commit `30f6af1`. This paragraph is the surviving record of the closure
choice; the implementation is recoverable from git history.

## A9 classical electron heat flux — RETAIN + limiter gate (audit 2026-07-23)

The parallel electron conduction above uses the classical Spitzer–Härm
$\kappa_{\parallel e} \propto T_e^{5/2}$ closure, and the whole-model audit
retained it **only under a standing gate**: at resolved gap faces the
unbounded Spitzer–Härm flux reaches 1.7–3.3× $n\,T_e\,v_{th,e}$ — above the
free-streaming scale a physically saturated flux must respect (electron mean
free path medians ~23–24 cm after settling, p95 ~42 cm, against
comparable-scale gradients). **Time-integration stability does not make a
constitutive law valid**: the implicit TR-BDF2 substep being well tested is
not evidence for the closure. The gate: any port-level or
boundary-power-transfer claim that leans on conduction must carry a
nonlocal/flux-limited closure **bracket**. The limited arm exists as the
`electron_heat_flux_limit` flag with `heat_flux_limiter_f` (free-streaming
fraction, $q_{sat} = f\,n\,T_e\,v_{th,e}$) and `heat_flux_limiter_exponent`
— see `core/config.py` for the authoritative semantics. Reference: Cowie &
McKee, ApJ 211 (1977) 135. *(The companion audit gate A11, on the
fluid↔circuit coupling, is stated in the R3 section below.)*

## R1 audited topology and configuration contract (2026-07-23)

Typed geometry now owns one authoritative plasma topology:
`plasma_active[cell]` and `plasma_face_live_cell[face]`. The live
`active_plasma_topology` stance uses that map at the assembled-operator
boundary. Plasma-dead plenum/obstruction rows are invariant,
plasma-coupled source rows and diagnostic reductions are zero there, timestep
bounds exclude them, and a closed internal face takes velocity/pressure only
from its live side. Pure neutral transport remains active in those volumes.
The unchanged checkpoint golden explicitly pins the historical selector-off
path and remains bit-exact; it is a regression anchor, not the live stance.

The repaired startup defaults are `Te0=0.21 eV` and provisional
`Ti0=0.125 eV`, both strictly above their unchanged `0.1 eV` numerical
floors. The electron seed is just above the exact bundled He ADF11 lower edge
(`0.200092... eV`). The ion seed is a numerical margin, not a neutral
temperature claim; the model still separately uses `Tn_K=300 K` and the
audited `Tn_fit=0.1 eV` collision temperature pending the A8/R4 repair.

Optional neutral states use their actual packed layout throughout evidence:
five rows for `(n, nn, M, Ee, Ei)`, then optional `M_n`, `nn_a`, `M_n_a`, and
`En`.
For two-zone runs the column and annulus inventories are
`nn*V_col + nn_a*V_ann`, with `V_col=V_p` and `V_ann=V_m-V_p`; `nn*V_m` is
never reported as a two-zone inventory. The same volume split applies to the
two neutral-momentum rows, and internal radial/zone transfers close exactly.

Configuration is resolved once at construction from the shared registry.
Unknown keys raise `ValueError`; `config_manifest()` exposes all 248 parameter
defaults and 47 flags with their defining groups (count current as of the
D3 deletion pass, 2026-08-21). The live but formerly
unregistered controls `b_anode_collection` and
`b_anode_advective_block` are registered at their pre-audit fallback
values; `drag_dt_fraction` was registered alongside them and then deleted
at D3 (2026-08-21), where it became the fixed `timestep.DRAG_DT_FRACTION`.
The disconnected source/end absorption enables/scales, deleted at D3, and the
compatibility-only `front_flux_model`, `D_amb_model`, `D_amb`, and
`cathode_model` accept only their checkpoint values; noncanonical values fail
at construction rather than acting as silent no-ops. Their replacement
operators belong to R2/R3, not R1.

The historical shared electron-birth default remains
`Te_birth_ionization="local"`, which is what the committed production golden
runs since the R2b re-anchor — inherited from the stance of record (which names
it) rather than pinned in the fixture. The config-complete M6 and mechanism-ladder
drivers now explicitly select `"floor"` so they cannot accidentally inherit
`"local"`. This is provenance repair, not endorsement of either physical
moment: the unified ionization particle/momentum/energy derivation remains R4.

Every saved result now carries an `atomic_rate_domain` ledger derived from the
actual bundled ADF11 grid. It records active-cell and active-volume fractions
below the lower edge, the active minimum `Te`, and first whole-run and
afterglow crossings. This is a validity annotation: rates still follow their
selected clamping/extension policy, and a below-grid sample is not silently
promoted to validated atomic physics.

## R2 conservative hyperbolic core (2026-07-24)

The default-off `hyperbolic_energy_consistent` selector makes the discrete
hyperbolic operator conserve the total plasma energy `K + Ee + Ei` (kinetic
plus electron and ion internal) to machine precision on a closed domain. It
combines three compatible pieces:

- a kinetic-energy-preserving convective momentum flux (the divergence-form
  `0.5(M_L u_L + M_R u_R)` becomes `{u}{M}`, Jameson 2008);
- deposit of the Rusanov `(n, M)` numerical kinetic-energy dissipation into the
  ion internal energy `Ei`. This is a numerical (scheme) viscosity, distinct
  from the physical ion-neutral drag (A7/R4); the model carries no physical ion
  viscous stress $\nabla\!\cdot\pi$.
- a kinetic-energy-preserving pressure-work discretization, `-u dM_press` per
  species, replacing `-p_s \nabla\!\cdot u`.

The signal speed used by the Rusanov dissipation and the CFL is the exact
linear acoustic speed of the implemented $\gamma=5/3$ two-species energy
system, `c = sqrt((5/3)(Te+Ti)/m_i)` (selector
`hyperbolic_wave_speed="adiabatic"`); the historical `sqrt(Te/m_i)`
under-bounded it. The sonic front-filling flux is retired from the repaired
stance (a mesh-vanishing diffusion). All three are default-off and the
checkpoint golden is bit-exact; the deliberate repaired stance is selected only
after the R3/R4 boundary and source ledgers close.

## R3 characteristic material boundaries + closed circuit power balance (2026-07-24)

The default-off `characteristic_boundary` selector rebuilds the plasma-terminating
(cathode/collector) surfaces as one control surface feeding both the fluid sink
and the circuit. Default-off is golden bit-exact; the audit findings are A1, A13,
A16 (plus the A11 convergence gate).

**R3.1 -- characteristic ghost-cell Bohm outflow (A1).** The closed reflecting
face plus one-sided volumetric absorber is replaced by a ghost-cell Bohm outflow
computed with the committed R2 KEP/Rusanov flux against a ghost state
(`n_se = n*presheath_alpha`, `u = c_s` into the wall, `Te`, `Ti`). The flux is
applied one-sidedly to the live cell (the shared face array would telescope the
removed plasma into the dead plenum), the advective flux carries nothing at those
faces (the ghost flux supplies `M u + p`), and the neutral return + cathode jet
are booked as before. This fixes the A1 wrong-sign momentum: the edge establishes
`u -> c_s` into the wall (a net energy sink) instead of the historical `+18.5 kW`
reconstructed-kinetic source.

**R3.2 -- one control-surface power balance (A16).** Electrode energy is booked to
three distinct sources: (1) the *circuit field-work* book -- the sheath-fall `phi`
and work function, sourced from the bank maintaining the potentials against the
loop current, deposited on the electrode, never through the plasma thermal store;
(2) the *plasma-thermal* book -- `2 Te` per electron and `Te/2` per ion (the
boxed transmission coefficients `gamma_e = 2 + phi/Te`, `gamma_i = 1/2 + phi/Te`,
Stangeby; the existing `_P_elec`/`_P_ion` forms), from the plasma; (3) the *plasma-
heating* book -- the beam `P_prim` and gap ohmic, into the plasma. The fluid
boundary removes only the plasma-thermal part; the sheath fall goes to the
electrode; both the fluid sink and the circuit read one mesh-independent
sheath-edge density `n_se` (`presheath_alpha`, shared; the circuit's flat
`exp(-1/2)` is upgraded, with the electron lift generalized `Lambda -> Lambda -
ln(alpha)`; the anode mesh keeps `exp(-1/2)`). The collector gains its own
floating `2 Te` electron sheath (previously absent). The load-power balance
`P_load = I_tot V_b = cathode + gap + anode field work` closes to machine
precision via the cathode Kirchhoff `I_tot = I_eth* + I_i - I_e_ret` (returning
electrons with a minus); the never-closing `P_net`/`P_net2` scalars are demoted to
deprecated. Measurement-plane aliases (`V_dis = V_b + V_series`, `I_bank =
I_plasma + I_parallel`, divergences zero now) set up the future effective-load
work (items 24/25). Honest limits: the anode uses the net-current ladder
(per-species anode is A15/R4), and the bracket-capped regime clamps `V_b` off the
ladder (a reported residual). The identity asserted here is not taken on faith:
`scripts/verify_sim1d_r3_routing.py` gate G6 evaluates the residual
`P_load - (cathode + gap + anode field work)` directly against the solver's own
terms and requires it below `1e-6 * P_load`, alongside the cathode Kirchhoff sum
the derivation rests on.

**R3.3 -- A13 controls deprecated.** The resolved-boundary surface-loss area
scales and per-face enables were 0D artifacts (they stood in for I_sat that the
lumped model could not separate between cathode and anode); the resolved geometry
measures the Bohm I_sat to each electrode face directly, so they are retired
(loud `DeprecationWarning` on non-default use), not wired.

**A11 (convergence gate, deferred).** The fluid SSPRK stages run at a loop current
frozen over the step; the fixed-`dt` refinement gate (`verify_sim1d_r3_a11.py`)
shows the coupled sheath observables `V_b`/`phi_c` do not converge cleanly at the
emission knee. The fix (a gated fluid<->circuit Picard iteration) is deferred to
R5's coupled-circuit-convergence validation.

## R4.1 anode-mesh beam interception (2026-07-24)

The `beam_anode_interception` selector adds the missing anode-mesh interception
event to the CSDA beam ray (audit A15). It is the correct csda physics, so it is
the **production default (on)**; like `beam_coulomb_model` / `beam_anomalous_model`
it is a csda control -- inert under `beam_deposition_model="beer_lambert"` (which
never launches the CSDA module) and where the resolved geometry has no anode faces.
The golden fixture runs it ON since the R2b re-anchor recaptured that fixture at
the stance of record, which inherits this default. Set it `False` for the
with/without-interception A/B. Without it the CSDA adapter launches the full emitted flux
`Gamma0 = I_eth_star/e` through the whole column, so the fluid deposits the entire
emitted beam (~470 kW on the settled artifact) while the current-driven circuit
books only `P_prim = (1 - eta*beam_bypass_fraction) * I_eth_star * phi_c` into the
plasma (~307 kW) plus the bypass power `eta*beam_bypass_fraction*I_eth_star*V_b` on
the anode. The ~164 kW difference is the long-mean-free-path beam the anode mesh
intercepts, which the fluid was wrongly depositing downstream.

With the selector on, `deposit_beam` carries a running flux `gamma` (initially
`Gamma0`); at the anode-face crossing the mesh solid fraction `eta` of the flux
still streaming there is intercepted -- booked to the anode surface
(`anode_intercepted_erg_s`), NOT to `plasma_heating_erg_s` -- and only `(1 - eta)`
transmits downstream, carrying the reduced flux through all subsequent deposition
and ionization. A ray that stops in the gap never reaches the face and intercepts
nothing, so exactly the survived (bypass) fraction is removed, consistent with the
circuit's `eta*beam_bypass_fraction`. Per-ray energy still closes to roundoff:

$$\Gamma_0 E_0 = \text{heating} + \text{radiated} + \text{cost} + \text{anode-intercepted} + \gamma_{\text{exit}} E_{\text{exit}}.$$

Active under `beam_deposition_model="csda"` with resolved anode faces; inert
otherwise (no construction error -- a csda control, like the beam Coulomb/anomalous
selectors). Production default on, and on in the golden fixture too since the
R2b re-anchor (`baseline_sim1d.py`). This removes the
+164 kW item-21 anode-interception error; the paired +43.1 kW ionization birth
energy (A14) is R4.2.

`anode_intercepted_erg_s` is the beam energy *removed from the plasma* (the launch
energy `phi_c` the fluid was over-depositing downstream), NOT the anode heat: the
intercepted electron decelerates through the anode sheath `phi_a` and strikes the
mesh with its arrival KE `V_b = phi_c + V_p - phi_a`, so the anode *surface* takes
`I_bypass*V_b` while `I_bypass*phi_a` returns to the anode-sheath field (circuit).
The total anode power loss is already booked by the current-driven circuit at `V_b`
(`_P_beam_bypass`, inside the R3.2 closed ledger `P_load = I_tot*V_b`); this term
only ensures the fluid stops depositing that beam into the column. The
plasma-removed / anode-heat / sheath split (settled artifact: 163.56 = 123.26 +
40.43 kW, residual ~0.1 kW = `V_p`) is what a scheme-efficiency analysis
(plasma-useful vs electrode-lost power) across I/V-ratio operating points will use.

## R4.2 unified ionization birth energy moments (2026-07-24)

The default-off `ionization_birth_energy_model` selector re-derives the bulk (and
beam) ionization birth energy moments from one particle/momentum/energy balance
(audit A14). Under the historical `"legacy"` booking the bulk electron birth adds
$\tfrac32 T_{e,\text{birth}} S_{iz}$ to $E_e$; with `Te_birth_ionization="local"`
this creates $\tfrac32 T_e$ of thermal energy per new electron (+43.1 kW on the
settled artifact), cancelling 92% of the ionization-potential cost $I_\text{ion}
S_{iz}$ -- unphysical, since a newly ionized electron carries no kinetic energy.

Under `"conservative"` the electron birth energy is **zero** (reconciled to the
beam's standing $E_e=0$ convention): the new electron is born cold, so $T_e =
E_e/\tfrac32 n$ falls by dilution as $n$ rises, on top of the unchanged potential
($-I_\text{ion} S_{iz}$) and radiation sinks. The ion mass-loading **mixing
energy** is booked explicitly to $E_i$: a neutral consumed at drift $u_n$ becomes
an ion that joins the bulk flow at $u_i$, dissipating

$$Q_\text{mix} = \tfrac12 m_i\,(u_i - u_n)^2\,S_{iz}$$

per unit volume. With the reconstructed bulk kinetic change $dK = u_i\,dM -
\tfrac12 m_i u_i^2\,dn$ (from the birth momentum $dM = m_i u_n S_{iz}$ and
$dn = S_{iz}$), the ion total energy then closes to the consumed neutral's energy,

$$dE_i + dK = \tfrac32 T_{i,\text{birth}} S_{iz} + \tfrac12 m_i u_n^2 S_{iz},$$

to machine precision -- the drift energy that `"legacy"` lost through the bulk
kinetic derivative is retained as ion heat. The beam ion birth (electron already
$E_e=0$) gains the same $Q_\text{mix}$ with its own $u_n$ (the two-momentum column
wind, or zero for the historical rest-birth). Under `"conservative"` the
`Te_birth_ionization` selector is inert (the electron birth energy is physically
zero regardless). Default `"legacy"` keeps the golden bit-exact.

With A15 (R4.1) and A14 (R4.2) both booked, the two item-21 structural errors
(+164 kW anode interception, +43.1 kW electron birth) that partially concealed one
another on the settled artifact are each in their correct book; the item-21 power
ledger can now be re-checked on that artifact. The `Te_birth_ionization=local`
caveat in the energy-source glossary above applies only to the `"legacy"` booking.

## R4.3 moment-closed ion-neutral collision operator (2026-07-25)

The default-off `ion_neutral_moment_closure` flag replaces the drag +
$Q_{\text{fric}}$ + $Q_{\text{eq,el}}$ + $Q_{cx}$ quartet with ONE moment-closed
reduced ion-neutral collision operator (audit A7 + A8). The legacy
`sigma_in_model="cx_derived"` arm (removed at D3, 2026-08-21) applied
$2\langle\sigma v\rangle_{cx}+k_L$ directly
as the lab drag frequency -- dropping the equal-mass reduced-mass factor
$\mu/m_i=\tfrac12$, so the CX drag was doubled -- and split the energy into $Q_{cx}$
(coefficient $1\,K_{cx}$) plus an elastic $Q_{\text{fric}}/Q_{\text{eq,el}}$ at
$\nu_{el}=\nu_{in}-\nu_{cx}=n_n(K_{cx}+K_L)$, mislabelling a full extra CX-rate
equivalent as elastic. The present thermal cooling was $1.51$--$2.02\times$ the
reduced-operator bracket, and the CX-sized frictional-heating residual the exact
swap moment requires was dropped.

**Rates (Phelps He$^+$/He, boxed literature).** The reduced operator uses the
Phelps database He$^+$-in-He isotropic + backscatter cross sections (Phelps
database, www.lxcat.net, retrieved on July 25, 2026; see
`funcs/_cross.py` for the full citation and `vars/he_ion_neutral_phelps_lxcat.txt`
for the archived download). They map to $\sigma_{cx}=Q_b$ (backscatter = charge
exchange) and $\sigma_{mt}=Q_i+2Q_b$ (momentum transfer: isotropic contributes
$\int(1-\cos\theta)=Q_i$, backscatter $180^\circ$ contributes $2Q_b$), and are
valid to thermal energies (no 0.1 eV clamp, unlike the IAEA `A_R531` table). The
isotropic rate is velocity-independent ($Q_i\propto E^{-1/2}$) and equals the
classic Langevin capture rate to 0.1% -- the ad-hoc additive Langevin was a good
proxy, now rigorous.

**Operator (equal mass).** With $k_b(T_\text{eff})=\langle Q_b v_\text{rel}\rangle$
and $k_\text{iso}(T_\text{eff})=\langle Q_i v_\text{rel}\rangle$ Maxwellian-averaged
at $T_\text{eff}=(T_i+T_n)/2$ (reduced mass $\mu=m_i/2$), both channels reduce to
the same Braginskii form through a single momentum-transfer frequency

$$\nu_{mt} = n_n\big[\,k_b(T_\text{eff}) + \tfrac12 k_\text{iso}(T_\text{eff})\,\big]$$

(the $\tfrac12$ on $k_\text{iso}$ and the CX $2Q_b\!\to\!k_b$ are the equal-mass
$\mu/m_i$ factors). That one frequency governs momentum, frictional heating, and
thermal equilibration:

$$\frac{dM}{dt} = -m_i\,n\,\nu_{mt}\,(u-u_n),\qquad \frac{dE_i}{dt} = \tfrac12 m_i\,n\,\nu_{mt}\,(u-u_n)^2 + \tfrac32\,n\,\nu_{mt}\,(T_n-T_i).$$

The neutral takes the exact mirror momentum source ($M_n$ through the plasma/neutral
volume ratio when the state carries it), so ion-neutral momentum exchange is
antisymmetric. The frictional term at the full $\nu_{mt}$ carries the CX-sized
residual the swap moment requires (it is not restricted to the elastic fraction).
At zero drift the operator is pure equilibration $\tfrac32 n\,\nu_{mt}(T_n-T_i)$ --
the CX thermal coefficient is $\tfrac32 K_{cx}$, ending the legacy $2.5\,K_{cx}$
double-count. Without an evolved neutral energy the neutral-side collisional
energy is dropped (as before); with the `neutral_energy` flag it is booked, and
the CX share of it is re-routed -- see the two-channel section below.

**A8 (neutral temperature).** The single cold-gas $T_n$ is the 300 K feed/wall
temperature (`Tn_K`), used consistently in both $(T_n-T_i)$ and $T_\text{eff}$; the
legacy `Tn_fit`$=0.1$ eV is not consulted on this path, ending the term-by-term
$T_{n,K}/T_{n,\text{fit}}$ mix.

Presence-gated: when on, the four legacy ion-neutral terms return zero and this
single term runs; when off it is a strict no-op (golden bit-exact). He-only (loud
`ValueError` at construction otherwise). The legacy `sigma_in_model` arms
(`"constant"`, `"cx_derived"`) were removed at D3, 2026-08-21: they were the
solver's only non-helium path, and `"phelps"` is now the sole accepted value.
Analytic identities
(`verify_sim1d_r4_collision.py`: momentum antisymmetry, friction $=-\tfrac12
u_\text{rel}\,dM$, zero-drift thermal) hold to roundoff. On the settled matched-M6
artifact (`probe_sim1d_r4_collision_bracket.py`) the operator's thermal cooling is
$-28.2$ kW -- inside the IAEA-based pre-registration bracket $[-30.40,-22.67]$ kW
and reduced from the present $-46.0$ kW, with no rate tuning (Phelps supersedes the
IAEA rate set; the bracket is a cross-check).


## Decoupled two-channel neutral gas (`neutral_energy`)

The neutral population in the column is bimodal. A cold bulk sits near the
vessel temperature; a minority born by resonant charge exchange sits at the
local ion temperature. In the **ionization-depleted discharge column**
($n_n \lesssim 10^{12}\,\mathrm{cm^{-3}}$, the state this channel was built
for) the neutral-neutral mean free path is orders of magnitude longer than the
column radius, so the two populations are **collisionally decoupled**: the hot
population's (much larger) partial pressure must never appear as a force on
the cold fluid.

**Regime limit (disclosed).** That separation belongs to the depleted column,
not to the machine. At FILL density it does not hold: $n_n \approx 2\times
10^{13}\,\mathrm{cm^{-3}}$ gives a He--He momentum-transfer mean free path of
$\approx 24.5$ cm against $R_p = 15$ cm, i.e. $\mathrm{mfp}/R_p \approx 1.6$ --
the same order, not orders of magnitude. Wherever the column density stays
near fill (before breakdown, and in any cell the discharge never depletes) the
hot channel carries an un-modeled elastic leak of $O(R_p/\mathrm{mfp})$ per
crossing, moving hot momentum and energy into the cold field. The sign is
hot-forces-cold: the true cold fluid would be pushed and heated slightly more
than the decoupled treatment allows. This bounds the closure's validity range;
it is not a defect inside that range.

**Cold channel.** A fluid with its own modest pressure $p_n = n_n k T_n =
\tfrac23 E_n$, transported by the Rusanov mini-flux described in
[`NUMERICS.md`](NUMERICS.md). Every particle current states the energy it
carries: the Knudsen exchanges carry the DONOR cell's energy per atom
$E_n/n_n$, which is the choice that leaves an isothermal gas isothermal under a
pure density gradient; the puff arrives at the wall temperature; the pump,
ionization, and the CX swap remove gas at the local per-atom energy; and the
column/annulus exchange leaves at $T_n$ and returns at $T_\text{wall}$ (the
ratified annulus-cold v1 cut). Only the ELASTIC share of $\nu_{mt}$ heats it.

**Hot channel.** Algebraic -- no packed row. The CX share
$\nu_{cx} = n_n k_b(T_\text{eff})$ of $\nu_{mt}$ is a population swap, not a
collision that warms anything: the cold gas loses an atom carrying its own
energy and momentum, the hot channel gains one carrying the ion's
$\tfrac32 kT_i + \tfrac12 m u_\text{rel}^2$ and $m u_i$. The standing
population follows the saturating balance $f_\text{hot} = x/(1+x)$,
$x = \nu_{cx}\tau_\text{hot}$, with $\tau_\text{hot}$ a ballistic
column-radius crossing; the competing rates (re-CX, in-flight ionization) set
the branching. Nothing in it is fitted: $k_b$ is the Phelps LXCat backscatter
rate whose sum with the half isotropic-elastic rate IS $\nu_{mt}$, and
$\tau_\text{hot}$ is geometry.

**Transport is a ballistic redistribution kernel** (`physics/hot_neutrals.py`),
the same kinematics `KN2ZoneJump._fly` integrates over a discrete velocity grid
-- a radially determined chord, axial hop $dz = R_p\,\mu/\sqrt{1-\mu^2}$ --
evaluated analytically over an isotropic volume birth. Flights end on the
column boundary (mass moved axially, the CX-ballistic **erosion** that relieves
an axial pile; energy left on the wall), in re-CX (momentum and energy handed
to the ions where the flight got to -- the nonlocal CX-recycling channel), or
ionized in flight (a plasma source there).

**What closes and what does not.** Particles, momentum, and energy close across
ion + cold + hot to machine precision, with the wall as the ONLY named leak:
the landed atoms rejoin the cold gas at $T_\text{wall}$ and their excess energy
and their whole directed momentum are absorbed by the surface. Under the v1
cut the annulus carries no energy field, so only the $\alpha_E$ share of that
excess is accommodation in the physical sense and the remainder is the cut
itself; both are reported by the run's hot-channel diagnostics. Recombination-
born neutrals are Ti-class and so physically hot, but they stay COLD by
decision: the operator hands the recombined ion's directed momentum to $M_n$ as
an exact mirror, and splitting the particle from its momentum would break that.
Gates: `verify_sim1d_nbl2_neutral_transport.py`.


## Clumpy-plasma coverage closure v2 (`coverage_closure`, default off)

Breakdown in the machine is azimuthally patchy: discrete channels carry the
discharge, and a 1D mean-field solver azimuthally averages that structure away.
The closure restores the leading consequence of it with a coverage fraction
$f_\text{cov}(z,t)\in(0,1]$ -- the fraction of the column cross-section the
plasma occupies, **per axial cell**. Channel-local densities are then the mean
divided by $f_\text{cov}(z)$, and the remaining $1-f_\text{cov}(z)$ is a
neutral reservoir.

The geometry is **patches, not a connected channel**: $f_\text{cov}(z)$ is the
local broken-down area fraction at each $z$, with no axial-connectivity
requirement and no threshold. Coverage feeds the physics continuously, exactly
as the scalar did.

**The coverage field.** $f_\text{cov}$ obeys the logistic law with a
deposition-keyed local rate,

$$\frac{\partial f_\text{cov}(z)}{\partial t}
= r_0\,w(z,t)\,f_\text{cov}(z)\,\bigl(1-f_\text{cov}(z)\bigr),$$

with $r_0$ = `coverage_growth_rate_per_s`. There is deliberately **no decay or
drop-out term**; that richness is a pre-registered open question for the
ensemble leg, not something to patch in here.

$w(z,t)$ is the local beam-ionization rate normalized to its own
**volume-weighted column mean**,

$$w_i = \frac{S_i}{\bigl(\sum_j S_j V_j\bigr)\big/\bigl(\sum_j V_j\bigr)},$$

with $S$ the per-cell beam ion birth density (the `n` row of
`beam_ionization_birth`, already carrying the active-plasma mask) and $V$ the
plasma cell volumes, both sums running over the **plasma-active** cells. So
normalized, $\langle w\rangle_V = 1$ identically: the rescaling is
parameter-free, it introduces **no new constant**, and $r_0$ keeps exactly the
meaning it had in v1 -- the column-mean growth rate, and still the F2
calibration target. What $w$ does is redistribute growth in $z$, not change how
much growth there is on average. **Degenerate case, stated:** when the beam
deposits no ionization anywhere -- no cathode solve this evaluation, no
emission, or a ray that dies in the gap -- the mean is zero, $w$ is $0/0$, and
the answer is $w\equiv0$, i.e. no growth. That is the physical statement
(nothing is breaking the column down), not a numerical guard.

**Co-integration.** This law takes feedback from the state -- the deposition
that drives it is shaped by the coverage it drives -- so v1's closed form is
gone, exactly as v1's own documentation said a feedback v2 would require. The
field is carried as **accepted-step state** and advanced on the SSPRK2 stage
structure, using the discipline the neutral deficit $D$ already uses: every RHS
stage adds $\tfrac{dt}{2}\,w_\text{stage}$ to a per-attempt accumulator, the
accumulator is carried on the step attempt (not on the solver), and the field is
advanced once per ACCEPTED step from $\bar w = w_\text{accum}/dt$ by the exact
solution of the frozen-driver logistic,

$$f' = \Bigl[1 + \bigl(1/f - 1\bigr)\,e^{-r_0\bar w\,dt}\Bigr]^{-1}.$$

That advance is unconditionally positive, cannot leave $(0,1]$ at any $dt$,
reduces to $f$ identically wherever $\bar w$ or $r_0$ is zero, and returns
exactly $1.0$ wherever $f$ is already 1. A **rejected** attempt drops its
accumulator with the attempt and leaves the field untouched; a re-tried step
re-derives its own driver. The packed state vector is **not** widened -- the
field is an auxiliary per-cell array beside $D$, so nothing downstream of
`pack_state` changes.

Within a step's stages the field itself is held frozen (its DRIVER is what is
stage-accumulated), which is the same first-order treatment the DVM's scoped
transfer and the frozen-$\kappa$ Picard-0 path already carry.

**The two-stream march (v2).** The beam is emitted over the WHOLE cathode face,
so it does not all enter the patches: it divides by area. With $f_\text{cov}$
varying in $z$ the two media exchange cross-sectional territory as the ray
advances, so the partition cannot be made once at emission -- it is re-made at
every cell. One ray is marched per cathode end
(`deposit_beam_two_stream`), and in each cell $k$:

1. it enters with a total surviving primary flux $\Gamma_k$ at mean energy
   $E_k$;
2. the flux re-splits by the LOCAL coverage -- $f_\text{cov}(z_k)\Gamma_k$ into
   the channel medium $(n/f_\text{cov}(z_k),\,n_{n,c})$ and
   $(1-f_\text{cov}(z_k))\Gamma_k$ into the reservoir medium
   $(n_\text{floor},\,n_{n,r})$, on cross-sections $f_\text{cov}A$ and
   $(1-f_\text{cov})A$;
3. each arm marches that cell's path length in its own medium and banks into
   its own per-cell arrays -- **per-cell deposition is the sum of the two
   arms' deposits there**;
4. the arms re-mix at the cell exit into one stream:
   $\Gamma_{k+1}=\gamma_c'+\gamma_r'$ and
   $E_{k+1}=(\gamma_c'E_c'+\gamma_r'E_r')/\Gamma_{k+1}$.

The reservoir arm's plasma density is the model's own "no plasma"
representation, the density floor, because the closure's premise is that plasma
lives in the covered fraction.

**The stated approximation.** Re-mixing at every cell is the statement that the
breakdown patches **decorrelate axially on the cell scale**: an electron that
crossed cell $k$ inside a patch has no memory of that on entering cell $k+1$, so
the population is re-randomised over the local cross-section there. The re-mix
is a ONE-GROUP closure on that population -- it carries the flux-weighted mean
energy forward rather than two separate energies -- and it conserves flux and
power identically at every re-partition boundary, since $\Gamma_{k+1}E_{k+1}$ is
by construction $\gamma_c'E_c'+\gamma_r'E_r'$. **No flux or beam energy is
created or lost at a re-partition boundary**; what the closure discards is the
SPREAD of the primary energy distribution, which the CSDA model does not carry
in the first place.

Note what the split does to the quasilinear beam density: each arm carries its
share of the flux over the matching share of the area, so $n_b=\Gamma/(Av)$ in
both and equal to the mean-field value. That is the physical statement -- the
emitted beam is uniform over the cathode face, and it is the MEDIUM that
differs between the arms, not the beam.

**Uniform $f_\text{cov}$ does NOT reduce to v1.1, and that is expected.**
v1.1 partitioned the emitted flux once at the cathode face and let each arm
keep its own energy to the end of the column; v2 pulls both arms back to the
common mixed energy at every cell face. Those are different models even when
the profile is flat, so a flat-profile v2 run will not reproduce a v1.1 run and
must not be expected to. Two consequences are worth stating explicitly:

* The per-arm energy share is **no longer** $(1-f_\text{cov})\Gamma_0E_0$:
  flux migrates between the arms all the way down the column, so the reservoir
  arm carries $\sum_k (1-f_k)\Gamma_k\,\Delta E_{\text{res},k}$. The smoke
  suite's beam-split gate asserted the v1.1 per-arm identity; under v2 that one
  assertion is replaced by the TOTAL closure at the same tolerance (both arms
  plus the re-mixing carry exactly the emitted beam), with the per-cell re-mix
  budget -- which IS still exact under v2 -- gated separately.
* Gap survival is now genuinely **fractional**. A single-medium CSDA ray either
  crosses the cathode-anode gap whole or dies inside it; a two-stream ray can
  lose one arm in the gap while the other carries its share past. The ledger
  tripwire's "ray" view therefore reads the march's own record of the flux
  ENTERING each cell (recorded before any anode-mesh interception, which removes
  primaries rather than stopping them) instead of the binary breakout test.

**The exact reduction** is $f_\text{cov}\equiv1$ in every cell: there is then no
uncovered medium anywhere, the caller marches the shipped single-medium
`deposit_beam` rather than entering the two-stream march at all, every
concentration factor is multiplication by exactly 1.0, and the flag-off
trajectory is reproduced bit-for-bit. A profile that is 1 in SOME cells is
handled cell-wise: those cells give the reservoir arm zero cross-section and
zero flux, and cost exactly what a single-medium cell costs.

**Initial condition and the ensemble hook.** Exactly one of
`coverage_initial_fraction` (one uniform $f_{\text{cov}0}$) and
`coverage_initial_profile` (a per-cell $f_{\text{cov}0}(z)$ of length `nx`) must
be given with the flag on. They are two spellings of the same initial condition
and neither modifies the other, so supplying both is a construction-time
`ValueError` rather than a precedence rule the reader has to remember. There is
**no RNG inside the solver**: an ensemble is generated by building profiles
externally and passing them here, one run per realization.

**The walk closures run on the MEAN state.** The walking
`beam_product_transport` values (`"nonlocal"`, `"terminal_nonlocal"`)
(WP-D) and `heating_anomalous_transport="tail_walk"` (WP-E) withhold banks
during the march and walk them afterwards. The fused two-stream march has one
post-march walk stage and two media, so the question is which medium that stage
runs on -- and the answer the closure forces is **neither**: it runs on the mean
plasma state. Both arms' withheld banks are per-arm per-cell by construction, so
the birth LOCATIONS are the march's own; they feed the single walk stage
together.

The justification is the decorrelation statement above, applied to a product
rather than to a primary. A field-aligned product's path samples the channel
medium with probability $f_\text{cov}(z)$ per cell and the reservoir with
$1-f_\text{cov}(z)$, so the stopping it sees on average is

$$f_\text{cov}\frac{n}{f_\text{cov}} + (1-f_\text{cov})\,n_\text{floor}
  \;\simeq\; n,$$

the MEAN density: the concentration cancels, exactly as it cancels in a
volumetric bilinear rate. Nothing is added to make this work -- no constant, no
configuration key. In particular the **free-stream-to-walk transition is
emergent**, not imposed: the walk's reach is set by the same density-dependent
Coulomb blocking the primary's drag law uses, so at pedestal densities a tail
product crosses the machine (it *is* a free-streamer there) and localizes on its
own as the mean density builds behind the initial clumps.

Two second-order misattributions are accepted and stated rather than corrected.
They point in opposite directions and are bounded by the same patch-scale
argument that licenses the re-mix:

1. A product born INSIDE a channel is briefly correlated with that channel,
   whose density exceeds the mean, so it thermalizes somewhat more locally than
   the mean-field walk predicts.
2. A product born in the RESERVOIR truly sees the floor density, so its true
   reach is longer than the mean walk gives it. This one vanishes at both ends
   of the closure's own life: early, when the mean IS near the floor and the two
   media barely differ; and late, when $1-f_\text{cov}\to0$ and there are no
   reservoir births left to misplace.

The mean-state hand-off is structural, not conventional: the walks' hoisted
`stopping_coefficient` is built on the mean $n$ in the cathode path and is
REQUIRED by `deposit_beam_two_stream` once either walk is active, which holds
two media and no mean and therefore refuses to choose one silently. Without a
coverage view the mean array IS the array the rays march through, so every walk
arm without coverage is bit-exact.

**Burn attribution follows the same partition.** Under
`heating_anomalous_tail_ionization="on"` a walker also removes neutrals, and the
medium it removed them from is fixed by the identical sampling statement: at
cell $z$ its path lies inside a channel for a fraction $f_\text{cov}(z)$ of the
cross-section and inside the reservoir for $1-f_\text{cov}(z)$, so a fraction
$f_\text{cov}(z)$ of its per-cell ionization events debits the covered column
and $1-f_\text{cov}(z)$ debits the reservoir. This is the third application of
the one registered closure rule -- after the primary's re-mix and the walk
medium -- and it introduces no constant and no key: the split is expressed by
banking the walker's events into the two ARMS with those weights, and the
reservoir arm's ionization rows are already what the deficit equation reads as
$B_\text{res}$. The walk itself still runs on the mean medium, the births are
still booked to the mean fields under the interpretation caveat recorded above,
and the two misattribution bounds carry over unchanged -- they are statements
about this same sampling argument, not separate approximations. The walker's
energy rows are NOT split by those weights: the walk ran ONCE on the mean state
for both arms' births, so its end ledgers and tail-power scalars are booked
WHOLE to the channel arm with the reservoir arm carrying $0$ -- which likewise
leaves their sum, the only thing any consumer reads, exactly where it shipped.

**Where the concentration factors go.** The rule is that a factor appears only
where it does not cancel. A volumetric rate that is bilinear in a plasma and a
neutral density gains $1/f_\text{cov}$ locally but acts over the fraction
$f_\text{cov}$ of the cell, so its MEAN is unchanged -- which is why the bulk
reaction terms take no factor. What survives is everything that is *not*
linear in the local density:

1. **Beam stopping / deposition.** Each arm's attenuation is exponential in
   its own medium's densities, so the coverage changes the deposition PROFILE.
   The amplitude is untouched: the module returns per-cell TOTALS, and within
   each arm the $1/f_\text{cov}$ (or $1/(1-f_\text{cov})$) in the local flux
   density cancels the matching factor in that medium's volume, so the callers
   keep the historical conversion to a mean volumetric source. The closure
   REFUSES `beam_deposition_model="beer_lambert"`: that path has no second ray
   to give the reservoir, so the whole beam would go through the channels while
   the closure's own premise says only $f_\text{cov}$ of it does -- silently
   inconsistent rather than inert. It refuses `beam_clump_fraction > 0` for the
   same class of reason (two beam splits over different neutral media, whose
   four-ray product this build does not define).
2. **The discharge-current channel, where it is density-bilinear.** That is the
   sheath solve's coupling length, $1/\ell_b = 1/\ell_{bi}(n_e)+\sigma_b n_n$,
   which sets the gap bypass fraction the circuit books. It is reached
   INDIRECTLY, and deliberately so. The sheath solve reads one $n_e$ and spends
   it on two different things: that bilinear coupling length, and the LINEAR
   Bohm ion current $I_i = A_c e\,n\,c_s$ over the full cathode area. Only the
   first may be concentrated -- the channel density rises by $1/f_\text{cov}$
   over an area that shrinks by $f_\text{cov}$, so the coverage cancels
   identically in the second, and handing the solve a concentrated argument
   would silently inflate $I_i$ by $1/f_\text{cov}$. The solve therefore keeps
   the mean fields, and the coverage reaches the circuit through
   $\sigma_\text{eff}$: the CSDA adapter calibrates that effective attenuation
   cross section so the frozen mean-density Beer-Lambert bypass reproduces the
   transmission the CHANNEL ray measured. The inversion and the item-35 gap
   ledger are on the mean densities for the same reason -- they must be
   self-consistent with the solve they calibrate -- while the transmission they
   reproduce is the channel's. The same cancellation is why the anode sample
   and the sheath $\alpha$ take no factor.

   The transmission $\sigma_\text{eff}$ reproduces is the survival of the whole
   emitted beam -- both media together, measured by a gap-clipped run of the
   SAME two-stream march -- so the circuit's bypass is the real one and not one
   medium's.

At $f_\text{cov}=1$ in every cell each factor above is multiplication by exactly
1.0 and the model reduces to the shipped one bit-for-bit.

**The reservoir, and why the budget cannot open.** The mean neutral field $n_n$
keeps its exact meaning and its exact equations: **every existing term is
untouched, so $n_n$ still carries every particle and total inventory is
conserved identically.** What the closure adds is one auxiliary per-cell scalar,
the covered column's neutral DEFICIT $D = n_n - n_{n,c}$, from which the
channel density $n_{n,c}$ is read and the reservoir density follows
algebraically as $n_{n,r} = n_n + f_\text{cov}D/(1-f_\text{cov})$. The reservoir
is therefore represented IMPLICITLY, as the complement of the covered column
inside the conserved mean, and is never integrated. Its budget is a
re-partition of what the mean already holds, so no operation on $D$ -- including
the clip to the two positivity conditions $n_{n,c}\ge0$ and $n_{n,r}\ge0$,
i.e. $D\in[-(1-f_\text{cov})n_n/f_\text{cov},\,n_n]$ -- can create or destroy a
particle. $D$ is SIGNED: it is positive where the plasma burns column gas
faster than the reservoir refills it, and negative where the covered region is
a net neutral source (a recombining cold column returns neutrals into the
covered fraction alone and enriches it above the mean). Its lower bound closes
onto zero as $f_\text{cov}\to1$, where there is no reservoir left to donate.

$D$ evolves under two effects. The covered column absorbs the whole cell's
plasma-driven neutral debit $B$ but holds only the fraction $f_\text{cov}$ of
its volume, so its local density falls $1/f_\text{cov}$ times as fast as the
mean's; and the reservoir relaxes the difference back on `coverage_backfill_time_s`:

$$\frac{dD}{dt} = B\,\frac{1-f_\text{cov}}{f_\text{cov}} - \frac{D}{\tau_\text{backfill}}.$$

The relaxation term is the exchange
$f_\text{cov}(1-f_\text{cov})(n_{n,r}-n_{n,c})/\tau_\text{backfill}$ written
out: it reduces ALGEBRAICALLY to $(n_n-n_{n,c})/\tau_\text{backfill}$, so no
reservoir density is ever formed and the $f_\text{cov}\to1$ limit is regular
rather than a $0/0$. The reservoir arm of the beam split debits the OTHER medium, which lowers the
mean while leaving the covered column alone and so closes the gap between
them; the deficit equation therefore carries both debits, at different weights:

$$\frac{dD(z)}{dt} = -B_\text{cov}(z)\frac{1-f_\text{cov}(z)}{f_\text{cov}(z)}
+ B_\text{res}(z) - \frac{D(z)}{\tau_\text{backfill}}$$

(both debits negative on a burn, so the first term grows the deficit and the
second shrinks it). Under v2 every $f_\text{cov}$ above is the LOCAL one: the
algebra is unchanged and elementwise, including the positivity floor
$-(1-f_\text{cov}(z))n_n/f_\text{cov}(z)$, which closes onto 0 in cells that are
fully covered. $B_\text{cov}$ is the step-integrated sum of the neutral
rows of the COVERED-ONLY channels -- the terms whose rate is proportional to a
plasma or beam density, so the reaction can only happen where the plasma is:
ionization birth, the CHANNEL arm's share of beam ionization birth, both
recombination returns, and the gas-puff local ionization. $B_\text{res}$ is the
reservoir arm's share, which the beam terms publish as a side channel the RHS
ledger never sees. Terms that act uniformly across the cross-section (the puff,
the pump, neutral transport, the zone and kinetic exchanges) and terms that
transfer no particles (the ion-neutral collision operators) are deliberately
absent. It is accumulated across the SSPRK2 stages at the same equal stage
weights the DVM ionization booking uses, carried on the step attempt, and
applied once per ACCEPTED step, so a rejected attempt or a Picard re-run cannot
double-count it.

**The reservoir-born plasma caveat, stated rather than hidden.** The reservoir
arm's ionization births are booked into the MEAN plasma fields, conservatively
and with their full energy cost, exactly as the channel arm's are. But the
closure then READS those mean fields as covered plasma: the very next
concentration divides them by $f_\text{cov}(z)$ along with everything else. So
plasma born in the uncovered fraction is, one step later, treated as though it
lived in the patches. This is the honest cost of carrying one plasma field for
two media. v2 addresses **half** of the pair v1.1 named: $f_\text{cov}$ is now
driven by that birth rather than by an imposed autonomous law, so the coverage
grows where the reservoir is actually being ionized. Giving the reservoir its
own plasma field remains **open** and is the leading candidate for v3. The
caveat is bounded in the regime this closure targets, where the reservoir is
tenuous and its births are a small share of the total, and it is NOT bounded
once the reservoir carries an appreciable plasma density -- which is precisely
when $f_\text{cov}$ should have grown anyway.

**Limitations, stated rather than absorbed.** The bulk fluid reactions run
on the mean $n_n$, which is exact when the column is backfilled
($n_{n,c}=n_n$) and understates their depletion sensitivity otherwise. The
deficit is carried on the chamber-mean $n_n$ rather than on a column-resolved
partition, so $\tau_\text{backfill}$ absorbs the column-to-chamber exchange as
well as the azimuthal one. The coverage field is held frozen within a step's
stages (only its DRIVER is stage-accumulated), so the feedback between coverage
and deposition is resolved at first order in $dt$ within the step, the same
treatment the DVM's scoped transfer carries. There is no azimuthal transport of
coverage: $f_\text{cov}(z)$ cells are independent, and patch spreading in $z$
enters only through the deposition profile that drives them. The closure
requires `neutral_model="moment"` (the kinetic arms take over the fluid $n_n$
rows once engaged, so the column would never deplete and the backfill would be
a silent no-op). All are construction-time `ValueError`s.

**The compiled kernels are permitted under coverage.** v1 refused
`CABLP_COMPILED_KERNELS=1` while the closure was on. That refusal was written
against a belief about what the opt-in reaches, and the belief was wrong: the
compiled CSDA march is bound only inside `deposit_beam`, the SINGLE-MEDIUM
ray. `deposit_beam_two_stream` -- the closure's own two-medium wrapper, its
per-cell re-split, its flux/energy re-mix and every one of its banking
arrays -- has no compiled branch and runs in pure Python whether the opt-in is
set or not. What the flag actually accelerates under coverage is (i) the
nested single-medium walker marches the two-stream wrapper issues, which are
exactly the ray shape the `tierA+csda` transcription was bit-verified over,
and (ii) the tier-A cathode kernels, which the closure does not touch at all.
Both were measured raw-uint64 IDENTICAL over coverage trajectories (tail-walk
and ionizing-tail arms, flat and $z$-varying $f_\text{cov}$ seeds) before the
refusal was lifted, and bit-identity is now the standing guard in its place:
the smoke suite runs a beam-live coverage arm on both paths and asserts the
raw state bytes match.

## Ad-hoc probe neutral source (`neutral_probe_source`, default off)

An **inference instrument, not a physical closure and never a validation
channel.** An arm with this on answers one question -- *if there were this many
neutrals here, at this time, what would the plasma do?* -- so a result produced
under it reports the hypothesis alongside the response, and agreement with data
is not evidence for the source. Everything below is capability; how it is used
is registered separately.

The term is a volumetric particle source on the neutral density equation
(eq. 2), separable by construction,

$$S_\text{probe}(z, t) = A\,p(z)\,w(t)
\qquad [\text{cm}^{-3}\,\text{s}^{-1}],$$

carried as its own named RHS row, `neutral_probe_source`. Nothing else in the
model changes: the plasma, momentum and energy rows of the term are identically
zero, and no conservative field is added.

**Normalization, so the amplitude means one thing.** $p(z)$ is a dimensionless
shape sampled at the cell centres and rescaled so its chamber-volume-weighted
mean over the whole grid is exactly 1,

$$\frac{\sum_i p_i V_i}{\sum_i V_i} = 1,
\qquad V = \texttt{neutral\_volume\_cm3}.$$

The caller therefore supplies a shape, not a magnitude: its own scale divides
out, $A$ is the volume-mean source rate at $w = 1$, and the volume-integrated
influx is $A\,w(t)\sum_i V_i$ [particles/s] independently of the grid, of the
profile's normalization, and (under two zones) of which zone is fed. $p$ is
built either from an explicit per-cell profile -- the hook for an externally
computed hypothesis, since the solver contains no randomness and does no file
I/O -- or from the one built-in family, a gaussian in $z$. Exactly one of the
two: they are two spellings of the same object.

$w(t)$ is dimensionless, on the **absolute solver clock**. Three registered
forms: `const`; `square`, one on the half-open $[t_\text{on}, t_\text{off})$
with hard edges and no smoothing constant anywhere; and `table`, linear between
tabulated $(t, w)$ nodes and exactly zero outside their span.

**The stages consume the waveform's exact step average, and that is what makes
the delivered inventory the stated hypothesis.** The explicit step is Heun: it
samples the RHS pointwise at $t_0$ and $t_0+\Delta t$ and averages the two with
equal weights, so a pointwise waveform would be integrated by the **trapezoid
rule**. Across a hard edge that is not merely second-order but wrong by a
finite amount -- a step ending at a rising edge books $\tfrac12\Delta t$ of
source from *outside* the window, one ending at a falling edge loses the same,
and the two cancel only when those steps carry equal $\Delta t$, which adaptive
stepping does not arrange. Measured on this build before the fix: $-1.9\times
10^{-2}$ of the stated inventory on an off-lattice window, $+2.9\times 10^{-2}$
on an unequal-$\Delta t$ lattice. Because the probe term is state-independent
and separable, feeding both stages

$$\bar w = \frac{1}{\Delta t}\int_{t_0}^{t_0+\Delta t} w\,\mathrm{d}t$$

-- closed form for all three waveforms -- repairs this *identically*: the two
stages carry the same value, the $\tfrac12/\tfrac12$ combination returns it
unchanged, and each accepted step delivers $A\,p\int w\,\mathrm{d}t$ exactly,
for any $\Delta t$, any edge placement and any asymmetry between adjacent
steps. The window is threaded to the term as an explicit argument, so a
rejected trial $\Delta t$ cannot be read by the attempt that follows it.

Every hard edge is still registered as a step boundary, but for a different
and smaller reason: it keeps the **applied rate** the square that was asked
for. A step straddling an edge applies a partial-window average across its
whole width, which smears the edge in the plasma's *response* -- never in the
delivered total. A diagnostic read of the term (a save, or `rhs_terms` called
directly) reports the *instantaneous* rate at that instant, which is $\bar w$
for a window of zero width.

**Injection conventions, both inherited from the gas puff rather than invented
here.** *Zero net momentum*: the momentum rows are identically zero, so the gas
arrives at rest in the lab frame. Where a neutral wind is evolved this DILUTES
it -- $u_n = M_n/(m_n n_n)$ falls as $n_n$ rises at fixed $M_n$ -- which is the
physical content of injecting at rest, not a separate drag. *Temperature*: the
moment model carries one neutral temperature, `Tn_K`, and no neutral energy
equation, so injected particles join that single cold-gas population exactly as
puffed particles do. There is deliberately no probe temperature key; a distinct
injection temperature would be a new field, not a new parameter.

**Where it is live.** Wherever the explicit RHS is evaluated in a plasma run.
It is identically zero -- and recorded as zero, so the saved term structure is
stable -- whenever the solver is on the implicit neutral-only stepper (the
`Plasma` flag off, or the `neutral_prebreakdown` phase), whose backward-Euler
neutral matrix the term deliberately does not enter. A probe can therefore
neither fuel a pre-shot fill nor reach a cached neutral-equilibration seed.

**Two zones: an explicit choice, not a default.** Under `neutral_two_zone` the
run must name `neutral_probe_zone`, `"column"` ($n_n$) or `"annulus"`
($n_{n,a}$). The two put the gas in different places and the plasma responds to
them differently, which is precisely the thing a probe arm is measuring, so
there is no defensible default. The per-cell rate is formed on the chamber
volume and then re-normalized to the target zone, so the total influx is the
same number either way; cells with no annulus route to the column, as the puff
does.

**Coverage composes; it is not refused.** The clumpy-plasma closure partitions
the mean $n_n$ into a covered column and a reservoir through a deficit that
only the `COVERAGE_BURN_TERMS` move -- terms whose rate is set by a plasma or
beam density. The probe is not one of those: its rate is set by the caller and
it acts **uniformly across the cross-section**, exactly like the gas puff and
the pump, which that ledger already names as deliberately absent. So a probe
raises the covered column and the reservoir by the same amount, leaves the
deficit untouched, and the partition identity
$f_\text{cov}n_\text{col} + (1-f_\text{cov})n_\text{res} = n_n$ keeps closing.
The answer to *"does probe-injected inventory belong to the reservoir or the
column?"* is therefore neither-and-both, in area proportion, and it is forced
by the injection convention rather than chosen -- which is why this is an
allowance with a statement rather than a guess.

**Refusals.** v1 is the **moment neutral model only**: both kinetic arms take
over the fluid $n_n$ rows once engaged, so a source written into those rows
would be stripped or double-counted rather than felt, and the probe would
silently inject nothing. Injecting into a distribution function is a different
instrument, not a flag. Beyond that: every one of the ten parameters set with
the flag off; a missing amplitude, shape or waveform; both shape spellings at
once or neither; a negative or non-finite amplitude; a profile of the wrong
length, with a negative or non-finite entry, or identically zero (the null
control is amplitude 0, a different key); a non-positive gaussian width; a
waveform key belonging to another waveform; an empty or inverted square window;
a table with fewer than two rows, non-increasing times or a negative $w$; and
the zone selector present without two zones or absent with them. All are
construction-time `ValueError`s.

## Circuit voltage bound on the sheath ceiling (`cathode_circuit_voltage_bound`, default off)

The current-driven sheath solve carries a ceiling `cathode_phi_c_cap_V` on the
net cathode drop $\phi_c$. That number is the top of the tabulated He EII cross
section — an **atomic-data domain guard**, not a voltage the device sustains —
and in the `capability_limited` branch it was also used as a FLOOR on the
device voltage, `V_b = max(V_b, phi_c_cap_V)`. On the pre-breakdown build leg
that makes the solve report $V_b \approx 1000$ V and launch a $\sim$keV beam
while the bank supplies $\approx 178$ V (measured $V_b/V_\text{dis} \approx
5.1$ on 27–53 % of build-leg saves; 0 % at the plateau, which is 100 %
`classical`).

The flag composes a second upper bound with the cap. The circuit-available
device voltage is the loop equation the circuit already integrates,

$$L\,\frac{dI}{dt} = V_\text{src} - I\,(R_\text{comp} + R_\text{mesh}) - V_b,$$

read at $dI/dt = 0$: $V_\text{avail}(I) = V_\text{src} - I\,(R_\text{comp} +
R_\text{mesh})$. The ceiling the sheath root is solved against becomes
$\min(\phi_{c,\text{cap}}, V_\text{avail})$, so $\phi_c$ — and the beam birth
energy keyed to it through `_compute_l_b` — is held at the supply, and the
capability-limited $V_b$ is clamped there instead of floored at the cap. The
cap's own refusal machinery is untouched and still composes as the other upper
bound. The escape invariant (a returned $\phi_c$ above the ceiling in any other
regime is a `RuntimeError`) now asserts against the composed ceiling.

**The inductor's back-EMF is deliberately not counted as available voltage.**
It is stored energy, not supply, and including it would make the bound
vacuous.

**What the bound does NOT constrain is the loop current** (corrected
2026-08-12). The bound is a ceiling on the sheath and beam objects; the
circuit integrates the sheath's *unbounded demand* $V_b^{\text{unb}}(I)$ —
the device voltage the sheath would require to carry $I$, whether or not the
loop can source it. That asymmetry is what keeps a restoring force above the
capability wall. Feeding the bounded voltage into the loop equation instead
makes $V_\text{dis}(I) \equiv V_\text{src} - I R$ wherever the bound binds
(at $R_\text{mesh} = 0$), so the residual is identically zero there, $dI/dt
\ge 0$ everywhere, and the loop current becomes a **ratchet** whose value is
the running maximum of the TR stage's explicit overshoot — measured 156.7 A
after a single $2\times10^{-5}$ s step against a dt-converged 0.9 A, growing
with $\Delta t$ and with the save cadence. Unbounded, the demand climbs to
the data cap past the wall, $f$ goes sharply negative, and the wall is an
attractor: the same scenario is now dt-invariant to six figures from
$\Delta t = 10^{-8}$ s up to a single-step traverse.

The runaway backstop the floor was originally added for is therefore still in
place — more robustly, since it now rests on a real restoring force rather
than on a null residual — and the circuit's stage root-find keeps $g'(I) > 1$
strictly rather than collapsing to exactly 1.

### The bound's object

What the circuit supplies is the *device* voltage $V_b = \phi_c - \phi_a +
V_p$, in which the anode fall **subtracts**; $\phi_c$ is only its proxy.
`cathode_circuit_bound_object` chooses between them.

`"device_voltage"` (shipped) makes the circuit member of the composed ceiling
the net cathode drop $\phi_c^\star$ at which

$$V_b(\psi) = \phi_c(\psi) + V_p(\psi) - \phi_a(\psi) = V_\text{avail},$$

located by a bracketed solve on the same monotone device relation the current
root uses, with $\phi_a$ and $V_p$ evaluated by the identical expressions that
assemble the returned result — so the bound's object and the reported object
cannot drift apart. Everything downstream is unchanged: the composition is
still a `min` with the data cap, the ladder is the same ladder, the escape
invariant still asserts against the composed ceiling, the `bound_active`
census still says which member bound, and because the circuit's contribution
is still a $\phi_c$ number the compiled root path is entered exactly as
before. On the build leg $\phi_a$ is small and *negative* (it adds to $V_b$),
so $\phi_c^\star$ sits slightly **below** $V_\text{avail}$; at a plateau-class
point $\phi_a$ is a real positive fall and $\phi_c^\star = V_\text{avail} +
\phi_a - V_p$ sits **above** it. That is the whole of the difference.

`"phi_c"` makes the circuit member $V_\text{avail}$ itself. This is the R1
composition, bit for bit, retained as an A/B arm. Where $\phi_a$ is not
negligible it clamps a physically correct solve to $\phi_c = V_\text{avail}$
and tags it `capability_limited`, **raising nothing** — only `bound_active`
records it — and the sheath drop, and the beam birth energy keyed to it, come
back low by about $\phi_a$.

**SCOPE — a full-window run with the flag on is IN contract** (2026-08-12).
Both exclusions that once narrowed it are gone. The $\phi_c$/$V_b$ mismatch
went with the object above. The back-EMF exclusion went with the integrand:
it said that while the bound binds, $V_b$ is held at $V_\text{avail}$, the
loop residual is identically zero and $dI/dt = 0$, so on a **falling** leg —
the main-discharge decay, where the physical $V_b$ exceeds $V_\text{avail}$
*precisely because the inductor is supplying* — the bound would freeze the
current instead of letting it decay. That reasoning was sound about the code
as it then stood, and it is what the ratchet above was the acute form of.
With the circuit integrating the unbounded demand the residual is no longer
null under the clamp, so $dI/dt$ is free while the bound is active.

Measured, not merely argued: on the ON-probe build leg (`r1vb_run_probe.py
--bound on`, both objects, read by `regime_vcm_onprobe_read.py`) the loop
current **falls on 12 of 33 bound saves** and spans 1.10 A while bound — a
state the old contract excluded as impossible — with every R1 registered
assertion still holding ($V_b \le V_\text{avail}$ 33/33, zero ceiling
escapes, $V_b/V_\text{dis}$ median 1.0002). The inductor's stored energy is
still not counted as supply; that exclusion now costs the falling leg
nothing, because it constrains only the reported and beam-facing objects.
The one caveat on the evidence: the probe window stops at 0.033 ms and never
reaches the plateau decay, so the decay leg specifically is covered by the
loop equation rather than by observation, and the `bound_active` census is
still worth reading on a run that gets there.

**Diagnostics.** Three per-solve values ride the cathode diagnostics:
`phi_c_ceiling_V` (the ceiling actually solved against), `circuit_V_avail_V`
(NaN where the bound is not in force) and `bound_active` — 0 the solve is a
free root, 1 it sat on the data cap, 2 it sat on the circuit bound. All three
are NaN on the voltage-driven (floating) solve, which has no ceiling.

## Vessel / common-mode node (`regime_vessel_node`, default off)

The cathode/anode system **floats** with respect to the machine wall. The
whole electrically connected stainless vessel is ONE wall conductor, and the
anode is referenced to it only through four feedthrough capacitors bridging
the ceramic gap insulators, whose parallel sum is `vessel_capacitance_F`
($C_\text{total}$). The capacitor **type is visually unresolved** — axial
polypropylene film on the second look, aluminium electrolytic on the first —
so `vessel_leak_resistance_ohm` is finite, ESTIMATED over a bracket spanning
both readings (2.5e7–1e11 Ω, defaulting to the film reading) with the bracket
as the claim, and a bench measurement resolves it.

**The structural fact does not depend on the type.** At BOTH bracket edges
$\tau_\text{leak} = R_\text{leak}C_\text{total} \gtrsim 10\ \text{s}$ against a
~25 ms discharge, so **within a shot the node is hard-float in kind either
way** and nothing a run measures moves with the leak. The shipped leak is a
*symmetric* linear resistor; if the parts are electrolytic their reverse-bias
asymmetry is a documented deviation rather than a modelled one, and if they
are film there is no polarity nuance at all (see `NUMERICS.md`).

The flag adds ONE state variable, the anode-to-wall potential $V_\text{cm}$:

$$C_\text{total}\,\frac{dV_\text{cm}}{dt}
  = I_{e,\text{wall}} - I_{i,\text{wall}} - \frac{V_\text{cm}}{R_\text{leak}},$$

with $I_{e,\text{wall}}$ the CSDA rays' transmitted primary flux (the far end
IS the vessel) and $I_{i,\text{wall}}$ the column's ion loss read off the live
plasma-terminating boundary term at the collector cells. Electrons landing on
the wall raise $V_\text{cm}$; ions lower it; the steady state is the floating
condition, zero net system-to-wall current.

$V_\text{cm}$ is the potential the transmitted beam must **climb** from the
mesh into the column, so the energy reaching column physics is
$\max(\phi_c - \max(V_\text{cm},0),\,0)$ with $\phi_c$ the circuit-bounded
sheath drop — never the atomic-data cap, which is why the node requires
`cathode_circuit_voltage_bound`. The launched **flux** is untouched: the same
electrons arrive, decelerated.

That is the **ion-loss bootstrap**. The system self-biases until transmitted
electrons are decelerated climbing from mesh to column; with gas present,
ionization produces an ion wall flux, and the floating constraint then permits
an equal electron leakage into the column. Column seeding becomes ion-loss
throttled rather than emission throttled.

$V_\text{cm}(t)$ is saved as a **prediction channel** and is deliberately not
scored here: the qualitative shape observed on the machine — high early
positive bias, decaying as the bootstrap relaxes it, plateauing at either sign
at the main discharge — is the eventual comparison, and this build supplies
the trace, not the verdict. There are no tuned constants: both config values
are hardware quantities, and $C_\text{total}$ is ESTIMATED with the bracket as
the claim.

The numerical method — the closed-form step, the charge ledger and its
conservation statement, the single shared launch energy, the restart handling
and the reported phase sequence — is `NUMERICS.md`, section "Vessel
common-mode node".

## End-region recycle routing (`end_recycle_to_annulus`, default off)

**What it changes: where the end-recycled atoms are put, not how much plasma
the end takes.** The plasma-terminating faces neutralize the flux that reaches
them and rebirth it as gas. By default that gas is booked into the same cell's
**column** — the plasma channel itself — at column density, which places the
whole recycle stream directly in the path of the plasma it just left. Under
this flag the faces whose live cell has the **collector** role instead deposit
into that cell's **annulus**,

$$\left.\frac{\partial n_{n,a}}{\partial t}\right|_\text{recycle}
= \frac{\dot N_\text{loss}}{V_\text{ann}},
\qquad V_\text{ann} = V_\text{m} - V_\text{p},$$

as thermal diffuse gas, and their column row is correspondingly zero. The
**cathode** faces are untouched — the ratified jet/debit closure owns them, and
routing is a statement about the far end.

The physical picture is the end region as a plenum the plasma terminates
*into*, rather than a mirror that returns gas along the flux tube: an atom
neutralized on a collector surface leaves it with a cosine-law thermal
distribution into the whole end volume, not preferentially back down the
column. What the column then receives is set by free-molecular re-entry —
which the two-zone exchange conductance already computes — instead of being
imposed as a delta function on the last cells.

**Momentum.** The routed atoms carry none, on either $M_n$ or $M_{n,a}$. A
diffuse thermal re-emission has no directed flux by construction, and the
chamber-mean wind is left exactly as it was. The plasma-side rows — the density
sink, the sonic momentum debit, and the $E_e$/$E_i$ sinks — are bit-identical
to the unrouted term, on both discretizations.

**Energy, booked exactly once.** The recycled atoms are wall-temperature gas,
and the model has two places that could say so. The neutral-energy routing
table classes both boundary terms as `"wall"` sources, which turns their
**column** $n_n$ row into a $\tfrac32 k T_\text{wall}$ credit on the column
$E_n$; and the zone-exchange convention re-supplies wall-temperature enthalpy
whenever annulus gas re-enters the column. Under this flag the routed
particles leave the column row, so the first credit leaves with them, and the
second is the one that fires when the gas actually arrives. Booking both would
plant the same energy twice. The annulus itself carries no energy field, so
there is nothing to book there.

**Conservation.** The routing is a transfer between two rows of one term: the
volume-integrated rebirth rate $\sum_i (\dot n_{n,i} V_{\text{p},i} +
\dot n_{n,a,i} V_{\text{ann},i})$ is unchanged by it, and the recycle
throughput is conserved as before — nothing is created or destroyed, only
placed.

**What it refuses.** `neutral_two_zone`, since the destination row is that
closure's; and any geometry whose routed collector cell has no annulus
($V_\text{ann} = 0$), since the routing would then deposit into nothing. Both
are construction-time errors rather than silent fallbacks — the alternative is
a future end-region flare quietly destroying the stream at exactly the cells
the closure exists to describe. (This is the one place the routing differs
deliberately from the anode mesh, which falls back to the column in
annulus-free cells: the anode's annulus is incidental, the collector's is the
closure.)

The flag applies to whichever plasma-terminating discretization the run
configured — the R3.1 characteristic ghost-cell outflow or the volumetric
absorber — and is bit-exact when off, on both.

## Prescribed flux-tube and vessel geometry (`prescribed_area_geometry`, default off)

**What it changes: the cross-sections the whole 1D column is written on.** By
default the plasma occupies a straight tube of radius $R_p$ inside a straight
bore of radius $R_m$, so every cell has the same area $A = \pi R_p^2$ and every
area factor in the discretization cancels. Under this flag the caller supplies
a per-cell effective radius vector `plasma_radius_profile_cm`, one entry per
**mesh** cell, and the geometry is rebuilt on it:

$$A_i = \pi r_i^2,\qquad V_{\mathrm{p},i} = A_i\,\Delta z_i,\qquad
A_{i+1/2} = \tfrac12\left(A_i + A_{i+1}\right),$$

with the external faces taking their end cell's area. The quantity being
prescribed is the **area** — $A(z)$ is the flux-tube variable, $AB =
\mathrm{const}$ along a field line, so a solved $B_z(z)$ maps to
$A \propto 1/B_z$. It is *supplied* as the radius $\sqrt{A/\pi}$ because that
parameterization makes a constant profile at $R_p$ reproduce the uniform
column bit for bit rather than to within a round-trip rounding: the area is
rebuilt with the same `pi * Rp_cm**2` expression the uniform path uses. Any
conversion from a measured or solved field happens outside — the solver does
no file I/O.

**The vessel too, optionally.** `machine_radius_profile_cm` is the same kind
of vector for $R_m$, and is optional: omitted, every cell keeps the scalar bore
exactly as before. Supplied, it sets the neutral open area $\pi R_m(z)^2$, the
neutral cell volume, and the hydraulic radius that fixes the free-molecular
face conductance — so a bore that **steps** partway along a block of cells is
expressible, which neither a single $R_m$ nor `end_expansion_machine_radius_cm`
(one value over the whole terminal block) can express. It composes with the
annular-duct and support-rod reductions rather than overriding them: an
obstruction cell keeps $\pi(R_m(z)^2 - R_{cs}^2)$ and hydraulic radius
$R_m(z) - R_{cs}$, a plenum keeps $\pi(R_m(z)^2 - R_\text{sup}^2)$. Neutral
face areas stay *restricting* apertures, so a step is seen from upstream as the
narrow side, as it is at any other change of bore.

This pair is the replacement for the built-in half-cosine flare
(`end_expansion_geometry`), whose zero slope at *both* ends is wrong for a
convex solved profile and whose one vessel radius spans the whole terminal
block. The two are refused together: they prescribe the same quantities over
the same cells and there is no composition rule.

**Mirror force.** A varying $A$ makes the momentum equation quasi-1D,

$$\partial_t (A\rho u) + \partial_z\!\left[A(\rho u^2 + p)\right]
= p\,\frac{\partial A}{\partial z} + A\,S_M,$$

and the $p\,\partial_z A$ term is the mirror force. It is not an *additional*
closure on top of a $-\mu\nabla B$ force: the Maxwellian average of
$-\mu\nabla_\parallel B$ **is** this term at $A \propto 1/B$, so building both
would count the expansion twice. Discretely
(`sources.flux_tube_geometry_rhs`) it is

$$\left.\frac{\partial M_i}{\partial t}\right|_\text{geom}
= \frac{p_i A_{i+1/2} - p_i A_{i-1/2}}{V_{\mathrm{p},i}},$$

written with the same multiply-then-subtract ordering as the area-weighted
pressure flux it pairs with. That ordering is the whole point: for a
stationary uniform-pressure plasma the momentum face flux is exactly $p$, its
divergence is exactly $-(p A_{i+1/2} - p A_{i-1/2})/V_{\mathrm{p},i}$, and the
two cancel **bit for bit** rather than to round-off. A flare cannot
manufacture momentum out of a constant-pressure state — the well-balanced
property, and the one the smoke suite asserts as an exact array equality. The
exceptions are the plasma-*terminating* cells, where the characteristic
ghost-cell outflow supplies the face momentum instead of a reflecting wall
pressure, and a uniform stationary state is deliberately not the equilibrium.

**Pressure work, in both energy equations.** The compression partner
$-p_s\,\nabla\!\cdot\mathbf{u}$ is already written on the face areas,

$$\nabla\!\cdot\mathbf{u}\big|_i = \frac{A_{i+1/2}u_{i+1/2} -
A_{i-1/2}u_{i-1/2}}{V_{\mathrm{p},i}},$$

so it becomes $\partial_z(Au)/A$ the moment $A$ varies
(`sources.velocity_divergence`, consumed by `pressure_work_rhs` for $E_e$ and
$E_i$ alike, and by the R2 KEP correction). Expansion cooling through the
flare is therefore carried by the same term that carries compression heating
in a straight tube; there is no separate mirror-cooling source, and none
should be added.

**Neutrals and volumes.** The column zone *is* the plasma volume, so a varying
$A$ moves it, and the two-zone annulus volume $V_\text{ann} = V_\mathrm{m} -
V_\mathrm{p}$ moves with **both** profiles cell by cell — as does the zone
exchange conductance $\tfrac14\bar v\,2\pi r_i \Delta z_i$, which is the
column's lateral surface. All of these are read off the geometry rather than
recomputed, so the bookkeeping stays consistent by construction; what a
profile *can* break is $V_\text{ann}$ itself, in two distinct ways.

The first is the **sign**: a plasma radius past the local vessel open area
would make $V_\text{ann}$ negative, and it is clipped at zero, so the mistake
would be silent. That is refused at construction.

The second is **collapse**, which is subtler because the volume stays
positive. A flux tube that nearly fills the bore leaves an annulus of a few
hundred cm³ inside a cell of a few hundred thousand, and $V_\text{ann}$ is a
**divisor**: the zone exchange and the hot-channel deposit $\dot N_\text{land}
V_\mathrm{p}/V_\text{ann}$ both scale as $1/V_\text{ann}$, so a vanishing
annulus does not switch off — it stiffens the step without ever tripping the
`V_ann > 0` gates every consumer already carries. Two controls address it, and
they are complements rather than alternatives:

- `neutral_annulus_volume_fraction_min` refuses, at construction, any
  two-zone cell whose $V_\text{ann}/V_\mathrm{m}$ falls below it *while
  remaining nonzero*. Cells with no annulus at all are exempt — an absent zone
  is inert by the same gates. The shipped value sits an order of magnitude
  below a straight column (which leaves $\approx 0.86$) and an order of
  magnitude above the collapse it exists to catch.
- `plasma_area_max_vessel_fraction` caps the plasma area at a stated fraction
  of the local vessel open area. This is a **declared regularization**, not a
  geometry: a configuration that sets it is stating the cap as part of its
  closure, and it binds before the sign refusal, so a capped run cannot also
  trip that error. It is the intended way to run a solved flux tube that would
  otherwise fill the bore.

The end-recycle routing's own refusal (a routed collector cell with
$V_\text{ann} = 0$) remains the check for a flare that fills the vessel exactly
where the routing deposits; the fraction guard is the wider one, covering every
two-zone cell rather than only the routed ones.

**What it refuses**, all at construction: any of the three profile parameters
without the flag, or the flag without `plasma_radius_profile_cm` (either way
they would be inert); a profile whose length is not the mesh cell count — note
that is the *mesh*, which carries the plenum, gap and end cells as well as the
$n_x$ column cells; a non-finite, zero or negative entry in either profile (a
zero-area cell divides the flux divergence by a zero volume); a vessel radius
narrower than the plasma radius in the same cell; a plasma area past the local
vessel open area; a ceiling outside $(0, 1]$; a collapsed two-zone annulus;
and `end_expansion_geometry`.

The flag is default off and bit-exact off — with it clear no profile object is
built and the geometric momentum source is not even in the term ledger — and a
constant plasma profile at $R_p$ with a constant vessel profile at $R_m$ is
bit-identical to it being off. Because it changes the geometry it re-keys the
cached neutral-equilibration seed: $V_\mathrm{p}$, $V_\text{ann}$, the
conductances and the exchange are all read by the neutral-only equilibration,
so a seed equilibrated on one machine is simply wrong for another.

(`neutral_annulus_volume_fraction_min` is the one key here that is *not*
gated on the flag: it constrains any two-zone geometry, including one built
from the scalar radii, because the collapse it refuses is a property of the
volumes and not of how they were specified.)
