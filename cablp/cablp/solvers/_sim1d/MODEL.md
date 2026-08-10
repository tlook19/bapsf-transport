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
  $$\nu_{in}(T_i) = \frac{8}{3}\,n_n\,\sigma_{in}\,\sqrt{\frac{T_i}{\pi\,m_i}}, \qquad \sigma_{in} = 5\times10^{-15}\ \text{cm}^2.$$
  Toggled by the `ion_neutral_drag` flag and scaled by `b_ion_neutral_drag`;
  $\sigma_{in}$ is the `sigma_in_cm2` parameter. Setting the
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
five rows for `(n, nn, M, Ee, Ei)`, then optional `M_n`, `nn_a`, and `M_n_a`.
For two-zone runs the column and annulus inventories are
`nn*V_col + nn_a*V_ann`, with `V_col=V_p` and `V_ann=V_m-V_p`; `nn*V_m` is
never reported as a two-zone inventory. The same volume split applies to the
two neutral-momentum rows, and internal radial/zone transfers close exactly.

Configuration is resolved once at construction from the shared registry.
Unknown keys raise `ValueError`; `config_manifest()` exposes all 184 parameter
defaults and 29 flags with their defining groups. The live but formerly
unregistered controls `drag_dt_fraction`, `b_anode_collection`, and
`b_anode_advective_block` are now registered at their pre-audit fallback
values. The disconnected source/end absorption enables/scales and the
compatibility-only `front_flux_model`, `D_amb_model`, `D_amb`, and
`cathode_model` accept only their checkpoint values; noncanonical values fail
at construction rather than acting as silent no-ops. Their replacement
operators belong to R2/R3, not R1.

The historical shared electron-birth default remains
`Te_birth_ionization="local"`, and the committed production golden pins that
recorded stance explicitly. The config-complete M6 and mechanism-ladder
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
The historical csda checkpoint golden pins it off explicitly (`baseline_sim1d.py`,
the R1 pattern) so its pre-A15 trajectory stays bit-exact. Set it `False` for the
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
selectors). Production default on; the csda checkpoint golden pins it off
(`baseline_sim1d.py`) so its pre-A15 trajectory stays bit-exact. This removes the
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
`sigma_in_model="cx_derived"` applied $2\langle\sigma v\rangle_{cx}+k_L$ directly
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
double-count. The neutral carries no energy field, so the neutral-side collisional
energy is dropped (as before).

**A8 (neutral temperature).** The single cold-gas $T_n$ is the 300 K feed/wall
temperature (`Tn_K`), used consistently in both $(T_n-T_i)$ and $T_\text{eff}$; the
legacy `Tn_fit`$=0.1$ eV is not consulted on this path, ending the term-by-term
$T_{n,K}/T_{n,\text{fit}}$ mix.

Presence-gated: when on, the four legacy ion-neutral terms return zero and this
single term runs; when off it is a strict no-op (golden bit-exact). He-only (loud
`ValueError` at construction otherwise). The legacy `sigma_in_model` arms
(`"constant"`, `"cx_derived"`) remain live A/B instruments. Analytic identities
(`verify_sim1d_r4_collision.py`: momentum antisymmetry, friction $=-\tfrac12
u_\text{rel}\,dM$, zero-drift thermal) hold to roundoff. On the settled matched-M6
artifact (`probe_sim1d_r4_collision_bracket.py`) the operator's thermal cooling is
$-28.2$ kW -- inside the IAEA-based pre-registration bracket $[-30.40,-22.67]$ kW
and reduced from the present $-46.0$ kW, with no rate tuning (Phelps supersedes the
IAEA rate set; the bracket is a cross-check).


## Clumpy-plasma coverage closure v1 (`coverage_closure`, default off)

Breakdown in the machine is azimuthally patchy: discrete channels carry the
discharge, and a 1D mean-field solver azimuthally averages that structure away.
v1 restores the leading consequence of it with a single **scalar** coverage
fraction $f_\text{cov}(t)\in(0,1]$ -- the fraction of the column cross-section
the plasma occupies. Channel-local densities are then the mean divided by
$f_\text{cov}$, and the remaining $1-f_\text{cov}$ is a neutral reservoir.

**The coverage field.** $f_\text{cov}$ obeys the logistic law

$$\frac{df_\text{cov}}{dt} = r\,f_\text{cov}\,(1-f_\text{cov}),\qquad
f_\text{cov}(t_0)=f_{\text{cov}0}$$

from the plasma-phase time origin, with $r$ =
`coverage_growth_rate_per_s` and $f_{\text{cov}0}$ =
`coverage_initial_fraction`. There is deliberately **no decay or drop-out
term** in v1; that richness is a pre-registered open question for the ensemble
leg, not something to patch in here. Because the law is autonomous and takes no
feedback from the state, the solver evaluates its closed form
$f_\text{cov}(t)=\left[1+(1/f_{\text{cov}0}-1)e^{-r(t-t_0)}\right]^{-1}$ at each
stage time rather than co-integrating it: exact, stage-time consistent through
the same SSPRK2 stage clock the gas-puff waveform rides, and reproducible
across step retries and Picard re-runs with no snapshot state. **A v2 that adds
feedback or a decay channel loses that property and must move to genuine
co-integration.**

**The two-medium beam split (v1.1).** The beam is emitted over the WHOLE
cathode face, so it does not all enter the channels: it divides by area. The
fraction $f_\text{cov}$ of the emitted flux enters the covered medium and the
remaining $1-f_\text{cov}$ enters the reservoir -- the part of the
cross-section the discharge has not broken down. Two rays are marched per
cathode end and their per-cell totals are summed,

$$\text{deposition} = \underbrace{\text{ray}\!\left(f_\text{cov}\Gamma_0;\;
n/f_\text{cov},\;n_{n,c},\;f_\text{cov}A\right)}_{\text{channel arm}}
\;+\;
\underbrace{\text{ray}\!\left((1-f_\text{cov})\Gamma_0;\;
n_\text{floor},\;n_{n,r},\;(1-f_\text{cov})A\right)}_{\text{reservoir arm}},$$

with the reservoir arm's plasma density at the model's own "no plasma"
representation, the density floor, because the closure's premise is that
plasma lives in the covered fraction. Note what the split does to the
quasilinear beam density: each arm carries its own share of the flux over its
own share of the area, so $n_b$ comes out the same in both and equal to the
mean-field value. That is the physical statement -- the emitted beam is
uniform over the cathode face, and it is the MEDIUM that differs between the
arms, not the beam. The v1 build routed the *whole* beam through the channels,
which is what made it a pure ignition accelerant; the reservoir arm is what
restores the machine-wide deposition the measured conducting phase shows.

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

   The transmission $\sigma_\text{eff}$ reproduces is the flux-weighted
   survival of BOTH arms, i.e. of the whole emitted beam, so the circuit's
   bypass is the real one and not one medium's.
3. **The anomalous tail channel.** The QL tail walkers are marched on the same
   CSDA machinery, so they inherit the channel $n_e$ (through the hoisted
   stopping coefficient) and the channel $n_n$ they ionize. No separate factor
   exists for them.

At $f_\text{cov}=1$ every factor above is multiplication by exactly 1.0 and the
model reduces to the shipped one; $f_{\text{cov}0}=1$ with $r=0$ reproduces the
flag-off trajectory bit-for-bit.

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

$$\frac{dD}{dt} = -B_\text{cov}\frac{1-f_\text{cov}}{f_\text{cov}}
+ B_\text{res} - \frac{D}{\tau_\text{backfill}}$$

(both debits negative on a burn, so the first term grows the deficit and the
second shrinks it). $B_\text{cov}$ is the step-integrated sum of the neutral
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

**The reservoir-born plasma caveat (v1.1), stated rather than hidden.** The
reservoir arm's ionization births are booked into the MEAN plasma fields,
conservatively and with their full energy cost, exactly as the channel arm's
are. But the closure then READS those mean fields as covered plasma: the very
next concentration divides them by $f_\text{cov}$ along with everything else.
So plasma born in the uncovered fraction is, one step later, treated as though
it lived in the channels. This is the honest cost of carrying one plasma field
for two media, and it is the leading candidate for what v2 has to fix -- either
by giving the reservoir its own plasma field, or by letting $f_\text{cov}$ be
driven by that birth rather than by an imposed logistic law. It is bounded in
the regime v1.1 targets, where the reservoir is tenuous and its births are a
small share of the total, and it is NOT bounded once the reservoir carries an
appreciable plasma density -- which is precisely when $f_\text{cov}$ should
have grown anyway.

**v1 limitations, stated rather than absorbed.** The bulk fluid reactions run
on the mean $n_n$, which is exact when the column is backfilled
($n_{n,c}=n_n$) and understates their depletion sensitivity otherwise. The
deficit is carried on the chamber-mean $n_n$ rather than on a column-resolved
partition, so $\tau_\text{backfill}$ absorbs the column-to-chamber exchange as
well as the azimuthal one. The closure requires `neutral_model="moment"` (the
kinetic arms take over the fluid $n_n$ rows once engaged, so the column would
never deplete and the backfill would be a silent no-op) and refuses the
compiled kernels while it is on (the beam deposition it concentrates has a
compiled transcription that v1 does not touch and that has never been
bit-compared under coverage). Both are construction-time `ValueError`s.
