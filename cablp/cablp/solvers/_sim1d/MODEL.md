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

$$\partial_t(A\rho u)+\partial_z[A(\rho u^2+p)]
  =p\,\partial_z A+A S_M.$$

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
  `ion_neutral_thermalization` flag (default off).
- $\tfrac32(T_\text{birth} - T)\,S_{iz}$ — thermal cost of injecting
  freshly-ionized particles at the birth temperature (vanishes for the default
  `birth="local"` electron choice).

## Reductions relative to full 3D Braginskii

Dropped in this model: ion viscosity/stress $\nabla\!\cdot\pi$, the $\mathbf{E}$
and $\mathbf{u}\times\mathbf{B}$ forces and diamagnetic/drift heat fluxes, and
perpendicular conduction — only the parallel (axial) dynamics are retained. Wall
and end losses (surface neutralization, gas puff, pumping) are folded in as 0D
boundary-cell source terms rather than bulk 3D terms.

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
work (THESIS items 24/25). Honest limits: the anode uses the net-current ladder
(per-species anode is A15/R4), and the bracket-capped regime clamps `V_b` off the
ladder (a reported residual). Full derivation: THESIS_NOTES "The circuit power
balance that closes."

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

The default-off `beam_anode_interception` selector adds the missing anode-mesh
interception event to the CSDA beam ray (audit A15). Without it the CSDA adapter
launches the full emitted flux `Gamma0 = I_eth_star/e` through the whole column, so
the fluid deposits the entire emitted beam (~470 kW on the settled artifact) while
the current-driven circuit books only `P_prim = (1 - eta*beam_bypass_fraction)
* I_eth_star * phi_c` into the plasma (~307 kW) plus the bypass power
`eta*beam_bypass_fraction*I_eth_star*V_b` on the anode. The ~164 kW difference is
the long-mean-free-path beam the anode mesh intercepts, which the fluid was
wrongly depositing downstream.

With the selector on, `deposit_beam` carries a running flux `gamma` (initially
`Gamma0`); at the anode-face crossing the mesh solid fraction `eta` of the flux
still streaming there is intercepted -- booked to the anode surface
(`anode_intercepted_erg_s`), NOT to `plasma_heating_erg_s` -- and only `(1 - eta)`
transmits downstream, carrying the reduced flux through all subsequent deposition
and ionization. A ray that stops in the gap never reaches the face and intercepts
nothing, so exactly the survived (bypass) fraction is removed, consistent with the
circuit's `eta*beam_bypass_fraction`. Per-ray energy still closes to roundoff:

$$\Gamma_0 E_0 = \text{heating} + \text{radiated} + \text{cost}
  + \text{anode-intercepted} + \gamma_{\text{exit}} E_{\text{exit}}.$$

Requires `beam_deposition_model="csda"` and resolved geometry with anode faces;
rejects at construction otherwise (it would be a silent no-op on the beer_lambert
path). Default off; the production golden runs beer_lambert (which never launches
the CSDA module) and stays bit-exact. This removes the +164 kW item-21 anode-
interception error; the paired +43.1 kW ionization birth energy (A14) is R4.2, and
the item-21 ledger re-check follows once both land.

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
