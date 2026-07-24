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
