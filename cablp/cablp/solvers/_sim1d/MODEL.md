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

The neutral flow is not solved dynamically — $\mathbf{u}_n$ is closed by
**molecular (Clausing) free-flow diffusion**, so in practice this reduces to
$\partial_t n_n = \nabla\!\cdot(D_n\nabla n_n) - S_{iz} + S_{rr} + S_{3b}$ plus
the gas-puff / pumping wall terms.

**3. Momentum** (ion inertia, total-pressure gradient; `physics/flux.py`,
`physics/reactions.py`, `physics/sources.py`)

$$m_i\,n\,\frac{D\mathbf{u}}{Dt} = -\,\nabla p \;-\; m_i\,\mathbf{u}\,S_{iz} \;-\; m_i\,\nu_{in}(T_i)\,n\,\mathbf{u}$$

- $-m_i\mathbf{u}\,S_{iz}$ — **ion-loading drag**: neutrals ionize at rest, so
  newly created cold ions mass-load and slow the flow. (The recombination and
  wall-loss momentum sinks cancel identically against their continuity
  contributions when moved to convective form.)
- $-m_i\,\nu_{in}\,n\,\mathbf{u}$ — **ion-neutral collisional drag** (friction on
  the flow from the neutral background), with momentum-transfer collision
  frequency
  $$\nu_{in}(T_i) = \frac{8}{3}\,n_n\,\sigma_{in}\,\sqrt{\frac{T_i}{\pi\,m_i}}, \qquad \sigma_{in} = 5\times10^{-15}\ \text{cm}^2.$$
  Toggled by the `ion_neutral_drag` flag and scaled by `b_ion_neutral_drag`;
  $\sigma_{in}$ is the `sigma_in_cm2` parameter.

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
