# sim1d Model Equations

The equations `LAPDSim1D` integrates: a conservative axial fluid plasma
coupled to a kinetic (discrete-velocity) neutral gas. Schemes are
[`NUMERICS.md`](NUMERICS.md); configuration-file form is
[`CONFIG_DECLARATIONS.md`](CONFIG_DECLARATIONS.md). Helium only —
`gas_type` other than `"He"` raises at construction.

## Notation and units

CGS with temperatures in eV: cm, s, cm<sup>-3</sup>, cm s<sup>-1</sup>,
erg cm<sup>-3</sup>, cm<sup>-3</sup> s<sup>-1</sup>, A, V.

| symbol | meaning |
|---|---|
| $n$ | plasma density, quasineutral and singly charged, $n_e=n_i\equiv n$ |
| $u$ | parallel plasma (ion) drift velocity |
| $M=m_inu$ | parallel plasma momentum density |
| $E_e=\tfrac32nT_e$, $E_i=\tfrac32nT_i$ | electron and ion energy densities |
| $p_e=nT_e$, $p_i=nT_i$, $p=p_e+p_i$ | pressures |
| $c_s=\sqrt{(T_e+T_i)/m_i}$ | ion sound speed |
| $\mathbf r$, $\mathbf v$ | position and velocity vector of a neutral |
| $z=\mathbf r\!\cdot\!\hat z$ | axial coordinate; $\hat z$ along the axis and $\mathbf B$ |
| $v_\parallel=\mathbf v\!\cdot\!\hat z$ | velocity parallel to $\mathbf B$ |
| $v_\perp=\lvert\mathbf v-v_\parallel\hat z\rvert$ | perpendicular SPEED (a magnitude, not a component) |
| $f_\text{col}$, $f_\text{ann}$ | neutral distribution, column and annulus zone |
| $n_n$, $E_n$, $T_n$ | neutral density, thermal energy density, temperature — moments of $f_\text{col}$ |
| $\phi_c$, $\phi_a$ | cathode and anode sheath falls |
| $V_b$, $V_\text{dis}$, $V_p$, $I$ | device voltage, discharge voltage, gap drop, loop current |
| $\Gamma_0$, $\gamma$, $E$ | beam launch flux, surviving flux, primary energy |
| $A$, $V_\mathrm{p}$, $V_\mathrm{m}$, $V_\text{ann}$ | plasma face area; plasma, chamber, annulus cell volume |

$+z$ runs from the cathode end toward the collector end; a flux is positive
toward $+z$; a source is positive into the field it is written on; $Q_{ie}$ is
a sink for electrons and a source for ions.

## Geometry and state

The machine is represented as a straight axial line of finite-volume cells.
The interior cells resolve the column in 1D; the two cells at either end are 0D
reservoirs rather than resolved fluid. The neutral gas fills the whole vessel,
while the plasma occupies only part of it — a plenum behind the cathode, and
any obstructed volume, carry neutral transport but no plasma at all. Every cell
and every face therefore records whether plasma lives there: a face at the
edge of the plasma is CLOSED to it, and the subset of those faces
where plasma is actually absorbed by a surface carries the outflow described
below. In the code that map is `plasma_active[cell]` and
`plasma_face_live_cell[face]`, and it is the single authority both the fluxes
and the source terms read.

Each cell carries two radial zones: the **column** of radius $R_p$ — the plasma
channel, $V_\mathrm{p}=A\Delta z$ — and the **annulus** between $R_p$ and the
bore $R_m$, $V_\text{ann}=V_\mathrm{m}-V_\mathrm{p}$. The neutral gas occupies
both, the plasma the column alone. Cross-sections may vary along $z$: with a
per-cell effective radius $r_i$, $A_i=\pi r_i^2$,
$V_{\mathrm{p},i}=A_i\Delta z_i$, $A_{i+1/2}=\tfrac12(A_i+A_{i+1})$, external
faces taking their end cell's area. $A$ is the flux-tube variable
($AB=\text{const}$ along a field line), so a solved $B_z(z)$ maps to
$A\propto1/B_z$. Thin annular baffles are neutral-transport surfaces of clear
radius $R_b\ge R_p$: the plasma channel passes through untouched and the discs
restrict the annulus through the open ring
$A_\text{open}=\pi(R_b^2-R_\text{col}^2)$.

The plasma carries four conservative rows per cell, $(n,M,E_e,E_i)$, with
$T_e$, $T_i$, $u$ recovered from them. The neutral gas is the distribution
$f_\text{col}$ on a discrete velocity grid, $f_\text{ann}$ its annulus
counterpart; $n_n$, $E_n$, $T_n$ are its moments rather than evolved rows.
(A fluid neutral closure exists in the code under the `neutral_model` selector
and is not described here.)

## Conservation laws

The plasma is written conservatively, $\partial_tU+\nabla\!\cdot\mathbf F(U)=S$.
The convective derivative is never discretized alone: each is fused with its
compression partner inside one face flux through
$\nabla\!\cdot(U\mathbf u)=\mathbf u\!\cdot\!\nabla U+U\nabla\!\cdot\mathbf u$,
giving $F_n=nu$, $F_M=Mu+p$, $F_{E_e}=E_eu$, $F_{E_i}=E_iu$. The advected energy
flux is the internal-energy flux $E_su$, not the enthalpy flux $(E_s+p_s)u$; the
missing $p_su$ returns as the explicit pressure work below.

$$\partial_tn+\nabla\!\cdot(n\mathbf u)=S_{iz}+S_{iz}^\text{beam}-S_\text{rec}+S_n^\text{bnd}$$

$$\partial_tM+\nabla\!\cdot(M\mathbf u+p)=m_i\mathbf u_{n,\text{eff}}S_{iz}+S_M^{n}+S_M^\text{geom}+S_M^\text{bnd}$$

$$\partial_tE_e+\nabla\!\cdot(E_e\mathbf u)=-p_e\nabla\!\cdot\mathbf u+\nabla\!\cdot\!\left(\kappa_{\parallel e}\nabla T_e\right)-Q_{ie}-C_e+Q_\text{beam}+S_{E_e}^\text{bnd}$$

$$\partial_tE_i+\nabla\!\cdot(E_i\mathbf u)=-p_i\nabla\!\cdot\mathbf u+\nabla\!\cdot\!\left(\kappa_{\parallel i}\nabla T_i\right)+Q_{ie}+Q_i^{n}+Q_\text{hyp}+S_{E_i}^\text{bnd}$$

With a varying area the momentum law is quasi-1D,
$\partial_t(A\rho u)+\partial_z[A(\rho u^2+p)]=p\,\partial_zA+AS_M$, and the
geometric source $S_M^\text{geom}=p\,\partial_zA$ **is** the Maxwellian average
of $-\mu\nabla_\parallel B$ at $A\propto1/B$: no separate mirror force is
carried, and adding one would count the expansion twice. Dropped relative to
full 3D Braginskii: ion viscous stress $\nabla\!\cdot\pi$, the $\mathbf E$ and
$\mathbf u\times\mathbf B$ forces, diamagnetic and drift heat fluxes, and
perpendicular conduction. Wall and end losses enter as boundary-cell terms.

**Neutral kinetic equation.** Let $F(\mathbf r,\mathbf v,t)$ be the full
six-dimensional neutral distribution and $\varphi_v$ the azimuth of
$\mathbf v-v_\parallel\hat z$ about $\hat z$. The evolved object is $F$
averaged over the zone's cross-section and integrated over $\varphi_v$, with
the polar Jacobian folded in:

$$f_\text{col}(z,v_\parallel,v_\perp,t)=\frac{1}{A_\text{col}(z)}\int_{A_\text{col}(z)}\!\!d^2r_\perp\int_0^{2\pi}\!\!d\varphi_v\;v_\perp\,F\!\left(\mathbf r,\left(v_\parallel,v_\perp\cos\varphi_v,v_\perp\sin\varphi_v\right),t\right)$$

normalised so moments are taken against the plain measure
$dv_\parallel dv_\perp$:

$$\int f_\text{col}\,dv_\parallel dv_\perp=n_\text{col}(z,t),\qquad E_n=\int\tfrac12m\left(v_\parallel^2+v_\perp^2\right)f_\text{col}\,dv_\parallel dv_\perp$$

$f_\text{ann}$ is the same construction with $A_\text{ann}(z)$. On the discrete
grid the arrays hold each bin's integral of $f$ — its particle content — so a
density is an unweighted sum over the two velocity axes and an energy the same
sum weighted by $\tfrac12m(v_\parallel^2+v_\perp^2)$ at bin centres. The code's
`v_z` is $v_\parallel$.

$v_\perp$ is a speed and is conserved in free flight; nothing forces a neutral, so
there is no $\partial/\partial\mathbf v$ term, and the cross-section average turns
perpendicular streaming into a boundary flux, leaving $v_\parallel\partial_zf$ the
only spatial derivative:

$$\partial_tf_\text{col}+v_\parallel\partial_zf_\text{col}=\nu_x'(v_\perp)f_\text{ann}-\nu_x(v_\perp)f_\text{col}+\mathcal C[f_\text{col}]+\mathcal S_\text{col}$$

$$\partial_tf_\text{ann}+v_\parallel\partial_zf_\text{ann}=\frac{V_\mathrm{p}}{V_\text{ann}}\nu_x(v_\perp)f_\text{col}-\nu_x'(v_\perp)f_\text{ann}-\nu_w(v_\perp)f_\text{ann}+\mathcal S_\text{ann}$$

$\nu_x$, $\nu_x'$, $\nu_w$ are the azimuth-averaged boundary-crossing rates of an
atom of perpendicular speed $v_\perp$ — each $\propto v_\perp$ per bin, the
coaxial-cylinder view factor and the cavity chord supplying the geometry. This
**column↔annulus exchange** conserves
$n_\text{col}V_\text{col}+n_\text{ann}V_\text{ann}$ exactly, with
$V_\text{col}=V_\mathrm{p}$. $\mathcal C$ is the collision operator below;
$\mathcal S$ carries wall and sheath rebirth, the surface jets, fueling and
pumping. Both velocity coordinates are needed: $\nu_x\propto v_\perp$ selects
fast-perpendicular atoms out of the column, the collision rates use
$(v_\parallel-u_i)^2+v_\perp^2$ per bin, and every rebirth channel enters with a
definite $(v_\parallel,v_\perp)$ spectrum. What the azimuth integral costs is
radial structure INSIDE the column — an atom entering from the annulus is spread
over the whole cross-section at once — and the two-zone split is this model's
radial description.

## Source and sink terms

### Ionization and recombination

$$S_{iz}=n\,n_n\langle\sigma v\rangle_{iz}(n,T_e),\qquad S_\text{rec}=\alpha_\text{rec}(n,T_e)\,n^2$$

$\langle\sigma v\rangle_{iz}$ and $\alpha_\text{rec}$ are ADAS effective
coefficients from the bundled helium `adf11` collisional–radiative files — the
`scd` class for ionization, `acd` for recombination, `plt` and `prb` for
radiated power, and the `adf15` `pec` class for line emission. Both are
collisional–radiative coefficients tabulated against DENSITY as well as
temperature, which is why each carries $n$ as an argument.

**There is no separate three-body sink.** `acd` already contains three-body
recombination at the tabulated density, so the whole recombination loss is the
quadratic term above and the cubic channel is identically zero; the
`recombination_3b_loss` row a result carries reads zero throughout. The
`atomic_rate_model = "janev"` arm instead uses the analytic fits and does split
the two, $\alpha_r(T_e)n^2$ radiative plus $\alpha_3(T_e)n^3$ three-body.
The coefficients are fixed inputs carrying no scale factor, and each result
records an `atomic_rate_domain` ledger of where the run sampled below the
tabulated $T_e$ edge.

$C_e$ is the electron inelastic and radiative cooling: line radiation
($\propto n\,n_n$), the ionization potential cost $I_\text{ion}S_{iz}$, and
recombination radiation ($\propto n^2$).

`ionization_birth_energy_model` selects the birth moments. Under
`"conservative"` the new electron is born cold — zero $E_e$ birth energy, so
$T_e$ falls by dilution as $n$ rises — and the ion mass-loading mixing energy
is booked explicitly as $Q_\text{mix}=\tfrac12m_i(u_i-u_n)^2S_{iz}$, so with
the reconstructed bulk kinetic change $dK=u_i\,dM-\tfrac12m_iu_i^2dn$ the ion
total energy closes on the consumed neutral's,

$$dE_i+dK=\tfrac32T_{i,\text{birth}}S_{iz}+\tfrac12m_iu_n^2S_{iz}.$$

Its thermal half holds at $T_{i,\text{birth}}=T_n$, which
`Ti_birth_ionization = "neutral"` selects; under `"floor"`, `"local"` or a
number the residue $\tfrac32k(T_n-T_{i,\text{birth}})S_{iz}$ leaves the model
and is reported per cell by the
`ionization_birth_thermal_deficit_*_W_cm3` rows. Under `"legacy"` the electron
birth instead adds $\tfrac32T_{e,\text{birth}}S_{iz}$ to $E_e$, with
$T_{e,\text{birth}}$ set by `Te_birth_ionization`.

### The neutral collision operator and its plasma coupling

$\mathcal C[f]$ carries ionization, resonant charge exchange, elastic
scattering and recombination, evaluated per velocity bin at the relative speed
$g_\text{eff}^2=(v_\parallel-u_i)^2+v_\perp^2+8kT_i/\pi m$. Charge exchange and
elastic scattering use the Phelps He<sup>+</sup>/He backscatter and isotropic
cross sections, mapped as $\sigma_{cx}=Q_b$ and $\sigma_{mt}=Q_i+2Q_b$ (the
isotropic channel contributing $\int(1-\cos\theta)=Q_i$ and backscatter at
$180^\circ$ contributing $2Q_b$); `atomic/cross_sections.py` carries the
citation and the archived table. Charge-exchange and elastic events return
their atoms to the same cell within the tick, at the ion Maxwellian;
ionization removes them.

The plasma receives **minus the measured moments** of those operators on its ion
momentum and energy rows, booked once per neutral clock tick; the electron-side
costs — ionization potential, radiation, excitation — stay on the plasma book.
The count the plasma books as ionization is exactly what leaves the column, so
both sides consume the same atoms by construction.

**The neutral clock tick.** The neutral gas is advanced on its own clock, whose
interval $\Delta t_\text{tick}$ is generally many plasma steps long. Everything
the plasma receives from the neutral side is measured over one such tick and
held constant across the plasma steps inside it. Over a tick, and per cell, the
kinetic operators remove a population whose moments are

$$N_\text{loss}\ [\text{atoms}],\qquad P_\text{loss}=\!\!\sum_{\text{lost}}\!\!m v_\parallel\ [\mathrm{g\,cm\,s^{-1}}],\qquad E_\text{loss}=\!\!\sum_{\text{lost}}\!\!\tfrac12m\lvert\mathbf v\rvert^2\ [\mathrm{erg}],$$

a count and two sums over the atoms that left, and $T_{n,\text{loss}}$ is that
population's own temperature — the second moment about ITS mean velocity
$P_\text{loss}/(mN_\text{loss})$, in eV.

The charge-exchange/elastic part of the booking is a **relaxation**, not a
source, with one rate $\nu$ per cell for the pair and targets set by those
moments, taken about the **ion** drift:

$$\frac{dE_i}{dt}=-\nu\left(E_i-E_i^\text{eq}\right),\qquad \frac{dM}{dt}=-\nu\left(M-M^\text{eq}\right)$$

$$u_{n,\text{eff}}=\frac{P_\text{loss}}{m\,N_\text{loss}},\qquad \tfrac32kT_\text{eff}=\frac{E_\text{loss}-u_iP_\text{loss}+\tfrac12mu_i^2N_\text{loss}}{N_\text{loss}}=\tfrac32kT_{n,\text{loss}}+\tfrac12m\left|u_{n,\text{eff}}-u_i\right|^2$$

with $M^\text{eq}=mn_iu_{n,\text{eff}}$ and
$E_i^\text{eq}=\tfrac32n_ikT_\text{eff}$. $T_\text{eff}$ is therefore **not**
the neutral gas temperature but that temperature plus the frictional term
$(m/3k)|u_{n,\text{eff}}-u_i|^2$, so an ion-energy equilibrium built from a
Maxwellian at $T_n$ would be wrong by the whole frictional heating. Ionization
and recombination are sources and are not part of this target; ionization
births sample the lost population's drift, which is the momentum equation's
$m_i\mathbf u_{n,\text{eff}}S_{iz}$ — ion loading drags the flow toward the gas
rather than toward rest.

### Electron–ion exchange, conduction and pressure work

$Q_{ie}$ is the Braginskii collisional exchange at the electron–ion
equipartition rate built from the collision times in `plasma/params.py`.

Parallel Braginskii conduction is $\mathbf q_s=-\kappa_{\parallel s}\nabla T_s$
with $\kappa_{\parallel e}\propto T_e^{5/2}$; perpendicular conduction is not
carried. The electron conductivity is scaled per cell by the harmonic limiter

$$\lambda=\frac{q_\text{sat}}{q_\text{sat}+q_{SH}},\qquad q_\text{sat}=f\,n\,T_e\,v_{th,e},\qquad q_{SH}=\kappa_e\left|\nabla T_e\right|$$

$f$ = `heat_flux_limiter_f` the free-streaming fraction,
`heat_flux_limiter_exponent` its blending exponent, $v_{th,e}$ the electron
thermal speed — so the flux caps at free-streaming where gradients are steep
and recovers the local Spitzer–Härm law where they are shallow, identically in
the explicit and implicit paths. `electron_heat_flux_limit = False` selects the
unlimited local law.

Pressure work is $-p_s\nabla\!\cdot\mathbf u$ for $s=e,i$ alike, with

$$\left.\nabla\!\cdot\mathbf u\right|_i=\frac{A_{i+1/2}u_{i+1/2}-A_{i-1/2}u_{i-1/2}}{V_{\mathrm{p},i}},$$

so expansion cooling through a flare is carried by the same term that carries
compression heating in a straight tube and there is no separate mirror-cooling
source. `hyperbolic_energy_consistent` replaces it with the
kinetic-energy-preserving $-u\,dM_\text{press}$ per species and deposits the
Rusanov $(n,M)$ numerical kinetic-energy dissipation into $E_i$ as
$Q_\text{hyp}$ — a scheme viscosity, distinct from the physical ion–neutral
friction.

### Beam deposition

Primaries are launched at the cathode with flux
$\Gamma_0=I_\text{eth}^\star/e$ and birth energy $e\phi_c$.
`beam_deposition_model = "beer_lambert"` attenuates them exponentially on one
absorption length. `"csda"` instead SLOWS each primary while carrying its flux
unattenuated — the march removes energy from the beam, never primaries from it:

$$\frac{dE}{dz}=-L_\text{tot}(E),\qquad \gamma=\Gamma_0\ \text{along the whole ray}$$

$$L_\text{tot}=\underbrace{n_n\sigma_\text{iz}I_\text{ion}}_\text{potential}+\underbrace{n_n\sigma_\text{iz}\langle W_\text{sec}\rangle}_\text{secondaries}+\underbrace{n_n\sigma_\text{exc}E_\text{rad}}_\text{excitation}+L_\text{coul}+L_\text{anom}$$

$\sigma_\text{iz}$ the He electron-impact ionization cross section,
$\langle W_\text{sec}\rangle$ the mean secondary energy, $\sigma_\text{exc}$
and $E_\text{rad}$ the excitation-manifold channel. Each cell banks
$Q_\text{beam}$, its potential cost and excitation radiation, and the beam
ionization birth as the RATE the surviving flux drives,
$S_{iz}^\text{beam}\,dV=\gamma\,n_n\sigma_\text{iz}(E)\,dz$ integrated over the
cell's path. Those births cost the primary its potential term but do not
remove it from the beam.

$\gamma$ changes at exactly ONE place along the ray: the anode-face crossing,
where the mesh solid fraction $\eta$ of the flux still streaming is booked to
the anode surface and $\gamma\leftarrow(1-\eta)\gamma$ carries downstream. A ray
ENDS when $E$ falls to the stopping floor $E_\text{stop}$, its remaining power
$\gamma E$ banked as local heating or walked. Per-ray power then closes:

$$\Gamma_0E_0=\text{heating}+\text{radiated}+\text{cost}+\text{anode-intercepted}+\gamma_\text{exit}E_\text{exit},$$

$\gamma_\text{exit}E_\text{exit}$ being what a ray that reaches the end of the
mesh carries out of it — with $\gamma_\text{exit}$ equal to $\Gamma_0$, or
$(1-\eta)\Gamma_0$ past the anode, and the whole term zero for a ray that
stopped inside.

The selectors below choose among equations for the two remaining stopping terms
and for where the anomalous bank is deposited; all are CSDA controls, inert
under `"beer_lambert"`.

| selector | value | equation selected |
|---|---|---|
| `beam_coulomb_model` | `"fast_electron"` | $L_\text{coul}=2\pi e^4n_e\ln\Lambda/E$, the CSDA electron–electron stopping power |
| | `"legacy_tau_ei"` | $L_\text{coul}=E/(v(E)\,\tau_{ei}(T_e,n_e))$ on the thermal collision time |
| `beam_anomalous_model` | `"none"` | $L_\text{anom}\equiv0$ |
| | `"quasilinear"` | $L_\text{anom}=E/l_{QL}$, $l_{QL}=(n_e/n_b)(v_b/\omega_{pe})\ln(n_e/n_b)$, $n_b=\gamma/(Av_b)$ |
| | `"ql_relaxation"` | $L_\text{anom}=f_\text{ext}E/L_\text{rel}$, $L_\text{rel}=c(n_e/n_b)v_b/\omega_{pe}$, trapped fraction $f_\text{ext}=C_\text{trap}\min(n_b/2n_e,1)^{1/3}$, gated per cell on $0.687\,\omega_{pe}\min(n_b/n_e,1)^{1/3}>\nu_{en}/2$ with $\omega_{pe}>\nu_{en}$ |
| `heating_anomalous_transport` | `"local"` | the anomalous bank heats the cell that drove it |
| | `"tail_walk"` | the bank is withheld and launched $50/50$ along $\pm B$ as fast-tail electrons, walked on the Coulomb-slowing kinematics until thermalized at $\tfrac32T_e$ or lost to an end |
| | `"plateau_multigroup"` | a solved plateau edge $E_1$ splits the bank into a wave/bulk share $(E_b-E_1)/2E_b$ deposited locally and a streaming share $(E_b+E_1)/2E_b$ divided into $N$ equal-power, $E^2$-uniform-edge groups walked at their own midpoint energies ($E_b=e\phi_c$) |
| `beam_product_transport` | `"local"` | BOTH product populations — the mean secondary energy $\langle W_\text{sec}\rangle$ per ionization and the primary's terminal sub-threshold residual — are banked in the cell where the event happened |
| | `"nonlocal"` | BOTH walk along $B$ from their birth cell on the same mini-CSDA Coulomb integral the primary uses, depositing until they thermalize at $\tfrac32T_e$ or leave an end; secondaries split $50/50$ into $\pm z$ half-weight walks, the terminal residual keeps the primary's direction |
| | `"terminal_nonlocal"` | ONLY the terminal residual walks; every along-ray product, secondaries included, stays banked in its birth cell |

### Cathode, anode and the circuit

The emitting surface, the anode mesh and the bank are one system; the electrode
surfaces are one control surface feeding both the fluid sink and the loop.

The sheath relations below are written in scaled variables. A potential is
scaled by the electron temperature, $\psi=\phi/T_e$; a current by the gap's own
resistance, $J=IR_p/T_e$; and the surface temperature by the electron
temperature, $\delta=k_BT_s/(eT_e)$. Four scaled quantities recur and are named
once here:

- $J_i$ — the cathode's Bohm ion current, scaled: $J_i=I_iR_p/T_e$.
- $J_{i,a}$ — the same for the anode mesh, $J_{i,a}=I_{i,a}R_p/T_e$.
- $\psi_\text{bank}$ — the scaled bank voltage, $V_\text{bank}/T_e$.
- $x$ — the EXTERNAL share of the compliance resistance
  (`R_comp_partition`), so $xR_\text{comp}$ is the part outside the reported
  discharge voltage and $(1-x)R_\text{comp}$ the part inside it.

**Emission.** The Richardson capability is
$I_\text{eth}=A_cC_RT_s^2\exp(-e\phi_\text{wf}/k_BT_s)$, with $A_c$ the
emitting area, $C_R$ the Richardson constant, $\phi_\text{wf}$ the work
function and $T_s$ the surface temperature. Space charge limits release at

$$J_\text{eth,crit}(\psi_+)=J_i\sqrt{\mu\,m_p/m_e}\;\frac{e^{-\psi_+}+\sqrt{1+2\psi_+}-2}{\sqrt{2\psi_+}},$$

$J_i$ the scaled ion current and $\mu$ the ion mass in proton masses, so the
allowed emission $J^\star$ is $J_\text{eth}$ clamped at $J_\text{eth,crit}$
with virtual-cathode barrier
$\psi_-=\delta\ln(J_\text{eth}/J_\text{eth,crit})$, zero on the unclamped
branch, and $I_\text{eth}^\star=J^\star T_e/R_p$.

**Sheaths.** The cathode root solves

$$0=\psi_+-\psi_-+(1+\gamma)J_\text{tot}(\psi_+)-\tau_a\Lambda+\tau_a\ln\!\left(1+\frac{J_\text{anode}}{J_{i,a}}\right)-\psi_\text{bank},\qquad J_\text{tot}=J_i\left(1-e^{\Lambda-\psi_+}\right)+J^\star(\psi_+)$$

with $\gamma=R_\text{comp}/R_p$, $\tau_a=T_{e,\text{anode}}/T_e$,
$\Lambda=\ln\sqrt{m_i/2\pi m_e}$ the electron lift, and
$\phi_c=(\psi_+-\psi_-)T_e$; the current-driven form imposes $I_\text{tot}$ and
roots the monotone $J_\text{tot}(\psi_+)=J_\text{imposed}$ instead. Each
electrode carries its own sheath-edge factor $\alpha_\text{se}$, the electron
lift generalizing as $\Lambda\to\Lambda-\ln\alpha_\text{se}$, the cathode and
anode factors independent. Equivalently the cathode Kirchhoff sum closes the loop current,
returning plasma electrons entering with a minus,
$I_\text{tot}=I_\text{eth}^\star+I_i-I_{e,\text{ret}}$.

The anode current the sheath must pass is the loop current less every directly
collected population,

$$J_\text{anode}=J_\text{tot}-\eta\,\beta_\text{bypass}\,J^\star-J_{\text{tail},a},$$

$\eta$ the mesh solid fraction, $\beta_\text{bypass}=e^{-L_\text{cath}/l_b}$
the beam's gap survival at mean free path $l_b$, and $J_{\text{tail},a}$ the
deposition module's collected tail-walker current. Its one consumer is the
anode sheath, so both subtracted populations raise $\phi_a$ logarithmically:

$$\psi_a=\Lambda_\text{anode}-\ln\!\left(\max\!\left(1+\frac{J_\text{anode}}{J_{i,a}},\epsilon\right)\right),\qquad \phi_a=\psi_aT_{e,\text{anode}}$$

$$V_b=\phi_c+V_p-\phi_a,\qquad V_\text{dis}=V_b+V_\text{series}$$

$V_p=I_\text{tot}R_p$ the ohmic gap drop across the Spitzer column
$R_p=L_\text{cath}/(\pi R_\text{cath}^2\sigma_\parallel)$, and
$V_\text{series}=I[(1-x)R_\text{comp}+R_\text{mesh}]$ the internal series drop
the discharge-voltage probe does not see.

**The loop.** `cathode_solver_model = "current_driven"` integrates

$$L\frac{dI}{dt}=V_\text{src}-I\,xR_\text{comp}-V_\text{dis}(I),\qquad C_\text{bank}\frac{dV_\text{cap}}{dt}=-I$$

$L$ the loop inductance, $R_\text{comp}$ the compliance resistor,
$R_\text{mesh}$ the anode-mesh series resistance, $V_\text{src}$ the bank
voltage, and $V_\text{dis}(I)$ the monotone device relation the sheath solve
returns. Only the EXTERNAL share $xR_\text{comp}$ appears beside
$V_\text{dis}$: the reported
$V_\text{dis}=V_b+I[(1-x)R_\text{comp}+R_\text{mesh}]$ already carries the
internal series drop, so subtracting the full $R_\text{comp}+R_\text{mesh}$
alongside it would count that drop twice.
Substituting $V_\text{dis}$ shows $x$ cancels identically,

$$L\frac{dI}{dt}=V_\text{src}-I\left(R_\text{comp}+R_\text{mesh}\right)-V_b(I),$$

so the loop current responds to the TOTAL series resistance while $x$ moves
only the reported $V_\text{dis}$. The load power closes:

$$P_\text{load}=I_\text{tot}V_b=\underbrace{I_\text{eth}^\star\phi_c+P_{c,i,\phi}-P_{c,e,\phi}}_\text{cathode field work}+\underbrace{I_\text{tot}V_p}_\text{gap ohmic}-\underbrace{I_\text{tot}\phi_a}_\text{anode field work}$$

The anode term SUBTRACTS, which is the same sign the device relation carries:
$V_b=\phi_c+V_p-\phi_a$, so multiplying through by $I_\text{tot}$ gives the
three regions with the anode negative.

Electrode energy is booked to three distinct sources: **circuit field work**
(sheath fall and work function, sourced from the bank and deposited on the
electrode, never through the plasma thermal store); the **plasma-thermal** book,
$2T_e$ per electron and $T_e/2$ per ion through the boxed transmission
coefficients $\gamma_e=2+\phi/T_e$ and $\gamma_i=\tfrac12+\phi/T_e$; and
**plasma heating**, the beam $P_\text{prim}$ and the gap ohmic. The fluid
boundary removes only the plasma-thermal part.

**Prescribed drive.** `cathode_solver_model = "prescribed_measured"` imposes
both loop quantities — $I(t)$ and $V_\text{dis}(t)$ interpolated from a
supplied trace onto the model clock — and consults nothing about the surface.
The same bookkeeping is solved for the cathode fall instead of the voltage,

$$\phi_c=\left(V_\text{dis}-V_\text{series}\right)+\phi_a(\phi_c)-V_p,$$

implicit through $\phi_a$ alone (the bypass fraction being a function of
$\phi_c$), so $\phi_c$ is the single bracketed root of
$\phi_c+V_p-\phi_a(\phi_c)-V_b$ on $(0,\phi_{c,\text{cap}}]$;
$d\phi_a/d\phi_c$ is positive and small, so the residual is strictly increasing
and the root unique. The emitted current is the loop current the ions do not
supply, $I_\text{eth}^\star=\max(I-I_i,0)$ — the deep-repelling-sheath limit of
the Kirchhoff sum above, there being no $\psi$ at which to evaluate the
returning-electron term; the dropped current is measured by the
`I_cathode_kirchhoff_residual` diagnostic. Before
`cathode_prescribed_start_s` the calibrated cathode runs unchanged, and
open-circuit phases withdraw the prescribed drive.

**Plasma-terminating boundary.** At each absorbing face a ghost state is set to
the Bohm outflow — $n_\text{se}=\alpha_\text{se}n$, $u=c_s$ into the wall, the live
cell's $T_e$ and $T_i$ — and the face flux between the interior cell and that
ghost is applied one-sidedly to the live cell. The sheath-edge factor

$$\alpha_\text{se}=\alpha_\text{ps}^{\,d/L_\text{ps}},\qquad \alpha_\text{ps}=e^{-1/2},\qquad L_\text{ps}\sim c_s/\nu_{in}$$

— $\alpha_\text{ps}$ the whole-presheath Boltzmann drop, raised to the sampling
cell length $d$ over the collisional presheath depth $L_\text{ps}$. The fluid
boundary and the circuit read the same $\alpha_\text{se}$. The advective flux carries nothing at those faces — the
ghost flux supplies $Mu+p$, and a wall pressure on top would count the wall
momentum twice. Every other face bounding the plasma is closed, carrying no
particle or thermal-energy flux while keeping the live cell's pressure as its
momentum flux: no flux through the wall, but the wall still pushes back.

**Collector.** The far-end faces take the same ghost-cell outflow with a
floating rather than a driven sheath. No circuit branch owns the electron
energy there, so the boundary term itself books the floating electron sheath at
$2T_e$ per collected ion, electron flux equalling ion flux at a floating
surface; at the cathode that row is owned by the circuit instead.
`end_recycle_to_annulus` routes the collector faces' neutralized flux into that
cell's annulus,
$\partial_tn_{n,a}|_\text{recycle}=\dot N_\text{loss}/V_\text{ann}$, as thermal
diffuse gas carrying no directed momentum.

### Wall return and jet rebirth spectra

Flux reaching a surface is neutralized and reborn as gas. The thermal rebirth
is the cosine half-flux at the live surface temperature, entered on the
velocity grid as $v_\perp\exp(-v_\perp^2/2s^2)$ — the azimuthally integrated 2D
Maxwellian — with one further power of $v_\perp$ from the cosine flux law. At
the cylindrical wall a landing splits into an accommodated share
$\alpha_\text{acc}$ re-emitted on that spectrum and a non-accommodated share
$1-\alpha_\text{acc}$;
`neutral_kinetic_dvm_wall_reflection` selects the second's treatment —
`"specular"` returns it in its incident bin (exact on an axisymmetric grid,
where a specular reflection off the cylinder reverses only the unresolved
radial component), `"diffuse_elastic"` on the same cosine shape at the
temperature carrying the retained share's own incident mean energy per atom.

The **surface jets** split a counted stream by a particle reflection fraction
$R_N$ and a **total** reflected energy fraction $R_E$, so the $R_N$
backscattered atoms carry all of $R_E$ and each leaves with

$$\varepsilon_\text{back}=\frac{R_E}{R_N}\left(\phi+T_i\right),$$

$\phi=\phi_c$ at the cathode and $\phi_a$ at the anode. BOTH electrode specs
clamp the incident energy at zero, $\max(\phi+T_i,0)$, and a cell whose clamped
incident energy is zero launches nothing; the collector jet reads its arrival
energy from $T_e$ and $T_i$ alone. The remaining $1-R_N$ keeps the thermal re-emission. Each surface's
energy book carries the reflected energy as a named loss row formed from the
same counted (particles, incident energy) pair the birth is formed from, so
what the surface gives up is what the gas receives. Anode launches are
DIRECTED away from the wires on the collection side ($-z$ below the mesh face,
$+z$ at or above it) and their thermal remainder is a volume rebirth, a wire
having no half-space to emit into; wire-intercepted *neutrals* are a different
population and keep their at-rest `mesh_blocked`/`mesh_reemit` pair. The
cathode jets launch only while ARMED — a hysteresis latch on the ion current
the accepted step's cathode solve booked, arming at
`neutral_jet_arm_current_A` and disarming below
`neutral_jet_disarm_current_A`, starting disarmed — and below the arming
current nothing is launched AND the surface is not debited, so the surface is
never charged for atoms that were never born. The anode jets are driven by the
anode-collected current and carry no such latch.

At an annular baffle the blocked share of the annulus flux crossing the face is
intercepted at the annulus transparency

$$t_f=\min\!\left(\frac{A_\text{open}}{A^{\,f}_\text{ann}},1\right),\qquad A^{\,f}_\text{ann}=\min\!\left(A^{\,f-1}_\text{ann},A^{\,f}_\text{ann}\right)$$

taken against the annulus throat the march transports through, so the
transmitted throughput $t_fF|v_\parallel|A^{\,f}_\text{ann}$ is exactly
$F|v_\parallel|A_\text{open}$ — the open area is what passes, which is the
whole content of a free-molecular orifice. Intercepted atoms are re-emitted at
$T_\text{wall}$ in the cell they were intercepted from, conserving particles
exactly; the column flux is untouched.

### Fueling and pumping

$$\mathcal S_\text{gp}=\dot N_\text{gp}(t)\,g(z)\,\chi(v_\parallel,v_\perp),\qquad \mathcal S_\text{pump}=-\frac{C_\text{pump}}{V}\,f$$

$\dot N_\text{gp}$ the measured inflow waveform, $g$ the axial placement
profile — normalized so every distributed form conserves the total inflow
exactly — and $\chi$ the injection spectrum at the feed temperature. Injected
gas carries no net directed momentum: it arrives at rest in the lab frame.
`gas_puff_profile` selects $g$: `"cell"` puts the whole flow in the role-tagged
cell; `"gaussian"` uses $\exp[-(z-z_0)^2/2\sigma^2]$ weighted by cell length;
`"cosine_pipe"` uses a Lambertian outlet's first-flight illumination
$[1+((z-z_0)/d)^2]^{-2}$ at throw $d$; `"orifice"` derives the row by ray
optics on a long tube's exit distribution — emit over the pipe-exit disc at the
vessel wall, weight directions by the transparent-regime long-tube angular
intensity, fly straight, and record where each ray first reaches the column
radius, or its perigee where it stays outside (the two coincide at grazing, so
the row is continuous).

### Instruments

Measurement constructs, not model physics, named so a result carrying them can
be read: $-\nu_\text{add}m_inu$ on the momentum row with its frictional work
$\nu_\text{add}m_inu^2$ into $E_i$ — **diagnostic probe, not a physical term.**
No collision process in this model supplies $\nu_\text{add}$; what it measures
is the parallel momentum loss, and the loss length $L=u/\nu_\text{add}$ going
with it, that a region would have to shed to reach a given profile.

## Floors

Clipping inequalities, not initial conditions. The temperature floors are
applied on the DERIVED quantity by `derive_state` on every read, so a saved
trajectory shows a field *at* a floor without showing whether it was clipped up
to one; the packed $E_e$, $E_i$ change only where the floored primitives are
used to rebuild them, at construction and at the end of every stage.

$$n\ge n_\text{floor},\qquad n_n\ge n_{n,\text{floor}},\qquad E_n\ge\tfrac32n_nkT_\text{wall}$$

$$T_e=\max\!\left(\frac{2E_e}{3n},\,T_{e,\text{floor}}\right),\qquad T_i=\max\!\left(\frac{2E_i}{3n},\,T_{i,\text{floor}}\right)$$

Momenta are not clipped: $u$ is recovered with the floored density and $M$
rebuilt from it, leaving $M$ numerically unchanged. Densities are floored
before the energies, so the neutral energy floor is taken against the
already-floored $n_n$ and the implied neutral temperature cannot fall below the
wall's. Clipping up to a floor injects mass or energy; every accepted repair
records its exact extensive debit — plasma and neutral particles added,
electron and ion energy added, on $V_\mathrm{p}$, $V_\text{col}$ and
$V_\text{ann}$ as appropriate — in `floor_ledger`, and a trajectory that never
clips carries an exactly zero ledger.

## Where each term is implemented

Rows a result carries in `rhs_terms`, for the model above.

| RHS row | implementation |
|---|---|
| `plasma_advective_flux` | `physics/flux.py:plasma_flux_rhs_terms` |
| `plasma_front_flux` | `physics/flux.py:front_filling_fluxes` |
| `characteristic_boundary` | `physics/sources.py:characteristic_boundary_rhs` |
| `pressure_work` | `physics/sources.py:pressure_work_rhs`, `velocity_divergence` |
| `hyperbolic_dissipation_heating` | `solver.py:hyperbolic_energy_correction_rhs` |
| `flux_tube_geometry` | `physics/sources.py:flux_tube_geometry_rhs` |
| `heat_conduction` | `physics/conduction.py:heat_conduction_rhs` |
| `ei_exchange` | `physics/energy.py:electron_ion_exchange_rhs` |
| `ionization_birth` | `physics/reactions.py:reaction_rhs_terms` |
| `ionization_energy_cost` | `physics/energy.py:electron_cooling_rhs_terms` |
| `electron_ion_cooling` | `physics/energy.py:electron_cooling_rhs_terms` |
| `electron_neutral_cooling` | `physics/energy.py:electron_cooling_rhs_terms` |
| `recombination_rad_loss` | `physics/reactions.py:reaction_rhs_terms` |
| `recombination_3b_loss` | `physics/reactions.py:reaction_rhs_terms` |
| `recombination_energy_return` | `solver.py:recombination_energy_return_rhs` |
| `cathode_surface_loss` | `physics/cathode.py:cathode_source_terms` |
| `anode_e_sheath_loss` | `physics/cathode.py:cathode_source_terms` (anode row) |
| `anode_collection` | `physics/sources.py:anode_collection_rhs` |
| `beam_ionization_birth` | `physics/cathode.py:beam_ionization_rhs_terms` |
| `beam_power_deposition` | `physics/cathode.py:beam_ionization_rhs_terms` |
| `beam_ionization_cost` | `physics/cathode.py:beam_ionization_rhs_terms` |
| `beam_excitation_radiation` | `physics/cathode.py:beam_ionization_rhs_terms` |
| `gas_puff_local_ionization` | `solver.py:gas_puff_local_ionization_rhs` |
| `neutral_kinetic_dvm_coupling` | `solver.py:neutral_kinetic_dvm_coupling_rhs` (moments from `physics/kinetic_dvm.py`) |
| `parallel_momentum_sink` | `physics/sources.py:parallel_momentum_sink_rhs` |
| `parallel_momentum_sink_heating` | `physics/sources.py:parallel_momentum_sink_heating_rhs` |

`boundary_absorption` and `surface_loss` are permanently zero rows kept for
saved-ledger schema stability, as is `recombination_3b_loss` under the ADAS
coefficients.

The model presented here is the equation set the reference configuration
integrates. A result may carry further rows that are not part of it: those of
the alternative fluid neutral closure, and the additional terms
`electron_drift_transport` and `neutral_probe_source`, all available in the
code and none described by this document.
Supporting modules: `cablp/atomic/` (cross sections, ADAS access, empirical
fits), `cablp/plasma/` (Braginskii conductivities, collision times),
`cablp/cathode/` (circuit solve, CSDA beam march, compiled kernels),
`cablp/constants.py`.
