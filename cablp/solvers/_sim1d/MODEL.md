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
| $p_e=nT_e$, $p_i=nT_i$, $p=p_e+p_i$ | pressures, formed on the floored $n$ |
| $c_s=\sqrt{T_e/m_i}$ | Bohm speed — the sound speed every boundary, collection and presheath term uses |
| $a=\sqrt{\tfrac53(T_e+T_i)/m_i}$ | the Rusanov signal speed, a scheme quantity ([`NUMERICS.md`](NUMERICS.md)) |
| $\mathbf r$, $\mathbf v$ | position and velocity vector of a neutral |
| $z=\mathbf r\!\cdot\!\hat z$ | axial coordinate; $\hat z$ along the axis and $\mathbf B$ |
| $v_\parallel=\mathbf v\!\cdot\!\hat z$ | parallel velocity — a SIGNED component, $-\infty<v_\parallel<\infty$ |
| $\mathbf v_\perp=\mathbf v-v_\parallel\hat z$ | perpendicular velocity VECTOR |
| $c_\perp=\lvert\mathbf v_\perp\rvert$ | perpendicular SPEED — a magnitude, $c_\perp\ge0$ |
| $f_\text{col}$, $f_\text{ann}$ | neutral distribution, column and annulus zone |
| $n_n$, $E_n$, $T_n$, $u_n$ | neutral density, thermal energy density, temperature, drift — moments of $f_\text{col}$ |
| $\phi_c$, $\phi_a$ | cathode and anode sheath falls |
| $V_b$, $V_\text{dis}$, $V_p$, $I$ | device voltage, discharge voltage, ohmic gap drop, loop current |
| $\Gamma_0$, $\Gamma$, $E$ | beam launch flux, surviving flux, primary energy |
| $A$ | plasma face area |
| $V_\text{col}$, $V_\text{ch}$, $V_\text{ann}$ | column (plasma), chamber and annulus cell volume, $V_\text{ann}=V_\text{ch}-V_\text{col}$ |
| $\eta$ | anode-mesh solid fraction |

Source aggregates used in the conservation laws, each defined in its own
section below:

| symbol | meaning |
|---|---|
| $S_{iz}$, $S_{iz}^\text{beam}$, $S_\text{rec}$ | thermal ionization, beam-impact ionization, recombination |
| $S_\text{an}$ | anode-mesh Bohm collection rate |
| $S_n^\text{out}$, $F^\text{out}$, $Q_e^\text{out}$, $Q_i^\text{out}$ | the plasma-terminating (absorbing) face terms — the outflow, one per evolved field |
| $Q_e^\text{elec}$ | the electrode electron-sheath energy term |
| $F^n$, $Q_i^n$ | the kinetic coupling terms — minus the measured moments of the ionization, charge-exchange, elastic AND recombination operators, so they carry the ionization birth and the recombination sink on $M$ and $E_i$ |
| $Q_\text{beam}$ | the NET electron gain from the beam: the gross deposition less the beam's OWN ionization cost and excitation radiation |
| $Q_\text{ohm}$ | the ohmic gap heating, booked with the beam deposition |
| $Q_\text{diss}$ | the ion kinetic energy the Rusanov face flux dissipates numerically, returned to $E_i$ |
| $F^\text{geom}$ | the quasi-1D geometric pressure force |

**A source symbol's letter is its dimension.** In the aggregates above and in
every conservation law below, $S$ denotes a particle-density rate
(cm<sup>-3</sup> s<sup>-1</sup>), $F$ a momentum-density rate — a force density
(dyn cm<sup>-3</sup> = g cm<sup>-2</sup> s<sup>-2</sup>) — and $Q$ an
energy-density rate (erg cm<sup>-3</sup> s<sup>-1</sup>). The one exception is
$S_\text{pump}$, which is a volumetric pumping SPEED
(cm<sup>3</sup> s<sup>-1</sup>) in the vacuum-engineering sense, not a density
rate. $\Gamma$ is a flux, never a source. The calligraphic $\mathcal S$ of the
kinetic equation is the phase-space counterpart of $S$, per unit
$dv_\parallel dc_\perp$, and the calligraphic $\mathcal F$ there is the
six-dimensional neutral distribution — upright $F$ is a force density and
nothing else.

**The two velocity coordinates are not the same kind of quantity:**
$v_\parallel$ is a SIGNED component along $\hat z$, so the discrete grid spans
$-v_\text{max}\ldots+v_\text{max}$ and a launch directed toward $-z$ has drift
$u<0$; $c_\perp$ is a SPEED, non-negative by construction, with the polar
Jacobian folded into the bin masses so no signed perpendicular component is
ever carried.

**Derivatives.** The model is ONE-DIMENSIONAL along $\mathbf B$: $z$ is the
only spatial coordinate and $\partial_z$ the only spatial derivative. A
parallel gradient is written $\partial_z$; the divergence of a parallel flux
$\Gamma$ on a tube of area $A(z)$ is written
$\nabla_\parallel\!\cdot \Gamma\equiv A^{-1}\partial_z(A\Gamma)$, which is the
$\Delta(A\Gamma)/V$ the code forms and reduces to $\partial_z\Gamma$ at constant
area. The material derivative along the flow is
$D/Dt\equiv\partial_t+u\,\partial_z$. No perpendicular derivative appears
anywhere in the model.

$+z$ runs from the cathode end toward the collector end; a flux is positive
toward $+z$; a source is positive into the field it is written on; $Q_{ie}$ is
a sink for electrons and a source for ions.

## Geometry and state

The machine is represented as a straight axial line of finite-volume cells.
The neutral gas fills the whole vessel, while the plasma occupies only part of
it — a plenum behind the cathode, and any obstructed volume, carry neutral
transport but no plasma. **The first cell is that neutral-only plenum; the last
is the collector cell, a live plasma cell terminated by an absorbing face.**
Both run the same operators as every other cell. Every cell and every face
records whether plasma lives there: a face at the edge of the plasma is CLOSED
to it, and the subset of those faces where plasma is absorbed by a surface
carries the outflow below. In the code that map is `plasma_active[cell]` and
`plasma_face_live_cell[face]`, and it is the single authority the fluxes and
the source terms both read.

Each cell carries two radial zones: the **column** of radius $R_p$ — the plasma
channel, $V_\text{col}=A\Delta z$ — and the **annulus** between $R_p$ and the
bore $R_m$. The neutral gas occupies both, the plasma the column alone.
Cross-sections may vary along $z$: with a per-cell effective radius $r_i$,
$A_i=\pi r_i^2$, $V_{\text{col},i}=A_i\Delta z_i$,
$A_{i+1/2}=\tfrac12(A_i+A_{i+1})$, external faces taking their end cell's area
(neutral faces instead take the MIN of the two, so a face is a restricting
aperture). $A$ is the flux-tube variable, $AB=\text{const}$ along a field line;
the profile is SUPPLIED per cell, and any conversion from a solved or measured
field happens outside the solver, which does no file I/O. Thin annular baffles
are neutral-transport surfaces of clear radius $R_b\ge R_p$: the plasma channel
passes through untouched and the discs restrict the annulus through the open
ring $A_\text{open}=\pi(R_b^2-R_\text{col}^2)$, $R_\text{col}$ the
face-averaged column radius.

The plasma carries the conservative fields $(n,M,E_e,E_i)$, with $T_e$, $T_i$,
$u$ recovered from them; the packed vector also carries the $n_n$ (and annulus
$n_{n,a}$) fields, which on the kinetic path are the republished moments of the
distribution rather than independently evolved fields. The neutral gas itself
is $f_\text{col}$ on a discrete velocity grid, $f_\text{ann}$ its annulus
counterpart. (A fluid neutral closure exists in the code under the
`neutral_model` selector and is not described here.)

**Phases and engagement.** A run passes through a pre-breakdown phase at a
small standing current, a main discharge held for its own duration, and an
afterglow with the cathode floating. The kinetic neutral arm engages at the
first accepted plasma step, seeded from the fluid neutral fill; the fluid
neutral operators run up to that point and their terms are stripped afterwards,
the coupling term below carrying them instead.

## Conservation laws

### The kinetic closure owns two of the four equations

One asymmetry has to be stated before the equations: it decides where several
terms are written. Once the kinetic neutral arm is engaged it TAKES OVER the
momentum and ion-energy contributions of the fluid ionization, recombination
and ion–neutral terms — those are zeroed and the coupling terms $F^n$,
$Q_i^n$ carry them instead, built from the measured moments of
the ionization, charge-exchange, elastic and recombination operators together.
The DENSITY and ELECTRON-ENERGY contributions of the same terms are NOT taken
over and stay explicit. So on $M$ and $E_i$ the ionization birth and the
recombination sink sit inside $F^n$ and $Q_i^n$; on $n$ both appear
explicitly; and on $E_e$ the recombination sink appears explicitly while the
ionization birth contributes NOTHING — the new electron is born cold, so its
birth energy is identically zero and $T_e$ falls by dilution alone. Their
decomposition is

$$F^n=m_iu_n\left(S_{iz}+S_{iz}^\text{beam}\right)-m_iu\,S_\text{rec}+F_\parallel^{cx},$$

$$Q_i^n=\left[\tfrac32T_n+\tfrac12m_i\left(u_i-u_n\right)^2\right]\left(S_{iz}+S_{iz}^\text{beam}\right)-\tfrac32T_i\,S_\text{rec}+Q_i^{cx},$$

$F_\parallel^{cx}$ and $Q_i^{cx}$ the charge-exchange/elastic friction and
heating of the relaxation described later. Ionization is velocity-blind, which
is why the births carry the column gas's own $u_n$ and $T_n$; recombination
hands the ion's directed momentum and thermal energy back to the gas at the
local $u$ and $T_i$.

### Braginskii form

The model is a two-fluid Braginskii plasma reduced to the parallel direction.
In primitive variables $(n,u,T_e,T_i)$ along the material derivative:

$$\frac{Dn}{Dt}=-n\,\nabla_\parallel\!\cdot u+S_{iz}+S_{iz}^\text{beam}-S_\text{rec}-S_\text{an}+S_n^\text{out}$$

$$m_in\frac{Du}{Dt}=-\partial_zp_i+enE_\parallel+m_i\left(u_n-u\right)\left(S_{iz}+S_{iz}^\text{beam}\right)+F_\parallel^{cx}+F^\text{geom}+F^\text{out}-m_iu\,S_n^\text{out}$$

$$\tfrac32n\frac{DT_e}{Dt}=-p_e\,\nabla_\parallel\!\cdot u-\nabla_\parallel\!\cdot q_{\parallel e}-Q_{ie}-C_e+Q_\text{beam}+Q_\text{ohm}-\tfrac32T_e\left(S_{iz}+S_{iz}^\text{beam}\right)+T_eS_\text{an}+Q_e^\text{elec}+Q_e^\text{out}-\tfrac32T_eS_n^\text{out}$$

$$\tfrac32n\frac{DT_i}{Dt}=-p_i\,\nabla_\parallel\!\cdot u-\nabla_\parallel\!\cdot q_{\parallel i}+Q_{ie}+Q_i^{cx}+\left[\tfrac32\left(T_n-T_i\right)+\tfrac12m_i\left(u_i-u_n\right)^2\right]\left(S_{iz}+S_{iz}^\text{beam}\right)-T_iS_\text{an}+Q_\text{diss}+Q_i^\text{out}-\tfrac32T_iS_n^\text{out}$$

with the parallel heat fluxes and the electron-momentum (ambipolar) closure

$$q_{\parallel s}=-\kappa_{\parallel s}\,\partial_zT_s,\qquad 0=-\partial_zp_e-enE_\parallel\ \Longrightarrow\ enE_\parallel=-\partial_zp_e.$$

**The electric field is not an independent field.** The electrons are taken
massless and carry no separately retained parallel friction, so their momentum
balance is the statement above; substituting it into the ion momentum equation
replaces $-\partial_zp_i+enE_\parallel$ by the TOTAL pressure gradient
$-\partial_zp$, which is the only form that appears below. That substitution
is why one velocity $u$ suffices: the two species move together, the ions
carrying the inertia.

Three features read off this form. Recombination and anode-mesh collection
remove particles at the LOCAL mean velocity and temperature, so their momentum
sinks cancel identically against their continuity contributions and neither
exerts a force. Ionization does not cancel: the new ions arrive at the neutral
drift, leaving the mass-loading drag
$m_i(u_n-u)(S_{iz}+S_{iz}^\text{beam})$ and, on the ion temperature, the birth
term $\tfrac32(T_n-T_i)$ plus the mixing energy per event. And the two
anode-collection debits change SIGN relative to the conservative terms, neither
of them removing energy at the mean the fluid carries: the electron energy is
debited $\tfrac12T_e$ per collected ION, BELOW the $\tfrac32T_e$ a mean
electron would take, so the collection RAISES $T_e$ by $T_eS_\text{an}$, while
the ions leave at the enthalpy $\tfrac52T_i$, ABOVE their $\tfrac32T_i$ mean,
so it LOWERS $T_i$ by $T_iS_\text{an}$.

### Conservative form

**The conservative form below is what the solver integrates**, in the variables
$n$, $M=m_inu$, $E_e=\tfrac32nT_e$ and $E_i=\tfrac32nT_i$; the primitive
equations above are that same system rewritten, and every term corresponds one
for one. Each equation is
$\partial_tU+\nabla_\parallel\!\cdot\boldsymbol\Gamma(U)=S$. The convective
derivative is never discretized alone: each is fused with its compression
partner inside one face flux through
$\nabla_\parallel\!\cdot(Uu)=u\,\partial_zU+U\,\nabla_\parallel\!\cdot u$,
giving $\Gamma_n=nu$, $\Gamma_M=Mu+p$, $\Gamma_{E_e}=E_eu$,
$\Gamma_{E_i}=E_iu$ — the momentum flux carrying the total pressure
$p=p_e+p_i$ the ambipolar closure produced. The advected energy flux is the
internal-energy flux $E_su$, not the enthalpy flux $(E_s+p_s)u$; the missing
$p_su$ returns as the explicit pressure work below.

$$\partial_tn+\nabla_\parallel\!\cdot(nu)=S_{iz}+S_{iz}^\text{beam}-S_\text{rec}-S_\text{an}+S_n^\text{out}$$

$$\partial_tM+\nabla_\parallel\!\cdot(Mu+p)=F^{n}-m_iu\,S_\text{an}+F^\text{geom}+F^\text{out}$$

$$\partial_tE_e+\nabla_\parallel\!\cdot(E_eu)=-u\,\partial_zp_e+\nabla_\parallel\!\cdot\!\left(\kappa_{\parallel e}\partial_zT_e\right)-Q_{ie}-C_e+Q_\text{beam}+Q_\text{ohm}-\tfrac32T_eS_\text{rec}-\tfrac12T_eS_\text{an}+Q_e^\text{elec}+Q_e^\text{out}$$

$$\partial_tE_i+\nabla_\parallel\!\cdot(E_iu)=-u\,\partial_zp_i+\nabla_\parallel\!\cdot\!\left(\kappa_{\parallel i}\partial_zT_i\right)+Q_{ie}+Q_i^{n}+Q_\text{diss}-\tfrac52T_iS_\text{an}+Q_i^\text{out}$$

The ionization birth and the recombination sink are absent from the $M$ and
$E_i$ equations for the reason given above — they are inside $F^n$ and
$Q_i^n$. On $n$ both are explicit, and on $E_e$ the recombination sink is
explicit while the ionization birth adds nothing at all, the electron being
born cold.

**The pressure work is the one term whose discretisation is not the literal
transcription of the primitive one.** The Braginskii form carries
$-p_s\,\nabla_\parallel\!\cdot u$; under `hyperbolic_energy_consistent` the
solver applies the kinetic-energy-preserving $-u\,\partial_zp_s$ instead,
reached by adding $p_s\,\nabla_\parallel\!\cdot u-u\,\partial_zp_s$ to the
uncorrected term, so the two differ by exactly that increment. The choice is a
discrete one: it puts the internal energy and the kinetic energy on the SAME
face pressure the momentum flux uses, so the two close against each other
rather than against two different discretisations of one term. The same
correction supplies $Q_\text{diss}$: the ion kinetic energy that the Rusanov
face flux's numerical dissipation removes from the momentum equation, measured
each step and returned to the ion internal energy so that total kinetic plus
thermal energy is conserved to roundoff. **It is a property of the
discretisation, not a physical process** — no collision, no viscosity of the
plasma itself — and it is distinct from the physical ion–neutral friction.
Clearing the selector restores
$-p_s\,\nabla_\parallel\!\cdot u$ literally, with

$$\left.\nabla_\parallel\!\cdot u\right|_i=\frac{A_{i+1/2}u_{i+1/2}-A_{i-1/2}u_{i-1/2}}{V_{\text{col},i}}.$$

Either way expansion cooling through a flare is carried by the same term that
carries compression heating in a straight tube; there is no separate
mirror-cooling source.

With a varying area the momentum law is quasi-1D,
$\partial_t(A\rho u)+\partial_z[A(\rho u^2+p)]=p\,\partial_zA+AF$, and the
geometric source is

$$F^\text{geom}=\frac{p\,A_{i+1/2}-p\,A_{i-1/2}}{V_{\text{col},i}},\qquad p=p_e+p_i,$$

the TOTAL pressure, the same $p$ the momentum flux carries.

It **is** the Maxwellian average of $-\mu\nabla_\parallel B$ at $A\propto1/B$ —
a derivation about the closure, not a second term the code carries — so no
separate mirror force appears anywhere in the RHS, and adding one would count
the expansion twice. Dropped relative to full 3D Braginskii: ion viscous stress
$\partial_z\pi_\parallel$, the perpendicular $\mathbf E\times\mathbf B$ drift,
diamagnetic and drift heat fluxes, and perpendicular conduction.

**Neutral kinetic equation.** Let $\mathcal F(\mathbf r,\mathbf v,t)$ be the
full six-dimensional neutral distribution,
$\mathbf v_\perp=\mathbf v-v_\parallel\hat z$ its perpendicular velocity
VECTOR, $c_\perp=\lvert\mathbf v_\perp\rvert$ that vector's SPEED, and
$\varphi_v$ the azimuth of $\mathbf v_\perp$ about $\hat z$.
The evolved object is $\mathcal F$ averaged over the zone's cross-section and
integrated over $\varphi_v$; the polar Jacobian $c_\perp$ folds into $f$, so
the surviving perpendicular coordinate is the SPEED $c_\perp\in[0,\infty)$ and
no signed perpendicular component is carried anywhere:

$$f_\text{col}(z,v_\parallel,c_\perp,t)=\frac{1}{A_\text{col}(z)}\int_{A_\text{col}(z)}\!\!d^2r_\perp\int_0^{2\pi}\!\!d\varphi_v\;c_\perp\,\mathcal F\!\left(\mathbf r,\left(v_\parallel,c_\perp\cos\varphi_v,c_\perp\sin\varphi_v\right),t\right)$$

normalised so moments are taken against the plain measure
$dv_\parallel dc_\perp$:

$$\int f_\text{col}\,dv_\parallel dc_\perp=n_\text{col}(z,t),\qquad E_n=\int\tfrac12m\left(v_\parallel^2+c_\perp^2\right)f_\text{col}\,dv_\parallel dc_\perp$$

$f_\text{ann}$ is the same construction with $A_\text{ann}(z)$. On the discrete
grid the arrays hold each bin's integral of $f$ — its particle content — so a
density is an unweighted sum over the two velocity axes and an energy the same
sum weighted by $\tfrac12m(v_\parallel^2+c_\perp^2)$ at bin centres. The code's
`v_z` is $v_\parallel$.

$c_\perp$ is a speed and is conserved in free flight; nothing forces a neutral,
so there is no $\partial/\partial\mathbf v$ term, and the cross-section average
turns perpendicular streaming into a boundary flux, leaving
$v_\parallel\partial_zf$ the only spatial derivative:

$$\partial_tf_\text{col}+v_\parallel\partial_zf_\text{col}=\nu_x(c_\perp)\left(f_\text{ann}-f_\text{col}\right)+\mathcal C[f_\text{col}]+\mathcal S_\text{col}$$

$$\partial_tf_\text{ann}+v_\parallel\partial_zf_\text{ann}=\nu_x'(c_\perp)\left(f_\text{col}-f_\text{ann}\right)-\nu_w(c_\perp)f_\text{ann}+\mathcal S_\text{ann}$$

**Both terms of the column pair carry $\nu_x$, and both of the annulus pair
carry $\nu_x'$**; the exchange conserves
$n_\text{col}V_\text{col}+n_\text{ann}V_\text{ann}$ exactly through
$V_\text{col}\nu_x=V_\text{ann}\nu_x'$. These are azimuth-averaged
boundary-crossing rates of an atom of perpendicular speed $c_\perp$, each
$\propto c_\perp$ per bin, with a Cauchy chord across the annular cavity
supplying the geometry and the $R_p/R_m$ split setting which surface a crossing
reaches; $\nu_w$ is the annulus's own vessel-wall rate. $\mathcal C$ is the
collision operator below; $\mathcal S$ carries wall and sheath rebirth, the
surface jets, fueling and pumping. Both velocity coordinates are needed:
$\nu_x\propto c_\perp$ selects fast-perpendicular atoms out of the column, the
collision rates use $(v_\parallel-u_i)^2+c_\perp^2$ per bin, and every rebirth
channel enters with a definite $(v_\parallel,c_\perp)$ spectrum. What the
azimuth integral costs is radial structure INSIDE the column — an atom entering
from the annulus is spread over the whole cross-section at once — and the
two-zone split is this model's radial description.

## Source and sink terms

### Ionization and recombination

$$S_{iz}=n\,n_n\langle\sigma v\rangle_{iz}(n,T_e),\qquad S_\text{rec}=\alpha_\text{rec}(n,T_e)\,n^2$$

$\langle\sigma v\rangle_{iz}$ and $\alpha_\text{rec}$ are ADAS effective
coefficients from the bundled helium `adf11` collisional–radiative files — the
`scd` class for ionization and `acd` for recombination, both tabulated against
DENSITY as well as temperature, which is why each carries $n$ as an argument.
The `plt` and `prb` classes supply the radiated power below; the `adf15` `pec`
class is read by the line-radiation instrument, not by the solver.

**There is no separate three-body sink.** `acd` already contains three-body
recombination at the tabulated density, so the whole recombination loss is the
quadratic term above and the cubic channel is identically zero; the
`recombination_3b_loss` term a result carries reads zero throughout. The
`atomic_rate_model = "janev"` arm instead uses the analytic fits and does split
the two, $\alpha_r(T_e)n^2$ radiative plus $\alpha_3(T_e)n^3$ three-body. The
BULK coefficients carry no scale factor; the beam excitation channel is the one
exception and carries `b_beam_excitation`. Each result records an
`atomic_rate_domain` ledger of where the run sampled below the tabulated $T_e$
edge.

**Recombination is a sink on every field**, at the local plasma moments: it
removes $S_\text{rec}$ particles, $m_i u S_\text{rec}$ of momentum, and
$\tfrac32T_eS_\text{rec}$, $\tfrac32T_iS_\text{rec}$ of electron and ion
energy, returning the particle to the gas.

$C_e$ is the electron inelastic and radiative cooling, three named channels:

$$C_e=\underbrace{I_\text{ion}S_{iz}}_\text{ionization cost}+\underbrace{\texttt{plt1}\,(T_e)\,n\,n_n}_\text{He line radiation}+\underbrace{\texttt{plt2}\,(T_e)\,n^2}_{\text{He}^+\text{ line radiation}}$$

The $n^2$ term is He<sup>+</sup> LINE radiation (`plt2`), not recombination
radiation; the recombination-radiation class `prb1` is added to it only under
`icool_recomb`.

Birth moments on the fluid path follow `ionization_birth_energy_model`. Under
`"conservative"` the new electron is born cold — zero $E_e$ birth energy, so
$T_e$ falls by dilution as $n$ rises — and the ion mass-loading mixing energy
is booked explicitly as $Q_\text{mix}=\tfrac12m_i(u_i-u_n)^2S_{iz}$, so with
the reconstructed bulk kinetic change $dK=u_i\,dM-\tfrac12m_iu_i^2dn$ the ion
total energy closes on the consumed neutral's,

$$dE_i+dK=\tfrac32T_{i,\text{birth}}S_{iz}+\tfrac12m_iu_n^2S_{iz}.$$

**On the kinetic path this bookkeeping is not a selector at all.** The ionized
atoms are removed from the distribution itself, so they carry their own
moments — the column gas's drift $u_n$ and temperature $T_n$ — and the coupling
term below books exactly $\left(\tfrac32kT_n+\tfrac12m_i(u_i-u_n)^2\right)S_{iz}$
onto $E_i$. `Ti_birth_ionization` and `Te_birth_ionization` govern the fluid
path; the `ionization_birth_thermal_deficit_*_W_cm3` diagnostics report what a fluid
birth temperature other than the gas temperature would leave unbooked.

### The neutral collision operator and its plasma coupling

$\mathcal C[f]$ carries ionization, resonant charge exchange, elastic
scattering and recombination, and the four are not all resolved the same way.
**Ionization is VELOCITY-BLIND**: one frequency per cell, attenuating the whole
distribution uniformly, which is why its births carry the gas's own moments.
Charge exchange and elastic scattering ARE resolved per velocity bin, at the
relative speed

$$g_\text{eff}^2=(v_\parallel-u_i)^2+c_\perp^2+\frac{8kT_i}{\pi m},$$

on the Phelps He<sup>+</sup>/He backscatter and isotropic cross sections; their
per-bin loss frequencies are

$$\nu_{cx}=n\,Q_b\,g_\text{eff},\qquad \nu_{el}=\tfrac12\,n\,Q_i\,g_\text{eff},$$

the $\tfrac12$ being the BGK momentum weighting of an isotropic scatter.
`atomic/cross_sections.py` carries the citation and the archived table.
Charge-exchange and elastic events return their atoms to the same cell within
the tick, at the ion Maxwellian; ionization removes them.

**The neutral clock tick.** The neutral gas is advanced on its own clock, whose
interval $\Delta t_\text{tick}$ is generally many plasma steps long. The plasma
side receives **minus the measured moments** of those operators on its momentum
and ion-energy equations, booked once per tick; the electron-side costs — ionization
potential, radiation, excitation — stay on the plasma book. The ionization and
recombination moments are held CONSTANT across the plasma steps inside a tick;
the charge-exchange/elastic pair is applied as the relaxation below, at the
rate and targets the tick froze (its hold and debt ledgers are
[`NUMERICS.md`](NUMERICS.md)). The count the plasma books as ionization is
exactly what leaves the column, so both sides consume the same atoms by
construction.

Over a tick, and per cell, the charge-exchange and elastic channels remove a
population whose moments are

$$N_\text{loss}\ [\text{atoms}],\qquad P_\text{loss}=\!\!\sum_{\text{lost}}\!\!m v_\parallel\ [\mathrm{g\,cm\,s^{-1}}],\qquad E_\text{loss}=\!\!\sum_{\text{lost}}\!\!\tfrac12m\lvert\mathbf v\rvert^2\ [\mathrm{erg}],$$

a count and two sums over the atoms that left, and $T_{n,\text{loss}}$ is that
population's own temperature — the second moment about ITS mean velocity
$P_\text{loss}/(mN_\text{loss})$, in eV. The pair relaxes at one rate $\nu$ per
cell toward targets set by those moments, taken about the **ion** drift:

$$\frac{dE_i}{dt}=-\nu\left(E_i-E_i^\text{eq}\right),\qquad \frac{dM}{dt}=-\nu\left(M-M^\text{eq}\right)$$

$$u_{n,\text{eff}}=\frac{P_\text{loss}}{m\,N_\text{loss}},\qquad \tfrac32kT_\text{eff}=\frac{E_\text{loss}-u_iP_\text{loss}+\tfrac12mu_i^2N_\text{loss}}{N_\text{loss}}=\tfrac32kT_{n,\text{loss}}+\tfrac12m\left|u_{n,\text{eff}}-u_i\right|^2$$

with $M^\text{eq}=mn_iu_{n,\text{eff}}$ and
$E_i^\text{eq}=\tfrac32n_ikT_\text{eff}$. $T_\text{eff}$ is therefore **not**
the neutral gas temperature but that temperature plus the frictional term
$(m/3k)|u_{n,\text{eff}}-u_i|^2$, so an ion-energy equilibrium built from a
Maxwellian at $T_n$ would be wrong by the whole frictional heating. Ionization
and recombination are sources, not part of this target: **ionization is
velocity-blind**, drawing uniformly from the cell's distribution, so its births
carry the COLUMN gas's own drift $u_n$ and temperature $T_n$ — which is the
$m_i\mathbf u_nS_{iz}$ of the momentum equation and the birth energy above.
$F^n$ and $Q_i^n$ are the momentum and ion-energy totals of this booking.

### Electron–ion exchange, conduction and pressure work

$$Q_{ie}=\frac{3n\left(T_e-T_i\right)}{\tau_e}\frac{m_e}{m_i}$$

the Braginskii collisional exchange at the electron collision time $\tau_e$
built in `plasma/params.py`, so $Q_{ie}\propto n^2\ln\Lambda\,T_e^{-3/2}$.

Parallel Braginskii conduction is $q_{\parallel s}=-\kappa_{\parallel s}\,\partial_zT_s$
with $\kappa_{\parallel e}=3.16\,n\tau_ev_{te}^2\propto T_e^{5/2}$ and
$\kappa_{\parallel i}=3.9\,n\tau_iv_{ti}^2$; perpendicular conduction is not
carried. The face conductivity is the arithmetic mean of its two cells, and
each face carries a transmission factor: zero at a plasma wall, $1-\eta$ across
the anode mesh — **parallel conduction is throttled by the mesh's open
fraction** — and one on an ordinary interior face. The electron conductivity is
scaled per cell by the harmonic limiter

$$\lambda=\frac{q_\text{sat}}{q_\text{sat}+q_{SH}},\qquad q_\text{sat}=f\,n\,T_e\,v_{th,e},\qquad q_{SH}=\kappa_e\left|\partial_zT_e\right|$$

$f$ = `heat_flux_limiter_f` the free-streaming fraction,
`heat_flux_limiter_exponent` its blending exponent (a value other than 1 gives
$1/(1+(q_{SH}/q_\text{sat})^p)$), and $v_{th,e}=\sqrt{T_e/m_e}$. The flux caps
at free-streaming where gradients are steep and recovers the local
Spitzer–Härm law where they are shallow; `electron_heat_flux_limit = False`
selects the unlimited local law.

### Anode-mesh collection

The mesh intercepts the Bohm flux reaching it. At each cell flanking the anode
face,

$$S_\text{an}=\eta\,\alpha_\text{ps}\,n\,c_s\,\frac{A}{V_\text{col}},$$

removing $S_\text{an}$ particles and $m_iuS_\text{an}$ of momentum. Under
`anode_sheath_full_debit` the energy debits are the SHEATH-EDGE moments,
both per COLLECTED ION: $\tfrac12T_e$ from the electrons (the presheath work
the collected pair has already done) and $\tfrac52T_i$ from the ions — the
ion enthalpy, not its thermal energy alone. Clearing that selector books
$\tfrac32T_e$ and $\tfrac32T_i$ instead. The neutralized atoms are returned to
the annulus, falling back to the column where a cell has no annulus.

### Beam deposition

Primaries are launched at the cathode with flux
$\Gamma_0=I_\text{eth}^\star/e$ and birth energy $e\phi_c$.
`beam_deposition_model = "beer_lambert"` attenuates the flux along the ray on
the local absorption length $l_b$. `"csda"` instead SLOWS each primary while
carrying its flux unattenuated — the march removes energy from the beam, never
primaries from it:

$$\frac{dE}{dz}=-L_\text{tot}(E),\qquad \Gamma=\Gamma_0\ \text{along the whole ray}$$

$$L_\text{tot}=\underbrace{n_n\sigma_\text{iz}I_\text{ion}}_\text{potential}+\underbrace{n_n\sigma_\text{iz}\langle W_\text{sec}\rangle}_\text{secondaries}+\underbrace{n_n\sigma_\text{exc}E_\text{rad}}_\text{excitation}+L_\text{coul}+L_\text{anom}$$

$\sigma_\text{iz}$ the He electron-impact ionization cross section,
$\langle W_\text{sec}\rangle$ the mean secondary energy, and
$\sigma_\text{exc}E_\text{rad}$ the excitation-manifold channel, which carries
the `b_beam_excitation` scale. The `beam_power_deposition` term a cell banks is
GROSS — heating, radiation and ionization cost together — and two further terms
take the last two back out of the electron energy: `beam_ionization_cost`
removes $I_\text{ion}S_{iz}^\text{beam}$ and `beam_excitation_radiation` removes
the excitation channel's energy, which leaves the plasma as He I light. Their
sum is the $Q_\text{beam}$ of the equations above, the NET electron gain.
$C_e$ does not cover either: its ionization cost is the BULK $I_\text{ion}S_{iz}$
alone, so the beam's own cost has to be, and is, booked separately. Each cell
also banks the beam ionization birth as the RATE the surviving flux drives,
$S_{iz}^\text{beam}\,dV=\Gamma\,n_n\sigma_\text{iz}(E)\,dz$ integrated over the
cell's path; those births cost the primary its potential term but do not remove
it from the beam.

$\Gamma$ changes at exactly ONE place along the ray: the anode-face crossing,
where the mesh solid fraction $\eta$ of the flux still streaming is booked to
the anode surface and $\Gamma\leftarrow(1-\eta)\Gamma$ carries downstream. A ray
ENDS when $E$ falls to the stopping floor $E_\text{stop}$ — the lowest He
inelastic threshold, $20.6158$ eV — its remaining power $\Gamma E$ banked as
local heating or walked. Per-ray power then closes:

$$\Gamma_0E_0=\text{heating}+\text{radiated}+\text{cost}+\text{anode-intercepted}+\Gamma_\text{exit}E_\text{exit},$$

$\Gamma_\text{exit}E_\text{exit}$ being what a ray that reaches the end of the
mesh carries out of it — with $\Gamma_\text{exit}$ equal to $\Gamma_0$, or
$(1-\eta)\Gamma_0$ past the anode, and the whole term zero for a ray that
stopped inside.

**Two bookings ride the same term.** The per-cell deposition densities are
smoothed by a conservative Gaussian of width `beam_deposition_smoothing_cm`
before they are written, which spreads the beam-range deposition without moving
its total; and the OHMIC GAP HEATING $Q_\text{ohm}$ — the circuit's
$I_\text{tot}V_p$ — is added afterwards, distributed over the cathode–anode gap
cells by Spitzer weights $\propto\Delta z/\sigma_\parallel(T_e,n)$ built from
the same conductivity the gap resistance uses. Both live inside the
`beam_power_deposition` term.

The selectors below choose among equations for the two remaining stopping terms
and for where the anomalous bank is deposited; all are CSDA controls, inert
under `"beer_lambert"`.

| selector | value | equation selected |
|---|---|---|
| `beam_coulomb_model` | `"fast_electron"` | $L_\text{coul}=2\pi e^4n_e\ln\Lambda/E$, the CSDA electron–electron stopping power |
| | `"legacy_tau_ei"` | $L_\text{coul}=E/(v(E)\,\tau_{ei}(T_e,n_e))$ on the thermal collision time |
| `beam_anomalous_model` | `"none"` | $L_\text{anom}\equiv0$ |
| | `"quasilinear"` | $L_\text{anom}=E/l_{QL}$, $l_{QL}=(n_e/n_b)(v_b/\omega_{pe})\ln(n_e/n_b)$, $n_b=\Gamma/(Av_b)$; the length is taken infinite for $n_b\ge n_e/10$, outside the weak-beam domain |
| | `"ql_relaxation"` | $L_\text{anom}=f_\text{ext}E/L_\text{rel}$, $L_\text{rel}=c(n_e/n_b)v_b/\omega_{pe}$, trapped fraction $f_\text{ext}=C_\text{trap}\min(n_b/2n_e,1)^{1/3}$, gated per cell on $0.687\,\omega_{pe}\min(n_b/n_e,1)^{1/3}>\nu_{en}/2$ with $\omega_{pe}>\nu_{en}$ |
| `heating_anomalous_transport` | `"local"` | the anomalous bank heats the cell that drove it |
| | `"tail_walk"` | the bank is withheld and launched $50/50$ along $\pm B$ as fast-tail electrons, walked on the Coulomb-slowing kinematics until thermalized at $\tfrac32T_e$ or lost to an end |
| | `"plateau_multigroup"` | a solved plateau edge $E_1$ splits the bank into a wave/bulk share $(E_b-E_1)/2E_b$ deposited locally and a streaming share $(E_b+E_1)/2E_b$ divided into $N$ equal-power, $E^2$-uniform-edge groups walked at their own midpoint energies ($E_b=e\phi_c$) |
| `beam_product_transport` | `"local"` | BOTH product populations — the mean secondary energy per ionization and the primary's terminal sub-threshold residual — are banked in the cell where the event happened |
| | `"nonlocal"` | BOTH walk along $B$ from their birth cell on the same mini-CSDA Coulomb integral the primary uses; secondaries split $50/50$ into $\pm z$ half-weight walks, the terminal residual keeps the primary's direction |
| | `"terminal_nonlocal"` | ONLY the terminal residual walks; every along-ray product stays banked in its birth cell |

$E_1$ is solved per extraction from the launch cell's own Maxwellian against the
emitted flux, $f_\text{M}(v_1)=m\,j_b/((E_b-E_1)\,\text{erg})$, and clamped at
$E_\text{stop}$ when the edge the equation asks for falls inside the bulk.
**The walkers are not passive.** Under the walking selectors they IONIZE the gas
they cross (`heating_anomalous_tail_ionization`), adding their own birth term,
and their treatment at the cathode face is a selector of its own
(`heating_anomalous_tail_cathode_boundary`) — reflecting them back into the
column rather than absorbing them.

### Cathode, anode and the circuit

The emitting surface, the anode mesh and the bank are one system; the electrode
surfaces are one control surface feeding both the fluid sink and the loop.

The sheath relations below are written in scaled variables. A potential is
scaled by the electron temperature, $\psi=\phi/T_e$; a current by the gap's own
resistance, $J=IR_p/T_e$; and the surface temperature by the electron
temperature, $\delta=k_BT_s/(eT_e)$. Four scaled quantities recur:

- $J_i$ — the cathode's Bohm ion current, scaled: $J_i=I_iR_p/T_e$.
- $J_{i,a}$ — the same for the anode mesh, $J_{i,a}=I_{i,a}R_p/T_e$.
- $\psi_\text{bank}$ — the scaled bank voltage $V_\text{bank}/T_e$, read by the
  voltage-driven solve.
- $x$ — the EXTERNAL share of the compliance resistance
  (`R_comp_partition`), so $xR_\text{comp}$ is the part outside the reported
  discharge voltage and $(1-x)R_\text{comp}$ the part inside it.

**Emission.** The Richardson capability is
$I_\text{eth}=A_cC_RT_s^2\exp(-e\phi_\text{wf}/k_BT_s)$, with $A_c$ the
emitting area, $C_R$ the Richardson constant, $\phi_\text{wf}$ the work
function and $T_s$ the surface temperature. Space charge limits release at

$$J_\text{eth,crit}(\psi_+)=J_i\sqrt{\mu\,m_p/m_e}\;\frac{e^{-\psi_+}+\sqrt{1+2\psi_+}-2}{\sqrt{2\psi_+}},$$

$\mu$ the ion mass in proton masses, and $I_\text{eth}^\star=J^\star T_e/R_p$.

Under `cathode_schottky` the surface field LOWERS the barrier before that test.
For a classical (unclamped) sheath the Child–Langmuir width and its emitter
field give

$$s_\text{CL}=\frac{\sqrt2}{3}\lambda_D\left(2\psi\right)^{3/4},\qquad E_s=\frac43\frac{\phi_c}{s_\text{CL}},\qquad \Delta\phi_S=\kappa_S\sqrt{E_s},$$

$\lambda_D$ the Debye length and $\kappa_S$ the Schottky constant, so the
effective emission is $J_\text{eff}=J_\text{eth}\exp(\Delta\phi_S/\delta T_e)$.
The released current $J^\star$ is then $J_\text{eff}$ where space charge allows
it, and $J_\text{eth,crit}$ where it does not. The virtual-cathode barrier is
$\psi_-=\delta\ln(J_\text{eth}/J_\text{eth,crit})$ on a deep space-charge clamp
(no surface field, no enhancement) and zero otherwise — including where the
enhancement is exactly eaten by space charge.

**The surface is not a constant.** Under `cathode_warming_model = "power_balance"`
its temperature obeys

$$C_\text{th}\frac{dT_s}{dt}=P_\text{heater}+P_\text{ion}-P_\text{rad}-P_\text{emis}-P_\text{cond}-P_\text{back},$$

with $P_\text{heater}$ pinned by the standby equilibrium (at the base
temperature the heater exactly balances radiation, so it is not free),
$P_\text{ion}$ the accepted solve's ion bombardment power,
$P_\text{rad}=\varepsilon\sigma_{SB}A_c(T_s^4-T_\text{env}^4)$ gray-body
radiation, $P_\text{emis}=I_\text{eth}^\star(\phi_\text{wf}+2k_BT_s)$
evaporative emission cooling — each emitted electron removing the barrier plus
its mean thermal energy over it — $P_\text{cond}=G(T_s-T_\text{base})$
conduction into the heater-held substrate, and $P_\text{back}$ the energy the
backscattered atoms of the cathode jet carry away, the $R_E$ share of the
incident ion energy that the gas receives and the surface therefore loses.

Under `cathode_surface_model = "ads_des"` the work function is not a constant
either: an adsorbate coverage $\theta\in[0,1]$ obeys

$$\frac{d\theta}{dt}=-\sigma_\text{cl}(E)\,\Gamma_i\,\theta,\qquad \phi_\text{wf,eff}=\phi_\text{clean}+\left(\phi_\text{wf}-\phi_\text{clean}\right)\theta,$$

$\Gamma_i=I_i/(eA_c)$ the ion flux density and $\sigma_\text{cl}$ the
ion-stimulated desorption cross section, carrying the near-threshold Bohdansky
factor $\left(1-(E_\text{th}/E)^{2/3}\right)\left(1-E_\text{th}/E\right)^2$ at
the mean deposited energy per ion $E=P_{c,i}/I_i$ and vanishing at or below
$E_\text{th}$. It is $\phi_\text{wf,eff}$ that enters the Richardson law.

**Sheaths.** The cathode root solves

$$0=\psi_+-\psi_-+(1+\gamma)J_\text{tot}(\psi_+)-\tau_a\Lambda+\tau_a\ln\!\left(1+\frac{J_\text{anode}}{J_{i,a}}\right)-\psi_\text{bank},\qquad J_\text{tot}=J_i\left(1-e^{\Lambda-\psi_+}\right)+J^\star(\psi_+)$$

with $\gamma=R_\text{comp}/R_p$, $\tau_a=T_{e,\text{anode}}/T_e$,
$\Lambda=\ln\sqrt{m_i/2\pi m_e}$ the electron lift, and
$\phi_c=(\psi_+-\psi_-)T_e$; the current-driven form imposes $I_\text{tot}$ and
roots the monotone $J_\text{tot}(\psi_+)=J_\text{imposed}$ instead, against a
ceiling `cathode_phi_c_cap_V` it clamps to and tags rather than exceeding. Each
electrode carries its own sheath-edge factor $\alpha_\text{se}$, the electron
lift generalizing as $\Lambda\to\Lambda-\ln\alpha_\text{se}$, the cathode and
anode factors independent (the anode keeps the flat $e^{-1/2}$). Equivalently
the cathode Kirchhoff sum closes the loop current, returning plasma electrons
entering with a minus, $I_\text{tot}=I_\text{eth}^\star+I_i-I_{e,\text{ret}}$.

The anode current the sheath must pass is the loop current less every directly
collected population,

$$J_\text{anode}=J_\text{tot}-\eta\,\beta_\text{bypass}\,J^\star-J_{\text{tail},a},$$

$\beta_\text{bypass}=e^{-L_\text{cath}/l_b}$ the beam's gap survival at the
coupling length $l_b$ ($1/l_b=1/(v_b\tau_{ei})+\sigma_bn_n$, zero for
$\phi_c\le0$), and $J_{\text{tail},a}$ the deposition module's collected
tail-walker current, lagged one step. Its one consumer is the anode sheath, so
both subtracted populations raise $\phi_a$ logarithmically:

$$\psi_a=\Lambda_\text{anode}-\ln\!\left(\max\!\left(1+\frac{J_\text{anode}}{J_{i,a}},\epsilon\right)\right),\qquad \phi_a=\psi_aT_{e,\text{anode}}$$

$$V_b=\phi_c+V_p-\phi_a,\qquad V_\text{dis}=V_b+V_\text{series}$$

$V_p=I_\text{tot}R_p$ the ohmic gap drop across the Spitzer column
$R_p=L_\text{cath}/(\pi R_\text{cath}^2\sigma_\parallel)$ with
$\sigma_\parallel\propto T_e^{3/2}/\ln\Lambda$, and
$V_\text{series}=I[(1-x)R_\text{comp}+R_\text{mesh}]$ the internal series drop
the discharge-voltage probe does not see.

**The loop.** `cathode_solver_model = "current_driven"` integrates

$$L\frac{dI}{dt}=V_\text{src}-I\,xR_\text{comp}-V_\text{dis}(I),\qquad C_\text{bank}\frac{dV_\text{cap}}{dt}=-I$$

$L$ the loop inductance, $R_\text{comp}$ the compliance resistor,
$R_\text{mesh}$ the anode-mesh series resistance, $V_\text{src}$ the bank
voltage, and $V_\text{dis}(I)$ the monotone device relation the sheath solve
returns; the current is clamped at $I\ge0$, the loop being a diode. Only the
EXTERNAL share $xR_\text{comp}$ appears beside $V_\text{dis}$: the reported
$V_\text{dis}=V_b+I[(1-x)R_\text{comp}+R_\text{mesh}]$ already carries the
internal series drop, so subtracting the full $R_\text{comp}+R_\text{mesh}$
alongside it would count that drop twice. Substituting $V_\text{dis}$ shows $x$
cancels identically,

$$L\frac{dI}{dt}=V_\text{src}-I\left(R_\text{comp}+R_\text{mesh}\right)-V_b(I),$$

so the loop current responds to the TOTAL series resistance while $x$ moves
only the reported $V_\text{dis}$. The load power closes:

$$P_\text{load}=I_\text{tot}V_b=\underbrace{I_\text{eth}^\star\phi_c+P_{c,i,\phi}-P_{c,e,\phi}}_\text{cathode field work}+\underbrace{I_\text{tot}V_p}_\text{gap ohmic}-\underbrace{I_\text{tot}\phi_a}_\text{anode field work}$$

The anode term SUBTRACTS, the same sign the device relation carries.

Electrode energy is booked to three distinct sources: **circuit field work**
(sheath fall and work function, sourced from the bank and deposited on the
electrode, never through the plasma thermal store); the **plasma-thermal**
book, $2T_e$ per collected electron and $T_e/2$ per ion through the boxed
transmission coefficients $\gamma_e=2+\phi/T_e$ and
$\gamma_i=\tfrac12+\phi/T_e$; and **plasma heating**, the beam and the gap
ohmic. $Q_e^\text{elec}$ is the plasma-thermal electron term of that split.

**What the plasma pays is not the same at the two electrodes.** At the cathode
it pays the thermal part alone. At the anode, under `anode_sheath_full_debit`
and a REPELLING sheath ($\phi_a>0$), the collected electrons climbed the fall
and the plasma pays $\phi_a$ per electron on top of the thermal $2T_e$; at an
ATTRACTING sheath ($\phi_a\le0$) the field does work ON the electrons, the bank
is the payer, and the thermal debit stands alone. A non-finite $\phi_a$ belongs
to neither regime and raises. The collected IONS leave with the enthalpy
$\tfrac52T_i$, not $\tfrac32T_i$ — the $S_\text{an}$ terms above.

**Prescribed drive.** `cathode_solver_model = "prescribed_measured"` imposes
both loop quantities — $I(t)$ and $V_\text{dis}(t)$ interpolated from a
supplied trace onto the model clock — and consults nothing about the surface.
The same bookkeeping is solved for the cathode fall instead of the voltage,

$$\phi_c=\left(V_\text{dis}-V_\text{series}\right)+\phi_a(\phi_c)-V_p,$$

implicit through $\phi_a$ alone, so $\phi_c$ is the single bracketed root of
$\phi_c+V_p-\phi_a(\phi_c)-V_b$ on $(0,\phi_{c,\text{cap}}]$. The emitted
current is the loop current the ions do not supply,
$I_\text{eth}^\star=\max(I-I_i,0)$ — the deep-repelling-sheath limit of the
Kirchhoff sum above, there being no $\psi$ at which to evaluate the
returning-electron term; the dropped current is measured by the
`I_cathode_kirchhoff_residual` diagnostic. A hand-off time
(`cathode_prescribed_start_s`) separates this mode from the calibrated cathode
that runs before it.

**Plasma-terminating boundary.** At each absorbing face a ghost state is set to
the Bohm outflow — $n_\text{se}=\alpha_\text{se}n$, $u=c_s=\sqrt{T_e/m_i}$ into
the wall, the live cell's $T_e$ and $T_i$ — and the face flux between the
interior cell and that ghost is applied one-sidedly to the live cell. Those
terms are the $S_n^\text{out}$, $F^\text{out}$, $Q_e^\text{out}$ and
$Q_i^\text{out}$ of the conservation laws. The sheath-edge
factor is

$$\alpha_\text{se}=\alpha_\text{ps}^{\,d/L_\text{ps}},\qquad \alpha_\text{ps}=e^{-1/2},\qquad L_\text{ps}\sim c_s/\nu_{in}$$

— $\alpha_\text{ps}$ the whole-presheath Boltzmann drop, raised to the sampling
cell length $d$ (capped at $L_\text{ps}$) over the collisional presheath depth.
The fluid boundary and the circuit read the same $\alpha_\text{se}$. The
advective flux carries nothing at those faces — the ghost flux supplies
$Mu+p$, and a wall pressure on top would count the wall momentum twice. Every
other face bounding the plasma is closed, carrying no particle or
thermal-energy flux while keeping the live cell's pressure as its momentum
flux: no flux through the wall, but the wall still pushes back.

**Collector.** The far-end faces take the same ghost-cell outflow with a
floating rather than a driven sheath. No circuit branch owns the electron
energy there, so the boundary term itself books the floating electron sheath at
$2T_e$ per collected ion, electron flux equalling ion flux at a floating
surface; at the cathode that term is owned by the circuit.
`end_recycle_to_annulus` routes the collector faces' neutralized flux into that
cell's annulus, $\partial_tn_{n,a}|_\text{recycle}=\dot N_\text{loss}/V_\text{ann}$,
as thermal diffuse gas carrying no directed momentum.

### Wall return and jet rebirth spectra

Flux reaching a surface is neutralized and reborn as gas. The thermal rebirth
is the cosine half-flux at that surface's own temperature, and which
temperature that is follows the surface: the LOW-$z$ end plane and the closed
faces flanking a cathode cell re-emit at the live cathode temperature $T_s$,
while the collector face, the cylindrical wall, the anode wires and the
baffles re-emit at the wall temperature. The spectrum is entered on the
velocity grid as $c_\perp\exp(-c_\perp^2/2s^2)$, the azimuthally integrated 2D
Maxwellian, with one further power of $c_\perp$ from the cosine flux law. At the cylindrical wall a landing splits into an
accommodated share $\alpha_\text{acc}$ re-emitted on that spectrum and a
non-accommodated share $1-\alpha_\text{acc}$;
`neutral_kinetic_dvm_wall_reflection` selects the second's treatment —
`"specular"` returns it in its incident bin (exact on an axisymmetric grid,
where a specular reflection off the cylinder reverses only the unresolved
radial component), `"diffuse_elastic"` on the same cosine shape at the
temperature carrying the retained share's own incident mean energy per atom.
**The two end planes take the same accommodation**, their non-accommodated
share returned $v_\parallel$-mirrored.

The **surface jets** split a counted stream by a particle reflection fraction
$R_N$ and a **total** reflected energy fraction $R_E$, so the $R_N$
backscattered atoms carry all of $R_E$ and each leaves with

$$\varepsilon_\text{back}=\frac{R_E}{R_N}\left(\phi+T_i\right),$$

$\phi=\phi_c$ at the cathode (clamped at zero before the sum) and $\phi_a$ at
the anode, the sum clamped at zero in both; the collector jet reads its arrival
energy from $T_e$ and $T_i$ alone. **The three channels handle a zero clamped
incident energy differently**: the anode and collector jets launch nothing from
such a cell, while the cathode jet is governed by its arming latch and its
launch builder REFUSES a counted launch at or below zero energy rather than
silently dropping it. The remaining $1-R_N$ keeps the thermal re-emission.
Each surface's energy book carries the reflected energy as a named loss term
formed from the same counted (particles, incident energy) pair the birth is
formed from, so what the surface gives up is what the gas receives.

A monoenergetic launch has to be represented on a discrete grid: the spectrum
is placed at a temperature tied to the axial bin containing the launch speed,
with its drift SOLVED from the energy so the DISCRETE mean energy is exactly
$\varepsilon_\text{back}$ rather than that plus the smear's own thermal
content, and the placed spectrum's moments are checked against their targets
(the construction is [`NUMERICS.md`](NUMERICS.md)).

Anode launches are DIRECTED away from the wires on the collection side — drift
$u<0$ in the cell below the mesh face, $u>0$ at or above it, $v_\parallel$ being
a signed coordinate — and their thermal remainder is a volume rebirth, a wire having no half-space to emit into; wire-intercepted
*neutrals* are a different population and keep their at-rest
`mesh_blocked`/`mesh_reemit` pair. The cathode jets launch only while ARMED — a
hysteresis latch on the ion current the accepted step's cathode solve booked,
arming at `neutral_jet_arm_current_A` and disarming below
`neutral_jet_disarm_current_A`, starting disarmed — and below the arming
current nothing is launched AND the surface is not debited, so the surface is
never charged for atoms that were never born. The anode jets are driven by the
anode-collected current and carry no such latch.

At an annular baffle the blocked share of the annulus flux crossing the face is
intercepted at the annulus transparency

$$t_f=\min\!\left(\frac{A_\text{open}}{A^{\,f}_\text{ann}},1\right),\qquad A^{\,f}_\text{ann}=\min\!\left(A_{\text{ann},f-1},A_{\text{ann},f}\right)$$

the throat being the smaller of the two flanking CELLS' annulus areas, so the
transmitted throughput per bin,
$t_f\,f_\text{ann}|v_\parallel|A^{\,f}_\text{ann}$, is exactly
$f_\text{ann}|v_\parallel|A_\text{open}$ — the bin content of the annulus
distribution times the open area is what passes, which is the whole content of
a free-molecular orifice. Intercepted atoms are re-emitted at $T_\text{wall}$
in the cell they were intercepted from, conserving particles exactly; the
column flux is untouched.

### Fueling and pumping

$$\mathcal S_\text{gp}=\dot N_\text{gp}(t)\,g(z)\,\chi(v_\parallel,c_\perp)$$

$\dot N_\text{gp}$ the measured inflow waveform — a square with erf rise and
close — $g$ the axial placement profile, normalized so every distributed form
conserves the total inflow exactly, and $\chi$ a wall-temperature Maxwellian at
rest, so injected gas carries no net directed momentum. **The puff is born in
the ANNULUS cells**, reaching the column through the zone exchange.
`gas_puff_profile` selects $g$: `"cell"` puts the whole flow in the role-tagged
cell; `"gaussian"` uses $\exp[-(z-z_0)^2/2\sigma^2]$ weighted by cell length;
`"cosine_pipe"` uses a Lambertian outlet's first-flight illumination
$[1+((z-z_0)/d)^2]^{-2}$ at throw $d$; `"orifice"` derives the profile by ray optics
on a long tube's exit distribution — emit over the pipe-exit disc at the vessel
wall, weight directions by the transparent-regime long-tube angular intensity,
fly straight, and record where each ray first reaches the column radius, or its
perigee where it stays outside.

**Pumping is a surface, not a volume.** Each end plane absorbs the fraction

$$s_{L,R}=\min\!\left(\frac{S_\text{pump}}{A_\text{end}\,\bar v/4},\,1\right)$$

of the free-molecular flux striking it — the pumping speed over the one-way
thermal flux through that end's own open neutral area, $\bar v$ the mean speed
of the wall-temperature gas. The remaining $1-s$ returns accommodated or
mirrored like any other end-plane landing. Where the end cell is a plenum, a
pump-elbow conductance folds into the speed in series first.

### Instruments

Measurement constructs, not model physics, named so a result carrying them can
be read: $-\nu_\text{add}m_inu$ on the momentum equation with its frictional work
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
rebuilt from it, leaving $M$ unchanged. Densities are floored before the
energies, so the neutral energy floor is taken against the already-floored
$n_n$ and the implied neutral temperature cannot fall below the wall's. On the
kinetic path the published $n_n$ and $n_{n,a}$ fields are a one-sided
$\max(\text{moment},\text{floor})$ rather than a ledgered clip. Clipping up to
a floor injects mass or energy; every accepted repair records its exact
extensive debit — plasma and neutral particles added, electron and ion energy
added — in `floor_ledger`, and a trajectory that never clips carries an exactly
zero ledger.

## Where each term is implemented

Terms a result carries in `rhs_terms`, for the model above.

| term | function |
|---|---|
| `plasma_advective_flux` | `physics/flux.py:plasma_flux_rhs_terms` |
| `plasma_front_flux` | `physics/flux.py:front_filling_fluxes` |
| `characteristic_boundary` | `physics/sources.py:characteristic_boundary_rhs` |
| `pressure_work` | `physics/sources.py:pressure_work_rhs`, `velocity_divergence`, with the pressure half of `hyperbolic_energy_correction_rhs` |
| `hyperbolic_dissipation_heating` | `physics/sources.py:hyperbolic_energy_correction_rhs` |
| `flux_tube_geometry` | `physics/sources.py:flux_tube_geometry_rhs` |
| `heat_conduction` | `physics/conduction.py:heat_conduction_rhs`; on the split path `implicit_heat_conduction_step` |
| `ei_exchange` | `physics/energy.py:electron_ion_exchange_rhs` |
| `ionization_birth` | `physics/reactions.py:reaction_rhs_terms` |
| `ionization_energy_cost` | `physics/energy.py:electron_cooling_rhs_terms` |
| `electron_ion_cooling` | `physics/energy.py:electron_cooling_rhs_terms` |
| `electron_neutral_cooling` | `physics/energy.py:electron_cooling_rhs_terms` |
| `recombination_rad_loss` | `physics/reactions.py:reaction_rhs_terms` |
| `recombination_3b_loss` | `physics/reactions.py:reaction_rhs_terms` |
| `recombination_energy_return` | `physics/reactions.py:recombination_energy_return_rhs` |
| `cathode_surface_loss` | `physics/cathode.py:cathode_source_terms` |
| `anode_e_sheath_loss` | `physics/cathode.py:cathode_source_terms` (anode part) |
| `anode_collection` | `physics/sources.py:anode_collection_rhs` |
| `beam_ionization_birth` | `physics/cathode.py:beam_ionization_rhs_terms` |
| `beam_power_deposition` | `physics/cathode.py:beam_ionization_rhs_terms` (beam banks, smoothing, and the ohmic gap booking) |
| `beam_ionization_cost` | `physics/cathode.py:beam_ionization_rhs_terms` |
| `beam_excitation_radiation` | `physics/cathode.py:beam_ionization_rhs_terms` |
| `gas_puff_local_ionization` | `physics/reactions.py:gas_puff_local_ionization_rhs` |
| `neutral_kinetic_dvm_coupling` | `solver.py:neutral_kinetic_dvm_coupling_rhs` (moments from `physics/kinetic_dvm.py:_book_transfer`) |
| `parallel_momentum_sink` | `physics/sources.py:parallel_momentum_sink_rhs` |
| `parallel_momentum_sink_heating` | `physics/sources.py:parallel_momentum_sink_heating_rhs` |

`boundary_absorption` and `surface_loss` are permanently zero terms kept for
saved-ledger schema stability, as is `recombination_3b_loss` under the ADAS
coefficients.

The model presented here is the equation set the reference configuration
integrates. A result may carry further terms that are not part of it: those of
the alternative fluid neutral closure, and the additional terms
`electron_drift_transport` and `neutral_probe_source`, all available in the
code and none described by this document.

Supporting modules: `cablp/atomic/` (cross sections, ADAS access, empirical
fits), `cablp/plasma/` (Braginskii conductivities, collision times),
`cablp/cathode/` (circuit solve, CSDA beam march, compiled kernels),
`cablp/constants.py`.
