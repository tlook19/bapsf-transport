"""Solver-agnostic beam deposition along a ray (B1).

Deterministic CSDA (continuous-slowing-down) integration of a monoenergetic
primary-electron beam through the column. **Pure function of the beam and the
column** — ``(E0, Gamma0, nn, ne, Te, ray)`` — with no solver state, so both
the voltage-driven and current-driven cathode formulations can consume it at
the same call sites.

Energy channels, per unit path length [eV/cm]:

- ionization: ``nn * sigma_ion(E) * (I_ion + <W_sec>(E))`` — the potential
  I_ion is banked as ionization cost (matching the solver's separate
  ``beam_ionization_cost`` term); the mean secondary energy <W_sec> is banked
  as plasma heating (the ejected electron thermalizes locally). <W_sec> uses
  the Opal-Peterson-Beaty differential shape ``dsigma/dW ~ 1/(W^2 + Ebar^2)``
  with the He shape parameter Ebar = 15.8 eV (Opal, Peterson & Beaty, At.
  Data 4, 209 (1971)) over W in [0, (E - I_ion)/2].
- excitation: ``nn * sigma_exc(E) * E_rad(E)`` — the summed Ralchenko singlet
  manifold (``_cross.He_beam_excitation_channel``); banked as radiation.
- Coulomb drag on the plasma, per ``coulomb_model``; banked as plasma heating.
- optional anomalous (beam-plasma instability) drag, per ``anomalous_model``;
  banked as plasma heating (Langmuir turbulence Landau-damps on the bulk
  near where it is driven).

Coulomb closures (both parameter-free):

- ``"fast_electron"`` (default): the physical CSDA stopping power on plasma
  electrons, ``dE/dx = 2 pi e^4 n_e lnLambda / E`` — the collision rate of a
  primary falls as 1/v^3, so at main-discharge conditions (150 eV, n_e =
  5e12) the energy e-fold is ~35 m.
- ``"legacy_tau_ei"``: ``dE/dx = E / (v(E) tau_ei(Te, ne))`` with the
  *thermal* NRL collision time — the historical `_cathode_solver._compute_l_b`
  form (~1.1 m at the same conditions). Provided for continuity experiments;
  its "Coulomb" label overestimates the classical drag ~30x.

Anomalous closure:

- ``"quasilinear"``: the beam-plasma (bump-on-tail) instability drives
  Langmuir waves at v_phase ~ v_b with cold-beam growth rate
  ``~(n_b/n_e)^(1/3) omega_pe`` (growth lengths ~mm-cm here, i.e. saturated
  essentially at the source); quasilinear diffusion then flattens the beam
  over the relaxation length

      l_QL = (n_e / n_b) * (v_b / omega_pe) * ln(n_e / n_b)

  (Vedenov-style estimate; ~0.05-0.1 m at production parameters). Modeled as
  mean-energy drag ``dE/dx = E / l_QL`` with the flux preserved (the module
  carries one mean energy, not a distribution — the plateau spread is a
  documented limitation). **Stated validity domain:** weak beam
  (``n_b << n_e``) with growth beating wave damping — the main-discharge
  column. The breakdown phase (n_b ~ n_e) is outside quasilinear theory and
  deliberately stays collisional. Requires ``beam_area_cm2`` to form
  ``n_b = Gamma0 / (A v_b)``. Order-of-magnitude closure: density-gradient
  detuning and saturation physics can lengthen it substantially; results
  using it must be presented per closure, like the drag story.

- ``"ql_relaxation"``: the same instability, booked on its RELAXATION physics
  instead of by fiat. Three ingredients, each literature-boxed:

  * reactive trapping extracts a fraction
    ``f_ext = C_trap min(n_b/2n_e, 1)^(1/3)`` of the beam energy per
    relaxation, ``C_trap = 1`` (O'Neil, Winfrey & Malmberg 1971);
  * the plateau forms over ``tau_QL = c (n_e/n_b)/omega_pe``, so the extracted
    power is spread over ``L_rel = tau_QL v_b`` -- and ``c``, the Vedenov-era
    O(10-100) coefficient, is the closure's ONE registered bracket constant
    (``ql_relaxation_coeff``, default 30, every headline quoted at 10 and 100
    as well);
  * the wave hands its energy to BULK electrons by collisional damping at
    ``nu_en/2``, so the deposition is bulk heating in the cell where the waves
    damp -- the same bank the fiat closure fills, reached for a stated reason
    rather than by assumption.

  Modeled as mean-energy drag ``dE/dx = f_ext E / L_rel``, flux preserved, and
  GATED on the boxed onset inequality: nothing is booked in a cell unless
  ``0.687 omega_pe min(n_b/n_e,1)^(1/3) > nu_en/2`` AND ``omega_pe > nu_en``,
  with ``nu_en = nn K_m(Te)`` on the boxed He e-n momentum-transfer table
  (``_cross.he_electron_momentum_transfer_rate_cm3_s``). Over the working range
  ``n_e = 1e8-1e11`` the gate is expected permanently open by a factor
  400-2500: linear onset is NOT the gating physics, which is precisely why the
  closure is built on relaxation. The gate is evaluated per cell anyway so the
  statement stays a computed property rather than a claim about a range.

  Unlike ``"quasilinear"`` this closure has NO weak-beam cutoff: the
  ``min(., 1)`` caps carry the ``n_b >~ n_e`` corner (a flagged inference), so
  it is defined across breakdown as well as in the main-discharge column.
  Requires ``beam_area_cm2`` and ``ql_relaxation_coeff``. The three closures
  ``{none, quasilinear, ql_relaxation}`` are a declared BRACKET and a result
  must state which one it used.

The primary is followed until it exits the domain (transmitted) or its
energy crosses ``E_stop`` (default: the lowest inelastic threshold, He 2^1S
at 20.6158 eV), where the sub-threshold residual is banked as plasma heating
in the crossing cell. Per-ray energy conservation
``Gamma0*E0 = heating + radiated + ionization cost + transmitted`` holds to
accumulated roundoff by construction (the energy decrement and the channel
banks are the same floating-point sums).

``plasma_heating_erg_s`` lumps four physically distinct deliveries, so the
result also carries them separately as DIAGNOSTIC arrays (``heating_coulomb``
/ ``heating_anomalous`` / ``heating_secondary`` / ``heating_terminal``). These
are bookkeeping only -- they re-add products the energy decrement already
formed, and the lumped bank the RHS consumes is untouched.

Non-local product transport (``product_transport``, WP-D)
---------------------------------------------------------

The two EVENT-PRODUCT channels above -- the mean secondary energy <W_sec> per
ionization, and the primary's terminal sub-threshold residual -- are banked in
their birth cell under ``product_transport="local"`` (the default, and the
historical behaviour). That is perfect local confinement, and at breakdown it
is the wrong limit: both products are BELOW every He inelastic threshold, so
their only loss channel is Coulomb slowing on the bulk, and at n_e ~ 1e10 that
costs ~1 eV per machine pass. They are near-collisionless along B exactly when
the model assumes they stop where they were born.

``product_transport="nonlocal"`` gives each product its own mini-CSDA walk
along B from its birth cell, depositing the same
``coulomb_stopping_eV_per_cm`` drag the primary feels (same ``coulomb_model``)
until it either thermalizes -- energy down to the local Maxwellian mean
``1.5*Te`` (floored at the module's own lnLambda temperature clamp, 0.1 eV),
below which the fast test-particle stopping formula has no meaning and the
electron is indistinguishable from the bulk -- or leaves an end of the domain,
where its remaining energy is booked to the END LEDGER
(``end_loss_low_erg_s`` / ``end_loss_high_erg_s``) and LEAVES the system.

Products and their directions:

- terminal residual: one walk from the stop cell in the PRIMARY's direction;
- secondaries: two half-weight walks (+z and -z) from each birth cell, the
  OPB emission being broadly isotropic -- a stated approximation.

Stated limitations of the walk: straight-line along B with no pitch-angle
diffusion; Coulomb slowing on the plasma is the ONLY loss channel (elastic
e-He transfers ~5 meV per collision, negligible); each birth cell's
secondaries walk at their flux-weighted MEAN energy, matching the module's
mean-energy treatment of the primary; a product is born at the near edge of
its birth cell along its own walk direction, so the walk traverses the whole
birth cell (cell-resolution granularity, as everywhere else in the module).

The end ledger ALSO books the transmitted primary's power
``Gamma_t * E_t`` -- computed by this module since B1 and never banked
anywhere. Under ``"nonlocal"`` the per-ray identity is therefore

    Gamma0*E0 = heating + radiated + ionization_cost + anode_intercepted
                + end_loss_low + end_loss_high

with the transmitted primary INSIDE the end-loss channel (its size is
reported separately as ``end_loss_transmitted_erg_s``, so the historical
form with ``transmitted`` as its own term is recoverable by subtraction).
``transmitted_flux`` / ``transmitted_energy_eV`` keep their meaning and their
values. Under ``"local"`` the end-loss fields are identically zero, nothing
new is booked, and the identity stays exactly the historical

    Gamma0*E0 = heating + radiated + ionization_cost + anode_intercepted
                + transmitted

**v1 is ENERGY-ONLY routing.** Particle and charge bookkeeping is UNCHANGED:
``ionization_events`` / ``excitation_events``, the fluid ``n`` rows they feed,
and the circuit currents are identical in both modes. Only where the product
ENERGY lands moves.

``product_transport="terminal_nonlocal"`` is the MIDDLE point between those
two, and it moves exactly ONE population. The terminal residual walks -- same
machinery, same thermalization floor, same end ledger as under ``"nonlocal"``
-- while every ALONG-RAY product stays where ``"local"`` puts it: the
secondaries are banked at their birth cell, and so are the Coulomb drag, the
ionization cost and the excitation radiation. The two selectors that walk
something therefore differ only in WHICH population walks, never in how.

What separates the terminal residual from the along-ray products is where it
is born and what it is. It is the WHOLE surviving primary flux arriving at one
cell carrying exactly ``E_stop`` each, so it is a point source rather than a
distributed one; and at pre-breakdown density its Coulomb slowing path is
hundreds of machine lengths while its stop cell can sit a small fraction of an
elastic mean free path from the end wall, so "thermalizes where it stopped" is
then a statement about a population that physically leaves. The secondaries
are a distributed source born along the whole ray and are a separate question,
which ``"nonlocal"`` answers and this value deliberately does not.

Two bookings follow from walking that population alone:

- the ESCAPING ENERGY goes to ``end_loss_low_erg_s`` / ``end_loss_high_erg_s``
  exactly as under ``"nonlocal"``. The transmitted primary is NOT added to
  those fields here (that is ``"nonlocal"``'s separate closure of a standing
  hole), so under ``"terminal_nonlocal"`` the end ledger contains the walked
  terminal escape and nothing else, ``end_loss_transmitted_erg_s`` stays 0.0,
  and the per-ray identity keeps ``transmitted`` as its own term::

      Gamma0*E0 = heating + radiated + ionization_cost + anode_intercepted
                  + end_loss_low + end_loss_high + transmitted

- the ESCAPING FLUX is reported in ``terminal_escape_flux_per_s``. Those
  electrons land on a terminating surface, so a caller running a wall-charge
  model can book their CURRENT there while their energy leaves through the
  ledger above. The field is filled whenever the terminal walk runs (under
  either walking value); the solver consumes it only under
  ``"terminal_nonlocal"``, where the terminal population is the only one that
  walks and the charge statement is therefore complete -- ``"nonlocal"``
  remains ENERGY-ONLY as documented above, its secondaries' escape being an
  equally real charge channel that v1 does not book.

QL heating locality (``anomalous_transport``, WP-E)
---------------------------------------------------

The anomalous channel above banks its drag as INSTANTANEOUS LOCAL bulk
heating: the Langmuir turbulence is Landau-damped near where it is driven, so
its energy is handed to the background electrons in the cell that drove it.
That is the right picture for the wave energy, but not for the electrons the
wave heats. Quasilinear diffusion does not warm a Maxwellian in place — it
fills a fast-tail PLATEAU first, and at breakdown densities a tail electron is
collisionally decoupled (Coulomb range ~km at n_e ~ 1e10, hundreds of machine
lengths) and free-streams along B. The bulk only sees the power once those
tail electrons slow down, wherever that happens to be.

``anomalous_transport="tail_walk"`` carries that step explicitly. The
per-cell anomalous power ``P_QL(z)`` is WITHHELD from its birth cell and
re-expressed as a population of tail electrons at a single energy
``tail_energy_eV`` (``E_tail``), i.e. an equivalent tail flux
``P_QL(z) / E_tail`` launched from that cell, split 50/50 along +B and -B
(the QL plateau is driven along B and the bump-on-tail resonance is genuinely
ONE-SIDED; the split does not claim otherwise, and it is NOT particle
scattering — the very Coulomb decoupling stated two paragraphs above makes
collisional isotropization orders of magnitude too slow to act at breakdown.
What the 50/50 stands in for is Langmuir-wave BACKSCATTER, which returns
resonant momentum to the opposite direction; an order-unity return share is
plausible, but the share itself is a kinetic quantity this module cannot
compute, so 50/50 is a STATED APPROXIMATION and not a derived branching
ratio, matching the secondaries' treatment above). Each population is then
walked with the SAME closed-form Coulomb machinery the WP-D products use
-- the module's own ``coulomb_stopping_eV_per_cm`` under the ray's own
``coulomb_model``, the same ``1.5*Te`` thermalization floor -- and deposits
into ``plasma_heating_erg_s`` where it actually slows. **No new physics
parameters beyond E_tail:** the walk introduces nothing the primary's drag law
does not already contain.

``E_tail`` cannot be pinned by a fluid model (the plateau energy is a kinetic
quantity), so per campaign policy it is an ASSUMPTION value and the BRACKET is
the claim: the registered central arm is 75 eV, with 30 and 150 eV as the
bracket arms, all three reported together and none of them ever fitted.

Energy that reaches a domain end still hot is booked to a SEPARATE end-surface
ledger, ``end_loss_tail_low_erg_s`` / ``end_loss_tail_high_erg_s``, and leaves
the system. It is deliberately NOT mixed into the WP-D ``end_loss_*`` fields:
those are documented, smoke-pinned, and already MEASURED (WP-D D2/D4) as the
escape of the two EVENT products, they are identically zero under
``product_transport="local"``, and the two closures switch independently — so
sharing one pair of fields would both break that invariant and silently
contaminate the WP-D diverted-fraction readout whenever both are on. The
per-ray identity under ``"tail_walk"`` gains two terms and nothing else::

    Gamma0*E0 = heating + radiated + ionization_cost + anode_intercepted
                + transmitted (or the WP-D end ledger)
                + end_loss_tail_low + end_loss_tail_high

The CONSERVATION IDENTITY the smoke suite pins is the sharper local statement:
the ray integration is bit-identical in both modes (``L_anom`` depends only on
the beam and the column, never on where its energy is banked), so the
anomalous power the ``"local"`` arm banks equals, to roundoff, what the
``"tail_walk"`` arm deposits along the walks plus what it books to the tail end
ledger.

``heating_anomalous_erg_s`` keeps its meaning as "the anomalous channel's
delivery to the electrons", so under ``"tail_walk"`` it reports the WALKED
deposition profile rather than the birth profile — the same convention WP-D
uses for ``heating_secondary`` / ``heating_terminal``.

Like WP-D this is ENERGY-ONLY: the tail electrons are an energy-transport
bookkeeping device, not a new particle species. ``ionization_events``,
``excitation_events``, every fluid particle row and every circuit current are
identical in both modes. Stated limitations, inherited from the walk: straight
lines along B, no pitch-angle diffusion, one mean energy rather than a plateau
distribution, and no sheath/ambipolar throttle at the ends — so ``"tail_walk"``
is a FREE-ESCAPE bound and the pair {local, tail_walk} brackets the truth.

Tail ionization (``tail_ionization``, K6)
-----------------------------------------

``tail_ionization="on"`` removes the ENERGY-ONLY restriction from the WP-E
tail walk: a tail electron at ``E_tail`` is above every He inelastic threshold
at all shipped rungs, so declaring that it may only Coulomb-slow is a modeling
choice, not a physical one. Under ``"on"`` each tail population is marched
with THIS MODULE'S OWN CSDA integration -- a recursive ``deposit_beam`` call
per birth cell and direction, at ``anomalous_model="none"`` (the plateau
electrons are the instability's product; they do not re-drive it) and
``product_transport="local"`` -- so it attenuates on the local COLUMN neutral
density with the same He ionization and excitation cross sections the primary
uses, at the walker's CURRENT energy, simultaneously with its Coulomb slowing.
Under ``neutral_two_zone`` the ``nn`` the caller passes IS the column channel,
and that is the density the walker attenuates on; the attic/annulus channel is
not on the walker's field line.

The walker's channels are booked into the SAME per-cell banks the primary
fills, so every downstream consumer -- the fluid particle rows, the ionization
cost sink, the radiation sink, the circuit -- sees tail-borne events through
the machinery it already has, with the ion/electron pair booked by exactly the
convention the primary's births use:

- each ionization event adds one pair to ``ionization_events`` (and its
  ``I_ion`` investment to ``ionization_cost_erg_s``);
- the mean secondary ``<W_sec>`` banks as local electron heat with the birth;
- each excitation event's threshold energy goes to ``radiated_erg_s``;
- the walker's Coulomb drag and its sub-threshold terminal residual bank as
  plasma heating where they occur;
- a walker still above ``E_stop`` at either face of its WALK WINDOW goes to the
  SAME tail end ledger the energy-only walk uses (``end_loss_tail_low/high``)
  and leaves the system. This keeps ``"tail_walk"`` a free-escape bound with or
  without the ionization channel, so ``tail_ionization`` moves ONE thing and
  the end convention is not silently swapped underneath it.

``tail_walk_window`` is REQUIRED under ``"on"`` and is the inclusive cell range
the walkers may traverse -- for the solver, the plasma-active window, so that
the cathode disc and anything behind it are a wall. It exists because the
default "the whole grid" is WRONG for a particle channel and wrong quietly: a
``-z`` walker launched near the source runs on into the cells behind the
cathode, and at K5a breakdown conditions 5-66% of the tail's ionization lands
there (measured; ``k6build_tailion_crosscheck.txt``) -- in rows the solver's
active-plasma mask zeroes, so those pairs are created and then deleted. The
energy-only walk shares the geometry but leaks only ~0.04% of ``P_QL`` into it,
which is why the defect had no visible consequence until the walk started
carrying particles. Bounding the window also makes the walk agree with the
registered offline estimate, which truncates ``-z`` walks at the cathode disc
for the same reason.

All of the walkers' heat lands in ``heating_anomalous_erg_s``, which keeps its
documented meaning ("the anomalous channel's delivery to the electrons"); the
primary's own ``heating_coulomb`` / ``heating_secondary`` / ``heating_terminal``
splits are untouched. The tail's contributions to the shared banks are ALSO
reported separately as ``ionization_events_tail``, ``excitation_events_tail``,
``ionization_cost_tail_erg_s`` and ``radiated_tail_erg_s`` -- pure bookkeeping,
identically zero under ``"off"``, and what makes the per-ray branching
statement checkable::

    P_QL = heating_anomalous + ionization_cost_tail + radiated_tail
           + end_loss_tail_low + end_loss_tail_high

i.e. every launched eV ends in exactly one of {bulk heat via thermalization,
ionization investment, secondary-birth heat, radiation, end ledger}. The
overall per-ray identity is UNCHANGED in form -- the tail's cost and radiation
join the terms that already carry the primary's.

**The depth-1 cascade truncation is MEASURED, not assumed.** Secondaries bank
locally rather than walking, which is only correct while ``<W_sec>(E_tail)``
sits below ``E_stop`` -- the point at which a secondary could itself do
something inelastic. That is true at every shipped fixed rung (1.35 / 9.89 /
16.82 eV at 30 / 75 / 150 eV against the 20.6158 eV floor) and stops being true
above ``E_tail`` = 221.406 eV, the ``<W_sec>`` crossing. Both that bar and the
``E_stop`` bar below it are COMPUTED from the thresholds themselves, never
tabulated for the shipped rungs.

Marching outside the band (``tail_ionization``, K7b)
-----------------------------------------------------

Under K7 the tail birth energy is keyed to the LIVE cathode drop,
``E_tail = f * e*phi_c(t)``, so a single run sweeps a wide range of ``E_tail``
as ``phi_c`` climbs from its cold value to its drive plateau. Refusing at the
two bars, which is what K6 did, made that unrunnable: the cold foot sits below
the lower bar and the top ``f`` arm sits above the upper one, so no keyed
ionizing arm could start. The bars now select a TREATMENT per ray instead:

- **Below** ``E_stop``: the march is REVERTED to the energy-only walk for that
  ray. This is exact physics rather than a fallback -- no He inelastic channel
  is open below the lowest threshold, so zero ionization is the answer and not
  an approximation of one. The reversion is EXACT in the implementation sense
  too: the ray takes the identical branch, over the identical floats, that
  ``tail_ionization="off"`` would take for the same call, including that
  closure's own domain convention (windowed under ``tail_reflect_face``, the
  whole grid without it). It is deliberately NOT silent: the reverted power is
  reported in ``tail_sub_threshold_power_erg_s``, so the foot-phase reversion
  is quotable from a saved trajectory.
- **Above** the ``<W_sec>`` crossing: the march is ALLOWED, with the depth-1
  truncation kept and its cost DISCLOSED rather than assumed away. Banking the
  mean secondary locally there UNDERSTATES the tail's ionization, because a
  secondary that clears ``E_stop`` could itself ionize; the understatement was
  MEASURED at <= 2.0% at ``f = 1.0`` (the 2026-08-05 sheathwalk read) and no
  cross-section extrapolation is involved in it -- the EII table covers the
  whole range. The exposure is reported in ``tail_above_bar_power_erg_s``.
- **Above the EII table edge** (``HE_EII_EPS_TOP * I_ion``, ~999.98 eV at the
  module's own ``I_ion``): still a REFUSAL, and the only one left. There
  ``He_EII_cross_lkup`` clamps to its last node, so the walk would attenuate
  on an extrapolated cross section. It IS reachable: a caller keying
  ``E_tail`` to the cathode drop at ``f = 1.0`` puts it on the edge whenever
  the sheath solve sits at its capability-limited ceiling. The edge itself is
  therefore INCLUSIVE within ``HE_EII_EDGE_REL_TOL`` (K7c) -- at the edge the
  clamped value IS the table's last node, so nothing is extrapolated and there
  is nothing to refuse -- and a genuine excess beyond that tolerance raises,
  reporting the measured relative excess.

``tail_power_erg_s`` carries the launched ``P_QL`` so the two exposures can be
read as fractions. In band both are identically zero, which is the statement
that K7b changed nothing that already ran: the split activates ONLY where the
previous code refused outright.

Stated limitations, additional to the walk's own: the walker carries one mean
energy rather than a plateau distribution, so its cross sections are evaluated
at that mean; and it is launched at the near edge of its birth cell, the same
cell-resolution granularity as everywhere else in the module.

Branched disposal of the extracted power (``anomalous_disposal``, pd1)
----------------------------------------------------------------------

``anomalous_transport`` is all-or-nothing: ``"local"`` books every extracted
eV as bulk heat in the cell that extracted it, ``"tail_walk"`` withholds every
extracted eV and walks it. Neither is what a Langmuir wave does. The wave the
beam drives loses its energy through TWO channels at once -- Landau damping on
the resonant electrons (which makes tail electrons, and is nonlocal) and
collisional damping of the wave itself (which makes bulk heat, and is local) --
and their ratio is a computed property of the local cell, not a choice.

``anomalous_disposal="landau_branched"`` books that ratio. Per cell::

    f_Landau = gamma_L / (gamma_L + nu_en/2)

with ``nu_en/2`` the collisional Langmuir amplitude damping already used by the
onset gate (:func:`ql_onset_open`) and ``gamma_L`` the Maxwellian Landau
damping rate at the beam-resonant phase velocity
(:func:`landau_branching_fraction`, which carries the formula and its
validity caveat). A fraction ``f_Landau`` of each cell's extracted power is
withheld and walked exactly as ``"tail_walk"`` walks all of it -- same birth
energy, same 50/50 launch, same Coulomb machinery, same cathode and collector
conventions, same tail end ledger -- and the remaining ``1 - f_Landau`` is
banked as local bulk heat, exactly as ``"local"`` banks all of it. The two
existing values are therefore the ``f_Landau ≡ 1`` and ``f_Landau ≡ 0``
corners of this one, which is why selecting the branch TOGETHER with
``anomalous_transport="tail_walk"`` is refused rather than composed: both
settings claim the same bank.

Mechanically the split is applied AFTER the march and BEFORE the walk stage,
over the withholding bank the tail walk already fills. The march therefore runs
in exactly the configuration ``"tail_walk"`` runs it in -- including the
compiled CSDA kernel, which supports the withholding bank -- and the branch
never enters a kernel. That is what makes the branched arm compiled-vs-pure
bit-identical for the same reason the tail-walk arm is.

The conservation identity is unchanged in FORM, because the locally-banked
share is booked into ``heating_anomalous_erg_s`` alongside the walked share
(both are the anomalous channel's delivery to the electrons)::

    P_QL = heating_anomalous + ionization_cost_tail + radiated_tail
           + end_loss_tail_low + end_loss_tail_high

``tail_power_erg_s`` keeps its documented meaning -- the power this ray
actually LAUNCHED as walkers -- so under the branch it reports the Landau
share rather than ``P_QL``.

Where the split is evaluated matters and is stated: ``f_Landau`` uses the
column state each ray marched (so under beam clumping the enhanced-``nn`` ray
branches on its own enhanced collisionality) and the ray's LAUNCH energy for
the resonant phase velocity, the same convention the pd0 read used. CSDA
slowing along the column is neglected in ``v_phi``.

**Not available under the two-stream (coverage) march**, which refuses it: see
:func:`deposit_beam_two_stream`.

The plateau's two heirs (``anomalous_transport="plateau_multigroup"``)
----------------------------------------------------------------------

Every value above carries the extracted power at ONE energy: ``"local"`` at
none (it is bulk heat), ``"tail_walk"`` and the branch at a single
``E_tail``. The quasilinear plateau is not one energy. In the flux frame the
relaxed distribution is FLAT over the resonant velocity band, so the tail the
wave leaves behind carries a flat differential FLUX ``dGamma/dE`` and
therefore a differential POWER ``dP/dE`` proportional to ``E``, running from
the plateau edge ``E_1`` up to the beam energy ``E_b = e*phi_c``. That
spectrum has two heirs, and a single-energy closure can only ever be one of
them:

* the STREAMING share, the electrons above the plateau edge, which leave the
  extraction region and deposit wherever their own range takes them;
* the WAVE/BULK share, the part of the extracted power that the plateau hands
  back to the bulk where it was driven.

Their sizes follow from the flat plateau alone, with nothing fitted. Matching
the plateau level to the launch cell's own Maxwellian at the edge --

    F_M(v_1) = m * j_b / ((E_b - E_1) * erg),    j_b = I_eth* / (e * A_cell)

with ``F_M`` the 1D-reduced Maxwellian of the launch cell and ``j_b`` the
emitted beam number flux -- fixes ``E_1`` as a STATE-DEPENDENT quantity,
solved by bisection at every extraction solve (see
:func:`plateau_edge_energy_eV`; it is monotone, so the root is unique). The
mean energy of a ``dP/dE ~ E`` spectrum over ``[E_1, E_b]`` then splits the
bank as

    streaming share = (E_b + E_1) / (2 E_b),
    wave/bulk share = (E_b - E_1) / (2 E_b).

The wave share is banked as local bulk heat in the extraction cells, exactly
the bank ``"local"`` fills and the same one the pd1 branch's collisional share
returns to. The streaming share is split into ``N`` EQUAL-POWER groups whose
edges are uniform in ``E^2``,

    E_i = sqrt(E_1^2 + (i/N) (E_b^2 - E_1^2)),   i = 0 .. N,

so ``w_i = (E_i+1^2 - E_i^2)/(E_b^2 - E_1^2) = 1/N`` by construction, and the
same edges are uniform in the CLASSICAL RANGE (which goes as ``E^2``) -- one
edge set that is simultaneously equal-power and equal-reach. Each group is
represented by the arithmetic midpoint ``Ehat_i = (E_i + E_i+1)/2`` and
launched at ``gamma_i = (P_stream/N) / (e * Ehat_i)``, then marched by the
EXISTING walk machinery at its own ``Ehat_i``: same 50/50 +-B split, same
reflection convention, same ionization channel, same tail end ledger. No walk
physics is added -- only ``N`` populations where there was one.
``PLATEAU_GROUP_COUNT`` is the shipped ``N = 8``.

The range law is UNCHANGED (classical Coulomb): the He inelastic channel is a
sub-percent correction to the column stopping over this band, so nothing about
the range map is re-derived for the groups.

Conservation is the same identity in the same FORM, because the wave share is
booked into ``heating_anomalous_erg_s`` alongside the walked groups::

    P_bank = sum_i (walked + cost + radiated + endloss)_i + P_wave

``tail_power_erg_s`` keeps its documented meaning -- the power actually
LAUNCHED as walkers -- so under this value it reports the STREAMING share, and
``plateau_wave_power_erg_s`` reports the other one, the two summing to the
withheld ``P_QL``. The edge ``E_1`` and its clamp census belong to the
EXTRACTION SOLVE rather than to one ray, so the solver carries them (see
``solvers._sim1d.physics.cathode``): the edge is a state-dependent solve, and
a run that spends frames on the clamp is running a different spectrum from one
that does not, so both must be readable per frame rather than assumed.

What this REPLACES: the single-line tail closure's two dials. Under this value
``heating_anomalous_tail_energy_eV``, the ``f`` fraction and the keying
selector are all INERT -- the birth spectrum is derived, not dialled -- so the
solver refuses them explicitly rather than ignoring them.

**Not available under the two-stream (coverage) march**, for the same reason
the pd1 branch is not: the withholding bank is shared between the channel and
reservoir arms and the reservoir carries the density FLOOR, so a plateau edge
solved on it would be an artifact of the floor convention.

Sheath reflection at a walk-window face (``tail_reflect_face``, K7)
-------------------------------------------------------------------

Everything above lets a walker that reaches a window face LEAVE, whatever its
energy. That is the free-escape bound, and at a face that is a biased emitting
surface it is the wrong limit rather than a conservative one: a cathode sitting
at an accelerating drop of a few hundred volts repels every electron in the
plateau, so the flux the free-escape convention deletes is in fact returned to
the column.

``tail_reflect_face`` names the ONE window face that reflects and
``tail_reflect_threshold_eV`` the energy it reflects below. A walker arriving
at that face with energy strictly below the threshold is turned around at the
SAME energy and keeps marching from the face cell; at or above the threshold it
escapes to the tail end ledger exactly as before. The comparison is general
even though the physical threshold (``e*phi_c``) exceeds every shipped plateau
energy, so the reflecting arm is not silently a "reflect everything" arm.

Consequences, all of them deliberate:

- naming a face makes ``tail_walk_window`` REQUIRED in both tail modes, and
  the energy-only walk becomes WINDOWED. Without a window the energy-only walk
  runs the whole grid and there is no face at the cathode to reflect at; with
  one, the same two faces bound both tail modes;
- the reflected walker re-crosses the face cell, the cell-resolution
  granularity every launch in this module has;
- only ONE face may reflect. Two reflecting faces trap the walker between them
  with no escape channel, which neither the closed-form walk nor the march has
  a termination convention for, and it is refused rather than approximated.

STANDING RIDER, unsized: the reflected fraction assumed here is 1. A real
cathode end is a grounded wall with an emitting disc in it, so some of the
returning tail misses the disc RADIALLY and is lost to the wall at its own
much smaller potential. This 1D walk has no radial coordinate and cannot size
that fraction; it is the one assumption that could pull the measured ~2x
reflection gain below exact. It is a documented limitation, NOT a knob -- there
is no partial-reflection coefficient to turn, precisely so that nothing can be
fitted through it.

STANDING RIDER, unchanged by this: the 30 eV bracket arm sits below the
``min(4*Te, 30 eV)`` free-escape convention the offline estimate uses at the
ends, and that convention is the estimate's, not this module's; the walk here
reflects on ``tail_reflect_threshold_eV`` alone and free-escapes on everything
else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .circuit import _c_log_ei
from ..atomic.cross_sections import (
    _HE_LOG_EPS,
    _HE_LOG_SIGMA,
    He_EII_cross_lkup,
    He_beam_excitation_channel_lkup,
    he_electron_momentum_transfer_rate_cm3_s,
    _he_beam_excitation_table,
)
from .beam_lane_march import (
    LANE_MARCH_MODELS,
    check_constants as _check_lane_constants,
    lane_march,
    lane_march_energy_ceiling_eV,
)
from .kernels import COMPILED_KERNELS as _COMPILED_KERNELS

_ERG_PER_EV = 1.602176634e-12
_ME_CGS = 9.1093837015e-28  # electron mass [g]
_E4_CGS = (4.80320425e-10) ** 4  # e^4 [esu^4]
_OMEGA_PE_COEFF = 5.64e4  # omega_pe = 5.64e4 sqrt(n_e) [rad/s] (NRL)

# --- ql_relaxation closure: the two literature-boxed coefficients -----------
# Neither is a knob and neither is exposed as config. The closure's ONE
# description-class constant is `ql_relaxation_coeff`, which is; both of these
# are cited values taken from the literature at the form written beside them.
#
# Cold-beam growth rate coefficient, gamma_r = 0.687 w_pe (n_b/n_e)^(1/3).
QL_GROWTH_COEFF = 0.687
# Reactive-trapping extraction coefficient, f_ext = C_trap (n_b/2n_e)^(1/3).
QL_TRAP_COEFF = 1.0

# Maxwellian Landau damping prefactor, sqrt(pi/8), of the Krall & Trivelpiece
# Sec. 8.6 expression used by `landau_branching_fraction`. A number the cited
# formula contains, not a description-class constant: nothing selects it and
# nothing may tune it.
_LANDAU_DAMPING_COEFF = math.sqrt(math.pi / 8.0)

# The declared anomalous-closure family. A bracket, not a default plus
# alternatives: a result states which arm produced it.
ANOMALOUS_MODELS = ("none", "quasilinear", "ql_relaxation")

#: Number of equal-power groups the plateau's streaming share is split into
#: under ``anomalous_transport="plateau_multigroup"``. Not a config key and
#: not a knob: the band functionals the closure exists to produce converge to
#: 0.39% at this value, so it is a RESOLUTION of the derived spectrum rather
#: than a parameter of it. Exposed as the ``plateau_groups`` argument of
#: :func:`deposit_beam` so a convergence read can vary it without a config
#: dial existing for a campaign run to move.
PLATEAU_GROUP_COUNT = 8

#: Bisection budget for the plateau-edge solve (:func:`plateau_edge_energy_eV`).
#: A fixed iteration count, so the solve is deterministic and reproducible
#: rather than tolerance-and-machine dependent. 200 halvings take any bracket
#: this model can pose (<= ~1e4 eV) below the double-precision spacing of its
#: own endpoints many times over, so the loop is exact-to-representation.
PLATEAU_EDGE_BISECTIONS = 200

# He first ionization potential [eV], the module's STANDALONE default for the
# ``I_ion_eV`` argument. Every solver path passes ``I_ion_eV`` explicitly from
# ``constants.I_ion``, so this default is dormant there; it is spelled to the
# same digits as that constant so that a caller invoking this module DIRECTLY
# reproduces a solver run's numbers bit for bit. It used to read 24.587, which
# differed in the 4th decimal and made the direct-call path silently
# irreproducible -- a trap that could only arm itself once someone used it.
HE_I_ION_EV = 24.58738793623
HE_E_STOP_EV = 20.6158  # lowest inelastic threshold (2^1S)
HE_OPB_EBAR_EV = 15.8  # Opal-Peterson-Beaty shape parameter for He

# The lane march (``beam_lane_march``) re-spells the four constants above that
# its expressions need, because it cannot import them from here without a
# cycle. Assert the two spellings agree at import, so a drift is a startup
# error rather than a silent divergence in the batched legs -- the same check
# the compiled kernel module answers with ``check_constants_beam``.
_check_lane_constants(_ERG_PER_EV, _ME_CGS, _E4_CGS, HE_OPB_EBAR_EV)

# Top of the tabulated He EII cross section, in the table's own reduced units
# eps = E / I_ion (``He_EII_cross_lkup``). Taken FROM the table rather than
# written down, so it cannot drift from the data it describes. Above it the
# lookup clamps to the last node -- i.e. it extrapolates a constant sigma --
# which is the one thing the marched tail walk must never be allowed to do
# (K7b). At the module's own I_ion this is ~999.98 eV.
HE_EII_EPS_TOP = float(np.exp(_HE_LOG_EPS[-1]))
# Fractional slack on that edge (K7c). The edge is INCLUSIVE and carries this
# tolerance: a tail energy whose relative excess over ``HE_EII_EPS_TOP *
# I_ion_eV`` is <= this clamps to the table's last node, which AT the edge is
# the node's own value and therefore not an extrapolation at all; a larger
# excess still raises. The number brackets two measurable scales:
#   floor -- the arithmetic that produces the comparison. ``E_tail`` and the
#     edge are each a product of floats (1 ULP = 1.14e-16 relative here), and
#     under phi_c keying at the ceiling ``E_tail`` also carries the
#     capability-limited sheath root-find, run at rtol = 1e-14
#     (``_cathode_solver_idriven``). That is the largest term, so the slack
#     must exceed ~1e-14 to admit an ``E_tail`` the caller meant to place AT
#     the edge;
#   ceiling -- the table's own resolution. The last two nodes are 3.7e-3 apart
#     in eps, so 1e-12 is 2.7e-10 of one node gap, and a log-linear
#     extrapolation over it (end slope dln(sigma)/dln(eps) = -0.764) would
#     move sigma by 7.6e-13 relative. Nothing physical lives in that window.
# 1e-12 sits two decades above the floor and ten below the ceiling.
HE_EII_EDGE_REL_TOL = 1.0e-12

# --- Non-local product transport (the walking product_transport values) -----
# Thermalization floor of a walking product, as a multiple of the local Te.
# 3/2 is the mean energy of the local Maxwellian: at that energy the product
# is statistically indistinguishable from the bulk it is dragging against, and
# the fast test-particle stopping power (which assumes W >> Te) has stopped
# being a description of anything. NOT a tunable -- it is the bulk mean
# energy, not a fitted number, and it is deliberately not exposed as a config
# key.
_PRODUCT_FLOOR_TE_MULTIPLE = 1.5
# Absolute floor [eV], reusing the module's OWN lnLambda temperature clamp
# (``_c_log_ei(max(Te, 0.1), ne)`` in coulomb_stopping_eV_per_cm): below it the
# Coulomb logarithm is already being evaluated at a substitute temperature, so
# a walk there would be tracking a formula outside its own domain. Also the
# guard that keeps the walk finite when Te -> 0.
_PRODUCT_FLOOR_MIN_EV = 0.1
# Energy exponent p of each closure's stopping power, dE/dx = A(ne,Te) * W**p.
# Both closures are exact power laws in W -- lnLambda depends only on (ne, Te)
# -- which is what makes the product walk integrable in closed form below.
_COULOMB_STOPPING_EXPONENT = {"fast_electron": -1.0, "legacy_tau_ei": 0.5}
# Lowest INCIDENT energy [eV] at which the reversed-walker rider evaluates its
# {R_e, eta_E} pair. A walker arriving at the anode plane below this is treated
# as ABSORBED whatever the pair says -- the per-incident particle/energy
# reflection convention the pair is stated in does not hold down there, and
# absorbing is the conservative direction for a cull. Not a config key: it is
# the domain edge of the convention, not an arm of it.
TAIL_ANODE_RIDER_MIN_ENERGY_EV = 50.0

# --- Compiled CSDA march (opt-in; see cablp.cathode.kernels) ------------------
# The cost read of 2026-08-02 measured the substep march at ~61% numpy SCALAR
# dispatch and Python call overhead -- ~873 sub-calls per ``deposit_beam``
# call -- so the compiled unit's boundary encloses the WHOLE double loop
# rather than any leaf. Nothing here changes on the default pure path: the
# module-level bind is one ``is None`` test at import, and ``deposit_beam``
# keeps its Python march verbatim as the fallback and the equivalence target.
#
# Model selectors are integers across the boundary because the transcription
# cannot raise ``coulomb_stopping_eV_per_cm``'s ValueError from the place the
# Python raises it; an unrecognised model is simply not offered to the kernel
# and takes the Python march, which raises exactly as it always did.
_COULOMB_MODEL_CODE = {"fast_electron": 0, "legacy_tau_ei": 1}

_CSDA_MARCH = None
if _COMPILED_KERNELS is not None:
    _COMPILED_KERNELS.check_constants_beam(
        _ERG_PER_EV, _ME_CGS, _E4_CGS, _OMEGA_PE_COEFF, math.pi
    )
    _CSDA_MARCH = _COMPILED_KERNELS.csda_march

_CSDA_TABLES = None
_CSDA_TABLES_SRC = None


def _csda_tables():
    """The compiled march's view of the three cross-section tables.

    Rebuilt only when ``_cross``'s lazily-built excitation table is replaced
    (it is cached there and only rebuilt for a different ``n_max``, which
    ``deposit_beam`` never asks for), so the steady state is an identity test.
    The log-log EII table is a module constant in ``_cross`` and never moves.
    """
    global _CSDA_TABLES, _CSDA_TABLES_SRC
    src = _he_beam_excitation_table(20)
    if _CSDA_TABLES_SRC is not src:
        _CSDA_TABLES = _COMPILED_KERNELS.CsdaTables(
            _HE_LOG_EPS, _HE_LOG_SIGMA, src[1], src[2], src[3]
        )
        _CSDA_TABLES_SRC = src
    return _CSDA_TABLES


def _coulomb_stopping_coefficient(ne, Te, model):
    """Per-cell ``A`` in ``dE/dx = A * W**p`` [eV/cm], from the module's own
    ``coulomb_stopping_eV_per_cm``.

    Evaluated at ``W = 1 eV``, where ``W**p == 1`` for either exponent, so the
    coefficient carries the closure's constants, ``ne``, ``Te`` and lnLambda
    exactly as the primary's stopping power does -- the walk cannot drift away
    from the primary's drag law without ``coulomb_stopping_eV_per_cm`` itself
    changing shape (which the smoke suite checks).
    """
    return np.array(
        [
            coulomb_stopping_eV_per_cm(1.0, float(n), float(t), model=model)
            for n, t in zip(np.asarray(ne).ravel(), np.asarray(Te).ravel())
        ],
        dtype=float,
    )


#: Empty cull tally, in the shape :func:`_walk_products_forward` returns.
_NO_CULL = (
    np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.intp),
    np.zeros(0, dtype=np.intp),
)


def _walk_products_forward(
    W0_eV, flux_per_s, coeff, dz_cm, floor_eV, q, cull=None
):
    """Coulomb slowing walk of per-cell product populations toward +index.

    ``W0_eV[s]`` / ``flux_per_s[s]`` are the birth energy and flux of the
    population born in cell ``s``; the walk proceeds s, s+1, ... Returns
    ``(deposited_eV_per_s, exit_eV_per_s, exit_state)``: per-cell deposited
    power in ``flux*eV`` units, the scalar power carried out of the last cell,
    and ``exit_state = (birth_cells, W_exit_eV, thermalized)`` -- the PER-BIRTH
    breakdown of that scalar, which a reflecting boundary needs because the
    populations arrive at the face with different energies and only some of
    them may be below its threshold. ``exit_state`` is pure bookkeeping: it
    re-reads quantities the walk already formed and no caller that ignores it
    sees any difference.

    The integration is EXACT, not substepped. With ``dE/dx = A_j W**p`` held
    constant across a cell, ``u = W**q`` with ``q = 1 - p`` obeys
    ``du/dz = -q A_j``, so ``u`` falls linearly and a cumulative sum gives the
    energy at every cell exit in closed form. Deposition in a cell is the
    difference of the entry and exit energies, and the entry energy of a cell
    is REUSED as the exit energy of its predecessor (the identical float), so
    the per-ray sum telescopes to the birth energy exactly rather than to
    accumulated roundoff.

    ``cull=(slots, eta)`` arms an OBSTRUCTION at the given traversal slots: a
    population that reaches the FIRST slot at or after its birth loses the
    fraction ``eta`` of its flux there, and the survivors carry ``1 - eta``
    through that slot and every later one (the obstruction fires once per
    walker -- first crossing only). ``slots`` must be sorted ascending; two
    entries describe one plane met twice by an unfolded reflected path, of
    which only the first is a crossing. The removed share is reported rather
    than deposited: the third return value becomes
    ``(culled_flux_per_s, W_at_crossing_eV, birth_slots, crossing_slots)``, one
    entry per population that actually crossed, so the caller books the flux,
    its arrival energy and WHICH crossing it was (an unfolded reflected path
    meets the plane in either of its two blocks, and the incident direction
    differs between them) on its own convention. ``cull=None`` (the default) is
    the historical walk and takes the historical arithmetic line for line.
    """
    cells = int(dz_cm.size)
    dep = np.zeros(cells)
    W0_eV = np.asarray(W0_eV, dtype=float)
    flux_per_s = np.asarray(flux_per_s, dtype=float)
    active = np.flatnonzero((flux_per_s > 0.0) & (W0_eV > 0.0))
    if active.size == 0:
        return (
            dep, 0.0, (active, np.zeros(0), np.zeros(0, dtype=bool)), _NO_CULL,
        )
    W0v = W0_eV[active]
    fluxv = flux_per_s[active]
    index = np.arange(cells)
    reach = index[None, :] >= active[:, None]
    # u after LEAVING each cell, per birth cell (cumulative sum starts at the
    # birth cell, so cells behind it contribute nothing and read back u = W0**q).
    delta = q * coeff * dz_cm
    u_out = np.power(W0v, q)[:, None] - np.cumsum(
        np.where(reach, delta[None, :], 0.0), axis=1
    )
    W_out = np.power(np.maximum(u_out, 0.0), 1.0 / q)
    W_in = np.empty_like(W_out)
    W_in[:, 0] = W0v
    W_in[:, 1:] = W_out[:, :-1]
    # The birth cell's entry energy is the birth energy by definition; pin it
    # rather than trusting (W0**q)**(1/q) to round back onto W0.
    W_in[np.arange(active.size), active] = W0v
    # Thermalized where the cell-exit energy has fallen to the local floor.
    stop = reach & (W_out <= floor_eV[None, :])
    has_stop = stop.any(axis=1)
    j_stop = np.where(has_stop, np.argmax(stop, axis=1), cells)
    before = reach & (index[None, :] < j_stop[:, None])
    at = reach & (index[None, :] == j_stop[:, None])
    contrib = np.where(before, W_in - W_out, 0.0) + np.where(at, W_in, 0.0)
    if cull is None:
        dep += (fluxv[:, None] * contrib).sum(axis=0)
        exit_eV = float(
            np.sum(fluxv * np.where(has_stop, 0.0, W_out[:, -1]))
        )
        return dep, exit_eV, (active, W_out[:, -1], has_stop), _NO_CULL
    slots, eta_cull = cull
    slots = np.asarray(slots, dtype=np.intp)
    # The first obstruction slot at or after each birth. A population born
    # PAST the last slot never meets one; a population that thermalizes short
    # of its slot never reaches it, and its flux is already gone.
    pos = np.searchsorted(slots, active, side="left")
    met = pos < slots.size
    slot_of = np.where(met, slots[np.minimum(pos, max(slots.size - 1, 0))], cells)
    crosses = met & (j_stop >= slot_of)
    survivor = np.ones((active.size, cells))
    rows = np.flatnonzero(crosses)
    if rows.size:
        survivor[rows] = np.where(
            index[None, :] >= slot_of[rows][:, None], 1.0 - eta_cull, 1.0
        )
    dep += (fluxv[:, None] * survivor * contrib).sum(axis=0)
    exit_eV = float(
        np.sum(
            fluxv * survivor[:, -1] * np.where(has_stop, 0.0, W_out[:, -1])
        )
    )
    return (
        dep,
        exit_eV,
        (active, W_out[:, -1], has_stop),
        (
            eta_cull * fluxv[rows],
            W_in[rows, slot_of[rows]] if rows.size else np.zeros(0),
            active[rows],
            slot_of[rows],
        ),
    )


def _walk_products(
    W0_eV, flux_per_s, direction, coeff, dz_cm, floor_eV, q, cull=None
):
    """Direction-aware wrapper around ``_walk_products_forward``.

    Returns ``(deposited_eV_per_s, exit_eV_per_s, exit_flux_per_s, cull)``; the
    exit power and flux leave the HIGH-index end for ``direction > 0`` and the
    LOW-index end otherwise. The exit FLUX is the population behind that power
    -- the walkers that reached the end without thermalizing, less whatever an
    armed obstruction removed -- read off the forward walk's own per-birth exit
    state, so it cannot disagree with the energy about who left. The rest of
    the per-birth state is dropped; the reflecting walk below indexes cells in
    its own traversal order and calls the forward walk directly.

    ``cull=(cell_indices, eta)`` names the obstruction in CELL indices; this
    wrapper maps them into the traversal order the walk actually runs in. The
    returned cull tally's birth cells are likewise mapped back to cell indices.
    """
    cells = int(np.asarray(dz_cm).size)
    if direction > 0:
        fwd_cull = (
            None if cull is None
            else (np.sort(np.asarray(cull[0], dtype=np.intp)), cull[1])
        )
        flux_ordered = np.asarray(flux_per_s, dtype=float)
        dep, exit_eV, (active, _W_exit, thermalized), tally = (
            _walk_products_forward(
                W0_eV, flux_per_s, coeff, dz_cm, floor_eV, q, cull=fwd_cull
            )
        )
        survive = np.ones(active.size)
        if cull is not None and tally[2].size:
            survive[np.searchsorted(active, tally[2])] = 1.0 - cull[1]
        return (
            dep,
            exit_eV,
            float(np.sum(flux_ordered[active][~thermalized]
                         * survive[~thermalized])),
            tally,
        )
    rev_cull = (
        None if cull is None
        else (np.sort(cells - 1 - np.asarray(cull[0], dtype=np.intp)), cull[1])
    )
    flux_ordered = np.asarray(flux_per_s, dtype=float)[::-1]
    dep, exit_eV, (active, _W_exit, thermalized), tally = (
        _walk_products_forward(
            W0_eV[::-1], flux_per_s[::-1], coeff[::-1], dz_cm[::-1],
            floor_eV[::-1], q, cull=rev_cull,
        )
    )
    survive = np.ones(active.size)
    if cull is not None and tally[2].size:
        survive[np.searchsorted(active, tally[2])] = 1.0 - cull[1]
    return (
        dep[::-1],
        exit_eV,
        float(np.sum(flux_ordered[active][~thermalized]
                     * survive[~thermalized])),
        (tally[0], tally[1], cells - 1 - tally[2], cells - 1 - tally[3]),
    )


def beam_speed_cm_s(E_eV: float) -> float:
    """Nonrelativistic primary speed [cm/s] at kinetic energy E_eV."""
    return math.sqrt(2.0 * E_eV * _ERG_PER_EV / _ME_CGS)


def he_mean_secondary_energy_eV(
    E_eV: float,
    I_ion_eV: float = HE_I_ION_EV,
    ebar_eV: float = HE_OPB_EBAR_EV,
) -> float:
    """Mean secondary-electron energy [eV] per He ionization at E_eV.

    OPB shape over W in [0, (E - I)/2]:
    <W> = Ebar * ln(1 + x^2) / (2 atan(x)), x = W_max / Ebar.
    """
    W_max = 0.5 * (E_eV - I_ion_eV)
    if W_max <= 0.0:
        return 0.0
    x = W_max / ebar_eV
    return ebar_eV * math.log1p(x * x) / (2.0 * math.atan(x))


def coulomb_stopping_eV_per_cm(
    E_eV: float, ne: float, Te: float, model: str = "fast_electron"
) -> float:
    """Coulomb energy-loss rate [eV/cm] of a primary at E_eV."""
    if ne <= 0.0 or E_eV <= 0.0:
        return 0.0
    lnL = _c_log_ei(max(Te, 0.1), ne)
    if model == "fast_electron":
        return 2.0 * math.pi * _E4_CGS * ne * lnL / (E_eV * _ERG_PER_EV) / _ERG_PER_EV
    if model == "legacy_tau_ei":
        tau_ei = 3.44e5 * Te**1.5 / ne / lnL
        return E_eV / (beam_speed_cm_s(E_eV) * tau_ei)
    raise ValueError(
        f"unknown coulomb_model {model!r}; "
        "expected 'fast_electron' or 'legacy_tau_ei'"
    )


def quasilinear_relaxation_length_cm(
    E_eV: float, ne: float, n_b: float
) -> float:
    """Quasilinear beam relaxation length [cm], l_QL = (ne/nb)(v_b/w_pe)ln(ne/nb).

    Valid for a weak beam (n_b << n_e); returns inf outside that domain
    (n_b >= n_e/10) rather than extrapolating the theory into the strong-beam
    regime, and inf for an absent beam or plasma.
    """
    if ne <= 0.0 or n_b <= 0.0 or n_b >= 0.1 * ne:
        return math.inf
    omega_pe = _OMEGA_PE_COEFF * math.sqrt(ne)
    ratio = ne / n_b
    return ratio * (beam_speed_cm_s(E_eV) / omega_pe) * math.log(ratio)


def beam_plasma_growth_rate_s(ne: float, n_b: float) -> float:
    """Cold-beam beam-plasma (bump-on-tail) growth rate [1/s].

    ``gamma_r = 0.687 * omega_pe * min(n_b/n_e, 1)**(1/3)`` with
    ``omega_pe = 5.64e4 sqrt(n_e)``. Zero for an absent beam or plasma.

    The ``min(., 1)`` cap keeps the expression finite where ``n_b`` approaches
    or exceeds ``n_e``; there the cold-beam derivation no longer applies at all
    and the cap holds the rate at ``0.687 omega_pe`` rather than continuing a
    curve past its own domain. The cap is a FLAGGED INFERENCE and travels as one
    -- it is not part of the cited result, which is the ``(n_b/n_e)**(1/3)``
    scaling in the weak-beam limit.
    """
    if ne <= 0.0 or n_b <= 0.0:
        return 0.0
    omega_pe = _OMEGA_PE_COEFF * math.sqrt(ne)
    return QL_GROWTH_COEFF * omega_pe * min(n_b / ne, 1.0) ** (1.0 / 3.0)


def ql_onset_open(ne: float, nn: float, Te_eV: float, n_b: float) -> bool:
    """Whether the boxed beam-plasma onset inequality is satisfied locally.

    Two conditions, both evaluated on the LOCAL cell state:

    * growth beats damping -- ``gamma_r > 0.5 * nu_en``, the half being the
      conversion from the momentum-transfer collision frequency to the
      AMPLITUDE damping rate of the Langmuir wave the instability drives;
    * the wave exists as a wave -- ``omega_pe > nu_en``, i.e. the oscillation
      completes before it is collisionally interrupted.

    ``nu_en = nn * K_m(Te)`` uses the boxed He e-n momentum-transfer rate
    coefficient (``_cross.he_electron_momentum_transfer_rate_cm3_s``).

    Expected permanently OPEN over the working range: onset is not the gating
    physics, relaxation is. It is evaluated anyway, and gates the booking, so
    that the statement is a computed property of each cell rather than an
    assertion made once about a range.
    """
    if ne <= 0.0 or n_b <= 0.0:
        return False
    omega_pe = _OMEGA_PE_COEFF * math.sqrt(ne)
    nu_en = max(nn, 0.0) * float(
        he_electron_momentum_transfer_rate_cm3_s(max(Te_eV, 0.0))
    )
    if not omega_pe > nu_en:
        return False
    return beam_plasma_growth_rate_s(ne, n_b) > 0.5 * nu_en


def landau_branching_fraction(ne, Te_eV, nn, E_beam_eV):
    """Landau share of the driven Langmuir-wave energy, per cell (in [0, 1]).

    ``f_Landau = gamma_L / (gamma_L + nu_en/2)`` -- the fraction of the wave
    energy the beam-plasma instability drives that is damped on the RESONANT
    electrons (Landau, which makes a fast tail) rather than on the wave itself
    (electron-neutral collisions, which make bulk heat). ``1 - f_Landau`` is
    the collisional share. Dimensionless; the two rates are both amplitude
    damping rates [1/s], so their units cancel.

    ``nu_en/2 = 0.5 * nn * K_m(Te)`` is the collisional Langmuir amplitude
    damping the onset gate already weighs against
    (:func:`ql_onset_open`), on the boxed He e-n momentum-transfer rate
    coefficient ``_cross.he_electron_momentum_transfer_rate_cm3_s``.

    ``gamma_L`` is the Maxwellian Landau damping rate evaluated at the
    beam-resonant phase velocity ``v_phi = v_b`` (the wave the beam drives has
    ``k = omega/v_b``), Krall & Trivelpiece Sec. 8.6 with the Bohm-Gross
    ``-3/2`` term::

        gamma_L = sqrt(pi/8) * omega_pe * (v_phi/v_te)**3
                  * exp(-v_phi**2 / (2 v_te**2) - 3/2)

    with ``omega_pe = 5.64e4 sqrt(ne)`` and ``v_te**2 = Te/m_e``, so
    ``(v_phi/v_te)**2 = 2 E_beam / Te`` in eV. NO new physical constant enters:
    ``omega_pe``'s coefficient and ``K_m`` are the module's existing boxed
    inputs and the rest is the cited formula's own arithmetic.

    ``E_beam_eV`` is the resonant beam energy [eV] -- the caller's launch
    energy; slowing along the column is not tracked in ``v_phi``.

    **Validity caveat, stated because the formula is used as-is:** the
    asymptotic Landau expression is a large-argument expansion and is
    quantitative for ``v_phi/v_te`` greater than roughly 2.4; below that the
    expansion is marginal and the value should be read as indicative. It is
    never extrapolated into a regime it changes sign in -- it is positive
    everywhere and monotone in ``Te`` at fixed ``E_beam``.

    Corner behaviour, all of it the collisional limit and none of it a
    fallback: where the exponential underflows (a cold or slow-tail cell) the
    Landau channel is genuinely dead and ``f_Landau`` is 0.0; where BOTH rates
    vanish (no plasma and no neutrals, i.e. no wave and no damping at all) the
    ratio is undefined and 0.0 is returned, which routes such a cell's power
    exactly where the shipped ``"local"`` closure routes it.

    Accepts per-cell arrays of ``ne``, ``Te_eV`` and ``nn`` sharing one shape
    and a scalar ``E_beam_eV``; returns an array of that shape.
    """
    ne = np.asarray(ne, dtype=float)
    Te = np.asarray(Te_eV, dtype=float)
    nn = np.asarray(nn, dtype=float)
    # (v_phi/v_te)**2. The positive floor on Te is an ARITHMETIC guard on the
    # division, not a temperature: the exponential below is already exactly
    # 0.0 many decades above it, so every value it can produce is the
    # collisional limit whatever the floor is set to.
    r2 = 2.0 * float(E_beam_eV) / np.maximum(Te, 1.0e-12)
    gamma_L = (
        _LANDAU_DAMPING_COEFF
        * _OMEGA_PE_COEFF
        * np.sqrt(np.maximum(ne, 0.0))
        * r2 ** 1.5
        * np.exp(-0.5 * r2 - 1.5)
    )
    nu_half = 0.5 * np.maximum(nn, 0.0) * (
        he_electron_momentum_transfer_rate_cm3_s(np.maximum(Te, 0.0))
    )
    total = gamma_L + nu_half
    # The 0/0 cell (no plasma AND no neutrals) is masked out of the division
    # itself rather than divided and repaired, so no invalid-value warning is
    # raised and no NaN is ever formed.
    live = total > 0.0
    return np.where(live, gamma_L / np.where(live, total, 1.0), 0.0)


def ql_trapped_fraction(ne: float, n_b: float) -> float:
    """Beam-energy fraction reactive trapping extracts per relaxation.

    ``f_ext = C_trap * min(n_b/(2 n_e), 1)**(1/3)`` -- the O'Neil, Winfrey &
    Malmberg (1971) trapping scaling, with the same flagged-inference cap the
    growth rate carries. Zero for an absent beam or plasma, and never above
    ``C_trap``.
    """
    if ne <= 0.0 or n_b <= 0.0:
        return 0.0
    return QL_TRAP_COEFF * min(n_b / (2.0 * ne), 1.0) ** (1.0 / 3.0)


def ql_relaxation_length_cm(
    E_eV: float, ne: float, n_b: float, coeff: float
) -> float:
    """Plateau-formation length [cm], ``L_rel = tau_QL * v_b``.

    ``tau_QL = coeff * (n_e/n_b) / omega_pe`` is the Vedenov-era quasilinear
    plateau-formation time with ``coeff`` the registered O(10-100) bracket
    constant (``ql_relaxation_coeff``); the beam covers ``v_b`` per unit time
    while it forms, so the extracted power is spread over that length rather
    than dumped at a point. Returns inf for an absent beam or plasma.

    Note what this does NOT contain: the ``ln(n_e/n_b)`` of
    :func:`quasilinear_relaxation_length_cm`. That factor is the fiat closure's
    own estimate of the same length; here the order-unity content of the
    estimate is carried explicitly by ``coeff``, whose bracket is reported with
    every result, instead of being fixed by a logarithm.
    """
    if ne <= 0.0 or n_b <= 0.0 or coeff <= 0.0:
        return math.inf
    omega_pe = _OMEGA_PE_COEFF * math.sqrt(ne)
    return coeff * (ne / n_b) * beam_speed_cm_s(E_eV) / omega_pe


def ql_relaxation_stopping_eV_per_cm(
    E_eV: float, ne: float, nn: float, Te_eV: float, n_b: float, coeff: float
) -> float:
    """Anomalous stopping [eV/cm] of the ``ql_relaxation`` closure.

    ``dE/dx = f_ext * E / L_rel`` -- the trapped fraction of the beam energy,
    delivered over the plateau-formation length. Identically 0.0 wherever the
    boxed onset inequality is not satisfied (:func:`ql_onset_open`), which is
    what makes the gate a property of the booking rather than a claim about it.

    The extracted energy leaves the beam here and is banked as BULK electron
    heating in the same cell: the beam's energy goes into Langmuir waves, and
    the waves hand it to the bulk by collisional damping at ``nu_en/2``, which
    is a bulk channel and not a tail one. Energy conservation is by
    construction -- the banked decrement and the beam's own energy decrement are
    the same product -- so extracted + retained-and-carried-out = launched.
    """
    if E_eV <= 0.0:
        return 0.0
    if not ql_onset_open(ne, nn, Te_eV, n_b):
        return 0.0
    L_rel = ql_relaxation_length_cm(E_eV, ne, n_b, coeff)
    if not (math.isfinite(L_rel) and L_rel > 0.0):
        return 0.0
    return ql_trapped_fraction(ne, n_b) * E_eV / L_rel


# --- The tail-walk legs, marched as LANES (R3) -------------------------------
# The ionizing tail walk gives every (plateau group, birth cell, direction) its
# own CSDA march, and a second march per reflection. Every one of those legs
# runs the SAME column in the SAME configuration and none of them interacts
# with any other, so they are marched together by
# ``cathode.beam_lane_march.lane_march`` -- one numpy call per quantity per
# round instead of one Python call per quantity per leg per substep. The legs
# are bit-identical either way (that module's docstring says by what), so which
# route runs is a COST decision and nothing else; the two constants below are
# where it is made, and the recursive route stays as the equivalence target and
# as the implementation for everything the lane march refuses.


#: Fewest legs worth batching. Below it the per-round numpy dispatch costs
#: more than the scalar substeps it replaces, so the recursive route is the
#: cheaper one. A cost threshold, not a correctness one: both routes produce
#: the same floats, which is what lets this be tuned without a recapture.
#:
#: MEASURED, on the reference corpus's own real deposition rays: the two routes
#: cross near 96 legs, so the value sits above the crossing with margin rather
#: than on it -- a batch just past a mis-set threshold pays the dispatch and
#: gets nothing back, and the corpus's main-discharge ray is exactly that case
#: at 64 legs. Re-measure with scripts/verify/r3lane_equivalence.py's timing companion
#: after any change to the round's op count.
LANE_MARCH_MIN_LEGS = 128

#: Substeps and legs the lane march absorbed since the counter was last reset.
#: The instruments that census the CSDA march count substeps by hooking the
#: per-substep excitation lookup and legs by hooking the recursive
#: ``deposit_beam`` call; a batched leg passes through neither hook, so it
#: reports itself here instead. Read-only bookkeeping -- nothing in the march
#: consumes it.
LANE_MARCH_COUNTS = {"substeps": 0, "legs": 0}


def _tail_anode_take(culled_flux, W_cross, R_e, eta_E):
    """Split one crossing tally into what the anode keeps and what returns.

    ``culled_flux[k]`` is the walker flux [1/s] the mesh removed from crossing
    ``k`` and ``W_cross[k]`` the energy [eV] that flux was carrying there.
    Returns the four scalars
    ``(culled_flux, culled_eV, returned_flux, returned_eV)`` -- fluxes in 1/s
    and energies in ``flux*eV``, which the caller converts. Only crossings at or
    above :data:`TAIL_ANODE_RIDER_MIN_ENERGY_EV` contribute to the return; below
    it the walker is ABSORBED whatever the pair says, so its whole culled share
    stays with the anode. ``R_e = 0`` (the rider off) returns zeros, which is
    the same statement with no threshold to test.
    """
    culled_flux = np.asarray(culled_flux, dtype=float)
    W_cross = np.asarray(W_cross, dtype=float)
    gross_flux = float(culled_flux.sum())
    gross_eV = float((culled_flux * W_cross).sum())
    if R_e <= 0.0 or culled_flux.size == 0:
        return gross_flux, gross_eV, 0.0, 0.0
    boxed = W_cross >= TAIL_ANODE_RIDER_MIN_ENERGY_EV
    return (
        gross_flux,
        gross_eV,
        float(R_e * culled_flux[boxed].sum()),
        float(eta_E * (culled_flux[boxed] * W_cross[boxed]).sum()),
    )


def _tail_lane_chains(
    plans, nn_w, ne_w, Te_w, dz_w, march_kwargs, tail_lo, tail_hi,
    reflect_face, E_reflect, cull=None,
):
    """The ionizing tail-walk legs of every population, in the legs' own order.

    ``plans`` is one ``(E_walk_eV, half_flux, ionizes)`` per launched
    population. Returns ``(layout, tally)``. ``layout`` is a list parallel to
    ``plans``: ``None`` for a population that does not ionize, otherwise a list
    of CHAINS -- one per (birth cell, direction) walker, each chain holding the
    legs that walker marched (one, or two when it reflected at the walk
    window's reflecting face), plus one further chain per REVERSED walker the
    anode rider launched. A leg is
    ``(banks, transmitted_flux, transmitted_energy_eV, direction)`` where
    ``banks`` is the five window-shaped arrays
    ``(ionization_events, excitation_events, ionization_cost_erg_s,
    radiated_erg_s, plasma_heating_erg_s)`` the caller banks; the caller books
    the escape of each chain's LAST leg and nothing else.

    ``tally`` is the four-scalar anode take of
    :func:`_tail_anode_take`, summed over every walker.

    ``cull=(local_cell, eta, R_e, eta_E)`` arms the anode cull on these legs;
    ``None`` leaves them exactly as they always marched.

    The order is the recursive route's exactly -- population, then birth cell
    ascending, then ``+1`` before ``-1``, then the reflected leg after the leg
    that produced it -- because the caller replays it into shared banks and
    float addition is not associative.
    """
    lanes_E0 = []
    lanes_flux = []
    lanes_launch = []
    lanes_dir = []
    layout = []
    for slot, (E_walk, half_flux, ionizes) in enumerate(plans):
        if not ionizes:
            layout.append(None)
            continue
        chains = []
        for birth in np.flatnonzero(half_flux > 0.0):
            for walk_direction in (1, -1):
                chains.append(len(lanes_E0))
                lanes_E0.append(float(E_walk))
                lanes_flux.append(float(half_flux[birth]))
                lanes_launch.append(int(birth) - tail_lo)
                lanes_dir.append(walk_direction)
        layout.append(chains)
    if not lanes_E0:
        return [None] * len(plans), (0.0, 0.0, 0.0, 0.0)

    use_lanes = (
        # The anode cull is a per-lane FLUX event at one cell, and the batched
        # marcher has no interception of its own. The armed arm takes the
        # recursive route instead -- the two agree bit for bit, so this is a
        # cost choice, and it costs only the arm that arms the cull.
        cull is None
        and
        # THE PURE PATH'S ROUTE. With the compiled march bound, a leg's whole
        # substep loop already runs in C at a fraction of a numpy round's
        # dispatch cost, and batching it is a large REGRESSION (measured 5.1 ms
        # -> 27.1 ms on the corpus's breakdown ray). The two routes agree bit
        # for bit, so which one a process takes is purely a cost question, and
        # the answer differs by whether a kernel is loaded.
        _CSDA_MARCH is None
        and len(lanes_E0) >= LANE_MARCH_MIN_LEGS
        and march_kwargs["coulomb_model"] in LANE_MARCH_MODELS
        and march_kwargs["I_ion_eV"] > 0.0
        and march_kwargs["E_stop_eV"] > 0.0
        and max(lanes_E0) < lane_march_energy_ceiling_eV()
    )
    if not use_lanes:
        return _tail_recursive_chains(
            plans, nn_w, ne_w, Te_w, dz_w, march_kwargs, tail_lo, tail_hi,
            reflect_face, E_reflect, cull=cull,
        )

    first = lane_march(
        np.array(lanes_E0, dtype=float),
        np.array(lanes_flux, dtype=float),
        np.array(lanes_launch, dtype=np.intp),
        np.array(lanes_dir, dtype=np.intp),
        nn_w, ne_w, Te_w, dz_w,
        I_ion_eV=march_kwargs["I_ion_eV"],
        E_stop_eV=march_kwargs["E_stop_eV"],
        coulomb_model=march_kwargs["coulomb_model"],
        max_energy_fraction_per_substep=march_kwargs[
            "max_energy_fraction_per_substep"
        ],
    )
    LANE_MARCH_COUNTS["legs"] += len(lanes_E0)
    LANE_MARCH_COUNTS["substeps"] += int(first.substeps.sum())
    # The bounce wave. Only the ONE named face reflects, so a reflected leg
    # heads at the other one and cannot bounce again; two waves close the loop.
    bounce_src = []
    if reflect_face is not None:
        for k in range(len(lanes_E0)):
            if (
                lanes_dir[k] == reflect_face
                and first.transmitted_flux[k] > 0.0
                and first.transmitted_energy_eV[k] < E_reflect
            ):
                bounce_src.append(k)
    second = None
    if bounce_src:
        bounce_launch = 0 if reflect_face < 0 else tail_hi - tail_lo
        second = lane_march(
            first.transmitted_energy_eV[bounce_src],
            first.transmitted_flux[bounce_src],
            np.full(len(bounce_src), bounce_launch, dtype=np.intp),
            np.array([-lanes_dir[k] for k in bounce_src], dtype=np.intp),
            nn_w, ne_w, Te_w, dz_w,
            I_ion_eV=march_kwargs["I_ion_eV"],
            E_stop_eV=march_kwargs["E_stop_eV"],
            coulomb_model=march_kwargs["coulomb_model"],
            max_energy_fraction_per_substep=march_kwargs[
                "max_energy_fraction_per_substep"
            ],
        )
        LANE_MARCH_COUNTS["legs"] += len(bounce_src)
        LANE_MARCH_COUNTS["substeps"] += int(second.substeps.sum())
    bounce_of = {k: n for n, k in enumerate(bounce_src)}

    def _leg(res, row, direction):
        return (
            (
                res.ionization_events[row],
                res.excitation_events[row],
                res.ionization_cost_erg_s[row],
                res.radiated_erg_s[row],
                res.plasma_heating_erg_s[row],
            ),
            float(res.transmitted_flux[row]),
            float(res.transmitted_energy_eV[row]),
            direction,
        )

    out = []
    for chains in layout:
        if chains is None:
            out.append(None)
            continue
        built = []
        for k in chains:
            chain = [_leg(first, k, lanes_dir[k])]
            if k in bounce_of:
                chain.append(
                    _leg(second, bounce_of[k], -lanes_dir[k])
                )
            built.append(chain)
        out.append(built)
    return out, (0.0, 0.0, 0.0, 0.0)


def _tail_recursive_chains(
    plans, nn_w, ne_w, Te_w, dz_w, march_kwargs, tail_lo, tail_hi,
    reflect_face, E_reflect, cull=None,
):
    """:func:`_tail_lane_chains`'s contract, one recursive march per leg.

    The equivalence target: one ``deposit_beam`` call per leg, exactly as the
    ionizing tail walk has always made them, and the route taken whenever the
    lane march is not worth its dispatch or does not cover the configuration --
    including whenever the anode cull is armed, which the batched marcher does
    not carry.

    With ``cull=(local_cell, eta, R_e, eta_E)`` each leg is marched with the
    module's OWN anode interception (``anode_cross_index``/``anode_eta``), which
    is the same event, the same convention and the same arithmetic the primary
    ray has used since A15 -- the cull is not re-derived here. It is armed
    until it fires: a walker born gap-side of the plane may only meet it after
    reflecting, and one that met it on its first leg must not be culled twice.
    The energy the cull removed is read back from the leg's own
    ``anode_intercepted_erg_s`` and the flux behind it from that energy and the
    leg's recorded entry energy at the plane, which is the pair the rider and
    the anode current are formed from.
    """
    out = []
    tally = [0.0, 0.0, 0.0, 0.0]
    if cull is None:
        cull_local = -1
        cull_eta = R_e = eta_E = 0.0
        cull_kwargs = {}
    else:
        cull_local, cull_eta, R_e, eta_E = cull
        cull_kwargs = dict(
            anode_cross_index=int(cull_local), anode_eta=float(cull_eta)
        )
    n_w = tail_hi - tail_lo + 1
    for E_walk, half_flux, ionizes in plans:
        if not ionizes:
            out.append(None)
            continue
        built = []
        for birth in np.flatnonzero(half_flux > 0.0):
            for walk_direction in (1, -1):
                leg_dir = walk_direction
                armed = cull is not None
                leg = deposit_beam(
                    E_walk, float(half_flux[birth]), nn_w, ne_w, Te_w,
                    int(birth) - tail_lo, leg_dir, dz_w, **march_kwargs,
                    **(cull_kwargs if armed else {}),
                )
                chain = []
                riders = []
                while True:
                    if armed and float(leg.anode_intercepted_erg_s) > 0.0:
                        # The cull fired on this leg. ``anode_intercepted`` is
                        # ``eta * gamma * E`` at the plane and ``E_entry`` at
                        # that cell is the same ``E``, so the removed FLUX is
                        # the one division that separates them.
                        armed = False
                        _E_cross = float(leg.E_entry_eV[cull_local])
                        _f_cull = (
                            float(leg.anode_intercepted_erg_s)
                            / (_E_cross * _ERG_PER_EV)
                        )
                        _g_f, _g_eV, _r_f, _r_eV = _tail_anode_take(
                            np.array([_f_cull]), np.array([_E_cross]),
                            R_e, eta_E,
                        )
                        tally[0] += _g_f
                        tally[1] += _g_eV
                        tally[2] += _r_f
                        tally[3] += _r_eV
                        if _r_f > 0.0:
                            # The reversed walker: launched from the plane cell
                            # back the way it came, at the flux-weighted mean
                            # energy the returned share carries. It is a chain
                            # of its own -- its escape is its own, and it does
                            # NOT meet the plane again (first crossing only).
                            riders.append(
                                (cull_local, -leg_dir, _r_f, _r_eV / _r_f)
                            )
                    if (
                        reflect_face is not None
                        and leg_dir == reflect_face
                        and float(leg.transmitted_flux) > 0.0
                        and float(leg.transmitted_energy_eV) < E_reflect
                    ):
                        _next = (
                            float(leg.transmitted_flux),
                            float(leg.transmitted_energy_eV),
                        )
                    else:
                        _next = None
                    chain.append(
                        (
                            (
                                leg.ionization_events,
                                leg.excitation_events,
                                leg.ionization_cost_erg_s,
                                leg.radiated_erg_s,
                                leg.plasma_heating_erg_s,
                            ),
                            float(leg.transmitted_flux),
                            float(leg.transmitted_energy_eV),
                            leg_dir,
                        )
                    )
                    if _next is None:
                        break
                    leg_flux, leg_E = _next
                    leg_dir = -leg_dir
                    leg = deposit_beam(
                        leg_E, leg_flux, nn_w, ne_w, Te_w,
                        0 if reflect_face < 0 else n_w - 1,
                        leg_dir, dz_w, **march_kwargs,
                        **(cull_kwargs if armed else {}),
                    )
                built.append(chain)
                for r_cell, r_dir, r_flux, r_E in riders:
                    built.append(
                        _tail_rider_chain(
                            r_cell, r_dir, r_flux, r_E, nn_w, ne_w, Te_w,
                            dz_w, march_kwargs, n_w, reflect_face, E_reflect,
                        )
                    )
        out.append(built)
    return out, tuple(tally)


def _tail_rider_chain(
    cell, direction, flux, E_eV, nn_w, ne_w, Te_w, dz_w, march_kwargs, n_w,
    reflect_face, E_reflect,
):
    """March one reversed walker the anode rider launched, as its own chain.

    Same legs, same banks and same reflecting-face convention as a born
    walker's chain -- the rider changes where a walker starts, not how it
    walks. It never meets the anode plane again: the cull is first-crossing
    only, and this walker's crossing is the one that made it.
    """
    leg_dir = int(direction)
    leg = deposit_beam(
        float(E_eV), float(flux), nn_w, ne_w, Te_w, int(cell), leg_dir, dz_w,
        **march_kwargs,
    )
    chain = []
    while True:
        chain.append(
            (
                (
                    leg.ionization_events,
                    leg.excitation_events,
                    leg.ionization_cost_erg_s,
                    leg.radiated_erg_s,
                    leg.plasma_heating_erg_s,
                ),
                float(leg.transmitted_flux),
                float(leg.transmitted_energy_eV),
                leg_dir,
            )
        )
        if (
            reflect_face is not None
            and leg_dir == reflect_face
            and chain[-1][1] > 0.0
            and chain[-1][2] < E_reflect
        ):
            leg_flux, leg_E = chain[-1][1], chain[-1][2]
            leg_dir = -leg_dir
            leg = deposit_beam(
                leg_E, leg_flux, nn_w, ne_w, Te_w,
                0 if reflect_face < 0 else n_w - 1,
                leg_dir, dz_w, **march_kwargs,
            )
            continue
        break
    return chain


def _tail_band(E_walk_eV, I_ion_eV, E_stop_eV, label):
    """K7b band treatment for ONE walker energy: ``(ionize, sub, above)``.

    The two depth-1 bars are COMPUTED from the thresholds themselves, and each
    selects a treatment rather than refusing (module docstring, K7b):

    * at or below ``E_stop_eV`` no He inelastic channel is open, so the
      ionizing march REVERTS to the energy-only walk -- exact physics, not a
      fallback, and bit-identical to what ``tail_ionization="off"`` would do;
    * above the ``<W_sec>`` crossing the march RUNS under the depth-1
      truncation, whose understatement is measured (<= 2.0%) and reported;
    * above the tabulated He EII edge it RAISES -- there the lookup clamps to
      its last node and the walk would attenuate on an extrapolated cross
      section. The edge itself is INCLUSIVE within ``HE_EII_EDGE_REL_TOL``
      (K7c): AT the edge the clamped value IS the table's endpoint, so nothing
      is extrapolated and there is nothing to refuse.

    ``label`` names the caller's quantity in that refusal, so a single-energy
    ray blames ``tail_energy_eV`` and a plateau group blames its own midpoint.
    """
    _E_table_top = HE_EII_EPS_TOP * I_ion_eV
    _edge_excess = (E_walk_eV - _E_table_top) / _E_table_top
    if _edge_excess > HE_EII_EDGE_REL_TOL:
        raise ValueError(
            "tail_ionization='on' marches the walkers on the tabulated "
            "He EII cross section, which ends at eps = E/I_ion = "
            f"{HE_EII_EPS_TOP:.6f} (i.e. "
            f"{_E_table_top:.2f} eV at I_ion_eV={I_ion_eV}); "
            f"at {label}={E_walk_eV} eV the lookup would clamp to its "
            "last node and the walk would attenuate on an extrapolated "
            "cross section. This is refused, not approximated (relative "
            f"excess {_edge_excess:.3e}, tolerated "
            f"{HE_EII_EDGE_REL_TOL:.1e})"
        )
    if E_walk_eV <= E_stop_eV:
        return False, True, False
    if he_mean_secondary_energy_eV(E_walk_eV, I_ion_eV=I_ion_eV) >= E_stop_eV:
        return True, False, True
    return True, False, False


def plateau_edge_energy_eV(
    E_b_eV: float,
    beam_flux_per_cm2_s: float,
    ne: float,
    Te_eV: float,
    E_stop_eV: float = HE_E_STOP_EV,
) -> tuple[float, int]:
    """Solve the quasilinear plateau EDGE ``E_1`` [eV]; ``(E_1, clamp)``.

    The relaxed distribution is flat in velocity from the plateau edge up to
    the beam, so its level is the launch cell's own Maxwellian evaluated AT
    the edge, and the flat band must carry the emitted beam's number flux.
    Those two statements are one equation,

        F_M(v_1) = m * j_b / ((E_b - E_1) * erg)

    with ``F_M(v) = ne sqrt(m / (2 pi Te)) exp(-m v^2 / 2 Te)`` the 1D-reduced
    Maxwellian (``Te`` in erg), ``v_1 = sqrt(2 E_1 / m)``, ``E_b = e*phi_c``
    the beam energy and ``j_b`` the emitted beam NUMBER flux
    ``I_eth* / (e A_cell)`` [1/cm^2/s]. The left side falls monotonically in
    ``E_1`` and the right side rises monotonically to ``+inf`` at ``E_b``, so
    the root is unique and bisection cannot land on the wrong one.

    ``clamp`` is ``0`` when the root was bracketed, and ``-1`` when it sits at
    or below ``E_stop_eV`` and the edge was clamped up to that floor. There is
    no ``+1``: the right side diverges at ``E_b``, so no root can exceed it.
    The clamp is reported rather than swallowed because ``E_1`` is a
    STATE-DEPENDENT solve -- a run that spends frames on the floor is running
    a different spectrum from one that does not, and that has to be visible in
    the diagnostics instead of inferred.

    Raises when the state cannot pose the question: a non-positive or
    non-finite ``E_b``/``Te``/``ne``, a non-positive flux, or ``E_b`` at or
    below ``E_stop_eV`` (there is no band to split).
    """
    E_b = float(E_b_eV)
    Te = float(Te_eV)
    n_e = float(ne)
    j_b = float(beam_flux_per_cm2_s)
    E_stop = float(E_stop_eV)
    if not (math.isfinite(E_b) and E_b > E_stop):
        raise ValueError(
            "plateau_edge_energy_eV needs a beam energy above the inelastic "
            f"floor E_stop_eV={E_stop} (got E_b_eV={E_b_eV!r}): below it the "
            "plateau has no band to split"
        )
    if not (math.isfinite(Te) and Te > 0.0):
        raise ValueError(
            f"plateau_edge_energy_eV needs a finite Te > 0 (got {Te_eV!r})"
        )
    if not (math.isfinite(n_e) and n_e > 0.0):
        raise ValueError(
            f"plateau_edge_energy_eV needs a finite ne > 0 (got {ne!r})"
        )
    if not (math.isfinite(j_b) and j_b > 0.0):
        raise ValueError(
            "plateau_edge_energy_eV needs a finite beam flux > 0 (got "
            f"{beam_flux_per_cm2_s!r})"
        )
    # Both sides in LOGS: F_M underflows to 0.0 for any edge more than a few
    # hundred Te above the bulk, which is the whole interesting range, and a
    # residual formed on the underflowed value would be flat and the bisection
    # would return the bracket midpoint rather than the root.
    ln_level = (
        math.log(n_e)
        + 0.5 * math.log(_ME_CGS / (2.0 * math.pi * Te * _ERG_PER_EV))
    )
    ln_demand = math.log(_ME_CGS * j_b / _ERG_PER_EV)

    def _residual(E1):
        # ln F_M(v_1) - ln[ m j_b / ((E_b - E_1) erg) ], strictly decreasing.
        gap = E_b - E1
        if gap <= 0.0:
            return -math.inf
        return (ln_level - E1 / Te) - (ln_demand - math.log(gap))

    if _residual(E_stop) <= 0.0:
        # The Maxwellian is already below the level the plateau would need at
        # the floor: the edge the equation asks for sits inside the bulk,
        # where this closure's flat-plateau picture does not hold. Clamped to
        # the floor and COUNTED.
        return E_stop, -1
    lo, hi = E_stop, E_b
    for _ in range(PLATEAU_EDGE_BISECTIONS):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return lo, 0


def plateau_group_edges_eV(E_1_eV: float, E_b_eV: float, groups: int):
    """``(edges, midpoints)`` of the equal-power plateau groups [eV].

    Edges uniform in ``E^2`` -- ``E_i = sqrt(E_1^2 + (i/N)(E_b^2 - E_1^2))``,
    ``i = 0 .. N`` -- which makes each group carry exactly ``1/N`` of the
    streaming power (``dP/dE ~ E``) AND span an equal interval of the
    classical range (which goes as ``E^2``). The representative energy is the
    arithmetic midpoint of each group's two edges.

    Returned as two arrays of length ``groups + 1`` and ``groups``.
    """
    N = int(groups)
    if N < 1:
        raise ValueError(f"plateau group count must be >= 1 (got {groups!r})")
    E_1 = float(E_1_eV)
    E_b = float(E_b_eV)
    if not (math.isfinite(E_1) and math.isfinite(E_b) and 0.0 < E_1 < E_b):
        raise ValueError(
            "plateau group edges need 0 < E_1 < E_b (got "
            f"E_1={E_1_eV!r}, E_b={E_b_eV!r})"
        )
    i = np.arange(N + 1, dtype=float)
    edges = np.sqrt(E_1 * E_1 + (i / N) * (E_b * E_b - E_1 * E_1))
    return edges, 0.5 * (edges[:-1] + edges[1:])


@dataclass(frozen=True)
class BeamDepositionResult:
    """Per-cell deposition of one beam ray; arrays have shape (cells,).

    ionization_events   : ionization events [1/s]
    excitation_events   : excitation events [1/s]
    plasma_heating_erg_s: Coulomb + anomalous drag + secondary-electron
                          energy + sub-threshold residual [erg/s]
    heating_coulomb_erg_s   : DIAGNOSTIC split of ``plasma_heating_erg_s`` --
                          the continuous Coulomb drag on plasma electrons
                          [erg/s]
    heating_anomalous_erg_s : DIAGNOSTIC split -- the anomalous (quasilinear
                          beam-plasma) drag [erg/s]; identically zero under
                          ``anomalous_model="none"``. Under
                          ``anomalous_transport="tail_walk"`` this is the
                          WALKED deposition profile (where the tail electrons
                          actually slowed), not the birth profile.
    heating_secondary_erg_s : DIAGNOSTIC split -- the inelastic-EVENT thermal
                          residue, i.e. the mean secondary-electron energy
                          <W_sec> carried away per ionization ABOVE the
                          ``I_ion`` potential cost [erg/s]
    heating_terminal_erg_s  : DIAGNOSTIC split -- the primary's end-of-range
                          terminal dump: the sub-threshold residual banked
                          whole in the cell where E crosses ``E_stop`` [erg/s]

    The four ``heating_*`` arrays are pure BOOKKEEPING of products the energy
    decrement already forms; they are accumulated alongside, never in place
    of, ``plasma_heating_erg_s``, which keeps its exact historical value and
    is the only one the solver RHS consumes. Their sum reproduces
    ``plasma_heating_erg_s`` to floating-point associativity only (the lumped
    bank adds the three per-substep pieces before multiplying), so compare
    them with a relative tolerance, never with ``==``.

    radiated_erg_s      : excitation line radiation [erg/s]
    ionization_cost_erg_s: I_ion * ionization events [erg/s] (kept separate
                          to map onto the solver's beam_ionization_cost term)
    transmitted_flux    : primary flux leaving the far end [1/s] (reduced by
                          the anode-mesh interception, if any)
    transmitted_energy_eV: mean primary energy at exit [eV] (0 if absorbed)
    anode_intercepted_erg_s: energy the anode mesh intercepts at the anode-face
                          crossing [erg/s] (audit A15). This leaves the plasma
                          (booked to the electrode, NOT plasma_heating_erg_s);
                          0 when no interception is requested or the ray stops
                          before the anode face. With the TAIL cull armed the
                          QL tail walkers' NET contribution -- what the mesh
                          removed from the walk less what the reversed-walker
                          rider returned to it -- is added to this same row:
                          one convention for everything the anode takes.
    tail_anode_culled_flux_per_s: WALKER flux [1/s] the mesh removed from the QL
                          tail walk at its first crossing of the anode plane.
                          0 unless the tail cull is armed.
    tail_anode_culled_erg_s: the energy [erg/s] that removed flux was carrying
                          at the crossing -- the equal-and-opposite partner of
                          the walk's own loss, and the gross the anode row is
                          formed from.
    tail_anode_returned_flux_per_s: WALKER flux [1/s] the reversed-walker rider
                          launched back off the anode plane. 0 when the rider
                          is off (its default), so the whole culled flux lands.
    tail_anode_returned_erg_s: the energy [erg/s] that returned flux carries,
                          i.e. what re-enters the plasma as reversed walkers
                          and is therefore NOT booked to the anode.
    E_entry_eV          : diagnostic: primary energy entering each cell [eV]
                          (0 for cells the ray never reaches)
    end_loss_low_erg_s  : END LEDGER, low-index end [erg/s]. Identically 0.0
                          under ``product_transport="local"``. Under
                          ``"nonlocal"`` it books (a) the remaining energy of
                          product walks that leave that end without
                          thermalizing and (b) the transmitted PRIMARY's power
                          ``Gamma_t * E_t`` when the ray exits there. This
                          energy LEAVES the system: like
                          ``anode_intercepted_erg_s`` it is NOT part of
                          ``plasma_heating_erg_s`` and must not enter any RHS
                          row.
    end_loss_high_erg_s : END LEDGER, high-index end [erg/s]; same content.
    end_loss_transmitted_erg_s: DIAGNOSTIC split -- how much of
                          ``end_loss_low + end_loss_high`` is the transmitted
                          primary rather than walked products. 0.0 under
                          ``"local"``, and 0.0 under ``"terminal_nonlocal"``
                          as well: that value books only the walked terminal
                          population, leaving the transmitted primary its own
                          term in the identity.
    terminal_escape_flux_per_s: the WALKED TERMINAL population that reached a
                          domain end without thermalizing [1/s]. Identically
                          0.0 unless the terminal walk ran, i.e. under
                          ``product_transport="local"`` and whenever the ray
                          stopped nowhere. These electrons land on a
                          terminating surface, so this is the CHARGE channel
                          matching the energy that went to the end ledger; it
                          is a flux, never an energy, and it is not part of
                          any RHS row. Under ``"nonlocal"`` it reports the
                          terminal walkers only -- the secondaries' escaping
                          flux is real and is deliberately not booked in v1.
    end_loss_tail_low_erg_s : WP-E TAIL END LEDGER, low-index end [erg/s].
                          Identically 0.0 under
                          ``anomalous_transport="local"`` (the default). Under
                          ``"tail_walk"`` it books the energy of QL tail
                          electrons that reach that end without thermalizing.
                          A SIBLING of ``end_loss_low_erg_s``, deliberately
                          kept separate so the WP-D product ledger keeps its
                          documented meaning and its measured values while the
                          two closures switch independently. Like the WP-D
                          ledger this energy LEAVES the system and must not
                          enter any RHS row.
    end_loss_tail_high_erg_s: WP-E TAIL END LEDGER, high-index end [erg/s];
                          same content.
    ionization_events_tail  : K6 DIAGNOSTIC SPLIT -- how much of
                          ``ionization_events`` the QL tail walkers contributed
                          [1/s]. Identically zero under ``tail_ionization="off"``
                          (the default), where the walkers cannot ionize at all.
    excitation_events_tail  : K6 DIAGNOSTIC SPLIT of ``excitation_events``
                          [1/s]; same status.
    ionization_cost_tail_erg_s: K6 DIAGNOSTIC SPLIT of ``ionization_cost_erg_s``
                          [erg/s]; same status.
    radiated_tail_erg_s     : K6 DIAGNOSTIC SPLIT of ``radiated_erg_s``
                          [erg/s]; same status.
    tail_power_erg_s        : K7b EXPOSURE LEDGER -- the total QL tail power
                          this ray launched into the walk [erg/s], i.e. the
                          withheld ``P_QL`` before any of it is deposited.
                          The DENOMINATOR of the two band fractions below.
                          Identically 0.0 under
                          ``anomalous_transport="local"``.
    tail_sub_threshold_power_erg_s: K7b -- how much of ``tail_power_erg_s`` was
                          marched with the ionizing channel REVERTED because
                          ``E_tail`` sits at or below ``E_stop_eV`` [erg/s].
                          Nonzero only under ``tail_ionization="on"``; it is
                          the whole tail power when it is nonzero, because
                          ``E_tail`` is one number per ray. What makes the
                          foot-phase reversion readable from a saved file
                          rather than being a silent no-op.
    tail_above_bar_power_erg_s: K7b -- how much of ``tail_power_erg_s`` was
                          marched ABOVE the depth-1 ``<W_sec>`` bar [erg/s],
                          i.e. under the <= 2.0% cascade understatement. Same
                          status and same all-or-nothing structure.
    plateau_wave_power_erg_s: MULTI-GROUP -- the WAVE/BULK share of the
                          withheld ``P_QL``, banked as local bulk heat in the
                          extraction cells [erg/s]. Identically 0.0 unless
                          ``anomalous_transport="plateau_multigroup"``. It is
                          already inside ``plasma_heating_erg_s`` and
                          ``heating_anomalous_erg_s`` (it IS the anomalous
                          channel's local delivery); this reports it apart so
                          the derived split of the bank -- ``tail_power_erg_s``
                          streaming, this wave -- is readable per frame.

    The plateau EDGE itself is deliberately NOT a field here: it is a property
    of the extraction solve rather than of one ray (both ends' rays and both
    halves of a clumping split share one edge), so the solver carries it and
    its clamp census on the cathode solve instead -- one home for one fact.

    The four ``*_tail`` arrays are bookkeeping, exactly like the four
    ``heating_*`` splits: they re-report a SUBSET of banks the shared arrays
    already carry, and nothing downstream consumes them in place of those
    arrays. Their purpose is that the tail channel's own energy branching
    (module docstring, K6) stays readable when the primary is filling the same
    banks.
    """

    ionization_events: np.ndarray
    excitation_events: np.ndarray
    plasma_heating_erg_s: np.ndarray
    radiated_erg_s: np.ndarray
    ionization_cost_erg_s: np.ndarray
    transmitted_flux: float
    transmitted_energy_eV: float
    anode_intercepted_erg_s: float
    E_entry_eV: np.ndarray
    heating_coulomb_erg_s: np.ndarray
    heating_anomalous_erg_s: np.ndarray
    heating_secondary_erg_s: np.ndarray
    heating_terminal_erg_s: np.ndarray
    ionization_events_tail: np.ndarray
    excitation_events_tail: np.ndarray
    ionization_cost_tail_erg_s: np.ndarray
    radiated_tail_erg_s: np.ndarray
    end_loss_low_erg_s: float = 0.0
    end_loss_high_erg_s: float = 0.0
    end_loss_transmitted_erg_s: float = 0.0
    terminal_escape_flux_per_s: float = 0.0
    end_loss_tail_low_erg_s: float = 0.0
    end_loss_tail_high_erg_s: float = 0.0
    tail_power_erg_s: float = 0.0
    tail_sub_threshold_power_erg_s: float = 0.0
    tail_above_bar_power_erg_s: float = 0.0
    plateau_wave_power_erg_s: float = 0.0
    tail_anode_culled_flux_per_s: float = 0.0
    tail_anode_culled_erg_s: float = 0.0
    tail_anode_returned_flux_per_s: float = 0.0
    tail_anode_returned_erg_s: float = 0.0


def deposit_beam(
    E0_eV: float,
    Gamma0_per_s: float,
    nn: np.ndarray,
    ne: np.ndarray,
    Te: np.ndarray,
    launch: int,
    direction: int,
    dz_cm: np.ndarray,
    *,
    I_ion_eV: float = HE_I_ION_EV,
    E_stop_eV: float = HE_E_STOP_EV,
    coulomb_model: str = "fast_electron",
    anomalous_model: str = "none",
    beam_area_cm2: np.ndarray | float | None = None,
    ql_relaxation_coeff: float | None = None,
    max_energy_fraction_per_substep: float = 0.02,
    anode_cross_index: int | None = None,
    anode_eta: float = 0.0,
    product_transport: str = "local",
    anomalous_transport: str = "local",
    anomalous_disposal: str = "local",
    tail_energy_eV: float | None = None,
    tail_ionization: str = "off",
    tail_walk_window: tuple[int, int] | None = None,
    tail_reflect_face: int | None = None,
    tail_reflect_threshold_eV: float | None = None,
    plateau_edge_eV: float | None = None,
    plateau_groups: int = PLATEAU_GROUP_COUNT,
    stopping_coefficient: np.ndarray | None = None,
    tail_anode_cross_index: int | None = None,
    tail_anode_eta: float = 0.0,
    tail_anode_reflected_particles: float = 0.0,
    tail_anode_reflected_energy: float = 0.0,
) -> BeamDepositionResult:
    """Deposit one monoenergetic beam ray through the column (He only).

    ``E0_eV`` is the accelerating sheath drop,
    ``Gamma0_per_s`` the accepted emitted electron flux, ``nn/ne/Te`` the
    per-cell column state, and the ray is ``(launch, direction, dz_cm)``
    with ``direction`` +1 (toward increasing index) or -1. Cells behind the
    launch point receive nothing.

    ``anomalous_model``: ``"none"`` (default), ``"quasilinear"`` or
    ``"ql_relaxation"``. Both non-default values require ``beam_area_cm2``
    (scalar or per-cell) to form n_b; ``"ql_relaxation"`` additionally requires
    ``ql_relaxation_coeff``, the registered O(10-100) plateau-formation bracket
    constant, and raises rather than substituting a default for it.
    ``ql_relaxation_coeff`` is INERT under the other two models.

    **Anode-mesh interception (audit A15).** ``anode_cross_index`` is the first
    cell on the far (column) side of the anode face along the ray; when it is
    given with ``anode_eta`` in ``[0, 1)`` the mesh intercepts the solid
    fraction ``anode_eta`` of the flux STILL STREAMING when the ray reaches
    that face (i.e. the long-mean-free-path beam that survived the gap). The
    intercepted power ``anode_eta * gamma * E`` is booked to
    ``anode_intercepted_erg_s`` (it leaves the plasma, landing on the anode,
    NOT in ``plasma_heating_erg_s``), and the surviving flux is reduced to
    ``(1 - anode_eta) * gamma`` for all subsequent deposition and ionization.
    A ray that stops in the gap never reaches the face and intercepts nothing,
    so only the survived (bypass) fraction is removed -- consistent with the
    circuit's ``eta * beam_bypass_fraction``. Per-ray energy still closes to
    roundoff::

        Gamma0*E0 = heating + radiated + cost + anode_intercepted + transmitted

    Off (``anode_cross_index is None`` or ``anode_eta == 0``) the running flux
    is the constant ``Gamma0_per_s`` throughout, so every bank is byte-for-byte
    the historical result.

    **Anode-mesh cull of the QL TAIL (A2a).** The primary interception above
    removes cathode-borne flux streaming OUT through the mesh; the tail walkers
    are born in the column and meet the same wires from whichever side they
    happen to approach. ``tail_anode_cross_index`` (a CELL index on the full
    grid, the same cell ``anode_cross_index`` names) with ``tail_anode_eta`` in
    ``[0, 1)`` removes that solid fraction of a walker's flux at its FIRST
    crossing of the anode plane and books the removed share to
    ``anode_intercepted_erg_s`` -- the same row the primary uses, no second
    convention. A walker that thermalizes short of the plane, or that is born
    past it and walks away, never crosses and loses nothing; a walker that
    reflects at the cathode face and re-crosses is culled once, on the first
    crossing only. The gross removed flux and the energy it carried are
    reported as ``tail_anode_culled_flux_per_s`` / ``tail_anode_culled_erg_s``.

    ``tail_anode_reflected_particles`` (``R_e``) and
    ``tail_anode_reflected_energy`` (``eta_E``) arm the REVERSED-WALKER RIDER on
    top of that cull. Both are PER INCIDENT and both default to 0.0, which is
    the rider off: nothing returns and the whole culled share lands on the
    anode. Armed, ``R_e`` of the culled walkers are launched back off the plane
    in the reversed direction carrying ``eta_E`` of the culled energy, so their
    mean energy per returned particle is ``(eta_E / R_e)`` times the incident
    energy -- which is why ``eta_E > R_e`` is refused rather than clamped. The
    box evaluation has a domain edge: a crossing whose incident energy is below
    ``TAIL_ANODE_RIDER_MIN_ENERGY_EV`` returns nothing whatever the pair says
    (that walker is absorbed). What returns is reported as
    ``tail_anode_returned_flux_per_s`` / ``tail_anode_returned_erg_s`` and is
    subtracted from the anode row, since it never landed. The rider requires
    the marched tail (``tail_ionization="on"``): the energy-only closed-form
    walk carries a whole cell sequence in one telescoping integral and has no
    per-walker launch to reverse, and the combination is refused rather than
    approximated.

    **Product transport (WP-D).** ``product_transport`` is ``"local"``
    (default), ``"nonlocal"`` or ``"terminal_nonlocal"``; see the module
    docstring for the physics. Off
    (the default, and the value the gap-transmission probe call sites leave
    untouched) not one branch below changes, so the result is byte-for-byte
    the historical one and ``end_loss_*`` are identically zero. Under
    ``"nonlocal"`` the
    secondary and terminal-residual banks are withheld from their birth cells
    and re-deposited along mini-CSDA walks, with what escapes an end booked to
    the end ledger. ``"terminal_nonlocal"`` withholds the TERMINAL residual
    alone and walks it identically; the secondary bank is banked locally,
    exactly as under ``"local"``, and the transmitted primary keeps its own
    identity term instead of joining the end ledger. Energy-only:
    ``ionization_events`` and
    ``excitation_events`` -- and therefore every particle and circuit row
    downstream -- are identical in all three modes. The escaping terminal
    FLUX is reported (``terminal_escape_flux_per_s``) for a caller that books
    wall charge; this function books no charge anywhere.

    **QL heating locality (WP-E).** ``anomalous_transport`` is ``"local"``
    (default), ``"tail_walk"`` or ``"plateau_multigroup"``; see the module
    docstring for the physics.
    Off, no branch below changes and the result is byte-for-byte the
    historical one with ``end_loss_tail_*`` identically zero. On, the
    anomalous channel's power is withheld from its birth cell, re-expressed as
    tail electrons at ``tail_energy_eV`` (required, finite, > 0) launched
    50/50 along +-B, and walked on the same Coulomb machinery as the WP-D
    products; escapes go to the SEPARATE tail end ledger. ``"tail_walk"``
    requires an active anomalous channel (``anomalous_model="quasilinear"``) --
    with no anomalous drag there is no power to carry and the setting would be
    a silent no-op. Energy-only, exactly like WP-D. The two closures are
    independent and compose: with both on, the event products walk on the WP-D
    ledger and the QL tails on the WP-E one.

    **Branched disposal (pd1).** ``anomalous_disposal`` is ``"local"``
    (default, bit-exact -- not one branch below changes) or
    ``"landau_branched"``, which splits each cell's extracted anomalous power
    by the COMPUTED ``landau_branching_fraction(ne, Te, nn, E0_eV)``: that
    share is walked exactly as ``anomalous_transport="tail_walk"`` walks all of
    it, the rest is banked locally exactly as ``"local"`` banks all of it. See
    the module docstring. It needs everything the walk needs
    (``anomalous_model`` active, ``tail_energy_eV``) and is REFUSED together
    with any non-``"local"`` ``anomalous_transport`` (``"tail_walk"`` is its
    ``f_Landau ≡ 1`` corner) -- both settings claim the same bank, so naming
    both states two dispositions for one quantity.

    **Multi-group plateau.** ``anomalous_transport="plateau_multigroup"``
    replaces the single birth energy with the plateau's own DERIVED spectrum
    (module docstring). It needs ``plateau_edge_eV``, the edge ``E_1`` solved
    for THIS extraction by :func:`plateau_edge_energy_eV` -- a state-dependent
    quantity with deliberately no default, and one this function does not
    solve itself because the edge belongs to the extraction rather than to a
    ray (both ends and both halves of a clumping split share one). It must lie
    strictly below the launch energy ``E0_eV``, which is the band's top.
    ``plateau_groups`` is the group count ``N`` (default
    ``PLATEAU_GROUP_COUNT``); it is an argument rather than a config key so a
    convergence read can vary the RESOLUTION of the derived spectrum without a
    dial existing that a campaign run could tune. ``tail_energy_eV`` is INERT
    here and is REFUSED rather than ignored, as is ``plateau_edge_eV`` under
    any other value. The wave/bulk share is banked locally and reported in
    ``plateau_wave_power_erg_s``; ``tail_power_erg_s`` then carries the
    streaming share alone, and the two sum to the withheld ``P_QL``.

    **Tail ionization (K6).** ``tail_ionization`` is ``"off"`` (default,
    bit-exact -- the walk stays energy-only) or ``"on"``, which marches each
    tail population on this module's own CSDA integration so it ionizes and
    excites the column gas on its way; see the module docstring. Requires
    ``anomalous_transport="tail_walk"`` (there is no other walk to give the
    channel to; ``"plateau_multigroup"`` gives it to every group) and
    ``tail_walk_window=(lo, hi)``, the inclusive cell range the
    walkers may traverse; see the module docstring for why the window has no
    safe default. Both are ValueErrors, not silent adjustments.

    **The band split (K7b).** ``tail_energy_eV`` no longer has to sit inside
    the depth-1 band; the two bars select a treatment for the ray instead of
    refusing it. At or below ``E_stop_eV`` the march REVERTS to the energy-only
    walk -- exactly the branch and exactly the floats ``"off"`` would produce
    for this call, because no inelastic channel is open there -- and the
    reverted power is reported in ``tail_sub_threshold_power_erg_s``. Above the
    ``<W_sec>(E_tail) >= E_stop_eV`` crossing the march RUNS with the depth-1
    truncation, whose cascade understatement is measured at <= 2.0%, and the
    exposed power is reported in ``tail_above_bar_power_erg_s``. Only the EII
    table edge (``HE_EII_EPS_TOP * I_ion_eV``) is still a ValueError: past it
    the ionization lookup clamps to its last node and the walk would be running
    on an extrapolated cross section. That edge is INCLUSIVE within
    ``HE_EII_EDGE_REL_TOL`` (K7c) -- ``E_tail`` there evaluates the cross
    section AT the table's last node, which is the node's own value rather than
    an extrapolation of it -- and a relative excess larger than the tolerance
    raises, reporting what it measured. In band both exposure fields are zero
    and nothing about the march changed.

    **Sheath reflection at a walk-window face (K7).** ``tail_reflect_face`` is
    ``None`` (default, bit-exact: both faces free-escape, as WP-E and K6 ship)
    or ``-1`` / ``+1``, naming the ONE face of ``tail_walk_window`` that
    reflects tail walkers. A walker arriving there is reflected -- same energy,
    reversed direction -- when its arrival energy is strictly below
    ``tail_reflect_threshold_eV`` (required, finite, > 0 with a face named),
    and free-escapes to the tail end ledger otherwise; see the module
    docstring. Naming a face also requires ``tail_walk_window``, whose faces
    become the walk's boundary in BOTH tail modes: the energy-only walk is
    windowed under reflection where it runs the whole grid without it, because
    a reflecting face has to be a face the walk actually stops at. Only one
    face may reflect: two reflecting faces trap the walker between them, which
    this closed-form walk has no termination convention for, and the caller is
    expected to refuse that configuration rather than have it silently
    approximated here.

    ``stopping_coefficient`` (cost read 2026-08-02, restructure C) is the
    per-cell ``A`` of ``dE/dx = A W**p`` that the walks below need, HOISTED to
    the caller. Left ``None`` -- the default, and every historical call -- it
    is built here by ``_coulomb_stopping_coefficient`` exactly as before.
    Supplied, it is used verbatim, so a caller that launches several rays or
    (under WP-F) several energy groups over the SAME ``(ne, Te, model)`` pays
    for the 262-iteration Python listcomp once instead of once per ray: it is
    100 us, half of the entire WP-E per-call surcharge. Bit-exact either way --
    the caller is expected to build it with this module's own
    ``_coulomb_stopping_coefficient``, and it is only read when a walk closure
    is active, so the default path never touches it.
    """
    if product_transport not in ("local", "nonlocal", "terminal_nonlocal"):
        raise ValueError(
            f"unknown product_transport {product_transport!r}; "
            "expected 'local', 'nonlocal' or 'terminal_nonlocal'"
        )
    if anomalous_transport not in (
        "local", "tail_walk", "plateau_multigroup"
    ):
        raise ValueError(
            f"unknown anomalous_transport {anomalous_transport!r}; "
            "expected 'local', 'tail_walk' or 'plateau_multigroup'"
        )
    if anomalous_disposal not in ("local", "landau_branched"):
        raise ValueError(
            f"unknown anomalous_disposal {anomalous_disposal!r}; "
            "expected 'local' or 'landau_branched'"
        )
    if anomalous_disposal == "landau_branched" and (
        anomalous_transport != "local"
    ):
        raise ValueError(
            "anomalous_disposal='landau_branched' cannot be combined with "
            f"anomalous_transport={anomalous_transport!r}: the branch already "
            "decides what share of the extracted power is walked, and "
            "'tail_walk' is its f_Landau == 1 corner, so naming both states "
            "two dispositions for one bank. Select the branch with "
            "anomalous_transport='local'"
        )
    branch_tail = anomalous_disposal == "landau_branched"
    multigroup = anomalous_transport == "plateau_multigroup"
    if tail_ionization not in ("off", "on"):
        raise ValueError(
            f"unknown tail_ionization {tail_ionization!r}; "
            "expected 'off' or 'on'"
        )
    if tail_ionization == "on" and not (
        anomalous_transport == "tail_walk" or branch_tail or multigroup
    ):
        raise ValueError(
            "tail_ionization='on' requires walkers to give the channel to: "
            "anomalous_transport='tail_walk', "
            "anomalous_transport='plateau_multigroup' or "
            "anomalous_disposal='landau_branched' (with both 'local' there "
            "are no walkers and the setting would do nothing). "
            "anomalous_transport accepts 'local', 'tail_walk' or "
            "'plateau_multigroup'; "
            "anomalous_disposal accepts 'local' or 'landau_branched'; "
            "tail_ionization accepts 'off' or 'on'"
        )
    if anode_eta != 0.0 and not (0.0 <= anode_eta < 1.0):
        raise ValueError(
            f"anode_eta must be in [0, 1) (got {anode_eta})"
        )
    nn = np.asarray(nn, dtype=float)
    ne = np.asarray(ne, dtype=float)
    Te = np.asarray(Te, dtype=float)
    dz_cm = np.asarray(dz_cm, dtype=float)
    cells = dz_cm.size
    if anode_cross_index is not None:
        anode_cross_index = int(anode_cross_index)
        if not 0 <= anode_cross_index < cells:
            raise ValueError(
                "anode_cross_index must index a cell in [0, cells) "
                f"(got {anode_cross_index}, cells={cells})"
            )
    if nn.shape != (cells,) or ne.shape != (cells,) or Te.shape != (cells,):
        raise ValueError("nn, ne, Te, dz_cm must share one shape (cells,)")
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1 (got {direction})")
    if anomalous_model not in ANOMALOUS_MODELS:
        raise ValueError(
            f"unknown anomalous_model {anomalous_model!r}; "
            f"expected one of {sorted(ANOMALOUS_MODELS)}"
        )
    if anomalous_model in ("quasilinear", "ql_relaxation"):
        if beam_area_cm2 is None:
            raise ValueError(
                f"anomalous_model={anomalous_model!r} needs beam_area_cm2"
            )
        area = np.broadcast_to(
            np.asarray(beam_area_cm2, dtype=float), (cells,)
        )
    ql_coeff = 0.0
    if anomalous_model == "ql_relaxation":
        # No default is substituted here. The plateau-formation coefficient is
        # a REGISTERED BRACKET whose endpoints every headline is quoted at, so
        # a caller that has not stated which arm it is on has not configured
        # the closure -- picking one silently would put an unreported bracket
        # arm behind a published number.
        if ql_relaxation_coeff is None:
            raise ValueError(
                "anomalous_model='ql_relaxation' needs ql_relaxation_coeff "
                "(the registered O(10-100) plateau-formation bracket "
                "constant); there is deliberately no default here"
            )
        ql_coeff = float(ql_relaxation_coeff)
        if not math.isfinite(ql_coeff) or ql_coeff <= 0.0:
            raise ValueError(
                "ql_relaxation_coeff must be finite and > 0 (got "
                f"{ql_relaxation_coeff})"
            )
    # The branched disposal fills and consumes the SAME withholding bank the
    # tail walk does -- it only scales it per cell between the march and the
    # walk -- so it enters the march in the tail walk's own configuration and
    # every requirement the walk states applies to it verbatim.
    walk_tail = (
        anomalous_transport == "tail_walk" or branch_tail or multigroup
    )
    _tail_sel = (
        "anomalous_disposal='landau_branched'" if branch_tail
        else "anomalous_transport='plateau_multigroup'" if multigroup
        else "anomalous_transport='tail_walk'"
    )
    E_tail = 0.0
    E_plateau_1 = 0.0
    plateau_edges = plateau_midpoints = None
    plateau_stream_share = plateau_wave_share = 0.0
    if walk_tail:
        # There must BE an anomalous channel for the walk to carry; without one
        # the setting is a silent no-op, which is exactly what the presence
        # gating exists to prevent.
        if anomalous_model == "none":
            raise ValueError(
                f"{_tail_sel} requires an active anomalous "
                "channel (anomalous_model='quasilinear' or "
                "'ql_relaxation'); with no anomalous "
                "drag there is no power to carry and the setting would do "
                "nothing"
            )
    if multigroup:
        # The birth spectrum is DERIVED (module docstring), so the single-line
        # rung is not merely unused here -- it is a different closure, and
        # accepting it would let a caller name a birth energy that nothing
        # reads. Refused rather than ignored.
        if tail_energy_eV is not None:
            raise ValueError(
                "anomalous_transport='plateau_multigroup' derives the birth "
                "spectrum from the plateau itself, so tail_energy_eV is inert "
                f"under it (got {tail_energy_eV!r}); drop it, or select "
                "anomalous_transport='tail_walk' to launch at one energy"
            )
        if plateau_edge_eV is None:
            raise ValueError(
                "anomalous_transport='plateau_multigroup' needs "
                "plateau_edge_eV, the plateau edge E_1 solved for THIS "
                "extraction (cathode.beam_deposition.plateau_edge_energy_eV); "
                "it is a state-dependent solve on the launch cell's own "
                "Maxwellian and the emitted beam flux, and there is "
                "deliberately no default for it"
            )
        E_plateau_1 = float(plateau_edge_eV)
        if not math.isfinite(E_plateau_1) or E_plateau_1 <= 0.0:
            raise ValueError(
                "plateau_edge_eV must be finite and > 0 (got "
                f"{plateau_edge_eV})"
            )
        if not E_plateau_1 < E0_eV:
            raise ValueError(
                f"plateau_edge_eV={E_plateau_1} must lie strictly below the "
                f"beam energy E_b={E0_eV} eV: the streaming band is "
                "[E_1, E_b] and at or above E_b there is no band"
            )
        plateau_edges, plateau_midpoints = plateau_group_edges_eV(
            E_plateau_1, float(E0_eV), plateau_groups
        )
        # The two heirs of a dP/dE ~ E plateau over [E_1, E_b]; they sum to 1
        # by construction, which is the statement that no eV of the withheld
        # bank is created or dropped by the split.
        plateau_stream_share = (float(E0_eV) + E_plateau_1) / (2.0 * float(E0_eV))
        plateau_wave_share = (float(E0_eV) - E_plateau_1) / (2.0 * float(E0_eV))
    elif walk_tail:
        if tail_energy_eV is None:
            raise ValueError(
                f"{_tail_sel} needs tail_energy_eV (the "
                "QL plateau energy the tail electrons are launched at)"
            )
        E_tail = float(tail_energy_eV)
        if not math.isfinite(E_tail) or E_tail <= 0.0:
            raise ValueError(
                "tail_energy_eV must be finite and > 0 (got "
                f"{tail_energy_eV})"
            )
    elif plateau_edge_eV is not None:
        raise ValueError(
            "plateau_edge_eV was given without "
            "anomalous_transport='plateau_multigroup'; the edge belongs to "
            "the multi-group plateau closure and on its own would silently "
            "do nothing"
        )
    ionize_tail = tail_ionization == "on"
    # K7b band diagnostics for THIS ray. ``tail_sub_threshold`` means the
    # requested channel was reverted to the energy-only walk because E_tail
    # sits at or below the lowest inelastic threshold; ``tail_above_bar``
    # means it marched with the depth-1 truncation past the <W_sec> crossing.
    # At most one can be true (E_tail is one number per ray) and both are
    # false in band, which is what makes the in-band path provably untouched.
    # Under the multi-group plateau these are the POWER-WEIGHTED shares of the
    # launched bank instead, because the ray launches several energies and
    # each group is banded on its own -- see the walk stage below.
    tail_sub_threshold = False
    tail_above_bar = False
    # K7 sheath reflection. The face and its threshold are one setting: a face
    # with no threshold cannot be tested and a threshold with no face has
    # nothing to test, so each without the other is a refusal rather than a
    # default.
    reflect_face = None
    E_reflect = 0.0
    if tail_reflect_face is not None:
        if not walk_tail:
            raise ValueError(
                "tail_reflect_face requires a selection that walks the QL "
                "tail (anomalous_transport='tail_walk' or "
                "anomalous_disposal='landau_branched'); with no walk "
                "there is nothing to reflect and the setting would do "
                "nothing"
            )
        reflect_face = int(tail_reflect_face)
        if reflect_face not in (-1, 1):
            raise ValueError(
                "tail_reflect_face must be -1 (the walk window's low-index "
                f"face) or +1 (its high-index face), got {tail_reflect_face!r}"
            )
        if tail_reflect_threshold_eV is None:
            raise ValueError(
                "tail_reflect_face needs tail_reflect_threshold_eV (the "
                "energy below which a walker arriving at that face is turned "
                "around instead of escaping)"
            )
        E_reflect = float(tail_reflect_threshold_eV)
        if not math.isfinite(E_reflect) or E_reflect <= 0.0:
            raise ValueError(
                "tail_reflect_threshold_eV must be finite and > 0 (got "
                f"{tail_reflect_threshold_eV})"
            )
    elif tail_reflect_threshold_eV is not None:
        raise ValueError(
            "tail_reflect_threshold_eV was given without tail_reflect_face; "
            "the threshold belongs to a named reflecting face and on its own "
            "would silently do nothing"
        )
    tail_lo, tail_hi = 0, cells - 1
    if reflect_face is not None and not ionize_tail:
        # The reflecting face IS a window face, so the window is required in
        # both tail modes once reflection is on -- see the docstring.
        if tail_walk_window is None:
            raise ValueError(
                "tail_reflect_face needs tail_walk_window=(lo, hi): the "
                "reflecting face is one of that window's two faces, and "
                "without it the walk has no face to reflect at"
            )
        tail_lo, tail_hi = (int(tail_walk_window[0]), int(tail_walk_window[1]))
        if not 0 <= tail_lo <= tail_hi < cells:
            raise ValueError(
                "tail_walk_window must be an inclusive (lo, hi) cell range "
                f"with 0 <= lo <= hi < cells={cells} (got "
                f"{tail_walk_window})"
            )
    if ionize_tail:
        # THE WALK DOMAIN IS REQUIRED, not defaulted to the whole grid. This
        # module is solver-agnostic and cannot know which cells are plasma, but
        # an ionizing walk that leaves the plasma is not a small error: a ``-z``
        # walker launched near the source runs into whatever cells sit behind
        # the cathode, and at K5a conditions that is where 5-66% of its
        # ionization lands (measured, k6build_tailion_crosscheck.txt) -- births
        # into rows the solver's active-plasma mask ZEROES, i.e. pairs created
        # and silently deleted. The energy-only walk survives the same geometry
        # only because it leaks ~0.04% of P_QL there rather than most of its
        # product; a PARTICLE channel cannot be built on that domain. So the
        # caller states the window and a caller that has not thought about it
        # gets an error, not a quiet sink.
        if tail_walk_window is None:
            raise ValueError(
                "tail_ionization='on' needs tail_walk_window=(lo, hi), the "
                "inclusive cell range the tail walkers may traverse (the "
                "plasma-active window: a walker leaving it hits a wall and is "
                "booked to the tail end ledger). Without it the walk would "
                "run off into cells whose plasma rows the solver zeroes and "
                "birth pairs that are then deleted"
            )
        tail_lo, tail_hi = (int(tail_walk_window[0]), int(tail_walk_window[1]))
        if not 0 <= tail_lo <= tail_hi < cells:
            raise ValueError(
                "tail_walk_window must be an inclusive (lo, hi) cell range "
                f"with 0 <= lo <= hi < cells={cells} (got "
                f"{tail_walk_window})"
            )
        # --- The K7b band split (see ``_tail_band``) ----------------------
        # One walker energy per ray under the single-line closures, so the
        # ray's whole tail power lands in one band. The multi-group plateau
        # launches several energies from one ray, so ITS bands are decided
        # per group in the walk stage below and the two flags become
        # power-weighted shares there.
        if not multigroup:
            ionize_tail, tail_sub_threshold, tail_above_bar = _tail_band(
                E_tail, I_ion_eV, E_stop_eV, "tail_energy_eV"
            )
    # --- A2a: the anode-mesh cull of the QL tail, and its rider -----------
    # Placed after the walk window is resolved, because the cull cell is stated
    # on the FULL grid and has to land inside the window the walkers traverse.
    R_e_tail = float(tail_anode_reflected_particles)
    eta_E_tail = float(tail_anode_reflected_energy)
    tail_cull = tail_anode_cross_index is not None and tail_anode_eta > 0.0
    tail_anode_local = -1
    if tail_anode_eta != 0.0 and not (0.0 <= tail_anode_eta < 1.0):
        raise ValueError(
            f"tail_anode_eta must be in [0, 1) (got {tail_anode_eta})"
        )
    for _name, _val in (
        ("tail_anode_reflected_particles", R_e_tail),
        ("tail_anode_reflected_energy", eta_E_tail),
    ):
        if not math.isfinite(_val) or not 0.0 <= _val <= 1.0:
            raise ValueError(f"{_name} must be in [0, 1] (got {_val})")
    if eta_E_tail > R_e_tail:
        raise ValueError(
            "tail_anode_reflected_energy must not exceed "
            "tail_anode_reflected_particles (got "
            f"{eta_E_tail} > {R_e_tail}): both are PER INCIDENT, so the "
            "returned energy fraction is the returned particle fraction times "
            "the mean returned energy in units of the incident energy, and "
            "that mean cannot exceed one"
        )
    if (R_e_tail > 0.0 or eta_E_tail > 0.0) and not tail_cull:
        raise ValueError(
            "the reversed-walker rider "
            "(tail_anode_reflected_particles/tail_anode_reflected_energy) "
            "needs the tail cull it rides on: give tail_anode_cross_index "
            "with tail_anode_eta > 0. Without the cull nothing is intercepted "
            "and the pair would be a silent no-op"
        )
    if R_e_tail > 0.0 and tail_ionization != "on":
        # Tested against the SELECTOR, not against the band-reverted
        # ``ionize_tail``: a run whose walker energy tracks phi_c(t) would
        # otherwise be accepted at one step and refused at the next. A walker
        # the K7b band does revert to the energy-only integral is below
        # E_stop, hence below the rider's own energy floor, so the clamp
        # already returns nothing for it -- and the walk asserts that rather
        # than assuming it.
        raise ValueError(
            "the reversed-walker rider requires tail_ionization='on': the "
            "energy-only closed-form walk carries a whole cell sequence in one "
            "telescoping integral and has no per-walker launch to reverse. "
            "The cull itself composes with either walk; only the return does "
            "not, and it is refused rather than approximated"
        )
    if tail_cull:
        if not walk_tail:
            raise ValueError(
                "tail_anode_cross_index/tail_anode_eta cull the QL TAIL "
                "walkers, so they need a walked tail: select "
                "anomalous_transport='tail_walk' or 'plateau_multigroup' (or "
                "anomalous_disposal='landau_branched'). With none of them "
                "there are no walkers to cull and the pair would do nothing"
            )
        tail_anode_local = int(tail_anode_cross_index) - tail_lo
        if not 0 <= tail_anode_local <= tail_hi - tail_lo:
            raise ValueError(
                f"tail_anode_cross_index={tail_anode_cross_index} lies outside "
                f"the tail walk window {(tail_lo, tail_hi)}; the walkers never "
                "reach that cell, so the cull would be a silent no-op"
            )
    if stopping_coefficient is not None:
        stopping_coefficient = np.asarray(stopping_coefficient, dtype=float)
        if stopping_coefficient.shape != (cells,):
            raise ValueError(
                "stopping_coefficient must have shape (cells,) = "
                f"{(cells,)} (got {stopping_coefficient.shape})"
            )
    frac = float(max_energy_fraction_per_substep)
    if not 0.0 < frac < 1.0:
        raise ValueError(
            "max_energy_fraction_per_substep must be in (0, 1), got "
            f"{max_energy_fraction_per_substep}"
        )

    ionization_events = np.zeros(cells)
    excitation_events = np.zeros(cells)
    heating = np.zeros(cells)  # erg/s
    radiated = np.zeros(cells)
    ionization_cost = np.zeros(cells)
    E_entry = np.zeros(cells)
    # Diagnostic splits of `heating` (see BeamDepositionResult). Accumulated
    # from the SAME products the lumped bank uses; nothing here feeds the RHS.
    heat_coulomb = np.zeros(cells)
    heat_anomalous = np.zeros(cells)
    heat_secondary = np.zeros(cells)
    heat_terminal = np.zeros(cells)
    # K6 diagnostic splits: the QL tail walkers' share of the four shared
    # banks they now write into. Allocated unconditionally (four cell-sized
    # arrays) so every result carries the same fields; they stay all-zero
    # unless the ionizing walk actually runs.
    ion_events_tail = np.zeros(cells)
    exc_events_tail = np.zeros(cells)
    ion_cost_tail = np.zeros(cells)
    radiated_tail = np.zeros(cells)
    # --- Non-local product transport (WP-D) -----------------------------
    # Under "nonlocal" the secondary and terminal-residual banks are WITHHELD
    # from their birth cells and accumulated here, then walked after the ray
    # is done. Under "local" none of this is touched or allocated.
    #
    # The selector names WHICH population walks, and each branch below tests
    # the population it belongs to rather than the selector: "nonlocal" walks
    # both, "terminal_nonlocal" walks the terminal residual alone, and the
    # secondary branches then take the identical path they take under "local"
    # -- which is what makes the along-ray banks byte-for-byte the local ones.
    # `book_transmitted` is "nonlocal"'s separate closure of the standing
    # transmitted-primary hole and travels with that value only.
    walk_secondaries = product_transport == "nonlocal"
    walk_terminal = product_transport in ("nonlocal", "terminal_nonlocal")
    book_transmitted = product_transport == "nonlocal"
    walk_products = walk_secondaries or walk_terminal
    end_loss_low = 0.0
    end_loss_high = 0.0
    end_loss_transmitted = 0.0
    terminal_escape_flux = 0.0
    if walk_products:
        # Both secondary banks are allocated whenever anything walks, so the
        # compiled march's argument list and the walk stage below keep one
        # shape; under "terminal_nonlocal" they simply stay all-zero.
        sec_flux = np.zeros(cells)  # secondary electrons born per cell [1/s]
        sec_power_eV = np.zeros(cells)  # their energy [eV/s]
        terminal_cell = -1
        terminal_flux = 0.0
        terminal_E = 0.0
    # --- QL heating locality (WP-E) --------------------------------------
    # Under "tail_walk" the anomalous drag is WITHHELD from its birth cell and
    # accumulated here [eV/s], then carried by tail electrons at E_tail after
    # the ray is done. Under "local" none of this is touched or allocated.
    end_loss_tail_low = 0.0
    end_loss_tail_high = 0.0
    # K7b exposure ledger [erg/s]: the QL tail power this ray launched, and
    # how much of it was marched outside the depth-1 band. Filled once the
    # withheld power is known, below.
    tail_power = 0.0
    tail_sub_threshold_power = 0.0
    tail_above_bar_power = 0.0
    # MULTI-GROUP: the wave/bulk heir of the withheld bank [erg/s]. Reported
    # apart from the streaming heir ``tail_power`` even though it is already
    # inside the heating banks, so the derived split is readable per frame.
    plateau_wave_power = 0.0
    # A2a anode cull of the tail (all four stay 0.0 unless the cull is armed).
    # ``culled`` is what the mesh took OUT of the walk at the plane; ``returned``
    # is what the rider put back. Their difference is the net the anode row
    # below books, and each is reported so the partner audit can name both
    # halves rather than reading a single net.
    tail_anode_culled_flux = 0.0
    tail_anode_culled_erg = 0.0
    tail_anode_returned_flux = 0.0
    tail_anode_returned_erg = 0.0
    if walk_tail:
        anom_power_eV = np.zeros(cells)

    order = range(launch, cells) if direction > 0 else range(launch, -1, -1)
    E = float(E0_eV)
    absorbed = False
    # Running flux [1/s]. Constant Gamma0 unless the anode mesh intercepts part
    # of the surviving beam at its face (audit A15); every bank below multiplies
    # by this, so the off path is bit-for-bit the historical constant-flux result.
    gamma = float(Gamma0_per_s)
    anode_intercepted = 0.0  # erg/s booked to the anode, not the plasma
    intercept_active = anode_cross_index is not None and anode_eta > 0.0

    if E <= E_stop_eV:
        # Sub-threshold source: nothing inelastic can happen; the module's
        # domain is beam energies, so pass it through untouched. Under
        # "nonlocal" that pass-through IS a transmitted primary, so the end
        # ledger books it and the identity closes here too. (Defensive path:
        # the cathode wiring only launches rays with phi_c > I_ion > E_stop.)
        sub_power = float(Gamma0_per_s) * E * _ERG_PER_EV
        if book_transmitted and sub_power > 0.0:
            end_loss_transmitted = sub_power
            if direction > 0:
                end_loss_high = sub_power
            else:
                end_loss_low = sub_power
        return BeamDepositionResult(
            ionization_events=ionization_events,
            excitation_events=excitation_events,
            plasma_heating_erg_s=heating,
            radiated_erg_s=radiated,
            ionization_cost_erg_s=ionization_cost,
            transmitted_flux=float(Gamma0_per_s),
            transmitted_energy_eV=E,
            anode_intercepted_erg_s=0.0,
            E_entry_eV=E_entry,
            heating_coulomb_erg_s=heat_coulomb,
            heating_anomalous_erg_s=heat_anomalous,
            heating_secondary_erg_s=heat_secondary,
            heating_terminal_erg_s=heat_terminal,
            ionization_events_tail=ion_events_tail,
            excitation_events_tail=exc_events_tail,
            ionization_cost_tail_erg_s=ion_cost_tail,
            radiated_tail_erg_s=radiated_tail,
            end_loss_low_erg_s=end_loss_low,
            end_loss_high_erg_s=end_loss_high,
            end_loss_transmitted_erg_s=end_loss_transmitted,
            # A sub-threshold source drives no anomalous drag, so neither the
            # tail ledger nor the K7b exposure ledger has anything to book on
            # this path.
            end_loss_tail_low_erg_s=end_loss_tail_low,
            end_loss_tail_high_erg_s=end_loss_tail_high,
            tail_power_erg_s=tail_power,
            tail_sub_threshold_power_erg_s=tail_sub_threshold_power,
            tail_above_bar_power_erg_s=tail_above_bar_power,
        )

    # --- Compiled CSDA march (opt-in) ------------------------------------
    # Runs the whole double loop below in C and then leaves ``order`` empty so
    # the Python march is skipped. Deliberately shaped as an INSERTION rather
    # than an ``else:`` wrapped around the loop: the Python march is the
    # equivalence target and has to stay byte-for-byte reviewable, and an
    # added indent level would touch every one of its ~170 lines.
    #
    # The five preconditions are the cases the transcription does not
    # reproduce, each routed back to Python so it behaves exactly as it always
    # has: an out-of-range ``launch`` (Python's ``range`` walks negative
    # indices or raises), ``I_ion_eV == 0`` (ZeroDivisionError), an unknown
    # coulomb model (ValueError from ``coulomb_stopping_eV_per_cm``), a
    # beam above the excitation table's ceiling (where the lookup falls back
    # to the exact manifold sum), and ``anomalous_model="ql_relaxation"``.
    # ``E`` only ever decreases along the march, so testing the launch energy
    # settles the ceiling for the whole ray.
    #
    # The anomalous precondition is a HARD one, not a performance choice: the
    # kernel takes the anomalous channel as a BOOLEAN and applies the fiat
    # quasilinear drag when it is set, so offering it ``ql_relaxation`` would
    # silently run the wrong closure. It takes the Python march instead, which
    # is where the new closure lives.
    #
    # ``product_transport="terminal_nonlocal"`` is refused for exactly the same
    # reason, and it is the same trap: the kernel takes product transport as
    # ONE boolean covering both withholding banks, so neither value it can be
    # given is this closure. False banks the terminal residual locally --
    # silently the "local" rule under a selector that asked for the walk --
    # and True additionally withholds the SECONDARIES, whose bank nothing then
    # walks, silently deleting that energy. The Python march below is the only
    # place the two populations are separable.
    _csda_ctx = None
    if (
        _CSDA_MARCH is not None
        and 0 <= launch < cells
        and I_ion_eV != 0.0
        and coulomb_model in _COULOMB_MODEL_CODE
        and anomalous_model in ("none", "quasilinear")
        and walk_secondaries == walk_terminal
    ):
        _csda_ctx = _csda_tables()
        if not E < _csda_ctx.exc_top:
            _csda_ctx = None
    if _csda_ctx is not None:
        (
            E,
            gamma,
            absorbed,
            anode_intercepted,
            _terminal_cell,
            _terminal_flux,
            _terminal_E,
        ) = _CSDA_MARCH(
            _csda_ctx,
            E,
            gamma,
            nn,
            ne,
            Te,
            dz_cm,
            launch,
            direction,
            I_ion_eV,
            E_stop_eV,
            frac,
            HE_OPB_EBAR_EV,
            _COULOMB_MODEL_CODE[coulomb_model],
            anomalous_model == "quasilinear",
            area if anomalous_model == "quasilinear" else None,
            anode_cross_index if intercept_active else -1,
            anode_eta,
            walk_products,
            walk_tail,
            ionization_events,
            excitation_events,
            heating,
            radiated,
            ionization_cost,
            E_entry,
            heat_coulomb,
            heat_anomalous,
            heat_secondary,
            heat_terminal,
            sec_flux if walk_products else None,
            sec_power_eV if walk_products else None,
            anom_power_eV if walk_tail else None,
        )
        if walk_products:
            terminal_cell = _terminal_cell
            terminal_flux = _terminal_flux
            terminal_E = _terminal_E
        order = ()

    for cell in order:
        # Anode-mesh interception (A15): the ray reaches the anode face only if
        # it survived the gap (a stopped beam breaks out before this cell), so
        # removing eta of the flux HERE removes exactly the long-mfp/bypass beam.
        # Book the intercepted primaries' remaining energy to the anode and carry
        # the reduced flux downstream.
        if intercept_active and cell == anode_cross_index:
            anode_intercepted += anode_eta * gamma * E * _ERG_PER_EV
            gamma *= 1.0 - anode_eta
            intercept_active = False
        E_entry[cell] = E
        remaining = float(dz_cm[cell])
        nn_c = float(nn[cell])
        ne_c = float(ne[cell])
        Te_c = float(Te[cell])
        # --- Per-cell banks, as Python floats (cost read 2026-08-02,
        # restructure B) ------------------------------------------------
        # The substep loop below used to write each bank straight into its
        # array: eight ``arr[cell] += scalar`` fancy-index stores per substep,
        # 0.825 us of a 5.70 us substep (14.5%), against ~0.01 us for a local
        # float. They accumulate here instead and are flushed once, at cell
        # exit.
        #
        # BIT-EXACT BY CONSTRUCTION, not by tolerance: ``for cell in order``
        # visits every cell EXACTLY once (a strictly monotone range) and every
        # target array starts at 0.0, so
        #
        #     arr[cell] += x1 ; arr[cell] += x2 ; ... ; arr[cell] += xn
        #
        # and
        #
        #     acc = 0.0 ; acc += x1 ; ... ; acc += xn ; arr[cell] += acc
        #
        # are the identical sequence of float64 additions from the identical
        # starting value. The flush stays a ``+=`` rather than a ``=`` so the
        # signed-zero case lands on the historical ``0.0`` too.
        acc_ionization_events = 0.0
        acc_excitation_events = 0.0
        acc_heating = 0.0
        acc_radiated = 0.0
        acc_ionization_cost = 0.0
        acc_heat_coulomb = 0.0
        acc_heat_anomalous = 0.0
        acc_heat_secondary = 0.0
        acc_heat_terminal = 0.0
        # WP-D / WP-E withholding banks; inert unless their closure is on.
        acc_sec_flux = 0.0
        acc_sec_power_eV = 0.0
        acc_anom_power_eV = 0.0
        while remaining > 0.0:
            sigma_i = (
                He_EII_cross_lkup(E / I_ion_eV) if E > I_ion_eV else 0.0
            )
            # Table-interpolated manifold channel (see _cross docstring):
            # exact-node table, ~1e-6 relative interp error, ~100x cheaper
            # than the scalar sums this loop used to spend ~80% of total
            # step time in (2026-07-21).
            sigma_x, E_rad = He_beam_excitation_channel_lkup(E)
            W_sec = he_mean_secondary_energy_eV(E, I_ion_eV=I_ion_eV)
            # channel loss rates [eV/cm]
            L_pot = nn_c * sigma_i * I_ion_eV
            L_sec = nn_c * sigma_i * W_sec
            L_exc = nn_c * sigma_x * E_rad
            L_coul = coulomb_stopping_eV_per_cm(
                E, ne_c, Te_c, model=coulomb_model
            )
            L_anom = 0.0
            if anomalous_model == "quasilinear":
                n_b = gamma / (
                    float(area[cell]) * beam_speed_cm_s(E)
                )
                l_ql = quasilinear_relaxation_length_cm(E, ne_c, n_b)
                if math.isfinite(l_ql) and l_ql > 0.0:
                    L_anom = E / l_ql
            elif anomalous_model == "ql_relaxation":
                n_b = gamma / (
                    float(area[cell]) * beam_speed_cm_s(E)
                )
                L_anom = ql_relaxation_stopping_eV_per_cm(
                    E, ne_c, nn_c, Te_c, n_b, ql_coeff
                )
            L_tot = L_pot + L_sec + L_exc + L_coul + L_anom
            if L_tot <= 0.0:
                break  # vacuum cell: free streaming
            dz_sub = min(remaining, frac * E / L_tot)
            # Land exactly on E_stop rather than overshooting through it.
            if E - L_tot * dz_sub <= E_stop_eV:
                dz_sub = (E - E_stop_eV) / L_tot
            if dz_sub <= 0.0:
                # E sits at E_stop to roundoff: absorb the residual here.
                if walk_terminal:
                    terminal_cell = cell
                    terminal_flux = gamma
                    terminal_E = E
                else:
                    acc_heating += gamma * E * _ERG_PER_EV
                    acc_heat_terminal += gamma * E * _ERG_PER_EV
                E = 0.0
                absorbed = True
                break
            # Bank each channel with the identical products the energy
            # decrement uses, so conservation closes to roundoff.
            d_pot = L_pot * dz_sub
            d_sec = L_sec * dz_sub
            d_exc = L_exc * dz_sub
            d_coul = L_coul * dz_sub
            d_anom = L_anom * dz_sub
            acc_ionization_cost += gamma * d_pot * _ERG_PER_EV
            # WP-E: under "tail_walk" the anomalous decrement is withheld from
            # every local bank in this cell (both the lumped one and the
            # diagnostic split) and accumulated for the tail walks below.
            # `d_anom_local` is `d_anom` when the walk is OFF, so the two
            # expressions below are then literally the historical ones --
            # which is what makes the default path bit-exact. The ray's own
            # energy decrement further down is UNCHANGED in both modes, so the
            # trajectory, the transmitted flux and every other channel are
            # bit-identical: only the destination of this one bank moves.
            if walk_tail:
                acc_anom_power_eV += gamma * d_anom
                d_anom_local = 0.0
            else:
                d_anom_local = d_anom
            if walk_secondaries:
                # Withhold the secondary bank from this cell; accumulate the
                # population (flux and energy) for the walks below. The flux
                # is the SAME product `ionization_events` uses, so the
                # particle rows are untouched by construction.
                acc_heating += gamma * (d_coul + d_anom_local) * _ERG_PER_EV
                acc_sec_flux += gamma * nn_c * sigma_i * dz_sub
                acc_sec_power_eV += gamma * d_sec
            else:
                acc_heating += (
                    gamma * (d_sec + d_coul + d_anom_local) * _ERG_PER_EV
                )
                acc_heat_secondary += gamma * d_sec * _ERG_PER_EV
            acc_heat_coulomb += gamma * d_coul * _ERG_PER_EV
            acc_heat_anomalous += gamma * d_anom_local * _ERG_PER_EV
            acc_radiated += gamma * d_exc * _ERG_PER_EV
            acc_ionization_events += gamma * nn_c * sigma_i * dz_sub
            acc_excitation_events += gamma * nn_c * sigma_x * dz_sub
            E -= d_pot + d_sec + d_exc + d_coul + d_anom
            remaining -= dz_sub
            if E <= E_stop_eV:
                # Sub-threshold residual: the primary can only Coulomb-drag
                # from here; bank the remainder as local plasma heating
                # (plan B1's stated closure) and end the ray. Under "nonlocal"
                # and "terminal_nonlocal" alike that same residual is instead
                # walked from this cell.
                if walk_terminal:
                    terminal_cell = cell
                    terminal_flux = gamma
                    terminal_E = E
                else:
                    acc_heating += gamma * E * _ERG_PER_EV
                    acc_heat_terminal += gamma * E * _ERG_PER_EV
                E = 0.0
                absorbed = True
                break
        # Flush this cell's banks (see the accumulator note above). Reached on
        # every exit from the substep loop -- the vacuum-cell break, both
        # absorption breaks, and running the cell's path out -- so no cell can
        # leave its banks unwritten.
        ionization_events[cell] += acc_ionization_events
        excitation_events[cell] += acc_excitation_events
        heating[cell] += acc_heating
        radiated[cell] += acc_radiated
        ionization_cost[cell] += acc_ionization_cost
        heat_coulomb[cell] += acc_heat_coulomb
        heat_anomalous[cell] += acc_heat_anomalous
        heat_secondary[cell] += acc_heat_secondary
        heat_terminal[cell] += acc_heat_terminal
        if walk_secondaries:
            sec_flux[cell] += acc_sec_flux
            sec_power_eV[cell] += acc_sec_power_eV
        if walk_tail:
            anom_power_eV[cell] += acc_anom_power_eV
        if absorbed:
            break

    if branch_tail:
        # --- pd1: the Landau/collisional branch ---------------------------
        # The march above withheld ALL of the anomalous power, exactly as the
        # tail walk does, so this is the ONE place the two dispositions differ
        # and it sits strictly between the march and the walk stage. Nothing
        # here reaches a kernel: the withholding bank is already final, and the
        # compiled and pure marches produce it bit-identically.
        #
        # The collisional share is returned to the local banks it was withheld
        # from -- the lumped one and the anomalous diagnostic split alike, the
        # same two the march's own `d_anom_local` feeds -- so the conservation
        # identity keeps its shipped FORM with the local share inside
        # `heating_anomalous`. The tail share is left in the bank for the walk
        # below, which then carries it with no knowledge that it was split.
        #
        # Cells the ray never reached carry 0.0 in the bank, so their local
        # credit is 0.0 whatever the branching says there.
        f_landau = landau_branching_fraction(ne, Te, nn, E0_eV)
        local_anom_erg = (1.0 - f_landau) * anom_power_eV * _ERG_PER_EV
        heating += local_anom_erg
        heat_anomalous += local_anom_erg
        anom_power_eV *= f_landau
    elif multigroup:
        # --- The plateau's two heirs (module docstring) -------------------
        # Structurally the pd1 branch's sibling and in the same place: the
        # march withheld ALL of the anomalous power, and this is where the
        # withheld bank is divided between a share that stays and a share that
        # streams. The difference is WHY -- the split here is the mean energy
        # of the derived dP/dE ~ E plateau over [E_1, E_b], not a local
        # damping ratio -- and that it is ONE number per ray rather than a
        # per-cell field, because the spectrum is a property of the extraction
        # and not of the cell the drag happened to be booked in.
        #
        # The wave/bulk share goes back to the local banks it was withheld
        # from, exactly as pd1's collisional share does, so the conservation
        # identity keeps its shipped FORM with the wave share inside
        # `heating_anomalous`. The streaming share is left in the bank for the
        # walk stage, which then splits it into groups.
        wave_anom_erg = plateau_wave_share * anom_power_eV * _ERG_PER_EV
        heating += wave_anom_erg
        heat_anomalous += wave_anom_erg
        plateau_wave_power = float(wave_anom_erg.sum())
        anom_power_eV *= plateau_stream_share

    if walk_products or walk_tail:
        # --- Product walks (WP-D) and QL tail walks (WP-E) ---------------
        # ONE set of walk machinery serves both closures: the same per-cell
        # stopping-power coefficient taken from the module's own
        # coulomb_stopping_eV_per_cm, the same closure energy exponent, and
        # the same thermalization floor; see _walk_products_forward for the
        # closed-form integration. WP-E introduces no walk physics of its own
        # -- only a different population to walk (see the module docstring).
        q = 1.0 - _COULOMB_STOPPING_EXPONENT[coulomb_model]
        # Restructure C: the caller may have built this already and be sharing
        # it across several rays / energy groups over the same (ne, Te, model).
        coeff = (
            _coulomb_stopping_coefficient(ne, Te, coulomb_model)
            if stopping_coefficient is None
            else stopping_coefficient
        )
        floor_eV = np.maximum(
            _PRODUCT_FLOOR_TE_MULTIPLE * Te, _PRODUCT_FLOOR_MIN_EV
        )

        def _walk_and_deposit(W0, flux, walk_direction, split, cull=None):
            """Walk one population; deposit it and RETURN what escaped.

            ``(escaping power [erg/s], escaping flux [1/s], cull tally)``. The
            caller books the escape to its own end ledger -- WP-D products to
            ``end_loss_*``, WP-E tails to ``end_loss_tail_*`` -- which is what
            keeps the two ledgers independently readable. ``cull`` is passed
            through to :func:`_walk_products`; the WP-D product walks never arm
            it (the anode mesh removes column-borne walkers, and a secondary or
            terminal residual is not one).
            """
            dep_eV, exit_eV, exit_flux, tally = _walk_products(
                W0, flux, walk_direction, coeff, dz_cm, floor_eV, q, cull=cull
            )
            dep_erg = dep_eV * _ERG_PER_EV
            heating[:] += dep_erg
            split[:] += dep_erg
            return exit_eV * _ERG_PER_EV, exit_flux, tally

        def _bank_walk(W0, flux, walk_direction, split):
            """Walk one product population and book its deposit and escape.

            Returns the escaping FLUX, which only the terminal population's
            caller reads (it is the charge that lands on the end surface).
            """
            nonlocal end_loss_low, end_loss_high
            exit_erg, exit_flux, _tally = _walk_and_deposit(
                W0, flux, walk_direction, split
            )
            if walk_direction > 0:
                end_loss_high += exit_erg
            else:
                end_loss_low += exit_erg
            return exit_flux

        if walk_secondaries and np.any(sec_flux > 0.0):
            # Flux-weighted mean secondary energy per birth cell (the module
            # carries mean energies, not distributions -- stated limitation),
            # emitted 50/50 along +z and -z (OPB emission is broadly
            # isotropic -- stated approximation).
            W_sec_cell = np.zeros(cells)
            born = sec_flux > 0.0
            W_sec_cell[born] = sec_power_eV[born] / sec_flux[born]
            half = 0.5 * sec_flux
            for walk_direction in (1, -1):
                _bank_walk(W_sec_cell, half, walk_direction, heat_secondary)
        if walk_terminal and terminal_flux > 0.0 and terminal_E > 0.0:
            # The terminal residual keeps the primary's direction. Its
            # escaping FLUX is kept: those electrons reached a terminating
            # surface, and a wall-charge model books their current there while
            # their energy left through the ledger above.
            term_flux = np.zeros(cells)
            term_W = np.zeros(cells)
            term_flux[terminal_cell] = terminal_flux
            term_W[terminal_cell] = terminal_E
            terminal_escape_flux = _bank_walk(
                term_W, term_flux, direction, heat_terminal
            )
        if walk_tail and np.any(anom_power_eV > 0.0):
            # WP-E: re-express each cell's withheld anomalous POWER as a flux
            # of tail electrons (flux = P / E, so flux*E returns the power to
            # roundoff), split 50/50 along +-B, and walk them on the shared
            # machinery above. The escape goes to the tail-only ledger, and
            # the deposition profile becomes the anomalous diagnostic split --
            # heating_anomalous now reports where the QL energy LANDS.
            #
            # This self-limits exactly like the WP-D walks: once n_e is high
            # the slowing length falls below a cell, the walker thermalizes in
            # its birth cell, and the closure collapses onto the local banking
            # it replaced.
            #
            # ONE population at the single plateau energy E_tail under the
            # single-line closures; under the multi-group plateau, N EQUAL-
            # POWER populations at the derived group midpoints, each carrying
            # 1/N of the streaming share (module docstring). The loop is the
            # only structural difference: every population goes through the
            # identical machinery below, so the single-line arms walk exactly
            # the floats they always did.
            # K7b: the tail power this ray actually launched, and which band
            # it was marched in. Read-only bookkeeping over a bank that is
            # already final -- nothing below consumes it. Under the multi-
            # group plateau the bank is already the STREAMING share alone (the
            # wave share was banked locally above), and the two band exposures
            # accumulate per group rather than being all-or-nothing.
            tail_power = float(anom_power_eV.sum()) * _ERG_PER_EV
            if multigroup:
                tail_populations = [
                    (float(E_hat), anom_power_eV / len(plateau_midpoints))
                    for E_hat in plateau_midpoints
                ]
            else:
                tail_populations = [(E_tail, anom_power_eV)]
            # Band, flux and window resolution runs for EVERY population
            # before any of them is walked, so the ionizing legs of the whole
            # ray form ONE lane batch below (a single group's legs are too few
            # to batch, and the plateau launches its groups from one ray).
            # Nothing is banked here and nothing is reordered: the two
            # exposure ledgers still fill in population order, and the window
            # refusal still raises on the population it always raised on.
            tail_plans = []
            for E_walk, walk_power_eV in tail_populations:
                half_flux = 0.5 * (walk_power_eV / E_walk)
                if multigroup:
                    # Each group is banded on its OWN energy: the bars are
                    # properties of the walker, and this ray launches several.
                    if tail_ionization == "on":
                        ionize_walk, _sub_g, _above_g = _tail_band(
                            E_walk, I_ion_eV, E_stop_eV,
                            "plateau group energy",
                        )
                    else:
                        ionize_walk, _sub_g, _above_g = False, False, False
                    _group_power = float(walk_power_eV.sum()) * _ERG_PER_EV
                    if _sub_g:
                        tail_sub_threshold_power += _group_power
                    elif _above_g:
                        tail_above_bar_power += _group_power
                else:
                    ionize_walk = ionize_tail
                    if tail_sub_threshold:
                        tail_sub_threshold_power = tail_power
                    elif tail_above_bar:
                        tail_above_bar_power = tail_power
                if ionize_walk or reflect_face is not None:
                    # Both WINDOWED closures stand on the same statement: the
                    # window must contain every cell the QL channel drives, or
                    # that cell's tail power would be dropped on the floor.
                    for birth in np.flatnonzero(half_flux > 0.0):
                        if not tail_lo <= birth <= tail_hi:
                            raise ValueError(
                                f"anomalous power in cell {int(birth)} lies "
                                f"outside tail_walk_window {(tail_lo, tail_hi)}; "
                                "the window must contain every cell the QL "
                                "channel drives, or that cell's tail power would "
                                "be silently dropped"
                            )
                tail_plans.append((E_walk, half_flux, ionize_walk))

            # K6: the walkers attenuate INELASTICALLY on the column gas as well
            # as Coulomb-slowing, so the closed-form integral above (which
            # knows only the Coulomb power law) cannot carry them. March them
            # on this module's own CSDA integration instead -- one leg per
            # birth cell and direction, the same instrument the primary uses,
            # so the cross sections, thresholds, <W_sec> convention and substep
            # control are the primary's by construction rather than by
            # transcription.
            #
            # anomalous_model="none": the plateau electrons ARE the
            # instability's product and do not re-drive it (and a walker that
            # drove its own QL drag would double-count the very power being
            # carried). product_transport="local": the depth-1 truncation
            # validated at construction -- secondaries and the sub-threshold
            # terminal residual bank where they are made. No anode
            # interception: these are born in the column, not streaming out of
            # the cathode through the mesh. No recursion risk: the nested call
            # takes anomalous_transport="local".
            #
            # The march runs on the WINDOWED domain, so the window's two faces
            # are walls: a walker that reaches one is transmitted out of the
            # sliced grid and booked to the tail end ledger, exactly as it
            # would be at a true domain end. That is what keeps every birth
            # inside cells the solver actually integrates.
            win = slice(tail_lo, tail_hi + 1)
            nn_w = nn[win]
            ne_w = ne[win]
            Te_w = Te[win]
            dz_w = dz_cm[win]
            march_kwargs = dict(
                I_ion_eV=I_ion_eV,
                E_stop_eV=E_stop_eV,
                coulomb_model=coulomb_model,
                anomalous_model="none",
                max_energy_fraction_per_substep=frac,
            )

            def _bank_tail_march(banks):
                """Book one marched tail LEG into the shared banks.

                Shared banks: the tail's events and energy join the
                primary's, which is what puts the born pair on the existing
                beam-ionization birth convention and its ``I_ion``
                investment on the existing cost sink.
                """
                leg_ion, leg_exc, leg_cost, leg_rad, leg_heat = banks
                ionization_events[win] += leg_ion
                excitation_events[win] += leg_exc
                ionization_cost[win] += leg_cost
                radiated[win] += leg_rad
                # All of the walker's HEAT (Coulomb drag, the local
                # <W_sec> secondaries, the terminal residual) is the
                # anomalous channel's delivery to the electrons, so it
                # lands in the lumped bank and in the anomalous split
                # -- never in the primary's coulomb/secondary/terminal
                # splits, which keep describing the primary alone.
                heating[win] += leg_heat
                heat_anomalous[win] += leg_heat
                # Diagnostic splits of the four shared banks.
                ion_events_tail[win] += leg_ion
                exc_events_tail[win] += leg_exc
                ion_cost_tail[win] += leg_cost
                radiated_tail[win] += leg_rad

            tail_chains, _take = _tail_lane_chains(
                tail_plans, nn_w, ne_w, Te_w, dz_w, march_kwargs,
                tail_lo, tail_hi, reflect_face, E_reflect,
                cull=(
                    None if not tail_cull
                    else (tail_anode_local, tail_anode_eta, R_e_tail,
                          eta_E_tail)
                ),
            )
            tail_anode_culled_flux += _take[0]
            tail_anode_culled_erg += _take[1] * _ERG_PER_EV
            tail_anode_returned_flux += _take[2]
            tail_anode_returned_erg += _take[3] * _ERG_PER_EV
            for (E_walk, half_flux, ionize_walk), chains in zip(
                tail_plans, tail_chains
            ):
                if ionize_walk:
                    for chain in chains:
                        # K7: at the ONE reflecting face, a walker whose
                        # arrival energy is below the sheath threshold is
                        # turned around at the same energy and marched back
                        # from the face cell. Only that face reflects, so the
                        # reversed leg cannot come back to it and a chain holds
                        # at most two legs.
                        for banks, _flux, _E_leg, _leg_dir in chain:
                            _bank_tail_march(banks)
                        _banks, leg_flux, leg_E, leg_dir = chain[-1]
                        # A walker still above E_stop at the window face it
                        # was heading for escapes, on the SAME free-escape
                        # convention the energy-only walk uses. Without a
                        # reflecting face no sheath or ambipolar throttle
                        # is applied at either end, which is what makes
                        # "tail_walk" the free-escape bound it is
                        # documented as with the channel on.
                        exit_erg = leg_flux * leg_E * _ERG_PER_EV
                        if leg_dir > 0:
                            end_loss_tail_high += exit_erg
                        else:
                            end_loss_tail_low += exit_erg
                elif reflect_face is not None:
                    # K7 energy-only walk with one reflecting window face. The
                    # closed-form integral above carries a population over a
                    # SEQUENCE of cells, so a reflection is expressed by UNFOLDING
                    # the path: the reflected leg is the window traversed back the
                    # other way, concatenated onto the incoming leg. The entry
                    # energy of the first cell of the second leg is the identical
                    # float that left the last cell of the first, so the walk still
                    # telescopes exactly across the bounce and the energy ledger
                    # closes at roundoff, as it does without reflection.
                    win = slice(tail_lo, tail_hi + 1)
                    n_w = tail_hi - tail_lo + 1
                    coeff_w = coeff[win]
                    dz_w = dz_cm[win]
                    floor_w = floor_eV[win]
                    flux_w = half_flux[win]
                    W0_w = np.full(n_w, E_walk)
                    # Window-local cell indices in traversal order for the arm that
                    # heads INTO the reflecting face, and for the arm that heads
                    # away from it (which is also the reflected leg's order).
                    order_hit = (
                        np.arange(n_w)[::-1] if reflect_face < 0 else np.arange(n_w)
                    )
                    order_away = order_hit[::-1]
                    escape_at_face = 0.0     # leaves through the reflecting face
                    escape_opposite = 0.0    # leaves through the other one

                    # The anode plane in each traversal order. ``_cull_at``
                    # returns the slot list the walk culls at (empty when the
                    # cull is off), so the plain arms name one crossing and the
                    # unfolded path names the two occurrences of one plane, of
                    # which the walk takes only the first each walker meets.
                    def _cull_at(*orders):
                        if not tail_cull:
                            return None
                        slots = []
                        for k, order in enumerate(orders):
                            hit = np.flatnonzero(order == tail_anode_local)
                            if hit.size:
                                slots.append(k * n_w + int(hit[0]))
                        return (np.array(sorted(slots), dtype=np.intp),
                                tail_anode_eta)

                    def _leg(order, W0, flux, cull=None):
                        return _walk_products_forward(
                            W0[order], flux[order], coeff_w[order], dz_w[order],
                            floor_w[order], q, cull=cull,
                        )

                    def _bank_tail_walk(dep_eV, *orders):
                        """Deposit one walked population, in ITS traversal order.

                        ``orders`` maps each block of ``dep_eV`` back onto window
                        cells: one block for a plain arm, two for the unfolded
                        reflected path. Banked per population, matching the arm-by-
                        arm conversion the unreflected walk does, so a walk in
                        which nothing reflects lands on the identical floats.
                        """
                        dep_win = np.zeros(n_w)
                        for k, order in enumerate(orders):
                            dep_win[order] += dep_eV[k * n_w:(k + 1) * n_w]
                        dep_erg = dep_win * _ERG_PER_EV
                        heating[win] += dep_erg
                        heat_anomalous[win] += dep_erg

                    def _bank_cull(tally):
                        nonlocal tail_anode_culled_flux, tail_anode_culled_erg
                        nonlocal tail_anode_returned_flux
                        nonlocal tail_anode_returned_erg
                        g_f, g_eV, r_f, r_eV = _tail_anode_take(
                            tally[0], tally[1], R_e_tail, eta_E_tail
                        )
                        if r_f > 0.0:
                            # Unreachable by construction: the rider is refused
                            # unless tail_ionization='on', and a population the
                            # K7b band reverts to THIS walk is below E_stop and
                            # so below the rider's energy floor. Asserted rather
                            # than assumed -- a silently dropped return would be
                            # a hole in the ledger, not a small error.
                            raise ValueError(
                                "the reversed-walker rider fired on the "
                                "energy-only tail walk, which has no launch to "
                                "reverse; this combination is refused at "
                                "construction and should be unreachable"
                            )
                        tail_anode_culled_flux += g_f
                        tail_anode_culled_erg += g_eV * _ERG_PER_EV

                    # The arm walking AWAY from the reflecting face never meets it.
                    dep_a, exit_a, _, cull_a = _leg(
                        order_away, W0_w, flux_w, cull=_cull_at(order_away)
                    )
                    _bank_tail_walk(dep_a, order_away)
                    escape_opposite += exit_a
                    if tail_cull:
                        _bank_cull(cull_a)
                    # The arm walking INTO it: test each population's ARRIVAL
                    # energy against the threshold. Populations born in different
                    # cells arrive with different energies, so this is a per-birth
                    # split, not a whole-arm switch.
                    dep_h, exit_h, (act_h, W_face, stop_h), cull_h = _leg(
                        order_hit, W0_w, flux_w, cull=_cull_at(order_hit)
                    )
                    bounced = (~stop_h) & (W_face < E_reflect)
                    if not bounced.any():
                        _bank_tail_walk(dep_h, order_hit)
                        escape_at_face += exit_h
                        if tail_cull:
                            _bank_cull(cull_h)
                    else:
                        flux_hit = flux_w[order_hit]
                        flux_bounce = np.zeros(n_w)
                        flux_escape = np.zeros(n_w)
                        flux_bounce[act_h[bounced]] = flux_hit[act_h[bounced]]
                        flux_escape[act_h[~bounced]] = flux_hit[act_h[~bounced]]
                        if np.any(flux_escape > 0.0):
                            dep_e, exit_e, _, cull_e = _walk_products_forward(
                                W0_w[order_hit], flux_escape, coeff_w[order_hit],
                                dz_w[order_hit], floor_w[order_hit], q,
                                cull=_cull_at(order_hit),
                            )
                            _bank_tail_walk(dep_e, order_hit)
                            escape_at_face += exit_e
                            if tail_cull:
                                _bank_cull(cull_e)
                        # The unfolded two-leg path. The face cell appears at the
                        # end of the first leg and again at the start of the second
                        # -- the reflected walker re-crosses it, the same
                        # cell-resolution granularity the marched walk has.
                        dep_u, exit_u, _, cull_u = _walk_products_forward(
                            np.concatenate([W0_w[order_hit], np.zeros(n_w)]),
                            np.concatenate([flux_bounce, np.zeros(n_w)]),
                            np.concatenate([coeff_w[order_hit], coeff_w[order_away]]),
                            np.concatenate([dz_w[order_hit], dz_w[order_away]]),
                            np.concatenate([floor_w[order_hit], floor_w[order_away]]),
                            q,
                            cull=_cull_at(order_hit, order_away),
                        )
                        _bank_tail_walk(dep_u, order_hit, order_away)
                        escape_opposite += exit_u
                        if tail_cull:
                            _bank_cull(cull_u)
                    if reflect_face > 0:
                        end_loss_tail_high += escape_at_face * _ERG_PER_EV
                        end_loss_tail_low += escape_opposite * _ERG_PER_EV
                    else:
                        end_loss_tail_low += escape_at_face * _ERG_PER_EV
                        end_loss_tail_high += escape_opposite * _ERG_PER_EV
                else:
                    tail_W = np.full(cells, E_walk)
                    # The cull is stated in FULL-GRID cell indices here: this
                    # arm walks the whole grid, not the window.
                    _cull_plain = (
                        None if not tail_cull
                        else (np.array([tail_anode_local + tail_lo],
                                       dtype=np.intp), tail_anode_eta)
                    )
                    for walk_direction in (1, -1):
                        exit_erg, _exit_flux, _tally = _walk_and_deposit(
                            tail_W, half_flux, walk_direction, heat_anomalous,
                            cull=_cull_plain,
                        )
                        if walk_direction > 0:
                            end_loss_tail_high += exit_erg
                        else:
                            end_loss_tail_low += exit_erg
                        if tail_cull:
                            g_f, g_eV, r_f, r_eV = _tail_anode_take(
                                _tally[0], _tally[1], R_e_tail, eta_E_tail
                            )
                            if r_f > 0.0:
                                raise ValueError(
                                    "the reversed-walker rider fired on the "
                                    "energy-only tail walk, which has no "
                                    "launch to reverse; this combination is "
                                    "refused at construction and should be "
                                    "unreachable"
                                )
                            tail_anode_culled_flux += g_f
                            tail_anode_culled_erg += g_eV * _ERG_PER_EV
        if book_transmitted and not absorbed and gamma > 0.0 and E > 0.0:
            # The transmitted primary: computed since B1, never banked. It
            # leaves through the end the ray was heading for.
            end_loss_transmitted = gamma * E * _ERG_PER_EV
            if direction > 0:
                end_loss_high += end_loss_transmitted
            else:
                end_loss_low += end_loss_transmitted

    # The tail's NET landing on the anode joins the primary's on the existing
    # row. Gross less what the rider sent back: the returned walkers re-enter
    # the plasma and are accounted for by the legs that walked them, so booking
    # the gross here would charge the anode for energy the column got.
    # Guarded rather than added unconditionally: with the cull off the whole
    # statement is skipped, so the primary's float reaches the result untouched
    # by construction and not by an argument about adding zero.
    if tail_cull:
        anode_intercepted += tail_anode_culled_erg - tail_anode_returned_erg

    return BeamDepositionResult(
        ionization_events=ionization_events,
        excitation_events=excitation_events,
        plasma_heating_erg_s=heating,
        radiated_erg_s=radiated,
        ionization_cost_erg_s=ionization_cost,
        transmitted_flux=0.0 if absorbed else gamma,
        transmitted_energy_eV=0.0 if absorbed else E,
        anode_intercepted_erg_s=anode_intercepted,
        E_entry_eV=E_entry,
        heating_coulomb_erg_s=heat_coulomb,
        heating_anomalous_erg_s=heat_anomalous,
        heating_secondary_erg_s=heat_secondary,
        heating_terminal_erg_s=heat_terminal,
        ionization_events_tail=ion_events_tail,
        excitation_events_tail=exc_events_tail,
        ionization_cost_tail_erg_s=ion_cost_tail,
        radiated_tail_erg_s=radiated_tail,
        end_loss_low_erg_s=end_loss_low,
        end_loss_high_erg_s=end_loss_high,
        end_loss_transmitted_erg_s=end_loss_transmitted,
        terminal_escape_flux_per_s=terminal_escape_flux,
        end_loss_tail_low_erg_s=end_loss_tail_low,
        end_loss_tail_high_erg_s=end_loss_tail_high,
        tail_power_erg_s=tail_power,
        tail_sub_threshold_power_erg_s=tail_sub_threshold_power,
        tail_above_bar_power_erg_s=tail_above_bar_power,
        plateau_wave_power_erg_s=plateau_wave_power,
        tail_anode_culled_flux_per_s=tail_anode_culled_flux,
        tail_anode_culled_erg_s=tail_anode_culled_erg,
        tail_anode_returned_flux_per_s=tail_anode_returned_flux,
        tail_anode_returned_erg_s=tail_anode_returned_erg,
    )


def _empty_deposition(cells, transmitted_flux=0.0, transmitted_energy_eV=0.0):
    """Return an all-zero :class:`BeamDepositionResult` of width ``cells``."""
    return BeamDepositionResult(
        ionization_events=np.zeros(cells),
        excitation_events=np.zeros(cells),
        plasma_heating_erg_s=np.zeros(cells),
        radiated_erg_s=np.zeros(cells),
        ionization_cost_erg_s=np.zeros(cells),
        transmitted_flux=float(transmitted_flux),
        transmitted_energy_eV=float(transmitted_energy_eV),
        anode_intercepted_erg_s=0.0,
        E_entry_eV=np.zeros(cells),
        heating_coulomb_erg_s=np.zeros(cells),
        heating_anomalous_erg_s=np.zeros(cells),
        heating_secondary_erg_s=np.zeros(cells),
        heating_terminal_erg_s=np.zeros(cells),
        ionization_events_tail=np.zeros(cells),
        excitation_events_tail=np.zeros(cells),
        ionization_cost_tail_erg_s=np.zeros(cells),
        radiated_tail_erg_s=np.zeros(cells),
    )


def deposit_beam_two_stream(
    E0_eV: float,
    Gamma0_per_s: float,
    *,
    f_cov: np.ndarray,
    nn_channel: np.ndarray,
    ne_channel: np.ndarray,
    nn_reservoir: np.ndarray,
    ne_reservoir: np.ndarray,
    Te: np.ndarray,
    launch: int,
    direction: int,
    dz_cm: np.ndarray,
    I_ion_eV: float = HE_I_ION_EV,
    E_stop_eV: float = HE_E_STOP_EV,
    coulomb_model: str = "fast_electron",
    anomalous_model: str = "none",
    beam_area_cm2: np.ndarray | float | None = None,
    ql_relaxation_coeff: float | None = None,
    max_energy_fraction_per_substep: float = 0.02,
    anode_cross_index: int | None = None,
    anode_eta: float = 0.0,
    product_transport: str = "local",
    anomalous_transport: str = "local",
    anomalous_disposal: str = "local",
    tail_energy_eV: float | None = None,
    tail_walk_window: tuple[int, int] | None = None,
    tail_ionization: str = "off",
    tail_reflect_face: int | None = None,
    tail_reflect_threshold_eV: float | None = None,
    stopping_coefficient: np.ndarray | None = None,
    nn_mean: np.ndarray | None = None,
    ne_mean: np.ndarray | None = None,
    tail_anode_cross_index: int | None = None,
    tail_anode_eta: float = 0.0,
    tail_anode_reflected_particles: float = 0.0,
    tail_anode_reflected_energy: float = 0.0,
):
    """March one beam ray through a Z-RESOLVED two-medium column (coverage v2).

    Returns ``(channel, reservoir, flux_entry_per_s)``: one
    :class:`BeamDepositionResult` per medium -- their per-cell banks ADD to the
    ray's total, which is what the caller consumes -- and the per-cell total
    primary flux ENTERING each cell [1/s] (zero for cells the ray never
    reaches), which is the ray's own probe-independent record of where its flux
    was lost.

    **What this is.** ``deposit_beam`` marches ONE stream through ONE medium.
    The coverage closure's v1.1 split ran it twice, partitioning the emitted
    flux once at the cathode face and letting each arm march its own medium to
    the end of the column. That is exact only while the covered fraction is
    uniform in z. With ``f_cov = f_cov(z)`` the two media exchange
    cross-sectional territory as the ray advances, so the partition has to be
    re-made at every cell:

    * enter cell ``k`` with a total surviving primary flux ``Gamma`` at mean
      energy ``E``;
    * re-split by the LOCAL coverage -- ``f_cov[k]*Gamma`` into the channel
      medium (``ne_channel``, ``nn_channel``) and ``(1-f_cov[k])*Gamma`` into
      the reservoir medium (``ne_reservoir``, ``nn_reservoir``);
    * march each arm across that cell's path length with its own substepping,
      banking into that arm's own per-cell arrays;
    * re-mix at cell exit: ``Gamma' = gamma_c' + gamma_r'`` and
      ``E' = (gamma_c'*E_c' + gamma_r'*E_r')/Gamma'``.

    **The stated approximation.** Re-mixing at every cell is the statement that
    the breakdown patches DECORRELATE AXIALLY ON THE CELL SCALE: an electron
    that crossed cell ``k`` inside a patch has no memory of that when it enters
    cell ``k+1``, so the population is re-randomised over the local
    cross-section there. The re-mix is a ONE-GROUP closure on that population --
    it carries the flux-weighted mean energy forward rather than two separate
    energies -- and it conserves both flux and power identically at every
    re-partition boundary (``Gamma'*E'`` is by construction
    ``gamma_c'*E_c' + gamma_r'*E_r'``), so nothing is created or lost by the
    re-partition itself. What it discards is the SPREAD of the primary energy
    distribution, which the CSDA model does not carry in the first place.

    **Uniform ``f_cov`` does NOT reproduce v1.1.** Re-mixing at every cell is a
    different model from partitioning once at emission, even when the profile
    is flat: v1.1's channel arm keeps whatever energy the channel medium left
    it, while here both arms are pulled back to the common mean at every cell
    face. Only ``f_cov == 1`` everywhere reduces exactly -- and then to the
    SHIPPED single-medium model, which the caller obtains by calling
    ``deposit_beam`` directly rather than by entering this function.

    **The quasilinear beam density is unchanged by the split.** Each arm
    carries ``f`` (or ``1-f``) of the flux over ``f`` (or ``1-f``) of the area,
    so ``n_b = Gamma/(A*v)`` in both and equal to the mean-field value.
    ``beam_area_cm2`` is therefore the FULL column area, exactly as it is for a
    mean-field ray; the arm areas are formed here.

    **The walk closures run ON THE MEAN STATE.** The walking
    ``product_transport`` values (``"nonlocal"``, ``"terminal_nonlocal"``)
    (WP-D) and ``anomalous_transport="tail_walk"`` (WP-E) withhold banks during
    the march and walk them afterwards. Both arms' per-cell withheld banks --
    which are per-arm per-cell by construction, so birth LOCATIONS are the
    march's own -- feed the ONE post-march walk stage, and that stage runs on
    the mean plasma state: neither the channel view ``n/f_cov`` nor the
    reservoir floor.

    That is the same patch-decorrelation closure the re-mix above stands on,
    applied to a product instead of a primary. A field-aligned product's path
    samples the channel medium with probability ``f_cov(z)`` per cell and the
    reservoir with ``1 - f_cov(z)``, so the stopping it sees on average is
    ``f_cov*(n/f_cov) + (1-f_cov)*n_floor``, i.e. the MEAN density: the
    concentration cancels exactly the way it cancels in a volumetric bilinear
    rate. The free-stream-to-walk transition is therefore EMERGENT rather than
    imposed -- the walk's reach is set by the same density-dependent Coulomb
    blocking the primary's drag law uses, so at pedestal densities a walker
    crosses the machine (it IS a free-streamer there) and localizes as the mean
    builds. No constant and no keyword is introduced for it.

    Two second-order misattributions are accepted, in opposite directions and
    both bounded by the patch-scale argument above:

    * a product born INSIDE a channel is briefly correlated with that channel,
      whose density is higher than the mean, so it thermalizes somewhat more
      locally than this mean-field walk predicts;
    * a product born in the RESERVOIR truly sees the floor density, so its reach
      is longer than the mean walk gives it. This one vanishes at both ends of
      the closure's own life: early, when the mean IS near the floor and the two
      media barely differ, and late, when ``1 - f_cov -> 0`` and there is no
      reservoir birth left to misplace.

    ``stopping_coefficient`` is REQUIRED once either walk is active and is the
    mean-state coefficient (see ``deposit_beam``'s hoisting note). It is not
    defaulted here: this function holds two media and no single ``(ne, Te)``
    pair it could honestly build one from, so a caller that has not decided
    which state the walk runs on gets an error rather than a quiet choice.

    **Burn attribution under ``tail_ionization="on"`` is THE SAME PARTITION.**
    An ionizing walker also removes neutrals, and the medium it removed them
    from follows from the identical statement: at cell ``z`` the walker's path
    lies inside a channel for a fraction ``f_cov(z)`` of its cross-section and
    inside the reservoir for ``1 - f_cov(z)``, so its per-cell ionization
    EVENTS are split that way -- ``f_cov`` of them debit the covered column and
    ``1 - f_cov`` the reservoir. That split is expressed by banking the walker's
    events into the two ARMS with those weights, which is why it needs no
    machinery of its own: the caller already reads the reservoir arm's
    ionization rows as the reservoir debit, and the sum of the arms as the
    total. The walk itself still runs on the mean medium (``nn_mean``,
    ``ne_mean``, required with the channel on), the births are still booked to
    the mean fields, and the two misattribution bounds above carry over
    unchanged -- they are statements about the same sampling argument.

    The walked HEAT is booked to the CHANNEL arm's banks, on the same
    convention the transmitted primary already uses above: after the walk the
    energy belongs to the mean field and no medium owns it, and the caller
    consumes the two arms' sum. Splitting the ionization cost and the radiated
    energy by the same weights therefore leaves the energy side exactly where
    it shipped -- only the sum is read.

    **``anomalous_disposal="landau_branched"`` is REFUSED here** (pd1), and the
    refusal is the design rather than a gap in it. Three code facts decide it:

    * the withholding bank is SHARED across the two arms and indexed by birth
      cell (see its comment below), because the one walk stage runs on the mean
      state for both. The power the RESERVOIR arm extracted is therefore not
      separable from the channel arm's at the point the branch would be
      applied, so a per-medium branching cannot be expressed without giving the
      bank a per-arm axis it deliberately does not have;
    * a single MEAN-FIELD ``f_Landau`` applied to that shared bank would branch
      the reservoir's extraction on a density that is not the reservoir's --
      and under ``anomalous_model="ql_relaxation"`` the reservoir arm is the
      DOMINANT extractor, so that is not a small misattribution;
    * a per-arm branch would evaluate the reservoir's ``f_Landau`` at
      ``ne = ne_floor``, a numerical floor constant standing for "no plasma",
      against the MEAN-FIELD ``Te`` this march shares between the media. That
      combination is not a physical state: at the floor with a hot ``Te`` it
      returns ``f_Landau`` ~ 0.84 (Te 25 eV) to ~0.98 (Te 55 eV), i.e. the
      branching in the reservoir would be an artifact of the floor convention
      rather than a measurement of it.

    The honest disposition is therefore to refuse the combination at
    construction and leave the coverage arms to a stance designed for them,
    rather than to ship a number the floor convention owns.
    """
    if anomalous_model not in ANOMALOUS_MODELS:
        raise ValueError(
            f"unknown anomalous_model {anomalous_model!r}; "
            f"expected one of {sorted(ANOMALOUS_MODELS)}"
        )
    if anomalous_disposal != "local":
        # See the docstring for the three code facts behind this refusal. The
        # value domain is checked first so a typo reads as a typo rather than
        # as the coverage refusal.
        if anomalous_disposal != "landau_branched":
            raise ValueError(
                f"unknown anomalous_disposal {anomalous_disposal!r}; "
                "expected 'local' or 'landau_branched'"
            )
        raise ValueError(
            "anomalous_disposal='landau_branched' is not available under the "
            "two-stream (coverage) march: the withholding bank is shared "
            "across the channel and reservoir arms by construction, so the "
            "reservoir's extracted power cannot be branched on the reservoir's "
            "own state -- and the reservoir carries ne = ne_floor, a numerical "
            "floor constant, against the mean-field Te, so any branching there "
            "would be an artifact of the floor convention rather than a "
            "measurement. The coverage arms are deferred until that stance is "
            "designed; run the branch without coverage_closure"
        )
    if anode_eta != 0.0 and not (0.0 <= anode_eta < 1.0):
        raise ValueError(f"anode_eta must be in [0, 1) (got {anode_eta})")
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1 (got {direction})")
    dz_cm = np.asarray(dz_cm, dtype=float)
    cells = dz_cm.size
    f_cov = np.asarray(f_cov, dtype=float)
    media = tuple(
        np.asarray(a, dtype=float)
        for a in (nn_channel, ne_channel, nn_reservoir, ne_reservoir, Te)
    )
    if f_cov.shape != (cells,) or any(a.shape != (cells,) for a in media):
        raise ValueError(
            "f_cov, nn_channel, ne_channel, nn_reservoir, ne_reservoir, Te "
            f"and dz_cm must share one shape (cells,) = {(cells,)}"
        )
    if not np.all(np.isfinite(f_cov)) or np.any(f_cov <= 0.0) or np.any(
        f_cov > 1.0
    ):
        raise ValueError(
            "f_cov must be finite and in (0, 1] per cell (got min "
            f"{float(np.min(f_cov)):.6g}, max {float(np.max(f_cov)):.6g})"
        )
    nn_ch, ne_ch, nn_res, ne_res, Te = media
    if anode_cross_index is not None:
        anode_cross_index = int(anode_cross_index)
        if not 0 <= anode_cross_index < cells:
            raise ValueError(
                "anode_cross_index must index a cell in [0, cells) "
                f"(got {anode_cross_index}, cells={cells})"
            )
    area = None
    if anomalous_model in ("quasilinear", "ql_relaxation"):
        if beam_area_cm2 is None:
            raise ValueError(
                f"anomalous_model={anomalous_model!r} needs beam_area_cm2"
            )
        area = np.broadcast_to(
            np.asarray(beam_area_cm2, dtype=float), (cells,)
        )
    ql_coeff = 0.0
    if anomalous_model == "ql_relaxation":
        # No default is substituted here. The plateau-formation coefficient is
        # a REGISTERED BRACKET whose endpoints every headline is quoted at, so
        # a caller that has not stated which arm it is on has not configured
        # the closure -- picking one silently would put an unreported bracket
        # arm behind a published number.
        if ql_relaxation_coeff is None:
            raise ValueError(
                "anomalous_model='ql_relaxation' needs ql_relaxation_coeff "
                "(the registered O(10-100) plateau-formation bracket "
                "constant); there is deliberately no default here"
            )
        ql_coeff = float(ql_relaxation_coeff)
        if not math.isfinite(ql_coeff) or ql_coeff <= 0.0:
            raise ValueError(
                "ql_relaxation_coeff must be finite and > 0 (got "
                f"{ql_relaxation_coeff})"
            )
    frac = float(max_energy_fraction_per_substep)
    if not 0.0 < frac < 1.0:
        raise ValueError(
            "max_energy_fraction_per_substep must be in (0, 1), got "
            f"{max_energy_fraction_per_substep}"
        )
    if product_transport not in ("local", "nonlocal", "terminal_nonlocal"):
        raise ValueError(
            f"unknown product_transport {product_transport!r}; "
            "expected 'local', 'nonlocal' or 'terminal_nonlocal'"
        )
    if anomalous_transport == "plateau_multigroup":
        # Refused for the same reason the pd1 branch is (see the block above):
        # this march shares ONE withholding bank between the channel and
        # reservoir arms, and the reservoir carries the density FLOOR against
        # the mean-field Te -- so a plateau edge solved on the launch cell of
        # a two-medium column would be an artifact of the floor convention
        # rather than a measurement of the plasma. The coverage arms of this
        # closure are deferred until that stance is designed.
        raise ValueError(
            "anomalous_transport='plateau_multigroup' does not support the "
            "two-stream (coverage) march: the withholding bank is shared "
            "between the channel and reservoir arms and the reservoir carries "
            "the density FLOOR, so the plateau edge E_1 solved there would be "
            "an artifact of the floor convention. Run the closure without "
            "coverage_closure"
        )
    if anomalous_transport not in ("local", "tail_walk"):
        raise ValueError(
            f"unknown anomalous_transport {anomalous_transport!r}; "
            "expected 'local' or 'tail_walk'"
        )
    if tail_ionization not in ("off", "on"):
        raise ValueError(
            f"unknown tail_ionization {tail_ionization!r}; "
            "expected 'off' or 'on'"
        )
    if tail_ionization == "on" and anomalous_transport != "tail_walk":
        raise ValueError(
            "tail_ionization='on' requires anomalous_transport='tail_walk' "
            "(the ionizing channel belongs to the QL tail walkers; with "
            "anomalous_transport='local' there are no walkers and the "
            "setting would do nothing). anomalous_transport accepts 'local' "
            "or 'tail_walk'; tail_ionization accepts 'off' or 'on'"
        )
    # Same population-by-population gating deposit_beam uses; see the comment
    # on its own booleans for why each branch tests the population and not the
    # selector.
    walk_secondaries = product_transport == "nonlocal"
    walk_terminal = product_transport in ("nonlocal", "terminal_nonlocal")
    book_transmitted = product_transport == "nonlocal"
    walk_products = walk_secondaries or walk_terminal
    walk_tail = anomalous_transport == "tail_walk"
    ionize_tail = tail_ionization == "on"
    tail_sub_threshold = False
    tail_above_bar = False
    E_tail = 0.0
    if walk_tail:
        # Same presence gating deposit_beam applies: with no anomalous channel
        # there is no power for the walk to carry and the setting is a no-op.
        if anomalous_model == "none":
            raise ValueError(
                "anomalous_transport='tail_walk' requires an active anomalous "
                "channel (anomalous_model='quasilinear' or "
                "'ql_relaxation'); with no anomalous "
                "drag there is no power to carry and the setting would do "
                "nothing"
            )
        if tail_energy_eV is None:
            raise ValueError(
                "anomalous_transport='tail_walk' needs tail_energy_eV (the "
                "QL plateau energy the tail electrons are launched at)"
            )
        E_tail = float(tail_energy_eV)
        if not math.isfinite(E_tail) or E_tail <= 0.0:
            raise ValueError(
                f"tail_energy_eV must be finite and > 0 (got {tail_energy_eV})"
            )
    reflect_face = None
    E_reflect = 0.0
    if tail_reflect_face is not None:
        if not walk_tail:
            raise ValueError(
                "tail_reflect_face requires anomalous_transport='tail_walk' "
                "(it reflects the QL tail walkers; with no walk there is "
                "nothing to reflect and the setting would do nothing)"
            )
        reflect_face = int(tail_reflect_face)
        if reflect_face not in (-1, 1):
            raise ValueError(
                "tail_reflect_face must be -1 (the walk window's low-index "
                f"face) or +1 (its high-index face), got {tail_reflect_face!r}"
            )
        if tail_reflect_threshold_eV is None:
            raise ValueError(
                "tail_reflect_face needs tail_reflect_threshold_eV (the "
                "energy below which a walker arriving at that face is turned "
                "around instead of escaping)"
            )
        E_reflect = float(tail_reflect_threshold_eV)
        if not math.isfinite(E_reflect) or E_reflect <= 0.0:
            raise ValueError(
                "tail_reflect_threshold_eV must be finite and > 0 (got "
                f"{tail_reflect_threshold_eV})"
            )
    elif tail_reflect_threshold_eV is not None:
        raise ValueError(
            "tail_reflect_threshold_eV was given without tail_reflect_face; "
            "the threshold belongs to a named reflecting face and on its own "
            "would silently do nothing"
        )
    tail_lo, tail_hi = 0, cells - 1
    if reflect_face is not None and not ionize_tail:
        if tail_walk_window is None:
            raise ValueError(
                "tail_reflect_face needs tail_walk_window=(lo, hi): the "
                "reflecting face is one of that window's two faces, and "
                "without it the walk has no face to reflect at"
            )
        tail_lo, tail_hi = (int(tail_walk_window[0]), int(tail_walk_window[1]))
        if not 0 <= tail_lo <= tail_hi < cells:
            raise ValueError(
                "tail_walk_window must be an inclusive (lo, hi) cell range "
                f"with 0 <= lo <= hi < cells={cells} (got {tail_walk_window})"
            )
    if ionize_tail:
        # The walk DOMAIN is required, not defaulted to the whole grid: an
        # ionizing walker that leaves the plasma births pairs into rows the
        # solver's active mask zeroes, i.e. pairs created and silently deleted.
        if tail_walk_window is None:
            raise ValueError(
                "tail_ionization='on' needs tail_walk_window=(lo, hi), the "
                "inclusive cell range the tail walkers may traverse (the "
                "plasma-active window: a walker leaving it hits a wall and is "
                "booked to the tail end ledger). Without it the walk would "
                "run off into cells whose plasma rows the solver zeroes and "
                "birth pairs that are then deleted"
            )
        tail_lo, tail_hi = (int(tail_walk_window[0]), int(tail_walk_window[1]))
        if not 0 <= tail_lo <= tail_hi < cells:
            raise ValueError(
                "tail_walk_window must be an inclusive (lo, hi) cell range "
                f"with 0 <= lo <= hi < cells={cells} (got {tail_walk_window})"
            )
        # THE MEAN MEDIUM the walkers march through. Required for the same
        # reason the stopping coefficient is: this march holds a channel view
        # and a reservoir view, and the ruling says the walk runs on neither.
        if nn_mean is None or ne_mean is None:
            raise ValueError(
                "tail_ionization='on' under a two-stream march needs nn_mean "
                "and ne_mean: the ionizing walkers march the MEAN state, and "
                "this march carries only the channel and reservoir views, so "
                "it cannot form the mean without silently choosing a medium"
            )
        nn_mean = np.asarray(nn_mean, dtype=float)
        ne_mean = np.asarray(ne_mean, dtype=float)
        if nn_mean.shape != (cells,) or ne_mean.shape != (cells,):
            raise ValueError(
                "nn_mean and ne_mean must have one entry per grid cell "
                f"(cells={cells}); got {nn_mean.shape} and {ne_mean.shape}"
            )
        # --- The K7b band split, transcribed from deposit_beam --------------
        # The two depth-1 bars select a TREATMENT rather than refusing; only
        # the EII table edge is still a refusal, and it is edge-INCLUSIVE
        # within HE_EII_EDGE_REL_TOL (at the edge the clamped cross section IS
        # the table's endpoint value, so nothing is extrapolated).
        _E_table_top = HE_EII_EPS_TOP * I_ion_eV
        _edge_excess = (E_tail - _E_table_top) / _E_table_top
        if _edge_excess > HE_EII_EDGE_REL_TOL:
            raise ValueError(
                "tail_ionization='on' marches the walkers on the tabulated "
                "He EII cross section, which ends at eps = E/I_ion = "
                f"{HE_EII_EPS_TOP:.6f} (i.e. "
                f"{_E_table_top:.2f} eV at I_ion_eV={I_ion_eV}); "
                f"at tail_energy_eV={E_tail} eV the lookup would clamp to its "
                "last node and the walk would attenuate on an extrapolated "
                "cross section. This is refused, not approximated (relative "
                f"excess {_edge_excess:.3e}, tolerated "
                f"{HE_EII_EDGE_REL_TOL:.1e})"
            )
        if E_tail <= E_stop_eV:
            # SUB-THRESHOLD: no He inelastic channel is open, so zero
            # ionization is the answer, not a modelling choice. Revert this ray
            # to the energy-only walk by clearing the flag.
            tail_sub_threshold = True
            ionize_tail = False
        elif (
            he_mean_secondary_energy_eV(E_tail, I_ion_eV=I_ion_eV)
            >= E_stop_eV
        ):
            # ABOVE THE DEPTH-1 BAR: the mean secondary can itself do something
            # inelastic, so banking it locally understates the cascade.
            # Allowed, with the understatement measured and the exposure
            # reported.
            tail_above_bar = True
    # --- A2a: the anode-mesh cull of the QL tail, and its rider -----------
    # Transcribed from ``deposit_beam``'s block, on the same window resolution
    # and with the same refusals -- the mesh is the same obstruction whichever
    # march carries the walkers past it.
    R_e_tail = float(tail_anode_reflected_particles)
    eta_E_tail = float(tail_anode_reflected_energy)
    tail_cull = tail_anode_cross_index is not None and tail_anode_eta > 0.0
    tail_anode_local = -1
    if tail_anode_eta != 0.0 and not (0.0 <= tail_anode_eta < 1.0):
        raise ValueError(
            f"tail_anode_eta must be in [0, 1) (got {tail_anode_eta})"
        )
    for _name, _val in (
        ("tail_anode_reflected_particles", R_e_tail),
        ("tail_anode_reflected_energy", eta_E_tail),
    ):
        if not math.isfinite(_val) or not 0.0 <= _val <= 1.0:
            raise ValueError(f"{_name} must be in [0, 1] (got {_val})")
    if eta_E_tail > R_e_tail:
        raise ValueError(
            "tail_anode_reflected_energy must not exceed "
            "tail_anode_reflected_particles (got "
            f"{eta_E_tail} > {R_e_tail}): both are PER INCIDENT, so the "
            "returned energy fraction is the returned particle fraction times "
            "the mean returned energy in units of the incident energy, and "
            "that mean cannot exceed one"
        )
    if (R_e_tail > 0.0 or eta_E_tail > 0.0) and not tail_cull:
        raise ValueError(
            "the reversed-walker rider "
            "(tail_anode_reflected_particles/tail_anode_reflected_energy) "
            "needs the tail cull it rides on: give tail_anode_cross_index "
            "with tail_anode_eta > 0. Without the cull nothing is intercepted "
            "and the pair would be a silent no-op"
        )
    if R_e_tail > 0.0 and tail_ionization != "on":
        raise ValueError(
            "the reversed-walker rider requires tail_ionization='on': the "
            "energy-only closed-form walk carries a whole cell sequence in one "
            "telescoping integral and has no per-walker launch to reverse. "
            "The cull itself composes with either walk; only the return does "
            "not, and it is refused rather than approximated"
        )
    if tail_cull:
        if not walk_tail:
            raise ValueError(
                "tail_anode_cross_index/tail_anode_eta cull the QL TAIL "
                "walkers, so they need a walked tail: select "
                "anomalous_transport='tail_walk' (or "
                "anomalous_disposal='landau_branched'). With neither there "
                "are no walkers to cull and the pair would do nothing"
            )
        tail_anode_local = int(tail_anode_cross_index) - tail_lo
        if not 0 <= tail_anode_local <= tail_hi - tail_lo:
            raise ValueError(
                f"tail_anode_cross_index={tail_anode_cross_index} lies outside "
                f"the tail walk window {(tail_lo, tail_hi)}; the walkers never "
                "reach that cell, so the cull would be a silent no-op"
            )
    tail_anode_culled_flux = 0.0
    tail_anode_culled_erg = 0.0
    tail_anode_returned_flux = 0.0
    tail_anode_returned_erg = 0.0
    if walk_products or walk_tail:
        # THE MEAN-STATE HAND-OFF, made structural. There is no medium here the
        # walk could fall back on: this function holds a channel view and a
        # reservoir view and the closure says the walk runs on NEITHER, so the
        # coefficient is required rather than built.
        if stopping_coefficient is None:
            raise ValueError(
                "the walk closures under a two-stream march need an explicit "
                "stopping_coefficient: the walk runs on the MEAN plasma state, "
                "and this march carries two media (channel n/f_cov and "
                "reservoir floor) but not the mean, so it cannot build one "
                "without silently choosing a medium. Pass the caller's "
                "mean-state _coulomb_stopping_coefficient(ne_mean, Te, model)"
            )
        stopping_coefficient = np.asarray(stopping_coefficient, dtype=float)
        if stopping_coefficient.shape != (cells,):
            raise ValueError(
                "stopping_coefficient must have one entry per grid cell "
                f"(cells={cells}); got shape {stopping_coefficient.shape}"
            )

    # Per-arm banks. Index 0 is the CHANNEL arm, 1 the RESERVOIR arm; they are
    # never summed here, so the caller keeps the reservoir arm's ionization
    # separable for the closure's deficit equation.
    banks = tuple(
        {
            name: np.zeros(cells)
            for name in (
                "ionization_events",
                "excitation_events",
                "heating",
                "radiated",
                "ionization_cost",
                "heat_coulomb",
                "heat_anomalous",
                "heat_secondary",
                "heat_terminal",
                "E_entry",
                # K6 diagnostic splits. Present (and zero) on both arms
                # whatever the closure, so the result shape never depends on
                # which branch ran.
                "ion_events_tail",
                "exc_events_tail",
                "ion_cost_tail",
                "radiated_tail",
            )
        }
        for _ in range(2)
    )
    flux_entry = np.zeros(cells)
    anode_intercepted = [0.0, 0.0]
    # WP-D / WP-E withholding banks, SHARED across the arms and indexed by
    # BIRTH CELL. Both arms withhold into the same per-cell slot because the
    # one walk stage below runs on the mean state for both, so a per-arm split
    # would only be summed again before it was used; the birth LOCATION, which
    # is the thing the walk consumes, is the march's own either way. The energy
    # each slot carries is a flux-weighted mean, exactly the convention
    # deposit_beam uses across the substeps within one cell.
    sec_flux = np.zeros(cells) if walk_secondaries else None
    sec_power_eV = np.zeros(cells) if walk_secondaries else None
    term_flux = np.zeros(cells) if walk_terminal else None
    term_power_eV = np.zeros(cells) if walk_terminal else None
    anom_power_eV = np.zeros(cells) if walk_tail else None
    end_loss_low = 0.0
    end_loss_high = 0.0
    end_loss_transmitted = 0.0
    terminal_escape_flux = 0.0
    end_loss_tail_low = 0.0
    end_loss_tail_high = 0.0
    tail_power = 0.0
    tail_sub_threshold_power = 0.0
    tail_above_bar_power = 0.0

    E = float(E0_eV)
    gamma_total = float(Gamma0_per_s)
    if E <= E_stop_eV:
        # Sub-threshold source: nothing inelastic can happen, so the ray passes
        # through untouched and the whole flux is transmitted. Mirrors
        # deposit_beam's own early return; the coverage wiring never launches
        # such a ray (phi_c > I_ion > E_stop).
        chan = _empty_deposition(cells, gamma_total, E)
        return chan, _empty_deposition(cells), flux_entry

    order = range(launch, cells) if direction > 0 else range(launch, -1, -1)
    intercept_active = anode_cross_index is not None and anode_eta > 0.0
    absorbed = False
    for cell in order:
        # Anode-mesh interception (A15). The mesh is a GEOMETRIC obstruction
        # across the whole cross-section, so it removes the same solid fraction
        # from both media; applying it to the mixed flux before the re-split is
        # therefore equivalent to applying it per arm, and books the
        # intercepted energy on the arm shares the cell is about to use.
        f = float(f_cov[cell])
        # Recorded BEFORE the mesh takes its share: ``flux_entry`` is what
        # ARRIVED at this cell, the flux counterpart of ``E_entry`` (which the
        # interception cannot move, since it removes primaries rather than
        # slowing them). A caller measuring where the ray lost flux to the
        # PLASMA -- the gap-survival ledger -- must not see a solid obstruction
        # counted as stopping.
        flux_entry[cell] = gamma_total
        if intercept_active and cell == anode_cross_index:
            lost = anode_eta * gamma_total * E * _ERG_PER_EV
            anode_intercepted[0] += f * lost
            anode_intercepted[1] += (1.0 - f) * lost
            gamma_total *= 1.0 - anode_eta
            intercept_active = False
        # The local re-split, by AREA. At f == 1 the reservoir arm has no
        # cross-section at all and is skipped entirely, so a fully-covered cell
        # costs exactly what a single-medium cell costs.
        arms = [(0, f, float(nn_ch[cell]), float(ne_ch[cell]))]
        if f < 1.0:
            arms.append(
                (1, 1.0 - f, float(nn_res[cell]), float(ne_res[cell]))
            )
        Te_c = float(Te[cell])
        dz_cell = float(dz_cm[cell])
        # n_b is formed on the TOTAL flux over the FULL area: each arm carries
        # its share of the flux over the matching share of the area, so the two
        # factors cancel and both arms see the mean-field beam density.
        area_c = 0.0 if area is None else float(area[cell])
        surviving_weight = 0.0
        surviving_energy_weight = 0.0
        for arm, weight, nn_c, ne_c in arms:
            bank = banks[arm]
            gamma = weight * gamma_total
            bank["E_entry"][cell] = E
            E_arm = E
            remaining = dz_cell
            arm_absorbed = False
            acc_ionization_events = 0.0
            acc_excitation_events = 0.0
            acc_heating = 0.0
            acc_radiated = 0.0
            acc_ionization_cost = 0.0
            acc_heat_coulomb = 0.0
            acc_heat_anomalous = 0.0
            acc_heat_secondary = 0.0
            acc_heat_terminal = 0.0
            # This arm's withheld populations in this cell; inert unless their
            # closure is on.
            acc_sec_flux = 0.0
            acc_sec_power_eV = 0.0
            acc_anom_power_eV = 0.0
            while remaining > 0.0:
                sigma_i = (
                    He_EII_cross_lkup(E_arm / I_ion_eV)
                    if E_arm > I_ion_eV
                    else 0.0
                )
                sigma_x, E_rad = He_beam_excitation_channel_lkup(E_arm)
                W_sec = he_mean_secondary_energy_eV(E_arm, I_ion_eV=I_ion_eV)
                L_pot = nn_c * sigma_i * I_ion_eV
                L_sec = nn_c * sigma_i * W_sec
                L_exc = nn_c * sigma_x * E_rad
                L_coul = coulomb_stopping_eV_per_cm(
                    E_arm, ne_c, Te_c, model=coulomb_model
                )
                L_anom = 0.0
                if anomalous_model == "quasilinear":
                    n_b = gamma_total / (area_c * beam_speed_cm_s(E_arm))
                    l_ql = quasilinear_relaxation_length_cm(E_arm, ne_c, n_b)
                    if math.isfinite(l_ql) and l_ql > 0.0:
                        L_anom = E_arm / l_ql
                elif anomalous_model == "ql_relaxation":
                    n_b = gamma_total / (area_c * beam_speed_cm_s(E_arm))
                    L_anom = ql_relaxation_stopping_eV_per_cm(
                        E_arm, ne_c, nn_c, Te_c, n_b, ql_coeff
                    )
                L_tot = L_pot + L_sec + L_exc + L_coul + L_anom
                if L_tot <= 0.0:
                    break  # vacuum cell: free streaming
                dz_sub = min(remaining, frac * E_arm / L_tot)
                if E_arm - L_tot * dz_sub <= E_stop_eV:
                    dz_sub = (E_arm - E_stop_eV) / L_tot
                if dz_sub <= 0.0:
                    if walk_terminal:
                        term_flux[cell] += gamma
                        term_power_eV[cell] += gamma * E_arm
                    else:
                        acc_heating += gamma * E_arm * _ERG_PER_EV
                        acc_heat_terminal += gamma * E_arm * _ERG_PER_EV
                    E_arm = 0.0
                    arm_absorbed = True
                    break
                d_pot = L_pot * dz_sub
                d_sec = L_sec * dz_sub
                d_exc = L_exc * dz_sub
                d_coul = L_coul * dz_sub
                d_anom = L_anom * dz_sub
                acc_ionization_cost += gamma * d_pot * _ERG_PER_EV
                # WP-E: under the tail walk the anomalous decrement is withheld
                # from every local bank in this cell and carried to the walk
                # stage. ``d_anom_local`` is ``d_anom`` with the walk off, so
                # the expressions below are then literally the ones this march
                # shipped with. The ARM's own energy decrement further down is
                # unchanged in both modes, so the trajectory, the re-mix and
                # every other channel are untouched: only this bank's
                # destination moves.
                if walk_tail:
                    acc_anom_power_eV += gamma * d_anom
                    d_anom_local = 0.0
                else:
                    d_anom_local = d_anom
                if walk_secondaries:
                    acc_heating += (
                        gamma * (d_coul + d_anom_local) * _ERG_PER_EV
                    )
                    acc_sec_flux += gamma * nn_c * sigma_i * dz_sub
                    acc_sec_power_eV += gamma * d_sec
                else:
                    acc_heating += (
                        gamma * (d_sec + d_coul + d_anom_local) * _ERG_PER_EV
                    )
                    acc_heat_secondary += gamma * d_sec * _ERG_PER_EV
                acc_heat_coulomb += gamma * d_coul * _ERG_PER_EV
                acc_heat_anomalous += gamma * d_anom_local * _ERG_PER_EV
                acc_radiated += gamma * d_exc * _ERG_PER_EV
                acc_ionization_events += gamma * nn_c * sigma_i * dz_sub
                acc_excitation_events += gamma * nn_c * sigma_x * dz_sub
                E_arm -= d_pot + d_sec + d_exc + d_coul + d_anom
                remaining -= dz_sub
                if E_arm <= E_stop_eV:
                    if walk_terminal:
                        term_flux[cell] += gamma
                        term_power_eV[cell] += gamma * E_arm
                    else:
                        acc_heating += gamma * E_arm * _ERG_PER_EV
                        acc_heat_terminal += gamma * E_arm * _ERG_PER_EV
                    E_arm = 0.0
                    arm_absorbed = True
                    break
            bank["ionization_events"][cell] += acc_ionization_events
            bank["excitation_events"][cell] += acc_excitation_events
            bank["heating"][cell] += acc_heating
            bank["radiated"][cell] += acc_radiated
            bank["ionization_cost"][cell] += acc_ionization_cost
            bank["heat_coulomb"][cell] += acc_heat_coulomb
            bank["heat_anomalous"][cell] += acc_heat_anomalous
            bank["heat_secondary"][cell] += acc_heat_secondary
            bank["heat_terminal"][cell] += acc_heat_terminal
            if walk_secondaries:
                sec_flux[cell] += acc_sec_flux
                sec_power_eV[cell] += acc_sec_power_eV
            if walk_tail:
                anom_power_eV[cell] += acc_anom_power_eV
            if not arm_absorbed:
                surviving_weight += weight
                surviving_energy_weight += weight * E_arm
        # --- Re-mix -----------------------------------------------------
        # Flux and power both carry across: the surviving total is the entering
        # total times the surviving AREA share, and the common energy is the
        # area-weighted mean over the surviving arms. Weighting by area rather
        # than by flux is the same weighting -- each arm's flux is its area
        # share times the entering total -- and it stays well defined on a
        # zero-flux ray, which is a legitimate energy-only trajectory (the
        # flux-independent probe). An arm that stopped inside this cell
        # contributes to neither sum and has already banked its residual as
        # terminal heating there.
        if surviving_weight > 0.0:
            gamma_total = gamma_total * surviving_weight
            E = surviving_energy_weight / surviving_weight
        else:
            gamma_total = 0.0
            E = 0.0
            absorbed = True
            break

    if walk_products or walk_tail:
        # --- THE ONE POST-MARCH WALK STAGE, ON THE MEAN STATE -------------
        # The same closed-form machinery deposit_beam's walks use, on the same
        # closure exponent and the same thermalization floor, fed by the shared
        # per-cell birth banks above. What is coverage-specific is the MEDIUM:
        # ``coeff`` is the caller's mean-state coefficient (required, validated
        # above), never the channel view -- see this function's docstring for
        # why the concentration cancels in a decorrelated walker's path.
        # ``floor_eV`` needs Te alone, which the two media share.
        q = 1.0 - _COULOMB_STOPPING_EXPONENT[coulomb_model]
        coeff = stopping_coefficient
        floor_eV = np.maximum(
            _PRODUCT_FLOOR_TE_MULTIPLE * Te, _PRODUCT_FLOOR_MIN_EV
        )
        chan = banks[0]

        def _walk_and_deposit(W0, flux, walk_direction, split, cull=None):
            """Walk one population; deposit it and RETURN what escaped.

            ``(escaping power [erg/s], escaping flux [1/s], cull tally)``. The
            deposit lands in the CHANNEL arm's banks: the walked energy is
            the mean field's, not either medium's, and the caller consumes the
            two arms' sum (see the docstring's booking note). ``cull`` is
            passed through to :func:`_walk_products`; the WP-D product walks
            never arm it.
            """
            dep_eV, exit_eV, exit_flux, tally = _walk_products(
                W0, flux, walk_direction, coeff, dz_cm, floor_eV, q, cull=cull
            )
            dep_erg = dep_eV * _ERG_PER_EV
            chan["heating"] += dep_erg
            chan[split] += dep_erg
            return exit_eV * _ERG_PER_EV, exit_flux, tally

        def _bank_walk(W0, flux, walk_direction, split):
            """Walk one product population and book its deposit and escape.

            Returns the escaping FLUX, read only by the terminal population's
            caller (the charge that lands on the end surface).
            """
            nonlocal end_loss_low, end_loss_high
            exit_erg, exit_flux, _tally = _walk_and_deposit(
                W0, flux, walk_direction, split
            )
            if walk_direction > 0:
                end_loss_high += exit_erg
            else:
                end_loss_low += exit_erg
            return exit_flux

        if walk_secondaries and np.any(sec_flux > 0.0):
            # Flux-weighted mean secondary energy per BIRTH cell, over both
            # arms' contributions to that cell, emitted 50/50 along +-z.
            W_sec_cell = np.zeros(cells)
            born = sec_flux > 0.0
            W_sec_cell[born] = sec_power_eV[born] / sec_flux[born]
            half = 0.5 * sec_flux
            for walk_direction in (1, -1):
                _bank_walk(W_sec_cell, half, walk_direction, "heat_secondary")
        if walk_terminal and np.any(term_flux > 0.0):
            # The terminal residual keeps the PRIMARY's direction. Unlike the
            # single-medium march there can be several terminal cells: each arm
            # runs out of energy where its own medium stops it, and an arm that
            # stops leaves the survivor marching on. Carried per cell at the
            # flux-weighted mean residual, the same convention as above. The
            # escaping flux sums those populations, which is the charge the
            # end surface collects.
            term_W = np.zeros(cells)
            stopped = term_flux > 0.0
            term_W[stopped] = term_power_eV[stopped] / term_flux[stopped]
            terminal_escape_flux = _bank_walk(
                term_W, term_flux, direction, "heat_terminal"
            )
        if walk_tail and np.any(anom_power_eV > 0.0):
            # WP-E: re-express each cell's withheld anomalous POWER as a flux of
            # tail electrons at the single plateau energy E_tail
            # (flux*E_tail returns the power to roundoff), split 50/50 along
            # +-B, and walk them. The deposition profile becomes the anomalous
            # diagnostic split: heating_anomalous reports where the QL energy
            # LANDS. This self-limits on the mean density -- machine-length at
            # pedestal densities, collapsing onto local banking once the mean
            # has built -- which is the emergent transition the closure claims.
            half_flux = 0.5 * (anom_power_eV / E_tail)
            tail_power = float(anom_power_eV.sum()) * _ERG_PER_EV
            if tail_sub_threshold:
                tail_sub_threshold_power = tail_power
            elif tail_above_bar:
                tail_above_bar_power = tail_power
            if ionize_tail or reflect_face is not None:
                # The windowed closure's standing requirement: the window must
                # contain every cell the QL channel drives, or that cell's tail
                # power would be dropped on the floor.
                for birth in np.flatnonzero(half_flux > 0.0):
                    if not tail_lo <= birth <= tail_hi:
                        raise ValueError(
                            f"anomalous power in cell {int(birth)} lies "
                            f"outside tail_walk_window {(tail_lo, tail_hi)}; "
                            "the window must contain every cell the QL "
                            "channel drives, or that cell's tail power would "
                            "be silently dropped"
                        )
            if ionize_tail:
                # K6 under coverage: the walkers attenuate INELASTICALLY on the
                # mean neutral column as well as Coulomb-slowing, so the
                # closed-form integral cannot carry them. March them on this
                # module's own CSDA integration over the WINDOWED MEAN medium
                # -- the same instrument the primary uses, so cross sections,
                # thresholds, <W_sec> and substep control are the primary's by
                # construction rather than by transcription.
                #
                # BURN ATTRIBUTION: every per-cell bank the marched walker
                # produces is split between the arms by the local coverage --
                # f_cov to the channel, (1 - f_cov) to the reservoir. That is
                # the decorrelation partition applied to the walker's path, and
                # it is the whole implementation: the caller already reads the
                # reservoir arm's ionization rows as the reservoir debit and
                # the arms' sum as the total, so the deficit equation picks the
                # split up with no machinery of its own. The energy rows are
                # split by the same weights, which leaves their SUM -- the only
                # thing read -- exactly where it shipped.
                win = slice(tail_lo, tail_hi + 1)
                nn_w = nn_mean[win]
                ne_w = ne_mean[win]
                Te_w = Te[win]
                dz_w = dz_cm[win]
                f_w = f_cov[win]

                def _bank_tail_march(res):
                    """Book one marched tail population, PARTITIONED by f_cov."""
                    for arm, weight in ((0, f_w), (1, 1.0 - f_w)):
                        bank = banks[arm]
                        bank["ionization_events"][win] += (
                            weight * res.ionization_events
                        )
                        bank["excitation_events"][win] += (
                            weight * res.excitation_events
                        )
                        bank["ionization_cost"][win] += (
                            weight * res.ionization_cost_erg_s
                        )
                        bank["radiated"][win] += weight * res.radiated_erg_s
                    # All of the walker's HEAT is the anomalous channel's
                    # delivery to the electrons, so it lands in the lumped bank
                    # and in the anomalous split -- never in the primary's
                    # coulomb/secondary/terminal splits, which keep describing
                    # the primary alone. Booked whole to the channel slot, like
                    # every other walked deposit above.
                    chan["heating"][win] += res.plasma_heating_erg_s
                    chan["heat_anomalous"][win] += res.plasma_heating_erg_s
                    # Diagnostic tail splits of the four shared banks, on the
                    # same partition as the banks they split.
                    for arm, weight in ((0, f_w), (1, 1.0 - f_w)):
                        bank = banks[arm]
                        bank["ion_events_tail"][win] += (
                            weight * res.ionization_events
                        )
                        bank["exc_events_tail"][win] += (
                            weight * res.excitation_events
                        )
                        bank["ion_cost_tail"][win] += (
                            weight * res.ionization_cost_erg_s
                        )
                        bank["radiated_tail"][win] += (
                            weight * res.radiated_erg_s
                        )

                march_kwargs = dict(
                    I_ion_eV=I_ion_eV,
                    E_stop_eV=E_stop_eV,
                    coulomb_model=coulomb_model,
                    anomalous_model="none",
                    max_energy_fraction_per_substep=frac,
                )
                # A2a: the cull rides the nested march's OWN anode interception
                # -- same event, same convention, same arithmetic as the
                # primary's -- armed until it fires, so a walker that meets the
                # plane only after reflecting still meets it and one that met
                # it on its first leg is not culled twice. The rider's reversed
                # walkers are marched here as their own legs.
                _cull_kwargs = (
                    {} if not tail_cull
                    else dict(anode_cross_index=int(tail_anode_local),
                              anode_eta=float(tail_anode_eta))
                )
                _rider_launches = []
                for birth in np.flatnonzero(half_flux > 0.0):
                    for walk_direction in (1, -1):
                        leg_dir = walk_direction
                        armed = tail_cull
                        leg = deposit_beam(
                            E_tail,
                            float(half_flux[birth]),
                            nn_w,
                            ne_w,
                            Te_w,
                            int(birth) - tail_lo,
                            leg_dir,
                            dz_w,
                            **march_kwargs,
                            **(_cull_kwargs if armed else {}),
                        )
                        while True:
                            _bank_tail_march(leg)
                            if armed and float(leg.anode_intercepted_erg_s) > 0.0:
                                armed = False
                                _E_cross = float(leg.E_entry_eV[tail_anode_local])
                                _f_cull = (
                                    float(leg.anode_intercepted_erg_s)
                                    / (_E_cross * _ERG_PER_EV)
                                )
                                _g_f, _g_eV, _r_f, _r_eV = _tail_anode_take(
                                    np.array([_f_cull]), np.array([_E_cross]),
                                    R_e_tail, eta_E_tail,
                                )
                                tail_anode_culled_flux += _g_f
                                tail_anode_culled_erg += _g_eV * _ERG_PER_EV
                                tail_anode_returned_flux += _r_f
                                tail_anode_returned_erg += _r_eV * _ERG_PER_EV
                                if _r_f > 0.0:
                                    _rider_launches.append(
                                        (int(tail_anode_local), -leg_dir, _r_f,
                                         _r_eV / _r_f)
                                    )
                            leg_flux = float(leg.transmitted_flux)
                            leg_E = float(leg.transmitted_energy_eV)
                            if (
                                reflect_face is not None
                                and leg_dir == reflect_face
                                and leg_flux > 0.0
                                and leg_E < E_reflect
                            ):
                                leg_dir = -leg_dir
                                leg = deposit_beam(
                                    leg_E,
                                    leg_flux,
                                    nn_w,
                                    ne_w,
                                    Te_w,
                                    0 if reflect_face < 0 else tail_hi - tail_lo,
                                    leg_dir,
                                    dz_w,
                                    **march_kwargs,
                                    **(_cull_kwargs if armed else {}),
                                )
                                continue
                            exit_erg = leg_flux * leg_E * _ERG_PER_EV
                            if leg_dir > 0:
                                end_loss_tail_high += exit_erg
                            else:
                                end_loss_tail_low += exit_erg
                            break
                for _r_cell, _r_dir, _r_flux, _r_E in _rider_launches:
                    # The reversed walker walks like any other: same march,
                    # same banks, same reflecting-face convention. It never
                    # meets the plane again -- the cull is first-crossing only
                    # and this walker's crossing is the one that made it.
                    leg_dir = _r_dir
                    leg = deposit_beam(
                        _r_E, _r_flux, nn_w, ne_w, Te_w, _r_cell, leg_dir,
                        dz_w, **march_kwargs,
                    )
                    while True:
                        _bank_tail_march(leg)
                        leg_flux = float(leg.transmitted_flux)
                        leg_E = float(leg.transmitted_energy_eV)
                        if (
                            reflect_face is not None
                            and leg_dir == reflect_face
                            and leg_flux > 0.0
                            and leg_E < E_reflect
                        ):
                            leg_dir = -leg_dir
                            leg = deposit_beam(
                                leg_E, leg_flux, nn_w, ne_w, Te_w,
                                0 if reflect_face < 0 else tail_hi - tail_lo,
                                leg_dir, dz_w, **march_kwargs,
                            )
                            continue
                        exit_erg = leg_flux * leg_E * _ERG_PER_EV
                        if leg_dir > 0:
                            end_loss_tail_high += exit_erg
                        else:
                            end_loss_tail_low += exit_erg
                        break
            elif reflect_face is not None:
                # K7 with one reflecting window face. A reflection is expressed
                # by UNFOLDING the path: the reflected leg is the window
                # traversed back the other way, concatenated onto the incoming
                # leg, so the walk still telescopes exactly across the bounce.
                win = slice(tail_lo, tail_hi + 1)
                n_w = tail_hi - tail_lo + 1
                coeff_w = coeff[win]
                dz_w = dz_cm[win]
                floor_w = floor_eV[win]
                flux_w = half_flux[win]
                W0_w = np.full(n_w, E_tail)
                order_hit = (
                    np.arange(n_w)[::-1] if reflect_face < 0 else np.arange(n_w)
                )
                order_away = order_hit[::-1]
                escape_at_face = 0.0     # leaves through the reflecting face
                escape_opposite = 0.0    # leaves through the other one

                def _cull_at(*orders):
                    if not tail_cull:
                        return None
                    slots = []
                    for k, order in enumerate(orders):
                        hit = np.flatnonzero(order == tail_anode_local)
                        if hit.size:
                            slots.append(k * n_w + int(hit[0]))
                    return (np.array(sorted(slots), dtype=np.intp),
                            tail_anode_eta)

                def _leg(order, W0, flux, cull=None):
                    return _walk_products_forward(
                        W0[order], flux[order], coeff_w[order], dz_w[order],
                        floor_w[order], q, cull=cull,
                    )

                def _bank_tail_walk(dep_eV, *orders):
                    """Deposit one walked population, in ITS traversal order."""
                    dep_win = np.zeros(n_w)
                    for k, order in enumerate(orders):
                        dep_win[order] += dep_eV[k * n_w:(k + 1) * n_w]
                    dep_erg = dep_win * _ERG_PER_EV
                    chan["heating"][win] += dep_erg
                    chan["heat_anomalous"][win] += dep_erg

                def _bank_cull(tally):
                    nonlocal tail_anode_culled_flux, tail_anode_culled_erg
                    g_f, g_eV, r_f, _r_eV = _tail_anode_take(
                        tally[0], tally[1], R_e_tail, eta_E_tail
                    )
                    if r_f > 0.0:
                        raise ValueError(
                            "the reversed-walker rider fired on the "
                            "energy-only tail walk, which has no launch to "
                            "reverse; this combination is refused at "
                            "construction and should be unreachable"
                        )
                    tail_anode_culled_flux += g_f
                    tail_anode_culled_erg += g_eV * _ERG_PER_EV

                # The arm walking AWAY from the reflecting face never meets it.
                dep_a, exit_a, _, cull_a = _leg(
                    order_away, W0_w, flux_w, cull=_cull_at(order_away)
                )
                _bank_tail_walk(dep_a, order_away)
                escape_opposite += exit_a
                if tail_cull:
                    _bank_cull(cull_a)
                # The arm walking INTO it: populations born in different cells
                # arrive with different energies, so the threshold test is a
                # per-birth split, not a whole-arm switch.
                dep_h, exit_h, (act_h, W_face, stop_h), cull_h = _leg(
                    order_hit, W0_w, flux_w, cull=_cull_at(order_hit)
                )
                bounced = (~stop_h) & (W_face < E_reflect)
                if not bounced.any():
                    _bank_tail_walk(dep_h, order_hit)
                    escape_at_face += exit_h
                    if tail_cull:
                        _bank_cull(cull_h)
                else:
                    flux_hit = flux_w[order_hit]
                    flux_bounce = np.zeros(n_w)
                    flux_escape = np.zeros(n_w)
                    flux_bounce[act_h[bounced]] = flux_hit[act_h[bounced]]
                    flux_escape[act_h[~bounced]] = flux_hit[act_h[~bounced]]
                    if np.any(flux_escape > 0.0):
                        dep_e, exit_e, _, cull_e = _walk_products_forward(
                            W0_w[order_hit], flux_escape, coeff_w[order_hit],
                            dz_w[order_hit], floor_w[order_hit], q,
                            cull=_cull_at(order_hit),
                        )
                        _bank_tail_walk(dep_e, order_hit)
                        escape_at_face += exit_e
                        if tail_cull:
                            _bank_cull(cull_e)
                    dep_u, exit_u, _, cull_u = _walk_products_forward(
                        np.concatenate([W0_w[order_hit], np.zeros(n_w)]),
                        np.concatenate([flux_bounce, np.zeros(n_w)]),
                        np.concatenate(
                            [coeff_w[order_hit], coeff_w[order_away]]
                        ),
                        np.concatenate([dz_w[order_hit], dz_w[order_away]]),
                        np.concatenate(
                            [floor_w[order_hit], floor_w[order_away]]
                        ),
                        q,
                        cull=_cull_at(order_hit, order_away),
                    )
                    _bank_tail_walk(dep_u, order_hit, order_away)
                    escape_opposite += exit_u
                    if tail_cull:
                        _bank_cull(cull_u)
                if reflect_face > 0:
                    end_loss_tail_high += escape_at_face * _ERG_PER_EV
                    end_loss_tail_low += escape_opposite * _ERG_PER_EV
                else:
                    end_loss_tail_low += escape_at_face * _ERG_PER_EV
                    end_loss_tail_high += escape_opposite * _ERG_PER_EV
            else:
                tail_W = np.full(cells, E_tail)
                # Stated in FULL-GRID cell indices: this arm walks the whole
                # grid, not the window.
                _cull_plain = (
                    None if not tail_cull
                    else (np.array([tail_anode_local + tail_lo], dtype=np.intp),
                          tail_anode_eta)
                )
                for walk_direction in (1, -1):
                    exit_erg, _exit_flux, _tally = _walk_and_deposit(
                        tail_W, half_flux, walk_direction, "heat_anomalous",
                        cull=_cull_plain,
                    )
                    if walk_direction > 0:
                        end_loss_tail_high += exit_erg
                    else:
                        end_loss_tail_low += exit_erg
                    if tail_cull:
                        g_f, g_eV, r_f, _r_eV = _tail_anode_take(
                            _tally[0], _tally[1], R_e_tail, eta_E_tail
                        )
                        if r_f > 0.0:
                            raise ValueError(
                                "the reversed-walker rider fired on the "
                                "energy-only tail walk, which has no launch "
                                "to reverse; this combination is refused at "
                                "construction and should be unreachable"
                            )
                        tail_anode_culled_flux += g_f
                        tail_anode_culled_erg += g_eV * _ERG_PER_EV
        if book_transmitted and not absorbed and gamma_total > 0.0 and E > 0.0:
            # The transmitted primary: computed by the march, never banked. It
            # leaves through the end the ray was heading for.
            end_loss_transmitted = gamma_total * E * _ERG_PER_EV
            if direction > 0:
                end_loss_high += end_loss_transmitted
            else:
                end_loss_low += end_loss_transmitted

    results = []
    for arm, bank in enumerate(banks):
        # The transmitted primary is the mixed stream leaving the far end; it
        # is booked to the CHANNEL slot rather than split, because after the
        # last re-mix there is one population and no medium owns it. The
        # reservoir slot's transmitted flux is 0 by that convention, which is
        # why the caller must read the SUM (or ``flux_entry``) when it wants
        # the ray's survival.
        transmitted = 0.0 if (absorbed or arm == 1) else gamma_total
        # A2a: the tail cull ran ONCE on the mean state for both arms' births,
        # exactly like the end ledgers below, so its net lands whole on the
        # channel slot's anode row and the reservoir slot carries none of it.
        # A caller reading what the anode took must read the SUM, which is what
        # ``_sum_beam_deposition`` gives it.
        anode_arm = anode_intercepted[arm]
        if tail_cull and arm == 0:
            anode_arm = (
                anode_arm + tail_anode_culled_erg - tail_anode_returned_erg
            )
        results.append(
            BeamDepositionResult(
                ionization_events=bank["ionization_events"],
                excitation_events=bank["excitation_events"],
                plasma_heating_erg_s=bank["heating"],
                radiated_erg_s=bank["radiated"],
                ionization_cost_erg_s=bank["ionization_cost"],
                transmitted_flux=transmitted,
                transmitted_energy_eV=0.0 if transmitted <= 0.0 else E,
                anode_intercepted_erg_s=anode_arm,
                E_entry_eV=bank["E_entry"],
                heating_coulomb_erg_s=bank["heat_coulomb"],
                heating_anomalous_erg_s=bank["heat_anomalous"],
                heating_secondary_erg_s=bank["heat_secondary"],
                heating_terminal_erg_s=bank["heat_terminal"],
                # K6 diagnostic splits, PARTITIONED by f_cov like the shared
                # banks they split, so the two arms' tail rows sum to the
                # walkers' own totals exactly as their parents do.
                ionization_events_tail=bank["ion_events_tail"],
                excitation_events_tail=bank["exc_events_tail"],
                ionization_cost_tail_erg_s=bank["ion_cost_tail"],
                radiated_tail_erg_s=bank["radiated_tail"],
                # The end ledgers belong to the WALK, which ran once on the
                # mean state for both arms' births, so they are booked whole to
                # the channel slot on the same convention the transmitted
                # primary uses above. The reservoir slot carries 0.0, which is
                # why a caller reading escape must read the SUM.
                end_loss_low_erg_s=0.0 if arm == 1 else end_loss_low,
                end_loss_high_erg_s=0.0 if arm == 1 else end_loss_high,
                end_loss_transmitted_erg_s=(
                    0.0 if arm == 1 else end_loss_transmitted
                ),
                terminal_escape_flux_per_s=(
                    0.0 if arm == 1 else terminal_escape_flux
                ),
                end_loss_tail_low_erg_s=0.0 if arm == 1 else end_loss_tail_low,
                end_loss_tail_high_erg_s=(
                    0.0 if arm == 1 else end_loss_tail_high
                ),
                tail_power_erg_s=0.0 if arm == 1 else tail_power,
                tail_sub_threshold_power_erg_s=(
                    0.0 if arm == 1 else tail_sub_threshold_power
                ),
                tail_above_bar_power_erg_s=(
                    0.0 if arm == 1 else tail_above_bar_power
                ),
                tail_anode_culled_flux_per_s=(
                    0.0 if arm == 1 else tail_anode_culled_flux
                ),
                tail_anode_culled_erg_s=(
                    0.0 if arm == 1 else tail_anode_culled_erg
                ),
                tail_anode_returned_flux_per_s=(
                    0.0 if arm == 1 else tail_anode_returned_flux
                ),
                tail_anode_returned_erg_s=(
                    0.0 if arm == 1 else tail_anode_returned_erg
                ),
            )
        )
    return results[0], results[1], flux_entry
