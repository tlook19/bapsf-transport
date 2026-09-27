"""The decoupled HOT neutral channel: a CX-born minority and its ballistic fate.

The neutral gas the column carries is bimodal. A cold bulk sits near the vessel
temperature; a minority born by resonant charge exchange sits at the local ion
temperature. Gas-gas collisions cannot merge them -- in the IONIZATION-DEPLETED
discharge column (nn <~ 1e12 cm^-3, the state this channel was built for) the
neutral-neutral mean free path is orders of magnitude longer than the column
radius -- so the two populations are COLLISIONALLY DECOUPLED, and the hot
population's pressure must never appear as a force on the cold fluid.

REGIME LIMIT (disclosed, not a rebuild). That separation is a property of the
DEPLETED column, not of the machine. At FILL density it is false: nn ~ 2e13
cm^-3 gives a He-He momentum-transfer mfp of ~24.5 cm against Rp = 15 cm, so
mfp/Rp ~ 1.6 -- the same order, not orders of magnitude. Wherever the column
density stays near fill (before breakdown, and in any cell the discharge never
depletes), the hot channel therefore carries an UN-MODELED elastic leak of
O(Rp/mfp) per crossing, transferring hot momentum and energy into the cold
field. The sign is hot-forces-cold: the true cold fluid would be pushed and
heated slightly more than this module's decoupled treatment allows. This is a
stated limitation of the closure's validity range, not a defect in what it
computes inside that range.

This module carries the hot population as an ALGEBRAIC standing quantity (no
new PDE state, no packed row) plus a ballistic redistribution of the flows it
carries.

The standing population follows the saturating two-state balance

    f_hot = x / (1 + x),   x = nu_cx * tau_hot,   nu_cx = n_i k_cx(T_eff)

whose ``tau_hot`` is a ballistic column-radius crossing. Everything in it is
already boxed: ``k_cx`` is the Phelps LXCat backscatter rate coefficient
(:func:`cablp.atomic.cross_sections.phelps_cx_rate_cm3_s`, the same table whose sum with
half the isotropic-elastic rate IS the collision operator's ``nu_mt``), and
``tau_hot`` is the column radius over the hot thermal speed. No constant in this
module is fitted.

A hot atom's flight ends one of three ways, and the rates it competes against
set the branching:

    1/tau_hot = v_hot/Rp + nu_recx + nu_ion

    ballistic  the atom crosses the column boundary and joins the cold gas
               outside it (the annulus), leaving its
               excess energy on the wall
    re-CX      the atom exchanges charge again, handing its momentum and energy
               to the ION channel at the cell it reached and launching a
               REPLACEMENT hot atom there -- the nonlocal CX-recycling channel
    ionization the atom is ionized in flight, becoming a plasma source at the
               cell it reached

Only the first and third remove hot atoms. Re-CX replaces every atom it
consumes, so it cancels out of the standing population exactly,

    nn_hot = S_cx / (v_hot/Rp + nu_ion)

which is the fts-validated ``f_hot = x/(1+x)`` form when ionization is slow.
It does NOT cancel out of the flows: the replacement is born where the flight
reached, not where the original started, so the hot-atom birth rate the flows
run on is the fixed point

    Shat = S_cx + (Shat * b_recx) @ residence

solved exactly below. That fixed point is what makes the whole channel close to
machine precision: ``Shat`` is, by construction, the fresh CX births plus the
replacements, and every energy and momentum the ions hand a replacement is
booked at the replacement's own cell.

Where the flights land is a geometry-only kernel. This is the same kinematics
:meth:`..kinetic_neutrals.KN2ZoneJump._fly` integrates over its discrete
velocity grid -- an atom flies a RADIALLY determined chord and the axial hop is
whatever the axial velocity carries in that time,

    dz = v_z * chord / v_perp = chord * mu / sqrt(1 - mu^2)

-- evaluated here analytically over the launch's direction distribution instead.
Three things differ from that machinery and are stated rather than hidden. The
launch is an ISOTROPIC volume birth (``mu`` uniform on ``[-1, 1]``) rather than
a cosine-law wall launch (``annulus_chord_classes``' ``s`` uniform on
``[0, 1]``), because a CX birth happens in the gas and not on a surface. The
speed cancels out of ``dz`` entirely, which is what makes the kernel a pure
function of the geometry and cacheable once per run. And flights that would
leave through an end plane are folded back onto the end cells, exactly as
``kn2zone.build_hop_kernels`` folds its own -- there is a wall there, so the
atom lands on it rather than leaving the inventory.

INTERNAL WALLS (``neutral_hot_internal_wall``). The two global end
planes are not the only surfaces a flight can hit. The plasma domain is bounded
INSIDE the neutral domain: wherever a plasma-dead cell abuts a live one there is
a closed face -- the cathode disc against its plenum is the canonical one -- and
the plasma-terminating (absorbing) faces are a refinement of the same set. The
flag makes every such face a wall for the flight, on exactly the treatment the
end planes already get: the flight is clipped to the wall plane and the atom
lands in the cell on its OWN side of it. Off, a flight launched in a live cell
next to the cathode disc sails over the plenum and lands there, and the caller's
plasma-topology mask -- which the hot channel's rows are subject to -- then
deletes the deposit, so the atoms leave the inventory without a surface having
absorbed them.
"""

import numpy as np

from cablp.atomic.cross_sections import phelps_cx_rate_cm3_s
from cablp.constants import ev_to_erg

from ..core.state import (
    ConservativeState1D,
    derive_state,
    neutral_energy_floor,
)
from .sources import (
    neutral_energy_volume_ratio,
    neutral_temperature_eV,
    neutral_wind_velocity,
)


#: Direction samples used to integrate the isotropic launch. The kernel is a
#: one-off geometry product, so this is a convergence knob rather than a
#: per-step cost; it is odd so the sample midpoints straddle ``mu = 0``
#: symmetrically and never evaluate the ``mu -> +-1`` grazing limit exactly.
BALLISTIC_DIRECTION_SAMPLES = 4001

#: The PER-CELL hot-channel diagnostics a trajectory saves, in save order.
#: Each is a reading of :func:`neutral_hot_channel_rhs`'s own rates rather than
#: a row of its RHS, and each is present only on a run carrying ``En``; a saved
#: file written before they were persisted has none of them, which readers must
#: treat as "never recorded" rather than zero. The names are the result
#: attribute names and the HDF5 dataset names alike.
HOT_CHANNEL_DIAGNOSTIC_FIELDS = (
    "nn_hot",
    "f_hot",
    "tau_hot",
    "hot_S_cx",
    "hot_births",
    "hot_wall",
    "hot_recx",
    "hot_ionized",
    "hot_end_fraction",
    "hot_Ei_recx",
    "hot_Ei_ionization",
)


def flight_wall_bounds(geometry, internal_wall=False):
    """Return ``(z_lo, z_hi, cell_lo, cell_hi)``: where a flight from each cell ends.

    One entry per cell. ``z_lo[i]`` / ``z_hi[i]`` are the axial positions of the
    two walls a flight born in cell ``i`` can reach, and ``cell_lo[i]`` /
    ``cell_hi[i]`` are the first and last cells between them -- the range a
    landing index is clipped into, so an atom stopped ON a wall plane is booked
    on its own side of it rather than in the cell across.

    With ``internal_wall`` FALSE the only walls are the two global end planes,
    so every row is ``(z_edges[0], z_edges[-1], 0, cells - 1)`` and the caller's
    clips reduce to the historical ones exactly.

    With it TRUE the walls are the CLOSED plasma faces
    (``geometry.plasma_open`` false: the two end planes plus every face where a
    plasma-dead cell abuts a live one) together with the plasma-absorbing faces,
    which are a refinement of the same set. Each cell is therefore confined to
    its own contiguous run of same-topology cells: a live cell to its live
    segment, and a plasma-dead cell (plenum, obstruction) to the dead block it
    sits in. Neither can reach the other, which is the point -- a live cell's
    flights never deposit into a masked cell, and a dead cell's never deposit
    out of one.
    """
    z_edges = np.asarray(geometry.z_edges_cm, dtype=float)
    cells = int(np.asarray(geometry.z_cm, dtype=float).size)
    if not internal_wall:
        return (
            np.full(cells, z_edges[0]),
            np.full(cells, z_edges[-1]),
            np.zeros(cells, dtype=int),
            np.full(cells, cells - 1, dtype=int),
        )
    closed = ~np.asarray(geometry.plasma_open, dtype=bool)
    closed = closed | np.asarray(geometry.plasma_absorbing, dtype=bool)
    # The two end planes are closed by construction, so every cell has a wall
    # on both sides and the searches below cannot run off either end.
    closed[0] = True
    closed[-1] = True
    faces = np.flatnonzero(closed)
    index = np.arange(cells)
    lo_face = faces[np.searchsorted(faces, index, side="right") - 1]
    hi_face = faces[np.searchsorted(faces, index + 1, side="left")]
    return z_edges[lo_face], z_edges[hi_face], lo_face, hi_face - 1


def ballistic_flight_kernels(
    geometry, samples=BALLISTIC_DIRECTION_SAMPLES, internal_wall=False
):
    """Return ``(landing, residence, end_fraction)`` for isotropic column births.

    ``landing[i, j]`` is the fraction of atoms born isotropically in cell ``i``
    that reach the column boundary while over cell ``j``. Flights that would
    cross an end plane land on the end cell instead, so every row closes::

        landing[i].sum() == 1     residence[i].sum() == 1

    to machine precision -- the solid-angle normalization identity. The
    ``end_fraction[i]`` that was folded back is returned alongside so the
    approximation is measurable rather than implicit.

    ``internal_wall`` adds the closed and absorbing plasma faces to the pair of
    end planes (see :func:`flight_wall_bounds`), so every row is confined to its
    own contiguous same-topology segment and ``end_fraction`` counts the folds
    at those walls too. False reproduces the two-end-plane kernel exactly.

    ``residence[i, j]`` is the fraction of a flight's in-domain path length
    spent over cell ``j``. Path length is proportional to time along a straight
    flight, so it is the weight with which an along-flight process (re-CX,
    ionization) samples the axis.

    Both are functions of the grid and the column radius alone -- the flight
    speed cancels from ``dz = chord * mu / sqrt(1 - mu^2)`` -- so this is built
    once per run and never re-entered.
    """
    z_edges = np.asarray(geometry.z_edges_cm, dtype=float)
    z_center = np.asarray(geometry.z_cm, dtype=float)
    chord = np.asarray(geometry.Rp_cm, dtype=float)
    cells = z_center.size
    count = int(samples)
    if count < 3:
        raise ValueError(
            "ballistic_flight_kernels needs at least 3 direction samples "
            f"(got {samples})"
        )
    # Midpoints of an even partition of the isotropic cosine mu in [-1, 1]:
    # equal solid angle per sample, and no sample sits on the grazing limit.
    mu = -1.0 + (np.arange(count, dtype=float) + 0.5) * (2.0 / count)
    ratio = mu / np.sqrt(1.0 - mu**2)
    weight = 1.0 / count
    wall_lo, wall_hi, cell_lo, cell_hi = flight_wall_bounds(
        geometry, internal_wall=internal_wall
    )

    landing = np.zeros((cells, cells), dtype=float)
    residence = np.zeros((cells, cells), dtype=float)
    end_fraction = np.zeros(cells, dtype=float)
    for i in range(cells):
        if chord[i] <= 0.0:
            # No column here, so no CX birth and no flight: a degenerate row
            # would divide by zero downstream. Keep the atom in place.
            landing[i, i] = 1.0
            residence[i, i] = 1.0
            continue
        z0 = z_center[i]
        z_raw = z0 + chord[i] * ratio
        outside = (z_raw < wall_lo[i]) | (z_raw > wall_hi[i])
        z1 = np.clip(z_raw, wall_lo[i], wall_hi[i])
        j = np.clip(np.searchsorted(z_edges, z1) - 1, cell_lo[i], cell_hi[i])
        np.add.at(landing[i], j, weight)
        end_fraction[i] = float(np.count_nonzero(outside)) * weight
        # Residence: uniform along the clipped path, normalized per flight so
        # each flight contributes one unit of time however long it is. The
        # lifetime is a single tau_hot per birth cell, so weighting flights by
        # their own duration would be a second, unstated lifetime model.
        lo = np.minimum(z0, z1)
        hi = np.maximum(z0, z1)
        overlap = np.clip(
            np.minimum(hi[:, None], z_edges[None, 1:])
            - np.maximum(lo[:, None], z_edges[None, :-1]),
            0.0,
            None,
        )
        span = overlap.sum(axis=1)
        # A perpendicular flight (mu ~ 0) has no axial span and spends its
        # whole life over its birth cell.
        degenerate = span <= 0.0
        if np.any(degenerate):
            overlap[degenerate, :] = 0.0
            overlap[degenerate, i] = 1.0
            span = np.where(degenerate, 1.0, span)
        residence[i] = weight * (overlap / span[:, None]).sum(axis=0)
    return landing, residence, end_fraction


def hot_thermal_speed(Ti_eV, ion_mass_g):
    """Return the hot atom's launch speed ``sqrt(2 k Ti / m)`` [cm/s].

    The single speed the whole channel runs on: it sets the ballistic crossing
    rate ``nu_ball = v_hot / Rp``.
    """
    return np.sqrt(2.0 * np.asarray(Ti_eV, dtype=float) * ev_to_erg / ion_mass_g)


def hot_channel_rates(
    state,
    floors,
    ion_mass_g,
    geometry,
    gas_type,
    Tn_eV,
    ionization_rate_per_neutral,
    residence,
    b_ion_neutral_drag=1.0,
):
    """Return the algebraic hot-channel rates, population, and segment flows.

    Every entry is per cell, and every rate density is on the PLASMA volume --
    the volume ``n`` and the collision operator already live on. Callers convert
    with :func:`~.sources.neutral_energy_volume_ratio` wherever a row lands on
    the cold gas instead.

    Keys:

    ``S_cx``      cold->hot transfer rate [cm^-3 s^-1], the CX event density
                  ``b n nn k_cx(T_eff)``; the SAME events whose momentum share
                  sits inside the collision operator's ``nu_mt``
    ``nu_ball``   ballistic column-boundary crossing rate ``v_hot / Rp`` [s^-1]
    ``nu_recx``   re-CX rate along the flight ``n k_cx(Ti)`` [s^-1]
    ``nu_ion``    in-flight ionization rate [s^-1] -- the caller's own
                  per-neutral ionization frequency, so the bulk and in-flight
                  channels cannot disagree about the rate
    ``tau_hot``   harmonic lifetime ``1/(nu_ball + nu_recx + nu_ion)`` [s]
    ``nn_hot``    standing hot density [cm^-3]; re-CX cancels from it exactly
    ``f_hot``     hot number fraction ``nn_hot / (nn_cold + nn_hot)``
    ``births``    the fixed-point segment birth rate ``Shat`` [cm^-3 s^-1]:
                  fresh CX births plus re-CX replacements booked where the
                  replacement is actually born
    ``wall`` / ``recx`` / ``ionized``  segment rates ending in each channel
    ``e_hot`` / ``p_hot``  energy [erg] and directed momentum [g cm s^-1] a hot
                  atom carries out of its birth cell. ``e_hot`` is
                  ``(3/2) k Ti + (1/2) m u_rel^2``: the ion's thermal energy
                  PLUS the slip kinetic energy, which is the CX half of the
                  frictional dissipation the collision operator books. Dropping
                  that half would leave the ion/cold/hot energy sum short by
                  exactly ``q_fric_cx``, so it is not a refinement -- it is what
                  closes the three-way budget.
    """
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    Tn = neutral_temperature_eV(state, floors=floors, Tn_eV=Tn_eV)
    nn = np.maximum(np.asarray(state.nn, dtype=float), floors["nn"])
    n = np.asarray(state.n, dtype=float)
    T_eff = 0.5 * (derived.Ti + Tn)
    S_cx = (
        abs(float(b_ion_neutral_drag))
        * n
        * nn
        * phelps_cx_rate_cm3_s(T_eff, gas_type=gas_type)
    )
    if state.M_n is None:
        u_n = np.zeros_like(derived.u)
    else:
        u_n = neutral_wind_velocity(
            state, floors=floors, ion_mass_g=ion_mass_g, geometry=geometry
        )
    u_rel = derived.u - u_n
    v_hot = hot_thermal_speed(derived.Ti, ion_mass_g)
    Rp = np.asarray(geometry.Rp_cm, dtype=float)
    nu_ball = np.where(Rp > 0.0, v_hot / np.maximum(Rp, 1e-300), 0.0)
    # The hot atom meets ions at the ion temperature on both sides of the
    # collision, so its own T_eff is Ti rather than the cold-gas mixture.
    nu_recx = n * phelps_cx_rate_cm3_s(derived.Ti, gas_type=gas_type)
    nu_ion = np.maximum(np.asarray(ionization_rate_per_neutral, dtype=float), 0.0)
    total = nu_ball + nu_recx + nu_ion
    live = total > 0.0
    tau_hot = np.where(live, 1.0 / np.maximum(total, 1e-300), 0.0)
    # Removal channels only: re-CX replaces the atom it consumes.
    nn_hot = np.where(live, S_cx / np.maximum(nu_ball + nu_ion, 1e-300), 0.0)

    b_wall = nu_ball * tau_hot
    b_recx = nu_recx * tau_hot
    b_ion = nu_ion * tau_hot
    Vp = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    # Shat_ext = S_ext + (Shat_ext * b_recx) @ residence, solved exactly. The
    # spectral radius of diag(b_recx) @ residence is at most max(b_recx) < 1
    # wherever a ballistic escape exists, so the system is well conditioned.
    matrix = np.eye(Vp.size) - b_recx[:, None] * residence
    births_ext = np.linalg.solve(matrix.T, S_cx * Vp)
    births = births_ext / Vp
    return {
        "S_cx": S_cx,
        "nu_ball": nu_ball,
        "nu_recx": nu_recx,
        "nu_ion": nu_ion,
        "tau_hot": tau_hot,
        "nn_hot": nn_hot,
        "f_hot": nn_hot / np.maximum(nn + nn_hot, 1e-300),
        "births": births,
        "wall": births * b_wall,
        "recx": births * b_recx,
        "ionized": births * b_ion,
        "e_hot": 1.5 * derived.Ti * ev_to_erg + 0.5 * ion_mass_g * u_rel**2,
        "p_hot": ion_mass_g * derived.u,
        "Ti": derived.Ti,
        "u_i": derived.u,
        "Tn": Tn,
    }


def neutral_hot_channel_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    gas_type,
    Tn_eV,
    ionization_rate_per_neutral,
    kernels,
    I_ion,
    b_ion_neutral_drag=1.0,
    internal_wall=False,
):
    """Return ``(rhs, diagnostics)`` for the hot channel's ballistic flows.

    The cold gas has ALREADY paid for the births: the CX decoupling correction
    (:func:`~.sources.neutral_cx_channel_rhs`) removes the transferred atoms
    from ``nn`` at their own per-particle energy and momentum and withdraws the
    CX share of the collision operator's cold heating. This term spends exactly
    what that transfer handed over, so the pair closes.

    Rows produced, all presence-gated on an ``En``-carrying state:

    ``nn`` / ``nn_a``   atoms whose flight ended on the column boundary,
                        deposited at the LANDING cell -- the CX-ballistic
                        erosion that relieves an axial pile. Under
                        ``internal_wall`` the landing cell is always on the
                        birth cell's own side of every closed plasma face, so
                        a live cell's deposit can never reach a plasma-dead
                        one and the caller's topology mask has nothing to
                        delete.
    ``En``              the landed atoms rejoin the cold gas fully accommodated
                        at the wall temperature (single-zone); the excess is a
                        disclosed wall loss
    ``n``, ``Ee``, ``Ei``, ``M``  the in-flight ionization and re-CX channels,
                        deposited at the RESIDENCE-weighted cells

    ENERGY LEFT ON THE WALL. An atom reaching the column boundary is thermally
    accommodated there. Under the ratified annulus-cold v1 cut the gas outside
    the column carries no energy field, so the whole excess
    ``e_hot - (3/2) k T_wall`` leaves the model. Only the ``alpha_E`` share of
    that is accommodation in the physical sense; the remainder is the v1 cut.
    Both are reported by ``diagnostics`` rather than silently dropped.

    MOMENTUM LEFT ON THE WALL. Landed atoms arrive with no directed momentum:
    the same cut wall-thermalizes them. The momentum they carried out of the
    column is absorbed by the surface, exactly as the momentum wall sink
    absorbs the cold wind's.

    THE PAIRING. Three terms touch the CX momentum and each owns exactly one
    leg:

    1. :func:`~.sources.ion_neutral_collision_rhs` books the ION side of the
       full ``nu_mt`` -- CX and elastic together -- as ``-m n nu_mt u_rel`` with
       the exact mirror on ``M_n``. The CX share of that ion debit is already
       the right answer for a swap (the ion fluid loses ``m u_i`` and gains
       ``m u_n``), so nothing here or downstream touches the ion side of a fresh
       CX event again.
    2. :func:`~.sources.neutral_cx_channel_rhs` repairs only the COLD side,
       replacing the mirror's ``+m S_cx u_rel`` with the atom's own
       ``-m S_cx u_n``; the two collapse to ``-m S_cx u_i``. Ion and cold fluid
       together are therefore short by exactly ``m u_i S_cx`` per cell, and that
       deficit IS the hot channel's income.
    3. This term spends it. ``p_hot = m u_i`` is the momentum ONE hot atom
       carries out of its birth cell, and the birth rate it multiplies is the
       fixed point ``births == wall + recx + ionized``, so income
       ``births * p_hot`` leaves as wall absorption plus the two residence-
       weighted returns to the ions, with the ``recx`` return net of the
       replacement drawn at the cell it reached.

    WHAT IS STILL CUT. Booking ``p_hot`` at the landing cell delivers the
    launch MEAN there, not each flight's own ``v_hot * mu`` excess. The
    mu-correlated half of the per-flight momentum is therefore transported
    without being resolved per landing cell. ``mu`` is uniform on ``[-1, 1]``,
    so the unresolved part integrates to zero over each birth cell's flights,
    which is why total momentum still closes.

    ``diagnostics`` is a READING of the term, never a row of it: nothing in it
    enters the RHS sum, and every array in it is the pre-mask quantity the rows
    were built from. Per cell it carries the standing population
    (``nn_hot``, ``f_hot``, ``tau_hot``), the four birth-cell fates
    (``hot_S_cx``, ``hot_births``, ``hot_wall``, ``hot_recx``,
    ``hot_ionized``), the kernel's end-plane fold fraction
    (``hot_end_fraction``), and the two separated halves of the ion-energy
    return (``hot_Ei_recx``, ``hot_Ei_ionization``, which sum bitwise to the
    ``Ei`` row). The remaining ``hot_*_per_s`` / ``hot_*_erg_s`` entries are
    run-wide scalars.
    """
    zeros = np.zeros_like(np.asarray(state.nn, dtype=float))
    if state.En is None:
        return (
            ConservativeState1D(
                n=zeros,
                nn=zeros.copy(),
                M=zeros.copy(),
                Ee=zeros.copy(),
                Ei=zeros.copy(),
            ),
            {},
        )
    landing, residence, end_fraction = kernels
    rates = hot_channel_rates(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        gas_type=gas_type,
        Tn_eV=Tn_eV,
        ionization_rate_per_neutral=ionization_rate_per_neutral,
        residence=residence,
        b_ion_neutral_drag=b_ion_neutral_drag,
    )
    Vp = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    ratio = neutral_energy_volume_ratio(state, geometry)

    def spread(rate, kernel):
        """Redistribute an extensive plasma-volume rate over a kernel row set."""
        return ((np.asarray(rate, dtype=float) * Vp) @ kernel) / Vp

    # --- (i)/(iv) column-boundary interception, spread by the landing kernel.
    landed = spread(rates["wall"], landing)
    landed_energy = neutral_energy_floor(landed)
    wall_energy_carried = rates["wall"] * rates["e_hot"]
    # --- (ii) re-CX along the flight: the atom's momentum and energy go to the
    # ions at the cell it reached; the replacement it launches there is drawn
    # from that cell's own ion population. ``births`` already contains the
    # replacement, so the two halves close exactly.
    recx_here = spread(rates["recx"], residence)
    # The replacement is drawn at the cell the flight REACHED and is born with
    # that cell's whole e_hot -- thermal AND slip. Debiting only the thermal
    # half would leave the three-way energy identity short by exactly the
    # replacement's slip term, which is how this was caught.
    dEi_recx = (
        spread(rates["recx"] * rates["e_hot"], residence)
        - recx_here * rates["e_hot"]
    )
    dM_recx = (
        spread(rates["recx"] * rates["p_hot"], residence)
        - recx_here * ion_mass_g * rates["u_i"]
    )
    # --- (iii) in-flight ionization: a plasma source at the cell reached,
    # carrying the atom's own thermal energy and momentum and paying the same
    # binding-energy cost the bulk channel pays.
    ionized_here = spread(rates["ionized"], residence)
    dEi_ion = spread(rates["ionized"] * rates["e_hot"], residence)
    dM_ion = spread(rates["ionized"] * rates["p_hot"], residence)
    dEe_ion = -float(I_ion) * ev_to_erg * ionized_here

    two_zone = state.nn_a is not None
    rhs = ConservativeState1D(
        n=ionized_here,
        nn=zeros.copy() if two_zone else landed * ratio,
        M=dM_recx + dM_ion,
        Ee=dEe_ion,
        Ei=dEi_recx + dEi_ion,
        M_n=None if state.M_n is None else zeros.copy(),
        nn_a=(landed * Vp / _annulus_volume(geometry)) if two_zone else None,
        M_n_a=None if state.M_n_a is None else zeros.copy(),
        En=zeros.copy() if two_zone else landed_energy * ratio,
    )
    # The returned energy is extensive on the plasma volume: the row is
    # landed_energy * (Vp/V_En) on V_En, so its inventory is landed_energy*Vp.
    returned = 0.0 if two_zone else float(np.sum(landed_energy * Vp))
    diagnostics = {
        "nn_hot": rates["nn_hot"],
        "f_hot": rates["f_hot"],
        "tau_hot": rates["tau_hot"],
        # --- per-cell readings of the channel, not rows of it ----------------
        # These are the SAME arrays the rows above were built from, handed back
        # so a saved trajectory can be read without reconstructing them. They
        # are pre-mask, like every other entry here: the plasma-topology mask is
        # applied to the RETURNED ROWS by the caller, so a reader comparing a
        # diagnostic against a saved row must expect them to agree only where
        # the mask left the row alone.
        #
        # The four fates are booked at the BIRTH cell (``births`` is the
        # fixed-point rate ``Shat``, so ``wall + recx + ionized == births``
        # per cell). The ledger's own ``nn_a`` and ``n`` rows carry the same
        # wall and ionization flows at the cell the flight REACHED, so the two
        # views together give the axial displacement the channel performs.
        "hot_S_cx": rates["S_cx"],
        "hot_births": rates["births"],
        "hot_wall": rates["wall"],
        "hot_recx": rates["recx"],
        "hot_ionized": rates["ionized"],
        # The kernel's end-plane fold: the fraction of cell ``i``'s isotropic
        # launches whose free flight would have left through an end plane and
        # was folded back onto the end cell instead. Geometry only, constant in
        # time, and NOT a loss -- it is what keeps the landing rows closing to
        # 1 -- so it is saved to make the approximation measurable per cell
        # rather than only as the run-wide scalar below. Under
        # ``internal_wall`` it counts the folds at the closed and absorbing
        # plasma faces on the same footing, so on a run with that flag the
        # reading is "folded at a wall", not "folded at an end plane".
        "hot_end_fraction": np.asarray(end_fraction, dtype=float),
        # The two halves of the ion-energy return, separated. Their sum is
        # bit-identical to the ``Ei`` row above by construction -- the row is
        # literally ``dEi_recx + dEi_ion`` and these are its two addends -- so
        # separating them here costs no arithmetic and changes no row.
        # ``hot_Ei_recx`` is the nonlocal CX-recycling power: energy the ions
        # hand back at the cell a hot atom's flight reached, net of the
        # replacement drawn there. ``hot_Ei_ionization`` is the thermal energy
        # an in-flight ionization deposits with the new ion.
        "hot_Ei_recx": dEi_recx,
        "hot_Ei_ionization": dEi_ion,
        "hot_births_per_s": float(np.sum(rates["births"] * Vp)),
        "hot_wall_energy_erg_s": float(np.sum(wall_energy_carried * Vp)),
        "hot_wall_energy_returned_erg_s": returned,
        "hot_end_fold_fraction": float(
            np.sum(end_fraction * rates["wall"] * Vp)
            / max(float(np.sum(rates["wall"] * Vp)), 1e-300)
        ),
    }
    return rhs, diagnostics


def _annulus_volume(geometry):
    """Return the per-cell annulus volume [cm^3], floored away from zero."""
    Vp = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    Vm = np.asarray(geometry.neutral_volume_cm3, dtype=float)
    return np.maximum(Vm - Vp, 1e-300)
