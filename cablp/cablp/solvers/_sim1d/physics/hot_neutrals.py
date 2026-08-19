"""The decoupled HOT neutral channel: a CX-born minority and its ballistic fate.

The neutral gas the column carries is bimodal. A cold bulk sits near the vessel
temperature; a minority born by resonant charge exchange sits at the local ion
temperature. Gas-gas collisions cannot merge them -- at column densities the
neutral-neutral mean free path is orders of magnitude longer than the column
radius -- so the two populations are COLLISIONALLY DECOUPLED, and the hot
population's pressure must never appear as a force on the cold fluid. This
module carries the hot population as an ALGEBRAIC standing quantity (no new PDE
state, no packed row) plus a ballistic redistribution of the flows it carries.

The standing population follows the saturating two-state balance

    f_hot = x / (1 + x),   x = nu_cx * tau_hot,   nu_cx = n_i k_cx(T_eff)

whose ``tau_hot`` is a ballistic column-radius crossing. Everything in it is
already boxed: ``k_cx`` is the Phelps LXCat backscatter rate coefficient
(:func:`cablp.funcs._cross.phelps_cx_rate_cm3_s`, the same table whose sum with
half the isotropic-elastic rate IS the collision operator's ``nu_mt``), and
``tau_hot`` is the column radius over the hot thermal speed. No constant in this
module is fitted.

A hot atom's flight ends one of three ways, and the rates it competes against
set the branching:

    1/tau_hot = v_hot/Rp + nu_recx + nu_ion

    ballistic  the atom crosses the column boundary and joins the cold gas
               outside it (the annulus under ``neutral_two_zone``), leaving its
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

INTERNAL WALLS (``neutral_hot_internal_wall``, default off). The two global end
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

DIRECTED BIRTHS (``neutral_hot_birth_drift``, default off). A resonant charge
exchange hands the atom the ion's WHOLE velocity, drift included, so the birth
is at the local ``(Ti, u_i)`` rather than at ``Ti`` alone. The flag restores the
drift to the flight kinematics::

    v_z = v_hot * mu + u_i        v_perp = v_hot * sqrt(1 - mu^2)
    dz  = chord * (mu + m) / sqrt(1 - mu^2),       m = u_i / v_hot

``mu`` stays uniform on ``[-1, 1]`` -- the birth is still a volume birth, and
the drift shifts the velocity, not the direction measure. ``v_perp`` is
untouched because the drift is purely axial, so the radial crossing rate
``nu_ball = v_hot / Rp``, and with it every branching ratio and the standing
population, are the SAME numbers the isotropic launch computes; what moves is
only WHERE the flights get to. Two consequences are stated rather than hidden.
The kernel is no longer a pure function of the geometry -- the speed no longer
cancels from ``dz`` -- so it is a function of the STATE, rebuilt on every
evaluation instead of once per run, and ``BALLISTIC_DIRECTION_SAMPLES`` is a
per-evaluation cost under this flag rather than a free convergence knob. And
the drift enters per BIRTH CELL, so the landing, residence and end-plane
matrices become row-wise asymmetric: an atom born in a cell flowing towards the
far end is carried that way, which is the whole point of the flag.
"""

import numpy as np

from cablp.funcs._cross import phelps_cx_rate_cm3_s
from cablp.vars._cons import ev_to_erg

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
    # The two STREAMING readings, appended 2026-08-14 with the directed-birth
    # flag. Unlike every entry above them these are gated a second time, on
    # ``neutral_hot_birth_drift``: they are built from the drift kernel's
    # mu-weighted residence, which the isotropic path does not construct, and
    # they read ZERO on a run with the flag off. Zero here therefore means "not
    # computed", exactly as absence means "never recorded" for the rest.
    #
    # ``hot_n_flight`` is the residence-resolved in-flight hot density over the
    # cell [cm^-3] -- the nonlocal counterpart of ``nn_hot``, which counts a
    # cell's own births only. ``hot_flux_z`` is that population's directed
    # axial number flux [cm^-2 s^-1], signed with +z. Their RATIO is the hot
    # population's mean axial velocity over the cell, which is the quantity a
    # measured directed flow is compared against; it is left to the reader
    # rather than saved, so the density it was divided by is always in hand.
    "hot_n_flight",
    "hot_flux_z",
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
    index = np.arange(cells)
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
    rate ``nu_ball = v_hot / Rp`` and, under ``neutral_hot_birth_drift``, the
    denominator of the birth drift ratio. One definition, so the branching
    ratios and the flight kinematics cannot disagree about how fast the atom is.
    """
    return np.sqrt(2.0 * np.asarray(Ti_eV, dtype=float) * ev_to_erg / ion_mass_g)


def hot_birth_drift_ratio(state, floors, ion_mass_g):
    """Return the per-cell birth drift ratio ``m = u_i / v_hot`` [dimensionless].

    The ion drift in units of the hot atom's own launch speed -- the ONLY new
    number the directed-birth flag introduces, and it is local state rather than
    a constant: ``u_i`` is the ion fluid velocity the collision operator already
    runs on and ``v_hot`` is :func:`hot_thermal_speed`. ``|m| < 1`` is a
    drift-subsonic birth whose flights still reach both ways; ``|m| >= 1`` means
    every atom is carried downstream, which the kernel handles without a branch.
    """
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    v_hot = hot_thermal_speed(derived.Ti, ion_mass_g)
    return derived.u / np.maximum(v_hot, 1e-300)


def directed_flight_kernels(
    geometry,
    drift_ratio,
    samples=BALLISTIC_DIRECTION_SAMPLES,
    isotropic=None,
    internal_wall=False,
):
    """Return ``(landing, residence, end_fraction, residence_mu)`` with drift.

    The drift-asymmetric generalization of :func:`ballistic_flight_kernels`:
    identical launch measure (``mu`` uniform on ``[-1, 1]``, same sample grid),
    identical end-plane fold, identical row normalization

        landing[i].sum() == 1     residence[i].sum() == 1

    but the axial hop carries the birth cell's own ion drift,

        dz = chord * (mu + m_i) / sqrt(1 - mu**2)

    so each ROW is built at its own ``drift_ratio[i]``. Setting every
    ``m_i = 0`` recovers the isotropic kernel.

    ``residence_mu[i, j]`` is the same residence average weighted by the
    flight's own ``mu``. It carries no rows of the RHS; it exists so the
    standing population's mean axial velocity, ``v_hot * residence_mu + u_i *
    residence``, is readable per cell. Its rows sum to ``E[mu] == 0`` for ANY
    drift, which is the statement that the mu-correlated part of the flight
    velocity transports momentum without creating any.

    ``internal_wall`` confines each row to its own contiguous same-topology
    segment, exactly as it does for :func:`ballistic_flight_kernels` and from
    the same :func:`flight_wall_bounds` faces; it must be passed the value the
    ``isotropic`` triple was built with, or the verbatim-row copy below would
    mix a walled row into an unwalled kernel.

    ``isotropic``, when given, is the ``(landing, residence, end_fraction)``
    triple :func:`ballistic_flight_kernels` already built for this geometry.
    Rows whose ``m_i`` is exactly zero are copied from it VERBATIM rather than
    recomputed, so a cell with no ion drift gets the isotropic kernel bit for
    bit -- which is both the physical statement and what makes the drift arm's
    pre-plasma phase (``u_i == 0`` everywhere) bit-identical to the isotropic
    arm's. The recomputed rows agree with the isotropic ones to roundoff rather
    than exactly, because this builder accumulates each row by binning the
    flights it launched instead of by summing a per-flight overlap matrix; the
    reorganization is what makes a per-evaluation rebuild affordable at all,
    and the two orderings are compared directly by
    ``scripts/verify_hbd_momentum.py``.

    Raises ``ValueError`` for fewer than three direction samples, or for a
    ``drift_ratio`` that is not one finite value per cell.
    """
    z_edges = np.asarray(geometry.z_edges_cm, dtype=float)
    z_center = np.asarray(geometry.z_cm, dtype=float)
    chord = np.asarray(geometry.Rp_cm, dtype=float)
    cells = z_center.size
    count = int(samples)
    if count < 3:
        raise ValueError(
            "directed_flight_kernels needs at least 3 direction samples "
            f"(got {samples})"
        )
    m = np.asarray(drift_ratio, dtype=float)
    if m.shape != (cells,):
        raise ValueError(
            "directed_flight_kernels needs one drift ratio per cell "
            f"(got shape {m.shape} for {cells} cells)"
        )
    if not np.all(np.isfinite(m)):
        raise ValueError(
            "directed_flight_kernels got a non-finite drift ratio; u_i / v_hot "
            "is finite wherever Ti is floored above zero, so this is a "
            "corrupted state rather than a configuration error"
        )
    # The SAME mu grid the isotropic kernel integrates on: equal solid angle
    # per sample, none of them on the grazing limit.
    mu = -1.0 + (np.arange(count, dtype=float) + 0.5) * (2.0 / count)
    weight = 1.0 / count
    inv_perp = 1.0 / np.sqrt(1.0 - mu**2)
    wall_lo, wall_hi, cell_lo, cell_hi = flight_wall_bounds(
        geometry, internal_wall=internal_wall
    )

    live = chord > 0.0
    # A cell with no column has no CX birth and no flight; its row is the
    # in-place identity, exactly as in the isotropic kernel.
    chord_eff = np.where(live, chord, 0.0)[:, None]
    z0 = z_center[:, None]
    z_raw = z0 + chord_eff * (
        mu[None, :] * inv_perp[None, :] + m[:, None] * inv_perp[None, :]
    )
    end_fraction = (
        np.count_nonzero(z_raw < wall_lo[:, None], axis=1)
        + np.count_nonzero(z_raw > wall_hi[:, None], axis=1)
    ) * weight
    z1 = np.clip(z_raw, wall_lo[:, None], wall_hi[:, None])
    cell_of = np.clip(
        (np.searchsorted(z_edges, z1.ravel()) - 1).reshape(cells, count),
        cell_lo[:, None],
        cell_hi[:, None],
    )
    flat = (np.arange(cells)[:, None] * cells + cell_of).ravel()

    def binned(values):
        """Accumulate a per-flight quantity into its landing cell, per row."""
        return np.bincount(
            flat, weights=values.ravel(), minlength=cells * cells
        ).reshape(cells, cells)

    landing = (
        np.bincount(flat, minlength=cells * cells).reshape(cells, cells) * weight
    )

    # Residence by accumulation rather than by a per-flight overlap matrix.
    # Every flight starts at the SAME point z0 and deposits a uniform density
    # 1/|z1 - z0| along the segment it covers, so a cell's residence splits into
    # (a) the flights that cross it entirely, which contribute their full width
    # times the density of everything landing beyond it, and (b) the flights
    # that END inside it, which contribute the part of the cell they reached.
    displacement = z1 - z0
    stalled = displacement == 0.0
    density = weight / np.where(stalled, 1.0, np.abs(displacement))
    # The near side of the landing cell as seen from the birth point -- the
    # lower edge for a flight that went up, the upper edge for one that went
    # down, and z0 itself for a flight that never left its birth cell. Taking
    # the difference against it PER FLIGHT keeps the near-perpendicular flights,
    # whose density diverges as their span shrinks, out of any cancellation.
    anchor = np.where(
        cell_of == np.arange(cells)[:, None],
        z0,
        z_edges[cell_of + (cell_of < np.arange(cells)[:, None])],
    )
    # A flight with no axial span spends its whole life over its birth cell.
    partial = np.where(stalled, weight, density * np.abs(z1 - anchor))
    crossing = np.where(stalled, 0.0, density)

    # Widths of the part of each cell that lies above / below the birth point.
    above_lo = np.maximum(z0, z_edges[None, :-1])
    above_hi = np.maximum(above_lo, z_edges[None, 1:])
    below_hi = np.minimum(z0, z_edges[None, 1:])
    below_lo = np.minimum(below_hi, z_edges[None, :-1])

    def beyond(values):
        """Suffix sums (up-going flights) and prefix sums (down-going)."""
        up = np.zeros_like(values)
        up[:, :-1] = np.cumsum(values[:, ::-1], axis=1)[:, ::-1][:, 1:]
        down = np.zeros_like(values)
        down[:, 1:] = np.cumsum(values, axis=1)[:, :-1]
        return up, down

    # An up-going flight can only land at or above its birth cell and a
    # down-going one at or below it, so ONE binned array carries both: the
    # suffix sums above the birth cell see only up-going flights, the prefix
    # sums below it only down-going ones, and the widths vanish on the wrong
    # side of the birth point.
    up, down = beyond(binned(crossing))
    residence = (above_hi - above_lo) * up + (below_hi - below_lo) * down
    residence = residence + binned(partial)
    up_mu, down_mu = beyond(binned(crossing * mu[None, :]))
    residence_mu = (above_hi - above_lo) * up_mu + (below_hi - below_lo) * down_mu
    residence_mu = residence_mu + binned(partial * mu[None, :])

    rows = np.arange(cells)
    dead = ~live
    if np.any(dead):
        landing[dead, :] = 0.0
        residence[dead, :] = 0.0
        residence_mu[dead, :] = 0.0
        end_fraction[dead] = 0.0
        landing[rows[dead], rows[dead]] = 1.0
        residence[rows[dead], rows[dead]] = 1.0
    if isotropic is not None:
        # Exactly-zero drift IS the isotropic launch; take those rows verbatim
        # so the reduction is bit-for-bit rather than merely to roundoff.
        undrifted = (m == 0.0) & live
        if np.any(undrifted):
            iso_landing, iso_residence, iso_end = isotropic
            landing[undrifted, :] = np.asarray(iso_landing, dtype=float)[
                undrifted, :
            ]
            residence[undrifted, :] = np.asarray(iso_residence, dtype=float)[
                undrifted, :
            ]
            end_fraction[undrifted] = np.asarray(iso_end, dtype=float)[undrifted]
    return landing, residence, end_fraction, residence_mu


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
    wind_column_factor=None,
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
        if wind_column_factor is not None:
            u_n = wind_column_factor * u_n
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
    wind_column_factor=None,
    birth_drift=False,
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
    absorbs the cold wind's. This is unchanged by ``birth_drift``, and the
    reason is worth stating, because a directed birth looks like it should need
    a second momentum source and does NOT.

    THE PAIRING, AND WHY THE DIRECTED CASE ADDS NO BOOKING. Three terms touch
    the CX momentum and each owns exactly one leg:

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

    Under ``birth_drift`` the hot atom's axial velocity is
    ``v_hot * mu + u_i`` and ``mu`` stays uniform on ``[-1, 1]``, so its MEAN
    over the launch is ``u_i`` -- unchanged. ``p_hot`` is that mean, so every
    debit above is already the directed one and adding a drift momentum on top
    would book the same ``m u_i`` twice. What the flag changes is only the
    kernel the returns are spread over: the same momentum, delivered where the
    drift actually carried it.

    WHAT IS STILL CUT. Booking ``p_hot`` at the landing cell delivers the
    launch MEAN there, not each flight's own ``v_hot * mu`` excess. The
    mu-correlated half of the per-flight momentum is therefore transported
    without being resolved per landing cell. That cut is pre-existing, it is
    exactly as large under drift as without it (``mu`` is uniform either way,
    so the unresolved part integrates to zero over each birth cell's flights,
    which is why total momentum still closes), and ``residence_mu`` -- built by
    :func:`directed_flight_kernels` for the streaming diagnostic -- is the
    object that would resolve it. Resolving it is a separate closure decision
    and is NOT taken here.

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
    if birth_drift:
        # State-dependent: the drift ratio is read from the same state the
        # rates below are built from, and the kernel is rebuilt for it.
        landing, residence, end_fraction, residence_mu = directed_flight_kernels(
            geometry,
            drift_ratio=hot_birth_drift_ratio(
                state, floors=floors, ion_mass_g=ion_mass_g
            ),
            isotropic=kernels,
            # The same wall the cached isotropic triple was built with: the
            # zero-drift rows are copied from it verbatim.
            internal_wall=internal_wall,
        )
    else:
        landing, residence, end_fraction = kernels
        residence_mu = None
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
        wind_column_factor=wind_column_factor,
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
    # --- the streaming reading, gated on the directed kernel ---------------
    # Atoms born at cell i live tau_hot[i] and spend residence[i, j] of it over
    # cell j, so births*Vp*tau_hot is the in-flight count and the residence
    # kernels carry it onto the axis. The velocity each flight holds is
    # v_hot*mu + u_i, constant along a ballistic flight, so the flux uses the
    # mu-weighted kernel for the thermal half and the plain one for the drift.
    if residence_mu is None:
        n_flight = zeros.copy()
        flux_z = zeros.copy()
    else:
        in_flight = rates["births"] * Vp * rates["tau_hot"]
        v_hot = hot_thermal_speed(rates["Ti"], ion_mass_g)
        n_flight = (in_flight @ residence) / Vp
        flux_z = (
            (in_flight * v_hot) @ residence_mu + (in_flight * rates["u_i"]) @ residence
        ) / Vp
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
        # The streaming pair. Zero unless ``birth_drift`` armed the mu-weighted
        # kernel; their ratio is the hot population's mean axial velocity.
        "hot_n_flight": n_flight,
        "hot_flux_z": flux_z,
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
