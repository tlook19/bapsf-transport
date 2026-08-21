"""The DIRECTED hot surface carrier: the cathode backscatter stream's fate.

The ``R_N`` share of the flux a cathode face recycles does not rejoin the cold
gas where it was born. It leaves the surface as a fast, DIRECTED beam of
neutral atoms at tens of eV per particle and flies down the column until
something stops it. This module carries that beam as an ALGEBRAIC
quasi-static attenuation profile -- no new PDE state, no packed row, no saved
trajectory field -- and books where the stream's particles, energy and
momentum actually land.

WHY NOT THE HOT CHANNEL (``hot_neutrals.py``). That module integrates an
ISOTROPIC VOLUME birth: ``mu`` uniform on ``[-1, 1]``, launched in the gas,
speed cancelling out of its flight kernel. The backscatter stream is a
DIRECTED SURFACE flux at a single speed, launched on a disc at one end of the
domain. Its premise is violated in every one of those respects, so its kernel
is not reused and nothing here touches it.

WHAT THE CARRIER IS. One quasi-static beam per cathode face, launched into the
plasma at

    v_fast = cathode_jet_backscatter_speed(...)      [the ONE spec]
    E_fast = (1/2) m v_fast^2

-- the same one spec the jet's ``M_n`` momentum booking and the solver's
``cathode_jet_neutral_energy`` term read, so the launch here can never
describe atoms moving at a different speed from the ones those terms describe.
The beam does not slow down: an atom either survives at ``v_fast`` or its
history ends, which is what makes the profile algebraic.

THREE ATTENUATION CHANNELS, per unit axial length:

``CX``          ``a_cx = n_i sigma_b(E_rel) g_eff / v_fast``. The Phelps
                He+/He backscatter cross section is tabulated against the
                RELATIVE collision energy of the pair, so its argument is
                ``E_rel = (1/2) mu g_eff^2`` with the equal-mass reduced mass
                ``mu = m/2``, and the RATE is ``n sigma <relative speed>``
                rather than ``n sigma v_lab``. ``g_eff`` is the standard
                equal-mass interpolation
                ``g_eff^2 = |v_fast - u_i|^2 + 16 k Ti/(pi m)``, transcribed
                from :meth:`..kinetic_dvm.TransientDVM.collision_frequencies`,
                the repo's existing correct consumer, including its ``Ti``
                clamp and its ``E_rel`` floor.
``ionization``  ``a_ion = nu_ion / v_fast`` with ``nu_ion`` the caller's OWN
                per-neutral ionization frequency ``n_e <sigma v>_SCD(Te)``.
                The rate stays AT ``Te`` and never sees the fast energy: the
                electrons are the Maxwellian species, and the beam's speed
                relative to a thermal electron shifts the electron-impact rate
                coefficient by well under a per cent. Only ``sigma_cx`` is
                evaluated at ``E_rel``.
``escape``      ``a_esc = 1 / lambda_esc``; see
                :func:`carrier_escape_length_cm`. A z-resolved GEOMETRIC
                channel with no fitted constant.

WHERE THE THREE END UP:

``CX``          resonant charge exchange is a swap. The fast atom becomes a
                fast ION carrying ``E_fast`` and ``m v_fast``; the ion it
                exchanged with becomes a neutral born at the LOCAL ION STATE,
                so the ions are debited ``(3/2) k Ti`` and ``m u_i`` per event
                and the cold gas is credited the same. ``n`` is unchanged --
                one ion in, one ion out. Booking either side alone would
                violate conservation while still passing a sum closure, which
                is why the pairwise audit checks this pair by name.
``ionization``  a plasma source: ``n += 1`` at the cell reached, the ion born
                with ``E_fast`` and ``m v_fast``, and the electrons paying the
                standard ADAS binding cost ``I_ion`` -- the same cost the bulk
                channel pays, from the same rate.
``escape``      the atom crosses the column boundary and joins the gas outside
                it (the annulus under ``neutral_two_zone``). Under the
                ratified annulus-cold v1 cut that gas carries no energy field,
                so the atom's whole energy leaves the model as a NAMED wall
                leak, reported rather than deleted. Its directed momentum is
                absorbed by the surface, exactly as the hot channel's landed
                atoms' is.

Two more fates end a history and each is its own named row. The anode MESH is
opaque to a fraction ``eta`` of the beam: at the beam's FIRST crossing of an
anode face that share is culled and re-emitted at the wall temperature on the
INCIDENT side. And whatever survives to the end of the plasma segment hits the
end wall: a named END leak, booked at the last cell it reached.

THE INCOME SIDE, AND THE ONE-OWNER RULE. Everything above is spent, and it is
spent exactly once. Three v1 bookings are WITHHELD at the source when this
carrier is armed, and the withdrawal is the carrier's whole income:

1. the ``R_N`` share of the cathode cell's ``nn`` rebirth (and with it the
   ``(3/2) k T_wall`` the generic surface booking would have credited it);
2. the ``R_N`` share of ``cathode_jet_neutral_energy``'s ``En`` excess -- the
   backscatter energy the v1 channel dumped into the one cathode-adjacent cold
   cell, which that term's own docstring names as the defect this build fixes;
3. the ``R_N v_back`` share of the boundary term's ``jet_M_n``, leaving the
   ``(1 - R_N) v_eff`` effusive share behind.

Leaving any of the three in place would plant that channel twice. The audit
script checks each withholding against the launch BY NAME rather than only
through a sum, because a compensated double-book closes a sum identity.
"""

import numpy as np

from cablp.funcs._cross import phelps_he_backscatter_cm2
from cablp.vars._cons import ev_to_erg

from ..core.state import (
    ConservativeState1D,
    derive_state,
    neutral_energy_floor,
)
from .sources import cathode_jet_backscatter_speed


#: The ``E_rel`` floor [eV] and the ``Ti`` clamp [eV] the CX cross section is
#: evaluated behind. Both are transcribed from
#: :meth:`..kinetic_dvm.TransientDVM.collision_frequencies` so the two
#: consumers of the same Phelps table cannot disagree about its argument.
CARRIER_E_REL_FLOOR_EV = 1e-9
CARRIER_TI_CLAMP_EV = 1e-6

#: erg/s per watt: the ledger rows below are quoted in watts, the rest of the
#: solver in CGS.
_ERG_S_PER_W = 1.0e7

#: The named ledger flows this term reports, in report order. Each is a
#: READING of the term, never a row of it: nothing here enters the RHS sum.
#: Reported as ``<row>_per_s`` [s^-1], ``<row>_W`` [W] and, where the flow
#: carries directed momentum, ``<row>_dyn`` [g cm s^-2].
CARRIER_LEDGER_ROWS = (
    "launch",
    "jet_ionization",
    "partner_exchange",
    "wall_leak",
    "end_leak",
    "mesh_cull",
    "v1_withdrawal",
)


def carrier_escape_length_cm(geometry):
    """Return the per-cell geometric escape length ``lambda_esc`` [cm].

    The mean AXIAL distance a cosine-launched fast atom covers before its first
    crossing of the plasma column surface. It is a pure geometry product with
    no fitted constant, and it is z-resolved through ``Rp(z)`` alone:

        lambda_esc(z) = <transverse chord to the column surface>
                        * <cot(theta)>_cosine
                      = (pi/2) Rp(z) * (2/pi)
                      = Rp(z)

    Both averages are standard. The mean chord of a convex 2D region from a
    uniformly distributed interior point in a uniformly distributed direction
    is ``pi A / P`` (Cauchy), which for a disc of radius ``Rp`` is
    ``pi Rp / 2`` -- the transverse path the atom must cover to get out. The
    axial distance it covers meanwhile is that chord times ``cot(theta)``, and
    for a Lambert (cosine) surface launch
    ``<cot(theta)> = int_0^{pi/2} cot(th) 2 cos(th) sin(th) dth = pi/2``,
    whose reciprocal ``2/pi`` cancels the chord's ``pi/2`` exactly.

    The result is the directed counterpart of the hot channel's ballistic
    crossing rate ``nu_ball = v_hot / Rp``: dividing that rate by an axial
    speed gives the same ``1 / Rp`` per unit length. The two channels are
    different populations and neither reads the other's kernel, but they agree
    on the geometry, which is the point.

    Cells with no column (``Rp <= 0``) are plasma-dead and are never marched
    over; they are returned as ``inf`` so an accidental read is a zero rate
    rather than a divide-by-zero.
    """
    Rp = np.asarray(geometry.Rp_cm, dtype=float)
    return np.where(Rp > 0.0, Rp, np.inf)


def carrier_attenuation_coefficients(
    state,
    floors,
    ion_mass_g,
    geometry,
    v_fast,
    direction,
    ionization_rate_per_neutral,
    derived=None,
):
    """Return ``(a_cx, a_ion, a_esc)`` per cell [cm^-1] for one beam.

    ``v_fast`` is the beam's (positive) launch speed [cm/s] and ``direction``
    its axial sign (``+1`` or ``-1``), so the relative velocity against the ion
    fluid is ``direction * v_fast - u_i``.

    Each entry is an attenuation coefficient per unit AXIAL length: a collision
    frequency divided by the axial speed the beam actually travels at. The beam
    is purely axial, so that speed is ``v_fast`` itself.

    ``derived`` is the caller's already-built :func:`derive_state` result; it
    is rebuilt here when omitted so the function stands alone for an audit.
    """
    if derived is None:
        derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    n_i = np.asarray(state.n, dtype=float)
    Ti = np.maximum(np.asarray(derived.Ti, dtype=float), CARRIER_TI_CLAMP_EV)
    w = float(direction) * float(v_fast) - np.asarray(derived.u, dtype=float)
    # The equal-mass interpolation between the drift-dominated and
    # thermal-dominated limits; mu = m/2, so 8 k T/(pi mu) = 16 k T/(pi m).
    g_eff = np.sqrt(w**2 + 16.0 * Ti * ev_to_erg / (np.pi * ion_mass_g))
    E_rel = np.maximum(
        0.25 * ion_mass_g * g_eff**2 / ev_to_erg, CARRIER_E_REL_FLOOR_EV
    )
    a_cx = n_i * phelps_he_backscatter_cm2(E_rel) * g_eff / float(v_fast)
    nu_ion = np.maximum(
        np.asarray(ionization_rate_per_neutral, dtype=float), 0.0
    )
    a_ion = nu_ion / float(v_fast)
    a_esc = 1.0 / carrier_escape_length_cm(geometry)
    return a_cx, a_ion, a_esc


def carrier_beam_segment(geometry, live, direction):
    """Return the ordered cell indices one beam marches over.

    The beam starts in the cathode face's live cell and runs until the plasma
    segment ends -- at the domain end, or at the first plasma-dead cell, which
    is a closed plasma face and therefore a wall. Confining the march this way
    is what guarantees no deposit ever lands on a cell the caller's
    plasma-topology mask would delete: a masked deposit is a silent particle
    loss, not a leak, and this term reports its leaks.
    """
    active = np.asarray(geometry.plasma_active, dtype=bool)
    cells = int(active.size)
    step = 1 if direction > 0 else -1
    idx = []
    cell = int(live)
    while 0 <= cell < cells and active[cell]:
        idx.append(cell)
        cell += step
    return np.asarray(idx, dtype=int)


def cathode_jet_carrier_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    cathode_jet,
    launch_per_s,
    ionization_rate_per_neutral,
    I_ion,
    eta=0.0,
):
    """Return ``(rhs, diagnostics)`` for the directed cathode backscatter beam.

    ``launch_per_s`` is the per-cell particle rate [s^-1] the boundary term
    WITHHELD from its cold rebirth for this carrier -- ``R_N`` times the
    cathode faces' own recycle flux, taken from the same evaluation, so the
    launch here and the withdrawal there are one number rather than two
    estimates of it.

    ``ionization_rate_per_neutral`` is the per-neutral ionization frequency
    the bulk reaction term is using on this same evaluation, threaded in so
    the beam and the bulk cannot disagree about the rate.

    ``eta`` is the anode mesh's solid fraction, applied as a ``(1 - eta)``
    first-crossing cull at each anode face (see the module docstring).

    ``diagnostics`` is a reading of the term, never a row of it. It carries
    the named ledger flows of :data:`CARRIER_LEDGER_ROWS` as
    ``<row>_per_s`` / ``<row>_W`` / ``<row>_dyn`` scalars, the per-cell
    deposition and escape profiles the TPMC read is validated against, and the
    launch kinematics. Every array in it is the PRE-MASK quantity the rows
    were built from.
    """
    cells = int(geometry.cells)
    zeros = np.zeros(cells, dtype=float)
    launch = np.asarray(launch_per_s, dtype=float)
    roles = np.asarray(geometry.cell_role)
    absorbing = np.asarray(
        getattr(geometry, "plasma_absorbing", np.zeros(0)), dtype=bool
    )
    anode_faces = set(
        int(face)
        for face in np.asarray(
            getattr(geometry, "anode_face_indices", ()), dtype=int
        )
    )
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    Ti = np.asarray(derived.Ti, dtype=float)
    u_i = np.asarray(derived.u, dtype=float)
    length = np.asarray(geometry.length_cm, dtype=float)

    # Particle rates [s^-1] per cell, by fate.
    dep_cx = zeros.copy()
    dep_ion = zeros.copy()
    escaped = zeros.copy()
    culled = zeros.copy()
    ended = zeros.copy()
    # The energy [erg/s] each fate carried into the cell it reached, and the
    # signed axial momentum [g cm s^-2] it carried there. Accumulated per cell
    # because a twin machine launches two beams, at their own speeds and with
    # opposite signs, into the same cells.
    eE_cx = zeros.copy()
    eE_ion = zeros.copy()
    eE_esc = zeros.copy()
    eE_cull = zeros.copy()
    eE_end = zeros.copy()
    p_in_cx = zeros.copy()
    p_in_ion = zeros.copy()
    p_leak = zeros.copy()
    flux_profile = zeros.copy()
    launch_erg_s = 0.0
    launch_dyn = 0.0
    v_fast_report = 0.0
    E_fast_report = 0.0

    for face in np.flatnonzero(absorbing):
        face = int(face)
        live = int(geometry.plasma_face_live_cell[face])
        if live < 0 or roles[live] != "cathode":
            continue
        rate = float(launch[live])
        if rate <= 0.0:
            continue
        # Outward normal: plasma on the high-z side of the surface (the live
        # cell IS the face index) flows to -z to reach it, so the beam leaves
        # the surface the other way. This is exactly the ``-outward`` the
        # boundary term directs its jet momentum along.
        direction = 1.0 if live == face else -1.0
        v_fast = float(
            cathode_jet_backscatter_speed(
                cathode_jet, float(Ti[live]), ion_mass_g
            )
        )
        if not np.isfinite(v_fast) or v_fast <= 0.0:
            continue
        E_fast = 0.5 * ion_mass_g * v_fast**2
        p_fast = direction * ion_mass_g * v_fast
        v_fast_report = v_fast
        E_fast_report = E_fast
        launch_erg_s += rate * E_fast
        launch_dyn += rate * p_fast

        idx = carrier_beam_segment(geometry, live, direction)
        if idx.size == 0:
            continue
        a_cx, a_ion, a_esc = carrier_attenuation_coefficients(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            geometry=geometry,
            v_fast=v_fast,
            direction=direction,
            ionization_rate_per_neutral=ionization_rate_per_neutral,
            derived=derived,
        )
        a_cx = a_cx[idx]
        a_ion = a_ion[idx]
        a_esc = a_esc[idx]
        a_tot = a_cx + a_ion + a_esc
        tau = a_tot * length[idx]
        # The mesh cull is resolved on ENTRY to a cell, at the face just
        # crossed; the first cell is entered off the surface, not through a
        # face, so it is never culled. Face ``f`` separates cell ``f-1`` from
        # cell ``f``, so a +z hop into cell ``c`` crosses face ``c`` and a -z
        # hop into cell ``c`` crosses face ``c+1``.
        offset = 0 if direction > 0 else 1
        crossed = np.asarray(
            [int(cell) + offset in anode_faces for cell in idx], dtype=bool
        )
        crossed[0] = False
        survive = np.exp(-tau)
        enter = (
            rate
            * np.cumprod(np.where(crossed, 1.0 - float(eta), 1.0))
            * np.exp(-(np.cumsum(tau) - tau))
        )
        leave = enter * survive
        absorbed = enter - leave
        # Split the absorbed flux between the three channels in proportion to
        # their coefficients; a_tot is strictly positive on every marched cell
        # (a_esc alone is 1/Rp > 0 there).
        share = np.where(a_tot > 0.0, absorbed / np.maximum(a_tot, 1e-300), 0.0)
        cx_k = share * a_cx
        ion_k = share * a_ion
        esc_k = share * a_esc
        # The culled share is taken off the flux LEAVING the previous cell and
        # re-emitted on that incident side.
        cull_k = np.zeros(idx.size, dtype=float)
        hit = np.flatnonzero(crossed)
        if hit.size:
            np.add.at(cull_k, hit - 1, leave[hit - 1] * float(eta))
        end_k = float(leave[-1])
        last = int(idx[-1])

        dep_cx[idx] += cx_k
        dep_ion[idx] += ion_k
        escaped[idx] += esc_k
        culled[idx] += cull_k
        ended[last] += end_k
        flux_profile[idx] += enter

        eE_cx[idx] += cx_k * E_fast
        eE_ion[idx] += ion_k * E_fast
        eE_esc[idx] += esc_k * E_fast
        eE_cull[idx] += cull_k * E_fast
        eE_end[last] += end_k * E_fast
        p_in_cx[idx] += cx_k * p_fast
        p_in_ion[idx] += ion_k * p_fast
        p_leak[idx] += (esc_k + cull_k) * p_fast
        p_leak[last] += end_k * p_fast

    Vp = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    Vm = np.asarray(geometry.neutral_volume_cm3, dtype=float)
    two_zone = state.nn_a is not None
    V_nn = Vp if two_zone else Vm
    V_Mn = Vp if state.M_n_a is not None else Vm
    V_ann = np.maximum(Vm - Vp, 1e-300)

    # The CX partner is born at the local ion state: it takes the ion's
    # thermal energy and its drift momentum with it.
    partner_En = 1.5 * dep_cx * Ti * ev_to_erg
    partner_Mn = dep_cx * ion_mass_g * u_i
    # Escaped and end-lost atoms join the gas OUTSIDE the column (the annulus
    # when the zones are split, which carries no energy field; the
    # chamber-mean row otherwise), wall-thermalized either way. Mesh-culled
    # atoms re-emit into the column on the side they arrived from.
    outside = escaped + ended
    column_return = culled if two_zone else culled + outside
    annulus_return = outside if two_zone else zeros
    returned_column = neutral_energy_floor(column_return)
    # What each leak actually takes OUT of the model: the energy it carried in,
    # less whatever the wall-temperature return credited back on an
    # energy-carrying row.
    leak_wall = float(np.sum(eE_esc)) - (
        0.0 if two_zone else float(np.sum(neutral_energy_floor(escaped)))
    )
    leak_end = float(np.sum(eE_end)) - (
        0.0 if two_zone else float(np.sum(neutral_energy_floor(ended)))
    )
    leak_mesh = float(np.sum(eE_cull)) - float(
        np.sum(neutral_energy_floor(culled))
    )

    rhs = ConservativeState1D(
        n=dep_ion / Vp,
        nn=(dep_cx + column_return) / V_nn,
        M=(p_in_cx - partner_Mn + p_in_ion) / Vp,
        Ee=-float(I_ion) * ev_to_erg * dep_ion / Vp,
        Ei=(eE_cx - partner_En + eE_ion) / Vp,
        M_n=None if state.M_n is None else partner_Mn / V_Mn,
        nn_a=None if not two_zone else annulus_return / V_ann,
        M_n_a=None if state.M_n_a is None else zeros.copy(),
        En=(
            None
            if state.En is None
            else (partner_En + returned_column) / V_nn
        ),
    )

    launched = float(np.sum(launch))
    n_cx = float(np.sum(dep_cx))
    n_ion = float(np.sum(dep_ion))
    interactions = n_cx + n_ion
    diagnostics = {
        # --- the named ledger flows ---------------------------------------
        "launch_per_s": launched,
        "launch_W": launch_erg_s / _ERG_S_PER_W,
        "launch_dyn": launch_dyn,
        "jet_ionization_per_s": n_ion,
        "jet_ionization_W": float(np.sum(eE_ion)) / _ERG_S_PER_W,
        "jet_ionization_dyn": float(np.sum(p_in_ion)),
        "jet_ionization_cost_W": (
            float(I_ion) * ev_to_erg * n_ion / _ERG_S_PER_W
        ),
        "partner_exchange_per_s": n_cx,
        "partner_exchange_W": (
            float(np.sum(eE_cx - partner_En)) / _ERG_S_PER_W
        ),
        "partner_exchange_dyn": float(np.sum(p_in_cx - partner_Mn)),
        "partner_exchange_carried_W": float(np.sum(eE_cx)) / _ERG_S_PER_W,
        "partner_exchange_En_W": float(np.sum(partner_En)) / _ERG_S_PER_W,
        "partner_exchange_Mn_dyn": float(np.sum(partner_Mn)),
        "wall_leak_per_s": float(np.sum(escaped)),
        "wall_leak_W": leak_wall / _ERG_S_PER_W,
        "end_leak_per_s": float(np.sum(ended)),
        "end_leak_W": leak_end / _ERG_S_PER_W,
        "mesh_cull_per_s": float(np.sum(culled)),
        "mesh_cull_W": leak_mesh / _ERG_S_PER_W,
        "leak_dyn": float(np.sum(p_leak)),
        "column_return_W": float(np.sum(returned_column)) / _ERG_S_PER_W,
        "v1_withdrawal_per_s": launched,
        "v1_withdrawal_W": launch_erg_s / _ERG_S_PER_W,
        "v1_withdrawal_dyn": launch_dyn,
        # --- kinematics and the TPMC-comparable readings -------------------
        "v_fast_cm_s": v_fast_report,
        "E_fast_eV": E_fast_report / ev_to_erg,
        "f_dep": interactions / max(launched, 1e-300),
        "cx_share": n_cx / max(interactions, 1e-300),
        # --- per-cell profiles, pre-mask -----------------------------------
        "carrier_flux": flux_profile,
        "carrier_cx": dep_cx,
        "carrier_ionization": dep_ion,
        "carrier_escape": escaped,
        "carrier_mesh_cull": culled,
        "carrier_end_leak": ended,
    }
    return rhs, diagnostics
