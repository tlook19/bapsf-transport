import numpy as np

from cablp.funcs._cross import charge_ex_react
from cablp.vars._cons import ev_to_erg

from .flux import ion_sound_speed
from ..core.geometry import PLASMA_DEAD_ROLES
from ..core.state import ConservativeState1D, derive_state


def velocity_divergence(state, floors, ion_mass_g, geometry):
    """Return finite-volume axial velocity divergence [s^-1]."""
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    face_u = np.zeros(geometry.cells + 1, dtype=float)
    face_u[1:-1] = 0.5 * (derived.u[:-1] + derived.u[1:])
    inventory_rate = geometry.plasma_face_area_cm2 * face_u
    return (inventory_rate[1:] - inventory_rate[:-1]) / geometry.plasma_volume_cm3


def pressure_work_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    electron_scale=1.0,
    ion_scale=1.0,
):
    """Return conservative electron/ion pressure-work energy sources."""
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    div_u = velocity_divergence(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
    )
    zeros = np.zeros(geometry.cells, dtype=float)
    return ConservativeState1D(
        n=zeros.copy(),
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=-float(electron_scale) * derived.pe * div_u,
        Ei=-float(ion_scale) * derived.pi * div_u,
    )


def surface_neutralization_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    alpha_isat=np.exp(-0.5),
    source_surface_area_scale=2.0,
    end_surface_area_scale=1.0,
    source_surface_loss_enabled=True,
    end_surface_loss_enabled=True,
    end_mode="collector",
    b_surface_loss=1.0,
):
    """Return conservative source/end surface plasma neutralization terms.

    Superseded wherever the geometry carries an absorbing face: a plasma-
    terminating surface removes its plasma through the face itself (plan §11
    decision 3), so applying this volumetric form as well would neutralize the
    same plasma twice. Legacy geometry has no absorbing faces and is unaffected.
    """
    zeros = np.zeros(geometry.cells, dtype=float)
    if np.any(np.asarray(getattr(geometry, "plasma_absorbing", ()), dtype=bool)):
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    if b_surface_loss == 0.0:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    if end_mode not in {"collector", "mirrored_source"}:
        raise ValueError(
            "end_mode must be 'collector' or 'mirrored_source' "
            f"(got {end_mode!r})"
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    dN_loss = np.zeros(geometry.cells, dtype=float)
    if source_surface_loss_enabled:
        dN_loss[0] = _cell_surface_particle_loss(
            n=state.n[0],
            Te=derived.Te[0],
            mu=mu,
            area_cm2=source_surface_area_scale * geometry.plasma_area_cm2[0],
            alpha_isat=alpha_isat,
        )
    if end_surface_loss_enabled:
        dN_loss[-1] = _cell_surface_particle_loss(
            n=state.n[-1],
            Te=derived.Te[-1],
            mu=mu,
            area_cm2=end_surface_area_scale * geometry.plasma_area_cm2[-1],
            alpha_isat=alpha_isat,
        )
    dN_loss *= float(b_surface_loss)

    plasma_loss_rate = dN_loss / geometry.plasma_volume_cm3
    neutral_gain_rate = dN_loss / geometry.neutral_volume_cm3
    return ConservativeState1D(
        n=-plasma_loss_rate,
        nn=neutral_gain_rate,
        M=-ion_mass_g * derived.u * plasma_loss_rate,
        Ee=-1.5 * ev_to_erg * derived.Te * plasma_loss_rate,
        Ei=-1.5 * ev_to_erg * derived.Ti * plasma_loss_rate,
    )


def presheath_length_cm(nn, Te, Ti, mu, ion_mass_g, sigma_in_cm2=5.0e-15):
    """Return the collisional presheath depth in front of a surface [cm].

    Ions are accelerated to ``c_s`` across the presheath, and cannot be freely
    accelerated over more than an ion-neutral momentum-transfer mean free path,
    so ``L_ps ~ c_s / nu_in``. In this device that runs from ~66 cm when the gas
    is cold and rarefied to ~5 cm once the discharge is hot and dense, which is
    what makes the sampling depth self-selecting rather than a tuned constant.
    """
    nu_in = ion_neutral_collision_frequency(
        nn=nn, Ti=Ti, ion_mass_g=ion_mass_g, sigma_in_cm2=sigma_in_cm2
    )
    if nu_in <= 0.0 or not np.isfinite(nu_in):
        return np.inf
    return float(ion_sound_speed(Te, mu) / nu_in)


def presheath_alpha(alpha_isat, cell_length_cm, presheath_cm):
    """Return the sheath-edge conversion factor for a *locally* sampled density.

    ``alpha_isat = exp(-1/2)`` is the Boltzmann drop across the whole presheath,
    ``n_se = n_0 * exp(-1/2)``, so it is only correct when the sampled density is
    the presheath-*entrance* density. Sampling at depth ``d`` inside the presheath
    catches only part of that drop -- for a linear potential profile
    ``n(d) = n_0 * exp(-(1/2)(1 - d/L_ps))`` -- leaving

        n_se = n(d) * exp(-(1/2) * d / L_ps)

    so the factor is ``alpha_isat ** (d / L_ps)`` with ``d`` capped at one
    presheath depth. The two limits are the physical ones:

    - presheath **fits inside** the cell (``L_ps <= d``): the cell average is the
      upstream reservoir, so the full ``exp(-1/2)`` applies.
    - presheath **much longer** than the cell: the cell already sits at the sheath
      edge, so no further reduction applies and the factor tends to 1.

    This is also self-consistently mesh-independent: refine the cell and the local
    density falls along the same Boltzmann profile that the exponent compensates
    for, leaving ``n_se`` unchanged. And since the factor never exceeds 1, the
    flux can never exceed what the cell can deliver at the sound speed.
    """
    if not np.isfinite(presheath_cm) or presheath_cm <= 0.0:
        return float(alpha_isat)
    fraction = min(float(cell_length_cm), float(presheath_cm)) / float(presheath_cm)
    return float(alpha_isat) ** fraction


def boundary_absorption_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    alpha_isat=np.exp(-0.5),
    b_surface_loss=1.0,
    sigma_in_cm2=5.0e-15,
    b_presheath_length=1.0,
):
    """Return the plasma absorbed by the plasma-terminating surfaces.

    The cathode and collector surfaces end the plasma domain, so the Bohm
    criterion applies to the face itself (plan §11 decision 3): plasma leaves at
    the sound speed and is neutralized on the surface.

    Applied one-sidedly to the live cell rather than as a face flux, because the
    flux array telescopes: an *interior* absorbing face would otherwise hand the
    plasma it removes to the plasma-dead plenum behind it, and kick that cell with
    sonic momentum while its density sits on the floor.

    The sonic condition is what distinguishes this from the historical volumetric
    surface term: momentum leaves at ``c_s`` directed *into* the surface, not at
    the cell's own drift ``u``, so the loss actually drives flow toward the wall
    instead of deleting plasma that was never moving there. Legacy geometry has no
    absorbing faces, so it keeps ``surface_neutralization_rhs`` unchanged.
    """
    zeros = np.zeros(geometry.cells, dtype=float)
    absorbing = np.asarray(
        getattr(geometry, "plasma_absorbing", np.zeros(0)), dtype=bool
    )
    if not np.any(absorbing) or b_surface_loss == 0.0:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    roles = np.asarray(geometry.cell_role)
    dead = np.asarray([role in PLASMA_DEAD_ROLES for role in roles], dtype=bool)
    cells = roles.size
    dN_loss = np.zeros(cells, dtype=float)
    sonic_momentum = np.zeros(cells, dtype=float)
    for face in np.flatnonzero(absorbing):
        face = int(face)
        left, right = face - 1, face
        live_is_right = left < 0 or (right < cells and not dead[right])
        live = right if live_is_right else left
        # Outward normal: plasma on the high-z side of the surface flows toward
        # -z to reach it, and vice versa.
        outward = -1.0 if live_is_right else 1.0
        cs = ion_sound_speed(derived.Te[live], mu)
        # Apply only the portion of the presheath drop that the cell actually
        # spans: the full exp(-1/2) is valid when the presheath fits inside the
        # cell, and tends to no correction when the cell is buried inside a
        # presheath much longer than it.
        presheath_cm = b_presheath_length * presheath_length_cm(
            nn=state.nn[live],
            Te=derived.Te[live],
            Ti=derived.Ti[live],
            mu=mu,
            ion_mass_g=ion_mass_g,
            sigma_in_cm2=sigma_in_cm2,
        )
        alpha_eff = presheath_alpha(
            alpha_isat=alpha_isat,
            cell_length_cm=geometry.length_cm[live],
            presheath_cm=presheath_cm,
        )
        loss = _cell_surface_particle_loss(
            n=state.n[live],
            Te=derived.Te[live],
            mu=mu,
            area_cm2=float(geometry.plasma_face_area_cm2[face]),
            alpha_isat=alpha_eff,
        )
        dN_loss[live] += loss
        sonic_momentum[live] += ion_mass_g * outward * cs * loss
    dN_loss *= float(b_surface_loss)
    sonic_momentum *= float(b_surface_loss)

    plasma_loss_rate = dN_loss / geometry.plasma_volume_cm3
    return ConservativeState1D(
        n=-plasma_loss_rate,
        nn=dN_loss / geometry.neutral_volume_cm3,
        M=-sonic_momentum / geometry.plasma_volume_cm3,
        Ee=-1.5 * ev_to_erg * derived.Te * plasma_loss_rate,
        Ei=-1.5 * ev_to_erg * derived.Ti * plasma_loss_rate,
    )


def anode_collection_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    eta,
    alpha_isat=np.exp(-0.5),
    b_anode_collection=1.0,
):
    """Return the plasma the anode mesh collects and neutralizes.

    A sheath forms on every mesh wire, so ions reach it at the **Bohm flux**
    ``exp(-0.5) * n * c_s`` -- set by the sheath, not by the bulk drift. A mesh
    sitting in stagnant plasma still collects; one in fast-flowing plasma does not
    collect proportionally faster. This is why the collection cannot be written as
    the intercepted directed flux ``eta * n * u``.

    The wires present the solid fraction ``eta`` of the plasma cross-section to
    *each* side, and each face is evaluated against the plasma actually on that
    side, so a mesh separating hot gap plasma from cooler column plasma collects
    asymmetrically -- the sum is the plan's ``2 * eta * I_i_a`` with each half
    sampled locally. Neutrals are released on the side they were collected from,
    since a wire blocks the path to the other side and the mesh throttles neutral
    flow between them (§7).

    The full ``alpha_isat`` applies here, and that is the *same* rule
    ``boundary_absorption_rhs`` uses rather than an exception to it. The factor is
    attenuated by how much of the presheath a cell spans (``presheath_alpha``),
    and a mesh's presheath is **geometric**, not collisional: only ``eta`` of the
    cross-section terminates and the rest streams past, so each wire carries its
    own presheath on the scale of the wire spacing -- sub-millimetre, thousands of
    times shorter than a cell. The presheath therefore always fits inside the
    cell, the fraction is 1, and the factor is the undiminished ``exp(-1/2)``.
    Equivalently: from the wires' point of view the cell average already *is* the
    upstream reservoir. The depletion the mesh does cause is between its two
    sides, which the grid resolves by sampling each flanking cell separately.

    Mass, momentum and thermal energy leave together as at any wall; the collected
    momentum is absorbed by the grounded anode structure rather than heating the
    ions (§5). ``eta = 0`` gives a transparent anode -- the legacy limit -- and
    legacy geometry has no anode faces at all.
    """
    zeros = np.zeros(geometry.cells, dtype=float)
    anode_faces = np.asarray(
        getattr(geometry, "anode_face_indices", ()), dtype=int
    )
    if anode_faces.size == 0 or eta <= 0.0 or b_anode_collection == 0.0:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    dN_loss = np.zeros(geometry.cells, dtype=float)
    for face in anode_faces:
        for cell in (int(face) - 1, int(face)):
            dN_loss[cell] += _cell_surface_particle_loss(
                n=state.n[cell],
                Te=derived.Te[cell],
                mu=mu,
                area_cm2=float(eta) * geometry.plasma_area_cm2[cell],
                alpha_isat=alpha_isat,
            )
    dN_loss *= float(b_anode_collection)

    plasma_loss_rate = dN_loss / geometry.plasma_volume_cm3
    neutral_gain_rate = dN_loss / geometry.neutral_volume_cm3
    return ConservativeState1D(
        n=-plasma_loss_rate,
        nn=neutral_gain_rate,
        M=-ion_mass_g * derived.u * plasma_loss_rate,
        Ee=-1.5 * ev_to_erg * derived.Te * plasma_loss_rate,
        Ei=-1.5 * ev_to_erg * derived.Ti * plasma_loss_rate,
    )


def _cell_surface_particle_loss(n, Te, mu, area_cm2, alpha_isat):
    return float(alpha_isat) * n * ion_sound_speed(Te, mu) * area_cm2


def ion_neutral_collision_frequency(nn, Ti, ion_mass_g, sigma_in_cm2=5.0e-15):
    """Return the ion-neutral momentum-transfer collision frequency [s^-1].

    ``nu_in = (8/3) * nn * sigma_in * sqrt(Ti / (pi * m_i))`` with ``Ti`` in eV
    (converted to erg here), ``m_i`` in grams, and ``sigma_in`` in cm^2, so the
    thermal-speed factor is in cm/s and ``nu_in`` in s^-1.
    """
    v_thi = np.sqrt(np.asarray(Ti, dtype=float) * ev_to_erg / (np.pi * ion_mass_g))
    return (8.0 / 3.0) * np.asarray(nn, dtype=float) * float(sigma_in_cm2) * v_thi


def ion_neutral_cx_frequency(nn, Ti, gas_type):
    """Return the resonant charge-exchange collision frequency [s^-1].

    ``nu_cx = nn * <sigma v>_cx(Ti)`` using the same rate table as the ``Q_cx``
    energy term.
    """
    return np.asarray(nn, dtype=float) * charge_ex_react(Ti, gas_type)


def ion_neutral_momentum_frequency(
    nn,
    Ti,
    ion_mass_g,
    gas_type,
    sigma_in_cm2=5.0e-15,
    cx_only=False,
):
    """Return the ion-neutral momentum-transfer frequency [s^-1].

    With ``cx_only=False`` this is the total rate ``nu_in`` from ``sigma_in``.
    With ``cx_only=True`` the drag is driven purely by the resonant
    charge-exchange rate ``nu_cx`` (the same rate the ``Q_cx`` energy term uses),
    for which there is no elastic momentum transfer.
    """
    if cx_only:
        return ion_neutral_cx_frequency(nn=nn, Ti=Ti, gas_type=gas_type)
    return ion_neutral_collision_frequency(
        nn=nn,
        Ti=Ti,
        ion_mass_g=ion_mass_g,
        sigma_in_cm2=sigma_in_cm2,
    )


def ion_neutral_drag_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    sigma_in_cm2=5.0e-15,
    b_ion_neutral_drag=1.0,
    cx_only=False,
):
    """Return the conservative ion-neutral drag momentum sink.

    The drag force density is ``-m_i * nu(Ti) * n * u`` [g cm^-2 s^-2], a
    friction on the plasma flow from collisions with the neutral background.
    ``nu`` is the total momentum-transfer rate, or the charge-exchange-only rate
    when ``cx_only`` is set. Only the momentum field is affected.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    if b_ion_neutral_drag == 0.0:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    nu = ion_neutral_momentum_frequency(
        nn=state.nn,
        Ti=derived.Ti,
        ion_mass_g=ion_mass_g,
        gas_type=gas_type,
        sigma_in_cm2=sigma_in_cm2,
        cx_only=cx_only,
    )
    drag = -float(b_ion_neutral_drag) * ion_mass_g * nu * state.n * derived.u
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=drag,
        Ee=zeros.copy(),
        Ei=zeros.copy(),
    )


def ion_neutral_elastic_frequency(
    nn,
    Ti,
    ion_mass_g,
    gas_type,
    sigma_in_cm2=5.0e-15,
    cx_only=False,
):
    """Return the elastic (non-CX) ion-neutral momentum-transfer frequency [s^-1].

    ``nu_el = max(nu_in - nu_cx, 0)`` where ``nu_in`` is the total (``sigma_in``)
    momentum-transfer rate and ``nu_cx = nn * <sigma v>_cx`` is the resonant
    charge-exchange rate shared with the ``Q_cx`` energy term. When ``cx_only``
    is set the drag carries no elastic fraction, so ``nu_el = 0``.
    """
    if cx_only:
        return np.zeros_like(np.asarray(nn, dtype=float))
    nu_in = ion_neutral_collision_frequency(
        nn=nn,
        Ti=Ti,
        ion_mass_g=ion_mass_g,
        sigma_in_cm2=sigma_in_cm2,
    )
    nu_cx = ion_neutral_cx_frequency(nn=nn, Ti=Ti, gas_type=gas_type)
    return np.maximum(nu_in - nu_cx, 0.0)


def ion_neutral_frictional_heating_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    sigma_in_cm2=5.0e-15,
    b_ion_neutral_drag=1.0,
    cx_only=False,
):
    """Return the conservative ion frictional-heating energy source.

    Elastic ion-neutral collisions thermalize the flow's directed energy; for
    equal masses half of the dissipated drift energy heats the ions, giving the
    ``Ei`` source ``+(1/2) m_i * nu_el(Ti) * n * u^2`` [erg cm^-3 s^-1]. The
    charge-exchange fraction carries its energy off with the fast neutral and is
    excluded via ``nu_el = nu_in - nu_cx``; when ``cx_only`` is set there is no
    elastic fraction so this source vanishes.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    if b_ion_neutral_drag == 0.0:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    nu_el = ion_neutral_elastic_frequency(
        nn=state.nn,
        Ti=derived.Ti,
        ion_mass_g=ion_mass_g,
        gas_type=gas_type,
        sigma_in_cm2=sigma_in_cm2,
        cx_only=cx_only,
    )
    q_fric = (
        0.5
        * float(b_ion_neutral_drag)
        * ion_mass_g
        * nu_el
        * state.n
        * derived.u**2
    )
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=q_fric,
    )


def ion_neutral_thermalization_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    Tn_fit=0.1,
    sigma_in_cm2=5.0e-15,
    b_ion_neutral_drag=1.0,
    cx_only=False,
):
    """Return the conservative elastic ion-neutral thermal-equilibration source.

    Elastic collisions relax ``Ti`` toward the neutral temperature at the elastic
    rate, giving the ``Ei`` source ``+(3/2) nu_el(Ti) * n * (Tn - Ti)``
    [erg cm^-3 s^-1]. This is the elastic companion to the CX ``Q_cx`` cooling and
    is gated separately by the ``ion_neutral_thermalization`` flag; when
    ``cx_only`` is set there is no elastic fraction so this source vanishes.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    if b_ion_neutral_drag == 0.0:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    nu_el = ion_neutral_elastic_frequency(
        nn=state.nn,
        Ti=derived.Ti,
        ion_mass_g=ion_mass_g,
        gas_type=gas_type,
        sigma_in_cm2=sigma_in_cm2,
        cx_only=cx_only,
    )
    q_eq = (
        1.5
        * float(b_ion_neutral_drag)
        * nu_el
        * state.n
        * (float(Tn_fit) - derived.Ti)
        * ev_to_erg
    )
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=q_eq,
    )


def add_state_rhs(left, right):
    """Return the sum of two conservative RHS bundles."""
    return ConservativeState1D(
        n=left.n + right.n,
        nn=left.nn + right.nn,
        M=left.M + right.M,
        Ee=left.Ee + right.Ee,
        Ei=left.Ei + right.Ei,
    )
