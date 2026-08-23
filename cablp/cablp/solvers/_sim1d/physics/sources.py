import numpy as np
from scipy.special import expn

from cablp.funcs._cross import (
    charge_ex_react,
    phelps_cx_rate_cm3_s,
    phelps_momentum_transfer_rate_cm3_s,
)
from cablp.vars._cons import ev_to_erg, kb_cgs

from .flux import (
    ion_sound_speed,
    plasma_wave_speed,
    _flux_divergence,
    kep_rusanov_face_scalar,
)
from ..core.state import (
    ConservativeState1D,
    derive_state,
    neutral_energy_floor,
)


def velocity_divergence(
    state, floors, ion_mass_g, geometry, active_plasma_topology=False
):
    """Return finite-volume axial velocity divergence [s^-1]."""
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    face_u = np.zeros(geometry.cells + 1, dtype=float)
    face_u[1:-1] = 0.5 * (derived.u[:-1] + derived.u[1:])
    if active_plasma_topology:
        for face in np.flatnonzero(~np.asarray(geometry.plasma_open, dtype=bool)):
            live = int(geometry.plasma_face_live_cell[face])
            face_u[face] = 0.0 if live < 0 else derived.u[live]
    inventory_rate = geometry.plasma_face_area_cm2 * face_u
    return (inventory_rate[1:] - inventory_rate[:-1]) / geometry.plasma_volume_cm3


def pressure_work_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    electron_scale=1.0,
    ion_scale=1.0,
    active_plasma_topology=False,
):
    """Return conservative electron/ion pressure-work energy sources."""
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    div_u = velocity_divergence(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        active_plasma_topology=active_plasma_topology,
    )
    zeros = np.zeros(geometry.cells, dtype=float)
    return ConservativeState1D(
        n=zeros.copy(),
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=-float(electron_scale) * derived.pe * div_u,
        Ei=-float(ion_scale) * derived.pi * div_u,
    )


def _pressure_flux_divergence(p, geometry, active_plasma_topology):
    """Divergence of the momentum pressure flux {p} for one species pressure.

    Interior faces carry the arithmetic-mean pressure ``0.5*(p_L+p_R)``; closed
    faces carry the live-side cell pressure, matching
    ``flux._apply_plasma_walls`` for the momentum row. Transmission is applied on
    partially blocked (anode-mesh) faces.
    """
    p = np.asarray(p, dtype=float)
    cells = geometry.cells
    face = np.zeros(cells + 1, dtype=float)
    face[1:-1] = 0.5 * (p[:-1] + p[1:])
    face = face * np.asarray(geometry.plasma_transmission, dtype=float)
    dead = ~np.asarray(geometry.plasma_active, dtype=bool)
    for f in np.flatnonzero(~np.asarray(geometry.plasma_open, dtype=bool)):
        f = int(f)
        if active_plasma_topology:
            live = int(geometry.plasma_face_live_cell[f])
            face[f] = 0.0 if live < 0 else p[live]
        else:
            left, right = f - 1, f
            live_is_right = left < 0 or (right < cells and not dead[right])
            live = right if live_is_right else left
            face[f] = p[live]
    return _flux_divergence(face, geometry)


def hyperbolic_energy_correction_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    wave_speed="isothermal",
    active_plasma_topology=False,
    electron_scale=1.0,
    ion_scale=1.0,
):
    """Return the R2 kinetic-energy-preserving energy-consistency correction.

    Added on top of the plasma advective flux and the ``-p div u`` pressure
    work, this term (i) deposits the Rusanov ``(n, M)`` numerical kinetic-energy
    dissipation into the ion internal energy and (ii) converts the electron and
    ion pressure work from ``-p_s div u`` to the kinetic-energy-preserving
    ``-u dM_press_s`` form. Combined with the KEP convective momentum flux
    (``flux._rusanov_raw_faces(energy_consistent=True)``) the flux plus
    pressure-work operator then conserves the closed-domain total plasma energy
    ``K + Ee + Ei`` to machine precision. Off-path callers never build it, so
    the operator is structurally inert when the selector is off.
    """
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    u = derived.u
    cells = geometry.cells
    n = np.asarray(state.n, dtype=float)
    M = np.asarray(state.M, dtype=float)

    cs = plasma_wave_speed(derived.Te, derived.Ti, mu, wave_speed)
    amax = np.maximum(np.abs(u[:-1]) + cs[:-1], np.abs(u[1:]) + cs[1:])
    open_faces = np.asarray(geometry.plasma_open, dtype=bool)
    transmission = np.asarray(geometry.plasma_transmission, dtype=float)

    def _dissipative_divergence(field):
        face = np.zeros(cells + 1, dtype=float)
        face[1:-1] = -0.5 * amax * (field[1:] - field[:-1])
        face = face * transmission
        face[~open_faces] = 0.0
        return _flux_divergence(face, geometry)

    dn_diss = _dissipative_divergence(n)
    dM_diss = _dissipative_divergence(M)
    dK_diss = u * dM_diss - 0.5 * ion_mass_g * u**2 * dn_diss

    dK_press_e = u * _pressure_flux_divergence(
        derived.pe, geometry, active_plasma_topology
    )
    dK_press_i = u * _pressure_flux_divergence(
        derived.pi, geometry, active_plasma_topology
    )

    div_u = velocity_divergence(
        state, floors, ion_mass_g, geometry,
        active_plasma_topology=active_plasma_topology,
    )

    zeros = np.zeros(cells, dtype=float)
    d_Ee = float(electron_scale) * (derived.pe * div_u - dK_press_e)
    d_Ei = float(ion_scale) * (derived.pi * div_u - dK_press_i) - dK_diss
    return ConservativeState1D(
        n=zeros.copy(),
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=d_Ee,
        Ei=d_Ei,
    )


def flux_tube_geometry_rhs(state, floors, ion_mass_g, geometry):
    """Return the quasi-1D pressure force for a variable-area flux tube.

    The conservative momentum equation is

        d(A rho u)/dt + d[A(rho u^2 + p)]/dz = p dA/dz + A S_M.

    ``physics.flux`` already carries the area-weighted flux divergence. This
    source supplies the matching ``p dA/dz`` term. Its discrete form exactly
    cancels the pressure-flux divergence for a uniform stationary plasma, so a
    geometric flare cannot create momentum from a constant-pressure state.
    """
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    zeros = np.zeros(geometry.cells, dtype=float)
    area = np.asarray(geometry.plasma_face_area_cm2, dtype=float)
    # Keep the same multiply-then-subtract ordering as the pressure-flux
    # divergence. That makes the uniform-state balance bit-exact instead of
    # merely algebraically equivalent up to roundoff.
    dM = (
        derived.p * area[1:] - derived.p * area[:-1]
    ) / np.asarray(geometry.plasma_volume_cm3, dtype=float)
    return ConservativeState1D(
        n=zeros.copy(),
        nn=zeros.copy(),
        M=dM,
        Ee=zeros.copy(),
        Ei=zeros.copy(),
        M_n=np.zeros_like(state.M_n) if state.M_n is not None else None,
        nn_a=np.zeros_like(state.nn_a) if state.nn_a is not None else None,
        M_n_a=np.zeros_like(state.M_n_a) if state.M_n_a is not None else None,
    )


def presheath_length_cm(
    nn,
    Te,
    Ti,
    mu,
    ion_mass_g,
    gas_type=None,
    Tn_eV=None,
):
    """Return the collisional presheath depth in front of a surface [cm].

    Ions are accelerated to ``c_s`` across the presheath, and cannot be freely
    accelerated over more than an ion-neutral momentum-transfer mean free path,
    so ``L_ps ~ c_s / nu_in``. In this device that runs from ~66 cm when the gas
    is cold and rarefied to ~5 cm once the discharge is hot and dense, which is
    what makes the sampling depth self-selecting rather than a tuned constant.

    ``Tn_eV`` is the neutral temperature entering the collisionality's
    ``T_eff = (Ti + Tn)/2``. ``None`` (every historical caller) leaves
    ``ion_neutral_collision_frequency`` on its own fixed cold-gas value, so
    the default path is unchanged bit for bit; a value is supplied only by
    the kinetic DVM arm's Tn-feedback switch, which measures ``Tn`` from the
    live distribution instead of assuming it.
    """
    nu_in = ion_neutral_collision_frequency(
        nn=nn,
        Ti=Ti,
        gas_type=gas_type,
        **({} if Tn_eV is None else {"Tn_eV": float(Tn_eV)}),
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


def electrode_sheath_alpha(
    nn,
    Te,
    Ti,
    cell_length_cm,
    mu,
    ion_mass_g,
    alpha_isat=np.exp(-0.5),
    b_presheath_length=1.0,
    gas_type=None,
):
    """Return the mesh-independent sheath-edge factor ``n_se/n`` at one cell.

    The single source of truth for the collisional-presheath sampling
    (R3.2 / A16): one mesh-independent sheath-edge density ``n_se``, SHARED by
    the fluid sink, the circuit current, and the power terms.
    Both the fluid characteristic boundary (``characteristic_boundary_rhs``) and
    the circuit's cathode current (``funcs._cathode_solver_idriven.solve_idriven``
    via the cathode adapter) call this so they cannot disagree about ``n_se``.
    The anode mesh is NOT sampled here: its presheath is geometric and always
    fits inside a cell, so its factor is the flat ``exp(-1/2)`` used unchanged by
    both ``anode_collection_rhs`` and ``anode_circuit_sample``.
    """
    presheath_cm = b_presheath_length * presheath_length_cm(
        nn=nn,
        Te=Te,
        Ti=Ti,
        mu=mu,
        ion_mass_g=ion_mass_g,
        gas_type=gas_type,
    )
    return presheath_alpha(
        alpha_isat=alpha_isat,
        cell_length_cm=cell_length_cm,
        presheath_cm=presheath_cm,
    )


def _annulus_deposit_row(dN_routed, annulus_volume_cm3):
    """Return the ``nn_a`` row [cm^-3 s^-1] for a routed particle stream.

    ``dN_routed`` is the per-cell routed rate [s^-1] and
    ``annulus_volume_cm3`` the per-cell annulus volume. Cells with no routed
    stream are left at exactly zero WITHOUT dividing, so a cell that has no
    annulus at all (``V_ann = 0``, the plenum and any cell the plasma fills to
    the wall) cannot produce a ``0/0``. Every cell the routing actually
    deposits into is guaranteed a positive annulus volume at construction, so
    the division that does run is always well posed.
    """
    dN_routed = np.asarray(dN_routed, dtype=float)
    row = np.zeros_like(dN_routed)
    np.divide(
        dN_routed,
        np.asarray(annulus_volume_cm3, dtype=float),
        out=row,
        where=dN_routed != 0.0,
    )
    return row


#: Accepted values of the cathode jet's ``energy_convention``, which fixes how
#: ``R_E`` is read when the backscattered atoms' launch speed is built.
CATHODE_JET_ENERGY_CONVENTIONS = ("legacy", "total_reflected")


def cathode_jet_backscatter_speed(cathode_jet, Ti_eV, ion_mass_g):
    """Return the cathode jet's backscatter launch speed [cm s^-1].

    THE ONE SPEC. Every consumer of the backscattered atoms' kinetic energy
    reads it here -- the directed momentum booked by
    :func:`boundary_absorption_rhs` and :func:`characteristic_boundary_rhs`,
    and the ``En`` the solver's ``cathode_jet_neutral_energy`` term hands the
    neutral gas -- so the momentum and the energy can never describe atoms
    moving at two different speeds.

    ``cathode_jet`` is the jet spec dict (``R_N``, ``R_E``, ``phi_c_V``,
    ``T_s_K``, and optionally ``energy_convention``); ``Ti_eV`` is the local
    ion temperature [eV], scalar or per-cell; ``ion_mass_g`` the ion mass [g].
    The incident per-particle energy is ``phi_c + Ti`` [eV], clamped at zero.

    ``energy_convention`` fixes what ``R_E`` means, and therefore how much
    energy one backscattered atom leaves with:

    ``"legacy"`` (the default when the key is absent)
        ``R_E`` is read PER BACKSCATTERED PARTICLE:
        ``v_back = sqrt(2 R_E (phi_c + Ti)/m)``. Only the ``R_N`` reflected
        fraction carries it, so the gas receives ``R_N R_E`` of the incident
        ion power.
    ``"total_reflected"``
        ``R_E`` is the TOTAL reflected energy fraction -- reflected energy
        over incident energy, summed over all particles, which is the
        convention :func:`~cablp.solvers._sim1d.solver.LAPDSim1D` debits the
        cathode surface by. The ``R_N`` reflected particles carry all of it,
        so each leaves with ``R_E/R_N`` of the incident energy:
        ``v_back = sqrt(2 (R_E/R_N) (phi_c + Ti)/m)`` and the gas receives
        ``R_E`` of the incident ion power.

    Raises ``ValueError`` for any other ``energy_convention`` string.
    """
    R_E = float(cathode_jet["R_E"])
    convention = cathode_jet.get("energy_convention", "legacy")
    if convention == "legacy":
        energy_fraction = R_E
    elif convention == "total_reflected":
        energy_fraction = R_E / float(cathode_jet["R_N"])
    else:
        raise ValueError(
            "cathode jet energy_convention must be one of "
            f"{CATHODE_JET_ENERGY_CONVENTIONS} (got {convention!r})"
        )
    return np.sqrt(
        2.0
        * energy_fraction
        * np.maximum(float(cathode_jet["phi_c_V"]) + Ti_eV, 0.0)
        * ev_to_erg
        / ion_mass_g
    )


#: Accepted values of the anode jet's ``energy_convention``, which fixes how
#: ``anode_jet_R_E`` is read when the backscattered atoms' launch speed is
#: built. ``None`` (the shipped default of the config key) is NOT a member: an
#: armed anode jet must declare its convention explicitly.
ANODE_JET_ENERGY_CONVENTIONS = ("legacy", "total_reflected")


def anode_jet_backscatter_speed(anode_jet, Ti_eV, ion_mass_g):
    """Return the anode jet's backscatter launch speed [cm s^-1].

    THE ONE SPEC for the anode channel, mirroring
    :func:`cathode_jet_backscatter_speed`: the momentum
    :func:`anode_collection_rhs` books reads the launch energy here and
    nowhere else, so no second site can pick a different convention.

    ``anode_jet`` is the jet spec dict (``R_N``, ``R_E``, ``phi_a_V``,
    ``energy_convention``); ``Ti_eV`` is the local ion temperature [eV] and
    ``ion_mass_g`` the ion mass [g]. The incident per-particle energy is
    ``phi_a + Ti`` [eV], clamped at zero -- the ions fall through the
    ion-attracting anode sheath before striking the wires.

    ``energy_convention`` fixes what ``R_E`` means, and therefore how fast one
    backscattered atom leaves:

    ``"legacy"``
        ``R_E`` is read PER BACKSCATTERED PARTICLE:
        ``v_back = sqrt(2 R_E (phi_a + Ti)/m)``. This is the reading the
        anode channel was hard-coded to before the convention key existed.
    ``"total_reflected"``
        ``R_E`` is the TOTAL reflected energy fraction -- reflected energy
        over incident energy, summed over all particles, which is the
        convention the tabulated reflection coefficients are published in.
        The ``R_N`` reflected particles carry all of it, so each leaves with
        ``R_E/R_N`` of the incident energy,
        ``v_back = sqrt(2 (R_E/R_N) (phi_a + Ti)/m)``.

    Raises ``ValueError`` for any other ``energy_convention`` value, including
    the undeclared ``None``.
    """
    R_E = float(anode_jet["R_E"])
    convention = anode_jet.get("energy_convention")
    if convention == "legacy":
        energy_fraction = R_E
    elif convention == "total_reflected":
        energy_fraction = R_E / float(anode_jet["R_N"])
    else:
        raise ValueError(
            "anode jet energy_convention must be one of "
            f"{ANODE_JET_ENERGY_CONVENTIONS} (got {convention!r})"
        )
    return np.sqrt(
        2.0
        * energy_fraction
        * np.maximum(float(anode_jet["phi_a_V"]) + Ti_eV, 0.0)
        * ev_to_erg
        / ion_mass_g
    )


def boundary_absorption_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    alpha_isat=np.exp(-0.5),
    b_surface_loss=1.0,
    b_presheath_length=1.0,
    gas_type=None,
    cathode_jet=None,
    Tn_presheath_eV=None,
    end_recycle_annulus_volume_cm3=None,
    cathode_carrier_out=None,
):
    """Return the plasma absorbed by the plasma-terminating surfaces.

    ``cathode_carrier_out``: when given (a dict), the directed hot surface
    carrier is ARMED and this term stops booking the backscatter share of the
    cathode recycle itself -- see
    :func:`~.jet_carrier.cathode_jet_carrier_rhs`, which spends it instead.
    Two things change on the cathode faces alone, and both are the same
    withholding: the ``R_N`` share of the recycle flux is removed from the
    neutral rebirth row (the implanted ``1 - R_N`` effusive share stays,
    cold at the surface temperature), and the jet's per-particle momentum
    drops from ``R_N v_back + (1 - R_N) v_eff`` to ``(1 - R_N) v_eff``. The
    withheld particle rate [s^-1] per cell is written back into the dict as
    ``"launch_per_s"``, so the carrier's launch and this withdrawal are ONE
    number rather than two estimates of it. ``None`` (the default, and every
    historical caller) leaves both bookings exactly as they were, bit for
    bit. The PLASMA sink is untouched either way: the surface absorbs the
    same flux, only its re-emission changes.

    ``end_recycle_annulus_volume_cm3``: when given (the per-cell annulus
    volume [cm^3], supplied only under the ``end_recycle_to_annulus``
    closure), the recycle stream rebirthed at faces whose live cell has the
    ``collector`` role is deposited into the ANNULUS row ``nn_a`` at
    ``dN_loss / V_ann`` instead of into the column row ``nn``. CATHODE faces
    are untouched, so the jet/debit closure that owns them is unchanged. The
    routed atoms are thermal and diffuse: no directed momentum is booked
    anywhere for them, on either ``M_n`` or ``M_n_a``. ``None`` (the default,
    and every historical caller) keeps the whole stream on the column row, so
    that path is unchanged bit for bit.

    ENERGY PAIRING for the routed stream. The recycled atoms are booked at
    the wall temperature exactly ONCE. This term's column ``nn`` row is what
    the ``"wall"`` entry of the solver's neutral-energy routing table turns
    into a ``(3/2) k T_wall`` column-``En`` credit, so moving the routed
    particles off that row removes their credit with them -- which is
    correct, because the annulus carries no energy field and the zone-exchange
    convention re-supplies wall-temperature enthalpy when annulus gas re-enters
    the column. Booking both would plant the same energy twice.

    ``Tn_presheath_eV``: optional PER-CELL neutral temperature [eV] for the
    presheath collisionality's ``T_eff``. ``None`` (the default, and every
    historical caller) keeps the fixed cold-gas value, so this path is
    unchanged bit for bit; the kinetic DVM arm supplies the measured
    ``Tn(z)`` here when its Tn-feedback switch is on.

    ``cathode_jet``: when given (a dict with
    ``R_N``, ``R_E``, ``phi_c_V``, ``T_s_K``, and optionally
    ``energy_convention``) and the state carries ``M_n``,
    the recycle flux rebirthed at a *cathode* face is a directed jet instead
    of gas at rest: the reflected fraction ``R_N`` backscatters at the
    ``v_back`` of :func:`cathode_jet_backscatter_speed` (which is also what
    the solver's ``En`` term books, so momentum and energy describe the same
    atoms) and the implanted remainder
    ``1 - R_N`` desorbs as a directed effusive flux off the hot disc at
    ``v_eff = sqrt(pi k T_s / (2 m))`` (the per-particle directed momentum
    of a cosine-law effusive flux). The momentum rides in the SAME term
    that rebirths the particles, so the two are consistent by construction;
    the surface absorbs the difference between the incoming sonic momentum
    and the re-emitted jet momentum, as any wall does. Collector faces stay
    momentum-free (their sheath is the ~Te-scale ambipolar drop, not the
    cathode fall). The reflected atoms' kinetic energy beyond the mean-flow
    momentum is NOT booked -- neutrals carry no energy field (the standing
    M2 convention); see the campaign log for the surface-debit sensitivity
    arm that bounds the omission.

    The cathode and collector surfaces end the plasma domain, so the Bohm
    criterion applies to the face itself: plasma leaves at
    the sound speed and is neutralized on the surface.

    Applied one-sidedly to the live cell rather than as a face flux, because the
    flux array telescopes: an *interior* absorbing face would otherwise hand the
    plasma it removes to the plasma-dead plenum behind it, and kick that cell with
    sonic momentum while its density sits on the floor.

    The sonic condition is what distinguishes this from the historical volumetric
    surface term: momentum leaves at ``c_s`` directed *into* the surface, not at
    the cell's own drift ``u``, so the loss actually drives flow toward the wall
    instead of deleting plasma that was never moving there. Legacy geometry has no
    absorbing faces, so it leaves resolved boundary absorption unchanged.
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
    active = np.asarray(geometry.plasma_active, dtype=bool)
    cells = roles.size
    dN_loss = np.zeros(cells, dtype=float)
    sonic_momentum = np.zeros(cells, dtype=float)
    route_active = end_recycle_annulus_volume_cm3 is not None
    dN_routed = np.zeros(cells, dtype=float) if route_active else None
    jet_active = cathode_jet is not None and state.M_n is not None
    jet_M_n = np.zeros(cells, dtype=float) if jet_active else None
    carrier_active = jet_active and cathode_carrier_out is not None
    dN_withheld = np.zeros(cells, dtype=float) if carrier_active else None
    if jet_active:
        # Directed effusive desorption off the hot disc: per-particle
        # momentum of a cosine-law effusive flux at the surface temperature.
        v_eff = np.sqrt(
            np.pi * kb_cgs * max(float(cathode_jet["T_s_K"]), 0.0)
            / (2.0 * ion_mass_g)
        )
    for face in np.flatnonzero(absorbing):
        face = int(face)
        live = int(geometry.plasma_face_live_cell[face])
        if live < 0:
            continue
        live_is_right = live == face
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
            gas_type=gas_type,
            Tn_eV=(
                None
                if Tn_presheath_eV is None
                else float(np.asarray(Tn_presheath_eV, dtype=float)[live])
            ),
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
        if route_active and roles[live] == "collector":
            dN_routed[live] += loss
        if jet_active and roles[live] == "cathode":
            v_back = cathode_jet_backscatter_speed(
                cathode_jet, derived.Ti[live], ion_mass_g
            )
            R_N = float(cathode_jet["R_N"])
            if carrier_active:
                # The carrier owns the backscatter share: it leaves as its own
                # directed beam instead of as this cell's cold rebirth.
                dN_withheld[live] += R_N * loss
                v_mix = (1.0 - R_N) * v_eff
            else:
                v_mix = R_N * v_back + (1.0 - R_N) * v_eff
            # Directed into the plasma: opposite the face's outward normal.
            jet_M_n[live] += (
                -outward
                * ion_mass_g
                * v_mix
                * loss
                / (
                    geometry.plasma_volume_cm3[live]
                    if state.M_n_a is not None
                    else geometry.neutral_volume_cm3[live]
                )
            )
    dN_loss *= float(b_surface_loss)
    sonic_momentum *= float(b_surface_loss)
    if route_active:
        dN_routed *= float(b_surface_loss)
    if jet_active:
        jet_M_n *= float(b_surface_loss)
    if carrier_active:
        dN_withheld *= float(b_surface_loss)
        cathode_carrier_out["launch_per_s"] = dN_withheld

    plasma_loss_rate = dN_loss / geometry.plasma_volume_cm3
    # Two-zone state: the cathode disc and collector are recycle faces and
    # feed the COLUMN (the jet momentum stays chamber-mean on M_n). Under the
    # end-recycle routing the collector's share is split off to the annulus
    # instead; the column row is then the remainder, exactly zero on a cell
    # whose only absorbing face is a collector one.
    dN_column = dN_loss if not route_active else dN_loss - dN_routed
    if carrier_active:
        dN_column = dN_column - dN_withheld
    nn_a_row = (
        None
        if not route_active
        else _annulus_deposit_row(dN_routed, end_recycle_annulus_volume_cm3)
    )
    return ConservativeState1D(
        n=-plasma_loss_rate,
        nn=dN_column
        / (
            geometry.plasma_volume_cm3
            if state.nn_a is not None
            else geometry.neutral_volume_cm3
        ),
        M=-sonic_momentum / geometry.plasma_volume_cm3,
        Ee=-1.5 * ev_to_erg * derived.Te * plasma_loss_rate,
        Ei=-1.5 * ev_to_erg * derived.Ti * plasma_loss_rate,
        M_n=jet_M_n,
        nn_a=nn_a_row,
    )


def characteristic_boundary_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    alpha_isat=np.exp(-0.5),
    b_surface_loss=1.0,
    b_presheath_length=1.0,
    gas_type=None,
    cathode_jet=None,
    wave_speed="isothermal",
    energy_consistent=False,
    sheath_energy_routing=False,
    end_recycle_annulus_volume_cm3=None,
    cathode_carrier_out=None,
):
    """Return the R3.1 characteristic ghost-cell Bohm outflow at absorbing faces.

    The R3.1 boundary approach (ghost-cell Bohm outflow; audit A1/A16)
    replaces the closed-reflecting-face + one-sided
    volumetric sink of ``boundary_absorption_rhs``. At each plasma-terminating
    (absorbing) face a ghost state is set to the Bohm outflow condition

        n_se = n * presheath_alpha,  u = c_s directed into the wall,  Te, Ti

    and the committed R2 KEP/Rusanov flux (``flux.kep_rusanov_face_scalar``) is
    evaluated between the interior live cell and the ghost. Unlike the historical
    volumetric sink -- which removed outward momentum at ``c_s`` from a cell that
    was already flowing outward and so drove ``u`` further outward, booking the
    absorbing wall as a net kinetic SOURCE (the A1/A16 ``+18.5 kW``) -- the ghost
    flux DRIVES the interior toward the Bohm state and is a net energy sink.

    The flux is applied **one-sidedly to the live cell**, exactly as the old sink
    was: the shared face-flux array telescopes, so an interior absorbing face
    would otherwise hand the removed plasma to the plasma-dead plenum behind it.
    When the flag is on the advective flux carries nothing at these faces
    (``flux._apply_plasma_walls`` zeroes them), so the ghost flux -- which
    includes its own pressure term ``M_g u_g + p_g`` -- is the complete face
    condition, not an addition to the reflecting wall pressure.

    The neutral return and the cathode jet are booked exactly as in
    ``boundary_absorption_rhs``, and so is the optional
    ``end_recycle_annulus_volume_cm3`` routing of the COLLECTOR faces' recycle
    stream into ``nn_a`` (including its energy pairing -- see that function's
    docstring; cathode faces are untouched there too). ``None`` is every
    historical caller and leaves this path unchanged bit for bit.

    ``cathode_carrier_out`` arms the directed hot surface carrier and is booked
    exactly as in ``boundary_absorption_rhs`` -- the ``R_N`` share of the
    cathode recycle is withheld from the neutral rebirth row and from the jet
    momentum, and the withheld rate is written back as ``"launch_per_s"``. See
    that function's docstring; ``None`` leaves this path unchanged bit for bit.

    The sheath-``phi`` -> electrode-surface power routing and the circuit's
    read of the same ``n_se`` are the R3.2 control-surface ledger, layered on
    top of this term. Default off; golden bit-exact.
    """
    cells = geometry.cells
    zeros = np.zeros(cells, dtype=float)
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
    Vp = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    area = np.asarray(geometry.plasma_face_area_cm2, dtype=float)

    d_n = np.zeros(cells, dtype=float)
    d_M = np.zeros(cells, dtype=float)
    d_Ee = np.zeros(cells, dtype=float)
    d_Ei = np.zeros(cells, dtype=float)
    loss_abs = np.zeros(cells, dtype=float)  # particles/s removed per cell
    route_active = end_recycle_annulus_volume_cm3 is not None
    routed_abs = np.zeros(cells, dtype=float) if route_active else None

    jet_active = cathode_jet is not None and state.M_n is not None
    jet_M_n = np.zeros(cells, dtype=float) if jet_active else None
    carrier_active = jet_active and cathode_carrier_out is not None
    withheld_abs = np.zeros(cells, dtype=float) if carrier_active else None
    if jet_active:
        v_eff = np.sqrt(
            np.pi * kb_cgs * max(float(cathode_jet["T_s_K"]), 0.0)
            / (2.0 * ion_mass_g)
        )

    for face in np.flatnonzero(absorbing):
        face = int(face)
        live = int(geometry.plasma_face_live_cell[face])
        if live < 0:
            continue
        live_is_right = live == face
        # Outward normal: plasma on the high-z side of the surface flows toward
        # -z to reach it (source cathode), and +z otherwise (collector).
        outward = -1.0 if live_is_right else 1.0

        Te_l = float(derived.Te[live])
        Ti_l = float(derived.Ti[live])
        cs = float(ion_sound_speed(Te_l, mu))

        # Shared mesh-independent sheath-edge sampling (presheath_alpha): the
        # SAME factor the circuit reads in R3.2 (via electrode_sheath_alpha).
        alpha_eff = electrode_sheath_alpha(
            nn=state.nn[live],
            Te=Te_l,
            Ti=Ti_l,
            cell_length_cm=float(geometry.length_cm[live]),
            mu=mu,
            ion_mass_g=ion_mass_g,
            alpha_isat=alpha_isat,
            b_presheath_length=b_presheath_length,
            gas_type=gas_type,
        )

        n_se = alpha_eff * float(state.n[live])
        u_g = outward * cs
        p_g = n_se * (Te_l + Ti_l) * ev_to_erg
        ghost = {
            "n": n_se,
            "M": ion_mass_g * n_se * u_g,
            "Ee": 1.5 * n_se * Te_l * ev_to_erg,
            "Ei": 1.5 * n_se * Ti_l * ev_to_erg,
            "u": u_g,
            "p": p_g,
            "Te": Te_l,
            "Ti": Ti_l,
        }
        interior = {
            "n": float(state.n[live]),
            "M": float(state.M[live]),
            "Ee": float(state.Ee[live]),
            "Ei": float(state.Ei[live]),
            "u": float(derived.u[live]),
            "p": float(derived.p[live]),
            "Te": Te_l,
            "Ti": Ti_l,
        }
        # Assemble the +z-oriented face: the interior sits on whichever side the
        # live cell occupies, the ghost (the surface) on the other.
        if live_is_right:
            left_state, right_state, signL = ghost, interior, 1.0
        else:
            left_state, right_state, signL = interior, ghost, -1.0

        f_n, f_M, f_Ee, f_Ei = kep_rusanov_face_scalar(
            left_state,
            right_state,
            mu=mu,
            ion_mass_g=ion_mass_g,
            wave_speed=wave_speed,
            energy_consistent=energy_consistent,
        )
        # One-sided divergence on the live cell (the plenum keeps its closed
        # face and never receives this flux).
        scale = signL * area[face] / Vp[live]
        d_n[live] += scale * f_n
        d_M[live] += scale * f_M
        d_Ei[live] += scale * f_Ei
        # Electron energy row. R3.1: the ghost enthalpy flux (~3/2 Te). R3.2/A16
        # sheath-transmission routing:
        # the electron wall loss is the flux-weighted sheath value 2 Te, and the
        # sheath-fall phi is electrode energy (never the plasma thermal store).
        # At DRIVEN electrodes (cathode) the circuit owns it -- it is booked once
        # by cathode_source_terms as P_cathode_e_thermal, so the boundary adds
        # nothing here (removing the R3.1 electron double-book). At the COLLECTOR
        # (floating zero-net-current exhaust, no circuit branch) the boundary IS
        # the electron sheath: 2 Te per electron at the Bohm flux (electron flux =
        # ion flux), the missing collector electron sheath power.
        if sheath_energy_routing:
            if roles[live] == "collector":
                d_Ee[live] += 2.0 * Te_l * ev_to_erg * (scale * f_n)
            # cathode / other driven electrode: electron energy owned by circuit.
        else:
            d_Ee[live] += scale * f_Ee

        # Particles/s leaving through this face (density sink-rate x cell volume).
        cell_loss = -scale * f_n * Vp[live]
        loss_abs[live] += cell_loss
        if route_active and roles[live] == "collector":
            routed_abs[live] += cell_loss
        if jet_active and roles[live] == "cathode":
            v_back = cathode_jet_backscatter_speed(
                cathode_jet, Ti_l, ion_mass_g
            )
            R_N = float(cathode_jet["R_N"])
            if carrier_active:
                # The carrier owns the backscatter share (see
                # boundary_absorption_rhs's ``cathode_carrier_out``).
                withheld_abs[live] += R_N * cell_loss
                v_mix = (1.0 - R_N) * v_eff
            else:
                v_mix = R_N * v_back + (1.0 - R_N) * v_eff
            jet_M_n[live] += (
                -outward
                * ion_mass_g
                * v_mix
                * cell_loss
                / (
                    geometry.plasma_volume_cm3[live]
                    if state.M_n_a is not None
                    else geometry.neutral_volume_cm3[live]
                )
            )

    scale_b = float(b_surface_loss)
    d_n *= scale_b
    d_M *= scale_b
    d_Ee *= scale_b
    d_Ei *= scale_b
    loss_abs *= scale_b
    if route_active:
        routed_abs *= scale_b
    if jet_active:
        jet_M_n *= scale_b
    if carrier_active:
        withheld_abs *= scale_b
        cathode_carrier_out["launch_per_s"] = withheld_abs

    # Neutral return: the absorbed plasma flux is rebirthed as neutrals on the
    # column (two-zone) or chamber-mean volume, exactly as boundary_absorption
    # -- and, under the end-recycle routing, the collector faces' share goes to
    # the annulus instead, leaving the column row exactly zero there.
    column_abs = loss_abs if not route_active else loss_abs - routed_abs
    if carrier_active:
        column_abs = column_abs - withheld_abs
    nn_return = column_abs / (
        geometry.plasma_volume_cm3
        if state.nn_a is not None
        else geometry.neutral_volume_cm3
    )
    return ConservativeState1D(
        n=d_n,
        nn=nn_return,
        M=d_M,
        Ee=d_Ee,
        Ei=d_Ei,
        M_n=jet_M_n,
        nn_a=(
            None
            if not route_active
            else _annulus_deposit_row(
                routed_abs, end_recycle_annulus_volume_cm3
            )
        ),
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
    anode_jet=None,
):
    """Return the plasma the anode mesh collects and neutralizes.

    ``anode_jet``: when given
    (a dict with ``R_N``, ``R_E``, ``phi_a_V``, ``energy_convention``) and the
    state carries ``M_n``, the backscattered fraction ``R_N`` of each side's
    collected flux re-emits as a directed jet AWAY from the mesh on the side it
    was collected from, at the launch speed
    :func:`anode_jet_backscatter_speed` builds from ``R_E`` under the declared
    convention -- the ions fall through the ion-attracting anode sheath
    ``phi_a`` before striking the wires. Unlike the cathode disc, the
    implanted-then-desorbed
    remainder ``1 - R_N`` re-emits from thin cylindrical wires with no net
    axial direction, so it stays momentum-free (gas at rest, as before).
    The gap-side jet points at the cathode (-z) and the column-side jet
    downstream (+z); each rides in the same term that rebirths its
    particles, so flux and momentum stay consistent per side.

    A sheath forms on every mesh wire, so ions reach it at the **Bohm flux**
    ``exp(-0.5) * n * c_s`` -- set by the sheath, not by the bulk drift. A mesh
    sitting in stagnant plasma still collects; one in fast-flowing plasma does not
    collect proportionally faster. This is why the collection cannot be written as
    the intercepted directed flux ``eta * n * u``.

    The wires present the solid fraction ``eta`` of the plasma cross-section to
    *each* side, and each face is evaluated against the plasma actually on that
    side, so a mesh separating hot gap plasma from cooler column plasma collects
    asymmetrically -- the sum is the historical ``2 * eta * I_i_a`` with each
    half
    sampled locally. Neutrals are released on the side they were collected from,
    since a wire blocks the path to the other side and the mesh throttles neutral
    flow between them.

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
    ions. ``eta = 0`` gives a transparent anode -- the legacy limit -- and
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
    jet_active = anode_jet is not None and state.M_n is not None
    jet_M_n = np.zeros(geometry.cells, dtype=float) if jet_active else None
    for face in anode_faces:
        for cell in (int(face) - 1, int(face)):
            loss = _cell_surface_particle_loss(
                n=state.n[cell],
                Te=derived.Te[cell],
                mu=mu,
                area_cm2=float(eta) * geometry.plasma_area_cm2[cell],
                alpha_isat=alpha_isat,
            )
            dN_loss[cell] += loss
            if jet_active:
                # Away from the mesh, on the side the ion was collected
                # from: -z for the low-z flanking cell, +z for the high-z.
                direction = -1.0 if cell == int(face) - 1 else 1.0
                v_back = anode_jet_backscatter_speed(
                    anode_jet, derived.Ti[cell], ion_mass_g
                )
                jet_volume = geometry.neutral_volume_cm3[cell]
                if state.M_n_a is not None:
                    jet_volume = max(
                        geometry.neutral_volume_cm3[cell]
                        - geometry.plasma_volume_cm3[cell],
                        1e-300,
                    )
                jet_M_n[cell] += (
                    direction
                    * float(anode_jet["R_N"])
                    * ion_mass_g
                    * v_back
                    * loss
                    / jet_volume
                )
    dN_loss *= float(b_anode_collection)
    if jet_active:
        jet_M_n *= float(b_anode_collection)

    plasma_loss_rate = dN_loss / geometry.plasma_volume_cm3
    # Two-zone state: the mesh feeds the ANNULUS, falling back to the column in
    # annulus-free cells; the jet momentum stays chamber-mean on M_n.
    if state.nn_a is not None:
        V_col = np.asarray(geometry.plasma_volume_cm3, dtype=float)
        V_ann = np.maximum(
            np.asarray(geometry.neutral_volume_cm3, dtype=float) - V_col, 0.0
        )
        into_annulus = V_ann > 0.0
        nn_gain = np.where(
            into_annulus, 0.0, dN_loss / np.maximum(V_col, 1e-300)
        )
        nn_a_gain = np.where(
            into_annulus, dN_loss / np.maximum(V_ann, 1e-300), 0.0
        )
    else:
        nn_gain = dN_loss / geometry.neutral_volume_cm3
        nn_a_gain = None
    return ConservativeState1D(
        n=-plasma_loss_rate,
        nn=nn_gain,
        M=-ion_mass_g * derived.u * plasma_loss_rate,
        Ee=-1.5 * ev_to_erg * derived.Te * plasma_loss_rate,
        Ei=-1.5 * ev_to_erg * derived.Ti * plasma_loss_rate,
        M_n=None if state.M_n_a is not None else jet_M_n,
        nn_a=nn_a_gain,
        M_n_a=jet_M_n if state.M_n_a is not None else None,
    )


def _cell_surface_particle_loss(n, Te, mu, area_cm2, alpha_isat):
    return float(alpha_isat) * n * ion_sound_speed(Te, mu) * area_cm2


def ion_neutral_collision_frequency(
    nn,
    Ti,
    gas_type=None,
    Tn_eV=0.025851,
):
    """Return the ion-neutral momentum-transfer collision frequency [s^-1].

    The DEFINITIVE momentum-transfer rate -- the same Phelps He+/He isotropic
    + backscatter cross section the ``ion_neutral_moment_closure`` operator
    uses, ``nu_in = nn * (k_b + 1/2 k_iso)(T_eff)`` with
    ``T_eff = (Ti + Tn)/2`` (A8 single cold-gas ``Tn`` = ``Tn_eV``, 300 K by
    default). This ties the R3.1 presheath sampling to the same collision
    physics as the drag. He-only; ``gas_type`` is required and the He gate
    lives in ``phelps_momentum_transfer_rate_cm3_s``.

    NB the presheath ``Tn`` is taken as the fixed A8 cold-gas value (Tn_eV);
    callers do not thread the config ``Tn_K`` because it is a fixed constant,
    not a tuned knob (thread it here if that ever changes).
    """
    if gas_type is None:
        raise ValueError(
            "the ion-neutral momentum-transfer rate requires gas_type"
        )
    T_eff = 0.5 * (np.asarray(Ti, dtype=float) + float(Tn_eV))
    return np.asarray(nn, dtype=float) * phelps_momentum_transfer_rate_cm3_s(
        T_eff, gas_type=gas_type
    )


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
        gas_type=gas_type,
    )


def ion_neutral_slip_factor(
    n,
    Ti,
    ion_mass_g,
    Rm_cm,
    Tn_eV=0.1,
    b_slip_entrainment=1.0,
    gas_type=None,
):
    """Return the local drag slip factor ``s = 1 - u_n/u_i = 1/(1 + E)``.

    The model has no neutral momentum field, so the drag term needs a closure
    for the neutral flow it drags against. A constant ``b_ion_neutral_drag``
    asserts a fixed slip everywhere; this closure instead balances, per cell,
    the rate at which ions entrain a neutral (``nu_ni = n_i * sigma_in * v_ti``,
    the same relative-speed integral as ``ion_neutral_collision_frequency`` with
    the densities swapped) against the free-molecular rate at which the neutral
    carries that momentum to the wall (``1/tau_wall = vbar_n / Rm``). The
    steady balance gives ``u_n/u_i = E/(1 + E)`` with ``E = nu_ni * tau_wall``,
    so the drag on the ions scales by ``s = 1/(1 + E)`` -- full drag in
    rarefied plasma (``E -> 0``), vanishing as the neutrals entrain
    (``E -> inf``). ``b_slip_entrainment`` scales ``E`` and absorbs the O(1)
    geometric factors this balance ignores.
    """
    nu_ni = ion_neutral_collision_frequency(
        nn=n,
        Ti=Ti,
        gas_type=gas_type,
    )
    vbar_n = np.sqrt(
        8.0 * float(Tn_eV) * ev_to_erg / (np.pi * ion_mass_g)
    )
    tau_wall = np.asarray(Rm_cm, dtype=float) / vbar_n
    entrainment = float(b_slip_entrainment) * nu_ni * tau_wall
    return 1.0 / (1.0 + entrainment)


def _resolve_slip_factor(
    state,
    derived,
    ion_mass_g,
    drag_model,
    Rm_cm,
    Tn_fit,
    b_slip_entrainment,
    gas_type=None,
):
    """Return the per-cell slip factor for ``drag_model``, or 1 for constant."""
    if drag_model == "constant":
        return 1.0
    if drag_model != "slip":
        raise ValueError(
            f"ion_neutral_drag_model must be 'constant' or 'slip' "
            f"(got {drag_model!r})"
        )
    if Rm_cm is None:
        raise ValueError("drag_model='slip' requires the machine radius Rm_cm")
    return ion_neutral_slip_factor(
        n=state.n,
        Ti=derived.Ti,
        ion_mass_g=ion_mass_g,
        Rm_cm=Rm_cm,
        Tn_eV=Tn_fit,
        b_slip_entrainment=b_slip_entrainment,
        gas_type=gas_type,
    )


def neutral_wind_velocity(state, floors, ion_mass_g, geometry=None):
    """Return the neutral drift ``u_n = M_n / (m * nn)`` [cm/s], or zeros.

    ``nn`` is floored before dividing, matching ``derive_state``'s treatment
    of the plasma velocity; a state without ``M_n`` has no wind.

    When ``M_n_a`` is present, ``M_n`` is the COLUMN momentum density and
    divides by the column density directly. Otherwise ``M_n`` is a
    CHAMBER-MEAN momentum density, so on a two-zone state
    (``nn_a`` present) the divisor must be the
    chamber-mean density ``(nn V_col + nn_a V_ann) / Vm`` -- dividing by
    the column ``nn`` alone would inflate the wind wherever the annulus
    holds the gas. That path requires ``geometry`` for the zone volumes.
    """
    if state.M_n is None:
        return np.zeros_like(np.asarray(state.nn, dtype=float))
    nn = np.asarray(state.nn, dtype=float)
    if state.nn_a is not None and state.M_n_a is None:
        if geometry is None:
            raise ValueError(
                "neutral_wind_velocity on a two-zone state requires "
                "geometry for the chamber-mean density"
            )
        V_col = np.asarray(geometry.plasma_volume_cm3, dtype=float)
        Vm = np.asarray(geometry.neutral_volume_cm3, dtype=float)
        V_ann = np.maximum(Vm - V_col, 0.0)
        nn = (nn * V_col + np.asarray(state.nn_a, dtype=float) * V_ann) / Vm
    nn_safe = np.maximum(nn, floors["nn"])
    return np.asarray(state.M_n, dtype=float) / (ion_mass_g * nn_safe)


def neutral_wind_two_zone_factors(geometry, Tn_eV, ion_mass_g):
    """Return per-cell ``(column_factor, wall_rate_1_s)`` for the two-zone closure.

    The chamber-mean ``M_n`` hides the radial structure of the wind: drag acts
    only inside the plasma column (radius ``Rp``), and the annulus gas outside
    it is held slow by diffuse wall reflection. Two well-mixed zones exchanging
    free-molecularly close that structure algebraically (no new state):

    - a column neutral escapes the column in ``1/nu_p``, ``nu_p = vbar/(2 Rp)``
      (flux ``n vbar/4`` through the lateral surface ``2 pi Rp`` per column
      area ``pi Rp^2``);
    - an annulus neutral thermalizes on the outer wall in ``1/nu_w``,
      ``nu_w = vbar Rm / (2 (Rm^2 - Rp^2))`` (same flux argument on the shell).

    Quasi-steady annulus momentum balance (``f = (Rp/Rm)^2`` the column volume
    fraction) gives the annulus/column wind ratio and the column enhancement
    over the chamber mean::

        r = f nu_p / (f nu_p + (1 - f) nu_w)      u_a = r * u_c
        c = 1 / (f + (1 - f) r)                   u_c = c * u_mean

    and the wall only sees annulus gas, so the sink on the chamber-mean
    momentum is ``-W M_n`` with ``W = (1 - f) r nu_w c``. On the production
    geometry (Rp 15, Rm 50, Tn 0.1 eV He) this gives ``c ~ 3.3`` and
    ``W ~ 1.9e3 1/s`` versus the uniform closure's ``vbar/Rm ~ 4.9e3 1/s``:
    the drag input shrinks (the column gas already rides near ``u_i``) far
    more than the sink weakens, so the chamber-mean wind slows.

    Cells without a genuine annulus (``Rp >= Rm``) fall back to the uniform
    closure: ``c = 1`` and ``W = vbar/Rm``.
    """
    vbar_n = np.sqrt(8.0 * float(Tn_eV) * ev_to_erg / (np.pi * ion_mass_g))
    Rp = np.asarray(geometry.Rp_cm, dtype=float)
    Rm = np.asarray(geometry.Rm_cm, dtype=float)
    if np.any(Rp <= 0.0) or np.any(Rm <= 0.0):
        raise ValueError("two-zone closure requires positive Rp_cm and Rm_cm")
    column_factor = np.ones_like(Rm)
    wall_rate = vbar_n / Rm
    mask = Rp < Rm
    if np.any(mask):
        f = (Rp[mask] / Rm[mask]) ** 2
        nu_p = vbar_n / (2.0 * Rp[mask])
        nu_w = vbar_n * Rm[mask] / (2.0 * (Rm[mask] ** 2 - Rp[mask] ** 2))
        r = f * nu_p / (f * nu_p + (1.0 - f) * nu_w)
        c = 1.0 / (f + (1.0 - f) * r)
        column_factor[mask] = c
        wall_rate[mask] = (1.0 - f) * r * nu_w * c
    return column_factor, wall_rate


def ion_neutral_drag_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    b_ion_neutral_drag=1.0,
    cx_only=False,
    drag_model="constant",
    b_slip_entrainment=1.0,
    Rm_cm=None,
    Tn_fit=0.1,
    geometry=None,
    wind_column_factor=None,
):
    """Return the conservative ion-neutral drag momentum exchange.

    The drag force density is ``-m_i * nu(Ti) * n * (u - u_n)`` [g cm^-2 s^-2],
    a friction on the plasma flow from collisions with the neutral background.
    ``nu`` is the total momentum-transfer rate, or the charge-exchange-only rate
    when ``cx_only`` is set.

    The neutral flow ``u_n`` comes from one of three closures. With
    ``drag_model="constant"`` it is the constant ``b_ion_neutral_drag``
    (asserting ``u_n = (1 - b)*u`` everywhere); with ``drag_model="slip"`` the
    relative velocity is closed per cell by ``ion_neutral_slip_factor``, and
    ``b_ion_neutral_drag`` remains an overall multiplier. When the state
    carries the evolved neutral momentum ``M_n`` (the ``neutral_momentum``
    flag) there is no closure at all: ``u_n = M_n / (m nn)`` is the chamber-
    mean wind, the plasma sink is ``-m nu n (u - u_n)``, and the same
    momentum lands in ``M_n`` through the ``(Vp/Vm)`` volume conversion --
    conserved between species exactly like particles. That mode requires
    ``geometry`` and rejects ``drag_model="slip"`` (whose closure is this
    exchange's own steady state against the wall sink). A non-``None``
    ``wind_column_factor`` (``neutral_wind_two_zone_factors``) scales the
    chamber-mean wind up to the in-column wind the drag actually pushes
    against; the species exchange stays exactly conservative because only
    the sampled velocity changes, not the transfer bookkeeping.
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
        cx_only=cx_only,
    )
    if state.M_n is not None:
        if drag_model == "slip":
            raise ValueError(
                "neutral_momentum is mutually exclusive with "
                "ion_neutral_drag_model='slip'"
            )
        if geometry is None:
            raise ValueError(
                "drag with an evolved M_n requires geometry for the "
                "plasma/neutral volume conversion"
            )
        u_n = neutral_wind_velocity(
            state, floors=floors, ion_mass_g=ion_mass_g, geometry=geometry
        )
        if wind_column_factor is not None:
            u_n = wind_column_factor * u_n
        drag = (
            -float(b_ion_neutral_drag)
            * ion_mass_g
            * nu
            * state.n
            * (derived.u - u_n)
        )
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=drag,
            Ee=zeros.copy(),
            Ei=zeros.copy(),
            # In the kinetic-derived two-momentum mode M_n lives on the
            # plasma/column volume, so no Vp/Vm conversion is needed.
            M_n=(
                -drag
                if state.M_n_a is not None
                else -drag * geometry.volume_ratio
            ),
            M_n_a=(
                np.zeros_like(state.M_n_a)
                if state.M_n_a is not None
                else None
            ),
        )
    slip = _resolve_slip_factor(
        state=state,
        derived=derived,
        ion_mass_g=ion_mass_g,
        drag_model=drag_model,
        Rm_cm=Rm_cm,
        Tn_fit=Tn_fit,
        b_slip_entrainment=b_slip_entrainment,
        gas_type=gas_type,
    )
    drag = (
        -float(b_ion_neutral_drag) * ion_mass_g * nu * state.n * derived.u * slip
    )
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
        gas_type=gas_type,
    )
    nu_cx = ion_neutral_cx_frequency(nn=nn, Ti=Ti, gas_type=gas_type)
    return np.maximum(nu_in - nu_cx, 0.0)


def ion_neutral_frictional_heating_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    b_ion_neutral_drag=1.0,
    cx_only=False,
    drag_model="constant",
    b_slip_entrainment=1.0,
    Rm_cm=None,
    Tn_fit=0.1,
    wind_column_factor=None,
    geometry=None,
):
    """Return the conservative ion frictional-heating energy source.

    Elastic ion-neutral collisions thermalize the flow's directed energy; for
    equal masses half of the dissipated drift energy heats the ions, giving the
    ``Ei`` source ``+(1/2) m_i * nu_el(Ti) * n * (u - u_n)^2`` [erg cm^-3 s^-1].
    The charge-exchange fraction carries its energy off with the fast neutral
    and is excluded via ``nu_el = nu_in - nu_cx``; when ``cx_only`` is set there
    is no elastic fraction so this source vanishes.

    With ``drag_model="slip"`` the relative velocity is ``u * s`` from
    ``ion_neutral_slip_factor``, so the dissipated power carries ``s**2``
    (it is quadratic in the slip, where the drag force is linear). With an
    evolved ``M_n`` on the state the relative velocity is ``u - u_n``
    directly, no closure. Either way only the ion half of the dissipated
    drift energy is booked; the neutral half has no energy equation to land
    in and is dropped, as ever.
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
        cx_only=cx_only,
    )
    if state.M_n is not None:
        u_n = neutral_wind_velocity(
            state, floors=floors, ion_mass_g=ion_mass_g, geometry=geometry
        )
        if wind_column_factor is not None:
            u_n = wind_column_factor * u_n
        u_rel = derived.u - u_n
    else:
        slip = _resolve_slip_factor(
            state=state,
            derived=derived,
            ion_mass_g=ion_mass_g,
            drag_model=drag_model,
            Rm_cm=Rm_cm,
            Tn_fit=Tn_fit,
            b_slip_entrainment=b_slip_entrainment,
            gas_type=gas_type,
        )
        u_rel = derived.u * slip
    q_fric = (
        0.5
        * float(b_ion_neutral_drag)
        * ion_mass_g
        * nu_el
        * state.n
        * u_rel**2
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
    b_ion_neutral_drag=1.0,
    cx_only=False,
    b_ion_neutral_thermalization=None,
):
    """Return the conservative elastic ion-neutral thermal-equilibration source.

    Elastic collisions relax ``Ti`` toward the neutral temperature at the elastic
    rate, giving the ``Ei`` source ``+(3/2) nu_el(Ti) * n * (Tn - Ti)``
    [erg cm^-3 s^-1]. This is the elastic companion to the CX ``Q_cx`` cooling and
    is gated separately by the ``ion_neutral_thermalization`` flag; when
    ``cx_only`` is set there is no elastic fraction so this source vanishes.

    ``b_ion_neutral_thermalization`` scales this term. Its ``None`` default
    inherits ``b_ion_neutral_drag`` -- the historical coupling, kept for
    reproducibility -- but the two terms are physically distinct: the drag
    scalar stands in for velocity slip, while this term relaxes *temperature*,
    so a slip correction has no business scaling it. Set an explicit value to
    decouple them (which also frees this term from the ``ion_neutral_drag``
    flag's zeroing of the drag scalar).
    """
    scale = (
        float(b_ion_neutral_drag)
        if b_ion_neutral_thermalization is None
        else float(b_ion_neutral_thermalization)
    )
    zeros = np.zeros_like(state.n, dtype=float)
    if scale == 0.0:
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
        cx_only=cx_only,
    )
    q_eq = (
        1.5
        * scale
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


def neutral_temperature_eV(state, floors, Tn_eV):
    """Return the neutral temperature [eV] the collision terms should use.

    With the optional ``En`` field present this is the PER-CELL field value
    ``Tn = (2/3) En / (nn k)`` (``nn`` floored before dividing, as
    ``derive_state`` floors ``n``); without it, the caller's single cold-gas
    scalar ``Tn_eV`` is returned unchanged.
    """
    if state.En is None:
        return float(Tn_eV)
    nn = np.maximum(np.asarray(state.nn, dtype=float), floors["nn"])
    return (2.0 / 3.0) * np.asarray(state.En, dtype=float) / (nn * ev_to_erg)


#: The per-cell, per-save DIAGNOSTIC rows disclosing the thermal energy that
#: an ionization birth deletes: what the ``En`` sink debits per particle minus
#: what the ``Ei`` birth partner books per particle, times the birth rate. The
#: pair is conservative only when the ion is born at the neutral temperature
#: (``Ti_birth_ionization = "neutral"``), where these rows read zero to
#: roundoff; under ``"floor"``/``"local"`` they are the size of the leak. Rows
#: are [W cm^-3] on the PLASMA volume (the volume ``Ei`` lives on), signed
#: POSITIVE for energy that leaves the model. Diagnostic only: nothing in the
#: state or the RHS ledger reads them.
IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS = (
    "ionization_birth_thermal_deficit_W_cm3",
    "ionization_birth_thermal_deficit_bulk_W_cm3",
    "ionization_birth_thermal_deficit_beam_W_cm3",
    "ionization_birth_thermal_deficit_puff_W_cm3",
)

#: Which RHS term each per-site deficit row above belongs to, in the order the
#: summed row adds them.
IONIZATION_BIRTH_DEFICIT_SITES = (
    ("ionization_birth", "ionization_birth_thermal_deficit_bulk_W_cm3"),
    ("beam_ionization_birth", "ionization_birth_thermal_deficit_beam_W_cm3"),
    (
        "gas_puff_local_ionization",
        "ionization_birth_thermal_deficit_puff_W_cm3",
    ),
)


def ionization_birth_neutral_temperature_eV(state, floors, Tn_K):
    """Return the neutral temperature [eV] an ionized atom is born carrying.

    This is the SAME per-cell quantity :func:`neutral_temperature_eV` hands the
    ``En`` sink, from the same ``Tn_K`` cold-gas scalar, so an ion born at it
    receives exactly the ``(3/2) k Tn`` the neutral energy field gives up.
    Without an evolved ``En`` the state has no local neutral temperature and
    the cold-gas scalar ``Tn_K`` is returned.
    """
    return neutral_temperature_eV(
        state, floors, Tn_eV=float(Tn_K) * kb_cgs / ev_to_erg
    )


def neutral_energy_volume_ratio(state, geometry):
    """Return the ``Vp / V_En`` factor converting a plasma-volume energy source
    into the volume ``En`` lives on.

    ``En`` sits on the same volume as ``nn``: the plasma column ``Vp`` when
    ``nn_a`` splits the zones (so the factor is exactly 1), and the chamber
    volume ``Vm`` otherwise (so the factor is ``geometry.volume_ratio``).
    """
    if state.nn_a is not None:
        return np.ones_like(np.asarray(state.nn, dtype=float))
    return np.asarray(geometry.volume_ratio, dtype=float)


def ion_neutral_cx_split_rates(nn, Ti, Tn, gas_type):
    """Return ``(nu_cx, nu_el)`` [s^-1]: the CX and elastic shares of ``nu_mt``.

    The collision operator's momentum-transfer frequency is
    ``nu_mt = nn (k_b + 0.5 k_iso)(T_eff)``, and the two summands ARE the two
    physical channels: ``k_b`` is the resonant charge-exchange (backscatter)
    rate coefficient and ``0.5 k_iso`` the polarization-elastic one, both
    already carrying the equal-mass lab-frame factor. The split is therefore
    exact and introduces no constant that was not already in ``nu_mt``:

        nu_cx = nn k_b(T_eff)          nu_el = nu_mt - nu_cx = nn 0.5 k_iso(T_eff)

    ``nu_cx`` doubles as the CX EVENT rate per ion, which is what makes it the
    cold->hot population-swap rate: the equal-mass ``mu/m_i = 1/2`` factor that
    turns ``2 Qb`` into ``k_b`` is exactly the factor that turns the
    momentum-transfer moment back into an event count.

    ``nu_el`` is floored at zero and the floor RAISES if it ever binds. It
    cannot: ``k_iso`` is a positive cross-section moment, so the difference is
    positive by construction. A bind would mean the two rate tables had stopped
    being the two halves of the same sum, which is a broken model rather than a
    small negative number to clip away.
    """
    T_eff = 0.5 * (np.asarray(Ti, dtype=float) + np.asarray(Tn, dtype=float))
    nn = np.asarray(nn, dtype=float)
    k_cx = phelps_cx_rate_cm3_s(T_eff, gas_type=gas_type)
    k_mt = phelps_momentum_transfer_rate_cm3_s(T_eff, gas_type=gas_type)
    elastic = k_mt - k_cx
    if np.any(elastic < 0.0):
        raise ValueError(
            "the elastic share of the ion-neutral momentum-transfer rate went "
            f"negative (worst k_mt - k_cx = {float(np.min(elastic)):.6e} "
            "cm^3/s): k_cx and k_mt are no longer the backscatter rate and its "
            "sum with the isotropic-elastic half, so the CX/elastic split has "
            "lost its meaning"
        )
    return nn * k_cx, nn * elastic


def neutral_energy_transfer_row(nn_row, Tn_eV_local, birth_energy_erg=None):
    """Return the ``En`` row [erg cm^-3 s^-1] accompanying a neutral-density row.

    ``En`` and ``nn`` share a volume, so an ``nn`` row of ``[cm^-3 s^-1]``
    becomes an ``En`` row by multiplying it, per cell, by the energy the
    particles it moves carry:

    - a REMOVAL carries the local per-particle energy ``(3/2) k Tn``, so a sink
      cannot change the temperature of what it leaves behind. This is what
      makes ionization, pumping, and the cold->hot swap temperature-preserving
      rather than temperature-shifting;
    - an ADDITION carries ``birth_energy_erg`` per particle, the stated
      temperature of whatever the source is (the wall for a puff or a recycled
      surface flux, the local ion temperature for a recombined ion).

    A row that adds particles without a stated birth energy raises: silently
    reusing the local energy would assert that fresh gas arrives at whatever
    temperature the cell already had.
    """
    nn_row = np.asarray(nn_row, dtype=float)
    local = 1.5 * np.asarray(Tn_eV_local, dtype=float) * ev_to_erg
    if birth_energy_erg is None:
        if np.any(nn_row > 0.0):
            raise ValueError(
                "neutral_energy_transfer_row was given a row that ADDS "
                "neutrals but no birth energy; a source must state the "
                "temperature its particles arrive at"
            )
        return nn_row * local
    birth = np.asarray(birth_energy_erg, dtype=float)
    return np.where(nn_row > 0.0, nn_row * birth, nn_row * local)


def neutral_cx_channel_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    Tn_eV,
    b_ion_neutral_drag=1.0,
    geometry=None,
    wind_column_factor=None,
):
    """Return the charge-exchange DECOUPLING correction on the cold channel.

    :func:`ion_neutral_collision_rhs` is left exactly as it was: its ion rows
    are correct for the full ``nu_mt`` (the ion really does feel both channels),
    and its pairwise identity is a property of that operator which this term
    does not disturb. What the pass-1 operator got wrong once the two neutral
    populations are recognised as decoupled is the NEUTRAL side of the CX share:
    it heated the cold gas with energy that in fact leaves it entirely.

    A resonant charge exchange is a population swap, not a collision that warms
    anything::

        ion(u_i, Ti) + cold(u_n, Tn)  ->  HOT(u_i, Ti) + ion(u_n, Tn)

    The cold gas loses one atom carrying its OWN per-particle energy and
    momentum -- so ``Tn`` is untouched by CX, which is the whole content of the
    decoupling ruling -- and the hot channel gains one atom carrying the ion's.
    This term therefore does two things, both restricted to the neutral rows:

    1. WITHDRAWS the CX share of the collision operator's cold-side booking,
       ``-(q_fric_cx - q_therm_cx)`` on ``En`` and the CX share of the momentum
       mirror on ``M_n``;
    2. BOOKS the swap itself: ``-S_cx`` on ``nn`` at the local per-particle
       energy on ``En``, and ``-m u_i S_cx`` on ``M_n`` -- the momentum the hot
       atom carries away, which is exactly what the two corrections sum to.

    The elastic share keeps the full pass-1 treatment: it is a real collision
    and it really does heat the cold gas.

    What the hot channel receives is the exact complement,
    ``S_cx (3/2 k Ti + 1/2 m u_rel^2)`` of energy and ``S_cx m u_i`` of
    momentum, so ion + cold + hot conserve both to roundoff. The frictional
    half is not thermal energy in the cold gas's sense: it is the slip kinetic
    energy the hot atom is born with, and
    :func:`~.hot_neutrals.hot_channel_rates` carries it in ``e_hot``.

    A state without ``En`` gets zeros -- the decoupling has no meaning without
    a neutral temperature to decouple.
    """
    zeros = np.zeros_like(np.asarray(state.n, dtype=float))
    if state.En is None or b_ion_neutral_drag == 0.0:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    if geometry is None:
        raise ValueError(
            "neutral_cx_channel_rhs requires geometry for the plasma/neutral "
            "volume conversion"
        )
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    Tn = neutral_temperature_eV(state, floors=floors, Tn_eV=Tn_eV)
    nu_cx, _nu_el = ion_neutral_cx_split_rates(
        nn=state.nn, Ti=derived.Ti, Tn=Tn, gas_type=gas_type
    )
    if state.M_n is not None:
        u_n = neutral_wind_velocity(
            state, floors=floors, ion_mass_g=ion_mass_g, geometry=geometry
        )
        if wind_column_factor is not None:
            u_n = wind_column_factor * u_n
    else:
        u_n = np.zeros_like(derived.u)
    u_rel = derived.u - u_n
    scale = float(b_ion_neutral_drag)
    S_cx = scale * nu_cx * np.asarray(state.n, dtype=float)
    ratio = neutral_energy_volume_ratio(state, geometry)
    # (1) the CX share of what pass-1 booked into the cold gas.
    q_fric_cx = 0.5 * ion_mass_g * S_cx * u_rel**2
    q_therm_cx = 1.5 * S_cx * (Tn - derived.Ti) * ev_to_erg
    # (2) the swap itself, at the cold gas's own per-particle energy.
    swap_energy = 1.5 * Tn * ev_to_erg * S_cx
    dEn = (-(q_fric_cx - q_therm_cx) - swap_energy) * ratio
    dM_n = None
    if state.M_n is not None:
        momentum_ratio = (
            np.ones_like(ratio)
            if state.M_n_a is not None
            else np.asarray(geometry.volume_ratio, dtype=float)
        )
        # The two corrections collapse to the hot atom's own momentum:
        # -(m u_n S_cx) - (+m S_cx u_rel) == -m u_i S_cx.
        dM_n = -ion_mass_g * S_cx * derived.u * momentum_ratio
    return ConservativeState1D(
        n=zeros,
        nn=-S_cx * ratio,
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=zeros.copy(),
        M_n=dM_n,
        nn_a=None if state.nn_a is None else zeros.copy(),
        M_n_a=None if state.M_n_a is None else zeros.copy(),
        En=dEn,
    )


def ion_neutral_collision_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    Tn_eV,
    b_ion_neutral_drag=1.0,
    geometry=None,
    wind_column_factor=None,
):
    """Return the R4.3 moment-closed reduced ion-neutral collision operator.

    Replaces the drag + frictional-heating + elastic-thermalization + CX-cooling
    quartet with ONE equal-mass (He⁺/He) Braginskii momentum-transfer operator
    built from the Phelps isotropic + backscatter rate coefficients (audit A7).
    With the momentum-transfer frequency

        nu_mt = nn * (k_b(T_eff) + 0.5*k_iso(T_eff)),   T_eff = (Ti + Tn)/2

    where ``k_b = <Qb v_rel>`` is the charge-exchange (backscatter) rate and
    ``k_iso = <Qi v_rel>`` the isotropic-elastic rate, the single frequency governs
    momentum, frictional heating, AND thermal equilibration (both channels reduce
    to the same 1/2 and 3/2 coefficients when expressed through their own nu_mt):

        dM/dt  = -m n nu_mt (u - u_n)                              [momentum sink]
        dEi/dt = 0.5 m n nu_mt (u - u_n)^2 + 1.5 n nu_mt (Tn - Ti) [friction + thermal]

    The neutral receives the exact mirror momentum source (``M_n`` when the state
    carries it, through the plasma/neutral volume ratio, exactly as the legacy
    drag), so ion-neutral momentum exchange is antisymmetric. The
    CX-sized frictional-heating residual the exact swap moment requires is present
    inside the single ``0.5 m n nu_mt (u-u_n)^2`` term (it is not restricted to the
    elastic fraction, unlike the legacy ``Q_fric``).

    ``Tn_eV`` is the single cold-gas neutral temperature (audit A8; 300 K feed/wall
    for production), used consistently in both ``(Tn - Ti)`` and ``T_eff``.

    When the state carries the optional ``En`` field (the ``neutral_energy``
    flag) the neutral temperature is instead the PER-CELL field value
    ``Tn = (2/3) En / (nn k)`` -- in ``(Tn - Ti)`` and in ``T_eff`` alike --
    and the neutral side of the collisional energy is booked rather than
    dropped, through the ``Vp/V_En`` volume conversion::

        dEn/dt = [1.5 n nu_mt (Ti - Tn) + 0.5 m n nu_mt (u - u_n)^2] Vp/V_En

    the exact mirror of the ion thermal channel plus the neutral half of the
    equal-mass frictional split. The operator is then PAIRWISE conservative in
    energy: ``dEi Vp + dEn V_En == -dM u_rel Vp`` per cell, the full dissipated
    drift power, to roundoff.
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
    Tn = neutral_temperature_eV(state, floors=floors, Tn_eV=Tn_eV)
    T_eff = 0.5 * (derived.Ti + Tn)
    nu_mt = np.asarray(state.nn, dtype=float) * phelps_momentum_transfer_rate_cm3_s(
        T_eff, gas_type=gas_type
    )
    if state.M_n is not None:
        if geometry is None:
            raise ValueError(
                "ion_neutral_collision_rhs with an evolved M_n requires geometry "
                "for the plasma/neutral volume conversion"
            )
        u_n = neutral_wind_velocity(
            state, floors=floors, ion_mass_g=ion_mass_g, geometry=geometry
        )
        if wind_column_factor is not None:
            u_n = wind_column_factor * u_n
    else:
        u_n = np.zeros_like(derived.u)
    u_rel = derived.u - u_n
    scale = float(b_ion_neutral_drag)
    drag = -scale * ion_mass_g * nu_mt * state.n * u_rel
    q_fric = 0.5 * scale * ion_mass_g * nu_mt * state.n * u_rel**2
    q_therm = 1.5 * scale * nu_mt * state.n * (Tn - derived.Ti) * ev_to_erg
    if state.En is None:
        dEn = None
    else:
        if geometry is None:
            raise ValueError(
                "ion_neutral_collision_rhs with an evolved En requires "
                "geometry for the plasma/neutral volume conversion"
            )
        dEn = (q_fric - q_therm) * neutral_energy_volume_ratio(state, geometry)
    if state.M_n is not None:
        # Mirror momentum source into the neutral wind (exactly conservative).
        # In the kinetic-derived two-momentum mode M_n lives on the plasma/column
        # volume, so no Vp/Vm conversion; otherwise convert by the volume ratio.
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=drag,
            Ee=zeros.copy(),
            Ei=q_fric + q_therm,
            M_n=(
                -drag
                if state.M_n_a is not None
                else -drag * geometry.volume_ratio
            ),
            M_n_a=(
                np.zeros_like(state.M_n_a)
                if state.M_n_a is not None
                else None
            ),
            En=dEn,
        )
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=drag,
        Ee=zeros.copy(),
        Ei=q_fric + q_therm,
        En=dEn,
    )


def neutral_momentum_wall_rhs(
    state,
    floors,
    ion_mass_g,
    Rm_cm,
    Tn_fit=0.1,
    wall_rate_1_s=None,
):
    """Return the neutral-wind wall-accommodation momentum sink.

    A free-molecular neutral carries its directed momentum to the chamber
    wall and thermalizes there in ``tau_wall = Rm / vbar_n(Tn)`` -- the same
    accommodation time the ``slip`` closure balances against, because the
    local steady state of drag reception vs. this sink *is* that closure.
    The rhs is ``-M_n / tau_wall`` on the
    neutral-momentum field only; a state without ``M_n`` gets zeros.

    A non-``None`` ``wall_rate_1_s`` (``neutral_wind_two_zone_factors``)
    replaces ``1/tau_wall`` with the two-zone effective rate, in which only
    the slow annulus gas touches the wall.
    """
    zeros = np.zeros_like(state.nn, dtype=float)
    if state.M_n is None:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    if state.M_n_a is not None:
        # The kinetic-derived operator books its annulus wall loss together
        # with the exactly conservative radial transfer.
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
            M_n=zeros.copy(),
            nn_a=zeros.copy(),
            M_n_a=zeros.copy(),
        )
    if wall_rate_1_s is None:
        vbar_n = np.sqrt(8.0 * float(Tn_fit) * ev_to_erg / (np.pi * ion_mass_g))
        tau_wall = np.asarray(Rm_cm, dtype=float) / vbar_n
        dM_n = -np.asarray(state.M_n, dtype=float) / tau_wall
    else:
        dM_n = -np.asarray(state.M_n, dtype=float) * np.asarray(
            wall_rate_1_s, dtype=float
        )
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=zeros.copy(),
        M_n=dM_n,
    )


def neutral_energy_wall_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    Rm_cm,
    alpha_E,
    Tn_fit=0.1,
    wall_rate_1_s=None,
):
    """Return the neutral-energy wall-accommodation sink.

    A neutral that reaches a vessel surface leaves part of its excess thermal
    energy there and returns partly re-thermalized. The rhs is

        dEn/dt = -alpha_E * nu_wall * (En - (3/2) nn k T_wall)

    on the ``En`` field only [erg cm^-3 s^-1]; a state without ``En`` gets
    zeros. ``alpha_E`` is the thermal accommodation coefficient, in [0, 1]
    (0 = perfectly specular, no energy exchange; 1 = full accommodation in one
    wall visit). The equilibrium it relaxes toward is
    :func:`~..core.state.neutral_energy_floor`, so the sink and the state
    floor agree by construction and the term can never push ``En`` below it.

    ``nu_wall`` is the free-molecular wall-visit rate, taken from the SAME
    geometry the momentum wall sinks use so the two channels see one surface
    model. ``Tn_fit`` is the temperature whose thermal speed sets that visit
    rate, and for THIS channel it is the wall's own: the gas that reaches a
    surface and exchanges energy with it is the near-wall gas, which the v1 cut
    holds at ``T_wall``. (The momentum wall sink keeps the 0.1 eV ``Tn_fit``
    closure it was calibrated with; the two terms are allowed to differ because
    they are answering different questions, and the solver passes each its
    own.) Radially the rate is ``vbar_n(Tn_fit)/Rm``, or the two-zone
    effective rate when
    ``wall_rate_1_s`` is supplied (``neutral_wind_two_zone_factors``, in which
    only the slow annulus gas touches the wall); plus, on the two end cells,
    the outward-wind end-face flux ``max(-+u_n, 0) * A_end / V``, the same
    form ``neutral_wind_advection_rhs`` applies to the momentum an outward
    wind carries into an end wall. Areas and volumes are the ones ``nn``
    (and so ``En``) lives on: the column under ``nn_a``, the chamber
    otherwise.
    """
    zeros = np.zeros_like(np.asarray(state.nn, dtype=float))
    if state.En is None:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    if wall_rate_1_s is None:
        vbar_n = np.sqrt(8.0 * float(Tn_fit) * ev_to_erg / (np.pi * ion_mass_g))
        nu_wall = vbar_n / np.asarray(Rm_cm, dtype=float)
    else:
        nu_wall = np.asarray(wall_rate_1_s, dtype=float)
    if state.nn_a is not None:
        area = np.asarray(geometry.plasma_face_area_cm2, dtype=float)
        volume = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    else:
        area = np.asarray(geometry.neutral_face_area_cm2, dtype=float)
        volume = np.asarray(geometry.neutral_volume_cm3, dtype=float)
    u_n = neutral_wind_velocity(
        state, floors=floors, ion_mass_g=ion_mass_g, geometry=geometry
    )
    nu_wall = nu_wall + _end_face_wall_rate(u_n, area, volume)
    excess = np.asarray(state.En, dtype=float) - neutral_energy_floor(state.nn)
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=zeros.copy(),
        En=-float(alpha_E) * nu_wall * excess,
    )


def _end_face_wall_rate(u_n, face_area_cm2, volume_cm3):
    """Return the end-cell outward-wind wall-visit rate [1/s], zero elsewhere.

    ``max(-+u_n, 0) * A_end / V`` on the first and last cells: the rate at
    which a wind directed INTO an end wall delivers the cell's contents to it.
    """
    rate = np.zeros_like(np.asarray(u_n, dtype=float))
    rate[0] = (
        max(-float(u_n[0]), 0.0)
        * float(face_area_cm2[0])
        / max(float(volume_cm3[0]), 1e-300)
    )
    rate[-1] = (
        max(float(u_n[-1]), 0.0)
        * float(face_area_cm2[-1])
        / max(float(volume_cm3[-1]), 1e-300)
    )
    return rate


def neutral_wall_partition_survival(geometry, nn_a, sigma_hehe_cm2):
    """Return ``(survival, tau, mfp_cm)`` for the wall-branch momentum partition.

    The free-molecular wall branch assumes every annulus atom flies to the
    vessel wall unimpeded. At finite gas density it does not: a He atom
    crossing the annulus of radial thickness ``d = Rm - Rp`` may collide with
    another He atom first, in which case its directed momentum stays in the
    gas instead of accommodating on the surface.

    The He--He momentum-transfer mean free path is
    ``mfp = 1 / (nn_a sigma_HeHe)`` [cm].
    An atom emitted at direction cosine ``mu`` to the radial normal traverses
    the slant path ``L = d / mu``, so its collisionless survival to the wall is
    ``exp(-L/mfp) = exp(-tau/mu)`` with the optical depth ``tau = d/mfp``.
    Averaging that over the cosine-weighted exit geometry (``2 mu dmu`` on
    ``[0, 1]``) gives the standard slab transmission

        survival = 2 * integral_0^1 mu exp(-tau/mu) dmu = 2 E_3(tau)

    with ``E_3`` the third exponential integral. ``survival`` is the fraction
    of wall-branch momentum that still reaches the wall; the complement
    ``1 - survival`` is retained by the annulus gas. The free-molecular limit
    is exact: ``2 E_3(0) = 1``, so a zero optical depth reproduces the
    unpartitioned ledger bit-for-bit.

    ``sigma_HeHe`` [cm^2] is the MOMENTUM-TRANSFER cross section ``sigma_mt``
    (the ``Omega^(1,1)``-derived moment), not a total elastic one. What is
    being attenuated here is DIRECTED MOMENTUM, not particle number: a
    small-angle He--He encounter barely deflects the atom and so barely
    removes its forward momentum, whereas a quantum-total cross section counts
    that encounter at full weight. Using the total would therefore overcount
    interception and over-suppress the wall branch. A literature box for
    ``sigma_mt`` is in flight.

    KERNEL-CONDITIONALITY (disclosed). ``2 E_3(tau)`` is the SURFACE-EMITTED
    single-flight transmission -- every atom starts at one face and crosses the
    full thickness ``d``. The wall-bound momentum pool is not surface-emitted:
    it is volume-distributed through the annulus, at a mean depth of about
    ``d/2``, so the survival number is conditional on which kernel of the
    family is chosen. At ``tau = 1.29`` (the production fill point) the three
    natural members give

        surface-emitted single flight   2 E_3(tau)                    ~ 0.149
        volume-averaged single flight   (2/tau) [1/3 - E_4(tau)]      ~ 0.424
        diffusive                       1 / (1 + 3 tau / 4)           ~ 0.508

    This implementation is the FIRST, which is the most retention-biased
    member of the family -- it is the one the registered
    "transverse-radial-exit, mu-averaged" wording specifies, and it
    over-suppresses wall loss as ``tau -> infinity``. Read the re-routed
    fraction as the retention-biased end of a kernel bracket, not as a point
    value.

    Cells with no annulus (``Rp >= Rm``) get ``tau = 0`` and unit survival,
    matching the wall rate the caller already zeroes there. A zero density or
    a zero cross section likewise gives unit survival and an infinite path.
    """
    Rp = np.asarray(geometry.Rp_cm, dtype=float)
    Rm = np.asarray(geometry.Rm_cm, dtype=float)
    sigma = float(sigma_hehe_cm2)
    dens = np.maximum(np.asarray(nn_a, dtype=float), 0.0)
    thickness = np.maximum(Rm - Rp, 0.0)
    inv_mfp = dens * sigma
    with np.errstate(divide="ignore"):
        mfp = np.where(inv_mfp > 0.0, 1.0 / np.maximum(inv_mfp, 1e-300), np.inf)
    tau = thickness * inv_mfp
    survival = 2.0 * expn(3, tau)
    return survival, tau, mfp


def neutral_momentum_two_zone_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    Tn_K=300.0,
    sigma_hehe_cm2=None,
):
    """Return conservative column/annulus radial momentum exchange and wall loss.

    This operator exists only when ``M_n_a`` is present. Column momentum
    escapes radially at the fast-ion thermal crossing rate
    ``vbar(Ti)/(2 Rp)``; cold annulus momentum returns at the 300-K
    free-molecular rate. Equal and opposite volume-integrated transfers make
    the radial exchange exact. Only annulus momentum accommodates on the
    vessel wall.

    A non-``None`` ``sigma_hehe_cm2`` [cm^2] arms the wall-branch momentum
    PARTITION (the ``neutral_wall_momentum_partition`` flag). The wall
    absorption ``nu_wall M_n_a`` is then split by the He--He survival weight of
    ``neutral_wall_partition_survival``: only ``survival * nu_wall M_n_a``
    accommodates on the surface, and the complement stays on the annulus
    momentum row. The split is a partition by construction -- the retained part
    is formed as the exact FP complement of the absorbed part -- so every
    increment this adds to ``M_n_a`` is matched by an equal decrement of its
    own partner, the wall-absorption term. Particle and energy channels are
    untouched: this partitions momentum only. ``None`` (the default) leaves the
    ledger byte-identical.
    """
    zeros = np.zeros_like(state.nn, dtype=float)
    if state.M_n_a is None:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    if state.M_n is None or state.nn_a is None:
        raise ValueError("M_n_a requires both M_n and nn_a")
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    Rp = np.asarray(geometry.Rp_cm, dtype=float)
    Rm = np.asarray(geometry.Rm_cm, dtype=float)
    Vc = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    Va = np.maximum(
        np.asarray(geometry.neutral_volume_cm3, dtype=float) - Vc, 0.0
    )
    live = Va > 0.0
    vbar_i = np.sqrt(
        8.0 * np.asarray(derived.Ti, dtype=float) * ev_to_erg
        / (np.pi * ion_mass_g)
    )
    vbar_n = np.sqrt(
        8.0 * float(Tn_K) * kb_cgs / (np.pi * ion_mass_g)
    )
    nu_ca = np.where(live, vbar_i / (2.0 * Rp), 0.0)
    ann_area = np.maximum(Rm**2 - Rp**2, 1e-300)
    nu_ac = np.where(live, vbar_n * Rp / (2.0 * ann_area), 0.0)
    nu_wall = np.where(live, vbar_n * Rm / (2.0 * ann_area), 0.0)
    Mc = np.asarray(state.M_n, dtype=float)
    Ma = np.asarray(state.M_n_a, dtype=float)
    transfer = -Vc * nu_ca * Mc + Va * nu_ac * Ma
    dMc = transfer / np.maximum(Vc, 1e-300)
    wall_total = nu_wall * Ma
    if sigma_hehe_cm2 is None:
        wall_absorbed = wall_total
    else:
        survival, _tau, _mfp = neutral_wall_partition_survival(
            geometry, state.nn_a, sigma_hehe_cm2
        )
        wall_absorbed = survival * wall_total
    dMa = -transfer / np.maximum(Va, 1e-300) - wall_absorbed
    dMa = np.where(live, dMa, 0.0)
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=zeros.copy(),
        M_n=dMc,
        nn_a=zeros.copy(),
        M_n_a=dMa,
    )


def _add_optional_rows(a, b):
    """Sum two optional RHS rows, treating a missing side as zeros."""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def add_state_rhs(left, right):
    """Return the sum of two conservative RHS bundles.

    A missing optional field (``M_n``, ``nn_a``, ``M_n_a``, ``En``) on either
    side counts as zeros when the other side carries one (most RHS terms do
    not touch them); both missing keeps the historical 5-field result.
    """
    return ConservativeState1D(
        n=left.n + right.n,
        nn=left.nn + right.nn,
        M=left.M + right.M,
        Ee=left.Ee + right.Ee,
        Ei=left.Ei + right.Ei,
        M_n=_add_optional_rows(left.M_n, right.M_n),
        nn_a=_add_optional_rows(left.nn_a, right.nn_a),
        M_n_a=_add_optional_rows(left.M_n_a, right.M_n_a),
        En=_add_optional_rows(left.En, right.En),
    )
