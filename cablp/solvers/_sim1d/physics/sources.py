import math

import numpy as np

from cablp.atomic.cross_sections import (
    phelps_cx_rate_cm3_s,
    phelps_momentum_transfer_rate_cm3_s,
)
from cablp.cathode.circuit_common import sheath_lift_lambda
from cablp.constants import ev_to_erg, kb_cgs

from .flux import (
    ion_sound_speed,
    plasma_wave_speed,
    _flux_divergence,
    physical_face_scalar,
)
from ..core.state import (
    ConservativeState1D,
    derive_state,
    neutral_energy_floor,
)


def velocity_divergence(
    state, floors, ion_mass_g, geometry, active_plasma_topology=False
):
    """Return finite-volume axial velocity divergence [s^-1].

    The face velocity rule, which is what makes the ``-p_s div u`` row the
    exact energy partner of the momentum equation's net pressure force:

    * an OPEN face carries the arithmetic mean ``0.5*(u_L + u_R)``;
    * a face that is closed AND plasma-ABSORBING carries its one live cell's
      ``u``: the plasma really does leave through it, at the velocity the
      momentum flux's wall pressure ``p[live]`` acts on, so the store pays
      ``p_s u A_f`` there and the kinetic energy receives it;
    * ANY OTHER closed face carries zero. A reflecting wall does no work: the
      fluid does not move through it, so no pressure work crosses it and the
      wall reaction in the momentum flux is cancelled by the quasi-1D
      geometric source at the same face.

    The last rule is reachable only with ``active_plasma_topology``, and only
    at a closed face that is not a plasma-terminating surface.
    """
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    face_u = np.zeros(geometry.cells + 1, dtype=float)
    face_u[1:-1] = 0.5 * (derived.u[:-1] + derived.u[1:])
    if active_plasma_topology:
        absorbing = np.asarray(geometry.plasma_absorbing, dtype=bool)
        for face in np.flatnonzero(~np.asarray(geometry.plasma_open, dtype=bool)):
            live = int(geometry.plasma_face_live_cell[face])
            face_u[face] = (
                derived.u[live] if (live >= 0 and absorbing[face]) else 0.0
            )
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


def hyperbolic_energy_correction_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    wave_speed="isothermal",
):
    """Return the Rusanov numerical-dissipation deposit, into ``Ei`` alone.

    The ``(n, M)`` numerical kinetic-energy dissipation the Rusanov face flux
    removes from the momentum equation is measured each evaluation and returned
    to the ion internal energy as

        ``Q_diss,i = -[u_i dM_diss,i - 0.5 m_i u_i^2 dn_diss,i]``,

    the exact partner of what the (n, M) dissipation took from ``K``. It is a
    NUMERICAL channel, not pressure work and not a physical viscosity, so it is
    booked as its own ledger row (``hyperbolic_dissipation_heating``) rather
    than folded into any physical term. All of it goes to the ions: the
    dissipation acts on ion momentum, and the electrons carry pressure but no
    inertia. Per cell it is a flux divergence contracted with the local
    velocity and is NOT sign-definite; the dissipation is non-negative in the
    volume-weighted total.

    It rides the SAME ``a_max``, transmission and closed-face zeroing as the
    flux whose dissipation it returns, and it is not extended to the ghost
    faces the boundary operator owns.

    **Pressure work is not here.** The ``pressure_work`` row is
    :func:`pressure_work_rhs` literally, ``-p_s (div u)_i``, and that row is
    already the exact energy partner of the momentum equation's net pressure
    force: with the face velocity of :func:`velocity_divergence` and the face
    pressure the momentum flux carries,

        ``-p_i V_i (div u)_i + u_i * (net pressure force)_i
             = -[A_f Pi_f]_{i-1/2}^{i+1/2},  Pi_f = 0.5 (p_L u_R + u_L p_R)``,

    for general states and variable area. Off-path callers never build this
    operator, so it is structurally inert when the selector is off.
    """
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    u = derived.u
    cells = geometry.cells
    n = np.asarray(state.n, dtype=float)
    M = np.asarray(state.M, dtype=float)

    cs = plasma_wave_speed(derived.Te, derived.Ti, ion_mass_g, wave_speed)
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

    zeros = np.zeros(cells, dtype=float)
    return ConservativeState1D(
        n=zeros.copy(),
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=-dK_diss,
    )


def flux_tube_geometry_rhs(state, floors, ion_mass_g, geometry):
    """Return the quasi-1D pressure force for a variable-area flux tube.

    The conservative momentum equation is

        d(A rho u)/dt + d[A(rho u^2 + p)]/dz = p dA/dz + A F.

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
    ion_mass_g,
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
        **({} if Tn_eV is None else {"Tn_eV": float(Tn_eV)}),
    )
    if nu_in <= 0.0 or not np.isfinite(nu_in):
        return np.inf
    return float(ion_sound_speed(Te, ion_mass_g) / nu_in)


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
    ion_mass_g,
    alpha_isat=np.exp(-0.5),
    b_presheath_length=1.0,
):
    """Return the mesh-independent sheath-edge factor ``n_se/n`` at one cell.

    The single source of truth for the collisional-presheath sampling:
    one mesh-independent sheath-edge density ``n_se``, SHARED by the fluid
    sink, the circuit current, and the power terms. Both the fluid
    characteristic boundary (``characteristic_boundary_rhs``) and the circuit's
    cathode current (``cathode.circuit_idriven.solve_idriven`` via the cathode
    adapter) call this, and both build the same flux ``alpha_eff n c_s`` on the
    same face area, so the ions the fluid removes at the cathode and the ions
    the circuit counts are ONE number -- up to the electrode sample smoothing,
    which is a sampling of the same formula and not a second one.
    The anode mesh is NOT sampled here: its presheath is geometric and always
    fits inside a cell, so its factor is the flat ``exp(-1/2)`` used unchanged by
    both ``anode_collection_rhs`` and ``anode_circuit_sample``.
    """
    presheath_cm = b_presheath_length * presheath_length_cm(
        nn=nn,
        Te=Te,
        Ti=Ti,
        ion_mass_g=ion_mass_g,
    )
    return presheath_alpha(
        alpha_isat=alpha_isat,
        cell_length_cm=cell_length_cm,
        presheath_cm=presheath_cm,
    )


#: Accepted values of the cathode jet's ``energy_convention``, which fixes how
#: ``R_E`` is read when the backscattered atoms' launch speed is built.
CATHODE_JET_ENERGY_CONVENTIONS = ("legacy", "total_reflected")


def cathode_jet_incident_energy_eV(phi_c_V, Te_eV):
    """Return the per-ion INCIDENT energy [eV] at the cathode face.

    THE ONE DEFINITION, read by the fluid jet's launch speed below and by
    the kinetic channel's incident-energy row, so the two arms cannot
    describe ions arriving with different energies.

    ``phi_c + Te/2``, clamped at zero: a Bohm ion enters the sheath with the
    half-``Te`` directed energy the presheath gave it and then falls through
    the cathode drop. That sum is exactly the circuit's own per-ion energy
    (``cablp.cathode.circuit_common.P_ion``), so the power the jet launches
    and the power ``P_cathode_i`` credits the surface with are one energy on
    one count.

    ``phi_c_V`` is the CLAMPED sheath drop the jet spec carries; ``Te_eV``
    the local electron temperature, scalar or per-cell.
    """
    return np.maximum(
        float(phi_c_V) + 0.5 * np.asarray(Te_eV, dtype=float), 0.0
    )


def cathode_jet_backscatter_speed(cathode_jet, Te_eV, ion_mass_g):
    """Return the cathode jet's backscatter launch speed [cm s^-1].

    THE ONE SPEC. Every consumer of the backscattered atoms' kinetic energy
    reads it here -- the directed momentum booked by
    :func:`characteristic_boundary_rhs`,
    and the ``En`` the solver's ``cathode_jet_neutral_energy`` term hands the
    neutral gas -- so the momentum and the energy can never describe atoms
    moving at two different speeds.

    ``cathode_jet`` is the jet spec dict (``R_N``, ``R_E``, ``phi_c_V``,
    ``T_s_K``, and optionally ``energy_convention``); ``Te_eV`` is the local
    electron temperature [eV], scalar or per-cell; ``ion_mass_g`` the ion mass
    [g]. The incident per-particle energy is
    :func:`cathode_jet_incident_energy_eV`, the circuit's own ``phi_c + Te/2``
    -- the SAME number the kinetic channel's incident-energy row reads.

    ``energy_convention`` fixes what ``R_E`` means, and therefore how much
    energy one backscattered atom leaves with:

    ``"legacy"`` (the default when the key is absent)
        ``R_E`` is read PER BACKSCATTERED PARTICLE:
        ``v_back = sqrt(2 R_E (phi_c + Te/2)/m)``. Only the ``R_N`` reflected
        fraction carries it, so the gas receives ``R_N R_E`` of the incident
        ion power.
    ``"total_reflected"``
        ``R_E`` is the TOTAL reflected energy fraction -- reflected energy
        over incident energy, summed over all particles, which is the
        convention :func:`~cablp.solvers._sim1d.solver.LAPDSim1D` debits the
        cathode surface by. The ``R_N`` reflected particles carry all of it,
        so each leaves with ``R_E/R_N`` of the incident energy:
        ``v_back = sqrt(2 (R_E/R_N)(phi_c + Te/2)/m)`` and the gas receives
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
        * cathode_jet_incident_energy_eV(cathode_jet["phi_c_V"], Te_eV)
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


def absorbing_face_states(
    state,
    derived,
    geometry,
    live,
    outward,
    ion_mass_g,
    alpha_isat=np.exp(-0.5),
    b_presheath_length=1.0,
):
    """Return ``(interior, ghost, alpha_eff)`` for one plasma-absorbing face.

    THE ghost builder of :func:`characteristic_boundary_rhs`, factored out so
    that a face flux and the sheath quantities booked on it are read from one
    construction rather than two views of it. ``live`` is the live plasma cell
    against the face and ``outward`` its outward normal (``+1`` when the plasma
    lies on the low-z side of the surface, ``-1`` when it lies on the high-z
    side), so the ghost's velocity always points INTO the wall.

    The two returned dicts carry the conservative and derived scalars
    (``n, M, Ee, Ei, u, p, Te, Ti``) a single-face flux reads. The GHOST is
    the Bohm outflow condition at the sheath edge -- density
    ``n_se = alpha_eff n``, velocity the ion sound speed
    :func:`~.flux.ion_sound_speed` directed outward, and the live cell's own
    ``Te`` and ``Ti`` -- and it is the state the face flux is evaluated at.
    ``interior`` is returned for the callers that report the live cell beside
    it; the flux itself reads the ghost alone. ``alpha_eff`` is the
    sheath-edge sampling factor :func:`electrode_sheath_alpha` returns for
    this cell -- returned rather than recomputed by the caller, so the flux
    this face delivers and any sheath barrier charged on it describe one
    sheath edge.
    """
    Te_l = float(derived.Te[live])
    Ti_l = float(derived.Ti[live])
    cs = float(ion_sound_speed(Te_l, ion_mass_g))

    # Shared mesh-independent sheath-edge sampling (presheath_alpha): the
    # SAME factor the circuit reads in R3.2 (via electrode_sheath_alpha).
    alpha_eff = electrode_sheath_alpha(
        nn=state.nn[live],
        Te=Te_l,
        Ti=Ti_l,
        cell_length_cm=float(geometry.length_cm[live]),
        ion_mass_g=ion_mass_g,
        alpha_isat=alpha_isat,
        b_presheath_length=b_presheath_length,
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
    return interior, ghost, alpha_eff


def characteristic_boundary_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    alpha_isat=np.exp(-0.5),
    b_surface_loss=1.0,
    b_presheath_length=1.0,
    cathode_jet=None,
    cathode_carrier_out=None,
    end_wall_sheath_climb_out=None,
):
    """Return the characteristic ghost-cell Bohm outflow at absorbing faces.

    THE plasma-terminating boundary operator: since the legacy volumetric
    absorber was retired (see commit 1fc05c9) this is the only
    discretization of the cathode/end wall surfaces, and it always runs.
    At each
    plasma-terminating (absorbing) face a ghost state is set to the Bohm
    outflow condition

        n_se = n * presheath_alpha,  u = c_s directed into the wall,  Te, Ti

    and the flux removed is the PHYSICAL flux at that sheath-edge state
    alone (``flux.physical_face_scalar`` on the ghost): the pure-upwind limit,
    with no live-cell central half and no dissipation term. A sheath sends no
    wave back into the plasma, and the density step from the live cell to
    ``n_se`` is a sub-grid presheath model rather than a discontinuity, so
    there is no Riemann problem at the surface to average across. The face
    flux DRIVES the interior toward the Bohm state and is a net energy sink,
    and its particle flux ``alpha_se n c_s`` per unit area is the SAME
    expression the cathode circuit books as its ion current.

    The flux is applied **one-sidedly to the live cell**: the shared face-flux
    array telescopes, so an interior absorbing face would otherwise hand the
    removed plasma to the plasma-dead plenum behind it. The advective flux
    carries nothing at these faces (``flux._apply_plasma_walls`` zeroes them),
    so the ghost flux -- which includes its own pressure term ``M_g u_g +
    p_g`` -- is the complete face condition, not an addition to a reflecting
    wall pressure.

    ELECTRON ENERGY ROW. The electron wall loss is the flux-weighted sheath
    value ``2 Te``, and the sheath-fall ``phi`` is electrode energy, never the
    plasma thermal store. At DRIVEN electrodes (the cathode) the circuit owns
    it -- it is booked once by ``cathode_source_terms`` as
    ``P_cathode_e_thermal`` -- so this term adds nothing there. At the
    END WALL (a floating zero-net-current exhaust with no circuit branch)
    this term IS the electron sheath: ``2 Te`` per electron at the Bohm flux
    (electron flux = ion flux).

    ``end_wall_sheath_climb_out``: when given (a dict), the END WALL's
    sheath-fall electron debit is computed and written back into it under the
    key ``"Ee"`` as a per-cell electron-energy row [erg cm^-3 s^-1], negative
    where it acts. It is the ``end_wall_sheath_full_debit`` closure's row
    and is NOT added to the returned state: the caller books it as its
    own named RHS row, so the two rows together are the sheath-edge
    ``(2 + Lambda_eff) Te`` per collected electron while this function's own
    row keeps its unconditional ``2 Te`` meaning. ``Lambda_eff = Lambda +
    ln(1/alpha)`` is the barrier those electrons climb at a surface drawing
    no net current: ``Lambda``
    (:func:`~cablp.cathode.circuit_common.sheath_lift_lambda` at this call's
    ion mass, the same lift the circuit's sheath currents ride)
    plus the presheath drop implied by the very ``alpha`` this face samples
    its Bohm flux at, so the two cannot describe different sheath edges. The
    fall is taken from the plasma electron store and handed to the ions,
    which deposit it on the floating surface -- there is no circuit branch
    here to supply it, which is what makes the end wall different from the
    driven electrodes above. CATHODE faces are untouched: the accelerated
    species there is the ion. ``None`` -- the default -- computes nothing and
    is the historical call, bit for bit.

    ``cathode_jet``: when given (a dict with ``R_N``, ``R_E``, ``phi_c_V``,
    ``T_s_K``, and optionally ``energy_convention``) and the state carries
    ``M_n``, the recycle flux rebirthed at a *cathode* face is a directed jet
    instead of gas at rest: the reflected fraction ``R_N`` backscatters at the
    ``v_back`` of :func:`cathode_jet_backscatter_speed` (which is also what the
    solver's ``En`` term books, so momentum and energy describe the same atoms)
    and the implanted remainder ``1 - R_N`` desorbs as a directed effusive flux
    off the hot disc at ``v_eff = sqrt(pi k T_s / (2 m))`` (the per-particle
    directed momentum of a cosine-law effusive flux). The momentum rides in the
    SAME term that rebirths the particles, so the two are consistent by
    construction. End wall faces stay momentum-free (their sheath is the
    ~Te-scale ambipolar drop, not the cathode fall). The reflected atoms'
    kinetic energy beyond the mean-flow momentum is NOT booked -- neutrals
    carry no energy field (the standing M2 convention).

    ``cathode_carrier_out``: when given (a dict), the directed hot surface
    carrier is ARMED and this term stops booking the backscatter share of the
    cathode recycle itself -- see
    :func:`~.jet_carrier.cathode_jet_carrier_rhs`, which spends it instead.
    Two things change on the cathode faces alone, and both are the same
    withholding: the ``R_N`` share of the recycle flux is removed from the
    neutral rebirth row (the implanted ``1 - R_N`` effusive share stays, cold
    at the surface temperature), and the jet's per-particle momentum drops from
    ``R_N v_back + (1 - R_N) v_eff`` to ``(1 - R_N) v_eff``. The withheld
    particle rate [s^-1] per cell is written back into the dict as
    ``"launch_per_s"``, so the carrier's launch and this withdrawal are ONE
    number rather than two estimates of it. ``None`` leaves both bookings
    unchanged, bit for bit. The PLASMA sink is untouched either way: the
    surface absorbs the same flux, only its re-emission changes.
    """
    cells = geometry.cells
    zeros = np.zeros(cells, dtype=float)
    absorbing = np.asarray(
        getattr(geometry, "plasma_absorbing", np.zeros(0)), dtype=bool
    )
    if not np.any(absorbing) or b_surface_loss == 0.0:
        if end_wall_sheath_climb_out is not None:
            # The write-back happens on EVERY path a dict is passed on, so
            # the caller reads one key and never has to decide what an absent
            # one meant. There is no collected flux to charge here.
            end_wall_sheath_climb_out["Ee"] = zeros.copy()
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

    jet_active = cathode_jet is not None and state.M_n is not None
    jet_M_n = np.zeros(cells, dtype=float) if jet_active else None
    carrier_active = jet_active and cathode_carrier_out is not None
    withheld_abs = np.zeros(cells, dtype=float) if carrier_active else None
    climb_active = end_wall_sheath_climb_out is not None
    climb_Ee = np.zeros(cells, dtype=float) if climb_active else None
    # The sheath lift is a property of the ion mass alone, so it is read once
    # here rather than per face -- and read from the circuit, which is where
    # the same barrier sets the sheath currents.
    lambda_lift = sheath_lift_lambda(ion_mass_g) if climb_active else None
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
        # -z to reach it (source cathode), and +z otherwise (end wall).
        outward = -1.0 if live_is_right else 1.0

        interior, ghost, alpha_eff = absorbing_face_states(
            state=state,
            derived=derived,
            geometry=geometry,
            live=live,
            outward=outward,
            ion_mass_g=ion_mass_g,
            alpha_isat=alpha_isat,
            b_presheath_length=b_presheath_length,
        )
        Te_l = interior["Te"]
        Ti_l = interior["Ti"]
        # The face is +z-oriented, so the one-sided divergence takes the
        # live cell's side: +1 where the plasma is the face's RIGHT state,
        # -1 where it is the left one.
        signL = 1.0 if live_is_right else -1.0

        # A MATERIAL surface removes the physical flux at the sheath-edge
        # state, evaluated at the ghost ALONE: the pure-upwind limit, with no
        # live-cell central half and no dissipation term. Every one of the
        # four rows this function books rides the single ``f_n`` that returns,
        # so the particle sink, the electron rows, the neutral rebirth and --
        # at the cathode -- the circuit's ion current describe one face flux.
        f_n, f_M, f_Ee, f_Ei = physical_face_scalar(ghost)
        # One-sided divergence on the live cell (the plenum keeps its closed
        # face and never receives this flux).
        scale = signL * area[face] / Vp[live]
        d_n[live] += scale * f_n
        d_M[live] += scale * f_M
        d_Ei[live] += scale * f_Ei
        # Electron energy row -- the sheath-transmission routing (A16), the
        # only routing since the pure-ghost enthalpy alternative was
        # retired; see commit 1fc05c9. See the module docstring's
        # ELECTRON ENERGY ROW.
        if roles[live] == "end_wall":
            d_Ee[live] += 2.0 * Te_l * ev_to_erg * (scale * f_n)
            if climb_active:
                # end_wall_sheath_full_debit. The fall those
                # electrons climbed, at the sheath edge THIS face sampled its
                # Bohm flux at: alpha_eff is the same factor, so the density
                # drop the flux was taken across and the drop the barrier is
                # measured across are one number.
                lambda_eff = lambda_lift - math.log(alpha_eff)
                climb_Ee[live] += lambda_eff * Te_l * ev_to_erg * (scale * f_n)
        # cathode / other driven electrode: electron energy owned by circuit.

        # Particles/s leaving through this face (density sink-rate x cell volume).
        cell_loss = -scale * f_n * Vp[live]
        loss_abs[live] += cell_loss
        if jet_active and roles[live] == "cathode":
            v_back = cathode_jet_backscatter_speed(
                cathode_jet, Te_l, ion_mass_g
            )
            R_N = float(cathode_jet["R_N"])
            if carrier_active:
                # The carrier owns the backscatter share (see this function's
                # ``cathode_carrier_out``).
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
    if climb_active:
        # Scaled with the row it rides: the climb charges the flux this
        # function actually books, so a scaled surface loss scales both.
        climb_Ee *= scale_b
        end_wall_sheath_climb_out["Ee"] = climb_Ee
    if jet_active:
        jet_M_n *= scale_b
    if carrier_active:
        withheld_abs *= scale_b
        cathode_carrier_out["launch_per_s"] = withheld_abs

    # Neutral return: the absorbed plasma flux is rebirthed as neutrals on the
    # column (two-zone) or chamber-mean volume.
    column_abs = loss_abs
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
    )


def anode_collection_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    eta,
    alpha_isat=np.exp(-0.5),
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
    ``characteristic_boundary_rhs`` uses rather than an exception to it. The factor is
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

    The energy rows are booked at the sheath-edge values of the collected
    flux. On the electron store that is ``Te/2`` per
    collected ion -- the presheath work accelerating it to the Bohm speed --
    because the collected ELECTRONS' own thermal transport is booked by the
    electrode sheath term, not twice here; on the ion store it is ``5/2 Ti``,
    the enthalpy flux (3/2 Ti internal plus the Ti of flow work) rather than
    the internal energy alone.
    """
    zeros = np.zeros(geometry.cells, dtype=float)
    anode_faces = np.asarray(
        getattr(geometry, "anode_face_indices", ()), dtype=int
    )
    if anode_faces.size == 0 or eta <= 0.0:
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
                ion_mass_g=ion_mass_g,
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
    d_Ee = -0.5 * ev_to_erg * derived.Te * plasma_loss_rate
    d_Ei = -2.5 * ev_to_erg * derived.Ti * plasma_loss_rate
    return ConservativeState1D(
        n=-plasma_loss_rate,
        nn=nn_gain,
        M=-ion_mass_g * derived.u * plasma_loss_rate,
        Ee=d_Ee,
        Ei=d_Ei,
        M_n=None if state.M_n_a is not None else jet_M_n,
        nn_a=nn_a_gain,
        M_n_a=jet_M_n if state.M_n_a is not None else None,
    )


def _cell_surface_particle_loss(n, Te, ion_mass_g, area_cm2, alpha_isat):
    return float(alpha_isat) * n * ion_sound_speed(Te, ion_mass_g) * area_cm2


def ion_neutral_collision_frequency(
    nn,
    Ti,
    Tn_eV=0.025851,
):
    """Return the ion-neutral momentum-transfer collision frequency [s^-1].

    The DEFINITIVE momentum-transfer rate -- the same Phelps He+/He isotropic
    + backscatter cross section the moment-closed ion-neutral collision
    operator uses, ``nu_in = nn * (k_b + 1/2 k_iso)(T_eff)`` with
    ``T_eff = (Ti + Tn)/2`` (A8 single cold-gas ``Tn`` = ``Tn_eV``, 300 K by
    default). This ties the R3.1 presheath sampling to the same collision
    physics as the drag.

    NB the presheath ``Tn`` is taken as the fixed A8 cold-gas value (Tn_eV);
    callers do not thread the config ``Tn_K`` because it is a fixed constant,
    not a tuned knob (thread it here if that ever changes).
    """
    T_eff = 0.5 * (np.asarray(Ti, dtype=float) + float(Tn_eV))
    return np.asarray(nn, dtype=float) * phelps_momentum_transfer_rate_cm3_s(
        T_eff
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


def parallel_momentum_sink_rhs(state, rate_s, cells):
    """Return the imposed parallel momentum sink [g cm^-2 s^-2].

    ``F = -nu_add * M`` on the cells of the boolean mask ``cells`` and
    exactly zero everywhere else. ``M`` is the parallel momentum density the
    solver evolves, so the rate acts on the parallel drift relative to zero:
    the force density is ``-nu_add * m_i n u``.

    This is a RESPONSE-MAP INSTRUMENT with no physical owner -- no collision
    process in this model supplies it -- and the configuration keys that arm
    it carry the whole statement of what that means and where it may appear
    (``core/config.parallel_momentum_sink_defaults``). Its frictional work is
    :func:`parallel_momentum_sink_heating_rhs`, which must be booked with it:
    the pair is what makes the term energy-closing.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    sink = np.where(
        np.asarray(cells, dtype=bool),
        -float(rate_s) * np.asarray(state.M, dtype=float),
        0.0,
    )
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=sink,
        Ee=zeros.copy(),
        Ei=zeros.copy(),
    )


def parallel_momentum_sink_heating_rhs(state, floors, ion_mass_g, rate_s, cells):
    """Return the imposed sink's frictional heating of the ions.

    ``Q_i = +nu_add * M * u = nu_add * m_i n u^2`` [erg cm^-3 s^-1] on the
    same cells the momentum sink acts on -- exactly the rate at which that
    sink destroys the flow's directed kinetic energy, since with ``n`` held
    fixed ``d((1/2) m_i n u^2)/dt = u dM/dt = -nu_add m_i n u^2``.

    The FULL dissipated drift energy is booked here, not half. An ion-neutral
    drag books half of it on the ions because its other half leaves with a
    neutral population that exists and has its own equations; this term has no partner species at all, so a half
    booking would be an unowned energy leak rather than a convention. The
    convention it mirrors is ``hyperbolic_dissipation_heating``, which
    likewise deposits the whole of a destroyed kinetic energy into ``Ei``.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    q = np.where(
        np.asarray(cells, dtype=bool),
        float(rate_s) * np.asarray(state.M, dtype=float) * derived.u,
        0.0,
    )
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=q,
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


def ion_neutral_cx_split_rates(nn, Ti, Tn):
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
    k_cx = phelps_cx_rate_cm3_s(T_eff)
    k_mt = phelps_momentum_transfer_rate_cm3_s(T_eff)
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
    Tn_eV,
    b_ion_neutral_drag=1.0,
    geometry=None,
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
        nn=state.nn, Ti=derived.Ti, Tn=Tn
    )
    if state.M_n is not None:
        u_n = neutral_wind_velocity(
            state, floors=floors, ion_mass_g=ion_mass_g, geometry=geometry
        )
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
    Tn_eV,
    b_ion_neutral_drag=1.0,
    geometry=None,
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

    ``nu_mt`` is formed on the COLD neutral field ``nn`` alone, so where the
    ``neutral_energy`` hot channel exists its standing population ``nn_hot``
    (:mod:`~..physics.hot_neutrals`) exerts no drag by construction, and a
    comparison of this closure's neutral density against a kinetic closure's
    must add the hot field to the cold one before the two are the same
    quantity.
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
        T_eff
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
):
    """Return the neutral-wind wall-accommodation momentum sink.

    A free-molecular neutral carries its directed momentum to the chamber
    wall and thermalizes there in ``tau_wall = Rm / vbar_n(Tn)`` -- the same
    accommodation time the ``slip`` closure balances against, because the
    local steady state of drag reception vs. this sink *is* that closure.
    The rhs is ``-M_n / tau_wall`` on the
    neutral-momentum field only; a state without ``M_n`` gets zeros.
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
    vbar_n = np.sqrt(8.0 * float(Tn_fit) * ev_to_erg / (np.pi * ion_mass_g))
    tau_wall = np.asarray(Rm_cm, dtype=float) / vbar_n
    dM_n = -np.asarray(state.M_n, dtype=float) / tau_wall
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
    own.) Radially the rate is ``vbar_n(Tn_fit)/Rm``; plus, on the two end cells,
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
    vbar_n = np.sqrt(8.0 * float(Tn_fit) * ev_to_erg / (np.pi * ion_mass_g))
    nu_wall = vbar_n / np.asarray(Rm_cm, dtype=float)
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
