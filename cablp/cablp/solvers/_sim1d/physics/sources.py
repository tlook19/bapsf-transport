import numpy as np

from cablp.funcs._cross import (
    charge_ex_react,
    phelps_cx_rate_cm3_s,
    phelps_iso_rate_cm3_s,
    phelps_momentum_transfer_rate_cm3_s,
)
from cablp.vars._cons import ev_to_erg, kb_cgs

from .flux import (
    ion_sound_speed,
    plasma_wave_speed,
    _flux_divergence,
    kep_rusanov_face_scalar,
)
from ..core.state import ConservativeState1D, derive_state


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
    the historical golden stays bit-exact.
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
    sigma_in_cm2=5.0e-15,
    sigma_in_model="constant",
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
        ion_mass_g=ion_mass_g,
        sigma_in_cm2=sigma_in_cm2,
        sigma_in_model=sigma_in_model,
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
    sigma_in_cm2=5.0e-15,
    sigma_in_model="constant",
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
        sigma_in_cm2=sigma_in_cm2,
        sigma_in_model=sigma_in_model,
        gas_type=gas_type,
    )
    return presheath_alpha(
        alpha_isat=alpha_isat,
        cell_length_cm=cell_length_cm,
        presheath_cm=presheath_cm,
    )


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
    sigma_in_model="constant",
    gas_type=None,
    cathode_jet=None,
    Tn_presheath_eV=None,
):
    """Return the plasma absorbed by the plasma-terminating surfaces.

    ``Tn_presheath_eV``: optional PER-CELL neutral temperature [eV] for the
    presheath collisionality's ``T_eff``. ``None`` (the default, and every
    historical caller) keeps the fixed cold-gas value, so this path is
    unchanged bit for bit; the kinetic DVM arm supplies the measured
    ``Tn(z)`` here when its Tn-feedback switch is on.

    ``cathode_jet``: when given (a dict with
    ``R_N``, ``R_E``, ``phi_c_V``, ``T_s_K``) and the state carries ``M_n``,
    the recycle flux rebirthed at a *cathode* face is a directed jet instead
    of gas at rest: the reflected fraction ``R_N`` backscatters at
    ``v_back = sqrt(2 R_E (phi_c + Ti)/m)`` and the implanted remainder
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
    jet_active = cathode_jet is not None and state.M_n is not None
    jet_M_n = np.zeros(cells, dtype=float) if jet_active else None
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
            sigma_in_cm2=sigma_in_cm2,
            sigma_in_model=sigma_in_model,
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
        if jet_active and roles[live] == "cathode":
            v_back = np.sqrt(
                2.0
                * float(cathode_jet["R_E"])
                * max(
                    float(cathode_jet["phi_c_V"]) + derived.Ti[live], 0.0
                )
                * ev_to_erg
                / ion_mass_g
            )
            R_N = float(cathode_jet["R_N"])
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
    if jet_active:
        jet_M_n *= float(b_surface_loss)

    plasma_loss_rate = dN_loss / geometry.plasma_volume_cm3
    # Two-zone state: the cathode disc and collector are recycle faces and
    # feed the COLUMN (the jet momentum stays chamber-mean on M_n).
    return ConservativeState1D(
        n=-plasma_loss_rate,
        nn=dN_loss
        / (
            geometry.plasma_volume_cm3
            if state.nn_a is not None
            else geometry.neutral_volume_cm3
        ),
        M=-sonic_momentum / geometry.plasma_volume_cm3,
        Ee=-1.5 * ev_to_erg * derived.Te * plasma_loss_rate,
        Ei=-1.5 * ev_to_erg * derived.Ti * plasma_loss_rate,
        M_n=jet_M_n,
    )


def characteristic_boundary_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    alpha_isat=np.exp(-0.5),
    b_surface_loss=1.0,
    sigma_in_cm2=5.0e-15,
    b_presheath_length=1.0,
    sigma_in_model="constant",
    gas_type=None,
    cathode_jet=None,
    wave_speed="isothermal",
    energy_consistent=False,
    sheath_energy_routing=False,
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
    ``boundary_absorption_rhs``. The sheath-``phi`` -> electrode-surface power
    routing and the circuit's read of the same ``n_se`` are the R3.2 control-
    surface ledger, layered on top of this term. Default off; golden bit-exact.
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

    jet_active = cathode_jet is not None and state.M_n is not None
    jet_M_n = np.zeros(cells, dtype=float) if jet_active else None
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
            sigma_in_cm2=sigma_in_cm2,
            sigma_in_model=sigma_in_model,
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
        if jet_active and roles[live] == "cathode":
            v_back = np.sqrt(
                2.0
                * float(cathode_jet["R_E"])
                * max(float(cathode_jet["phi_c_V"]) + Ti_l, 0.0)
                * ev_to_erg
                / ion_mass_g
            )
            R_N = float(cathode_jet["R_N"])
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
    if jet_active:
        jet_M_n *= scale_b

    # Neutral return: the absorbed plasma flux is rebirthed as neutrals on the
    # column (two-zone) or chamber-mean volume, exactly as boundary_absorption.
    nn_return = loss_abs / (
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
    mu,
    geometry,
    eta,
    alpha_isat=np.exp(-0.5),
    b_anode_collection=1.0,
    anode_jet=None,
):
    """Return the plasma the anode mesh collects and neutralizes.

    ``anode_jet``: when given
    (a dict with ``R_N``, ``R_E``, ``phi_a_V``) and the state carries
    ``M_n``, the backscattered fraction ``R_N`` of each side's collected
    flux re-emits as a directed jet AWAY from the mesh on the side it was
    collected from, at ``v_back = sqrt(2 R_E (phi_a + Ti)/m)`` -- the ions
    fall through the ion-attracting anode sheath ``phi_a`` before striking
    the wires. Unlike the cathode disc, the implanted-then-desorbed
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
                v_back = np.sqrt(
                    2.0
                    * float(anode_jet["R_E"])
                    * max(
                        float(anode_jet["phi_a_V"]) + derived.Ti[cell], 0.0
                    )
                    * ev_to_erg
                    / ion_mass_g
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


# Dipole polarizability of the neutral [cm^3], for the Langevin capture rate.
_NEUTRAL_POLARIZABILITY_CM3 = {"He": 2.05e-25, "H": 6.67e-25}
_QE_CGS = 4.803e-10  # electron charge [esu], matches cablp.vars._cons.qe_cgs


def langevin_rate_cm3_s(gas_type, ion_mass_g):
    """Return the Langevin (polarization capture) rate coefficient [cm^3/s].

    ``k_L = 2 * pi * e * sqrt(alpha / mu_r)`` with the neutral's dipole
    polarizability ``alpha`` and the reduced mass ``mu_r = m/2`` of the
    symmetric ion-atom pair. Velocity-independent, so it is the natural
    low-energy floor for the elastic momentum-transfer channel.
    """
    try:
        alpha = _NEUTRAL_POLARIZABILITY_CM3[gas_type]
    except KeyError:
        raise ValueError(
            f"no polarizability tabulated for gas_type {gas_type!r}"
        ) from None
    return 2.0 * np.pi * _QE_CGS * np.sqrt(alpha / (0.5 * ion_mass_g))


def ion_neutral_collision_frequency(
    nn,
    Ti,
    ion_mass_g,
    sigma_in_cm2=5.0e-15,
    sigma_in_model="constant",
    gas_type=None,
    Tn_eV=0.025851,
):
    """Return the ion-neutral momentum-transfer collision frequency [s^-1].

    ``sigma_in_model = "phelps"`` (default as of the R5 stance flip): the
    DEFINITIVE momentum-transfer rate -- the same Phelps He+/He isotropic +
    backscatter cross section the ``ion_neutral_moment_closure`` operator uses,
    ``nu_in = nn * (k_b + 1/2 k_iso)(T_eff)`` with ``T_eff = (Ti + Tn)/2`` (A8
    single cold-gas ``Tn`` = ``Tn_eV``, 300 K by default). Ties the R3.1
    presheath sampling to the same collision physics as the drag, so
    ``sigma_in_cm2`` / the legacy ``constant`` / ``cx_derived`` arms are inert
    on the production path. He-only (gated at construction).

    ``sigma_in_model = "constant"`` (legacy A/B): ``nu_in = (8/3) *
    nn * sigma_in * sqrt(Ti / (pi * m_i))`` with ``Ti`` in eV (converted to
    erg here), ``m_i`` in grams, and ``sigma_in`` in cm^2, so the
    thermal-speed factor is in cm/s and ``nu_in`` in s^-1.

    ``sigma_in_model = "cx_derived"`` (legacy A/B): for a symmetric resonant
    pair the momentum transfer is dominated by charge exchange, each event
    handing over essentially the full momentum, so ``sigma_mt ~ 2*sigma_cx``.
    The rate is built from the same CX table the energy channel uses --
    ``nu_in = nn * (2*<sigma v>_cx(Ti) + k_Langevin)``. The Langevin term is
    the velocity-independent polarization-elastic floor. Requires ``gas_type``.

    NB the presheath ``Tn`` is taken as the fixed A8 cold-gas value (Tn_eV);
    callers do not thread the config ``Tn_K`` because it is a fixed constant,
    not a tuned knob (thread it here if that ever changes).
    """
    if sigma_in_model == "phelps":
        if gas_type is None:
            raise ValueError("sigma_in_model='phelps' requires gas_type")
        T_eff = 0.5 * (np.asarray(Ti, dtype=float) + float(Tn_eV))
        return np.asarray(nn, dtype=float) * phelps_momentum_transfer_rate_cm3_s(
            T_eff, gas_type=gas_type
        )
    if sigma_in_model == "constant":
        v_thi = np.sqrt(
            np.asarray(Ti, dtype=float) * ev_to_erg / (np.pi * ion_mass_g)
        )
        return (
            (8.0 / 3.0) * np.asarray(nn, dtype=float) * float(sigma_in_cm2) * v_thi
        )
    if sigma_in_model == "cx_derived":
        if gas_type is None:
            raise ValueError("sigma_in_model='cx_derived' requires gas_type")
        return np.asarray(nn, dtype=float) * (
            2.0 * charge_ex_react(Ti, gas_type)
            + langevin_rate_cm3_s(gas_type, ion_mass_g)
        )
    raise ValueError(
        "sigma_in_model must be 'phelps', 'constant', or 'cx_derived' "
        f"(got {sigma_in_model!r})"
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
    sigma_in_cm2=5.0e-15,
    cx_only=False,
    sigma_in_model="constant",
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
        sigma_in_model=sigma_in_model,
        gas_type=gas_type,
    )


def ion_neutral_slip_factor(
    n,
    Ti,
    ion_mass_g,
    Rm_cm,
    Tn_eV=0.1,
    sigma_in_cm2=5.0e-15,
    b_slip_entrainment=1.0,
    sigma_in_model="constant",
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
        ion_mass_g=ion_mass_g,
        sigma_in_cm2=sigma_in_cm2,
        sigma_in_model=sigma_in_model,
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
    sigma_in_cm2,
    b_slip_entrainment,
    sigma_in_model="constant",
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
        sigma_in_cm2=sigma_in_cm2,
        b_slip_entrainment=b_slip_entrainment,
        sigma_in_model=sigma_in_model,
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
    sigma_in_cm2=5.0e-15,
    b_ion_neutral_drag=1.0,
    cx_only=False,
    drag_model="constant",
    b_slip_entrainment=1.0,
    Rm_cm=None,
    Tn_fit=0.1,
    sigma_in_model="constant",
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
        sigma_in_cm2=sigma_in_cm2,
        cx_only=cx_only,
        sigma_in_model=sigma_in_model,
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
        sigma_in_cm2=sigma_in_cm2,
        b_slip_entrainment=b_slip_entrainment,
        sigma_in_model=sigma_in_model,
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
    sigma_in_cm2=5.0e-15,
    cx_only=False,
    sigma_in_model="constant",
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
        sigma_in_model=sigma_in_model,
        gas_type=gas_type,
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
    drag_model="constant",
    b_slip_entrainment=1.0,
    Rm_cm=None,
    Tn_fit=0.1,
    sigma_in_model="constant",
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
        sigma_in_cm2=sigma_in_cm2,
        cx_only=cx_only,
        sigma_in_model=sigma_in_model,
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
            sigma_in_cm2=sigma_in_cm2,
            b_slip_entrainment=b_slip_entrainment,
            sigma_in_model=sigma_in_model,
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
    sigma_in_cm2=5.0e-15,
    b_ion_neutral_drag=1.0,
    cx_only=False,
    b_ion_neutral_thermalization=None,
    sigma_in_model="constant",
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
        sigma_in_cm2=sigma_in_cm2,
        cx_only=cx_only,
        sigma_in_model=sigma_in_model,
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
    drag), so ion-neutral momentum exchange is antisymmetric. The neutral has no
    energy field, so the neutral-side collisional energy is dropped as ever; the
    CX-sized frictional-heating residual the exact swap moment requires is present
    inside the single ``0.5 m n nu_mt (u-u_n)^2`` term (it is not restricted to the
    elastic fraction, unlike the legacy ``Q_fric``).

    ``Tn_eV`` is the single cold-gas neutral temperature (audit A8; 300 K feed/wall
    for production), used consistently in both ``(Tn - Ti)`` and ``T_eff``.
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
    Tn = float(Tn_eV)
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
        )
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=drag,
        Ee=zeros.copy(),
        Ei=q_fric + q_therm,
    )


def radial_recycling_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    tau_s=None,
):
    """Return the radial-loss + wall-recycling proxy source terms.

    **This is a deliberate stand-in for radial physics the 1D model lacks**,
    and this docstring is its statement of record: the axial model resolves no
    radial coordinate, so radial confinement and the wall recycling it drives
    cannot emerge -- they are imposed here through the single knob ``tau_s``.
    Plasma is lost radially at ``-n / tau_s``, the
    wall neutralizes it, and the neutral returns *locally* as cold gas. Per
    cell, with ``S = n/tau``: the plasma channel loses particles, momentum,
    and thermal energy (the wall keeps all three -- this is a radial energy
    loss channel too), and the neutral inventory gains ``S * Vp/Vm``. Total
    particle inventory is conserved exactly; the returned gas is cold, so no
    energy comes back.

    The motivation: the model's mid-column neutral burnout canyon has no
    refill channel, because the physical refill -- wall recycling of
    radially-lost plasma, a *distributed* neutral source -- is radial. This
    term is that channel through one named knob. ``tau_s`` is therefore a
    calibrated quantity, not one derived from anything else in the model, and
    a result that uses it must say so. Its honesty test: LAPD radial
    confinement is of
    order 5-25 ms, so a fitted ``tau_s`` in the low-ms range is plausible
    compensation, and anything far outside is a documented failure.

    ``tau_s = None`` or ``<= 0`` disables the term (the default; the golden
    baseline never sees it). The implicit neutral-only step deliberately
    omits this term: it runs only before plasma launch, where ``n`` sits on
    its floor and the term is ~1e-7 of the gas inventory.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    if tau_s is None or float(tau_s) <= 0.0:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    S = state.n / float(tau_s)
    volume_ratio = geometry.plasma_volume_cm3 / geometry.neutral_volume_cm3
    # Two-zone state: this term IS wall recycling, so the returned gas
    # lands in the ANNULUS (falling back to the column where none exists).
    if state.nn_a is not None:
        V_col = np.asarray(geometry.plasma_volume_cm3, dtype=float)
        V_ann = np.maximum(
            np.asarray(geometry.neutral_volume_cm3, dtype=float) - V_col, 0.0
        )
        particles = S * V_col
        into_annulus = V_ann > 0.0
        nn_gain = np.where(
            into_annulus, 0.0, particles / np.maximum(V_col, 1e-300)
        )
        nn_a_gain = np.where(
            into_annulus, particles / np.maximum(V_ann, 1e-300), 0.0
        )
    else:
        nn_gain = S * volume_ratio
        nn_a_gain = None
    return ConservativeState1D(
        n=-S,
        nn=nn_gain,
        M=-ion_mass_g * derived.u * S,
        Ee=-1.5 * ev_to_erg * derived.Te * S,
        Ei=-1.5 * ev_to_erg * derived.Ti * S,
        nn_a=nn_a_gain,
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


def neutral_momentum_two_zone_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    Tn_K=300.0,
):
    """Return conservative column/annulus radial momentum exchange and wall loss.

    This operator exists only when ``M_n_a`` is present. Column momentum
    escapes radially at the fast-ion thermal crossing rate
    ``vbar(Ti)/(2 Rp)``; cold annulus momentum returns at the 300-K
    free-molecular rate. Equal and opposite volume-integrated transfers make
    the radial exchange exact. Only annulus momentum accommodates on the
    vessel wall.
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
    dMa = -transfer / np.maximum(Va, 1e-300) - nu_wall * Ma
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

    A missing optional field (``M_n``, ``nn_a``, ``M_n_a``) on either side counts as
    zeros when the other side carries one (most RHS terms do not touch
    them); both missing keeps the historical 5-field result.
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
    )
