import math

import numpy as np

from ..core.geometry import is_plenum_cell, puff_cell_indices, pump_cell_indices
from ..core.state import ConservativeState1D
from .sources import neutral_wind_velocity
from cablp.vars._cons import kb_cgs, m_p_cgs


def neutral_thermal_speed(Tn_K, mu_neutral):
    """Return neutral thermal speed [cm/s] using the _sim3 convention."""
    if Tn_K <= 0.0:
        raise ValueError(f"Tn_K must be positive (got {Tn_K})")
    if mu_neutral <= 0.0:
        raise ValueError(f"mu_neutral must be positive (got {mu_neutral})")
    return np.sqrt(8.0 * kb_cgs * Tn_K / (np.pi * mu_neutral * m_p_cgs))


def knudsen_flow_coefficients(
    geometry,
    Tn_K,
    mu_neutral,
    clausing_scale=1.0,
):
    """Return internal-face conductances from Knudsen diffusion [cm^3/s].

    ``molecular_flow`` applies the Clausing duct formula to every face, which is
    the wrong object for cell-to-cell exchange *inside* a continuous tube: the
    implied axial diffusivity is ``D = 0.25*v_th*P(dz)*dz``, which **vanishes as
    the mesh is refined**. That is not a consistent discretization of diffusion --
    refine it and neutral transport disappears rather than converging. It only
    approaches the physical value when ``dz >> R``, so a 50 cm-radius machine
    needs cells far longer than 50 cm for the historical model to be right.

    Free-molecular transport along a long tube is Fickian with the Knudsen
    diffusivity ``D = (2/3) * v_th * R``. Discretized over the centre-to-centre
    distance this gives ``C = D * A / dz``, which is mesh-independent and, for a
    circular tube, reproduces the textbook long-tube conductance
    ``C = (2*pi/3) * v_th * R^3 / L`` exactly.

    An anode mesh is a genuinely thin aperture rather than a tube segment, so it
    keeps an orifice conductance ``0.25 * v_th * A_open`` combined in series with
    the tube on either side. The annular obstruction needs no special case: it is
    a real cell, so its own reduced area and hydraulic radius
    flow through the tube formula naturally.
    """
    if clausing_scale < 0.0:
        raise ValueError(f"clausing_scale must be non-negative (got {clausing_scale})")
    v_th_n = neutral_thermal_speed(Tn_K=Tn_K, mu_neutral=mu_neutral)
    R_face = np.asarray(geometry.neutral_face_hydraulic_radius_cm[1:-1], dtype=float)
    if np.any(R_face <= 0.0):
        raise ValueError("neutral hydraulic radii must be positive")

    # Tube segment: the unrestricted cross-section, since any aperture restriction
    # is applied separately below rather than folded into the tube area.
    cell_area = np.asarray(geometry.neutral_area_cm2, dtype=float)
    tube_area = np.minimum(cell_area[:-1], cell_area[1:])
    diffusivity = (2.0 / 3.0) * v_th_n * R_face
    coefficients = (
        float(clausing_scale)
        * diffusivity
        * tube_area
        / np.asarray(geometry.center_distance_cm, dtype=float)
    )

    # Thin apertures (the anode mesh) in series with the tube on either side.
    face_area = np.asarray(geometry.neutral_face_area_cm2, dtype=float)
    for face in np.asarray(
        getattr(geometry, "anode_face_indices", ()), dtype=int
    ):
        interior = int(face) - 1
        if not 0 <= interior < coefficients.size:
            continue
        orifice = float(clausing_scale) * 0.25 * v_th_n * face_area[int(face)]
        if orifice <= 0.0:
            coefficients[interior] = 0.0
            continue
        coefficients[interior] = 1.0 / (
            1.0 / coefficients[interior] + 1.0 / orifice
        )
    # Thin annular baffles leave the central plasma channel open but restrict
    # the full neutral field to the stored circular aperture. Their zero axial
    # thickness is a series orifice, not a narrowed finite-length tube.
    for face in np.asarray(
        getattr(geometry, "neutral_baffle_face_indices", ()), dtype=int
    ):
        interior = int(face) - 1
        if not 0 <= interior < coefficients.size:
            continue
        orifice = (
            float(clausing_scale)
            * 0.25
            * v_th_n
            * face_area[int(face)]
        )
        if orifice <= 0.0:
            coefficients[interior] = 0.0
            continue
        coefficients[interior] = 1.0 / (
            1.0 / coefficients[interior] + 1.0 / orifice
        )
    return coefficients


def neutral_exchange_coefficients(
    geometry,
    model,
    constant_coeff_cm3_s,
    Tn_K,
    mu_neutral,
    clausing_scale=1.0,
):
    """Return internal-face neutral exchange coefficients [cm^3/s]."""
    if model == "constant":
        return _as_face_coefficients(constant_coeff_cm3_s, geometry)
    if model == "knudsen":
        coefficients = knudsen_flow_coefficients(
            geometry=geometry,
            Tn_K=Tn_K,
            mu_neutral=mu_neutral,
            clausing_scale=clausing_scale,
        )
    else:
        raise ValueError(
            f"neutral_exchange_model must be 'constant' or 'knudsen' (got {model!r})"
        )
    # Escape hatch: a face whose conductance is known directly rather than
    # geometrically overrides the computed value (NaN => keep the computed one).
    prescribed = np.asarray(geometry.neutral_face_conductance_cm3_s[1:-1], dtype=float)
    return np.where(np.isnan(prescribed), coefficients, prescribed)


def neutral_zone_volumes(geometry):
    """Return per-cell ``(V_col, V_ann)`` zone volumes [cm^3].

    The column IS the plasma volume (``pi Rp^2 dz``), so the column neutral
    field lives on the same volume as the plasma and the ionization
    bookkeeping's ``Vp/V_col`` conversion is exactly unity. The annulus is
    whatever neutral volume remains; cells without one (``V_ann = 0``)
    carry an inert ``nn_a``.
    """
    V_col = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    V_ann = np.maximum(
        np.asarray(geometry.neutral_volume_cm3, dtype=float) - V_col, 0.0
    )
    return V_col, V_ann


def neutral_zone_exchange_conductance(geometry, Tn_K, mu_neutral):
    """Return the per-cell column/annulus exchange conductance [cm^3/s].

    Free-molecular exchange through the column's lateral surface: one
    symmetric conductance

        K_r = (vbar / 4) * 2 pi Rp dz

    so ``K_r (nn_a - nn)`` is the net particle flow into the column. Equal
    densities give zero net flux -- exact detailed balance, the
    free-molecular equilibrium -- and the implied rates reproduce the M5
    momentum-closure geometry (``nu_c->a = vbar/(2 Rp)``,
    ``nu_a->c = vbar Rp / (2 (Rm^2 - Rp^2))``). Cells without an annulus
    get zero (nothing to exchange with).
    """
    v_th_n = neutral_thermal_speed(Tn_K=Tn_K, mu_neutral=mu_neutral)
    Rp = np.asarray(geometry.Rp_cm, dtype=float)
    length = np.asarray(geometry.length_cm, dtype=float)
    _, V_ann = neutral_zone_volumes(geometry)
    conductance = 0.25 * v_th_n * 2.0 * np.pi * Rp * length
    return np.where(V_ann > 0.0, conductance, 0.0)


def neutral_zone_exchange_rhs(state, geometry, conductance_cm3_s):
    """Return the conservative column/annulus free-molecular exchange.

    A state without ``nn_a`` gets zeros (the term is presence-gated, like
    every optional-field term).
    """
    zeros = np.zeros(geometry.cells, dtype=float)
    if state.nn_a is None:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    V_col, V_ann = neutral_zone_volumes(geometry)
    flow = np.asarray(conductance_cm3_s, dtype=float) * (
        np.asarray(state.nn_a, dtype=float) - np.asarray(state.nn, dtype=float)
    )
    dnn = np.where(V_col > 0.0, flow / np.maximum(V_col, 1e-300), 0.0)
    dnn_a = np.where(V_ann > 0.0, -flow / np.maximum(V_ann, 1e-300), 0.0)
    return ConservativeState1D(
        n=zeros,
        nn=dnn,
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=zeros.copy(),
        nn_a=dnn_a,
    )


def _zone_face_average(values):
    values = np.asarray(values, dtype=float)
    return 0.5 * (values[:-1] + values[1:])


def two_zone_knudsen_coefficients(geometry, Tn_K, mu_neutral, clausing_scale=1.0):
    """Return per-zone internal-face Knudsen conductances [cm^3/s].

    The column channel diffuses at ``D = (2/3) v_th Rp`` through the plasma
    cross-section; the annulus channel at ``D = (2/3) v_th (Rm - Rp)`` (its
    hydraulic radius ``2A/P``) through the annulus cross-section -- the
    same mesh-independent ``C = D A / dz`` form as
    ``knudsen_flow_coefficients``, split by zone.

    The anode mesh keeps its series orifice with the open-area budget the
    single-field geometry already carries (``_anode_neutral_transparency``):
    the disc must cover the plasma channel, so the column's open area is
    ``(1 - eta_face) A_col`` with ``eta_face`` inferred from the stored
    open face area, and the annulus gets the REMAINDER of the stored open
    area -- the two zones together conserve the existing budget exactly,
    and an annulus outside a column-sized disc is free, as the geometry
    docstring promises. A prescribed face conductance
    (``neutral_face_conductance_cm3_s``) is split between the zones in
    proportion to their computed conductances.

    Faces where a zone pinches off (zero area or radius) get zero
    conductance in that zone.
    """
    if clausing_scale < 0.0:
        raise ValueError(f"clausing_scale must be non-negative (got {clausing_scale})")
    v_th_n = neutral_thermal_speed(Tn_K=Tn_K, mu_neutral=mu_neutral)
    Rp = np.asarray(geometry.Rp_cm, dtype=float)
    Rm = np.asarray(geometry.Rm_cm, dtype=float)
    area_col = np.asarray(geometry.plasma_area_cm2, dtype=float)
    area_ann = np.maximum(
        np.asarray(geometry.neutral_area_cm2, dtype=float) - area_col, 0.0
    )
    distance = np.asarray(geometry.center_distance_cm, dtype=float)

    R_col = _zone_face_average(Rp)
    R_ann = np.maximum(_zone_face_average(Rm) - R_col, 0.0)
    tube_col = np.minimum(area_col[:-1], area_col[1:])
    tube_ann = np.minimum(area_ann[:-1], area_ann[1:])
    coeff_col = (
        float(clausing_scale) * (2.0 / 3.0) * v_th_n * R_col * tube_col / distance
    )
    coeff_ann = (
        float(clausing_scale) * (2.0 / 3.0) * v_th_n * R_ann * tube_ann / distance
    )

    face_area = np.asarray(geometry.neutral_face_area_cm2, dtype=float)
    for face in np.asarray(getattr(geometry, "anode_face_indices", ()), dtype=int):
        interior = int(face) - 1
        if not 0 <= interior < coeff_col.size:
            continue
        # Chamber cross-section and column cross-section at the face; the
        # stored face area is the OPEN area (chamber area times the open
        # fraction), from which the disc's opacity is recovered.
        chamber = np.pi * _zone_face_average(Rm)[interior] ** 2
        col = np.pi * R_col[interior] ** 2
        open_total = face_area[int(face)]
        if chamber <= 0.0 or col <= 0.0:
            continue
        eta_face = np.clip((chamber - open_total) / chamber, 0.0, 1.0)
        # The disc covers the plasma channel: the column sees 1 - eta of
        # its own area, the annulus keeps whatever open area remains.
        open_col = np.clip((1.0 - eta_face) * col, 0.0, col)
        open_ann = np.clip(open_total - open_col, 0.0, None)
        for coeffs, open_area in ((coeff_col, open_col), (coeff_ann, open_ann)):
            if coeffs[interior] <= 0.0:
                continue
            orifice = float(clausing_scale) * 0.25 * v_th_n * open_area
            if orifice <= 0.0:
                coeffs[interior] = 0.0
                continue
            coeffs[interior] = 1.0 / (1.0 / coeffs[interior] + 1.0 / orifice)

    # A vessel-wall baffle blocks only the outer annulus: its clear radius is
    # at least the local plasma radius, so the column conductance is unchanged.
    # The open annulus is the ring Rp < r < R_clear and enters as a thin
    # series-orifice resistance.
    baffle_faces = np.asarray(
        getattr(geometry, "neutral_baffle_face_indices", ()), dtype=int
    )
    baffle_radii = np.asarray(
        getattr(geometry, "neutral_baffle_clear_radius_cm", ()), dtype=float
    )
    if baffle_faces.shape != baffle_radii.shape:
        raise ValueError("neutral baffle face and radius arrays must match")
    for face, clear in zip(baffle_faces, baffle_radii):
        interior = int(face) - 1
        if not 0 <= interior < coeff_ann.size:
            continue
        open_ann = np.pi * max(float(clear) ** 2 - R_col[interior] ** 2, 0.0)
        orifice = float(clausing_scale) * 0.25 * v_th_n * open_ann
        if orifice <= 0.0:
            coeff_ann[interior] = 0.0
            continue
        coeff_ann[interior] = 1.0 / (
            1.0 / coeff_ann[interior] + 1.0 / orifice
        )

    prescribed = np.asarray(
        geometry.neutral_face_conductance_cm3_s[1:-1], dtype=float
    )
    override = np.isfinite(prescribed)
    if np.any(override):
        total = coeff_col + coeff_ann
        share_col = np.where(total > 0.0, coeff_col / np.maximum(total, 1e-300), 1.0)
        coeff_col = np.where(override, prescribed * share_col, coeff_col)
        coeff_ann = np.where(override, prescribed * (1.0 - share_col), coeff_ann)
    return coeff_col, coeff_ann


def neutral_exchange_two_zone_rhs(state, geometry, column_coeff_cm3_s, annulus_coeff_cm3_s):
    """Return conservative per-zone axial Knudsen exchange.

    The same pairwise-face form as ``neutral_exchange_rhs``, run
    independently per zone on the zone's own volumes. Faces where a zone's
    conductance is zero pass nothing (a pinched-off channel).
    """
    if state.nn_a is None:
        raise ValueError(
            "neutral_exchange_two_zone_rhs requires a state carrying nn_a"
        )
    V_col, V_ann = neutral_zone_volumes(geometry)
    zeros = np.zeros(geometry.cells, dtype=float)
    dnn = zeros.copy()
    dnn_a = zeros.copy()
    for coeff, values, volumes, out in (
        (column_coeff_cm3_s, state.nn, V_col, dnn),
        (annulus_coeff_cm3_s, state.nn_a, V_ann, dnn_a),
    ):
        face_rates = np.asarray(coeff, dtype=float) * (
            np.asarray(values[:-1], dtype=float)
            - np.asarray(values[1:], dtype=float)
        )
        safe = np.maximum(volumes, 1e-300)
        out[:-1] -= np.where(volumes[:-1] > 0.0, face_rates / safe[:-1], 0.0)
        out[1:] += np.where(volumes[1:] > 0.0, face_rates / safe[1:], 0.0)
    return ConservativeState1D(
        n=zeros,
        nn=dnn,
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=zeros.copy(),
        nn_a=dnn_a,
    )


def neutral_exchange_face_rates(nn, geometry, exchange_coeff_cm3_s):
    """Return neutral inventory rates across internal faces [particles/s]."""
    coeff = _as_face_coefficients(exchange_coeff_cm3_s, geometry)
    return coeff * (
        np.asarray(nn[:-1], dtype=float) - np.asarray(nn[1:], dtype=float)
    )


def neutral_exchange_rhs(state, geometry, exchange_coeff_cm3_s):
    """Return conservative RHS for pairwise neutral exchange."""
    face_rates = neutral_exchange_face_rates(
        nn=state.nn,
        geometry=geometry,
        exchange_coeff_cm3_s=exchange_coeff_cm3_s,
    )
    dnn = np.zeros(geometry.cells, dtype=float)
    dnn[:-1] -= face_rates / geometry.neutral_volume_cm3[:-1]
    dnn[1:] += face_rates / geometry.neutral_volume_cm3[1:]
    zeros = np.zeros(geometry.cells, dtype=float)
    return ConservativeState1D(
        n=zeros.copy(),
        nn=dnn,
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=zeros.copy(),
    )


def neutral_wind_advection_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    mesh_faces=None,
    mesh_blocked_area_cm2=None,
):
    """Return conservative upwind advection of ``nn`` and ``M_n`` by the wind.

    The drag-driven neutral wind ``u_n = M_n / (m nn)`` carries gas and its
    own momentum along the axis (M3), on top of the
    diffusive Knudsen exchange that carries the thermal transport. First-order
    donor-cell upwind on the internal neutral faces: the face velocity is the
    adjacent-cell average, the donor is the upwind cell, and the face area is
    the restricting ``neutral_face_area_cm2`` (so constrictions throttle the
    wind exactly as they throttle the diffusive exchange). Interior fluxes
    cancel in the inventory exactly.

    End faces pass no particles -- the end wall re-emits impinging gas
    thermally in place (pumping is a separate, named sink) -- but the
    momentum an *outward* wind carries into an end wall accommodates there:
    a sink ``-max(+/-u_n, 0) * A_end / V_end * M_n`` on the end cells, the
    same free-molecular accommodation the radial wall term applies. A state
    without ``M_n`` gets zeros.

    ``mesh_faces`` / ``mesh_blocked_area_cm2``: the anode mesh's *open* area
    already throttles what the
    wind carries across, but the momentum the wires intercept has to land on
    the anode structure, not stay in the gas -- without this sink the gap
    recirculation set up by opposing surface jets is artificially elastic.
    For each listed face, the wind flowing INTO the mesh from either flanking
    cell accommodates on the blocked area: ``-max(+/-u_n, 0) * A_blocked / V
    * M_n``, the exact form of the end-wall sink (sign-safe because ``u_n``
    and ``M_n`` share a sign).
    """
    zeros = np.zeros(geometry.cells, dtype=float)
    if state.M_n is None:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    if state.M_n_a is not None:
        if state.nn_a is None:
            raise ValueError("M_n_a requires nn_a")
        Vc = np.asarray(geometry.plasma_volume_cm3, dtype=float)
        Va = np.maximum(
            np.asarray(geometry.neutral_volume_cm3, dtype=float) - Vc, 0.0
        )
        Mc = np.asarray(state.M_n, dtype=float)
        Ma = np.asarray(state.M_n_a, dtype=float)
        nn_c = np.asarray(state.nn, dtype=float)
        nn_a = np.asarray(state.nn_a, dtype=float)
        uc = Mc / (
            ion_mass_g * np.maximum(nn_c, floors["nn"])
        )
        ua = Ma / (
            ion_mass_g * np.maximum(nn_a, floors["nn"])
        )

        def advect_zone(density, momentum, velocity, face_area, volume):
            u_face = 0.5 * (velocity[:-1] + velocity[1:])
            donor_n = np.where(u_face > 0.0, density[:-1], density[1:])
            donor_M = np.where(u_face > 0.0, momentum[:-1], momentum[1:])
            flux_n = u_face * donor_n * face_area[1:-1]
            flux_M = u_face * donor_M * face_area[1:-1]
            dn = np.zeros_like(density)
            dM = np.zeros_like(momentum)
            live = volume > 0.0
            dn[:-1] -= np.where(
                live[:-1], flux_n / np.maximum(volume[:-1], 1e-300), 0.0
            )
            dn[1:] += np.where(
                live[1:], flux_n / np.maximum(volume[1:], 1e-300), 0.0
            )
            dM[:-1] -= np.where(
                live[:-1], flux_M / np.maximum(volume[:-1], 1e-300), 0.0
            )
            dM[1:] += np.where(
                live[1:], flux_M / np.maximum(volume[1:], 1e-300), 0.0
            )
            dM[0] -= (
                max(-velocity[0], 0.0)
                * face_area[0]
                * momentum[0]
                / max(volume[0], 1e-300)
            )
            dM[-1] -= (
                max(velocity[-1], 0.0)
                * face_area[-1]
                * momentum[-1]
                / max(volume[-1], 1e-300)
            )
            return dn, dM

        area_c = np.asarray(geometry.plasma_face_area_cm2, dtype=float)
        area_a = np.maximum(
            np.asarray(geometry.neutral_face_area_cm2, dtype=float) - area_c,
            0.0,
        )
        dnn, dMc = advect_zone(nn_c, Mc, uc, area_c, Vc)
        dnn_a, dMa = advect_zone(nn_a, Ma, ua, area_a, Va)

        # A thin annular baffle keeps the plasma aperture open, throttles the
        # annulus flux through its clear area above Rp, and diffusely
        # accommodates momentum intercepted by the blocked ring. The 300-K
        # incident flux vbar/4 makes this a sign-safe local loss.
        vbar_n = np.sqrt(
            8.0 * 300.0 * kb_cgs / (np.pi * ion_mass_g)
        )
        for face in np.asarray(
            geometry.neutral_baffle_face_indices, dtype=int
        ):
            left, right = int(face) - 1, int(face)
            full_area = min(
                np.pi * geometry.Rm_cm[left] ** 2,
                np.pi * geometry.Rm_cm[right] ** 2,
            )
            blocked = max(
                full_area - geometry.neutral_face_area_cm2[int(face)], 0.0
            )
            for cell in (left, right):
                if Va[cell] > 0.0:
                    dMa[cell] -= (
                        0.25 * vbar_n * blocked / Va[cell] * Ma[cell]
                    )
        if mesh_faces is not None:
            for face, blocked in zip(
                np.asarray(mesh_faces, dtype=int),
                np.asarray(mesh_blocked_area_cm2, dtype=float),
            ):
                left, right = int(face) - 1, int(face)
                if Va[left] > 0.0:
                    dMa[left] -= (
                        max(ua[left], 0.0)
                        * blocked
                        * Ma[left]
                        / Va[left]
                    )
                if Va[right] > 0.0:
                    dMa[right] -= (
                        max(-ua[right], 0.0)
                        * blocked
                        * Ma[right]
                        / Va[right]
                    )
        return ConservativeState1D(
            n=zeros,
            nn=dnn,
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
            M_n=dMc,
            nn_a=dnn_a,
            M_n_a=dMa,
        )
    u_n = neutral_wind_velocity(
        state, floors=floors, ion_mass_g=ion_mass_g, geometry=geometry
    )
    nn = np.asarray(state.nn, dtype=float)
    M_n = np.asarray(state.M_n, dtype=float)
    u_face = 0.5 * (u_n[:-1] + u_n[1:])
    donor_nn = np.where(u_face > 0.0, nn[:-1], nn[1:])
    donor_M_n = np.where(u_face > 0.0, M_n[:-1], M_n[1:])
    area = geometry.neutral_face_area_cm2[1:-1]
    # Two-zone state: the drag-driven wind lives in the column (the M5
    # radial argument), so it advects the COLUMN gas through the plasma
    # face area on the column volumes. M_n stays a chamber-mean field on
    # the chamber areas/volumes.
    if state.nn_a is not None:
        area_nn = geometry.plasma_face_area_cm2[1:-1]
        volume_nn = geometry.plasma_volume_cm3
    else:
        area_nn = area
        volume_nn = geometry.neutral_volume_cm3
    flux_nn = u_face * donor_nn * area_nn
    flux_M_n = u_face * donor_M_n * area
    dnn = zeros.copy()
    dM_n = np.zeros(geometry.cells, dtype=float)
    dnn[:-1] -= flux_nn / volume_nn[:-1]
    dnn[1:] += flux_nn / volume_nn[1:]
    dM_n[:-1] -= flux_M_n / geometry.neutral_volume_cm3[:-1]
    dM_n[1:] += flux_M_n / geometry.neutral_volume_cm3[1:]
    # u_n and M_n share a sign, so these sinks only ever relax M_n toward
    # zero; an inward wind at an end face contributes nothing.
    dM_n[0] -= (
        max(-u_n[0], 0.0)
        * geometry.neutral_face_area_cm2[0]
        * M_n[0]
        / geometry.neutral_volume_cm3[0]
    )
    dM_n[-1] -= (
        max(u_n[-1], 0.0)
        * geometry.neutral_face_area_cm2[-1]
        * M_n[-1]
        / geometry.neutral_volume_cm3[-1]
    )
    if mesh_faces is not None:
        for face, blocked in zip(
            np.asarray(mesh_faces, dtype=int),
            np.asarray(mesh_blocked_area_cm2, dtype=float),
        ):
            left, right = int(face) - 1, int(face)
            dM_n[left] -= (
                max(u_n[left], 0.0)
                * blocked
                * M_n[left]
                / geometry.neutral_volume_cm3[left]
            )
            dM_n[right] -= (
                max(-u_n[right], 0.0)
                * blocked
                * M_n[right]
                / geometry.neutral_volume_cm3[right]
            )
    return ConservativeState1D(
        n=zeros,
        nn=dnn,
        M=np.zeros(geometry.cells, dtype=float),
        Ee=np.zeros(geometry.cells, dtype=float),
        Ei=np.zeros(geometry.cells, dtype=float),
        M_n=dM_n,
    )


def neutral_source_sink_rhs(
    state,
    geometry,
    S_gp,
    Twin_S_gp,
    S_pump_L,
    S_pump_R,
    twin_cathode=False,
    gas_puff_enabled=True,
    pump_enabled=True,
    gas_puff_valves=2,
    pump_elbow_conductance_lps=None,
    gas_puff_profile="cell",
    gas_puff_z_cm=None,
    gas_puff_sigma_cm=50.0,
    gas_puff_throw_cm=100.0,
):
    """Return conservative RHS for neutral gas puff and pump terms.

    Both terms are anchored by ``cell_role``, not by ``[0]``/``[-1]``: the
    puff lands on its puff cell and each pump on the plenum/collector at its end.
    Legacy roles resolve to the source and end cells, reproducing today exactly.
    The puff's axial shape comes from ``gas_puff_rate_profile``.

    On a two-zone state (``nn_a`` present) the puff
    feeds the ANNULUS first -- the pipe enters at the wall -- re-normalized
    from the profile's chamber volume to the annulus volume so the total
    inflow is conserved exactly (annulus-free cells fall back to the
    column). The pumps keep their chamber-volume rate coefficient applied
    to BOTH zone densities, which reproduces the single-zone ``S * n_port``
    exactly at the well-mixed equilibrium.
    """
    dnn = np.zeros(geometry.cells, dtype=float)
    two_zone = state.nn_a is not None
    dnn_a = np.zeros(geometry.cells, dtype=float) if two_zone else None
    pump_left_index, pump_right_index = pump_cell_indices(geometry)
    if gas_puff_enabled:
        puff = gas_puff_rate_profile(
            geometry,
            S_gp,
            gas_puff_valves,
            profile=gas_puff_profile,
            z_cm=gas_puff_z_cm,
            sigma_cm=gas_puff_sigma_cm,
            throw_cm=gas_puff_throw_cm,
            end=0,
        )
        if twin_cathode:
            puff = puff + gas_puff_rate_profile(
                geometry,
                Twin_S_gp,
                gas_puff_valves,
                profile=gas_puff_profile,
                z_cm=gas_puff_z_cm,
                sigma_cm=gas_puff_sigma_cm,
                throw_cm=gas_puff_throw_cm,
                end=-1,
            )
        if two_zone:
            V_col, V_ann = neutral_zone_volumes(geometry)
            particles = puff * np.asarray(
                geometry.neutral_volume_cm3, dtype=float
            )
            into_annulus = V_ann > 0.0
            dnn_a += np.where(
                into_annulus, particles / np.maximum(V_ann, 1e-300), 0.0
            )
            dnn += np.where(
                into_annulus, 0.0, particles / np.maximum(V_col, 1e-300)
            )
        else:
            dnn += puff
    if pump_enabled:
        # The unmodeled pump elbow folds into an effective speed on the plenum
        # a collector-side pump has no elbow in front of it.
        S_left = _effective_pump_speed(
            S_pump_L,
            pump_elbow_conductance_lps if is_plenum_cell(geometry, pump_left_index)
            else None,
        )
        S_right = _effective_pump_speed(
            S_pump_R,
            pump_elbow_conductance_lps if is_plenum_cell(geometry, pump_right_index)
            else None,
        )
        rate_left = pump_rate(S_left, geometry.neutral_volume_cm3[pump_left_index])
        rate_right = pump_rate(
            S_right, geometry.neutral_volume_cm3[pump_right_index]
        )
        dnn[pump_left_index] -= rate_left * state.nn[pump_left_index]
        dnn[pump_right_index] -= rate_right * state.nn[pump_right_index]
        if two_zone:
            dnn_a[pump_left_index] -= rate_left * state.nn_a[pump_left_index]
            dnn_a[pump_right_index] -= (
                rate_right * state.nn_a[pump_right_index]
            )
    zeros = np.zeros(geometry.cells, dtype=float)
    # An evolved neutral wind (state carries M_n) leaves through the pump at
    # the same rate as the gas, so the pumped-out neutrals take their
    # momentum with them and u_n does not inflate at the pump cells. The
    # puff needs no companion: cold gas arrives with zero directed momentum.
    dM_n = None
    dM_n_a = None
    if state.M_n is not None:
        dM_n = zeros.copy()
        if pump_enabled:
            dM_n[pump_left_index] -= rate_left * state.M_n[pump_left_index]
            dM_n[pump_right_index] -= rate_right * state.M_n[pump_right_index]
    if state.M_n_a is not None:
        dM_n_a = zeros.copy()
        if pump_enabled:
            dM_n_a[pump_left_index] -= (
                rate_left * state.M_n_a[pump_left_index]
            )
            dM_n_a[pump_right_index] -= (
                rate_right * state.M_n_a[pump_right_index]
            )
    return ConservativeState1D(
        n=zeros.copy(),
        nn=dnn,
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=zeros.copy(),
        M_n=dM_n,
        nn_a=dnn_a,
        M_n_a=dM_n_a,
    )


# --- Saved per-sample gas-puff waveform fields ------------------------------
# The EFFECTIVE puff the solver actually applied at each save: the configured
# S_gp after waveform shaping (``gas_puff_mode``) AND the phase gate, not the
# nominal input_dict level. Recording it removes the need to reconstruct the
# waveform from the phase switch plus the mode's formula after the fact.
# Pure recording -- nothing here feeds an RHS row.
GAS_PUFF_DIAGNOSTIC_FIELDS = (
    # Source-end and twin-end valve rates [sccm] as applied; zero whenever the
    # phase gate is shut, and the twin entry is zero without TwinCathode.
    "S_gp_sccm",
    "Twin_S_gp_sccm",
    # Volume-integrated particle influx [s^-1] of the same instant's puff
    # profile -- the quantity whose time integral is the delivered fuel.
    "puff_particles_per_s",
)


def puff_rate(sccm, valves, chamber_vol):
    """Return gas puff source rate [cm^-3 s^-1] using _sim3 conversion."""
    if chamber_vol <= 0.0:
        raise ValueError(f"chamber_vol must be positive (got {chamber_vol})")
    return 4.477962e17 * float(sccm) * float(valves) / float(chamber_vol)


# Roles a distributed gas puff may land on: the main plasma chamber, not the
# plenum/obstruction behind the cathode, the cathode-anode gap, or the
# collector region.
_PUFF_ELIGIBLE_ROLES = frozenset({"puff", "column", "source", "domain", "end"})


def gas_puff_rate_profile(
    geometry,
    sccm,
    valves,
    profile="cell",
    z_cm=None,
    sigma_cm=50.0,
    throw_cm=100.0,
    end=0,
):
    """Return the per-cell puff source rate array [cm^-3 s^-1].

    This is the single implementation behind BOTH puff sites -- the explicit
    RHS (``neutral_source_sink_rhs``) and the implicit backward-Euler neutral
    matrix in the solver -- so the two cannot desync (the historical trap was
    two independently maintained copies of this shape).

    ``profile = "cell"`` (default) reproduces the historical behaviour
    bit-exactly: the whole flow lands in the role-tagged puff cell.

    ``profile = "cosine_pipe"`` is the physical source: a small pipe at the
    chamber wall pointing radially inward with a Lambertian (cosine) outlet.
    Its first-flight deposition along the wall is the standard cosine-lobe
    illumination ``[1 + ((z - z0)/d)^2]^-2`` with throw distance ``d``
    (``throw_cm``) of order the chord across the chamber (~2*Rm), after which
    the neutral transport model does the spreading. Centre and width both
    come from geometry, so this adds no free shape parameter.

    ``profile = "gaussian"`` is the generic tunable shape,
    ``exp(-(z - z0)^2 / (2 sigma^2))``.

    All distributed profiles weight by cell length, land only on
    main-chamber cells, and are normalized to conserve the total inflow
    exactly. ``z_cm = None`` centres on the puff cell; ``end = -1`` selects
    the twin puff cell and mirrors an explicit centre through the machine
    midpoint.
    """
    dnn = np.zeros(geometry.cells, dtype=float)
    puff_index, puff_twin_index = puff_cell_indices(geometry)
    index = puff_twin_index if end == -1 else puff_index
    if profile == "cell":
        dnn[index] = puff_rate(
            sccm, valves, geometry.neutral_volume_cm3[index]
        )
        return dnn
    if profile not in ("gaussian", "cosine_pipe"):
        raise ValueError(
            "gas_puff_profile must be 'cell', 'gaussian', or 'cosine_pipe' "
            f"(got {profile!r})"
        )

    roles = np.asarray(geometry.cell_role)
    eligible = np.asarray(
        [role in _PUFF_ELIGIBLE_ROLES for role in roles], dtype=bool
    )
    z_centers = np.asarray(geometry.z_cm, dtype=float)
    if z_cm is None:
        z0 = float(z_centers[index])
    else:
        z0 = float(z_cm)
        if end == -1:
            z_lo = float(np.min(z_centers[eligible]))
            z_hi = float(np.max(z_centers[eligible]))
            z0 = z_lo + z_hi - z0  # mirror through the chamber midpoint

    weights = np.zeros(geometry.cells, dtype=float)
    if profile == "gaussian":
        sigma = float(sigma_cm)
        if sigma <= 0.0:
            raise ValueError(f"gas_puff_sigma_cm must be positive (got {sigma})")
        shape = np.exp(-0.5 * ((z_centers[eligible] - z0) / sigma) ** 2)
    else:  # cosine_pipe
        throw = float(throw_cm)
        if throw <= 0.0:
            raise ValueError(f"gas_puff_throw_cm must be positive (got {throw})")
        shape = 1.0 / (1.0 + ((z_centers[eligible] - z0) / throw) ** 2) ** 2
    weights[eligible] = shape * np.asarray(
        geometry.length_cm, dtype=float
    )[eligible]
    total_weight = weights.sum()
    if total_weight <= 0.0:
        # Profile centred far outside the chamber: fall back to the puff cell
        # rather than silently deleting the fueling.
        dnn[index] = puff_rate(
            sccm, valves, geometry.neutral_volume_cm3[index]
        )
        return dnn
    total_particles_per_s = 4.477962e17 * float(sccm) * float(valves)
    dnn = (
        total_particles_per_s
        * (weights / total_weight)
        / geometry.neutral_volume_cm3
    )
    return dnn


def neutral_initial_profile_values(geometry, profile, key):
    """Return the validated per-cell initial neutral density [cm^-3].

    ``profile`` is a sequence of length ``geometry.cells``; ``key`` names the
    config key it came from and appears in every message. The array is
    returned as VALUES -- absolute densities, copied and cast to float, with
    no normalization, rescaling or role masking of any kind. That is what
    separates an initial condition from :func:`neutral_probe_profile_weights`,
    whose input is a shape whose scale divides out.

    Every entry must be finite and strictly ``> 0``: a neutral density is
    positive, and zero would be a hole in the fill that the state floor would
    silently paper over at the first read. Raises ``ValueError`` on a
    wrong-length, non-finite, or non-positive profile.
    """
    cells = int(geometry.cells)
    raw = np.array(profile, dtype=float).reshape(-1)
    if raw.size != cells:
        raise ValueError(
            f"{key} must have one entry per grid cell (nx={cells}); got "
            f"{raw.size}. It is a per-cell initial condition, not a shape to "
            "be resampled"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError(
            f"every {key} entry must be finite; got "
            f"{int(np.count_nonzero(~np.isfinite(raw)))} non-finite of {cells}"
        )
    if np.any(raw <= 0.0):
        raise ValueError(
            f"every {key} entry must be > 0 (a neutral density is positive, "
            "and a zero cell would be clipped to nn_floor on the first read "
            f"rather than starting where it was asked to); got min "
            f"{float(np.min(raw)):.6g}"
        )
    return raw


#: Registered axial shape families for the ad-hoc probe source. The membership
#: is deliberately tiny: a family exists so a cheap arm needs no profile file,
#: and anything richer is expressible as an explicit per-cell profile, which
#: carries its own provenance instead of hiding a shape behind a name.
NEUTRAL_PROBE_SHAPES = ("gaussian",)

#: Registered time dependences for the ad-hoc probe source.
NEUTRAL_PROBE_WAVEFORMS = ("const", "square", "table")


def neutral_probe_profile_weights(
    geometry, profile=None, shape=None, center_cm=None, width_cm=None
):
    """Return the ad-hoc probe source's normalized axial shape ``p(z)`` [1].

    Exactly one of ``profile`` (a per-cell sequence of length
    ``geometry.cells``) and ``shape`` (a member of
    :data:`NEUTRAL_PROBE_SHAPES`, with its own parameters) selects the raw
    shape; supplying both or neither raises. ``"gaussian"`` is
    ``exp(-(z - center_cm)^2 / (2 width_cm^2))`` sampled at the cell centres
    ``geometry.z_cm``.

    The raw shape is then rescaled so its CHAMBER-VOLUME-WEIGHTED MEAN over the
    whole grid is exactly 1,

        sum_i p_i V_i / sum_i V_i == 1,   V = geometry.neutral_volume_cm3,

    to roundoff. The input is therefore a shape and not a magnitude: its
    overall scale divides out, the amplitude carries the whole scale, and the
    volume-integrated influx ``A * w * sum(V)`` depends on neither the grid nor
    the profile's normalization.

    No cell role is excluded and no cell length weighting is applied: ``p`` is
    a field sampled at cell centres, so a refinement of the grid converges to
    the continuum shape rather than redistributing it. Placing the source is
    the caller's job, which is what makes this an instrument.

    Raises ``ValueError`` on a wrong-length, non-finite, negative or
    identically-zero profile, on an unknown shape name, and on a
    non-positive or non-finite width.
    """
    cells = int(geometry.cells)
    if (profile is None) == (shape is None):
        raise ValueError(
            "the probe source needs EXACTLY ONE axial shape: "
            "neutral_probe_profile (a per-cell sequence of length "
            f"nx={cells}) or neutral_probe_shape (a parametric family, one of "
            f"{list(NEUTRAL_PROBE_SHAPES)}). "
            + (
                "Both were given; they are two spellings of the same p(z) and "
                "neither modifies the other, so there is no composition rule "
                "to apply -- drop one."
                if profile is not None
                else "Neither was given; there is no default profile -- a "
                "probe source with no stated placement measures nothing."
            )
        )
    if profile is not None:
        raw = np.asarray(profile, dtype=float).reshape(-1)
        if raw.size != cells:
            raise ValueError(
                "neutral_probe_profile must have one entry per grid cell "
                f"(nx={cells}); got {raw.size}"
            )
        if not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
            raise ValueError(
                "every neutral_probe_profile entry must be finite and >= 0 "
                f"(got min {float(np.min(raw)):.6g}, max "
                f"{float(np.max(raw)):.6g})"
            )
    else:
        if shape not in NEUTRAL_PROBE_SHAPES:
            raise ValueError(
                "neutral_probe_shape must be one of "
                f"{list(NEUTRAL_PROBE_SHAPES)} (got {shape!r})"
            )
        z0 = float(center_cm)
        sigma = float(width_cm)
        if not math.isfinite(z0):
            raise ValueError(
                f"neutral_probe_center_cm must be finite (got {center_cm!r})"
            )
        if not (math.isfinite(sigma) and sigma > 0.0):
            raise ValueError(
                "neutral_probe_width_cm must be finite and > 0 (got "
                f"{width_cm!r})"
            )
        z = np.asarray(geometry.z_cm, dtype=float)
        raw = np.exp(-0.5 * ((z - z0) / sigma) ** 2)
    volume = np.asarray(geometry.neutral_volume_cm3, dtype=float)
    weighted = float(np.sum(raw * volume))
    if not (math.isfinite(weighted) and weighted > 0.0):
        # An all-zero profile, or a gaussian so far outside the grid that every
        # sample underflows, is a source that injects nothing anywhere. That is
        # a misconfiguration, not a null control: the amplitude is the null
        # control and it is a separate key.
        raise ValueError(
            "the probe source's axial profile carries no weight on this grid "
            f"(volume-weighted sum {weighted!r}): every cell sample is zero, "
            "so the normalization is undefined. Use "
            "neutral_probe_amplitude_cm3_s = 0 for a null-control arm"
        )
    return raw * (float(np.sum(volume)) / weighted)


def neutral_probe_waveform_value(
    time, waveform, t_on_s=None, t_off_s=None, table=None
):
    """Return the ad-hoc probe source's dimensionless ``w(t)`` at ``time`` [s].

    The INSTANTANEOUS value, i.e. the ``dt -> 0`` limit of
    :func:`neutral_probe_waveform_mean`. This is what a diagnostic read of the
    term reports at a save instant; it is NOT what an integration step
    consumes, which is the step average -- see that function for why the
    difference matters.

    ``time`` is the ABSOLUTE solver clock. ``waveform`` is a member of
    :data:`NEUTRAL_PROBE_WAVEFORMS`:

    * ``"const"`` -- 1.0 everywhere.
    * ``"square"`` -- 1.0 on the half-open interval ``[t_on_s, t_off_s)`` and
      0.0 outside, with hard edges. Half-open so the two edges are
      unambiguous and ``int w dt`` over any window is unambiguous with them.
    * ``"table"`` -- linear interpolation between the rows of ``table``
      (``[t_s, w]`` pairs, ``t`` strictly increasing) and exactly 0.0 strictly
      outside their span.

    The table's validity is checked by :func:`neutral_probe_waveform_table`
    at construction; this reads an already-validated array.
    """
    t = float(time)
    if waveform == "const":
        return 1.0
    if waveform == "square":
        return 1.0 if float(t_on_s) <= t < float(t_off_s) else 0.0
    if waveform == "table":
        nodes = np.asarray(table, dtype=float)
        if t < nodes[0, 0] or t > nodes[-1, 0]:
            return 0.0
        return float(np.interp(t, nodes[:, 0], nodes[:, 1]))
    raise ValueError(
        "neutral_probe_waveform must be one of "
        f"{list(NEUTRAL_PROBE_WAVEFORMS)} (got {waveform!r})"
    )


def neutral_probe_waveform_mean(
    t0, dt, waveform, t_on_s=None, t_off_s=None, table=None,
    table_cumulative=None,
):
    """Return the probe waveform's EXACT step average over ``[t0, t0 + dt]``.

    ``w_bar = (1/dt) * int_{t0}^{t0+dt} w(t) dt``, in closed form for every
    registered waveform. ``dt <= 0`` returns the instantaneous value, which is
    the same quantity's ``dt -> 0`` limit.

    THIS, not the instantaneous value, is what an integration step consumes,
    and the reason is a property of the integrator rather than of the source.
    The explicit step is Heun: it samples the RHS pointwise at ``t0`` and
    ``t0 + dt`` and averages the two with equal weights, so a pointwise
    waveform would be integrated by the TRAPEZOID RULE. On a smooth waveform
    that is merely second-order; across a hard edge it is WRONG BY A FINITE
    AMOUNT, because the trapezoid reads a step that merely touches an edge as
    half a step of delivery. A step ending exactly at a rising edge books
    ``0.5*dt`` of source from outside the window, and one ending at a falling
    edge loses the same, so the error cancels only when the two edge-adjacent
    steps happen to carry equal ``dt`` -- which adaptive stepping does not
    arrange (measured: -1.9e-2 of the stated inventory on an off-lattice
    window, +2.9e-2 on an unequal-dt lattice).

    The probe term is state-independent and separable, so consuming ``w_bar``
    in BOTH Heun stages fixes this identically rather than approximately: the
    stages then carry the same value, the ``0.5/0.5`` combination returns it
    unchanged, and the step delivers ``A * p * int w dt`` exactly. That holds
    for any ``dt``, any edge placement, and any asymmetry between adjacent
    steps, so the delivered inventory is the stated hypothesis and not an
    artifact of where the stepper put its samples.

    ``table_cumulative`` is the companion returned by
    :func:`neutral_probe_waveform_table`: ``cum[i]`` is the integral from the
    first node to node ``i``, so a window costs two interpolations and one
    subtraction instead of a walk over the table.
    """
    t0 = float(t0)
    dt = float(dt)
    if dt <= 0.0:
        return neutral_probe_waveform_value(
            t0, waveform, t_on_s=t_on_s, t_off_s=t_off_s, table=table
        )
    if waveform == "const":
        return 1.0
    if waveform == "square":
        overlap = min(t0 + dt, float(t_off_s)) - max(t0, float(t_on_s))
        return max(overlap, 0.0) / dt
    if waveform == "table":
        nodes = np.asarray(table, dtype=float)
        cum = np.asarray(table_cumulative, dtype=float)
        lo = max(t0, float(nodes[0, 0]))
        hi = min(t0 + dt, float(nodes[-1, 0]))
        if hi <= lo:
            return 0.0
        return (
            _table_integral(nodes, cum, hi) - _table_integral(nodes, cum, lo)
        ) / dt
    raise ValueError(
        "neutral_probe_waveform must be one of "
        f"{list(NEUTRAL_PROBE_WAVEFORMS)} (got {waveform!r})"
    )


def _table_integral(nodes, cum, x):
    """Return ``int`` of the tabulated waveform from its first node to ``x``.

    ``x`` must already lie within the tabulated span. Exact for the piecewise
    linear interpolant: whole segments come from the precomputed cumulative
    sum and the partial one is a single trapezoid.
    """
    t = nodes[:, 0]
    w = nodes[:, 1]
    i = int(np.searchsorted(t, x, side="right")) - 1
    if i >= t.size - 1:
        return float(cum[-1])
    span = t[i + 1] - t[i]
    frac = (x - t[i]) / span
    w_x = w[i] + frac * (w[i + 1] - w[i])
    return float(cum[i] + 0.5 * (w[i] + w_x) * (x - t[i]))


def neutral_probe_waveform_table(table):
    """Return the validated probe waveform table and its cumulative integral.

    Returns ``(nodes, cumulative)``: ``nodes`` is the ``(N, 2)`` float array of
    ``[t_s, w]`` rows, and ``cumulative[i]`` is the exact integral of the
    piecewise-linear interpolant from the first node to node ``i`` (so
    ``cumulative[0] == 0``). The cumulative half is precomputed here, once at
    construction, because :func:`neutral_probe_waveform_mean` reads it on every
    integration stage.

    Requires at least two rows, strictly increasing times, all entries finite,
    and every ``w >= 0``. Raises ``ValueError`` otherwise.
    """
    nodes = np.asarray(table, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] < 2:
        raise ValueError(
            "neutral_probe_waveform_table must be a sequence of at least two "
            "[t_s, w] pairs, i.e. shape (N >= 2, 2); got shape "
            f"{tuple(nodes.shape)}"
        )
    if not np.all(np.isfinite(nodes)):
        raise ValueError(
            "every neutral_probe_waveform_table entry must be finite"
        )
    if not np.all(np.diff(nodes[:, 0]) > 0.0):
        raise ValueError(
            "neutral_probe_waveform_table times must be strictly increasing "
            f"(got {nodes[:, 0].tolist()})"
        )
    if np.any(nodes[:, 1] < 0.0):
        raise ValueError(
            "every neutral_probe_waveform_table w must be >= 0 (got min "
            f"{float(np.min(nodes[:, 1])):.6g})"
        )
    segments = 0.5 * (nodes[:-1, 1] + nodes[1:, 1]) * np.diff(nodes[:, 0])
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    return nodes, cumulative


def neutral_probe_source_rhs(
    state, geometry, amplitude_cm3_s, weights, waveform_value, zone=None
):
    """Return the conservative RHS of the ad-hoc probe neutral source.

    The term is ``S_probe(z, t) = A * p(z) * w(t)`` [cm^-3 s^-1] on the neutral
    density row and nothing else. ``weights`` is ``p`` as returned by
    :func:`neutral_probe_profile_weights` (chamber-volume-weighted mean 1), so
    the volume-integrated influx is ``A * w * sum(neutral_volume_cm3)``
    [particles/s] exactly.

    ``waveform_value`` is the waveform factor the CALLER resolved, and which
    one it resolves decides what the term means. An integration stage passes
    the step average over the step it is taking
    (:func:`neutral_probe_waveform_mean`), which is what makes the delivered
    inventory exactly the stated hypothesis; a diagnostic read passes the
    instantaneous value at its instant
    (:func:`neutral_probe_waveform_value`). The two differ only on a step that
    straddles a hard edge.

    INJECTION CONVENTIONS, both inherited unchanged from the gas puff rather
    than invented here:

    * ZERO NET MOMENTUM. The momentum rows are identically zero: the injected
      gas arrives at rest in the lab frame, so it carries no directed momentum
      of its own. Where a neutral wind is evolved (``M_n`` present) this
      DILUTES it -- ``u_n = M_n / (m_n n_n)`` falls as ``n_n`` rises at fixed
      ``M_n`` -- which is the physical content of injecting at rest and is not
      a separate drag term.
    * TEMPERATURE. The moment neutral model carries ONE neutral temperature,
      the config's ``Tn_K``, and no neutral energy equation, so injected
      particles join that single cold-gas population exactly as gas-puff
      particles do. There is deliberately no probe temperature key: a distinct
      injection temperature would be a new field, not a new parameter.

    Under the two-zone closure ``zone`` selects which neutral field is fed,
    ``"column"`` (``nn``) or ``"annulus"`` (``nn_a``). The per-cell particle
    rate is formed on the CHAMBER volume first and then re-normalized to the
    target zone's volume, so the total influx is the same number whichever
    zone is chosen and the amplitude keeps one meaning. Cells with no annulus
    (``V_ann = 0``) route to the column, as the gas puff does.
    """
    cells = int(geometry.cells)
    zeros = np.zeros(cells, dtype=float)
    rate = (
        float(amplitude_cm3_s)
        * float(waveform_value)
        * np.asarray(weights, dtype=float)
    )
    dnn = np.zeros(cells, dtype=float)
    two_zone = state.nn_a is not None
    dnn_a = np.zeros(cells, dtype=float) if two_zone else None
    if two_zone:
        V_col, V_ann = neutral_zone_volumes(geometry)
        particles = rate * np.asarray(geometry.neutral_volume_cm3, dtype=float)
        if zone == "annulus":
            into_annulus = V_ann > 0.0
            dnn_a += np.where(
                into_annulus, particles / np.maximum(V_ann, 1e-300), 0.0
            )
            dnn += np.where(
                into_annulus, 0.0, particles / np.maximum(V_col, 1e-300)
            )
        elif zone == "column":
            dnn += particles / np.maximum(V_col, 1e-300)
        else:
            raise ValueError(
                "neutral_probe_zone must be 'column' or 'annulus' under the "
                f"two-zone closure (got {zone!r})"
            )
    else:
        dnn += rate
    return ConservativeState1D(
        n=zeros.copy(),
        nn=dnn,
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=zeros.copy(),
        M_n=None if state.M_n is None else zeros.copy(),
        nn_a=dnn_a,
        M_n_a=None if state.M_n_a is None else zeros.copy(),
    )


def _effective_pump_speed(lps, elbow_conductance_lps):
    """Return the pump speed seen by the plenum after the unmodeled elbow [L/s].

    Series conductance, ``1/S_eff = 1/S_pump + 1/C_elbow``. ``None`` or a
    non-positive conductance means no elbow restriction and returns ``S_pump``
    unchanged -- short-circuited rather than computed as ``1/(1/S + 1/inf)`` so
    the legacy path stays bit-exact.
    """
    speed = float(lps)
    if elbow_conductance_lps is None:
        return speed
    conductance = float(elbow_conductance_lps)
    if conductance <= 0.0 or speed <= 0.0:
        return speed
    return 1.0 / (1.0 / speed + 1.0 / conductance)


def pump_rate(lps, chamber_vol):
    """Return pump sink rate coefficient [s^-1] using _sim3 conversion."""
    if chamber_vol <= 0.0:
        raise ValueError(f"chamber_vol must be positive (got {chamber_vol})")
    return float(lps) * 1e3 / float(chamber_vol)


def neutral_inventory_rate(rhs, geometry):
    """Return total neutral inventory rate [particles/s] from a neutral RHS."""
    return math.fsum((rhs.nn * geometry.neutral_volume_cm3).tolist())


def _as_face_coefficients(exchange_coeff_cm3_s, geometry):
    coeff = np.asarray(exchange_coeff_cm3_s, dtype=float)
    if coeff.ndim == 0:
        coeff = np.full(geometry.cells - 1, float(coeff))
    if coeff.shape != (geometry.cells - 1,):
        raise ValueError(
            "exchange_coeff_cm3_s must be scalar or have shape "
            f"({geometry.cells - 1},), got {coeff.shape}"
        )
    if np.any(coeff < 0.0):
        raise ValueError("exchange_coeff_cm3_s must be non-negative")
    return coeff
