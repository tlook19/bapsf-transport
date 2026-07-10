import numpy as np

from cablp.funcs._cross import charge_ex_react
from cablp.vars._cons import ev_to_erg

from .flux import ion_sound_speed
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
    """Return conservative source/end surface plasma neutralization terms."""
    zeros = np.zeros(geometry.cells, dtype=float)
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


def ion_neutral_drag_rhs(
    state,
    floors,
    ion_mass_g,
    sigma_in_cm2=5.0e-15,
    b_ion_neutral_drag=1.0,
):
    """Return the conservative ion-neutral drag momentum sink.

    The drag force density is ``-m_i * nu_in(Ti) * n * u`` [g cm^-2 s^-2], a
    friction on the plasma flow from collisions with the neutral background.
    Only the momentum field is affected.
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
    nu_in = ion_neutral_collision_frequency(
        nn=state.nn,
        Ti=derived.Ti,
        ion_mass_g=ion_mass_g,
        sigma_in_cm2=sigma_in_cm2,
    )
    drag = -float(b_ion_neutral_drag) * ion_mass_g * nu_in * state.n * derived.u
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
):
    """Return the elastic (non-CX) ion-neutral momentum-transfer frequency [s^-1].

    ``nu_el = max(nu_in - nu_cx, 0)`` where ``nu_in`` is the total (``sigma_in``)
    momentum-transfer rate and ``nu_cx = nn * <sigma v>_cx`` is the resonant
    charge-exchange rate shared with the ``Q_cx`` energy term.
    """
    nu_in = ion_neutral_collision_frequency(
        nn=nn,
        Ti=Ti,
        ion_mass_g=ion_mass_g,
        sigma_in_cm2=sigma_in_cm2,
    )
    nu_cx = np.asarray(nn, dtype=float) * charge_ex_react(Ti, gas_type)
    return np.maximum(nu_in - nu_cx, 0.0)


def ion_neutral_frictional_heating_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    sigma_in_cm2=5.0e-15,
    b_ion_neutral_drag=1.0,
):
    """Return the conservative ion frictional-heating energy source.

    Elastic ion-neutral collisions thermalize the flow's directed energy; for
    equal masses half of the dissipated drift energy heats the ions, giving the
    ``Ei`` source ``+(1/2) m_i * nu_el(Ti) * n * u^2`` [erg cm^-3 s^-1]. The
    charge-exchange fraction carries its energy off with the fast neutral and is
    excluded via ``nu_el = nu_in - nu_cx``.
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
):
    """Return the conservative elastic ion-neutral thermal-equilibration source.

    Elastic collisions relax ``Ti`` toward the neutral temperature at the elastic
    rate, giving the ``Ei`` source ``+(3/2) nu_el(Ti) * n * (Tn - Ti)``
    [erg cm^-3 s^-1]. This is the elastic companion to the CX ``Q_cx`` cooling and
    is gated separately by the ``ion_neutral_thermalization`` flag.
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
