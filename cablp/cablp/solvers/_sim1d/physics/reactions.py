import math

import numpy as np

from cablp.funcs._adas import he_rates
from cablp.funcs._cross import He_ion_rate_lkup, alpha_3, alpha_r
from cablp.funcs._fits import rate_coeff
from cablp.vars._cons import ev_to_erg

from ..core.state import ConservativeState1D, derive_state
from .sources import neutral_wind_velocity


H_ION_COEFF = (1e-5, 6.0)

ATOMIC_RATE_MODELS = ("janev", "adas")


def _check_atomic_rate_model(atomic_rate_model, gas_type):
    if atomic_rate_model not in ATOMIC_RATE_MODELS:
        raise ValueError(
            f"atomic_rate_model must be one of {ATOMIC_RATE_MODELS} "
            f"(got {atomic_rate_model!r})"
        )
    if atomic_rate_model == "adas" and gas_type != "He":
        raise ValueError(
            "atomic_rate_model='adas' is only wired for gas_type 'He' "
            f"(got {gas_type!r})"
        )


def reaction_rates(
    state,
    floors,
    ion_mass_g,
    gas_type,
    I_ion,
    b_ioniz=1.0,
    b_rec_rad=1.0,
    b_rec_3b=1.0,
    atomic_rate_model="janev",
    adas_low_te_extension=False,
):
    """Return bulk ionization and recombination density rates [cm^-3 s^-1].

    ``atomic_rate_model`` selects the coefficient source. ``"janev"`` (the
    historical default) uses the direct ground-state ionization rate and the
    separate radiative/three-body recombination coefficients. ``"adas"`` uses
    the OPEN-ADAS GCR effective coefficients (``cablp.funcs._adas``): SCD for
    ionization -- which includes the stepwise/metastable channel the direct
    rate lacks (up to ~3-6x at 3-5 eV, LAPD densities) -- and ACD for
    recombination. ACD already contains three-body recombination at the
    tabulated density, so in adas mode the whole sink is reported through the
    ``S_rec_rad`` slot scaled by ``b_rec_rad``, and ``b_rec_3b`` is inert.
    """
    _check_atomic_rate_model(atomic_rate_model, gas_type)
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    if atomic_rate_model == "adas":
        n_safe = np.maximum(state.n, floors["n"])
        rates = he_rates(
            n_safe, derived.Te, ("scd", "acd"),
            low_te_extension=adas_low_te_extension,
        )
        S_ion = float(b_ioniz) * state.n * state.nn * rates["scd"]
        S_rec_rad = float(b_rec_rad) * state.n * state.n * rates["acd"]
        S_rec_3b = np.zeros_like(state.n, dtype=float)
        return S_ion, S_rec_rad, S_rec_3b

    if gas_type == "He":
        ion_rate = He_ion_rate_lkup(derived.Te)
    elif gas_type == "H":
        ion_rate = rate_coeff(derived.Te, I_ion, *H_ION_COEFF)
    else:
        raise ValueError(f"unsupported gas_type {gas_type!r}; expected 'He' or 'H'")

    S_ion = float(b_ioniz) * state.n * state.nn * ion_rate
    S_rec_rad = float(b_rec_rad) * state.n * state.n * alpha_r(derived.Te, I=I_ion)
    S_rec_3b = float(b_rec_3b) * state.n * state.n * state.n * alpha_3(derived.Te)
    return S_ion, S_rec_rad, S_rec_3b


def reaction_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    gas_type,
    I_ion,
    b_ioniz=1.0,
    b_rec_rad=1.0,
    b_rec_3b=1.0,
    atomic_rate_model="janev",
    adas_low_te_extension=False,
    Te_birth_ionization="local",
    Ti_birth_ionization="floor",
    wind_column_factor=None,
):
    """Return conservative source terms for local bulk plasma reactions."""
    terms = reaction_rhs_terms(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        gas_type=gas_type,
        I_ion=I_ion,
        b_ioniz=b_ioniz,
        b_rec_rad=b_rec_rad,
        b_rec_3b=b_rec_3b,
        atomic_rate_model=atomic_rate_model,
        adas_low_te_extension=adas_low_te_extension,
        Te_birth_ionization=Te_birth_ionization,
        Ti_birth_ionization=Ti_birth_ionization,
        wind_column_factor=wind_column_factor,
    )
    ionization = terms["ionization_birth"]
    recombination_rad = terms["recombination_rad_loss"]
    recombination_3b = terms["recombination_3b_loss"]
    return ConservativeState1D(
        n=ionization.n + recombination_rad.n + recombination_3b.n,
        nn=ionization.nn + recombination_rad.nn + recombination_3b.nn,
        M=ionization.M + recombination_rad.M + recombination_3b.M,
        Ee=ionization.Ee + recombination_rad.Ee + recombination_3b.Ee,
        Ei=ionization.Ei + recombination_rad.Ei + recombination_3b.Ei,
    )


def reaction_rhs_terms(
    state,
    floors,
    ion_mass_g,
    geometry,
    gas_type,
    I_ion,
    b_ioniz=1.0,
    b_rec_rad=1.0,
    b_rec_3b=1.0,
    atomic_rate_model="janev",
    adas_low_te_extension=False,
    Te_birth_ionization="local",
    Ti_birth_ionization="floor",
    wind_column_factor=None,
):
    """Return ionization and recombination conservative source terms."""
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    S_ion, S_rec_rad, S_rec_3b = reaction_rates(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        gas_type=gas_type,
        I_ion=I_ion,
        b_ioniz=b_ioniz,
        b_rec_rad=b_rec_rad,
        b_rec_3b=b_rec_3b,
        atomic_rate_model=atomic_rate_model,
        adas_low_te_extension=adas_low_te_extension,
    )
    volume_ratio = geometry.plasma_volume_cm3 / geometry.neutral_volume_cm3
    # On a two-zone state (NEUTRAL_TWOZONE_PLAN.md) nn is the COLUMN density
    # on the column volume, which IS the plasma volume -- the Vp/V_col
    # conversion on the neutral-density rows is exactly unity, and the rates
    # above (n * nn * <sv>) already sample column gas by construction. The
    # M_n exchanges keep Vp/Vm: the wind stays a chamber-mean field.
    nn_ratio = (
        np.ones_like(volume_ratio) if state.nn_a is not None else volume_ratio
    )

    Te_birth = _birth_temperature(Te_birth_ionization, derived.Te, floors["Te"])
    Ti_birth = _birth_temperature(Ti_birth_ionization, derived.Ti, floors["Ti"])

    zeros = np.zeros_like(state.n, dtype=float)
    # With an evolved neutral wind (state carries M_n), reactions exchange
    # momentum between the species: an ionized neutral is born drifting at
    # u_n (fixing the historical zero-drift birth), and a recombined ion
    # hands its momentum to the wind. Both close M*Vp + M_n*Vm exactly
    # through the same (Vp/Vm) conversion the particles use. Ionization only
    # ever consumes *column* gas, so the two-zone closure's column factor
    # (when given) scales the sampled wind up from the chamber mean.
    if state.M_n is not None:
        u_n = neutral_wind_velocity(
            state, floors=floors, ion_mass_g=ion_mass_g, geometry=geometry
        )
        if wind_column_factor is not None:
            u_n = wind_column_factor * u_n
        M_birth = ion_mass_g * u_n * S_ion
        M_n_birth = -M_birth * volume_ratio
    else:
        M_birth = zeros
        M_n_birth = None
    ionization = ConservativeState1D(
        n=S_ion,
        nn=-S_ion * nn_ratio,
        M=M_birth,
        Ee=1.5 * ev_to_erg * Te_birth * S_ion,
        Ei=1.5 * ev_to_erg * Ti_birth * S_ion,
        M_n=M_n_birth,
    )
    with_wind = state.M_n is not None
    return {
        "ionization_birth": ionization,
        "recombination_rad_loss": _recombination_loss(
            S_rec_rad,
            volume_ratio,
            ion_mass_g,
            derived,
            with_wind=with_wind,
            nn_ratio=nn_ratio,
        ),
        "recombination_3b_loss": _recombination_loss(
            S_rec_3b,
            volume_ratio,
            ion_mass_g,
            derived,
            with_wind=with_wind,
            nn_ratio=nn_ratio,
        ),
    }


def _recombination_loss(
    S_rec, volume_ratio, ion_mass_g, derived, with_wind=False, nn_ratio=None
):
    M_loss = -ion_mass_g * derived.u * S_rec
    if nn_ratio is None:
        nn_ratio = volume_ratio
    return ConservativeState1D(
        n=-S_rec,
        nn=S_rec * nn_ratio,
        M=M_loss,
        Ee=-1.5 * ev_to_erg * derived.Te * S_rec,
        Ei=-1.5 * ev_to_erg * derived.Ti * S_rec,
        M_n=(-M_loss * volume_ratio) if with_wind else None,
    )


def recombination_energy_return_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    I_ion,
    b_rec_rad=1.0,
    atomic_rate_model="janev",
    enabled=False,
    adas_low_te_extension=False,
):
    """Return the GCR-consistent recombination energy pair (electron fluid).

    Per recombination event the electron fluid is credited the binding
    energy ``I_ion`` (paid at ionization via ``I_ion * S_ion`` and never
    returned in the standard booking) and charged the full ADAS PRB --
    the photons that actually leave (recombination radiation,
    bremsstrahlung, cascade). Net per event ``I_ion - E_rad``: heating in
    the recombining afterglow (ADAS says E_rad ~ 18 of 24.6 eV at LAPD
    afterglow conditions -- three-body capture into Rydberg states with
    partial radiative cascade), a small extra sink in the ionizing plateau
    (E_rad/event > I_ion there). The ``3/2 Te S_rec`` capture-KE loss in
    the recombination terms stays booked -- it cancels in the net; this
    pair adds ``I_ion*S_rec - P_PRB`` on top. The PAIR is the consistent
    unit (PRB alone double-charges -- the ``icool_recomb`` audit); both
    halves scale with ``b_rec_rad`` so the credit tracks the particle
    equation's actual sink. ADAS ('adas' rate model) only: the janev path
    has no PRB booking. Grid lookups clamp at the adf11 edges (0.2 eV Te
    floor), nearest-edge.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    if not enabled:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )
    if atomic_rate_model != "adas":
        raise ValueError(
            "recombination_energy_return requires atomic_rate_model='adas' "
            "(the PRB radiated-power booking has no janev counterpart)"
        )
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    n_safe = np.maximum(state.n, floors["n"])
    rates = he_rates(
        n_safe, derived.Te, ("acd", "prb1"),
        low_te_extension=adas_low_te_extension,
    )
    # Mirror reaction_rates' adas branch exactly: the credit is I_ion per
    # particle the particle equation actually recombines.
    S_rec = float(b_rec_rad) * state.n * state.n * rates["acd"]
    P_prb_eV = float(b_rec_rad) * state.n * state.n * rates["prb1"]
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=ev_to_erg * (I_ion * S_rec - P_prb_eV),
        Ei=zeros.copy(),
    )


def particle_inventory_rate(rhs, geometry):
    """Return total plasma-plus-neutral particle inventory rate [particles/s]."""
    terms = rhs.n * geometry.plasma_volume_cm3 + rhs.nn * geometry.neutral_volume_cm3
    return math.fsum(terms.tolist())


def _birth_temperature(value, local_temperature, floor_temperature):
    if isinstance(value, str):
        if value == "local":
            return local_temperature
        if value == "floor":
            return np.full_like(local_temperature, floor_temperature, dtype=float)
        raise ValueError(
            "birth temperature must be 'local', 'floor', or a numeric eV value "
            f"(got {value!r})"
        )
    return np.full_like(local_temperature, float(value), dtype=float)
