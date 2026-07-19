import math

import numpy as np

from cablp.funcs._adas import he_rates
from cablp.funcs._cross import He_ion_rate_lkup, alpha_3, alpha_r
from cablp.funcs._fits import rate_coeff
from cablp.vars._cons import ev_to_erg

from ..core.state import ConservativeState1D, derive_state


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
        rates = he_rates(n_safe, derived.Te, ("scd", "acd"))
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
    Te_birth_ionization="local",
    Ti_birth_ionization="floor",
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
        Te_birth_ionization=Te_birth_ionization,
        Ti_birth_ionization=Ti_birth_ionization,
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
    Te_birth_ionization="local",
    Ti_birth_ionization="floor",
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
    )
    volume_ratio = geometry.plasma_volume_cm3 / geometry.neutral_volume_cm3

    Te_birth = _birth_temperature(Te_birth_ionization, derived.Te, floors["Te"])
    Ti_birth = _birth_temperature(Ti_birth_ionization, derived.Ti, floors["Ti"])

    zeros = np.zeros_like(state.n, dtype=float)
    ionization = ConservativeState1D(
        n=S_ion,
        nn=-S_ion * volume_ratio,
        M=zeros,
        Ee=1.5 * ev_to_erg * Te_birth * S_ion,
        Ei=1.5 * ev_to_erg * Ti_birth * S_ion,
    )
    return {
        "ionization_birth": ionization,
        "recombination_rad_loss": _recombination_loss(
            S_rec_rad,
            volume_ratio,
            ion_mass_g,
            derived,
        ),
        "recombination_3b_loss": _recombination_loss(
            S_rec_3b,
            volume_ratio,
            ion_mass_g,
            derived,
        ),
    }


def _recombination_loss(S_rec, volume_ratio, ion_mass_g, derived):
    return ConservativeState1D(
        n=-S_rec,
        nn=S_rec * volume_ratio,
        M=-ion_mass_g * derived.u * S_rec,
        Ee=-1.5 * ev_to_erg * derived.Te * S_rec,
        Ei=-1.5 * ev_to_erg * derived.Ti * S_rec,
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
