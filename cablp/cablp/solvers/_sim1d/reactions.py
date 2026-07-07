import math

import numpy as np

from cablp.funcs._cross import He_ion_rate_lkup, alpha_3, alpha_r
from cablp.funcs._fits import rate_coeff
from cablp.vars._cons import ev_to_erg

from .state import ConservativeState1D, derive_state


H_ION_COEFF = (1e-5, 6.0)


def reaction_rates(
    state,
    floors,
    ion_mass_g,
    gas_type,
    I_ion,
    b_ioniz=1.0,
    b_rec_rad=1.0,
    b_rec_3b=1.0,
):
    """Return bulk ionization and recombination density rates [cm^-3 s^-1]."""
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
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
    Te_birth_ionization="local",
    Ti_birth_ionization="floor",
):
    """Return conservative source terms for local bulk plasma reactions."""
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
    )
    S_rec = S_rec_rad + S_rec_3b
    volume_ratio = geometry.plasma_volume_cm3 / geometry.neutral_volume_cm3

    Te_birth = _birth_temperature(Te_birth_ionization, derived.Te, floors["Te"])
    Ti_birth = _birth_temperature(Ti_birth_ionization, derived.Ti, floors["Ti"])

    dn = S_ion - S_rec
    dnn = -S_ion * volume_ratio + S_rec * volume_ratio
    dM = -ion_mass_g * derived.u * S_rec
    dEe = 1.5 * ev_to_erg * (Te_birth * S_ion - derived.Te * S_rec)
    dEi = 1.5 * ev_to_erg * (Ti_birth * S_ion - derived.Ti * S_rec)
    return ConservativeState1D(n=dn, nn=dnn, M=dM, Ee=dEe, Ei=dEi)


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
