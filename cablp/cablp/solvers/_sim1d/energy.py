import numpy as np

from cablp.funcs._heat import Q_ie
from cablp.funcs._plasmaparams import c_log
from cablp.vars._cons import ev_to_erg

from .state import ConservativeState1D, derive_state


def electron_ion_exchange_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    b_Qie=1.0,
    ln_lambda_min=1.0,
):
    """Return conservative electron-ion thermal exchange sources.

    ``Q_ie`` is positive when electrons transfer energy to ions. The helper
    returns eV cm^-3 s^-1 with ``per_particle=False``; conservative energies are
    stored as erg cm^-3.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    if b_Qie == 0.0:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    n = np.maximum(state.n, floors["n"])
    ln_lambda = np.maximum(c_log(derived.Te, n, kind="ei"), ln_lambda_min)
    q_e_to_i = (
        float(b_Qie)
        * Q_ie(
            derived.Te,
            derived.Ti,
            n,
            mu,
            ln_lambda,
            per_particle=False,
        )
        * ev_to_erg
    )
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=-q_e_to_i,
        Ei=q_e_to_i,
    )
