import numpy as np

from cablp.atomic.adas import he_rates
from cablp.atomic.fits import IAEA_exp1, IAEA_exp4, IAEA_exp6
from cablp.plasma.heat import Q_cx_He, Q_ie
from cablp.plasma.params import LN_LAMBDA_MIN, c_log
from cablp.atomic.coefficients import aHII, aHI, aHeI, aHeII
from cablp.constants import ev_to_erg

from .reactions import _check_atomic_rate_model, reaction_rates
from ..core.state import ConservativeState1D, derive_state


def electron_ion_exchange_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    b_Qie=1.0,
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
    ln_lambda = np.maximum(c_log(derived.Te, n, kind="ei"), LN_LAMBDA_MIN)
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


def electron_cooling_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    I_ion,
    b_ioniz=1.0,
    b_rec_rad=1.0,
    b_rec_3b=1.0,
    b_ionization_energy_cost=1.0,
    b_Qei=1.0,
    b_Qen=1.0,
    b_Qei_Te_exp=0.0,
    b_Qen_Te_exp=0.0,
    b_Q_Te_ref_eV=5.0,
    atomic_rate_model="janev",
    ionization_energy_cost=True,
    icool_recomb=False,
    adas_low_te_extension=False,
):
    """Return conservative electron inelastic/radiative cooling sources.

    Cooling terms are volumetric electron-energy sinks. The rate helpers
    return eV-rate coefficients, so the accumulated loss is converted to
    conservative ``erg cm^-3 s^-1`` before being applied to ``Ee``.
    """
    terms = electron_cooling_rhs_terms(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        gas_type=gas_type,
        I_ion=I_ion,
        b_ioniz=b_ioniz,
        b_rec_rad=b_rec_rad,
        b_rec_3b=b_rec_3b,
        b_ionization_energy_cost=b_ionization_energy_cost,
        b_Qei=b_Qei,
        b_Qen=b_Qen,
        b_Qei_Te_exp=b_Qei_Te_exp,
        b_Qen_Te_exp=b_Qen_Te_exp,
        b_Q_Te_ref_eV=b_Q_Te_ref_eV,
        atomic_rate_model=atomic_rate_model,
        ionization_energy_cost=ionization_energy_cost,
        icool_recomb=icool_recomb,
        adas_low_te_extension=adas_low_te_extension,
    )
    rhs = terms["ionization_energy_cost"]
    for term in (
        terms["electron_ion_cooling"],
        terms["electron_neutral_cooling"],
    ):
        rhs = ConservativeState1D(
            n=rhs.n + term.n,
            nn=rhs.nn + term.nn,
            M=rhs.M + term.M,
            Ee=rhs.Ee + term.Ee,
            Ei=rhs.Ei + term.Ei,
        )
    return rhs


def electron_cooling_rhs_terms(
    state,
    floors,
    ion_mass_g,
    gas_type,
    I_ion,
    b_ioniz=1.0,
    b_rec_rad=1.0,
    b_rec_3b=1.0,
    b_ionization_energy_cost=1.0,
    b_Qei=1.0,
    b_Qen=1.0,
    b_Qei_Te_exp=0.0,
    b_Qen_Te_exp=0.0,
    b_Q_Te_ref_eV=5.0,
    atomic_rate_model="janev",
    ionization_energy_cost=True,
    icool_recomb=False,
    adas_low_te_extension=False,
):
    """Return split conservative electron cooling source terms.

    ``atomic_rate_model`` selects the cooling coefficients. ``"janev"`` (the
    historical default) uses the IAEA fit expressions -- note the He I fit
    *includes* the ionization-potential loss, so combined with the separate
    ``ionization_energy_cost`` term it double-counts that channel unless
    ``b_Qen`` compensates. ``"adas"`` uses the OPEN-ADAS radiated-power
    coefficients (PLT; plus PRB for ``icool_recomb``), which are radiation
    only and therefore consistent with the separate ionization-cost term.

    The ``b_Qei``/``b_Qen`` scalars carry an optional Te-dependent shape:
    a nonzero ``b_Q*_Te_exp`` multiplies the corresponding cooling term by
    ``(Te / b_Q_Te_ref_eV) ** exp``, so the correction is ``b_Q*`` exactly at
    the reference temperature. The IAEA fits are believed good to only a
    factor ~2 at low Te, and a constant scalar cannot express an error that
    varies over the 2-12 eV discharge range; the exponent hook lets a
    literature- or decay-calibrated shape in without touching the fits. The
    default exponent of 0 skips the factor entirely (bit-exact legacy).
    """
    _check_atomic_rate_model(atomic_rate_model, gas_type)
    zeros = np.zeros_like(state.n, dtype=float)
    ionization_cost_eV = zeros.copy()
    electron_ion_cooling_eV = zeros.copy()
    electron_neutral_cooling_eV = zeros.copy()
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    use_adas = atomic_rate_model == "adas"

    want_cost = ionization_energy_cost and b_ionization_energy_cost != 0.0
    want_qei = b_Qei != 0.0
    want_qen = b_Qen != 0.0

    adas = {}
    if use_adas and (want_cost or want_qei or want_qen):
        quantities = []
        if want_cost:
            quantities.append("scd")
        if want_qei:
            quantities.append("plt2")
            if icool_recomb:
                quantities.append("prb1")
        if want_qen:
            quantities.append("plt1")
        n_safe = np.maximum(state.n, floors["n"])
        # A18/R5.3: honor the low-Te extension here too, so prb1 (recombination
        # radiated power) matches acd (recombination rate, particle path) below
        # the 0.2 eV edge -- one consistent low-Te package. No effect off, or
        # unless prb1 is requested (icool_recomb) at sub-edge Te.
        adas = he_rates(
            n_safe, derived.Te, quantities,
            low_te_extension=adas_low_te_extension,
        )

    if want_cost:
        if use_adas:
            # Must mirror reaction_rates' adas branch exactly: the cost is
            # I_ion per particle actually created by the particle equation.
            S_ion = float(b_ioniz) * state.n * state.nn * adas["scd"]
        else:
            S_ion, _, _ = reaction_rates(
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
        ionization_cost_eV = float(b_ionization_energy_cost) * I_ion * S_ion

    if want_qei:
        if use_adas:
            coeff = adas["plt2"]
            if icool_recomb:
                coeff = coeff + adas["prb1"]
            electron_ion_cooling_eV = float(b_Qei) * coeff * state.n * state.n
        else:
            electron_ion_cooling_eV = float(b_Qei) * _ion_inelastic_cooling_eV(
                derived.Te,
                state.n,
                gas_type=gas_type,
                recomb=icool_recomb,
            )
        if b_Qei_Te_exp != 0.0:
            electron_ion_cooling_eV = electron_ion_cooling_eV * (
                derived.Te / float(b_Q_Te_ref_eV)
            ) ** float(b_Qei_Te_exp)

    if want_qen:
        if use_adas:
            electron_neutral_cooling_eV = (
                float(b_Qen) * adas["plt1"] * state.n * state.nn
            )
        else:
            electron_neutral_cooling_eV = float(b_Qen) * _neutral_inelastic_cooling_eV(
                derived.Te,
                state.n,
                state.nn,
                gas_type=gas_type,
            )
        if b_Qen_Te_exp != 0.0:
            electron_neutral_cooling_eV = electron_neutral_cooling_eV * (
                derived.Te / float(b_Q_Te_ref_eV)
            ) ** float(b_Qen_Te_exp)

    return {
        "ionization_energy_cost": _electron_energy_sink(
            zeros,
            ionization_cost_eV,
        ),
        "electron_ion_cooling": _electron_energy_sink(
            zeros,
            electron_ion_cooling_eV,
        ),
        "electron_neutral_cooling": _electron_energy_sink(
            zeros,
            electron_neutral_cooling_eV,
        ),
    }


def _electron_energy_sink(zeros, loss_eV_cm3_s):
    return ConservativeState1D(
        n=zeros.copy(),
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=-loss_eV_cm3_s * ev_to_erg,
        Ei=zeros.copy(),
    )


def _ion_inelastic_cooling_eV(Te, n, gas_type, recomb=False):
    """Return electron-ion inelastic/radiative cooling [eV cm^-3 s^-1]."""
    if gas_type == "He":
        return IAEA_exp4(Te, aHeII, recomb=recomb) * n * n
    if gas_type == "H":
        return IAEA_exp6(Te, aHII) * n * n
    raise ValueError(f"unsupported gas_type {gas_type!r}; expected 'He' or 'H'")


def ion_charge_exchange_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    Tn_fit=0.1,
    b_Qcx=1.0,
    cx=True,
):
    """Return conservative ion charge-exchange energy sources."""
    zeros = np.zeros_like(state.n, dtype=float)
    if not cx or b_Qcx == 0.0:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    q_cx = (
        float(b_Qcx)
        * Q_cx_He(
            state.n,
            state.nn,
            derived.Ti,
            float(Tn_fit),
            gas_type=gas_type,
            per_particle=False,
        )
        * ev_to_erg
    )
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=-q_cx,
    )


def _neutral_inelastic_cooling_eV(Te, n, nn, gas_type):
    """Return electron-neutral inelastic cooling [eV cm^-3 s^-1]."""
    if gas_type == "He":
        return IAEA_exp1(Te, aHeI) * n * nn
    if gas_type == "H":
        return IAEA_exp1(Te, aHI) * n * nn
    raise ValueError(f"unsupported gas_type {gas_type!r}; expected 'He' or 'H'")
