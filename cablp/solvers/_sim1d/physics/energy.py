import numpy as np

from cablp.atomic.adas import he_rates
from cablp.plasma.heat import Q_cx_He, Q_ie
from cablp.plasma.params import LN_LAMBDA_MIN, c_log
from cablp.constants import ev_to_erg

from ..core.state import ConservativeState1D, derive_state


def electron_ion_exchange_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
):
    """Return conservative electron-ion thermal exchange sources.

    ``Q_ie`` is positive when electrons transfer energy to ions. The helper
    returns eV cm^-3 s^-1 with ``per_particle=False``; conservative energies are
    stored as erg cm^-3.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    n = np.maximum(state.n, floors["n"])
    ln_lambda = np.maximum(c_log(derived.Te, n, kind="ei"), LN_LAMBDA_MIN)
    q_e_to_i = (
        Q_ie(
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


def electron_ion_relaxation_rate(
    state,
    floors,
    ion_mass_g,
    mu,
):
    """Return the per-cell electron-ion thermal relaxation rate [s^-1].

    ``nu_eq`` is the rate at which ``electron_ion_exchange_rhs`` above relaxes
    ONE species' temperature toward the other: with ``Ee = 3/2 n k Te`` the
    exchange term gives ``dTe/dt = -nu_eq (Te - Ti)`` and, with equal electron
    and ion heat capacities, ``dTi/dt = +nu_eq (Te - Ti)``, so the DIFFERENCE
    ``Te - Ti`` relaxes at ``2 nu_eq``.

    The rate is read back out of ``plasma.heat.Q_ie`` -- the same expression
    the exchange term itself calls -- rather than restated here, so the two
    cannot drift apart. ``Q_ie`` is linear in ``Te - Ti`` with a collision
    time that depends on ``Te`` and ``n`` only, so evaluating it at ``Ti = 0``
    returns ``3/2 n nu_eq Te`` volumetrically and the division below recovers
    ``nu_eq``. The derived temperatures, the floored density and the Coulomb
    logarithm are formed exactly as ``electron_ion_exchange_rhs`` forms them,
    so the rate describes the term as applied and not an idealization of it.

    Returns a positive array in ``s^-1``: ``Te`` and ``n`` are floored, so no
    cell can divide by zero and no cell can return a negative rate.
    """
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    n = np.maximum(state.n, floors["n"])
    ln_lambda = np.maximum(c_log(derived.Te, n, kind="ei"), LN_LAMBDA_MIN)
    unit_difference = Q_ie(
        derived.Te,
        0.0,
        n,
        mu,
        ln_lambda,
        per_particle=False,
    )
    return unit_difference / (1.5 * n * derived.Te)


def electron_cooling_rhs(
    state,
    floors,
    ion_mass_g,
    I_ion,
    b_ionization_energy_cost=1.0,
    ionization_energy_cost=True,
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
        I_ion=I_ion,
        b_ionization_energy_cost=b_ionization_energy_cost,
        ionization_energy_cost=ionization_energy_cost,
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
    I_ion,
    b_ionization_energy_cost=1.0,
    ionization_energy_cost=True,
):
    """Return split conservative electron cooling source terms.

    The cooling coefficients are the OPEN-ADAS radiated-power coefficients
    (PLT), which are radiation only and therefore consistent with the
    separate ionization-cost term.

    The cooling coefficients are applied unscaled: atomic rates are fixed
    inputs, not knobs.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    ionization_cost_eV = zeros.copy()
    electron_ion_cooling_eV = zeros.copy()
    electron_neutral_cooling_eV = zeros.copy()
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)

    want_cost = ionization_energy_cost and b_ionization_energy_cost != 0.0

    quantities = []
    if want_cost:
        quantities.append("scd")
    quantities.append("plt2")
    quantities.append("plt1")
    n_safe = np.maximum(state.n, floors["n"])
    adas = he_rates(n_safe, derived.Te, quantities)

    if want_cost:
        # Must mirror reaction_rates exactly: the cost is I_ion per particle
        # actually created by the particle equation.
        S_ion = state.n * state.nn * adas["scd"]
        ionization_cost_eV = float(b_ionization_energy_cost) * I_ion * S_ion

    electron_ion_cooling_eV = adas["plt2"] * state.n * state.n
    electron_neutral_cooling_eV = adas["plt1"] * state.n * state.nn

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


def ion_charge_exchange_rhs(
    state,
    floors,
    ion_mass_g,
    Tn_fit=0.1,
):
    """Return conservative ion charge-exchange energy sources.

    Not a row of the RHS -- the moment-closed ion-neutral collision operator
    carries CX cooling -- but the rate the ``ion_charge_exchange`` timestep
    bound is built from.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    q_cx = (
        Q_cx_He(
            state.n,
            state.nn,
            derived.Ti,
            float(Tn_fit),
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
