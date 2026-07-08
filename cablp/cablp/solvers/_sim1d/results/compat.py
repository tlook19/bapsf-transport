from types import SimpleNamespace

import numpy as np


_CATHODE_COMPAT_FIELDS = (
    "phi_c_plus",
    "phi_c_minus",
    "phi_c",
    "phi_a",
    "V_p",
    "V_b",
    "R_p",
    "I_i",
    "I_e",
    "I_eth",
    "I_eth_star",
    "I_tot",
    "P_wall",
    "P_load",
    "P_comp",
    "P_prim",
    "P_ohmic",
    "P_cathode_e",
    "P_cathode_i",
    "P_cathode_i_pl",
    "P_anode_e",
    "P_anode_i",
    "P_anode_i_pl",
    "P_net",
    "P_net2",
    "P_loss",
    "beam_bypass_fraction",
    "l_b",
    "long_mfp",
    "regime",
)


def add_sim3_compat_aliases(result):
    """Attach _sim3-style result aliases to a sim1d result namespace.

    The conservative result fields remain the source of truth.  Energy aliases
    are saved as W/cm^3 power-density diagnostics, not _sim3 primitive
    temperature-rate terms.
    """
    result.ne = result.n
    result.v_plasma = result.u

    cathode_diagnostics = getattr(result, "cathode_diagnostics", {})
    result.n_beam = _diagnostic_or_zeros(cathode_diagnostics, "n_beam", result.n)
    result.cathode = _cathode_namespace(cathode_diagnostics, "source", result.time)
    result.cathode_twin = _cathode_namespace(cathode_diagnostics, "end", result.time)

    rhs_terms = getattr(result, "rhs_terms", {})
    result.Ne_flux = _sum_rhs_fields(
        rhs_terms,
        "n",
        result.n,
        (
            "plasma_advective_flux",
            "plasma_front_flux",
            "surface_loss",
            "cathode_surface_loss",
        ),
    )
    result.Nn_flux = _sum_rhs_fields(
        rhs_terms,
        "nn",
        result.n,
        (
            "neutral_exchange",
            "surface_loss",
            "cathode_surface_loss",
        ),
    )
    result.S_ion_bulk = _rhs_field(rhs_terms, "ionization_birth", "n", result.n)
    result.S_ion_beam = _rhs_field(rhs_terms, "beam_ionization_birth", "n", result.n)
    result.S_rec_rad = -_rhs_field(rhs_terms, "recombination_loss", "n", result.n)
    result.S_rec_3b = np.zeros_like(result.S_rec_rad)

    electron_terms = getattr(result, "electron_energy_terms_W_cm3", {})
    ion_terms = getattr(result, "ion_energy_terms_W_cm3", {})
    zeros = np.zeros_like(result.n, dtype=float)
    result.Qie = _term_or_zeros(ion_terms, "ei_exchange", zeros)
    result.Qei = -_term_or_zeros(electron_terms, "electron_cooling", zeros)
    result.Qen = np.zeros_like(result.Qei)
    result.Qcx = -_term_or_zeros(ion_terms, "ion_charge_exchange", zeros)
    result.Qeb = (
        _term_or_zeros(electron_terms, "beam_power_deposition", zeros)
        + _term_or_zeros(electron_terms, "beam_ionization_cost", zeros)
        + _term_or_zeros(electron_terms, "cathode_surface_loss", zeros)
    )
    result.Qib = (
        _term_or_zeros(ion_terms, "surface_loss", zeros)
        + _term_or_zeros(ion_terms, "cathode_surface_loss", zeros)
    )
    result.sim3_compat_units = {
        "energy_terms": "W/cm^3",
        "density_terms": "cm^-3 s^-1",
        "time": "s",
    }
    result.sim3_compat_notes = {
        "Qei": "coarse electron_cooling sink; ion/neutral pieces are not split yet",
        "Qen": "zero placeholder until electron_cooling is split by target",
        "Qeb": "net electron beam/cathode power-density mapping",
        "Qib": "net ion surface/cathode loss power-density mapping",
        "S_rec_rad": "combined recombination loss; rad/3body split is not saved yet",
        "S_rec_3b": "zero placeholder until recombination split is saved",
    }
    return result


def _sum_rhs_fields(rhs_terms, field_name, shape_like, term_names):
    total = np.zeros_like(shape_like, dtype=float)
    for term_name in term_names:
        total = total + _rhs_field(rhs_terms, term_name, field_name, shape_like)
    return total


def _rhs_field(rhs_terms, term_name, field_name, shape_like):
    if term_name not in rhs_terms:
        return np.zeros_like(shape_like, dtype=float)
    return np.asarray(rhs_terms[term_name][field_name], dtype=float)


def _term_or_zeros(terms, term_name, shape_like):
    if term_name not in terms:
        return np.zeros_like(shape_like, dtype=float)
    return np.asarray(terms[term_name], dtype=float)


def _diagnostic_or_zeros(diagnostics, name, shape_like):
    if name not in diagnostics:
        return np.zeros_like(shape_like, dtype=float)
    return np.asarray(diagnostics[name], dtype=float)


def _cathode_namespace(diagnostics, prefix, time):
    fields = {}
    for name in _CATHODE_COMPAT_FIELDS:
        key = f"{prefix}_{name}"
        if key in diagnostics:
            fields[name] = np.asarray(diagnostics[key]).copy()
        elif name == "regime":
            fields[name] = np.full_like(time, "none", dtype=object)
        else:
            fields[name] = np.full_like(time, np.nan, dtype=float)
    return SimpleNamespace(**fields)
