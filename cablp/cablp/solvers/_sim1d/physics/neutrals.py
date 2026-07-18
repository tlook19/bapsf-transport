import math

import numpy as np

from ..core.geometry import is_plenum_cell, puff_cell_indices, pump_cell_indices
from ..core.state import ConservativeState1D
from cablp.vars._cons import kb_cgs, m_p_cgs


def neutral_thermal_speed(Tn_K, mu_neutral):
    """Return neutral thermal speed [cm/s] using the _sim3 convention."""
    if Tn_K <= 0.0:
        raise ValueError(f"Tn_K must be positive (got {Tn_K})")
    if mu_neutral <= 0.0:
        raise ValueError(f"mu_neutral must be positive (got {mu_neutral})")
    return np.sqrt(8.0 * kb_cgs * Tn_K / (np.pi * mu_neutral * m_p_cgs))


def molecular_flow_coefficients(
    geometry,
    Tn_K,
    mu_neutral,
    clausing_scale=1.0,
):
    """Return internal-face molecular-flow conductances [cm^3/s]."""
    if clausing_scale < 0.0:
        raise ValueError(f"clausing_scale must be non-negative (got {clausing_scale})")
    v_th_n = neutral_thermal_speed(Tn_K=Tn_K, mu_neutral=mu_neutral)
    L_eff = 0.5 * (geometry.length_cm[:-1] + geometry.length_cm[1:])
    # Generalized Clausing (BOUNDARY_REGIONS_PLAN.md §4): the aperture area and
    # the hydraulic radius are carried per face and may differ -- an annular
    # obstruction reduces both, independently. On legacy geometry these reduce to
    # today's pi*Rm^2 and 0.5*(Rm[:-1] + Rm[1:]).
    R_face = geometry.neutral_face_hydraulic_radius_cm[1:-1]
    if np.any(R_face <= 0.0):
        raise ValueError("neutral hydraulic radii must be positive")
    clausing = 1.0 / (1.0 + (3.0 / 8.0) * L_eff / R_face)
    coefficients = (
        float(clausing_scale)
        * 0.25
        * v_th_n
        * geometry.neutral_face_area_cm2[1:-1]
        * clausing
    )
    # Escape hatch: a face whose conductance is known directly rather than
    # geometrically overrides the computed value (NaN => keep the computed one).
    prescribed = np.asarray(geometry.neutral_face_conductance_cm3_s[1:-1], dtype=float)
    return np.where(np.isnan(prescribed), coefficients, prescribed)


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
    if model == "molecular_flow":
        return molecular_flow_coefficients(
            geometry=geometry,
            Tn_K=Tn_K,
            mu_neutral=mu_neutral,
            clausing_scale=clausing_scale,
        )
    raise ValueError(
        "neutral_exchange_model must be 'constant' or 'molecular_flow' "
        f"(got {model!r})"
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
):
    """Return conservative RHS for neutral gas puff and pump terms.

    Both terms are anchored by ``cell_role`` (§8), not by ``[0]``/``[-1]``: the
    puff lands on its puff cell and each pump on the plenum/collector at its end.
    Legacy roles resolve to the source and end cells, reproducing today exactly.
    """
    dnn = np.zeros(geometry.cells, dtype=float)
    puff_index, puff_twin_index = puff_cell_indices(geometry)
    pump_left_index, pump_right_index = pump_cell_indices(geometry)
    if gas_puff_enabled:
        dnn[puff_index] += puff_rate(
            S_gp, gas_puff_valves, geometry.neutral_volume_cm3[puff_index]
        )
        if twin_cathode:
            dnn[puff_twin_index] += puff_rate(
                Twin_S_gp,
                gas_puff_valves,
                geometry.neutral_volume_cm3[puff_twin_index],
            )
    if pump_enabled:
        # The unmodeled pump elbow folds into an effective speed on the plenum
        # (§4); a collector-side pump has no elbow in front of it.
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
        dnn[pump_left_index] -= (
            pump_rate(S_left, geometry.neutral_volume_cm3[pump_left_index])
            * state.nn[pump_left_index]
        )
        dnn[pump_right_index] -= (
            pump_rate(S_right, geometry.neutral_volume_cm3[pump_right_index])
            * state.nn[pump_right_index]
        )
    zeros = np.zeros(geometry.cells, dtype=float)
    return ConservativeState1D(
        n=zeros.copy(),
        nn=dnn,
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=zeros.copy(),
    )


def puff_rate(sccm, valves, chamber_vol):
    """Return gas puff source rate [cm^-3 s^-1] using _sim3 conversion."""
    if chamber_vol <= 0.0:
        raise ValueError(f"chamber_vol must be positive (got {chamber_vol})")
    return 4.477962e17 * float(sccm) * float(valves) / float(chamber_vol)


def _effective_pump_speed(lps, elbow_conductance_lps):
    """Return the pump speed seen by the plenum after the unmodeled elbow [L/s].

    Series conductance, ``1/S_eff = 1/S_pump + 1/C_elbow`` (§4). ``None`` or a
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
