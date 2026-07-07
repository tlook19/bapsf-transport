import numpy as np

from .state import ConservativeState1D


def neutral_exchange_face_rates(nn, geometry, exchange_coeff_cm3_s):
    """Return neutral inventory rates across internal faces [particles/s]."""
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


def neutral_inventory_rate(rhs, geometry):
    """Return total neutral inventory rate [particles/s] from a neutral RHS."""
    return float(np.sum(rhs.nn * geometry.neutral_volume_cm3))
