from dataclasses import dataclass

import numpy as np

from cablp.vars._cons import ev_to_erg


STATE_NAMES_1D = ("n", "nn", "M", "Ee", "Ei")


@dataclass(frozen=True)
class ConservativeState1D:
    n: np.ndarray
    nn: np.ndarray
    M: np.ndarray
    Ee: np.ndarray
    Ei: np.ndarray


@dataclass(frozen=True)
class DerivedState1D:
    u: np.ndarray
    Te: np.ndarray
    Ti: np.ndarray
    pe: np.ndarray
    pi: np.ndarray
    p: np.ndarray


def pack_state(state):
    """Pack conservative state arrays into a flat solver vector."""
    return np.vstack([state.n, state.nn, state.M, state.Ee, state.Ei]).ravel()


def unpack_state(y, cells):
    """Unpack a flat solver vector into conservative state arrays."""
    arr = np.asarray(y, dtype=float).reshape((len(STATE_NAMES_1D), cells))
    return ConservativeState1D(*(row.copy() for row in arr))


def conservative_from_primitives(n, nn, u, Te, Ti, ion_mass_g):
    """Build conservative variables from primitive CGS/eV quantities."""
    n = np.asarray(n, dtype=float)
    nn = np.asarray(nn, dtype=float)
    u = np.asarray(u, dtype=float)
    Te = np.asarray(Te, dtype=float)
    Ti = np.asarray(Ti, dtype=float)
    M = ion_mass_g * n * u
    Ee = 1.5 * n * Te * ev_to_erg
    Ei = 1.5 * n * Ti * ev_to_erg
    return ConservativeState1D(n=n.copy(), nn=nn.copy(), M=M, Ee=Ee, Ei=Ei)


def apply_state_floors(state, floors, ion_mass_g):
    """Return a conservative state with density and temperature floors applied."""
    n = np.maximum(np.asarray(state.n, dtype=float), floors["n"])
    nn = np.maximum(np.asarray(state.nn, dtype=float), floors["nn"])
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    u = derived.u
    Te = np.maximum(derived.Te, floors["Te"])
    Ti = np.maximum(derived.Ti, floors["Ti"])
    return conservative_from_primitives(n, nn, u, Te, Ti, ion_mass_g)


def derive_state(state, floors, ion_mass_g):
    """Compute primitive and pressure fields from conservative variables."""
    n_safe = np.maximum(np.asarray(state.n, dtype=float), floors["n"])
    M = np.asarray(state.M, dtype=float)
    Ee = np.asarray(state.Ee, dtype=float)
    Ei = np.asarray(state.Ei, dtype=float)

    u = M / (ion_mass_g * n_safe)
    Te = (2.0 / 3.0) * Ee / (n_safe * ev_to_erg)
    Ti = (2.0 / 3.0) * Ei / (n_safe * ev_to_erg)
    Te = np.maximum(Te, floors["Te"])
    Ti = np.maximum(Ti, floors["Ti"])
    pe = n_safe * Te * ev_to_erg
    pi = n_safe * Ti * ev_to_erg
    return DerivedState1D(u=u, Te=Te, Ti=Ti, pe=pe, pi=pi, p=pe + pi)


def assert_finite_state(state, derived=None):
    """Raise ValueError if any conservative or derived state field is non-finite."""
    for name in STATE_NAMES_1D:
        values = getattr(state, name)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite values in state field {name!r}")
    if derived is None:
        return
    for name in ("u", "Te", "Ti", "pe", "pi", "p"):
        values = getattr(derived, name)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite values in derived field {name!r}")
