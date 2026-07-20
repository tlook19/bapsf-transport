from dataclasses import dataclass

import numpy as np

from cablp.vars._cons import ev_to_erg


STATE_NAMES_1D = ("n", "nn", "M", "Ee", "Ei")
# The optional sixth field (NEUTRAL_MOMENTUM_PLAN.md): axial neutral momentum,
# present only when the `neutral_momentum` flag builds it. STATE_NAMES_1D
# stays the 5-field tuple because it anchors the historical packed layout,
# the golden fixture, and the HDF5 format.
NEUTRAL_MOMENTUM_NAME = "M_n"


@dataclass(frozen=True)
class ConservativeState1D:
    n: np.ndarray
    nn: np.ndarray
    M: np.ndarray
    Ee: np.ndarray
    Ei: np.ndarray
    # Axial neutral momentum [g cm^-2 s^-1] on the neutral volume; None means
    # the field is absent (the historical 5-field state). Every existing
    # 5-argument constructor call remains valid.
    M_n: np.ndarray | None = None


@dataclass(frozen=True)
class DerivedState1D:
    u: np.ndarray
    Te: np.ndarray
    Ti: np.ndarray
    pe: np.ndarray
    pi: np.ndarray
    p: np.ndarray


def pack_state(state, neutral_momentum=None):
    """Pack conservative state arrays into a flat solver vector.

    The vector is ``5*cells`` long historically and ``6*cells`` with the
    optional neutral-momentum field. ``neutral_momentum=None`` follows the
    state (pack ``M_n`` iff present); ``True`` forces the 6-field layout,
    padding a missing ``M_n`` with zeros (used when summing per-term RHS
    states, most of which do not touch neutral momentum); ``False`` forces
    the historical 5-field layout (and requires the state not to carry
    ``M_n``, so information is never silently dropped).
    """
    rows = [state.n, state.nn, state.M, state.Ee, state.Ei]
    include = state.M_n is not None if neutral_momentum is None else bool(neutral_momentum)
    if include:
        rows.append(
            state.M_n
            if state.M_n is not None
            else np.zeros_like(np.asarray(state.n, dtype=float))
        )
    elif state.M_n is not None:
        raise ValueError(
            "pack_state(neutral_momentum=False) would drop a present M_n field"
        )
    return np.vstack(rows).ravel()


def unpack_state(y, cells):
    """Unpack a flat solver vector into conservative state arrays.

    Infers the field count from the vector length: ``5*cells`` is the
    historical layout, ``6*cells`` carries the optional neutral momentum.
    """
    y = np.asarray(y, dtype=float)
    cells = int(cells)
    if y.size == len(STATE_NAMES_1D) * cells:
        arr = y.reshape((len(STATE_NAMES_1D), cells))
        return ConservativeState1D(*(row.copy() for row in arr))
    if y.size == (len(STATE_NAMES_1D) + 1) * cells:
        arr = y.reshape((len(STATE_NAMES_1D) + 1, cells))
        return ConservativeState1D(
            *(row.copy() for row in arr[:-1]), M_n=arr[-1].copy()
        )
    raise ValueError(
        f"state vector of size {y.size} does not match 5 or 6 fields of "
        f"{cells} cells"
    )


def conservative_from_primitives(n, nn, u, Te, Ti, ion_mass_g, un=None):
    """Build conservative variables from primitive CGS/eV quantities.

    ``un`` (neutral drift [cm/s]) attaches the optional neutral-momentum
    field; ``None`` keeps the historical 5-field state.
    """
    n = np.asarray(n, dtype=float)
    nn = np.asarray(nn, dtype=float)
    u = np.asarray(u, dtype=float)
    Te = np.asarray(Te, dtype=float)
    Ti = np.asarray(Ti, dtype=float)
    M = ion_mass_g * n * u
    Ee = 1.5 * n * Te * ev_to_erg
    Ei = 1.5 * n * Ti * ev_to_erg
    M_n = None
    if un is not None:
        M_n = ion_mass_g * nn * np.asarray(un, dtype=float)
    return ConservativeState1D(
        n=n.copy(), nn=nn.copy(), M=M, Ee=Ee, Ei=Ei, M_n=M_n
    )


def apply_state_floors(state, floors, ion_mass_g):
    """Return a conservative state with density and temperature floors applied.

    ``M_n`` has no floor and passes through unchanged (momentum may carry
    either sign); flooring ``nn`` upward leaves the momentum untouched, which
    slightly *reduces* the implied ``u_n`` there -- the conservative choice.
    """
    n = np.maximum(np.asarray(state.n, dtype=float), floors["n"])
    nn = np.maximum(np.asarray(state.nn, dtype=float), floors["nn"])
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    u = derived.u
    Te = np.maximum(derived.Te, floors["Te"])
    Ti = np.maximum(derived.Ti, floors["Ti"])
    floored = conservative_from_primitives(n, nn, u, Te, Ti, ion_mass_g)
    if state.M_n is None:
        return floored
    return ConservativeState1D(
        n=floored.n,
        nn=floored.nn,
        M=floored.M,
        Ee=floored.Ee,
        Ei=floored.Ei,
        M_n=np.asarray(state.M_n, dtype=float).copy(),
    )


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
    if state.M_n is not None and not np.all(np.isfinite(state.M_n)):
        raise ValueError("non-finite values in state field 'M_n'")
    if derived is None:
        return
    for name in ("u", "Te", "Ti", "pe", "pi", "p"):
        values = getattr(derived, name)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite values in derived field {name!r}")
