from dataclasses import dataclass

import numpy as np

from cablp.vars._cons import ev_to_erg


STATE_NAMES_1D = ("n", "nn", "M", "Ee", "Ei")
# The optional sixth field (NEUTRAL_MOMENTUM_PLAN.md): axial neutral momentum,
# present only when the `neutral_momentum` flag builds it. STATE_NAMES_1D
# stays the 5-field tuple because it anchors the historical packed layout,
# the golden fixture, and the HDF5 format.
NEUTRAL_MOMENTUM_NAME = "M_n"
# The optional annulus neutral density (NEUTRAL_TWOZONE_PLAN.md): present only
# when the `neutral_two_zone` flag builds it, and `nn` is then the COLUMN
# density. Packed after M_n in flag-introduction order.
NEUTRAL_ANNULUS_NAME = "nn_a"
# Optional annulus axial momentum for the kinetic-derived two-momentum
# reduction. It is packed last so every existing 5/6/7-field layout remains
# byte-for-byte unchanged.
NEUTRAL_ANNULUS_MOMENTUM_NAME = "M_n_a"


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
    # Annulus neutral density [cm^-3] on the annulus volume Vm - Vp; None
    # means the field is absent and `nn` keeps its chamber-mean meaning.
    nn_a: np.ndarray | None = None
    # Annulus axial neutral momentum [g cm^-2 s^-1] on Vm - Vp. Presence
    # requires both M_n (then the column momentum) and nn_a.
    M_n_a: np.ndarray | None = None


@dataclass(frozen=True)
class DerivedState1D:
    u: np.ndarray
    Te: np.ndarray
    Ti: np.ndarray
    pe: np.ndarray
    pi: np.ndarray
    p: np.ndarray


def pack_state(
    state,
    neutral_momentum=None,
    neutral_two_zone=None,
    neutral_annulus_momentum=None,
):
    """Pack conservative state arrays into a flat solver vector.

    The vector is ``5*cells`` long historically, plus one row per present
    optional field, packed in flag-introduction order: ``M_n`` (neutral
    momentum), then ``nn_a`` (annulus density). For each optional field the
    corresponding keyword follows the same contract: ``None`` follows the
    state (pack iff present); ``True`` forces the row, padding a missing
    field with zeros (used when summing per-term RHS states, most of which
    do not touch the optional fields); ``False`` forces it out and requires
    the state not to carry it, so information is never silently dropped.
    """
    rows = [state.n, state.nn, state.M, state.Ee, state.Ei]
    for name, value, flag in (
        (NEUTRAL_MOMENTUM_NAME, state.M_n, neutral_momentum),
        (NEUTRAL_ANNULUS_NAME, state.nn_a, neutral_two_zone),
        (
            NEUTRAL_ANNULUS_MOMENTUM_NAME,
            state.M_n_a,
            neutral_annulus_momentum,
        ),
    ):
        include = value is not None if flag is None else bool(flag)
        if include:
            rows.append(
                value
                if value is not None
                else np.zeros_like(np.asarray(state.n, dtype=float))
            )
        elif value is not None:
            raise ValueError(
                f"pack_state({name}=False) would drop a present "
                f"{name} field"
            )
    return np.vstack(rows).ravel()


def unpack_state(
    y,
    cells,
    neutral_momentum=None,
    neutral_two_zone=None,
    neutral_annulus_momentum=None,
):
    """Unpack a flat solver vector into conservative state arrays.

    The optional fields make bare width inference ambiguous at 6 fields
    (``5 + M_n`` vs ``5 + nn_a``), so the keywords declare which optional
    fields the layout carries (the solver passes its own flags). ``None``
    means "infer": a bare 6-field vector keeps its HISTORICAL meaning of
    ``M_n`` — every pre-two-zone call site reads exactly as before — so a
    two-zone-only vector is only reachable by passing
    ``neutral_two_zone=True``. 5- and 7-field vectors are unambiguous.
    """
    y = np.asarray(y, dtype=float)
    cells = int(cells)
    base = len(STATE_NAMES_1D)
    if y.size % cells:
        raise ValueError(
            f"state vector of size {y.size} is not a multiple of "
            f"{cells} cells"
        )
    fields = y.size // cells
    candidates = [
        (has_mn, has_2z, has_mna)
        for has_mn in (False, True)
        for has_2z in (False, True)
        for has_mna in (False, True)
        if (neutral_momentum is None or has_mn == bool(neutral_momentum))
        and (neutral_two_zone is None or has_2z == bool(neutral_two_zone))
        and (
            neutral_annulus_momentum is None
            or has_mna == bool(neutral_annulus_momentum)
        )
        and (not has_mna or (has_mn and has_2z))
        and base + has_mn + has_2z + has_mna == fields
    ]
    if not candidates:
        raise ValueError(
            f"state vector of size {y.size} does not match the declared "
            f"optional fields for {cells} cells"
        )
    if len(candidates) > 1:
        # Only reachable at 6 fields with both hints None: the historical
        # reading is M_n.
        candidates = [(True, False, False)]
    has_mn, has_2z, has_mna = candidates[0]
    arr = y.reshape((fields, cells))
    optional = {}
    row = base
    if has_mn:
        optional["M_n"] = arr[row].copy()
        row += 1
    if has_2z:
        optional["nn_a"] = arr[row].copy()
        row += 1
    if has_mna:
        optional["M_n_a"] = arr[row].copy()
    return ConservativeState1D(
        *(r.copy() for r in arr[:base]), **optional
    )


def conservative_from_primitives(
    n,
    nn,
    u,
    Te,
    Ti,
    ion_mass_g,
    un=None,
    nn_a=None,
    un_a=None,
):
    """Build conservative variables from primitive CGS/eV quantities.

    ``un`` (neutral drift [cm/s]) attaches the optional neutral-momentum
    field; ``nn_a`` (annulus density [cm^-3]) attaches the optional
    two-zone field (``nn`` is then the column density); ``None`` keeps
    each absent.
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
    if un_a is not None and nn_a is None:
        raise ValueError("un_a requires nn_a")
    return ConservativeState1D(
        n=n.copy(),
        nn=nn.copy(),
        M=M,
        Ee=Ee,
        Ei=Ei,
        M_n=M_n,
        nn_a=None if nn_a is None else np.asarray(nn_a, dtype=float).copy(),
        M_n_a=(
            None
            if un_a is None
            else ion_mass_g
            * np.asarray(nn_a, dtype=float)
            * np.asarray(un_a, dtype=float)
        ),
    )


def apply_state_floors(state, floors, ion_mass_g):
    """Return a conservative state with density and temperature floors applied.

    ``M_n`` has no floor and passes through unchanged (momentum may carry
    either sign); flooring ``nn`` upward leaves the momentum untouched, which
    slightly *reduces* the implied ``u_n`` there -- the conservative choice.
    ``nn_a`` is a density and takes the ``nn`` floor.
    """
    n = np.maximum(np.asarray(state.n, dtype=float), floors["n"])
    nn = np.maximum(np.asarray(state.nn, dtype=float), floors["nn"])
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    u = derived.u
    Te = np.maximum(derived.Te, floors["Te"])
    Ti = np.maximum(derived.Ti, floors["Ti"])
    floored = conservative_from_primitives(n, nn, u, Te, Ti, ion_mass_g)
    if state.M_n is None and state.nn_a is None and state.M_n_a is None:
        return floored
    return ConservativeState1D(
        n=floored.n,
        nn=floored.nn,
        M=floored.M,
        Ee=floored.Ee,
        Ei=floored.Ei,
        M_n=(
            None
            if state.M_n is None
            else np.asarray(state.M_n, dtype=float).copy()
        ),
        nn_a=(
            None
            if state.nn_a is None
            else np.maximum(
                np.asarray(state.nn_a, dtype=float), floors["nn"]
            )
        ),
        M_n_a=(
            None
            if state.M_n_a is None
            else np.asarray(state.M_n_a, dtype=float).copy()
        ),
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
    if state.nn_a is not None and not np.all(np.isfinite(state.nn_a)):
        raise ValueError("non-finite values in state field 'nn_a'")
    if state.M_n_a is not None and not np.all(np.isfinite(state.M_n_a)):
        raise ValueError("non-finite values in state field 'M_n_a'")
    if derived is None:
        return
    for name in ("u", "Te", "Ti", "pe", "pi", "p"):
        values = getattr(derived, name)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite values in derived field {name!r}")
