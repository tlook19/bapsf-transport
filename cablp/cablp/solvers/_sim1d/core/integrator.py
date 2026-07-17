import numpy as np

from .state import ConservativeState1D, pack_state, unpack_state


def add_scaled_state(state, rhs, scale):
    """Return state + scale * rhs for conservative state arrays."""
    return ConservativeState1D(
        n=state.n + scale * rhs.n,
        nn=state.nn + scale * rhs.nn,
        M=state.M + scale * rhs.M,
        Ee=state.Ee + scale * rhs.Ee,
        Ei=state.Ei + scale * rhs.Ei,
    )


def add_scaled_vector(y, rhs, scale):
    """Return y + scale * rhs for packed conservative state vectors."""
    return np.asarray(y, dtype=float) + scale * np.asarray(rhs, dtype=float)


def _call_rhs(rhs_func, y, time):
    """Evaluate ``rhs_func`` at ``y``, passing ``time`` only when supplied."""
    if time is None:
        return np.asarray(rhs_func(y), dtype=float)
    return np.asarray(rhs_func(y, time), dtype=float)


def ssprk2_step(y0, dt, rhs_func, floor_func, time=None):
    """Advance one explicit SSPRK2 step with stage-end floor enforcement.

    When ``time`` is given, the two Heun stages are evaluated at ``time`` and
    ``time + dt`` and ``rhs_func`` is called as ``rhs_func(y, stage_time)``.
    This keeps second-order accuracy for explicitly time-dependent forcing
    (e.g. the gas-puff schedule). When ``time`` is ``None`` the stage time is
    omitted and ``rhs_func`` is called as ``rhs_func(y)``, which freezes any
    such forcing at the step start and is only first-order accurate in it.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive (got {dt})")

    y0 = np.asarray(y0, dtype=float)
    k0 = _call_rhs(rhs_func, y0, time)
    y1 = floor_func(add_scaled_vector(y0, k0, dt))

    stage_time = None if time is None else float(time) + float(dt)
    k1 = _call_rhs(rhs_func, y1, stage_time)
    y2_raw = 0.5 * y0 + 0.5 * add_scaled_vector(y1, k1, dt)
    return floor_func(y2_raw)


def floor_state_vector(y, cells, floors, ion_mass_g):
    """Apply density and temperature floors to a packed conservative vector."""
    from .state import apply_state_floors

    state = unpack_state(y, cells)
    floored = apply_state_floors(state, floors=floors, ion_mass_g=ion_mass_g)
    return pack_state(floored)
