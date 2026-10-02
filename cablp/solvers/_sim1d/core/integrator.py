import numpy as np

from .state import pack_state, unpack_state


def add_scaled_vector(y, rhs, scale):
    """Return y + scale * rhs for packed conservative state vectors."""
    return np.asarray(y, dtype=float) + scale * np.asarray(rhs, dtype=float)


def _call_rhs(rhs_func, y, time):
    """Evaluate ``rhs_func`` at ``y``, passing ``time`` only when supplied."""
    if time is None:
        return np.asarray(rhs_func(y), dtype=float)
    return np.asarray(rhs_func(y, time), dtype=float)


#: Weight with which each SSPRK2 floor call's additions enter the vector
#: ``ssprk2_step`` returns. The first stage's floored vector ``y1`` enters
#: ``y2_raw = 0.5*y0 + 0.5*(y1 + dt*k1)`` with coefficient 1/2, so an amount
#: the first floor call adds reaches the returned vector at half weight (its
#: effect through ``k1`` is a change of the right-hand side, booked by the
#: terms that make it up). The second floor call acts on the returned vector
#: itself, so its additions enter at full weight.
SSPRK2_FLOOR_WEIGHTS = {"ssprk_stage_1": 0.5, "ssprk_stage_2": 1.0}


def ssprk2_step(
    y0,
    dt,
    rhs_func,
    floor_func=None,
    time=None,
    raw_stage_func=None,
    weighted_floor_func=None,
):
    """Advance one explicit SSPRK2 step with stage-end floor enforcement.

    When ``time`` is given, the two Heun stages are evaluated at ``time`` and
    ``time + dt`` and ``rhs_func`` is called as ``rhs_func(y, stage_time)``.
    This keeps second-order accuracy for explicitly time-dependent forcing
    (e.g. the gas-puff schedule). When ``time`` is ``None`` the stage time is
    omitted and ``rhs_func`` is called as ``rhs_func(y)``, which freezes any
    such forcing at the step start and is only first-order accurate in it.

    Exactly one of ``floor_func`` and ``weighted_floor_func`` is given.
    ``floor_func(y)`` returns the floored vector. ``weighted_floor_func(y,
    weight)`` returns the same floored vector and is told the weight with
    which that call's additions enter the returned vector
    (``SSPRK2_FLOOR_WEIGHTS``), which is what a caller keeping a ledger of
    the floors' additions needs; the step it returns is the same either way.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive (got {dt})")
    if (floor_func is None) == (weighted_floor_func is None):
        raise ValueError(
            "ssprk2_step needs exactly one of floor_func and "
            "weighted_floor_func"
        )

    def _floor(y, stage):
        if weighted_floor_func is not None:
            return weighted_floor_func(y, SSPRK2_FLOOR_WEIGHTS[stage])
        return floor_func(y)

    y0 = np.asarray(y0, dtype=float)
    k0 = _call_rhs(rhs_func, y0, time)
    y1_raw = add_scaled_vector(y0, k0, dt)
    if raw_stage_func is not None:
        raw_stage_func(y1_raw, "ssprk_stage_1")
    y1 = _floor(y1_raw, "ssprk_stage_1")

    stage_time = None if time is None else float(time) + float(dt)
    k1 = _call_rhs(rhs_func, y1, stage_time)
    y2_raw = 0.5 * y0 + 0.5 * add_scaled_vector(y1, k1, dt)
    if raw_stage_func is not None:
        raw_stage_func(y2_raw, "ssprk_stage_2")
    return _floor(y2_raw, "ssprk_stage_2")


def floor_state_vector(
    y,
    cells,
    floors,
    ion_mass_g,
    neutral_momentum=None,
    neutral_two_zone=None,
    neutral_annulus_momentum=None,
    neutral_energy=None,
):
    """Apply density and temperature floors to a packed conservative vector.

    The optional-field hints resolve the packed-width ambiguity exactly as
    in ``unpack_state`` (a bare 6-field vector reads as ``M_n``); the solver
    passes its own flags.
    """
    from .state import apply_state_floors

    state = unpack_state(
        y,
        cells,
        neutral_momentum=neutral_momentum,
        neutral_two_zone=neutral_two_zone,
        neutral_annulus_momentum=neutral_annulus_momentum,
        neutral_energy=neutral_energy,
    )
    floored = apply_state_floors(state, floors=floors, ion_mass_g=ion_mass_g)
    return pack_state(floored)
