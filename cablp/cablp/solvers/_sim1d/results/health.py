from collections import Counter
from types import SimpleNamespace

import numpy as np


def summarize_result(result):
    """Return lightweight health diagnostics for a saved sim1d trajectory."""
    finite_fields = _finite_fields(result)
    plasma_active = _plasma_active(result)
    plasma_volume = np.asarray(result.plasma_volume_cm3, dtype=float) * plasma_active
    plasma_inventory = _inventory(result.n, plasma_volume)
    if hasattr(result, "nn_a"):
        column_volume = np.asarray(result.plasma_volume_cm3, dtype=float)
        annulus_volume = (
            np.asarray(result.neutral_volume_cm3, dtype=float) - column_volume
        )
        neutral_column_inventory = _inventory(result.nn, column_volume)
        neutral_annulus_inventory = _inventory(result.nn_a, annulus_volume)
        neutral_inventory = (
            neutral_column_inventory + neutral_annulus_inventory
        )
    else:
        neutral_column_inventory = _inventory(
            result.nn, result.neutral_volume_cm3
        )
        neutral_annulus_inventory = np.zeros_like(neutral_column_inventory)
        neutral_inventory = neutral_column_inventory
    thermal_energy = _inventory(result.Ee + result.Ei, plasma_volume)
    neutral_momentum_column_inventory = np.zeros_like(neutral_inventory)
    neutral_momentum_annulus_inventory = np.zeros_like(neutral_inventory)
    if hasattr(result, "M_n"):
        if hasattr(result, "M_n_a"):
            neutral_momentum_column_inventory = _inventory(
                result.M_n, result.plasma_volume_cm3
            )
            neutral_momentum_annulus_inventory = _inventory(
                result.M_n_a,
                np.asarray(result.neutral_volume_cm3, dtype=float)
                - np.asarray(result.plasma_volume_cm3, dtype=float),
            )
        else:
            neutral_momentum_column_inventory = _inventory(
                result.M_n, result.neutral_volume_cm3
            )
    neutral_momentum_inventory = (
        neutral_momentum_column_inventory
        + neutral_momentum_annulus_inventory
    )
    constraint_counts = Counter(
        diag.active_constraint for diag in getattr(result, "diagnostics", ())
    )
    diagnostic_phase_counts = Counter(
        diag.phase
        for diag in getattr(result, "diagnostics", ())
        if getattr(diag, "phase", "")
    )
    step_cap_counts = Counter(
        diag.step_cap
        for diag in getattr(result, "diagnostics", ())
        if getattr(diag, "step_cap", "")
    )
    rejection_reason_counts = Counter(
        diag.rejection_reason
        for diag in getattr(result, "diagnostics", ())
        if getattr(diag, "rejection_reason", "")
    )
    retry_counts = np.asarray(
        [
            getattr(diag, "retry_count", 0)
            for diag in getattr(result, "diagnostics", ())
        ],
        dtype=int,
    )
    accepted_dt = np.asarray(
        [
            diag.accepted_dt
            for diag in getattr(result, "diagnostics", ())
            if np.isfinite(getattr(diag, "accepted_dt", np.nan))
        ],
        dtype=float,
    )
    dt_min_clamp_summary = _dt_min_clamp_summary(result)
    phase_event_summary = _phase_event_summary(result)
    rejection_event_summary = _timestep_rejection_event_summary(result)
    current_trigger_summary = _current_trigger_sample_summary(result)
    return SimpleNamespace(
        finite=all(finite_fields.values()),
        finite_fields=finite_fields,
        samples=len(result.time),
        steps=int(result.steps),
        final_time=float(result.final_time),
        n_min=float(np.min(result.n)) if result.n.size else np.nan,
        n_max=float(np.max(result.n)) if result.n.size else np.nan,
        nn_min=float(np.min(result.nn)) if result.nn.size else np.nan,
        nn_max=float(np.max(result.nn)) if result.nn.size else np.nan,
        Te_min=float(np.min(result.Te)) if result.Te.size else np.nan,
        Te_max=float(np.max(result.Te)) if result.Te.size else np.nan,
        Ti_min=float(np.min(result.Ti)) if result.Ti.size else np.nan,
        Ti_max=float(np.max(result.Ti)) if result.Ti.size else np.nan,
        plasma_inventory=plasma_inventory,
        neutral_inventory=neutral_inventory,
        neutral_column_inventory=neutral_column_inventory,
        neutral_annulus_inventory=neutral_annulus_inventory,
        neutral_momentum_inventory=neutral_momentum_inventory,
        neutral_momentum_column_inventory=neutral_momentum_column_inventory,
        neutral_momentum_annulus_inventory=neutral_momentum_annulus_inventory,
        total_particle_inventory=plasma_inventory + neutral_inventory,
        thermal_energy=thermal_energy,
        plasma_inventory_relative_drift=_relative_drift(plasma_inventory),
        neutral_inventory_relative_drift=_relative_drift(neutral_inventory),
        total_particle_inventory_relative_drift=_relative_drift(
            plasma_inventory + neutral_inventory
        ),
        thermal_energy_relative_drift=_relative_drift(thermal_energy),
        phase_counts=_value_counts(getattr(result, "phase", ())),
        diagnostic_phase_counts=dict(sorted(diagnostic_phase_counts.items())),
        accepted_dt_min=float(np.min(accepted_dt)) if accepted_dt.size else np.nan,
        accepted_dt_max=float(np.max(accepted_dt)) if accepted_dt.size else np.nan,
        step_cap_counts=dict(sorted(step_cap_counts.items())),
        retrying_step_count=int(np.count_nonzero(retry_counts))
        if retry_counts.size
        else 0,
        total_retry_count=int(np.sum(retry_counts)) if retry_counts.size else 0,
        max_retry_count=int(np.max(retry_counts)) if retry_counts.size else 0,
        rejection_reason_counts=dict(sorted(rejection_reason_counts.items())),
        phase_event_count=phase_event_summary["count"],
        phase_event_phase_counts=phase_event_summary["phase_counts"],
        phase_event_reason_counts=phase_event_summary["reason_counts"],
        last_phase_event=phase_event_summary["last_event"],
        timestep_rejection_event_count=rejection_event_summary["count"],
        timestep_rejection_reason_counts=rejection_event_summary["reason_counts"],
        last_timestep_rejection_event=rejection_event_summary["last_event"],
        current_trigger_sample_count=current_trigger_summary["count"],
        last_current_trigger_sample=current_trigger_summary["last_sample"],
        phase_switch_fractions=_phase_switch_fractions(result),
        cathode_diagnostic_fractions=_cathode_diagnostic_fractions(result),
        constraint_counts=dict(sorted(constraint_counts.items())),
        dt_min_clamped_step_count=dt_min_clamp_summary["clamped"],
        dt_min_hard_zero_step_count=dt_min_clamp_summary["hard_zero"],
        max_consecutive_dt_min_clamped_steps=dt_min_clamp_summary["max_run"],
        # Accepted steps BELOW dt_min: the post-clamp step caps can shrink a
        # step under the floor, which the clamp census above cannot see.
        below_dt_min_step_count=dt_min_clamp_summary["below_dt_min"],
        below_dt_min_known=dt_min_clamp_summary["below_dt_min_known"],
        below_dt_min_min_accepted_dt=dt_min_clamp_summary[
            "below_dt_min_min_accepted"
        ],
        below_dt_min_step_cap_counts=dt_min_clamp_summary[
            "below_dt_min_step_caps"
        ],
        dvm_arm_configured=_dvm_arm_configured(result),
        dvm_transfer_ledger_census=_dvm_ledger_census_summary(result),
    )


def _finite_fields(result):
    names = (
        "time",
        "y",
        "n",
        "nn",
        "M",
        "Ee",
        "Ei",
        "u",
        "Te",
        "Ti",
        "pe",
        "pi",
        "p",
        "M_n",
        "u_n",
        "nn_a",
        "M_n_a",
        "u_n_a",
        "En",
        "Tn",
    )
    return {
        name: bool(np.all(np.isfinite(np.asarray(getattr(result, name)))))
        for name in names
        if hasattr(result, name)
    }


def _plasma_active(result):
    if hasattr(result, "plasma_active"):
        return np.asarray(result.plasma_active, dtype=bool)
    roles = np.asarray(getattr(result, "cell_role", ()), dtype=object)
    if roles.size:
        return ~np.isin(roles, ("plenum", "obstruction"))
    return np.ones(np.asarray(result.plasma_volume_cm3).shape, dtype=bool)


def _inventory(density, volume):
    density = np.asarray(density, dtype=float)
    volume = np.asarray(volume, dtype=float)
    if density.size == 0:
        return np.empty((0,), dtype=float)
    return np.sum(density * volume[None, :], axis=1)


def _relative_drift(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    reference = values[0]
    scale = max(abs(reference), 1.0)
    return float((values[-1] - reference) / scale)


def _value_counts(values):
    values = np.asarray(values, dtype=object)
    if values.size == 0:
        return {}
    counts = Counter(str(value) for value in values)
    return dict(sorted(counts.items()))


def _phase_switch_fractions(result):
    names = {
        "cathode_enabled": "phase_cathode_enabled",
        "gas_puff_enabled": "phase_gas_puff_enabled",
        "floating": "phase_floating",
    }
    return {
        summary_name: _mean_fraction(getattr(result, field_name))
        for summary_name, field_name in names.items()
        if hasattr(result, field_name)
    }


def _dt_min_clamp_summary(result):
    """Census the dt_min clamp over a trajectory's per-step diagnostics.

    Reports how many steps were clamped up to ``dt_min``, how many of those
    had a bound request of exactly zero (``dt_raw == 0.0``, the drained
    floor-pinned signature), and the longest CONSECUTIVE clamped run -- the
    quantity that separates a self-releasing clamp episode from a lock.

    Back-compatibility: results written before 2026-08-05 carry no
    ``clamped_to_dt_min`` field, but their ``active_constraint`` was
    OVERWRITTEN with ``"dt_min"`` on exactly the clamped steps, so that label
    is read as the clamp flag. The fallback is unambiguous because the current
    solver never emits that label. ``dt_raw`` has no such fallback, so the
    hard-zero count reads 0 for those older files.
    """
    diagnostics = getattr(result, "diagnostics", ())
    clamped = np.asarray(
        [
            bool(getattr(diag, "clamped_to_dt_min", 0.0))
            or getattr(diag, "active_constraint", "") == "dt_min"
            for diag in diagnostics
        ],
        dtype=bool,
    )
    hard_zero = int(
        sum(
            1
            for diag in diagnostics
            if getattr(diag, "dt_raw", np.nan) == 0.0
        )
    )
    below = _below_dt_min_summary(result, diagnostics)
    if not clamped.size or not np.any(clamped):
        return {"clamped": 0, "hard_zero": hard_zero, "max_run": 0, **below}
    edges = np.diff(np.concatenate(([False], clamped, [False])).astype(np.int8))
    run_lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return {
        "clamped": int(np.count_nonzero(clamped)),
        "hard_zero": hard_zero,
        "max_run": int(np.max(run_lengths)),
        **below,
    }


def _below_dt_min_summary(result, diagnostics):
    """Census accepted steps that landed BELOW ``dt_min``.

    A separate category from the clamp count, and deliberately not folded into
    it: these are the opposite fact. The clamp lifts a bound's request UP to
    ``dt_min`` inside ``suggest_timestep``; the step caps (``dt_growth``,
    ``save_time``, ``phase_boundary``, ``t_end``) are applied AFTERWARDS in the
    run loop and can only shrink the step, so an accepted step can end up
    strictly below ``dt_min`` while ``clamped_to_dt_min`` reads either way.
    Such a step was invisible to this census, which is why a production run
    (K6d) accepted dt = 9.239e-11 against a configured ``dt_min`` of 1e-10 and
    nothing recorded it.

    ``step_caps`` names which cap was responsible, which is the whole
    diagnostic value: it distinguishes a benign landing on ``t_end`` or a save
    time from a ``dt_growth`` ramp genuinely driving the step under the floor.

    Returns zero counts and NaN when the run records no ``dt_min`` (results
    carrying no params). That is "cannot tell", and is reported as such rather
    than as an absence of below-floor steps.
    """
    params = getattr(result, "params", None)
    dt_min = None
    if isinstance(params, dict) and "dt_min" in params:
        try:
            candidate = float(params["dt_min"])
        except (TypeError, ValueError):
            candidate = np.nan
        if np.isfinite(candidate) and candidate > 0.0:
            dt_min = candidate
    if dt_min is None:
        return {
            "below_dt_min": 0,
            "below_dt_min_known": False,
            "below_dt_min_min_accepted": np.nan,
            "below_dt_min_step_caps": {},
        }
    accepted = np.asarray(
        [getattr(diag, "accepted_dt", np.nan) for diag in diagnostics],
        dtype=float,
    )
    below = np.isfinite(accepted) & (accepted < dt_min)
    caps = _value_counts(
        [
            getattr(diag, "step_cap", "")
            for diag, is_below in zip(diagnostics, below)
            if is_below
        ]
    )
    return {
        "below_dt_min": int(np.count_nonzero(below)),
        "below_dt_min_known": True,
        "below_dt_min_min_accepted": (
            float(np.min(accepted[below])) if np.any(below) else np.nan
        ),
        "below_dt_min_step_caps": caps,
    }


def _dvm_arm_configured(result):
    """Return whether the run built the K2d DVM arm, or None when unknowable.

    Read from the run's own ``neutral_model`` parameter. None is returned for
    a result carrying no params at all -- that is "cannot tell", never "no".
    """
    params = getattr(result, "params", None)
    if not isinstance(params, dict) or "neutral_model" not in params:
        return None
    return str(params["neutral_model"]) == "kinetic_dvm"


def _dvm_ledger_census_summary(result):
    """Return the quotable K2d transfer-ledger census, or None if not recorded.

    None is the UNQUOTABLE reading and must never be presented as zero: every
    DVM artifact written before the census was persisted (2026-08-05) carries
    no ledger group at all, and a moment-model run has no ledger to carry.
    Pair this with ``dvm_arm_configured`` to tell those two apart -- an arm
    that ran with no census recorded is the case a report has to flag.

    Scalars only: the per-cell arrays stay on ``result.dvm_transfer_ledger``
    for anyone localizing the debt. ``*_total`` are volume-integrated (erg for
    the Ei channel, g cm/s for M); ``*_max_abs`` are per-cell densities. The
    ``ion_*`` entries are the particle handshake and are already totals in
    PARTICLES; ``ion_residual_rel`` is the coupled system's
    particle-conservation residual. Artifacts written before 2026-08-06 have
    the transfer channels but not the particle one, so read every name as
    optional -- as this function already does.
    """
    census = getattr(result, "dvm_transfer_ledger", None)
    if not census:
        return None
    names = (
        "engaged",
        "relax_steps",
        "relax_limited_steps",
        "limited_cells",
        "Ei_debt_total",
        "Ei_debt_max_abs",
        "Ei_booked_total",
        "Ei_applied_total",
        "Ei_residual_rel",
        "M_debt_total",
        "M_debt_max_abs",
        "M_booked_total",
        "M_applied_total",
        "M_residual_rel",
        "ion_booked_total",
        "ion_removed_total",
        "ion_debt_total",
        "ion_debt_max_abs",
        "ion_residual_rel",
        "ion_shortfall_updates",
    )
    return {name: census[name] for name in names if name in census}


def _phase_event_summary(result):
    phase_events = getattr(result, "phase_events", {})
    times = np.asarray(phase_events.get("time", ()), dtype=float)
    phases = np.asarray(phase_events.get("phase", ()), dtype=object)
    reasons = np.asarray(phase_events.get("reason", ()), dtype=object)
    if times.size == 0:
        return {
            "count": 0,
            "phase_counts": {},
            "reason_counts": {},
            "last_event": None,
        }
    last_index = times.size - 1
    return {
        "count": int(times.size),
        "phase_counts": _value_counts(phases),
        "reason_counts": _value_counts(reasons),
        "last_event": {
            "time": float(times[last_index]),
            "phase": str(phases[last_index]),
            "reason": str(reasons[last_index]),
        },
    }


def _timestep_rejection_event_summary(result):
    events = getattr(result, "timestep_rejection_events", {})
    times = np.asarray(events.get("time", ()), dtype=float)
    attempted_dt = np.asarray(events.get("attempted_dt", ()), dtype=float)
    retry_index = np.asarray(events.get("retry_index", ()), dtype=float)
    reasons = np.asarray(events.get("reason", ()), dtype=object)
    phases = np.asarray(events.get("phase", ()), dtype=object)
    constraints = np.asarray(events.get("active_constraint", ()), dtype=object)
    if times.size == 0:
        return {
            "count": 0,
            "reason_counts": {},
            "last_event": None,
        }
    last_index = times.size - 1
    return {
        "count": int(times.size),
        "reason_counts": _value_counts(reasons),
        "last_event": {
            "time": float(times[last_index]),
            "attempted_dt": float(attempted_dt[last_index]),
            "retry_index": int(retry_index[last_index]),
            "reason": str(reasons[last_index]),
            "phase": str(phases[last_index]),
            "active_constraint": str(constraints[last_index]),
        },
    }


def _current_trigger_sample_summary(result):
    samples = getattr(result, "current_trigger_samples", {})
    times = np.asarray(samples.get("time", ()), dtype=float)
    currents = np.asarray(samples.get("I_tot", ()), dtype=float)
    if times.size == 0:
        return {
            "count": 0,
            "last_sample": None,
        }
    last_index = times.size - 1
    return {
        "count": int(times.size),
        "last_sample": {
            "time": float(times[last_index]),
            "I_tot": float(currents[last_index]),
        },
    }


def _cathode_diagnostic_fractions(result):
    diagnostics = getattr(result, "cathode_diagnostics", {})
    names = (
        "configured",
        "phase_enabled",
        "rhs_enabled",
        "solve_enabled",
        "floating",
        "has_solution",
        "has_twin_solution",
    )
    return {
        name: _mean_fraction(diagnostics[name])
        for name in names
        if name in diagnostics
    }


def _mean_fraction(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.mean(values))
