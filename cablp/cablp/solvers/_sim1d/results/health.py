from collections import Counter
from types import SimpleNamespace

import numpy as np


def summarize_result(result):
    """Return lightweight health diagnostics for a saved sim1d trajectory."""
    finite_fields = _finite_fields(result)
    plasma_inventory = _inventory(result.n, result.plasma_volume_cm3)
    neutral_inventory = _inventory(result.nn, result.neutral_volume_cm3)
    thermal_energy = _inventory(result.Ee + result.Ei, result.plasma_volume_cm3)
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
    )
    return {
        name: bool(np.all(np.isfinite(np.asarray(getattr(result, name)))))
        for name in names
        if hasattr(result, name)
    }


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
