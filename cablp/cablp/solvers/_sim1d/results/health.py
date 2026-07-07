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
