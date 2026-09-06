"""Export a solver's end state and rebuild it as a later run's initial condition.

The payload written here is NOT a trajectory: it carries one instant, at full
precision, including every continuation cache, latch, accumulator and run-loop
controller value a bit-identical continuation needs. The inventory, with the
mutation site of each member and the justification for every deliberate
omission, is THIS module together with the restart state carried in
``solver.py``. The resume contract it implements is stated in
``_sim1d/NUMERICS.md``, section "Restart".

The format string is independent of the trajectory format: a restart payload
and a run result answer different questions and version separately.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from ..core.config import resolve_config


RESTART_VERSION = "sim1d-restart-v1"

#: Config keys whose value decides what the payload's members MEAN, as opposed
#: to merely how large a number is. A restart across a difference in any of
#: these is refused: the stored fields would be reinterpreted under a closure
#: they were not produced by. Everything outside this set is free to differ,
#: which is what lets a conducting-phase stage hand off to a differently
#: configured main arm.
STRUCTURAL_FLAG_KEYS = (
    "Plasma",
    "TwinCathode",
    "cathode_coupling",
    # The lit-area fraction rides the ``cathode`` group, and only when the
    # emitting-area closure is armed. Resuming across a change of this flag
    # would either drop an evolved fraction or leave an armed closure sitting
    # at its seed, so the structural check refuses instead.
    "cathode_emitting_area",
    "coverage_closure",
    "neutral_momentum",
    "neutral_two_zone",
    # En is a packed row, so the layout check already refuses a mismatch; the
    # flag is listed for the same reason the two above are, so the refusal
    # names the closure that changed rather than only its width.
    "neutral_energy",
    # The vessel node's V_cm and its charge ledger ride the ``circuit`` group,
    # and only when the node is armed. Resuming across a change of this flag
    # would either drop an evolved potential or leave one unread, so the
    # structural check refuses instead.
    "regime_vessel_node",
)
STRUCTURAL_PARAM_KEYS = (
    "cathode_sample_smoothing",
    "cathode_surface_model",
    "cathode_warming_model",
    "neutral_model",
    "phase_transition_mode",
)

#: Neutral models whose evolving state is a distribution function this payload
#: does not serialise. Combining one with a restart raises rather than silently
#: reseeding the kinetic half from a Maxwellian.
REFUSED_NEUTRAL_MODELS = ("kinetic", "kinetic_dvm")


def save_restart_state(path, sim):
    """Write ``sim``'s complete evolving state to a restart payload.

    ``sim`` is a :class:`LAPDSim1D` that has been run; the payload describes the
    instant ``sim.time``. Returns the written path.
    """
    payload = sim.restart_payload()
    path = Path(path)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = RESTART_VERSION
        h5.attrs["solver"] = "LAPDSim1D"
        h5.attrs["time"] = float(payload["time"])
        h5.attrs["cells"] = int(payload["cells"])
        h5.attrs["state_fields"] = json.dumps(list(payload["state_fields"]))
        h5.attrs["params_json"] = _json_dumps(payload["params"])
        h5.attrs["flags_json"] = _json_dumps(payload["flags"])
        h5.attrs["compiled_kernels"] = str(payload["compiled_kernels"])
        h5.create_dataset("y", data=np.asarray(payload["y"], dtype=float))
        for group_name in ("cathode", "circuit", "coverage", "triggers",
                           "ignition", "ledgers", "sample_ema", "run_loop"):
            _write_mapping(h5.create_group(group_name), payload[group_name])
    return path


def load_restart_state(path):
    """Read a :func:`save_restart_state` payload back into its mapping."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"restart_from points at a missing file: {path}")
    with h5py.File(path, "r") as h5:
        file_format = _decode(h5.attrs.get("format", ""))
        if file_format != RESTART_VERSION:
            raise ValueError(
                f"unsupported sim1d restart payload format {file_format!r}; "
                f"expected {RESTART_VERSION!r} (file: {path})"
            )
        payload = {
            "path": path,
            "time": float(h5.attrs["time"]),
            "cells": int(h5.attrs["cells"]),
            "state_fields": tuple(json.loads(_decode(h5.attrs["state_fields"]))),
            "params": json.loads(_decode(h5.attrs["params_json"])),
            "flags": json.loads(_decode(h5.attrs["flags_json"])),
            "compiled_kernels": _decode(h5.attrs["compiled_kernels"]),
            "y": np.asarray(h5["y"][()], dtype=float),
        }
        for group_name in ("cathode", "circuit", "coverage", "triggers",
                           "ignition", "ledgers", "sample_ema", "run_loop"):
            payload[group_name] = _read_mapping(h5[group_name])
    return payload


def check_restart_compatibility(payload, cells, state_fields, params, flags):
    """Raise ``ValueError`` unless ``payload`` describes this solver's problem.

    Compares the grid, the packed field layout, and the structural config keys
    (:data:`STRUCTURAL_PARAM_KEYS`, :data:`STRUCTURAL_FLAG_KEYS`). A mismatch in
    any of them means the stored members would be read under a closure that did
    not produce them, so the load refuses instead of proceeding partially.
    """
    problems = []
    if int(payload["cells"]) != int(cells):
        problems.append(
            f"cells: payload {int(payload['cells'])} != solver {int(cells)}"
        )
    if tuple(payload["state_fields"]) != tuple(state_fields):
        problems.append(
            f"packed state fields: payload {tuple(payload['state_fields'])} "
            f"!= solver {tuple(state_fields)}"
        )
    expected_len = int(payload["cells"]) * len(payload["state_fields"])
    if payload["y"].size != expected_len:
        problems.append(
            f"packed state length: payload holds {payload['y'].size} values, "
            f"its own header describes {expected_len}"
        )
    stored_params, stored_flags = resolve_config(
        params=payload["params"], flags=payload["flags"]
    )
    for key in STRUCTURAL_PARAM_KEYS:
        if stored_params.get(key) != params.get(key):
            problems.append(
                f"params[{key!r}]: payload {stored_params.get(key)!r} != "
                f"solver {params.get(key)!r}"
            )
    for key in STRUCTURAL_FLAG_KEYS:
        if bool(stored_flags.get(key)) != bool(flags.get(key)):
            problems.append(
                f"flags[{key!r}]: payload {stored_flags.get(key)!r} != "
                f"solver {flags.get(key)!r}"
            )
    if problems:
        raise ValueError(
            "restart payload is incompatible with this LAPDSim1D "
            f"configuration ({payload['path']}); a restart may change any "
            "config key EXCEPT the grid, the packed state layout, and the "
            "structural keys listed in results/restart.py. Mismatches: "
            + "; ".join(problems)
        )


def _write_mapping(group, mapping):
    """Write a flat mapping: arrays as datasets, scalars/strings/None as attrs."""
    for name, value in mapping.items():
        if value is None:
            group.attrs[f"{name}__none"] = True
        elif isinstance(value, np.ndarray):
            group.create_dataset(name, data=np.asarray(value, dtype=float))
        elif isinstance(value, str):
            group.attrs[name] = value
        elif isinstance(value, (bool, np.bool_)):
            group.attrs[f"{name}__bool"] = bool(value)
        elif isinstance(value, (int, np.integer)):
            group.attrs[f"{name}__int"] = int(value)
        else:
            group.attrs[name] = float(value)


def _read_mapping(group):
    """Invert :func:`_write_mapping`, restoring None/bool/int/float/str/array."""
    mapping = {name: np.asarray(dataset[()], dtype=float)
               for name, dataset in group.items()}
    for name, value in group.attrs.items():
        if name.endswith("__none"):
            mapping[name[: -len("__none")]] = None
        elif name.endswith("__bool"):
            mapping[name[: -len("__bool")]] = bool(value)
        elif name.endswith("__int"):
            mapping[name[: -len("__int")]] = int(value)
        elif isinstance(value, (bytes, str)):
            mapping[name] = _decode(value)
        else:
            mapping[name] = float(value)
    return mapping


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _json_dumps(value):
    return json.dumps(value, sort_keys=True, default=_json_default)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")
