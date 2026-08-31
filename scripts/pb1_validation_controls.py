"""[perf-batch-1] item 3 -- negative controls for the validation happy-path skips.

Item 3 stops re-scanning the unpacked state fields for non-finiteness once the
PACKED candidate has already been proved finite (``unpack_state`` returns
``.copy()`` of rows of ``y``, so a finite ``y`` cannot yield a non-finite
field). The saving is only legitimate if every raise and every rejection
reason still fires with an IDENTICAL message and an identical detail payload,
including on the paths where the packed vector is NOT clean.

This prints a canonical, diffable transcript of six controls across both call
sites -- ``validate_raw_stage`` and ``LAPDSim1D._step_rejection_info``:

  A  raw stage, NaN injected into the packed candidate
  B  raw stage, negative density
  C  raw stage, negative energy
  D  rejection info, NaN in the candidate (the per-field branch the skip must
     still take when the packed pass did NOT prove finiteness)
  E  rejection info, negative density
  F  rejection info, a step-fraction rejection (no bad values at all)

Run it before and after the change and diff the two transcripts; the gate is
that they are byte-identical.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cablp.solvers._sim1d import LAPDSim1D, default_config  # noqa: E402
from cablp.solvers._sim1d.core.state import state_field_names  # noqa: E402
from cablp.solvers._sim1d.core.validation import (  # noqa: E402
    _RawStageError,
    validate_raw_stage,
)

DT = 1.0e-10


def _config():
    params, flags = default_config()
    params.update(
        {
            "nx": 8,
            "nx_gap": 2,
            "ne0": 2.0e10,
            "nn0": 2.0e12,
            "Te0": 1.0,
            "Ti0": 0.5,
            "phase_transition_mode": "scheduled",
            "tau_prebreakdown": 0.0,
            "tau_breakdown": 0.0,
            "tau_discharge": 1.0e-6,
        }
    )
    flags.update(
        {
            "active_plasma_topology": True,
            "cathode_coupling": False,
            "neutral_prebreakdown": False,
            "neutral_equilibration": False,
            "launch_plasma_after_equilibration": False,
            "raw_stage_validation": True,
        }
    )
    return params, flags


def _canonical(obj):
    """Render a detail payload deterministically, floats by exact repr."""
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return repr(obj.item())
    return obj


def _emit(label, payload):
    print(f"--- {label}")
    print(json.dumps(_canonical(payload), indent=2, sort_keys=True))


def _row_slice(sim, field):
    cells = sim.geometry.cells
    row = state_field_names(sim.state).index(field)
    return slice(row * cells, (row + 1) * cells)


def _raw_stage_control(label, mutate):
    """Drive validate_raw_stage on a mutated candidate and report the raise."""
    params, flags = _config()
    sim = LAPDSim1D(params, flags)
    y = np.asarray(sim._y, dtype=float).copy()
    mutate(sim, y)
    try:
        validate_raw_stage(y, "pb1_control_stage", sim._unpack)
    except _RawStageError as exc:
        _emit(
            label,
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": exc.stage,
                "reason": exc.reason,
                "detail": exc.detail,
            },
        )
    else:
        _emit(label, {"raised": False})


def _rejection_control(label, mutate, overrides=None):
    """Drive _step_rejection_info on a mutated candidate and report the verdict."""
    params, flags = _config()
    params.update(overrides or {})
    sim = LAPDSim1D(params, flags)
    y0 = np.asarray(sim._y, dtype=float).copy()
    y1 = y0.copy()
    mutate(sim, y1)
    attempt = SimpleNamespace(
        y=y1, raw_rejection_reason="", raw_rejection_detail=None
    )
    reason, detail = sim._step_rejection_info(attempt, y0=y0)
    _emit(label, {"reason": reason, "detail": detail})


def main():
    _raw_stage_control(
        "A raw_stage nonfinite (NaN in packed y)",
        lambda sim, y: y.__setitem__(_row_slice(sim, "n"), np.nan),
    )
    _raw_stage_control(
        "B raw_stage negative density",
        lambda sim, y: y.__setitem__(
            _row_slice(sim, "nn"), -np.abs(y[_row_slice(sim, "nn")]) - 1.0
        ),
    )
    _raw_stage_control(
        "C raw_stage negative energy",
        lambda sim, y: y.__setitem__(
            _row_slice(sim, "Ee"), -np.abs(y[_row_slice(sim, "Ee")]) - 1.0
        ),
    )
    _rejection_control(
        "D rejection_info nonfinite (NaN in candidate)",
        lambda sim, y: y.__setitem__(_row_slice(sim, "Ei"), np.inf),
    )
    _rejection_control(
        "E rejection_info negative density",
        lambda sim, y: y.__setitem__(
            _row_slice(sim, "n"), -np.abs(y[_row_slice(sim, "n")]) - 1.0
        ),
    )
    _rejection_control(
        "F rejection_info forced step-fraction rejection",
        lambda sim, y: y.__setitem__(
            _row_slice(sim, "n"), y[_row_slice(sim, "n")] * 1.0e6
        ),
        overrides={"max_density_step_fraction": 0.5},
    )
    _rejection_control(
        "G rejection_info accepted step (nothing wrong)",
        lambda sim, y: None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
