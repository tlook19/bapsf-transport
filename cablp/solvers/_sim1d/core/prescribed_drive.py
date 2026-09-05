"""Measured discharge drive for ``cathode_solver_model="prescribed_measured"``.

The current-driven cathode PREDICTS the drive: Richardson emission at the
cathode's surface temperature sets what the sheath can carry, and the bank
loop integrates ``L dI/dt = V_src - I*R - V_dis(I)`` to find the current.
The prescribed-measured mode does not predict it. The operator set each rung's
discharge current by hand -- the heater was raised until the discharge reached
a target power and then held -- so the drive LEVEL is a machine input, and this
module supplies it from the rung's own measured overlay trace instead.

**The trace file.** ``cathode_prescribed_trace_path`` names an ``.npz`` in the
ES-overlay schema, i.e. exactly the artifacts
``scripts/data/es{N}_sim1d_overlay.npz`` the scorer and the circuit fits
already read. Three arrays are required, and a file missing any of them is
refused rather than partially read:

===============================================  =====  =============================
key                                              unit   what it is
===============================================  =====  =============================
``discharge_time_ms``                            ms     the trace's own clock
``discharge_current_mean_a``                     A      shot-mean discharge current
``discharge_voltage_positive_mean_v``            V      shot-mean discharge voltage
===============================================  =====  =============================

The voltage array is the overlay's POSITIVE convention (the overlay records
``discharge_voltage_sign`` = "positive overlay is -1 times raw cathode-anode
voltage"), which is the sign the model's own ``V_dis`` carries, so it is read
as-is.

**The two clocks.** The overlay's ``discharge_time_ms`` is referenced to the
start of the main discharge; the model's ``result.time`` is seconds since the
simulation began and reaches the main discharge only after its pre-breakdown
and breakdown phases. ``scripts/score/compare_sim1d_es1.py`` reconciles the two
post-hoc by shifting the MODEL clock, ``t_ms = (t_model - origin)*1e3`` with
``origin`` the model time of the first ``main_discharge`` frame. This module
reuses that convention with the origin supplied as configuration, because a
solver stepping forward cannot read a quantity a saved trajectory is searched
for:

    t_trace_ms = (t_model_s - cathode_prescribed_t0_s) * 1e3

``cathode_prescribed_t0_s`` is therefore the model time that the trace's own
``t = 0`` names. It is REQUIRED: there is no defensible default, and a guessed
origin would slide the whole measured drive against the column.

**The foot.** Prescribing from ``t = 0`` would impose the plateau current on a
column that has not broken down, where the sheath cannot carry it and the solve
sits at the ``cathode_phi_c_cap_V`` ceiling for the whole build leg. So the
prescribed drive begins at ``cathode_prescribed_start_s`` (also REQUIRED) and
the CALIBRATED cathode -- Richardson emission, the bank loop, the warming model,
exactly as configured -- runs everything before it. A run in this mode therefore
needs a VALID CALIBRATED CONFIGURATION as well as a trace: the foot is a real
current-driven discharge and every key it reads must still be set.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The value of ``cathode_solver_model`` this module serves.
PRESCRIBED_MEASURED = "prescribed_measured"

#: The three ``input_dict`` keys that carry the measured drive. They are
#: refused under any other ``cathode_solver_model`` (a trace nothing reads is
#: exactly the silent inert control the config surface forbids), and required
#: under this one.
PRESCRIBED_DRIVE_KEYS = (
    "cathode_prescribed_trace_path",
    "cathode_prescribed_t0_s",
    "cathode_prescribed_start_s",
)

#: Array names read out of the overlay ``.npz``, in (time, current, voltage)
#: order. Stated once so the loader and its refusal message cannot disagree
#: about which columns a trace file must carry.
TRACE_TIME_KEY = "discharge_time_ms"
TRACE_CURRENT_KEY = "discharge_current_mean_a"
TRACE_VOLTAGE_KEY = "discharge_voltage_positive_mean_v"
TRACE_REQUIRED_KEYS = (TRACE_TIME_KEY, TRACE_CURRENT_KEY, TRACE_VOLTAGE_KEY)

#: The CALIBRATED cathode's own keys: the emission constant, the surface
#: temperature the emission model reads, the bank loop, and the heater/warming
#: package. The prescribed drive reads none of them -- and yet a run in this
#: mode normally still NEEDS them, because its foot is a real current-driven
#: discharge (see the module docstring). So the refusal below is conditional on
#: there being no foot: with the hand-off at or before the model clock's own
#: origin, the calibrated cathode never runs at all, these keys are read by
#: nothing for the whole run, and a non-default value among them is exactly the
#: silent inert control the config surface forbids.
CALIBRATED_ONLY_KEYS = (
    "C_R",
    "cathode_Ts_base_K",
    "V_bank",
    "C_bank_F",
    "L_parasitic_H",
    "cathode_warming_model",
    "cathode_heat_capacity_J_per_K",
    "cathode_conduction_W_per_K",
    "cathode_emissivity",
)

#: Relative hand-off discontinuity above which the switch is announced as a
#: JUMP rather than a hand-off. Not a tolerance and not a gate: nothing is
#: smoothed, clipped or rejected at any size: the number only decides whether
#: the switch line is accompanied by a loud one. A calibrated cathode that
#: already reproduces its rung's drive lands well inside it.
HANDOFF_JUMP_WARN_FRACTION = 0.10


@dataclass(frozen=True)
class PrescribedDrivePoint:
    """The measured drive one step is integrated at.

    Two numbers, frozen for the step: the discharge current [A] and the
    discharge voltage [V]. It exists as a type rather than a tuple because it
    is the PRESENCE GATE on the prescribed sheath branch -- the dispatched
    cathode solve takes it or ``None``, and ``None`` cannot reach the branch.
    """

    I_A: float
    V_dis_V: float


@dataclass(frozen=True)
class PrescribedDriveTrace:
    """A measured discharge trace, resolved onto the MODEL clock.

    ``time_s`` is ``t0_s + 1e-3*discharge_time_ms``: the trace's samples
    already shifted onto model time, so every consumer interpolates against
    the solver's own clock and the conversion happens once, here.
    ``current_A`` and ``V_dis_V`` are the trace's own columns, unchanged.

    ``sha256`` is the digest of the FILE's bytes, not of the arrays: it is
    what a saved trajectory records so an artifact can be matched back to the
    exact measured product that drove it.
    """

    path: str
    sha256: str
    t0_s: float
    start_s: float
    time_s: np.ndarray
    current_A: np.ndarray
    V_dis_V: np.ndarray

    def active(self, time_s):
        """Return whether the prescribed drive has taken over at ``time_s``."""
        return float(time_s) >= self.start_s

    def at(self, time_s):
        """Return the measured ``(I [A], V_dis [V])`` at one model time.

        Linear interpolation between the trace's samples, and its endpoint
        values outside them. The endpoint hold is a clamp, not an
        extrapolation: the ES overlays span from ~20 ms before the trigger to
        ~140 ms after it, far outside any campaign run's window, so the clamp
        is unreachable at every configuration the campaign runs -- and where a
        shorter trace did end early, holding its last measured value is the
        only reading of it that is still a measurement.
        """
        t = float(time_s)
        return (
            float(np.interp(t, self.time_s, self.current_A)),
            float(np.interp(t, self.time_s, self.V_dis_V)),
        )


def _file_sha256(path):
    """Return the sha256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse(message):
    raise ValueError(
        f"cathode_solver_model='{PRESCRIBED_MEASURED}': {message}"
    )


def _required_float(input_dict, key):
    value = input_dict.get(key)
    if value is None:
        _refuse(
            f"{key} is REQUIRED and unset. The measured overlay clock and the "
            "model clock have different origins, and the hand-off time is a "
            "decision about the run, so neither has a default"
        )
    value = float(value)
    if not np.isfinite(value):
        _refuse(f"{key} must be finite (got {value!r})")
    return value


def _load_trace_arrays(path):
    """Return the three trace columns, refusing anything else loudly."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            available = list(archive.files)
            missing = [k for k in TRACE_REQUIRED_KEYS if k not in available]
            if missing:
                _refuse(
                    f"the trace file {str(path)!r} does not carry "
                    f"{missing}. A prescribed-drive trace is read in the ES "
                    "overlay schema and must supply "
                    f"{list(TRACE_REQUIRED_KEYS)}; this file carries "
                    f"{sorted(available)}"
                )
            columns = tuple(
                np.asarray(archive[key], dtype=float)
                for key in TRACE_REQUIRED_KEYS
            )
    except ValueError:
        raise
    except Exception as error:
        _refuse(
            f"the trace file {str(path)!r} could not be read as an .npz "
            f"archive ({type(error).__name__}: {error})"
        )
    time_ms, current_A, V_dis_V = columns
    for key, column in zip(TRACE_REQUIRED_KEYS, columns):
        if column.ndim != 1:
            _refuse(
                f"trace column {key!r} must be one-dimensional (got shape "
                f"{column.shape})"
            )
        if not np.all(np.isfinite(column)):
            _refuse(f"trace column {key!r} carries non-finite samples")
    if not (time_ms.size == current_A.size == V_dis_V.size):
        _refuse(
            "the three trace columns must have equal length (got "
            f"{TRACE_TIME_KEY}={time_ms.size}, "
            f"{TRACE_CURRENT_KEY}={current_A.size}, "
            f"{TRACE_VOLTAGE_KEY}={V_dis_V.size})"
        )
    if time_ms.size < 2:
        _refuse(
            f"the trace carries {time_ms.size} sample(s); interpolating a "
            "drive needs at least two"
        )
    if not np.all(np.diff(time_ms) > 0.0):
        _refuse(
            f"trace column {TRACE_TIME_KEY!r} must be strictly increasing; "
            "an unsorted or repeated time base has no unique value to "
            "interpolate at"
        )
    return time_ms, current_A, V_dis_V


def resolve_prescribed_drive(input_dict, input_flags, solver_model):
    """Return the resolved :class:`PrescribedDriveTrace`, or ``None``.

    ``None`` is the OFF path and is what every configuration that does not
    select ``prescribed_measured`` gets: the three drive keys are then required
    to sit at their ``None`` defaults, so a trace, an origin or a hand-off time
    can never be configured into a run that would silently ignore it.

    Called once, at solver construction. Every refusal below therefore costs a
    misconfigured run nothing but the construction: a trace that cannot be
    read, a clock that cannot be reconciled, or a hand-off that cannot be
    honoured is stated before any compute is spent discovering it.
    """
    supplied = [
        key for key in PRESCRIBED_DRIVE_KEYS if input_dict.get(key) is not None
    ]
    if solver_model != PRESCRIBED_MEASURED:
        if supplied:
            raise ValueError(
                f"{supplied} are read only under "
                f"cathode_solver_model='{PRESCRIBED_MEASURED}' (got "
                f"{solver_model!r}); a measured drive nothing reads is a "
                "silent inert control"
            )
        return None
    if bool(input_flags.get("cathode_circuit_voltage_bound", False)):
        _refuse(
            "cathode_circuit_voltage_bound bounds the sheath against what the "
            "BANK LOOP can supply, and this mode has no loop equation -- the "
            "device voltage is measured, not sourced. Turn the flag off"
        )

    raw_path = input_dict.get("cathode_prescribed_trace_path")
    if raw_path is None:
        _refuse(
            "cathode_prescribed_trace_path is REQUIRED and unset: the mode IS "
            "the measured trace, so there is nothing to fall back to"
        )
    path = Path(str(raw_path)).expanduser()
    if not path.is_file():
        _refuse(f"the trace file {str(path)!r} does not exist")

    t0_s = _required_float(input_dict, "cathode_prescribed_t0_s")
    start_s = _required_float(input_dict, "cathode_prescribed_start_s")

    time_ms, current_A, V_dis_V = _load_trace_arrays(path)
    time_s = t0_s + 1.0e-3 * time_ms

    if start_s < t0_s:
        _refuse(
            f"cathode_prescribed_start_s={start_s!r} s is before "
            f"cathode_prescribed_t0_s={t0_s!r} s. The origin is the model time "
            "the trace's own t = 0 names, so a hand-off below it would drive "
            "the column from the trace's pre-trigger branch, which is the "
            "quiescent baseline and not a drive"
        )
    if start_s <= 0.0:
        # NO FOOT: the prescribed drive is in force from the first step, so
        # the calibrated cathode never runs and its keys are read by nothing.
        # (With a foot -- every configuration the campaign actually runs --
        # they ARE read, and are required to be a valid calibrated
        # configuration; see the module docstring.)
        from .config import input_dict_template_1d

        offenders = sorted(
            key
            for key in CALIBRATED_ONLY_KEYS
            if key in input_dict_template_1d
            and input_dict.get(key) != input_dict_template_1d[key]
        )
        if offenders:
            _refuse(
                f"cathode_prescribed_start_s={start_s!r} s puts the hand-off "
                "at or before the model clock's origin, so the calibrated "
                "cathode never runs -- but "
                f"{offenders} are set away from their defaults. The emission "
                "constant, the surface temperature, the bank loop and the "
                "heater package are read by NOTHING on a run with no foot. "
                "Either restore them, or move the hand-off after the "
                "breakdown so the foot that reads them exists"
            )
    if start_s > float(time_s[-1]):
        _refuse(
            f"cathode_prescribed_start_s={start_s!r} s is past the end of the "
            f"trace on the model clock ({float(time_s[-1])!r} s), so the "
            "prescribed drive would never take over"
        )
    return PrescribedDriveTrace(
        path=str(path),
        sha256=_file_sha256(path),
        t0_s=t0_s,
        start_s=start_s,
        time_s=time_s,
        current_A=current_A,
        V_dis_V=V_dis_V,
    )
