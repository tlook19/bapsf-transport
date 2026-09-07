"""Census of the afterglow tail's hand-off to open circuit.

After the bank transistors open, the parasitic inductance keeps the discharge
loop driven at zero source volts -- the measured freewheel tail. The loop
returns to OPEN CIRCUIT when either end condition is met on the last accepted
step:

    I_prev <= 1 A          the current has decayed to negligible stored energy
    V_dis_step <= 0        the device voltage has turned non-positive, so the
                           load would have to drive the loop; the bank is
                           already open and the diode blocks the reversal

This instrument reports WHEN that happened and asserts it did not happen too
early. The assertion is the point: the hand-off is a late-afterglow event, and
a run whose tail ends inside a scored window is reporting a different physics
question than the one that was asked.

TWO ROUTES
----------
``--from-h5 RUN.h5`` reads a saved trajectory. Resolution is the SAVE lattice,
not the step lattice: the reported hand-off time is the first SAVE at which
the phase reads open-circuit, so the true firing lies in the preceding save
interval. It names no configuration, on the same convention as the scorers --
the file records its own.

``--config <name>`` runs a configuration live and censuses every accepted step,
which is the exact instrument. It names a configuration like every other run
entry point.

Exit 0 = the assertion holds (or none was asked for); exit 1 = it does not.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, ProgressPrinter1D

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from stance_config import (  # noqa: E402
    STANCE_DIR,
    SUFFIX,
    available_stances,
    load_configuration,
)

REFERENCE_CONFIGURATION = STANCE_DIR / f"g1atrim{SUFFIX}"

# The two end conditions, in the solver's own terms. Kept here so the census
# can NAME which one fired; the solver owns the decision itself.
HANDOFF_CURRENT_A = 1.0


def _criterion(I_prev, V_dis_step):
    """Return the label of the end condition(s) met, or ``None``."""
    met = []
    if I_prev <= HANDOFF_CURRENT_A:
        met.append("current")
    if V_dis_step <= 0.0:
        met.append("voltage")
    return "+".join(met) if met else None


class _StepCensus:
    """Per-accepted-step record of the tail phase.

    Wraps the solver's accepted-step entry point and reads, at the state each
    step STARTS from, the phase that step will be integrated under and the two
    circuit quantities the hand-off criterion is tested against. Recording at
    the start is what makes the row describe the step's own decision rather
    than the state it left behind.
    """

    def __init__(self, sim):
        self.sim = sim
        self.rows = []
        self._inner = sim._accept_step_with_picard
        sim._accept_step_with_picard = self._wrapped

    def _wrapped(self, generate_attempt):
        t0 = float(self.sim._time)
        I_prev = float(self.sim._circuit_I_prev)
        V_dis_step = float(self.sim._circuit_V_dis_step)
        phase = self.sim._cathode_phase_options(time=t0)
        self.rows.append(
            (t0, I_prev, V_dis_step, bool(phase["floating"]),
             bool(phase["inductive_tail"]), bool(phase["cathode_enabled"]))
        )
        return self._inner(generate_attempt)

    def release(self):
        del self.sim._accept_step_with_picard


def _report(times_s, floating, driven_tail, I_loop, V_dis, resolution,
            assert_before_ms):
    """Print the census and return the exit status.

    ``floating`` and ``driven_tail`` are boolean arrays over the same lattice
    as ``times_s``; the first is the open-circuit phase, the second the driven
    freewheel. ``I_loop`` and ``V_dis`` are per-entry circuit readings, used
    only to describe the firing.
    """
    times_ms = np.asarray(times_s, dtype=float) * 1.0e3
    floating = np.asarray(floating, dtype=bool)
    driven_tail = np.asarray(driven_tail, dtype=bool)
    n_tail = int(driven_tail.sum())
    n_float = int(floating.sum())
    print(f"tail census: resolution={resolution}, entries={times_ms.size}")
    if times_ms.size:
        print(
            f"tail census: span t=[{times_ms[0]:.4f}, {times_ms[-1]:.4f}] ms"
        )
    print(
        f"tail census: driven-tail entries={n_tail}, "
        f"open-circuit entries={n_float}"
    )
    if n_tail == 0 and n_float == 0:
        print(
            "tail census NOTE: the record contains no afterglow at all, so it "
            "says nothing about the hand-off -- the assertion below passes "
            "vacuously"
        )
    fired = np.flatnonzero(floating)
    if fired.size == 0:
        print("tail census: hand-off NEVER fires in this record")
        first_ms = None
    else:
        k = int(fired[0])
        first_ms = float(times_ms[k])
        which = _criterion(float(I_loop[k]), float(V_dis[k]))
        if which is None:
            which = "not reconstructible at this resolution"
        print(
            f"tail census: first hand-off at index {k}, t={first_ms:.4f} ms, "
            f"I={float(I_loop[k]):.4f} A, V_dis={float(V_dis[k]):.6f} V, "
            f"criterion={which}"
        )
    if assert_before_ms is None:
        print("tail census: no earliest-firing assertion requested")
        return 0
    if first_ms is not None and first_ms < assert_before_ms:
        print(
            f"tail census: FAIL -- hand-off fired at {first_ms:.4f} ms, "
            f"before the registered {assert_before_ms:.4f} ms"
        )
        return 1
    print(
        f"tail census: PASS -- zero hand-off firings before "
        f"{assert_before_ms:.4f} ms"
    )
    return 0


def _run_from_h5(args):
    import h5py

    path = Path(args.from_h5)
    with h5py.File(path, "r") as h:
        time_s = np.asarray(h["time"][:], dtype=float)
        diag = h["cathode_diagnostics"]
        if "floating" not in diag:
            raise SystemExit(
                f"{path}: no cathode_diagnostics/floating dataset; this file "
                "was written before the phase flag was exported and the "
                "census cannot be taken from it"
            )
        floating = np.asarray(diag["floating"][:], dtype=float) > 0.5
        I_loop = np.asarray(diag["circuit_I_loop"][:], dtype=float)
        V_dis = np.asarray(diag["circuit_V_dis_step"][:], dtype=float)
        # phase_floating is the SCHEDULE's afterglow flag; the driven tail is
        # the afterglow with the loop still integrated.
        if "phase_floating" in h:
            afterglow = np.asarray(h["phase_floating"][:], dtype=float) > 0.5
        else:
            afterglow = np.zeros_like(floating)
        name = h.attrs.get("configuration_name", None)
    print(f"tail census route=from-h5 file={path}")
    print(f"tail census: configuration={name if name is not None else 'None'}")
    print(
        "tail census NOTE: the saved circuit_V_dis_step is the dt-weighted "
        "SAVE-INTERVAL average, not the per-step value the criterion reads, "
        "so the criterion label at the firing is indicative only."
    )
    return _report(
        time_s, floating, afterglow & ~floating, I_loop, V_dis,
        "save", args.assert_no_handoff_before_ms,
    )


def _run_live(args):
    if args.config is None:
        raise SystemExit(
            "census_afterglow_tail_handoff: name the configuration to run. "
            "Pass --config <name> for a committed configuration (available: "
            f"{', '.join(available_stances()) or '(none committed)'}) or "
            "--config <path.toml> for a configuration file, derived or not. "
            f"The LAPD reference configuration is {REFERENCE_CONFIGURATION}."
        )
    params, flags, configuration = load_configuration(args.config)
    if args.nx is not None:
        params["nx"] = args.nx
    configuration = configuration.with_identity(params, flags)
    sim = LAPDSim1D(params, flags, configuration=configuration)
    census = _StepCensus(sim)
    tracker = ProgressPrinter1D() if args.progress else None
    sim.start_simulation(
        t_end=args.t_end,
        max_steps=args.max_steps,
        progress_tracker=tracker,
    )
    census.release()
    if args.output is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        sim.save_result(out, sim.get_results(), params=params, flags=flags)
        print(f"tail census: trajectory saved to {out}")
    rows = census.rows
    print(f"tail census route=live configuration={configuration.name}")
    print(f"tail census: identity={configuration.identity}")
    times = np.array([r[0] for r in rows], dtype=float)
    I_prev = np.array([r[1] for r in rows], dtype=float)
    V_dis = np.array([r[2] for r in rows], dtype=float)
    floating = np.array([r[3] for r in rows], dtype=bool)
    tail = np.array([r[4] for r in rows], dtype=bool)
    return _report(
        times, floating, tail, I_prev, V_dis,
        "accepted step", args.assert_no_handoff_before_ms,
    )


def main(argv=None):
    args = _parse_args(argv)
    if args.from_h5 is not None:
        return _run_from_h5(args)
    return _run_live(args)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Census the afterglow tail's hand-off to open circuit."
    )
    parser.add_argument(
        "--from-h5", default=None,
        help="Census a saved trajectory instead of running one. Names no "
             "configuration: the file records its own.",
    )
    parser.add_argument(
        "--config", default=None,
        help="REQUIRED on the live route. The configuration to run: a "
             "committed configuration NAME, or the path of a configuration "
             "file. The LAPD reference configuration is "
             "scripts/stances/g1atrim.toml.",
    )
    parser.add_argument("--nx", type=int, default=None, help="Cell count.")
    parser.add_argument("--t-end", type=float, default=None,
                        help="Final time [s] on the live route.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Step cap on the live route.")
    parser.add_argument("--output", default=None,
                        help="Optional HDF5 to save the live run into.")
    parser.add_argument("--progress", action="store_true",
                        help="Print run progress on the live route.")
    parser.add_argument(
        "--assert-no-handoff-before-ms", type=float, default=None,
        metavar="MS",
        help="Fail (exit 1) if the hand-off fires before this time [ms]. The "
             "registered value for the LAPD reference configuration is 21.5, "
             "the end of the scored window.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
