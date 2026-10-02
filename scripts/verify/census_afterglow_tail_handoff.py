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
early, and prints the emitter surface temperature either side of the firing --
the surface ledger crosses the hand-off too, and a kink in ``T_s`` there would
mean it does not. The assertion is the point: the hand-off is a late-afterglow
event, and a run whose tail ends inside a scored window is reporting a
different physics question than the one that was asked.

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

Run without ``--assert-no-handoff-before-ms`` the census measures and asserts
nothing, so its exit status carries no verdict. With the flag: exit 0 = the
record covers the window up to the assertion time and the hand-off did not
fire before it; exit 1 = the hand-off fired before it; exit 2 = DID NOT RUN,
either because the record contains no afterglow or because it ends before the
assertion time without a firing, so the window the assertion names was not
all examined.

``--self-test`` builds synthetic saved records in a temporary directory and
checks the exit status of the ``--from-h5`` route on each.
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
    load_configuration_or_exit,
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
        self._inner = sim._accept_step_attempt
        sim._accept_step_attempt = self._wrapped

    def _wrapped(self, attempt):
        t0 = float(self.sim._time)
        I_prev = float(self.sim._circuit_I_prev)
        V_dis_step = float(self.sim._circuit_V_dis_step)
        phase = self.sim._cathode_phase_options(time=t0)
        # T_s is recorded because the SURFACE ledger crosses the hand-off too:
        # the emitting face keeps cooling at its released current on both
        # sides of it, so the temperature trace should show no kink there.
        T_s = self.sim._cathode_Ts_K
        self.rows.append(
            (t0, I_prev, V_dis_step, bool(phase["floating"]),
             bool(phase["inductive_tail"]), bool(phase["cathode_enabled"]),
             float(T_s))
        )
        return self._inner(attempt)

    def release(self):
        del self.sim._accept_step_attempt


def _report(times_s, floating, driven_tail, I_loop, V_dis, resolution,
            assert_before_ms, T_s=None, end_s=None):
    """Print the census and return the exit status.

    ``floating`` and ``driven_tail`` are boolean arrays over the same lattice
    as ``times_s``; the first is the open-circuit phase, the second the driven
    freewheel. ``I_loop`` and ``V_dis`` are per-entry circuit readings, used
    only to describe the firing. ``end_s`` is the last time the record covers
    [s]; it defaults to the last entry of ``times_s``.
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
    no_afterglow = n_tail == 0 and n_float == 0
    if no_afterglow:
        print(
            "tail census NOTE: the record contains no afterglow at all, so it "
            "says nothing about the hand-off"
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
        if T_s is not None:
            # The surface temperature either side of the firing, and the
            # per-entry increments beside it: a discontinuity in the surface
            # ledger shows up as a step in dT, not in T_s itself.
            T_s = np.asarray(T_s, dtype=float)
            lo, hi = max(k - 4, 0), min(k + 5, T_s.size)
            print("tail census: T_s across the hand-off "
                  "(index, t [ms], phase, T_s [K], dT from previous [K])")
            for j in range(lo, hi):
                dT = (T_s[j] - T_s[j - 1]) if j > 0 else float("nan")
                mark = "open" if floating[j] else "driven"
                star = " <- first open-circuit entry" if j == k else ""
                print(f"    {j:6d}  {times_ms[j]:9.5f}  {mark:6s}  "
                      f"{T_s[j]:12.7f}  {dT:+.3e}{star}")
    if assert_before_ms is None:
        print("tail census: no earliest-firing assertion requested")
        print("tail census: measurement only -- nothing was asserted, so the "
              "exit status carries no verdict")
        return 0
    if no_afterglow:
        print(
            "tail census: DID NOT RUN -- the record contains no afterglow, so "
            f"the assertion before {assert_before_ms:.4f} ms examined nothing"
        )
        return 2
    if first_ms is not None and first_ms < assert_before_ms:
        print(
            f"tail census: FAIL -- hand-off fired at {first_ms:.4f} ms, "
            f"before the registered {assert_before_ms:.4f} ms"
        )
        return 1
    if end_s is not None:
        end_ms = float(end_s) * 1.0e3
    elif times_ms.size:
        end_ms = float(times_ms[-1])
    else:
        end_ms = float("-inf")
    if first_ms is None and end_ms < assert_before_ms:
        print(
            f"tail census: DID NOT RUN -- the record ends at {end_ms:.4f} ms "
            "without a hand-off, before the assertion time "
            f"{assert_before_ms:.4f} ms, so the window up to it was not all "
            "examined"
        )
        return 2
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
        T_s_surface = (
            np.asarray(diag["T_s_surface"][:], dtype=float)
            if "T_s_surface" in diag
            else None
        )
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
        "save", args.assert_no_handoff_before_ms, T_s=T_s_surface,
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
    params, flags, configuration = load_configuration_or_exit(args.config)
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
    T_s = np.array([r[6] for r in rows], dtype=float)
    # The rows hold each accepted step's START time; the run covers up to the
    # end of its last step.
    return _report(
        times, floating, tail, I_prev, V_dis,
        "accepted step", args.assert_no_handoff_before_ms,
        T_s=None if np.all(np.isnan(T_s)) else T_s,
        end_s=float(sim._time),
    )


def _self_test():
    """Check the ``--from-h5`` exit status on constructed records.

    Each record is written with only the datasets ``_run_from_h5`` reads, on
    a 1 ms save lattice from 0 ms. Its expected status follows from how it is
    built: a firing is a save with ``floating`` set, the afterglow is
    ``phase_floating``, and the record ends at its last save.
    """
    import contextlib
    import io
    import tempfile

    import h5py

    def write(path, end_ms, afterglow_from_ms, floating_from_ms):
        t_ms = np.arange(0.0, end_ms + 0.5, 1.0)
        pf = (np.zeros(t_ms.size, bool) if afterglow_from_ms is None
              else t_ms >= afterglow_from_ms)
        fl = (np.zeros(t_ms.size, bool) if floating_from_ms is None
              else t_ms >= floating_from_ms)
        with h5py.File(path, "w") as h:
            h["time"] = t_ms * 1.0e-3
            h["phase_floating"] = pf.astype(float)
            d = h.create_group("cathode_diagnostics")
            d["floating"] = fl.astype(float)
            d["circuit_I_loop"] = np.linspace(2000.0, 0.1, t_ms.size)
            d["circuit_V_dis_step"] = np.linspace(60.0, -1.0, t_ms.size)
            h.attrs["configuration_name"] = "synthetic"

    # (name, record end [ms], afterglow from [ms], open circuit from [ms],
    #  assertion time [ms], expected exit status)
    cases = (
        ("tail-ends-before-T", 21.0, 20.0, None, 21.5, 2),
        ("tail-reaches-past-T", 25.0, 20.0, None, 21.5, 0),
        ("tail-ends-exactly-at-T", 21.0, 20.0, None, 21.0, 0),
        ("fires-before-T", 25.0, 10.0, 15.0, 21.5, 1),
        ("no-afterglow", 25.0, None, None, 21.5, 2),
    )
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, end_ms, ag_ms, fl_ms, T_ms, want in cases:
            path = Path(tmp) / f"{name}.h5"
            write(path, end_ms, ag_ms, fl_ms)
            with contextlib.redirect_stdout(io.StringIO()):
                got = main(["--from-h5", str(path),
                            "--assert-no-handoff-before-ms", repr(T_ms)])
            ok = got == want
            bad += not ok
            print(f"census self-test {name}: end={end_ms} ms T={T_ms} ms "
                  f"expected exit {want}, got {got} -- "
                  f"{'ok' if ok else 'MISMATCH'}")
    print("census self-test: "
          + ("PASS" if bad == 0 else f"FAIL ({bad} mismatch)"))
    return 0 if bad == 0 else 1


def main(argv=None):
    args = _parse_args(argv)
    if args.self_test:
        return _self_test()
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
        help="Fail (exit 1) if the hand-off fires before this time [ms]; exit "
             "2 (did not run) if the record contains no afterglow, or ends "
             "before this time without a firing. The "
             "registered value for the LAPD reference configuration is 21.5, "
             "the end of the scored window.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Check the exit status on synthetic saved records and exit.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
