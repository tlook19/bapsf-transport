"""P2 pre-flight: NO-SOLVE config diff of the mn1z ES2/ES3 arms vs their
per-rung reference h5.

Run artifact (untracked).  Adapted from scripts/mn_p1_preflight_diffcfg.py.
Difference from P1: the P2 rung references were produced by run_m6_point.py
(not compare_sim1d_es1.py's native path), so this harness does NOT
re-implement the config build -- it STUBS LAPDSim1D and calls
run_m6_point.main() with the real argv, capturing the driver's fully-resolved
params/flags at the construction call, before any solve.

run_m6_point.py exposes no --drag-closure.  The M8/P1 mechanism
(compare_sim1d_es1.run_model, drag_closure="neutral_momentum") is:
    params["ion_neutral_drag_model"] = "constant"
    params["b_ion_neutral_drag"]     = 1.0
    flags["neutral_momentum"]        = True
The two param writes are already the reference values on both rungs (verified
by the CONTROL diff below), so the arm is expressed through the driver's
existing --extra-flag passthrough as `--extra-flag neutral_momentum=true`,
which is net-identical.  The diff below is what proves it.

Usage:
    python scripts/mn_p2_preflight_diffcfg.py <es> <reference.h5>
"""
import json
import sys
from pathlib import Path

import h5py

REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/scripts")

import compare_sim1d_es1 as cmp_es1  # noqa: E402
import run_m6_point  # noqa: E402


class _Captured(Exception):
    def __init__(self, params, flags):
        super().__init__("captured")
        self.params = params
        self.flags = flags


class _StubSim:
    """Stands in for LAPDSim1D: captures the resolved config, never solves."""

    def __init__(self, params, flags, *a, **kw):
        raise _Captured(dict(params), dict(flags))


def build(argv):
    """Run the REAL driver up to the LAPDSim1D construction call."""
    real = cmp_es1.LAPDSim1D
    cmp_es1.LAPDSim1D = _StubSim
    try:
        run_m6_point.main(argv)
    except _Captured as c:
        return c.params, c.flags
    finally:
        cmp_es1.LAPDSim1D = real
    raise SystemExit("harness ERROR: driver did not construct LAPDSim1D")


def read_ref(path):
    with h5py.File(path, "r") as f:
        p = json.loads(f.attrs["params_json"])
        fl = json.loads(f.attrs["flags_json"])
    return p, fl


def diff(name, ref, new):
    changed, added, removed = [], [], []
    for k in sorted(set(ref) | set(new)):
        if k not in ref:
            added.append((k, new[k]))
        elif k not in new:
            removed.append((k, ref[k]))
        elif ref[k] != new[k]:
            changed.append((k, ref[k], new[k]))
    print(f"  {name}: {len(changed)} CHANGED, {len(added)} ADDED, "
          f"{len(removed)} REMOVED")
    for k, a, b in changed:
        print(f"    CHANGED  {k}: {a!r} -> {b!r}")
    for k, v in added:
        print(f"    ADDED    {k}: <absent> -> {v!r}")
    for k, v in removed:
        print(f"    REMOVED  {k}: {v!r} -> <absent>")
    return len(changed) + len(added) + len(removed)


BASE = [
    "--nx", "240",
    "--sgp", "3000",
    "--extra", "tau_afterglow=0.006", "T_s=1998.15",
    'Te_birth_ionization="local"',
]


def main():
    es = sys.argv[1]
    ref_path = sys.argv[2]
    rp, rf = read_ref(ref_path)
    print(f"ES{es} reference: {ref_path}")
    print(f"  ref params keys={len(rp)}  flags keys={len(rf)}")
    print(f"  PRODUCTION_NX = {cmp_es1.PRODUCTION_NX}")
    print(f"  ES_OPERATING[{es}] = {run_m6_point.ES_OPERATING[int(es)]}")

    sink = "/dev/null"
    control_argv = ["--es", es] + BASE + ["--save-h5", sink]
    arm_argv = (["--es", es] + BASE
                + ["--extra-flag", "neutral_momentum=true", "--save-h5", sink])

    print(f"\n  control argv: {' '.join(control_argv)}")
    print(f"  arm     argv: {' '.join(arm_argv)}")

    total = {}
    for label, argv in (("CONTROL rebuild (no neutral_momentum)", control_argv),
                        ("ARM rebuild (--extra-flag neutral_momentum=true)",
                         arm_argv)):
        print(f"\n=== {label} ===")
        p, fl = build(argv)
        np_ = diff("params", rp, p)
        nf_ = diff("flags", rf, fl)
        total[label] = (np_, nf_)

    print("\n=== VERDICT ===")
    c = total["CONTROL rebuild (no neutral_momentum)"]
    a = total["ARM rebuild (--extra-flag neutral_momentum=true)"]
    print(f"  control: params {c[0]} deltas, flags {c[1]} deltas "
          f"(REQUIRED 0 / 0)  -> {'OK' if c == (0, 0) else 'STOP'}")
    print(f"  arm:     params {a[0]} deltas, flags {a[1]} deltas "
          f"(REQUIRED 0 / 1)  -> {'OK' if a == (0, 1) else 'STOP'}")
    ok = c == (0, 0) and a == (0, 1)
    print(f"  PRE-FLIGHT: {'PASS' if ok else 'FAIL -- DO NOT RUN'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
