"""fnb3 -- FLAG-ON PHYSICS IDENTITY: before/after digest of a short real run.

The NBL hot-channel diagnostics pass claims ZERO physics change. This is the
proof instrument. It runs the fnb2 flag-ON smoke configuration (the same one
the pass-2 build was smoked on) and dumps EVERY float array the result carries
-- top-level fields, the whole ``rhs_terms`` ledger field by field, the energy
term ledgers, ``total_rhs``, the floor ledger and the timestep diagnostics --
into an npz keyed by a flat path name.

``--compare BEFORE.npz AFTER.npz`` then compares the two at RAW UINT64, never a
tolerance, and classifies every key:

    IDENTICAL   same bits (what a diagnostics-only pass must produce for
                every pre-existing key)
    MISMATCH    the physics moved -- a FAILURE of the pass's central claim
    ONLY-IN-B   a key the pass added (the diagnostics; expected)
    ONLY-IN-A   a key the pass dropped (a regression unless deliberate)

Nothing is fitted, tuned or written back into the model. Run with
PYTHONPATH=<worktree>/cablp.
"""
import sys

import numpy as np

import cablp
from cablp.solvers._sim1d import LAPDSim1D, default_config

MAX_STEPS = 20000
T_END = 3.0e-3

# The config echoes are inputs, not outputs, and the pass is expected to leave
# them alone; they are dicts of scalars/strings and are excluded by name.
CONFIG_ECHOES = {"params", "flags"}


def flag_on_config():
    """Return the fnb2 flag-ON smoke configuration (params, flags)."""
    params, flags = default_config()
    params["nx"] = 24
    params["S_gp"] = 9010
    params["max_steps_action"] = "stop"
    flags["neutral_momentum"] = True
    flags["neutral_two_zone"] = True
    flags["neutral_energy"] = True
    return params, flags


def run():
    params, flags = flag_on_config()
    sim = LAPDSim1D(input_dict=params, input_flags=flags)
    sim.start_simulation(t_end=T_END, dt=None, operator_split=None,
                         max_steps=MAX_STEPS)
    return sim.get_results()


def _collect(prefix, value, out):
    """Flatten anything float-bearing on the result into ``out[path] = array``."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f":
            out[prefix] = np.ascontiguousarray(value)
        return
    if isinstance(value, (float, np.floating)):
        out[prefix] = np.ascontiguousarray(np.asarray([float(value)]))
        return
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        out[prefix] = np.ascontiguousarray(np.asarray([float(value)]))
        return
    if isinstance(value, dict):
        for key in sorted(value):
            _collect(f"{prefix}/{key}", value[key], out)
        return
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _collect(f"{prefix}/{name}", getattr(value, name), out)
        return
    if isinstance(value, (list, tuple)) and value:
        if hasattr(value[0], "__dataclass_fields__"):
            for name in value[0].__dataclass_fields__:
                column = [getattr(item, name) for item in value]
                if all(isinstance(x, (float, int, np.floating)) and
                       not isinstance(x, bool) for x in column):
                    out[f"{prefix}/{name}"] = np.ascontiguousarray(
                        np.asarray(column, dtype=float)
                    )
        return


def digest(result):
    out = {}
    for name in sorted(vars(result)):
        if name in CONFIG_ECHOES:
            continue
        value = getattr(result, name)
        if callable(value):
            continue
        _collect(name, value, out)
    return out


def dump(path):
    print("cablp package file:", cablp.__file__)
    result = run()
    print(f"status={result.run_status} steps={result.steps} "
          f"t_end={result.time[-1]:.12e} s")
    data = digest(result)
    print(f"digest keys: {len(data)}")
    np.savez(path, **data)
    print(f"wrote {path}")


def raw(a):
    return np.ascontiguousarray(np.asarray(a, dtype=float)).view(np.uint64)


def compare(path_a, path_b):
    a = np.load(path_a)
    b = np.load(path_b)
    keys_a, keys_b = set(a.files), set(b.files)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    shared = sorted(keys_a & keys_b)
    bad = []
    for key in shared:
        va, vb = a[key], b[key]
        if va.shape != vb.shape or not np.array_equal(raw(va), raw(vb)):
            bad.append(key)
    print(f"A = {path_a}   ({len(keys_a)} keys)")
    print(f"B = {path_b}   ({len(keys_b)} keys)")
    print(f"shared keys compared at raw uint64: {len(shared)}")
    print(f"  IDENTICAL : {len(shared) - len(bad)}")
    print(f"  MISMATCH  : {len(bad)}")
    for key in bad:
        va, vb = a[key], b[key]
        if va.shape == vb.shape:
            worst = float(np.max(np.abs(va - vb)))
            scale = max(float(np.max(np.abs(va))), 1e-300)
            print(f"      ! {key}  max|dA-B|={worst:.6e}  rel={worst/scale:.3e}")
        else:
            print(f"      ! {key}  shape {va.shape} vs {vb.shape}")
    print(f"  ONLY-IN-A : {len(only_a)}")
    for key in only_a:
        print(f"      - {key}")
    print(f"  ONLY-IN-B : {len(only_b)}")
    for key in only_b:
        print(f"      + {key}")
    ok = not bad and not only_a
    print()
    print("PHYSICS IDENTITY:", "PASS -- every pre-existing key is bit-identical; "
          "only new diagnostic keys appear" if ok else
          "FAIL -- the trajectory or a pre-existing row moved")
    return ok


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--dump":
        dump(sys.argv[2])
        return
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        raise SystemExit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
    raise SystemExit(
        "usage: fnb3_physics_identity.py --dump OUT.npz\n"
        "       fnb3_physics_identity.py --compare BEFORE.npz AFTER.npz"
    )


if __name__ == "__main__":
    main()
