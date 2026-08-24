"""Bit-inertness probe for the density-preserving Maxwellian projection.

The R11 fix in ``VGrid.maxwellian(exact_moments=True)`` is required to be
BIT-EXACT wherever the density invariant already held. This probe measures
that directly, on the arms where it has to hold: it runs the registered B0c
windows and, for EVERY ``maxwellian`` call the arm makes, digests the call's
arguments and the raw bit pattern of the returned bin masses (uint64 view, no
tolerance anywhere), in call order. It also digests the arm's final state.

Two modes:

  --bank PATH   run the arms and write the digest chains to PATH (npz).
  --check PATH  run the arms again and compare against a banked PATH,
                reporting the FIRST differing call with its arguments.

The evidence pair is: bank on the pre-fix source, check on the post-fix
source. A clean --check means every call in every window returned an
identical bit pattern, so the arms themselves are bit-identical.

The arms probed are the ladder/fixture grids on which inertness is claimed:
the B0c base and grid rungs, and the k2_dvm suite's own (48,12) fixture
grid. The failing ``cad_3.125e-06`` arm is deliberately NOT probed -- that
is the arm the fix exists to change.

Usage (from <checkout>/cablp, PYTHONPATH set to that same cablp):
    python scripts/b0cfx_bitinertness.py --bank scripts/b0cfx_bitinertness.npz
    python scripts/b0cfx_bitinertness.py --check scripts/b0cfx_bitinertness.npz
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from verify_sim1d_k2_dvm import advance_one_step, make_sim  # noqa: E402

from cablp.solvers._sim1d.physics import kinetic_neutrals as KN  # noqa: E402

# (name, cadence_s, nvz, nvp, ticks). The tick counts are the registered
# B0c N_k at t* = t_engage + 2.0 ms for the shipped cadence.
ARMS = (
    ("base_2.5e-05_16x6", 2.5e-05, 16, 6, 80),
    ("grid_32x12", 2.5e-05, 32, 12, 80),
    ("grid_64x24", 2.5e-05, 64, 24, 80),
    ("k2_fixture_48x12", 2.5e-05, 48, 12, 80),
)

STATE_KEYS = ("nn", "nn_a")


def _bits(arr):
    """Raw bit pattern of a float64 array, as bytes. No tolerance."""
    return np.ascontiguousarray(arr, dtype=float).view(np.uint64).tobytes()


def _digest(*chunks):
    h = hashlib.blake2b(digest_size=8)
    for chunk in chunks:
        h.update(chunk)
    return np.frombuffer(h.digest(), dtype=np.uint64)[0]


def run_arm(cadence_s, nvz, nvp, ticks):
    """Run one arm, returning (per-call digests, argument record, state digest)."""
    calls = []
    args = []
    original = KN.VGrid.maxwellian

    def _spy(self, T_eV, u_drift, exact_moments=True):
        out = original(self, T_eV, u_drift, exact_moments)
        args.append((float(T_eV), float(u_drift), float(exact_moments)))
        calls.append(_digest(
            _bits(np.array([float(T_eV), float(u_drift),
                            float(exact_moments)])),
            _bits(out),
        ))
        return out

    KN.VGrid.maxwellian = _spy
    try:
        sim = make_sim(
            neutral_kinetic_dvm_cadence_s=cadence_s,
            neutral_kinetic_dvm_nvz=nvz,
            neutral_kinetic_dvm_nvp=nvp,
        )
        steps = 0
        while sim._dvm.updates < ticks and steps < 200_000:
            advance_one_step(sim)
            steps += 1
    finally:
        KN.VGrid.maxwellian = original
    state = _digest(*(_bits(getattr(sim.state, key)) for key in STATE_KEYS))
    return (np.asarray(calls, dtype=np.uint64),
            np.asarray(args, dtype=float),
            np.uint64(state),
            int(steps),
            int(sim._dvm.updates))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bank", metavar="PATH",
                      help="run the arms and write the digest chains")
    mode.add_argument("--check", metavar="PATH",
                      help="run the arms and compare against a banked file")
    args = p.parse_args(argv)

    banked = None if args.bank else np.load(args.check, allow_pickle=False)
    out = {}
    failures = []
    for name, cadence_s, nvz, nvp, ticks in ARMS:
        digests, call_args, state, steps, done = run_arm(
            cadence_s, nvz, nvp, ticks
        )
        print(f"{name}: ({nvz},{nvp}) cadence {cadence_s:g} s -- "
              f"{done} ticks, {steps} steps, {digests.size} maxwellian calls, "
              f"state digest {int(state):016x}", flush=True)
        if banked is None:
            out[f"calls_{name}"] = digests
            out[f"args_{name}"] = call_args
            out[f"state_{name}"] = np.asarray([state], dtype=np.uint64)
            continue
        ref = banked[f"calls_{name}"]
        ref_state = np.uint64(banked[f"state_{name}"][0])
        if ref.size != digests.size:
            failures.append(
                f"{name}: call COUNT moved, {ref.size} banked vs "
                f"{digests.size} now -- the arm itself diverged"
            )
            continue
        differing = np.flatnonzero(ref != digests)
        if differing.size:
            i = int(differing[0])
            T_eV, u_drift, exact = call_args[i]
            failures.append(
                f"{name}: {differing.size}/{digests.size} calls differ; "
                f"first at call {i} (T_eV={T_eV!r}, u_drift={u_drift!r}, "
                f"exact_moments={bool(exact)})"
            )
        elif state != ref_state:
            failures.append(
                f"{name}: every maxwellian call is bit-identical but the "
                f"final state digest moved ({int(ref_state):016x} banked vs "
                f"{int(state):016x} now)"
            )
        else:
            print(f"    BIT-EXACT: {digests.size}/{digests.size} calls and "
                  f"the final ({', '.join(STATE_KEYS)}) state match the "
                  "banked run at raw uint64", flush=True)

    if banked is None:
        np.savez(args.bank, **out)
        print(f"\nbanked {len(ARMS)} arms to {args.bank}")
        return 0
    print("")
    if failures:
        for line in failures:
            print(f"  DIFFERS  {line}")
        print(f"\nBIT-INERTNESS: FAIL on {len(failures)}/{len(ARMS)} arms")
        return 1
    print(f"BIT-INERTNESS: PASS -- all {len(ARMS)} arms bit-identical at "
          "raw uint64")
    return 0


if __name__ == "__main__":
    sys.exit(main())
