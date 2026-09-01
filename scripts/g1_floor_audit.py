"""Floor-activation audit REPLICA for the G1 arms (the floor-activation bin).

``audit_sim1d_floor_activation.py`` runs only its own production config, and
the G1 registration requires the audit ON the arm configs -- runtime
instrumentation a saved trajectory cannot provide (``derive_state`` floors
Te/Ti on every read). This driver rebuilds an arm's exact resolved config from
its artifact's ``params_json``/``flags_json`` attrs (JSON round-trips IEEE
doubles exactly), installs the parent script's two clip-site probes, re-runs
the identical solve invocation (``start_simulation(t_end=None, dt=None,
operator_split=None, max_steps=...)`` -- the ``run_model`` call the arm
itself used), and reports with the parent's reporter.

The replica claim is GATED, not assumed: after the run, the replica's final
frame must be bit-identical (raw uint64) to the artifact's on every compared
state channel, and the frame count and final time must match. A failed gate
voids the audit numbers and the report says so loudly.

The conduction probe deliberately differs from the parent's: the value
RETURNED to the solver is the real-floor solve, untouched -- the -inf-floor
solve runs a second time purely as the diagnostic. (The parent's probe
reconstructs the clipped result from the -inf solve through a divide/multiply
round-trip, which is faithful but not guaranteed bit-exact; here bit-exactness
of the probed path is what the identity gate rests on.) Cost: the implicit
conduction solves run twice.

Usage:
    python scripts/g1_floor_audit.py scripts/g1a_foot45_cr6p94.h5 \
        --max-steps 300000 --out scripts/g1a_flooraudit.txt
"""

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import audit_sim1d_floor_activation as parent  # noqa: E402
import cablp.solvers._sim1d.physics.conduction as conduction_mod  # noqa: E402
from cablp.solvers._sim1d import LAPDSim1D, load_result_hdf5  # noqa: E402

#: State channels gated bit-identical between replica and artifact. Every name
#: must exist on both result objects; a missing one is an error, not a skip.
IDENTITY_CHANNELS = ("n", "nn", "M", "Ee", "Ei")


def _final_row(result, name):
    value = getattr(result, name)
    return np.asarray(value, dtype=float)[-1]


def _bitwise_equal(a, b):
    a = np.ascontiguousarray(np.asarray(a, dtype=float))
    b = np.ascontiguousarray(np.asarray(b, dtype=float))
    if a.shape != b.shape:
        return False, -1
    differing = int(np.sum(a.view(np.uint64) != b.view(np.uint64)))
    return differing == 0, differing


def make_probes(recorder, true_orig_species):
    """Install the state probe from the parent and the bit-exact conduction
    probe here; return the parent's restore callable."""
    restore = parent.install_probes(recorder)

    def probed_species(
        energy, capacity, temperature_floor, conductivity, geometry, dt, **kwargs
    ):
        recorder.bump_calls("conduction")
        clipped_result = true_orig_species(
            energy=energy,
            capacity=capacity,
            temperature_floor=temperature_floor,
            conductivity=conductivity,
            geometry=geometry,
            dt=dt,
            **kwargs,
        )
        unclipped = true_orig_species(
            energy=energy,
            capacity=capacity,
            temperature_floor=-np.inf,
            conductivity=conductivity,
            geometry=geometry,
            dt=dt,
            **kwargs,
        )
        raw_T = np.asarray(unclipped, dtype=float) / capacity
        lo = temperature_floor * (1.0 - parent.FLOOR_RTOL)
        clipped = raw_T < lo
        injected = (
            float(np.sum(capacity[clipped] * (temperature_floor - raw_T[clipped])))
            if np.any(clipped)
            else 0.0
        )
        field = "Te" if recorder.cond_parity % 2 == 0 else "Ti"
        recorder.cond_parity += 1
        recorder.record(
            "conduction",
            field,
            clipped,
            injected,
            resting=(raw_T >= lo)
            & (raw_T <= temperature_floor * (1.0 + parent.FLOOR_RTOL)),
        )
        return clipped_result

    conduction_mod._implicit_species_energy = probed_species
    return restore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", help="the arm artifact to replicate and audit")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300000,
        help="the accepted-step cap the arm ran with (a solve argument, not "
        "a params key, so it is not recoverable from the attrs)",
    )
    parser.add_argument("--out", default=None, help="report file (also printed)")
    args = parser.parse_args(argv)

    import h5py

    with h5py.File(args.h5, "r") as handle:
        params = json.loads(handle.attrs["params_json"])
        flags = json.loads(handle.attrs["flags_json"])

    true_orig_species = conduction_mod._implicit_species_energy

    sim = LAPDSim1D(params, flags)
    recorder = parent.FloorRecorder(
        cells=sim._geometry.cells, scheme=params.get("implicit_heat_scheme")
    )
    recorder.time_getter = lambda: sim._time
    restore = make_probes(recorder, true_orig_species)
    try:
        sim.start_simulation(
            t_end=None, dt=None, operator_split=None, max_steps=args.max_steps
        )
        result = sim.get_results()
    finally:
        restore()

    saved = load_result_hdf5(args.h5)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"artifact : {os.path.abspath(args.h5)}")
        print(f"replica  : LAPDSim1D from the artifact's params_json/flags_json")
        print(
            f"solve    : start_simulation(t_end=None, dt=None, "
            f"operator_split=None, max_steps={args.max_steps})"
        )

        print("\n=== REPLICA IDENTITY GATE (raw uint64, final frame) ===")
        gate_pass = True
        steps_equal = int(result.steps) == int(saved.steps) and str(
            result.run_status
        ) == str(saved.run_status)
        print(
            f"steps    : replica {result.steps} ({result.run_status}), artifact "
            f"{saved.steps} ({saved.run_status})  "
            f"{'PASS' if steps_equal else 'FAIL'}"
        )
        gate_pass &= steps_equal
        t_replica = np.asarray(result.time, dtype=float)
        t_saved = np.asarray(saved.time, dtype=float)
        frames_equal = t_replica.size == t_saved.size
        print(
            f"frames   : replica {t_replica.size}, artifact {t_saved.size}  "
            f"{'PASS' if frames_equal else 'FAIL'}"
        )
        gate_pass &= frames_equal
        equal, _ = _bitwise_equal(t_replica[-1], t_saved[-1])
        print(
            f"final t  : replica {t_replica[-1]!r}, artifact {t_saved[-1]!r}  "
            f"{'PASS' if equal else 'FAIL'}"
        )
        gate_pass &= equal
        for name in IDENTITY_CHANNELS:
            equal, differing = _bitwise_equal(
                _final_row(result, name), _final_row(saved, name)
            )
            print(
                f"{name:8s} : {'PASS bit-identical' if equal else f'FAIL ({differing} differing)'}"
            )
            gate_pass &= equal
        if getattr(saved, "nn_a", None) is not None and getattr(
            result, "nn_a", None
        ) is not None:
            equal, differing = _bitwise_equal(
                _final_row(result, "nn_a"), _final_row(saved, "nn_a")
            )
            print(
                f"{'nn_a':8s} : {'PASS bit-identical' if equal else f'FAIL ({differing} differing)'}"
            )
            gate_pass &= equal

        if gate_pass:
            print("IDENTITY GATE: PASS -- the audit below is the ARM's audit.")
        else:
            print(
                "IDENTITY GATE: FAIL -- THE AUDIT BELOW IS VOID: the replica "
                "did not track the artifact and its clip statistics describe "
                "a different trajectory. Do not quote them."
            )

        parent.report(recorder, sim, result)

    text = buffer.getvalue()
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
        print(f"wrote {args.out}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
