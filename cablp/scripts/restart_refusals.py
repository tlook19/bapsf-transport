"""Construction-time refusal and off-path checks for the restart machinery.

Every misconfiguration below must raise ``ValueError`` at CONSTRUCTION, before
any stepping, and every accepted case must load. The last two checks are the
off-path ones: a solver built without ``restart_from`` must carry no restart
state at all.

Usage:  python scripts/restart_refusals.py
"""

import sys
import tempfile
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))

from restart_bitidentity import scenario_config  # noqa: E402

from cablp.solvers._sim1d import (  # noqa: E402
    LAPDSim1D,
    save_restart_state,
)


def main():
    failures = []
    tmp = Path(tempfile.mkdtemp())
    params, flags, t_mid, _ = scenario_config("meanfield")
    sim = LAPDSim1D(dict(params), dict(flags))
    sim.run(t_end=t_mid)
    good = tmp / "good.restart.h5"
    save_restart_state(good, sim)
    print(f"reference payload written at t={sim.time:.9e} s\n")

    def expect_raise(label, override_params, override_flags=None):
        use_flags = dict(flags if override_flags is None else override_flags)
        try:
            LAPDSim1D(dict(override_params), use_flags)
        except ValueError as error:
            text = " ".join(str(error).split())
            print(f"[RAISED ValueError] {label}\n    {text[:280]}\n")
            return
        except Exception as error:  # noqa: BLE001
            print(f"[WRONG EXCEPTION TYPE] {label}: "
                  f"{type(error).__name__}: {error}\n")
            failures.append(label)
            return
        print(f"[NO RAISE -- DEFECT] {label}\n")
        failures.append(label)

    expect_raise("missing payload file",
                 {**params, "restart_from": str(tmp / "absent.h5")})

    foreign = tmp / "foreign.h5"
    with h5py.File(foreign, "w") as h5:
        h5.attrs["format"] = "sim1d-hdf5-v1"
    expect_raise("foreign file format",
                 {**params, "restart_from": str(foreign)})

    expect_raise("grid change (nx 24 -> 30)",
                 {**params, "restart_from": str(good), "nx": 30})

    # phase_transition_mode, not neutral_model: the only other neutral_model
    # values are the two the REFUSED list rejects outright, so that structural
    # comparison is unreachable belt-and-braces, and this exercises a key a
    # two-stage hybrid could realistically differ on.
    expect_raise("structural param (phase_transition_mode)",
                 {**params, "restart_from": str(good),
                  "phase_transition_mode": "current"})

    coverage_flags = dict(flags)
    coverage_flags["coverage_closure"] = True
    expect_raise("structural flag (coverage_closure)",
                 {**params, "restart_from": str(good),
                  "coverage_initial_fraction": 0.3,
                  "beam_deposition_model": "csda"},
                 coverage_flags)

    equilibration_flags = dict(flags)
    equilibration_flags["neutral_equilibration"] = True
    expect_raise("neutral_equilibration would overwrite the restored IC",
                 {**params, "restart_from": str(good)}, equilibration_flags)

    # neutral_two_zone is set so the kinetic arm's own prerequisite validator
    # passes and the RESTART refusal is what actually fires.
    dvm_flags = dict(flags)
    dvm_flags["neutral_two_zone"] = True
    expect_raise("kinetic neutral_model (distribution not carried)",
                 {**params, "restart_from": str(good),
                  "neutral_model": "kinetic_dvm"}, dvm_flags)

    # Accepted: a NON-structural difference is exactly what a two-stage hybrid
    # needs, so it must load rather than refuse.
    resumed = LAPDSim1D(
        {**params, "restart_from": str(good), "tau_discharge": 2.0},
        dict(flags),
    )
    if resumed.time != sim.time:
        print(f"[DEFECT] resumed clock {resumed.time!r} != exported "
              f"{sim.time!r}\n")
        failures.append("clock round trip")
    else:
        print("[LOADED] non-structural change (tau_discharge) accepted, and "
              f"the clock round-trips exactly (t={resumed.time:.9e} s)\n")

    # Off path: no restart configured, no restart state.
    plain = LAPDSim1D(dict(params), dict(flags))
    if plain._restart_run_loop is not None or plain.time != 0.0:
        print(f"[DEFECT] off path carries restart state: "
              f"{plain._restart_run_loop!r}, t={plain.time!r}\n")
        failures.append("off path clean")
    else:
        print("[OFF PATH] restart_from unset: no resume state, clock at 0.0\n")

    if failures:
        print(f"OVERALL: FAIL -- {len(failures)} check(s): {failures}")
        return 1
    print("OVERALL: PASS -- every misconfiguration raises at construction, "
          "the accepted case loads, and the off path is untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
