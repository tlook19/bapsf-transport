"""Identity-rotation evidence for the accommodation adoption: ONE solve.

The claim this pass has to prove about the golden is narrow and specific:
the moment path never READS the DVM keys, so re-valuing
``neutral_kinetic_dvm_accommodation`` must move the resolved-config IDENTITY
and NOTHING ELSE -- the trajectory chain bit-identical at every checkpoint,
the final digest unchanged, and a stripped-key reproduction returning the
base identity bit-for-bit.

``golden_digest_gate.py`` decides that question but does not SHOW it: on a
config-identity failure it prints the identity line and stops, so a reader
cannot tell from its transcript whether the trajectory also moved. This
wrapper runs the gate's own ``main()`` UNMODIFIED -- same verdict, same
verbatim transcript -- and only tees the record ``baseline_digest_record``
produced, so the field-by-field comparison costs no second solve.

It also runs the STRIP-RESTORE CONTROL, which needs no solve at all: rebuild
the golden config on this tree, put the DVM keys back to their base values,
recompute the identity, and check it returns to the base reference's.

Run from the worktree root with ``PYTHONPATH=<worktree>``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cablp  # noqa: E402
from cablp.cathode.kernels import PROVENANCE as KERNEL_PROVENANCE  # noqa: E402

import golden_digest_gate as gate  # noqa: E402
from baseline_sim1d import build_baseline_config  # noqa: E402

#: The base commit this branch is built on.
BASE_REV = "3e32967"

#: Every DVM key, with the value the BASE tree resolved it to. Restoring the
#: whole family (not only the re-valued one) is what makes the control say
#: "the DVM keys own the entire identity move" rather than only "this one
#: key moves it".
BASE_DVM_VALUES = {
    "neutral_kinetic_dvm_accommodation": 1.0,
    "neutral_kinetic_dvm_wall_reflection": "specular",
    "neutral_kinetic_dvm_cathode_jet": False,
    "neutral_kinetic_dvm_cathode_jet_R_N": 0.34,
    "neutral_kinetic_dvm_cathode_jet_R_E": 0.18,
    "neutral_kinetic_dvm_cathode_jet_T_launch_eV": None,
}


def _provenance():
    """In-process import-provenance assertion (AGENTS.md worktree bring-up)."""
    pkg = os.path.abspath(cablp.__file__)
    print(f"cablp.__file__ = {pkg}")
    print(f"worktree root  = {ROOT}")
    print(f"KERNEL_ID      = {KERNEL_PROVENANCE}")
    inside = pkg.startswith(str(ROOT) + os.sep)
    print(f"import resolves inside the worktree: {inside}")
    assert inside, "PYTHONPATH TRAP: cablp resolved outside the worktree"


def _base_reference(rev):
    rel = "scripts/baselines/golden_digest_4k.json"
    out = subprocess.run(
        ["git", "show", f"{rev}:{rel}"],
        capture_output=True, text=True, check=True, cwd=str(ROOT),
    ).stdout
    return json.loads(out)


def strip_restore_control(base_ref):
    """Recompute the golden config identity with the DVM keys at base."""
    print()
    print("=" * 78)
    print("STRIP-RESTORE CONTROL (no solve: identity is a pure config hash)")
    print("=" * 78)
    params, flags = build_baseline_config(gate.DIGEST_PARAM_OVERRIDES)
    live = gate.config_identity(params, flags)
    restored = dict(params)
    for key, value in BASE_DVM_VALUES.items():
        print(f"  restore {key:46s} {restored.get(key)!r} -> {value!r}")
        restored[key] = value
    back = gate.config_identity(restored, flags)
    want = base_ref["config_identity"]
    print()
    print(f"  identity on this tree as built      : {live}")
    print(f"  identity with the DVM keys restored : {back}")
    print(f"  base {BASE_REV} committed identity      : {want}")
    ok = back == want
    print(
        f"  restores the base identity bit-for-bit: {ok}"
    )
    print(
        f"  and the live identity differs from it : {live != want}"
    )
    return ok


def compare_record(record, base_ref):
    """Field-by-field comparison of the fresh record against the base."""
    print()
    print("=" * 78)
    print(f"FRESH RECORD vs BASE REFERENCE ({BASE_REV})")
    print("=" * 78)
    traj_ok = True

    for field in (
        "digest_format", "steps", "checkpoint_interval",
        "cells", "fields_per_cell", "final_time", "digest",
    ):
        a, b = base_ref.get(field), record.get(field)
        same = a == b
        if field != "digest_format":
            traj_ok = traj_ok and same
        print(f"  {field:20s} {'IDENTICAL' if same else 'MOVED':10s} "
              f"{a}  |  {b}")

    print("  checkpoints (the trajectory chain, per 1000 accepted steps):")
    for key in sorted(base_ref["checkpoints"], key=int):
        a = base_ref["checkpoints"][key]
        b = record["checkpoints"].get(key)
        same = a == b
        traj_ok = traj_ok and same
        print(f"    step {key:>5s}  {'IDENTICAL' if same else 'MOVED':10s} {a}")
        if not same:
            print(f"                        fresh: {b}")

    a, b = base_ref["config_identity"], record["config_identity"]
    moved = a != b
    print(f"  {'config_identity':20s} {'MOVED' if moved else 'IDENTICAL':10s}")
    print(f"    base  {a}")
    print(f"    fresh {b}")

    print()
    print(f"  TRAJECTORY BIT-UNCHANGED (checkpoints + digest + time): {traj_ok}")
    print(f"  CONFIG IDENTITY ROTATED                              : {moved}")
    print(
        "  => identity-only rotation"
        if (traj_ok and moved)
        else "  => NOT an identity-only rotation -- read the rows above"
    )
    return traj_ok and moved


def main():
    _provenance()
    base_ref = _base_reference(BASE_REV)

    captured = {}
    real = gate.baseline_digest_record

    def teed(*args, **kwargs):
        record = real(*args, **kwargs)
        captured["record"] = record
        return record

    gate.baseline_digest_record = teed
    print()
    print("=" * 78)
    print("GATE TRANSCRIPT (scripts/golden_digest_gate.py, logic UNMODIFIED)")
    print("=" * 78)
    try:
        rc = gate.main([])
    finally:
        gate.baseline_digest_record = real
    print(f"gate exit code: {rc}")

    record = captured.get("record")
    if record is None:
        print("no record captured -- the gate did not reach the solve")
        return 2
    Path(ROOT / "scripts" / "dacc_digest_fresh_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )

    identity_only = compare_record(record, base_ref)
    control_ok = strip_restore_control(base_ref)

    print()
    print("=" * 78)
    print(f"identity-only rotation demonstrated : {identity_only}")
    print(f"strip-restore control returns base  : {control_ok}")
    print(
        "EVIDENCE PACKAGE COMPLETE"
        if (identity_only and control_ok)
        else "EVIDENCE PACKAGE INCOMPLETE"
    )
    print(
        "NOTE: the gate's own exit code is 1 BY DESIGN here -- a config "
        "identity move is a recapture question, and rotating "
        "scripts/baselines/ is the REVIEWER's action, not this branch's."
    )
    return 0 if (identity_only and control_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
