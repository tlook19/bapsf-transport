"""Show that the edt digest-gate move is CONFIG IDENTITY ONLY.

``golden_digest_gate.py`` prints the failure lines it collects and not the
digests it agreed on, so on the identity-only rotation class its transcript
says what moved and leaves what did not to be inferred from an absence. This
says both, from ONE run of the gate's own machinery -- no second implementation
of the config layering or of the digest:

1. the fresh record at the current tree (checkpoints and final digest), beside
   the committed reference, row by row;
2. the STRIP-3-KEYS CONTROL: the same config with this branch's three added
   keys removed from BOTH namespaces -- two params and one flag -- hashed
   through ``golden_digest_gate.config_identity``, which must return the base
   commit's identity BIT-FOR-BIT. That is what makes "the identity moved
   because three keys were added" a measurement rather than an inference from
   the key count. The control runs through the DIGEST GATE'S OWN expression,
   because the two golden references legitimately carry different identities
   and a control computed through the other one matches neither.

The control is a hash, not a solve; the record costs one 4,000-step run. The
in-process import provenance is printed with it, so a transcript captured in a
worktree states which ``cablp`` it actually exercised.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/edt_digest_rotation.py
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import cablp  # noqa: E402
from cablp.cathode.kernels import KERNEL_ID, PROVENANCE  # noqa: E402

from baseline_sim1d import build_baseline_config  # noqa: E402
from golden_digest_gate import (  # noqa: E402
    DEFAULT_REFERENCE,
    DIGEST_PARAM_OVERRIDES,
    baseline_digest_record,
    config_identity,
)

#: The params-namespace keys this branch adds.
ADDED_PARAMS = (
    "electron_drift_charge_death",
    "electron_drift_anode_handshake",
)

#: The flags-namespace key this branch adds.
ADDED_FLAGS = ("electron_drift_transport",)


def main():
    print("[edt] digest-gate rotation: identity moves, trajectory does not")
    print("=" * 78)
    print(f"cablp.__file__ = {cablp.__file__}")
    print(f"KERNEL_ID      = {KERNEL_ID}  (provenance {PROVENANCE})")
    print("=" * 78)

    reference = json.loads(Path(DEFAULT_REFERENCE).read_text())
    params, flags = build_baseline_config(DIGEST_PARAM_OVERRIDES)

    stripped_params = {
        k: v for k, v in params.items() if k not in ADDED_PARAMS
    }
    stripped_flags = {k: v for k, v in flags.items() if k not in ADDED_FLAGS}
    control = config_identity(stripped_params, stripped_flags)
    print("STRIP-3-KEYS CONTROL (no solve)")
    print(f"  branch identity, all keys      : {config_identity(params, flags)}")
    print(f"  branch identity, 3 keys removed: {control}")
    print(f"  committed reference identity   : {reference['config_identity']}")
    print(
        "  control reproduces the base identity bit-for-bit: "
        f"{control == reference['config_identity']}"
    )
    print(
        f"  params {len(params)} -> {len(stripped_params)}; "
        f"flags {len(flags)} -> {len(stripped_flags)}"
    )
    print("=" * 78)

    record = baseline_digest_record(steps=int(reference["steps"]))
    print(
        f"TRAJECTORY at {record['steps']} accepted steps "
        f"(kernels={record['kernel_provenance']})"
    )
    same = True
    for key in sorted(reference["checkpoints"], key=int):
        fresh = record["checkpoints"].get(key)
        want = reference["checkpoints"][key]
        match = fresh == want
        same = same and match
        print(f"  checkpoint {key:>4}: {'==' if match else '!='}  {fresh}")
    digest_match = record["digest"] == reference["digest"]
    same = same and digest_match
    print(
        f"  final digest    : {'==' if digest_match else '!='}  "
        f"{record['digest']}"
    )
    print(f"  reference digest:      {reference['digest']}")
    print(f"  final_time {record['final_time']!r}, cells {record['cells']}")
    print(f"  reference final_time {reference['final_time']!r}, "
          f"cells {reference['cells']}")
    print("=" * 78)
    ok = same and control == reference["config_identity"]
    print(
        "IDENTITY-ONLY ROTATION: trajectory unchanged, identity explained by "
        "the three added keys"
        if ok
        else "NOT AN IDENTITY-ONLY ROTATION -- read the rows above"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
