"""Show that the B4 digest-gate move is CONFIG IDENTITY ONLY.

``golden_digest_gate.py`` stops at the first failure line it collects and does
not print the digests it agreed on, so on the identity-only rotation class its
transcript says what moved and not what did not. This says both, from ONE run
of the gate's own machinery -- no second implementation of the config layering
or of the digest:

1. the fresh record at the current tree (checkpoints and final digest), beside
   the committed reference, row by row;
2. the STRIP-4-KEYS CONTROL: the same config with this branch's four added
   keys removed, hashed through ``golden_digest_gate.config_identity``, which
   must return the base commit's identity BIT-FOR-BIT. That is what makes
   "the identity moved because four keys were added" a measurement rather than
   an inference from the key count.

The control is a hash, not a solve; the record costs one 4,000-step run.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/b4aj_digest_rotation.py
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import cablp  # noqa: E402
from cablp.cathode.kernels import KERNEL_ID  # noqa: E402

from baseline_sim1d import build_baseline_config  # noqa: E402
from golden_digest_gate import (  # noqa: E402
    DEFAULT_REFERENCE,
    DIGEST_PARAM_OVERRIDES,
    baseline_digest_record,
    config_identity,
)

#: The four keys this branch adds, all params-namespace.
ADDED_KEYS = (
    "neutral_kinetic_dvm_anode_jet",
    "neutral_kinetic_dvm_anode_jet_R_N",
    "neutral_kinetic_dvm_anode_jet_R_E",
    "neutral_kinetic_dvm_anode_jet_T_launch_eV",
)


def main():
    print("[B4] digest-gate rotation: identity moves, trajectory does not")
    print("=" * 78)
    print(f"cablp.__file__ = {cablp.__file__}")
    print(f"KERNEL_ID      = {KERNEL_ID}")
    print("=" * 78)

    reference = json.loads(Path(DEFAULT_REFERENCE).read_text())
    params, flags = build_baseline_config(DIGEST_PARAM_OVERRIDES)

    stripped = {k: v for k, v in params.items() if k not in ADDED_KEYS}
    control = config_identity(stripped, flags)
    print("STRIP-4-KEYS CONTROL (no solve)")
    print(f"  branch identity, all keys      : {config_identity(params, flags)}")
    print(f"  branch identity, 4 keys removed: {control}")
    print(f"  committed reference identity   : {reference['config_identity']}")
    print(
        "  control reproduces the base identity bit-for-bit: "
        f"{control == reference['config_identity']}"
    )
    print(
        f"  params {len(params)} -> {len(stripped)} with the four removed; "
        f"flags {len(flags)} (unchanged)"
    )
    print("=" * 78)

    record = baseline_digest_record(steps=int(reference["steps"]))
    print(f"TRAJECTORY at {record['steps']} accepted steps "
          f"(kernels={record['kernel_provenance']})")
    same = True
    for key in sorted(reference["checkpoints"], key=int):
        fresh = record["checkpoints"].get(key)
        want = reference["checkpoints"][key]
        match = fresh == want
        same = same and match
        print(f"  checkpoint {key:>4}: {'==' if match else '!='}  {fresh}")
    digest_match = record["digest"] == reference["digest"]
    same = same and digest_match
    print(f"  final digest    : {'==' if digest_match else '!='}  "
          f"{record['digest']}")
    print(f"  reference digest:      {reference['digest']}")
    print(f"  final_time {record['final_time']!r}, cells {record['cells']}")
    print("=" * 78)
    ok = same and control == reference["config_identity"]
    print(
        "IDENTITY-ONLY ROTATION: trajectory unchanged, identity explained by "
        "the four added keys"
        if ok
        else "NOT AN IDENTITY-ONLY ROTATION -- read the rows above"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
