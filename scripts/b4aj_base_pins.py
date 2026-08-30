"""[B4] Measure the pre-registered pins AT BASE, before the channel exists.

The 2026-08-30 standing rule: a pre-registered numeric pin is measured at the
base commit before it is gated on. This prints, in one transcript:

* the golden digest gate's CONFIG IDENTITY, computed through the gate's own
  expression (``config_identity(*build_baseline_config(DIGEST_PARAM_OVERRIDES))``)
  rather than a re-implementation of the layering;
* the committed digest reference's identity, final digest and checkpoints, so
  the "identity moves, trajectory does not" statement has both halves on
  record;
* the in-process import provenance (``cablp.__file__``, ``KERNEL_ID``), the
  worktree-gate requirement.

It reads the tree and computes hashes; it runs no solver and writes nothing.
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
    config_identity,
)


def main():
    print("[B4] base pin measurement (no solve)")
    print("=" * 78)
    print(f"cablp.__file__ = {cablp.__file__}")
    print(f"KERNEL_ID      = {KERNEL_ID}")
    print("=" * 78)

    params, flags = build_baseline_config(DIGEST_PARAM_OVERRIDES)
    measured = config_identity(params, flags)
    print(f"MEASURED config identity (through the gate's own expression):")
    print(f"  {measured}")
    print(f"  params keys: {len(params)}   flags keys: {len(flags)}")

    reference = json.loads(Path(DEFAULT_REFERENCE).read_text())
    print("committed reference scripts/baselines/golden_digest_4k.json:")
    print(f"  config_identity = {reference['config_identity']}")
    print(f"  digest          = {reference['digest']}")
    print(f"  steps           = {reference['steps']}")
    for step, value in sorted(
        reference["checkpoints"].items(), key=lambda kv: int(kv[0])
    ):
        print(f"  checkpoint {step:>4} = {value}")
    print(
        "  identity matches the fresh build: "
        f"{measured == reference['config_identity']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
