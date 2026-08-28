"""Per-step state digests for the [b-removal] bit-exactness A/B.

Runs a short fixed-configuration LAPDSim1D and prints a SHA-256 over the raw
float64 packed state after every accepted step, plus a running chain digest
over all of them. Two trees (base 39cb395 vs the removal branch) must produce
byte-identical output.

The production golden cannot serve as the gate at this tip -- the R3 window
suspends it by design -- so this self-controlled A/B is the bit-exactness
evidence: identical code paths are proven by identical raw state bytes.

Two configurations are covered:

* ``--config default``  -- ``default_config()`` unchanged.
* ``--config g1atrim``  -- the golden's own config, via
  ``baseline_sim1d.build_baseline_config()`` (``default_config()`` + the
  committed ``scripts/stances/g1atrim.toml`` minus its mesh-sized package +
  ``nx = 60``). Nothing under ``scripts/baselines/`` is read or written.

Usage::

    PYTHONPATH=<tree> python -P scripts/bremoval_ab_digest.py \
        --config default --steps 400
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from cablp.solvers._sim1d import LAPDSim1D, default_config  # noqa: E402
from baseline_sim1d import build_baseline_config  # noqa: E402


def build(config_name):
    if config_name == "default":
        return default_config()
    if config_name == "g1atrim":
        return build_baseline_config()
    raise ValueError(f"unknown config {config_name!r}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=("default", "g1atrim"))
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args(argv)

    params, flags = build(args.config)
    # Keep the A/B cheap and self-contained: no cached neutral seed and no
    # equilibration, so the digests describe the stepped solver and nothing
    # about a cache file that may or may not exist beside either tree.
    flags["use_cached_neutral_seed"] = False
    flags["neutral_equilibration"] = False

    import cablp

    print(f"cablp={cablp.__file__}")
    print(f"config={args.config} steps={args.steps}")

    sim = LAPDSim1D(params, flags)
    running = hashlib.sha256()
    for index in range(1, args.steps + 1):
        dt = sim.suggest_timestep().dt
        snapshot = sim.advance_one_step(dt)
        raw = np.ascontiguousarray(snapshot.y, dtype=np.float64).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        running.update(digest.encode("ascii"))
        if index <= 3 or index % 100 == 0 or index == args.steps:
            print(f"step {index:5d} t={sim.time:.17e} {digest}")
    print(f"STEPS={args.steps}")
    print(f"CHAIN={running.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
