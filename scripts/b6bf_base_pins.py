"""[B6] Measure the pre-registered pins AT BASE, before the channel exists.

The 2026-08-30 standing rule: a pre-registered numeric pin is measured at the
base commit before it is gated on. This prints, in one transcript:

* the golden digest gate's CONFIG IDENTITY, computed through the gate's own
  expression (``config_identity(*build_baseline_config(DIGEST_PARAM_OVERRIDES))``)
  rather than a re-implementation of the layering -- the two committed golden
  references legitimately carry DIFFERENT identities, and a control computed
  through the wrong expression matches neither;
* the committed digest reference's identity, final digest and checkpoints, so
  the "identity moves, trajectory does not" statement has both halves on
  record;
* the in-process import provenance (``cablp.__file__``, ``KERNEL_ID``), which
  is the worktree-gate requirement: the editable install's ``cablp.pth``
  points at the MAIN checkout and can serve its code under a worktree
  ``PYTHONPATH``;
* the g1atrim baffle geometry B6's plan gate is registered on -- the face the
  stance's one baffle maps to, its clear radius, the face-average column
  radius there, the open ring, the annulus face area and the ratio between
  them, which is the plan's "~1.75x" as this stance actually realizes it.

It reads the tree, builds one geometry and computes hashes; it runs no solver
and writes nothing.
"""

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import cablp  # noqa: E402
from cablp.cathode.kernels import KERNEL_ID, PROVENANCE  # noqa: E402
from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402
from cablp.solvers._sim1d.physics.kinetic_dvm import _throat_areas  # noqa: E402
from cablp.solvers._sim1d.physics.neutrals import (  # noqa: E402
    neutral_zone_volumes,
)

from baseline_sim1d import build_baseline_config  # noqa: E402
from golden_digest_gate import (  # noqa: E402
    DEFAULT_REFERENCE,
    DIGEST_PARAM_OVERRIDES,
    config_identity,
)
from verify_sim1d_k2_dvm import (  # noqa: E402
    PRODUCTION_GEOMETRY_KEYS,
    arm_config,
)


def main():
    print("[B6] base pin measurement (no solve)")
    print("=" * 78)
    resolved = Path(cablp.__file__).resolve()
    root = SCRIPT_DIR.parent
    print(f"cablp.__file__ = {resolved}")
    print(f"checkout root  = {root}")
    print(f"import is inside this checkout: {root in resolved.parents}")
    print(f"KERNEL_ID      = {KERNEL_ID}")
    print(f"PROVENANCE     = {PROVENANCE}")
    print("=" * 78)

    params, flags = build_baseline_config(DIGEST_PARAM_OVERRIDES)
    measured = config_identity(params, flags)
    print("MEASURED config identity (through the digest gate's own expression):")
    print(f"  {measured}")
    print(f"  params keys: {len(params)}   flags keys: {len(flags)}")

    reference = json.loads(Path(DEFAULT_REFERENCE).read_text())
    print("committed reference scripts/baselines/golden_digest_4k.json:")
    print(f"  config_identity = {reference['config_identity']}")
    print(f"  digest          = {reference['digest']}")
    for step in sorted(reference["checkpoints"], key=int):
        print(f"  checkpoint {step:>5} = {reference['checkpoints'][step]}")
    print(
        "  identity matches the measured one: "
        f"{measured == reference['config_identity']}"
    )
    print("=" * 78)

    d, fl = arm_config(**PRODUCTION_GEOMETRY_KEYS)
    geom = build_geometry(d, fl)
    faces = np.asarray(geom.neutral_baffle_face_indices, dtype=int)
    radii = np.asarray(geom.neutral_baffle_clear_radius_cm, dtype=float)
    Rp = np.asarray(geom.Rp_cm, dtype=float)
    Rm = np.asarray(geom.Rm_cm, dtype=float)
    _V_col, V_ann = neutral_zone_volumes(geom)
    face_a = _throat_areas(V_ann / np.asarray(geom.length_cm, dtype=float))
    z_edges = np.asarray(geom.z_edges_cm, dtype=float)
    print(f"g1atrim geometry: {geom.cells} cells, {faces.size} baffle(s)")
    for face, clear in zip(faces, radii):
        face = int(face)
        R_col = 0.5 * (float(Rp[face - 1]) + float(Rp[face]))
        open_ann = np.pi * (float(clear) ** 2 - R_col**2)
        area = float(face_a[face])
        print(
            f"  face {face} at z = {z_edges[face]:.4f} cm: R_clear "
            f"{clear} cm, R_col (face average) {R_col:.6f} cm, Rm "
            f"{float(Rm[face]):.4f} cm"
        )
        print(
            f"    open_ann {open_ann:.6f} cm^2, A_ann (throat) {area:.6f} "
            f"cm^2, t_f = {open_ann / area:.12f}, A_ann / open_ann = "
            f"{area / open_ann:.12f}"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
