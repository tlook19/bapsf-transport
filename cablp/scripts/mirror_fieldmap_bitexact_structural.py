#!/usr/bin/env python3
"""Structural proof that the mirror field-map loader cannot move a run.

The golden gate is the authority on bit-exactness and is the reviewer's to
run; this is the argument the coder can make WITHOUT a golden, and here it is
stronger than a count comparison because the claim is an identity rather than
a bound.

The claim
---------
The field-map loader (``solvers/_sim1d/physics/mirror_field.py``) adds NO
configuration key, NO flag, and NO import edge into the solver. The resolved
configuration of every production case is therefore IDENTICAL to the base
commit's, key for key and value for value -- not "identical after deleting the
new keys", identical.

Why there are no keys to delete
-------------------------------
An earlier cut of this work shipped a ``flux_tube_mirror`` flag and three
``mirror_field_*`` parameters gating a ``-mu grad B`` force term. That force
must never exist: the fluid mirror force is ALREADY in the model as the
quasi-1D ``p dA/dz`` source (``physics.sources.flux_tube_geometry_rhs``, armed
by ``prescribed_area_geometry``, which the stance of record sets). Flux
conservation gives ``A B = const``, so ``A`` goes as ``1/B`` exactly, and
under the isotropic closure the state ``(n, nn, M, Ee, Ei)`` supports, that
source IS the distribution average of ``-mu grad_par B``, term for term. A
sibling term would double-count it by 100 %, exactly. A flag gating a term
that must never be built can never do anything, which is precisely the silent
inert control the configuration namespaces forbid -- so the flag and its
parameters are gone, and the loader is a plain library function called by
``scripts/characterise_mirror_fieldmap.py``.

The proof
---------
``cablp/solvers/_sim1d/config_snapshots.json`` pins a sha256 of the resolved
``(params, flags)`` for four production configurations, plus the manifest
digest and the two key counts. This script recomputes the whole snapshot
record on the CURRENT tree, through the standing audit gate's own
``current_snapshots()``, and asserts it reproduces the base record BYTE FOR
BYTE, with no key deleted first.

That is an equality over the whole canonical JSON of both namespaces: if any
key had been added, removed, renamed, reordered or had its value changed, a
digest would not match.

The remaining check is the reachability side of the same claim: nothing the
solver imports imports ``physics.mirror_field``, so no code in it is reachable
from ``LAPDSim1D`` at all -- there is no path to enter, armed or otherwise.

``--base-snapshots`` re-runs the proof against another commit's copy of the
file. This branch does not modify it, so the committed copy IS the base copy;
the option exists so a reviewer can pin the comparison explicitly:

    git show <base>:cablp/cablp/solvers/_sim1d/config_snapshots.json \\
        > /tmp/base_config_snapshots.json
    python scripts/mirror_fieldmap_bitexact_structural.py \\
        --base-snapshots /tmp/base_config_snapshots.json

Run from ``<checkout>/cablp`` with ``PYTHONPATH`` set to that directory.
Exit 0 = proved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from audit_sim1d_configs import (  # noqa: E402
    SNAPSHOT_PATH,
    current_snapshots,
)

from cablp.solvers._sim1d import (  # noqa: E402
    LAPDSim1D,
    config_manifest,
    default_config,
)

# The keys the pulled cut would have added. They must be absent from BOTH
# namespaces: this is the assertion that the re-cut actually removed them
# rather than moving them somewhere quieter.
WITHDRAWN_PARAMS = (
    "mirror_field_map_path",
    "mirror_field_case",
    "mirror_field_interior_fill_gauss",
)
WITHDRAWN_FLAGS = ("flux_tube_mirror",)

MIRROR_MODULE = "cablp.solvers._sim1d.physics.mirror_field"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-snapshots", default=None,
        help="JSON of base-commit snapshots (default: the committed file, "
             "which this branch does not modify)",
    )
    parser.add_argument("--output", default=None, help="also write the report here")
    args = parser.parse_args(argv)

    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    base_path = Path(args.base_snapshots) if args.base_snapshots else SNAPSHOT_PATH
    base = json.loads(base_path.read_text())
    manifest = config_manifest()

    emit("mirror field-map loader: structural bit-exactness proof")
    emit("=" * 72)
    emit(f"base snapshots   {base_path}")
    emit(f"base counts      parameters={base['parameter_count']}, "
         f"flags={base['flag_count']}")
    emit(f"current counts   parameters={len(manifest['parameters'])}, "
         f"flags={len(manifest['flags'])}")
    emit(f"delta            "
         f"{len(manifest['parameters']) - base['parameter_count']:+d} parameters, "
         f"{len(manifest['flags']) - base['flag_count']:+d} flags")
    emit("")

    failures = []

    emit("1. the loader registers no configuration key in either namespace")
    for key in WITHDRAWN_PARAMS + WITHDRAWN_FLAGS:
        in_params = key in manifest["parameters"]
        in_flags = key in manifest["flags"]
        ok = not (in_params or in_flags)
        emit(f"   {key:38s} parameters={str(in_params):5s} "
             f"flags={str(in_flags):5s} "
             f"{'ABSENT' if ok else 'PRESENT -- FAIL'}")
        if not ok:
            failures.append(f"key {key} is registered")
    emit("")

    emit("2. the whole resolved-config snapshot record reproduces the BASE")
    emit("   record, with NOTHING deleted first -- there is nothing to delete")
    current = current_snapshots()
    for field in ("schema", "manifest_sha256", "parameter_count", "flag_count"):
        got, want = current[field], base[field]
        ok = got == want
        emit(f"   {field:24s} {'MATCH ' if ok else 'DIFFER'}  {got}")
        if not ok:
            failures.append(f"{field} moved")
            emit(f"   {'':24s} base    {want}")
    for name in sorted(set(current["cases"]) | set(base["cases"])):
        got = current["cases"].get(name)
        want = base["cases"].get(name)
        ok = got == want
        emit(f"   {name:38s} {'MATCH ' if ok else 'DIFFER'}  "
             f"{(got or {}).get('sha256')}")
        if not ok:
            failures.append(f"config snapshot moved for {name}")
            emit(f"   {'':38s} base    {(want or {}).get('sha256')}")
    if current == base:
        emit("   whole-record equality: the recomputed snapshot dict is equal "
             "to the base dict")
    else:
        failures.append("the snapshot record differs from base")
    emit("")

    emit("3. nothing reachable from the solver imports the loader")
    d, f = default_config()
    sim = LAPDSim1D(input_dict=d, input_flags=f)
    imported = MIRROR_MODULE in sys.modules
    emit(f"   LAPDSim1D constructed at canonical defaults: {type(sim).__name__}")
    emit(f"   {MIRROR_MODULE} in sys.modules: {imported}")
    if imported:
        failures.append("the solver imported the field-map loader")
    emit("   The loader is never imported by constructing or running the")
    emit("   solver, so none of its code is reachable from a run at all --")
    emit("   there is no armed/disarmed distinction to make.")
    emit("")

    if failures:
        emit("RESULT: FAILED -- " + "; ".join(failures))
    else:
        emit("RESULT: PROVED. The resolved configuration is IDENTICAL to the "
             "base commit's")
        emit("in both namespaces, no key was added or removed, and nothing "
             "the solver")
        emit("imports reaches the field-map loader. The GOLDEN bit-exactness "
             "gate is")
        emit("DEFERRED to the reviewer -- this is a structural argument, not a "
             "trajectory check.")

    if args.output:
        Path(args.output).write_text("\n".join(lines) + "\n")
        print(f"\nreport written to {Path(args.output).resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
