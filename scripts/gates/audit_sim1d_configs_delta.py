"""Say what the reference-configuration layer does to each driver's config.

The rotation record for ``audit_sim1d_configs.py``. That gate answers
pass/fail; a diff of a hash file answers nothing at all. This one answers WHICH
CASE DIGEST MOVED AND WHY, in resolved values, so a reviewer can read a
legitimate rotation rather than take one on trust.

Since the "no default plasma" ruling (2026-09-03) neither campaign driver has a
bare mode, so both driver cases name the reference configuration and their
digests rotated with the rename. For each of them this prints:

* the digest WITHOUT the reference layer -- the pre-rotation value, rebuilt
  from the same code that builds the rotated one so the rotation can be read
  rather than taken on trust;
* the digest WITH it, checked by name against both the committed snapshot and
  the current one;
* every resolved key the layer moves, with both values.

A case this script names but the snapshot does not hold is a failure, not a
False row: the script exits nonzero naming the case.

It also reads the committed snapshot and reports, name by name, which cases the
rotation removed, added, or left alone.

Run from the repository root::

    python scripts/gates/audit_sim1d_configs_delta.py
    python scripts/gates/audit_sim1d_configs_delta.py --revision <rev>
"""

import argparse
import json
import subprocess
from pathlib import Path

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from cablp.solvers._sim1d import resolve_config  # noqa: E402

import audit_sim1d_configs as audit  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Git path of the committed snapshot, for the "against a revision" read.
SNAPSHOT_GIT_PATH = str(audit.SNAPSHOT_PATH.relative_to(REPO_ROOT))

#: Current case name -> the name that case carried BEFORE the rotation. The
#: keys are looked up in the snapshot; the values are HISTORICAL as of
#: 2026-09-03 -- they name no case in any snapshot from that date on, and are
#: printed as section headings only, never looked up.
PRE_ROTATION_NAMES = {
    "run_mechanism_ladder_es1_stance_g1atrim":
        "run_mechanism_ladder_es1_defaults",
    "run_m6_point_es1_stance_g1atrim":
        "run_m6_point_es1_sgp3649_defaults",
}


def committed_snapshot(revision):
    """Return the snapshot JSON as of ``revision``."""
    blob = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{revision}:{SNAPSHOT_GIT_PATH}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(blob)


def unstanced_cases():
    """Return the two driver cases WITHOUT the reference-configuration layer.

    Built by calling ``audit_sim1d_configs.config_cases`` with the layer turned
    into a no-op, so this reproduces the pre-rotation configuration from the
    SAME code that builds the rotated one -- no transcribed second copy that
    could disagree with it.
    """
    original = audit._apply_reference_configuration
    audit._apply_reference_configuration = lambda params, flags: (params, flags)
    try:
        return audit.config_cases()
    finally:
        audit._apply_reference_configuration = original


def _deltas(old, new):
    """Return ``(key, old, new)`` for every resolved value that differs."""
    rows = []
    for space, old_map, new_map in (
        ("params", old[0], new[0]),
        ("flags", old[1], new[1]),
    ):
        for key in sorted(set(old_map) | set(new_map)):
            if old_map.get(key) != new_map.get(key):
                rows.append((f"{space}:{key}", old_map.get(key), new_map.get(key)))
    return rows


def _brief(value):
    text = repr(value)
    return text if len(text) <= 72 else text[:69] + "..."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", default="HEAD",
        help="git revision to read the committed snapshot from (default HEAD)",
    )
    args = parser.parse_args(argv)

    committed = committed_snapshot(args.revision)
    current = audit.current_snapshots()
    old_digests = {k: v["sha256"] for k, v in committed["cases"].items()}
    new_digests = {k: v["sha256"] for k, v in current["cases"].items()}

    print(f"# committed snapshot: {args.revision}:{SNAPSHOT_GIT_PATH}")
    print(
        f"# manifest_sha256 {committed['manifest_sha256']} -> "
        f"{current['manifest_sha256']}  "
        f"({'unchanged' if committed['manifest_sha256'] == current['manifest_sha256'] else 'MOVED'})"
    )
    print(
        f"# parameter_count {committed['parameter_count']} -> "
        f"{current['parameter_count']}, flag_count {committed['flag_count']} -> "
        f"{current['flag_count']}"
    )
    print()
    print("## case roster")
    for name in sorted(set(old_digests) | set(new_digests)):
        if name not in new_digests:
            print(f"REMOVED  {name}  {old_digests[name]}")
        elif name not in old_digests:
            print(f"ADDED    {name}  {new_digests[name]}")
        elif old_digests[name] == new_digests[name]:
            print(f"UNMOVED  {name}  {new_digests[name]}")
        else:
            print(f"MOVED    {name}  {old_digests[name]} -> {new_digests[name]}")

    print()
    print("## why each driver case moved: the reference-configuration layer")
    missing = sorted(n for n in PRE_ROTATION_NAMES if n not in old_digests)
    if missing:
        print(
            "ERROR: case(s) this script explains are absent from "
            f"{args.revision}:{SNAPSHOT_GIT_PATH}: {', '.join(missing)}"
        )
        return 1
    bare = unstanced_cases()
    stanced = audit.config_cases()
    for name, was_called in sorted(PRE_ROTATION_NAMES.items()):
        before = resolve_config(*bare[name])
        after = resolve_config(*stanced[name])
        before_digest = audit.config_digest(*before)
        after_digest = audit.config_digest(*after)
        print()
        print(f"### {was_called} -> {name}")
        print(f"  without the reference layer: {before_digest}"
              "  (pre-rotation value)")
        print(f"  with the reference layer:    {after_digest}")
        print(
            "    matches the committed digest for "
            f"{name}: {after_digest == old_digests[name]}"
        )
        print(
            "    matches the rotated digest: "
            f"{after_digest == new_digests.get(name)}"
        )
        rows = _deltas(before, after)
        print(f"  resolved keys the layer moves: {len(rows)}")
        for key, was, now in rows:
            print(f"    {key}: {_brief(was)} -> {_brief(now)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
