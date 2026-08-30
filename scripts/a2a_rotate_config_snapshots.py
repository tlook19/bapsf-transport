"""Regenerate ``cablp/solvers/_sim1d/config_snapshots.json`` for the A2a keys.

The resolved-config snapshot is a config-IDENTITY fixture: it pins the default
manifest's parameter and flag counts and a digest of each driver's resolved
config, so that an unannounced configuration change is loud. Registering new
keys is exactly the announced kind, and the fixture rotates in the same change
set -- the practice the b-removal, B3 and B5 commits already follow.

It is regenerated PROGRAMMATICALLY, from the audit module's own
``current_snapshots()``, never hand-edited. The delta this rotation is entitled
to make is printed before it is written, and the script REFUSES to write if the
manifest moved by anything other than the three A2a keys at their off defaults.

Run from the worktree root with PYTHONPATH=<worktree>.
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import audit_sim1d_configs as A  # noqa: E402

#: The keys this rotation is entitled to add, and the default each must carry.
ENTITLED = {
    "flags": {"beam_tail_anode_interception": False},
    "parameters": {
        "beam_tail_anode_reflected_particles": 0.0,
        "beam_tail_anode_reflected_energy": 0.0,
    },
}


def main():
    base_manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    manifest = A.config_manifest()
    if base_manifest_path is not None:
        base = json.loads(base_manifest_path.read_text())
        ok = True
        for section, added in ENTITLED.items():
            b = set(base[section])
            c = set(manifest[section])
            gained = c - b
            lost = b - c
            print(f"{section}: gained {sorted(gained)}, lost {sorted(lost)}")
            if gained != set(added) or lost:
                ok = False
            for key in sorted(b & c):
                if base[section][key] != manifest[section][key]:
                    print(f"  MOVED: {section}.{key}: "
                          f"{base[section][key]} -> {manifest[section][key]}")
                    ok = False
            for key, want in added.items():
                got = manifest[section].get(key, {}).get("default")
                print(f"  new {section}.{key} default = {got!r} "
                      f"(entitled {want!r})")
                if got != want:
                    ok = False
        if not ok:
            print("REFUSING TO WRITE: the manifest moved beyond the three A2a "
                  "keys at their off defaults")
            return 1
        print("manifest delta is exactly the entitled addition")

    snapshots = A.current_snapshots()
    old = json.loads(A.SNAPSHOT_PATH.read_text())
    print(f"parameter_count {old['parameter_count']} -> "
          f"{snapshots['parameter_count']}")
    print(f"flag_count      {old['flag_count']} -> {snapshots['flag_count']}")
    for name in sorted(snapshots["cases"]):
        print(f"{name}: {old['cases'][name]['sha256']} -> "
              f"{snapshots['cases'][name]['sha256']}")
    A.SNAPSHOT_PATH.write_text(
        json.dumps(snapshots, sort_keys=True, indent=2) + "\n"
    )
    print(f"wrote {A.SNAPSHOT_PATH}")
    A.verify_snapshots()
    print("verify_snapshots() OK at the rotated fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
