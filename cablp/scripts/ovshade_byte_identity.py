"""Pre-registered check for the discharge-spread overlay candidate.

Compares each candidate NPZ against the PROMOTED product that gates scoring
(``cablp/scripts/data/esN_sim1d_overlay.npz``) and asserts the promotion-safety
property claimed for the change:

  every array present in BOTH files is byte-identical at raw buffer level
  (dtype, shape and ``tobytes()``), the candidate ADDS only the declared
  spread keys, and it REMOVES nothing.

``schema_version`` is the one declared exception: it is the version marker
whose whole job is to change when fields are added, exactly as at the v6
promotion (``ac55eda``, "+ te_window_spread_frac, all else byte-identical").
It is reported separately and never counted as an incidental difference.

Usage:
  python scripts/ovshade_byte_identity.py --candidate-dir <dir>
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROMOTED_DIR = HERE / "data"

# Keys the change is allowed to add. Anything else new is a finding.
EXPECTED_NEW = {
    "discharge_current_sd_a",
    "discharge_voltage_sd_v",
    "discharge_spread_alignment",
}
# The version marker, excepted by declaration rather than by discovery.
SCHEMA_KEY = "schema_version"


def _raw_equal(a, b):
    """Raw-buffer equality: same dtype, same shape, same bytes."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return a.tobytes() == b.tobytes()


def check_one(es, candidate_path, out_lines):
    promoted_path = PROMOTED_DIR / f"es{es}_sim1d_overlay.npz"
    promoted = np.load(promoted_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)

    p_keys = set(promoted.files)
    c_keys = set(candidate.files)
    added = c_keys - p_keys
    removed = p_keys - c_keys
    shared = sorted(p_keys & c_keys)

    out_lines.append(f"=== ES{es} ===")
    out_lines.append(f"promoted : {promoted_path}")
    out_lines.append(f"candidate: {candidate_path}")
    out_lines.append(
        f"keys: promoted {len(p_keys)}, candidate {len(c_keys)}, "
        f"shared {len(shared)}, added {len(added)}, removed {len(removed)}"
    )

    ok = True

    if removed:
        ok = False
        out_lines.append(f"FAIL removed keys: {sorted(removed)}")
    if added != EXPECTED_NEW:
        ok = False
        out_lines.append(
            f"FAIL added keys {sorted(added)} != expected {sorted(EXPECTED_NEW)}"
        )
    else:
        out_lines.append(f"added keys as declared: {sorted(added)}")

    differing = []
    for key in shared:
        if key == SCHEMA_KEY:
            continue
        if not _raw_equal(promoted[key], candidate[key]):
            differing.append(key)

    n_compared = len(shared) - (1 if SCHEMA_KEY in shared else 0)
    if differing:
        ok = False
        out_lines.append(
            f"FAIL {len(differing)}/{n_compared} shared arrays differ: {differing}"
        )
    else:
        out_lines.append(
            f"byte-identical: {n_compared}/{n_compared} shared arrays "
            "(dtype, shape, raw bytes)"
        )

    out_lines.append(
        f"{SCHEMA_KEY} (declared exception): promoted "
        f"{int(promoted[SCHEMA_KEY])} -> candidate {int(candidate[SCHEMA_KEY])}"
    )

    # Report the new band so the numbers are on the record, not just the verdict.
    if "discharge_current_sd_a" in c_keys:
        sd = np.asarray(candidate["discharge_current_sd_a"], dtype=float)
        sem = np.asarray(candidate["discharge_current_sem_a"], dtype=float)
        n = int(candidate["discharge_n_traces"])
        ratio = sd / np.where(sem > 0, sem, np.nan)
        out_lines.append(
            f"I_dis sd: max {np.nanmax(sd):.1f} A, median {np.nanmedian(sd):.2f} A; "
            f"sd/sem median {np.nanmedian(ratio):.4f} vs sqrt(n)={np.sqrt(n):.4f}"
        )
        vsd = np.asarray(candidate["discharge_voltage_sd_v"], dtype=float)
        out_lines.append(
            f"V_dis sd: max {np.nanmax(vsd):.1f} V, median {np.nanmedian(vsd):.2f} V"
        )

    out_lines.append("VERDICT: PASS" if ok else "VERDICT: FAIL")
    out_lines.append("")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-dir", required=True, type=Path)
    ap.add_argument("--sets", default="1,2,3,4")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lines = []
    all_ok = True
    for es in [int(s) for s in args.sets.split(",")]:
        path = args.candidate_dir / f"es{es}_sim1d_overlay_spread_candidate.npz"
        all_ok &= check_one(es, path, lines)

    lines.append(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
