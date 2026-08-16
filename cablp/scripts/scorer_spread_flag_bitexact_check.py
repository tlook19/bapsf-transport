"""Raw-uint64 bit-exactness of the scorer's stage-(ii) numbers.

The printed table is decimal-rounded, so identical text is weaker than
identical arithmetic. This loads the PRE-change scorer (HEAD~1) and the
post-change one side by side, runs compare() with each over the same result
and the same ES1-4 overlays, and compares every float the rows carry at raw
uint64.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

WT = Path("/Users/tlook/bapsf/bapsf-transport/.claude/worktrees/"
          "agent-ab06290979dde7b57/cablp")
SCRATCH = Path("/private/tmp/claude-501/-Users-tlook-bapsf-bapsf-transport/"
               "0db6b570-25fe-4ec9-8f34-fd79511820b8/scratchpad")
sys.path.insert(0, str(WT / "scripts"))

from cablp.solvers._sim1d import load_result_hdf5  # noqa: E402
import compare_sim1d_es1 as NEW  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "compare_sim1d_es1_base", SCRATCH / "compare_base.py"
)
OLD = importlib.util.module_from_spec(spec)
sys.modules["compare_sim1d_es1_base"] = OLD
spec.loader.exec_module(OLD)
print(f"OLD module: {OLD.__file__}")
print(f"NEW module: {NEW.__file__}")
assert not hasattr(OLD, "TE_SPREAD_SEMIQUANT_FRAC"), "base already has the flag"
assert hasattr(NEW, "TE_SPREAD_SEMIQUANT_FRAC")

H5 = ("/Users/tlook/bapsf/bapsf-transport/cablp/scripts/"
      "es1_prod_circuit_nx240.h5")
print("loading result h5 ...", flush=True)
result = load_result_hdf5(H5)
print("loaded.", flush=True)

FLOAT_KEYS = ("z", "model", "exp", "ratio", "rms_rel", "sigma")


def bits(x):
    return int(np.float64(x).view(np.uint64))


ok = True
for es in (1, 2, 3, 4):
    overlay = np.load(WT / "scripts" / "data" / f"es{es}_sim1d_overlay.npz",
                      allow_pickle=False)
    old_rows = OLD.compare(result, None, overlay)
    new_rows = NEW.compare(result, None, overlay)

    n_ok = len(old_rows) == len(new_rows)
    mismatches = []
    for a, b in zip(old_rows, new_rows):
        if (a["field"], a["port"]) != (b["field"], b["port"]):
            mismatches.append(f"row identity {a['field']}/{a['port']}")
            continue
        for k in FLOAT_KEYS:
            if bits(a[k]) != bits(b[k]):
                mismatches.append(
                    f"{a['field']} p{a['port']} {k}: "
                    f"0x{bits(a[k]):016x} -> 0x{bits(b[k]):016x}"
                )
    same = n_ok and not mismatches
    ok = ok and same
    print(f"ES{es}: rows {len(old_rows)}->{len(new_rows)}, "
          f"{len(old_rows) * len(FLOAT_KEYS)} floats compared at raw uint64, "
          f"bit-identical={same}")
    for m in mismatches[:5]:
        print("   MISMATCH:", m)

    # And the flag change, reported explicitly.
    for a, b in zip(old_rows, new_rows):
        marks = ""
        if b.get("semiquant_spread"):
            marks += "~"
        if b.get("semiquant_te"):
            marks += "*"
        if b.get("spread_undetermined"):
            marks += "?"
        was = "~" if a["semiquant"] else ""
        if was != marks:
            print(f"   flag {a['field']:>4} p{a['port']:<3} "
                  f"{was or '(none)':>7} -> {marks:<3}")

print("\nRAW-UINT64 BIT-EXACTNESS:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
