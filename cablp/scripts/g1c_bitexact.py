"""Raw-uint64 dataset comparison between two sim1d HDF5 results.

The off-path gate: a build that adds default-off machinery must reproduce an
archived run BIT for BIT. Floating-point equality is not the test -- NaN
compares unequal to itself and -0.0 compares equal to +0.0 -- so every dataset
is viewed as raw ``uint64`` (``uint32`` / ``uint8`` for narrower dtypes) and
compared as integers. Top-level attrs are compared separately, because new
config-template keys legitimately appear there.

Usage:

    python scripts/g1c_bitexact.py NEW.h5 ARCHIVED.h5
"""

import json
import sys

import h5py
import numpy as np


def collect(handle):
    """Return {path: dataset} for every dataset in the file."""
    found = {}

    def visit(name, node):
        if isinstance(node, h5py.Dataset):
            found[name] = node

    handle.visititems(visit)
    return found


def raw_view(array):
    """Return the array's bit pattern as an unsigned-integer view."""
    array = np.asarray(array)
    if array.dtype.kind in "SUO":
        return array
    size = array.dtype.itemsize
    kind = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(size)
    if kind is None:
        return array
    return array.view(kind)


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    a_path, b_path = argv[1], argv[2]
    print(f"COMMAND: python scripts/g1c_bitexact.py {a_path} {b_path}")
    print()
    with h5py.File(a_path, "r") as a, h5py.File(b_path, "r") as b:
        print(f"A (new/control) : {a_path}")
        print(f"B (archived ref): {b_path}")
        da, db = collect(a), collect(b)
        common = sorted(set(da) & set(db))
        only_a = sorted(set(da) - set(db))
        only_b = sorted(set(db) - set(da))
        print(f"datasets: A={len(da)}  B={len(db)}  common={len(common)}")
        print(f"only in A: {len(only_a)}  {only_a}")
        print(f"only in B: {len(only_b)}  {only_b}")
        print()

        print("--- top-level attrs compared ---")
        ka, kb = set(a.attrs), set(b.attrs)
        print(
            f"attrs: A={len(ka)} B={len(kb)} onlyA={sorted(ka - kb)} "
            f"onlyB={sorted(kb - ka)}"
        )
        differing_attrs = []
        for key in sorted(ka & kb):
            if not np.array_equal(np.asarray(a.attrs[key]), np.asarray(b.attrs[key])):
                differing_attrs.append(key)
                print(f"  ATTR DIFFERS  {key}")
        print()

        print("--- RAW uint64 DATASET COMPARISON ---")
        identical = differ = mismatched = 0
        first_diff = None
        for name in common:
            xa, xb = da[name][()], db[name][()]
            xa, xb = np.asarray(xa), np.asarray(xb)
            if xa.shape != xb.shape or xa.dtype != xb.dtype:
                mismatched += 1
                if first_diff is None:
                    first_diff = f"{name} (shape/dtype {xa.shape}/{xa.dtype} vs {xb.shape}/{xb.dtype})"
                continue
            if np.array_equal(raw_view(xa), raw_view(xb)):
                identical += 1
            else:
                differ += 1
                if first_diff is None:
                    first_diff = name
        print(f"datasets bit-identical : {identical}")
        print(f"datasets differing     : {differ}")
        print(f"datasets shape/dtype mismatch : {mismatched}")
        print(f"first differing dataset: {first_diff}")
        print()
        verdict = "PASS (all datasets bit-identical)" if (
            differ == 0 and mismatched == 0 and not only_a and not only_b
        ) else "FAIL"
        print(f"BIT-EXACT VERDICT: {verdict}")

        for key in differing_attrs:
            if not key.endswith("_json"):
                continue
            print()
            print(f"--- {key} attr delta detail ---")
            ja = json.loads(a.attrs[key])
            jb = json.loads(b.attrs[key])
            print(f"only in new h5   : { {k: ja[k] for k in sorted(set(ja) - set(jb))} }")
            print(f"only in archived : { {k: jb[k] for k in sorted(set(jb) - set(ja))} }")
            print(
                "changed in common: "
                f"{ {k: (jb[k], ja[k]) for k in sorted(set(ja) & set(jb)) if ja[k] != jb[k]} }"
            )


if __name__ == "__main__":
    main(sys.argv)
