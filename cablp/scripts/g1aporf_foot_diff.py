"""Three-way comparison of the g1atrim shaped foot fill.

The stance's initial neutral fill was rebuilt when the fluid puff moved to the
tube-beamed ``gas_puff_profile = "orifice"`` row. Two things moved at once, and
this separates them:

* STALE BANKED -- ``scripts/g1afix_foot45.npz``, built 2026-08-19, before the
  CAD-span gap re-anchor moved every cell downstream of the cathode face and
  before ``SCCM_TO_PARTICLES_PER_S`` was re-referenced to the flow meter's own
  20 C / 1013 mbar standard. It sits on a DIFFERENT z grid.
* COSINE CURRENT -- the same recipe rebuilt at the current tip on the current
  geometry, still on the fluid ``"cosine_pipe"`` lobe. This is the
  STALENESS-ONLY control: banked vs this is the re-anchor, nothing else.
* ORIFICE CURRENT -- the adopted fill, same tip and geometry, on the orifice
  row. Cosine-current vs this is the PROFILE change, nothing else.

Reading the two deltas against the wrong partner would attribute the re-anchor
to the profile or vice versa, which is the whole reason the cosine control is
rebuilt rather than read off the banked file.

Usage:
    python scripts/g1aporf_foot_diff.py > scripts/g1aporf_foot_diff.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sp3_build_nn0 as sp3  # noqa: E402

from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402
from cablp.solvers._sim1d.physics.neutrals import (  # noqa: E402
    neutral_zone_volumes,
)

#: ES1 measurement stations [cm] and their LAPD port numbers. Port 11 is the
#: near-cathode station and port 29 the mid-machine one; the mid-port trough is
#: the residual this campaign tracks, so both are reported explicitly.
PORTS = ((11, 470.05), (29, 1045.15))

#: Axial bands for the inventory census [cm]. 103.25 is the far edge of the
#: fixed source region; 500 splits the near column from the far machine. Same
#: bands as scripts/foot_orifice_probe.txt, so the two tables can be read
#: against each other.
BANDS = ((-np.inf, 103.25), (103.25, 500.0), (500.0, np.inf))


def load(path):
    """Return one foot fill: its arrays, its z grid and its build ledger."""
    with np.load(path, allow_pickle=False) as data:
        out = {
            "path": path,
            "column": np.asarray(data["nn0_profile"], dtype=float),
            "annulus": (
                np.asarray(data["nn0_annulus_profile"], dtype=float)
                if "nn0_annulus_profile" in data
                else None
            ),
            "z_cm": np.asarray(data["z_cm"], dtype=float),
            "ledger": json.loads(str(data["provenance"])),
        }
    return out


def zone_volumes(entry):
    """Column and annulus cell volumes for the mesh this fill was built on.

    Rebuilt from the fill's OWN recorded ledger, not from the current stance:
    the banked file sits on the pre-re-anchor mesh, and weighting it with
    today's volumes would fold a second difference into the inventory.
    """
    ledger = entry["ledger"]
    extra = dict(ledger.get("extra_params") or {})
    # Array-valued overrides are recorded by REFERENCE ("file.npz:name"), not
    # by value, so they are resolved from the same files the build read.
    for key, meta in (ledger.get("extra_params_from_npz") or {}).items():
        path, _, array_name = str(meta["source"]).rpartition(":")
        with np.load(path, allow_pickle=False) as data:
            array = np.asarray(data[array_name])
        extra[key] = array.item() if array.ndim == 0 else array.tolist()
    params, flags = sp3.stance_config(
        int(ledger["es"]),
        int(ledger["nx"]),
        float(ledger["S_gp_sccm"]),
        bool(ledger.get("two_zone", False)),
        extra_params=extra,
        extra_flags=ledger.get("extra_flags"),
    )
    geometry = build_geometry(params, flags)
    if geometry.cells != entry["column"].size:
        raise ValueError(
            f"{entry['path']}: rebuilt geometry has {geometry.cells} cells for "
            f"a {entry['column'].size}-cell fill; the ledger does not describe "
            "the mesh this file was written on"
        )
    return neutral_zone_volumes(geometry)


def inventory(entry):
    """Per-cell atom inventory of the fill [atoms], both zones together."""
    V_col, V_ann = zone_volumes(entry)
    total = entry["column"] * np.asarray(V_col, dtype=float)
    if entry["annulus"] is not None:
        total = total + entry["annulus"] * np.asarray(V_ann, dtype=float)
    return total


def at_z(entry, z_target):
    """Nearest cell to a measurement station: (index, its z, column density)."""
    idx = int(np.argmin(np.abs(entry["z_cm"] - z_target)))
    return idx, float(entry["z_cm"][idx]), float(entry["column"][idx])


def rel(a, b):
    return "n/a" if b == 0.0 else f"{a / b - 1.0:+.6e}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banked", default="scripts/g1afix_foot45.npz")
    p.add_argument("--cosine", default="scripts/g1acos_foot45.npz")
    p.add_argument("--orifice", default="scripts/g1aporf_foot45.npz")
    args = p.parse_args(argv)

    entries = [
        ("stale banked  ", load(args.banked)),
        ("cosine current", load(args.cosine)),
        ("orifice curr. ", load(args.orifice)),
    ]

    print("=== g1aporf_foot_diff: the g1atrim shaped foot, three ways ===")
    for label, e in entries:
        led = e["ledger"]
        print(
            f"{label}  {e['path']}\n"
            f"                  cells={e['column'].size} nx={led['nx']} "
            f"profile={led['gas_puff_profile']!r} "
            f"dt_foot={led['dt_foot_s']} s kernel={led['kernel']!r}\n"
            f"                  base={led.get('base_from_h5')} "
            f"injected={led['injected_atoms_as_applied']:.9e} atoms"
        )

    z_ref = entries[1][1]["z_cm"]
    print("\n--- the z grid ---")
    for label, e in entries:
        d = e["z_cm"] - z_ref
        print(
            f"{label}  z[0]={e['z_cm'][0]:10.4f}  z[-1]={e['z_cm'][-1]:10.4f}  "
            f"max |z - cosine_current| = {np.max(np.abs(d)):.4f} cm"
        )
    print(
        "  the banked file's shift IS the CAD-span gap re-anchor; the two "
        "current files share one grid exactly"
    )

    print("\n--- column density at the measurement stations [cm^-3] ---")
    header = f"{'':16s} {'port':>5} {'z_target':>9} {'cell':>5} {'z_cell':>10} {'nn0':>16}"
    print(header)
    values = {}
    for port, z_target in PORTS:
        for label, e in entries:
            idx, z_cell, nn = at_z(e, z_target)
            values[(port, label)] = nn
            print(f"{label:16s} {port:5d} {z_target:9.2f} {idx:5d} "
                  f"{z_cell:10.4f} {nn:16.9e}")
    print("\n  relative differences, each against the partner that isolates one cause:")
    for port, _ in PORTS:
        banked = values[(port, "stale banked  ")]
        cosine = values[(port, "cosine current")]
        orif = values[(port, "orifice curr. ")]
        print(
            f"    port {port:2d}: staleness (banked / cosine-current - 1) "
            f"{rel(banked, cosine)} | profile (orifice / cosine-current - 1) "
            f"{rel(orif, cosine)}"
        )

    print("\n--- peak cell of the column fill ---")
    for label, e in entries:
        i = int(np.argmax(e["column"]))
        print(f"{label}  cell {i:4d}  z={e['z_cm'][i]:10.4f} cm  "
              f"nn0={e['column'][i]:.9e} cm^-3")

    print("\n--- per-cell spread against the cosine-current control ---")
    cosine_col = entries[1][1]["column"]
    for label, e in entries:
        if e is entries[1][1]:
            continue
        if e["column"].size != cosine_col.size:
            print(f"{label}  (different cell count -- not comparable per cell)")
            continue
        r = np.abs(e["column"] / cosine_col - 1.0)
        i = int(np.argmax(r))
        print(f"{label}  max |rel| {r[i]:.6e} at z={e['z_cm'][i]:.4f} cm, "
              f"mean |rel| {float(np.mean(r)):.6e}")
    print(
        "  NB the banked row is compared cell-INDEX to cell-index; its grid is "
        "shifted, so this number carries the z shift as well as the fill change"
    )

    print("\n--- band inventories [atoms], both zones ---")
    print(f"{'':16s} {'band [cm]':>22} {'cells':>6} {'atoms':>16} {'share':>9}")
    inventories = {}
    for label, e in entries:
        inv = inventory(e)
        inventories[label] = inv
        total = float(inv.sum())
        for lo, hi in BANDS:
            mask = (e["z_cm"] >= lo) & (e["z_cm"] < hi)
            band = float(inv[mask].sum())
            span = f"[{lo:>8.2f},{hi:>8.2f})"
            print(f"{label:16s} {span:>22} {int(mask.sum()):6d} "
                  f"{band:16.9e} {100.0 * band / total:8.4f}%")
        print(f"{label:16s} {'TOTAL':>22} {e['column'].size:6d} {total:16.9e}")

    print("\n  band-by-band relative differences:")
    for lo, hi in BANDS:
        row = {}
        for label, e in entries:
            mask = (e["z_cm"] >= lo) & (e["z_cm"] < hi)
            row[label] = float(inventories[label][mask].sum())
        span = f"[{lo:>8.2f},{hi:>8.2f})"
        print(
            f"    {span}  staleness {rel(row['stale banked  '], row['cosine current'])}"
            f" | profile {rel(row['orifice curr. '], row['cosine current'])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
