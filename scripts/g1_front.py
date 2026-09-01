"""G1 front/e-fold reader, the registered variant of l6a1_front.py.

Implements the "G1 BINS OF RECORD" spec registered 2026-08-18. The bins, as
this script computes them:

- z_front: threshold nn* = 2e12 cm^-3 on column nn, LAST crossing restricted
  to cell centers z <= 1900 cm (the C1 cap -- the bare last-crossing rule
  could wander onto non-shared far-chamber cells). Crossings beyond 1900 are
  listed separately as the report-only far pass. The legacy 1e12 threshold is
  retired (no crossing on the control); the uncapped result is printed for
  transparency but is NOT the registered read.
- last-2-m e-fold: window on CELL CENTERS with z_end = the terminal cell
  center; window selection FIRST, then exclusions dropped from the fit with
  no re-anchoring of z_end. Three quotes, each with ln-rms: (a) full window;
  (b) excluding the terminal cell (the A/B comparator); (c) additionally
  excluding the throat set -- the cells whose registered plasma-radius
  profile sits BELOW the flat column radius, matched by z against
  g1_profiles.npz (z-keyed per C4; empty for the 'off' case and the control
  by construction).
- The e-fold runs on TWO channels: column nn (the l6a1 instrument's channel)
  and plasma n (the parallel quote the session-23 riders require -- the
  throat/flux-conservation rationale lives in plasma n). z_front stays
  nn-only, as registered.

Usage:
    python scripts/g1_front.py <h5> <lo_ms> <hi_ms> --case droop_min
    python scripts/g1_front.py <h5> <lo_ms> <hi_ms> --case off
    python scripts/g1_front.py <h5> <lo_ms> <hi_ms> --case none   # control
"""
import argparse
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

PORTS = {"p11": 470.1, "p21": 789.5, "p29": 1045.2, "p41": 1428.5,
         "p50": 1716.1}
NN_STAR_REGISTERED = 2.0e12
Z_FRONT_CAP_CM = 1900.0
RP_FLAT_CM = 18.415

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("h5")
parser.add_argument("lo_ms", type=float)
parser.add_argument("hi_ms", type=float)
parser.add_argument(
    "--case",
    choices=("droop_min", "off", "none"),
    required=True,
    help="which registered plasma profile defines the throat set; 'none' for "
    "the control (empty set)",
)
parser.add_argument(
    "--profiles-npz", default=os.path.join(HERE, "g1_profiles.npz")
)
args = parser.parse_args()

with h5py.File(args.h5, "r") as f:
    t0 = float(f.attrs["t_breakdown_trigger"])
    t_ms = (f["time"][:] - t0) * 1e3
    m = (t_ms >= args.lo_ms) & (t_ms <= args.hi_ms)
    z = f["geometry/z_cm"][:]
    roles = [r.decode() if isinstance(r, bytes) else str(r)
             for r in f["geometry/cell_role"][:]]
    nn = np.mean(f["nn"][:][m], axis=0)
    n_pl = np.mean(f["n"][:][m], axis=0)
    nn_a = np.mean(f["nn_a"][:][m], axis=0) if "nn_a" in f else None

print(f"artifact : {args.h5}")
print(f"window   : {args.lo_ms}-{args.hi_ms} ms after breakdown "
      f"({int(m.sum())} saves)")
print(f"case     : {args.case}")

# The throat set, z-keyed against the registered profile (C4: never by index).
if args.case == "none":
    throat_z = np.empty(0)
else:
    prof = np.load(args.profiles_npz)
    r_prof = prof[f"plasma_radius_profile_cm_{args.case}"]
    throat_z = np.asarray(prof["z_cm"])[r_prof < RP_FLAT_CM]
print(f"throat set (profile r < {RP_FLAT_CM}): "
      + (f"{throat_z.size} cells, z {throat_z.min():.2f}..{throat_z.max():.2f}"
         if throat_z.size else "empty"))

i_cath = [i for i, r in enumerate(roles) if r == "cathode"][-1]
i_coll = [i for i, r in enumerate(roles) if r == "collector"][-1]
sl = slice(i_cath, i_coll + 1)
zc = z[sl]


def crossings(values, target):
    """All crossing positions of `target`, linear interpolation in z."""
    idx = np.flatnonzero(((values[:-1] - target) * (values[1:] - target)) < 0.0)
    out = []
    for j in idx:
        a, b = values[j], values[j + 1]
        frac = (target - a) / (b - a)
        out.append((float(zc[j] + frac * (zc[j + 1] - zc[j])), int(j)))
    return out


nc = nn[sl]
print(f"\n=== z_front, REGISTERED read (column nn, nn* = "
      f"{NN_STAR_REGISTERED:.1e}, last crossing at cell centers z <= "
      f"{Z_FRONT_CAP_CM:.0f} cm) ===")
print(f"  nn range over [cathode, collector]: {nc.min():.6e} .. "
      f"{nc.max():.6e} cm^-3")
all_cross = crossings(nc, NN_STAR_REGISTERED)
capped = [c for c in all_cross
          if zc[c[1]] <= Z_FRONT_CAP_CM and zc[c[1] + 1] <= Z_FRONT_CAP_CM]
beyond = [c for c in all_cross if c not in capped]
if capped:
    zf, j = capped[-1]
    print(f"  z_front = {zf:.3f} cm  (this-mesh cells {j}/{j + 1}, z centers "
          f"{zc[j]:.2f}/{zc[j + 1]:.2f})")
else:
    print("  z_front = NONE (no crossing at z <= 1900)")
print(f"  crossings beyond {Z_FRONT_CAP_CM:.0f} cm (report-only far pass): "
      + (", ".join(f"{c[0]:.3f}" for c in beyond) if beyond else "none"))
uncapped = all_cross[-1][0] if all_cross else None
print(f"  [transparency] uncapped last crossing: "
      + (f"{uncapped:.3f} cm" if uncapped is not None else "NONE"))


def efold(channel_name, values):
    vals = values[sl]
    z_end = zc[-1]
    window = zc >= (z_end - 200.0)
    zz, vv = zc[window], vals[window]
    print(f"\n=== last-2-m e-fold ({channel_name}) ===")
    print(f"  window: centers {zz[0]:.2f} .. {zz[-1]:.2f} cm "
          f"({int(window.sum())} cells; z_end anchor {z_end:.2f}, fixed)")
    quotes = {
        "(a) full window": np.ones(zz.size, dtype=bool),
        "(b) excl. terminal cell": zz < zz[-1],
    }
    in_throat = np.isin(zz, throat_z)
    quotes["(c) excl. terminal + throat set"] = (zz < zz[-1]) & ~in_throat
    print(f"  throat cells inside window: {int(in_throat.sum())}")
    for label, keep in quotes.items():
        good = keep & (vv > 0.0)
        if good.sum() < 2:
            print(f"  {label}: insufficient positive samples")
            continue
        p = np.polyfit(zz[good], np.log(vv[good]), 1)
        L = 1.0 / p[0]
        resid = np.log(vv[good]) - np.polyval(p, zz[good])
        rms = float(np.sqrt((resid ** 2).mean()))
        print(f"  {label}: L = {L:+.4f} cm ({L / 100.0:+.4f} m), "
              f"d(ln)/dz = {p[0]:+.6e} cm^-1, ln-rms = {rms:.4e}, "
              f"cells = {int(good.sum())}")


efold("column nn", nn)
efold("plasma n", n_pl)

print(f"\n=== FAR-nn REPORT-ONLY: column nn and annulus nn_a at the ports ===")
print(f"{'port':>6}{'z_cm':>10}{'cell':>6}{'nn [cm^-3]':>18}"
      f"{'nn_a [cm^-3]':>18}{'nn_a/nn':>12}")
for name, zp in PORTS.items():
    i = int(np.argmin(np.abs(z - zp)))
    a = nn_a[i] if nn_a is not None else float("nan")
    print(f"{name:>6}{zp:>10.1f}{i:>6}{nn[i]:>18.6e}{a:>18.6e}"
          f"{(a / nn[i] if nn[i] else float('nan')):>12.4f}")
