"""L6 piece-1: boundary-layer front z_front(nn* ) on the COLUMN channel,
nn* sensitivity x{0.5,1,2}, and the last-2-m e-fold length. Plus the
report-only far-nn read at the ES1 ports. Report-only.
"""
import sys

import h5py
import numpy as np

path = sys.argv[1]
lo, hi = float(sys.argv[2]), float(sys.argv[3])

PORTS = {"p11": 470.1, "p21": 789.5, "p29": 1045.2, "p41": 1428.5,
         "p50": 1716.1}
NN_STAR = 1.0e12

with h5py.File(path, "r") as f:
    t0 = float(f.attrs["t_breakdown_trigger"])
    t_ms = (f["time"][:] - t0) * 1e3
    m = (t_ms >= lo) & (t_ms <= hi)
    z = f["geometry/z_cm"][:]
    roles = [r.decode() if isinstance(r, bytes) else str(r)
             for r in f["geometry/cell_role"][:]]
    nn = np.mean(f["nn"][:][m], axis=0)
    nn_a = np.mean(f["nn_a"][:][m], axis=0) if "nn_a" in f else None

print(f"artifact : {path}")
print(f"window   : {lo}-{hi} ms after breakdown ({int(m.sum())} saves)")
print(f"channel  : column nn (time-averaged over the window)")

# Restrict to the plasma column between the cathode and the collector.
i_cath = [i for i, r in enumerate(roles) if r == "cathode"][-1]
i_coll = [i for i, r in enumerate(roles) if r == "collector"][-1]
sl = slice(i_cath, i_coll + 1)
zc, nc = z[sl], nn[sl]


def z_front(target):
    """Largest z at which the column nn last crosses `target` from below,
    scanning from the collector end inward; linear interpolation in z."""
    idx = np.flatnonzero(
        ((nc[:-1] - target) * (nc[1:] - target)) < 0.0
    )
    if idx.size == 0:
        return None, "no crossing in [cathode, collector]"
    j = int(idx[-1])
    a, b = nc[j], nc[j + 1]
    frac = (target - a) / (b - a)
    return float(zc[j] + frac * (zc[j + 1] - zc[j])), f"cells {j}/{j+1}"


print(f"\n=== z_front on the column channel ===")
print(f"  nn range over [cathode, collector]: "
      f"{nc.min():.6e} .. {nc.max():.6e} cm^-3")
print(f"{'nn* [cm^-3]':>16}{'factor':>9}{'z_front [cm]':>16}   note")
for fac in (0.5, 1.0, 2.0):
    tgt = NN_STAR * fac
    zf, note = z_front(tgt)
    zs = f"{zf:.3f}" if zf is not None else "NONE"
    print(f"{tgt:>16.4e}{fac:>9.1f}{zs:>16}   {note}")

# --- last-2-m e-fold length on the column channel (200 cm ending at the
# collector cell), log-linear least squares fit nn ~ exp(z / L).
z_end = zc[-1]
sel = zc >= (z_end - 200.0)
zz, vv = zc[sel], nc[sel]
good = vv > 0.0
print(f"\n=== last-2-m e-fold length (column nn) ===")
print(f"  span z = {zz[0]:.1f} .. {zz[-1]:.1f} cm  ({int(sel.sum())} cells, "
      f"{int(good.sum())} with nn>0)")
if good.sum() >= 2:
    p = np.polyfit(zz[good], np.log(vv[good]), 1)
    L = 1.0 / p[0]
    fit = np.exp(np.polyval(p, zz[good]))
    resid = np.log(vv[good]) - np.log(fit)
    print(f"  d(ln nn)/dz = {p[0]:+.6e} cm^-1")
    print(f"  E-FOLD LENGTH L = {L:+.4f} cm  ({L/100.0:+.4f} m)")
    print(f"  fit residual (ln) rms = {float(np.sqrt((resid**2).mean())):.4e}")
    print(f"  endpoint ratio nn(z_end)/nn(z_end-2m) = "
          f"{float(vv[good][-1]/vv[good][0]):.6f}")
else:
    print("  insufficient positive samples for a fit")

print(f"\n=== FAR-nn REPORT-ONLY: column nn and annulus nn_a at the ports ===")
print(f"{'port':>6}{'z_cm':>10}{'cell':>6}{'nn [cm^-3]':>18}"
      f"{'nn_a [cm^-3]':>18}{'nn_a/nn':>12}")
for name, zp in PORTS.items():
    i = int(np.argmin(np.abs(z - zp)))
    a = nn_a[i] if nn_a is not None else float("nan")
    print(f"{name:>6}{zp:>10.1f}{i:>6}{nn[i]:>18.6e}{a:>18.6e}"
          f"{(a/nn[i] if nn[i] else float('nan')):>12.4f}")
