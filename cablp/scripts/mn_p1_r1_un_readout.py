"""P1 R1 readout: arm plateau-mean u_n(z) at the five Mach ports vs the M7
demanded field.

Run artifact (untracked), MECHANICAL READOUT ONLY -- no fitting, no tuning.
Conventions mirror scripts/invert_un_m7.py exactly:
  * plateau window 15.0-19.5 ms on the MAIN-DISCHARGE clock
    (origin = first SAVED frame whose phase == 'main_discharge');
  * port -> cell by argmin |z_model - z_port|, z_port from the scorer's
    own overlay file;
  * time-mean of the saved u_n field over the window.

Usage:
    python scripts/mn_p1_r1_un_readout.py <arm.h5> [--png OUT.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
OVERLAY = HERE / "data" / "es1_sim1d_overlay.npz"

PLATEAU_MS = (15.0, 19.5)
INTERIOR_PORTS = (11, 21, 29, 41)
PORTS = (11, 21, 29, 41, 50)

# M7 demanded field, read from scripts/invert_un_m7_results.txt
# (the MINUS-sign implementation, which is the one that script emits).
# port -> (u_n [km/s], +hi, -lo)
M7_DEMANDED = {
    11: (0.894, 1.093, 2.220),
    21: (0.919, 1.084, 1.758),
    29: (1.679, 1.282, 2.073),
    41: (3.910, 1.319, 2.114),
    50: (7.887, 1.794, 3.350),  # EXCLUDED from kill tests (M7 precedent)
}


def _decode(arr):
    return np.array(
        [s.decode() if isinstance(s, bytes) else str(s) for s in np.asarray(arr)]
    )


def port_z_map(overlay_path):
    ov = np.load(overlay_path, allow_pickle=False)
    ports = np.asarray(ov["port"], dtype=int)
    z_cm = np.asarray(ov["z_cm"], dtype=float)
    return {int(p): float(z_cm[i]) for i, p in enumerate(ports)}


def window_mean(h5_path, t_lo_ms, t_hi_ms):
    with h5py.File(h5_path, "r") as f:
        if "u_n" not in f:
            raise SystemExit(
                "ERROR: no 'u_n' dataset in the artifact -- the arm did not "
                "evolve M_n (neutral_momentum off?).  STOP."
            )
        t = f["time"][:]
        phase = _decode(f["phase"][:])
        hits = np.flatnonzero(phase == "main_discharge")
        if hits.size == 0:
            raise SystemExit("no saved frame with phase == 'main_discharge'")
        origin = float(t[hits[0]])
        t_ms = (t - origin) * 1.0e3
        w = (t_ms >= t_lo_ms) & (t_ms <= t_hi_ms)
        if not np.any(w):
            raise SystemExit("plateau window empty")
        out = {
            "origin_s": origin,
            "n_frames": int(w.sum()),
            "t_first_ms": float(t_ms[w][0]),
            "t_last_ms": float(t_ms[w][-1]),
            "z_cm": f["geometry/z_cm"][:],
            "u_n": f["u_n"][:][w].mean(axis=0),
            "u": f["u"][:][w].mean(axis=0),
            "M_n": f["M_n"][:][w].mean(axis=0) if "M_n" in f else None,
            "nn": f["nn"][:][w].mean(axis=0),
            "n_total_frames": int(t.size),
            "has_M_n_a": "M_n_a" in f,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--png", default=None)
    args = ap.parse_args()

    zmap = port_z_map(OVERLAY)
    fld = window_mean(args.h5, *PLATEAU_MS)

    print("=" * 78)
    print("P1 R1 -- arm plateau u_n(z) vs the M7 demanded field")
    print("MECHANICAL READOUT ONLY.  No tuning, no fitting.")
    print("=" * 78)
    print(f"arm h5   : {args.h5}")
    print(f"overlay  : {OVERLAY}")
    print(f"two-zone radial field present (M_n_a): {fld['has_M_n_a']}")
    print("\n--- plateau window (main-discharge clock) ---")
    print(f"  main_discharge origin (first saved frame) = {fld['origin_s']:.9f} s")
    print(f"  window {PLATEAU_MS[0]}-{PLATEAU_MS[1]} ms -> {fld['n_frames']} frames "
          f"[{fld['t_first_ms']:.3f}, {fld['t_last_ms']:.3f}] ms "
          f"(of {fld['n_total_frames']} saved frames)")

    z_model = fld["z_cm"]
    rows = []
    for p in PORTS:
        z_port = zmap[p]
        iz = int(np.argmin(np.abs(z_model - z_port)))
        un_kms = float(fld["u_n"][iz]) / 1.0e5
        ui_kms = float(fld["u"][iz]) / 1.0e5
        dem, hi, lo = M7_DEMANDED[p]
        in_lo, in_hi = dem - lo, dem + hi
        inside = bool(in_lo <= un_kms <= in_hi)
        rows.append({
            "port": p, "z_port": z_port, "iz": iz, "z_cell": float(z_model[iz]),
            "un": un_kms, "ui": ui_kms, "dem": dem, "hi": hi, "lo": lo,
            "in_lo": in_lo, "in_hi": in_hi, "inside": inside,
            "nn": float(fld["nn"][iz]),
        })

    print("\n--- R1 per-port table (velocities in km/s, +z = toward collector) ---")
    print(f"{'port':>5} {'z_cm':>8} {'cell':>5} {'z_cell':>8} {'arm u_n':>9} "
          f"{'demanded':>9} {'+hi':>6} {'-lo':>6} {'bracket':>17} {'in?':>5} "
          f"{'arm u_i':>9}")
    print("-" * 100)
    for r in rows:
        tag = "IN" if r["inside"] else "OUT"
        if r["port"] == 50:
            tag += "*"
        print(f"{r['port']:>5} {r['z_port']:>8.2f} {r['iz']:>5} {r['z_cell']:>8.2f} "
              f"{r['un']:>9.3f} {r['dem']:>9.3f} {r['hi']:>6.3f} {r['lo']:>6.3f} "
              f"[{r['in_lo']:>7.3f},{r['in_hi']:>7.3f}] {tag:>5} {r['ui']:>9.3f}")
    print("  * p50 is reported but EXCLUDED from kill tests (M7 precedent).")

    interior = [r for r in rows if r["port"] in INTERIOR_PORTS]
    n_in = sum(1 for r in interior if r["inside"])
    print(f"\n  interior ports (p11-p41) inside bracket: {n_in} / {len(interior)}")

    print("\n--- SHAPE fact (flat vs downstream-rising) ---")
    un_i = np.array([r["un"] for r in interior])
    dem_i = np.array([r["dem"] for r in interior])
    print("  arm u_n at p11/p21/p29/p41 [km/s]:      "
          + " / ".join(f"{v:.3f}" for v in un_i))
    print("  demanded  at p11/p21/p29/p41 [km/s]:    "
          + " / ".join(f"{v:.3f}" for v in dem_i))
    span_arm = float(un_i[-1] - un_i[0])
    span_dem = float(dem_i[-1] - dem_i[0])
    ratio_arm = float(un_i[-1] / un_i[0]) if un_i[0] != 0 else float("nan")
    ratio_dem = float(dem_i[-1] / dem_i[0]) if dem_i[0] != 0 else float("nan")
    print(f"  arm      p41-p11 span = {span_arm:+.3f} km/s   p41/p11 = {ratio_arm:.3f}")
    print(f"  demanded p41-p11 span = {span_dem:+.3f} km/s   p41/p11 = {ratio_dem:.3f}")
    monotone = bool(np.all(np.diff(un_i) > 0))
    print(f"  arm u_n monotonically rising across p11->p41: {monotone}")
    print("  per-step arm deltas [km/s]: "
          + " ".join(f"p{a['port']}->p{b['port']} {b['un'] - a['un']:+.3f}"
                     for a, b in zip(interior[:-1], interior[1:])))
    print("  per-step demanded deltas:   "
          + " ".join(f"p{a['port']}->p{b['port']} {b['dem'] - a['dem']:+.3f}"
                     for a, b in zip(interior[:-1], interior[1:])))
    print("  (SHAPE verdict is the orchestrator's read; numbers only here.)")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        z_all = np.array([r["z_cell"] for r in rows])
        un_all = np.array([r["un"] for r in rows])
        dem_all = np.array([r["dem"] for r in rows])
        hi_all = np.array([r["hi"] for r in rows])
        lo_all = np.array([r["lo"] for r in rows])

        ax.plot(z_model, fld["u_n"] / 1e5, color="0.75", lw=1.0, zorder=1,
                label="arm $u_n(z)$, full grid (plateau mean)")
        ax.errorbar(z_all, dem_all, yerr=[lo_all, hi_all], fmt="s",
                    color="tab:red", capsize=4, ms=7, lw=1.6, zorder=3,
                    label="M7 demanded $u_n$ (bracket)")
        ax.plot(z_all, un_all, "o-", color="tab:blue", ms=7, lw=1.8, zorder=4,
                label="arm $u_n$ at Mach ports")
        for r in rows:
            ax.annotate(f"p{r['port']}", (r["z_cell"], r["un"]),
                        textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=8, color="tab:blue")
        ax.axvline(1716.1, color="0.6", ls=":", lw=1.0)
        ax.annotate("p50 excluded\nfrom kill tests", (1716.1, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(-6, -28), ha="right",
                    fontsize=8, color="0.4")
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlabel("z [cm]  (+z toward the collector)")
        ax.set_ylabel(r"$u_n$ [km/s]")
        ax.set_title("P1 single-zone arm: plateau neutral wind vs M7 demanded field\n"
                     f"{Path(args.h5).name}, 15.0-19.5 ms main-discharge plateau",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.png, dpi=150)
        print(f"\nfigure written: {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
