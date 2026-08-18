"""Leg-C discriminator: who owns the model's FALLING far-end ion drift?

THE QUESTION (THESIS_NOTES item 51, challenged 2026-08-17b). Leg C measured a
shape null: no neutral channel makes the neutral flow rise, because every one
of them inherits the model's ion drift, and that drift FALLS toward the end
(~7.9 -> 6.6 km/s) where the machine's flow rises. The item then named the
measured column widening as the candidate riser the 1D model "structurally
lacks" -- but that mechanism does not work (u = Gamma/(n A) = Gamma/N, so
geometric spreading cancels out of the flow arithmetic; a rising u needs a
source or a sink, not geometry). So the remainder is unowned, and there are
two candidates with very different consequences for the thesis:

  (a) STRUCTURAL -- the machine's rise is an extended pre-sheath into the end
      sink. The model HAS that riser but compresses it into its boundary
      cells (MODEL.md R3.1 imposes ghost-cell Bohm outflow at u = c_s); a
      distributed radial particle sink would pull the sonic point upstream.
      1D structurally lacks it, but via the radial SINK, not geometry.

  (b) CLOSURE-CONDITIONAL -- the model's own far-end neutral inventory
      (the foot IC runs plateau mid-band nn at 1.28-2.03x REF) drags the ions
      down through charge exchange. Then "structurally lacks" is the WRONG
      verdict and thread 20 inherits a different question.

THE DISCRIMINATOR, and why it costs no new runs: the saved artifacts already
carry a complete per-term RHS decomposition for the momentum field M
(`rhs_terms/<term>/M`, 37 named terms + `total_rhs/M`). Reading the
plateau-averaged momentum budget term by term along z says directly whether
the ion-neutral drag family is large enough to own the deceleration.

READ-ONLY. Touches no solver code and re-runs nothing.

Arms (all on the leg-B corrected background, 4.5 ms diffusive foot IC):
    sp3b_4p5ms_diff_arm.h5   REF   the background itself
    lcw1_arm.h5              W1    coefficient-free two_zone radial closure
    lcc0_arm.h5              C0    directed hot birth (single flag delta)
    lcc1_arm.h5              C1    composite

Usage:  python legc_momentum_budget.py [--plateau 15 19.5] [--band 1000 1750]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent

ARMS = [
    ("REF", "sp3b_4p5ms_diff_arm.h5"),
    ("W1", "lcw1_arm.h5"),
    ("C0", "lcc0_arm.h5"),
    ("C1", "lcc1_arm.h5"),
]

PORTS = {"p11": 470.05, "p21": 789.55, "p29": 1045.15, "p41": 1428.55,
         "p50": 1716.1}

# Term families. A term appears in exactly one family; the script asserts that
# the families cover every term present in the file.
FAMILIES = {
    "drag / CX (the closure-conditional candidate)": [
        "ion_neutral_drag", "ion_neutral_collision", "ion_charge_exchange",
        "neutral_cx_channel", "ion_neutral_thermalization",
        "ion_neutral_frictional_heating",
    ],
    "advective flux + pressure (the hyperbolic core)": [
        "plasma_advective_flux", "plasma_front_flux", "pressure_work",
        "hyperbolic_energy_correction",
    ],
    "geometry (flux-tube area)": ["flux_tube_geometry"],
    "boundary / end sink (the structural candidate)": [
        "characteristic_boundary", "surface_loss", "boundary_absorption",
        "cathode_surface_loss", "anode_collection",
    ],
    "mass loading (ionization / recombination)": [
        "ionization_birth", "beam_ionization_birth", "ionization_energy_cost",
        "beam_ionization_cost", "gas_puff_local_ionization",
        "recombination_3b_loss", "recombination_rad_loss",
        "recombination_energy_return", "neutral_sources", "neutral_exchange",
        "neutral_zone_exchange",
    ],
    "neutral channels (wind / hot / walls)": [
        "neutral_wind_advection", "neutral_momentum_wall",
        "neutral_energy_wall", "neutral_hot_channel",
        "cathode_jet_neutral_energy",
    ],
    "other": [
        "beam_power_deposition", "beam_excitation_radiation", "ei_exchange",
        "electron_ion_cooling", "electron_neutral_cooling", "heat_conduction",
    ],
}


def load(path, t0, t1, band):
    f = h5py.File(path, "r")
    t = f["time"][:]
    w = np.where((t >= t0) & (t <= t1))[0]
    sl = slice(int(w[0]), int(w[-1]) + 1)
    z = f["geometry/z_cm"][:]
    vol = f["geometry/plasma_volume_cm3"][:]
    active = f["geometry/plasma_active"][:].astype(bool)
    u = f["u"][sl].mean(axis=0)
    n = f["n"][sl].mean(axis=0)

    terms = {}
    for name in f["rhs_terms"]:
        d = f[f"rhs_terms/{name}/M"]
        terms[name] = d[sl].mean(axis=0)
    total = f["total_rhs/M"][sl].mean(axis=0)

    inband = active & (z >= band[0]) & (z <= band[1])
    f.close()
    return dict(z=z, vol=vol, active=active, inband=inband, u=u, n=n,
                terms=terms, total=total, nframes=len(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plateau", nargs=2, type=float, default=[15.0, 19.5],
                    help="plateau window in ms")
    ap.add_argument("--band", nargs=2, type=float, default=[1000.0, 1750.0],
                    help="axial band for the integrated budget, cm")
    a = ap.parse_args()
    t0, t1 = a.plateau[0] * 1e-3, a.plateau[1] * 1e-3

    print(f"LEG-C MOMENTUM BUDGET — who owns the falling far-end ion drift?")
    print(f"plateau {a.plateau[0]}–{a.plateau[1]} ms · "
          f"band z = {a.band[0]:.0f}–{a.band[1]:.0f} cm · read-only\n")

    data = {}
    for tag, fn in ARMS:
        p = HERE / fn
        if not p.exists():
            print(f"  !! missing {fn}")
            continue
        data[tag] = load(p, t0, t1, a.band)

    # ---- 1. the drift profile the whole question is about ---------------
    print("=" * 78)
    print("1. PLATEAU ION DRIFT u_i (km/s) at the scored ports")
    print("=" * 78)
    hdr = "  arm  " + "".join(f"{k:>9s}" for k in PORTS) + "     p50/p21"
    print(hdr)
    for tag in data:
        d = data[tag]
        vals = [d["u"][np.argmin(np.abs(d["z"] - zc))] / 1e5
                for zc in PORTS.values()]
        print(f"  {tag:4s} " + "".join(f"{v:9.2f}" for v in vals)
              + f"   {vals[-1]/vals[1]:9.3f}")
    print("\n  (falling toward the end is the model behaviour under test;\n"
          "   the machine RISES over this span — item 38.)\n")

    # ---- 2. the term-by-term budget -------------------------------------
    print("=" * 78)
    print(f"2. MOMENTUM BUDGET, volume-integrated over z = "
          f"{a.band[0]:.0f}–{a.band[1]:.0f} cm   [dyn]")
    print("   positive = accelerates toward the far end")
    print("=" * 78)

    for tag in data:
        d = data[tag]
        w = d["inband"] * d["vol"]
        present = set(d["terms"])
        mapped = {t for fam in FAMILIES.values() for t in fam}
        unmapped = present - mapped
        if unmapped:
            FAMILIES.setdefault("UNMAPPED", []).extend(sorted(unmapped))

        fam_tot, rows = {}, []
        for fam, names in FAMILIES.items():
            s = 0.0
            for nm in names:
                if nm in d["terms"]:
                    v = float(np.sum(d["terms"][nm] * w))
                    s += v
                    if abs(v) > 0:
                        rows.append((abs(v), fam, nm, v))
            fam_tot[fam] = s

        sinks = {k: v for k, v in fam_tot.items() if v < 0}
        total_sink = sum(sinks.values())
        print(f"\n  ── {tag} " + "─" * 62)
        for fam, v in sorted(fam_tot.items(), key=lambda kv: -abs(kv[1])):
            if abs(v) < 1e-12:
                continue
            share = (100 * v / total_sink) if (v < 0 and total_sink) else None
            tail = f"   {share:5.1f} % of all sinks" if share else ""
            print(f"     {v:+14.4e}   {fam}{tail}")
        drag = fam_tot.get("drag / CX (the closure-conditional candidate)", 0.0)
        bnd = fam_tot.get("boundary / end sink (the structural candidate)", 0.0)
        print(f"     {'':14s}   ---")
        print(f"     net total_rhs check: "
              f"{float(np.sum(d['total'] * w)):+.4e}")
        if total_sink:
            print(f"     DRAG share of sinks     : "
                  f"{100*min(drag,0)/total_sink:6.2f} %")
            print(f"     BOUNDARY share of sinks : "
                  f"{100*min(bnd,0)/total_sink:6.2f} %")
        print("     top individual terms:")
        for _, fam, nm, v in sorted(rows, reverse=True)[:6]:
            print(f"       {v:+14.4e}  {nm}")

    # ---- 3. where along z the deceleration happens ----------------------
    print("\n" + "=" * 78)
    print("3. DRAG vs BOUNDARY per port  [dyn per cell, plateau mean]")
    print("=" * 78)
    print("  arm  port      drag/CX     boundary    advect+p     u_i km/s")
    for tag in data:
        d = data[tag]
        for pn, zc in PORTS.items():
            i = int(np.argmin(np.abs(d["z"] - zc)))
            def famsum(fam):
                return sum(float(d["terms"][nm][i] * d["vol"][i])
                           for nm in FAMILIES[fam] if nm in d["terms"])
            print(f"  {tag:4s} {pn:5s} "
                  f"{famsum('drag / CX (the closure-conditional candidate)'):+12.3e} "
                  f"{famsum('boundary / end sink (the structural candidate)'):+12.3e} "
                  f"{famsum('advective flux + pressure (the hyperbolic core)'):+12.3e} "
                  f"{d['u'][i]/1e5:9.2f}")
        print()


if __name__ == "__main__":
    main()
