"""fab -- THE CHOKE READ: does the p27 annular baffle hold gas upstream?

*** SURROGATE DISCLOSURE (FIRST) ***
The baffle is a deliberately CRUDE, AZIMUTHALLY-SYMMETRIC SURROGATE for an
ASYMMETRIC antenna array around LAPD port 27 (a couple of ports wide). A real
array blocks part of the AZIMUTH; this model replaces it with a full annular
iris, which cannot represent azimuthal bypass -- in reality gas flows AROUND
the obstruction. The surrogate BOUNDS the axial-choke effect from above; it is
not a model of the array. Every number below inherits that caveat.

Read-only. Zone conventions (SOURCE/BAND/MID/FAR) are fa2_diag's, reused
verbatim so the numbers compose with the rest of the campaign; the CHOKE bins
are this probe's own and are stated where used.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent

# fa2_diag zone conventions, verbatim (scripts/fa2_diag.py:38-47).
SOURCE_Z = (-1e9, 100.0)
BAND_Z = (100.0, 790.0)
MID_Z = (790.0, 1045.0)
FAR_Z = (1045.0, 1900.0)
END_Z = (1900.0, 1e9)
ZONES = (
    ("SOURCE z<=100", SOURCE_Z),
    ("BAND 100-790", BAND_Z),
    ("MID 790-1045", MID_Z),
    ("FAR 1045-1900", FAR_Z),
    ("END z>1900", END_Z),
)

# This probe's own CHOKE bins, straddling the realized baffle span
# (faces 124/129/135 at z = 940.0 / 977.5 / 1022.5 cm).
BAF_LO, BAF_HI = 940.0, 1022.5
CHOKE = (
    ("UPSTREAM 790-940", (790.0, BAF_LO)),
    ("INSIDE 940-1022.5", (BAF_LO, BAF_HI)),
    ("DOWNSTREAM 1022.5-1150", (BAF_HI, 1150.0)),
)
PORTS = {11: 470.05, 21: 789.55, 27: 981.25, 29: 1045.15, 41: 1428.55, 50: 1716.1}


class Run:
    def __init__(self, path, tag):
        self.tag = tag
        self.h5 = h5py.File(path, "r")
        g = self.h5["geometry"]
        self.z = g["z_cm"][:]
        self.V_col = g["plasma_volume_cm3"][:]
        self.V_ann = g["neutral_volume_cm3"][:] - self.V_col
        self.t_ms = self.h5["time"][:] * 1e3
        self.params = json.loads(self.h5.attrs["params_json"])
        self.flags = json.loads(self.h5.attrs["flags_json"])

    def zmask(self, lo, hi):
        return (self.z >= lo) & (self.z <= hi)

    def isnap(self, t_ms):
        return int(np.argmin(np.abs(self.t_ms - t_ms)))

    def inv_col(self, m, j):
        return float(self.h5["nn"][j, m] @ self.V_col[m])

    def inv_ann(self, m, j):
        return float(self.h5["nn_a"][j, m] @ self.V_ann[m])

    def close(self):
        self.h5.close()


def banner(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def zone_table(arm, ctl, t_ms, zones):
    ja, jc = arm.isnap(t_ms), ctl.isnap(t_ms)
    print(
        f"\n  t = {arm.t_ms[ja]:.2f} ms (arm) / {ctl.t_ms[jc]:.2f} ms (fa4j)"
    )
    print(
        f"    {'zone':<24} {'COLUMN arm':>12} {'COLUMN fa4j':>12} {'x':>7}"
        f" {'ANNULUS arm':>12} {'ANNULUS fa4j':>12} {'x':>7}"
    )
    tot = {}
    for name, (lo, hi) in zones:
        ma, mc = arm.zmask(lo, hi), ctl.zmask(lo, hi)
        ca, cc = arm.inv_col(ma, ja), ctl.inv_col(mc, jc)
        aa, ac = arm.inv_ann(ma, ja), ctl.inv_ann(mc, jc)
        tot[name] = (ca, cc, aa, ac)
        print(
            f"    {name:<24} {ca:>12.4e} {cc:>12.4e} "
            f"{ca / cc if cc else float('nan'):>7.3f}"
            f" {aa:>12.4e} {ac:>12.4e} "
            f"{aa / ac if ac else float('nan'):>7.3f}"
        )
    ca = sum(v[0] for v in tot.values())
    cc = sum(v[1] for v in tot.values())
    aa = sum(v[2] for v in tot.values())
    ac = sum(v[3] for v in tot.values())
    print(
        f"    {'TOTAL':<24} {ca:>12.4e} {cc:>12.4e} {ca / cc:>7.3f}"
        f" {aa:>12.4e} {ac:>12.4e} {aa / ac:>7.3f}"
    )
    print(
        f"    whole-domain neutral inventory: arm {ca + aa:.4e}  "
        f"fa4j {cc + ac:.4e}  x {(ca + aa) / (cc + ac):.4f}"
    )


def main():
    arm = Run(_SCRIPTS / "fab_arm.h5", "fab")
    ctl = Run(_SCRIPTS / "fa4j_arm.h5", "fa4j")

    banner(
        "A. THE BAFFLE AS BUILT (from the arm's OWN stored config)\n"
        "   Azimuthally-symmetric surrogate for the port-27 antenna array."
    )
    print(f"  neutral_baffles flag       : {arm.flags.get('neutral_baffles')}")
    print(f"  positions_cm (requested)   : {arm.params.get('neutral_baffle_positions_cm')}")
    print(f"  clear_radii_cm             : {arm.params.get('neutral_baffle_clear_radii_cm')}")
    print(f"  fa4j control baffles       : {ctl.flags.get('neutral_baffles')} "
          f"(positions {ctl.params.get('neutral_baffle_positions_cm')})")
    print(f"  realized span used for bins: {BAF_LO} - {BAF_HI} cm "
          f"(faces 124/129/135; port 27 = {PORTS[27]} cm)")

    banner(
        "B. THE EQUILIBRATED BASE (t = 0). The baffle is GEOMETRY-CLASS: it\n"
        "   throttles annular spreading DURING equilibration, so the base MAY\n"
        "   legitimately move. Quantified, not assumed (the fa3 pattern) --\n"
        "   downstream deltas must be attributed base-vs-discharge."
    )
    zone_table(arm, ctl, 0.0, ZONES)
    ja, jc = 0, 0
    na, nc = arm.h5["nn"][ja], ctl.h5["nn"][jc]
    aa_, ac_ = arm.h5["nn_a"][ja], ctl.h5["nn_a"][jc]
    for label, va, vc in (("nn (column)", na, nc), ("nn_a (annulus)", aa_, ac_)):
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(vc > 0, np.abs(va - vc) / np.maximum(vc, 1e-300), 0.0)
        print(f"  base SHAPE {label:<16}: max |rel dev| {rel.max():.4e} "
              f"at z {arm.z[int(np.argmax(rel))]:.1f} cm")

    banner(
        "C. THE ANNULAR FLOW ACROSS THE CHOKE. Does inventory pile UPSTREAM\n"
        "   of the baffle and thin DOWNSTREAM? (Tom's question, part b.)"
    )
    for t in (5.0, 10.0, 15.0, 20.0):
        zone_table(arm, ctl, t, CHOKE)

    banner("D. nn_a(z) ANNULUS PROFILE ACROSS THE CHOKE [cm^-3]")
    for t in (5.0, 10.0, 20.0):
        ja, jc = arm.isnap(t), ctl.isnap(t)
        m = arm.zmask(850.0, 1150.0)
        za = arm.z[m]
        va = arm.h5["nn_a"][ja, m]
        vc = ctl.h5["nn_a"][jc, m]
        print(f"\n  t = {t:.1f} ms   {'z':>8} {'nn_a arm':>12} {'nn_a fa4j':>12} {'x':>7}")
        for k in range(0, za.size, 2):
            mark = " <<BAF" if BAF_LO - 4 <= za[k] <= BAF_HI + 4 else ""
            print(f"              {za[k]:>8.1f} {va[k]:>12.4e} {vc[k]:>12.4e} "
                  f"{va[k] / vc[k] if vc[k] else float('nan'):>7.3f}{mark}")

    banner("E. FULL REGIONAL INVENTORIES, COLUMN vs ANNULUS -- where does the\n"
           "   choked gas go? (fa2_diag zones.)")
    for t in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        zone_table(arm, ctl, t, ZONES)

    banner("F. THE MID-PORT ROWS AND THE PILE (raw model values at the port z)")
    for t in (5.0, 10.0, 20.0):
        ja, jc = arm.isnap(t), ctl.isnap(t)
        print(f"\n  t = {t:.1f} ms  {'port':>5} {'z':>8} {'n arm':>12} "
              f"{'n fa4j':>12} {'x':>7} {'Te arm':>8} {'Te fa4j':>8} "
              f"{'nn_a arm':>11} {'nn_a fa4j':>11} {'x':>7}")
        for p, z in PORTS.items():
            i = int(np.argmin(np.abs(arm.z - z)))
            na_ = arm.h5["n"][ja, i]
            nc_ = ctl.h5["n"][jc, i]
            ta_ = arm.h5["Te"][ja, i]
            tc_ = ctl.h5["Te"][jc, i]
            aa2 = arm.h5["nn_a"][ja, i]
            ac2 = ctl.h5["nn_a"][jc, i]
            print(f"          {p:>5} {z:>8.1f} {na_:>12.4e} {nc_:>12.4e} "
                  f"{na_ / nc_ if nc_ else float('nan'):>7.3f} {ta_:>8.3f} "
                  f"{tc_:>8.3f} {aa2:>11.4e} {ac2:>11.4e} "
                  f"{aa2 / ac2 if ac2 else float('nan'):>7.3f}")

    banner("G. THE HOT CHANNEL. The kernel deposits column hot deaths into the\n"
           "   ANNULUS at their own z (a LOCAL source); the baffle acts only on\n"
           "   the AXIAL annulus conductance/area/momentum. PREDICTION from the\n"
           "   code read: the baffle cannot change WHERE hot atoms deposit, only\n"
           "   whether they are trapped axially afterwards. Tested here.")
    for t in (5.0, 10.0, 20.0):
        ja, jc = arm.isnap(t), ctl.isnap(t)
        print(f"\n  t = {t:.1f} ms")
        print(f"    {'zone':<24} {'f_hot arm':>11} {'f_hot fa4j':>11} "
              f"{'nn_hot arm':>12} {'nn_hot fa4j':>12} {'x':>7} "
              f"{'births arm':>12} {'births fa4j':>12} {'x':>7}")
        for name, (lo, hi) in ZONES:
            ma, mc = arm.zmask(lo, hi), ctl.zmask(lo, hi)
            if not ma.any():
                continue
            fa = float(np.mean(arm.h5["f_hot"][ja, ma]))
            fc = float(np.mean(ctl.h5["f_hot"][jc, mc]))
            ha = float(arm.h5["nn_hot"][ja, ma] @ arm.V_col[ma])
            hc = float(ctl.h5["nn_hot"][jc, mc] @ ctl.V_col[mc])
            ba = float(arm.h5["hot_births"][ja, ma] @ arm.V_col[ma])
            bc = float(ctl.h5["hot_births"][jc, mc] @ ctl.V_col[mc])
            print(f"    {name:<24} {fa:>11.4e} {fc:>11.4e} {ha:>12.4e} "
                  f"{hc:>12.4e} {ha / hc if hc else float('nan'):>7.3f} "
                  f"{ba:>12.4e} {bc:>12.4e} "
                  f"{ba / bc if bc else float('nan'):>7.3f}")

    banner("H. Tn / u_n / M_n AROUND THE CHOKE (two-channel + momentum reads)")
    for t in (5.0, 20.0):
        ja, jc = arm.isnap(t), ctl.isnap(t)
        print(f"\n  t = {t:.1f} ms  {'z':>8} {'Tn arm':>10} {'Tn fa4j':>10} "
              f"{'u_n arm':>12} {'u_n fa4j':>12} {'M_n arm':>12} {'M_n fa4j':>12}")
        for z in (850.0, 900.0, 940.0, 977.5, 1022.5, 1060.0, 1100.0):
            i = int(np.argmin(np.abs(arm.z - z)))
            print(f"          {arm.z[i]:>8.1f} {arm.h5['Tn'][ja, i]:>10.3f} "
                  f"{ctl.h5['Tn'][jc, i]:>10.3f} {arm.h5['u_n'][ja, i]:>12.4e} "
                  f"{ctl.h5['u_n'][jc, i]:>12.4e} {arm.h5['M_n'][ja, i]:>12.4e} "
                  f"{ctl.h5['M_n'][jc, i]:>12.4e}")

    arm.close()
    ctl.close()


if __name__ == "__main__":
    main()
