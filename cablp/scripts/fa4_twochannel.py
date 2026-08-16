"""fa4 -- THE TWO-CHANNEL NEUTRAL DIAGNOSTICS (neutral_energy=True).

READ-ONLY instrument. No tuning, no fitting, no state mutation; it only opens
saved trajectories. Nothing is written back to any artifact.

Built for the fa4 NBL-stage verdict run against its fa2 control.

--------------------------------------------------------------------------
WHAT THE ARTIFACT CAN AND CANNOT ANSWER  (established before writing this;
the distinction is load-bearing and is printed in the output, not hidden)
--------------------------------------------------------------------------
RECOVERABLE from a saved h5:
  * Tn_cold(z,t)   -- top-level 'Tn' dataset, ALREADY IN eV,
                      Tn = (2/3) En / (nn_floored * ev_to_erg).
  * S_cx           -- the CX transfer rate, as the cold-side debit
                      rhs_terms/neutral_cx_channel/nn = -S_cx * ratio.
  * wall deaths    -- rhs_terms/neutral_hot_channel/nn_a (two-zone landing
                      deposit into the annulus).
  * in-flight ionization -- rhs_terms/neutral_hot_channel/n (= ionized_here),
                      with its binding cost on .../Ee = -I_ion*ev_to_erg*n.
  * the COMBINED hot->ion energy return -- rhs_terms/neutral_hot_channel/Ei.

NOT RECOVERABLE (measurement limits, reported as such):
  * f_hot / nn_hot / tau_hot are attached to the LIVE result
    (solver.py:10015-10017) but are NOT in io.py's write list, so
    save_result_hdf5 DROPS them. They are absent from the h5.
  * The scalar hot diagnostics -- 'hot_births_per_s',
    'hot_wall_energy_erg_s', 'hot_wall_energy_returned_erg_s',
    'hot_end_fold_fraction' -- are cached on the solver and never reach the
    result at all.
  * The re-CX ion-return power is NOT separable from the in-flight
    ionization deposit: the Ei row is the SUM dEi_recx + dEi_ion, and both
    are nonlocal 'spread' integrals over the residence kernel, so
    dEi_ion != ionized_here * e_hot_local. Only the sum is a fact of the
    artifact. This instrument reports the sum and says so.

VOLUME CONVENTION (the easy way to get this wrong):
  En and its rhs rows live on V_En = neutral_energy_volume_ratio's volume:
  the CHAMBER (neutral_volume_cm3) in single-zone runs, the COLUMN
  (plasma_volume_cm3) in two-zone runs. fa4 is TWO-ZONE, so V_En = V_col --
  but the choice is made from the artifact, never assumed.

UNITS: rhs energy rows (Ee, Ei, En) are CGS erg cm^-3 s^-1 -> /1e7 for W.
Particle rows (n, nn, nn_a) are cm^-3 s^-1 -> integrate to 1/s directly.
"""

import json
import sys

import h5py
import numpy as np

ERG_PER_J = 1.0e7

REGIONS = [
    ("SOURCE z<=100", -1e9, 100.0),
    ("100<z<790", 100.0, 790.0),
    ("MID 790-1045", 790.0, 1045.0),
    ("FAR 1045-1900", 1045.0, 1900.0),
]
PORTS = [(11, 470.0), (21, 790.0), (29, 1045.0), (41, 1429.0), (50, 1716.0)]
WINDOWS = [(2.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 19.5)]

FTS_LO_EV, FTS_HI_EV = 0.3, 0.8   # fts/TPMC frozen-field column class
WALL_EV = 300.0 * 1.380649e-16 / 1.602176634e-12   # ~0.02585 eV


class Run:
    def __init__(self, path, tag):
        self.tag, self.path = tag, path
        self.h5 = h5py.File(path, "r")
        g = self.h5["geometry"]
        self.z = g["z_cm"][:]
        self.V_col = g["plasma_volume_cm3"][:]
        self.V_neu = g["neutral_volume_cm3"][:]
        self.V_ann = self.V_neu - self.V_col
        self.t_ms = self.h5["time"][:] * 1e3
        self.params = json.loads(self.h5.attrs["params_json"])
        self.flags = json.loads(self.h5.attrs["flags_json"])
        self.two_zone = "nn_a" in self.h5
        # The En volume is decided FROM the artifact, not assumed.
        self.V_En = self.V_col if self.two_zone else self.V_neu

    def close(self):
        self.h5.close()

    def has(self, name):
        return name in self.h5

    def zmask(self, lo, hi):
        return (self.z >= lo) & (self.z <= hi)

    def wsl(self, lo_ms, hi_ms):
        i0 = int(np.searchsorted(self.t_ms, lo_ms))
        i1 = int(np.searchsorted(self.t_ms, hi_ms))
        return slice(i0, min(i1 + 1, self.t_ms.size))

    def isnap(self, t_ms):
        return int(np.argmin(np.abs(self.t_ms - t_ms)))

    def tree(self):
        if "rhs_terms" not in self.h5:
            return {}
        return {t: sorted(self.h5["rhs_terms"][t].keys())
                for t in self.h5["rhs_terms"].keys()}

    def _vol_for(self, group):
        if group in ("Ee", "Ei", "n", "M"):
            return self.V_col
        if group in ("En", "nn", "M_n"):
            return self.V_En if group == "En" else (
                self.V_col if self.two_zone else self.V_neu)
        if group.endswith("_a"):
            return self.V_ann
        return self.V_col

    def row_int(self, term, group, mask=None):
        """Volume-integrated rhs row. Energy groups -> W, particle -> 1/s."""
        m = np.ones_like(self.z, dtype=bool) if mask is None else mask
        grp = self.h5["rhs_terms"]
        if term not in grp or group not in grp[term]:
            return None
        vol = self._vol_for(group)
        out = grp[term][group][:, m] @ vol[m]
        if group in ("Ee", "Ei", "En"):
            out = out / ERG_PER_J
        return out

    def wmean(self, dset, lo, hi, mask, vol=None):
        if dset not in self.h5:
            return None
        v = self.V_col if vol is None else vol
        a = self.h5[dset][self.wsl(lo, hi)][:, mask]
        w = v[mask]
        return float((a @ w).mean() / w.sum())


def banner(t, note=""):
    print("=" * 110)
    print(t)
    if note:
        for line in note.split("\n"):
            print("   " + line)
    print("=" * 110)


def sec0(arm, ctl):
    banner("0. TERM DISCOVERY -- names READ from the artifacts, never assumed")
    at, ct = arm.tree(), ctl.tree()
    new = sorted(set(at) - set(ct))
    print(f"  ARM {arm.tag}: {len(at)} rhs terms   "
          f"CTL {ctl.tag}: {len(ct)} rhs terms")
    print(f"  two_zone: arm={arm.two_zone} ctl={ctl.two_zone}")
    print(f"  V_En basis: arm={'V_col (two-zone)' if arm.two_zone else 'V_chamber'}")
    print(f"\n  *** ARM-ONLY TERMS ({len(new)}) ***")
    for t in new:
        print(f"      + {t:<46} groups={at[t]}")
    ch = [t for t in sorted(set(at) & set(ct)) if at[t] != ct[t]]
    if ch:
        print(f"\n  *** SHARED TERMS WHOSE GROUP SET CHANGED ({len(ch)}) ***")
        for t in ch:
            print(f"      ~ {t:<34}")
            print(f"          ctl={ct[t]}")
            print(f"          arm={at[t]}")
    print("\n  TOP-LEVEL two-channel datasets:")
    for nm in ("En", "Tn", "nn_hot", "f_hot", "tau_hot", "u_n", "M_n",
               "nn", "nn_a", "n", "Te", "Ti"):
        print(f"      {nm:<9} arm={arm.has(nm)!s:<6} ctl={ctl.has(nm)}")
    missing = [n for n in ("nn_hot", "f_hot", "tau_hot") if not arm.has(n)]
    if missing:
        print(f"\n  !! MEASUREMENT LIMIT: {missing} are computed by the solver "
              f"and\n     attached to the live result, but are NOT in io.py's "
              f"write list, so\n     save_result_hdf5 DROPS them. f_hot(z,t) "
              f"is NOT recoverable from this\n     artifact. Reported as a "
              f"gap, not silently skipped.")
    return new


def sec1_tn(arm):
    banner("1. Tn_cold(z,t): THE COLD CHANNEL TEMPERATURE [eV]",
           f"fts/TPMC frozen-field column class = {FTS_LO_EV}-{FTS_HI_EV} eV.\n"
           f"wall/feed reference = {WALL_EV:.5f} eV (300 K).")
    if not arm.has("Tn"):
        print("  ARM carries no Tn field.")
        return
    print(f"  {'window[ms]':>14}" + "".join(f"{r[0]:>18}" for r in REGIONS))
    for lo, hi in WINDOWS:
        cells = []
        for _, a, b in REGIONS:
            v = arm.wmean("Tn", lo, hi, arm.zmask(a, b))
            cells.append(f"{v:18.4f}" if v is not None else f"{'--':>18}")
        print(f"  {lo:6.1f}-{hi:6.1f}" + "".join(cells))
    print("\n  VERDICT vs the fts/TPMC class (plateau 15.0-19.5 ms):")
    for nm, a, b in REGIONS:
        v = arm.wmean("Tn", 15.0, 19.5, arm.zmask(a, b))
        if v is None:
            continue
        w = ("INSIDE" if FTS_LO_EV <= v <= FTS_HI_EV
             else ("BELOW" if v < FTS_LO_EV else "ABOVE"))
        print(f"      {nm:<16} Tn={v:9.4f} eV  -> {w} the "
              f"{FTS_LO_EV}-{FTS_HI_EV} eV class "
              f"({v / WALL_EV:7.2f}x the wall value)")
    print("\n  1b. Tn vs Ti in the MID column (the B1 mechanism read):")
    mid = arm.zmask(790.0, 1045.0)
    print(f"      {'window[ms]':>14} {'Tn[eV]':>10} {'Ti[eV]':>10} "
          f"{'Tn-Ti':>10} {'Tn/Ti':>8}  direction")
    for lo, hi in WINDOWS:
        tn, ti = arm.wmean("Tn", lo, hi, mid), arm.wmean("Ti", lo, hi, mid)
        if tn is None or ti is None:
            continue
        d = ("Tn>Ti: ions HEATED by neutrals" if tn > ti
             else "Tn<Ti: ions COOLED by neutrals (the dump)")
        print(f"      {lo:6.1f}-{hi:6.1f} {tn:10.4f} {ti:10.4f} "
              f"{tn - ti:+10.4f} {tn / ti if ti else np.nan:8.3f}  {d}")
    print("\n  1c. EARLY MID Tn rise (the ~0.4 ms equilibration scale):")
    print(f"      {'t[ms]':>8} {'Tn[eV]':>10} {'Ti[eV]':>10} {'Tn/Ti':>8}")
    w = arm.V_col[mid]
    for t in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0):
        i = arm.isnap(t)
        tn = float(arm.h5["Tn"][i][mid] @ w / w.sum())
        ti = float(arm.h5["Ti"][i][mid] @ w / w.sum())
        print(f"      {arm.t_ms[i]:8.3f} {tn:10.4f} {ti:10.4f} "
              f"{tn / ti if ti else np.nan:8.3f}")


def sec2_hot(arm):
    banner("2. THE HOT CHANNEL",
           "f_hot/nn_hot/tau_hot are NOT saved (see section 0). What follows "
           "is what the\nartifact DOES carry: the hot channel's own rhs rows.")
    tree = arm.tree()
    if "neutral_hot_channel" not in tree:
        print("  no neutral_hot_channel term present.")
        return
    print(f"  neutral_hot_channel groups: {tree['neutral_hot_channel']}")
    print("\n  2a. THE FOUR FATES, whole column, per window.")
    print("      S_cx      = -neutral_cx_channel.nn        [1/s]  CX births")
    print("      wall      =  neutral_hot_channel.nn_a     [1/s]  wall deaths")
    print("      ionized   =  neutral_hot_channel.n        [1/s]  in-flight ion.")
    print("      CLOSURE: total births = wall + ionized (re-CX replaces its own"
          " atom),\n      so S_cx == wall + ionized is an IDENTITY of the model"
          " -- checked here.\n")
    print(f"      {'window[ms]':>14} {'S_cx[1/s]':>16} {'wall[1/s]':>16} "
          f"{'ionized[1/s]':>16} {'wall+ion':>16} {'ratio':>9}")
    for lo, hi in WINDOWS:
        sl = arm.wsl(lo, hi)
        scx = arm.row_int("neutral_cx_channel", "nn")
        wall = arm.row_int("neutral_hot_channel", "nn_a")
        if wall is None:
            wall = arm.row_int("neutral_hot_channel", "nn")
        ion = arm.row_int("neutral_hot_channel", "n")
        if scx is None or wall is None or ion is None:
            print("      (a required row is absent)")
            break
        s = -float(scx[sl].mean())
        w_ = float(wall[sl].mean())
        i_ = float(ion[sl].mean())
        print(f"      {lo:6.1f}-{hi:6.1f} {s:16.4e} {w_:16.4e} {i_:16.4e} "
              f"{w_ + i_:16.4e} {(w_ + i_) / s if s else np.nan:9.5f}")
    print("\n  2b. HOT-CHANNEL FATES per region (plateau 15.0-19.5 ms) [1/s]")
    print(f"      {'quantity':<34}" + "".join(f"{r[0]:>18}" for r in REGIONS))
    for label, term, group, sign in (
            ("S_cx (CX births)", "neutral_cx_channel", "nn", -1.0),
            ("wall deaths -> annulus", "neutral_hot_channel", "nn_a", 1.0),
            ("in-flight ionization", "neutral_hot_channel", "n", 1.0)):
        cells = []
        for _, a, b in REGIONS:
            r = arm.row_int(term, group, arm.zmask(a, b))
            cells.append(f"{sign * float(r[arm.wsl(15.0, 19.5)].mean()):18.4e}"
                         if r is not None else f"{'--':>18}")
        print(f"      {label:<34}" + "".join(cells))


def sec3_recycling(arm):
    banner("3. THE CX-RECYCLING POWER (the registered measurement)",
           "JETLESS ARM: the recycling here is CX-TAIL-ONLY. It is NOT "
           "comparable\nlike-for-like with the ~60 kW jet-scale prediction "
           "context.\n"
           "SEPARABILITY LIMIT: neutral_hot_channel.Ei = dEi_recx + dEi_ion. "
           "Both are\nnonlocal 'spread' integrals over the residence kernel, "
           "so the re-CX part is\nNOT separable from the artifact. The SUM is "
           "the measurable quantity.")
    ei = arm.row_int("neutral_hot_channel", "Ei")
    ee = arm.row_int("neutral_hot_channel", "Ee")
    if ei is None:
        print("  no neutral_hot_channel.Ei row.")
        return
    print(f"  {'window[ms]':>14} {'hot->ion Ei [kW]':>20} "
          f"{'hot ioniz. Ee [kW]':>22} {'net hot->plasma [kW]':>22}")
    for lo, hi in WINDOWS:
        sl = arm.wsl(lo, hi)
        a = float(ei[sl].mean()) / 1e3
        b = float(ee[sl].mean()) / 1e3 if ee is not None else np.nan
        print(f"  {lo:6.1f}-{hi:6.1f} {a:20.5f} {b:22.5f} {a + b:22.5f}")
    print("\n  3b. hot->ion return power PER REGION (plateau) [kW]")
    print(f"      {'row':<26}" + "".join(f"{r[0]:>18}" for r in REGIONS))
    for label, group in (("neutral_hot_channel.Ei", "Ei"),
                         ("neutral_hot_channel.Ee", "Ee")):
        cells = []
        for _, a, b in REGIONS:
            r = arm.row_int("neutral_hot_channel", group, arm.zmask(a, b))
            cells.append(f"{float(r[arm.wsl(15.0, 19.5)].mean()) / 1e3:18.5f}"
                         if r is not None else f"{'--':>18}")
        print(f"      {label:<26}" + "".join(cells))
    en = arm.row_int("neutral_hot_channel", "En")
    if en is not None:
        v = float(en[arm.wsl(15.0, 19.5)].mean())
        print(f"\n  3c. neutral_hot_channel.En (energy RETURNED to the cold "
              f"gas) = {v:.6e} W")
        if arm.two_zone and abs(v) == 0.0:
            print("      EXACTLY ZERO, and that is STRUCTURAL, not a null "
                  "measurement:\n      under two-zone the landed atoms deposit "
                  "into the ANNULUS, which carries\n      no energy field (the "
                  "ratified annulus-cold v1 cut). The whole hot excess\n"
                  "      energy is left on the wall; none returns. "
                  "'hot_wall_energy_returned_erg_s'\n      is likewise 0.0 by "
                  "construction on this arm (hot_neutrals.py:405).")


def sec4_erosion(arm, ctl):
    banner("4. EROSION AT THE PILE",
           "the CX-ballistic axial mass movement that relieves the pile.\n"
           "The fa2 control has NO hot channel, so its pile is static by "
           "construction.")
    far = arm.zmask(1045.0, 1900.0)
    allm = np.ones_like(arm.z, dtype=bool)

    def inv(r, i, m):
        return float(r.h5["nn"][i][m] @ r.V_col[m]
                     + r.h5["nn_a"][i][m] @ r.V_ann[m])

    print("  4a. FAR-region neutral inventory and share [particles]")
    print(f"      {'t[ms]':>8} {'ARM far':>15} {'CTL far':>15} {'ratio':>8} "
          f"{'ARM share':>11} {'CTL share':>11} {'d(share)':>10}")
    for t in (2.0, 5.0, 10.0, 15.0, 19.0, 19.5):
        ia, ic = arm.isnap(t), ctl.isnap(t)
        fa, fc = inv(arm, ia, far), inv(ctl, ic, far)
        wa, wc = inv(arm, ia, allm), inv(ctl, ic, allm)
        sa, sc = 100 * fa / wa, 100 * fc / wc
        print(f"      {arm.t_ms[ia]:8.2f} {fa:15.4e} {fc:15.4e} "
              f"{fa / fc if fc else np.nan:8.4f} {sa:10.2f}% {sc:10.2f}% "
              f"{sa - sc:+9.2f}pp")
    print("\n  4b. PEAK column nn and its location (does the pile move?)")
    print(f"      {'t[ms]':>8} {'ARM peak':>14} {'z ARM':>8} {'CTL peak':>14} "
          f"{'z CTL':>8} {'ratio':>8} {'dz[cm]':>8}")
    for t in (2.0, 5.0, 10.0, 15.0, 19.0, 19.5):
        ia, ic = arm.isnap(t), ctl.isnap(t)
        na, nc = arm.h5["nn"][ia], ctl.h5["nn"][ic]
        ja, jc = int(np.argmax(na)), int(np.argmax(nc))
        print(f"      {arm.t_ms[ia]:8.2f} {na[ja]:14.4e} {arm.z[ja]:8.1f} "
              f"{nc[jc]:14.4e} {ctl.z[jc]:8.1f} {na[ja] / nc[jc]:8.4f} "
              f"{arm.z[ja] - ctl.z[jc]:+8.1f}")
    print("\n  4c. HOT-CHANNEL activity resident IN the FAR pile [1/s]")
    print(f"      {'t-window':>14} {'S_cx far':>16} {'wall far':>16} "
          f"{'ionized far':>16}")
    for lo, hi in WINDOWS:
        sl = arm.wsl(lo, hi)
        s = arm.row_int("neutral_cx_channel", "nn", far)
        w = arm.row_int("neutral_hot_channel", "nn_a", far)
        i_ = arm.row_int("neutral_hot_channel", "n", far)
        print(f"      {lo:6.1f}-{hi:6.1f} "
              f"{-float(s[sl].mean()) if s is not None else np.nan:16.4e} "
              f"{float(w[sl].mean()) if w is not None else np.nan:16.4e} "
              f"{float(i_[sl].mean()) if i_ is not None else np.nan:16.4e}")


def sec5_health(arm, ctl):
    banner("5. TWO-CHANNEL RUN HEALTH",
           "En floor debits, finiteness, and the dt_neutral_energy headroom.")
    for r in (arm, ctl):
        print(f"\n  --- {r.tag}: {r.path} ---")
        print(f"    run_status {r.h5.attrs.get('run_status')}  "
              f"steps {r.h5.attrs.get('steps')}")
        if "floor_ledger" in r.h5:
            fl = r.h5["floor_ledger"]
            print("    floor ledger:")
            for k in sorted(fl.keys()):
                print(f"        {k:<30} {float(fl[k][()]):.6e}")
        for nm in ("En", "Tn", "nn", "nn_a", "n", "Te", "Ti", "u_n"):
            if r.has(nm):
                a = r.h5[nm][:]
                print(f"    finiteness {nm:<6}: finite="
                      f"{bool(np.isfinite(a).all())!s:<6} "
                      f"min={np.nanmin(a):.4e} max={np.nanmax(a):.4e}")
        d = r.h5["diagnostics"]
        if "dt_neutral_energy" in d:
            dte = d["dt_neutral_energy"][:]
            raw = d["dt_raw"][:]
            good = np.isfinite(dte) & np.isfinite(raw) & (raw > 0)
            hr = dte[good] / raw[good]
            print(f"    dt_neutral_energy: min={np.nanmin(dte):.4e} "
                  f"median={np.nanmedian(dte):.4e}")
            print(f"    HEADROOM dt_neutral_energy/dt_raw: "
                  f"min={hr.min():.4f} p5={np.percentile(hr, 5):.4f} "
                  f"median={np.median(hr):.4f}")
            print(f"      (>1 means the En bound was NOT the binding "
                  f"constraint; the 7x watch)")
        else:
            print("    dt_neutral_energy: ABSENT (pre-merge artifact)")
        if "active_constraint" in d:
            ac = np.array([x.decode() if isinstance(x, (bytes, np.bytes_))
                           else str(x) for x in d["active_constraint"][:]])
            adt = d["accepted_dt"][:] if "accepted_dt" in d else d["dt"][:]
            print("    active timestep constraint census:")
            u, c = np.unique(ac, return_counts=True)
            for k, v in sorted(zip(u, c), key=lambda x: -x[1]):
                tfrac = adt[ac == k].sum() / adt.sum()
                print(f"        {k:<26} {v:8d} ({100 * v / c.sum():6.2f}% of "
                      f"steps, {100 * tfrac:6.2f}% of elapsed)")


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: fa4_twochannel.py ARM.h5 CTL.h5")
    arm, ctl = Run(sys.argv[1], "ARM(fa4)"), Run(sys.argv[2], "CTL(fa2)")
    print("=" * 110)
    print("fa4 -- TWO-CHANNEL NEUTRAL DIAGNOSTICS")
    print(f"  ARM {arm.path}")
    print(f"      neutral_energy={arm.flags.get('neutral_energy')} "
          f"neutral_momentum={arm.flags.get('neutral_momentum')} "
          f"two_zone={arm.flags.get('neutral_two_zone')}")
    print(f"      alpha_E={arm.params.get('neutral_energy_wall_accommodation')} "
          f"knudsen_T={arm.params.get('neutral_knudsen_temperature')!r} "
          f"Tn_K={arm.params.get('Tn_K')}")
    print(f"  CTL {ctl.path}")
    print(f"      neutral_energy={ctl.flags.get('neutral_energy')} "
          f"neutral_momentum={ctl.flags.get('neutral_momentum')}")
    print("=" * 110 + "\n")
    new = sec0(arm, ctl)
    print()
    sec1_tn(arm)
    print()
    sec2_hot(arm)
    print()
    sec3_recycling(arm)
    print()
    sec4_erosion(arm, ctl)
    print()
    sec5_health(arm, ctl)
    arm.close()
    ctl.close()


if __name__ == "__main__":
    main()
