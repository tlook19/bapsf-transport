"""fa3 END-FLARE reader: the geometry, the ion flow through it, and the end pile.

READ-ONLY instrument for the END-FLARE EXPANSION PROBE (fueling-anchor
campaign, thread 19). It complements ``fa2_diag.py`` -- which already covers
the equilibrated base, the NBL, the first-metre heat flux, the quench front,
the mid-column state, the dump channel, the neutral wind field and run health
-- by measuring the things that are specific to the flare and that no
existing instrument reports:

  A. the BUILT flare geometry, read back from the artifacts (never assumed):
     Rp_cm, plasma cross-sectional area and cell volumes over the end block,
     plus the cell-count identity check between arm and control;
  B. the EQUILIBRATED BASE difference arm vs control, computed with each
     run's OWN cell volumes -- the flare changes plasma cell volumes, so the
     geometry-consistent seed necessarily differs and the difference must be
     quantified before any discharge-phase delta is attributed;
  C. THE MECHANISM UNDER TEST: the plasma/ion flow u(z) through the end
     block, as a fraction of the LOCAL ion sound speed, with the local Te
     and Ti stated alongside it;
  D. the ion PARTICLE FLUX n*u*A through the block -- whether area expansion
     accelerates the ions or merely dilutes them at fixed throughput;
  E. the END-BLOCK and FAR-REGION neutral inventories -- whether the pile at
     the end evacuates, shrinks, or moves.

***** LOWER BOUND BY CONSTRUCTION *****
The model carries NO MIRROR FORCE. The end-flare option expands the plasma
flux-tube AREA only. Every acceleration number below is therefore a lower
bound on what real end field expansion could do.

DECLARED DEFINITIONS (fixed before the run was scored; the conventions are
taken from fa2_diag.py so the two instruments compose):

  plateau window   15.0-19.5 ms, sample-mean over saved frames.
  FAR region       1045 <= z <= 1900 cm (fa2_diag's FAR_Z).
  end block        the cells whose ``geometry/cell_role`` is 'end' or
                   'collector' -- i.e. exactly the ``end_expansion_cells``
                   block the flare acts on (9 'end' + 1 'collector' = 10
                   at this stance, verified per-artifact below).
  plasma area      A(z) = pi * Rp_cm(z)^2 [cm^2], read from the artifact's
                   own Rp_cm, not recomputed from the config.
  ion sound speed  TWO declared forms, both reported, neither privileged:
                     c_s(Te)    = sqrt(e*Te / m_i)            (cold-ion Bohm)
                     c_s(Te+Ti) = sqrt(e*(Te+Ti) / m_i)
                   with m_i = 4.002602 u (He+). Velocities in the artifact
                   are CGS (cm/s); /1e5 gives km/s.
  neutral inv.     nn @ V_col + nn_a @ V_ann, each run using its OWN
                   volumes (the two-zone particle channel's inventory, the
                   same definition fa2_diag uses).
  FAR SHARE        TWO denominators, both reported. The WHOLE-DOMAIN one
                   (column + plenum/obstruction/cathode cells at z < 0) is
                   the one that reproduces the fa2 control's quoted 50.5 %
                   at 19 ms and is therefore the PRIMARY, comparable figure;
                   the column-only denominator (60.4 % on the same control
                   frame) is reported beside it so neither convention has to
                   be guessed at later. The plenum holds a large reservoir
                   at this fill, which is the whole difference between them.
  PEAK nn          column-zone nn only (not nn + nn_a), reported as a TIME
                   SERIES rather than one frame: the pile MOVES inward
                   through the plateau (control 5.06e13 @ z=1416 at 15 ms ->
                   5.55e13 @ z=1349 at 19.5 ms), so a single-frame quote is
                   a convention, not a fact, and the series makes the A/B
                   independent of which frame is quoted.
  ion flux         Gamma(z) = n(z) * u(z) * A(z) [s^-1], signed: POSITIVE is
                   flow TOWARD the end wall (+z).

Usage:
    python scripts/fa3_flare_read.py --arm scripts/fa3_arm.h5 \\
        --ctl scripts/fa2_arm.h5
"""

import argparse
import json

import h5py
import numpy as np

FAR_Z = (1045.0, 1900.0)
PLATEAU = (15.0, 19.5)
SNAPS_MS = (2.0, 5.0, 10.0, 15.0, 19.5)

# He+ mass [g] and the erg-per-eV conversion, CGS throughout.
# SUPERSEDED 2026-08-21: the unified helium mass is cablp.vars._cons
# .m_He_cgs = 6.6464790809e-24 g (Ar(4He)*u, CODATA 2022). The literal
# below is 0.31 ppm low and is left AS A RECORD of what this dated script ran.
M_I_G = 4.002602 * 1.66053906660e-24
ERG_PER_EV = 1.602176634e-12


def _s(x):
    if isinstance(x, bytes):
        return x.decode()
    return str(x)


class Run:
    def __init__(self, path, tag):
        self.path = path
        self.tag = tag
        self.h5 = h5py.File(path, "r")
        g = self.h5["geometry"]
        self.z = g["z_cm"][:]
        self.Rp = g["Rp_cm"][:]
        self.Rm = g["Rm_cm"][:]
        self.role = np.array([_s(s) for s in g["cell_role"][:]])
        self.V_col = g["plasma_volume_cm3"][:]
        self.V_ann = g["neutral_volume_cm3"][:] - self.V_col
        self.active = np.asarray(g["plasma_active"], bool)
        self.A = np.pi * self.Rp ** 2
        self.t_ms = self.h5["time"][:] * 1e3
        self.params = json.loads(self.h5.attrs["params_json"])
        self.flags = json.loads(self.h5.attrs["flags_json"])
        self.end = np.isin(self.role, ("end", "collector"))
        self.col = self.z >= 0.0

    def close(self):
        self.h5.close()

    def wsl(self, lo_ms, hi_ms):
        i0 = int(np.searchsorted(self.t_ms, lo_ms))
        i1 = int(np.searchsorted(self.t_ms, hi_ms))
        return slice(i0, min(i1 + 1, self.t_ms.size))

    def isnap(self, t_ms):
        return int(np.argmin(np.abs(self.t_ms - t_ms)))

    def zmask(self, lo, hi):
        return (self.z >= lo) & (self.z <= hi)

    def pmean(self, name, sl):
        return np.asarray(self.h5[name][sl]).mean(axis=0)

    def neutral_inventory(self, mask=None):
        m = np.ones_like(self.z, dtype=bool) if mask is None else mask
        return (self.h5["nn"][:, m] @ self.V_col[m]
                + self.h5["nn_a"][:, m] @ self.V_ann[m])

    def plasma_inventory(self, mask=None):
        m = np.ones_like(self.z, dtype=bool) if mask is None else mask
        return self.h5["n"][:, m] @ self.V_col[m]


def banner(title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)


# --------------------------------------------------------------------------
def section_geometry(arm, ctl):
    banner("A. THE BUILT FLARE  (read back FROM the artifacts, never assumed)")
    for r in (arm, ctl):
        print(f"  {r.tag:<6} end_expansion_geometry      = "
              f"{r.flags.get('end_expansion_geometry', '<absent>')}")
        print(f"  {r.tag:<6} end_expansion_cells         = "
              f"{r.params.get('end_expansion_cells', '<absent>')}")
        print(f"  {r.tag:<6} end_expansion_machine_R_cm  = "
              f"{r.params.get('end_expansion_machine_radius_cm', '<absent>')}")
        print(f"  {r.tag:<6} end_expansion_plasma_R_cm   = "
              f"{r.params.get('end_expansion_plasma_radius_cm', '<absent>')}")
        print(f"  {r.tag:<6} Rp (column)                 = "
              f"{r.params.get('Rp', '<absent>')}")
    print()
    print(f"  total cells            arm {arm.z.size:5d}   ctl {ctl.z.size:5d}"
          f"   IDENTICAL={arm.z.size == ctl.z.size}")
    print(f"  end-block cells        arm {int(arm.end.sum()):5d}   "
          f"ctl {int(ctl.end.sum()):5d}   "
          f"IDENTICAL={int(arm.end.sum()) == int(ctl.end.sum())}")
    same_z = (arm.z.size == ctl.z.size
              and float(np.abs(arm.z - ctl.z).max()) == 0.0)
    print(f"  cell centres z_cm      bit-identical = {same_z}")
    print(f"  vessel radii Rm_cm     bit-identical = "
          f"{arm.Rm.size == ctl.Rm.size and float(np.abs(arm.Rm - ctl.Rm).max()) == 0.0}")
    print(f"  neutral volumes        bit-identical = "
          f"{float(np.abs(arm.V_col + arm.V_ann - ctl.V_col - ctl.V_ann).max()):.6e}"
          " (max abs diff cm^3)")

    print("\n  END-BLOCK PROFILE  (the 10 cells the flare acts on)")
    print(f"      {'idx':>4} {'role':>10} {'z[cm]':>9} {'Rp arm':>8} "
          f"{'Rp ctl':>8} {'A arm':>11} {'A ctl':>11} {'A ratio':>8} "
          f"{'Vcol arm':>12} {'Vcol ctl':>12}")
    idx = np.where(arm.end)[0]
    for i in idx:
        print(f"      {i:4d} {arm.role[i]:>10} {arm.z[i]:9.2f} "
              f"{arm.Rp[i]:8.3f} {ctl.Rp[i]:8.3f} {arm.A[i]:11.2f} "
              f"{ctl.A[i]:11.2f} {arm.A[i] / ctl.A[i]:8.4f} "
              f"{arm.V_col[i]:12.2f} {ctl.V_col[i]:12.2f}")
    print(f"      end-block plasma volume  arm {arm.V_col[arm.end].sum():.4e}"
          f"  ctl {ctl.V_col[ctl.end].sum():.4e}"
          f"  ratio {arm.V_col[arm.end].sum() / ctl.V_col[ctl.end].sum():.4f} cm^3")
    print(f"      terminal area ratio A_end/A_column = "
          f"{arm.A[idx[-1]] / arm.A[arm.col & ~arm.end][-1]:.4f} (arm)   "
          f"{ctl.A[idx[-1]] / ctl.A[ctl.col & ~ctl.end][-1]:.4f} (ctl)")
    print(f"      whole-column plasma volume arm {arm.V_col[arm.col].sum():.6e}"
          f"  ctl {ctl.V_col[ctl.col].sum():.6e}"
          f"  ratio {arm.V_col[arm.col].sum() / ctl.V_col[ctl.col].sum():.6f}")


# --------------------------------------------------------------------------
def section_base(arm, ctl):
    banner("B. THE EQUILIBRATED BASE (t = 0 saved frame), arm vs control\n"
           "   The flare changes PLASMA cell volumes, so a geometry-consistent\n"
           "   seed necessarily differs. Quantified here so discharge-phase\n"
           "   deltas can be attributed (flare-on-base vs flare-on-discharge).")
    for name, m_a, m_c in (("whole column", arm.col, ctl.col),
                           ("FAR 1045-1900", arm.zmask(*FAR_Z),
                            ctl.zmask(*FAR_Z)),
                           ("end block", arm.end, ctl.end)):
        ia = float(arm.neutral_inventory(m_a)[0])
        ic = float(ctl.neutral_inventory(m_c)[0])
        print(f"  neutral inventory  {name:<16} arm {ia:.6e}  ctl {ic:.6e}"
              f"  ratio {ia / ic:.6f}")
    print()
    na = arm.h5["nn"][0]
    nc = ctl.h5["nn"][0]
    tot_a = float(arm.neutral_inventory(arm.col)[0])
    tot_c = float(ctl.neutral_inventory(ctl.col)[0])
    print("  SHAPE test: column nn(z,0) each normalised to its OWN column")
    print("  inventory; max |relative deviation| arm vs ctl over the column:")
    sa = na[arm.col] / tot_a
    sc = nc[ctl.col] / tot_c
    dev = np.abs(sa - sc) / np.maximum(np.abs(sc), 1e-300)
    print(f"      max |rel dev| = {dev.max():.6e}  at z = "
          f"{arm.z[arm.col][int(np.argmax(dev))]:.1f} cm")
    print(f"      mean |rel dev| = {dev.mean():.6e}")
    print("\n  base nn(z,0) at the end block [cm^-3]:")
    print(f"      {'z[cm]':>9} {'nn arm':>13} {'nn ctl':>13} {'ratio':>9} "
          f"{'nn_a arm':>13} {'nn_a ctl':>13}")
    for i in np.where(arm.end)[0]:
        print(f"      {arm.z[i]:9.2f} {na[i]:13.4e} {nc[i]:13.4e} "
              f"{na[i] / nc[i]:9.4f} {arm.h5['nn_a'][0][i]:13.4e} "
              f"{ctl.h5['nn_a'][0][i]:13.4e}")


# --------------------------------------------------------------------------
def section_ionflow(arm, ctl):
    banner("C. THE MECHANISM UNDER TEST: plasma/ion flow through the end block\n"
           "   Does AREA EXPANSION accelerate the ions? Reported as a fraction\n"
           "   of the LOCAL ion sound speed, with local Te and Ti stated.\n"
           "   NO MIRROR FORCE IN THE MODEL -- this is the area half only,\n"
           "   hence a LOWER BOUND.")
    for r in (arm, ctl):
        sl = r.wsl(*PLATEAU)
        u = r.pmean("u", sl)
        Te = r.pmean("Te", sl)
        Ti = r.pmean("Ti", sl)
        n = r.pmean("n", sl)
        cs_e = np.sqrt(ERG_PER_EV * np.maximum(Te, 0.0) / M_I_G)
        cs_ei = np.sqrt(ERG_PER_EV * np.maximum(Te + Ti, 0.0) / M_I_G)
        print(f"\n  --- {r.tag} --- plateau {PLATEAU[0]}-{PLATEAU[1]} ms, "
              "last 6 column cells then the end block")
        print(f"      {'z[cm]':>9} {'role':>10} {'u[km/s]':>10} "
              f"{'Te[eV]':>8} {'Ti[eV]':>8} {'cs(Te)':>9} {'cs(Te+Ti)':>10} "
              f"{'u/cs(Te)':>9} {'u/cs(TeTi)':>11} {'n[cm^-3]':>12}")
        colidx = np.where(r.col & ~r.end)[0][-6:]
        for i in list(colidx) + list(np.where(r.end)[0]):
            print(f"      {r.z[i]:9.2f} {r.role[i]:>10} {u[i] / 1e5:10.4f} "
                  f"{Te[i]:8.4f} {Ti[i]:8.4f} {cs_e[i] / 1e5:9.4f} "
                  f"{cs_ei[i] / 1e5:10.4f} {u[i] / max(cs_e[i], 1e-300):9.4f} "
                  f"{u[i] / max(cs_ei[i], 1e-300):11.4f} {n[i]:12.4e}")

    print("\n  ARM vs CTL at the end block (plateau means):")
    sla, slc = arm.wsl(*PLATEAU), ctl.wsl(*PLATEAU)
    ua, uc = arm.pmean("u", sla), ctl.pmean("u", slc)
    Tea, Tec = arm.pmean("Te", sla), ctl.pmean("Te", slc)
    Tia, Tic = arm.pmean("Ti", sla), ctl.pmean("Ti", slc)
    print(f"      {'z[cm]':>9} {'u arm':>10} {'u ctl':>10} {'delta':>10} "
          f"{'ratio':>9} {'M arm':>8} {'M ctl':>8}  (u km/s, M = u/cs(Te+Ti))")
    for i in np.where(arm.end)[0]:
        ca = np.sqrt(ERG_PER_EV * max(Tea[i] + Tia[i], 0.0) / M_I_G)
        cc = np.sqrt(ERG_PER_EV * max(Tec[i] + Tic[i], 0.0) / M_I_G)
        print(f"      {arm.z[i]:9.2f} {ua[i] / 1e5:10.4f} {uc[i] / 1e5:10.4f} "
              f"{(ua[i] - uc[i]) / 1e5:+10.4f} "
              f"{ua[i] / uc[i] if uc[i] != 0 else float('nan'):9.4f} "
              f"{ua[i] / max(ca, 1e-300):8.4f} {uc[i] / max(cc, 1e-300):8.4f}")

    print("\n  TIME EVOLUTION of the terminal-cell ion speed [km/s] and Mach:")
    print(f"      {'t[ms]':>7} {'u arm':>10} {'M arm':>8} {'u ctl':>10} "
          f"{'M ctl':>8}")
    ia = np.where(arm.end)[0][-1]
    ic = np.where(ctl.end)[0][-1]
    for t in SNAPS_MS:
        ja, jc = arm.isnap(t), ctl.isnap(t)
        uaa = arm.h5["u"][ja, ia]
        ucc = ctl.h5["u"][jc, ic]
        caa = np.sqrt(ERG_PER_EV * max(arm.h5["Te"][ja, ia]
                                       + arm.h5["Ti"][ja, ia], 0.0) / M_I_G)
        ccc = np.sqrt(ERG_PER_EV * max(ctl.h5["Te"][jc, ic]
                                       + ctl.h5["Ti"][jc, ic], 0.0) / M_I_G)
        print(f"      {t:7.2f} {uaa / 1e5:10.4f} "
              f"{uaa / max(caa, 1e-300):8.4f} {ucc / 1e5:10.4f} "
              f"{ucc / max(ccc, 1e-300):8.4f}")


# --------------------------------------------------------------------------
def section_flux(arm, ctl):
    banner("D. ION PARTICLE FLUX through the end block: Gamma = n*u*A [s^-1]\n"
           "   Distinguishes ACCELERATION (Gamma held while u rises) from mere\n"
           "   DILUTION (u rises only because n falls at fixed throughput).")
    for r in (arm, ctl):
        sl = r.wsl(*PLATEAU)
        u = r.pmean("u", sl)
        n = r.pmean("n", sl)
        G = n * u * r.A
        print(f"\n  --- {r.tag} --- plateau means")
        print(f"      {'z[cm]':>9} {'role':>10} {'Gamma[s^-1]':>14} "
              f"{'n[cm^-3]':>12} {'u[km/s]':>10} {'A[cm^2]':>10}")
        colidx = np.where(r.col & ~r.end)[0][-4:]
        for i in list(colidx) + list(np.where(r.end)[0]):
            print(f"      {r.z[i]:9.2f} {r.role[i]:>10} {G[i]:14.4e} "
                  f"{n[i]:12.4e} {u[i] / 1e5:10.4f} {r.A[i]:10.2f}")
    sla, slc = arm.wsl(*PLATEAU), ctl.wsl(*PLATEAU)
    Ga = (arm.pmean("n", sla) * arm.pmean("u", sla) * arm.A)
    Gc = (ctl.pmean("n", slc) * ctl.pmean("u", slc) * ctl.A)
    ia = np.where(arm.end)[0][0]
    ic = np.where(ctl.end)[0][0]
    print(f"\n  Gamma at the block ENTRANCE  arm {Ga[ia]:.4e}  "
          f"ctl {Gc[ic]:.4e}  ratio {Ga[ia] / Gc[ic]:.4f}")
    ja = np.where(arm.end)[0][-1]
    jc = np.where(ctl.end)[0][-1]
    print(f"  Gamma at the block TERMINUS  arm {Ga[ja]:.4e}  "
          f"ctl {Gc[jc]:.4e}  ratio {Ga[ja] / Gc[jc]:.4f}")
    print(f"  plasma inventory in the end block  arm "
          f"{float(arm.plasma_inventory(arm.end)[arm.wsl(*PLATEAU)].mean()):.4e}"
          f"  ctl "
          f"{float(ctl.plasma_inventory(ctl.end)[ctl.wsl(*PLATEAU)].mean()):.4e}")


# --------------------------------------------------------------------------
def section_pile(arm, ctl):
    banner("E. THE END PILE: neutral inventory in the FAR region and the end\n"
           "   block through time. Does the flare EVACUATE it, SHRINK it, or\n"
           "   MOVE it? (fa2 control: FAR share 50.5 % at 19 ms.)")
    for r in (arm, ctl):
        r._far = r.neutral_inventory(r.zmask(*FAR_Z))
        r._tot = r.neutral_inventory(r.col)
        r._dom = r.neutral_inventory(None)
        r._end = r.neutral_inventory(r.end)
    print("  FAR inventory and share. PRIMARY share = FAR / WHOLE DOMAIN")
    print("  (the convention that reproduces the fa2 control's 50.5 % at")
    print("  19 ms); the column-only denominator follows it.")
    print(f"      {'t[ms]':>7} {'FAR arm':>14} {'FAR ctl':>14} "
          f"{'sh/dom arm':>11} {'sh/dom ctl':>11} {'sh/col arm':>11} "
          f"{'sh/col ctl':>11} {'endblk arm':>13} {'endblk ctl':>13}")
    for t in sorted(SNAPS_MS + (19.0,)):
        ja, jc = arm.isnap(t), ctl.isnap(t)
        print(f"      {t:7.2f} {arm._far[ja]:14.4e} {ctl._far[jc]:14.4e} "
              f"{100 * arm._far[ja] / arm._dom[ja]:10.1f}% "
              f"{100 * ctl._far[jc] / ctl._dom[jc]:10.1f}% "
              f"{100 * arm._far[ja] / arm._tot[ja]:10.1f}% "
              f"{100 * ctl._far[jc] / ctl._tot[jc]:10.1f}% "
              f"{arm._end[ja]:13.4e} {ctl._end[jc]:13.4e}")
    print("\n  PEAK column-zone nn THROUGH TIME (the pile moves inward, so a")
    print("  single-frame quote is a convention; the series is the fact):")
    print(f"      {'t[ms]':>7} {'peak arm':>13} {'z arm':>9} "
          f"{'peak ctl':>13} {'z ctl':>9} {'ratio':>8} {'dz[cm]':>9}")
    for t in sorted(set(SNAPS_MS) | {17.0, 18.0, 19.0, 20.0}):
        ja, jc = arm.isnap(t), ctl.isnap(t)
        nna_, nnc_ = arm.h5["nn"][ja], ctl.h5["nn"][jc]
        ia = np.where(arm.col)[0][int(np.argmax(nna_[arm.col]))]
        ic = np.where(ctl.col)[0][int(np.argmax(nnc_[ctl.col]))]
        print(f"      {t:7.2f} {nna_[ia]:13.4e} {arm.z[ia]:9.1f} "
              f"{nnc_[ic]:13.4e} {ctl.z[ic]:9.1f} "
              f"{nna_[ia] / nnc_[ic]:8.4f} {arm.z[ia] - ctl.z[ic]:+9.1f}")
    print("\n  PEAK column-zone nn (plateau-mean profile):")
    for r in (arm, ctl):
        sl = r.wsl(*PLATEAU)
        nn = r.pmean("nn", sl)
        m = r.col
        i = np.where(m)[0][int(np.argmax(nn[m]))]
        print(f"      {r.tag:<6} peak nn = {nn[i]:.4e} cm^-3 at z = "
              f"{r.z[i]:.1f} cm")
    print("\n  nn(z) plateau-mean profile over the far half [cm^-3]:")
    sla, slc = arm.wsl(*PLATEAU), ctl.wsl(*PLATEAU)
    nna, nnc = arm.pmean("nn", sla), ctl.pmean("nn", slc)
    sel = np.where(arm.zmask(1000.0, 2000.0))[0][::3]
    print(f"      {'z[cm]':>9} {'nn arm':>13} {'nn ctl':>13} {'ratio':>9}")
    for i in sel:
        print(f"      {arm.z[i]:9.1f} {nna[i]:13.4e} {nnc[i]:13.4e} "
              f"{nna[i] / nnc[i] if nnc[i] else float('nan'):9.4f}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ctl", required=True)
    a = ap.parse_args()
    arm = Run(a.arm, "fa3")
    ctl = Run(a.ctl, "fa2")
    print("=" * 104)
    print("fa3 END-FLARE EXPANSION PROBE -- flare geometry, ion flow, end pile")
    print("  arm : " + a.arm)
    print("  ctl : " + a.ctl + "   (the fa2 arm = flare-OFF control)")
    print("  ONE config delta: end_expansion_plasma_radius_cm 15.0 -> 25.0")
    print("  LOWER BOUND: no mirror force in the model; area expansion only.")
    print("=" * 104)
    section_geometry(arm, ctl)
    section_base(arm, ctl)
    section_ionflow(arm, ctl)
    section_flux(arm, ctl)
    section_pile(arm, ctl)
    arm.close()
    ctl.close()


if __name__ == "__main__":
    main()
