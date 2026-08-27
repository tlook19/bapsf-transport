"""fa4j -- THE JET-SPECIFIC READS (cathode_neutral_jet + surface debit ON).

READ-ONLY. Opens saved artifacts, computes, prints. Solves nothing, writes
nothing back, fits nothing, tunes nothing.

WHAT THIS ADDS over fa4_twochannel.py (which stays the primary two-channel
instrument and is run unchanged): the hot-diag merge (9e5aae6) persists
``hot_Ei_recx`` and ``hot_Ei_ionization`` as separate per-cell datasets.
hot_neutrals.py:426 builds the rhs row as ``Ei = dEi_recx + dEi_ion`` and
:472-473 saves those two addends, so the SUM that fa4 was forced to report is
now SEPARABLE. fa4_twochannel.py predates the merge and reads only the summed
rhs row, so the separation is done here instead of by editing that instrument.

UNITS: per-cell rhs/diagnostic energy rows are CGS erg cm^-3 s^-1. Integrate
on the plasma-column volume and divide by 1e7 for W. Two-zone runs put En on
V_col (fa4_twochannel.py:38-42), which every arm here is.

Reads covered: 1 (CX-recycling power, separated), 2 (the cathode-adjacent
seam Tn), 3 (topology-mask deleted deposit, absolute), 4 (the cathode thermal
response under the debit).

Usage (PYTHONPATH=<checkout>/cablp):
    fa4j_jetread.py ARM.h5 [CTL.h5 [ALT.h5]]
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
WINDOWS = [(2.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 19.5)]
WALL_EV = 300.0 * 1.380649e-16 / 1.602176634e-12   # ~0.02585 eV
SEAM_DISCLOSED_EV = 11.0   # the pass-2 review's bookkeeping figure, at R_N=R_E=0.5


def rule(s):
    print("\n" + "=" * 110)
    print(s)
    print("=" * 110)


class Run:
    def __init__(self, path, tag):
        self.tag, self.path = tag, path
        self.h5 = h5py.File(path, "r")
        g = self.h5["geometry"]
        self.z = g["z_cm"][:]
        self.V_col = g["plasma_volume_cm3"][:]
        self.V_neu = g["neutral_volume_cm3"][:]
        self.V_ann = self.V_neu - self.V_col
        self.roles = np.array(
            [r.decode() if isinstance(r, bytes) else str(r)
             for r in g["cell_role"][:]]
        )
        self.pa = np.asarray(g["plasma_active"][:], dtype=bool) \
            if "plasma_active" in g else np.ones_like(self.z, dtype=bool)
        self.t_ms = self.h5["time"][:] * 1e3
        self.params = json.loads(self.h5.attrs["params_json"])
        self.flags = json.loads(self.h5.attrs["flags_json"])
        self.two_zone = "nn_a" in self.h5

    def close(self):
        self.h5.close()

    def has(self, n):
        return n in self.h5

    def zmask(self, lo, hi):
        return (self.z >= lo) & (self.z <= hi)

    def wsl(self, lo, hi):
        i0 = int(np.searchsorted(self.t_ms, lo))
        i1 = int(np.searchsorted(self.t_ms, hi))
        return slice(i0, min(i1 + 1, self.t_ms.size))

    def isnap(self, t):
        return int(np.argmin(np.abs(self.t_ms - t)))

    def dset_int_kW(self, name, lo, hi, mask):
        """Time-mean, volume-integrated energy density row -> kW."""
        if name not in self.h5:
            return None
        a = self.h5[name][self.wsl(lo, hi)][:, mask]
        return float((a @ self.V_col[mask]).mean() / ERG_PER_J / 1e3)

    def rhs_int_kW(self, term, group, lo, hi, mask):
        g = self.h5.get("rhs_terms")
        if g is None or term not in g or group not in g[term]:
            return None
        vol = self.V_ann if group.endswith("_a") else self.V_col
        a = g[term][group][self.wsl(lo, hi)][:, mask]
        return float((a @ vol[mask]).mean() / ERG_PER_J / 1e3)

    def cd(self, name):
        g = self.h5.get("cathode_diagnostics")
        if g is None or name not in g:
            return None
        return np.asarray(g[name][:], dtype=float)

    def cd_mean(self, name, lo, hi):
        v = self.cd(name)
        if v is None:
            return None
        tt = self.t_ms
        if v.shape[0] != tt.shape[0]:
            tt = np.linspace(self.t_ms[0], self.t_ms[-1], v.shape[0])
        m = (tt >= lo) & (tt <= hi)
        return float(np.nanmean(v[m])) if np.any(m) else None


def fmt(v, w=14, p=5):
    return f"{'--':>{w}}" if v is None else f"{v:{w}.{p}f}"


# ---------------------------------------------------------------- read 1
def read1(arm, ctl):
    rule("READ 1. THE CX-RECYCLING POWER, SEPARATED\n"
         "   hot_Ei_recx      = the NONLOCAL CX-recycling power (ions give up a\n"
         "                      fast atom and draw a cold replacement)\n"
         "   hot_Ei_ionization= the thermal deposit of in-flight ionization\n"
         "   Both are per-cell erg cm^-3 s^-1 and sum BITWISE to\n"
         "   rhs_terms/neutral_hot_channel/Ei (hot_neutrals.py:426,472-473).")
    if not arm.has("hot_Ei_recx"):
        print("  !! ARM carries no hot_Ei_recx -- artifact predates the "
              "hot-diag merge. Not separable.")
        return
    print("\n  1a. WHOLE COLUMN, per window [kW]")
    print(f"  {'window[ms]':>14}{'recx':>14}{'ionization':>14}"
          f"{'sum':>14}{'rhs Ei':>14}{'|sum-rhs|':>13}")
    allm = np.ones_like(arm.z, dtype=bool)
    for lo, hi in WINDOWS:
        a = arm.dset_int_kW("hot_Ei_recx", lo, hi, allm)
        b = arm.dset_int_kW("hot_Ei_ionization", lo, hi, allm)
        r = arm.rhs_int_kW("neutral_hot_channel", "Ei", lo, hi, allm)
        s = None if (a is None or b is None) else a + b
        d = None if (s is None or r is None) else abs(s - r)
        print(f"  {lo:6.1f}-{hi:6.1f}{fmt(a)}{fmt(b)}{fmt(s)}{fmt(r)}"
              f"{'--' if d is None else f'{d:13.3e}'}")
    print("\n  1b. PER REGION, plateau 15.0-19.5 ms [kW]")
    print(f"  {'row':<26}" + "".join(f"{r[0]:>18}" for r in REGIONS))
    for nm, key in (("hot_Ei_recx", "hot_Ei_recx"),
                    ("hot_Ei_ionization", "hot_Ei_ionization")):
        cells = []
        for _, a_, b_ in REGIONS:
            v = arm.dset_int_kW(key, 15.0, 19.5, arm.zmask(a_, b_))
            cells.append(fmt(v, 18))
        print(f"  {nm:<26}" + "".join(cells))
    for nm, grp in (("rhs hot_channel.Ei", "Ei"), ("rhs hot_channel.Ee", "Ee"),
                    ("rhs hot_channel.En", "En")):
        cells = []
        for _, a_, b_ in REGIONS:
            v = arm.rhs_int_kW("neutral_hot_channel", grp, 15.0, 19.5,
                               arm.zmask(a_, b_))
            cells.append(fmt(v, 18))
        print(f"  {nm:<26}" + "".join(cells))
    print("\n  1c. THROUGH TIME, recx only, per region [kW]")
    print(f"  {'window[ms]':>14}" + "".join(f"{r[0]:>18}" for r in REGIONS))
    for lo, hi in WINDOWS:
        cells = [fmt(arm.dset_int_kW("hot_Ei_recx", lo, hi, arm.zmask(a_, b_)),
                     18) for _, a_, b_ in REGIONS]
        print(f"  {lo:6.1f}-{hi:6.1f}" + "".join(cells))
    print("\n  1d. THE AVAILABLE-POWER CONTEXT (not a gate -- a scale check)")
    R_E = float(arm.params.get("cathode_jet_R_E", 0.0))
    for tag, rn in (("ARM", arm), ("CTL", ctl)):
        if rn is None:
            continue
        p_i = rn.cd_mean("source_P_cathode_i", 15.0, 19.5)
        if p_i is None:
            continue
        print(f"      {tag:<4} source_P_cathode_i = {p_i / 1e3:10.3f} kW  "
              f"R_E={R_E:.2f} -> R_E*P = {R_E * p_i / 1e3:9.3f} kW")
    tot = arm.dset_int_kW("hot_Ei_recx", 15.0, 19.5, allm)
    p_i = arm.cd_mean("source_P_cathode_i", 15.0, 19.5)
    if tot is not None and p_i:
        print(f"      measured plateau recx / (R_E*P_cathode_i) = "
              f"{tot / (R_E * p_i / 1e3):.4f}")


# ---------------------------------------------------------------- read 2
def read2(arm, ctl):
    rule("READ 2. THE SEAM WATCH -- cathode-adjacent cell Tn\n"
         f"   Disclosed pass-2 bookkeeping figure: ~{SEAM_DISCLOSED_EV:.0f} eV,\n"
         "   measured at R_N = R_E = 0.5.  THIS arm's boxed values are\n"
         "   R_N = 0.5 / R_E = 0.2, so the disclosure is an UPPER-side\n"
         "   reference, not a like-for-like prediction.")
    if not arm.has("Tn"):
        print("  ARM carries no Tn field.")
        return
    cath = np.where(arm.roles == "cathode")[0]
    print(f"  cathode-role cells: idx={list(cath)} z={list(arm.z[cath])}")
    print(f"  R_N={arm.params.get('cathode_jet_R_N')} "
          f"R_E={arm.params.get('cathode_jet_R_E')} "
          f"jet={arm.params.get('cathode_neutral_jet')} "
          f"debit={arm.params.get('cathode_jet_surface_debit')}")
    idxs = sorted(set(list(cath[:2]) + [0, 1, 2, 3, 4]))
    print(f"\n  2a. Tn [eV] at the seam cells through time  "
          f"(wall reference {WALL_EV:.5f} eV)")
    hdr = "".join(f"{'z=' + f'{arm.z[i]:.1f}':>14}" for i in idxs)
    print(f"  {'t[ms]':>8}{hdr}")
    for t in (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 19.0, 19.5):
        i = arm.isnap(t)
        row = "".join(f"{float(arm.h5['Tn'][i][j]):14.4f}" for j in idxs)
        print(f"  {arm.t_ms[i]:8.3f}{row}")
    print("\n  2b. PLATEAU (15.0-19.5 ms) seam Tn, ARM vs CTL")
    print(f"  {'z[cm]':>10}{'ARM Tn[eV]':>14}{'CTL Tn[eV]':>14}"
          f"{'delta':>12}{'ARM/disclosed':>16}")
    for j in idxs:
        sl = arm.wsl(15.0, 19.5)
        a = float(np.mean(arm.h5["Tn"][sl][:, j]))
        c = None
        if ctl is not None and ctl.has("Tn"):
            c = float(np.mean(ctl.h5["Tn"][ctl.wsl(15.0, 19.5)][:, j]))
        print(f"  {arm.z[j]:10.1f}{a:14.4f}"
              f"{'--' if c is None else f'{c:14.4f}'}"
              f"{'--' if c is None else f'{a - c:12.4f}'}"
              f"{a / SEAM_DISCLOSED_EV:16.4f}")
    sl = arm.wsl(15.0, 19.5)
    peak = float(np.max(arm.h5["Tn"][sl]))
    jpk = int(np.argmax(arm.h5["Tn"][sl].mean(axis=0)))
    print(f"\n      plateau MAX Tn anywhere in the column = {peak:.4f} eV")
    print(f"      plateau-mean Tn peaks at z={arm.z[jpk]:.1f} cm "
          f"(role={arm.roles[jpk]})")
    print(f"      VERDICT vs the ~{SEAM_DISCLOSED_EV:.0f} eV disclosure: "
          f"seam max is {peak / SEAM_DISCLOSED_EV:.3f}x it")


# ---------------------------------------------------------------- read 3
def read3(arm):
    rule("READ 3. THE TOPOLOGY-MASK SEAM (the fnb3 root cause)\n"
         "   _apply_active_plasma_topology zeros the hot rows on plasma-dead\n"
         "   cells AFTER the ballistic spread, so a deposit landing there is\n"
         "   DELETED. The jet multiplies the cathode cell's hot birth rate,\n"
         "   so the absolute deleted rate is the quantity of interest.")
    print(f"  plasma-dead cells: idx={list(np.where(~arm.pa)[0])} "
          f"z={list(arm.z[~arm.pa])} roles={list(arm.roles[~arm.pa])}")
    allm = np.ones_like(arm.z, dtype=bool)
    print("\n  3a. THE BRANCHING CLOSURE D/A on the saved rows "
          "(fa4 published 0.9697)")
    g = arm.h5.get("rhs_terms")
    if g is not None and "neutral_cx_channel" in g:
        cx_d = g["neutral_cx_channel"]["nn"]
        wl_d = g["neutral_hot_channel"]["nn_a"]
        io_d = g["neutral_hot_channel"]["n"]
        src = arm.z <= 100.0
        for lo, hi in ((15.0, 19.5), (15.0, float(arm.t_ms[-1]))):
            sl = arm.wsl(lo, hi)
            cx, wl, io_ = -cx_d[sl], wl_d[sl], io_d[sl]
            sl = slice(None)
            A = float((cx[sl] @ arm.V_col).mean())
            D = float((wl[sl] @ arm.V_ann).mean()
                      + (io_[sl] @ arm.V_col).mean())
            A_s = float((cx[sl][:, src] @ arm.V_col[src]).mean())
            D_s = float((wl[sl][:, src] @ arm.V_ann[src]).mean()
                        + (io_[sl][:, src] @ arm.V_col[src]).mean())
            print(f"      window {lo:5.1f}-{hi:5.1f} ms: A={A:.6e} "
                  f"D={D:.6e}  D/A={D / A:.6f}  deficit={A - D:.6e}")
            print(f"          SOURCE z<=100 alone: deficit={A_s - D_s:.6e} = "
                  f"{100 * (A_s - D_s) / max(A - D, 1e-300):.2f}% of the "
                  f"column deficit")
    print("\n  3b. hot_end_fraction (the end-plane fold), from the artifact")
    if arm.has("hot_end_fraction"):
        ef = arm.h5["hot_end_fraction"][arm.isnap(17.0)]
        ef = np.asarray(ef, dtype=float)
        print(f"      mean={float(np.mean(ef)):.6f} max={float(np.max(ef)):.6f}"
              f" at z={arm.z[int(np.argmax(ef))]:.1f}")
        print(f"      at the cathode cell(s): "
              f"{[f'{float(ef[i]):.6f}' for i in np.where(arm.roles == 'cathode')[0]]}")
    else:
        print("      hot_end_fraction absent from this artifact.")
    print("\n  3c. THE CATHODE CELL'S HOT BIRTH RATE (what the jet multiplies)")
    if arm.has("hot_births"):
        for lo, hi in WINDOWS:
            sl = arm.wsl(lo, hi)
            hb = arm.h5["hot_births"][sl]
            tot = float((hb @ arm.V_col).mean())
            cath = np.where(arm.roles == "cathode")[0]
            cthr = float((hb[:, cath] @ arm.V_col[cath]).mean())
            print(f"      {lo:5.1f}-{hi:5.1f} ms: column {tot:.6e} 1/s | "
                  f"cathode cell {cthr:.6e} 1/s ({100 * cthr / max(tot, 1e-300):.3f}%)")
    else:
        print("      hot_births absent from this artifact.")
    print("\n  3d. THE DELETED DEPOSIT, absolute [1/s]  (deficit = A - D above)")
    print("      The kernel-side attribution (which live birth cells send how\n"
          "      much mass onto dead cells) is produced by fnb3_closure.py on\n"
          "      this same artifact -- see fa4j_fnb3.txt. It is geometry-only\n"
          "      and so is identical between fa4 and fa4j; what differs, and\n"
          "      is reported above, is the RATE it multiplies.")


# ---------------------------------------------------------------- read 4
def read4(arm, ctl, alt):
    rule("READ 4. THE CATHODE THERMAL RESPONSE UNDER THE DEBIT\n"
         "   fa4j is the FIRST debit-ON arm. solver.py:8893-8895 sets\n"
         "   _cathode_surface_ion_retention = 1 - R_E when the debit is on,\n"
         "   so the surface keeps only (1-R_E) of P_cathode_i. faj ran\n"
         "   debit-OFF (retention 1.0), fa4 has no jet at all.")
    runs = [r for r in (arm, ctl, alt) if r is not None]
    print(f"  {'key':<34}" + "".join(f"{r.tag:>20}" for r in runs))
    for k in ("cathode_neutral_jet", "cathode_jet_surface_debit",
              "cathode_jet_R_N", "cathode_jet_R_E"):
        print(f"  {k:<34}" + "".join(
            f"{str(r.params.get(k, '<absent>')):>20}" for r in runs))
    for k in ("neutral_energy", "neutral_momentum"):
        print(f"  flags {k:<28}" + "".join(
            f"{str(r.flags.get(k, '<absent>')):>20}" for r in runs))
    print("\n  4a. PLATEAU (15.0-19.5 ms) CATHODE + DRIVE SCALARS")
    rows = [
        ("T_s_surface [K]", "T_s_surface", 1.0),
        ("source_P_cathode_i [kW]", "source_P_cathode_i", 1e-3),
        ("source_P_cathode_e [kW]", "source_P_cathode_e", 1e-3),
        ("source_P_net [kW]", "source_P_net", 1e-3),
        ("source_P_loss [kW]", "source_P_loss", 1e-3),
        ("source_I_eth [A]", "source_I_eth", 1.0),
        ("source_I_tot [A]", "source_I_tot", 1.0),
        ("source_phi_c [V]", "source_phi_c", 1.0),
        ("circuit_I_loop [A]", "circuit_I_loop", 1.0),
        ("circuit_V_dis_dt_int [V]", "circuit_V_dis_dt_integral", 1.0),
        ("warming_E_ion_J [J]", "warming_E_ion_J", 1.0),
        ("warming_E_emis_J [J]", "warming_E_emis_J", 1.0),
        ("warming_E_cond_J [J]", "warming_E_cond_J", 1.0),
        ("warming_E_rad_J [J]", "warming_E_rad_J", 1.0),
        ("warming_E_heater_J [J]", "warming_E_heater_J", 1.0),
    ]
    print(f"  {'quantity':<28}" + "".join(f"{r.tag:>20}" for r in runs)
          + f"{'ARM/CTL':>12}")
    for lbl, key, sc in rows:
        vs = [r.cd_mean(key, 15.0, 19.5) for r in runs]
        vs = [None if v is None else v * sc for v in vs]
        rat = ("--" if (len(vs) < 2 or vs[0] is None or not vs[1])
               else f"{vs[0] / vs[1]:.4f}")
        print(f"  {lbl:<28}"
              + "".join(f"{'--':>20}" if v is None else f"{v:20.5f}" for v in vs)
              + f"{rat:>12}")
    print("\n  4b. T_s TRAJECTORY [K]")
    print(f"  {'t[ms]':>8}" + "".join(f"{r.tag:>20}" for r in runs))
    for t in (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 19.0, 19.5, 25.0):
        cells = []
        for r in runs:
            v = r.cd("T_s_surface")
            if v is None:
                cells.append(f"{'--':>20}")
                continue
            tt = (r.t_ms if v.shape[0] == r.t_ms.shape[0]
                  else np.linspace(r.t_ms[0], r.t_ms[-1], v.shape[0]))
            cells.append(f"{float(v[int(np.argmin(np.abs(tt - t)))]):20.4f}")
        print(f"  {t:8.2f}" + "".join(cells))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    arm = Run(sys.argv[1], "ARM(fa4j)")
    ctl = Run(sys.argv[2], "CTL(fa4)") if len(sys.argv) > 2 else None
    alt = Run(sys.argv[3], "ALT(faj)") if len(sys.argv) > 3 else None
    rule("fa4j JET READS -- artifacts")
    for r in (arm, ctl, alt):
        if r is None:
            continue
        print(f"  {r.tag:<12} {r.path}")
        print(f"      run_status={r.h5.attrs.get('run_status')} "
              f"steps={r.h5.attrs.get('steps')} "
              f"two_zone={r.two_zone} "
              f"kernels={r.h5.attrs.get('compiled_kernels')}")
    read1(arm, ctl)
    read2(arm, ctl)
    read3(arm)
    read4(arm, ctl, alt)
    for r in (arm, ctl, alt):
        if r is not None:
            r.close()


main()
