"""fa2 -- NBL MOMENTUM PREVIEW read instrument: ARM vs A1 vs REF.

READ-ONLY. Pure h5 reads of three saved runs; no solver is constructed, no
config is rebuilt, nothing in the repo is mutated. Conventions are INHERITED
from the validated readers so this read cannot drift from the numbers the A1
and faq reads produced:

  * volume/inventory conventions  <- scripts/fa1_diag.py
        V_col = geometry/plasma_volume_cm3                (nn lives here)
        V_ann = neutral_volume_cm3 - plasma_volume_cm3    (nn_a lives here)
  * region windows                <- scripts/fa1_diag.py
        SOURCE z<=100 | 100<z<790 | MID 790-1045 | FAR 1045-1900
  * Ee/Ei ledger sign convention  <- faq_ledger.txt / faq_quenchfront.txt
        volume-integrated rhs_terms, + = source, - = sink, WATTS
  * S_net quench front            <- faq_quenchfront.txt section 1
        S_net = (ionization_birth + beam_ionization_birth
                 + gas_puff_local_ionization + recombination_rad_loss
                 + recombination_3b_loss).n     [cm^-3 s^-1]
  * walker axial span             <- faq_quenchfront.txt section 3
        cathode_diagnostics/beam_heat_anomalous_W per z-band

The faq probe scripts themselves lived in an ephemeral session scratchpad and
no longer exist (faq_run.cmd records them as "<scratchpad>/..."), so the
methods above are REBUILT from their recorded output format. The beam
deposition-depth definition is DECLARED here rather than inherited, because
the vanished probe's exact window is not recoverable -- see DEPOSITION note.

Usage:
    python scripts/fa2_diag.py --arm fa2_arm.h5 --a1 fa1_arm.h5 --ref sp1_ref.h5
"""

import argparse
import json

import h5py
import numpy as np

SOURCE_Z = (-1e9, 100.0)
BAND_Z = (100.0, 790.0)
MID_Z = (790.0, 1045.0)
FAR_Z = (1045.0, 1900.0)

REGIONS = (
    ("SOURCE z<=100", SOURCE_Z),
    ("100<z<790", BAND_Z),
    ("MID 790-1045", MID_Z),
    ("FAR 1045-1900", FAR_Z),
)

WINDOWS = ((0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0),
           (10.0, 15.0), (15.0, 19.5))

PORTS = ((11, 470.0), (21, 790.0), (29, 1045.0), (41, 1429.0), (50, 1716.0))

S_NET_TERMS = ("ionization_birth", "beam_ionization_birth",
               "gas_puff_local_ionization", "recombination_rad_loss",
               "recombination_3b_loss")

WALKER_BANDS = ((0.0, 100.0), (100.0, 200.0), (200.0, 400.0), (400.0, 600.0),
                (600.0, 790.0), (790.0, 1045.0), (1045.0, 1400.0),
                (1400.0, 1900.0))

# Helium mass [g] for the neutral thermal speed / Mach number.
# SUPERSEDED 2026-08-21: the unified helium mass is cablp.constants
# .m_He_cgs = 6.6464790809e-24 g (Ar(4He)*u, CODATA 2022). The literal
# below is 0.90 ppm low and is left AS A RECORD of what this dated script ran.
M_HE_G = 6.6464731e-24
K_B_ERG = 1.380649e-16
# The saved rhs energy terms are CGS: erg cm^-3 s^-1. 1 W = 1e7 erg/s.
ERG_PER_J = 1.0e7


def _s(x):
    if isinstance(x, bytes):
        return x.decode()
    s = str(x)
    return s[2:-1] if s.startswith("b'") and s.endswith("'") else s


class Run:
    def __init__(self, path, tag):
        self.path = path
        self.tag = tag
        self.h5 = h5py.File(path, "r")
        g = self.h5["geometry"]
        self.z = g["z_cm"][:]
        self.V_col = g["plasma_volume_cm3"][:]
        self.V_ann = g["neutral_volume_cm3"][:] - self.V_col
        self.active = np.asarray(g["plasma_active"], bool)
        self.t = self.h5["time"][:]
        self.t_ms = self.t * 1e3
        self.params = json.loads(self.h5.attrs["params_json"])
        self.flags = json.loads(self.h5.attrs["flags_json"])
        self.col = self.z >= 0.0

    def close(self):
        self.h5.close()

    def zmask(self, lo, hi):
        return (self.z >= lo) & (self.z <= hi)

    def wsl(self, lo_ms, hi_ms):
        i0 = int(np.searchsorted(self.t_ms, lo_ms))
        i1 = int(np.searchsorted(self.t_ms, hi_ms))
        return slice(i0, min(i1 + 1, self.t_ms.size))

    def isnap(self, t_ms):
        return int(np.argmin(np.abs(self.t_ms - t_ms)))

    def neutral_inventory(self, mask=None):
        m = np.ones_like(self.z, dtype=bool) if mask is None else mask
        return (self.h5["nn"][:, m] @ self.V_col[m]
                + self.h5["nn_a"][:, m] @ self.V_ann[m])

    def term_W(self, group, term, mask=None):
        """Volume-integrated power of one rhs term in one energy field [W].

        The saved rhs energy terms are CGS volumetric rates (erg cm^-3 s^-1),
        so the volume integral is erg/s and the /ERG_PER_J conversion to WATTS
        is mandatory. Verified against faq_increment.txt section B, which
        this reproduces to 4 significant figures on every window once the
        conversion is applied (that agreement is also what identifies the
        band the A1 read called the flipping "source band" as 100<z<790).
        """
        m = np.ones_like(self.z, dtype=bool) if mask is None else mask
        grp = self.h5["rhs_terms"]
        if term not in grp or group not in grp[term]:
            return None
        return (grp[term][group][:, m] @ self.V_col[m]) / ERG_PER_J

    def term_rate(self, term, channel="n", mask=None):
        m = np.ones_like(self.z, dtype=bool) if mask is None else mask
        grp = self.h5["rhs_terms"]
        if term not in grp or channel not in grp[term]:
            return None
        vol = self.V_col[m] if channel in ("n", "nn") else self.V_ann[m]
        return grp[term][channel][:, m] @ vol

    def s_net(self, isamp):
        """Per-cell net volumetric ionization at one sample [cm^-3 s^-1]."""
        out = np.zeros_like(self.z)
        for term in S_NET_TERMS:
            grp = self.h5["rhs_terms"]
            if term in grp and "n" in grp[term]:
                out = out + grp[term]["n"][isamp]
        return out

    def has(self, name):
        return name in self.h5


# --------------------------------------------------------------------------
def section_provenance(runs):
    print("\n" + "=" * 108)
    print("0. PROVENANCE AND CLOSURE READBACK  (read FROM the artifacts, not assumed)")
    print("=" * 108)
    keys_f = ("neutral_momentum", "neutral_two_zone", "ion_neutral_drag",
              "ion_neutral_moment_closure", "neutral_equilibration",
              "use_cached_neutral_seed")
    keys_p = ("neutral_momentum_radial", "ion_neutral_drag_model",
              "b_ion_neutral_drag", "neutral_mesh_accommodation",
              "S_gp", "nx", "C_R", "T_s", "cathode_Ts_base_K", "Tn_K",
              "tau_afterglow", "gas_puff_delivery_fraction",
              "heating_anomalous_transport", "max_steps_action")
    hdr = f"  {'key':<34}" + "".join(f"{r.tag:>22}" for r in runs)
    print(hdr)
    print("  -- flags --")
    for k in keys_f:
        print(f"  {k:<34}" + "".join(f"{str(r.flags.get(k, '<absent>')):>22}"
                                     for r in runs))
    print("  -- params --")
    for k in keys_p:
        print(f"  {k:<34}" + "".join(f"{str(r.params.get(k, '<absent>')):>22}"
                                     for r in runs))
    print("  -- artifact attrs --")
    for k in ("run_status", "steps", "final_time", "compiled_kernels",
              "t_prebreakdown_trigger", "t_breakdown_trigger"):
        vals = []
        for r in runs:
            v = r.h5.attrs.get(k, "<absent>")
            vals.append(_s(v) if isinstance(v, bytes) else str(v))
        print(f"  {k:<34}" + "".join(f"{v:>22}" for v in vals))
    print("  -- momentum-closure state fields present --")
    for name in ("u_n", "M_n", "M_n_a"):
        print(f"  {name:<34}"
              + "".join(f"{str(r.has(name)):>22}" for r in runs))


def section_base(runs):
    print("\n" + "=" * 108)
    print("1. EQUILIBRATED BASE (t = 0 saved frame): does the closure change the fill?")
    print("=" * 108)
    print("  The momentum closure MAY participate in the neutral-only")
    print("  equilibration; if it does, the equilibrated base differs from A1's.")
    print()
    base = {}
    for r in runs:
        nn0 = r.h5["nn"][0]
        nna0 = r.h5["nn_a"][0]
        inv = float(nn0 @ r.V_col + nna0 @ r.V_ann)
        base[r.tag] = (nn0, nna0, inv)
        print(f"  {r.tag:<12} seed inventory {inv:.6e} particles | "
              f"nn min {nn0.min():.4e} max {nn0.max():.4e} cm^-3")
    tags = [r.tag for r in runs]
    ref_inv = base[tags[-1]][2]
    print()
    for r in runs:
        print(f"  {r.tag:<12} / {tags[-1]:<10} inventory ratio = "
              f"{base[r.tag][2] / ref_inv:.6f}")
    print("  (A1 recorded exactly 1.73269x REF with identical shape.)")
    print()
    print("  -- SHAPE test: nn(z,0) normalised to its own inventory, vs REF --")
    print("  max |relative shape deviation| over the column, and per-region "
          "seed ratio")
    nnr = base[tags[-1]][0]
    for r in runs:
        nn0 = base[r.tag][0]
        sa = nn0 / nn0.sum()
        sr = nnr / nnr.sum()
        dev = np.max(np.abs(sa - sr) / np.where(sr > 0, sr, np.inf))
        parts = []
        for name, (lo, hi) in REGIONS:
            m = r.zmask(lo, hi)
            ia = float(nn0[m] @ r.V_col[m] + base[r.tag][1][m] @ r.V_ann[m])
            mr = runs[-1].zmask(lo, hi)
            ir = float(nnr[mr] @ runs[-1].V_col[mr]
                       + base[tags[-1]][1][mr] @ runs[-1].V_ann[mr])
            parts.append(f"{name.split()[0]}={ia / ir:.4f}")
        print(f"    {r.tag:<12} max shape dev {dev:9.3e}   " + "  ".join(parts))


def _depths(z, w, fracs=(0.5, 0.9)):
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return [float("nan")] * len(fracs)
    c = np.cumsum(w) / tot
    out = []
    for fr in fracs:
        i = int(np.searchsorted(c, fr))
        if i == 0:
            out.append(float(z[0]))
        elif i >= len(z):
            out.append(float(z[-1]))
        else:
            c0, c1 = c[i - 1], c[i]
            t = 0.0 if c1 == c0 else (fr - c0) / (c1 - c0)
            out.append(float(z[i - 1] + t * (z[i] - z[i - 1])))
    return out


def section_nbl(runs):
    print("\n" + "=" * 108)
    print("2. THE NEUTRAL BOUNDARY LAYER")
    print("=" * 108)
    print("\n  2a. NEAR-SOURCE nn(z) THROUGH TIME [cm^-3], column zone")
    zc = runs[0].z
    near = np.flatnonzero((zc >= 0.0) & (zc <= 300.0))
    show = near[:: max(1, len(near) // 12)]
    for t_ms in (1.0, 2.5, 5.0, 10.0, 15.0, 19.0):
        print(f"\n    --- t = {t_ms:.1f} ms ---")
        print("      " + f"{'z[cm]':>8}"
              + "".join(f"{r.tag:>14}" for r in runs)
              + f"{'ARM/A1':>9}{'ARM/REF':>9}")
        for i in show:
            vals = [r.h5["nn"][r.isnap(t_ms)][i] for r in runs]
            print(f"      {zc[i]:8.1f}"
                  + "".join(f"{v:14.4e}" for v in vals)
                  + f"{vals[0] / vals[1]:9.3f}{vals[0] / vals[2]:9.3f}")

    print("\n  2b. NBL THICKNESS: z at which column nn(z) falls to 1/e and 1/10")
    print("      of its own near-source (z<=100 cm) volume-mean, scanning outward.")
    for t_ms in (2.5, 5.0, 10.0, 15.0, 19.0):
        row = f"    t={t_ms:5.1f} ms  "
        for r in runs:
            i = r.isnap(t_ms)
            nn = r.h5["nn"][i]
            m = r.zmask(*SOURCE_Z) & r.col
            n0 = float(nn[m] @ r.V_col[m] / r.V_col[m].sum())
            cz = r.z[r.col]
            cn = nn[r.col]
            out = []
            for fr in (1 / np.e, 0.1):
                hit = np.flatnonzero(cn <= fr * n0)
                out.append(cz[hit[0]] if hit.size else np.nan)
            row += f"| {r.tag} n0={n0:.3e} 1/e@{out[0]:7.1f} 1/10@{out[1]:7.1f} "
        print(row)

    print("\n  2c. BEAM DEPOSITION DEPTH [cm] -- 50% / 90% of the column-integrated")
    print("      rhs_terms/beam_power_deposition.Ee * V_col, cumulative from z=0.")
    print("      DEPOSITION note: the vanished faq probe's exact window is not")
    print("      recoverable. This DECLARED definition reproduces the A1 read's")
    print("      quoted pair (45/111 vs REF 75/149) to within ~5% on the")
    print("      2.0-5.0 ms window and reproduces its ORDERING on every window,")
    print("      so the A/B below is internally consistent even where the")
    print("      absolute definition differs slightly from the original.")
    print(f"      {'window[ms]':>14}"
          + "".join(f"{r.tag + ' 50/90':>26}" for r in runs))
    for lo, hi in WINDOWS[2:]:
        cells = []
        for r in runs:
            sl = r.wsl(lo, hi)
            w = r.h5["rhs_terms"]["beam_power_deposition"]["Ee"][sl]
            w = np.clip(w[:, r.col].mean(0) * r.V_col[r.col], 0, None)
            d = _depths(r.z[r.col], w)
            cells.append(f"{d[0]:11.1f} /{d[1]:11.1f}")
        print(f"      {lo:6.1f}-{hi:6.1f}" + "".join(f"{c:>26}" for c in cells))
    print("      snapshots (same definition, single frame):")
    for ts in (2.5, 3.0, 5.0, 10.0, 15.0, 19.0):
        cells = []
        for r in runs:
            i = r.isnap(ts)
            w = np.clip(r.h5["rhs_terms"]["beam_power_deposition"]["Ee"][i][r.col]
                        * r.V_col[r.col], 0, None)
            d = _depths(r.z[r.col], w)
            cells.append(f"{d[0]:11.1f} /{d[1]:11.1f}")
        print(f"      {ts:13.1f} " + "".join(f"{c:>26}" for c in cells))

    print("\n  2d. ANOMALOUS-TAIL WALKER AXIAL SPAN "
          "(cathode_diagnostics/beam_heat_anomalous_W per band [W])")
    for lo, hi in ((2.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 19.5)):
        print(f"\n      === {lo}-{hi} ms ===")
        print(f"        {'band[cm]':>16}"
              + "".join(f"{r.tag + '[W]':>16}{'%':>8}" for r in runs))
        tot = []
        prof = []
        for r in runs:
            sl = r.wsl(lo, hi)
            p = r.h5["cathode_diagnostics"]["beam_heat_anomalous_W"][sl].mean(0)
            prof.append(p)
            tot.append(p.sum())
        for blo, bhi in WALKER_BANDS:
            cells = ""
            for r, p, tt in zip(runs, prof, tot):
                m = r.zmask(blo, bhi)
                v = float(p[m].sum())
                cells += f"{v:16.4e}{(100 * v / tt if tt else 0):8.2f}"
            print(f"        {blo:7.0f}-{bhi:7.0f}" + cells)
        print(f"        {'TOTAL':>16}"
              + "".join(f"{tt:16.4e}{100.0:8.2f}" for tt in tot))
        # span = outermost band edge carrying >1% of the walker power
        for r, p, tt in zip(runs, prof, tot):
            if tt <= 0:
                continue
            edge = 0.0
            for blo, bhi in WALKER_BANDS:
                m = r.zmask(blo, bhi)
                if float(p[m].sum()) / tt > 0.01:
                    edge = bhi
            print(f"        {r.tag} walker span (outermost band >1% of total)"
                  f" = {edge:.0f} cm")


def section_heatflux(runs):
    print("\n" + "=" * 108)
    print("3. FIRST-METRE HEAT FLUX: region-integrated heat_conduction on Ee [W]")
    print("   sign: NEGATIVE = the region EXPORTS heat (net conduction sink);")
    print("         POSITIVE = the region IMPORTS heat.")
    print("   A1 measured a band flipping exporter -> importer:")
    print("         REF -2.77e4 W  ->  A1 +5.17e4 W  at 2-5 ms.")
    print("   IDENTIFICATION: those two numbers are the 100<z<790 cm band, not")
    print("   z<=100. Established by reproducing faq_increment.txt section B to")
    print("   4 s.f. on all six windows (REF -2.7632e4, A1 +5.1754e4 at 2-5 ms);")
    print("   the z<=100 band is reported alongside for completeness.")
    print("=" * 108)
    for name, (lo_z, hi_z) in REGIONS[:3]:
        print(f"\n  --- {name} ---")
        print(f"      {'window[ms]':>14}"
              + "".join(f"{r.tag:>18}" for r in runs)
              + f"{'ARM-A1':>16}{'ARM-REF':>16}")
        for lo, hi in WINDOWS:
            vals = []
            for r in runs:
                m = r.zmask(lo_z, hi_z)
                w = r.term_W("Ee", "heat_conduction", m)
                vals.append(float(w[r.wsl(lo, hi)].mean()))
            print(f"      {lo:6.1f}-{hi:6.1f}"
                  + "".join(f"{v:18.4e}" for v in vals)
                  + f"{vals[0] - vals[1]:16.4e}{vals[0] - vals[2]:16.4e}")

    print("\n  3b. WHOLE-COLUMN Ee SINK PARTITION: share consumed in z<=100")
    print(f"      {'window[ms]':>14}"
          + "".join(f"{r.tag + ' src%':>18}" for r in runs))
    for lo, hi in WINDOWS:
        cells = []
        for r in runs:
            sl = r.wsl(lo, hi)
            msrc = r.zmask(*SOURCE_Z)
            tot_s = 0.0
            tot_a = 0.0
            for term in r.h5["rhs_terms"].keys():
                w = r.term_W("Ee", term, msrc)
                if w is not None:
                    v = float(w[sl].mean())
                    if v < 0:
                        tot_s += -v
                wa = r.term_W("Ee", term)
                if wa is not None:
                    v = float(wa[sl].mean())
                    if v < 0:
                        tot_a += -v
            cells.append(100 * tot_s / tot_a if tot_a else np.nan)
        print(f"      {lo:6.1f}-{hi:6.1f}" + "".join(f"{c:18.2f}" for c in cells))


def section_front(runs):
    print("\n" + "=" * 108)
    print("4. THE QUENCH FRONT: S_net(z,t) zero crossing")
    print("   front = outermost column cell reached by the CONTIGUOUS positive-")
    print("   S_net region starting at the source. A1: detaches 1.0-1.5 ms,")
    print("   marches inward at 43 cm/ms to z = 419 cm by 19.5 ms.")
    print("=" * 108)
    times = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5,
             15.0, 17.5, 19.0, 19.5)
    print(f"\n      {'t[ms]':>8}" + "".join(f"{r.tag + ' front[cm]':>22}"
                                           for r in runs))
    hist = {r.tag: [] for r in runs}
    for t_ms in times:
        cells = []
        for r in runs:
            i = r.isnap(t_ms)
            s = r.s_net(i)
            cz = r.z[r.col]
            cs = s[r.col]
            pos = cs > 0
            if not pos[0]:
                fz = np.nan
            else:
                k = int(np.argmin(pos)) if (~pos).any() else len(pos)
                fz = cz[k - 1] if k > 0 else np.nan
            hist[r.tag].append(fz)
            cells.append(fz)
        print(f"      {t_ms:8.2f}" + "".join(f"{c:22.1f}" for c in cells))
    print("\n      march rate over 1.5 -> 19.5 ms [cm/ms]:")
    for r in runs:
        h = np.array(hist[r.tag], dtype=float)
        i0 = times.index(1.5)
        v = (h[i0] - h[-1]) / (19.5 - 1.5)
        print(f"        {r.tag:<12} {h[i0]:8.1f} -> {h[-1]:8.1f} cm   "
              f"inward rate {v:7.2f} cm/ms")


def section_mid(runs):
    print("\n" + "=" * 108)
    print("5. MID-COLUMN (790-1045 cm) STATE AND IONIZATION")
    print("   A1: nn x290 REF, ionization -> 0 by 15 ms, Te -> floor.")
    print("=" * 108)
    print(f"\n      {'window[ms]':>12} {'field':>6}"
          + "".join(f"{r.tag:>16}" for r in runs)
          + f"{'ARM/A1':>10}{'ARM/REF':>10}")
    for lo, hi in WINDOWS[2:]:
        for field in ("nn", "Te", "n"):
            vals = []
            for r in runs:
                m = r.zmask(*MID_Z) & r.col
                sl = r.wsl(lo, hi)
                arr = r.h5[field][sl][:, m]
                vals.append(float((arr @ r.V_col[m]).mean() / r.V_col[m].sum()))
            print(f"      {lo:5.1f}-{hi:5.1f} {field:>6}"
                  + "".join(f"{v:16.4e}" for v in vals)
                  + f"{vals[0] / vals[1]:10.3f}{vals[0] / vals[2]:10.3f}")
        # volume-integrated bulk ionization
        vals = []
        for r in runs:
            m = r.zmask(*MID_Z)
            rate = r.term_rate("ionization_birth", "n", m)
            vals.append(float(rate[r.wsl(lo, hi)].mean()))
        print(f"      {lo:5.1f}-{hi:5.1f} {'S_ion':>6}"
              + "".join(f"{v:16.4e}" for v in vals)
              + f"{vals[0] / vals[1] if vals[1] else np.nan:10.3f}"
              + f"{vals[0] / vals[2] if vals[2] else np.nan:10.3f}")
        print()


def section_dump(runs):
    print("\n" + "=" * 108)
    print("6. THE DUMP CHANNEL: mid-column ion energy ledger")
    print("   A1: mid ei_exchange -> ion_neutral_collision = -8.9e3 W, 74% of sink.")
    print("   Does thinning the neutral pile shrink it?")
    print("=" * 108)
    for lo, hi in ((2.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 19.5)):
        print(f"\n    === {lo}-{hi} ms, MID 790-1045 ===")
        print(f"      {'term':<36}" + "".join(f"{r.tag + '[W]':>18}" for r in runs))
        terms = sorted(runs[0].h5["rhs_terms"].keys())
        rows = []
        sinks = [0.0] * len(runs)
        for term in terms:
            vals = []
            ok = True
            for r in runs:
                m = r.zmask(*MID_Z)
                w = r.term_W("Ei", term, m)
                if w is None:
                    ok = False
                    break
                vals.append(float(w[r.wsl(lo, hi)].mean()))
            if not ok or all(abs(v) < 1.0 for v in vals):
                continue
            rows.append((max(abs(v) for v in vals), term, vals))
            for k, v in enumerate(vals):
                if v < 0:
                    sinks[k] += -v
        rows.sort(reverse=True)
        for _, term, vals in rows:
            print(f"      {term:<36}" + "".join(f"{v:18.4e}" for v in vals))
        print(f"      {'TOTAL Ei SINK':<36}"
              + "".join(f"{-s:18.4e}" for s in sinks))
        print(f"      {'ion_neutral_collision % of sink':<36}", end="")
        for r, s in zip(runs, sinks):
            m = r.zmask(*MID_Z)
            w = r.term_W("Ei", "ion_neutral_collision", m)
            v = float(w[r.wsl(lo, hi)].mean()) if w is not None else 0.0
            print(f"{(100 * abs(v) / s if s else np.nan):18.2f}", end="")
        print()


def section_wind(runs):
    print("\n" + "=" * 108)
    print("7. THE NEUTRAL FLOW FIELD (the closure's own u_n / M_n)")
    print("   Mach uses the ISOTHERMAL neutral sound speed at the FIXED bath")
    print("   temperature Tn_K (300 K): c_n = sqrt(k*Tn/m_He).")
    print("=" * 108)
    arm = runs[0]
    if not arm.has("u_n"):
        print("  ARM carries no u_n field -- the closure did not build it.")
        return
    Tn = float(arm.params.get("Tn_K", 300.0))
    c_n = np.sqrt(K_B_ERG * Tn / M_HE_G)  # cm/s
    print(f"\n  Tn_K = {Tn:.1f} K   ->   c_n = {c_n / 1e5:.4f} km/s")
    print("\n  7a. PLATEAU-MEAN (15.0-19.5 ms) u_n(z) AND MACH, ARM")
    sl = arm.wsl(15.0, 19.5)
    un = arm.h5["u_n"][sl].mean(0)
    print(f"      {'z[cm]':>9} {'u_n[km/s]':>12} {'Mach':>9} "
          f"{'nn[cm^-3]':>12} {'u_plasma[km/s]':>16}")
    up = arm.h5["u"][sl].mean(0)
    nn = arm.h5["nn"][sl].mean(0)
    idx = np.flatnonzero(arm.col)
    for i in idx[:: max(1, len(idx) // 20)]:
        print(f"      {arm.z[i]:9.1f} {un[i] / 1e5:12.4f} "
              f"{un[i] / c_n:9.3f} {nn[i]:12.4e} {up[i] / 1e5:16.4f}")
    print("\n  7b. AT THE FIVE ES1 PORTS (plateau mean)")
    print(f"      {'port':>6} {'z[cm]':>8} {'u_n[km/s]':>12} {'Mach':>9} "
          f"{'u_plasma[km/s]':>16}")
    for p, zp in PORTS:
        i = int(np.argmin(np.abs(arm.z - zp)))
        print(f"      {p:6d} {arm.z[i]:8.1f} {un[i] / 1e5:12.4f} "
              f"{un[i] / c_n:9.3f} {up[i] / 1e5:16.4f}")
    print("\n      measured stance-robust far-port residual: "
          "+1.8 km/s (p41), +5.4 km/s (p50)")
    for p, zp in ((41, 1429.0), (50, 1716.0)):
        i = int(np.argmin(np.abs(arm.z - zp)))
        meas = 1.8 if p == 41 else 5.4
        print(f"      p{p}: model u_n {un[i] / 1e5:+.4f} km/s vs measured "
              f"{meas:+.1f} km/s   -> ratio {un[i] / 1e5 / meas:+.4f}")
    print("\n  7c. WIND EVOLUTION: volume-mean u_n per region [km/s]")
    print(f"      {'window[ms]':>12}" + "".join(f"{n.split()[0]:>16}"
                                               for n, _ in REGIONS))
    for lo, hi in WINDOWS[2:]:
        s = arm.wsl(lo, hi)
        u = arm.h5["u_n"][s].mean(0)
        cells = []
        for _, (lo_z, hi_z) in REGIONS:
            m = arm.zmask(lo_z, hi_z) & arm.col
            cells.append(float(u[m] @ arm.V_col[m] / arm.V_col[m].sum()) / 1e5)
        print(f"      {lo:5.1f}-{hi:5.1f}" + "".join(f"{c:16.4f}" for c in cells))
    print("\n  7d. MOMENTUM-CLOSURE RHS CHANNELS on Ee [W], whole column")
    print(f"      {'window[ms]':>12} {'neutral_wind_advection':>24} "
          f"{'neutral_momentum_wall':>24}")
    for lo, hi in WINDOWS[2:]:
        s = arm.wsl(lo, hi)
        a = arm.term_W("Ee", "neutral_wind_advection")
        b = arm.term_W("Ee", "neutral_momentum_wall")
        av = float(a[s].mean()) if a is not None else np.nan
        bv = float(b[s].mean()) if b is not None else np.nan
        print(f"      {lo:5.1f}-{hi:5.1f} {av:24.4e} {bv:24.4e}")


def section_health(runs):
    print("\n" + "=" * 108)
    print("8. RUN HEALTH")
    print("=" * 108)
    for r in runs:
        print(f"\n  --- {r.tag}: {r.path} ---")
        d = r.h5["diagnostics"]
        dt = d["accepted_dt"][:]
        print(f"    accepted steps {dt.size}   dt min {dt.min():.4e} "
              f"median {np.median(dt):.4e} max {dt.max():.4e}")
        clamp = d["clamped_to_dt_min"][:]
        print(f"    clamped to dt_min: {int(clamp.sum())} "
              f"({100 * clamp.mean():.3f}%)")
        ac = np.array([_s(x) for x in d["active_constraint"][:]])
        vals, cnts = np.unique(ac, return_counts=True)
        order = np.argsort(-cnts)
        print("    active timestep constraint census:")
        for i in order[:8]:
            print(f"      {vals[i]:28s} {cnts[i]:8d}  "
                  f"({100 * cnts[i] / ac.size:5.2f}%)")
        rr = np.array([_s(x) for x in d["rejection_reason"][:]])
        vals, cnts = np.unique(rr, return_counts=True)
        print("    rejection reasons: "
              + ", ".join(f"{v}={c}" for v, c in zip(vals, cnts)))
        fl = r.h5["floor_ledger"]
        print("    floor ledger: "
              + ", ".join(f"{k}={float(fl[k][()]):.4e}" for k in sorted(fl.keys())))
        ard = r.h5["atomic_rate_domain"]
        f_below = ard["active_volume_fraction_below"][:]
        for t_ms in (2.0, 5.0, 10.0, 15.0, 19.5):
            i = r.isnap(t_ms)
            print(f"      below-ADAS-edge active VOLUME fraction @ {t_ms:5.1f} ms"
                  f" = {f_below[i]:.4f}")
        for name in ("n", "nn", "nn_a", "Te", "Ti", "u_n", "M_n"):
            if not r.has(name):
                continue
            chunk = r.h5[name][::37]
            print(f"    finiteness {name:5s}: all finite = "
                  f"{bool(np.isfinite(chunk).all())}  "
                  f"min {np.nanmin(chunk):.4e} max {np.nanmax(chunk):.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--a1", required=True)
    ap.add_argument("--ref", required=True)
    args = ap.parse_args()
    runs = [Run(args.arm, "ARM(fa2)"), Run(args.a1, "A1(fa1)"),
            Run(args.ref, "REF(sp1)")]
    print("=" * 108)
    print("fa2 -- NBL MOMENTUM PREVIEW A/B: 1-zone fluid neutral momentum ON")
    print("  DISCLOSED INSTRUMENT. The closure was REJECTED as a production")
    print("  candidate 2026-08-04 (M_n campaign); it is")
    print("  used here as leg C's reserved vehicle, NOT as a candidate.")
    print("  CARRIED CAVEAT: Tn is FIXED at 300 K in this closure -- the energy")
    print("  back-reaction is absent, so this preview tests the MOMENTUM HALF")
    print("  ONLY and may under-push.")
    for r in runs:
        print(f"    {r.tag:<12} {r.path}")
    print("=" * 108)
    section_provenance(runs)
    section_base(runs)
    section_nbl(runs)
    section_heatflux(runs)
    section_front(runs)
    section_mid(runs)
    section_dump(runs)
    section_wind(runs)
    section_health(runs)
    for r in runs:
        r.close()


if __name__ == "__main__":
    main()
