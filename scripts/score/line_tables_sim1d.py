#!/usr/bin/env python3
"""Line-power tables of a saved sim1d run, as LaTeX table bodies.

READ-ONLY over a saved sim1d HDF5 artifact.  The per-line powers are those of
``line_radiation_sim1d.py``, reused unchanged: ``read_window`` loads the
window, ``evaluate_stage`` evaluates every adf15 EXCIT line of the He II and
He I photon-emissivity files at the saved state and returns the window-mean
emissivity per line and cell, the machine total and the clamp census.  The
port of each row is placed by ``port_radiance_sim1d.locate_cell`` (the linear
port-to-z law of the committed ES1 overlay and the nearest-cell rule).  The
atomic data of each transition (wavelength, levels, Einstein A) are read from
the committed ``scripts/data/he_lines_atomic.json``, which records its NIST
ASD source and conventions.

Four tables are written, each as a ``tabular`` body only (booktabs rules, no
``table`` environment, no caption, a leading ``% TODO(Tom): caption`` line),
with a Markdown twin of each:

    he2_lines.tex         He II transitions: wavelength, region, photon energy,
                          upper-level energy, A, share of the He II line power,
                          power [W]; ordered by power
    he1_lines.tex         the same for He I, with the upper level's offset above
                          the same-spin metastable
    tail_fractions.tex    per port and the cathode-adjacent cell: T_e, n_e and
                          the Maxwellian fraction above four excitation
                          thresholds, Gamma(3/2, E/T_e) / Gamma(3/2)
    axial_line_power.tex  per port and the cathode-adjacent cell: z, T_e, n_e,
                          n_n, the He II and He I line-power densities, their
                          ratio and the 468.7 nm and 587.7 nm power densities;
                          closing rows with the machine totals and the z by
                          which 50 % and 90 % of each stage's power is emitted

and ``tables_summary.txt`` with the machine totals, the adf11 completeness and
the clamp census at full precision.

The two He II transitions the adf15 file lacks (6 -> 3 at 273.4 nm and 5 -> 4
at 1012.6 nm) are listed with a footnote mark.  The 5 -> 4 power is estimated
from the 5 -> 3 row by the branching ratio and photon energies,
P(5->4) = P(5->3) * A(5->4)/A(5->3) * E(5->4)/E(5->3), which holds because the
two lines share an upper level; the 6 -> 3 line shares no upper level with any
row of the file, so no power is estimated for it.  Estimated rows are outside
the stage total, and their share is relative to it.

Species are written He~II (the ion, He+) and He~I (the neutral).  Window means
are plain means over the saves in the window, as ``evaluate_stage`` takes them.

    line_tables_sim1d.py --h5 RUN.h5 --window-ms 15 19.5 --outdir DIR
                         [--ports 11 21 22 27 29 41 50]
    line_tables_sim1d.py --h5 RUN.h5 --window-ms 15 19.5 --self-test

``--self-test`` builds the tables in memory and checks (a) that every adf15
line has a JSON entry and every JSON entry marked in the file has an adf15
line, (b) that the power column sums to the ``evaluate_stage`` machine total
and the share column to one over the adf15 rows, each share times the total
reproducing its row's power, (c) the Maxwellian fraction against its closed
form, and (d) that every LaTeX row has the column count of its ``tabular``
specification and balanced braces.  Exit 0 when every check passes, 1
otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.special import gammaincc

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

import line_radiation_sim1d as LR  # noqa: E402
import port_radiance_sim1d as PRS  # noqa: E402

ATOMIC_JSON = Path(__file__).resolve().parents[1] / "data" / "he_lines_atomic.json"
DEFAULT_PORTS = (11, 21, 22, 27, 29, 41, 50)

#: Spectral-region names by vacuum wavelength [nm], half-open [lo, hi).
REGIONS = (
    ("EUV", 0.0, 100.0),
    ("VUV", 100.0, 200.0),
    ("near-UV", 200.0, 380.0),
    ("visible", 380.0, 750.0),
    ("near-IR", 750.0, math.inf),
)

#: Transitions whose power density the axial table carries, by JSON key.
AXIAL_LINES = (("HeII_4-3", "He~II"), ("HeI_1s3d_3D-1s2p_3Po", "He~I"))

#: Threshold energies of the tail table: the upper levels of these entries.
TAIL_THRESHOLDS = (
    "HeI_1s3d_3D-1s2p_3Po",
    "HeII_2-1",
    "HeII_3-1",
    "HeII_4-3",
)

STAGE_SPECIES = {"he1": "He II", "he0": "He I"}

#: Relative tolerance of the summation checks.  The power column and the
#: machine total are sums of at most 15 non-negative doubles taken in
#: different orders, so they agree to n * 2**-53 ~ 2e-15; 1e-12 is the bound
#: the table contract states and holds three decades of margin.
SUM_RTOL = 1.0e-12

#: Relative tolerance of the closed-form tail check: gammaincc and the
#: erfc-based closed form are each accurate to a few ulp, so 1e-13 (~500 ulp)
#: bounds their difference with margin.
TAIL_RTOL = 1.0e-13


# --- atomic data -----------------------------------------------------------


def load_atomic(path=ATOMIC_JSON):
    """Return the committed atomic data and a key -> transition map."""
    doc = json.loads(Path(path).read_text())
    return doc, {t["key"]: t for t in doc["transitions"]}


def region_of(lam_nm):
    """Spectral-region name of a vacuum wavelength [nm]."""
    for name, lo, hi in REGIONS:
        if lo <= lam_nm < hi:
            return name
    raise ValueError(f"{lam_nm} nm is outside every region")


def adf15_index(atomic):
    """``(adf15 file, ISEL) -> transition`` for every entry in the files."""
    return {
        (t["adf15"]["file"], t["adf15"]["isel"]): t
        for t in atomic["transitions"]
        if t["in_adf15"]
    }


def tail_fraction(E_eV, Te_eV):
    """Fraction of a Maxwellian's electrons above energy E: Gamma(3/2, E/T)/Gamma(3/2)."""
    return gammaincc(1.5, np.asarray(E_eV, dtype=float) / np.asarray(Te_eV, dtype=float))


# --- reduction ---------------------------------------------------------------


def port_cells(h5_path, ports, active_index):
    """``[(port, active-cell index)]`` under the overlay's port law and nearest cell."""
    law = PRS.port_axial_law()
    out = []
    with h5py.File(h5_path, "r") as f:
        for p in ports:
            rec = PRS.locate_cell(f, p, law)
            hit = np.flatnonzero(active_index == rec["cell"])
            if not hit.size:
                raise LR.ArtifactRefused(
                    f"port {p} maps to cell {rec['cell']}, which is not plasma-active"
                )
            out.append((int(p), int(hit[0])))
    return out


def reduce_run(h5_path, window_ms, ports):
    """Everything the tables need from one run, as plain arrays and dicts."""
    data = LR.read_window(h5_path, tuple(window_ms))
    res = {k: LR.evaluate_stage(k, data) for k in LR.STAGE_ORDER}
    st, geo = data["state"], data["geometry"]
    return {
        "h5": str(h5_path),
        "window_ms": tuple(window_ms),
        "frames": data["frames"],
        "res": res,
        "Te": st["Te"].mean(axis=0),
        "ne": st["n"].mean(axis=0),
        "nn": st["nn"].mean(axis=0),
        "z_cm": geo["z_cm"],
        "length_cm": geo["length_cm"],
        "volume_cm3": geo["volume_cm3"],
        "ports": port_cells(h5_path, ports, geo["active_index"]),
    }


def line_rows(red, atomic, stage):
    """Rows of one stage's line table, adf15 lines first, ordered by power."""
    doc, by_key = atomic
    idx = adf15_index(doc)
    r = red["res"][stage]
    total = r["machine_W_total"]
    rows = []
    for k, block in enumerate(r["lines"]):
        t = idx[(r["spec_file"], block["isel"])]
        P = float(r["machine_W"][k])
        rows.append({"t": t, "power_W": P, "share": P / total, "estimated": False})
    rows.sort(key=lambda row: -row["power_W"])
    extra = []
    if stage == "he1":
        p53 = next(row for row in rows if row["t"]["key"] == "HeII_5-3")
        for key in ("HeII_5-4", "HeII_6-3"):
            t = by_key[key]
            P = None
            if key == "HeII_5-4":
                t53 = p53["t"]
                P = (p53["power_W"] * t["einstein_A_s"] / t53["einstein_A_s"]
                     * t["photon_eV"] / t53["photon_eV"])
            extra.append({"t": t, "power_W": P,
                          "share": None if P is None else P / total,
                          "estimated": True})
        extra.sort(key=lambda row: -(row["power_W"] or 0.0))
    return rows + extra


def emitted_by_z_m(red, cell_W, frac):
    """z [m] by which ``frac`` of the axial sum of ``cell_W`` is emitted.

    The cumulative sum is complete at each cell's downstream face, so it is
    interpolated linearly against the face positions, starting from zero at
    the first active cell's upstream face.
    """
    z, L = red["z_cm"], red["length_cm"]
    faces = np.concatenate([[z[0] - 0.5 * L[0]], z + 0.5 * L])
    cum = np.concatenate([[0.0], np.cumsum(cell_W)]) / np.sum(cell_W)
    return float(np.interp(frac, cum, faces)) / 100.0


# --- formatting ------------------------------------------------------------


def _sig(x, sig):
    """``(mantissa, exponent)`` strings of x to ``sig`` significant figures, or a plain number."""
    if x == 0.0:
        return "0", None
    e = int(math.floor(math.log10(abs(x))))
    m = round(x / 10.0 ** e, sig - 1)
    if abs(m) >= 10.0:
        m /= 10.0
        e += 1
    if -2 <= e <= 2:
        decimals = max(sig - 1 - e, 0)
        return f"{m * 10.0 ** e:.{decimals}f}", None
    return f"{m:.{sig - 1}f}", e


def num_tex(x, sig=2):
    if x is None:
        return "---"
    m, e = _sig(x, sig)
    return rf"\num{{{m}}}" if e is None else rf"\num{{{m}e{e}}}"


def num_md(x, sig=2):
    if x is None:
        return "—"
    m, e = _sig(x, sig)
    return m if e is None else f"{m} × 10^{e}"


def fixed(x, decimals):
    return f"{x:.{decimals}f}"


class Cell:
    """One table cell: a number with its significant figures, or text in both markups."""

    def __init__(self, value=None, sig=2, tex=None, md=None, fixed_decimals=None):
        self.value, self.sig = value, sig
        self.tex_text, self.md_text = tex, md
        self.fixed_decimals = fixed_decimals

    def tex(self):
        if self.tex_text is not None:
            return self.tex_text
        if self.fixed_decimals is not None and self.value is not None:
            return rf"\num{{{fixed(self.value, self.fixed_decimals)}}}"
        return num_tex(self.value, self.sig)

    def md(self):
        if self.md_text is not None:
            return self.md_text
        if self.fixed_decimals is not None and self.value is not None:
            return fixed(self.value, self.fixed_decimals)
        return num_md(self.value, self.sig)


def T(tex, md=None):
    return Cell(tex=tex, md=tex if md is None else md)


class Table:
    """A table as header cells, column spec, row blocks and comment lines."""

    def __init__(self, name, colspec, header, comments=()):
        self.name, self.colspec, self.header = name, colspec, header
        self.comments = list(comments)
        self.blocks = [[]]
        self.footnotes = []

    def add(self, row):
        self.blocks[-1].append(row)

    def rule(self):
        self.blocks.append([])

    def tex(self):
        out = ["% TODO(Tom): caption"]
        out += [f"% {c}" for c in self.comments]
        out.append(rf"\begin{{tabular}}{{{self.colspec}}}")
        out.append(r"\toprule")
        out.append(" & ".join(c.tex() for c in self.header) + r" \\")
        out.append(r"\midrule")
        for i, block in enumerate(b for b in self.blocks if b):
            if i:
                out.append(r"\midrule")
            out += [" & ".join(c.tex() for c in row) + r" \\" for row in block]
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out += [f"% {f[0]}" for f in self.footnotes]
        return "\n".join(out) + "\n"

    def md(self):
        out = [f"<!-- {c} -->" for c in self.comments]
        out.append("| " + " | ".join(c.md() for c in self.header) + " |")
        out.append("|" + "---|" * len(self.header))
        for block in self.blocks:
            out += ["| " + " | ".join(c.md() for c in row) + " |" for row in block]
        out += ["", *[f[1] for f in self.footnotes]]
        return "\n".join(out) + "\n"


def he1_level_tex(lvl):
    return rf"{lvl['n']}\,{{}}^{{{lvl['multiplicity']}}}\mathrm{{{lvl['L']}}}"


def he1_level_md(lvl):
    sup = str(lvl["multiplicity"]).translate(str.maketrans("13", "¹³"))
    return f"{lvl['n']}{sup}{lvl['L']}"


def transition_cells(t):
    if t["species"] == "He II":
        u, l = t["upper"]["n"], t["lower"]["n"]
        return rf"${u} \rightarrow {l}$", f"{u} → {l}"
    return (rf"${he1_level_tex(t['upper'])} \rightarrow {he1_level_tex(t['lower'])}$",
            f"{he1_level_md(t['upper'])} → {he1_level_md(t['lower'])}")


# --- tables ------------------------------------------------------------------


def lines_table(red, atomic, stage):
    rows = line_rows(red, atomic, stage)
    r = red["res"][stage]
    he1 = stage == "he0"
    name = "he1_lines" if he1 else "he2_lines"
    sp = "He~I" if he1 else "He~II"
    header = [T("Transition"), T(r"$\lambda$ (nm)", "λ (nm)"), T("Region"),
              T(r"$h\nu$ (eV)", "hν (eV)"), T(r"$E_{u}$ (eV)", "E_u (eV)")]
    if he1:
        header.append(T(r"$E_{u}-E_{m}$ (eV)", "E_u − E_m (eV)"))
    header += [T(r"$A$ (s$^{-1}$)", "A (s^-1)"), T(f"Share of {sp}", f"Share of {sp.replace('~', ' ')}"),
               T("Power (W)")]
    comments = [
        f"Species notation: {sp} ({'neutral helium' if he1 else 'singly ionized helium'}).",
        "Columns: vacuum wavelength (multiplet centroid); spectral region; photon energy;",
        "upper-level energy above the species ground state" + (
            "; offset of the upper level above the same-spin metastable (2^3S or 2^1S)" if he1 else "")
        + "; Einstein A of the multiplet; share of the stage's line power; power.",
        f"Window {red['window_ms'][0]}-{red['window_ms'][1]} ms on the main-discharge clock, "
        f"{red['frames']} saves; stage total {r['machine_W_total']:.6e} W over {len(r['lines'])} "
        "adf15 lines.",
    ]
    if not he1:
        comments.append(
            "Estimated rows: P(5->4) = P(5->3) * A(5->4)/A(5->3) * E_ph(5->4)/E_ph(5->3) "
            "(shared upper level); share = P / stage total.")
    tab = Table(name, "l" + "r" * (len(header) - 1), header, comments)
    marks = {"HeII_5-4": "a", "HeII_6-3": "b"}
    estimated_started = False
    for row in rows:
        t = row["t"]
        tt, tm = transition_cells(t)
        if row["estimated"]:
            mk = marks[t["key"]]
            tt += rf"\textsuperscript{{{mk}}}"
            tm += f" ({mk})"
        cells = [T(tt, tm),
                 Cell(t["wavelength_vac_nm"],
                      fixed_decimals=2 if t["wavelength_vac_nm"] < 100.0 else 1),
                 T(region_of(t["wavelength_vac_nm"])),
                 Cell(t["photon_eV"], fixed_decimals=2),
                 Cell(t["upper_level_eV"], fixed_decimals=2)]
        if he1:
            m = t["metastable"]
            cells.append(T(
                rf"\num{{{fixed(t['metastable_offset_eV'], 2)}}} above ${he1_level_tex(m)}$",
                f"{fixed(t['metastable_offset_eV'], 2)} above {he1_level_md(m)}"))
        cells += [Cell(t["einstein_A_s"], 2), Cell(row["share"], 2), Cell(row["power_W"], 2)]
        if row["estimated"] and not estimated_started:
            tab.rule()  # the estimated rows sit below a rule of their own
            estimated_started = True
        tab.add(cells)
    if not he1:
        tab.footnotes = [
            ("a: not in the adf15 file; estimated from the 5->3 row by A(5->4)/A(5->3) "
             "and the photon-energy ratio.",
             "(a) Not in the adf15 file; estimated from the 5 → 3 row by A(5→4)/A(5→3) "
             "and the photon-energy ratio."),
            ("b: not in the adf15 file; no row of the file shares its upper level (n = 6), "
             "so no power is estimated.",
             "(b) Not in the adf15 file; no row of the file shares its upper level (n = 6), "
             "so no power is estimated."),
        ]
    else:
        ms = atomic[0]["metastables"]
        tab.comments.append("Metastables: " + "; ".join(
            f"{m['level']['n']}^{m['level']['multiplicity']}{m['level']['L']} at "
            f"{m['level_eV']:.2f} eV, A to the ground state "
            + ("not in the line list" if m["einstein_A_s"] is None else f"{m['einstein_A_s']:.2e} s^-1")
            for m in ms) + ".")
    tab.rows = rows
    return tab


def location_rows(red):
    """``[(tex label, md label, active cell)]``: the cathode-adjacent cell, then each port."""
    out = [("cathode-adjacent cell", "cathode-adjacent cell", 0)]
    out += [(f"port {p}", f"port {p}", i) for p, i in red["ports"]]
    return out


def tail_table(red, atomic):
    by_key = atomic[1]
    E = [by_key[k]["upper_level_eV"] for k in TAIL_THRESHOLDS]
    header = [T("Location"), T(r"$z$ (m)", "z (m)"), T(r"$T_e$ (eV)", "T_e (eV)"),
              T(r"$n_e$ (cm$^{-3}$)", "n_e (cm^-3)")]
    header += [T(rf"$>\,$\num{{{fixed(e, 2)}}}~eV", f"> {fixed(e, 2)} eV") for e in E]
    comments = [
        "Fraction of a Maxwellian at the window-mean local T_e above each threshold, "
        "Gamma(3/2, E/T_e)/Gamma(3/2).",
        "Thresholds: the upper levels of He~I 3^3D (587.7 nm), He~II n = 2 (30.4 nm), "
        "n = 3 (25.6 nm) and n = 4 (468.7 nm).",
        f"Window {red['window_ms'][0]}-{red['window_ms'][1]} ms on the main-discharge clock.",
    ]
    tab = Table("tail_fractions", "l" + "r" * (len(header) - 1), header, comments)
    for lt, lm, i in location_rows(red):
        Te = float(red["Te"][i])
        cells = [T(lt, lm), Cell(red["z_cm"][i] / 100.0, fixed_decimals=2),
                 Cell(Te, fixed_decimals=1), Cell(float(red["ne"][i]), 2)]
        cells += [Cell(float(tail_fraction(e, Te)), 2) for e in E]
        tab.add(cells)
    return tab


def axial_table(red, atomic):
    res = red["res"]
    by_key = atomic[1]
    P2 = res["he1"]["eps_W_mean"].sum(axis=0)
    P1 = res["he0"]["eps_W_mean"].sum(axis=0)
    sel = []
    for key, _ in AXIAL_LINES:
        t = by_key[key]
        stage = "he1" if t["species"] == "He II" else "he0"
        k = [b["isel"] for b in res[stage]["lines"]].index(t["adf15"]["isel"])
        sel.append((t, res[stage]["eps_W_mean"][k], res[stage]["machine_W"][k]))
    unit = r"(W\,cm$^{-3}$)"
    header = [T("Location"), T(r"$z$ (m)", "z (m)"), T(r"$T_e$ (eV)", "T_e (eV)"),
              T(r"$n_e$ (cm$^{-3}$)", "n_e (cm^-3)"), T(r"$n_n$ (cm$^{-3}$)", "n_n (cm^-3)"),
              T(f"He~II {unit}", "He II (W cm^-3)"), T(f"He~I {unit}", "He I (W cm^-3)"),
              T("He~II/He~I", "He II/He I")]
    for t, _, _ in sel:
        sp = "He~II" if t["species"] == "He II" else "He~I"
        lam = f"{t['wavelength_vac_nm']:.1f}"
        header.append(T(f"{sp} {lam}~nm {unit}", f"{sp.replace('~', ' ')} {lam} nm (W cm^-3)"))
    comments = [
        "Window-mean values at the cell holding each port; line-power densities summed "
        "over the adf15 lines of each stage; n_n the in-column neutral density.",
        "Closing rows: machine totals (W) and the z by which 50 % / 90 % of each stage's "
        "(or line's) power is emitted, interpolated on cell faces.",
        f"Window {red['window_ms'][0]}-{red['window_ms'][1]} ms on the main-discharge clock.",
    ]
    tab = Table("axial_line_power", "l" + "r" * (len(header) - 1), header, comments)
    for lt, lm, i in location_rows(red):
        ratio = P2[i] / P1[i] if P1[i] > 0 else None
        cells = [T(lt, lm), Cell(red["z_cm"][i] / 100.0, fixed_decimals=2),
                 Cell(float(red["Te"][i]), fixed_decimals=1),
                 Cell(float(red["ne"][i]), 2), Cell(float(red["nn"][i]), 2),
                 Cell(float(P2[i]), 2), Cell(float(P1[i]), 2), Cell(ratio, 2)]
        cells += [Cell(float(eps[i]), 2) for _, eps, _ in sel]
        tab.add(cells)
    tab.rule()
    vol = red["volume_cm3"]
    W2, W1 = res["he1"]["machine_W_total"], res["he0"]["machine_W_total"]
    tot = [T("machine total (W)"), T(""), T(""), T(""), T(""),
           Cell(W2, 3), Cell(W1, 3), Cell(W2 / W1, 2)]
    tot += [Cell(float(Wl), 2) for _, _, Wl in sel]
    tab.add(tot)
    cols = [P2 * vol, P1 * vol] + [eps * vol for _, eps, _ in sel]
    zrow = [T(r"emitted by $z$ (m), 50\,\% / 90\,\%", "emitted by z (m), 50 % / 90 %"),
            T(""), T(""), T(""), T("")]
    for j, cw in enumerate(cols):
        a, b = emitted_by_z_m(red, cw, 0.5), emitted_by_z_m(red, cw, 0.9)
        zrow.append(T(rf"\num{{{a:.1f}}} / \num{{{b:.1f}}}", f"{a:.1f} / {b:.1f}"))
        if j == 1:
            zrow.append(T(""))
    tab.add(zrow)
    return tab


def summary_text(red, atomic):
    res = red["res"]
    L = [f"h5: {red['h5']}",
         f"window: {red['window_ms'][0]}-{red['window_ms'][1]} ms on the main-discharge clock, "
         f"{red['frames']} saves; {red['z_cm'].size} plasma-active cells",
         f"atomic data: {ATOMIC_JSON.name}, NIST ASD {atomic[0]['source']['version']} "
         f"retrieved {atomic[0]['source']['retrieved']}",
         "ports -> active cell: " + ", ".join(f"{p}->{i} (z={red['z_cm'][i]:.2f} cm)"
                                             for p, i in red["ports"]),
         ""]
    for k in LR.STAGE_ORDER:
        r = res[k]
        cath = r["cell_W"].sum(axis=0)[0] / r["machine_W_total"]
        L += [f"[{STAGE_SPECIES[k]}] adf15 file {r['spec_file']}, {len(r['lines'])} EXCIT lines",
              f"  machine total (sum of adf15 lines)  {r['machine_W_total']:.9e} W",
              f"  adf11 line-power channel            {r['adf11_machine_W']:.9e} W",
              f"  completeness (adf15 / adf11)        {r['completeness_machine']:.6f}",
              f"  share from the cathode-adjacent cell {cath:.6f}",
              f"  adf15 clamp census: {json.dumps(r['adf15_clamp'])}",
              f"  adf11 clamp census: {json.dumps(r['adf11_clamp'])}",
              f"  below the quotable T_e floor: {json.dumps(r['sub_quotable'])}",
              ""]
    return "\n".join(L) + "\n"


def build_tables(red, atomic):
    return [lines_table(red, atomic, "he1"), lines_table(red, atomic, "he0"),
            tail_table(red, atomic), axial_table(red, atomic)]


# --- self-test -------------------------------------------------------------


def _tail_closed_form(x):
    """Gamma(3/2, x)/Gamma(3/2) = erfc(sqrt x) + 2 sqrt(x/pi) exp(-x)."""
    return math.erfc(math.sqrt(x)) + 2.0 * math.sqrt(x / math.pi) * math.exp(-x)


def self_test(red, atomic, tables):
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    doc = atomic[0]
    idx = adf15_index(doc)
    for k in LR.STAGE_ORDER:
        r = red["res"][k]
        have = {(r["spec_file"], b["isel"]): b["wavelength_A"] for b in r["lines"]}
        missing = [key for key in have if key not in idx]
        wrong = [key for key in have if key in idx and idx[key]["adf15"]["wavelength_A"] != have[key]]
        extra = [key for key in idx if key[0] == r["spec_file"] and key not in have]
        check(f"(a) {STAGE_SPECIES[k]} adf15 lines in the JSON",
              not (missing or wrong or extra),
              f"{len(have)} lines; missing {missing}, wavelength mismatch {wrong}, "
              f"JSON-only {extra}")

    for tab, k in ((tables[0], "he1"), (tables[1], "he0")):
        total = red["res"][k]["machine_W_total"]
        adf = [row for row in tab.rows if not row["estimated"]]
        Psum = math.fsum(row["power_W"] for row in adf)
        Ssum = math.fsum(row["share"] for row in adf)
        worst = max(abs(row["share"] * total - row["power_W"]) / row["power_W"] for row in adf)
        rel = abs(Psum / total - 1.0)
        check(f"(b) {STAGE_SPECIES[k]} power column vs evaluate_stage total",
              rel <= SUM_RTOL, f"sum {Psum:.12e} W, total {total:.12e} W, rel {rel:.2e} (tol {SUM_RTOL:g})")
        check(f"(b) {STAGE_SPECIES[k]} share column",
              abs(Ssum - 1.0) <= SUM_RTOL and worst <= SUM_RTOL,
              f"sum {Ssum:.15f}, worst |share*total - power|/power {worst:.2e} (tol {SUM_RTOL:g})")

    worst = 0.0
    cases = [(0.0, 7.0, 1.0)] + [(x * 6.5, 6.5, _tail_closed_form(x)) for x in (0.5, 1.0, 2.0, 5.0, 10.0)]
    for E, Te, want in cases:
        got = float(tail_fraction(E, Te))
        worst = max(worst, abs(got / want - 1.0))
    check("(c) Maxwellian fraction vs closed form", worst <= TAIL_RTOL,
          f"E/T_e in (0, 0.5, 1, 2, 5, 10); worst rel {worst:.2e} (tol {TAIL_RTOL:g}); "
          f"above T_e: {_tail_closed_form(1.0):.15f}")

    for tab in tables:
        text = tab.tex()
        ncol = len(tab.colspec)
        bad = []
        depth = 0
        for ln in text.splitlines():
            if ln.startswith("%"):
                continue
            depth += ln.count("{") - ln.count("}")
            if ln.endswith(r"\\"):
                n_amp = ln.replace(r"\&", "").count("&")
                if n_amp != ncol - 1:
                    bad.append((n_amp, ln[:60]))
        check(f"(d) {tab.name}.tex rows and braces", not bad and depth == 0,
              f"{ncol} columns; bad rows {bad}; brace depth {depth}")
    return ok


# --- entry point -----------------------------------------------------------


def _parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h5", required=True, type=Path)
    ap.add_argument("--window-ms", nargs=2, type=float, default=list(LR.DEFAULT_WINDOW_MS))
    ap.add_argument("--outdir", type=Path)
    ap.add_argument("--ports", nargs="+", type=int, default=list(DEFAULT_PORTS))
    ap.add_argument("--self-test", action="store_true")
    return ap


def main(argv=None):
    args = _parser().parse_args(argv)
    if not args.self_test and args.outdir is None:
        raise SystemExit("--outdir is required unless --self-test is given")
    atomic = load_atomic()
    red = reduce_run(args.h5, args.window_ms, args.ports)
    tables = build_tables(red, atomic)
    if args.self_test:
        ok = self_test(red, atomic, tables)
        print("SELF-TEST", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    args.outdir.mkdir(parents=True, exist_ok=True)
    for tab in tables:
        (args.outdir / f"{tab.name}.tex").write_text(tab.tex())
        (args.outdir / f"{tab.name}.md").write_text(tab.md())
    (args.outdir / "tables_summary.txt").write_text(summary_text(red, atomic))
    print(f"wrote {len(tables)} tables to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
