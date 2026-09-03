#!/usr/bin/env python
"""Split the He adf11 line power into optical bands using adf15 PECs.

Answers one question: of the He line power the transport model books through
the ADAS PLT coefficients, what fraction actually reaches a detector through
an LAPD port window? PLT is a single number per stage -- it carries no
spectral information -- so the split has to come from the line-resolved adf15
photon emissivity coefficients (`pec96#he_pju#he{0,1}.dat`, see the README in
``cablp/atomic/data/adas``).

Method, per emitting stage:

  * take the EXCIT blocks only. PLT is the EXCITATION-driven line power; the
    recombination-driven emission tabulated in the RECOM blocks is booked by
    the model in the PRB channel instead, so mixing RECOM PECs in here would
    double-count against PLT.
  * convert each line's PEC [photons cm^3 s^-1] to a power coefficient
    P_i = PEC_i * (hc / lambda_vac) [eV cm^3 s^-1].
  * bin the lines by vacuum wavelength into the four window bands and report
    each band's share of Sum_i P_i and of the repo's own adf11 PLT
    interpolator.

The ratio C = Sum_all / PLT is a completeness check on the adf15 file, not a
physics result: the adf15 file carries a hand-picked line list, PLT carries
the whole spectrum. C ~ 1 says the file's lines are the radiator; C < 1 says
they are not, and then the observable fraction is only bracketed (see the
He0 bracket printed below).
"""

import math
import re
from pathlib import Path

import numpy as np

from cablp.atomic.adas import (
    ADAS_DIR,
    he_ion_line_power,
    he_neutral_line_power,
)

# hc in eV nm, so P_i [eV cm^3/s] = PEC_i [ph cm^3/s] * HC_EV_NM / lambda_nm.
HC_EV_NM = 1239.842

NE_GRID_CM3 = (1.0e12, 3.0e12, 1.0e13)
TE_GRID_EV = (1.0, 2.0, 3.0, 5.0, 10.0)

# LAPD port-window transmission bands, by VACUUM wavelength. Half-open
# [lo, hi) so every line lands in exactly one band.
BANDS = (
    ("<170 nm", None, 170.0, "no window"),
    ("170-300 nm", 170.0, 300.0, "fused silica only"),
    ("300-350 nm", 300.0, 350.0, "marginal borosilicate"),
    (">350 nm", 350.0, None, "glass-transmissible"),
)
BAND_NAMES = tuple(b[0] for b in BANDS)
VISIBLE_BAND = ">350 nm"

SPECIES = (
    {
        "key": "he0",
        "label": "He0 (He I)",
        "file": "pec96_he_pju_he0.dat",
        "n_excit": 15,
        "plt_fn": he_neutral_line_power,
        "plt_label": "he_neutral_line_power (adf11 PLT, z1=1)",
    },
    {
        "key": "he1",
        "label": "He+ (He II)",
        "file": "pec96_he_pju_he1.dat",
        "n_excit": 9,
        "plt_fn": he_ion_line_power,
        "plt_label": "he_ion_line_power (adf11 PLT, z1=2)",
    },
)

# "  4686.5 A  24  24 /FILMEM = n#he1   /TYPE = EXCIT    /INDM = T/ISEL =  8"
_BLOCK_HEADER = re.compile(
    r"^\s*([0-9.]+)\s*A\s+(\d+)\s+(\d+)\s*/"
    r".*?/\s*TYPE\s*=\s*(\S+)"
    r".*?/\s*ISEL\s*=\s*(\d+)"
)


def _missing_data_file_message(path):
    """Return the fetch-instruction text for a missing OPEN-ADAS data file.

    Mirrors ``cablp.atomic.adas._missing_data_file_message`` so this script
    fails the same loud way on a files-absent clone, worded for the adf15
    PEC files it reads rather than the adf11 masters.
    """
    return (
        f"OPEN-ADAS data file not found: {path}\n"
        "The ADAS .dat files are NOT tracked in this repository -- OPEN-ADAS's "
        "terms forbid redistributing them on a public website -- so the adf15 "
        "PEC files this script reads must be fetched by hand into "
        "cablp/atomic/data/adas/ before it can run.\n"
        "See cablp/atomic/data/adas/README.md for the per-file download URL, the "
        "local filename to save as, and the checksum to verify."
    )


def read_adf15(path):
    """Parse an adf15 photon-emissivity-coefficient file.

    Returns a list of block dicts with keys ``isel``, ``wavelength_A``,
    ``kind`` (``EXCIT``/``RECOM``), ``log_ne``, ``log_te`` and ``log_pec``.
    ``log_pec`` is ``(nte, ndens)`` -- transposed from the file's own
    density-slow/temperature-fast storage into the ``table[iy, ix]``
    convention the interpolator below shares with ``cablp.atomic.adas``.

    Unlike adf11, adf15 tabulates the coefficient LINEARLY; the log10 is
    taken here so the interpolation is log-log. 1.00E-74 is the format's
    zero sentinel and survives as log10 = -74.
    """
    try:
        text = Path(path).read_text()
    except FileNotFoundError as exc:
        raise RuntimeError(_missing_data_file_message(path)) from exc
    lines = text.splitlines()
    n_expected = int(lines[0].split()[0])

    blocks = []
    i = 1
    while i < len(lines):
        line = lines[i]
        if line.startswith("C"):
            break  # trailing comment section
        match = _BLOCK_HEADER.match(line)
        if match is None:
            i += 1
            continue
        wavelength_A = float(match.group(1))
        ndens, nte = int(match.group(2)), int(match.group(3))
        kind, isel = match.group(4), int(match.group(5))

        need = ndens + nte + ndens * nte
        values = []
        i += 1
        while i < len(lines) and len(values) < need:
            nxt = lines[i]
            if nxt.startswith("C") or _BLOCK_HEADER.match(nxt):
                break
            values.extend(float(tok) for tok in nxt.split())
            i += 1
        if len(values) != need:
            raise ValueError(
                f"{path}: ISEL {isel} block holds {len(values)} values, "
                f"expected {need}"
            )

        ne = np.array(values[:ndens], dtype=float)
        te = np.array(values[ndens : ndens + nte], dtype=float)
        pec = np.array(values[ndens + nte :], dtype=float).reshape(ndens, nte)

        if not (np.all(np.diff(ne) > 0) and np.all(np.diff(te) > 0)):
            raise ValueError(f"{path}: ISEL {isel} axes are not increasing")

        log_pec = np.log10(pec)
        # ORIENTATION GUARD. ndens == nte == 24 in both files, so a
        # transposed reshape would pass every shape check and silently
        # return PEC(Te, ne) swapped. It cannot pass this one: an
        # EXCITATION PEC climbs tens of decades across the Te axis and at
        # most ~1 decade across the ne axis. Checked on the EXCIT blocks
        # only -- orientation is a property of the file, these are the
        # blocks this script consumes, and a RECOM PEC's Te dependence is
        # far too gentle (~2 decades) to discriminate the two axes.
        if kind == "EXCIT":
            span_te = float(np.ptp(log_pec[0, :]))  # Te sweep, lowest ne
            span_ne = float(np.ptp(log_pec[:, -1]))  # ne sweep, highest Te
            if not span_te > 10.0 * max(span_ne, 1.0):
                raise ValueError(
                    f"{path}: ISEL {isel} PEC block does not vary "
                    f"temperature-fast (Te span {span_te:.1f} decades, ne "
                    f"span {span_ne:.1f}); the value ordering is not what "
                    "this parser assumes"
                )

        blocks.append(
            {
                "isel": isel,
                "wavelength_A": wavelength_A,
                "kind": kind,
                "log_ne": np.log10(ne),
                "log_te": np.log10(te),
                "log_pec": log_pec.T,
            }
        )

    if len(blocks) != n_expected:
        raise ValueError(
            f"{path}: header declares {n_expected} blocks, parsed {len(blocks)}"
        )
    return blocks


def _interp_coords(log_x_grid, log_y_grid, log_x, log_y):
    """Clamped bilinear indices and weights on one file's own axes."""
    x = np.clip(log_x, log_x_grid[0], log_x_grid[-1])
    y = np.clip(log_y, log_y_grid[0], log_y_grid[-1])
    ix = np.clip(np.searchsorted(log_x_grid, x) - 1, 0, log_x_grid.size - 2)
    iy = np.clip(np.searchsorted(log_y_grid, y) - 1, 0, log_y_grid.size - 2)
    fx = (x - log_x_grid[ix]) / (log_x_grid[ix + 1] - log_x_grid[ix])
    fy = (y - log_y_grid[iy]) / (log_y_grid[iy + 1] - log_y_grid[iy])
    return ix, iy, fx, fy


def _interp_blend(table, ix, iy, fx, fy):
    """Blend ``table[y, x]`` at precomputed bilinear coordinates."""
    return (
        table[iy, ix] * (1.0 - fy) * (1.0 - fx)
        + table[iy, ix + 1] * (1.0 - fy) * fx
        + table[iy + 1, ix] * fy * (1.0 - fx)
        + table[iy + 1, ix + 1] * fy * fx
    )


def pec_at(block, ne_cm3, Te_eV):
    """Interpolate one transition's PEC [photons cm^3/s] at (ne, Te).

    Bilinear in (log10 ne, log10 Te) on THIS BLOCK'S OWN grid, clamped at the
    edges. The arithmetic deliberately mirrors ``cablp.atomic.adas`` so the
    PEC side and the PLT side of every ratio below share one interpolation
    convention -- but the grid must not be shared with it: the two adf15
    grids differ from each other and from the adf11 grid.
    """
    ix, iy, fx, fy = _interp_coords(
        block["log_ne"],
        block["log_te"],
        math.log10(ne_cm3),
        math.log10(Te_eV),
    )
    return 10.0 ** _interp_blend(block["log_pec"], ix, iy, fx, fy)


def band_of(wavelength_nm):
    """Return the window-band name holding a vacuum wavelength [nm]."""
    for name, lo, hi, _ in BANDS:
        if (lo is None or wavelength_nm >= lo) and (
            hi is None or wavelength_nm < hi
        ):
            return name
    raise ValueError(f"{wavelength_nm} nm falls outside every band")


def excitation_lines(spec):
    """Return this species' EXCIT blocks, count-checked, wavelength-sorted."""
    blocks = read_adf15(ADAS_DIR / spec["file"])
    excit = [b for b in blocks if b["kind"] == "EXCIT"]
    if len(excit) != spec["n_excit"]:
        raise ValueError(
            f"{spec['file']}: expected {spec['n_excit']} EXCIT transitions, "
            f"found {len(excit)}"
        )
    for block in excit:
        block["wavelength_nm"] = block["wavelength_A"] / 10.0
        block["band"] = band_of(block["wavelength_nm"])
    return sorted(excit, key=lambda b: b["wavelength_nm"])


def evaluate(spec, lines):
    """Band sums, total and PLT for every (ne, Te) on the report grid."""
    rows = []
    for ne in NE_GRID_CM3:
        for Te in TE_GRID_EV:
            per_band = dict.fromkeys(BAND_NAMES, 0.0)
            for block in lines:
                power = pec_at(block, ne, Te) * (
                    HC_EV_NM / block["wavelength_nm"]
                )
                per_band[block["band"]] += power
            total = sum(per_band.values())
            plt = float(spec["plt_fn"](ne, Te))
            rows.append(
                {
                    "ne": ne,
                    "Te": Te,
                    "band": per_band,
                    "total": total,
                    "plt": plt,
                    "completeness": total / plt,
                }
            )
    return rows


def fmt(x):
    """Compact fixed/scientific formatting for a fraction or coefficient."""
    if x == 0.0:
        return "0"
    if 1.0e-3 <= abs(x) < 1.0e4:
        return f"{x:.4f}"
    return f"{x:.3e}"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out


def report(spec, lines, rows):
    """Render one species' section as a list of markdown/stdout lines."""
    out = [f"## {spec['label']} -- `{spec['file']}`", ""]

    out.append(
        f"{len(lines)} EXCIT transitions (RECOM blocks excluded: that "
        "emission belongs to the PRB channel, not PLT)."
    )
    out.append("")
    out.extend(
        _table(
            ["ISEL", "lambda_vac [A]", "lambda_vac [nm]", "band"],
            [
                [
                    str(b["isel"]),
                    f"{b['wavelength_A']:.1f}",
                    f"{b['wavelength_nm']:.2f}",
                    b["band"],
                ]
                for b in lines
            ],
        )
    )
    out.append("")

    out.append("### Band shares of Sum_all (the adf15 line list itself)")
    out.append("")
    out.extend(
        _table(
            ["ne [cm^-3]", "Te [eV]", "Sum_all [eV cm^3/s]", *BAND_NAMES],
            [
                [
                    f"{r['ne']:.0e}",
                    f"{r['Te']:g}",
                    fmt(r["total"]),
                    *(fmt(r["band"][n] / r["total"]) for n in BAND_NAMES),
                ]
                for r in rows
            ],
        )
    )
    out.append("")

    out.append(f"### Band shares of the adf11 PLT -- `{spec['plt_label']}`")
    out.append("")
    out.extend(
        _table(
            [
                "ne [cm^-3]",
                "Te [eV]",
                "PLT [eV cm^3/s]",
                "C = Sum_all/PLT",
                *BAND_NAMES,
            ],
            [
                [
                    f"{r['ne']:.0e}",
                    f"{r['Te']:g}",
                    fmt(r["plt"]),
                    fmt(r["completeness"]),
                    *(fmt(r["band"][n] / r["plt"]) for n in BAND_NAMES),
                ]
                for r in rows
            ],
        )
    )
    out.append("")

    out.append(
        f"### Glass-transmissible ({VISIBLE_BAND}) bracket -- "
        "Sum_vis/PLT to Sum_vis/Sum_all"
    )
    out.append("")
    out.append(
        "Lower bound charges the visible sum against the FULL PLT line power "
        "(every line the adf15 file omits assumed non-transmissible); upper "
        "bound charges it against the tabulated lines only (omitted lines "
        "assumed to share the tabulated visible fraction). The two collapse "
        "onto one number as C -> 1."
    )
    out.append("")
    out.extend(
        _table(
            [
                "ne [cm^-3]",
                "Te [eV]",
                "Sum_vis [eV cm^3/s]",
                "lower = Sum_vis/PLT",
                "upper = Sum_vis/Sum_all",
            ],
            [
                [
                    f"{r['ne']:.0e}",
                    f"{r['Te']:g}",
                    fmt(r["band"][VISIBLE_BAND]),
                    fmt(r["band"][VISIBLE_BAND] / r["plt"]),
                    fmt(r["band"][VISIBLE_BAND] / r["total"]),
                ]
                for r in rows
            ],
        )
    )
    out.append("")
    return out


def headline(results):
    """The two numbers this script exists to produce, over Te >= 2 eV."""
    out = ["## Headline", ""]
    for spec, _lines, rows in results:
        quotable = [r for r in rows if r["Te"] >= 2.0]
        vis_plt = [r["band"][VISIBLE_BAND] / r["plt"] for r in quotable]
        vis_tot = [r["band"][VISIBLE_BAND] / r["total"] for r in quotable]
        comp = [r["completeness"] for r in quotable]
        out.append(
            f"* **{spec['label']}** -- glass-transmissible ({VISIBLE_BAND}) "
            f"share of the adf11 line power is "
            f"{fmt(min(vis_plt))} to {fmt(max(vis_plt))} across the "
            f"report grid at Te >= 2 eV; adf15 completeness "
            f"C = {fmt(min(comp))} to {fmt(max(comp))}. Bracket "
            f"[Sum_vis/PLT, Sum_vis/Sum_all] = "
            f"[{fmt(min(vis_plt))}, {fmt(max(vis_tot))}]."
        )
    out.append("")
    return out


def main():
    results = []
    for spec in SPECIES:
        lines = excitation_lines(spec)
        results.append((spec, lines, evaluate(spec, lines)))

    out = [
        "# He line-power band fractions",
        "",
        "Regenerated by `scripts/score/pec_band_fractions.py` from the OPEN-ADAS "
        "adf15 files `pec96#he_pju#he{0,1}.dat` (EXCIT blocks only, vacuum "
        "wavelengths, hc = 1239.842 eV nm) against this repo's own adf11 PLT "
        "interpolators in `cablp/atomic/adas.py`. Do not edit by hand.",
        "",
    ]
    out.extend(headline(results))
    for spec, lines, rows in results:
        out.extend(report(spec, lines, rows))

    out.extend(
        [
            "## Reading these numbers",
            "",
            "* No interpolation cell used on this report grid touches the "
            "adf15 1e-74 zero sentinel, so every number above is an "
            "interpolation between real tabulated values.",
            "* The He+ Te = 1 eV row is still not quotable. Every He II line "
            "needs ~40 eV to excite, so that row sits in a grid cell "
            "spanning ~11 decades of PEC (0.689 -> 1.03 eV), where a "
            "log-linear interpolant stands in for a steep Arrhenius "
            "exponential -- order of magnitude at best. It is also "
            "irrelevant: at 1 eV the He+ line power is ~8 decades below the "
            "He0 line power at the same point. Quote the Te >= 5 eV rows.",
            "* Interpolation is clamped at each file's own grid edges, so a "
            "(ne, Te) outside a file's coverage returns its nearest edge "
            "value rather than an extrapolation. Both adf15 grids cover the "
            "whole report grid, so no clamping engages here.",
            "* The band edges are window-transmission boundaries, not "
            "spectroscopic ones: <170 nm has no window at all, 170-300 nm "
            "needs fused silica, 300-350 nm is marginal through "
            "borosilicate, >350 nm passes ordinary glass.",
            "",
        ]
    )

    text = "\n".join(out) + "\n"
    print(text, end="")
    dest = Path(__file__).resolve().parent / "pec_band_fractions.md"
    dest.write_text(text)
    print(f"[wrote] {dest}")


if __name__ == "__main__":
    main()
