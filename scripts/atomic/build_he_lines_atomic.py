"""Build ``scripts/data/he_lines_atomic.json`` from NIST ASD tab-delimited output.

Run from the repository root with the project environment::

    python scripts/atomic/build_he_lines_atomic.py \\
        --he1-lines HE1_LINES.tsv --he2-lines HE2_LINES.tsv \\
        --he1-levels HE1_LEVELS.tsv --he2-levels HE2_LEVELS.tsv \\
        --out scripts/data/he_lines_atomic.json [--adf15-dir DIR]

The four inputs are the ASD responses to the four queries in ``QUERIES``
(``format=3``, tab-delimited; the line queries use ``show_av=3``, vacuum
wavelengths in every range), saved as fetched.  The JSON records those
queries, the ASD version and the retrieval date of the tables it was built
from.  Nothing is typed by hand except the transition identities (which upper
and lower term each entry is) and the adf15 ISEL map, which is checked against
the wavelengths in the adf15 block headers (``--adf15-dir``, default the
package's OPEN-ADAS directory).  Every multiplet quantity is derived from the
component rows, which the JSON also carries; the conventions are written into
the JSON's ``conventions`` block.
"""
import argparse
import csv
import json
import re
from pathlib import Path

from scipy.constants import c, e, h

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ADF15_DIR = REPO / "cablp" / "atomic" / "data" / "adas"
HC_EV_NM = h * c / e * 1e9

ASD_VERSION = "5.12"
RETRIEVED = "2026-10-06"

_LINES_Q = (
    "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra={sp}&limits_type=0&low_w=20"
    "&upp_w=1100&unit=1&de=0&format=3&line_out=0&remove_js=on&en_unit=1&output=0"
    "&page_size=15&show_obs_wl=1&show_calc_wl=1&order_out=0&show_av=3&tsb_value=0"
    "&A_out=0&allowed_out=1&forbid_out=1&conf_out=on&term_out=on&enrg_out=on&J_out=on"
    "&g_out=on&submit=Retrieve+Data"
)
_LEVELS_Q = (
    "https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0&spectrum={sp}"
    "&submit=Retrieve+Data&units=1&format=3&output=0&page_size=15&multiplet_ordered=0"
    "&conf_out=on&term_out=on&level_out=on&unc_out=1&j_out=on&g_out=on&lande_out=on"
    "&perc_out=on&biblio=on&temp="
)
#: The ASD queries whose responses the four input tables are.
QUERIES = {
    "He I lines": _LINES_Q.format(sp="He+I"),
    "He II lines": _LINES_Q.format(sp="He+II"),
    "He I levels": _LEVELS_Q.format(sp="He+I"),
    "He II levels": _LEVELS_Q.format(sp="He+II"),
}

LINES = {}
LEVELS = {}


def strip(x):
    return x.strip().strip('"').strip()


def num(x):
    x = strip(x).strip("[]()")
    return float(x) if x else None


def read_tsv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    head = [strip(h_) for h_ in rows[0]]
    out = []
    seen = set()
    for r in rows[1:]:
        if not any(strip(x) for x in r):
            continue
        d = {head[i]: strip(r[i]) for i in range(min(len(head), len(r)))}
        key = tuple(sorted(d.items()))
        if key in seen:  # ASD repeats a few rows verbatim
            continue
        seen.add(key)
        out.append(d)
    return out


def he2_n(conf):
    m = re.fullmatch(r"(\d+)[a-z]", conf)
    return int(m.group(1)) if m else None


def he2_match_level(n):
    return lambda lv: he2_n(lv.get("Configuration", "")) == n


def he1_match_level(conf, term):
    return lambda lv: lv.get("Configuration") == conf and lv.get("Term") == term


def he2_match_line(nu, nl):
    return lambda r: he2_n(r["conf_k"]) == nu and he2_n(r["conf_i"]) == nl


def he1_match_line(cu, tu, cl, tl):
    return lambda r: (
        r["conf_k"] == cu and r["term_k"] == tu and r["conf_i"] == cl and r["term_i"] == tl
    )


def level_centroid(species, match):
    lv = [x for x in LEVELS[species] if match(x) and num(x.get("Level (eV)", "")) is not None]
    g = [int(x["g"]) for x in lv]
    E = [num(x["Level (eV)"]) for x in lv]
    if not lv:
        return None, None, 0
    G = sum(g)
    return sum(gi * Ei for gi, Ei in zip(g, E)) / G, G, len(lv)


def multiplet(species, line_match, level_match_u, level_match_l):
    comps = [r for r in LINES[species] if line_match(r)]
    Eu, gU, nlev_u = level_centroid(species, level_match_u)
    El, gL, _ = level_centroid(species, level_match_l)
    rec = {
        "upper_level_eV": Eu,
        "lower_level_eV": El,
        "upper_statistical_weight": gU,
        "n_components": len(comps),
        "components": [],
        "einstein_A_s": None,
        "wavelength_vac_nm": None,
        "photon_eV": None,
    }
    if not comps:
        return rec
    gA_sum = 0.0
    gA_lam = 0.0
    for r in comps:
        lam = num(r["ritz_wl_vac(nm)"]) or num(r["obs_wl_vac(nm)"])
        A = num(r["Aki(s^-1)"])
        gk = int(r["g_k"])
        rec["components"].append(
            {
                "lower": f'{r["conf_i"]} {r["term_i"]} J={r["J_i"]}',
                "upper": f'{r["conf_k"]} {r["term_k"]} J={r["J_k"]}',
                "g_lower": int(r["g_i"]),
                "g_upper": gk,
                "wavelength_vac_nm": lam,
                "wavelength_kind": "ritz" if num(r["ritz_wl_vac(nm)"]) else "observed",
                "A_s": A,
                "accuracy": r.get("Acc", ""),
                "type": r.get("Type", "") or "E1",
            }
        )
        if A is not None:
            gA_sum += gk * A
            gA_lam += gk * A * lam
    if gA_sum > 0 and gU:
        rec["einstein_A_s"] = gA_sum / gU
        rec["wavelength_vac_nm"] = gA_lam / gA_sum
        rec["photon_eV"] = HC_EV_NM / rec["wavelength_vac_nm"]
    return rec


def he1_term(conf, term):
    n = int(re.search(r"(\d+)[a-z]$", conf).group(1)) if conf != "1s2" else 1
    mult = int(term[0])
    L = term[1]
    return {"config": conf, "term": term, "n": n, "multiplicity": mult, "L": L}


# (key, n_upper, n_lower, adf15 ISEL or None)
HE2 = [
    (2, 1, 1), (3, 1, 2), (4, 1, 3), (5, 1, 4), (3, 2, 5), (4, 2, 6), (5, 2, 7),
    (4, 3, 8), (5, 3, 9), (6, 3, None), (5, 4, None),
]
# (upper conf, upper term, lower conf, lower term, adf15 ISEL)
HE1 = [
    ("1s.2p", "1P*", "1s2", "1S", 1),
    ("1s.3p", "1P*", "1s2", "1S", 2),
    ("1s.4p", "1P*", "1s2", "1S", 3),
    ("1s.3p", "1P*", "1s.2s", "1S", 4),
    ("1s.4p", "1P*", "1s.2s", "1S", 5),
    ("1s.3p", "3P*", "1s.2s", "3S", 6),
    ("1s.4p", "3P*", "1s.2s", "3S", 7),
    ("1s.3s", "3S", "1s.2p", "3P*", 8),
    ("1s.3d", "3D", "1s.2p", "3P*", 9),
    ("1s.4s", "3S", "1s.2p", "3P*", 10),
    ("1s.4d", "3D", "1s.2p", "3P*", 11),
    ("1s.3s", "1S", "1s.2p", "1P*", 12),
    ("1s.3d", "1D", "1s.2p", "1P*", 13),
    ("1s.4s", "1S", "1s.2p", "1P*", 14),
    ("1s.4d", "1D", "1s.2p", "1P*", 15),
]
ADF15 = {"He II": "pec96_he_pju_he1.dat", "He I": "pec96_he_pju_he0.dat"}


def adf15_wavelengths(adf15_dir, fname):
    """ISEL -> wavelength [A] read from the adf15 block headers."""
    path = Path(adf15_dir) / fname
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*([0-9.]+)\s*A\s+\d+\s+\d+\s*/.*TYPE\s*=\s*EXCIT.*ISEL\s*=\s*(\d+)", line)
        if m:
            out[int(m.group(2))] = float(m.group(1))
    return out


def _parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--he1-lines", required=True, type=Path, help="ASD He I lines table")
    ap.add_argument("--he2-lines", required=True, type=Path, help="ASD He II lines table")
    ap.add_argument("--he1-levels", required=True, type=Path, help="ASD He I levels table")
    ap.add_argument("--he2-levels", required=True, type=Path, help="ASD He II levels table")
    ap.add_argument("--out", required=True, type=Path, help="output JSON path")
    ap.add_argument("--adf15-dir", type=Path, default=DEFAULT_ADF15_DIR,
                    help="directory holding the two adf15 files")
    return ap


def main(argv=None):
    args = _parser().parse_args(argv)
    LINES.update({"He I": read_tsv(args.he1_lines), "He II": read_tsv(args.he2_lines)})
    LEVELS.update({"He I": read_tsv(args.he1_levels), "He II": read_tsv(args.he2_levels)})
    out_path = args.out
    adf = {sp: adf15_wavelengths(args.adf15_dir, f) for sp, f in ADF15.items()}
    trans = []
    for nu, nl, isel in HE2:
        rec = multiplet("He II", he2_match_line(nu, nl), he2_match_level(nu), he2_match_level(nl))
        assert rec["upper_statistical_weight"] == 2 * nu * nu, (nu, rec["upper_statistical_weight"])
        entry = {
            "key": f"HeII_{nu}-{nl}",
            "species": "He II",
            "upper": {"n": nu},
            "lower": {"n": nl},
            "in_adf15": isel is not None,
            "adf15": None if isel is None else {
                "file": ADF15["He II"], "isel": isel, "wavelength_A": adf["He II"][isel]},
            "metastable": None,
            "metastable_offset_eV": None,
            "einstein_A_origin": "NIST ASD He II line list (the ion's own entries)",
        }
        entry.update(rec)
        trans.append(entry)
    meta = {}
    for conf, term in (("1s.2s", "3S"), ("1s.2s", "1S")):
        E, g, _ = level_centroid("He I", he1_match_level(conf, term))
        meta[term[0]] = (conf, term, E)
    for cu, tu, cl, tl, isel in HE1:
        rec = multiplet("He I", he1_match_line(cu, tu, cl, tl), he1_match_level(cu, tu), he1_match_level(cl, tl))
        mconf, mterm, mE = meta[tu[0]]
        assert rec["upper_statistical_weight"] == int(tu[0]) * (2 * "SPDFG".index(tu[1]) + 1), (cu, tu)
        entry = {
            "key": f"HeI_{cu}_{tu}-{cl}_{tl}".replace("*", "o").replace(".", ""),
            "species": "He I",
            "upper": he1_term(cu, tu),
            "lower": he1_term(cl, tl),
            "in_adf15": True,
            "adf15": {"file": ADF15["He I"], "isel": isel, "wavelength_A": adf["He I"][isel]},
            "metastable": he1_term(mconf, mterm),
            "metastable_offset_eV": None,
            "einstein_A_origin": "NIST ASD He I line list",
        }
        entry.update(rec)
        entry["metastable_offset_eV"] = entry["upper_level_eV"] - mE
        trans.append(entry)
    for t in trans:
        if t["adf15"]:
            d = abs(t["adf15"]["wavelength_A"] / 10 - t["wavelength_vac_nm"])
            assert d < 0.2, (t["key"], t["adf15"], t["wavelength_vac_nm"])
    metastables = []
    for mult in ("3", "1"):
        conf, term, E = meta[mult]
        rec = multiplet("He I", he1_match_line(conf, term, "1s2", "1S"),
                        he1_match_level(conf, term), he1_match_level("1s2", "1S"))
        m = {"species": "He I", "level": he1_term(conf, term), "level_eV": E,
             "decay_to": he1_term("1s2", "1S")}
        m.update({k: rec[k] for k in ("einstein_A_s", "wavelength_vac_nm", "n_components", "components")})
        if rec["n_components"] == 0:
            m["einstein_A_note"] = ("null: the ASD line list returned no 1s.2s 1S - 1s2 1S entry "
                                    "(the decay is two-photon and carries no single wavelength)")
        metastables.append(m)
    doc = {
        "description": (
            "Atomic data for the He II and He I transitions the line-power tables report: "
            "every EXCIT transition in the two OPEN-ADAS adf15 files the model's line "
            "decomposition reads, two He II transitions those files lack, and the two He I "
            "metastables."),
        "source": {
            "database": "NIST Atomic Spectra Database",
            "version": ASD_VERSION,
            "citation": ("Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team (2024). "
                         f"NIST Atomic Spectra Database (ver. {ASD_VERSION}). National Institute of "
                         "Standards and Technology, Gaithersburg, MD."),
            "doi": "https://doi.org/10.18434/T4W30F",
            "retrieved": RETRIEVED,
            "queries": dict(QUERIES),
        },
        "conventions": {
            "wavelength_vac_nm": ("vacuum; the g_k A_ki-weighted mean of the multiplet's "
                                  "component Ritz wavelengths (observed where ASD gives no Ritz value)"),
            "photon_eV": "h c / wavelength_vac_nm, CODATA h, c, e (scipy.constants)",
            "einstein_A_s": ("sum over the lower components and statistical-weight average over "
                             "the upper ones: sum_k sum_i g_k A_ki / g_upper, with g_upper the summed "
                             "weight of every component of the upper term (2 n^2 for a He II shell, "
                             "so orbital states with no decay to the lower shell count in g_upper)"),
            "upper_level_eV": ("g-weighted centroid of the upper term's (He II: shell's) ASD levels, "
                               "above the species' own ground state"),
            "metastable_offset_eV": ("He I only: upper_level_eV minus the level of the same-spin "
                                     "metastable (2 3S for triplets, 2 1S for singlets)"),
            "he2_einstein_A": ("NIST ASD's own He II entries (accuracy class AA), not Z^4-scaled "
                               "hydrogen values"),
        },
        "transitions": trans,
        "metastables": metastables,
    }
    out_path.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
