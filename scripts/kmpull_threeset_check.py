"""Three-set verification of the shipped He e-n momentum-transfer nodes
against the LXCat TXT pull of 2026-08-13 (~/Downloads/crosssec.txt:
Biagi + IST-Lisbon + Morgan, He ELASTIC momentum transfer each).

This is the pull OF RECORD (standard LXCat TXT format; supersedes the
XML pull checked in kmpull_biagi_check.py, whose own header flagged the
XML export as under development). Shipped nodes: funcs/_cross.py
HE_EN_MT_* (5 eV and 25 eV, values + brackets).

Read-only on the repo; stdout only (tee to kmpull_threeset_check.txt).
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from cablp.funcs._cross import (  # noqa: E402
    HE_EN_MT_NODE_EV,
    HE_EN_MT_SIGMA_CM2,
    HE_EN_MT_SIGMA_BRACKET_CM2,
)

TXT = str(Path.home() / "Downloads" / "crosssec.txt")
text = open(TXT).read()

# Each DB section: "DATABASE:  <name>" ... ELASTIC block with a
# dashed-line-delimited two-column table.
sets = {}
for m in re.finditer(
    r"DATABASE:\s+(\S[^\n]*?)\n.*?ELASTIC\nHe\n.*?-{5,}\n(.*?)\n\s*-{5,}",
    text,
    re.S,
):
    name = m.group(1).split()[0].rstrip(",")
    rows = np.array(
        [
            [float(a), float(b)]
            for a, b in (ln.split() for ln in m.group(2).strip().splitlines())
        ]
    )
    sets[name] = (rows[:, 0], rows[:, 1] * 1.0e4)  # eV, cm^2

print(f"# Parsed {len(sets)} He ELASTIC momentum-transfer sets from the "
      f"TXT pull of record (retrieved 2026-08-13): {', '.join(sets)}")
print("# Pedigrees: Biagi = Magboltz 8.97 (Bartschat 1998 / Ralchenko 2008");
print("#   >30 eV); IST-Lisbon = Crompton 1970 / Milloy & Crompton 1977 /")
print("#   Register 1980 / Belmonte 2007 (the memo's low-E pedigree);")
print("#   Morgan = Kinema Research compilation.")
print()

print("== Node verification, all sets (linear interpolation, the format's")
print("   own stated convention) ==")
for node, shipped, (lo, hi) in zip(
    HE_EN_MT_NODE_EV, HE_EN_MT_SIGMA_CM2, HE_EN_MT_SIGMA_BRACKET_CM2
):
    vals = {}
    for name, (E, s) in sets.items():
        vals[name] = float(np.interp(node, E, s))
    spread_lo, spread_hi = min(vals.values()), max(vals.values())
    print(f"  {node:g} eV  (shipped {shipped:.2e}, bracket "
          f"[{lo:.2e}, {hi:.2e}] cm^2):")
    for name, v in vals.items():
        flag = "IN" if lo <= v <= hi else "OUT OF"
        print(f"    {name:11s} {v:.3e} cm^2  ({100 * (v / shipped - 1):+.1f} "
              f"% vs shipped, {flag} bracket)")
    print(f"    set-to-set spread {spread_lo:.3e} - {spread_hi:.3e} "
          f"({100 * (spread_hi / spread_lo - 1):.1f} %)")
    print()
