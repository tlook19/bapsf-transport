# ADAS data for helium, carbon, oxygen, boron, molybdenum, and tungsten

Two ADAS data formats live here: adf11 iso-nuclear master files (the bulk of
the directory, consumed by the transport model) and, for helium only, two
adf15 photon-emissivity-coefficient files (diagnostic-side, see the adf15
section at the end).

## adf11 — iso-nuclear master files

Iso-nuclear master files from OPEN-ADAS (https://open.adas.ac.uk),
unresolved (stage-to-stage) form, from
`https://open.adas.ac.uk/download/adf11/<class><yy>/<class><yy>_<element>.dat`.
Helium ('96 GCR series) retrieved 2026-07-18; carbon/oxygen ('96) and
boron/molybdenum/tungsten ('89 Abels-van Maanen series — no '96 files
exist for these) retrieved 2026-07-21. The '89 series is older average-ion-era
data; treat it as order-of-magnitude at LAPD temperatures. Lanthanum
(the other LaB6 constituent) has NO adf11 data on OPEN-ADAS in any
series; tungsten is kept as the heavy-element analog for La-class
radiators.

| file | class | quantity | units | normalization |
|---|---|---|---|---|
| `scd96_he.dat` | SCD | effective ionization coefficient, stage z1-1 -> z1 | cm^3 s^-1 | per n_e * n(z1-1) |
| `acd96_he.dat` | ACD | effective recombination coefficient, stage z1 -> z1-1 (radiative + dielectronic + three-body) | cm^3 s^-1 | per n_e * n(z1) |
| `plt96_he.dat` | PLT | line power driven by excitation of ions of charge z1-1 | W cm^3 | per n_e * n(z1-1) |
| `prb96_he.dat` | PRB | recombination + bremsstrahlung power of ions of charge z1 | W cm^3 | per n_e * n(z1) |

For helium (Z = 2) each file carries stages z1 = 1, 2. The transport model
uses z1 = 1 of SCD/ACD/PLT/PRB (He0 ionization, He+ recombination, He0 line
power, He+ recombination power) and z1 = 2 of PLT (He+ line power).

The carbon (`*96_c.dat`, z1 = 1-6), oxygen (`*96_o.dat`, z1 = 1-8), boron
(`*89_b.dat`, z1 = 1-5), molybdenum (`*89_mo.dat`, z1 = 1-42), and
tungsten (`*89_w.dat`, z1 = 1-74) files were
added for the impurity-radiation scoping study
(`scripts/scope_impurity_radiation.py`): equilibrium stage balance from
SCD/ACD, total radiated power L_z from PLT+PRB. No model path consumes them
as of 2026-07-21 — the scoping verdict (required n_z/n_e ~ 4-10 % at
equilibrium for every species tested >> the ppm hypothesis) stopped the
campaign before any sink term was wired in.

These are generalized collisional-radiative (GCR) coefficients: they are
tabulated on a log10(n_e) x log10(T_e) grid (24 x 30; 5e7-2e15 cm^-3,
0.2-1.5e4 eV) and include finite-density effects (stepwise ionization via
metastables, collisional de-excitation), which the coronal Janev-era fits in
`cablp.funcs._fits` / `cablp.funcs._cross` do not. Selected by the sim1d
`atomic_rate_model = "adas"` input; the historical fits remain available as
`"janev"` (the default).

File format: adf11 (see `cablp.funcs._adas.read_adf11`). All tabulated
values are log10 of the coefficient in the units above.

## adf15 — helium photon emissivity coefficients (PEC)

Line-resolved PECs from the OPEN-ADAS `pec96#he` series, `pju` (unresolved,
projected-to-unresolved) variant — the variant that matches the unresolved
adf11 masters above. The metastable-**resolved** `pjr` sibling is
deliberately NOT held here: it would not compose with the unresolved PLT.
Both files retrieved 2026-08-26. OPEN-ADAS serves them under names containing
`#`, which is awkward in shell and Python paths, so they are stored with `#`
replaced by `_`; the canonical name is given below and is also recoverable
from each file's own trailing SCCS header line.

| file | canonical name | URL |
|---|---|---|
| `pec96_he_pju_he0.dat` | `pec96#he_pju#he0.dat` | `https://open.adas.ac.uk/download/adf15/pec96][he/pec96][he_pju][he0.dat` |
| `pec96_he_pju_he1.dat` | `pec96#he_pju#he1.dat` | `https://open.adas.ac.uk/download/adf15/pec96][he/pec96][he_pju][he1.dat` |

(The `][` in the URLs is OPEN-ADAS's own escaping of `#` in its download
paths; `curl` needs `-g/--globoff` to fetch them literally.)

Produced by population code ADAS208, producer M. O'Mullane, dated
01.11.1999. `he0` (He I) carries 15 transitions, `he1` (He II) 9, each
tabulated for both `EXCIT` (excitation-driven) and `RECOM`
(recombination-driven) emission, so 30 and 18 blocks respectively.
Wavelengths are in Angstrom, VACUUM. PEC units are photons cm^3 s^-1, to be
multiplied by n_e * n(emitting stage). Values are tabulated linearly (NOT
log10, unlike adf11) on a 24 x 24 (n_e, T_e) grid, and **the two adf15 grids
differ from each other and from the shared adf11 grid** — `he0` spans
1e1-1e15 cm^-3 and 0.0431-259 eV, `he1` spans 1.28e3-1.28e17 cm^-3 and
0.172-1030 eV — so each file must be interpolated on its own axes. A value
of 1.00E-74 is the format's zero sentinel.

No model path consumes these: they were added for the window-transmission
band-fraction analysis (`scripts/pec_band_fractions.py`), which splits the
adf11 PLT line power into optical bands to establish what fraction of the
He radiated power is observable through the LAPD port windows.

Citation: H.P. Summers, "The ADAS User Manual, version 2.6" (2004),
http://www.adas.ac.uk; data via OPEN-ADAS, ADAS Project.
