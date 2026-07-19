# ADAS adf11 data for helium

Iso-nuclear master files from OPEN-ADAS (https://open.adas.ac.uk), ADAS
"GCR project" '96 series, unresolved (stage-to-stage) form. Retrieved
2026-07-18 from `https://open.adas.ac.uk/download/adf11/<class>96/<class>96_he.dat`.

| file | class | quantity | units | normalization |
|---|---|---|---|---|
| `scd96_he.dat` | SCD | effective ionization coefficient, stage z1-1 -> z1 | cm^3 s^-1 | per n_e * n(z1-1) |
| `acd96_he.dat` | ACD | effective recombination coefficient, stage z1 -> z1-1 (radiative + dielectronic + three-body) | cm^3 s^-1 | per n_e * n(z1) |
| `plt96_he.dat` | PLT | line power driven by excitation of ions of charge z1-1 | W cm^3 | per n_e * n(z1-1) |
| `prb96_he.dat` | PRB | recombination + bremsstrahlung power of ions of charge z1 | W cm^3 | per n_e * n(z1) |

For helium (Z = 2) each file carries stages z1 = 1, 2. The transport model
uses z1 = 1 of SCD/ACD/PLT/PRB (He0 ionization, He+ recombination, He0 line
power, He+ recombination power) and z1 = 2 of PLT (He+ line power).

These are generalized collisional-radiative (GCR) coefficients: they are
tabulated on a log10(n_e) x log10(T_e) grid (24 x 30; 5e7-2e15 cm^-3,
0.2-1.5e4 eV) and include finite-density effects (stepwise ionization via
metastables, collisional de-excitation), which the coronal Janev-era fits in
`cablp.funcs._fits` / `cablp.funcs._cross` do not. Selected by the sim1d
`atomic_rate_model = "adas"` input; the historical fits remain available as
`"janev"` (the default).

File format: adf11 (see `cablp.funcs._adas.read_adf11`). All tabulated
values are log10 of the coefficient in the units above.

Citation: H.P. Summers, "The ADAS User Manual, version 2.6" (2004),
http://www.adas.ac.uk; data via OPEN-ADAS, ADAS Project.
