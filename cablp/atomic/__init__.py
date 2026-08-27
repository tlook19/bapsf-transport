"""Atomic data and reaction physics for helium (and the retained hydrogen arms).

Members are imported by their own module path -- this package deliberately
re-exports nothing, so there is exactly one name for every symbol.

- :mod:`cablp.atomic.cross_sections` -- Janev/Phelps cross sections and the
  Maxwellian rate integrals.
- :mod:`cablp.atomic.fits` -- IAEA empirical reaction-rate coefficient fits.
- :mod:`cablp.atomic.coefficients` -- the IAEA expression coefficients those
  fits evaluate.
- :mod:`cablp.atomic.adas` -- OPEN-ADAS adf11/adf15 readers and the GCR
  effective coefficients.

``data/`` holds the tabulated inputs these modules read at import: the EII
cross-section CSVs, the helium ionization-rate table, the Phelps LXCat
momentum-transfer text, and ``data/adas/``, whose adf11/adf15 master files are
untracked by licence (see ``data/adas/README.md``).
"""
