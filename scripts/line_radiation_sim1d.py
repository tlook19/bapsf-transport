#!/usr/bin/env python3
"""Line-resolved synthetic spectroscopy over a saved sim1d HDF5 artifact.

READ-ONLY.  Nothing is fitted, nothing is written back to the artifact, and
the solver is never constructed.  Every plasma number is a re-evaluation of
the run's OWN saved state through the run's own ADAS closure, so this
instrument cannot report a physics the run did not carry.

WHAT IS COMPUTED.  ``scripts/port_radiance_sim1d.py`` reports the two adf11
PLT channels, which carry NO spectral information -- one number per emitting
stage.  This script resolves those channels into individual transitions using
the OPEN-ADAS adf15 photon emissivity coefficients, reusing the reader,
interpolator, line list and band convention of
``scripts/pec_band_fractions.py`` unchanged:

    He I  (e-n)   eps_i = PEC_i(ne, Te) * n_e * nn      [photons cm^-3 s^-1]
    He II (e-i)   eps_i = PEC_i(ne, Te) * n_e * n_e     [photons cm^-3 s^-1]

with the per-line power ``eps_i * hc / lambda_i`` in W cm^-3.  ``nn`` is the
IN-COLUMN neutral density -- the same field ``physics/energy.py`` multiplies
into ``electron_neutral_cooling`` and the same field ``port_radiance_sim1d``
reads; where a run also carries ``nn_a`` that is the annulus density and is
NOT the emitter of the column's He I light.

EXCIT BLOCKS ONLY, and the reason is the one ``pec_band_fractions`` states:
PLT is the excitation-driven line power, while the recombination-driven
emission of the RECOM blocks is booked by the model in its PRB channel, so
mixing RECOM PECs in here would double-count against PLT.

DENSITY CONVENTION.  The PEC lookup is evaluated at ``max(n, ne_floor)``
while the density product uses the raw ``n``, mirroring
``electron_cooling_rhs_terms`` term for term so the line sum and the adf11
channel it is compared against share one convention.

CLAMP POLICY.  Both adf15 grids are interpolated bilinearly in
(log10 ne, log10 Te) and CLAMPED NEAREST-EDGE outside their own coverage,
identically to the adf11 path in ``cablp/atomic/adas.py``.  A clamped point
is not an extrapolation and not a physical value: below a grid's low Te edge
the coefficient is HELD while the true emission keeps collapsing, so a mean
containing clamped points is biased HIGH.  The census is measured per stage
per axis and printed, never left for the reader to infer.

TIME BASE.  The averaging window is on the MAIN-DISCHARGE clock, t = 0 at the
first save carrying the ``main_discharge`` phase label -- the clock
``compare_sim1d_es1.py`` scores on.  The window mean is an unweighted frame
mean over the saves that land inside it.

MACHINE TOTALS.  Per-line cell power is ``emissivity * plasma_volume_cm3``
using the artifact's own geometry volumes, summed over the plasma-active
cells.  No radial assumption enters a machine total.

CHORD AND ITS ASSUMPTION.  The model is 1D: one (ne, nn, Te) per axial cell
and no radial structure whatsoever.  A chord therefore requires an explicit
radial construction, and the honest one for a 1D model is the flattest --
emissivity RADIALLY UNIFORM across the plasma disc of radius Rp(z), zero
outside it -- so a radial line of sight through the column axis has
L_chord = 2 Rp(z) and radiance L_i = eps_i * L_chord / (4 pi).  This is a
stated construction, not a measurement, and it is repeated in every product
this script writes.

SYNTHETIC FIBER -- AN UPPER BOUND, AND IT SITS OUTSIDE THE WINDOW.  The
collection model is a BARE fiber looking into the machine THROUGH a port
window, with NO collection optics, and the imaged spot at the plasma is taken
to be approximately the fiber core width.  Two quantities follow and are
never conflated:

    collected     P_i = L_i * G                     [W, at the fiber face]
    transmitted   P_i * T_window(lambda) * T_fiber(lambda, L)   [W]

with G = A * Omega the fiber etendue, A = pi*(d/2)^2 the core area and
Omega = pi * NA^2 the acceptance solid angle; photon rate is
N_i = P_i * lambda_i / (hc).  The TRANSMITTED column is the photon-counter
number: the fiber cannot see a line its window absorbs, and the He resonance
lines that carry almost all the radiated power are exactly those lines.
T_window and T_fiber are datasheet curve readings tabulated in
``WINDOW_TRANSMISSION`` and ``FIBER_ATTENUATION``; the fiber's two silica/air
end faces are applied separately because the attenuation curve is bulk fiber
only.  This remains an UPPER BOUND: a real train loses light at every further
surface, and adding a collimating lens cannot raise it, because a lens
CONSERVES etendue -- it trades angular acceptance for collection area,
changing the field of view rather than G.  The fiber core, its NA and the
run length are MEASURED -- the FT1000UMT spec sheet and the operator's own
reading of the run -- while the window material and the attenuation curve
applied to the fiber remain ASSUMED; see ``ASSUMPTIONS``, which is emitted
verbatim into the markdown product.

WINDOW CUTOFFS.  The three 50 % transmission cutoffs drawn on the figures are
ASSUMED representative values for generic commercial parts, each carrying its
datasheet source in ``ASSUMPTIONS``.  The material actually installed on the
LAPD viewports is NOT known to this script and is not claimed.  The fiber on
the bench IS identified, but its manufacturer publishes no attenuation figure
at the 320.37 nm He II line -- the typical family curve begins near 400 nm --
so the bulk attenuation applied here is still an ASSUMED representative
curve.

    line_radiation_sim1d.py [--h5 RUN.h5] [--ports 22 27]
                            [--window-ms 15 19.5] [--output-stem STEM]
                            [--fiber-core-um 1000] [--fiber-na 0.39]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

#: Silica/air end-face transmission, one face.  DERIVED from R = ((n-1)/(n+1))^2
#: at n = 1.46 (Heraeus Suprasil n(lambda) table, 546-633 nm): R = 0.0350.
FIBER_END_FACE_TRANSMISSION = 0.965

FIBER_END_FACE_SOURCE = (
    "DERIVED from R = ((n-1)/(n+1))^2 with n from the Suprasil refractive-"
    "index table in the Heraeus datasheet below (n = 1.46008 at 546.07 nm, "
    "1.45702 at 632.8 nm; Fiberguide's own core index 1.457 at 633 nm agrees "
    "to 2e-5). Held FLAT: across 200-700 nm the true per-face value runs "
    "0.9534 to 0.9656 (two faces 0.9090 to 0.9324), so a flat two-face 0.931 "
    "is within about +/-1.3 %, far inside the fiber attenuation's own "
    "uncertainty. Wavelength-dependent Fresnel would be false precision "
    "here. Multiple-reflection form (1-R)/(1+R) differs from (1-R)^2 by "
    "0.1 % and is ignored; NA 0.22 launch angles change R by < 0.05 % "
    "absolute, so normal incidence is used."
)

#: Port-window EXTERNAL transmission curves, read off manufacturer figures.
#: ASSUMED representative parts -- the LAPD viewport material is NOT known
#: here.  Neither datasheet tabulates transmission against wavelength, so
#: every point is a curve reading or a value derived from numbers the same
#: document does tabulate; the ``tag`` on each point says which.
WINDOW_TRANSMISSION = {
    "fused_silica": {
        "material": "UV-grade fused silica (Heraeus Suprasil 1/2 Grade A)",
        "short": "UV fused silica",
        "thickness": "10 mm",
        "basis": (
            "EXTERNAL transmission, Fresnel included -- the panel header "
            "reads 'Measured transmission including Fresnel reflection "
            "losses (1-R)^2' and its legend 'Sample thickness: 10 mm'"
        ),
        "figure": (
            "Heraeus, 'Quartz Glass for Optics -- Data and Properties', "
            "HQS-SO, footer '05.2011/W1-A_E', p.7 of 8, MIDDLE panel of "
            "three; blue 'Suprasil 1 / 2 Grade A' curve; axes 'Transmission "
            "(%)' 0-100 and 'Wavelength (nm)' 150-300 linear"
        ),
        "source": (
            "https://sites.astro.caltech.edu/sedm/_downloads/"
            "2562e19ff76ec4ab03f0598d537f8428/Heraeus_SiO2-May-2011.pdf"
        ),
        "caveat": (
            "There is NO flat plateau: above about 210 nm the curve lies on "
            "the document's own (1-R)^2 Fresnel line, which RISES with "
            "dispersion from 0.909 at 200 nm to 0.934 at 1000 nm. Points at "
            "and above 193.4 nm are therefore DERIVED as (Fresnel ceiling "
            "from the tabulated n) x (tabulated internal transmission: "
            "98.50 % at 193.4 nm, 99.50 % at 248.4 nm, 99.90 % at 266 nm, "
            "10 mm) rather than read off the plot -- they are better than a "
            "curve read. The 150-190 nm points ARE curve reads, off a page "
            "raster at about 1.3 px/nm, and the VUV edge is steep: the 50 % "
            "point is 168 +/- 4 nm and the 165 nm reading alone spans "
            "0.05-0.45. A THINNER window cuts bluer; 10 mm is the "
            "datasheet's sample, not an LAPD viewport thickness."
        ),
        "opaque_below_nm": 150.0,
        "points_nm_T": (
            (150.0, 0.000),
            (160.0, 0.020),
            (165.0, 0.250),
            (170.0, 0.650),
            (175.0, 0.840),
            (180.0, 0.870),
            (190.0, 0.890),
            (193.4, 0.893),
            (200.0, 0.900),
            (248.4, 0.915),
            (266.0, 0.921),
            (350.0, 0.927),
            (500.0, 0.931),
            (700.0, 0.932),
            (1000.0, 0.934),
        ),
    },
    "borosilicate": {
        "material": "borosilicate glass (SCHOTT BOROFLOAT 33)",
        "short": "borosilicate",
        "thickness": "3.30 mm",
        "basis": (
            "external transmission, Fresnel included (INFERRED, not stated: "
            "the axis reads only 'Transmission [%]', but its 0.91-0.92 "
            "plateau sits just under the (1-R)^2 = 0.9286 ceiling computed "
            "from the sheet's own n_d = 1.47140, whereas an internal axis "
            "would plateau near 0.99)"
        ),
        "figure": (
            "SCHOTT Technical Glass Solutions, 'BOROFLOAT 33 -- Optical "
            "Properties' (2014, PDF internal title "
            "'140827_schott_b33_opt_en.indd'), p.1 lower-right panel "
            "'Transmission in UV range', 3.30 mm curve of the five-thickness "
            "legend; axes 'Transmission [%]' 0-100 and 'Wavelength lambda "
            "[nm]' 250-400. The 500 and 700 nm points come off the same "
            "sheet's coarse 0-6000 nm 'Transmission' panel."
        ),
        "source": (
            "https://www.schott.com/en-gb/products/borofloat/-/media/project/"
            "onex/products/b/borofloat/downloads/borofloat33_opt_eng_web.pdf"
        ),
        "caveat": (
            "The sheet's own tables are marked 'Reference values, not "
            "guaranteed values.' Its FIVE-THICKNESS panel is used here and "
            "is self-consistent across all five curves within reading error. "
            "Its SEPARATE 6.5 mm panel is NOT: that curve reaches 50 % at "
            "about 300 nm, bluer than both the 5.00 mm (316-320 nm) and the "
            "3.30 mm (306-312 nm) curves, which is impossible for one glass "
            "-- Beer-Lambert from the 3.30/5.00 pair predicts T(6.5 mm, "
            "310 nm) = 0.27 against the 0.68 plotted. The 6.5 mm panel is "
            "DISCARDED, in both the 2014 sheet and the older brochure that "
            "reproduces it; neither edition comments on the discrepancy. "
            "3.30 mm is the datasheet's thickness, not an LAPD viewport's."
        ),
        "opaque_below_nm": 265.0,
        "points_nm_T": (
            (265.0, 0.000),
            (270.0, 0.020),
            (280.0, 0.070),
            (290.0, 0.170),
            (300.0, 0.320),
            (303.0, 0.380),
            (310.0, 0.500),
            (320.0, 0.680),
            (350.0, 0.880),
            (400.0, 0.915),
            (500.0, 0.920),
            (700.0, 0.920),
        ),
    },
}

#: Fiber bulk attenuation, read off the manufacturer figure and converted
#: from its plotted dB/km to dB/m.  ASSUMED representative part.
FIBER_ATTENUATION = {
    "material": "solarization-resistant high-OH UV fiber (Fiberguide "
    "Solarguide, silica core / F-doped clad, hydrogen infused)",
    "basis": (
        "BULK fiber attenuation, end faces EXCLUDED -- an INFERENCE, not a "
        "datasheet statement: the sheet never says what the curve measures, "
        "but dB/km is by definition a per-length loss and spectral "
        "attenuation is conventionally measured by cut-back, which cancels "
        "end-face reflection. The two silica/air faces are therefore applied "
        "separately here."
    ),
    "figure": (
        "Fiberguide Industries, 'Solarguide Solarization Resistant UV "
        "Fiber', p.2 UPPER plot; y-axis labelled verbatim 'Attenuation "
        "(dB/km)' on a log scale, x-axis 'Wavelength (nm)' 0-1800 linear; "
        "values below divided by 1000 to reach dB/m. The plot carries NO "
        "title, NO fiber length, NO core size and NO typ/max designation."
    ),
    "source": (
        "https://shop.amstechnologies.com/media/27/3f/d4/1720716082/"
        "SolarguideTM-Solarization-Resistant-MM-Fibers-Fiberguide-"
        "Datasheetkn5dgcLgOXSSu.pdf"
    ),
    "caveat": (
        "Read off a log page raster at about 0.29 px/nm and 46 px/decade, so "
        "the readings carry +/-0.15 decade (x/ 1.4) over 250-600 nm and "
        "+/-0.25 decade on the steep 200-220 nm flank. 190 nm is "
        "UNAVAILABLE: the plotted curve begins near 195-200 nm even though "
        "the spec table claims '190nm ~ 1250nm', so this instrument sets the "
        "transmission to ZERO below 195 nm rather than extrapolate into a "
        "region the figure does not cover. The sheet's own bottom log "
        "gridline is mislabelled '0' where it must be 0.1. The 700 and "
        "1000 nm points sit on OH-absorption flanks and are the least "
        "certain (+/-0.25 and +/-0.4 decade). Separately, an un-stabilised "
        "high-OH fiber SOLARIZES and its effective UV cutoff drifts red with "
        "dose; this hydrogen-infused product is sold to resist that, but no "
        "aged curve is given. Finally, this is NOT the identified bench "
        "fiber's own curve: Thorlabs publishes no attenuation figure for the "
        "FT1000UMT at 320.37 nm and its typical family plot begins near "
        "400 nm, so a representative solarization-resistant UV curve stands "
        "in and the fiber attenuation stays ASSUMED. The FT1000UMT readings "
        "that do exist are tabulated separately, beside this one."
    ),
    "opaque_below_nm": 195.0,
    "points_nm_db_per_m": (
        (200.0, 2.9),
        (210.0, 1.8),
        (220.0, 1.3),
        (250.0, 0.65),
        (300.0, 0.23),
        (350.0, 0.13),
        (400.0, 0.08),
        (500.0, 0.033),
        (600.0, 0.015),
        (700.0, 0.02),
        (1000.0, 0.03),
    ),
}

#: The collection fiber actually on the bench, and the cells its own spec
#: sheet states.  Core diameter, NA and the run length are MEASURED from
#: here; the bulk attenuation is NOT, because the sheet publishes no figure
#: at the He II line this instrument is scanned onto -- ``FIBER_ATTENUATION``
#: above is what the transmission chain applies.
FIBER_DATASHEET = {
    "model": "Thorlabs FT1000UMT",
    "datasheet": "Thorlabs spec sheet TTN004598-S01 Rev A",
    "core_um": 1000.0,
    "core_tolerance_um": 15.0,
    "na": 0.39,
    "construction": (
        "pure silica core / TECS hard clad, Tefzel coat; operating range "
        "300-1200 nm"
    ),
    "guaranteed_attenuation": (
        "12 dB/km max at 808 nm -- the ONLY guaranteed attenuation figure on "
        "the sheet"
    ),
    "plot_start_nm": 400.0,
    "caveat": (
        "The typical-family attenuation plot BEGINS at about 400 nm, so the "
        "sheet carries NO attenuation at the 320.37 nm He II line. The "
        "readings below are reads off that typical curve, with the bracket "
        "the plot raster supports; they are not guaranteed values and they "
        "are not what this script applies. NA carries no tolerance on the "
        "sheet."
    ),
    #: (lambda [nm], read [dB/km], bracket low, bracket high, note)
    "plot_readings_db_per_km": (
        (400.0, 110.0, 78.0, 155.0, "plot reading at the curve's blue end"),
        (587.0, 13.0, 8.0, 21.0, "plot reading"),
        (706.0, 20.0, 8.0, 50.0, "plot reading, on an OH-peak edge"),
    ),
}

#: Fiber run length [m].  MEASURED: the operator's own reading of the
#: installed run, quoted as the lower bound "> 140 ft".
DEFAULT_FIBER_LENGTH_M = 43.0

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pec_band_fractions as PBF  # noqa: E402
import port_radiance_sim1d as PRS  # noqa: E402

from cablp.atomic.adas import (  # noqa: E402
    he_rate_temperature_range_eV,
    he_rates,
)
from cablp.constants import qe_SI  # noqa: E402
from cablp.solvers._sim1d.physics.kinetic_neutrals import (  # noqa: E402
    T_WALL_K,
)

#: Default subject: the kinetic-neutral arm on the corrected field profile.
DEFAULT_H5 = SCRIPT_DIR / "m1_arm2_es1.h5"

#: Plateau window on the MAIN-DISCHARGE clock [ms].
DEFAULT_WINDOW_MS = (15.0, 19.5)

#: (ion-stage port, neutral-stage port) for the synthetic fiber.
DEFAULT_PORTS = (22, 27)

#: In-repo band-split record whose rules this script inherits.
BAND_SPLIT_NOTE = "scripts/pec_band_fractions.md"

#: Below this Te the adf15 line list is not quotable as physics.
QUOTABLE_TE_MIN_EV = 2.0

#: Number of strongest lines labelled per panel in Figure A.
LABELLED_LINES = 8

STAGE = {
    "he1": {
        "spec_key": "he1",
        "label": "He II (e-i, ion stage)",
        "short": "He II",
        "partner": "n",
        "partner_label": "n_e",
        "plt_key": "plt2",
        "ledger_row": "electron_ion_cooling",
        "emitter_temp": "Ti",
    },
    "he0": {
        "spec_key": "he0",
        "label": "He I (e-n, neutral stage)",
        "short": "He I",
        "partner": "nn",
        "partner_label": "nn (in-column)",
        "plt_key": "plt1",
        "ledger_row": "electron_neutral_cooling",
        "emitter_temp": "wall",
    },
}

#: Panel order: ions first, neutrals second, as the figures are specified.
STAGE_ORDER = ("he1", "he0")


# --- provenance -----------------------------------------------------------
#
# PROVENANCE BLOCK.  Every number here that is not read from the artifact or
# from an OPEN-ADAS file.  ``class`` follows the repo's provenance vocabulary:
# MEASURED / DERIVED / FITTED / ASSUMED.  This structure is emitted verbatim
# into the markdown product, so the assumptions travel with the numbers.

ASSUMPTIONS = (
    {
        "quantity": "fiber core diameter",
        "value": (
            "1000 +/- 15 um (default); 400 and 600 um reported as comparison "
            "columns"
        ),
        "class": "MEASURED",
        "source": (
            "Thorlabs FT1000UMT, spec sheet TTN004598-S01 Rev A: core "
            "1000 +/- 15 um, pure silica core / TECS hard clad, Tefzel coat. "
            "The 400 and 600 um columns are NOT hardware -- they are kept so "
            "the core-scaling argument (collected flux ~ core^2 against "
            "f_slit ~ 1/core) stays readable. Substitutable with "
            "--fiber-core-um."
        ),
    },
    {
        "quantity": "fiber numerical aperture",
        "value": "NA = 0.39 (default); NA 0.12 reported as a comparison row",
        "class": "MEASURED",
        "source": (
            "Thorlabs FT1000UMT, spec sheet TTN004598-S01 Rev A: NA 0.39, "
            "with NO tolerance stated on the sheet. The NA 0.12 row is a "
            "comparison, not hardware. Substitutable with --fiber-na."
        ),
    },
    {
        "quantity": "acceptance solid angle Omega",
        "value": "pi * NA^2 (0.1520 sr at NA = 0.22)",
        "class": "DERIVED",
        "source": (
            "Small-angle form of the step-index acceptance cone, Omega = "
            "pi * sin^2(theta_max) with sin(theta_max) = NA in vacuum. At "
            "NA = 0.39 the exact cone 2*pi*(1 - cos(theta_max)) is 3.2 % "
            "below pi*NA^2, so the widest bracket member is the least "
            "accurate; the default and the narrow member are within 0.5 %."
        ),
    },
    {
        "quantity": "collection model",
        "value": (
            "UPPER BOUND: spot at the plasma ~ core width; no collection "
            "optics assumed"
        ),
        "class": "ASSUMED",
        "source": (
            "Stated bench assumption: light enters a bare fiber and the "
            "imaged spot at the plasma is approximately the core width, so "
            "the collection area IS the core area. Every real train loses "
            "light against this, and a collimating lens cannot beat it: a "
            "lens CONSERVES etendue, trading acceptance angle for area."
        ),
    },
    {
        "quantity": "etendue G",
        "value": "A_core * Omega (1.911e-04 cm^2 sr at 400 um / NA 0.22)",
        "class": "DERIVED",
        "source": "G = pi*(d/2)^2 * pi*NA^2, under the collection model above.",
    },
    {
        "quantity": "port window material",
        "value": "fused_silica default; borosilicate selectable (--window)",
        "class": "ASSUMED",
        "source": (
            "The fiber looks THROUGH a port window, so its datasheet "
            "external-transmission curve multiplies every line. The material "
            "actually installed on the LAPD viewports is NOT known to this "
            "script. Curve readings and sources are tabulated in the Window "
            "and fiber transmission section."
        ),
    },
    {
        "quantity": "fiber run length",
        "value": "43 m (default; --fiber-length-m)",
        "class": "MEASURED",
        "source": (
            "Operator's own reading of the installed run, quoted as the "
            "lower bound '> 140 ft'. Sets the bulk attenuation "
            "10^(-alpha L / 10), so it is one of the largest levers in the "
            "chain: attenuation readings and their source are tabulated in "
            "the Window and fiber transmission section."
        ),
    },
    {
        "quantity": "fiber end-face Fresnel loss",
        "value": (
            "T = 0.96 per silica/air face, applied twice (0.92 total)"
        ),
        "class": "DERIVED",
        "source": FIBER_END_FACE_SOURCE,
    },
    {
        "quantity": "radial emissivity profile",
        "value": "uniform across the disc of radius Rp(z), zero outside",
        "class": "ASSUMED",
        "source": (
            "Forced construction: the 1D model carries no radial structure "
            "at all, so a chord integral has no measured profile to use. "
            "This is the flattest and most honest choice, and it is a "
            "construction, not a claim about the LAPD radial profile."
        ),
    },
    {
        "quantity": "chord length",
        "value": "2 * Rp(z) from the artifact's own geometry",
        "class": "DERIVED",
        "source": (
            "Radial line of sight through the column axis under the uniform "
            "disc assumption above. Rp(z) is the run's geometry, which on "
            "the current stance comes from the measured axial field profile."
        ),
    },
    {
        "quantity": "port -> z map",
        "value": "the linear law port_radiance_sim1d.py verifies and uses",
        "class": "DERIVED",
        "source": (
            "Anchored on the committed overlay `scripts/data/"
            "es1_sim1d_overlay.npz` -- the same port/z table "
            "`compare_sim1d_es1.py` scores against -- and checked exactly "
            "collinear before the port pitch places a port the overlay does "
            "not carry. A CAD-ladder alternative exists in this repo; it is "
            "deliberately NOT used here, so a number from this instrument "
            "sits on the scorer's frame."
        ),
    },
    {
        "quantity": "hc",
        "value": "1239.842 eV nm",
        "class": "ASSUMED",
        "source": (
            "Inherited unchanged from `scripts/pec_band_fractions.py` so the "
            "photon/power conversion is identical on both sides of every "
            "completeness ratio below."
        ),
    },
    {
        "quantity": "window / fiber transmission curves",
        "value": (
            "per-line T_window(lambda) and T_fiber(lambda, L), from "
            "datasheet curve readings"
        ),
        "class": "ASSUMED",
        "source": (
            "Tabulated point by point, with the figure, thickness or length, "
            "external-vs-internal basis and caveat, in `WINDOW_TRANSMISSION` "
            "and `FIBER_ATTENUATION` -- reproduced in full in the Window and "
            "fiber transmission section below. NEITHER datasheet tabulates "
            "transmission against wavelength, so every point is a reading "
            "off a plotted curve or a value derived from numbers the same "
            "document does tabulate. Datasheet-sourced but for REPRESENTATIVE "
            "parts: the material on the LAPD viewports is not known to this "
            "script, and while the collection fiber IS identified (Thorlabs "
            "FT1000UMT), its sheet publishes no attenuation at 320.37 nm and "
            "its typical curve begins near 400 nm, so a stand-in curve is "
            "applied and the applied attenuation stays ASSUMED."
        ),
    },
    {
        "quantity": "50 % transmission cutoffs",
        "value": "DERIVED by inverting the curves above at the applied length",
        "class": "DERIVED",
        "source": (
            "Computed from `WINDOW_TRANSMISSION` / `FIBER_ATTENUATION` "
            "rather than asserted separately, so a cutoff drawn on a figure "
            "cannot disagree with the transmission its own numbers went "
            "through. The fiber's cutoff MOVES WITH THE RUN LENGTH. "
            "Uncertainties are the reading bars of the curves the point is "
            "computed from."
        ),
    },
    {
        "quantity": "reciprocal linear dispersion",
        "value": "0.31 nm/mm at 2400 g/mm (0.62 nm/mm at 1200 g/mm)",
        "class": "MEASURED",
        "source": (
            "McPherson Model 209 (1.33 m Czerny-Turner) specification sheet, "
            "p.1 grating table. The same sheet gives resolution 0.005 nm at "
            "2400 g/mm, 'typically measured at 313.1 nm'. The 1.5 / 2.5 / "
            "4.0 nm/mm columns this script carried before the instrument was "
            "identified are placeholders, and are reachable only through "
            "--dispersion-bracket."
        ),
    },
    {
        "quantity": "entrance / exit slit width",
        "value": "30 um, equal slits",
        "class": "MEASURED",
        "source": (
            "Instrument setting as read by the operator; the micrometer "
            "readout is in 10-micron thimble divisions (datasheet), and the "
            "slits are 'continuously adjustable by precision micrometer from "
            "5 to 4000 um' with a 2-20 mm height."
        ),
    },
    {
        "quantity": "instrument f-number",
        "value": "f/9.4 (value of record); bracket [f/7, f/9.4]",
        "class": "MEASURED",
        "source": (
            "McPherson Model 209 specification sheet. Its p.1 states f/9.4 "
            "('11.6 with smaller grating') while its p.3 drawing title block "
            "states f/7. THE SHEET CONTRADICTS ITSELF and this script does "
            "not resolve it: f/9.4 is used and both members are carried "
            "through the acceptance factor below."
        ),
    },
    {
        "quantity": "fiber-to-slit acceptance",
        "value": (
            "(NA_instr / NA_fiber)^2 = 1.9 % at f/9.4 and NA 0.39 (3.3 % at "
            "f/7)"
        ),
        "class": "DERIVED",
        "source": (
            "The fiber butts against the slit with NO coupling optics, so "
            "the instrument accepts only the part of the fiber's output cone "
            "that falls inside its own NA_instr = 1 / (2 f/#). Computed from "
            "two MEASURED datasheet values -- the fiber NA and the "
            "instrument f-number -- in the same small-angle Omega ~ NA^2 "
            "form used for the etendue above, and capped at 1. Replaces the "
            "NA-into-f/# half of the single ASSUMED spectrometer throughput "
            "this script used to carry."
        ),
    },
    {
        "quantity": "grating x mirror efficiency",
        "value": "0.4",
        "class": "ASSUMED",
        "source": (
            "What is left of the spectrometer's optical throughput once the "
            "acceptance above is computed separately. The Model 209's optics "
            "are Al + MgF2 and McPherson states NO efficiency, so this is a "
            "placed number, not a datasheet reading. Every count rate scales "
            "linearly with it."
        ),
    },
    {
        "quantity": "detector counting efficiency",
        "value": (
            "0.136 at 320.37 nm, 0.0286 at 587.75 nm; ZERO outside "
            "185-680 nm"
        ),
        "class": "DERIVED",
        "source": (
            "Hamamatsu H8259 (plain variant) datasheet count sensitivity S "
            "[s^-1 pW^-1] -- 2.1e5 at 300 nm, 2.6e5 at 400, 1.9e5 at 500, "
            "7.5e4 at 600, 1.5e3 at 700 -- divided by the photon rate one "
            "picowatt carries at that wavelength, 1 pW / (hc/lambda), with S "
            "interpolated LOG-LINEARLY between cells. The datasheet gives "
            "these only as 'Typ.' with NO min/max, so the result carries no "
            "tolerance. It folds cathode efficiency, collection and counting "
            "into one number exactly as the datasheet cell does, which is "
            "why it is called a counting efficiency and NOT a QE. Outside "
            "the head's rated 185-680 nm spectral response it is ZERO: the "
            "tube is not specified there, and at 706 nm the plain head is "
            "out of range (order 3e-4 if extrapolated)."
        ),
    },
)

#: Doppler-width coefficient: FWHM = DOPPLER_COEFF * lambda * sqrt(T/M),
#: T in eV, M in amu, FWHM in the units of lambda.
DOPPLER_COEFF = 7.716e-5

#: Helium mass [amu], the emitter of every line in both adf15 files.
HE_MASS_AMU = 4.0026

#: Boltzmann constant [eV/K], for the cold-neutral wall temperature.
K_B_EV_PER_K = 8.617333262e-5

#: Gaussian sigma per unit FWHM.
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

#: Spectrum drawing window [nm].
SPECTRUM_RANGE_NM = (20.0, 750.0)

#: Line each zoom panel resolves.  He II 320.37 nm is the n = 5 -> 3
#: transition; He I 587.75 nm is the visible workhorse.
INSET_LINE_NM = {"he1": 320.37, "he0": 587.75}

#: Half-width of the SHARED zoom axis [nm], as an offset from line centre.
#: Both panels use it so the two widths are read against one ruler; it holds
#: the widest Gaussian drawn (He II 320.37 nm at ~7.4 eV, FWHM ~0.034 nm) to
#: well beyond +/-3 sigma.
INSET_HALF_WIDTH_NM = 0.1

#: How close an adf15 line must lie to a named zoom/sweep target before it
#: may stand in for it [nm].  Loose against the adf15 wavelengths, which are
#: quoted to 0.01 nm, and far tighter than the gaps between the He lines
#: these panels resolve.
LINE_MATCH_TOL_NM = 0.05

#: Fixed temperature of the dashed comparison trace in the He II zoom [eV].
INSET_COMPARISON_T_EV = 1.0

#: adf15 lines kept in the line-shape record but no longer drawn, with the
#: reason.  Retained so a reader of the table is not left wondering.
INSET_RETIRED_LINES_NM = {
    "he1": (
        468.65,
        "superseded as the He II zoom subject by 320.37 nm; its FWHM is "
        "still tabulated for the record",
    ),
}

# --- synthetic monochromator sweep ----------------------------------------

#: Wavelength range the sweep is scanned over [nm].
SWEEP_RANGE_NM = (300.0, 750.0)

#: Minimum lambda0 samples per bandpass, and per Gaussian sigma, on the
#: sweep grid.  The narrow-line bound matters: the cold He I lines are
#: ~0.0036 nm FWHM, far narrower than any bandpass here, and a grid that
#: resolves only the bandpass would misrepresent their area.
SWEEP_POINTS_PER_BANDPASS = 25
SWEEP_POINTS_PER_SIGMA = 4

#: The monochromator actually on the bench, and the cells its own spec sheet
#: states.  Every scalar below that is read off this sheet says so in its own
#: comment; the sheet's internal contradiction on f-number is recorded here
#: rather than resolved.
MONOCHROMATOR = {
    "model": "McPherson Model 209, 1.33 m Czerny-Turner",
    "datasheet": "McPherson 'Model 209' specification sheet",
    "dispersion": (
        "reciprocal linear dispersion 0.31 nm/mm at 2400 g/mm (0.62 nm/mm at "
        "1200 g/mm), p.1 grating table"
    ),
    "resolution": (
        "0.005 nm at 2400 g/mm, carrying the sheet's own qualifier "
        "'typically measured at 313.1 nm'"
    ),
    "f_number": (
        "p.1 states f/9.4 ('11.6 with smaller grating'); the p.3 drawing "
        "title block states f/7. The sheet contradicts itself and this "
        "script does not resolve it: f/9.4 is the value of record and "
        "[f/7, f/9.4] is the disclosed bracket"
    ),
    "wavelength": (
        "wavelength accuracy 0.05 nm; reproducibility +/-0.005 nm at "
        "1200 g/mm"
    ),
    "slits": (
        "'continuously adjustable by precision micrometer from 5 to "
        "4000 um', 'micrometer readout is in 10-micron thimble divisions'; "
        "slit height 2-20 mm"
    ),
}

#: Entrance/exit slit width [um].  MEASURED: the instrument setting as read
#: by the operator, on a micrometer whose divisions are 10 um.
DEFAULT_SLIT_UM = 30.0

#: Reciprocal linear dispersion [nm/mm].  MEASURED: the datasheet grating
#: table cell for the 2400 g/mm grating.
DEFAULT_DISPERSION_NM_PER_MM = 0.31

#: Placeholder dispersions the sweep tables carried before the instrument was
#: identified.  Reachable only through --dispersion-bracket, and NOT a
#: property of the Model 209.
DISPERSION_BRACKET_NM_PER_MM = (1.5, 2.5, 4.0)

#: Instrument f-number and the bracket the datasheet's own contradiction
#: forces.  MEASURED cells; see MONOCHROMATOR["f_number"].
DEFAULT_F_NUMBER = 9.4
F_NUMBER_BRACKET = (7.0, 9.4)

#: Grating x mirror efficiency, the part of the spectrometer's throughput
#: that is left once the fiber-to-slit acceptance is computed from the
#: f-number.  ASSUMED: the 209's optics are Al + MgF2 and McPherson states
#: no efficiency.
DEFAULT_GRATING_MIRROR_EFFICIENCY = 0.4

#: The photon-counting head, and the cells its own datasheet states.
PMT = {
    "model": "Hamamatsu H8259 (plain variant)",
    "datasheet": "Hamamatsu H8259 photon-counting head datasheet",
    "spectral_response_nm": (185.0, 680.0),
    "peak_nm": 400.0,
    "effective_area": "4 x 20 mm",
    "linearity": (
        "count linearity 2.5e6 s^-1 at 10 % loss; pulse-pair resolution 35 ns"
    ),
    "dark_count": "typ 30 s^-1, max 80 s^-1 at 25 degC",
}

#: Datasheet count sensitivity S [s^-1 pW^-1] against wavelength [nm].  The
#: sheet gives these as 'Typ.' with NO min/max, so the counting efficiency
#: derived from them carries no tolerance either.
PMT_COUNT_SENSITIVITY_PER_PW = (
    (300.0, 2.1e5),
    (400.0, 2.6e5),
    (500.0, 1.9e5),
    (600.0, 7.5e4),
    (700.0, 1.5e3),
)

#: Flat fiber transmission quoted by the operator (2-3x the datasheet
#: chain), printed alongside the datasheet result rather than replacing it.
QUOTED_FLAT_FIBER_TRANSMISSION = 0.4

#: Core diameters the sweep tabulates [um]: the measured core first, then the
#: two comparison columns the core-scaling argument needs.
SWEEP_CORE_COLUMNS_UM = (1000.0, 600.0, 400.0)

#: Line the sweep figure's zoom panel resolves.
SWEEP_ZOOM_LINE_NM = 320.37

#: Half-width of that zoom, in bandpasses.
SWEEP_ZOOM_BANDPASSES = 6.0

#: Core diameters reported as columns [um]; the measured core is the default
#: and the other two are comparison columns.
FIBER_CORE_COLUMNS_UM = (400.0, 600.0, 1000.0)

#: NA values reported as comparison rows beside the measured default.
FIBER_NA_BRACKET = (0.12, 0.39)


class ArtifactRefused(SystemExit):
    """Raised with an explanatory message when the artifact cannot be read."""


def resolve_target_line(lam_nm, target_nm, what, atol_nm=LINE_MATCH_TOL_NM):
    """Index of the line at ``target_nm``, refusing a silent substitution.

    A bare ``argmin`` over ``|lambda - target|`` ALWAYS returns an index, so a
    target the line list does not carry is answered with whatever line
    happens to be nearest -- and a panel captioned 320.37 nm would then be
    drawn from a different transition, with no diagnostic anywhere.  This
    matches within ``atol_nm`` and raises ``ArtifactRefused`` otherwise,
    naming the target, the tolerance and the nearest line actually available.

    ``what`` names the caller in the refusal, since the same target is
    resolved for the markdown, the figure and the console.
    """
    lam = np.asarray(lam_nm, dtype=float)
    k = int(np.argmin(np.abs(lam - float(target_nm))))
    if not np.isclose(lam[k], float(target_nm), rtol=0.0, atol=atol_nm):
        raise ArtifactRefused(
            f"{what}: no line within {atol_nm:g} nm of the "
            f"{float(target_nm):g} nm target; the nearest available line is "
            f"{lam[k]:.4f} nm"
        )
    return k


# --- adf15 evaluation -----------------------------------------------------


def stage_lines(stage_key):
    """Return the EXCIT line blocks for one emitting stage.

    Delegates entirely to ``pec_band_fractions.excitation_lines``, so the
    line list, the count check, the wavelength ordering, the vacuum-
    wavelength convention and the band assignment are the ones the band-split
    record was produced with.
    """
    spec = next(
        s for s in PBF.SPECIES if s["key"] == STAGE[stage_key]["spec_key"]
    )
    return spec, PBF.excitation_lines(spec)


def pec_stack(lines, ne, Te):
    """Interpolate every line's PEC at once onto array ``(ne, Te)``.

    Returns ``(n_lines,) + ne.shape`` in photons cm^3 s^-1.  All blocks in an
    adf15 file share one (ne, Te) grid -- asserted here rather than assumed --
    so the bilinear coordinates are computed once and reused, and the blend is
    ``pec_band_fractions``'s own ``_interp_blend`` on its own ``_interp_coords``.
    Evaluating the scalar ``pec_band_fractions.pec_at`` point by point gives
    the same answer; the equality is checked in the self-consistency section.
    """
    grid_ne, grid_te = lines[0]["log_ne"], lines[0]["log_te"]
    for block in lines[1:]:
        if not (
            np.array_equal(block["log_ne"], grid_ne)
            and np.array_equal(block["log_te"], grid_te)
        ):
            raise ArtifactRefused(
                f"adf15 ISEL {block['isel']} is tabulated on a different "
                "(ne, Te) grid from the file's first block; the shared-grid "
                "shortcut this evaluation takes is not valid for that file"
            )
    ix, iy, fx, fy = PBF._interp_coords(
        grid_ne, grid_te, np.log10(ne), np.log10(Te)
    )
    return np.stack(
        [
            10.0 ** PBF._interp_blend(b["log_pec"], ix, iy, fx, fy)
            for b in lines
        ]
    )


def adf15_clamp_census(lines, ne, Te):
    """Count evaluation points pushed onto an adf15 grid edge by the clamp."""
    grid_ne, grid_te = lines[0]["log_ne"], lines[0]["log_te"]
    log_ne, log_te = np.log10(ne), np.log10(Te)
    total = int(log_ne.size)
    below_te = int(np.count_nonzero(log_te < grid_te[0]))
    above_te = int(np.count_nonzero(log_te > grid_te[-1]))
    below_ne = int(np.count_nonzero(log_ne < grid_ne[0]))
    above_ne = int(np.count_nonzero(log_ne > grid_ne[-1]))
    clamped = int(
        np.count_nonzero(
            (log_te < grid_te[0])
            | (log_te > grid_te[-1])
            | (log_ne < grid_ne[0])
            | (log_ne > grid_ne[-1])
        )
    )
    return {
        "points": total,
        "grid_Te_min_eV": float(10.0 ** grid_te[0]),
        "grid_Te_max_eV": float(10.0 ** grid_te[-1]),
        "grid_ne_min_cm3": float(10.0 ** grid_ne[0]),
        "grid_ne_max_cm3": float(10.0 ** grid_ne[-1]),
        "points_below_Te_edge": below_te,
        "points_above_Te_edge": above_te,
        "points_below_ne_edge": below_ne,
        "points_above_ne_edge": above_ne,
        "points_clamped": clamped,
        "fraction_clamped": (clamped / total) if total else None,
        "min_Te_eV": float(np.min(Te)),
        "max_Te_eV": float(np.max(Te)),
        "min_ne_cm3": float(np.min(ne)),
        "max_ne_cm3": float(np.max(ne)),
    }


def adf11_clamp_census(ne, Te):
    """Count evaluation points outside the adf11 Te grid the PLT path uses."""
    te_lo, te_hi = he_rate_temperature_range_eV()
    total = int(Te.size)
    below = int(np.count_nonzero(Te < te_lo))
    above = int(np.count_nonzero(Te > te_hi))
    return {
        "points": total,
        "table_Te_min_eV": te_lo,
        "table_Te_max_eV": te_hi,
        "points_below_Te_edge": below,
        "points_above_Te_edge": above,
        "fraction_clamped": ((below + above) / total) if total else None,
        "min_Te_eV": float(np.min(Te)),
    }


def sub_quotable_census(Te):
    """Count points below the Te floor the band-split record calls quotable."""
    total = int(Te.size)
    below = int(np.count_nonzero(Te < QUOTABLE_TE_MIN_EV))
    return {
        "points": total,
        "Te_min_quotable_eV": QUOTABLE_TE_MIN_EV,
        "points_below": below,
        "fraction_below": (below / total) if total else None,
    }


# --- artifact reduction ---------------------------------------------------


def main_discharge_origin_ms(t_ms, phase):
    """Model time [ms] of the first save labelled ``main_discharge``."""
    hits = np.flatnonzero(np.asarray(phase, dtype=str) == "main_discharge")
    if not hits.size:
        raise ArtifactRefused(
            "NON-IGNITED RUN: no save carries the 'main_discharge' phase "
            "label, so the main-discharge clock this instrument averages on "
            "has no origin. Nothing here is defined for such a run."
        )
    return float(t_ms[hits[0]])


def read_window(h5_path, window_ms):
    """Load the plateau-window state and geometry this instrument needs.

    Only the saves inside the window are read off disk: a production artifact
    is multi-gigabyte and the plateau is a small slice of it.
    """
    with h5py.File(h5_path, "r") as f:
        params = json.loads(f.attrs.get("params_json", "{}"))
        flags = json.loads(f.attrs.get("flags_json", "{}"))
        closure = PRS.require_adas(params)
        closure["icool_recomb"] = bool(flags.get("icool_recomb", False))

        t_ms = f["time"][:] * 1.0e3
        phase = np.array(
            [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f["phase"][:]
            ]
        )
        origin_ms = main_discharge_origin_ms(t_ms, phase)
        t_md_ms = t_ms - origin_ms
        frames = np.flatnonzero(
            (t_md_ms >= window_ms[0]) & (t_md_ms <= window_ms[1])
        )
        if not frames.size:
            raise ArtifactRefused(
                f"no save lands in {window_ms[0]}-{window_ms[1]} ms on the "
                f"main-discharge clock (this run spans {t_md_ms.min():.4f} "
                f"to {t_md_ms.max():.4f} ms on that clock)"
            )
        active = np.flatnonzero(f["geometry/plasma_active"][:])
        if not active.size:
            raise ArtifactRefused("artifact has no plasma-active cells")

        sl = slice(int(frames[0]), int(frames[-1]) + 1)
        take = np.ix_(np.arange(sl.start, sl.stop), active)
        state = {
            "n": f["n"][sl, :][:, active],
            "nn": f["nn"][sl, :][:, active],
            "Te": f["Te"][sl, :][:, active],
            "Ti": f["Ti"][sl, :][:, active],
        }
        del take
        geom = {
            "z_cm": f["geometry/z_cm"][:][active],
            "Rp_cm": f["geometry/Rp_cm"][:][active],
            "length_cm": f["geometry/length_cm"][:][active],
            "volume_cm3": f["geometry/plasma_volume_cm3"][:][active],
            "active_index": active,
        }
        run = {
            "steps": int(f.attrs["steps"]),
            "saves": int(t_ms.size),
            "run_status": str(f.attrs.get("run_status", "")),
            "final_time_ms": float(f.attrs["final_time"]) * 1.0e3,
            "compiled_kernels": bool(f.attrs.get("compiled_kernels", False)),
            "nx": params.get("nx"),
        }
        has_ledger = "electron_energy_terms_W_cm3" in f
        ledger = {}
        if has_ledger:
            for key, cfg in STAGE.items():
                path = f"electron_energy_terms_W_cm3/{cfg['ledger_row']}"
                if path in f:
                    ledger[key] = -f[path][sl, :][:, active]

    return {
        "params": params,
        "flags": flags,
        "closure": closure,
        "origin_ms": origin_ms,
        "t_md_ms": t_md_ms[sl],
        "frames": int(sl.stop - sl.start),
        "state": state,
        "geometry": geom,
        "run": run,
        "ledger": ledger,
        "window_ms": tuple(window_ms),
        "carries_nn_a": None,
    }


def evaluate_stage(stage_key, data):
    """Per-line window-mean emissivity, cell power and machine total."""
    cfg = STAGE[stage_key]
    spec, lines = stage_lines(stage_key)
    n = data["state"]["n"]
    partner = data["state"][cfg["partner"]]
    Te = data["state"]["Te"]
    ne_lookup = np.maximum(n, data["closure"]["ne_floor"])

    pec = pec_stack(lines, ne_lookup, Te)  # (n_lines, n_frames, n_cells)
    lam_nm = np.array([b["wavelength_nm"] for b in lines])
    photon_eV = PBF.HC_EV_NM / lam_nm
    photon_J = photon_eV * qe_SI

    eps_photons = pec * (n * partner)  # photons cm^-3 s^-1
    eps_W = eps_photons * photon_J[:, None, None]  # W cm^-3

    eps_W_mean = eps_W.mean(axis=1)  # (n_lines, n_cells)
    eps_ph_mean = eps_photons.mean(axis=1)
    volume = data["geometry"]["volume_cm3"]
    cell_W = eps_W_mean * volume  # W per line per cell
    machine_W = cell_W.sum(axis=1)  # W per line

    # The same adf11 channel, evaluated exactly as the solver's cooling row
    # builds it, for the completeness ratio and the ledger cross-check.
    coeff = he_rates(
        ne_lookup,
        Te,
        (cfg["plt_key"],),
        low_te_extension=data["closure"]["adas_low_te_extension"],
    )[cfg["plt_key"]]
    adf11_W = coeff * n * partner * qe_SI  # W cm^-3
    adf11_W_mean = adf11_W.mean(axis=0)
    adf11_cell_W = adf11_W_mean * volume
    adf11_machine_W = float(adf11_cell_W.sum())

    # Per-cell completeness under the window mean, and the machine-integrated
    # completeness.  Neither is a physics result: C measures how much of the
    # adf11 line power the adf15 file's hand-picked line list accounts for.
    with np.errstate(divide="ignore", invalid="ignore"):
        cell_C = np.where(adf11_cell_W > 0.0, cell_W.sum(axis=0) / adf11_cell_W, np.nan)

    return {
        "stage": stage_key,
        "label": cfg["label"],
        "short": cfg["short"],
        "spec_file": spec["file"],
        "partner_label": cfg["partner_label"],
        "plt_key": cfg["plt_key"],
        "ledger_row": cfg["ledger_row"],
        "lines": lines,
        "lambda_nm": lam_nm,
        "photon_J": photon_J,
        "band": [b["band"] for b in lines],
        "isel": [b["isel"] for b in lines],
        "eps_W_mean": eps_W_mean,
        "eps_photons_mean": eps_ph_mean,
        "cell_W": cell_W,
        "machine_W": machine_W,
        "machine_W_total": float(machine_W.sum()),
        "adf11_eps_W_mean": adf11_W_mean,
        "adf11_cell_W": adf11_cell_W,
        "adf11_machine_W": adf11_machine_W,
        "completeness_machine": (
            float(machine_W.sum() / adf11_machine_W)
            if adf11_machine_W > 0.0
            else None
        ),
        "completeness_cell": cell_C,
        "adf15_clamp": adf15_clamp_census(lines, ne_lookup, Te),
        "adf11_clamp": adf11_clamp_census(ne_lookup, Te),
        "sub_quotable": sub_quotable_census(Te),
        "ledger_check": ledger_check(data, stage_key, adf11_W),
    }


def ledger_check(data, stage_key, adf11_W):
    """Compare the reconstructed adf11 channel with the run's own ledger row.

    The row is an electron energy SINK, so it is negated on read.  A match
    here is what licenses the line decomposition: it says the adf11 channel
    this script splits is the very channel the run booked.
    """
    row = data["ledger"].get(stage_key)
    if row is None:
        return {
            "available": False,
            "reason": (
                f"artifact carries no "
                f"'electron_energy_terms_W_cm3/{STAGE[stage_key]['ledger_row']}' "
                "row over the window"
            ),
        }
    scale = np.maximum(np.abs(row), np.abs(adf11_W))
    good = scale > 0.0
    rel = np.zeros_like(scale)
    rel[good] = np.abs(adf11_W[good] - row[good]) / scale[good]
    mean_recon = float(adf11_W.mean())
    mean_ledger = float(row.mean())
    return {
        "available": True,
        "ledger_row": STAGE[stage_key]["ledger_row"],
        "window_mean_reconstructed_W_cm3": mean_recon,
        "window_mean_ledger_W_cm3": mean_ledger,
        "window_mean_rel_dev": (
            float(mean_recon / mean_ledger - 1.0) if mean_ledger else None
        ),
        "max_point_rel_dev": float(rel.max()),
    }


# --- self-consistency -----------------------------------------------------


def vectorization_check(stage, data, n_probe=64):
    """Assert the stacked PEC evaluation equals the scalar reference path.

    ``pec_stack`` reuses ``pec_band_fractions``'s interpolation coordinates
    and blend but evaluates them on arrays; this walks a sample of the actual
    (ne, Te) points through the scalar ``pec_band_fractions.pec_at`` and
    requires exact agreement to floating-point round-off.  Without it the
    array path could silently drift from the record's own numbers.
    """
    cfg = STAGE[stage["stage"]]
    n = data["state"]["n"]
    Te = data["state"]["Te"]
    ne_lookup = np.maximum(n, data["closure"]["ne_floor"])
    rng = np.random.default_rng(0)
    flat_ne, flat_te = ne_lookup.ravel(), Te.ravel()
    idx = rng.choice(flat_ne.size, size=min(n_probe, flat_ne.size), replace=False)

    worst = 0.0
    for j in idx:
        ne_j, te_j = float(flat_ne[j]), float(flat_te[j])
        stacked = pec_stack(stage["lines"], np.array([ne_j]), np.array([te_j]))
        for k, block in enumerate(stage["lines"]):
            ref = PBF.pec_at(block, ne_j, te_j)
            got = float(stacked[k, 0])
            denom = max(abs(ref), abs(got))
            if denom > 0.0:
                worst = max(worst, abs(got - ref) / denom)
    return {
        "probe_points": int(idx.size),
        "lines": len(stage["lines"]),
        "max_rel_dev_vs_pec_at": worst,
        "plt_key": cfg["plt_key"],
    }


def completeness_identity_check(stage):
    """Assert the line sum equals C times the adf11 cell power, cell by cell.

    This is the identity the brief pins: the per-cell He II line sum must be
    ``C * (PLT2 * ne^2 * V)`` with C the completeness ratio the band-split
    convention defines at that state.  It is an identity by construction of
    ``completeness_cell``, and the check exists to catch a broken construction
    (a mismatched volume, a dropped line, a wrong photon energy), not to
    discover physics.
    """
    lhs = stage["cell_W"].sum(axis=0)
    rhs = stage["completeness_cell"] * stage["adf11_cell_W"]
    good = np.isfinite(rhs) & (np.abs(lhs) > 0.0)
    if not np.any(good):
        return {"cells": 0, "max_rel_dev": None}
    rel = np.abs(lhs[good] - rhs[good]) / np.abs(lhs[good])
    return {
        "cells": int(np.count_nonzero(good)),
        "max_rel_dev": float(rel.max()),
        "C_min": float(np.nanmin(stage["completeness_cell"])),
        "C_max": float(np.nanmax(stage["completeness_cell"])),
    }


def port_radiance_crosscheck(h5_path, port):
    """Reproduce ``port_radiance_sim1d``'s e-i number on the same artifact.

    Runs that instrument's own ``build_report`` in-process and re-derives the
    same quantity here from the artifact, on ITS window (run clock) and ITS
    cell, so the two numbers are the same measurement made twice.  A match
    ties this script's adf11 side to the instrument already in the record.
    """
    rep = PRS.build_report(h5_path, port)
    theirs = rep["stats"]["emissivity"]["plt2"]["drive_mean"]
    iz = rep["geometry"]["cell"]

    with h5py.File(h5_path, "r") as f:
        params = json.loads(f.attrs.get("params_json", "{}"))
        flags = json.loads(f.attrs.get("flags_json", "{}"))
        t_ms = f["time"][:] * 1.0e3
        mask = (t_ms >= PRS.DRIVE_WINDOW_MS[0]) & (t_ms <= PRS.DRIVE_WINDOW_MS[1])
        sel = np.flatnonzero(mask)
        sl = slice(int(sel[0]), int(sel[-1]) + 1)
        n = f["n"][sl, iz]
        Te = f["Te"][sl, iz]
    ne_floor = float(params.get("ne_floor", 0.0))
    ext = bool(params.get("adas_low_te_extension", False))
    coeff = he_rates(
        np.maximum(n, ne_floor), Te, ("plt2",), low_te_extension=ext
    )["plt2"]
    mine = float(np.mean(coeff * n * n * qe_SI))

    return {
        "port": int(port),
        "cell": int(iz),
        "z_cell_cm": rep["geometry"]["z_cell_cm"],
        "window_ms_run_clock": list(PRS.DRIVE_WINDOW_MS),
        "port_radiance_drive_mean_W_cm3": theirs,
        "reconstructed_drive_mean_W_cm3": mine,
        "rel_dev": (
            float(mine / theirs - 1.0) if theirs else None
        ),
        "icool_recomb": bool(flags.get("icool_recomb", False)),
    }


# --- synthetic fiber ------------------------------------------------------


def etendue_cm2_sr(core_um, na):
    """Fiber etendue [cm^2 sr] from core diameter [um] and NA."""
    radius_cm = 0.5 * core_um * 1.0e-4
    area_cm2 = np.pi * radius_cm * radius_cm
    omega_sr = np.pi * na * na
    return {
        "core_um": float(core_um),
        "na": float(na),
        "core_area_cm2": float(area_cm2),
        "omega_sr": float(omega_sr),
        "etendue_cm2_sr": float(area_cm2 * omega_sr),
    }


def fiber_at_port(
    stage, data, port, law, fibers, material, length_m, flat_fiber_T=None
):
    """Per-line chord radiance, collected power and TRANSMITTED signal.

    Two distinct quantities are carried per line and never conflated.
    COLLECTED is the flux inside the fiber's acceptance at its face,
    ``L * G``; TRANSMITTED is what survives the port window and the fiber
    run to reach a detector, ``collected * T_window * T_fiber``.  The fiber
    sits OUTSIDE the window looking in, so the transmitted column is the one
    a photon counter would register, and the collected column is shown
    beside it only to make the loss visible.
    """
    z_want = PRS.port_to_z_cm(port, law)
    z_cells = data["geometry"]["z_cm"]
    j = int(np.argmin(np.abs(z_cells - z_want)))
    Rp = float(data["geometry"]["Rp_cm"][j])
    chord = 2.0 * Rp
    eps = stage["eps_W_mean"][:, j]  # W cm^-3
    brightness = eps * chord  # W cm^-2, the chord integral of emissivity
    radiance = brightness / (4.0 * np.pi)  # W cm^-2 sr^-1

    # Emitter temperature for the thermal line shape.  He II radiates from
    # the ION population, so its width is set by the run's own saved Ti at
    # this cell; He I radiates from the COLD neutral population the engine
    # books at the wall temperature.
    cfg = STAGE[stage["stage"]]
    if cfg["emitter_temp"] == "Ti":
        emitter_T_eV = float(data["state"]["Ti"][:, j].mean())
        emitter_T_source = (
            "plateau-mean saved Ti at this cell (the run's own ion "
            "temperature)"
        )
    else:
        emitter_T_eV = T_WALL_K * K_B_EV_PER_K
        emitter_T_source = (
            f"cold neutral population at the engine's T_WALL_K = "
            f"{T_WALL_K:g} K"
        )
    fwhm_nm = doppler_fwhm_nm(stage["lambda_nm"], emitter_T_eV)
    sigma_nm = fwhm_nm * FWHM_TO_SIGMA

    t_window = window_transmission(material, stage["lambda_nm"])
    if flat_fiber_T is None:
        t_fiber = fiber_transmission(stage["lambda_nm"], length_m)
        t_fiber_source = f"datasheet chain at {length_m:g} m"
    else:
        # A flat override still respects the fiber's own transmission edge:
        # a single number quoted for the visible says nothing about a line
        # the fiber does not carry at all.
        t_fiber = np.where(
            np.asarray(stage["lambda_nm"])
            < FIBER_ATTENUATION["opaque_below_nm"],
            0.0,
            float(flat_fiber_T),
        )
        t_fiber_source = (
            f"FLAT override {float(flat_fiber_T):g} (datasheet chain "
            "bypassed)"
        )
    t_total = t_window * t_fiber

    per_fiber = {}
    for fib in fibers:
        power = radiance * fib["etendue_cm2_sr"]  # W, at the fiber face
        photons = power / stage["photon_J"]  # s^-1, at the fiber face
        power_t = power * t_total  # W, past window + fiber
        photons_t = photons * t_total  # s^-1, past window + fiber
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                stage["machine_W"] > 0.0, power_t / stage["machine_W"], np.nan
            )
        per_fiber[fiber_tag(fib)] = {
            "fiber": fib,
            "power_W": power,
            "photons_per_s": photons,
            "power_W_transmitted": power_t,
            "photons_per_s_transmitted": photons_t,
            "ratio_to_machine": ratio,
            "power_W_total": float(power.sum()),
            "photons_per_s_total": float(photons.sum()),
            "power_W_transmitted_total": float(power_t.sum()),
            "photons_per_s_transmitted_total": float(photons_t.sum()),
            "ratio_to_machine_total": (
                float(power_t.sum() / stage["machine_W_total"])
                if stage["machine_W_total"] > 0.0
                else None
            ),
        }
    return {
        "port": int(port),
        "z_want_cm": float(z_want),
        "cell_in_active": j,
        "z_cell_cm": float(z_cells[j]),
        "Rp_cm": Rp,
        "chord_length_cm": chord,
        "brightness_W_cm2": brightness,
        "radiance_W_cm2_sr": radiance,
        "emitter_T_eV": emitter_T_eV,
        "emitter_T_source": emitter_T_source,
        "doppler_fwhm_nm": fwhm_nm,
        "doppler_sigma_nm": sigma_nm,
        "T_window": t_window,
        "T_fiber": t_fiber,
        "T_fiber_source": t_fiber_source,
        "T_total": t_total,
        "fibers": per_fiber,
    }


def fiber_tag(fib):
    """Short identifier for a fiber configuration."""
    return f"{fib['core_um']:.0f}um_NA{fib['na']:g}"


# --- window cutoffs -------------------------------------------------------


#: Reading uncertainty on each derived 50 % point [nm], from the curve
#: readings the point is computed from -- see each spec's own ``caveat``.
CUTOFF_UNCERTAINTY_NM = {
    "fused_silica": 4.0,
    "borosilicate": 4.0,
    "fiber": 10.0,
}


def _half_transmission_nm(fn, lo=140.0, hi=1200.0, n=212001):
    """Bluest wavelength at which ``fn`` first reaches 50 % transmission."""
    grid = np.linspace(lo, hi, n)
    hits = np.flatnonzero(np.asarray(fn(grid)) >= 0.5)
    return float(grid[hits[0]]) if hits.size else None


def cutoff_table(length_m):
    """50 % transmission points, DERIVED from the curves actually applied.

    Computed by inverting ``window_transmission`` and ``fiber_transmission``
    rather than asserted separately, so a cutoff drawn on a figure cannot
    disagree with the transmission the same figure's numbers went through.
    The fiber's point moves with the run length, because its bulk
    attenuation does.
    """
    out = []
    for key, spec in WINDOW_TRANSMISSION.items():
        out.append(
            {
                "material": spec["material"],
                "short": spec["short"],
                "cutoff_nm": _half_transmission_nm(
                    lambda g, k=key: window_transmission(k, g)
                ),
                "uncertainty_nm": CUTOFF_UNCERTAINTY_NM[key],
                "basis": spec["thickness"] + "; " + spec["basis"],
                "figure": spec["figure"],
                "source": spec["source"],
                "caveat": spec["caveat"],
            }
        )
    out.append(
        {
            "material": FIBER_ATTENUATION["material"],
            "short": f"UV fiber ({length_m:g} m)",
            "cutoff_nm": _half_transmission_nm(
                lambda g: fiber_transmission(g, length_m)
            ),
            "uncertainty_nm": CUTOFF_UNCERTAINTY_NM["fiber"],
            "basis": (
                f"{length_m:g} m of fiber, bulk attenuation AND the two "
                "end faces"
            ),
            "figure": FIBER_ATTENUATION["figure"],
            "source": FIBER_ATTENUATION["source"],
            "caveat": FIBER_ATTENUATION["caveat"],
        }
    )
    return tuple(sorted(out, key=lambda c: c["cutoff_nm"]))


def transmits(cutoff, lam_nm):
    """Boolean mask: which lines sit red-ward of a cutoff."""
    return np.asarray(lam_nm) >= cutoff["cutoff_nm"]


# --- thermal line shape ----------------------------------------------------


def doppler_fwhm_nm(lam_nm, temp_eV):
    """Doppler FWHM [nm] of a helium line at emitter temperature ``temp_eV``.

    ``FWHM = 7.716e-5 * lambda * sqrt(T[eV] / M[amu])`` with M = 4.0026, the
    standard thermal-broadening width.  This is the ONLY broadening applied:
    it is the thermal envelope of the adf15 line as that file lists it, and
    the instrumental, Stark, fine-structure and Zeeman contributions are all
    absent by construction.
    """
    return (
        DOPPLER_COEFF
        * np.asarray(lam_nm, dtype=float)
        * np.sqrt(float(temp_eV) / HE_MASS_AMU)
    )


def spectrum_grid(lam_nm, sigma_nm, lo, hi, n_base=3000, half=8.0, per_sigma=14):
    """Wavelength grid that resolves every line it carries.

    A uniform grid over 20-750 nm cannot represent a 0.004 nm line: the peak
    would fall between samples and the drawn height would be an artifact of
    the sampling.  The grid is therefore a coarse base plus a dense local
    patch around each line, so the plotted peak IS the analytic peak.
    """
    parts = [np.linspace(lo, hi, n_base)]
    for centre, sigma in zip(np.atleast_1d(lam_nm), np.atleast_1d(sigma_nm)):
        if sigma <= 0.0:
            continue
        parts.append(
            np.linspace(
                centre - half * sigma,
                centre + half * sigma,
                int(2 * half * per_sigma) + 1,
            )
        )
    grid = np.unique(np.concatenate(parts))
    return grid[(grid >= lo) & (grid <= hi)]


def spectral_density(grid_nm, lam_nm, area, sigma_nm):
    """Sum of Gaussians whose AREAS are ``area``, evaluated on ``grid_nm``.

    Each line contributes ``area / (sigma sqrt(2 pi)) * exp(-...)``, so the
    integral of the returned curve over wavelength reproduces the summed
    per-line quantity exactly and the units gain a ``nm^-1``.
    """
    out = np.zeros_like(np.asarray(grid_nm, dtype=float))
    for centre, a, sigma in zip(
        np.atleast_1d(lam_nm), np.atleast_1d(area), np.atleast_1d(sigma_nm)
    ):
        if sigma <= 0.0 or a == 0.0:
            continue
        out += (a / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((np.asarray(grid_nm, dtype=float) - centre) / sigma) ** 2
        )
    return out


# --- monochromator sweep ---------------------------------------------------


def bandpass_nm(slit_um, dispersion_nm_per_mm):
    """Spectral bandpass [nm] of equal entrance/exit slits.

    ``slit_um * 1e-3`` converts the slit width to mm, which the reciprocal
    linear dispersion then turns into nm.
    """
    return float(slit_um) * 1.0e-3 * float(dispersion_nm_per_mm)


def slit_fraction(slit_um, core_um):
    """Fraction of a butt-coupled fiber core the entrance slit admits.

    1:1 imaging, so the core image is the core width and the slit crops it
    in one dimension; a slit wider than the core admits all of it.
    """
    return min(1.0, float(slit_um) / float(core_um))


def slit_acceptance_fraction(fiber_na, f_number):
    """Fraction of the fiber's output cone the spectrometer's f/# accepts.

    The fiber butts against the entrance slit with NO coupling optics, so the
    instrument sees only the part of the fiber's emission cone that falls
    inside its own acceptance, ``NA_instr = 1 / (2 f/#)``.  Solid angles go
    as ``NA^2`` in the small-angle form this script uses everywhere else, so
    the accepted fraction is ``(NA_instr / NA_fiber)^2``, capped at 1 when
    the instrument is faster than the fiber.

    Raises ``ArtifactRefused`` on a non-positive f-number or an NA outside
    (0, 1], because both would make the ratio meaningless rather than merely
    wrong.
    """
    if float(f_number) <= 0.0:
        raise ArtifactRefused(f"f-number must be positive; got {f_number}")
    if not (0.0 < float(fiber_na) <= 1.0):
        raise ArtifactRefused(f"fiber NA must lie in (0, 1]; got {fiber_na}")
    na_instr = 0.5 / float(f_number)
    return float(min(1.0, (na_instr / float(fiber_na)) ** 2))


def pmt_counting_efficiency(lam_nm):
    """Counting efficiency per INCIDENT photon at each wavelength.

    The datasheet quantity is a count sensitivity ``S`` in counts per second
    per picowatt, which already folds photocathode efficiency, collection and
    discriminator counting into ONE number.  Dividing it by the photon rate a
    picowatt carries at that wavelength, ``1 pW / (hc/lambda)``, turns it into
    a dimensionless efficiency -- which is why this is a counting efficiency
    and NOT a quantum efficiency.

    ``S`` is interpolated LOG-LINEARLY between the tabulated cells, as the
    fiber attenuation is, because it spans two and a half decades across the
    band.  Outside the head's rated spectral response the efficiency is ZERO:
    the tube is not specified there and returning a number would put counts
    on lines it cannot register.  Inside the rated range but outside the
    tabulated span the nearest cell is HELD; with the shipped sweep range
    that hold never fires.

    Accepts a scalar or an array and returns the matching shape.
    """
    lam = np.atleast_1d(np.asarray(lam_nm, dtype=float))
    cells_nm = np.array([c[0] for c in PMT_COUNT_SENSITIVITY_PER_PW])
    cells_s = np.array([c[1] for c in PMT_COUNT_SENSITIVITY_PER_PW])
    s_per_pw = 10.0 ** np.interp(lam, cells_nm, np.log10(cells_s))
    photon_J = (PBF.HC_EV_NM / lam) * qe_SI
    photons_per_pw = 1.0e-12 / photon_J
    eta = s_per_pw / photons_per_pw
    lo, hi = PMT["spectral_response_nm"]
    eta = np.where((lam >= lo) & (lam <= hi), eta, 0.0)
    if np.ndim(lam_nm) == 0:
        return float(eta[0])
    return eta


def instrument_triangle(offsets_nm, bandpass):
    """Equal-slit instrument function, PEAK-NORMALIZED to 1.

    Two slits of equal width give a triangular slit function of base
    ``2 * bandpass``.  It is peak-normalized, NOT area-normalized, and that
    is the convention that makes the sweep come out in counts/s: a line much
    narrower than the bandpass then passes ENTIRELY when the monochromator
    sits on it, which is what a real instrument does.  (An area-normalized
    triangle would carry units of 1/nm and leave the sweep in counts/s/nm --
    a spectral density, not a count rate.)
    """
    x = np.abs(np.asarray(offsets_nm, dtype=float)) / float(bandpass)
    return np.maximum(0.0, 1.0 - x)


def sweep_curve(lam_nm, area_per_s, sigma_nm, bandpass, span=SWEEP_RANGE_NM):
    """Scan a monochromator across the line set and return ``(lambda0, S)``.

    ``S(lambda0)`` is the photon rate reaching the exit slit, in s^-1: the
    line spectrum ``E(lambda) = sum_i G_i(lambda)`` (each Gaussian carrying
    its own line's rate as its AREA) convolved with the peak-normalized slit
    triangle.  No instrument efficiency is folded in here -- the caller
    applies the slit, spectrometer and detector factors, so the same curve
    serves both the fiber-exit and the PMT scale.

    The grid resolves BOTH the bandpass and the narrowest line; the cold
    He I lines are an order of magnitude narrower than any bandpass here, so
    a bandpass-only criterion would under-sample them and lose their area.
    """
    lam_nm = np.asarray(lam_nm, dtype=float)
    sigma_nm = np.asarray(sigma_nm, dtype=float)
    lo, hi = span
    step = min(
        bandpass / SWEEP_POINTS_PER_BANDPASS,
        float(sigma_nm.min()) / SWEEP_POINTS_PER_SIGMA,
    )
    grid = np.arange(lo, hi + step, step)
    emission = spectral_density(grid, lam_nm, area_per_s, sigma_nm)

    half = int(np.ceil(bandpass / step))
    kernel = instrument_triangle(np.arange(-half, half + 1) * step, bandpass)
    swept = np.convolve(emission, kernel, mode="same") * step
    return grid, swept, step


def merge_features(lam_nm, bandpass):
    """Group lines the slit cannot separate: gaps of one bandpass or less."""
    order = np.argsort(np.asarray(lam_nm, dtype=float))
    groups, current = [], [int(order[0])]
    for a, b in zip(order[:-1], order[1:]):
        if lam_nm[b] - lam_nm[a] <= bandpass:
            current.append(int(b))
        else:
            groups.append(current)
            current = [int(b)]
    groups.append(current)
    return groups


def sweep_at_port(stage, port_record, fibers, knobs):
    """Peak count rates per resolvable feature, per dispersion and core.

    Only lines the window and fiber actually transmit enter the sweep -- a
    monochromator cannot scan onto a line the port window absorbed.

    The instrument gain splits in two.  ``f_slit * acceptance * grating x
    mirror`` is wavelength-flat and multiplies the whole curve; the
    detector's counting efficiency is NOT, so it is evaluated at each
    feature's centre for the tables and over the whole grid for the curve the
    figure draws.
    """
    lam_all = np.asarray(stage["lambda_nm"])
    sigma_all = np.asarray(port_record["doppler_sigma_nm"])
    lo, hi = SWEEP_RANGE_NM
    out = {"dispersions": {}, "knobs": dict(knobs)}

    for disp in knobs["dispersions"]:
        bp = bandpass_nm(knobs["slit_um"], disp)
        per_core = {}
        curve_for_figure = None
        for fib in fibers:
            if fib["na"] != fibers[0]["na"]:
                continue
            tag = fiber_tag(fib)
            rates = np.asarray(
                port_record["fibers"][tag]["photons_per_s_transmitted"]
            )
            keep = np.flatnonzero(
                (rates > 0.0) & (lam_all >= lo) & (lam_all <= hi)
            )
            if not keep.size:
                per_core[tag] = {"fiber": fib, "features": [], "empty": True}
                continue
            grid, swept, step = sweep_curve(
                lam_all[keep], rates[keep], sigma_all[keep], bp
            )
            f_slit = slit_fraction(knobs["slit_um"], fib["core_um"])
            acceptance = slit_acceptance_fraction(
                fib["na"], knobs["f_number"]
            )
            gain_optical = (
                f_slit * acceptance * knobs["grating_mirror_efficiency"]
            )
            feats = []
            for grp in merge_features(lam_all[keep], bp):
                idx = keep[grp]
                centre = float(np.mean(lam_all[idx]))
                halfwin = bp + 6.0 * float(sigma_all[idx].max())
                sel = np.flatnonzero(np.abs(grid - centre) <= halfwin)
                k = sel[int(np.argmax(swept[sel]))]
                eta_count = float(pmt_counting_efficiency(centre))
                feats.append(
                    {
                        "lines_nm": [float(v) for v in lam_all[idx]],
                        "centre_nm": centre,
                        "peak_lambda0_nm": float(grid[k]),
                        "counting_efficiency": eta_count,
                        "exit_counts_per_s": float(swept[k]),
                        "pmt_counts_per_s": float(
                            swept[k] * gain_optical * eta_count
                        ),
                        "merged": len(grp) > 1,
                    }
                )
            per_core[tag] = {
                "fiber": fib,
                "f_slit": f_slit,
                "acceptance": acceptance,
                "gain_optical": gain_optical,
                "features": feats,
                "empty": False,
            }
            if fib is fibers[0]:
                curve_for_figure = (
                    grid,
                    swept,
                    gain_optical * pmt_counting_efficiency(grid),
                    step,
                )
        out["dispersions"][f"{disp:g}"] = {
            "dispersion_nm_per_mm": float(disp),
            "bandpass_nm": bp,
            "cores": per_core,
            "curve": curve_for_figure,
        }
    return out


# --- datasheet transmission ------------------------------------------------


def window_transmission(material, lam_nm):
    """External transmission of the port window at each wavelength.

    Linear interpolation in wavelength on the datasheet curve reading in
    ``WINDOW_TRANSMISSION`` -- linear because the tabulated quantity is a
    transmission fraction on a smooth S-shaped edge, so a linear interpolant
    between adjacent readings is bounded by them.  Below the material's
    opaque edge the transmission is ZERO, not the edge value: a clamp there
    would invent light the window does not pass.  Above the reddest reading
    the plateau is held, which is what the datasheet curve does.
    """
    spec = WINDOW_TRANSMISSION[material]
    grid = np.array([w for w, _ in spec["points_nm_T"]], dtype=float)
    vals = np.array([t for _, t in spec["points_nm_T"]], dtype=float)
    lam = np.asarray(lam_nm, dtype=float)
    out = np.interp(lam, grid, vals, left=0.0, right=vals[-1])
    return np.where(lam < spec["opaque_below_nm"], 0.0, out)


def fiber_transmission(lam_nm, length_m):
    """Transmission of ``length_m`` of fiber, end-face Fresnel included.

    Bulk attenuation is ``10 ** (-alpha(lambda) * L / 10)`` with alpha in
    dB/m interpolated LOG-LINEARLY (linear in log10 alpha) on the datasheet
    reading in ``FIBER_ATTENUATION`` -- log-linear because alpha spans
    orders of magnitude across this range, where a linear interpolant
    between two readings would sit far above the curve it is standing in for.
    The datasheet's attenuation curve is BULK fiber only, so the two
    silica/air end faces are applied separately as a flat Fresnel factor.
    Below the fiber's stated transmission edge the result is ZERO.
    """
    spec = FIBER_ATTENUATION
    grid = np.array([w for w, _ in spec["points_nm_db_per_m"]], dtype=float)
    alpha_db = np.array(
        [a for _, a in spec["points_nm_db_per_m"]], dtype=float
    )
    lam = np.asarray(lam_nm, dtype=float)
    log_alpha = np.interp(
        lam,
        grid,
        np.log10(alpha_db),
        left=np.log10(alpha_db[0]),
        right=np.log10(alpha_db[-1]),
    )
    alpha = 10.0 ** log_alpha
    bulk = 10.0 ** (-alpha * float(length_m) / 10.0)
    ends = FIBER_END_FACE_TRANSMISSION ** 2
    out = bulk * ends
    return np.where(lam < spec["opaque_below_nm"], 0.0, out)


# --- products -------------------------------------------------------------


def _fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if x == 0.0:
        return "0"
    if 1.0e-3 <= abs(x) < 1.0e4:
        return f"{x:.4f}"
    return f"{x:.4e}"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out


def markdown_report(rep):
    """Assemble the markdown product."""
    L = []
    data = rep["data"]
    g = data["geometry"]
    L.append(f"# Line-resolved synthetic spectroscopy -- {Path(rep['h5']).name}")
    L.append("")
    L.append(
        "Produced by `scripts/line_radiation_sim1d.py`, READ-ONLY over the "
        "saved artifact. The adf11 PLT channels of "
        "`scripts/port_radiance_sim1d.py` are resolved into individual "
        "transitions with the OPEN-ADAS adf15 line list, reader and band "
        f"convention of `scripts/pec_band_fractions.py` (record: "
        f"`{BAND_SPLIT_NOTE}`). Do not edit by hand."
    )
    L.append("")

    # --- placement
    L.append("## Run and window")
    L.append("")
    r = data["run"]
    L.append(f"* artifact `{rep['h5']}`")
    L.append(
        f"* run: {r['steps']} steps, {r['saves']} saves, "
        f"{r['final_time_ms']:.4f} ms, status `{r['run_status']}`, "
        f"nx {r['nx']}, compiled kernels {r['compiled_kernels']}"
    )
    L.append(
        f"* plateau window {data['window_ms'][0]:g}-{data['window_ms'][1]:g} "
        f"ms on the MAIN-DISCHARGE clock (t = 0 at the first "
        f"`main_discharge` save, which sits at {data['origin_ms']:.4f} ms on "
        f"the run clock) -- the clock `compare_sim1d_es1.py` scores on. "
        f"{data['frames']} saves land in it; the window mean is an "
        "unweighted frame mean over them."
    )
    L.append(
        f"* {g['z_cm'].size} plasma-active cells, z "
        f"{g['z_cm'].min():.2f} to {g['z_cm'].max():.2f} cm, Rp "
        f"{g['Rp_cm'].min():.2f} to {g['Rp_cm'].max():.2f} cm, cell length "
        f"{g['length_cm'].min():.2f} to {g['length_cm'].max():.2f} cm"
    )
    c = data["closure"]
    L.append(
        f"* closure: atomic_rate_model `{c['atomic_rate_model']}`, "
        f"adas_low_te_extension {c['adas_low_te_extension']}, ne_floor "
        f"{c['ne_floor']:.3e} cm^-3, icool_recomb {c['icool_recomb']}"
    )
    L.append(
        "* the He I stage multiplies the IN-COLUMN neutral density `nn` -- "
        "the field `physics/energy.py` puts in `electron_neutral_cooling` "
        "and the field `port_radiance_sim1d.py` reads. Where a run also "
        "carries `nn_a` that is the annulus density and does not emit the "
        "column's He I light."
    )
    L.append("")

    # --- machine totals
    L.append("## Machine-total per-line radiated power")
    L.append("")
    L.append(
        "Window-mean emissivity times the artifact's own "
        "`geometry/plasma_volume_cm3`, summed over the plasma-active cells. "
        "No radial assumption enters these numbers."
    )
    L.append("")
    for key in STAGE_ORDER:
        s = rep["stages"][key]
        L.append(f"### {s['label']} -- `{s['spec_file']}`")
        L.append("")
        L.append(
            f"{len(s['lines'])} EXCIT transitions "
            f"(emissivity = PEC * n_e * {s['partner_label']}). "
            f"adf15 line sum {s['machine_W_total']:.4e} W; adf11 "
            f"`{s['plt_key']}` integral {s['adf11_machine_W']:.4e} W; "
            f"machine-integrated completeness C = "
            f"{_fmt(s['completeness_machine'])}."
        )
        L.append("")
        order = np.argsort(-s["machine_W"])
        rows = []
        for i in order:
            rows.append(
                [
                    str(s["isel"][i]),
                    f"{s['lambda_nm'][i]:.2f}",
                    s["band"][i],
                    _fmt(float(s["machine_W"][i])),
                    _fmt(float(s["machine_W"][i] / s["machine_W_total"])),
                    _fmt(float(s["machine_W"][i] / s["adf11_machine_W"])),
                ]
            )
        L.extend(
            _table(
                [
                    "ISEL",
                    "lambda_vac [nm]",
                    "band",
                    "machine power [W]",
                    "share of line sum",
                    "share of adf11 PLT",
                ],
                rows,
            )
        )
        L.append("")

    # --- window-transmissible totals
    L.append("## Window-transmissible machine totals")
    L.append("")
    L.append(
        "Each material passes the lines red-ward of its 50 % transmission "
        "cutoff. This is a HARD CUT at the cutoff, not a transmission curve: "
        "it brackets what reaches a detector, it does not model the roll-off. "
        "The cutoffs are ASSUMED representative commercial parts -- the "
        "material actually on the LAPD viewports and the fiber actually on "
        "the bench are not known here."
    )
    L.append("")
    rows = []
    for key in STAGE_ORDER:
        s = rep["stages"][key]
        for cut in rep["cutoffs"]:
            mask = transmits(cut, s["lambda_nm"])
            passed = float(s["machine_W"][mask].sum())
            rows.append(
                [
                    s["short"],
                    cut["material"],
                    f"{cut['cutoff_nm']:.0f}",
                    str(int(mask.sum())) + f"/{mask.size}",
                    _fmt(passed),
                    _fmt(passed / s["machine_W_total"])
                    if s["machine_W_total"] > 0.0
                    else "n/a",
                ]
            )
    L.extend(
        _table(
            [
                "stage",
                "material",
                "50 % cutoff [nm]",
                "lines passed",
                "transmitted power [W]",
                "share of stage line sum",
            ],
            rows,
        )
    )
    L.append("")
    for key in STAGE_ORDER:
        s = rep["stages"][key]
        sets = {
            tuple(transmits(cut, s["lambda_nm"]).tolist())
            for cut in rep["cutoffs"]
        }
        if len(sets) == 1:
            L.append(
                f"* {s['short']}: all three materials pass the SAME lines, so "
                "the choice among them does not change this table. The reason "
                "is the line list, not the cutoffs -- these adf15 files carry "
                "no He transition at all between 170 and 300 nm (the band "
                f"`{BAND_SPLIT_NOTE}` reports as exactly 0 at every point of "
                "its grid), so all three cutoffs fall in the same empty gap. "
                "A window-material decision cannot be made on these numbers; "
                "what decides it is whether the instrument needs the EUV "
                "lines below 170 nm, which NONE of the three passes."
            )
        else:
            L.append(
                f"* {s['short']}: the three materials pass different line "
                "sets, so the choice among them changes this table."
            )
    L.append("")

    # --- fiber
    L.append("## Synthetic fiber signal: collected, then transmitted")
    L.append("")
    win = WINDOW_TRANSMISSION[rep["window_material"]]
    L.append(
        "**The fiber sits OUTSIDE the port window looking in**, so two "
        "different quantities are reported per line and never conflated:"
    )
    L.append("")
    L.append(
        "* **collected** -- the flux inside the fiber's acceptance AT ITS "
        "FACE, `P_i = L_i * G`. This is what arrives; it is not what a "
        "detector sees."
    )
    L.append(
        "* **transmitted** -- what survives the window and the fiber run, "
        "`P_i * T_window(lambda) * T_fiber(lambda, L)`. **This is the "
        "photon-counter number.**"
    )
    L.append("")
    L.append(
        "**Upper bound: spot ~ core width, bare fiber, no collection "
        "optics.** Light enters a BARE fiber and the imaged spot at the "
        "plasma is taken to be approximately the fiber core width, so the "
        "collection area is the core area A = pi (d/2)^2 and the acceptance "
        "solid angle is Omega = pi NA^2. Radiance "
        "L_i = eps_i * L_chord / (4 pi) under the RADIALLY UNIFORM "
        "emissivity construction (see Assumptions); photon rate "
        "N_i = P_i * lambda_i / (hc). A real optical train loses light "
        "against this bound at every surface, and a collimating lens cannot "
        "beat it: a lens CONSERVES etendue, trading acceptance angle for "
        "collection area, so it changes the field of view, not G and not "
        "P_i."
    )
    L.append("")
    L.append(
        f"Window applied: **{win['material']}** ({win['thickness']}). Fiber "
        f"run: **{rep['fiber_length_m']:g} m**. Both are ASSUMED "
        "representative parts; the curves and their sources are tabulated "
        "in the next section."
    )
    L.append("")
    for k, fib in enumerate(rep["fibers"]):
        tag = "DEFAULT" if k == 0 else "reported alongside"
        L.append(
            f"* {tag}: core {fib['core_um']:.0f} um, NA {fib['na']:g}, "
            f"A = {fib['core_area_cm2']:.4e} cm^2, "
            f"Omega = {fib['omega_sr']:.4f} sr, "
            f"G = {fib['etendue_cm2_sr']:.4e} cm^2 sr"
        )
    L.append("")
    for pk in rep["fiber_ports"]:
        fp = rep["fiber_ports"][pk]
        L.append(
            f"### Port {fp['port']} -- z {fp['z_cell_cm']:.2f} cm "
            f"(law target {fp['z_want_cm']:.2f} cm), Rp "
            f"{fp['Rp_cm']:.2f} cm, chord {fp['chord_length_cm']:.2f} cm"
        )
        L.append("")
        for key in STAGE_ORDER:
            s = rep["stages"][key]
            got = fp["stages"][key]
            dflt = got["fibers"][fiber_tag(rep["fibers"][0])]
            L.append(
                f"**{s['label']}** -- collected vs transmitted at "
                f"{rep['fibers'][0]['core_um']:.0f} um / NA "
                f"{rep['fibers'][0]['na']:g}, sorted by transmitted rate."
            )
            L.append("")
            order = np.argsort(-np.asarray(dflt["photons_per_s_transmitted"]))
            rows = []
            for i in order:
                rows.append(
                    [
                        f"{s['lambda_nm'][i]:.2f}",
                        s["band"][i],
                        _fmt(float(got["radiance_W_cm2_sr"][i])),
                        _fmt(float(dflt["power_W"][i])),
                        _fmt(float(dflt["photons_per_s"][i])),
                        _fmt(float(got["T_window"][i])),
                        _fmt(float(got["T_fiber"][i])),
                        _fmt(float(got["T_total"][i])),
                        _fmt(float(dflt["power_W_transmitted"][i])),
                        _fmt(float(dflt["photons_per_s_transmitted"][i])),
                    ]
                )
            L.extend(
                _table(
                    [
                        "lambda_vac [nm]",
                        "band",
                        "radiance [W cm^-2 sr^-1]",
                        "collected [W]",
                        "collected [ph/s]",
                        "T_window",
                        "T_fiber",
                        "T_total",
                        "transmitted [W]",
                        "transmitted [ph/s]",
                    ],
                    rows,
                )
            )
            L.append("")
            brk = []
            for fib in rep["fibers"]:
                v = got["fibers"][fiber_tag(fib)]
                brk.append(
                    [
                        f"{fib['core_um']:.0f} um / NA {fib['na']:g}",
                        _fmt(fib["etendue_cm2_sr"]),
                        _fmt(v["power_W_total"]),
                        _fmt(v["photons_per_s_total"]),
                        _fmt(v["power_W_transmitted_total"]),
                        _fmt(v["photons_per_s_transmitted_total"]),
                        _fmt(v["ratio_to_machine_total"]),
                    ]
                )
            L.extend(
                _table(
                    [
                        "fiber (bare; spot ~ core width)",
                        "G [cm^2 sr]",
                        "collected [W]",
                        "collected [ph/s]",
                        "transmitted [W]",
                        "transmitted [ph/s]",
                        "P_transmitted / P_machine",
                    ],
                    brk,
                )
            )
            L.append("")

    # --- line shape
    L.append("## Synthetic line shape (Figure A)")
    L.append("")
    L.append(
        "Figure A draws each adf15 line as a Gaussian whose AREA is that "
        "line's chord radiance and whose width is the Doppler width at the "
        "EMITTING population's temperature, "
        "`FWHM = 7.716e-5 * lambda * sqrt(T[eV] / M[amu])` with "
        f"M = {HE_MASS_AMU:g}. The curve therefore integrates back to the "
        "per-line numbers tabulated above, and its HEIGHT encodes the line "
        "width: `peak = area / (sigma sqrt(2 pi))`."
    )
    L.append("")
    L.append(
        "**He II is broadened at the run's own ION temperature**, the "
        "plateau-mean saved `Ti` at that port's cell. **He I is broadened at "
        "the COLD neutral temperature** -- the engine's "
        f"`T_WALL_K = {T_WALL_K:g} K` = "
        f"{T_WALL_K * K_B_EV_PER_K:.4f} eV, imported from "
        "`physics/kinetic_neutrals.py` so it cannot drift from the engine. "
        "**The hot neutral channel is NOT represented**: a run that carries "
        "a hot population would emit a broader He I line than this figure "
        "draws, and nothing here brackets that."
    )
    L.append("")
    L.append(
        f"Both zoom panels share ONE offset axis (+/-"
        f"{INSET_HALF_WIDTH_NM:g} nm about the line centre) and ONE "
        "normalization, so the two thermal widths are read against a single "
        "ruler. The peaks differ by more than two orders of magnitude "
        "between the stages, which on a shared linear axis would leave the "
        "narrower line invisible, so both traces are PEAK-NORMALIZED and the "
        "axis label says so. The dashed He II trace carries the SAME AREA as "
        "the solid one and is divided by the SAME factor, so a colder ion "
        "population reads correctly as a taller, narrower line."
    )
    L.append("")
    rows = []
    drawn = {}
    for pk, key in rep["fiber_panels"]:
        fp = rep["fiber_ports"][pk]
        s_ = rep["stages"][key]
        got = fp["stages"][key]
        lam_all = np.asarray(s_["lambda_nm"])
        target = INSET_LINE_NM[key]
        m = resolve_target_line(
            lam_all, target, f"{s_['short']} zoom line (Figure A, markdown)"
        )
        drawn[key] = (fp, s_, got, m)
        rows.append(
            [
                str(fp["port"]),
                s_["short"],
                f"{lam_all[m]:.2f}",
                f"{got['emitter_T_eV']:.4f}",
                got["emitter_T_source"],
                f"{got['doppler_fwhm_nm'][m]:.5f}",
                "yes (solid)",
            ]
        )
        if key == "he1":
            t_cmp = INSET_COMPARISON_T_EV
            rows.append(
                [
                    str(fp["port"]),
                    s_["short"],
                    f"{lam_all[m]:.2f}",
                    f"{t_cmp:.4f}",
                    "fixed comparison temperature, not from the run",
                    f"{float(doppler_fwhm_nm(lam_all[m], t_cmp)):.5f}",
                    "yes (dashed)",
                ]
            )
        retired = INSET_RETIRED_LINES_NM.get(key)
        if retired is not None:
            r_nm, r_why = retired
            k = resolve_target_line(
                lam_all,
                r_nm,
                f"{s_['short']} retired zoom line (Figure A, markdown)",
            )
            rows.append(
                [
                    str(fp["port"]),
                    s_["short"],
                    f"{lam_all[k]:.2f}",
                    f"{got['emitter_T_eV']:.4f}",
                    got["emitter_T_source"],
                    f"{got['doppler_fwhm_nm'][k]:.5f}",
                    f"NOT DRAWN -- {r_why}",
                ]
            )
    L.extend(
        _table(
            [
                "port",
                "stage",
                "line [nm]",
                "T [eV]",
                "source of that T",
                "Doppler FWHM [nm]",
                "drawn in the zoom panel?",
            ],
            rows,
        )
    )
    L.append("")
    fp1, s1, got1, m1 = drawn["he1"]
    zoom_nm = float(np.asarray(s1["lambda_nm"])[m1])
    zoom_fwhm = float(got1["doppler_fwhm_nm"][m1])
    k468 = int(np.argmin(np.abs(np.asarray(s1["lambda_nm"]) - 468.65)))
    fwhm468 = float(got1["doppler_fwhm_nm"][k468])
    L.append(
        "**What the line shape omits.** Doppler broadening is the ONLY "
        "mechanism applied, to the adf15 line AS THAT FILE LISTS IT; Stark "
        "and instrumental widths are absent, and so is any fine or magnetic "
        "structure. Two consequences worth stating separately:"
    )
    L.append("")
    L.append(
        f"* The He II **468.65 nm** line (tabulated above, NOT drawn) is in "
        "reality a FINE-STRUCTURE MULTIPLET spread over about 0.07 nm, and "
        "at the LAPD's 1.4 kG it is additionally ZEEMAN-SPLIT by about "
        f"0.01 nm. That 0.07 nm spread EXCEEDS the {fwhm468:.5f} nm thermal "
        "FWHM tabulated for it, so a single Gaussian there would be the "
        "thermal envelope of a blended feature rather than a line profile -- "
        "no temperature could be read off its width without unfolding the "
        "multiplet first."
    )
    L.append(
        f"* The He II **{zoom_nm:.2f} nm** line now drawn in the zoom panel "
        "is the n = 5 -> 3 transition. It carries its own fine structure and "
        f"the same ~0.01 nm Zeeman splitting at 1.4 kG, and neither is "
        f"represented here either; against its {zoom_fwhm:.5f} nm thermal "
        "FWHM the Zeeman term is the smaller correction, but this remains a "
        "thermal envelope of the adf15 line as listed, not a synthetic "
        "profile to fit."
    )
    L.append("")

    # --- transmission curves
    L.append("## Window and fiber transmission curves applied")
    L.append("")
    L.append(
        "Every point below is a reading off a named manufacturer figure. "
        "The window curve is interpolated LINEARLY in wavelength (the "
        "tabulated quantity is a transmission fraction on a smooth edge, so "
        "a linear interpolant is bounded by its neighbours); the fiber "
        "attenuation is interpolated LOG-LINEARLY in alpha (it spans orders "
        "of magnitude, where a linear interpolant would sit far above the "
        "curve). Below a material's opaque edge the transmission is set to "
        "ZERO rather than clamped, because clamping there would invent "
        "light the window does not pass."
    )
    L.append("")
    L.append(f"### Window: {win['material']}")
    L.append("")
    L.append(f"* thickness: {win['thickness']}")
    L.append(f"* basis: {win['basis']}")
    L.append(f"* figure: {win['figure']}")
    L.append(f"* source: {win['source']}")
    L.append(f"* opaque below {win['opaque_below_nm']:g} nm (T set to 0)")
    L.append(f"* caveat: {win['caveat']}")
    L.append("")
    L.extend(
        _table(
            ["lambda_vac [nm]", "T_window (external)"],
            [[f"{w:g}", f"{t:.3f}"] for w, t in win["points_nm_T"]],
        )
    )
    L.append("")
    fa = FIBER_ATTENUATION
    L.append(f"### Fiber: {fa['material']}")
    L.append("")
    L.append(f"* run length applied: {rep['fiber_length_m']:g} m")
    L.append(f"* basis: {fa['basis']}")
    L.append(f"* figure: {fa['figure']}")
    L.append(f"* source: {fa['source']}")
    L.append(f"* opaque below {fa['opaque_below_nm']:g} nm (T set to 0)")
    L.append(f"* caveat: {fa['caveat']}")
    L.append(
        f"* end faces: T = {FIBER_END_FACE_TRANSMISSION:.3f} per silica/air "
        f"face, applied TWICE "
        f"({FIBER_END_FACE_TRANSMISSION ** 2:.3f} total). "
        f"{FIBER_END_FACE_SOURCE}"
    )
    L.append("")
    fd = FIBER_DATASHEET
    L.append(f"#### Identified bench fiber: {fd['model']}")
    L.append("")
    L.append(
        f"The collection fiber IS identified ({fd['datasheet']}) and its "
        f"core ({fd['core_um']:g} +/- {fd['core_tolerance_um']:g} um) and NA "
        f"({fd['na']:g}) are MEASURED cells used as this script's defaults. "
        "Its bulk ATTENUATION is not: the curve applied above is the "
        "representative stand-in, and the readings below are recorded only "
        "so the difference is visible."
    )
    L.append("")
    L.append(f"* construction: {fd['construction']}")
    L.append(f"* guaranteed attenuation: {fd['guaranteed_attenuation']}")
    L.append(f"* caveat: {fd['caveat']}")
    L.append("")
    L.extend(
        _table(
            [
                "lambda [nm]",
                "read [dB/km]",
                "bracket [dB/km]",
                "class",
                "note",
            ],
            [
                [
                    f"{lam_r:g}",
                    f"{val:g}",
                    f"{blo:g} - {bhi:g}",
                    "MEASURED",
                    note,
                ]
                for lam_r, val, blo, bhi, note in fd[
                    "plot_readings_db_per_km"
                ]
            ],
        )
    )
    L.append("")
    L.append(
        "There is NO row at 320.37 nm because the sheet's plot does not "
        f"reach it: the typical family curve begins near "
        f"{fd['plot_start_nm']:g} nm. The attenuation this instrument "
        "applies at that line stays ASSUMED."
    )
    L.append("")
    L.extend(
        _table(
            [
                "lambda_vac [nm]",
                "alpha [dB/m]",
                f"T_bulk at {rep['fiber_length_m']:g} m",
            ],
            [
                [
                    f"{w:g}",
                    f"{a:.4g}",
                    f"{10.0 ** (-a * rep['fiber_length_m'] / 10.0):.4f}",
                ]
                for w, a in fa["points_nm_db_per_m"]
            ],
        )
    )
    L.append("")

    L.append("## Synthetic collimated-fiber diagnostic")
    L.append("")
    L.append(
        "**Upper bound: spot ~ core width; no collection optics assumed.** "
        "Light enters a BARE fiber and the imaged spot at the plasma is "
        "taken to be approximately the fiber core width, so the collection "
        "area is the core area A = pi (d/2)^2 and the acceptance solid angle "
        "is Omega = pi NA^2. Radiance L_i = eps_i * L_chord / (4 pi) under "
        "the RADIALLY UNIFORM emissivity construction (see Assumptions); "
        "collected power P_i = L_i * G with G = A Omega; photon rate "
        "N_i = P_i * lambda_i / (hc). A real optical train loses light "
        "against this bound at every surface, and a collimating lens cannot "
        "beat it: a lens CONSERVES etendue, trading acceptance angle for "
        "collection area, so it changes the field of view, not G and not "
        "P_i."
    )
    L.append("")
    for k, fib in enumerate(rep["fibers"]):
        tag = "DEFAULT" if k == 0 else "reported alongside"
        L.append(
            f"* {tag}: core {fib['core_um']:.0f} um, NA {fib['na']:g}, "
            f"A = {fib['core_area_cm2']:.4e} cm^2, "
            f"Omega = {fib['omega_sr']:.4f} sr, "
            f"G = {fib['etendue_cm2_sr']:.4e} cm^2 sr"
        )
    L.append("")
    for pk in rep["fiber_ports"]:
        fp = rep["fiber_ports"][pk]
        L.append(
            f"### Port {fp['port']} -- z {fp['z_cell_cm']:.2f} cm "
            f"(law target {fp['z_want_cm']:.2f} cm), Rp "
            f"{fp['Rp_cm']:.2f} cm, chord {fp['chord_length_cm']:.2f} cm"
        )
        L.append("")
        for key in STAGE_ORDER:
            s = rep["stages"][key]
            got = fp["stages"][key]
            dflt = got["fibers"][fiber_tag(rep["fibers"][0])]
            L.append(f"**{s['label']}**")
            L.append("")
            order = np.argsort(-dflt["power_W"])
            rows = []
            for i in order:
                rows.append(
                    [
                        f"{s['lambda_nm'][i]:.2f}",
                        s["band"][i],
                        _fmt(float(fp["stages"][key]["radiance_W_cm2_sr"][i])),
                        _fmt(float(dflt["power_W"][i])),
                        _fmt(float(dflt["photons_per_s"][i])),
                        _fmt(float(dflt["ratio_to_machine"][i])),
                    ]
                )
            L.extend(
                _table(
                    [
                        "lambda_vac [nm]",
                        "band",
                        "radiance [W cm^-2 sr^-1]",
                        "fiber power [W]",
                        "fiber rate [photons/s]",
                        "P_fiber / P_machine",
                    ],
                    rows,
                )
            )
            L.append("")
            brk = []
            for fib in rep["fibers"]:
                v = got["fibers"][fiber_tag(fib)]
                brk.append(
                    [
                        f"{fib['core_um']:.0f} um / NA {fib['na']:g}",
                        _fmt(fib["etendue_cm2_sr"]),
                        _fmt(v["power_W_total"]),
                        _fmt(v["photons_per_s_total"]),
                        _fmt(v["ratio_to_machine_total"]),
                    ]
                )
            L.extend(
                _table(
                    [
                        "fiber (bare; spot ~ core width)",
                        "G [cm^2 sr]",
                        "stage power at fiber [W]",
                        "stage rate [photons/s]",
                        "P_fiber / P_machine",
                    ],
                    brk,
                )
            )
            L.append("")

    # --- self-consistency
    L.append("## Self-consistency")
    L.append("")
    pr = rep["port_radiance_crosscheck"]
    L.append(
        f"**Against `port_radiance_sim1d.py`.** Its e-i (PLT2) drive-window "
        f"mean emissivity at port {pr['port']} (cell {pr['cell']}, z "
        f"{pr['z_cell_cm']:.2f} cm, window "
        f"{pr['window_ms_run_clock'][0]}-{pr['window_ms_run_clock'][1]} ms "
        f"RUN clock) is {pr['port_radiance_drive_mean_W_cm3']:.6e} W cm^-3; "
        f"re-derived here on the same cell and window, "
        f"{pr['reconstructed_drive_mean_W_cm3']:.6e} W cm^-3 "
        f"(relative deviation {_fmt(pr['rel_dev'])})."
    )
    L.append("")
    L.append(
        "**Line sum vs C x adf11 cell power.** The identity the line "
        "decomposition rests on, checked cell by cell over the window."
    )
    L.append("")
    rows = []
    for key in STAGE_ORDER:
        s = rep["stages"][key]
        ci = s["identity_check"]
        vc = s["vectorization_check"]
        lc = s["ledger_check"]
        rows.append(
            [
                s["short"],
                str(ci["cells"]),
                _fmt(ci["max_rel_dev"]),
                f"{_fmt(ci['C_min'])} - {_fmt(ci['C_max'])}",
                _fmt(vc["max_rel_dev_vs_pec_at"]),
                (
                    _fmt(lc["window_mean_rel_dev"])
                    if lc["available"]
                    else "n/a"
                ),
                (
                    _fmt(lc["max_point_rel_dev"])
                    if lc["available"]
                    else lc["reason"]
                ),
            ]
        )
    L.extend(
        _table(
            [
                "stage",
                "cells checked",
                "max rel dev, line sum vs C x adf11",
                "per-cell C range",
                "max rel dev vs scalar pec_at",
                "adf11 vs ledger row, window mean",
                "adf11 vs ledger row, worst point",
            ],
            rows,
        )
    )
    L.append("")

    # --- clamp census
    L.append("## Clamp census")
    L.append("")
    L.append(
        "Every interpolation in this instrument CLAMPS NEAREST-EDGE outside "
        "its own grid, adf15 and adf11 alike. A clamped point is neither an "
        "extrapolation nor a physical value: below a low Te edge the "
        "coefficient is held while the true emission keeps collapsing, so a "
        "mean containing clamped points is BIASED HIGH."
    )
    L.append("")
    rows = []
    for key in STAGE_ORDER:
        s = rep["stages"][key]
        a15, a11 = s["adf15_clamp"], s["adf11_clamp"]
        rows.append(
            [
                s["short"],
                "adf15 " + s["spec_file"],
                f"{a15['grid_Te_min_eV']:g}-{a15['grid_Te_max_eV']:g}",
                f"{a15['grid_ne_min_cm3']:.3g}-{a15['grid_ne_max_cm3']:.3g}",
                str(a15["points"]),
                str(a15["points_clamped"]),
                _fmt(a15["fraction_clamped"]),
            ]
        )
        rows.append(
            [
                s["short"],
                f"adf11 PLT ({s['plt_key']})",
                f"{a11['table_Te_min_eV']:.4g}-{a11['table_Te_max_eV']:.4g}",
                "(shared adf11 grid)",
                str(a11["points"]),
                str(a11["points_below_Te_edge"] + a11["points_above_Te_edge"]),
                _fmt(a11["fraction_clamped"]),
            ]
        )
    L.extend(
        _table(
            [
                "stage",
                "table",
                "Te grid [eV]",
                "ne grid [cm^-3]",
                "points",
                "clamped",
                "fraction",
            ],
            rows,
        )
    )
    L.append("")
    for key in STAGE_ORDER:
        s = rep["stages"][key]
        a15 = s["adf15_clamp"]
        L.append(
            f"* {s['short']}: window state spans Te "
            f"{a15['min_Te_eV']:.4f}-{a15['max_Te_eV']:.4f} eV and lookup ne "
            f"{a15['min_ne_cm3']:.4e}-{a15['max_ne_cm3']:.4e} cm^-3. "
            + (
                "No point is clamped on either axis of either table, so every "
                "coefficient above is an interpolation between real tabulated "
                "values."
                if a15["points_clamped"] == 0
                and s["adf11_clamp"]["points_below_Te_edge"] == 0
                and s["adf11_clamp"]["points_above_Te_edge"] == 0
                else "CLAMPED POINTS ARE PRESENT -- the numbers for this "
                "stage are biased high by an amount this instrument cannot "
                "bound from the tables alone."
            )
        )
    L.append("")

    # --- caveats
    L.append("## Caveats inherited from the band-split record")
    L.append("")
    L.append(
        f"These are the rules `{BAND_SPLIT_NOTE}` states, quoted verbatim; "
        "they govern every number above, because this instrument evaluates "
        "the same line list through the same interpolator."
    )
    L.append("")
    L.append("**Te floor for quotability.**")
    L.append("")
    L.append(
        "> The He+ Te = 1 eV row is still not quotable. Every He II line "
        "needs ~40 eV to excite, so that row sits in a grid cell spanning "
        "~11 decades of PEC (0.689 -> 1.03 eV), where a log-linear "
        "interpolant stands in for a steep Arrhenius exponential -- order of "
        "magnitude at best. It is also irrelevant: at 1 eV the He+ line "
        "power is ~8 decades below the He0 line power at the same point. "
        "Quote the Te >= 5 eV rows."
    )
    L.append("")
    L.append(
        "The record's own headline numbers are stated \"at Te >= 2 eV\". "
        "Over this window:"
    )
    L.append("")
    for key in STAGE_ORDER:
        s = rep["stages"][key]
        sq = s["sub_quotable"]
        L.append(
            f"* {s['short']}: {sq['points_below']}/{sq['points']} "
            f"({100.0 * sq['fraction_below']:.1f} %) of the (frame, cell) "
            f"evaluation points sit below Te = "
            f"{sq['Te_min_quotable_eV']:g} eV."
            + (
                " Those points carry He II line power that is orders of "
                "magnitude below the emitting cells and contribute "
                "negligibly to the totals, but they are inside the "
                "log-linear-across-a-steep-exponential regime the rule "
                "indicts and are not separately quotable."
                if key == "he1" and sq["points_below"]
                else ""
            )
        )
    L.append("")
    L.append("**He I bracket.**")
    L.append("")
    L.append(
        "> Lower bound charges the visible sum against the FULL PLT line "
        "power (every line the adf15 file omits assumed non-transmissible); "
        "upper bound charges it against the tabulated lines only (omitted "
        "lines assumed to share the tabulated visible fraction). The two "
        "collapse onto one number as C -> 1."
    )
    L.append("")
    he0 = rep["stages"]["he0"]
    lam_max = float(np.max(he0["lambda_nm"]))
    L.append(
        f"This bracket is WIDE for He I and it matters here. The "
        f"`pec96#he_pju#he0` line list carries {len(he0['lines'])} EXCIT "
        f"transitions reaching only to {lam_max:.2f} nm: it holds no "
        "1083 nm line (the 2s3S-2p3P triplet, the brightest He I feature in "
        "many discharges) and no transitions from n >= 5 upper levels. Its "
        "machine-integrated completeness on this window is C = "
        f"{_fmt(he0['completeness_machine'])}, so the missing "
        f"{_fmt(1.0 - (he0['completeness_machine'] or 0.0))} of the adf11 "
        "He I line power is unaccounted for by name. A per-line He I number "
        "in this document is therefore a LOWER BOUND on that line's stage: "
        "the omitted lines could add to any band, and nothing here places "
        "them."
    )
    L.append("")
    he1 = rep["stages"]["he1"]
    L.append(
        f"The He II list is the more complete of the two (C = "
        f"{_fmt(he1['completeness_machine'])}) but is overwhelmingly EUV: "
        "the 30.4 nm class resonance line dominates PLT2, so an He II "
        "RADIATED POWER is not a visible-light prediction. The "
        "window-transmissible table above is the number to quote for a "
        "glass- or silica-windowed instrument."
    )
    L.append("")

    # --- monochromator sweep
    if rep["sweeps"] is not None:
        L.extend(markdown_sweep(rep))

    # --- window cutoff provenance
    L.append("## Window cutoffs and their sources")
    L.append("")
    L.append(
        "Each is an ASSUMED representative commercial part. **The material "
        "on the LAPD viewports and the identity of the collection fiber are "
        "NOT known to this script and nothing here is a statement about the "
        "installed hardware.** No manufacturer tabulates a 50 % point for "
        "any of the three, so every number below is a curve reading or an "
        "interpolation and carries its own bar."
    )
    L.append("")
    for cut in rep["cutoffs"]:
        L.append(
            f"### {cut['material']} -- 50 % at "
            f"{cut['cutoff_nm']:.0f} +/- {cut['uncertainty_nm']:.0f} nm"
        )
        L.append("")
        L.append(f"* basis: {cut['basis']}")
        L.append(f"* source: {cut['source']}")
        L.append(f"* caveat: {cut['caveat']}")
        L.append("")

    # --- assumptions
    L.append("## Assumptions")
    L.append("")
    L.extend(
        _table(
            ["quantity", "value", "class", "source / basis"],
            [
                [a["quantity"], a["value"], a["class"], a["source"]]
                for a in ASSUMPTIONS
            ],
        )
    )
    L.append("")
    L.append(
        "The three window cutoffs are ASSUMED representative parts. The "
        "material of the LAPD viewports is NOT known to this script, and "
        "neither is the attenuation of the identified collection fiber at "
        "the 320.37 nm He II line, so nothing in that section is a "
        "statement about the installed viewport or about the FT1000UMT's "
        "own transmission."
    )
    L.append("")
    return "\n".join(L)


# --- figures --------------------------------------------------------------

SWEEP_KNOB_CLASS = (
    (
        "slit width",
        "slit_um",
        "um",
        "MEASURED",
        "instrument setting as read by the operator; the micrometer readout "
        "is in 10-micron thimble divisions (datasheet). Equal entrance and "
        "exit slits, butt-coupled fiber, 1:1 imaging",
    ),
    (
        "reciprocal linear dispersion",
        "dispersion_nm_per_mm",
        "nm/mm",
        "MEASURED",
        "McPherson Model 209 datasheet grating table, 2400 g/mm cell "
        "(0.62 nm/mm at 1200 g/mm). Pass --dispersion-bracket to carry the "
        "placeholder 1.5 / 2.5 / 4.0 nm/mm columns beside it",
    ),
    (
        "instrument f-number",
        "f_number",
        "-",
        "MEASURED",
        "McPherson Model 209 datasheet p.1 ('11.6 with smaller grating'); "
        "its p.3 drawing title block says f/7 instead. The sheet contradicts "
        "itself, f/9.4 is the value of record and [f/7, f/9.4] the disclosed "
        "bracket",
    ),
    (
        "grating x mirror efficiency",
        "grating_mirror_efficiency",
        "-",
        "ASSUMED",
        "what is left of the spectrometer's optical throughput once the "
        "fiber-to-slit acceptance is computed from the f-number; the 209's "
        "optics are Al + MgF2 and McPherson states no efficiency",
    ),
)


def markdown_sweep(rep):
    """The synthetic-monochromator section of the markdown product."""
    knobs = rep["sweep_knobs"]
    L = ["## Synthetic monochromator sweep (Figure C)", ""]
    L.append(
        "What a slit-and-PMT monochromator would record as it is scanned "
        "through lambda0, built ONLY from lines the port window and the "
        "fiber actually transmit -- a monochromator cannot scan onto a line "
        "the window absorbed, so the EUV that carries most of the radiated "
        "power is absent by construction."
    )
    L.append("")
    L.append(
        "`S(lambda0) = f_slit * a_slit * eta_gm * eta_count(lambda0) * "
        "sum_i integral G_i(lambda) I(lambda - lambda0) dlambda`, with `G_i` "
        "the thermal Gaussian of Figure A carrying that line's TRANSMITTED "
        "photon rate as its AREA, and `I` the equal-slit triangle of base "
        "`2 * bandpass`. `a_slit` is the fiber-to-slit acceptance, `eta_gm` "
        "the grating x mirror efficiency and `eta_count` the detector's "
        "counting efficiency -- the only one of the four that varies with "
        "wavelength."
    )
    L.append("")
    L.append(
        "**Slit-function normalization.** `I` is PEAK-normalized to 1, not "
        "area-normalized. That is the convention that leaves `S` in "
        "counts/s: a line much narrower than the bandpass then passes "
        "ENTIRELY when the monochromator sits on it, which is what a real "
        "instrument does. Dividing by `integral I dlambda` as well would "
        "carry a spare factor of the bandpass and leave a spectral DENSITY "
        "in counts/s/nm rather than a count rate."
    )
    L.append("")
    L.append(
        "**`f_slit = min(1, slit / core)`** is the SPATIAL crop only: the "
        "butt-coupled core is imaged 1:1 onto the slit and the slit is "
        "narrower than the core at every combination tabulated here, so most "
        "of the collected light never enters the spectrometer. Note the "
        "consequence in the tables: collected flux grows as core^2 while "
        "`f_slit` falls as 1/core, so the PMT rate grows only LINEARLY with "
        "core diameter once the core exceeds the slit."
    )
    L.append("")
    L.append("### Knobs")
    L.append("")
    rows = []
    for label, key, unit, cls, note in SWEEP_KNOB_CLASS:
        rows.append([label, f"{knobs[key]:g}", unit, cls, note])
    L.extend(_table(["knob", "value", "unit", "class", "note"], rows))
    L.append("")
    L.append(
        f"Instrument: **{MONOCHROMATOR['model']}** "
        f"({MONOCHROMATOR['datasheet']}); detector "
        f"**{PMT['model']}** ({PMT['datasheet']}). Datasheet cells not "
        "otherwise used here, recorded so the instrument is identified: "
        f"{MONOCHROMATOR['resolution']}; {MONOCHROMATOR['wavelength']}; "
        f"slits {MONOCHROMATOR['slits']}. Detector: spectral response "
        f"{PMT['spectral_response_nm'][0]:g}-"
        f"{PMT['spectral_response_nm'][1]:g} nm peaking at "
        f"{PMT['peak_nm']:g} nm, effective area {PMT['effective_area']}, "
        f"{PMT['linearity']}, dark count {PMT['dark_count']}."
    )
    L.append("")
    L.append("### Fiber-to-slit acceptance")
    L.append("")
    L.append(
        "The fiber butts against the entrance slit with NO coupling optics, "
        "so the spectrometer accepts only the part of the fiber's output "
        "cone inside its own `NA_instr = 1 / (2 f/#)`: "
        "`a_slit = (NA_instr / NA_fiber)^2`, DERIVED from two MEASURED "
        "datasheet values and capped at 1. Both members of the datasheet's "
        "own f-number contradiction are carried."
    )
    L.append("")
    na_f = float(rep["fibers"][0]["na"])
    rows = []
    for fn in F_NUMBER_BRACKET:
        rows.append(
            [
                f"f/{fn:g}"
                + (" (of record)" if fn == knobs["f_number"] else ""),
                f"{0.5 / fn:.4f}",
                f"{na_f:g}",
                f"{slit_acceptance_fraction(na_f, fn):.4f}",
                f"{100.0 * slit_acceptance_fraction(na_f, fn):.2f} %",
            ]
        )
    L.extend(
        _table(
            [
                "f-number",
                "NA_instr = 1/(2 f/#)",
                "NA_fiber",
                "a_slit",
                "accepted",
            ],
            rows,
        )
    )
    L.append("")
    a_rec = slit_acceptance_fraction(na_f, knobs["f_number"])
    L.append(
        f"Combined spectrometer factor at the defaults: `a_slit * eta_gm = "
        f"{a_rec:.4f} * {knobs['grating_mirror_efficiency']:g} = "
        f"{a_rec * knobs['grating_mirror_efficiency']:.4e}`. The single "
        "ASSUMED throughput this script carried before the instrument was "
        "identified was 0.12, so every count rate below is smaller by a "
        f"factor "
        f"{0.12 / (a_rec * knobs['grating_mirror_efficiency']):.3g} on this "
        "factor alone."
    )
    L.append("")
    L.append("### Detector counting efficiency")
    L.append("")
    L.append(
        "The datasheet gives a count sensitivity `S` in counts per second "
        "per picowatt, which already folds photocathode efficiency, "
        "collection and counting into one number. `eta_count = S / (1 pW / "
        "(hc/lambda))` turns it into a dimensionless efficiency per INCIDENT "
        "photon -- hence counting efficiency, NOT quantum efficiency. `S` is "
        "interpolated LOG-LINEARLY between the cells below; the sheet gives "
        "them only as `Typ.` with no min/max, so `eta_count` carries no "
        "tolerance. Outside the head's rated "
        f"{PMT['spectral_response_nm'][0]:g}-"
        f"{PMT['spectral_response_nm'][1]:g} nm response `eta_count` is "
        "ZERO: the tube is not specified there, so the sweep cannot book "
        "counts on a line it could not register."
    )
    L.append("")
    rows = []
    for lam_c, s_c in PMT_COUNT_SENSITIVITY_PER_PW:
        lo_r, hi_r = PMT["spectral_response_nm"]
        rows.append(
            [
                f"{lam_c:g}",
                f"{s_c:.3g}",
                f"{pmt_counting_efficiency(lam_c):.4f}",
                "rated" if lo_r <= lam_c <= hi_r else "OUTSIDE rated range",
            ]
        )
    L.extend(
        _table(
            [
                "lambda [nm]",
                "S [s^-1 pW^-1] (Typ.)",
                "eta_count",
                "within rated response?",
            ],
            rows,
        )
    )
    L.append("")
    L.append(
        f"Fiber transmission applied: **{rep['fiber_ports'][rep['fiber_panels'][0][0]]['stages'][rep['fiber_panels'][0][1]]['T_fiber_source']}**. "
        f"The operator-quoted flat figure is "
        f"{QUOTED_FLAT_FIBER_TRANSMISSION:g} (about 2-3x the datasheet "
        "chain); pass `--fiber-transmission "
        f"{QUOTED_FLAT_FIBER_TRANSMISSION:g}` to substitute it, which scales "
        "every count rate below by the ratio of the two."
    )
    L.append("")
    L.append("### Bandpass per dispersion")
    L.append("")
    rows = []
    for disp in knobs["dispersions"]:
        rows.append(
            [
                f"{disp:g}",
                f"{knobs['slit_um']:g}",
                f"{bandpass_nm(knobs['slit_um'], disp):.4f}",
                f"{2.0 * bandpass_nm(knobs['slit_um'], disp):.4f}",
            ]
        )
    L.extend(
        _table(
            [
                "dispersion [nm/mm]",
                "slit [um]",
                "bandpass Dlambda_bp [nm]",
                "slit-function base [nm]",
            ],
            rows,
        )
    )
    L.append("")
    L.append("### Peak count rate per resolvable feature")
    L.append("")
    L.append(
        "Peak of the sweep at each feature, in counts/ms. `fiber exit` sets "
        "`f_slit = a_slit = eta_gm = eta_count = 1` and is what arrives at "
        "the spectrometer entrance; `PMT` applies all four. `eta_count` is "
        "evaluated at the feature centre, so a merged feature carries the "
        "efficiency of its centroid."
    )
    L.append("")
    for pkey, key in rep["fiber_panels"]:
        st = rep["stages"][key]
        fp = rep["fiber_ports"][pkey]
        sw = rep["sweeps"][pkey]
        L.append(f"**Port {fp['port']} -- {st['label']}**")
        L.append("")
        rows = []
        for disp in knobs["dispersions"]:
            block = sw["dispersions"][f"{disp:g}"]
            for core in SWEEP_CORE_COLUMNS_UM:
                tag = f"{core:.0f}um_NA{rep['fibers'][0]['na']:g}"
                rec = block["cores"].get(tag)
                if rec is None or rec.get("empty"):
                    continue
                for feat in sorted(
                    rec["features"], key=lambda f: -f["pmt_counts_per_s"]
                ):
                    rows.append(
                        [
                            f"{disp:g}",
                            f"{block['bandpass_nm']:.4f}",
                            f"{core:.0f}",
                            f"{rec['f_slit']:.4f}",
                            ", ".join(f"{v:.2f}" for v in feat["lines_nm"]),
                            "yes" if feat["merged"] else "no",
                            f"{feat['counting_efficiency']:.4f}",
                            _fmt(feat["exit_counts_per_s"] * 1.0e-3),
                            _fmt(feat["pmt_counts_per_s"] * 1.0e-3),
                        ]
                    )
        L.extend(
            _table(
                [
                    "dispersion [nm/mm]",
                    "bandpass [nm]",
                    "core [um]",
                    "f_slit",
                    "line(s) [nm]",
                    "merged?",
                    "eta_count",
                    "fiber exit [counts/ms]",
                    "PMT [counts/ms]",
                ],
                rows,
            )
        )
        L.append("")
    return L


CUTOFF_STYLE = ("tab:purple", "tab:olive", "tab:brown")


def figure_chord_power(rep, path_stem, dpi=180):
    """Figure A: synthetic thermally-broadened spectrum at each port.

    Every adf15 line is drawn as a Gaussian whose AREA is that line's chord
    radiance and whose FWHM is the Doppler width at the emitting population's
    temperature, so the curve integrates back to the tabulated per-line
    numbers and its HEIGHT encodes the line width.  On a 20-750 nm axis the
    lines are spikes; that is the intended reading, and the insets resolve
    one line each.  No transmission is applied -- this is the emission at the
    chord, and the cutoff separators show what is out of optical reach.
    """
    cuts = rep["cutoffs"]
    lo, hi = SPECTRUM_RANGE_NM
    fig = plt.figure(figsize=(13.6, 8.8), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=(3.0, 1.18))
    axes = [fig.add_subplot(gs[r, 0]) for r in range(2)]
    zooms = [fig.add_subplot(gs[r, 1]) for r in range(2)]

    zoom_ymax = 1.0
    for ax, axin, (pkey, key) in zip(axes, zooms, rep["fiber_panels"]):
        fp = rep["fiber_ports"][pkey]
        s = rep["stages"][key]
        got = fp["stages"][key]
        lam = np.asarray(s["lambda_nm"])
        area = np.asarray(got["radiance_W_cm2_sr"])
        sigma = np.asarray(got["doppler_sigma_nm"])
        fwhm = np.asarray(got["doppler_fwhm_nm"])
        inside = (lam >= lo) & (lam <= hi)

        grid = spectrum_grid(lam[inside], sigma[inside], lo, hi)
        dens = spectral_density(grid, lam[inside], area[inside], sigma[inside])
        peak = max(float(dens.max()), 1.0e-30)
        floor = peak * 1.0e-8
        ax.plot(grid, np.maximum(dens, floor), color="tab:blue", lw=0.9,
                zorder=3)
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(floor, peak * 30.0)
        ax.grid(True, alpha=0.22, zorder=0)
        ax.set_ylabel(
            "spectral radiance\n[W cm$^{-2}$ sr$^{-1}$ nm$^{-1}$]", fontsize=9
        )

        for k, cut in enumerate(cuts):
            if not (lo <= cut["cutoff_nm"] <= hi):
                continue
            ax.axvline(cut["cutoff_nm"], color=CUTOFF_STYLE[k], ls="--",
                       lw=1.5, zorder=5)
            ax.annotate(
                f"{cut['short']} {cut['cutoff_nm']:.0f} nm",
                xy=(cut["cutoff_nm"], 0.985),
                xycoords=("data", "axes fraction"),
                xytext=(-3, 0),
                textcoords="offset points",
                fontsize=6.8,
                color=CUTOFF_STYLE[k],
                ha="right",
                va="top",
                rotation=90,
                zorder=7,
            )
        placed = []
        for idx in np.argsort(-np.where(inside, area, 0.0))[:LABELLED_LINES]:
            if not inside[idx]:
                continue
            if any(abs(lam[idx] - q) < 0.022 * (hi - lo) for q in placed):
                continue
            placed.append(lam[idx])
            k = int(np.argmin(np.abs(grid - lam[idx])))
            ax.annotate(
                f"{lam[idx]:.1f} nm",
                xy=(lam[idx], max(dens[k], floor)),
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=6.4,
                rotation=90,
                ha="center",
                va="bottom",
                color="0.2",
                zorder=6,
            )
        ax.set_title(
            f"Port {fp['port']} ({s['label']})  --  z {fp['z_cell_cm']:.1f} "
            f"cm, chord {fp['chord_length_cm']:.1f} cm;  emitter T = "
            f"{got['emitter_T_eV']:.4g} eV ({got['emitter_T_source']})",
            fontsize=9.5,
        )

        # --- zoom: one line resolved, on the SHARED offset axis.
        # Both panels use one Delta-lambda span and one normalization, so the
        # two thermal widths are read against a single ruler.  The peaks
        # themselves differ by more than two orders of magnitude between the
        # stages, which would leave the narrower line invisible on a shared
        # linear axis -- hence peak-normalized, stated in the axis label.
        target = INSET_LINE_NM[key]
        m = resolve_target_line(
            lam, target, f"{s['short']} zoom line (Figure A)"
        )
        half = INSET_HALF_WIDTH_NM
        dl = np.linspace(-half, half, 2400)
        di = spectral_density(dl + lam[m], [lam[m]], [area[m]], [sigma[m]])
        norm = max(float(di.max()), 1.0e-300)
        axin.plot(dl, di / norm, color="tab:red", lw=1.5,
                  label=f"T = {got['emitter_T_eV']:.3g} eV")
        note = (
            f"FWHM {fwhm[m]:.4f} nm @ {got['emitter_T_eV']:.3g} eV"
        )
        if key == "he1":
            t_cmp = INSET_COMPARISON_T_EV
            f_cmp = float(doppler_fwhm_nm(lam[m], t_cmp))
            s_cmp = f_cmp * FWHM_TO_SIGMA
            d_cmp = spectral_density(dl + lam[m], [lam[m]], [area[m]], [s_cmp])
            # Same AREA, normalized by the SAME factor as the solid trace, so
            # the colder line reads correctly as taller and narrower.
            axin.plot(dl, d_cmp / norm, color="0.35", ls="--", lw=1.3,
                      label=f"T = {t_cmp:g} eV")
            zoom_ymax = max(zoom_ymax, float((d_cmp / norm).max()))
            note += f"\nFWHM {f_cmp:.4f} nm @ {t_cmp:g} eV"
            axin.legend(fontsize=6.4, loc="upper right", framealpha=0.9)
        axin.set_xlim(-half, half)
        axin.set_title(
            f"resolved: {lam[m]:.2f} nm", fontsize=8.5
        )
        axin.tick_params(labelsize=6.6)
        axin.set_xlabel(
            r"$\Delta\lambda$ from line centre [nm]", fontsize=7.2,
            labelpad=1.0,
        )
        axin.set_ylabel(
            "spectral radiance,\npeak-normalized", fontsize=7.2
        )
        axin.grid(True, alpha=0.2)
        axin.annotate(
            note,
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            fontsize=6.4,
            va="top",
            color="0.25",
        )

    for axin in zooms:
        axin.set_ylim(0.0, zoom_ymax * 1.12)

    axes[1].set_xlabel(r"$\lambda_{vac}$ [nm]")
    fig.suptitle(
        f"Synthetic thermal spectrum at the chord -- {Path(rep['h5']).name}\n"
        f"each adf15 line is a Gaussian of AREA = its chord radiance and "
        f"Doppler FWHM = 7.716e-5 $\\lambda\\sqrt{{T/M}}$ (M = "
        f"{HE_MASS_AMU:g}); plateau "
        f"{rep['data']['window_ms'][0]:g}-"
        f"{rep['data']['window_ms'][1]:g} ms, main-discharge clock",
        fontsize=10.2,
    )
    fig.get_layout_engine().set(rect=(0.0, 0.058, 1.0, 0.938))
    fig.text(
        0.5,
        0.030,
        "Doppler broadening ONLY, on the adf15 line as listed: fine "
        "structure and the ~0.01 nm Zeeman splitting at 1.4 kG are NOT "
        "represented, nor is any instrumental width. The He II 468.65 nm "
        "multiplet (~0.07 nm spread, wider than its own thermal width) is "
        "tabulated in the markdown but deliberately not drawn.",
        fontsize=7.0,
        color="0.35",
        ha="center",
    )
    fig.text(
        0.5,
        0.017,
        "Radiance assumes emissivity RADIALLY UNIFORM across the plasma disc "
        "-- the 1D model has no radial profile. No transmission is applied "
        "here; this is the emission at the chord.",
        fontsize=7.0,
        color="0.35",
        ha="center",
    )
    fig.text(
        0.5,
        0.004,
        "Cutoffs are ASSUMED representative commercial parts read off "
        "manufacturer curves, NOT LAPD hardware; see the markdown product "
        "for each source, thickness/length and caveat.",
        fontsize=7.0,
        color="0.35",
        ha="center",
    )
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=dpi)
    plt.close(fig)



#: Bar colours for the three reported core diameters, bluest = smallest.
CORE_COLORS = ("#9ecae1", "#4292c6", "#08519c")


def figure_photon_counter(rep, path_stem, dpi=180):
    """Figure B: transmitted photon rate per line, transmissible lines only.

    Bars are what a counter at the far end of the fiber would register:
    collected flux times the datasheet window and fiber transmissions.
    Lines the window blocks are absent by construction -- they contribute
    nothing to a photon counter and their place is Figure A.
    """
    cuts = rep["cutoffs"]
    cores = [f for f in rep["fibers"] if f["na"] == rep["fibers"][0]["na"]]
    cores = sorted(cores, key=lambda f: f["core_um"])
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.8), layout="constrained")

    for ax, (pkey, key) in zip(axes, rep["fiber_panels"]):
        fp = rep["fiber_ports"][pkey]
        s = rep["stages"][key]
        got = fp["stages"][key]
        keep = np.flatnonzero(np.asarray(got["T_total"]) > 0.0)
        if not keep.size:
            ax.text(
                0.5,
                0.5,
                "no line of this stage is transmitted by "
                f"{WINDOW_TRANSMISSION[rep['window_material']]['short']} "
                f"+ {rep['fiber_length_m']:g} m of fiber",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="0.3",
            )
            ax.set_yscale("log")
            continue
        lam = s["lambda_nm"][keep]
        order = np.argsort(lam)
        x = np.arange(keep.size, dtype=float)
        width = 0.8 / len(cores)
        top, bottom = 0.0, np.inf
        for c, fib in enumerate(cores):
            v = np.asarray(
                got["fibers"][fiber_tag(fib)]["photons_per_s_transmitted"]
            )[keep][order]
            top = max(top, float(v.max()))
            pos = v[v > 0.0]
            if pos.size:
                bottom = min(bottom, float(pos.min()))
            ax.bar(
                x + (c - (len(cores) - 1) / 2.0) * width,
                v,
                width=width,
                color=CORE_COLORS[c % len(CORE_COLORS)],
                edgecolor="black",
                linewidth=0.6,
                label=f"{fib['core_um']:.0f} $\\mu$m core",
                zorder=3,
            )
        if not np.isfinite(bottom):
            bottom = top * 1.0e-6
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{v:.1f}" for v in lam[order]], fontsize=7.5, rotation=45
        )
        ax.set_yscale("log")
        ax.set_ylim(max(bottom / 6.0, top * 1.0e-9), top * 30.0)
        # A stage with only a line or two would otherwise draw bars a third
        # of the axis wide; hold a minimum number of slots so the bar width
        # reads the same on every panel.
        ax.set_xlim(-0.75, max(float(keep.size), 5.0) - 0.25)
        ax.grid(True, axis="y", alpha=0.22, zorder=0)
        ax.set_ylabel("transmitted photons s$^{-1}$")
        ax.legend(fontsize=7.0, ncol=len(cores), loc="upper left",
                  framealpha=0.92)
        dflt = got["fibers"][fiber_tag(rep["fibers"][0])]
        ax.set_title(
            f"Port {fp['port']} ({s['label']})  --  z {fp['z_cell_cm']:.1f} "
            f"cm; {keep.size} of {s['lambda_nm'].size} lines transmitted; "
            f"stage total at {rep['fibers'][0]['core_um']:.0f} "
            f"$\\mu$m / NA {rep['fibers'][0]['na']:g}: "
            f"{dflt['photons_per_s_transmitted_total']:.3e} ph/s, "
            f"{dflt['power_W_transmitted_total']:.3e} W",
            fontsize=9.5,
        )
    axes[1].set_xlabel(r"$\lambda_{vac}$ [nm]")
    win = WINDOW_TRANSMISSION[rep["window_material"]]
    fig.suptitle(
        f"Synthetic photon counter -- {Path(rep['h5']).name}\n"
        f"upper bound: spot $\\approx$ core width, bare fiber outside the "
        f"window, no collection optics\n"
        f"datasheet {win['short']} window ({win['thickness']}) + "
        f"{rep['fiber_length_m']:g} m fiber transmission applied; "
        f"NA {rep['fibers'][0]['na']:g}; plateau "
        f"{rep['data']['window_ms'][0]:g}-"
        f"{rep['data']['window_ms'][1]:g} ms, main-discharge clock",
        fontsize=9.8,
    )
    fig.get_layout_engine().set(rect=(0.0, 0.038, 1.0, 0.912))
    fig.text(
        0.5,
        0.012,
        "Only lines the window and fiber actually pass are shown; the "
        "blocked lines (EUV) are in Figure A. Window and fiber curves are "
        "ASSUMED representative commercial parts, NOT LAPD hardware.",
        fontsize=7.0,
        color="0.35",
        ha="center",
    )
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=dpi)
    plt.close(fig)


def figure_sweep(rep, path_stem, dpi=180):
    """Figure C: the monochromator sweep as a slit + PMT would record it.

    Count rate at the PMT against the wavelength the monochromator is set to,
    at the default knobs.  A companion column resolves one feature so the
    instrument-limited shape is visible: at these slit widths the bandpass is
    comparable to the thermal width, so the recorded feature is neither the
    line nor the slit function but their convolution.
    """
    cuts = rep["cutoffs"]
    knobs = rep["sweep_knobs"]
    disp_key = f"{knobs['dispersion_nm_per_mm']:g}"
    lo, hi = SWEEP_RANGE_NM

    fig = plt.figure(figsize=(13.6, 8.8), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=(3.0, 1.18))
    axes = [fig.add_subplot(gs[r, 0]) for r in range(2)]
    zooms = [fig.add_subplot(gs[r, 1]) for r in range(2)]

    for ax, axz, (pkey, key) in zip(axes, zooms, rep["fiber_panels"]):
        st = rep["stages"][key]
        fp = rep["fiber_ports"][pkey]
        sw = rep["sweeps"][pkey]["dispersions"][disp_key]
        bp = sw["bandpass_nm"]
        curve = sw["curve"]
        if curve is None:
            for a_ in (ax, axz):
                a_.text(
                    0.5, 0.5,
                    "no transmitted line in the sweep range",
                    transform=a_.transAxes, ha="center", va="center",
                    fontsize=10, color="0.3",
                )
            continue
        grid, swept, gain_lambda, _step = curve
        counts_ms = swept * gain_lambda * 1.0e-3
        top = max(float(counts_ms.max()), 1.0e-30)
        floor = top * 1.0e-7
        ax.plot(grid, np.maximum(counts_ms, floor), color="tab:blue", lw=0.9,
                zorder=3)
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(floor, top * 30.0)
        ax.grid(True, alpha=0.22, zorder=0)
        ax.set_ylabel("PMT rate [counts ms$^{-1}$]", fontsize=9)

        for k, cut in enumerate(cuts):
            if not (lo <= cut["cutoff_nm"] <= hi):
                continue
            ax.axvline(cut["cutoff_nm"], color=CUTOFF_STYLE[k], ls="--",
                       lw=1.5, zorder=5)
            ax.annotate(
                f"{cut['short']} {cut['cutoff_nm']:.0f} nm",
                xy=(cut["cutoff_nm"], 0.985),
                xycoords=("data", "axes fraction"),
                xytext=(-3, 0), textcoords="offset points",
                fontsize=6.8, color=CUTOFF_STYLE[k], ha="right", va="top",
                rotation=90, zorder=7,
            )
        tagd = fiber_tag(rep["fibers"][0])
        feats = sw["cores"][tagd]["features"]
        placed = []
        for feat in sorted(feats, key=lambda f: -f["pmt_counts_per_s"])[:8]:
            c = feat["centre_nm"]
            if any(abs(c - q) < 0.022 * (hi - lo) for q in placed):
                continue
            placed.append(c)
            ax.annotate(
                f"{c:.1f} nm",
                xy=(feat["peak_lambda0_nm"],
                    max(feat["pmt_counts_per_s"] * 1.0e-3, floor)),
                xytext=(0, 4), textcoords="offset points",
                fontsize=6.4, rotation=90, ha="center", va="bottom",
                color="0.2", zorder=6,
            )
        ax.set_title(
            f"Port {fp['port']} ({st['label']})  --  slit "
            f"{knobs['slit_um']:g} um, {knobs['dispersion_nm_per_mm']:g} "
            f"nm/mm, bandpass {bp:.4f} nm;  f_slit "
            f"{sw['cores'][tagd]['f_slit']:.3f} x a_slit "
            f"{sw['cores'][tagd]['acceptance']:.4f} x eta_gm "
            f"{knobs['grating_mirror_efficiency']:g} x "
            r"$\eta_{\mathrm{count}}(\lambda)$",
            fontsize=9.0,
        )

        # companion: one feature resolved
        target = SWEEP_ZOOM_LINE_NM if key == "he1" else INSET_LINE_NM[key]
        feat = min(feats, key=lambda f: abs(f["centre_nm"] - target))
        half = SWEEP_ZOOM_BANDPASSES * bp
        sel = np.flatnonzero(np.abs(grid - feat["peak_lambda0_nm"]) <= half)
        axz.plot(grid[sel] - feat["peak_lambda0_nm"],
                 counts_ms[sel], color="tab:red", lw=1.5)
        axz.axvspan(-bp, bp, color="0.85", zorder=0,
                    label=f"bandpass $\\pm${bp:.3f} nm")
        axz.set_xlim(-half, half)
        axz.set_ylim(0.0, max(float(counts_ms[sel].max()) * 1.15, 1.0e-30))
        lines_txt = ", ".join(f"{v:.2f}" for v in feat["lines_nm"])
        axz.set_title(
            f"resolved: {feat['centre_nm']:.2f} nm"
            + (f"\n({len(feat['lines_nm'])} lines merged)"
               if feat["merged"] else ""),
            fontsize=8.5,
        )
        axz.set_xlabel(r"$\Delta\lambda_0$ from peak [nm]", fontsize=7.2,
                       labelpad=1.0)
        axz.set_ylabel("PMT rate [counts ms$^{-1}$]", fontsize=7.2)
        axz.tick_params(labelsize=6.6)
        axz.grid(True, alpha=0.2)
        axz.legend(fontsize=6.4, loc="upper right", framealpha=0.9)
        axz.annotate(
            f"lines: {lines_txt} nm\npeak "
            f"{feat['pmt_counts_per_s'] * 1.0e-3:.3e} counts/ms",
            xy=(0.03, 0.97), xycoords="axes fraction", fontsize=6.2,
            va="top", color="0.25",
        )

    axes[1].set_xlabel(r"$\lambda_0$, monochromator setting [nm]")
    fig.suptitle(
        f"Synthetic monochromator sweep -- {Path(rep['h5']).name}\n"
        f"{MONOCHROMATOR['model']}: slit {knobs['slit_um']:g} $\\mu$m, "
        f"dispersion {knobs['dispersion_nm_per_mm']:g} nm/mm (datasheet, "
        f"2400 g/mm), f/{knobs['f_number']:g}; core "
        f"{rep['fibers'][0]['core_um']:.0f} $\\mu$m NA "
        f"{rep['fibers'][0]['na']:g}, {rep['fiber_length_m']:g} m; "
        f"eta_gm {knobs['grating_mirror_efficiency']:g}, "
        f"{PMT['model']} counting efficiency",
        fontsize=10.2,
    )
    fig.get_layout_engine().set(rect=(0.0, 0.048, 1.0, 0.938))
    fig.text(
        0.5, 0.030,
        "Only lines the port window and the fiber transmit can be scanned "
        "onto; the EUV lines that carry most of the radiated power are "
        "absent by construction.",
        fontsize=7.0, color="0.35", ha="center",
    )
    fig.text(
        0.5, 0.017,
        "Slit function is a peak-normalized triangle of base 2 x bandpass "
        "(equal slits); the recorded feature is its convolution with the "
        "thermal line, not either one alone.",
        fontsize=7.0, color="0.35", ha="center",
    )
    fig.text(
        0.5, 0.004,
        "Slit, dispersion and f-number are datasheet MEASURED; the "
        "fiber-to-slit acceptance is DERIVED from them; the grating x mirror "
        "efficiency is ASSUMED and every rate scales linearly with it. The "
        "detector counting efficiency is DERIVED from datasheet 'Typ.' "
        "cells and is ZERO outside the head's rated response.",
        fontsize=7.0, color="0.35", ha="center",
    )
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=dpi)
    plt.close(fig)


# --- driver ---------------------------------------------------------------


def build(
    h5_path,
    ports,
    window_ms,
    fiber_core_um,
    fiber_na,
    material,
    length_m,
    flat_fiber_T=None,
    sweep_knobs=None,
):
    """Assemble every number and every product input from the artifact."""
    data = read_window(h5_path, window_ms)
    stages = {}
    for key in STAGE_ORDER:
        s = evaluate_stage(key, data)
        s["vectorization_check"] = vectorization_check(s, data)
        s["identity_check"] = completeness_identity_check(s)
        stages[key] = s

    fibers = [etendue_cm2_sr(fiber_core_um, fiber_na)]
    seen = {(fiber_core_um, fiber_na)}
    for d in FIBER_CORE_COLUMNS_UM:
        if (d, fiber_na) not in seen:
            seen.add((d, fiber_na))
            fibers.append(etendue_cm2_sr(d, fiber_na))
    for na in FIBER_NA_BRACKET:
        if (fiber_core_um, na) not in seen:
            seen.add((fiber_core_um, na))
            fibers.append(etendue_cm2_sr(fiber_core_um, na))

    law = PRS.port_axial_law()
    fiber_ports = {}
    for port in ports:
        fiber_ports[str(port)] = {
            "port": int(port),
            "stages": {
                key: fiber_at_port(
                    stages[key], data, port, law, fibers, material, length_m,
                    flat_fiber_T,
                )
                for key in STAGE_ORDER
            },
        }
        any_stage = fiber_ports[str(port)]["stages"][STAGE_ORDER[0]]
        fiber_ports[str(port)].update(
            {
                "z_want_cm": any_stage["z_want_cm"],
                "z_cell_cm": any_stage["z_cell_cm"],
                "Rp_cm": any_stage["Rp_cm"],
                "chord_length_cm": any_stage["chord_length_cm"],
            }
        )

    # Figure B panels follow the brief's assignment: the first port carries
    # the ion stage, the second the neutral stage.
    panels = [(str(ports[0]), "he1"), (str(ports[-1]), "he0")]

    sweeps = None
    if sweep_knobs is not None:
        sweeps = {
            pk: sweep_at_port(
                stages[key], fiber_ports[pk]["stages"][key], fibers,
                sweep_knobs,
            )
            for pk, key in panels
        }

    return {
        "h5": str(h5_path),
        "data": data,
        "stages": stages,
        "fibers": fibers,
        "fiber_ports": fiber_ports,
        "fiber_panels": panels,
        "port_law": {"z0_cm": law[0], "pitch_cm": law[1]},
        "window_material": material,
        "fiber_length_m": float(length_m),
        "flat_fiber_T": flat_fiber_T,
        "cutoffs": cutoff_table(length_m),
        "sweep_knobs": sweep_knobs,
        "sweeps": sweeps,
        "port_radiance_crosscheck": port_radiance_crosscheck(h5_path, ports[-1]),
    }


def print_console(rep):
    """Print the numbers this instrument exists to produce."""
    print(f"# line_radiation_sim1d -- {rep['h5']}")
    d = rep["data"]
    print(
        f"window {d['window_ms'][0]:g}-{d['window_ms'][1]:g} ms "
        f"(main-discharge clock, origin {d['origin_ms']:.4f} ms run clock), "
        f"{d['frames']} saves, {d['geometry']['z_cm'].size} active cells"
    )
    for key in STAGE_ORDER:
        s = rep["stages"][key]
        print("")
        print(
            f"[{s['short']}] machine line sum {s['machine_W_total']:.6e} W ; "
            f"adf11 {s['plt_key']} integral {s['adf11_machine_W']:.6e} W ; "
            f"C = {_fmt(s['completeness_machine'])}"
        )
        a15, a11 = s["adf15_clamp"], s["adf11_clamp"]
        print(
            f"  clamp census: adf15 {a15['points_clamped']}/{a15['points']} "
            f"points clamped ; adf11 Te "
            f"{a11['points_below_Te_edge'] + a11['points_above_Te_edge']}"
            f"/{a11['points']} ; window Te "
            f"{a15['min_Te_eV']:.4f}-{a15['max_Te_eV']:.4f} eV"
        )
        sq = s["sub_quotable"]
        print(
            f"  Te < {sq['Te_min_quotable_eV']:g} eV at "
            f"{sq['points_below']}/{sq['points']} points "
            f"({100.0 * sq['fraction_below']:.1f} %)"
        )
        order = np.argsort(-s["machine_W"])[:5]
        for i in order:
            print(
                f"    {s['lambda_nm'][i]:8.2f} nm  {s['band'][i]:<12s} "
                f"{s['machine_W'][i]:.4e} W  "
                f"({s['machine_W'][i] / s['machine_W_total']:.4f} of sum)"
            )

    print("")
    print("[self-consistency]")
    pr = rep["port_radiance_crosscheck"]
    print(
        f"  port_radiance_sim1d p{pr['port']} e-i drive mean "
        f"{pr['port_radiance_drive_mean_W_cm3']:.6e} W cm^-3 ; here "
        f"{pr['reconstructed_drive_mean_W_cm3']:.6e} W cm^-3 ; rel dev "
        f"{_fmt(pr['rel_dev'])}"
    )
    for key in STAGE_ORDER:
        s = rep["stages"][key]
        ci, vc, lc = s["identity_check"], s["vectorization_check"], s["ledger_check"]
        print(
            f"  {s['short']}: line sum vs C x adf11 cell power, max rel dev "
            f"{_fmt(ci['max_rel_dev'])} over {ci['cells']} cells ; "
            f"vs scalar pec_at {_fmt(vc['max_rel_dev_vs_pec_at'])} ; "
            + (
                f"adf11 vs `{lc['ledger_row']}` window mean "
                f"{_fmt(lc['window_mean_rel_dev'])}, worst point "
                f"{_fmt(lc['max_point_rel_dev'])}"
                if lc["available"]
                else lc["reason"]
            )
        )

    print("")
    win = WINDOW_TRANSMISSION[rep["window_material"]]
    print(
        "[synthetic fiber -- UPPER BOUND: spot ~ core width, bare fiber "
        "OUTSIDE the window, no collection optics]"
    )
    print(
        f"  window {win['material']} ({win['thickness']}); fiber run "
        f"{rep['fiber_length_m']:g} m; end faces "
        f"{FIBER_END_FACE_TRANSMISSION:.3f}^2 = "
        f"{FIBER_END_FACE_TRANSMISSION ** 2:.3f}"
    )
    for fib in rep["fibers"]:
        print(
            f"  fiber core {fib['core_um']:.0f} um NA {fib['na']:g}: "
            f"A {fib['core_area_cm2']:.4e} cm^2, Omega {fib['omega_sr']:.4f} "
            f"sr, G {fib['etendue_cm2_sr']:.4e} cm^2 sr"
        )
    default_tag = fiber_tag(rep["fibers"][0])
    for pkey, key in rep["fiber_panels"]:
        fp = rep["fiber_ports"][pkey]
        s = rep["stages"][key]
        got = fp["stages"][key]
        v = got["fibers"][default_tag]
        print("")
        print(
            f"  port {fp['port']} / {s['short']} -- z {fp['z_cell_cm']:.2f} "
            f"cm, chord {fp['chord_length_cm']:.2f} cm  "
            f"(collected -> transmitted)"
        )
        target = INSET_LINE_NM[key]
        lam_all = np.asarray(s["lambda_nm"])
        m = resolve_target_line(
            lam_all, target, f"{s['short']} zoom line (Figure A, console)"
        )
        print(
            f"    emitter T {got['emitter_T_eV']:.4f} eV "
            f"({got['emitter_T_source']}) ; Doppler FWHM at "
            f"{lam_all[m]:.2f} nm = {got['doppler_fwhm_nm'][m]:.5f} nm"
        )
        if key == "he1":
            t_cmp = INSET_COMPARISON_T_EV
            print(
                f"      zoom comparison: FWHM at {lam_all[m]:.2f} nm, "
                f"T = {t_cmp:g} eV = "
                f"{float(doppler_fwhm_nm(lam_all[m], t_cmp)):.5f} nm"
            )
            retired = INSET_RETIRED_LINES_NM.get(key)
            if retired is not None:
                k = resolve_target_line(
                    lam_all,
                    retired[0],
                    f"{s['short']} retired zoom line (Figure A, console)",
                )
                print(
                    f"      not drawn (record only): FWHM at "
                    f"{lam_all[k]:.2f} nm = "
                    f"{got['doppler_fwhm_nm'][k]:.5f} nm"
                )
        order = np.argsort(-np.asarray(v["photons_per_s_transmitted"]))[:5]
        for i in order:
            print(
                f"    {s['lambda_nm'][i]:8.2f} nm  "
                f"coll {v['power_W'][i]:.4e} W {v['photons_per_s'][i]:.4e} "
                f"ph/s  |  T_win {got['T_window'][i]:.4f} T_fib "
                f"{got['T_fiber'][i]:.4f} T_tot {got['T_total'][i]:.4e}  |  "
                f"trans {v['power_W_transmitted'][i]:.4e} W "
                f"{v['photons_per_s_transmitted'][i]:.4e} ph/s"
            )
        n_pass = int(np.count_nonzero(np.asarray(got["T_total"]) > 0.0))
        print(
            f"    stage total: collected {v['power_W_total']:.4e} W / "
            f"{v['photons_per_s_total']:.4e} ph/s -> transmitted "
            f"{v['power_W_transmitted_total']:.4e} W / "
            f"{v['photons_per_s_transmitted_total']:.4e} ph/s "
            f"({n_pass}/{s['lambda_nm'].size} lines pass) ; "
            f"P_trans/P_mach {_fmt(v['ratio_to_machine_total'])}"
        )
        for fib in rep["fibers"][1:]:
            b = fp["stages"][key]["fibers"][fiber_tag(fib)]
            print(
                f"    {fib['core_um']:.0f} um / NA {fib['na']:g}: "
                f"transmitted {b['power_W_transmitted_total']:.4e} W, "
                f"{b['photons_per_s_transmitted_total']:.4e} ph/s"
            )

    if rep["sweeps"] is not None:
        knobs = rep["sweep_knobs"]
        print("")
        print(
            f"[synthetic monochromator sweep -- {MONOCHROMATOR['model']}; "
            f"slit {knobs['slit_um']:g} um, f/{knobs['f_number']:g}, eta_gm "
            f"{knobs['grating_mirror_efficiency']:g}; "
            f"{PMT['model']} counting efficiency]"
        )
        for disp in knobs["dispersions"]:
            print(
                f"  dispersion {disp:g} nm/mm -> bandpass "
                f"{bandpass_nm(knobs['slit_um'], disp):.4f} nm "
                f"(slit-function base {2.0 * bandpass_nm(knobs['slit_um'], disp):.4f} nm)"
            )
        for pkey, key in rep["fiber_panels"]:
            st = rep["stages"][key]
            fp = rep["fiber_ports"][pkey]
            sw = rep["sweeps"][pkey]
            print("")
            print(f"  port {fp['port']} / {st['short']} -- peak counts/ms")
            for disp in knobs["dispersions"]:
                block = sw["dispersions"][f"{disp:g}"]
                for core in SWEEP_CORE_COLUMNS_UM:
                    tag = f"{core:.0f}um_NA{rep['fibers'][0]['na']:g}"
                    rec = block["cores"].get(tag)
                    if rec is None or rec.get("empty"):
                        continue
                    for feat in sorted(
                        rec["features"], key=lambda f: -f["pmt_counts_per_s"]
                    ):
                        lines_txt = ",".join(
                            f"{v:.2f}" for v in feat["lines_nm"]
                        )
                        print(
                            f"    disp {disp:>4g} bp "
                            f"{block['bandpass_nm']:.4f} nm  core "
                            f"{core:>4.0f} um  f_slit {rec['f_slit']:.4f}  "
                            f"a_slit {rec['acceptance']:.4f}  "
                            f"[{lines_txt}]"
                            f"{' MERGED' if feat['merged'] else ''}  "
                            f"eta_count "
                            f"{feat['counting_efficiency']:.4f}  "
                            f"exit {feat['exit_counts_per_s'] * 1e-3:.4e}  "
                            f"PMT {feat['pmt_counts_per_s'] * 1e-3:.4e}"
                        )

    print("")
    print("[window cutoffs -- ASSUMED representative parts, not LAPD hardware]")
    for cut in rep["cutoffs"]:
        print(
            f"  {cut['material']}: 50 % at {cut['cutoff_nm']:.0f} +/- "
            f"{cut['uncertainty_nm']:.0f} nm ({cut['basis']})"
        )
        print(f"    source: {cut['source']}")
        print(f"    caveat: {cut['caveat']}")
    for key in STAGE_ORDER:
        s = rep["stages"][key]
        for cut in rep["cutoffs"]:
            mask = transmits(cut, s["lambda_nm"])
            print(
                f"  {s['short']} through {cut['short']}: "
                f"{int(mask.sum())}/{mask.size} lines, "
                f"{s['machine_W'][mask].sum():.4e} W "
                f"({s['machine_W'][mask].sum() / s['machine_W_total']:.4e} "
                "of the stage line sum)"
            )


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument(
        "--ports",
        type=int,
        nargs=2,
        default=list(DEFAULT_PORTS),
        metavar=("ION_PORT", "NEUTRAL_PORT"),
        help="two ports; the first carries the He II panel of the fiber "
        "figure, the second the He I panel",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        nargs=2,
        default=list(DEFAULT_WINDOW_MS),
        metavar=("LO", "HI"),
        help="plateau window on the MAIN-DISCHARGE clock [ms]",
    )
    parser.add_argument("--output-stem", type=Path, default=None)
    parser.add_argument(
        "--fiber-core-um",
        type=float,
        default=FIBER_DATASHEET["core_um"],
        help="fiber core diameter [um]; the default is the datasheet core of "
        f"the {FIBER_DATASHEET['model']}",
    )
    parser.add_argument(
        "--fiber-na",
        type=float,
        default=FIBER_DATASHEET["na"],
        help="fiber numerical aperture; the default is the datasheet NA of "
        f"the {FIBER_DATASHEET['model']}",
    )
    parser.add_argument(
        "--window",
        choices=sorted(WINDOW_TRANSMISSION),
        default="fused_silica",
        help="port window material whose datasheet transmission curve is "
        "applied to every line",
    )
    parser.add_argument(
        "--fiber-length-m",
        type=float,
        default=DEFAULT_FIBER_LENGTH_M,
        help="fiber run length [m]; sets the bulk attenuation. The default "
        "is the operator's reading of the installed run",
    )
    parser.add_argument(
        "--fiber-transmission",
        type=float,
        default=None,
        help="flat fiber transmission replacing the datasheet chain "
        f"(the operator-quoted figure is "
        f"{QUOTED_FLAT_FIBER_TRANSMISSION:g}); the datasheet result is "
        "printed either way",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also run the synthetic monochromator sweep and write Figure C",
    )
    parser.add_argument(
        "--slit-um",
        type=float,
        default=DEFAULT_SLIT_UM,
        help="entrance/exit slit width [um] (equal slits)",
    )
    parser.add_argument(
        "--dispersion-nm-per-mm",
        type=float,
        default=DEFAULT_DISPERSION_NM_PER_MM,
        help="reciprocal linear dispersion [nm/mm]; the default is the "
        f"{MONOCHROMATOR['model']} datasheet cell at 2400 g/mm",
    )
    parser.add_argument(
        "--dispersion-bracket",
        action="store_true",
        help="also carry the placeholder "
        f"{'/'.join(f'{d:g}' for d in DISPERSION_BRACKET_NM_PER_MM)} nm/mm "
        "columns the sweep tables held before the instrument was identified",
    )
    parser.add_argument(
        "--f-number",
        type=float,
        default=DEFAULT_F_NUMBER,
        help="spectrometer f-number, which sets the fiber-to-slit "
        f"acceptance; the datasheet contradicts itself and the bracket is "
        f"[f/{F_NUMBER_BRACKET[0]:g}, f/{F_NUMBER_BRACKET[1]:g}]",
    )
    parser.add_argument(
        "--grating-mirror-efficiency",
        type=float,
        default=DEFAULT_GRATING_MIRROR_EFFICIENCY,
        help="grating x mirror efficiency, the ASSUMED residue of the "
        "spectrometer throughput once the acceptance is computed",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if not args.h5.exists():
        raise ArtifactRefused(f"artifact not found: {args.h5}")
    if args.window_ms[0] >= args.window_ms[1]:
        raise ArtifactRefused(
            f"--window-ms must be increasing, got {args.window_ms}"
        )
    if args.fiber_core_um <= 0.0 or not (0.0 < args.fiber_na < 1.0):
        raise ArtifactRefused(
            f"--fiber-core-um must be positive and --fiber-na in (0, 1); got "
            f"{args.fiber_core_um} um, NA {args.fiber_na}"
        )

    if args.fiber_length_m <= 0.0:
        raise ArtifactRefused(
            f"--fiber-length-m must be positive; got {args.fiber_length_m}"
        )
    if args.fiber_transmission is not None and not (
        0.0 < args.fiber_transmission <= 1.0
    ):
        raise ArtifactRefused(
            "--fiber-transmission must lie in (0, 1]; got "
            f"{args.fiber_transmission}"
        )
    for name, value in (
        ("--slit-um", args.slit_um),
        ("--dispersion-nm-per-mm", args.dispersion_nm_per_mm),
        ("--f-number", args.f_number),
    ):
        if value <= 0.0:
            raise ArtifactRefused(f"{name} must be positive; got {value}")
    if not (0.0 < args.grating_mirror_efficiency <= 1.0):
        raise ArtifactRefused(
            "--grating-mirror-efficiency must lie in (0, 1]; got "
            f"{args.grating_mirror_efficiency}"
        )

    sweep_knobs = None
    if args.sweep:
        # The measured dispersion is always present, so the figure can key
        # off it; --dispersion-bracket only ADDS the placeholder columns.
        dispersions = [float(args.dispersion_nm_per_mm)]
        if args.dispersion_bracket:
            for d in DISPERSION_BRACKET_NM_PER_MM:
                if f"{d:g}" not in {f"{x:g}" for x in dispersions}:
                    dispersions.append(float(d))
        sweep_knobs = {
            "slit_um": args.slit_um,
            "dispersion_nm_per_mm": args.dispersion_nm_per_mm,
            "dispersions": tuple(sorted(dispersions)),
            "f_number": args.f_number,
            "grating_mirror_efficiency": args.grating_mirror_efficiency,
        }

    rep = build(
        args.h5,
        [int(p) for p in args.ports],
        tuple(float(w) for w in args.window_ms),
        args.fiber_core_um,
        args.fiber_na,
        args.window,
        args.fiber_length_m,
        args.fiber_transmission,
        sweep_knobs,
    )

    stem = args.output_stem
    if stem is None:
        stem = args.h5.resolve().parent / f"line_radiation_{args.h5.stem}"
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    tag = f"p{args.ports[0]}_p{args.ports[1]}"
    figure_chord_power(rep, f"{stem}_chord_power_{tag}", dpi=args.dpi)
    figure_photon_counter(rep, f"{stem}_photon_counter_{tag}", dpi=args.dpi)
    if rep["sweeps"] is not None:
        figure_sweep(rep, f"{stem}_synthetic_sweep_{tag}", dpi=args.dpi)
    md = Path(f"{stem}_line_radiation.md")
    md.write_text(markdown_report(rep))

    print_console(rep)
    print("")
    print(f"wrote {stem}_chord_power_{tag}.pdf")
    print(f"wrote {stem}_chord_power_{tag}.png")
    print(f"wrote {stem}_photon_counter_{tag}.pdf")
    print(f"wrote {stem}_photon_counter_{tag}.png")
    if rep["sweeps"] is not None:
        print(f"wrote {stem}_synthetic_sweep_{tag}.pdf")
        print(f"wrote {stem}_synthetic_sweep_{tag}.png")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
