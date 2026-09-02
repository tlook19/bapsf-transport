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

SYNTHETIC FIBER -- AN UPPER BOUND.  The collection model is a BARE fiber
looking at the plasma with NO collection optics, and the imaged spot at the
plasma is taken to be approximately the fiber core width.  Collected power is
then P_i = L_i * G with G = A * Omega the fiber etendue, A = pi*(d/2)^2 the
core area and Omega = pi * NA^2 the acceptance solid angle; photon rate is
N_i = P_i * lambda_i / (hc).  This is an UPPER BOUND on collection: a real
train loses light at every surface, and adding a collimating lens cannot
raise it, because a lens CONSERVES etendue -- it trades angular acceptance
for collection area, changing the field of view rather than G.  The fiber
geometry is ASSUMED hardware, not measured -- see ``ASSUMPTIONS``, which is
emitted verbatim into the markdown product.

WINDOW CUTOFFS.  The three 50 % transmission cutoffs drawn on the figures are
ASSUMED representative values for generic commercial parts, each carrying its
datasheet source in ``ASSUMPTIONS``.  The material actually installed on the
LAPD viewports and the fiber actually on the bench are NOT known to this
script and are not claimed.

    line_radiation_sim1d.py [--h5 RUN.h5] [--ports 22 27]
                            [--window-ms 15 19.5] [--output-stem STEM]
                            [--fiber-core-um 400] [--fiber-na 0.22]
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
    },
    "he0": {
        "spec_key": "he0",
        "label": "He I (e-n, neutral stage)",
        "short": "He I",
        "partner": "nn",
        "partner_label": "nn (in-column)",
        "plt_key": "plt1",
        "ledger_row": "electron_neutral_cooling",
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
        "value": "400 um default; 200 / 400 / 600 um reported as columns",
        "class": "ASSUMED",
        "source": (
            "Bare multimode spectroscopy fiber in the 200-600 um class. NOT "
            "the fiber on the LAPD bench, which is unknown to this script. "
            "Substitutable with --fiber-core-um."
        ),
    },
    {
        "quantity": "fiber numerical aperture",
        "value": "NA = 0.22 default; bracketed by NA 0.12 and NA 0.39",
        "class": "ASSUMED",
        "source": (
            "Standard step-index silica/silica NA. NOT a measured value for "
            "the LAPD hardware. Substitutable with --fiber-na."
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
        "quantity": "window / fiber 50 % transmission cutoffs",
        "value": (
            "UV fused silica 170 +/- 5 nm; borosilicate 303 +/- 8 nm; "
            "UV fiber 200 +/- 10 nm"
        ),
        "class": "ASSUMED (datasheet-sourced, representative parts)",
        "source": (
            "Each carries its own datasheet, thickness/length, "
            "external-vs-internal basis and caveat in `WINDOW_CUTOFFS`, "
            "reproduced in full in the Window cutoffs section below. No "
            "manufacturer tabulates a 50 % point for any of the three, so "
            "every one is a curve reading or an interpolation. The material "
            "on the LAPD viewports and the identity of the collection fiber "
            "are NOT known to this script."
        ),
    },
)

#: Core diameters reported as columns alongside the default [um].
FIBER_CORE_COLUMNS_UM = (200.0, 400.0, 600.0)

#: NA values reported as a bracket around the default.
FIBER_NA_BRACKET = (0.12, 0.39)

#: 50 % transmission cutoffs drawn on both figures.  Every one of these is an
#: ASSUMED representative commercial part -- NOT the material on the LAPD
#: viewports, and NOT the fiber on the bench, neither of which is known here.
#: No manufacturer tabulates a 50 % point for any of the three, so each number
#: below is a curve reading or an interpolation and carries its own bar.
WINDOW_CUTOFFS = (
    {
        "material": "UV-grade fused silica window (synthetic, high-OH)",
        "short": "UV fused silica",
        "cutoff_nm": 170.0,
        "uncertainty_nm": 5.0,
        "basis": "10 mm, EXTERNAL transmission (Fresnel included)",
        "source": (
            "Heraeus, 'Quartz Glass for Optics -- Data and Properties' "
            "(HQS-SO, May 2011), Suprasil 1/2 Grade A transmission panel, "
            "'Sample thickness: 10 mm', figure header 'Measured transmission "
            "including Fresnel reflection losses (1-R)^2'. Read off the "
            "plotted curve at 50 %; the +/-5 nm is the plotted-line width on "
            "a steep edge (0 to ~88 % over ~163-180 nm). "
            "https://sites.astro.caltech.edu/sedm/_downloads/"
            "2562e19ff76ec4ab03f0598d537f8428/Heraeus_SiO2-May-2011.pdf"
        ),
        "caveat": (
            "Cross-check, Corning HPFS 7980 Standard Grade: 'certified to "
            "meet T external >= 80%/cm@185nm' -- a floor, not a 50 % point, "
            "and consistent with 170 nm at 10 mm. Thinner windows cut bluer: "
            "scaling tau_i(d) = tau_i(10 mm)^(d/10) puts a 3 mm window's "
            "50 % point near 165 nm (DERIVED, not from any datasheet)."
        ),
    },
    {
        "material": "borosilicate glass window (SCHOTT BOROFLOAT 33)",
        "short": "borosilicate",
        "cutoff_nm": 303.0,
        "uncertainty_nm": 8.0,
        "basis": "3.30 mm, external transmission (inferred)",
        "source": (
            "SCHOTT, 'BOROFLOAT 33 -- Optical Properties', panel "
            "'Transmission in UV range', 3.30 mm curve, read at 50 %. The "
            "axis is labelled only 'Transmission [%]', but its ~91-92 % "
            "plateau matches (1-R)^2 = 92.9 % for n_d = 1.471, so Fresnel "
            "appears to be included. The sheet's own tables are marked "
            "'Reference values, not guaranteed values.' "
            "https://www.schott.com/en-gb/products/borofloat/-/media/project/"
            "onex/products/b/borofloat/downloads/borofloat33_opt_eng_web.pdf"
        ),
        "caveat": (
            "This number has an unresolved internal inconsistency in its own "
            "source: the same sheet's 6.5 mm comparison panel reads "
            "~288-290 nm, which cannot be bluer than its 5.00 mm curve "
            "(~312 nm). Honest bracket across 3.3-6.5 mm: 288-312 nm. "
            "N-BK7 is a DIFFERENT glass and lands nearby -- its tabulated "
            "internal transmittance (tau_i = 0.574 at 310 nm, 0.292 at "
            "300 nm, 10 mm) interpolates to a 50 % internal point at "
            "307-308 nm."
        ),
    },
    {
        "material": "solarization-resistant high-OH UV fiber",
        "short": "UV fiber (1 m)",
        "cutoff_nm": 200.0,
        "uncertainty_nm": 10.0,
        "basis": "1 m of fiber, bulk attenuation only (no end-face Fresnel)",
        "source": (
            "Fiberguide Solarguide (silica core / F-doped clad, "
            "hydrogen-infused), spec 'Wavelength UV-VIS (High OH) 190nm ~ "
            "1250nm'; attenuation curve read at ~3000 dB/km = 3 dB/m at "
            "200 nm, which is exactly the 50 % point for 1 m. The "
            "datasheet's transmission panel states NO length; 1 m is "
            "INFERRED from that consistency. "
            "https://shop.amstechnologies.com/media/27/3f/d4/1720716082/"
            "SolarguideTM-Solarization-Resistant-MM-Fibers-Fiberguide-"
            "Datasheetkn5dgcLgOXSSu.pdf"
        ),
        "caveat": (
            "LENGTH-DEPENDENT and manufacturer-dependent. At 200 nm the "
            "attenuation read off three makers' curves spans 0.7 dB/m "
            "(CeramOptec Optran UV/UVNS) to 1.2 dB/m (Ocean Optics UV/SR-VIS) "
            "to 3 dB/m (Fiberguide), a factor ~4; at 1 m all three put the "
            "50 % point at or below 200 nm, at 2 m it moves to ~200-220 nm. "
            "Separately, an un-stabilised high-OH fiber SOLARIZES: Molex "
            "Polymicro's 214 nm exposure data has standard FVP falling to "
            "~15 % of its initial transmission after 8 h of deuterium-lamp "
            "dose, so an installed fiber's effective cutoff drifts red with "
            "use."
        ),
    },
)


class ArtifactRefused(SystemExit):
    """Raised with an explanatory message when the artifact cannot be read."""


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


def fiber_at_port(stage, data, port, law, fibers):
    """Per-line radiance, collected power and photon rate at one port."""
    z_want = PRS.port_to_z_cm(port, law)
    z_cells = data["geometry"]["z_cm"]
    j = int(np.argmin(np.abs(z_cells - z_want)))
    Rp = float(data["geometry"]["Rp_cm"][j])
    chord = 2.0 * Rp
    eps = stage["eps_W_mean"][:, j]  # W cm^-3
    radiance = eps * chord / (4.0 * np.pi)  # W cm^-2 sr^-1

    per_fiber = {}
    for fib in fibers:
        power = radiance * fib["etendue_cm2_sr"]  # W
        photons = power / stage["photon_J"]  # s^-1
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                stage["machine_W"] > 0.0, power / stage["machine_W"], np.nan
            )
        per_fiber[fiber_tag(fib)] = {
            "fiber": fib,
            "power_W": power,
            "photons_per_s": photons,
            "ratio_to_machine": ratio,
            "power_W_total": float(power.sum()),
            "photons_per_s_total": float(photons.sum()),
            "ratio_to_machine_total": (
                float(power.sum() / stage["machine_W_total"])
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
        "radiance_W_cm2_sr": radiance,
        "fibers": per_fiber,
    }


def fiber_tag(fib):
    """Short identifier for a fiber configuration."""
    return f"{fib['core_um']:.0f}um_NA{fib['na']:g}"


# --- window cutoffs -------------------------------------------------------


def cutoff_table():
    """Return the three sourced 50 % transmission cutoffs, ordered blue-ward."""
    return tuple(
        sorted(WINDOW_CUTOFFS, key=lambda c: c["cutoff_nm"])
    )


def transmits(cutoff, lam_nm):
    """Boolean mask: which lines sit red-ward of a cutoff."""
    return np.asarray(lam_nm) >= cutoff["cutoff_nm"]


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
        for cut in cutoff_table():
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
            for cut in cutoff_table()
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
    for cut in cutoff_table():
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
        "material of the LAPD viewports and the identity of the collection "
        "fiber are NOT known to this script; nothing here is a statement "
        "about the installed hardware."
    )
    L.append("")
    return "\n".join(L)


# --- figures --------------------------------------------------------------

CUTOFF_STYLE = ("tab:purple", "tab:olive", "tab:brown")


def _line_colors(n):
    return plt.get_cmap("turbo")(np.linspace(0.08, 0.95, max(n, 1)))


def figure_machine(rep, path_stem, dpi=180):
    """Figure A: per-line axial power profile plus its wavelength spectrum."""
    fig = plt.figure(figsize=(13.0, 8.8), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=(2.7, 1.35))
    cuts = cutoff_table()
    z = rep["data"]["geometry"]["z_cm"]
    length = rep["data"]["geometry"]["length_cm"]

    for row, key in enumerate(STAGE_ORDER):
        s = rep["stages"][key]
        ax = fig.add_subplot(gs[row, 0])
        axs = fig.add_subplot(gs[row, 1])
        order = np.argsort(-s["machine_W"])
        colors = {}
        rank = np.argsort(np.argsort(-s["machine_W"]))
        cmap = _line_colors(len(s["lines"]))
        for i in range(len(s["lines"])):
            colors[i] = cmap[rank[i]]

        # dP/dz is the grid-independent axial density of radiated power: the
        # mesh is non-uniform, so a per-cell W profile would show mesh
        # structure rather than physics.  Integrating it over z reproduces
        # the machine totals tabulated in the markdown product.
        dpdz = s["cell_W"] / length
        peak = float(np.max(dpdz)) if dpdz.size else 1.0
        floor = peak * 10.0**-PRS.PLOT_DECADES
        for i in range(len(s["lines"])):
            ax.plot(
                z,
                np.maximum(dpdz[i], floor),
                color=colors[i],
                lw=1.2,
                alpha=0.9 if i in order[:LABELLED_LINES] else 0.35,
            )
        handles = [
            plt.Line2D(
                [],
                [],
                color=colors[i],
                lw=2.0,
                label=(
                    f"{s['lambda_nm'][i]:.1f} nm  "
                    f"({s['machine_W'][i]:.2e} W)"
                ),
            )
            for i in order[:LABELLED_LINES]
        ]
        ax.legend(
            handles=handles,
            fontsize=6.6,
            ncol=2,
            loc="lower left",
            framealpha=0.92,
            title=f"strongest {len(handles)} lines",
            title_fontsize=6.6,
        )
        ax.set_yscale("log")
        ax.set_ylim(floor, peak * 8.0)
        ax.grid(True, alpha=0.22)
        ax.set_ylabel("dP/dz  [W cm$^{-1}$]")
        ax.set_title(
            f"{s['label']}  --  {len(s['lines'])} adf15 EXCIT lines, "
            f"machine total {s['machine_W_total']:.3e} W "
            f"(C = {_fmt(s['completeness_machine'])} of adf11 "
            f"{s['plt_key']})",
            fontsize=9.5,
        )
        if row == 1:
            ax.set_xlabel("z [cm]")

        # Right panel: per-line machine total against wavelength, with the
        # sourced 50 % window cutoffs as shaded bands.
        lam = s["lambda_nm"]
        mw = np.maximum(s["machine_W"], s["machine_W"].max() * 1.0e-12)
        edges = [1.0] + [c["cutoff_nm"] for c in cuts] + [3000.0]
        shades = ("0.55", "0.72", "0.85", "0.96")
        for k in range(len(edges) - 1):
            axs.axvspan(edges[k], edges[k + 1], color=shades[k], lw=0, zorder=0)
        for k, cut in enumerate(cuts):
            axs.axvline(
                cut["cutoff_nm"],
                color=CUTOFF_STYLE[k],
                ls="--",
                lw=1.4,
                zorder=3,
                label=f"{cut['short']} {cut['cutoff_nm']:.0f} nm",
            )
        for i in range(len(lam)):
            axs.plot(
                [lam[i], lam[i]],
                [mw.max() * 1.0e-12, mw[i]],
                color=colors[i],
                lw=1.6,
                zorder=4,
            )
            axs.plot(lam[i], mw[i], "o", ms=3.4, color=colors[i], zorder=5)
        axs.set_xscale("log")
        axs.set_yscale("log")
        axs.set_xlim(15.0, 2000.0)
        axs.set_ylim(mw.max() * 1.0e-10, mw.max() * 8.0)
        axs.set_ylabel("machine total [W]")
        axs.grid(True, alpha=0.22, zorder=1)
        axs.legend(fontsize=6.2, loc="lower left", framealpha=0.92)
        if row == 1:
            axs.set_xlabel(r"$\lambda_{vac}$ [nm]")
        axs.set_title(
            "per-line machine total vs wavelength;\n"
            "shaded = window transmission bands (50 % cutoffs)",
            fontsize=8,
        )

    fig.suptitle(
        f"Line-resolved radiated power -- {Path(rep['h5']).name}   "
        f"(plateau {rep['data']['window_ms'][0]:g}-"
        f"{rep['data']['window_ms'][1]:g} ms, main-discharge clock; "
        f"{rep['data']['frames']} saves)",
        fontsize=11,
    )
    fig.get_layout_engine().set(rect=(0.0, 0.045, 1.0, 0.952))
    fig.text(
        0.5,
        0.020,
        "dP/dz plotted rather than W-per-cell because the axial mesh is "
        "non-uniform; its z-integral is the tabulated machine total. Traces "
        f"clipped {PRS.PLOT_DECADES:.0f} decades below the panel peak "
        "(drawing only).",
        fontsize=7.0,
        color="0.35",
        ha="center",
    )
    fig.text(
        0.5,
        0.006,
        "Window cutoffs are ASSUMED representative commercial parts read off "
        "manufacturer curves, NOT LAPD hardware; see the markdown product "
        "for each source, thickness/length and caveat.",
        fontsize=7.0,
        color="0.35",
        ha="center",
    )
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=dpi)
    plt.close(fig)


def figure_fiber(rep, path_stem, dpi=180):
    """Figure B: per-line photon rate at the fiber, against the cutoffs."""
    cuts = cutoff_table()
    default_tag = fiber_tag(rep["fibers"][0])
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 9.0), layout="constrained")

    for ax, (pkey, key) in zip(axes, rep["fiber_panels"]):
        fp = rep["fiber_ports"][pkey]
        s = rep["stages"][key]
        v = fp["stages"][key]["fibers"][default_tag]
        lam = s["lambda_nm"]
        rate = np.asarray(v["photons_per_s"])
        top = max(float(rate.max()), 1.0e-30)
        floor = top * 1.0e-12

        best = cuts[0]
        passes = transmits(best, lam)
        for i in range(lam.size):
            ax.bar(
                lam[i],
                max(rate[i], floor),
                width=lam[i] * 0.055,
                color="tab:blue" if passes[i] else "0.78",
                edgecolor="black" if passes[i] else "0.55",
                linewidth=0.7,
                hatch=None if passes[i] else "///",
                zorder=3,
            )
        # Labelling every bar collides wherever the line list crowds (the
        # He I 471/492/505 nm group); the strongest few carry the reading and
        # the markdown table carries all of them.
        placed = [np.log10(c["cutoff_nm"]) for c in cuts]
        for i in np.argsort(-rate)[:LABELLED_LINES]:
            x = float(np.log10(lam[i]))
            if any(abs(x - q) < 0.022 for q in placed):
                continue  # would collide with a neighbour or a cutoff label
            placed.append(x)
            ax.annotate(
                f"{lam[i]:.1f} nm",
                xy=(lam[i], max(rate[i], floor)),
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=6.4,
                rotation=90,
                ha="center",
                va="bottom",
                color="0.2",
                zorder=6,
            )
        for k, cut in enumerate(cuts):
            ax.axvline(
                cut["cutoff_nm"], color=CUTOFF_STYLE[k], ls="--", lw=1.5, zorder=5
            )
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
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(15.0, 2000.0)
        ax.set_ylim(floor, top * 60.0)
        ax.grid(True, alpha=0.22, zorder=0)
        ax.set_ylabel("photons s$^{-1}$ at the fiber")
        ax.set_title(
            f"Port {fp['port']} ({s['label']})  --  z {fp['z_cell_cm']:.1f} cm, "
            f"chord {fp['chord_length_cm']:.1f} cm; "
            f"stage total {v['photons_per_s_total']:.3e} ph/s, "
            f"{v['power_W_total']:.3e} W, "
            f"P_fiber/P_machine = {_fmt(v['ratio_to_machine_total'])}",
            fontsize=9.5,
        )
    axes[1].set_xlabel(r"$\lambda_{vac}$ [nm]")
    solid = (
        f"Solid blue = red-ward of the bluest cutoff shown "
        f"({cuts[0]['short']}, {cuts[0]['cutoff_nm']:.0f} nm), i.e. "
        "potentially collectable; hatched grey = blocked by every material "
        "shown."
    )
    fib = rep["fibers"][0]
    fig.suptitle(
        f"Synthetic fiber signal -- {Path(rep['h5']).name}\n"
        f"UPPER BOUND: spot $\\approx$ core width, no collection optics "
        f"assumed  |  core {fib['core_um']:.0f} $\\mu$m, NA {fib['na']:g}, "
        f"G = {fib['etendue_cm2_sr']:.3e} cm$^2$ sr (ASSUMED hardware)\n"
        f"plateau {rep['data']['window_ms'][0]:g}-"
        f"{rep['data']['window_ms'][1]:g} ms, main-discharge clock",
        fontsize=10.0,
    )
    fig.get_layout_engine().set(rect=(0.0, 0.048, 1.0, 0.945))
    fig.text(0.5, 0.022, solid, fontsize=7.0, color="0.35", ha="center")
    fig.text(
        0.5,
        0.007,
        "Radiance assumes emissivity RADIALLY UNIFORM across the plasma disc "
        "-- the 1D model has no radial profile. Cutoffs are ASSUMED "
        "representative commercial parts, NOT LAPD hardware.",
        fontsize=7.0,
        color="0.35",
        ha="center",
    )
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=dpi)
    plt.close(fig)


# --- driver ---------------------------------------------------------------


def build(h5_path, ports, window_ms, fiber_core_um, fiber_na):
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
                key: fiber_at_port(stages[key], data, port, law, fibers)
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

    return {
        "h5": str(h5_path),
        "data": data,
        "stages": stages,
        "fibers": fibers,
        "fiber_ports": fiber_ports,
        "fiber_panels": panels,
        "port_law": {"z0_cm": law[0], "pitch_cm": law[1]},
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
    print(
        "[synthetic fiber -- UPPER BOUND: spot ~ core width, no collection "
        "optics assumed]"
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
        v = fp["stages"][key]["fibers"][default_tag]
        print("")
        print(
            f"  port {fp['port']} / {s['short']} -- z {fp['z_cell_cm']:.2f} "
            f"cm, chord {fp['chord_length_cm']:.2f} cm"
        )
        order = np.argsort(-np.asarray(v["power_W"]))[:5]
        for i in order:
            print(
                f"    {s['lambda_nm'][i]:8.2f} nm  {v['power_W'][i]:.4e} W  "
                f"{v['photons_per_s'][i]:.4e} ph/s  "
                f"P_fib/P_mach {v['ratio_to_machine'][i]:.4e}"
            )
        print(
            f"    stage total {v['power_W_total']:.4e} W, "
            f"{v['photons_per_s_total']:.4e} ph/s, "
            f"P_fib/P_mach {_fmt(v['ratio_to_machine_total'])}"
        )
        for fib in rep["fibers"][1:]:
            b = fp["stages"][key]["fibers"][fiber_tag(fib)]
            print(
                f"    {fib['core_um']:.0f} um / NA {fib['na']:g}: "
                f"{b['power_W_total']:.4e} W, "
                f"{b['photons_per_s_total']:.4e} ph/s"
            )

    print("")
    print("[window cutoffs -- ASSUMED representative parts, not LAPD hardware]")
    for cut in cutoff_table():
        print(
            f"  {cut['material']}: 50 % at {cut['cutoff_nm']:.0f} +/- "
            f"{cut['uncertainty_nm']:.0f} nm ({cut['basis']})"
        )
        print(f"    source: {cut['source']}")
        print(f"    caveat: {cut['caveat']}")
    for key in STAGE_ORDER:
        s = rep["stages"][key]
        for cut in cutoff_table():
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
    parser.add_argument("--fiber-core-um", type=float, default=400.0)
    parser.add_argument("--fiber-na", type=float, default=0.22)
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

    rep = build(
        args.h5,
        [int(p) for p in args.ports],
        tuple(float(w) for w in args.window_ms),
        args.fiber_core_um,
        args.fiber_na,
    )

    stem = args.output_stem
    if stem is None:
        stem = args.h5.resolve().parent / f"line_radiation_{args.h5.stem}"
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    figure_machine(rep, f"{stem}_line_power_machine", dpi=args.dpi)
    figure_fiber(rep, f"{stem}_synthetic_fiber_p{args.ports[0]}_p{args.ports[1]}", dpi=args.dpi)
    md = Path(f"{stem}_line_radiation.md")
    md.write_text(markdown_report(rep))

    print_console(rep)
    print("")
    print(f"wrote {stem}_line_power_machine.pdf")
    print(f"wrote {stem}_line_power_machine.png")
    print(f"wrote {stem}_synthetic_fiber_p{args.ports[0]}_p{args.ports[1]}.pdf")
    print(f"wrote {stem}_synthetic_fiber_p{args.ports[0]}_p{args.ports[1]}.png")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
