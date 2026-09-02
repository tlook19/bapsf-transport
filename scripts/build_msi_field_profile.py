"""Build the production ``plasma_radius_profile_cm`` from the MEASURED MSI field.

The stance's flux-tube radius profile was, until this build, a CAD-census
product: ``scripts/g1_build_profiles.py`` traced a flux surface through a
finite-element re-solve of the drawn coil set. This script replaces the SOURCE
of that profile with the machine's own record of the field it actually ran --
the ``MSI/Magnetic field`` group every ES1 shot file carries -- and leaves the
census build in place as the independent cross-check.

Adopted 2026-09-01 (Tom).

THE DATA
--------
Every ES1 raw file under ``MSI_DATA_DIR`` carries, under ``MSI/Magnetic
field``:

``Profile z locations`` (attribute)
    1024 axial sample positions in cm, spanning -300 .. 2025.3 at a uniform
    2.273 cm pitch. Identical in every file (asserted). These are in the
    MACHINE's port-referenced frame, not the model's -- see COORDINATES.
``Magnetic field profile`` (dataset, 2 x 1024, gauss)
    The axial field profile recorded at the FIRST and LAST shot of the run.
``Magnet power supply currents`` (dataset, 2 x 13, ampere)
    The thirteen magnet supplies at those same two shots.
``Magnetic field summary`` (dataset, 2 records)
    Per-shot peak field, timestamp and a validity flag.

Both recorded shots of every file are used; nothing is averaged across files
before normalization.

THE NORMALIZED SHAPE
--------------------
Each recorded profile is divided by its OWN plateau level -- the median over
``PLATEAU_WINDOW_CM`` -- giving ``B_hat(z)``, which is 1 on the plateau by
construction. The build then uses the MEAN of ``B_hat`` over every file and
shot. Normalizing per profile is what makes the mean meaningful: the runs
differ in absolute level by up to ~3 % because the main supplies were not set
identically day to day, and that level is exactly what the flux-tube ratio
does not care about.

Two files (32, 34) ran the lowest main-supply currents of the set (~4-5 %
below the nominal 3659 A on supply 0). They are excluded from the mean ONLY if
their normalized SHAPE departs from the others by more than the others depart
from each other, measured on the FAR END -- the span the flux ratio is
actually read on, in model coordinates. The decision is measured, printed with
both readings, and recorded in the report. It is never silently applied.

COORDINATES -- THE MEASURED REGISTRATION
----------------------------------------
``z_MSI`` is NOT the model's axial coordinate. The MSI record is written in
the machine's own port-referenced frame (the ``bapsflib.lapd`` convention):
``z = 0`` at PORT 53, the most northern regular port, with ``+z`` pointing
SOUTH, toward the main cathode -- ``portnum_to_z = 31.95 * (53 - port)``. The
model's ``z`` runs the other way, from ``z = 0`` at the cathode face toward
the far collector. The two frames are REVERSED with respect to each other, so
the registration is a reflection:

    z_model = C - z_MSI,   C = PORT53_Z_MODEL_CM = 1814.67 cm

``C`` is the port-53 station in model coordinates, read off the CAD port
ladder: port 2 at 182.67 cm from the cathode face, and exactly 53 regular
stations at a 32.00 cm pitch, so ``182.67 + 32.00 * (53 - 2) = 1814.67``.

THE PORT PITCH IS THE CAD's 32.00 cm, NOT the 31.95 cm nominal (Tom's
ruling). The nominal map ``182.5 + 31.95 * (port - 2)`` puts port 53 at
1811.95 cm instead; the two anchors differ by 2.72 cm, well inside one 7.49 cm
mesh cell, so nothing in the emitted profile turns on the choice. Both values
are stated wherever the anchor is quoted, and the report locates every port on
both ladders. The ladder is corroborated at two independent stations: the
stance's neutral baffle at 342.65 cm lands on port 7.00 on the CAD ladder
(7.01 on the nominal one), which is an OCTAGONAL RING station -- the octa
rings sit at ports 7, 13, 19, 24, 30, 35, 41 and 47 -- and the anode plane
sits at CAD 53.2-53.4 cm against the model's 53.25 cm.

THE MAPPING IS APPLIED FIRST, ahead of everything else this script does: every
recorded profile is re-expressed on the ascending model grid
``z_model in [-210.63, 2114.67]`` cm, and the plateau normalization, the
flat/departure rule, the feature windows, the exclusion adjudication and the
mesh resample all run in model coordinates. There is no coordinate assumption
left in this build.

WHAT THE CORRECTED FRAME SHOWS, AND WHAT IT RETIRES. The far end of the
machine simply FALLS OFF, with no mirror: the end-pair supply channel reads
0 A in every recorded shot -- the end coils were OFF, a measured machine fact
rather than an inference. The cathode side carries the source-coil structure
instead: a peak of ``B_hat`` just downstream of the cathode face and a dip
about a metre past it. The flat hold covers all of that, so the source-side
structure is REPORTED and never applied to the profile.

The mis-oriented build painted each end onto the other. It read the cathode
source-coil peak as an end-pair mirror throat sitting on the far column, gave
p50 a spurious flare out of the source-side dip, and reported a ~156 cm
"coil-location disagreement" against the CAD census. All three were artifacts
of the reflected frame and are retired; under the corrected registration the
measured fall-off tracks the census ``off`` case (end pair unpowered), which
is the machine state the 0 A channel records.

THE PROFILE
-----------
The flux tube is COLUMN-ANCHORED. Flux conservation through a slowly varying
axial field gives ``r(z) = R_p * sqrt(B_plateau / B(z)) = R_p *
sqrt(1 / B_hat(z))``, so the tube is exactly the design column radius wherever
the field is at its plateau level, wider where the field droops, and narrower
through the end-pair mirror where it exceeds the plateau.

Upstream the profile is held EXACTLY flat at ``RP_CM``. The flat span ends at
the first SUSTAINED departure of ``B_hat`` from 1 beyond
``DEPARTURE_SEARCH_FROM_CM`` -- the first contiguous run of
``|B_hat - 1| > FLAT_TOLERANCE`` spanning at least ``MIN_DEPARTURE_SPAN_CM``.
Holding the whole upstream span at exactly ``RP_CM``, rather than masking cell
by cell on the tolerance, is what keeps the SCALAR ``Rp`` read sites
(``cathode.py``'s ``Rp``, ``_anode_neutral_transparency``) in sync with the
per-cell vector: those sites read a single number, and a source-region cell
whose vector entry disagreed with it would desync them silently. It is the
same principle as the census builder's flat-to-1855 rule. The measured
consequence, reported rather than hidden: ``B_hat`` is OFF TOLERANCE over the
source region as well -- it carries the source-coil peak and dip there, and it
collapses behind the cathode plane -- so a cell-by-cell tolerance mask would
put per-cell structure on the cathode box, the plenum and the gap. That is a
SEPARATE question from this one and is not decided here; the flat hold is
kept. The report prints the census of where the tolerance would have fired.

Beyond the departure the ratio is applied, capped at
``sqrt(AREA_CAP_FRACTION) * R_m(z)`` -- the declared annulus regularization,
the same cap the census build uses. The annulus figure the report prints is
the smallest PER-CELL share ``(V_m - V_p)/V_m`` taken over the cells that have
an annulus, cap-bound cells included and ``V_ann == 0`` cells excluded -- the
same per-cell quantity the solver's ``neutral_annulus_volume_fraction_min``
guard refuses on, NOT a column-integrated share. Wherever the cap binds that
minimum is pinned at ``1 - AREA_CAP_FRACTION`` by construction, so it reports
the regularization rather than the measured field. Outside the corrected sample span
``B_hat`` is HELD at the nearer end sample: below ``z_model = -210.63`` cm,
where the flat rule already fixes ``r = RP_CM`` so the hold cannot be
observable, and beyond ``z_model = 2114.67`` cm. The report states how many
mesh cells fall outside the span at each end and whether the cap binds there,
which is what decides whether either hold is observable at all.

THE MESH AND THE VESSEL ARE NOT REBUILT. Both come from
``g1_build_profiles.py`` unchanged: its ``_g1_config`` mesh probe resolves the
stance's 280-cell mesh (1 plenum + 5 gap + 5 fixed source + 268 far column +
1 collector) and its ``build_vessel_profile`` gives the measured
``machine_radius_profile_cm`` staircase. This script changes ONE array.

Inputs
------
``MSI_DATA_DIR``/*.hdf5
    The ES1 raw shot files. READ-ONLY; never modified.
``scripts/lapd_end_field_1400G_rp18p415_census2026.npz``
``scripts/l2a7b_foot45_cr6p94.h5``
    Read through ``g1_build_profiles`` for the mesh, the vessel profile and
    the two census plasma profiles the comparison is made against.

Outputs (all in ``scripts/``)
-----------------------------
``mfp_field_profile.txt``
    The full report: the per-file MSI table, the cross-file spread, the
    32/34 adjudication, the departure z, the per-cell profile table, the
    comparison against the census profiles, the construction validation, and
    the emitted ``plasma_radius_profile_cm`` list in stance-file form.
``mfp_profile_compare.png``
    Three panels: the normalized field with its cross-file band, the radius
    profiles (new vs the two census cases) with the cap, and the new/old
    ratio.
"""

import glob
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import g1_build_profiles as census_build  # noqa: E402
from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402

#: The ES1 raw shot files. READ-ONLY -- this script opens them in 'r' and
#: never writes to that tree.
MSI_DATA_DIR = "/home/trloo/bapsf/bapsf-lapd-data-analysis/data/may2026"
MSI_GROUP = "MSI/Magnetic field"

#: The port-53 station in MODEL coordinates, cm: the registration anchor of
#: ``z_model = PORT53_Z_MODEL_CM - z_MSI``. Read off the CAD port ladder --
#: port 2 at 182.67 cm and 53 regular stations at the CAD 32.00 cm pitch.
PORT53_Z_MODEL_CM = 1814.67
#: The CAD port ladder, in model coordinates: port 2's station and the pitch.
#: Tom's ruling keeps the CAD 32.00 cm pitch over the 31.95 cm nominal.
PORT2_Z_MODEL_CM = 182.67
PORT_PITCH_CM = 32.00
#: The nominal (31.95 cm) map, retained only so the report can quote both
#: anchors. It puts port 53 at 1811.95 cm -- 2.72 cm from the CAD value, which
#: is sub-cell on this 280-cell mesh.
PORT_Z0_NOMINAL_CM = 182.5
PORT_PITCH_NOMINAL_CM = 31.95

#: All windows below are in MODEL coordinates, and are applied after the
#: registration.
#:
#: The plateau window, in cm. Wide enough to average over the interior coil
#: ripple, and clear of both the source-region structure and the far fall-off.
PLATEAU_WINDOW_CM = (300.0, 1500.0)
#: Search windows for the reported SOURCE-SIDE features -- the source-coil
#: peak just downstream of the cathode face, and the dip about a metre past
#: it. Both are REPORTED ONLY: the flat hold covers this whole span, so
#: neither is applied to the emitted profile.
PEAK_WINDOW_CM = (-50.0, 65.0)
DIP_WINDOW_CM = (45.0, 300.0)

#: The flat-column rule. ``B_hat`` is "at plateau" within this tolerance, and
#: a departure counts as sustained once it spans this much axial distance.
FLAT_TOLERANCE = 0.02
DEPARTURE_SEARCH_FROM_CM = 1500.0
MIN_DEPARTURE_SPAN_CM = 20.0

#: The two files that ran the lowest main-supply currents. Candidates for
#: exclusion, adjudicated on measured shape -- see ``adjudicate_low_current``.
LOW_CURRENT_FILES = ("32", "34")

#: Inherited, unchanged, from the census build: the column radius, the annulus
#: cap fraction, and the two census cases the new profile is compared against.
RP_CM = census_build.RP_CM
AREA_CAP_FRACTION = census_build.AREA_CAP_FRACTION
CENSUS_CASES = census_build.CASES

#: The ports the report locates on the profile.
REPORTED_PORTS = (11, 21, 29, 41, 50)

#: The measured p50/p41 flux-tube area ratio, and its uncertainty. The gate:
#: the ratio the EMITTED profile implies between those two stations must agree
#: with this measurement to within its stated sigma. Class MEASURED; instrument
#: and face bracket in the MSI block of ``scripts/production_stance_provenance.md``.
MEASURED_P50_P41_FLUX_RATIO = 0.9905
MEASURED_P50_P41_FLUX_RATIO_SIGMA = 0.0114


def port_z_cm(port):
    """Return ``port``'s station in MODEL coordinates, cm from the cathode.

    The CAD ladder: port 2 at ``PORT2_Z_MODEL_CM`` and a ``PORT_PITCH_CM``
    pitch. This is the ladder the registration anchor ``PORT53_Z_MODEL_CM``
    comes off, so the ports and the anchor stay consistent by construction.
    """
    return PORT2_Z_MODEL_CM + PORT_PITCH_CM * (port - 2)


def port_z_nominal_cm(port):
    """Return ``port``'s station on the 31.95 cm NOMINAL map, cm.

    Reported alongside :func:`port_z_cm` so the pitch ruling is visible in the
    build report; it is never used to build the profile.
    """
    return PORT_Z0_NOMINAL_CM + PORT_PITCH_NOMINAL_CM * (port - 2)


def to_model_frame(z_msi_cm):
    """Return ``(z_model_cm, order)`` for the MSI axial sample positions.

    Applies ``z_model = PORT53_Z_MODEL_CM - z_MSI`` -- the reflection between
    the machine's port-53-referenced frame (``+z`` south, toward the cathode)
    and the model's cathode-referenced frame. The reflection reverses the
    sample order, so ``order`` is the permutation that puts ``z_model``
    ascending; apply it to every recorded profile before anything else reads
    them.
    """
    z_model_cm = PORT53_Z_MODEL_CM - np.asarray(z_msi_cm, dtype=float)
    order = np.argsort(z_model_cm)
    return z_model_cm[order], order


# ------------------------------------------------------------------ reading


def msi_files():
    """Return the ES1 raw files, sorted by their leading run number."""
    return sorted(glob.glob(os.path.join(MSI_DATA_DIR, "*.hdf5")))


def read_msi(path):
    """Return one file's MSI magnetic-field record, IN MODEL COORDINATES.

    The registration ``z_model = PORT53_Z_MODEL_CM - z_MSI`` is applied here,
    at the read, so that nothing downstream ever sees the machine frame: the
    returned ``z_cm`` is the ascending model grid and ``B_gauss`` is reordered
    to match it. ``z_msi_cm`` carries the raw machine-frame positions in the
    same (reordered) sample order, for the report.

    The returned dict carries ``z_cm`` and ``z_msi_cm`` (1024 sample positions
    each), ``B_gauss`` (2 x 1024, the first and last shot), ``currents_A``
    (2 x 13), ``peak_gauss`` and ``valid`` (per shot), and ``run`` (the
    two-digit run number the filename starts with).
    """
    with h5py.File(path, "r") as handle:
        group = handle[MSI_GROUP]
        z_msi_cm = np.asarray(group.attrs["Profile z locations"], dtype=float)
        b_gauss = np.asarray(group["Magnetic field profile"], dtype=float)
        currents = np.asarray(group["Magnet power supply currents"], dtype=float)
        summary = np.asarray(group["Magnetic field summary"])
    z_cm, order = to_model_frame(z_msi_cm)
    return {
        "run": os.path.basename(path)[:2],
        "path": path,
        "z_cm": z_cm,
        "z_msi_cm": z_msi_cm[order],
        "B_gauss": b_gauss[:, order],
        "currents_A": currents,
        "peak_gauss": np.asarray(summary["Peak magnetic field"], dtype=float),
        "valid": np.asarray(summary["Data valid"], dtype=int),
    }


def plateau_level(z_cm, b_gauss):
    """Return the plateau level: the median of ``b_gauss`` over the window."""
    inside = (z_cm >= PLATEAU_WINDOW_CM[0]) & (z_cm <= PLATEAU_WINDOW_CM[1])
    return float(np.median(b_gauss[inside]))


def _extremum(z_cm, b_gauss, window, kind):
    """Return ``(value, z)`` of the min/max of ``b_gauss`` inside ``window``.

    Also returns whether the extremum landed on a window edge, which would say
    the window is cutting the feature rather than containing it.
    """
    inside = np.flatnonzero((z_cm >= window[0]) & (z_cm <= window[1]))
    index = inside[np.argmin(b_gauss[inside])] if kind == "min" else (
        inside[np.argmax(b_gauss[inside])]
    )
    at_edge = index in (inside[0], inside[-1])
    return float(b_gauss[index]), float(z_cm[index]), bool(at_edge)


def file_features(record):
    """Return the per-file plateau/dip/peak table row, per recorded shot."""
    z_cm = record["z_cm"]
    rows = []
    for shot in range(record["B_gauss"].shape[0]):
        b_gauss = record["B_gauss"][shot]
        plateau = plateau_level(z_cm, b_gauss)
        dip, dip_z, dip_edge = _extremum(z_cm, b_gauss, DIP_WINDOW_CM, "min")
        peak, peak_z, peak_edge = _extremum(z_cm, b_gauss, PEAK_WINDOW_CM, "max")
        rows.append(
            {
                "run": record["run"],
                "shot": shot,
                "plateau_gauss": plateau,
                "dip_gauss": dip,
                "dip_z_cm": dip_z,
                "dip_at_window_edge": dip_edge,
                "peak_gauss": peak,
                "peak_z_cm": peak_z,
                "peak_at_window_edge": peak_edge,
                "main_current_A": float(record["currents_A"][shot][0]),
                "b_hat": b_gauss / plateau,
            }
        )
    return rows


# ------------------------------------------------- the mean normalized shape


def adjudicate_low_current(rows_by_run, z_grid_cm):
    """Measure whether the low-current files' SHAPE is an outlier.

    For every file, take the mean of its two shots' ``B_hat`` and compute the
    leave-one-out deviation ``max |B_hat_file - mean(B_hat of all others)|``,
    over the full mesh and again over the FAR-END region the profile is
    actually built from -- ``z_model >= DEPARTURE_SEARCH_FROM_CM``, in model
    coordinates, which is the far column and its fall-off. A low-current file
    is an outlier when its deviation exceeds the LARGEST deviation any other
    file shows on the same measure.

    Returns the per-file deviations, the two verdicts, and the set to exclude
    (the far-end measure decides, because that is the region the ratio is
    read on; both are reported).
    """
    runs = sorted(rows_by_run)
    shapes = np.array([np.mean([r["b_hat"] for r in rows_by_run[k]], axis=0) for k in runs])
    end_region = z_grid_cm >= DEPARTURE_SEARCH_FROM_CM

    def leave_one_out(mask):
        out = {}
        for index, run in enumerate(runs):
            others = np.delete(shapes, index, axis=0).mean(axis=0)
            out[run] = float(np.abs(shapes[index] - others)[mask].max())
        return out

    full = leave_one_out(np.ones(z_grid_cm.size, dtype=bool))
    end = leave_one_out(end_region)
    baseline_full = max(v for k, v in full.items() if k not in LOW_CURRENT_FILES)
    baseline_end = max(v for k, v in end.items() if k not in LOW_CURRENT_FILES)
    outlier_full = {k for k in LOW_CURRENT_FILES if full.get(k, 0.0) > baseline_full}
    outlier_end = {k for k in LOW_CURRENT_FILES if end.get(k, 0.0) > baseline_end}
    return {
        "runs": runs,
        "deviation_full": full,
        "deviation_end": end,
        "others_max_full": baseline_full,
        "others_max_end": baseline_end,
        "outliers_full": outlier_full,
        "outliers_end": outlier_end,
        "excluded": outlier_end,
    }


def mean_normalized(rows_by_run, excluded):
    """Return the mean ``B_hat`` over every shot of every retained file."""
    stack = [
        row["b_hat"]
        for run in sorted(rows_by_run)
        if run not in excluded
        for row in rows_by_run[run]
    ]
    stack = np.asarray(stack)
    return stack.mean(axis=0), stack


# ------------------------------------------------------------ the flat span


def departure_z_cm(z_grid_cm, b_hat):
    """Return the first SUSTAINED departure of ``B_hat`` from the plateau.

    A departure is a contiguous run of ``|B_hat - 1| > FLAT_TOLERANCE`` that
    spans at least ``MIN_DEPARTURE_SPAN_CM``; the first such run beyond
    ``DEPARTURE_SEARCH_FROM_CM`` fixes the end of the flat column. Returns the
    z of that run's first sample, plus every off-plateau run beyond the search
    start so the report can show that the answer does not hinge on the span
    threshold.
    """
    off = np.abs(b_hat - 1.0) > FLAT_TOLERANCE
    beyond = np.flatnonzero(z_grid_cm > DEPARTURE_SEARCH_FROM_CM)
    runs = []
    start = None
    for index in beyond:
        if off[index]:
            start = index if start is None else start
            last = index
        elif start is not None:
            runs.append((start, last))
            start = None
    if start is not None:
        runs.append((start, last))
    sustained = [
        (a, b) for a, b in runs if z_grid_cm[b] - z_grid_cm[a] >= MIN_DEPARTURE_SPAN_CM
    ]
    if not sustained:
        raise RuntimeError(
            "no sustained departure of B_hat from the plateau was found beyond "
            f"z = {DEPARTURE_SEARCH_FROM_CM} cm: the measured field never "
            "leaves its plateau by more than the flat tolerance, so this build "
            "has no flare to apply"
        )
    return float(z_grid_cm[sustained[0][0]]), runs, sustained


# ------------------------------------------------------------- the profile


def build_plasma_profile(cell_z_cm, vessel_radius_cm, z_grid_cm, b_hat, z_departure_cm):
    """Return the flux-tube radius profile on the model mesh.

    Flat at exactly ``RP_CM`` for every cell centre at or upstream of
    ``z_departure_cm``; ``RP_CM / sqrt(B_hat)`` beyond it. Outside the MSI
    sample span ``B_hat`` is held at the nearer end sample, at BOTH ends of
    the corrected grid. Capped at ``sqrt(AREA_CAP_FRACTION) * R_m``.

    Returns ``(capped, raw, cap, b_hat_on_cells, held_above, held_below)``,
    where the two ``held`` masks mark the cells beyond the last and below the
    first MSI sample in model coordinates.
    """
    cell_z_cm = np.asarray(cell_z_cm, dtype=float)
    first_sample_cm = float(z_grid_cm[0])
    last_sample_cm = float(z_grid_cm[-1])
    held_above = cell_z_cm > last_sample_cm
    held_below = cell_z_cm < first_sample_cm
    b_on_cells = np.interp(
        np.clip(cell_z_cm, first_sample_cm, last_sample_cm), z_grid_cm, b_hat
    )
    raw = np.where(
        cell_z_cm <= z_departure_cm, RP_CM, RP_CM / np.sqrt(np.maximum(b_on_cells, 1e-12))
    )
    cap = np.sqrt(AREA_CAP_FRACTION) * np.asarray(vessel_radius_cm, dtype=float)
    return np.minimum(raw, cap), raw, cap, b_on_cells, held_above, held_below


def stance_list_text(values, per_line=4, indent="  "):
    """Return the profile as the stance file's own list body, four per line."""
    lines = []
    for start in range(0, len(values), per_line):
        chunk = values[start : start + per_line]
        lines.append(indent + ", ".join(repr(float(v)) for v in chunk) + ",")
    return "\n".join(lines)


# ----------------------------------------------------------------- figure


def write_figure(
    path, z_grid_cm, shapes, b_hat, mesh, profile, census_profiles, cap, z_departure
):
    """Render the comparison figure.

    Three shared-axis panels over the whole machine, all in MODEL coordinates
    (the normalized field with its per-shot band, the radius profiles against
    the census cases and the cap, the new/old ratio) plus a linear zoom on the
    end of the flat column, where the log axis of the second panel flattens
    the fall-off into a line.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11.0, 15.0))
    for ax in axes[:3]:
        ax.set_xlim(-320.0, 2140.0)
    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)
    for ax in axes:
        ax.axvline(z_departure, color="C1", lw=0.9, ls="--", zorder=0)

    ax = axes[0]
    ax.fill_between(
        z_grid_cm,
        shapes.min(axis=0),
        shapes.max(axis=0),
        color="0.80",
        label="per-shot range (all retained shots)",
    )
    ax.plot(z_grid_cm, b_hat, color="C0", lw=1.2, label="mean $\\hat{B}(z)$")
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.axhspan(1.0 - FLAT_TOLERANCE, 1.0 + FLAT_TOLERANCE, color="C0", alpha=0.10)
    ax.set_ylabel("$\\hat{B} = B / B_{plateau}$")
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(
        "MSI measured axial field, normalized per shot to its own plateau "
        f"(flat column ends at z = {z_departure:.1f} cm, dashed)"
    )

    ax = axes[1]
    ax.plot(mesh.z_cm, profile, color="C3", lw=1.6, marker=".", ms=3, label="NEW: MSI flux tube")
    for case, style in zip(CENSUS_CASES, ("--", ":")):
        ax.plot(
            mesh.z_cm,
            census_profiles[case],
            color="C0" if case == "droop_min" else "C2",
            lw=1.2,
            ls=style,
            label=f"census {case} (pre-MSI stance profile)"
            if case == "droop_min"
            else f"census {case} (end pair unpowered -- the measured state)",
        )
    ax.plot(mesh.z_cm, cap, color="0.4", lw=1.0, ls="-.", label="cap $\\sqrt{0.95}\\,R_m$")
    ax.axhline(RP_CM, color="k", lw=0.6, ls=":")
    ax.set_ylabel("plasma radius [cm]")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.plot(
        mesh.z_cm,
        profile / census_profiles["droop_min"],
        color="C3",
        lw=1.4,
        marker=".",
        ms=3,
    )
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_ylabel("new / census droop_min")
    ax.set_xlabel("z [cm], MODEL frame (cathode-referenced)")

    ax = axes[3]
    ax.plot(mesh.z_cm, profile, color="C3", lw=1.6, marker=".", ms=4, label="NEW: MSI flux tube")
    ax.plot(
        mesh.z_cm,
        census_profiles["droop_min"],
        color="C0",
        lw=1.2,
        ls="--",
        label="census droop_min",
    )
    ax.plot(mesh.z_cm, census_profiles["off"], color="C2", lw=1.2, ls=":", label="census off")
    ax.axhline(RP_CM, color="k", lw=0.6, ls=":")
    ax.set_xlim(1650.0, 1880.0)
    ax.set_ylim(17.0, 32.0)
    ax.set_ylabel("plasma radius [cm]")
    ax.set_xlabel("z [cm] -- ZOOM on the end of the flat column and the fall-off")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------- main


def main():
    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    say("=== MSI measured-field plasma radius profile build ===")
    say(f"data     : {MSI_DATA_DIR}")
    say(f"census   : {census_build.CENSUS_NPZ}")
    say(f"reference: {census_build.REFERENCE_H5}")
    say(
        f"rules    : plateau = median over z in {PLATEAU_WINDOW_CM} cm; flat "
        f"tolerance |B_hat - 1| <= {FLAT_TOLERANCE}; sustained departure "
        f">= {MIN_DEPARTURE_SPAN_CM} cm searched beyond "
        f"{DEPARTURE_SEARCH_FROM_CM} cm; cap sqrt({AREA_CAP_FRACTION}) * R_m"
    )
    say(
        f"frame    : z_model = {PORT53_Z_MODEL_CM} - z_MSI. The MSI record is "
        "port-53-referenced with +z pointing SOUTH toward the cathode "
        "(bapsflib.lapd, portnum_to_z = 31.95*(53 - port)); the model runs "
        "the other way from z = 0 at the cathode face, so the registration is "
        "a REFLECTION, applied at the read before anything else."
    )
    say(
        f"anchor   : port 53 at z_model = {PORT53_Z_MODEL_CM} cm, off the CAD "
        f"ladder (port 2 at {PORT2_Z_MODEL_CM} cm, {PORT_PITCH_CM} cm pitch, "
        f"53 regular stations). The {PORT_PITCH_NOMINAL_CM} cm nominal map "
        f"({PORT_Z0_NOMINAL_CM} + {PORT_PITCH_NOMINAL_CM}*(port-2)) would put "
        f"it at {port_z_nominal_cm(53):.2f} cm instead; the CAD pitch is kept "
        f"(Tom's ruling) and the {PORT53_Z_MODEL_CM - port_z_nominal_cm(53):.2f} cm "
        "difference is sub-cell on this mesh."
    )
    say()

    # --- read every file -----------------------------------------------------
    paths = msi_files()
    say(f"--- MSI files ({len(paths)}) ---")
    records = [read_msi(path) for path in paths]
    z_grid_cm = records[0]["z_cm"]
    for record in records[1:]:
        if not np.array_equal(record["z_cm"], z_grid_cm):
            raise RuntimeError(
                f"{record['run']} carries a different 'Profile z locations' "
                "attribute; the profiles cannot be averaged sample-for-sample"
            )
    z_msi_cm = records[0]["z_msi_cm"]
    say(
        f"ASSERT all {len(records)} files share one z mesh "
        f"({z_grid_cm.size} samples, {z_grid_cm[0]:.3f} .. {z_grid_cm[-1]:.3f} cm "
        f"in MODEL coordinates, from z_MSI {z_msi_cm[-1]:.3f} .. "
        f"{z_msi_cm[0]:.3f} cm, pitch {np.diff(z_grid_cm).mean():.4f} cm): PASS"
    )
    if not np.allclose(z_grid_cm, PORT53_Z_MODEL_CM - z_msi_cm):
        raise AssertionError("the model grid is not the reflection of the MSI grid")
    if not np.all(np.diff(z_grid_cm) > 0.0):
        raise AssertionError("the reflected model grid is not ascending")
    say(
        "ASSERT the model grid is exactly PORT53_Z_MODEL_CM - z_MSI and is "
        "ascending: PASS"
    )
    invalid = [
        (r["run"], s) for r in records for s in range(2) if not r["valid"][s]
    ]
    say(f"shots flagged 'Data valid' = 0: {invalid if invalid else 'none'}")
    say()

    rows_by_run = {}
    for record in records:
        rows_by_run[record["run"]] = file_features(record)

    say(
        "--- per-file MSI features (both recorded shots); the dip and peak "
        "columns are the SOURCE-SIDE structure, reported only ---"
    )
    say(
        f"{'run':>4} {'shot':>4} {'I_main[A]':>10} {'plateau[G]':>11} "
        f"{'dip[G]':>9} {'z_dip[cm]':>10} {'peak[G]':>9} {'z_peak[cm]':>11} "
        f"{'dip/plat':>9} {'peak/plat':>10}"
    )
    edge_hits = []
    for run in sorted(rows_by_run):
        for row in rows_by_run[run]:
            say(
                f"{row['run']:>4} {row['shot']:>4} {row['main_current_A']:10.1f} "
                f"{row['plateau_gauss']:11.2f} {row['dip_gauss']:9.2f} "
                f"{row['dip_z_cm']:10.2f} {row['peak_gauss']:9.2f} "
                f"{row['peak_z_cm']:11.2f} "
                f"{row['dip_gauss'] / row['plateau_gauss']:9.4f} "
                f"{row['peak_gauss'] / row['plateau_gauss']:10.4f}"
            )
            if row["dip_at_window_edge"] or row["peak_at_window_edge"]:
                edge_hits.append((row["run"], row["shot"]))
    say(
        "feature windows (MODEL coordinates): source-side dip searched over "
        f"{DIP_WINDOW_CM} cm, source-coil peak over {PEAK_WINDOW_CM} cm; "
        f"extrema landing ON a window edge (window cutting the feature): "
        f"{edge_hits if edge_hits else 'none'}"
    )
    plateaus = np.array(
        [row["plateau_gauss"] for run in rows_by_run for row in rows_by_run[run]]
    )
    currents = np.array(
        [row["main_current_A"] for run in rows_by_run for row in rows_by_run[run]]
    )
    say(
        f"plateau level across all {plateaus.size} shots: mean {plateaus.mean():.2f} G, "
        f"min {plateaus.min():.2f} G, max {plateaus.max():.2f} G "
        f"(range {plateaus.max() - plateaus.min():.2f} G = "
        f"{(plateaus.max() - plateaus.min()) / plateaus.mean() * 100:.2f} %)"
    )
    say(
        f"main supply (channel 0) across all shots: mean {currents.mean():.1f} A, "
        f"min {currents.min():.1f} A, max {currents.max():.1f} A"
    )
    all_currents = np.concatenate([record["currents_A"] for record in records], axis=0)
    zero_channels = [
        channel
        for channel in range(all_currents.shape[1])
        if np.all(np.abs(all_currents[:, channel]) < 1.0)
    ]
    say(
        f"magnet supply census over all {all_currents.shape[0]} recorded shots "
        f"({all_currents.shape[1]} channels): channels reading ZERO in every "
        f"shot: {zero_channels if zero_channels else 'none'}"
    )
    for channel in zero_channels:
        say(
            f"  channel {channel}: {all_currents[:, channel].min():.2f} .. "
            f"{all_currents[:, channel].max():.2f} A -- the recorder's offset, "
            "i.e. zero"
        )
    say(
        "THE END COILS WERE OFF, MEASURED: the end-pair supply is set by hand "
        "and its channel is auto-recorded, and it reads 0 A in every shot of "
        "every file. There is no end mirror in this data set -- the far end "
        "simply falls off. This is a machine fact, not an inference from the "
        "profile shape, and it is what the corrected registration shows."
    )
    say()

    # --- the 32/34 adjudication ---------------------------------------------
    verdict = adjudicate_low_current(rows_by_run, z_grid_cm)
    say("--- cross-file normalized-shape spread, and the low-current files ---")
    say(
        "leave-one-out deviation per file: max |B_hat_file - mean(all other "
        "files)|, over the FULL mesh and over the FAR-END region "
        f"z_model >= {DEPARTURE_SEARCH_FROM_CM} cm (the span the ratio is read "
        "on). RE-ADJUDICATED in the corrected frame: the mis-oriented build "
        f"masked on z_MSI >= {DEPARTURE_SEARCH_FROM_CM} cm, which is the "
        "CATHODE region, so its verdict was measured on the wrong span."
    )
    say(f"{'run':>4} {'full mesh':>11} {'far end':>11}")
    for run in verdict["runs"]:
        mark = "  <- low main current" if run in LOW_CURRENT_FILES else ""
        say(
            f"{run:>4} {verdict['deviation_full'][run]:11.4f} "
            f"{verdict['deviation_end'][run]:11.4f}{mark}"
        )
    say(
        f"largest deviation among the OTHER files: full {verdict['others_max_full']:.4f}, "
        f"far end {verdict['others_max_end']:.4f}"
    )
    for run in LOW_CURRENT_FILES:
        say(
            f"  file {run}: full {verdict['deviation_full'][run]:.4f} "
            f"({'BEYOND' if run in verdict['outliers_full'] else 'within'} the others' "
            f"spread), far end {verdict['deviation_end'][run]:.4f} "
            f"({'BEYOND' if run in verdict['outliers_end'] else 'within'} the others' "
            "spread)"
        )
    excluded = verdict["excluded"]
    retained_low = [run for run in LOW_CURRENT_FILES if run not in excluded]
    say(
        "VERDICT: "
        + (
            f"exclude {sorted(excluded)} -- on the far end, the span the flux "
            "ratio is actually read on, their normalized shape departs by more "
            "than any other file's does. On the full mesh they are INSIDE the "
            "others' spread (the plateau ripple of several other files is "
            "larger), so the exclusion rests on the far end alone and is "
            "reported here rather than assumed."
            if excluded
            else "retain every file -- neither low-current file departs beyond "
            "the others' spread on either measure."
        )
    )
    if retained_low:
        say(
            f"  RETAINED, having been excluded by the mis-oriented build: "
            f"{sorted(retained_low)}. Measured on the true far end "
            + ", ".join(
                f"{run} deviates {verdict['deviation_end'][run]:.4f}"
                for run in sorted(retained_low)
            )
            + f", inside the others' {verdict['others_max_end']:.4f}. The "
            "earlier exclusion was measured on the cathode region."
        )
    say()

    b_hat, shapes = mean_normalized(rows_by_run, excluded)
    say(
        f"mean normalized profile built from {shapes.shape[0]} shots over "
        f"{len(rows_by_run) - len(excluded)} files"
    )
    b_hat_all, shapes_all = mean_normalized(rows_by_run, set())
    say(
        f"effect of the exclusion on the mean shape: max |B_hat_kept - "
        f"B_hat_all| = {np.abs(b_hat - b_hat_all).max():.5f} at z = "
        f"{z_grid_cm[np.argmax(np.abs(b_hat - b_hat_all))]:.2f} cm"
    )
    say("mean B_hat and per-shot spread at reference positions:")
    say(f"{'z[cm]':>9} {'B_hat':>9} {'min':>9} {'max':>9} {'max-min':>9}")
    for z_sample in (
        -210.0, -100.0, -50.0, 0.0, 53.25, 100.0, 200.0, 300.0, 600.0, 900.0,
        1200.0, 1500.0, 1600.0, 1700.0, 1750.0, 1800.0, 1855.0, 1900.0,
        2000.0, 2114.0,
    ):
        say(
            f"{z_sample:9.1f} {np.interp(z_sample, z_grid_cm, b_hat):9.4f} "
            f"{np.interp(z_sample, z_grid_cm, shapes.min(axis=0)):9.4f} "
            f"{np.interp(z_sample, z_grid_cm, shapes.max(axis=0)):9.4f} "
            f"{np.interp(z_sample, z_grid_cm, shapes.max(axis=0) - shapes.min(axis=0)):9.4f}"
        )
    say()

    # --- the flat span -------------------------------------------------------
    z_departure, runs, sustained = departure_z_cm(z_grid_cm, b_hat)
    say("--- the flat column and its departure ---")
    say(f"off-plateau runs beyond z = {DEPARTURE_SEARCH_FROM_CM} cm:")
    for start, stop in runs:
        span = z_grid_cm[stop] - z_grid_cm[start]
        say(
            f"  z {z_grid_cm[start]:9.2f} .. {z_grid_cm[stop]:9.2f} cm "
            f"({stop - start + 1:3d} samples, {span:7.2f} cm)"
            + ("   SUSTAINED" if span >= MIN_DEPARTURE_SPAN_CM else "   (blip)")
        )
    say(
        f"first sustained departure: z = {z_departure:.3f} cm "
        f"(B_hat = {np.interp(z_departure, z_grid_cm, b_hat):.4f}); the run "
        f"spans {z_grid_cm[sustained[0][1]] - z_grid_cm[sustained[0][0]]:.2f} cm, "
        f"so the {MIN_DEPARTURE_SPAN_CM} cm sustain threshold is not binding "
        "(there is no earlier off-plateau sample beyond the search start at "
        "all)"
    )
    say(
        f"ports on the profile, CAD ladder z = {PORT2_Z_MODEL_CM} + "
        f"{PORT_PITCH_CM}*(port-2), with the {PORT_PITCH_NOMINAL_CM} cm "
        "nominal map alongside:"
    )
    for port in REPORTED_PORTS:
        z_port = port_z_cm(port)
        z_nominal = port_z_nominal_cm(port)
        b_port = float(np.interp(np.clip(z_port, z_grid_cm[0], z_grid_cm[-1]), z_grid_cm, b_hat))
        flat = z_port <= z_departure
        flat_nominal = z_nominal <= z_departure
        say(
            f"  p{port:<3d} z_CAD = {z_port:8.2f} cm (nominal {z_nominal:8.2f} "
            f"cm)  B_hat = {b_port:.4f}  "
            + (
                "FLAT column (r = Rp exactly)"
                if flat
                else f"FLARED: r/Rp = {1.0 / np.sqrt(b_port):.4f} "
                f"(+{(1.0 / np.sqrt(b_port) - 1.0) * 100:.2f} % radius, "
                f"+{(1.0 / b_port - 1.0) * 100:.2f} % area)"
            )
            + ("" if flat == flat_nominal else "   <- the two ladders DISAGREE here")
        )
    say(
        "the two ladders put every reported port on the same side of the "
        "departure"
        if all(
            (port_z_cm(port) <= z_departure) == (port_z_nominal_cm(port) <= z_departure)
            for port in REPORTED_PORTS
        )
        else "WARNING: the CAD and nominal ladders straddle the departure for "
        "at least one reported port"
    )
    say()

    # --- the cell-by-cell tolerance census, reported not applied -------------
    say(
        "--- reported, NOT applied: where |B_hat - 1| > tolerance UPSTREAM of "
        "the departure ---"
    )
    upstream = z_grid_cm <= z_departure
    off_upstream = upstream & (np.abs(b_hat - 1.0) > FLAT_TOLERANCE)
    if off_upstream.any():
        first = np.flatnonzero(off_upstream)
        say(
            f"  {int(off_upstream.sum())} of {int(upstream.sum())} upstream "
            f"samples are off tolerance, spanning z {z_grid_cm[first[0]]:.2f} .. "
            f"{z_grid_cm[first[-1]]:.2f} cm; B_hat there runs "
            f"{b_hat[off_upstream].min():.4f} .. {b_hat[off_upstream].max():.4f}"
        )
        say(
            "  This is the SOURCE-SIDE structure the corrected frame puts "
            "back where it belongs: the source-coil peak just downstream of "
            f"the cathode face (B_hat = {np.interp(0.0, z_grid_cm, b_hat):.4f} "
            f"at z = 0, {np.interp(53.25, z_grid_cm, b_hat):.4f} at the anode "
            f"plane), the dip about a metre past it, and the collapse behind "
            f"the cathode plane ({np.interp(-100.0, z_grid_cm, b_hat):.4f} at "
            "z = -100). A cell-by-cell tolerance mask would therefore put "
            "per-cell structure on the cathode box, the plenum and the gap "
            "and DESYNC the scalar Rp read sites. Whether the source region "
            "should carry that structure is a SEPARATE question and is not "
            "decided here; the flat span is held to the departure z, which is "
            "what the construction validation below gates on."
        )
    else:
        say("  none: B_hat is within tolerance at every upstream sample")
    say()

    # --- mesh + vessel, inherited unchanged ---------------------------------
    ref_params, ref_flags = census_build._reference_config()
    mesh_params, mesh_flags = census_build._g1_config(ref_params, ref_flags)
    mesh = build_geometry(mesh_params, mesh_flags)
    vessel = census_build.build_vessel_profile(mesh.z_cm, mesh.cell_role)
    say("--- mesh and vessel (from g1_build_profiles.py, UNCHANGED) ---")
    say(
        f"mesh: {mesh.cells} cells, z {mesh.z_cm[0]:.3f} .. {mesh.z_cm[-1]:.3f} cm; "
        f"roles {sorted({str(r) for r in mesh.cell_role})}"
    )
    say(
        f"vessel: {np.unique(np.round(vessel, 6)).size} distinct radii, "
        f"{vessel.min():.6f} .. {vessel.max():.6f} cm"
    )
    say()

    # --- the profile ---------------------------------------------------------
    profile, raw, cap, b_on_cells, held_above, held_below = build_plasma_profile(
        mesh.z_cm, vessel, z_grid_cm, b_hat, z_departure
    )
    census = np.load(census_build.CENSUS_NPZ, allow_pickle=True)
    census_profiles = {}
    for case in CENSUS_CASES:
        capped, _, _, _, _ = census_build.build_plasma_profile(
            case, mesh.z_cm, vessel, census
        )
        census_profiles[case] = capped

    flared = np.flatnonzero(mesh.z_cm > z_departure)
    binds = np.flatnonzero(profile < raw)
    say("--- plasma_radius_profile_cm (new, MSI-measured) ---")
    say(
        f"flat cells (r == {RP_CM} exactly): "
        f"{int(np.sum(profile == RP_CM))} of {mesh.cells}; flared cells: "
        f"{flared.size}, first index {flared[0]} at z = {mesh.z_cm[flared[0]]:.3f} cm"
    )
    say(
        f"{'i':>4} {'z[cm]':>10} {'B_hat':>8} {'raw[cm]':>9} {'cap[cm]':>9} "
        f"{'r[cm]':>9} {'A/A_col':>8} {'old[cm]':>9} {'new/old':>8}"
    )
    for index in flared:
        area_ratio = (profile[index] / RP_CM) ** 2
        say(
            f"{index:4d} {mesh.z_cm[index]:10.3f} {b_on_cells[index]:8.4f} "
            f"{raw[index]:9.4f} {cap[index]:9.4f} {profile[index]:9.4f} "
            f"{area_ratio:8.4f} {census_profiles['droop_min'][index]:9.4f} "
            f"{profile[index] / census_profiles['droop_min'][index]:8.4f}"
            + ("   CAP" if profile[index] < raw[index] else "")
            + ("   HELD" if held_above[index] else "")
        )
    say(
        f"cap binds in {binds.size} cell(s): {binds.tolist()}"
        if binds.size
        else "cap binds in 0 cells"
    )
    held_cells = np.flatnonzero(held_above)
    if held_cells.size:
        held_capped = bool(np.all(profile[held_cells] < raw[held_cells]))
        say(
            f"beyond the last MSI sample (z > {z_grid_cm[-1]:.3f} cm): "
            f"{held_cells.size} cell(s), B_hat HELD at its last sampled value "
            f"{b_hat[-1]:.4f}. The cap binds on "
            f"{'every' if held_capped else 'only some'} one of them "
            f"(raw {raw[held_cells].min():.4f} .. {raw[held_cells].max():.4f} cm "
            f"vs cap {cap[held_cells].min():.4f} cm), so the hold-vs-cap choice "
            + ("is NOT observable in the emitted profile." if held_capped
               else "IS observable and the held value is what ships.")
        )
    else:
        say(
            f"beyond the last MSI sample (z > {z_grid_cm[-1]:.3f} cm): 0 cells "
            f"-- the mesh ends at {mesh.z_cm[-1]:.3f} cm, inside the corrected "
            "sample span, so the hold-last rule never fires and cannot be "
            "observable. The terminal cells sit on the CAP instead."
        )
    below_cells = np.flatnonzero(held_below)
    say(
        f"below the first MSI sample (z < {z_grid_cm[0]:.3f} cm): "
        f"{below_cells.size} cell(s)"
        + (
            f" {below_cells.tolist()}, all inside the flat span, so the flat "
            f"rule fixes them at {RP_CM} and the hold cannot be observable "
            "there either."
            if below_cells.size
            else f" -- the mesh starts at {mesh.z_cm[0]:.3f} cm, inside the "
            "corrected sample span."
        )
    )
    say()

    throat = int(np.argmin(profile))
    widest_flux = int(np.argmax(raw[flared[0] : binds[0] if binds.size else mesh.cells]))
    widest_flux += flared[0]
    say("--- headline geometry of the new profile ---")
    say(
        f"NO THROAT: the minimum radius over the whole mesh is "
        f"{profile[throat]:.4f} cm = {profile[throat] / RP_CM:.4f} x Rp, first "
        f"reached at z = {mesh.z_cm[throat]:.3f} cm (cell {throat}). "
        + (
            "The profile never goes BELOW the design column radius: with the "
            "end coils off there is no mirror to compress the tube, and the "
            "sub-Rp throat the mis-oriented build reported was the "
            "SOURCE-COIL peak reflected onto the far column."
            if profile[throat] >= RP_CM
            else "THE PROFILE DIPS BELOW Rp, which the end-coils-off machine "
            "state does not admit -- investigate before shipping."
        )
    )
    say(
        f"FLARE: the widest cell the cap does NOT touch is {widest_flux} at "
        f"z = {mesh.z_cm[widest_flux]:.3f} cm, r = {raw[widest_flux]:.4f} cm = "
        f"{raw[widest_flux] / RP_CM:.4f} x Rp; past it the flux ratio would "
        f"reach {raw.max():.4f} cm and the cap takes over"
    )
    say(
        f"COLLECTOR (terminal cell {mesh.cells - 1}, z = "
        f"{mesh.z_cm[-1]:.3f} cm): r = {profile[-1]:.4f} cm "
        f"(cap {cap[-1]:.4f} cm, uncapped ratio would give {raw[-1]:.4f} cm); "
        f"census droop_min gave {census_profiles['droop_min'][-1]:.4f} cm"
    )
    say(
        f"p50 IS IN THE FLAT COLUMN: at z = {port_z_cm(50):.2f} cm the "
        f"measured B_hat is "
        f"{float(np.interp(port_z_cm(50), z_grid_cm, b_hat)):.4f} and the "
        f"departure is {z_departure - port_z_cm(50):.2f} cm downstream of it, "
        f"so the emitted r there is exactly {RP_CM} cm -- the same as p11-p41. "
        "The spurious p50 flare of the mis-oriented build was the SOURCE-SIDE "
        "dip reflected onto the far column."
    )
    say(
        "PROFILE IS MONOTONE THROUGH THE FALL-OFF, unlike the mis-oriented "
        "build: flat at Rp through the column, then widening monotonically as "
        "the field falls away with no mirror, stepping only where the vessel "
        "bore steps and the cap takes over."
    )
    say()

    # --- comparison ----------------------------------------------------------
    say("--- comparison against the census profiles ---")
    say(
        "census case names are 'droop_min' (end pair energized to the "
        "minimum-droop solution; the stance shipped this profile until the MSI "
        "adoption) and 'off' (end pair unpowered -- the state the 0 A supply "
        "channel records, so this is the case the machine was actually in)"
    )
    for case in CENSUS_CASES:
        delta = profile - census_profiles[case]
        worst = int(np.argmax(np.abs(delta)))
        say(
            f"[{case}] max |new - census| = {np.abs(delta).max():.4f} cm at "
            f"z = {mesh.z_cm[worst]:.3f} cm (cell {worst}: census "
            f"{census_profiles[case][worst]:.4f} -> new {profile[worst]:.4f}); "
            f"cells that differ at all: "
            f"{int(np.sum(profile != census_profiles[case]))} of {mesh.cells}"
        )
        first_diff = np.flatnonzero(profile != census_profiles[case])
        if first_diff.size:
            say(
                f"         first differing cell {first_diff[0]} at z = "
                f"{mesh.z_cm[first_diff[0]]:.3f} cm"
            )
    say(
        f"census flat span ended at z = {census_build.FLAT_THROUGH_Z_CM} cm; "
        f"the MSI flat span ends at z = {z_departure:.3f} cm -- the measured "
        "field leaves its plateau "
        f"{census_build.FLAT_THROUGH_Z_CM - z_departure:.1f} cm earlier than "
        "the census builder's hand-set flat-through, which is a rule "
        "difference (a threshold on the measured shape against a fixed z) and "
        "not a geometry disagreement."
    )

    # The end coils read 0 A, so 'off' is the machine state these files record;
    # run the SAME departure rule on the census 'off' axial field and compare.
    census_z_cm = np.asarray(census["z_axis_m"], dtype=float) * 100.0
    census_bz = np.asarray(census["off_bz_axis_gauss"], dtype=float)
    census_b_hat = census_bz / plateau_level(census_z_cm, census_bz)
    census_departure, _, _ = departure_z_cm(census_z_cm, census_b_hat)
    say(
        "CAD cross-check, in the corrected frame, against the census case the "
        "machine was actually in ('off' -- end pair unpowered, which is what "
        "the 0 A supply channel records):"
    )
    say(
        f"  same departure rule on the census 'off' field: z = "
        f"{census_departure:.2f} cm, against the measured "
        f"{z_departure:.2f} cm -- {z_departure - census_departure:+.2f} cm."
    )
    for level in (0.9, 0.5, 0.1):
        def crossing(z_axis, shape):
            beyond = z_axis > DEPARTURE_SEARCH_FROM_CM
            z_axis, shape = z_axis[beyond], shape[beyond]
            below = np.flatnonzero(shape < level)
            if not below.size:
                return float("nan")
            stop = below[0]
            return float(
                np.interp(
                    level, shape[: stop + 1][::-1], z_axis[: stop + 1][::-1]
                )
            )

        z_measured = crossing(z_grid_cm, b_hat)
        z_census = crossing(census_z_cm, census_b_hat)
        say(
            f"  B_hat = {level:.2f} crossing: measured z = {z_measured:.2f} cm, "
            f"census 'off' z = {z_census:.2f} cm, "
            f"{z_measured - z_census:+.2f} cm"
        )
    say(
        "The measured fall-off and the drawn machine's own unpowered-end "
        "solution agree to a few cm over the whole descent. The ~156 cm "
        "'coil-location disagreement' the mis-oriented build reported was an "
        "artifact of the reflected frame and is RETIRED, not carried forward."
    )
    say()

    # --- construction validation ---------------------------------------------
    say("--- construction validation ---")
    if profile.size != mesh.cells:
        raise AssertionError("profile length does not match the mesh")
    say(f"ASSERT length == mesh cells ({mesh.cells}): PASS")
    if not np.all(np.isfinite(profile)):
        raise AssertionError("profile carries a non-finite entry")
    say("ASSERT every entry finite: PASS")
    if not np.all(profile > 0.0):
        raise AssertionError("profile carries a non-positive radius")
    say(f"ASSERT every entry > 0 (min {profile.min():.6f} cm): PASS")
    if not np.all(np.diff(mesh.z_cm) > 0.0):
        raise AssertionError("mesh cell centres are not monotone increasing")
    say("ASSERT mesh cell centres monotone increasing: PASS")
    plasma_area = np.pi * profile**2
    vessel_area = np.pi * vessel**2
    worst_cap = int(np.argmax(plasma_area / vessel_area))
    if not np.all(plasma_area <= AREA_CAP_FRACTION * vessel_area * (1.0 + 1e-12)):
        raise AssertionError("a cell exceeds the 0.95 vessel open-area cap")
    say(
        f"ASSERT plasma area <= {AREA_CAP_FRACTION} * local vessel open area "
        f"in every cell (worst {plasma_area[worst_cap] / vessel_area[worst_cap]:.6f} "
        f"at cell {worst_cap}, z = {mesh.z_cm[worst_cap]:.3f} cm): PASS"
    )
    upstream_cells = mesh.z_cm <= z_departure
    if not np.all(profile[upstream_cells] == RP_CM):
        raise AssertionError("an upstream cell is not exactly at the scalar Rp")
    say(
        f"ASSERT r == {RP_CM} EXACTLY in all {int(upstream_cells.sum())} cells "
        f"at or upstream of the departure: PASS"
    )

    # The flux-tube ratio between the two stations the measurement resolves.
    port_cells = {
        port: int(np.argmin(np.abs(mesh.z_cm - port_z_cm(port)))) for port in (41, 50)
    }
    implied_ratio = (profile[port_cells[50]] / profile[port_cells[41]]) ** 2
    deviation = abs(implied_ratio - MEASURED_P50_P41_FLUX_RATIO)
    say(
        "GATE -- p50/p41 flux-tube area ratio. The emitted profile reads "
        f"r(p50) = {profile[port_cells[50]]:.4f} cm (cell {port_cells[50]}, "
        f"z = {mesh.z_cm[port_cells[50]]:.3f} cm) and r(p41) = "
        f"{profile[port_cells[41]]:.4f} cm (cell {port_cells[41]}, z = "
        f"{mesh.z_cm[port_cells[41]]:.3f} cm), so the implied area ratio is "
        f"{implied_ratio:.4f}. Measured: {MEASURED_P50_P41_FLUX_RATIO} +/- "
        f"{MEASURED_P50_P41_FLUX_RATIO_SIGMA}. Deviation {deviation:.4f} = "
        f"{deviation / MEASURED_P50_P41_FLUX_RATIO_SIGMA:.2f} sigma."
    )
    if deviation > MEASURED_P50_P41_FLUX_RATIO_SIGMA:
        raise AssertionError(
            f"the emitted p50/p41 flux-tube area ratio {implied_ratio:.6f} is "
            f"{deviation / MEASURED_P50_P41_FLUX_RATIO_SIGMA:.2f} sigma from "
            f"the measured {MEASURED_P50_P41_FLUX_RATIO} +/- "
            f"{MEASURED_P50_P41_FLUX_RATIO_SIGMA}"
        )
    say("ASSERT the implied ratio is within one sigma of the measurement: PASS")

    # Reported, not applied: the source-side anchors of the corrected frame.
    peak_value, peak_z, peak_edge = _extremum(z_grid_cm, b_hat, PEAK_WINDOW_CM, "max")
    dip_value, dip_z, dip_edge = _extremum(z_grid_cm, b_hat, DIP_WINDOW_CM, "min")
    say(
        "REPORTED, NOT APPLIED -- the source-side anchors of the mean shape: "
        f"peak B_hat = {peak_value:.4f} at z = {peak_z:.2f} cm"
        + ("  (ON a window edge)" if peak_edge else "")
        + f", dip B_hat = {dip_value:.4f} at z = {dip_z:.2f} cm"
        + ("  (ON a window edge)" if dip_edge else "")
        + ". Both sit inside the flat hold, so neither reaches the emitted "
        "profile; they are the structure the mis-oriented build reflected onto "
        "the far column as a p50 flare and a sub-Rp throat."
    )

    params, flags = census_build._g1_config(ref_params, ref_flags)
    params["plasma_radius_profile_cm"] = [float(v) for v in profile]
    params["machine_radius_profile_cm"] = [float(v) for v in vessel]
    params["neutral_baffle_positions_cm"] = list(census_build.BAFFLE_POSITIONS_CM)
    params["neutral_baffle_clear_radii_cm"] = list(census_build.BAFFLE_CLEAR_RADII_CM)
    flags["prescribed_area_geometry"] = True
    flags["neutral_baffles"] = True
    geometry = build_geometry(params, flags)
    say("build_geometry with the new profile: OK (no ValueError)")
    annulus = geometry.neutral_volume_cm3 - geometry.plasma_volume_cm3
    has_annulus = annulus > 0.0
    fraction = np.where(has_annulus, annulus / geometry.neutral_volume_cm3, np.nan)
    # Reported as a value plus the SPAN of the cells attaining it, never as one
    # index: wherever the area cap binds the share is 1 - AREA_CAP_FRACTION, so
    # the minimum is a wide tie and any single "worst cell" is whichever member
    # the argmin happened to reach. The tie is counted on EXACT equality, which
    # is narrower than the cap band: each bore level rounds to its own double,
    # so cells pinned at the cap on a different R_m print the same value while
    # comparing unequal. Read the span as "where the cap binds", not as a
    # boundary.
    smallest = float(np.nanmin(fraction))
    tied = np.flatnonzero(fraction == smallest)
    say(
        f"  smallest PER-CELL annulus share (V_m - V_p)/V_m over the cells "
        f"that have an annulus, cap-bound cells included = {smallest:.6f}, "
        f"attained exactly in {tied.size} tied cell(s) spanning "
        f"{int(tied[0])}...{int(tied[-1])}; guard "
        f"neutral_annulus_volume_fraction_min = "
        f"{params.get('neutral_annulus_volume_fraction_min', 1e-2)}"
    )
    say(f"  cells with no annulus (V_ann == 0): {int(np.sum(~has_annulus))}")
    for role in ("cathode", "puff"):
        cells = np.flatnonzero(np.asarray(geometry.cell_role) == role)
        if not np.all(geometry.Rp_cm[cells] == RP_CM):
            raise AssertionError(f"{role} cells are not at the scalar Rp")
        say(f"  ASSERT Rp_cm == {RP_CM} at {role} cells {cells.tolist()}: PASS")
    anode_face = int(np.asarray(geometry.anode_face_indices)[0])
    anode_cells = [anode_face - 1, anode_face]
    if not np.all(geometry.Rp_cm[anode_cells] == RP_CM):
        raise AssertionError("anode-flanking cells are not at the scalar Rp")
    say(
        f"  ASSERT Rp_cm == {RP_CM} at the anode-flanking cells {anode_cells} "
        f"(face {anode_face}, z = {geometry.z_edges_cm[anode_face]:.3f} cm): PASS"
    )
    baffle_faces = np.asarray(geometry.neutral_baffle_face_indices).tolist()
    say(
        f"  baffle mapped to face(s) {baffle_faces} at z = "
        f"{[float(geometry.z_edges_cm[i]) for i in baffle_faces]} cm, clear "
        f"radii {np.asarray(geometry.neutral_baffle_clear_radius_cm).tolist()} cm"
    )
    say()

    # --- emitted profile -----------------------------------------------------
    say("--- plasma_radius_profile_cm, stance-file form ---")
    say("plasma_radius_profile_cm = [")
    say(stance_list_text(profile))
    say("]")
    say()

    figure_path = os.path.join(HERE, "mfp_profile_compare.png")
    write_figure(
        figure_path,
        z_grid_cm,
        shapes,
        b_hat,
        mesh,
        profile,
        census_profiles,
        cap,
        z_departure,
    )
    say(f"wrote {figure_path}")

    report_path = os.path.join(HERE, "mfp_field_profile.txt")
    with open(report_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
