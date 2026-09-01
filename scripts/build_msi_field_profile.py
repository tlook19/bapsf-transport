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
    2.273 cm pitch. Identical in every file (asserted).
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
from each other; the decision is measured, printed with both readings, and
recorded in the report. It is never silently applied.

COORDINATES -- AN ASSUMPTION, NOT A MEASUREMENT
-----------------------------------------------
This build ASSUMES ``z_MSI == z_model``: both are taken as distances from the
cathode along the axis, so an MSI sample at z is applied to the model cell
whose centre is at the same z. The assumption is supported on the measurement
side by the analysis repo's port-to-axis map, whose ``z_from_port = 182.5 +
31.95 * (port - 2)`` is documented as a nominal distance from the cathode, and
on the model side by the solver's own convention (z = 0 at the cathode face).

THE DISAGREEMENT THIS DISCLOSES, AND DOES NOT RESOLVE. Under that assumption
the machine's end-pair mirror sits at a DIFFERENT place than the CAD census
puts it. The MSI record's end-pair peak is at z ~= 1791 cm; the census coil
table's end-pair centroid is at z ~= 1947 cm in model coordinates, and the
census re-solve's own axial field peaks near 1940 cm. That is a ~156 cm
coil-location disagreement between the drawn machine and the machine's
programmed magnet positions. It cannot be adjudicated from the data in hand:
neither record carries an independent axial fiducial that would fix the other.
Both readings are reported. This build takes the MEASURED positions, because
the profile it produces is a statement about the field the ES1 runs were
taken in.

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
consequence, reported rather than hidden: ``B_hat`` is BELOW tolerance over
the source region as well (it falls through the cathode plane and is ~0.91 at
z = 0), so a cell-by-cell tolerance mask would flare the plasma inside the
cathode box and the plenum. The report prints that census.

Beyond the departure the ratio is applied, capped at
``sqrt(AREA_CAP_FRACTION) * R_m(z)`` -- the declared annulus regularization,
the same cap the census build uses. Beyond the last MSI sample (2025.3 cm)
``B_hat`` is HELD at its last sampled value; the report states whether the cap
binds over that span, which decides whether the choice is observable at all.

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

#: The plateau window, in cm. Wide enough to average over the interior coil
#: ripple, and clear of both the source-region rise and the end structure.
PLATEAU_WINDOW_CM = (300.0, 1500.0)
#: Search windows for the reported end features, in cm.
DIP_WINDOW_CM = (1500.0, 1770.0)
PEAK_WINDOW_CM = (1750.0, 1860.0)

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

#: Port axis map of the analysis repo, quoted for the coordinate assumption:
#: nominal distance from the cathode, in cm.
PORT_Z0_CM = 182.5
PORT_PITCH_CM = 31.95
#: The ports the report locates on the profile.
REPORTED_PORTS = (11, 21, 29, 41, 50)


def port_z_cm(port):
    """Return the nominal axial position of ``port``, in cm from the cathode."""
    return PORT_Z0_CM + PORT_PITCH_CM * (port - 2)


# ------------------------------------------------------------------ reading


def msi_files():
    """Return the ES1 raw files, sorted by their leading run number."""
    return sorted(glob.glob(os.path.join(MSI_DATA_DIR, "*.hdf5")))


def read_msi(path):
    """Return one file's MSI magnetic-field record.

    The returned dict carries ``z_cm`` (1024 sample positions), ``B_gauss``
    (2 x 1024, the first and last shot), ``currents_A`` (2 x 13),
    ``peak_gauss`` and ``valid`` (per shot), and ``run`` (the two-digit run
    number the filename starts with).
    """
    with h5py.File(path, "r") as handle:
        group = handle[MSI_GROUP]
        z_cm = np.asarray(group.attrs["Profile z locations"], dtype=float)
        b_gauss = np.asarray(group["Magnetic field profile"], dtype=float)
        currents = np.asarray(group["Magnet power supply currents"], dtype=float)
        summary = np.asarray(group["Magnetic field summary"])
    return {
        "run": os.path.basename(path)[:2],
        "path": path,
        "z_cm": z_cm,
        "B_gauss": b_gauss,
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
    over the full mesh and again over the end region the profile is actually
    built from. A low-current file is an outlier when its deviation exceeds
    the LARGEST deviation any other file shows on the same measure.

    Returns the per-file deviations, the two verdicts, and the set to exclude
    (the end-region measure decides, because that is the region the ratio is
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
    ``z_departure_cm``; ``RP_CM / sqrt(B_hat)`` beyond it, with ``B_hat`` held
    at its last sampled value past the end of the MSI mesh; capped at
    ``sqrt(AREA_CAP_FRACTION) * R_m``.

    Returns ``(capped, raw, cap, b_hat_on_cells, held)`` where ``held`` marks
    the cells beyond the last MSI sample.
    """
    cell_z_cm = np.asarray(cell_z_cm, dtype=float)
    last_sample_cm = float(z_grid_cm[-1])
    held = cell_z_cm > last_sample_cm
    b_on_cells = np.interp(np.minimum(cell_z_cm, last_sample_cm), z_grid_cm, b_hat)
    raw = np.where(
        cell_z_cm <= z_departure_cm, RP_CM, RP_CM / np.sqrt(np.maximum(b_on_cells, 1e-12))
    )
    cap = np.sqrt(AREA_CAP_FRACTION) * np.asarray(vessel_radius_cm, dtype=float)
    return np.minimum(raw, cap), raw, cap, b_on_cells, held


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

    Three shared-axis panels over the whole machine (the normalized field with
    its per-shot band, the radius profiles against the census cases and the
    cap, the new/old ratio) plus a linear zoom on the end region, where the
    log axis of the second panel flattens the droop and the mirror throat into
    a line.
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
            label=f"census {case} (previous stance profile)"
            if case == "droop_min"
            else f"census {case}",
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
    ax.set_xlabel("z [cm], cathode-referenced")

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
    ax.set_xlim(1600.0, 1900.0)
    ax.set_ylim(17.0, 21.0)
    ax.set_ylabel("plasma radius [cm]")
    ax.set_xlabel("z [cm] -- ZOOM on the droop and the mirror throat")
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
        "coordinate: z_MSI == z_model ASSUMED (both cathode-referenced). See "
        "the module docstring for the CAD end-pair disagreement this leaves "
        "open and does not resolve."
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
    say(
        f"ASSERT all {len(records)} files share one z mesh "
        f"({z_grid_cm.size} samples, {z_grid_cm[0]:.3f} .. {z_grid_cm[-1]:.3f} cm, "
        f"pitch {np.diff(z_grid_cm).mean():.4f} cm): PASS"
    )
    invalid = [
        (r["run"], s) for r in records for s in range(2) if not r["valid"][s]
    ]
    say(f"shots flagged 'Data valid' = 0: {invalid if invalid else 'none'}")
    say()

    rows_by_run = {}
    for record in records:
        rows_by_run[record["run"]] = file_features(record)

    say("--- per-file MSI features (both recorded shots) ---")
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
        "feature windows: dip searched over "
        f"{DIP_WINDOW_CM} cm, peak over {PEAK_WINDOW_CM} cm; extrema landing "
        f"ON a window edge (window cutting the feature): "
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
    say()

    # --- the 32/34 adjudication ---------------------------------------------
    verdict = adjudicate_low_current(rows_by_run, z_grid_cm)
    say("--- cross-file normalized-shape spread, and the low-current files ---")
    say(
        "leave-one-out deviation per file: max |B_hat_file - mean(all other "
        "files)|, over the FULL mesh and over the end region "
        f"z >= {DEPARTURE_SEARCH_FROM_CM} cm (the span the ratio is read on)"
    )
    say(f"{'run':>4} {'full mesh':>11} {'z>=1500':>11}")
    for run in verdict["runs"]:
        mark = "  <- low main current" if run in LOW_CURRENT_FILES else ""
        say(
            f"{run:>4} {verdict['deviation_full'][run]:11.4f} "
            f"{verdict['deviation_end'][run]:11.4f}{mark}"
        )
    say(
        f"largest deviation among the OTHER files: full {verdict['others_max_full']:.4f}, "
        f"end region {verdict['others_max_end']:.4f}"
    )
    for run in LOW_CURRENT_FILES:
        say(
            f"  file {run}: full {verdict['deviation_full'][run]:.4f} "
            f"({'BEYOND' if run in verdict['outliers_full'] else 'within'} the others' "
            f"spread), end region {verdict['deviation_end'][run]:.4f} "
            f"({'BEYOND' if run in verdict['outliers_end'] else 'within'} the others' "
            "spread)"
        )
    excluded = verdict["excluded"]
    say(
        "VERDICT: "
        + (
            f"exclude {sorted(excluded)} -- on the end region, the span the flux "
            "ratio is actually read on, their normalized shape departs by more "
            "than any other file's does. On the full mesh they are INSIDE the "
            "others' spread (the plateau ripple of several other files is "
            "larger), so the exclusion rests on the end region alone and is "
            "reported here rather than assumed."
            if excluded
            else "retain every file -- neither low-current file departs beyond "
            "the others' spread on either measure."
        )
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
        -300.0, -100.0, 0.0, 100.0, 300.0, 600.0, 900.0, 1200.0, 1500.0,
        1600.0, 1684.0, 1718.0, 1791.0, 1850.0, 1891.0, 1950.0, 2000.0, 2025.0,
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
    say("ports on the profile (z_from_port = 182.5 + 31.95*(port-2)):")
    for port in REPORTED_PORTS:
        z_port = port_z_cm(port)
        b_port = float(np.interp(min(z_port, z_grid_cm[-1]), z_grid_cm, b_hat))
        flat = z_port <= z_departure
        say(
            f"  p{port:<3d} z = {z_port:8.2f} cm  B_hat = {b_port:.4f}  "
            + (
                "FLAT column (r = Rp exactly)"
                if flat
                else f"FLARED: r/Rp = {1.0 / np.sqrt(b_port):.4f} "
                f"(+{(1.0 / np.sqrt(b_port) - 1.0) * 100:.2f} % radius, "
                f"+{(1.0 / b_port - 1.0) * 100:.2f} % area)"
            )
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
            "  This is the source-region rise: the field falls through and "
            f"behind the cathode plane (B_hat = "
            f"{np.interp(0.0, z_grid_cm, b_hat):.4f} at z = 0, "
            f"{np.interp(-100.0, z_grid_cm, b_hat):.4f} at z = -100). A "
            "cell-by-cell tolerance mask would therefore flare the plasma "
            "inside the cathode box and the plenum and DESYNC the scalar Rp "
            "read sites; the flat span is held to the departure z instead, "
            "which is what the construction validation below gates on."
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
    profile, raw, cap, b_on_cells, held = build_plasma_profile(
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
            + ("   HELD" if held[index] else "")
        )
    say(
        f"cap binds in {binds.size} cell(s): {binds.tolist()}"
        if binds.size
        else "cap binds in 0 cells"
    )
    held_cells = np.flatnonzero(held)
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
    say()

    throat = int(np.argmin(profile))
    widest_flux = int(np.argmax(raw[flared[0] : binds[0] if binds.size else mesh.cells]))
    widest_flux += flared[0]
    say("--- headline geometry of the new profile ---")
    say(
        f"THROAT: min radius {profile[throat]:.4f} cm at z = "
        f"{mesh.z_cm[throat]:.3f} cm (cell {throat}) = "
        f"{profile[throat] / RP_CM:.4f} x Rp, area "
        f"{(profile[throat] / RP_CM) ** 2:.4f} x the column -- the end-pair "
        "mirror compresses the tube BELOW the design column radius"
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
        f"MID-DROOP: at p50 (z = {port_z_cm(50):.2f} cm) the flux tube is "
        f"{1.0 / np.sqrt(float(np.interp(port_z_cm(50), z_grid_cm, b_hat))):.4f} "
        "x Rp -- the ~10 % field droop between the last main coil and the "
        "end-pair mirror sits directly under the p50 measurement station, so "
        "the modelled column there is ~5 % WIDER in radius (~10 % in area) "
        "than the design column, where the census profile held it flat."
    )
    say(
        "PROFILE IS NOT MONOTONE, by construction and by physics: it flares "
        "into the droop, narrows through the mirror throat, flares again "
        "past it, and steps where the vessel bore steps and the cap takes "
        "over. The census profile was not monotone either."
    )
    say()

    # --- comparison ----------------------------------------------------------
    say("--- comparison against the census profiles ---")
    say(
        "census case names are 'droop_min' (end pair energized to the "
        "minimum-droop solution; this was the shipped stance profile) and "
        "'off' (end pair unpowered)"
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
        f"{census_build.FLAT_THROUGH_Z_CM - z_departure:.1f} cm EARLIER than "
        "the census trace did, which is the same "
        "coil-location disagreement the coordinate note discloses, read on "
        "the upstream side."
    )
    say(
        "CAD cross-check (from the census file, unchanged): end-pair coil "
        f"centroid at z = "
        f"{float(np.mean(census['coil_centers_end_pair_m'])) * 100:.2f} cm in "
        "model coordinates. The MSI end-pair peak of the mean profile is at "
        f"z = "
        f"{z_grid_cm[np.argmax(np.where((z_grid_cm > PEAK_WINDOW_CM[0]) & (z_grid_cm < PEAK_WINDOW_CM[1]), b_hat, -np.inf))]:.2f}"
        " cm. DISCLOSED, NOT RESOLVED: no fiducial in either record fixes the "
        "other."
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
    worst = int(np.nanargmin(fraction))
    say(
        f"  min V_ann/V_neutral over annulus cells = {fraction[worst]:.6f} at "
        f"cell {worst} (z = {geometry.z_cm[worst]:.3f} cm, role "
        f"{geometry.cell_role[worst]}); guard "
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
