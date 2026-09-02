"""Build the G1a/G1b prescribed per-cell geometry profiles for LAPDSim1D.

WHAT THIS SCRIPT STILL OWNS, SINCE 2026-09-01. The vessel profile
``machine_radius_profile_cm`` is this script's, and the stance ships it
unchanged. The SHIPPED ``plasma_radius_profile_cm`` is no longer: it now comes
from ``scripts/build_msi_field_profile.py``, which builds the flux tube from
the measured MSI machine-state field record instead of from the CAD census
re-solve below. The two plasma profiles this script emits are retained as the
INDEPENDENT CROSS-CHECK against that build -- the MSI builder reuses this
script's mesh probe and vessel profile verbatim and compares its result against
both cases here. The comparison is reported in ``mfp_field_profile.txt``.

WHICH CASE IS THE MACHINE'S. The end-pair supply channel reads 0 A in every
recorded ES1 shot, so ``off`` -- not ``droop_min`` -- is the case the machine
was actually in, and under the corrected MSI registration (2026-09-01) the
measured fall-off tracks this script's ``off`` axial field to a few cm all the
way down. The ~156 cm "coil-location disagreement" reported against
``droop_min`` earlier that day was an artifact of a mirror-imaged coordinate
assumption in the MSI build and is retired; see
``scripts/production_stance_provenance.md``.

G1 adopts the measured CAD machine geometry at the l2a7b operating point.
The solver takes the geometry as RADII, one entry per mesh cell, under the
``prescribed_area_geometry`` flag: ``plasma_radius_profile_cm`` (the flux-tube
radius, so the flux-tube AREA is ``pi r^2``) and ``machine_radius_profile_cm``
(the vessel bore). This script emits both, for the two end-field cases the
census re-solve resolved, plus the mesh comparison against the l2a7b
reference and the construction validation that must pass before any arm
launches. The annulus figure that validation prints is the smallest PER-CELL
share ``(V_m - V_p)/V_m`` over the cells that have an annulus, cap-bound cells
included and ``V_ann == 0`` cells excluded -- the same per-cell quantity the
solver's ``neutral_annulus_volume_fraction_min`` guard refuses on, NOT a
column-integrated share.

Inputs
------
``scripts/lapd_end_field_1400G_rp18p415_census2026.npz``
    The measured-census field re-solve. Per case ``c`` in
    ``{droop_min, off}`` it carries ``c_z_flux_m`` / ``c_flux_radius_m``
    (the traced flux-surface radius of the 18.415 cm column), the trace
    terminus ``c_trace_end_z_m``, and the vessel-wall crossings
    ``c_crossing_z_m`` / ``c_crossing_radii_m``.
``scripts/l2a7b_foot45_cr6p94.h5``
    The l2a7b operating point, read for its resolved config only. It is the
    comparison base for the mesh report, NOT an identity target: it was one
    until the 2026-08-24 CAD-span gap adoption moved
    ``cathode_anode_gap_cm`` 50.0 -> 53.25, which moves the anode face and
    with it every cell downstream of the cathode face. The grid of record
    (2026-08-18rrr) is ``Lm = 2117.8, collector_length_cm = 7.8, nx = 268``,
    and the terminal cell is the 7.8 cm collector at the flange. BEHIND the
    cathode face the mesh deliberately changes (the 2026-08-18sss fidelity
    package): the
    guessed ``Rcs 40 / Lcs 25`` obstruction is RETIRED (the obstruction cell
    is omitted at ``Lcs = 0``) and the plenum is the measured source chamber,
    ``plenum_length_cm = 166`` at machine radius 40 cm (reservoir volume
    8.34e5 cm^3 exactly). The measured cathode box enters solely as the
    machine-radius stages at the cathode/gap cells (annulus areas
    1350.1 / 1847.6 cm^2 exact) -- Tom's abstracted faithful conductance.

Plasma profile (per case)
-------------------------
Flat at exactly ``Rp`` = 18.415 cm for every cell centre at z <= 1855 cm --
through every port, so the scalar-read sites (cathode.py's ``Rp``,
``_anode_neutral_transparency``) stay in sync with the vector. Past 1855 cm
the flux-RATIO scaling ``Rp * r_flux(z) / r_flux(18.55 m)`` carries the
measured flare; taking the ratio removes the traced surface's anchored-sag
offset, so the profile is continuous with the flat column by construction.
The result is capped at ``sqrt(0.95) * R_m(z)`` -- the declared B2 annulus
regularization, which is also what clips the surface at the vessel wall --
and the trace is never read past ``c_trace_end_z_m``.

Vessel profile
--------------
Measured bore: 40.0 cm source chamber, 50.0 cm main shell to the 19.65 m
step, 76.2 cm far source chamber beyond it. Over the cathode box the bore is
replaced by the annulus-area-equivalent radius that reproduces the measured
clear area around the box (1350.1 cm^2 at the cathode cell, 1847.6 cm^2 at
the first gap cell) against the 18.415 cm column. The plenum cell sits at
negative z inside the first bore stage, so it carries the measured source
chamber's 40.0 cm directly.

Outputs (all in ``scripts/``)
-----------------------------
``g1_profiles.npz``  the two plasma profiles, the vessel profile, the mesh.
``g1_profiles.txt``  the human-readable tables and validation report.
``g1a_extra_args.txt`` / ``g1b_extra_args.txt``  the literal
    ``--extra key=value`` JSON the arm commands paste in.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402

CENSUS_NPZ = os.path.join(HERE, "lapd_end_field_1400G_rp18p415_census2026.npz")
REFERENCE_H5 = os.path.join(HERE, "l2a7b_foot45_cr6p94.h5")

#: The G1 grid of record (2026-08-18rrr ruling, supersedes the qqq
#: collector-217.8 draft): the end flange is the wall, the outer column runs
#: at a uniform dz through z = 2110 cm, and the terminal cell is the 7.8 cm
#: collector at the flange where the 0.95 cap binds flat. That far-column dz
#: is (Lm - gap - collector - source span)/nx, so the CAD-span gap adoption
#: moved it 7.5 -> 7.487873... cm; the build report carries the measured
#: value.
LM_CM = 2117.8
COLLECTOR_LENGTH_CM = 7.8
NX = 268
#: The 2026-08-18sss fidelity package: the guessed Rcs 40 / Lcs 25 cathode
#: box is RETIRED (Lcs = 0 omits the obstruction cell); the plenum is the
#: measured CAD source chamber (166 cm at bore 40 -> 8.34e5 cm^3 exactly).
RCS_CM = 0.0
LCS_CM = 0.0
PLENUM_LENGTH_CM = 166.0
#: Measured mid-plane puff ports at the anode stack (supersedes 60.0).
GAS_PUFF_Z_CM = 86.3
#: The cathode-anode gap, CAD-span midpoint of 0.531-0.534 m (supersedes the
#: 50.0 of record). The anode is a mesh FACE at this z, so the nx_gap = 5 gap
#: cells stretch to 10.65 cm and every cell downstream shifts with it.
CATHODE_ANODE_GAP_CM = 53.25
#: The fixed source region rides the anode face: its SPAN is unchanged (50 cm
#: = 5 x source_region_dz_cm), so its far end shifts by the same +3.25 cm and
#: the cell count is unchanged.
SOURCE_REGION_LENGTH_CM = 103.25
#: The port-7 annular ring (`TomLook-Aperature`), z 3.401-3.452 m; the
#: CAD-span midpoint is 342.65 cm (the 342.6 of record was that same midpoint
#: stale-rounded).
BAFFLE_POSITIONS_CM = [342.65]
BAFFLE_CLEAR_RADII_CM = [39.75]

#: The plasma column radius (design spec of record; [18.10, 18.415] bracket
#: carried analytically on aperture-sensitive claims).
RP_CM = 18.415
#: Flat-column span: the profile is exactly RP_CM for every cell centre at or
#: below this z, and the flux-ratio reference is taken at its far edge.
FLAT_THROUGH_Z_CM = 1855.0
FLUX_REFERENCE_Z_M = 18.55
#: B2 area cap: plasma area <= 0.95 * local vessel open area.
AREA_CAP_FRACTION = 0.95

#: Measured vessel bore, as (z_upper_cm, radius_cm) stages, applied by cell
#: centre. The last stage is the far source chamber (r 762 mm).
BORE_STAGES_CM = ((100.0, 40.0), (1965.0, 50.0), (np.inf, 76.2))
#: Cathode-box conductance encoding (sss): measured clear areas around the
#: box, expressed as the vessel radius whose ANNULUS about the column
#: reproduces them, at the cathode cell (front plate) and the first gap cell
#: (box body). Applied by cell centre over (z_low, z_high] downstream of
#: z = 0 -- the ranges land exactly on those two 10 cm cells.
BOX_STAGES = (
    ("front plate / cathode cell", 0.0, 10.0, 1350.1),
    ("box body / gap cell", 10.0, 20.0, 1847.6),
)
#: The plenum reservoir volume the measured source chamber must reproduce.
PLENUM_VOLUME_CM3 = np.pi * 40.0**2 * PLENUM_LENGTH_CM

CASES = ("droop_min", "off")


def _reference_config():
    """Return the l2a7b (params, flags) pair, read from the archived run."""
    import h5py

    with h5py.File(REFERENCE_H5, "r") as handle:
        params = json.loads(handle.attrs["params_json"])
        flags = json.loads(handle.attrs["flags_json"])
    return params, flags


def _g1_config(params, flags):
    """Return the G1 (params, flags) pair carrying geometry deltas only.

    The profile arrays are NOT filled in here: this is the mesh probe, built
    with ``prescribed_area_geometry`` off, whose only job is to resolve the
    cell centres the profiles are then evaluated on.
    """
    p = dict(params)
    f = dict(flags)
    p["Lm"] = LM_CM
    p["collector_length_cm"] = COLLECTOR_LENGTH_CM
    p["nx"] = NX
    p["gas_puff_z_cm"] = GAS_PUFF_Z_CM
    p["cathode_anode_gap_cm"] = CATHODE_ANODE_GAP_CM
    p["source_region_length_cm"] = SOURCE_REGION_LENGTH_CM
    # The sss fidelity package: guessed cathode box retired, measured plenum.
    p["Rcs"] = RCS_CM
    p["Lcs"] = LCS_CM
    p["plenum_length_cm"] = PLENUM_LENGTH_CM
    # The prescribed profile REPLACES the built-in half-cosine end flare; the
    # two refuse to compose, and the stale parameters refuse the off flag.
    p["end_expansion_cells"] = None
    p["end_expansion_machine_radius_cm"] = None
    p["end_expansion_plasma_radius_cm"] = None
    f["end_expansion_geometry"] = False
    return p, f


def build_vessel_profile(z_cm, cell_role):
    """Return ``machine_radius_profile_cm``: the measured bore, per cell.

    The plenum cell (negative z) falls inside the first bore stage, so the
    measured source chamber's 40.0 cm applies there with no special case --
    that is the sss encoding, giving the measured reservoir volume exactly.
    """
    z_cm = np.asarray(z_cm, dtype=float)
    radius = np.empty(z_cm.size, dtype=float)
    for index, z in enumerate(z_cm):
        for upper, value in BORE_STAGES_CM:
            if z <= upper:
                radius[index] = value
                break
    for _, z_low, z_high, clear_area in BOX_STAGES:
        inside = (z_cm > z_low) & (z_cm <= z_high)
        radius[inside] = np.sqrt(RP_CM**2 + clear_area / np.pi)
    return radius


def build_plasma_profile(case, z_cm, vessel_radius_cm, census):
    """Return ``plasma_radius_profile_cm`` for one end-field case.

    Flat at ``RP_CM`` through ``FLAT_THROUGH_Z_CM``; past it the measured
    flux-radius RATIO against the 18.55 m reference, capped at
    ``sqrt(AREA_CAP_FRACTION) * R_m(z)``.
    """
    z_cm = np.asarray(z_cm, dtype=float)
    z_flux_m = np.asarray(census[f"{case}_z_flux_m"], dtype=float)
    r_flux_m = np.asarray(census[f"{case}_flux_radius_m"], dtype=float)
    trace_end_m = float(census[f"{case}_trace_end_z_m"])

    def flux_radius_m(z_m):
        """Interpolate on the case's OWN grid, never past its trace end."""
        z_m = np.minimum(np.asarray(z_m, dtype=float), trace_end_m)
        return np.interp(z_m, z_flux_m, r_flux_m)

    reference_m = float(flux_radius_m(FLUX_REFERENCE_Z_M))
    radius = np.full(z_cm.size, RP_CM, dtype=float)
    flared = z_cm > FLAT_THROUGH_Z_CM
    radius[flared] = RP_CM * flux_radius_m(z_cm[flared] / 100.0) / reference_m

    cap = np.sqrt(AREA_CAP_FRACTION) * np.asarray(vessel_radius_cm, dtype=float)
    capped = np.minimum(radius, cap)
    return capped, radius, cap, reference_m, trace_end_m


def _fmt_array(values):
    """Compact JSON for a --extra value: full float repr, no spaces."""
    return json.dumps([float(v) for v in values], separators=(",", ":"))


def main():
    census = np.load(CENSUS_NPZ, allow_pickle=True)
    ref_params, ref_flags = _reference_config()
    ref_geometry = build_geometry(ref_params, ref_flags)

    mesh_params, mesh_flags = _g1_config(ref_params, ref_flags)
    mesh = build_geometry(mesh_params, mesh_flags)

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    say("=== G1 prescribed-geometry profile build ===")
    say(f"census   : {CENSUS_NPZ}")
    say(f"reference: {REFERENCE_H5}")
    say(
        f"grid     : Lm {ref_params['Lm']} -> {LM_CM} cm, collector "
        f"{ref_params['collector_length_cm']} -> {COLLECTOR_LENGTH_CM} cm, "
        f"nx {ref_params['nx']} -> {NX} (the rrr grid of record: the outer "
        f"column extends at its own dz through 2110 cm; terminal cell "
        f"{COLLECTOR_LENGTH_CM} cm at the flange)"
    )
    say(
        f"source   : Rcs {ref_params['Rcs']} -> {RCS_CM}, Lcs "
        f"{ref_params['Lcs']} -> {LCS_CM}, plenum_length_cm "
        f"{ref_params['plenum_length_cm']} -> {PLENUM_LENGTH_CM} "
        f"(the sss fidelity package)"
    )
    say()

    # --- mesh comparison against l2a7b ---------------------------------------
    # THE BIT-IDENTITY CLAIM IS RETIRED (2026-08-24 CAD-span gap adoption).
    # Until the gap moved, the G1 mesh reproduced l2a7b's cell centres and
    # edges bit-for-bit over 0 <= z <= 1900 cm, and this block ASSERTED it.
    # ``cathode_anode_gap_cm`` 50.0 -> 53.25 moves the anode face, so every
    # cell downstream of the cathode face moves with it and no such identity
    # can hold: the gap cells stretch 10.0 -> 10.65 cm, the fixed source
    # region rides the face (+3.25 cm, span and cell count unchanged), and the
    # far column re-divides the shortened remainder. The comparison is kept
    # and REPORTED so the size and shape of the shift are on the record, but
    # asserting the old identity would now be asserting something the ruling
    # deliberately broke.
    shared = np.flatnonzero((mesh.z_cm >= 0.0) & (mesh.z_cm <= 1900.0))
    ref_shared = np.flatnonzero(
        (ref_geometry.z_cm >= 0.0) & (ref_geometry.z_cm <= 1900.0)
    )
    say("--- mesh comparison vs l2a7b (cells 0 <= z <= 1900 cm) ---")
    say(f"reference cells {ref_geometry.cells}, G1 cells {mesh.cells}")
    ref_behind = np.flatnonzero(ref_geometry.z_cm < 0.0)
    g1_behind = np.flatnonzero(mesh.z_cm < 0.0)
    say(
        f"behind-cathode cells (deliberate sss change): reference "
        f"{[str(r) for r in np.asarray(ref_geometry.cell_role)[ref_behind]]} "
        f"lengths {ref_geometry.length_cm[ref_behind].tolist()} cm -> G1 "
        f"{[str(r) for r in np.asarray(mesh.cell_role)[g1_behind]]} "
        f"lengths {mesh.length_cm[g1_behind].tolist()} cm"
    )
    if [str(r) for r in np.asarray(mesh.cell_role)[g1_behind]] != ["plenum"]:
        raise AssertionError(
            "G1 behind-cathode cells must be the plenum alone (Lcs = 0 "
            "omits the obstruction cell)"
        )
    say(f"windowed cell count: reference {ref_shared.size}, G1 {shared.size}")
    say(
        f"anode face z: reference {ref_params['cathode_anode_gap_cm']} cm -> "
        f"G1 {CATHODE_ANODE_GAP_CM} cm; gap cell dz "
        f"{ref_params['cathode_anode_gap_cm'] / mesh_params['nx_gap']:.6g} -> "
        f"{mesh.length_cm[1]:.6g} cm over nx_gap = {mesh_params['nx_gap']}"
    )
    say(
        f"source region: reference [{ref_params['cathode_anode_gap_cm']}, "
        f"{ref_params['source_region_length_cm']}) -> G1 "
        f"[{CATHODE_ANODE_GAP_CM}, {SOURCE_REGION_LENGTH_CM}) cm; span "
        f"{ref_params['source_region_length_cm'] - ref_params['cathode_anode_gap_cm']:.6g}"
        f" -> {SOURCE_REGION_LENGTH_CM - CATHODE_ANODE_GAP_CM:.6g} cm at "
        f"source_region_dz_cm = {mesh_params['source_region_dz_cm']} cm "
        f"(cell count unchanged)"
    )
    # The shift, reported rather than asserted away. Over the windowed cells
    # the centres move by the gap delta inside the fixed-size source region
    # and by a growing amount down the far column, whose dz re-divides the
    # shortened remainder.
    if ref_shared.size == shared.size:
        centre_delta = mesh.z_cm[shared] - ref_geometry.z_cm[ref_shared]
        say(
            f"cell-centre shift over the window: min {centre_delta.min():+.6g} "
            f"cm, max {centre_delta.max():+.6g} cm, "
            f"{int(np.sum(centre_delta != 0.0))} of {shared.size} cell(s) moved"
        )
        length_delta = mesh.length_cm[shared] - ref_geometry.length_cm[ref_shared]
        say(
            f"cell-length delta over the window: min {length_delta.min():+.6g} "
            f"cm, max {length_delta.max():+.6g} cm, "
            f"{int(np.sum(length_delta != 0.0))} of {shared.size} cell(s) changed"
        )
    else:
        say(
            "windowed cell counts differ, so no per-cell delta is reported "
            "(the 1900 cm window is a fixed z cut, not a cell cut)"
        )
    ref_roles = np.asarray(ref_geometry.cell_role[ref_shared], dtype=object)
    g1_roles = np.asarray(mesh.cell_role[shared], dtype=object)
    role_moved = np.flatnonzero(ref_roles != g1_roles)
    say(
        f"role changes over shared cells: {role_moved.tolist()} "
        f"({[str(ref_roles[i]) for i in role_moved]} -> "
        f"{[str(g1_roles[i]) for i in role_moved]}) -- the intended "
        f"gas_puff_z_cm {ref_params['gas_puff_z_cm']} -> {GAS_PUFF_Z_CM} move"
    )
    moved_roles = {str(ref_roles[i]) for i in role_moved} | {
        str(g1_roles[i]) for i in role_moved
    }
    if not moved_roles <= {"puff", "column"}:
        raise AssertionError(f"unexpected role changes: {sorted(moved_roles)}")
    say("ASSERT role changes confined to the puff/column swap: PASS")
    end = np.flatnonzero(
        np.isin(np.asarray(mesh.cell_role), np.asarray(["end", "collector"], dtype=object))
    )
    say(
        f"n_end = {end.size}; end dz = "
        f"{', '.join(f'{v:.6g}' for v in mesh.length_cm[end])} cm; "
        f"end z-centres = {', '.join(f'{v:.6g}' for v in mesh.z_cm[end])} cm"
    )
    say(
        f"column dz (far column) = {mesh.length_cm[mesh.cells - end.size - 1]:.6g} cm; "
        f"last column cell centre = {mesh.z_cm[mesh.cells - end.size - 1]:.6g} cm"
    )
    say()

    # --- vessel profile -----------------------------------------------------
    vessel = build_vessel_profile(mesh.z_cm, mesh.cell_role)
    say("--- machine_radius_profile_cm (measured bore + box encoding) ---")
    edges = np.flatnonzero(np.diff(vessel) != 0.0)
    say("stage table (cell index range -> radius):")
    start = 0
    for boundary in list(edges) + [mesh.cells - 1]:
        stop = int(boundary)
        say(
            f"  cells {start:3d}-{stop:3d}  z {mesh.z_edges_cm[start]:9.3f} .. "
            f"{mesh.z_edges_cm[stop + 1]:9.3f} cm   R_m = {vessel[start]:.6f} cm"
            f"   [{mesh.cell_role[start]}]"
        )
        start = stop + 1
    for label, z_low, z_high, clear_area in BOX_STAGES:
        inside = np.flatnonzero((mesh.z_cm > z_low) & (mesh.z_cm <= z_high))
        for index in inside:
            annulus = np.pi * (vessel[index] ** 2 - RP_CM**2)
            say(
                f"  box stage '{label}': cell {index} (z {mesh.z_cm[index]:.4g} cm) "
                f"R_m = {vessel[index]:.6f} cm, annulus area = {annulus:.4f} cm^2 "
                f"(target {clear_area})"
            )
    plenum_cells = np.flatnonzero(np.asarray(mesh.cell_role) == "plenum")
    plenum_volume = float(
        np.sum(np.pi * vessel[plenum_cells] ** 2 * mesh.length_cm[plenum_cells])
    )
    say(
        f"plenum reservoir volume = {plenum_volume:.6e} cm^3 "
        f"(measured source chamber {PLENUM_VOLUME_CM3:.6e} cm^3)"
    )
    if plenum_volume != PLENUM_VOLUME_CM3:
        raise AssertionError(
            "plenum reservoir volume does not reproduce the measured source chamber"
        )
    say("ASSERT plenum reservoir volume == measured 8.34e5 cm^3: PASS")
    say()

    # --- plasma profiles ----------------------------------------------------
    profiles = {}
    say("--- plasma_radius_profile_cm, per end-field case ---")
    for case in CASES:
        capped, raw, cap, reference_m, trace_end_m = build_plasma_profile(
            case, mesh.z_cm, vessel, census
        )
        profiles[case] = capped
        binds = np.flatnonzero(capped < raw)
        say(f"[{case}]")
        say(
            f"  r_flux({FLUX_REFERENCE_Z_M} m) = {reference_m:.6f} m; "
            f"trace end = {trace_end_m:.6f} m; "
            f"crossings z = {np.asarray(census[case + '_crossing_z_m'])} m "
            f"at radii {np.asarray(census[case + '_crossing_radii_m'])} m"
        )
        say(f"  flat cells (r == {RP_CM}): {int(np.sum(capped == RP_CM))} of {mesh.cells}")
        flared = np.flatnonzero(mesh.z_cm > FLAT_THROUGH_Z_CM)
        say("  flared cells:")
        for index in flared:
            area = np.pi * capped[index] ** 2
            say(
                f"    cell {index:3d}  z = {mesh.z_cm[index]:9.3f} cm  "
                f"r_raw = {raw[index]:9.4f}  cap = {cap[index]:9.4f}  "
                f"r = {capped[index]:9.4f} cm  A = {area:12.4f} cm^2  "
                f"A/A_col = {area / (np.pi * RP_CM**2):7.3f}"
                + ("   <- CAP BINDS" if capped[index] < raw[index] else "")
            )
        if binds.size:
            say(
                f"  0.95 cap binds in {binds.size} cell(s): indices "
                f"{binds.tolist()} (z {mesh.z_cm[binds[0]]:.3f} .. "
                f"{mesh.z_cm[binds[-1]]:.3f} cm)"
            )
        else:
            say("  0.95 cap binds in 0 cells")
        terminal = mesh.cells - 1
        previous = terminal - 1
        area_terminal = np.pi * capped[terminal] ** 2
        area_previous = np.pi * capped[previous] ** 2
        say(
            f"  TERMINATING CELL: index {terminal}, z = {mesh.z_cm[terminal]:.3f} cm, "
            f"r = {capped[terminal]:.6f} cm, A = {area_terminal:.4f} cm^2"
        )
        say(
            f"  dA across the terminating cell (A[{terminal}] - A[{previous}]) = "
            f"{area_terminal - area_previous:.4f} cm^2 "
            f"(rel {(area_terminal - area_previous) / area_terminal:+.6f})"
        )
        # Sub-cell diagnostic, report-only: the terminal cell is the 7.8 cm
        # collector at the flange (the rrr grid of record), so a centre
        # sample and a volume average of the same profile should now nearly
        # agree. Both are stated; the arms carry the centre sample.
        fine_z_cm = np.linspace(
            mesh.z_edges_cm[terminal], mesh.z_edges_cm[terminal + 1], 4001
        )
        fine_raw = RP_CM * (
            np.interp(
                np.minimum(fine_z_cm / 100.0, trace_end_m),
                np.asarray(census[f"{case}_z_flux_m"], dtype=float),
                np.asarray(census[f"{case}_flux_radius_m"], dtype=float),
            )
            / reference_m
        )
        fine_cap = np.sqrt(AREA_CAP_FRACTION) * vessel[terminal]
        fine_r = np.minimum(fine_raw, fine_cap)
        fine_area = np.pi * fine_r**2
        binding = np.flatnonzero(fine_raw > fine_cap)
        say(
            f"  [report-only] volume-averaged area over the terminal cell = "
            f"{np.trapezoid(fine_area, fine_z_cm) / (fine_z_cm[-1] - fine_z_cm[0]):.4f} "
            f"cm^2 = {np.trapezoid(fine_area, fine_z_cm) / (fine_z_cm[-1] - fine_z_cm[0]) / (np.pi * RP_CM**2):.3f}"
            f" x column (centre sample gives {area_terminal / (np.pi * RP_CM**2):.3f} x)"
        )
        say(
            "  [report-only] the 0.95 cap would start binding at z = "
            + (
                f"{fine_z_cm[binding[0]]:.2f} cm"
                if binding.size
                else "nowhere inside the terminal cell"
            )
            + f"; area at the flange (z = {mesh.z_edges_cm[terminal + 1]:.1f} cm) = "
            f"{fine_area[-1]:.4f} cm^2 = {fine_area[-1] / (np.pi * RP_CM**2):.3f} x column"
        )
        say()

    # --- construction validation -------------------------------------------
    say("--- construction validation (build_geometry with the profiles) ---")
    arm_configs = {}
    for case, arm in (("droop_min", "G1a"), ("off", "G1b")):
        params, flags = _g1_config(ref_params, ref_flags)
        params["plasma_radius_profile_cm"] = [float(v) for v in profiles[case]]
        params["machine_radius_profile_cm"] = [float(v) for v in vessel]
        params["neutral_baffle_positions_cm"] = list(BAFFLE_POSITIONS_CM)
        params["neutral_baffle_clear_radii_cm"] = list(BAFFLE_CLEAR_RADII_CM)
        flags["prescribed_area_geometry"] = True
        flags["neutral_baffles"] = True
        arm_configs[arm] = (params, flags)
        geometry = build_geometry(params, flags)
        say(f"[{arm} / {case}] build_geometry: OK (no ValueError)")

        annulus = geometry.neutral_volume_cm3 - geometry.plasma_volume_cm3
        has_annulus = annulus > 0.0
        fraction = np.where(
            has_annulus, annulus / geometry.neutral_volume_cm3, np.nan
        )
        # Value plus the CAP SET, never one index. In a cap-bound cell the
        # share is 1 - AREA_CAP_FRACTION only up to the per-cell rounding of
        # the dz-dependent volumes this ratio is built from, and dz varies
        # across the mesh, so those cells do NOT all carry one double: the
        # minimum is whichever rounded lowest and locates nothing. The cap set
        # comes from the geometry's own radii against the cap rule, not from
        # comparing shares to each other.
        smallest = float(np.nanmin(fraction))
        cap_bound = np.flatnonzero(
            geometry.Rp_cm
            >= np.sqrt(AREA_CAP_FRACTION) * geometry.Rm_cm - 1e-9
        )
        if cap_bound.size:
            runs = np.split(
                cap_bound, np.flatnonzero(np.diff(cap_bound) != 1) + 1
            )
            cap_ulp = int(np.abs(
                fraction[cap_bound].view(np.int64)
                - np.float64(smallest).view(np.int64)
            ).max())
            cap_note = (
                f"the area cap binds in {cap_bound.size} cells ("
                + ", ".join(
                    f"{int(r[0])}-{int(r[-1])}" if r.size > 1
                    else f"{int(r[0])}"
                    for r in runs
                )
                + f"), whose shares all agree with this value to within "
                f"{cap_ulp} ULP"
            )
        else:
            cap_note = "the area cap binds in no cell"
        say(
            f"  smallest PER-CELL annulus share (V_m - V_p)/V_m over the "
            f"cells that have an annulus, cap-bound cells included = "
            f"{smallest:.6f}; {cap_note}; "
            f"guard neutral_annulus_volume_fraction_min = "
            f"{params.get('neutral_annulus_volume_fraction_min', 1e-2)}"
        )
        say(
            f"  cells with no annulus (V_ann == 0): "
            f"{int(np.sum(~has_annulus))}"
        )
        terminal_face = geometry.plasma_face_area_cm2[-1]
        column_face = np.pi * RP_CM**2
        say(
            f"  terminal plasma face area = {terminal_face:.4f} cm^2 = "
            f"{terminal_face / column_face:.4f} x the column face "
            f"({column_face:.4f} cm^2)"
        )
        # Scalar-read desync guard: the vector must agree with the scalar Rp
        # everywhere a scalar-Rp site reads the geometry.
        guard_roles = ("cathode", "puff")
        for role in guard_roles:
            cells = np.flatnonzero(np.asarray(geometry.cell_role) == role)
            values = geometry.Rp_cm[cells]
            if not np.all(values == RP_CM):
                raise AssertionError(f"{role} cells are not at the scalar Rp")
            say(
                f"  ASSERT Rp_cm == {RP_CM} at {role} cells "
                f"{cells.tolist()}: PASS"
            )
        anode_face = int(np.asarray(geometry.anode_face_indices)[0])
        anode_cells = [anode_face - 1, anode_face]
        if not np.all(geometry.Rp_cm[anode_cells] == RP_CM):
            raise AssertionError("anode-flanking cells are not at the scalar Rp")
        say(
            f"  ASSERT Rp_cm == {RP_CM} at the anode-flanking cells "
            f"{anode_cells} (face {anode_face}, z = "
            f"{geometry.z_edges_cm[anode_face]:.3f} cm): PASS"
        )
        baffle_faces = np.asarray(geometry.neutral_baffle_face_indices).tolist()
        say(
            f"  baffle mapped to face(s) {baffle_faces} at z = "
            f"{[float(geometry.z_edges_cm[i]) for i in baffle_faces]} cm, "
            f"clear radii {np.asarray(geometry.neutral_baffle_clear_radius_cm).tolist()} cm"
        )
        say()

    # --- emitted arm arguments ---------------------------------------------
    say("--- --extra payloads ---")
    for arm, case in (("G1a", "droop_min"), ("G1b", "off")):
        payload = [
            f"Lm={LM_CM}",
            f"collector_length_cm={COLLECTOR_LENGTH_CM}",
            f"nx={NX}",
            f"gas_puff_z_cm={GAS_PUFF_Z_CM}",
            f"Rcs={RCS_CM}",
            f"Lcs={LCS_CM}",
            f"plenum_length_cm={PLENUM_LENGTH_CM}",
            "end_expansion_cells=null",
            "end_expansion_machine_radius_cm=null",
            "end_expansion_plasma_radius_cm=null",
            f"neutral_baffle_positions_cm={_fmt_array(BAFFLE_POSITIONS_CM)}",
            f"neutral_baffle_clear_radii_cm={_fmt_array(BAFFLE_CLEAR_RADII_CM)}",
            f"plasma_radius_profile_cm={_fmt_array(profiles[case])}",
            f"machine_radius_profile_cm={_fmt_array(vessel)}",
        ]
        flag_payload = [
            "prescribed_area_geometry=true",
            "neutral_baffles=true",
            "end_expansion_geometry=false",
        ]
        path = os.path.join(HERE, f"{arm.lower()}_extra_args.txt")
        with open(path, "w") as handle:
            handle.write("--extra " + " ".join(payload) + "\n\n")
            handle.write("--extra-flag " + " ".join(flag_payload) + "\n")
        say(f"{arm}: wrote {path} ({os.path.getsize(path)} bytes)")
        say(f"  --extra-flag {' '.join(flag_payload)}")
        say(f"  --extra {' '.join(payload[:8])} <profile arrays in the file>")
    say()

    npz_path = os.path.join(HERE, "g1_profiles.npz")
    np.savez(
        npz_path,
        z_cm=mesh.z_cm,
        z_edges_cm=mesh.z_edges_cm,
        length_cm=mesh.length_cm,
        cell_role=np.asarray([str(r) for r in mesh.cell_role]),
        machine_radius_profile_cm=vessel,
        plasma_radius_profile_cm_droop_min=profiles["droop_min"],
        plasma_radius_profile_cm_off=profiles["off"],
        Lm_cm=LM_CM,
        collector_length_cm=COLLECTOR_LENGTH_CM,
        nx=NX,
        gas_puff_z_cm=GAS_PUFF_Z_CM,
        Rcs_cm=RCS_CM,
        Lcs_cm=LCS_CM,
        plenum_length_cm=PLENUM_LENGTH_CM,
        area_cap_fraction=AREA_CAP_FRACTION,
        flat_through_z_cm=FLAT_THROUGH_Z_CM,
        flux_reference_z_m=FLUX_REFERENCE_Z_M,
    )
    say(f"saved {npz_path}")

    with open(os.path.join(HERE, "g1_profiles.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
