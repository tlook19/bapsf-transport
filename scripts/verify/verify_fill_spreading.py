"""Verify the initial-fill builder's spreading members against their stated properties.

``scripts/stance/sp3_build_nn0.py`` offers three spreading members: two MATRIX
kernels (``diffusive``, ``ballistic``), whose targets are weighted by cell
LENGTH and which are applied to an inventory deposited whole at the start of
the foot, and ``knudsen``, a conservative finite-volume axial diffusion solve
on the builder's own mesh and zone volumes, weighted by cell VOLUME, with the
source spread over the foot. This script is the acceptance instrument for the
finite-volume member and for the property that separates the two families.

Each gate names the instrument that decides it; every gate prints its own
numbers, and the script exits non-zero if any selected gate fails.

DEFAULT RUN (no arguments) -- everything decidable inside this repository:

``G1 uniform-source control``
    A source uniform per unit VOLUME over the active set, on a bore-step tube
    and on the production configuration's own mesh. The finite-volume member's
    added density must be uniform to ``UNIFORM_SOURCE_REL_TOL``: a uniform
    density is in the operator's null space, so a source that is uniform per
    unit volume can produce no gradient, and in particular none at a bore
    step. The LEGACY length-weighted route is the NEGATIVE CONTROL on the same
    source: its added density goes as 1/A, so it steps by the area ratio --
    on the tube, exactly ``LEGACY_STEP_AREA_RATIO``. Both are printed. A gate
    whose negative control also passes is not deciding anything.
``G2 equilibrium limit``
    The same mesh, the real lobe, ``deposit_t0``, run for
    ``EQUILIBRIUM_TIME_S``: the added density must be uniform to
    ``EQUILIBRIUM_REL_TOL`` and equal to the injected inventory divided by the
    total active volume. The operator's only steady state under a zero-flux
    boundary is the one its null space allows.
``G3 volume reciprocity``
    The one-substep propagator ``P = (I + h K V^-1)^-1``, built column by
    column, must satisfy detailed balance with respect to VOLUME:
    ``V_i P_ji = V_j P_ij`` to ``RECIPROCITY_REL_TOL``. This is the discrete
    statement that the measure is volume, and it is independent of G1: a
    reciprocal operator with the wrong measure would still fail G1, and an
    operator that passes G1 by normalization would fail this.
``G4 free-space variance``
    A uniform-bore tube, a single-cell ``deposit_t0``, nothing reaching the
    ends: the variance of the inventory must equal ``2 D t`` to
    ``FREE_SPACE_REL_TOL``. This is the only gate that checks the operator
    carries the diffusivity it was given, rather than merely conserving and
    spreading something.
``G5 convergence``
    The probe rows of the added density must not move when the mesh is
    refined (``< GRID_CONVERGENCE_REL``) or the substep count doubled
    (``< SUBSTEP_CONVERGENCE_REL``). The mesh leg runs on the builder's own
    parametric configuration, which resolves at any ``nx``; the substep leg
    runs on the production configuration, whose per-cell radius profile is
    sized to one mesh and is not this script's to re-grid.

OPTIONAL MODES -- each reads data that does not live in this repository, so
neither the default run nor the smoke suite depends on it:

``--legacy-rows FILE --base-h5 FILE``
    G0 LEGACY BIT-IDENTITY. Rebuilds the committed initial-fill rows of a
    configuration file through the legacy ballistic route and compares them at
    RAW UINT64 -- float64 bit patterns read as integers, so a one-ulp move is
    a difference. The bar is zero differing values on both rows. This is the
    gate that says a change to the builder moved nothing that is committed.
``--tpmc-record FILE``
    G6 TPMC RECORD GATE. Scores candidate members against a banked
    test-particle Monte Carlo record of the same puff on the same geometry, by
    the total-variation distance of the normalized inventory profile and by
    ``z90``. The pre-registered bins are :data:`TPMC_BINS`; the legacy top-hat
    is scored on the same instrument and must fail them.

Usage (from the repo root, PYTHONPATH set to the repo root):

    python scripts/verify/verify_fill_spreading.py
    python scripts/verify/verify_fill_spreading.py --fast
    python scripts/verify/verify_fill_spreading.py --tpmc-record RECORD.npz
"""

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(Path(__file__).resolve().parents[1] / _sub)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sp3_build_nn0 as sp3  # noqa: E402
from stance_config import stance_config  # noqa: E402

from cablp.constants import m_He_cgs  # noqa: E402
from cablp.solvers._sim1d import LAPDSim1D  # noqa: E402
from cablp.solvers._sim1d.physics.neutrals import (  # noqa: E402
    gas_puff_rate_profile,
)

#: The configuration the production legs of G1, G2 and G5 run on.
PRODUCTION_STANCE = "g1atrim"
#: Neutral temperature the gates evaluate the mean speed at [K].
GATE_TN_K = 300.0
#: Foot duration the transport gates integrate over [s]. Any positive time
#: states the same properties; this is the ES1 registered foot's order.
GATE_DT_S = 5.83e-3

#: G1 bar on ``(max - min) / mean`` of the added density over the active set
#: for a source uniform per unit volume [1].
UNIFORM_SOURCE_REL_TOL = 1.0e-10
#: G1 negative control: the area ratio of the synthetic tube's bore step, and
#: therefore the step the length-weighted route puts in the added density [1].
LEGACY_STEP_AREA_RATIO = 1.5625
#: G1 negative control bar: the legacy route must miss the uniformity bar by
#: this much at least, or it is not controlling anything [1].
LEGACY_CONTROL_MIN_SPREAD = 0.1

#: G2 integration time [s] and bar on the resulting uniformity [1].
EQUILIBRIUM_TIME_S = 10.0
EQUILIBRIUM_REL_TOL = 1.0e-8

#: G3 bar on the propagator's volume-detailed-balance residual [1].
RECIPROCITY_REL_TOL = 1.0e-10

#: G4 bar on the relative miss of the inventory variance against ``2 D t`` [1].
FREE_SPACE_REL_TOL = 0.02
#: G4 bar on the share of inventory allowed to reach the tube's end cells,
#: above which the walls are in play and the free-space statement does not
#: hold [1].
FREE_SPACE_END_LEAK = 1.0e-8

#: G5 bars on the probe rows' relative movement under mesh refinement and
#: under a doubled substep count [1].
GRID_CONVERGENCE_REL = 0.01
SUBSTEP_CONVERGENCE_REL = 0.005
#: The two meshes the G5 mesh leg compares, as ``nx`` [1].
GRID_CONVERGENCE_NX = (268, 536)

#: G6 pre-registered bins, keyed by the member as this script names it. Each
#: entry is ``(max total-variation distance, min z90 ratio, max z90 ratio)``
#: against the record, applying at every reported time.
TPMC_BINS = {
    "knudsen kappa=2/3": (0.09, 1.00, 1.12),
    "knudsen kappa=0.5": (0.06, 0.95, 1.05),
}
#: G6: the member that must FAIL every bin above.
TPMC_NEGATIVE_MEMBER = "legacy ballistic top-hat"
#: G6 report times [s]. Both are exact samples of the record's own grid.
TPMC_REPORT_TIMES_S = (4.5e-3, 7.0e-3)
#: G6: the geometry file the record was produced on, found beside the record.
TPMC_GEOMETRY_NAME = "eqmap_demo_es1_nx240.npz"


def _face_open_from_areas(area_cm2):
    """Return the per-face open area of a plain tube [cm^2], length cells + 1.

    The restricting ``min`` of the two cells a face separates, which is what
    the mesh's own face-area array carries before any mesh or baffle throttles
    it. The two domain-boundary faces take their one neighbour's area; nothing
    flows through them, because the operator's ends are zero-flux.
    """
    area = np.asarray(area_cm2, dtype=float)
    face = np.empty(area.size + 1, dtype=float)
    face[1:-1] = np.minimum(area[:-1], area[1:])
    face[0] = area[0]
    face[-1] = area[-1]
    return face


def _tube(length_cm, radius_cm, cell_role):
    """Return the arrays the operator reads for a straight tube of given radii."""
    length = np.asarray(length_cm, dtype=float)
    radius = np.asarray(radius_cm, dtype=float)
    edges = np.concatenate(([0.0], np.cumsum(length)))
    area = math.pi * radius ** 2
    return SimpleNamespace(
        cells=int(length.size),
        z_cm=0.5 * (edges[:-1] + edges[1:]),
        length_cm=length,
        cell_role=np.asarray(list(cell_role), dtype=object),
        neutral_volume_cm3=area * length,
        neutral_face_area_cm2=_face_open_from_areas(area),
    )


def _bore_step_tube(cells_per_leg=30, cell_length_cm=10.0,
                    narrow_radius_cm=40.0, wide_radius_cm=50.0):
    """Return a tube that steps from a narrow bore to a wide one.

    The radii are chosen so the step's area ratio is
    :data:`LEGACY_STEP_AREA_RATIO`, which is the number the length-weighted
    route puts into the added density there and the number G1's negative
    control reports. The roles carry a plenum, a cathode and two gap cells in
    front and an end wall behind, so the active mask has cells to exclude in
    both modes and the gap-coupled variant has something to open.
    """
    radii = np.concatenate((
        np.full(4, wide_radius_cm),
        np.full(cells_per_leg, narrow_radius_cm),
        np.full(cells_per_leg, wide_radius_cm),
        [wide_radius_cm],
    ))
    roles = (["plenum", "cathode", "gap", "gap"] + ["puff"]
             + ["column"] * (2 * cells_per_leg - 1) + ["end_wall"])
    return _tube(np.full(radii.size, cell_length_cm), radii, roles)


def _volume_uniform_source(mesh, active, total_atoms=1.0):
    """Return a deposit proportional to cell VOLUME over the active set."""
    volume = np.asarray(mesh.neutral_volume_cm3, dtype=float)
    deposit = np.where(np.asarray(active, dtype=bool), volume, 0.0)
    return deposit * (float(total_atoms) / float(deposit.sum()))


def _spread_relative(values):
    """Return ``(max - min) / mean`` of an array [1]."""
    values = np.asarray(values, dtype=float)
    return float((values.max() - values.min()) / values.mean())


def _production_mesh(nx=None):
    """Return the production configuration's geometry, optionally re-meshed."""
    params, flags = stance_config(PRODUCTION_STANCE)
    if nx is not None:
        params["nx"] = int(nx)
    return LAPDSim1D(dict(params), dict(flags)).geometry, params, flags


def _lobe(geometry, params, dt_s):
    """Return the puff's first-flight deposit over ``dt_s`` [particles/cell]."""
    rate = gas_puff_rate_profile(
        geometry,
        params["S_gp"],
        params["gas_puff_valves"],
        profile=params["gas_puff_profile"],
        z_cm=params["gas_puff_z_cm"],
        sigma_cm=params["gas_puff_sigma_cm"],
        throw_cm=params["gas_puff_throw_cm"],
        orifice_id_cm=params["gas_puff_orifice_id_cm"],
        orifice_length_cm=params["gas_puff_orifice_length_cm"],
        end=0,
    )
    return rate * np.asarray(geometry.neutral_volume_cm3, dtype=float) * float(dt_s)


def _quantile_z(z_cm, inventory, fraction):
    """Return the z below which ``fraction`` of the inventory sits [cm]."""
    inventory = np.asarray(inventory, dtype=float)
    cumulative = np.cumsum(inventory) / float(inventory.sum())
    return float(np.interp(fraction, cumulative, np.asarray(z_cm, dtype=float)))


def _total_variation(first, second):
    """Return the total-variation distance of two inventory profiles [1]."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    return float(
        0.5 * np.sum(np.abs(first / first.sum() - second / second.sum()))
    )


# ----------------------------------------------------------------------
# G1 -- uniform-source control, with the legacy route as negative control
# ----------------------------------------------------------------------
def gate_uniform_source(vbar_cm_s, production=True):
    """Return ``(ok, lines)`` for the uniform-source density-continuity gate."""
    lines = []
    ok = True
    meshes = [("bore-step tube", _bore_step_tube(), True)]
    if production:
        geometry, _, _ = _production_mesh()
        meshes.append((f"{PRODUCTION_STANCE} mesh", geometry, False))
    for label, mesh, is_tube in meshes:
        for gap_coupling in (False, True):
            active = sp3.knudsen_active_mask(mesh.cell_role, gap_coupling)
            if gap_coupling and not np.any(
                active & ~sp3.knudsen_active_mask(mesh.cell_role, False)
            ):
                continue
            deposit = _volume_uniform_source(mesh, active)
            for kappa in (sp3.KNUDSEN_KAPPA_DEFAULT, 0.5):
                accumulated, _ = sp3.knudsen_spread(
                    mesh.z_cm, mesh.length_cm, mesh.neutral_volume_cm3,
                    mesh.neutral_face_area_cm2, active, deposit,
                    GATE_DT_S, vbar_cm_s, kappa=kappa,
                )
                density = (
                    accumulated[active]
                    / np.asarray(mesh.neutral_volume_cm3, dtype=float)[active]
                )
                spread = _spread_relative(density)
                good = spread < UNIFORM_SOURCE_REL_TOL
                ok = ok and good
                lines.append(
                    f"  [{'ok' if good else 'FAIL'}] {label}, "
                    f"gap_coupling={gap_coupling}, kappa={kappa:.6g}: "
                    f"added-density spread {spread:.3e} "
                    f"(bar {UNIFORM_SOURCE_REL_TOL:.0e})"
                )
        # THE NEGATIVE CONTROL: the same uniform source through the legacy
        # length-weighted route, whose added density goes as 1/A.
        active = sp3.knudsen_active_mask(mesh.cell_role, False)
        deposit = _volume_uniform_source(mesh, active)
        width = float(np.sum(mesh.length_cm))
        legacy = sp3.spread_matrix(mesh, "ballistic", width) @ deposit
        volume = np.asarray(mesh.neutral_volume_cm3, dtype=float)
        density = legacy[active] / volume[active]
        spread = _spread_relative(density)
        controlled = spread > LEGACY_CONTROL_MIN_SPREAD
        ok = ok and controlled
        lines.append(
            f"  [{'ok' if controlled else 'FAIL'}] NEGATIVE CONTROL "
            f"{label}, legacy length-weighted ballistic: added-density "
            f"spread {spread:.6g} (must exceed "
            f"{LEGACY_CONTROL_MIN_SPREAD:.6g})"
        )
        if is_tube:
            # ...and on the tube the step is the area ratio exactly.
            area = volume / np.asarray(mesh.length_cm, dtype=float)
            index = np.flatnonzero(active)
            lo, hi = index[:-1], index[1:]
            step = int(np.argmax(
                np.maximum(area[lo], area[hi]) / np.minimum(area[lo], area[hi])
            ))
            ratio = float(
                (legacy[lo[step]] / volume[lo[step]])
                / (legacy[hi[step]] / volume[hi[step]])
            )
            exact = abs(ratio / LEGACY_STEP_AREA_RATIO - 1.0) < 1.0e-12
            ok = ok and exact
            lines.append(
                f"  [{'ok' if exact else 'FAIL'}] NEGATIVE CONTROL "
                f"{label}, step ratio {ratio:.10f} against the area ratio "
                f"{LEGACY_STEP_AREA_RATIO:.10f}"
            )
    return ok, lines


# ----------------------------------------------------------------------
# G2 -- equilibrium limit
# ----------------------------------------------------------------------
def gate_equilibrium(vbar_cm_s, production=True):
    """Return ``(ok, lines)`` for the long-time uniformity gate."""
    lines = []
    ok = True
    mesh = _bore_step_tube()
    cases = [("bore-step tube", mesh, np.where(
        np.asarray(mesh.cell_role, dtype=object) == "puff", 1.0, 0.0
    ))]
    if production:
        geometry, params, _ = _production_mesh()
        cases.append((
            f"{PRODUCTION_STANCE} mesh, real lobe",
            geometry,
            _lobe(geometry, params, GATE_DT_S),
        ))
    for label, case, deposit in cases:
        active = sp3.knudsen_active_mask(case.cell_role, False)
        accumulated, _ = sp3.knudsen_spread(
            case.z_cm, case.length_cm, case.neutral_volume_cm3,
            case.neutral_face_area_cm2, active, deposit,
            EQUILIBRIUM_TIME_S, vbar_cm_s,
            source_convention="deposit_t0",
        )
        volume = np.asarray(case.neutral_volume_cm3, dtype=float)
        density = accumulated[active] / volume[active]
        spread = _spread_relative(density)
        expected = float(np.sum(deposit)) / float(np.sum(volume[active]))
        miss = abs(float(np.mean(density)) / expected - 1.0)
        good = spread < EQUILIBRIUM_REL_TOL and miss < EQUILIBRIUM_REL_TOL
        ok = ok and good
        lines.append(
            f"  [{'ok' if good else 'FAIL'}] {label} at t = "
            f"{EQUILIBRIUM_TIME_S:g} s: spread {spread:.3e}, level "
            f"{float(np.mean(density)):.9e} against inventory/volume "
            f"{expected:.9e} (miss {miss:.3e}, bar "
            f"{EQUILIBRIUM_REL_TOL:.0e})"
        )
    return ok, lines


# ----------------------------------------------------------------------
# G3 -- volume reciprocity of the one-substep propagator
# ----------------------------------------------------------------------
def gate_reciprocity(vbar_cm_s):
    """Return ``(ok, lines)`` for the propagator's volume detailed balance."""
    mesh = _bore_step_tube()
    active = sp3.knudsen_active_mask(mesh.cell_role, False)
    index, conductance = sp3.knudsen_face_conductances(
        mesh.z_cm, mesh.length_cm, mesh.neutral_volume_cm3,
        mesh.neutral_face_area_cm2, active, vbar_cm_s,
        sp3.KNUDSEN_KAPPA_DEFAULT,
    )
    volume = np.asarray(mesh.neutral_volume_cm3, dtype=float)[index]
    size = int(index.size)
    step = GATE_DT_S / sp3.KNUDSEN_SUBSTEPS_DEFAULT
    diag = np.ones(size, dtype=float)
    diag[:-1] += step * conductance / volume[:-1]
    diag[1:] += step * conductance / volume[1:]
    upper = -step * conductance / volume[1:]
    lower = -step * conductance / volume[:-1]
    propagator = np.empty((size, size), dtype=float)
    for column in range(size):
        unit = np.zeros(size, dtype=float)
        unit[column] = 1.0
        propagator[:, column] = sp3._thomas_sweep(lower, diag, upper, unit)
    weighted = propagator * volume[None, :]
    residual = float(np.max(np.abs(weighted - weighted.T)))
    scale = float(np.max(np.abs(weighted)))
    relative = residual / scale
    ok = relative < RECIPROCITY_REL_TOL
    return ok, [
        f"  [{'ok' if ok else 'FAIL'}] propagator {size}x{size}: "
        f"max |V_j P_ij - V_i P_ji| / max |V_j P_ij| = {relative:.3e} "
        f"(bar {RECIPROCITY_REL_TOL:.0e})"
    ]


# ----------------------------------------------------------------------
# G4 -- free-space variance
# ----------------------------------------------------------------------
def gate_free_space(vbar_cm_s):
    """Return ``(ok, lines)`` for the ``variance = 2 D t`` gate."""
    lines = []
    ok = True
    cells, cell_length, radius = 301, 15.0, 50.0
    mesh = _tube(
        np.full(cells, cell_length), np.full(cells, radius),
        ["column"] * cells,
    )
    active = sp3.knudsen_active_mask(mesh.cell_role, False)
    deposit = np.zeros(cells, dtype=float)
    deposit[cells // 2] = 1.0
    # ONE duration for both members, so the two readings differ by their
    # diffusivity and not by the time they were given: the gate is that the
    # operator carries the D it was handed, which a duration scaled by D
    # would hide. It is chosen to put the tube's end six standard deviations
    # out, where the reflecting ends cannot reach the second moment.
    duration = (
        (cells * cell_length / 12.0) ** 2
        / (2.0 * sp3.KNUDSEN_KAPPA_DEFAULT * radius * vbar_cm_s)
    )
    for kappa in (sp3.KNUDSEN_KAPPA_DEFAULT, 0.5):
        diffusivity = kappa * radius * vbar_cm_s
        accumulated, _ = sp3.knudsen_spread(
            mesh.z_cm, mesh.length_cm, mesh.neutral_volume_cm3,
            mesh.neutral_face_area_cm2, active, deposit,
            duration, vbar_cm_s, kappa=kappa,
            source_convention="deposit_t0",
        )
        z = np.asarray(mesh.z_cm, dtype=float)
        share = accumulated / accumulated.sum()
        centre = float(np.sum(share * z))
        variance = float(np.sum(share * (z - centre) ** 2))
        expected = 2.0 * diffusivity * duration
        miss = abs(variance / expected - 1.0)
        leak = float(share[0] + share[-1])
        good = miss < FREE_SPACE_REL_TOL and leak < FREE_SPACE_END_LEAK
        ok = ok and good
        lines.append(
            f"  [{'ok' if good else 'FAIL'}] kappa={kappa:.6g}, "
            f"t={duration:.6g} s: variance {variance:.6g} against 2 D t "
            f"{expected:.6g} cm^2 (miss {miss:.3e}, bar "
            f"{FREE_SPACE_REL_TOL:g}); end-cell share {leak:.3e}"
        )
    return ok, lines


# ----------------------------------------------------------------------
# G5 -- convergence in mesh and in substeps
# ----------------------------------------------------------------------
def _probe_density(mesh, accumulated):
    """Return the added density at :data:`sp3.KNUDSEN_PROBE_Z_CM` [cm^-3].

    Two readings of the same profile: interpolated TO each station, and the
    value of whichever cell is nearest it. A refinement moves the cell centres,
    so the nearest-cell reading changes by the local gradient times that shift
    even when the solution has not moved at all -- it measures the sampling,
    not the operator. The interpolated reading is what the convergence gate
    decides on, and the nearest-cell one is reported beside it so the size of
    that sampling shift is visible rather than mistaken for a result. The
    builder's own ledger quotes cells, which is the right reading there,
    because it is describing one mesh.
    """
    z = np.asarray(mesh.z_cm, dtype=float)
    density = accumulated / np.asarray(mesh.neutral_volume_cm3, dtype=float)
    stations = np.asarray(sp3.KNUDSEN_PROBE_Z_CM, dtype=float)
    nearest = np.array(
        [density[int(np.argmin(np.abs(z - station)))] for station in stations],
        dtype=float,
    )
    return np.interp(stations, z, density), nearest


def _solve_for(geometry, params, substeps, vbar_cm_s):
    """Return the added density at the probe stations for one configuration."""
    active = sp3.knudsen_active_mask(geometry.cell_role, False)
    deposit = _lobe(geometry, params, GATE_DT_S)
    accumulated, _ = sp3.knudsen_spread(
        geometry.z_cm, geometry.length_cm, geometry.neutral_volume_cm3,
        geometry.neutral_face_area_cm2, active, deposit,
        GATE_DT_S, vbar_cm_s, substeps=substeps,
    )
    return _probe_density(geometry, accumulated)


def gate_convergence(vbar_cm_s):
    """Return ``(ok, lines)`` for the mesh and substep convergence gate."""
    lines = []
    ok = True
    # MESH LEG. The production configuration's per-cell radius profile is
    # sized to one mesh, and re-gridding it belongs to the tool that builds
    # it, so the mesh leg runs on the builder's own parametric configuration
    # at the same operating point, which resolves at any nx.
    rows = []
    for nx in GRID_CONVERGENCE_NX:
        params, flags = sp3.stance_config(1, nx, 9010.0, True)
        geometry = LAPDSim1D(dict(params), dict(flags)).geometry
        rows.append(
            _solve_for(geometry, params, sp3.KNUDSEN_SUBSTEPS_DEFAULT,
                       vbar_cm_s)
        )
    (coarse, coarse_cell), (fine, fine_cell) = rows
    moved = float(np.max(np.abs(fine / coarse - 1.0)))
    sampled = float(np.max(np.abs(fine_cell / coarse_cell - 1.0)))
    good = moved < GRID_CONVERGENCE_REL
    ok = ok and good
    lines.append(
        f"  [{'ok' if good else 'FAIL'}] mesh nx "
        f"{GRID_CONVERGENCE_NX[0]} -> {GRID_CONVERGENCE_NX[1]} on the "
        f"builder's parametric configuration: probe rows move "
        f"{moved:.3e} (bar {GRID_CONVERGENCE_REL:g}); the same rows read at "
        f"the nearest CELL instead move {sampled:.3e}, which is the cell "
        f"centres shifting under the refinement"
    )
    lines.append("      coarse " + " ".join(f"{v:.6g}" for v in coarse))
    lines.append("      fine   " + " ".join(f"{v:.6g}" for v in fine))
    # SUBSTEP LEG, on the production mesh. The mesh does not move here, so the
    # two readings agree and only the interpolated one is printed.
    geometry, params, _ = _production_mesh()
    base_rows, _ = _solve_for(geometry, params,
                              sp3.KNUDSEN_SUBSTEPS_DEFAULT, vbar_cm_s)
    doubled_rows, _ = _solve_for(geometry, params,
                                 2 * sp3.KNUDSEN_SUBSTEPS_DEFAULT, vbar_cm_s)
    moved = float(np.max(np.abs(doubled_rows / base_rows - 1.0)))
    good = moved < SUBSTEP_CONVERGENCE_REL
    ok = ok and good
    lines.append(
        f"  [{'ok' if good else 'FAIL'}] substeps "
        f"{sp3.KNUDSEN_SUBSTEPS_DEFAULT} -> "
        f"{2 * sp3.KNUDSEN_SUBSTEPS_DEFAULT} on the {PRODUCTION_STANCE} "
        f"mesh: probe rows move {moved:.3e} (bar "
        f"{SUBSTEP_CONVERGENCE_REL:g})"
    )
    lines.append("      base    " + " ".join(f"{v:.6g}" for v in base_rows))
    lines.append("      doubled " + " ".join(f"{v:.6g}" for v in doubled_rows))
    return ok, lines


# ----------------------------------------------------------------------
# G0 -- legacy bit-identity against committed rows
# ----------------------------------------------------------------------
def _differing_uint64(got, want):
    """Return how many raw uint64 bit patterns differ between two arrays."""
    got = np.ascontiguousarray(np.asarray(got, dtype=float)).view(np.uint64)
    want = np.ascontiguousarray(np.asarray(want, dtype=float)).view(np.uint64)
    if got.size != want.size:
        return max(got.size, want.size)
    return int(np.count_nonzero(got != want))


def gate_legacy_rows(rows_path, base_h5, es, nx, sgp, geometry_npz):
    """Return ``(ok, lines)`` for the committed-row bit-identity gate."""
    import tomllib

    with open(rows_path, "rb") as handle:
        document = tomllib.load(handle)
    block = document["models"]["initial_neutral_state"]
    keys = [str(geometry_npz) + ":" + name for name in (
        "plasma_radius_profile_cm", "machine_radius_profile_cm",
        "neutral_baffle_positions_cm", "neutral_baffle_clear_radii_cm",
    )]
    args = SimpleNamespace(
        es=int(es), nx=int(nx), sgp=float(sgp), two_zone=True, zone="chamber",
        base_from_h5=str(base_h5),
        dt_foot_s=sp3.registered_foot_s(int(es)),
        kernel="ballistic",
        sigma_hehe_cm2=sp3.SIGMA_HE_HE_CM2, mfp_cm=None, tn_k=None,
        extra=["gas_puff_profile=orifice", "gas_puff_orifice_id_cm=3.95",
               "gas_puff_orifice_length_cm=22.0"],
        extra_flag=["prescribed_area_geometry=true", "neutral_baffles=true"],
        extra_npz=[
            "plasma_radius_profile_cm=" + keys[0],
            "machine_radius_profile_cm=" + keys[1],
            "neutral_baffle_positions_cm=" + keys[2],
            "neutral_baffle_clear_radii_cm=" + keys[3],
        ],
    )
    column, annulus = sp3.build(args)[:2]
    lines = []
    ok = True
    for name, got in (("nn0_profile", column),
                      ("nn0_annulus_profile", annulus)):
        want = np.asarray(block[name], dtype=float)
        differing = _differing_uint64(got, want)
        good = differing == 0
        ok = ok and good
        lines.append(
            f"  [{'ok' if good else 'FAIL'}] {name}: {differing} of "
            f"{want.size} raw uint64 values differ, max |delta| "
            f"{float(np.max(np.abs(np.asarray(got, dtype=float) - want))):.3e}"
        )
    return ok, lines


# ----------------------------------------------------------------------
# G6 -- the banked test-particle record
# ----------------------------------------------------------------------
def gate_tpmc(record_paths, geometry_path, vbar_cm_s):
    """Return ``(ok, lines)`` for the record comparison."""
    with np.load(geometry_path, allow_pickle=True) as data:
        map_z = np.asarray(data["z_cm"], dtype=float)
        map_length = np.asarray(data["length_cm"], dtype=float)
        map_volume = np.asarray(data["neutral_volume_cm3"], dtype=float)
        map_roles = np.asarray(data["cell_role"], dtype=object)
        import json
        provenance = json.loads(str(data["provenance"]))
    mesh = SimpleNamespace(
        cells=int(map_z.size), z_cm=map_z, length_cm=map_length,
        cell_role=map_roles, neutral_volume_cm3=map_volume,
        neutral_face_area_cm2=_face_open_from_areas(map_volume / map_length),
    )
    active = sp3.knudsen_active_mask(map_roles, False)
    rate = gas_puff_rate_profile(
        mesh,
        provenance["S_gp_sccm"],
        provenance["gas_puff_valves"],
        profile=provenance["gas_puff_profile"],
        z_cm=provenance["gas_puff_z_cm"],
        throw_cm=provenance["gas_puff_throw_cm"],
        end=0,
    )
    # THE ACTIVE SET THE BINS WERE REGISTERED AGAINST. The record's own domain
    # starts at the cathode cell and includes the gap, so a comparison run on
    # the operator's DEFAULT active set is scoring a model that places no gas
    # at all in a region where the record places a sixth of it. That deficit
    # is a fixed shape difference no diffusivity can absorb -- it is measured
    # and reported below as ``gap-closed``, alongside the share of the record
    # it accounts for -- so the bins are decided on the gap-coupled set, which
    # is the comparison they describe. Note the geometry file carries no mesh
    # transparency, so the gap is coupled here through an unthrottled face;
    # the resulting gap fill is reported, not gated.
    records = []
    for path in record_paths:
        with np.load(path, allow_pickle=True) as data:
            records.append((
                Path(path).name,
                np.asarray(data["z"], dtype=float),
                np.asarray(data["report_times_s"], dtype=float),
                np.asarray(data["nn_mean_t"], dtype=float),
            ))
    lines = [
        f"  geometry {Path(geometry_path).name}: {mesh.cells} cells, puff "
        f"{provenance['gas_puff_profile']} at z="
        f"{provenance['gas_puff_z_cm']:g} cm, throw "
        f"{provenance['gas_puff_throw_cm']:g} cm, "
        f"{provenance['S_gp_sccm']:g} sccm x "
        f"{provenance['gas_puff_valves']} valves",
        f"  records: " + ", ".join(name for name, _, _, _ in records),
    ]
    coupled = sp3.knudsen_active_mask(map_roles, True)
    ok = True
    for duration in TPMC_REPORT_TIMES_S:
        deposit = rate * map_volume * duration
        members = {}
        disclosed = {}
        for label, kappa in (("knudsen kappa=2/3", sp3.KNUDSEN_KAPPA_DEFAULT),
                             ("knudsen kappa=0.5", 0.5)):
            members[label], _ = sp3.knudsen_spread(
                mesh.z_cm, mesh.length_cm, mesh.neutral_volume_cm3,
                mesh.neutral_face_area_cm2, coupled, deposit,
                duration, vbar_cm_s, kappa=kappa,
            )
            disclosed[label], _ = sp3.knudsen_spread(
                mesh.z_cm, mesh.length_cm, mesh.neutral_volume_cm3,
                mesh.neutral_face_area_cm2, active, deposit,
                duration, vbar_cm_s, kappa=kappa,
            )
        members[TPMC_NEGATIVE_MEMBER] = (
            sp3.spread_matrix(mesh, "ballistic", vbar_cm_s * duration)
            @ deposit
        )
        for name, record_z, times, profiles in records:
            sample = int(np.argmin(np.abs(times - duration)))
            if abs(float(times[sample]) - duration) > 1.0e-9:
                raise ValueError(
                    f"{name} carries no sample at t = {duration:g} s; it "
                    f"reports {times.tolist()}"
                )
            common = np.array(
                [int(np.argmin(np.abs(map_z - value))) for value in record_z],
                dtype=int,
            )
            if np.max(np.abs(map_z[common] - record_z)) > 1.0e-9:
                raise ValueError(
                    f"{name}'s z axis does not sit on the geometry's cells"
                )
            reference = profiles[sample] * map_volume[common]
            reference_z90 = _quantile_z(record_z, reference, 0.9)
            # THE DISCLOSURE: the same members on the DEFAULT active set, with
            # the share of the record that sits in the region they leave
            # empty. Reported, never gated.
            behind = map_z[common] < float(np.min(map_z[active]))
            record_behind = float(
                reference[behind].sum() / reference.sum()
            )
            for label, accumulated in disclosed.items():
                candidate = accumulated[common]
                lines.append(
                    f"       (disclosed, not gated) t={duration * 1e3:g} ms "
                    f"{name} {label} on the DEFAULT gap-closed active set: "
                    f"TV {_total_variation(candidate, reference):.4f}, z90 "
                    f"{_quantile_z(record_z, candidate, 0.9):.1f} cm; the "
                    f"record puts {record_behind:.4f} of its inventory behind "
                    f"the mesh, where that set puts "
                    f"{float(candidate[behind].sum() / candidate.sum()):.4f}"
                )
            gap_share = float(
                members["knudsen kappa=2/3"][common][behind].sum()
                / members["knudsen kappa=2/3"][common].sum()
            )
            lines.append(
                f"       (disclosed, not gated) t={duration * 1e3:g} ms "
                f"{name}: gap-coupled kappa=2/3 puts {gap_share:.4f} of its "
                f"inventory behind the mesh against the record's "
                f"{record_behind:.4f}, through an unthrottled face"
            )
            for label, accumulated in members.items():
                candidate = accumulated[common]
                distance = _total_variation(candidate, reference)
                z90 = _quantile_z(record_z, candidate, 0.9)
                ratio = z90 / reference_z90
                bins = TPMC_BINS.get(label)
                if bins is None:
                    passes = all(
                        distance <= limit and low <= ratio <= high
                        for limit, low, high in TPMC_BINS.values()
                    )
                    good = not passes
                    verdict = "FAILS the bins" if good else "PASSES a bin"
                    ok = ok and good
                    lines.append(
                        f"  [{'ok' if good else 'FAIL'}] t={duration * 1e3:g} "
                        f"ms {name} {label}: TV {distance:.4f}, z90 "
                        f"{z90:.1f} cm = {ratio:.4f} x record "
                        f"{reference_z90:.1f} cm -- {verdict} (it must fail)"
                    )
                    continue
                limit, low, high = bins
                good = distance <= limit and low <= ratio <= high
                ok = ok and good
                lines.append(
                    f"  [{'ok' if good else 'FAIL'}] t={duration * 1e3:g} ms "
                    f"{name} {label}: TV {distance:.4f} (bar {limit:g}), z90 "
                    f"{z90:.1f} cm = {ratio:.4f} x record {reference_z90:.1f} "
                    f"cm (bin [{low:g}, {high:g}])"
                )
    return ok, lines


# ----------------------------------------------------------------------
# the subsets
# ----------------------------------------------------------------------
def fast_gates():
    """Return ``[(name, ok, lines), ...]`` for the subset the smoke suite runs.

    Everything decidable on a synthetic mesh in milliseconds: the gates that
    need a production configuration built, or a file from outside this
    repository, are not here.
    """
    vbar = sp3.mean_speed_cm_s(GATE_TN_K, m_He_cgs)
    results = []
    ok, lines = gate_uniform_source(vbar, production=False)
    results.append(("G1 uniform-source control", ok, lines))
    ok, lines = gate_equilibrium(vbar, production=False)
    results.append(("G2 equilibrium limit", ok, lines))
    ok, lines = gate_reciprocity(vbar)
    results.append(("G3 volume reciprocity", ok, lines))
    ok, lines = gate_free_space(vbar)
    results.append(("G4 free-space variance", ok, lines))
    return results


def default_gates():
    """Return ``[(name, ok, lines), ...]`` for every in-repo gate."""
    vbar = sp3.mean_speed_cm_s(GATE_TN_K, m_He_cgs)
    results = []
    ok, lines = gate_uniform_source(vbar, production=True)
    results.append(("G1 uniform-source control", ok, lines))
    ok, lines = gate_equilibrium(vbar, production=True)
    results.append(("G2 equilibrium limit", ok, lines))
    ok, lines = gate_reciprocity(vbar)
    results.append(("G3 volume reciprocity", ok, lines))
    ok, lines = gate_free_space(vbar)
    results.append(("G4 free-space variance", ok, lines))
    ok, lines = gate_convergence(vbar)
    results.append(("G5 convergence", ok, lines))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast", action="store_true",
        help="run only the in-repo subset that needs no configuration build "
             "(the subset the smoke suite runs); the optional modes below "
             "still run if they are named",
    )
    parser.add_argument(
        "--legacy-rows", type=Path, default=None,
        help="G0: a configuration file whose committed initial-fill rows the "
             "legacy ballistic route must reproduce at raw uint64",
    )
    parser.add_argument(
        "--base-h5", type=Path, default=None,
        help="G0: the equilibrated result whose t = 0 frames are the base "
             "those rows were built on",
    )
    parser.add_argument(
        "--legacy-geometry-npz", type=Path, default=None,
        help="G0: the .npz carrying the per-cell geometry profiles those rows "
             "were built on",
    )
    parser.add_argument("--legacy-es", type=int, default=1)
    parser.add_argument("--legacy-nx", type=int, default=268)
    parser.add_argument("--legacy-sgp", type=float, default=9010.0)
    parser.add_argument(
        "--tpmc-record", type=Path, action="append", default=None,
        help="G6: a banked test-particle record to score against; repeat for "
             "several seeds",
    )
    parser.add_argument(
        "--tpmc-geometry", type=Path, default=None,
        help=f"G6: the geometry the record was produced on (default: "
             f"{TPMC_GEOMETRY_NAME} beside the first record)",
    )
    args = parser.parse_args(argv)

    # --fast chooses which of the IN-REPO gates run; the optional modes below
    # are selected by their own arguments and compose with either subset.
    results = fast_gates() if args.fast else default_gates()

    if args.legacy_rows is not None:
        if args.base_h5 is None or args.legacy_geometry_npz is None:
            parser.error(
                "--legacy-rows needs --base-h5 and --legacy-geometry-npz: the "
                "committed rows are a base plus a foot, and both are named"
            )
        ok, lines = gate_legacy_rows(
            args.legacy_rows, args.base_h5, args.legacy_es, args.legacy_nx,
            args.legacy_sgp, args.legacy_geometry_npz,
        )
        results.insert(0, ("G0 legacy bit-identity", ok, lines))

    if args.tpmc_record:
        geometry = args.tpmc_geometry
        if geometry is None:
            geometry = args.tpmc_record[0].parent / TPMC_GEOMETRY_NAME
        if not Path(geometry).exists():
            parser.error(
                f"the record's geometry {geometry} does not exist; name it "
                "with --tpmc-geometry"
            )
        ok, lines = gate_tpmc(
            args.tpmc_record, geometry,
            sp3.mean_speed_cm_s(GATE_TN_K, m_He_cgs),
        )
        results.append(("G6 TPMC record gate", ok, lines))

    print("=== initial-fill spreading verification ===")
    failures = 0
    for name, ok, lines in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        for line in lines:
            print(line)
        failures += 0 if ok else 1
    print(
        f"--- {len(results) - failures} of {len(results)} gates pass ---"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
