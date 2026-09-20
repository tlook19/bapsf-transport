"""Verify the initial-fill builder's spreading members against their stated properties.

``scripts/stance/sp3_build_nn0.py`` offers three spreading members. The
REGISTERED one, which an omitted ``--kernel`` builds, is ``knudsen``: a
conservative finite-volume axial diffusion of the foot inventory away from the
puff row on the builder's own mesh and zone volumes, weighted by cell VOLUME,
with the source released continuously over the foot, gap-coupled through the
anode mesh face, at the registered coefficient
``sp3.KNUDSEN_KAPPA_REFERENCE``; its own registration is the three named
coefficient members. The other two, ``diffusive`` and ``ballistic``, are
MATRIX kernels, whose targets are weighted by cell LENGTH and which are
applied to an inventory deposited whole at the start of the foot; they are
retained, by explicit ``--kernel``, as the legacy reproduction route. This
script is the acceptance instrument for the finite-volume member and for the
property that separates the two families.

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
    G0 ROW BIT-IDENTITY, two legs, both compared at RAW UINT64 -- float64 bit
    patterns read as integers, so a one-ulp move is a difference, and the bar
    is zero differing values on both rows. ``G0a`` rebuilds the committed
    initial-fill rows of a configuration file through the builder's
    REGISTERED member, naming no kernel, which is the gate that says a change
    to the builder moved nothing that is committed. ``G0b`` rebuilds the
    LEGACY rows in :data:`LEGACY_ROWS_FIXTURE` through ``--kernel
    ballistic``, which is the gate that says the legacy reproduction route
    still reproduces. Both legs need the equilibrated base and the per-cell
    geometry profiles, which do not live in this repository, so both ride
    this optional mode rather than the default run.
``--tpmc-record FILE``
    G6 TPMC RECORD GATE. Scores candidate members against a banked
    test-particle Monte Carlo record of the same puff on the same geometry, by
    the total-variation distance of the normalized inventory profile and by
    ``z90``. The pre-registered bins are :data:`TPMC_BINS`; the legacy top-hat
    is scored on the same instrument and must miss EVERY one of them.
``--tpmc-production FILE --tpmc-production-geometry-npz FILE``
    G7 TPMC PRODUCTION COMPARISON. Scores the REGISTERED members -- the
    reference, the two ends of its bracket and the instrument-match member --
    against a reduced test-particle record of the same puff on the PRODUCTION
    geometry, at each of the three rungs' registered feet. The operator runs
    gap-coupled with a continuous source and with the neutral baffle's face
    opened, because the record's instrument carries no baffle; the production
    rows keep it. Seven reductions of the two profiles (the near-field and
    full-domain total-variation distances, the three quantile ratios, the bore
    ratio across the source-bore step and the gap ratio behind the anode mesh)
    are measured; :data:`TPMC_PRODUCTION_BINS` says which of them are GATED for
    which member, and the rest are printed. The legacy top-hat must miss
    :data:`TPMC_PRODUCTION_NEGATIVE_BINS`, ``z90`` must rise with the
    coefficient, the record's ``z90`` must fall between the slow member and the
    reference, and the gap-CLOSED set is disclosed beside its own failure.

Usage (from the repo root, PYTHONPATH set to the repo root):

    python scripts/verify/verify_fill_spreading.py
    python scripts/verify/verify_fill_spreading.py --fast
    python scripts/verify/verify_fill_spreading.py --tpmc-record RECORD.npz
    python scripts/verify/verify_fill_spreading.py \\
        --tpmc-production PROFILES.npz \\
        --tpmc-production-geometry-npz GEOMETRY.npz
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
#: G0b: the committed fixture holding the LEGACY initial-fill rows -- the two
#: rows the reference configuration carried before the finite-volume member
#: became the builder's registered one, with the z grid they sit on and a
#: provenance string naming the builder arguments that reproduce them.
LEGACY_ROWS_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "data" / "nn0_legacy_ballistic_reference.npz"
)
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
#: G6: the member that must MISS every bin above -- all of them, not one.
TPMC_NEGATIVE_MEMBER = "legacy ballistic top-hat"
#: G6 report times [s]. Both are exact samples of the record's own grid.
TPMC_REPORT_TIMES_S = (4.5e-3, 7.0e-3)
#: G6: the geometry file the record was produced on, found beside the record.
TPMC_GEOMETRY_NAME = "eqmap_demo_es1_nx240.npz"

#: G7: the ES rung whose configuration the production test-particle run was
#: made on, and therefore the one the comparison builds its geometry from. The
#: three rungs differ only in bank voltage and standby surface temperature,
#: neither of which enters the mesh, the zone volumes or the puff row, so the
#: three feet are compared on ONE geometry and differ only in duration.
TPMC_PRODUCTION_GEOMETRY_ES = 1
#: G7: the ES rungs compared, each at its own registered foot.
TPMC_PRODUCTION_RUNGS = (1, 2, 3)
#: G7: the axial split the near-field total-variation distance is taken over
#: [cm]. Everything past it is an upper bound in the record, so the profile
#: below it is where the comparison carries weight; the full-domain distance is
#: reported beside it.
TPMC_PRODUCTION_SPLIT_Z_CM = 342.65
#: G7: the two 20 cm bands whose mean added densities form the bore ratio
#: [cm] -- the narrow-bore side of the source-bore step and the wide side.
TPMC_PRODUCTION_BORE_BANDS_CM = ((83.25, 103.25), (103.25, 123.25))
#: G7: the members scored, as ``(label, kappa)``. ``kappa=0.50`` is the
#: INSTRUMENT-MATCH member -- the one required to reproduce a collisionless
#: test-particle run -- and the other three are the registered reference and
#: the two ends of its bracket, which are required to sit around it by a
#: bounded amount rather than to match.
TPMC_PRODUCTION_MEMBERS = (
    ("knudsen kappa=0.50", 0.50),
    ("knudsen kappa=0.45", sp3.KNUDSEN_KAPPA_SLOW),
    ("knudsen kappa=2/3", sp3.KNUDSEN_KAPPA_REFERENCE),
    ("knudsen kappa=0.90", sp3.KNUDSEN_KAPPA_FAST),
)
#: G7 PRE-REGISTERED BINS, keyed by member label and then by metric. Each entry
#: is ``(low, high)`` with ``None`` for an open end, and applies at every rung.
#: A metric absent from a member's entry is REPORTED, not gated.
TPMC_PRODUCTION_BINS = {
    "knudsen kappa=0.50": {
        "TV<342": (None, 0.038),
        "z50 ratio": (0.99, 1.05),
        "z90 ratio": (0.97, 1.03),
        "bore ratio": (0.91, 0.98),
    },
    "knudsen kappa=0.45": {
        "z90 ratio": (0.93, 0.99),
    },
    "knudsen kappa=2/3": {
        "TV<342": (None, 0.063),
        "z50 ratio": (1.06, 1.13),
        "z90 ratio": (1.08, 1.15),
    },
    "knudsen kappa=0.90": {
        "z90 ratio": (1.21, 1.29),
    },
}
#: G7: the member that must MISS its bins, and the bins it must miss. A
#: negative control that passes is not controlling anything.
TPMC_PRODUCTION_NEGATIVE_MEMBER = "legacy ballistic top-hat"
TPMC_PRODUCTION_NEGATIVE_BINS = {
    "TV<342": (0.25, None),
    "TV full": (0.50, None),
    "z90 ratio": (2.0, None),
}
#: G7: how close the gap-CLOSED set's near-field total-variation distance is
#: expected to sit to the share of the record's inventory that set leaves
#: empty [1]. Reported, never gated: it is the statement that the gap-closed
#: failure is a fixed missing region rather than a diffusivity that could be
#: retuned.
TPMC_PRODUCTION_GAP_CLOSED_TOL = 0.015


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
            for kappa in (sp3.KNUDSEN_KAPPA_REFERENCE, 0.5):
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
        sp3.KNUDSEN_KAPPA_REFERENCE,
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
        / (2.0 * sp3.KNUDSEN_KAPPA_REFERENCE * radius * vbar_cm_s)
    )
    for kappa in (sp3.KNUDSEN_KAPPA_REFERENCE, 0.5):
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


def _rebuild_rows(base_h5, es, nx, sgp, geometry_npz, kernel):
    """Return ``(column, annulus)`` rebuilt by the builder through ``kernel``.

    Every input the committed rows were built on, stated here once: the
    operating point, the rung's REGISTERED foot, the equilibrated base, the
    per-cell geometry profiles and the orifice puff row. ``kernel`` is the
    one thing that varies between the two legs -- ``None`` leaves it at the
    builder's own registered member, which is the point of G0a.
    """
    keys = [str(geometry_npz) + ":" + name for name in (
        "plasma_radius_profile_cm", "machine_radius_profile_cm",
        "neutral_baffle_positions_cm", "neutral_baffle_clear_radii_cm",
    )]
    args = SimpleNamespace(
        es=int(es), nx=int(nx), sgp=float(sgp), two_zone=True, zone="chamber",
        base_from_h5=str(base_h5),
        dt_foot_s=sp3.registered_foot_s(int(es)),
        kernel=sp3.KERNEL_REGISTERED if kernel is None else str(kernel),
        knudsen_kappa=sp3.KNUDSEN_KAPPA_REFERENCE,
        knudsen_substeps=sp3.KNUDSEN_SUBSTEPS_DEFAULT,
        knudsen_gap_coupling=sp3.KNUDSEN_GAP_COUPLING_REGISTERED,
        knudsen_source_convention=sp3.KNUDSEN_SOURCE_CONVENTIONS[0],
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
    return sp3.build(args)[:2]


def _compare_rows(label, got_pair, want_pair):
    """Return ``(ok, lines)`` for one raw-uint64 comparison of the two rows."""
    lines = []
    ok = True
    for name, got, want in (
        ("nn0_profile", got_pair[0], want_pair[0]),
        ("nn0_annulus_profile", got_pair[1], want_pair[1]),
    ):
        want = np.asarray(want, dtype=float)
        differing = _differing_uint64(got, want)
        good = differing == 0
        ok = ok and good
        lines.append(
            f"  [{'ok' if good else 'FAIL'}] {label} {name}: {differing} of "
            f"{want.size} raw uint64 values differ, max |delta| "
            f"{float(np.max(np.abs(np.asarray(got, dtype=float) - want))):.3e}"
        )
    return ok, lines


def gate_legacy_rows(rows_path, base_h5, es, nx, sgp, geometry_npz):
    """Return ``(ok, lines)`` for the two committed-row bit-identity legs.

    ``G0a`` rebuilds the configuration's own committed rows through the
    builder's REGISTERED member -- no kernel named, exactly as the committed
    rows were built -- and requires raw-uint64 identity. That is the gate
    that says a change to the builder moved nothing that is committed.

    ``G0b`` rebuilds the LEGACY rows the configuration carried before the
    finite-volume member was registered, through ``--kernel ballistic`` at
    the same registered foot, and requires raw-uint64 identity against the
    committed fixture :data:`LEGACY_ROWS_FIXTURE`. That is the gate that says
    the legacy reproduction route still reproduces.

    Both legs need the equilibrated base result and the per-cell geometry
    profiles, neither of which lives in this repository, so both ride the
    optional ``--legacy-rows`` mode rather than the default run.
    """
    import tomllib

    with open(rows_path, "rb") as handle:
        document = tomllib.load(handle)
    block = document["models"]["initial_neutral_state"]
    committed = (
        np.asarray(block["nn0_profile"], dtype=float),
        np.asarray(block["nn0_annulus_profile"], dtype=float),
    )
    ok, lines = _compare_rows(
        "G0a registered route vs the committed rows:",
        _rebuild_rows(base_h5, es, nx, sgp, geometry_npz, None),
        committed,
    )
    with np.load(LEGACY_ROWS_FIXTURE, allow_pickle=False) as fixture:
        legacy = (
            np.asarray(fixture["nn0_profile"], dtype=float),
            np.asarray(fixture["nn0_annulus_profile"], dtype=float),
        )
    ok_b, lines_b = _compare_rows(
        f"G0b --kernel ballistic vs {LEGACY_ROWS_FIXTURE.name}:",
        _rebuild_rows(base_h5, es, nx, sgp, geometry_npz, "ballistic"),
        legacy,
    )
    return ok and ok_b, lines + lines_b


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
        for label, kappa in (("knudsen kappa=2/3", sp3.KNUDSEN_KAPPA_REFERENCE),
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
                    # THE NEGATIVE CONTROL, held to the bar G7 holds its own
                    # to: it must miss EVERY bin it is scored against, not
                    # merely one of them. Satisfying one member's bin means
                    # the control scores like that member on the instrument
                    # that is supposed to separate them, and a control that
                    # is only required to miss one bin would pass anyway on
                    # the strength of the others.
                    satisfied = [
                        member
                        for member, (limit, low, high) in TPMC_BINS.items()
                        if distance <= limit and low <= ratio <= high
                    ]
                    good = not satisfied
                    verdict = (
                        f"misses all {len(TPMC_BINS)} bins"
                        if good
                        else "SATISFIES " + ", ".join(sorted(satisfied))
                    )
                    ok = ok and good
                    lines.append(
                        f"  [{'ok' if good else 'FAIL'}] t={duration * 1e3:g} "
                        f"ms {name} {label}: TV {distance:.4f}, z90 "
                        f"{z90:.1f} cm = {ratio:.4f} x record "
                        f"{reference_z90:.1f} cm -- {verdict} (it must miss "
                        f"every one)"
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
# G7 -- the production-geometry test-particle comparison
# ----------------------------------------------------------------------
def _production_geometry(geometry_npz, es, nx, sgp):
    """Return ``(geometry, params)`` for the configuration the record was run on.

    The same configuration the legacy-row gate rebuilds: the stance at the
    named operating point, with the per-cell radius and baffle profiles read
    out of ``geometry_npz`` and the orifice puff row the record's source was
    sampled from.
    """
    values, _ = sp3.parse_npz_overrides([
        f"{name}={geometry_npz}:{name}" for name in (
            "plasma_radius_profile_cm", "machine_radius_profile_cm",
            "neutral_baffle_positions_cm", "neutral_baffle_clear_radii_cm",
        )
    ])
    values.update(sp3.parse_extra_overrides(
        ["gas_puff_profile=orifice", "gas_puff_orifice_id_cm=3.95",
         "gas_puff_orifice_length_cm=22.0"], "--extra",
    ))
    params, flags = sp3.stance_config(
        int(es), int(nx), float(sgp), True,
        extra_params=values,
        extra_flags=sp3.parse_extra_overrides(
            ["prescribed_area_geometry=true", "neutral_baffles=true"],
            "--extra-flag",
        ),
    )
    return LAPDSim1D(dict(params), dict(flags)).geometry, params


def _baffle_opened_faces(geometry, params):
    """Return ``(face_open_area, opened)`` with the baffle faces unthrottled.

    The record's instrument carries no neutral baffle, so scoring the operator
    against it with one in the way would be comparing two different vessels.
    Every face that is throttled below the smaller of the two cell areas AND
    sits at a registered baffle position is opened to that smaller area, which
    is what the face would carry with nothing in it. The anode mesh, which the
    record DOES carry, is throttled at no baffle position and so is untouched.
    ``opened`` lists the faces changed, as ``(index, z)``, for the printout.
    """
    z = np.asarray(geometry.z_cm, dtype=float)
    length = np.asarray(geometry.length_cm, dtype=float)
    area = np.asarray(geometry.neutral_volume_cm3, dtype=float) / length
    face_open = np.asarray(geometry.neutral_face_area_cm2, dtype=float).copy()
    positions = np.atleast_1d(
        np.asarray(params["neutral_baffle_positions_cm"], dtype=float)
    )
    opened = []
    for face in range(1, z.size):
        smaller = min(area[face - 1], area[face])
        if face_open[face] >= smaller * (1.0 - 1.0e-9):
            continue
        edge = z[face - 1] + 0.5 * length[face - 1]
        if np.min(np.abs(positions - edge)) <= 0.5 * length[face - 1]:
            face_open[face] = smaller
            opened.append((face, edge))
    return face_open, opened


def _edge_quantile_z(edges_cm, inventory, fraction):
    """Return the z below which ``fraction`` of the inventory sits [cm].

    The record's own convention: the cumulative sum of the per-cell inventory
    carried on the cell EDGES, which starts at zero on the domain's left edge
    and reaches one on its right, linearly interpolated inside the straddling
    cell. It differs from the cell-centre reading by half a cell and is what
    the record's quantiles were reduced with.
    """
    inventory = np.asarray(inventory, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(inventory)))
    cumulative = cumulative / cumulative[-1]
    return float(
        np.interp(fraction, cumulative, np.asarray(edges_cm, dtype=float))
    )


def _band_mean_density(edges_cm, area_cm2, density_cm3, low_cm, high_cm):
    """Return the volume-weighted mean density over ``[low, high]`` [cm^-3].

    Cells are split at the band edges -- the weight of a cell is its own open
    cross section times the length of its overlap with the band -- so the band
    is the interval named rather than whichever cells happen to fall inside it.
    """
    edges_cm = np.asarray(edges_cm, dtype=float)
    overlap = np.clip(
        np.minimum(edges_cm[1:], high_cm) - np.maximum(edges_cm[:-1], low_cm),
        0.0, None,
    )
    weight = overlap * np.asarray(area_cm2, dtype=float)
    return float(np.sum(np.asarray(density_cm3, dtype=float) * weight)
                 / np.sum(weight))


def _production_metrics(inventory, mesh):
    """Return the reduced comparison metrics of one inventory profile."""
    density = np.asarray(inventory, dtype=float) / mesh.volume
    (bore_low, bore_high) = TPMC_PRODUCTION_BORE_BANDS_CM
    return {
        "z50": _edge_quantile_z(mesh.edges, inventory, 0.5),
        "z90": _edge_quantile_z(mesh.edges, inventory, 0.9),
        "z99": _edge_quantile_z(mesh.edges, inventory, 0.99),
        "bore": (
            _band_mean_density(mesh.edges, mesh.area, density, *bore_low)
            / _band_mean_density(mesh.edges, mesh.area, density, *bore_high)
        ),
        "gap": float(density[mesh.gap].mean() / density[mesh.first_chamber]),
        "behind": float(
            np.sum(np.asarray(inventory, dtype=float)[mesh.behind_mesh])
            / np.sum(np.asarray(inventory, dtype=float))
        ),
    }


def _in_bin(value, bounds):
    """Return whether ``value`` lies in ``(low, high)``, either end open."""
    low, high = bounds
    return (low is None or value >= low) and (high is None or value <= high)


def _bin_text(bounds):
    """Return the human reading of a ``(low, high)`` bin."""
    low, high = bounds
    if low is None:
        return f"<= {high:g}"
    if high is None:
        return f">= {low:g}"
    return f"[{low:g}, {high:g}]"


def gate_tpmc_production(record_path, geometry_npz, nx, sgp, vbar_cm_s):
    """Return ``(ok, lines)`` for the production-geometry record comparison.

    The operator is run gap-coupled with a continuous source, at each rung's
    REGISTERED foot, against the record's snapshot at that same time, with the
    neutral baffle's face opened for the comparison only -- the record's
    instrument has no baffle, and the production rows keep it. The builder's
    added inventory is restricted to the record's own domain, which drops the
    plenum cell behind the cathode.

    Seven reductions of the two profiles: the total-variation distance of the
    normalised inventory over the full domain and over the near field
    (``z < TPMC_PRODUCTION_SPLIT_Z_CM``, renormalised there), the z50/z90/z99
    quantiles as ratios to the record's, the bore ratio across the source-bore
    step and the gap ratio behind the anode mesh. The bins in
    :data:`TPMC_PRODUCTION_BINS` are pre-registered; a metric no member's entry
    names is printed and not gated.
    """
    with np.load(record_path, allow_pickle=True) as data:
        record_z = np.asarray(data["z_abs_cm"], dtype=float)
        record_edges = np.asarray(data["z_edges_abs_cm"], dtype=float)
        record_roles = np.array([str(role) for role in data["roles"]])
        record_inventory = np.asarray(
            data["inventory_atoms_per_seed"], dtype=float
        ).mean(axis=0)
        record_times = np.asarray(data["report_times_s"], dtype=float)

    geometry, params = _production_geometry(
        geometry_npz, TPMC_PRODUCTION_GEOMETRY_ES, nx, sgp,
    )
    z = np.asarray(geometry.z_cm, dtype=float)
    length = np.asarray(geometry.length_cm, dtype=float)
    volume = np.asarray(geometry.neutral_volume_cm3, dtype=float)
    offset = z.size - record_z.size
    if offset < 0 or np.max(np.abs(z[offset:] - record_z)) > 1.0e-9:
        raise ValueError(
            f"the record's {record_z.size} cells do not sit on the last "
            f"{record_z.size} of the configuration's {z.size}; the comparison "
            "has no common domain"
        )
    face_open, opened = _baffle_opened_faces(geometry, params)
    gap = record_roles == "gap"
    if not np.any(gap):
        raise ValueError(
            "the record carries no gap-role cell, so the gap ratio the bins "
            "are stated in has no denominator"
        )
    mesh = SimpleNamespace(
        edges=record_edges,
        volume=volume[offset:],
        area=volume[offset:] / length[offset:],
        gap=gap,
        first_chamber=int(np.flatnonzero(gap)[-1] + 1),
        behind_mesh=record_z < record_z[int(np.flatnonzero(gap)[-1] + 1)],
        near=record_z < TPMC_PRODUCTION_SPLIT_Z_CM,
    )
    lines = [
        f"  configuration: ES{TPMC_PRODUCTION_GEOMETRY_ES} operating point, "
        f"nx={nx}, S_gp={sgp:g} sccm, geometry profiles from "
        f"{Path(geometry_npz).name}; {z.size} cells against the record's "
        f"{record_z.size} ({offset} dropped: "
        f"{', '.join(str(role) for role in geometry.cell_role[:offset])})",
        f"  operator: gap-coupled, continuous source, "
        f"{sp3.KNUDSEN_SUBSTEPS_DEFAULT} substeps, vbar {vbar_cm_s:.6g} cm/s "
        f"at Tn {params['Tn_K']:g} K",
        "  baffle faces OPENED for the comparison only (the record's "
        "instrument carries none; the production rows keep them): "
        + (", ".join(f"face {face} at z={edge:.2f} cm"
                     for face, edge in opened) or "none"),
    ]

    ok = True
    for es in TPMC_PRODUCTION_RUNGS:
        foot = sp3.registered_foot_s(es)
        sample = int(np.argmin(np.abs(record_times - foot)))
        if abs(float(record_times[sample]) - foot) > 1.0e-9:
            raise ValueError(
                f"the record carries no snapshot at ES{es}'s registered foot "
                f"t = {foot:g} s; it reports "
                f"{np.round(record_times * 1e3, 6).tolist()} ms"
            )
        reference = record_inventory[sample]
        record = _production_metrics(reference, mesh)
        deposit = _lobe(geometry, params, foot)
        lines.append(
            f"  ES{es} registered foot {foot * 1e3:.2f} ms against the "
            f"record's {float(record_times[sample]) * 1e3:.2f} ms snapshot: "
            f"record z50 {record['z50']:.1f}, z90 {record['z90']:.1f}, z99 "
            f"{record['z99']:.1f} cm, bore {record['bore']:.4f}, gap "
            f"{record['gap']:.4f}, behind the mesh {record['behind']:.4f}"
        )
        z90_by_kappa = []
        for label, kappa in TPMC_PRODUCTION_MEMBERS:
            active = sp3.knudsen_active_mask(geometry.cell_role, True)
            accumulated, _ = sp3.knudsen_spread(
                z, length, volume, face_open, active, deposit, foot,
                vbar_cm_s, kappa=kappa,
            )
            candidate = accumulated[offset:]
            model = _production_metrics(candidate, mesh)
            z90_by_kappa.append((kappa, model["z90"]))
            measured = {
                "TV<342": _total_variation(
                    candidate[mesh.near], reference[mesh.near]
                ),
                "TV full": _total_variation(candidate, reference),
                "z50 ratio": model["z50"] / record["z50"],
                "z90 ratio": model["z90"] / record["z90"],
                "z99 ratio": model["z99"] / record["z99"],
                "bore ratio": model["bore"] / record["bore"],
                "gap ratio": model["gap"] / record["gap"],
            }
            bins = TPMC_PRODUCTION_BINS[label]
            for metric, value in measured.items():
                if metric not in bins:
                    lines.append(
                        f"       (reported, not gated) ES{es} {label} "
                        f"{metric}: {value:.4f}"
                    )
                    continue
                good = _in_bin(value, bins[metric])
                ok = ok and good
                lines.append(
                    f"  [{'ok' if good else 'FAIL'}] ES{es} {label} {metric}: "
                    f"{value:.4f} against bin {_bin_text(bins[metric])}"
                )
            lines.append(
                f"       ES{es} {label} raw: z50 {model['z50']:.1f}, z90 "
                f"{model['z90']:.1f}, z99 {model['z99']:.1f} cm, bore "
                f"{model['bore']:.4f}, gap {model['gap']:.4f}, behind the "
                f"mesh {model['behind']:.4f}"
            )
        # THE NEGATIVE CONTROL, on the same instrument and the same bins it
        # must miss.
        candidate = (
            sp3.spread_matrix(geometry, "ballistic", vbar_cm_s * foot)
            @ deposit
        )[offset:]
        model = _production_metrics(candidate, mesh)
        measured = {
            "TV<342": _total_variation(
                candidate[mesh.near], reference[mesh.near]
            ),
            "TV full": _total_variation(candidate, reference),
            "z90 ratio": model["z90"] / record["z90"],
        }
        for metric, bounds in TPMC_PRODUCTION_NEGATIVE_BINS.items():
            good = _in_bin(measured[metric], bounds)
            ok = ok and good
            lines.append(
                f"  [{'ok' if good else 'FAIL'}] ES{es} "
                f"{TPMC_PRODUCTION_NEGATIVE_MEMBER} {metric}: "
                f"{measured[metric]:.4f} must be {_bin_text(bounds)} -- it is "
                f"the control and has to miss the members' bins"
            )
        # THE ORDERING the bracket asserts, and where the record falls in it.
        # A bracket whose ends do not straddle its reference in reach is not a
        # bracket, whatever its members individually score.
        by_reach = sorted(z90_by_kappa)
        ordered = all(
            lower[1] < upper[1] for lower, upper in zip(by_reach, by_reach[1:])
        )
        by_kappa = dict(z90_by_kappa)
        ok = ok and ordered
        lines.append(
            f"  [{'ok' if ordered else 'FAIL'}] ES{es} z90 rises with kappa: "
            + " < ".join(
                f"{value:.1f} (kappa {kappa:.4g})" for kappa, value in by_reach
            )
        )
        bracketed = (
            by_kappa[sp3.KNUDSEN_KAPPA_SLOW] < record["z90"]
            < by_kappa[sp3.KNUDSEN_KAPPA_REFERENCE]
        )
        ok = ok and bracketed
        lines.append(
            f"  [{'ok' if bracketed else 'FAIL'}] ES{es} the record's z90 "
            f"{record['z90']:.1f} cm lies between the kappa="
            f"{sp3.KNUDSEN_KAPPA_SLOW:g} member "
            f"({by_kappa[sp3.KNUDSEN_KAPPA_SLOW]:.1f} cm) and the reference "
            f"({by_kappa[sp3.KNUDSEN_KAPPA_REFERENCE]:.1f} cm)"
        )
        # THE GAP-CLOSED DISCLOSURE. Reported, never gated: the set that leaves
        # the region behind the mesh empty misses by about the share of the
        # record that sits there, whatever the diffusivity.
        for label, kappa in TPMC_PRODUCTION_MEMBERS:
            active = sp3.knudsen_active_mask(geometry.cell_role, False)
            accumulated, _ = sp3.knudsen_spread(
                z, length, volume, face_open, active, deposit, foot,
                vbar_cm_s, kappa=kappa,
            )
            candidate = accumulated[offset:]
            distance = _total_variation(
                candidate[mesh.near], reference[mesh.near]
            )
            miss = distance - record["behind"]
            lines.append(
                f"       (disclosed, not gated) ES{es} {label} on the "
                f"GAP-CLOSED set: TV<342 {distance:.4f} against the record's "
                f"{record['behind']:.4f} behind the mesh, "
                f"{miss:+.4f} away -- "
                f"{'inside' if abs(miss) <= TPMC_PRODUCTION_GAP_CLOSED_TOL else 'OUTSIDE'}"
                f" +-{TPMC_PRODUCTION_GAP_CLOSED_TOL:g}"
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
             "builder's REGISTERED member must reproduce at raw uint64 "
             "(G0a), and against which the legacy ballistic route is checked "
             "at raw uint64 on the committed legacy fixture (G0b)",
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
    parser.add_argument(
        "--tpmc-production", type=Path, default=None,
        help="G7: a reduced test-particle record of the same puff on the "
             "PRODUCTION geometry, carrying its per-seed inventory profiles "
             "and its own cell edges, to score the registered members against "
             "at each rung's registered foot",
    )
    parser.add_argument(
        "--tpmc-production-geometry-npz", type=Path, default=None,
        help="G7: the .npz carrying the per-cell geometry profiles that "
             "record was produced on",
    )
    parser.add_argument("--tpmc-production-nx", type=int, default=268)
    parser.add_argument("--tpmc-production-sgp", type=float, default=9010.0)
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
        results.insert(0, ("G0 row bit-identity (G0a registered, G0b legacy)",
                           ok, lines))

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

    if args.tpmc_production is not None:
        if args.tpmc_production_geometry_npz is None:
            parser.error(
                "--tpmc-production needs --tpmc-production-geometry-npz: the "
                "comparison runs the operator on the configuration the record "
                "was produced on, and that configuration's per-cell profiles "
                "are named"
            )
        ok, lines = gate_tpmc_production(
            args.tpmc_production, args.tpmc_production_geometry_npz,
            args.tpmc_production_nx, args.tpmc_production_sgp,
            sp3.mean_speed_cm_s(GATE_TN_K, m_He_cgs),
        )
        results.append(("G7 TPMC production comparison", ok, lines))

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
