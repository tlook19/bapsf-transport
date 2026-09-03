"""Build the sp3 shaped initial neutral profile nn0(z) for the foot-shape arm.

INSTRUMENT, not repo physics: this script is the VALUE PRODUCER behind the
solver's ``neutral_initial_profile`` capability, which ships no number of its
own. It writes an ``.npz`` that ``run_m6_point.py --nn0-profile-npz`` hands to
the solver as ``nn0_profile`` / ``nn0_annulus_profile``.

THE CONSTRUCTION (leg 3a of the sp campaign):

    nn0(z) = base + spread( first-flight lobe x throughput x dt_foot )

* ``base`` -- ONE OF TWO, and a result states which:

  - ``--base-from-h5 RUN.h5`` (the verdict-arm base): the ``t = 0`` column
    ``nn`` frame of an existing result, and its ``nn_a`` frame for the
    annulus. For the sp3 verdict arm that h5 is the sp1 fluid REFERENCE, so
    the base IS the reference run's own equilibrated initial profile and the
    arm's SINGLE DELTA is the foot addition on top of it. This matters
    numerically: the REF's equilibrated fill is ~2.6e12 cm^-3, about 7.5x
    BELOW the uniform ``nn0`` convention, so the two bases are not
    interchangeable.
  - the uniform ``resolve_nn0`` value for the stance config (the default).
    Retained for stances that do not start from an equilibrated seed
    (the SS/G-class conducting stances), where there is no reference frame to
    read and the configured scalar is the honest base.

    That value is now the configured ``nn0`` and nothing else. It used to
    have a fallback -- a ``None`` was resolved from a frozen gas-puff lookup
    table -- and the table is RETIRED, so a stance that leaves ``nn0 = None``
    (which is what a stance arming the per-cell ``nn0_profile`` does) is
    REFUSED here with a ValueError naming what ``nn0`` accepts, rather than
    being handed an interpolation on a superseded sccm convention. Such a
    stance has a per-cell profile to read and should be given one through
    ``--base-from-h5``.

  The two are mutually exclusive: passing ``--base-from-h5`` replaces the
  uniform base entirely, and the ledger records which was used.
* the lobe -- the gas puff's first-flight axial deposition, taken from the
  repo's own ``gas_puff_rate_profile`` at the stance's own puff keys
  (``cosine_pipe``, its centre and throw). It is imported, never re-derived,
  so the accumulated shape is by construction the shape the running model
  deposits.
* the throughput -- AS-APPLIED, valves included: the same
  ``4.171431e17 * sccm * valves`` [particles/s] the solver applies, obtained
  from the repo's ``puff_rate`` rather than restated here. The ledger also
  prints the per-valve-nominal half, because both conventions are on the
  campaign record and a quoted number is incomplete without its convention.
* ``dt_foot`` -- the duration of the current foot the model forecloses. A
  DISCLOSED BRACKET, not a fit: ``{2.0e-3, 4.5e-3}`` s.
* ``spread`` -- a 1D kernel carrying the deposited inventory away from the
  lobe over ``dt_foot``. A DISCLOSED BRACKET, not a fit:

      diffusive  gaussian, sigma = sqrt(2 D dt),  D = lambda vbar / 3
      ballistic  top-hat,  half-width = vbar dt

  Both conserve the injected inventory on the grid exactly (asserted).

NOTHING HERE IS FITTED. Every input is hardware-anchored (S_gp, valves, the
puff placement), code-anchored (the lobe, the throughput constant, the base
fill), or literature-boxed (the He-He collision cross section, printed with
its source and overridable from the command line). The two brackets are the
claim's declared spread, and both members are run.

The kernels are stated, not assumed to be right:

* DIFFUSIVE is the random-walk limit -- the foot gas is collisional against
  the background fill, so it spreads as sqrt(t) with the elementary kinetic
  self-diffusion coefficient ``D = lambda vbar / 3``. Its reach is the
  SHORT end of the bracket.
* BALLISTIC is the collisionless limit -- the foot gas free-streams for
  ``dt_foot``, so its support is the interval it can physically reach,
  ``vbar dt``. A TOP-HAT fills that interval flatly, which makes the
  registered reach literally the profile's support and keeps the bracket's
  "short-of-trough vs trough-reaching" reading checkable off the array. It is
  a deliberate idealization: an exactly free-streaming 3D Maxwellian projects
  onto a gaussian of sigma ``t sqrt(kT/m)`` = 0.63 ``vbar t``, a narrower core
  with tails past the top-hat edge. The top-hat is the flatter, more
  spread-out member and so is the honest opposite end of the bracket.

THE STANCE IS OVERRIDABLE. ``--extra k=v`` / ``--extra-flag k=v`` carry
arbitrary ``input_dict`` / ``input_flags`` overrides into the stance the
builder assembles -- the same passthrough ``run_m6_point.py`` gives the RUN,
spelled the same way, so a geometry the arm runs on is a geometry the foot is
built on. They are applied LAST, after the whole stance is assembled, so the
grid, the puff lobe, the spread targets and the zone volumes all see them.
Keys are never screened here: an unknown or misfiled key reaches
``LAPDSim1D``'s own construction-time refusal, unchanged. Array-valued keys
(``plasma_radius_profile_cm`` and friends, one entry per mesh cell) come from
a file rather than a kilobyte of argv, via ``--extra-npz KEY=path.npz:array``.

Usage:

    python scripts/sp3_build_nn0.py --sgp 5200 --two-zone \
        --dt-foot-s 4.5e-3 --kernel ballistic --out scripts/sp3_nn0_b45.npz
"""

import argparse
import json
import math

import numpy as np

from compare_sim1d_es1 import PRODUCTION_NX, PARAM_OVERRIDES, FLAG_OVERRIDES
from run_mechanism_ladder import ES_OPERATING

from cablp.solvers._sim1d import LAPDSim1D, default_config, load_result_hdf5
from cablp.solvers._sim1d.core.config import resolve_nn0
from cablp.solvers._sim1d.physics.neutrals import (
    # The eligibility mask the puff itself uses. Imported rather than
    # restated: the spread must not carry gas into a cell the source is
    # forbidden from reaching (the plenum behind the cathode, the gap, the
    # collector region), and "which cells those are" has exactly one owner.
    _PUFF_ELIGIBLE_ROLES,
    gas_puff_rate_profile,
    neutral_zone_volumes,
    puff_rate,
)
from cablp.constants import kb_cgs, m_He_cgs

#: He-He collision cross section [cm^2], hard-sphere from the Lennard-Jones
#: collision diameter sigma_LJ = 2.551 Angstrom for helium (Hirschfelder,
#: Curtiss & Bird, *Molecular Theory of Gases and Liquids*, the standard
#: viscosity-fitted He parameters; the same value is tabulated in Bird,
#: Stewart & Lightfoot, *Transport Phenomena*, App. E):
#:
#:     sigma_c = pi sigma_LJ^2 = pi (2.551e-8 cm)^2 = 2.044e-15 cm^2
#:
#: LITERATURE-BOXED, never fitted, and overridable with --sigma-hehe-cm2 (or
#: bypassed entirely with --mfp-cm) so the bracket's sensitivity to it is a
#: command-line question rather than a code edit. The diffusive reach goes as
#: sqrt(lambda) and so as sigma^-1/2: a factor 2 in the cross section is a
#: factor 1.4 in reach, which is inside the bracket's own width.
SIGMA_HE_HE_CM2 = 2.044e-15
SIGMA_HE_HE_SOURCE = (
    "hard sphere from the Lennard-Jones He collision diameter "
    "sigma_LJ = 2.551 Angstrom (Hirschfelder/Curtiss/Bird; Bird/Stewart/"
    "Lightfoot App. E): sigma_c = pi sigma_LJ^2"
)

#: The registered sp3 brackets, printed with every ledger so a run always
#: shows which corner of the 2x2 it is.
DT_FOOT_BRACKET_S = (2.0e-3, 4.5e-3)
KERNELS = ("diffusive", "ballistic")

#: Axial band the sp1 response map named as the required-source location
#: [cm]; reported for orientation only, nothing keys off it.
SP1_BAND_Z_CM = (790.0, 1045.0)


def parse_kv_overrides(items):
    """Return ``{key: value}`` parsed from ``k=v`` strings.

    The value rule is ``run_m6_point.py``'s, replicated so that a key spelled
    the same way on the builder and on the run means the same thing: the value
    is read as JSON when it parses (numbers, ``true``/``false``/``null``,
    lists) and kept as the raw string when it does not, which is what lets a
    bare selector name like ``tail_walk`` be written without quoting.

    Keys are NOT screened. A key this stance does not own -- misspelled, or
    filed into the wrong one of the two namespaces -- must reach ``LAPDSim1D``
    and raise there, because the solver's construction-time refusal is the one
    authority on which template owns which key.
    """
    overrides = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(
                f"override {item!r} is not of the form key=value"
            )
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return overrides


def parse_npz_overrides(items):
    """Return ``({key: value}, {key: provenance})`` from ``KEY=path.npz:array``.

    The array-valued route into the stance. A per-mesh-cell profile
    (``plasma_radius_profile_cm``, ``machine_radius_profile_cm``, ...) is
    hundreds of numbers and belongs in a file, not in argv; this reads the
    named array out of the named ``.npz`` and hands it over as a plain Python
    list, exactly the form the config templates take. A 0-d entry (an ``.npz``
    may hold scalars alongside its profiles) becomes the scalar itself, so one
    file can carry a whole geometry -- the machine length and the puff centre
    as readily as the profiles. KEY is the CONFIG key and ``arrayname`` the
    name inside the file; they need not agree, which is what lets one file
    hold several candidate profiles under distinguishing names.

    The script -- not the solver -- does the file I/O, as everywhere else in
    this campaign. The returned provenance records where each value came from
    and its shape, never the values themselves, which would bloat the ledger
    the output ``.npz`` carries.
    """
    values, provenance = {}, {}
    for item in items:
        key, sep, reference = item.partition("=")
        if not sep or not key or not reference:
            raise ValueError(
                f"npz override {item!r} is not of the form "
                "key=path.npz:arrayname"
            )
        path, sep, name = reference.rpartition(":")
        if not sep or not path or not name:
            raise ValueError(
                f"npz override {item!r} names no array: the value must be "
                "path.npz:arrayname"
            )
        with np.load(path, allow_pickle=False) as data:
            if name not in data:
                raise ValueError(
                    f"{path} carries no array {name!r}; it holds "
                    f"{sorted(data.files)}"
                )
            array = np.asarray(data[name])
        values[key] = array.item() if array.ndim == 0 else array.tolist()
        provenance[key] = {
            "source": reference,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    return values, provenance


def stance_config(es, nx, sgp, two_zone, extra_params=None, extra_flags=None):
    """Return (params, flags) for the production stance, as run_model builds it.

    ``extra_params`` and ``extra_flags`` are applied LAST, after the stance is
    fully assembled, so every consumer downstream of this function -- the
    solver geometry, the puff lobe, the spreading kernel's eligible targets,
    the zone volumes and the uniform base -- reads the overridden values. That
    ordering is the point: a geometry override that arrived earlier could be
    overwritten by the stance itself.
    """
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    op = ES_OPERATING[es]
    params["nx"] = nx
    params["S_gp"] = float(sgp)
    params["V_bank"] = op["V_bank"]
    if two_zone:
        flags["neutral_two_zone"] = True
    if extra_params:
        params.update(extra_params)
    if extra_flags:
        flags.update(extra_flags)
    # The geometry is all this script needs from the solver, and the flags
    # that decide it are already set. Equilibration is a start_simulation()
    # behaviour and never runs at construction, so this costs one build.
    return params, flags


def base_profiles_from_h5(path, cells, two_zone):
    """Return ``(base_column, base_annulus)`` from a result's ``t = 0`` frames.

    The base is the run's INITIAL neutral state, so the first saved frame must
    actually be the initial one: a result whose first sample sits at ``t > 0``
    (a ``t_save_start`` beyond zero) is refused rather than silently treated as
    an initial condition it is not.

    ``base_annulus`` is the ``nn_a`` frame under the two-zone closure and
    ``None`` without it. The closure must MATCH: a two-zone build needs an
    ``nn_a`` to read, and a single-field build refuses a two-zone source,
    because collapsing two zones into one field is a modelling choice this
    script does not get to make silently.
    """
    result = load_result_hdf5(path)
    time = np.asarray(result.time, dtype=float)
    if time.size == 0 or time[0] != 0.0:
        raise ValueError(
            f"{path} does not save a t = 0 frame (first saved time "
            f"{time[0] if time.size else 'none'!r}), so its first frame is "
            "not an initial condition; rerun the source with t_save_start = 0"
        )
    base_col = np.array(result.nn[0], dtype=float).reshape(-1)
    if base_col.size != int(cells):
        raise ValueError(
            f"{path} has {base_col.size} cells, this stance has {cells}; the "
            "base profile and the run it seeds must be on the same grid "
            "(check --nx and the geometry keys)"
        )
    source_nn_a = getattr(result, "nn_a", None)
    if two_zone:
        if source_nn_a is None:
            raise ValueError(
                f"--two-zone needs an annulus base, but {path} carries no "
                "nn_a (it is a single-field run). Use a two-zone source, or "
                "drop --two-zone"
            )
        base_ann = np.array(source_nn_a[0], dtype=float).reshape(-1)
        if base_ann.size != int(cells):
            raise ValueError(
                f"{path} nn_a has {base_ann.size} cells, expected {cells}"
            )
    else:
        if source_nn_a is not None:
            raise ValueError(
                f"{path} is a TWO-ZONE run but this build is single-field; "
                "folding its two zones into one neutral field is a modelling "
                "choice, not a conversion. Pass --two-zone"
            )
        base_ann = None
    for label, arr in (("nn", base_col), ("nn_a", base_ann)):
        if arr is None:
            continue
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(
                f"{path} frame 0 {label} must be finite and > 0 (got min "
                f"{float(np.min(arr)):.6g}); the solver refuses such a profile "
                "as an initial condition and so does this"
            )
    return base_col, base_ann


def mean_speed_cm_s(T_K, mass_g):
    """Return the Maxwellian mean speed sqrt(8 k T / (pi m)) [cm/s]."""
    return math.sqrt(8.0 * kb_cgs * float(T_K) / (math.pi * float(mass_g)))


def spread_matrix(geometry, kernel, width_cm):
    """Return the inventory-conserving spreading operator ``W`` [1].

    ``W[i, j]`` is the fraction of the inventory deposited in source cell
    ``j`` that ends up in target cell ``i``. Columns sum to exactly 1 (to
    roundoff), which is what makes the spread conservative on the grid
    regardless of the kernel's shape or the domain's finite extent -- mass
    that would leave the ends is returned to the reachable cells rather than
    deleted, the discrete stand-in for reflecting walls.

    Targets are restricted to the puff's own eligible roles, weighted by cell
    length so a refinement of the grid converges rather than redistributing.

    A source cell from which the kernel reaches NO eligible target gets an
    identically-zero column. That is a real possibility on a coarse grid with
    a narrow kernel, and it is harmless exactly when such a cell carries no
    inventory -- which is the caller's to check, because only the caller knows
    the deposit. Silence here would delete particles; the conservation check in
    :func:`build` is what turns that into a loud failure.
    """
    if not (math.isfinite(width_cm) and width_cm > 0.0):
        raise ValueError(
            "the spreading kernel needs a finite width > 0 (got "
            f"{width_cm!r}); a zero-width spread is the NULL CONTROL and is "
            "handled by dt_foot = 0, which deposits nothing and never reaches "
            "this function"
        )
    z = np.asarray(geometry.z_cm, dtype=float)
    length = np.asarray(geometry.length_cm, dtype=float)
    eligible = np.array(
        [role in _PUFF_ELIGIBLE_ROLES for role in geometry.cell_role], dtype=bool
    )
    dz = z[:, None] - z[None, :]
    if kernel == "diffusive":
        raw = np.exp(-0.5 * (dz / float(width_cm)) ** 2)
    elif kernel == "ballistic":
        raw = (np.abs(dz) <= float(width_cm)).astype(float)
    else:
        raise ValueError(f"kernel must be one of {list(KERNELS)} (got {kernel!r})")
    raw = raw * (length * eligible)[:, None]
    column_sum = raw.sum(axis=0)
    return np.divide(
        raw, column_sum, out=np.zeros_like(raw), where=column_sum > 0.0
    )


def build(args):
    """Return (profiles, ledger) for the requested corner of the bracket."""
    # npz-sourced values first, so an inline --extra can still override any of
    # them -- the same precedence run_m6_point.py gives its file-sourced nn0.
    npz_params, npz_provenance = parse_npz_overrides(args.extra_npz)
    inline_params = parse_kv_overrides(args.extra)
    extra_params = dict(npz_params)
    extra_params.update(inline_params)
    extra_flags = parse_kv_overrides(args.extra_flag)
    params, flags = stance_config(
        args.es, args.nx, args.sgp, args.two_zone,
        extra_params=extra_params, extra_flags=extra_flags,
    )
    if params["gas_type"] != "He":
        raise ValueError(
            "sp3_build_nn0 is helium-only: the collision cross section and the "
            f"thermal speed are both He-He (stance gas_type={params['gas_type']!r})"
        )
    geometry = LAPDSim1D(dict(params), dict(flags)).geometry
    cells = int(geometry.cells)

    V_chamber_all = np.asarray(geometry.neutral_volume_cm3, dtype=float)
    V_col_all, V_ann_all = neutral_zone_volumes(geometry)
    if args.base_from_h5 is None:
        base_col_profile, base_ann_profile = None, None
        base_scalar = float(resolve_nn0(params))
        base_col = np.full(cells, base_scalar)
        base_ann = np.full(cells, base_scalar) if args.two_zone else None
        base_source = "resolve_nn0 at the stance config (shipped convention)"
    else:
        base_col_profile, base_ann_profile = base_profiles_from_h5(
            args.base_from_h5, cells, args.two_zone
        )
        base_col = base_col_profile
        base_ann = base_ann_profile
        base_source = f"t=0 frames of {args.base_from_h5}"
    # The scalar density the mean free path is evaluated at. For a profile
    # base that is the CHAMBER-VOLUME-WEIGHTED MEAN over the whole grid (total
    # neutral particles / total neutral volume, both zones counted) -- the
    # honest single number for a spread that is computed once for the grid.
    # lambda goes as 1/n and the diffusive reach as sqrt(lambda), so the choice
    # is a weak one, and --mfp-cm overrides it outright.
    if base_ann is None:
        base_particles = float(np.sum(base_col * V_chamber_all))
    else:
        base_particles = float(
            np.sum(base_col * V_col_all + base_ann * V_ann_all)
        )
    base_density = base_particles / float(np.sum(V_chamber_all))
    Tn_K = float(args.tn_k if args.tn_k is not None else params["Tn_K"])
    vbar = mean_speed_cm_s(Tn_K, m_He_cgs)

    # --- the deposited inventory -------------------------------------------
    # gas_puff_rate_profile returns [cm^-3 s^-1] against the CHAMBER volume,
    # normalized so that sum(rate * V_chamber) is the whole throughput.
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
    V_chamber = V_chamber_all
    deposited = rate * V_chamber * float(args.dt_foot_s)  # particles per cell

    throughput_applied = puff_rate(params["S_gp"], params["gas_puff_valves"], 1.0)
    throughput_nominal = puff_rate(params["S_gp"], 1, 1.0)
    injected_applied = throughput_applied * float(args.dt_foot_s)
    injected_nominal = throughput_nominal * float(args.dt_foot_s)

    # --- the spread ---------------------------------------------------------
    if args.mfp_cm is not None:
        mfp = float(args.mfp_cm)
        mfp_source = "explicit --mfp-cm"
    else:
        # Like-particle mean free path: the sqrt(2) is the relative-speed
        # correction for a test particle moving through its own species.
        mfp = 1.0 / (math.sqrt(2.0) * base_density * float(args.sigma_hehe_cm2))
        mfp_source = (
            f"1 / (sqrt(2) n sigma) at n = the base's chamber-volume-weighted "
            f"mean = {base_density:.6g} cm^-3, "
            f"sigma = {args.sigma_hehe_cm2:.6g} cm^2 [{SIGMA_HE_HE_SOURCE}]"
        )
    D_cm2_s = mfp * vbar / 3.0
    if args.kernel == "diffusive":
        width = math.sqrt(2.0 * D_cm2_s * float(args.dt_foot_s))
        width_label = "gaussian sigma = sqrt(2 D dt)"
    else:
        width = vbar * float(args.dt_foot_s)
        width_label = "top-hat half-width = vbar dt"

    if width == 0.0:
        # THE NULL CONTROL (dt_foot = 0): nothing was deposited, so there is
        # nothing to spread and no kernel is built. Short-circuited rather
        # than passed through a zero-width kernel, which is undefined.
        accumulated = np.zeros(cells, dtype=float)
    else:
        spread = spread_matrix(geometry, args.kernel, width)
        # A source cell the kernel cannot carry out of is only safe if it
        # holds nothing; otherwise the spread would delete its particles
        # silently.
        unreachable = spread.sum(axis=0) <= 0.0
        if np.any(deposited[unreachable] != 0.0):
            raise ValueError(
                "the spreading kernel reaches no eligible cell from a source "
                "cell that carries deposited gas, so the spread would delete "
                f"it: {int(np.count_nonzero(deposited[unreachable] != 0.0))} "
                f"such cells at kernel width {width:.6g} cm. Widen the kernel "
                "or refine the grid"
            )
        accumulated = spread @ deposited  # particles per cell after spreading

    grid_in = float(deposited.sum())
    grid_out = float(accumulated.sum())
    conservation_rel = abs(grid_out - grid_in) / max(grid_in, 1e-300)
    assert conservation_rel < 1e-12, (
        f"spreading kernel lost inventory: in {grid_in:.9e}, out "
        f"{grid_out:.9e}, rel {conservation_rel:.3e}"
    )

    # --- routing into the neutral field(s) ---------------------------------
    V_col, V_ann = V_col_all, V_ann_all
    add_col = np.zeros(cells, dtype=float)
    add_ann = np.zeros(cells, dtype=float) if args.two_zone else None
    if not args.two_zone:
        # One chamber-mean neutral field: the whole cell volume holds it.
        add_col = accumulated / V_chamber
    elif args.zone == "chamber":
        # Radially well-mixed: both zones rise by the same density, so the
        # particle count is conserved cell by cell (V_col + V_ann = V_chamber).
        add_col = accumulated / V_chamber
        add_ann = accumulated / V_chamber
    elif args.zone == "annulus":
        # The shipped first-flight routing: the pipe enters at the wall, so
        # the puff feeds the annulus first and annulus-free cells fall back to
        # the column -- exactly as neutral_source_sink_rhs routes it.
        has_ann = V_ann > 0.0
        add_ann = np.where(has_ann, accumulated / np.maximum(V_ann, 1e-300), 0.0)
        add_col = np.where(has_ann, 0.0, accumulated / np.maximum(V_col, 1e-300))
    elif args.zone == "column":
        add_col = accumulated / np.maximum(V_col, 1e-300)
    else:
        raise ValueError(f"unknown --zone {args.zone!r}")

    nn0_profile = base_col + add_col
    nn0_annulus_profile = None if add_ann is None else base_ann + add_ann

    # Round-trip particle check: the densities written out must hold the
    # inventory the spread produced, in whichever zone(s) it was routed to.
    if args.two_zone:
        held = float(np.sum(add_col * V_col + add_ann * V_ann))
    else:
        held = float(np.sum(add_col * V_chamber))
    routing_rel = abs(held - grid_in) / max(grid_in, 1e-300)
    assert routing_rel < 1e-10, (
        f"zone routing lost inventory: spread {grid_in:.9e}, held {held:.9e}, "
        f"rel {routing_rel:.3e}"
    )

    ledger = {
        "es": args.es,
        "nx": args.nx,
        "cells": cells,
        "S_gp_sccm": float(params["S_gp"]),
        "gas_puff_valves": int(params["gas_puff_valves"]),
        "gas_puff_profile": params["gas_puff_profile"],
        "gas_puff_z_cm": float(params["gas_puff_z_cm"]),
        "gas_puff_throw_cm": float(params["gas_puff_throw_cm"]),
        "two_zone": bool(args.two_zone),
        "zone": args.zone if args.two_zone else "single-field",
        "base_kind": "uniform" if args.base_from_h5 is None else "profile_h5",
        "base_source": base_source,
        "base_from_h5": args.base_from_h5,
        "base_mean_density_cm3": base_density,
        "base_column_min_cm3": float(np.min(base_col)),
        "base_column_max_cm3": float(np.max(base_col)),
        "base_annulus_min_cm3": (
            None if base_ann is None else float(np.min(base_ann))
        ),
        "base_annulus_max_cm3": (
            None if base_ann is None else float(np.max(base_ann))
        ),
        "Tn_K": Tn_K,
        "vbar_cm_s": vbar,
        "dt_foot_s": float(args.dt_foot_s),
        "dt_foot_bracket_s": list(DT_FOOT_BRACKET_S),
        "kernel": args.kernel,
        "kernel_bracket": list(KERNELS),
        "kernel_width_cm": width,
        "kernel_width_label": width_label,
        "mfp_cm": mfp,
        "mfp_source": mfp_source,
        "sigma_hehe_cm2": float(args.sigma_hehe_cm2),
        "D_cm2_s": D_cm2_s,
        "throughput_as_applied_per_s": throughput_applied,
        "throughput_nominal_per_valve_per_s": throughput_nominal,
        "injected_atoms_as_applied": injected_applied,
        "injected_atoms_nominal_per_valve": injected_nominal,
        "grid_inventory_before_spread": grid_in,
        "grid_inventory_after_spread": grid_out,
        "spread_conservation_rel": conservation_rel,
        "zone_routing_conservation_rel": routing_rel,
    }
    # The stance overrides, PRESENCE-GATED: an invocation that overrides
    # nothing writes exactly the ledger it always wrote, and so exactly the
    # same output bytes. The inline and file-sourced overrides are recorded
    # separately because they are separate provenance -- a literal value given
    # on the command line, versus a named array in a named file (recorded as
    # source, shape and dtype; the values themselves would bloat the ledger
    # the output npz carries, and they are already IN the output's own grid).
    if inline_params:
        ledger["extra_params"] = inline_params
    if npz_provenance:
        ledger["extra_params_from_npz"] = npz_provenance
    if extra_flags:
        ledger["extra_flags"] = extra_flags
    return (nn0_profile, nn0_annulus_profile, base_col, base_ann,
            geometry, ledger)


def print_ledger(
    nn0_profile, nn0_annulus_profile, base_col, base_ann, geometry, ledger
):
    """Print the inventory ledger and the profile's headline numbers."""
    z = np.asarray(geometry.z_cm, dtype=float)
    print("=== sp3 shaped-nn0 construction ===")
    print(
        f"stance: ES{ledger['es']} nx={ledger['nx']} cells={ledger['cells']} "
        f"S_gp={ledger['S_gp_sccm']:g} sccm x {ledger['gas_puff_valves']} valves, "
        f"puff {ledger['gas_puff_profile']} at z={ledger['gas_puff_z_cm']:g} cm, "
        f"throw {ledger['gas_puff_throw_cm']:g} cm"
    )
    # Presence-gated exactly as the ledger entries are: an invocation that
    # overrides nothing prints what it always printed.
    for label, key in (
        ("stance overrides [params]", "extra_params"),
        ("stance overrides [flags]", "extra_flags"),
    ):
        if key in ledger:
            print(f"{label}: " + ", ".join(
                f"{k}={v!r}" for k, v in sorted(ledger[key].items())
            ))
    if "extra_params_from_npz" in ledger:
        print("stance overrides [params from npz]: " + ", ".join(
            f"{k}<-{spec['source']} {spec['dtype']}{tuple(spec['shape'])}"
            for k, spec in sorted(ledger["extra_params_from_npz"].items())
        ))
    print(
        f"base [{ledger['base_kind']}]: {ledger['base_source']}; "
        f"column {ledger['base_column_min_cm3']:.6g}..."
        f"{ledger['base_column_max_cm3']:.6g}"
        + (
            ""
            if ledger["base_annulus_min_cm3"] is None
            else f", annulus {ledger['base_annulus_min_cm3']:.6g}..."
            f"{ledger['base_annulus_max_cm3']:.6g}"
        )
        + f"; chamber-mean {ledger['base_mean_density_cm3']:.6g} cm^-3"
    )
    print(
        f"bracket corner: dt_foot={ledger['dt_foot_s']:.6g} s of "
        f"{ledger['dt_foot_bracket_s']}, kernel={ledger['kernel']!r} of "
        f"{ledger['kernel_bracket']}"
    )
    print(
        f"thermal: Tn={ledger['Tn_K']:g} K, vbar={ledger['vbar_cm_s']:.6g} cm/s; "
        f"mfp={ledger['mfp_cm']:.6g} cm ({ledger['mfp_source']}); "
        f"D={ledger['D_cm2_s']:.6g} cm^2/s"
    )
    print(
        f"kernel width: {ledger['kernel_width_cm']:.6g} cm "
        f"({ledger['kernel_width_label']})"
    )
    print("--- inventory ledger ---")
    print(
        f"throughput as-applied (valves in): "
        f"{ledger['throughput_as_applied_per_s']:.6g} /s"
        f"  ->  S_gp x dt_foot = {ledger['injected_atoms_as_applied']:.6g} atoms"
    )
    print(
        f"throughput per-valve-nominal:      "
        f"{ledger['throughput_nominal_per_valve_per_s']:.6g} /s"
        f"  ->  S_gp x dt_foot = {ledger['injected_atoms_nominal_per_valve']:.6g} atoms"
    )
    print(
        f"deposited on grid (first-flight lobe): "
        f"{ledger['grid_inventory_before_spread']:.6g} atoms"
    )
    print(
        f"after spreading:                       "
        f"{ledger['grid_inventory_after_spread']:.6g} atoms  "
        f"(rel err {ledger['spread_conservation_rel']:.3e})"
    )
    print(
        f"held by the written densities:         "
        f"{ledger['grid_inventory_after_spread']:.6g} atoms  "
        f"(rel err {ledger['zone_routing_conservation_rel']:.3e})"
    )
    print("--- profile (enhancement is CELL-LOCAL: profile / base at that cell) ---")
    for label, prof, base in (
        ("column" if ledger["two_zone"] else "nn", nn0_profile, base_col),
        ("annulus", nn0_annulus_profile, base_ann),
    ):
        if prof is None:
            continue
        ratio = np.asarray(prof, dtype=float) / np.asarray(base, dtype=float)
        print(
            f"{label:>8}: min {float(np.min(prof)):.6g}  max "
            f"{float(np.max(prof)):.6g}  mean {float(np.mean(prof)):.6g} cm^-3 "
            f"(peak enhancement x{float(np.max(ratio)):.4g})"
        )
        for z_band in SP1_BAND_Z_CM:
            i = int(np.argmin(np.abs(z - z_band)))
            print(
                f"          z={z[i]:8.2f} cm (sp1 band {z_band:g}): "
                f"{float(prof[i]):.6g} cm^-3 = base x {float(ratio[i]):.6g} "
                f"(base {float(base[i]):.6g})"
            )


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build the sp3 shaped initial neutral profile npz."
    )
    p.add_argument("--es", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--nx", type=int, default=PRODUCTION_NX)
    p.add_argument("--sgp", type=float, required=True,
                   help="gas puff level [sccm]; must match the verdict run's")
    p.add_argument("--two-zone", action="store_true",
                   help="build for the neutral_two_zone closure (writes an "
                        "annulus profile as well); must match the run's")
    p.add_argument("--zone", choices=("chamber", "annulus", "column"),
                   default="chamber",
                   help="two-zone routing of the accumulated inventory. "
                        "'chamber' (default) raises both zones by the same "
                        "density -- the radially well-mixed convention, whose "
                        "justification is that the free-molecular zone "
                        "exchange time is ms-class, comparable to the foot. "
                        "'annulus' is the un-mixed extreme and reproduces the "
                        "shipped first-flight routing exactly; 'column' is the "
                        "fully-mixed-inward extreme. The routing is a "
                        "DISCLOSED DEGENERACY (sp2), so a result states it")
    p.add_argument("--base-from-h5", default=None,
                   help="read the base profile from an existing result's t=0 "
                        "frames (column nn, and nn_a for the annulus) instead "
                        "of the uniform resolve_nn0 convention. THE VERDICT-ARM "
                        "BASE: with the sp1 fluid reference here, the base is "
                        "that run's own equilibrated initial profile and the "
                        "arm's single delta is the foot addition on top of it. "
                        "Mutually exclusive with the uniform base by "
                        "construction -- passing this replaces it, and the "
                        "ledger records which was used")
    p.add_argument("--dt-foot-s", type=float, default=4.5e-3,
                   help=f"foot duration [s]; registered bracket "
                        f"{DT_FOOT_BRACKET_S} (default: the pedestal-floor "
                        f"end). 0 is the explicit NULL CONTROL: no foot "
                        f"addition at all, so the output is the base itself")
    p.add_argument("--kernel", choices=KERNELS, default="diffusive",
                   help="spreading kernel; registered bracket is both members "
                        "(default: the short-reach end)")
    p.add_argument("--sigma-hehe-cm2", type=float, default=SIGMA_HE_HE_CM2,
                   help="He-He collision cross section [cm^2] setting the mean "
                        f"free path (default {SIGMA_HE_HE_CM2:g}: "
                        f"{SIGMA_HE_HE_SOURCE})")
    p.add_argument("--mfp-cm", type=float, default=None,
                   help="mean free path [cm] stated directly, bypassing the "
                        "cross section; for auditing the diffusive reach "
                        "against an independently quoted lambda")
    p.add_argument("--tn-k", type=float, default=None,
                   help="neutral temperature [K]; default is the stance Tn_K")
    p.add_argument("--selfcheck", action="store_true",
                   help="NULL-CONSTRUCTION SELF-CHECK: requires --base-from-h5 "
                        "and --dt-foot-s 0. After writing, re-reads the npz "
                        "and the source h5 and asserts the written profiles "
                        "equal the h5's t=0 frames EXACTLY, in every zone. It "
                        "checks the plumbing end to end -- that the base "
                        "really came from the h5, survived the routing, and "
                        "round-tripped through the file -- rather than the "
                        "arithmetic of adding zero")
    p.add_argument("--extra", nargs="*", default=(),
                   help="additional k=v input_dict (params) overrides, "
                        "JSON-parsed values, spelled exactly as "
                        "run_m6_point.py --extra spells them. Applied AFTER "
                        "the whole stance is assembled, so the geometry keys "
                        "(Lm, collector_length_cm, gas_puff_z_cm, ...) take "
                        "effect everywhere this script reads the config. "
                        "Unknown or misfiled keys are NOT screened here -- "
                        "they raise at LAPDSim1D construction, which is the "
                        "one authority on which template owns a key")
    p.add_argument("--extra-flag", nargs="*", default=(),
                   help="additional k=v input_flags overrides (JSON-parsed), "
                        "as run_m6_point.py --extra-flag; same ordering and "
                        "same no-screening rule as --extra")
    p.add_argument("--extra-npz", nargs="*", default=(),
                   help="array-valued params override, KEY=path.npz:arrayname "
                        "-- reads the named array out of the named .npz and "
                        "files it under KEY, e.g. "
                        "plasma_radius_profile_cm=scripts/g1_profiles.npz:"
                        "plasma_radius_profile_cm_off. This is the route for "
                        "the per-mesh-cell geometry profiles, which are "
                        "hundreds of numbers and do not belong in argv; a 0-d "
                        "entry in the .npz becomes the scalar itself, so one "
                        "file can carry a whole geometry. Applied BEFORE "
                        "--extra, so an inline value still overrides a "
                        "file-sourced one")
    p.add_argument("--out", required=True, help="output .npz path")
    args = p.parse_args(argv)

    if args.dt_foot_s < 0.0 or not math.isfinite(args.dt_foot_s):
        p.error("--dt-foot-s must be finite and >= 0 (0 is the null control)")
    if args.zone != "chamber" and not args.two_zone:
        p.error("--zone is a two-zone routing choice; pass --two-zone or drop it")
    if args.selfcheck and (
        args.base_from_h5 is None or args.dt_foot_s != 0.0
    ):
        p.error(
            "--selfcheck is the null construction: it requires "
            "--base-from-h5 and --dt-foot-s 0"
        )

    nn0_profile, nn0_annulus_profile, base_col, base_ann, geometry, ledger = (
        build(args)
    )
    print_ledger(
        nn0_profile, nn0_annulus_profile, base_col, base_ann, geometry, ledger
    )

    payload = {
        "nn0_profile": nn0_profile,
        "z_cm": np.asarray(geometry.z_cm, dtype=float),
        "provenance": json.dumps(ledger, sort_keys=True),
    }
    if nn0_annulus_profile is not None:
        payload["nn0_annulus_profile"] = nn0_annulus_profile
    np.savez(args.out, **payload)
    print(f"saved {args.out}")

    if args.selfcheck:
        source = load_result_hdf5(args.base_from_h5)
        with np.load(args.out, allow_pickle=False) as written:
            checks = [(
                "column",
                np.asarray(written["nn0_profile"], dtype=float),
                np.asarray(source.nn[0], dtype=float),
            )]
            if "nn0_annulus_profile" in written:
                checks.append((
                    "annulus",
                    np.asarray(written["nn0_annulus_profile"], dtype=float),
                    np.asarray(source.nn_a[0], dtype=float),
                ))
            ok = True
            for label, got, want in checks:
                same = np.array_equal(got, want)
                ok = ok and same
                print(
                    f"null-construction self-check [{label}]: "
                    f"{'EXACT' if same else 'DIFFERS'} vs "
                    f"{args.base_from_h5} t=0 ({got.size} cells, "
                    f"max |delta| {float(np.max(np.abs(got - want))):.3e})"
                )
        print("SELFCHECK:", "PASS" if ok else "FAIL")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
