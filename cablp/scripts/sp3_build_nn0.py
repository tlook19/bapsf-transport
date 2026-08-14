"""Build the sp3 shaped initial neutral profile nn0(z) for the foot-shape arm.

INSTRUMENT, not repo physics: this script is the VALUE PRODUCER behind the
solver's ``neutral_initial_profile`` capability, which ships no number of its
own. It writes an ``.npz`` that ``run_m6_point.py --nn0-profile-npz`` hands to
the solver as ``nn0_profile`` / ``nn0_annulus_profile``.

THE CONSTRUCTION (leg 3a of the sp campaign):

    nn0(z) = base + spread( first-flight lobe x throughput x dt_foot )

* ``base`` -- the shipped-convention uniform fill the stance would otherwise
  start from, i.e. exactly what ``resolve_nn0`` returns for the stance config.
* the lobe -- the gas puff's first-flight axial deposition, taken from the
  repo's own ``gas_puff_rate_profile`` at the stance's own puff keys
  (``cosine_pipe``, its centre and throw). It is imported, never re-derived,
  so the accumulated shape is by construction the shape the running model
  deposits.
* the throughput -- AS-APPLIED, valves included: the same
  ``4.477962e17 * sccm * valves`` [particles/s] the solver applies, obtained
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

from cablp.solvers._sim1d import LAPDSim1D, default_config
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
from cablp.vars._cons import kb_cgs, m_He_cgs

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


def stance_config(es, nx, sgp, two_zone):
    """Return (params, flags) for the production stance, as run_model builds it."""
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    op = ES_OPERATING[es]
    params["nx"] = nx
    params["S_gp"] = float(sgp)
    params["V_bank"] = op["V_bank"]
    if two_zone:
        flags["neutral_two_zone"] = True
    # The geometry is all this script needs from the solver, and the flags
    # that decide it are already set. Equilibration is a start_simulation()
    # behaviour and never runs at construction, so this costs one build.
    return params, flags


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
    """
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
    if not np.all(column_sum > 0.0):
        raise ValueError(
            "the spreading kernel reaches no eligible cell from at least one "
            "source cell; widen the kernel or check the geometry"
        )
    return raw / column_sum


def build(args):
    """Return (profiles, ledger) for the requested corner of the bracket."""
    params, flags = stance_config(args.es, args.nx, args.sgp, args.two_zone)
    if params["gas_type"] != "He":
        raise ValueError(
            "sp3_build_nn0 is helium-only: the collision cross section and the "
            f"thermal speed are both He-He (stance gas_type={params['gas_type']!r})"
        )
    geometry = LAPDSim1D(dict(params), dict(flags)).geometry
    cells = int(geometry.cells)

    base = float(resolve_nn0(params, flags))
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
        end=0,
    )
    V_chamber = np.asarray(geometry.neutral_volume_cm3, dtype=float)
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
        mfp = 1.0 / (math.sqrt(2.0) * base * float(args.sigma_hehe_cm2))
        mfp_source = (
            f"1 / (sqrt(2) n sigma) at n = base = {base:.6g} cm^-3, "
            f"sigma = {args.sigma_hehe_cm2:.6g} cm^2 [{SIGMA_HE_HE_SOURCE}]"
        )
    D_cm2_s = mfp * vbar / 3.0
    if args.kernel == "diffusive":
        width = math.sqrt(2.0 * D_cm2_s * float(args.dt_foot_s))
        width_label = "gaussian sigma = sqrt(2 D dt)"
    else:
        width = vbar * float(args.dt_foot_s)
        width_label = "top-hat half-width = vbar dt"

    spread = spread_matrix(geometry, args.kernel, width)
    accumulated = spread @ deposited  # particles per cell after spreading

    grid_in = float(deposited.sum())
    grid_out = float(accumulated.sum())
    conservation_rel = abs(grid_out - grid_in) / max(grid_in, 1e-300)
    assert conservation_rel < 1e-12, (
        f"spreading kernel lost inventory: in {grid_in:.9e}, out "
        f"{grid_out:.9e}, rel {conservation_rel:.3e}"
    )

    # --- routing into the neutral field(s) ---------------------------------
    V_col, V_ann = neutral_zone_volumes(geometry)
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

    nn0_profile = base + add_col
    nn0_annulus_profile = None if add_ann is None else base + add_ann

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
        "base_nn0_cm3": base,
        "base_source": "resolve_nn0 at the stance config (shipped convention)",
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
    return (nn0_profile, nn0_annulus_profile, geometry, ledger)


def print_ledger(nn0_profile, nn0_annulus_profile, geometry, ledger):
    """Print the inventory ledger and the profile's headline numbers."""
    z = np.asarray(geometry.z_cm, dtype=float)
    print("=== sp3 shaped-nn0 construction ===")
    print(
        f"stance: ES{ledger['es']} nx={ledger['nx']} cells={ledger['cells']} "
        f"S_gp={ledger['S_gp_sccm']:g} sccm x {ledger['gas_puff_valves']} valves, "
        f"puff {ledger['gas_puff_profile']} at z={ledger['gas_puff_z_cm']:g} cm, "
        f"throw {ledger['gas_puff_throw_cm']:g} cm"
    )
    print(
        f"base fill: {ledger['base_nn0_cm3']:.6g} cm^-3 ({ledger['base_source']})"
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
    print("--- profile ---")
    for label, prof in (
        ("column" if ledger["two_zone"] else "nn", nn0_profile),
        ("annulus", nn0_annulus_profile),
    ):
        if prof is None:
            continue
        print(
            f"{label:>8}: min {float(np.min(prof)):.6g}  max "
            f"{float(np.max(prof)):.6g}  mean {float(np.mean(prof)):.6g} cm^-3 "
            f"(base x {float(np.max(prof)) / ledger['base_nn0_cm3']:.4g} at peak)"
        )
        for z_band in SP1_BAND_Z_CM:
            i = int(np.argmin(np.abs(z - z_band)))
            print(
                f"          z={z[i]:8.2f} cm (sp1 band {z_band:g}): "
                f"{float(prof[i]):.6g} cm^-3 = base x "
                f"{float(prof[i]) / ledger['base_nn0_cm3']:.6g}"
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
    p.add_argument("--dt-foot-s", type=float, default=4.5e-3,
                   help=f"foot duration [s]; registered bracket "
                        f"{DT_FOOT_BRACKET_S} (default: the pedestal-floor end)")
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
    p.add_argument("--out", required=True, help="output .npz path")
    args = p.parse_args(argv)

    if args.dt_foot_s <= 0.0 or not math.isfinite(args.dt_foot_s):
        p.error("--dt-foot-s must be finite and > 0")
    if args.zone != "chamber" and not args.two_zone:
        p.error("--zone is a two-zone routing choice; pass --two-zone or drop it")

    nn0_profile, nn0_annulus_profile, geometry, ledger = build(args)
    print_ledger(nn0_profile, nn0_annulus_profile, geometry, ledger)

    payload = {
        "nn0_profile": nn0_profile,
        "z_cm": np.asarray(geometry.z_cm, dtype=float),
        "provenance": json.dumps(ledger, sort_keys=True),
    }
    if nn0_annulus_profile is not None:
        payload["nn0_annulus_profile"] = nn0_annulus_profile
    np.savez(args.out, **payload)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
