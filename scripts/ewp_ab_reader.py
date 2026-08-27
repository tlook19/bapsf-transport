"""Saved-trajectory reader for the wall-branch momentum partition A/B.

WHY THIS EXISTS. ``scripts/ewp_demo_run.py`` reports the partition's numbers
at a CONSTRUCT-TIME state -- one short window, one evolved snapshot, read off a
live solver. That is enough for the mechanism's arithmetic and not enough for
the question the A/B actually asks, which is what the partition does to a
production trajectory over a declared plateau. This reader answers the second
question from SAVED artifacts: it re-runs nothing, and reads one or two
``sim1d-hdf5-v1`` files plus each run's own saved config.

ONE IMPLEMENTATION, TWO READERS. Every formula here that the demonstration
also uses is IMPORTED from it, never restated:

  * ``neutral_wall_partition_survival`` (``physics.sources``) is the in-solver
    authority for the optical depth, the mean free path and the survival
    weight, and both scripts call it;
  * ``zone_volumes``, ``wall_branch_pool``, ``partition_split``,
    ``drag_family_keys`` and ``drag_family_integral`` come from
    ``ewp_demo_run``, which owns them.

The demonstration reads a live ``TermRHS``; this reader reads a saved HDF5
term group. The shared functions take their rows through a ``get_row``
accessor, so the difference is the accessor and nothing else.

WHAT IT REPORTS, per save and averaged over each declared time window:

  A  the ion-neutral drag-family momentum ledger, as booked -- the family's
     |dM| size (the demonstration's own quantity) and each channel's signed
     contribution on the ion, column-neutral and annulus-neutral rows;
  B  the wall-branch pool ``nu_wall * M_n_a``, and, for a file whose config
     ARMS the partition, the re-routed part of that pool as a fraction of the
     pool and of the whole drag family;
  C  the optical depth, mean free path and survival at the SAVED per-cell
     ``nn_a`` and the run's own cross section, reported as the kernel BRACKET
     the survival docstring discloses, not as a point value;
  D  column and annulus neutral drift, their Mach numbers against the 300-K
     mean thermal speed, and the far-band means; in pair mode the on/off Mach
     ratio and the far-band drift delta;
  E  in pair mode, the ion-drift response: the p50 delta, the largest cell
     delta with its index and cell role, and the p50/p41 fall on both arms.

WINDOW AND BAND DEFAULTS are the item-51 far-band ones, taken from
``scripts/legc_momentum_budget.py``: the 15--19.5 ms plateau, and the far band
spanning ports p41 (z = 1428.55 cm) to p50 (z = 1716.1 cm). ``--window`` is
repeatable and overrides the plateau.

FAILURE IS LOUD. A file whose config does not run the two-moment closure, or
an ``--arm`` file whose config does not arm the partition flag, is refused by
name rather than silently reported on. So is a window that contains no save.

Usage (from <checkout>/cablp, PYTHONPATH set to that same cablp):

    python scripts/ewp_ab_reader.py --base BASE.h5 [--arm ARM.h5]
                                    [--window 15,19.5] [--json OUT.json]
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cablp.solvers._sim1d import load_result_hdf5  # noqa: E402
from cablp.solvers._sim1d.physics.sources import (  # noqa: E402
    neutral_wall_partition_survival,
)
from cablp.vars._cons import kb_cgs, m_He_cgs  # noqa: E402

from ewp_demo_run import (  # noqa: E402
    WALL_BRANCH_T_K,
    drag_family_integral,
    drag_family_keys,
    partition_split,
    wall_branch_pool,
    zone_volumes,
)

#: The item-51 plateau window [ms], from legc_momentum_budget.py --plateau.
DEFAULT_WINDOW_MS = (15.0, 19.5)

#: Far-band edges [cm]: the scored ports p41 and p50 of
#: legc_momentum_budget.py PORTS, which is where item 51's falling far-end ion
#: drift is read.
PORT_P41_CM = 1428.55
PORT_P50_CM = 1716.1

#: Cell roles that are 0D boundary cells rather than 1D interior ones. A
#: u = M/n reading in one of these is a boundary artifact, not a finding.
END_ROLES = frozenset({"plenum", "collector", "end", "obstruction"})

#: The radial closure the partition is defined on. Anything else cannot carry
#: an annulus momentum row for the wall branch to act on.
REQUIRED_RADIAL_CLOSURE = "kinetic_two_moment"

_PARTITION_FLAG = "neutral_wall_momentum_partition"
_SIGMA_KEY = "neutral_wall_partition_sigma_hehe_cm2"


# ----------------------------------------------------------------------
# Survival kernel bracket
# ----------------------------------------------------------------------
def survival_kernel_bracket(tau):
    """Return the three-member survival kernel family at optical depth ``tau``.

    ``physics.sources.neutral_wall_partition_survival`` implements ONE member
    of a family and says so: ``2 E_3(tau)`` is the SURFACE-EMITTED single
    flight, in which every atom starts at one face and crosses the full
    annulus thickness. The wall-bound pool is volume-distributed instead, at a
    mean depth of about ``d/2``, so the survival number is conditional on the
    kernel. The two other natural members named in that docstring are

        volume-averaged single flight   (2/tau) [1/3 - E_4(tau)]
        diffusive                       1 / (1 + 3 tau / 4)

    and the implemented member is the most retention-biased of the three. This
    function reports all three so the re-routed fraction is read as a bracket.
    Every member is exactly 1 at ``tau = 0``, the free-molecular limit.

    Returns a dict keyed ``surface_2E3`` / ``volume_avg`` / ``diffusive``.
    """
    tau = np.asarray(tau, dtype=float)
    surface = 2.0 * expn(3, tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        volume = np.where(
            tau > 0.0,
            (2.0 / np.maximum(tau, 1e-300)) * (1.0 / 3.0 - expn(4, tau)),
            1.0,
        )
    diffusive = 1.0 / (1.0 + 0.75 * tau)
    return {
        "surface_2E3": surface,
        "volume_avg": volume,
        "diffusive": diffusive,
    }


# ----------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------
def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_arm(path, role):
    """Load one saved run and refuse it loudly if it cannot answer the A/B.

    ``role`` is ``"base"`` or ``"arm"``; only the arm is required to have the
    partition flag armed.
    """
    path = Path(path)
    _require(path.exists(), f"--{role} file does not exist: {path}")
    result = load_result_hdf5(path)
    params = dict(getattr(result, "params", {}) or {})
    flags = dict(getattr(result, "flags", {}) or {})

    closure = params.get("neutral_momentum_radial")
    _require(
        closure == REQUIRED_RADIAL_CLOSURE,
        f"--{role} {path.name}: this reader requires "
        f"neutral_momentum_radial={REQUIRED_RADIAL_CLOSURE!r} (the only "
        f"radial closure carrying the annulus momentum row M_n_a the "
        f"wall-branch pool is built from); the file's saved config has "
        f"neutral_momentum_radial={closure!r}",
    )
    armed = bool(flags.get(_PARTITION_FLAG, False))
    if role == "arm":
        _require(
            armed,
            f"--arm {path.name}: the arm of an A/B must have the "
            f"{_PARTITION_FLAG} flag armed in its saved config; this file has "
            f"{_PARTITION_FLAG}={flags.get(_PARTITION_FLAG, False)!r}. Pass "
            f"the partitioned run as --arm and the reference as --base.",
        )
    _require(
        params.get("gas_type", "He") == "He",
        f"--{role} {path.name}: the He-He wall-branch algebra is helium-only; "
        f"the file's saved config has gas_type="
        f"{params.get('gas_type')!r}",
    )

    for field in ("nn_a", "M_n_a", "u_n", "u_n_a", "u", "n", "M", "time"):
        _require(
            getattr(result, field, None) is not None,
            f"--{role} {path.name}: the saved trajectory carries no {field!r} "
            f"array; the run did not evolve it (absence means never "
            f"persisted, never zero)",
        )

    geometry = SimpleNamespace(
        Rp_cm=np.asarray(result.Rp_cm, dtype=float),
        Rm_cm=np.asarray(result.Rm_cm, dtype=float),
        plasma_volume_cm3=np.asarray(result.plasma_volume_cm3, dtype=float),
        neutral_volume_cm3=np.asarray(result.neutral_volume_cm3, dtype=float),
    )
    return SimpleNamespace(
        role=role,
        path=path,
        result=result,
        params=params,
        flags=flags,
        armed=armed,
        geometry=geometry,
        z=np.asarray(result.z_cm, dtype=float),
        active=np.asarray(result.plasma_active, dtype=bool),
        cell_role=np.asarray(result.cell_role),
        time=np.asarray(result.time, dtype=float),
        sigma=None,
        sigma_source=None,
    )


def resolve_sigma(arm, override, partner_sigma):
    """Settle which He-He cross section this file's optical depth is read at.

    A flag-OFF run cannot carry one: the solver refuses ``sigma`` without the
    flag (an inert control), so the reference arm of an A/B has no cross
    section of its own and must borrow the partitioned arm's, or be given one
    on the command line. Which one was used is reported.
    """
    if override is not None:
        return float(override), "--sigma-hehe-cm2 (command line)"
    own = arm.params.get(_SIGMA_KEY)
    if own is not None:
        return float(own), f"saved config {_SIGMA_KEY}"
    if partner_sigma is not None:
        return float(partner_sigma), "the --arm file's saved config"
    raise ValueError(
        f"--{arm.role} {arm.path.name}: no He-He cross section is available. "
        f"The file's config has {_SIGMA_KEY}=None (the solver forbids it with "
        f"{_PARTITION_FLAG} off), no --arm supplied one, and no "
        f"--sigma-hehe-cm2 was given. The solver carries no default: there is "
        f"no boxed value. Supply --sigma-hehe-cm2."
    )


def window_indices(arm, t0_ms, t1_ms):
    """Return the save indices inside ``[t0, t1]`` ms, or refuse loudly."""
    t = arm.time
    idx = np.flatnonzero((t >= t0_ms * 1e-3) & (t <= t1_ms * 1e-3))
    _require(
        idx.size > 0,
        f"--{arm.role} {arm.path.name}: window {t0_ms:g}-{t1_ms:g} ms "
        f"contains no save; the file spans {t.min() * 1e3:.6g}-"
        f"{t.max() * 1e3:.6g} ms over {t.size} saves. Pass --window inside "
        f"that span.",
    )
    return idx


# ----------------------------------------------------------------------
# Per-save quantities
# ----------------------------------------------------------------------
def _row_getter(save_index):
    """Return a ``get_row`` reading one save out of a saved term group."""
    def get_row(term_fields, field):
        arr = term_fields.get(field)
        return None if arr is None else np.asarray(arr[save_index], dtype=float)
    return get_row


def per_save_quantities(arm, sigma, far_band):
    """Compute every per-save scalar this reader reports, for one file.

    Returns a dict of arrays indexed by save. Nothing here is window-aware:
    the windows average these afterwards, so a window change costs no reread.
    """
    res = arm.result
    geom = arm.geometry
    Vc, Va = zone_volumes(geom)
    vbar_300 = np.sqrt(8.0 * WALL_BRANCH_T_K * kb_cgs / (np.pi * m_He_cgs))

    nn_a = np.asarray(res.nn_a, dtype=float)
    M_n_a = np.asarray(res.M_n_a, dtype=float)
    u_n = np.asarray(res.u_n, dtype=float)
    u_n_a = np.asarray(res.u_n_a, dtype=float)
    saves = nn_a.shape[0]

    terms = res.rhs_terms
    keys = drag_family_keys(terms, get_row=_row_getter(0))

    out = {
        "pool_dyn": np.zeros(saves),
        "retained_dyn": np.zeros(saves),
        "drag_family_dyn": np.zeros(saves),
        "tau_min": np.zeros(saves),
        "tau_med": np.zeros(saves),
        "tau_max": np.zeros(saves),
        "mfp_med_cm": np.zeros(saves),
        "nn_a_med": np.zeros(saves),
        "u_n_mean_kms": np.zeros(saves),
        "u_n_a_mean_kms": np.zeros(saves),
        "u_n_far_kms": np.zeros(saves),
        "u_n_a_far_kms": np.zeros(saves),
    }
    for member in survival_kernel_bracket(np.zeros(1)):
        out[f"survival_{member}_med"] = np.zeros(saves)
        out[f"reroute_frac_{member}"] = np.zeros(saves)
    channels = {
        k: {"M": np.zeros(saves), "M_n": np.zeros(saves),
            "M_n_a": np.zeros(saves)}
        for k in keys
    }

    live = Va > 0.0
    band = arm.active & (arm.z >= far_band[0]) & (arm.z <= far_band[1])
    _require(
        band.any(),
        f"--{arm.role} {arm.path.name}: the far band "
        f"{far_band[0]:g}-{far_band[1]:g} cm contains no active cell.",
    )

    for s in range(saves):
        dens = nn_a[s]
        survival, tau, mfp = neutral_wall_partition_survival(geom, dens, sigma)
        _nu_wall, pool = wall_branch_pool(geom, M_n_a[s], m_He_cgs)
        _absorbed, retained = partition_split(pool, survival)
        out["pool_dyn"][s] = float(np.sum(np.abs(pool) * Va))
        out["retained_dyn"][s] = float(np.sum(np.abs(retained) * Va))

        get_row = _row_getter(s)
        out["drag_family_dyn"][s] = drag_family_integral(
            terms, keys, Vc, Va, get_row=get_row
        )
        for k in keys:
            channels[k]["M"][s] = float(
                np.sum(np.asarray(terms[k]["M"][s], dtype=float) * Vc)
            )
            channels[k]["M_n"][s] = float(
                np.sum(np.asarray(terms[k]["M_n"][s], dtype=float) * Vc)
            )
            row_a = terms[k].get("M_n_a")
            channels[k]["M_n_a"][s] = (
                0.0 if row_a is None
                else float(np.sum(np.asarray(row_a[s], dtype=float) * Va))
            )

        sel = live & (dens > 0.0)
        out["tau_min"][s] = float(tau[sel].min()) if sel.any() else np.nan
        out["tau_med"][s] = float(np.median(tau[sel])) if sel.any() else np.nan
        out["tau_max"][s] = float(tau[sel].max()) if sel.any() else np.nan
        out["mfp_med_cm"][s] = (
            float(np.median(mfp[sel])) if sel.any() else np.nan
        )
        out["nn_a_med"][s] = float(np.median(dens[live])) if live.any() else np.nan

        bracket = survival_kernel_bracket(tau)
        for member, weight in bracket.items():
            out[f"survival_{member}_med"][s] = (
                float(np.median(weight[sel])) if sel.any() else np.nan
            )
            kept = pool - weight * pool
            out[f"reroute_frac_{member}"][s] = float(
                np.sum(np.abs(kept) * Va)
            ) / max(out["pool_dyn"][s], 1e-300)

        out["u_n_mean_kms"][s] = float(np.mean(u_n[s][arm.active])) / 1.0e5
        out["u_n_a_mean_kms"][s] = float(np.mean(u_n_a[s][arm.active])) / 1.0e5
        out["u_n_far_kms"][s] = float(np.mean(u_n[s][band])) / 1.0e5
        out["u_n_a_far_kms"][s] = float(np.mean(u_n_a[s][band])) / 1.0e5

    out["Ma_n_far"] = out["u_n_far_kms"] * 1.0e5 / vbar_300
    out["Ma_n_a_far"] = out["u_n_a_far_kms"] * 1.0e5 / vbar_300
    return SimpleNamespace(
        keys=keys, channels=channels, scalars=out, band=band,
        vbar_300_cms=vbar_300, Vc=Vc, Va=Va,
    )


def port_index(arm, z_cm):
    """Index of the cell nearest ``z_cm``, the legc port convention."""
    return int(np.argmin(np.abs(arm.z - z_cm)))


def is_end_cell(arm, index):
    """True when ``index`` is a 0D boundary cell or a domain end."""
    role = str(arm.cell_role[index])
    return role in END_ROLES or index in (0, arm.z.size - 1)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def _mean(values, idx):
    return float(np.mean(np.asarray(values)[idx]))


def _win_label(window):
    return f"{window[0]:g}-{window[1]:g} ms"


def report(base, arm, windows, dump):
    """Print every block and return the structured record."""
    arms = [base] + ([arm] if arm is not None else [])
    record = {
        "windows_ms": [list(w) for w in windows],
        "far_band_cm": [PORT_P41_CM, PORT_P50_CM],
        "arms": {},
    }

    print("=" * 78)
    print("ewp_ab_reader: wall-branch momentum partition, saved-trajectory A/B")
    print("=" * 78)
    for a in arms:
        print(f"  --{a.role:4s} {a.path}")
        print(
            f"         cells {a.z.size}  saves {a.time.size}  "
            f"t {a.time.min() * 1e3:.4g}-{a.time.max() * 1e3:.4g} ms  "
            f"{_PARTITION_FLAG}={a.armed}"
        )
        print(
            f"         sigma_HeHe = {a.sigma:.6e} cm^2  "
            f"[source: {a.sigma_source}]"
        )
    print(
        f"  far band z = {PORT_P41_CM:.2f}-{PORT_P50_CM:.2f} cm "
        f"(ports p41-p50); windows: "
        + ", ".join(_win_label(w) for w in windows)
    )
    print(
        f"  vbar(300 K) = {base.q.vbar_300_cms / 1.0e5:.6f} km/s; every "
        f"ledger integral is volume-weighted [dyn]"
    )
    print()

    # ---- A: drag-family ledger -----------------------------------------
    print("-" * 78)
    print("A. ION-NEUTRAL DRAG-FAMILY MOMENTUM LEDGER  [dyn]")
    print("-" * 78)
    for a in arms:
        print(f"  {a.role}: family terms = {a.q.keys}")
    print(
        "  family |dM|dV is the demonstration's own quantity (|M_n| on Vc "
        "plus |M_n_a| on Va);\n  the channel columns are SIGNED integrals of "
        "each row."
    )
    print()
    print(
        f"  {'run':5s} {'window':12s} {'family |dM|dV':>15s} "
        f"{'sum M dVc':>14s} {'sum M_n dVc':>14s} {'sum M_n_a dVa':>14s}"
    )
    for a in arms:
        for w in windows:
            idx = a.windows[tuple(w)]
            tot_M = sum(_mean(c["M"], idx) for c in a.q.channels.values())
            tot_Mn = sum(_mean(c["M_n"], idx) for c in a.q.channels.values())
            tot_Mna = sum(
                _mean(c["M_n_a"], idx) for c in a.q.channels.values()
            )
            print(
                f"  {a.role:5s} {_win_label(w):12s} "
                f"{_mean(a.q.scalars['drag_family_dyn'], idx):15.6e} "
                f"{tot_M:14.5e} {tot_Mn:14.5e} {tot_Mna:14.5e}"
            )
    print()
    print(f"  per channel (signed, window mean):")
    print(
        f"  {'run':5s} {'window':12s} {'channel':32s} "
        f"{'M dVc':>13s} {'M_n dVc':>13s} {'M_n_a dVa':>13s}"
    )
    for a in arms:
        for w in windows:
            idx = a.windows[tuple(w)]
            for k, c in a.q.channels.items():
                print(
                    f"  {a.role:5s} {_win_label(w):12s} {k:32s} "
                    f"{_mean(c['M'], idx):13.4e} {_mean(c['M_n'], idx):13.4e} "
                    f"{_mean(c['M_n_a'], idx):13.4e}"
                )
    print()

    # ---- B: wall-branch pool and re-routing ----------------------------
    print("-" * 78)
    print("B. WALL-BRANCH POOL nu_wall*M_n_a AND RE-ROUTING  [dyn]")
    print("-" * 78)
    print(
        "  The re-routed columns are reported ONLY for a file whose config "
        "arms the\n  partition: on a flag-off run the same arithmetic would "
        "be a counterfactual,\n  not a booking."
    )
    print()
    print(
        f"  {'run':5s} {'window':12s} {'pool |dV|':>15s} "
        f"{'re-routed':>15s} {'% of pool':>11s} {'% of family':>13s}"
    )
    for a in arms:
        for w in windows:
            idx = a.windows[tuple(w)]
            pool = _mean(a.q.scalars["pool_dyn"], idx)
            if not a.armed:
                print(
                    f"  {a.role:5s} {_win_label(w):12s} {pool:15.6e} "
                    f"{'--':>15s} {'--':>11s} {'--':>13s}   (flag off)"
                )
                continue
            kept = _mean(a.q.scalars["retained_dyn"], idx)
            fam = _mean(a.q.scalars["drag_family_dyn"], idx)
            print(
                f"  {a.role:5s} {_win_label(w):12s} {pool:15.6e} "
                f"{kept:15.6e} {100.0 * kept / max(pool, 1e-300):11.3f} "
                f"{100.0 * kept / max(fam, 1e-300):13.3f}"
            )
    print()

    # ---- C: optical depth, mfp, survival bracket -----------------------
    print("-" * 78)
    print("C. OPTICAL DEPTH, MEAN FREE PATH AND SURVIVAL BRACKET")
    print("-" * 78)
    print(
        "  tau = (Rm-Rp)*nn_a*sigma at the SAVED per-cell nn_a. The solver "
        "implements the\n  surface-emitted member 2*E_3(tau); the other two "
        "are the family named in\n  physics.sources.neutral_wall_partition_"
        "survival. Read the bracket, not a point."
    )
    print()
    print(
        f"  {'run':5s} {'window':12s} {'nn_a med':>11s} {'mfp med':>9s} "
        f"{'tau min':>9s} {'tau med':>9s} {'tau max':>9s} "
        f"{'2E3':>8s} {'volavg':>8s} {'diffus':>8s}"
    )
    for a in arms:
        for w in windows:
            idx = a.windows[tuple(w)]
            s = a.q.scalars
            print(
                f"  {a.role:5s} {_win_label(w):12s} "
                f"{_mean(s['nn_a_med'], idx):11.4e} "
                f"{_mean(s['mfp_med_cm'], idx):9.4g} "
                f"{_mean(s['tau_min'], idx):9.5f} "
                f"{_mean(s['tau_med'], idx):9.5f} "
                f"{_mean(s['tau_max'], idx):9.5f} "
                f"{_mean(s['survival_surface_2E3_med'], idx):8.5f} "
                f"{_mean(s['survival_volume_avg_med'], idx):8.5f} "
                f"{_mean(s['survival_diffusive_med'], idx):8.5f}"
            )
    print()
    print(
        f"  re-routed fraction of the pool under each kernel "
        f"(window mean, all files):"
    )
    print(
        f"  {'run':5s} {'window':12s} {'2E3':>10s} {'volavg':>10s} "
        f"{'diffus':>10s}"
    )
    for a in arms:
        for w in windows:
            idx = a.windows[tuple(w)]
            s = a.q.scalars
            print(
                f"  {a.role:5s} {_win_label(w):12s} "
                f"{100 * _mean(s['reroute_frac_surface_2E3'], idx):9.3f}% "
                f"{100 * _mean(s['reroute_frac_volume_avg'], idx):9.3f}% "
                f"{100 * _mean(s['reroute_frac_diffusive'], idx):9.3f}%"
            )
    print()

    # ---- D: neutral drift and Mach --------------------------------------
    print("-" * 78)
    print("D. NEUTRAL DRIFT AND MACH NUMBER vs vbar(300 K)")
    print("-" * 78)
    print(
        f"  Ma = u_n / vbar(300 K), vbar = "
        f"{base.q.vbar_300_cms / 1.0e5:.6f} km/s. 'all' averages the active "
        f"cells;\n  'far' averages the p41-p50 band."
    )
    print()
    print(
        f"  {'run':5s} {'window':12s} {'u_n all':>10s} {'u_n far':>10s} "
        f"{'u_n_a all':>10s} {'u_n_a far':>10s} {'Ma far':>9s} "
        f"{'Ma_a far':>9s}"
    )
    for a in arms:
        for w in windows:
            idx = a.windows[tuple(w)]
            s = a.q.scalars
            print(
                f"  {a.role:5s} {_win_label(w):12s} "
                f"{_mean(s['u_n_mean_kms'], idx):10.3e} "
                f"{_mean(s['u_n_far_kms'], idx):10.3e} "
                f"{_mean(s['u_n_a_mean_kms'], idx):10.3e} "
                f"{_mean(s['u_n_a_far_kms'], idx):10.3e} "
                f"{_mean(s['Ma_n_far'], idx):9.3e} "
                f"{_mean(s['Ma_n_a_far'], idx):9.3e}"
            )
    if arm is not None:
        print()
        print("  pair deltas (arm vs base), far band, km/s:")
        print(
            f"  {'':5s} {'window':12s} {'Ma far ratio':>14s} "
            f"{'Ma_a far ratio':>16s} {'d u_n far':>13s} {'d u_n_a far':>13s}"
        )
        for w in windows:
            ib = base.windows[tuple(w)]
            ia = arm.windows[tuple(w)]
            mb = _mean(base.q.scalars["Ma_n_far"], ib)
            ma = _mean(arm.q.scalars["Ma_n_far"], ia)
            mb_a = _mean(base.q.scalars["Ma_n_a_far"], ib)
            ma_a = _mean(arm.q.scalars["Ma_n_a_far"], ia)
            du = (
                _mean(arm.q.scalars["u_n_far_kms"], ia)
                - _mean(base.q.scalars["u_n_far_kms"], ib)
            )
            du_a = (
                _mean(arm.q.scalars["u_n_a_far_kms"], ia)
                - _mean(base.q.scalars["u_n_a_far_kms"], ib)
            )
            print(
                f"  {'':5s} {_win_label(w):12s} "
                f"{ma / mb if mb else np.nan:14.6f} "
                f"{ma_a / mb_a if mb_a else np.nan:16.6f} "
                f"{du:13.5e} {du_a:13.5e}"
            )
    print()

    # ---- E: ion drift response ------------------------------------------
    if arm is not None:
        print("-" * 78)
        print("E. ION DRIFT RESPONSE (pair mode)  [km/s]")
        print("-" * 78)
        i41 = port_index(base, PORT_P41_CM)
        i50 = port_index(base, PORT_P50_CM)
        _require(
            port_index(arm, PORT_P41_CM) == i41
            and port_index(arm, PORT_P50_CM) == i50,
            "base and arm do not share a mesh: the p41/p50 port cells differ, "
            "so a per-cell difference is not defined.",
        )
        print(
            f"  ports: p41 cell {i41} (z = {base.z[i41]:.2f} cm), "
            f"p50 cell {i50} (z = {base.z[i50]:.2f} cm)"
        )
        print(
            f"  {'window':12s} {'d u_i(p50)':>13s} {'max|d u_i|':>13s} "
            f"{'cell':>6s} {'z cm':>9s} {'role':>10s} "
            f"{'base p50/p41':>13s} {'arm p50/p41':>12s}"
        )
        for w in windows:
            ib = base.windows[tuple(w)]
            ia = arm.windows[tuple(w)]
            ub = np.asarray(base.result.u, dtype=float)[ib].mean(axis=0)
            ua = np.asarray(arm.result.u, dtype=float)[ia].mean(axis=0)
            du = (ua - ub) / 1.0e5
            imax = int(np.argmax(np.abs(du)))
            flag = "  <- END CELL (u = M/n boundary artifact, not a finding)" \
                if is_end_cell(base, imax) else ""
            print(
                f"  {_win_label(w):12s} {du[i50]:13.5e} "
                f"{du[imax]:13.5e} {imax:6d} {base.z[imax]:9.2f} "
                f"{str(base.cell_role[imax]):>10s} "
                f"{ub[i50] / ub[i41] if ub[i41] else np.nan:13.5f} "
                f"{ua[i50] / ua[i41] if ua[i41] else np.nan:12.5f}"
                + flag
            )
            print(
                f"  {'':12s} base u_i(p41) {ub[i41] / 1e5:9.4f}  "
                f"u_i(p50) {ub[i50] / 1e5:9.4f}   |   "
                f"arm u_i(p41) {ua[i41] / 1e5:9.4f}  "
                f"u_i(p50) {ua[i50] / 1e5:9.4f}"
            )
        print()

    # ---- structured record ----------------------------------------------
    for a in arms:
        record["arms"][a.role] = {
            "path": str(a.path),
            "armed": bool(a.armed),
            "sigma_hehe_cm2": a.sigma,
            "sigma_source": a.sigma_source,
            "cells": int(a.z.size),
            "saves": int(a.time.size),
            "drag_family_terms": list(a.q.keys),
            "vbar_300K_cm_s": a.q.vbar_300_cms,
            "time_s": a.time.tolist(),
            "per_save": {k: np.asarray(v).tolist()
                         for k, v in a.q.scalars.items()},
            "per_save_channels": {
                k: {row: np.asarray(vals).tolist()
                    for row, vals in rows.items()}
                for k, rows in a.q.channels.items()
            },
            "window_means": {
                _win_label(w): {
                    k: _mean(v, a.windows[tuple(w)])
                    for k, v in a.q.scalars.items()
                }
                for w in windows
            },
        }
    if dump is not None:
        Path(dump).write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"structured dump -> {dump}")
    return record


def _parse_window(text):
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--window takes 't0,t1' in milliseconds; got {text!r}"
        )
    t0, t1 = (float(p) for p in parts)
    if not t1 > t0:
        raise argparse.ArgumentTypeError(
            f"--window {text!r}: t1 must exceed t0"
        )
    return (t0, t1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="reference run (HDF5)")
    ap.add_argument("--arm", default=None, help="partitioned run (HDF5)")
    ap.add_argument(
        "--window", action="append", type=_parse_window, default=None,
        help="time window 't0,t1' in ms, repeatable; default is the item-51 "
             "plateau 15,19.5",
    )
    ap.add_argument(
        "--sigma-hehe-cm2", type=float, default=None,
        help="override the He-He momentum-transfer cross section [cm^2] for "
             "BOTH files; by default each file uses its own saved config and "
             "a flag-off base borrows the arm's",
    )
    ap.add_argument("--json", default=None, help="write a structured dump")
    args = ap.parse_args()

    windows = args.window or [DEFAULT_WINDOW_MS]
    base = load_arm(args.base, "base")
    arm = load_arm(args.arm, "arm") if args.arm else None

    arm_sigma = None
    if arm is not None:
        arm.sigma, arm.sigma_source = resolve_sigma(
            arm, args.sigma_hehe_cm2, None
        )
        arm_sigma = arm.sigma
    base.sigma, base.sigma_source = resolve_sigma(
        base, args.sigma_hehe_cm2, arm_sigma
    )

    far_band = (PORT_P41_CM, PORT_P50_CM)
    for a in [base] + ([arm] if arm is not None else []):
        a.q = per_save_quantities(a, a.sigma, far_band)
        a.windows = {tuple(w): window_indices(a, *w) for w in windows}

    report(base, arm, windows, args.json)


if __name__ == "__main__":
    main()
