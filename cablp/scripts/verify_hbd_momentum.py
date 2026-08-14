"""Verify the directed-hot-birth kernel and its momentum closure.

Five checks, each a pass/fail line, over the ``neutral_hot_birth_drift`` flag:

1. KERNEL IDENTITY -- the drift kernel at ``m = 0`` reduces to the isotropic
   one bit for bit when handed the isotropic triple, and to roundoff without
   it; the fast accumulation agrees with a slow reference that is the existing
   ``ballistic_flight_kernels`` loop generalized with ``(mu + m)``; and the
   rows normalize (``landing``, ``residence`` to 1, ``residence_mu`` to 0).
2. CONSTRUCTION GUARD -- the flag without ``neutral_energy`` raises.
3. FLAG-OFF BIT-EXACTNESS -- a two-channel trajectory with the flag ABSENT and
   with it PRESENT-FALSE agree on every saved array at raw ``uint64``.
4. MOMENTUM CLOSURE -- the ion / cold / hot / wall ledger sums to zero on a
   real drifting state, with the flag on AND off.
5. STREAMING READING -- the ``hot_n_flight`` / ``hot_flux_z`` pair is present
   under the flag, absent (zero) without it, and its mean velocity is bounded
   by the launch speed it was built from.

Run from ``<checkout>/cablp`` with ``PYTHONPATH`` pointing at the same tree.
Exit 0 iff every check passes.
"""

import argparse
import sys
import warnings

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.geometry import build_geometry
from cablp.solvers._sim1d.physics.hot_neutrals import (
    BALLISTIC_DIRECTION_SAMPLES,
    ballistic_flight_kernels,
    directed_flight_kernels,
    hot_birth_drift_ratio,
)

FAILURES = []


def check(label, ok, detail=""):
    """Record and print one pass/fail line."""
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAILURES.append(label)
    print(f"  [{status}] {label}{(' -- ' + detail) if detail else ''}")


def slow_reference_kernels(geometry, drift_ratio, samples=BALLISTIC_DIRECTION_SAMPLES):
    """The shipped isotropic loop, generalized with ``(mu + m)``, verbatim.

    This is ``ballistic_flight_kernels`` with exactly one edit -- ``ratio``
    carries the per-cell drift -- so it is the definition the fast accumulating
    builder in ``directed_flight_kernels`` is answerable to. It is O(cells^2 *
    samples) and exists only for this comparison.
    """
    z_edges = np.asarray(geometry.z_edges_cm, dtype=float)
    z_center = np.asarray(geometry.z_cm, dtype=float)
    chord = np.asarray(geometry.Rp_cm, dtype=float)
    m = np.asarray(drift_ratio, dtype=float)
    cells = z_center.size
    count = int(samples)
    mu = -1.0 + (np.arange(count, dtype=float) + 0.5) * (2.0 / count)
    weight = 1.0 / count
    landing = np.zeros((cells, cells), dtype=float)
    residence = np.zeros((cells, cells), dtype=float)
    residence_mu = np.zeros((cells, cells), dtype=float)
    end_fraction = np.zeros(cells, dtype=float)
    for i in range(cells):
        if chord[i] <= 0.0:
            landing[i, i] = 1.0
            residence[i, i] = 1.0
            continue
        ratio = (mu + m[i]) / np.sqrt(1.0 - mu**2)
        z0 = z_center[i]
        z_raw = z0 + chord[i] * ratio
        outside = (z_raw < z_edges[0]) | (z_raw > z_edges[-1])
        z1 = np.clip(z_raw, z_edges[0], z_edges[-1])
        j = np.clip(np.searchsorted(z_edges, z1) - 1, 0, cells - 1)
        np.add.at(landing[i], j, weight)
        end_fraction[i] = float(np.count_nonzero(outside)) * weight
        lo = np.minimum(z0, z1)
        hi = np.maximum(z0, z1)
        overlap = np.clip(
            np.minimum(hi[:, None], z_edges[None, 1:])
            - np.maximum(lo[:, None], z_edges[None, :-1]),
            0.0,
            None,
        )
        span = overlap.sum(axis=1)
        degenerate = span <= 0.0
        if np.any(degenerate):
            overlap[degenerate, :] = 0.0
            overlap[degenerate, i] = 1.0
            span = np.where(degenerate, 1.0, span)
        residence[i] = weight * (overlap / span[:, None]).sum(axis=0)
        residence_mu[i] = weight * (
            mu[:, None] * overlap / span[:, None]
        ).sum(axis=0)
    return landing, residence, end_fraction, residence_mu


def two_channel_config(nx):
    """Return the fa4-class two-channel ``(params, flags)`` pair."""
    params, flags = default_config()
    params = dict(params)
    flags = dict(flags)
    params["nx"] = int(nx)
    flags["neutral_momentum"] = True
    flags["neutral_energy"] = True
    return params, flags


def section_kernel(nx):
    """Check 1: the kernel's reduction, faithfulness and normalization."""
    print("1. KERNEL IDENTITY")
    params, flags = two_channel_config(nx)
    geometry = build_geometry(params, flags)
    cells = np.asarray(geometry.z_cm).size
    isotropic = ballistic_flight_kernels(geometry)
    zero = np.zeros(cells)

    land, res, end, res_mu = directed_flight_kernels(
        geometry, zero, isotropic=isotropic
    )
    exact = (
        np.array_equal(land, isotropic[0])
        and np.array_equal(res, isotropic[1])
        and np.array_equal(end, isotropic[2])
    )
    check(
        "m=0 reduces to the isotropic kernel BIT-EXACTLY (isotropic supplied)",
        exact,
        "landing/residence/end_fraction identical at raw float",
    )

    land0, res0, end0, _ = directed_flight_kernels(geometry, zero)
    worst = max(
        float(np.abs(land0 - isotropic[0]).max()),
        float(np.abs(res0 - isotropic[1]).max()),
        float(np.abs(end0 - isotropic[2]).max()),
    )
    check(
        "m=0 recomputed from scratch agrees with the isotropic kernel",
        worst < 1e-12,
        f"max |delta| = {worst:.3e}",
    )

    worst_ref = 0.0
    for value in (0.25, -0.4, 0.9, 1.6, -2.3):
        drift = np.full(cells, value)
        drift[3] = -drift[3]
        drift[cells // 2] = 0.0
        fast = directed_flight_kernels(geometry, drift)
        ref = slow_reference_kernels(geometry, drift)
        worst_ref = max(
            worst_ref, *(float(np.abs(a - b).max()) for a, b in zip(fast, ref))
        )
    check(
        "fast accumulation matches the slow generalized reference",
        worst_ref < 1e-10,
        f"max |delta| over m in [-2.3, 1.6] = {worst_ref:.3e}",
    )

    worst_norm = 0.0
    worst_mu = 0.0
    for value in (0.0, 0.35, -0.7, 1.4):
        land, res, end, res_mu = directed_flight_kernels(
            geometry, np.full(cells, value)
        )
        worst_norm = max(
            worst_norm,
            float(np.abs(land.sum(axis=1) - 1.0).max()),
            float(np.abs(res.sum(axis=1) - 1.0).max()),
        )
        worst_mu = max(worst_mu, float(np.abs(res_mu.sum(axis=1)).max()))
    check(
        "landing and residence rows normalize to 1 under drift",
        worst_norm < 1e-11,
        f"max |rowsum - 1| = {worst_norm:.3e}",
    )
    check(
        "residence_mu rows sum to 0 (E[mu] = 0 for any drift)",
        worst_mu < 1e-11,
        f"max |rowsum| = {worst_mu:.3e}",
    )

    # The point of the flag: births in a drifting cell are carried downstream.
    # Measured in cm, on the production grid, against the isotropic control.
    z_cm = np.asarray(geometry.z_cm, dtype=float)
    drift = np.full(cells, 0.35)
    _, res_p, _, _ = directed_flight_kernels(geometry, drift)
    _, res_n, _, _ = directed_flight_kernels(geometry, -drift)
    mid = cells // 2
    shift_p = float((res_p[mid] * z_cm).sum() - z_cm[mid])
    shift_n = float((res_n[mid] * z_cm).sum() - z_cm[mid])
    shift_i = float((np.asarray(isotropic[1])[mid] * z_cm).sum() - z_cm[mid])
    check(
        "drift displaces the residence centroid downstream, isotropic does not",
        shift_p > 0.1 and shift_n < -0.1 and abs(shift_i) < 1e-9,
        f"centroid shift +m {shift_p:+.6f} cm, -m {shift_n:+.6f} cm, "
        f"isotropic {shift_i:+.2e} cm",
    )

    # Antisymmetry under m -> -m is exact only where the grid itself is
    # symmetric about the birth cell; the production grid runs 10 to 100 cm
    # cells, so it is checked on a uniform stub where the statement is true.
    class _UniformGrid:
        """A uniform, wide, constant-radius axial grid: geometry only."""

        def __init__(self, count):
            self.z_edges_cm = np.linspace(-1000.0, 1000.0, count + 1)
            self.z_cm = 0.5 * (self.z_edges_cm[:-1] + self.z_edges_cm[1:])
            self.Rp_cm = np.full(count, 20.0)

    stub = _UniformGrid(41)
    _, up, _, _ = directed_flight_kernels(stub, np.full(41, 0.35))
    _, un, _, _ = directed_flight_kernels(stub, np.full(41, -0.35))
    mirror = float(np.abs(up[20] - un[20][::-1]).max())
    check(
        "on a symmetric grid the kernel mirrors exactly under m -> -m",
        mirror < 1e-14,
        f"max |residence(+m) - residence(-m) reversed| = {mirror:.3e}",
    )


def section_guard(nx):
    """Check 2: the presence guard fires at construction."""
    print("2. CONSTRUCTION GUARD")
    params, _ = two_channel_config(nx)
    _, shipped = default_config()
    flags = dict(shipped)
    flags["neutral_hot_birth_drift"] = True
    try:
        LAPDSim1D(input_dict=params, input_flags=flags)
    except ValueError as error:
        check(
            "neutral_hot_birth_drift without neutral_energy raises ValueError",
            "requires neutral_energy" in str(error),
            str(error).split(":")[0],
        )
    else:
        check(
            "neutral_hot_birth_drift without neutral_energy raises ValueError",
            False,
            "constructed silently",
        )

    params2, flags2 = two_channel_config(nx)
    flags2["neutral_hot_birth_drift"] = True
    try:
        LAPDSim1D(input_dict=params2, input_flags=flags2)
        ok, detail = True, "constructs under neutral_energy"
    except ValueError as error:  # pragma: no cover - a regression would show here
        ok, detail = False, str(error)[:80]
    check("the armed flag constructs alongside neutral_energy", ok, detail)


def run_trajectory(flags, params, t_end):
    """Run one short two-channel trajectory and return its saved arrays."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = LAPDSim1D(input_dict=params, input_flags=flags)
        result = sim.run(t_end=t_end)
    return sim, result


def section_bitexact(nx, t_end):
    """Check 3: absent vs present-False is bit-identical."""
    print("3. FLAG-OFF BIT-EXACTNESS")
    params, flags = two_channel_config(nx)
    absent = dict(flags)
    absent.pop("neutral_hot_birth_drift", None)
    present = dict(flags)
    present["neutral_hot_birth_drift"] = False

    _, a = run_trajectory(absent, params, t_end)
    _, b = run_trajectory(present, params, t_end)

    names = [
        name
        for name in dir(a)
        if not name.startswith("_")
        and isinstance(getattr(a, name, None), np.ndarray)
    ]
    mismatched = []
    for name in names:
        left = np.asarray(getattr(a, name))
        right = np.asarray(getattr(b, name))
        if left.shape != right.shape:
            mismatched.append(f"{name}(shape)")
            continue
        if left.dtype.kind != "f":
            if not np.array_equal(left, right):
                mismatched.append(name)
            continue
        if not np.array_equal(left.view(np.uint64), right.view(np.uint64)):
            mismatched.append(name)
    check(
        "flag ABSENT vs PRESENT-FALSE trajectories identical at raw uint64",
        not mismatched,
        f"{len(names)} saved arrays compared"
        + ("" if not mismatched else f"; differing: {mismatched}"),
    )


def ionization_rate_per_neutral(sim, terms):
    """Reconstruct the per-neutral ionization frequency the solver threads in."""
    births = np.asarray(terms["ionization_birth"].n, dtype=float)
    nn = np.maximum(np.asarray(sim.state.nn, dtype=float), sim._floors["nn"])
    return births / nn


def momentum_ledger(sim):
    """Return the four extensive axial-momentum legs [g cm s^-2] of the CX chain.

    ``ion`` and ``cold`` are the fluid rows' inventories -- the ion row rides
    the plasma volume, the cold row the neutral volume through the same
    ``volume_ratio`` its source used -- and ``wall`` is the momentum the hot
    atoms carry onto the column boundary, which the v1 cut absorbs there. The
    three must sum to zero: nothing else in the chain touches axial momentum.
    """
    state = sim.state
    geometry = sim._geometry
    Vp = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    Vn = np.asarray(geometry.neutral_volume_cm3, dtype=float)
    terms = sim.rhs_terms()
    rate = ionization_rate_per_neutral(sim, terms)

    collision = sim.ion_neutral_collision_rhs(state=state)
    cx = sim.neutral_cx_channel_rhs(state=state)
    hot = sim.neutral_hot_channel_rhs(state=state, ionization_rate=rate)
    rates = sim._hot_channel_diagnostics

    ion = float(np.sum((np.asarray(collision.M) + np.asarray(hot.M)) * Vp))
    cold = float(
        np.sum((np.asarray(collision.M_n) + np.asarray(cx.M_n)) * Vn)
    )
    ion_mass_g = sim._ion_mass_g
    p_hot = ion_mass_g * np.asarray(sim.derived.u, dtype=float)
    wall = float(np.sum(np.asarray(rates["hot_wall"], dtype=float) * p_hot * Vp))
    scale = max(abs(ion), abs(cold), abs(wall), 1e-300)
    return ion, cold, wall, scale, rates


def section_momentum(nx, t_end):
    """Check 4: the ion / cold / hot / wall ledger closes, drifted and not."""
    print("4. MOMENTUM CLOSURE")
    params, flags = two_channel_config(nx)
    for armed in (True, False):
        run_flags = dict(flags)
        run_flags["neutral_hot_birth_drift"] = armed
        sim, _ = run_trajectory(run_flags, params, t_end)
        label = "drift ON" if armed else "drift OFF (control)"

        drift = hot_birth_drift_ratio(
            sim.state, floors=sim._floors, ion_mass_g=sim._ion_mass_g
        )
        ion, cold, wall, scale, rates = momentum_ledger(sim)
        residual = ion + cold + wall
        check(
            f"{label}: ion + cold + wall axial momentum sums to zero",
            abs(residual) / scale < 1e-12,
            f"ion {ion:+.6e}, cold {cold:+.6e}, wall {wall:+.6e}, "
            f"residual/scale {residual / scale:+.3e}",
        )

        births = np.asarray(rates["hot_births"], dtype=float)
        fates = (
            np.asarray(rates["hot_wall"], dtype=float)
            + np.asarray(rates["hot_recx"], dtype=float)
            + np.asarray(rates["hot_ionized"], dtype=float)
        )
        worst = float(
            np.abs(births - fates).max() / max(float(np.abs(births).max()), 1e-300)
        )
        check(
            f"{label}: births == wall + recx + ionized per cell",
            worst < 1e-12,
            f"max relative |delta| = {worst:.3e}",
        )
        print(
            f"       drift ratio m in [{drift.min():+.4f}, {drift.max():+.4f}]; "
            f"|m| > 1 in {int(np.count_nonzero(np.abs(drift) > 1.0))} cells"
        )


def section_streaming(nx, t_end):
    """Check 5: the streaming reading is present, gated, and bounded."""
    print("5. STREAMING READING")
    params, flags = two_channel_config(nx)

    off_flags = dict(flags)
    off_flags["neutral_hot_birth_drift"] = False
    sim_off, result_off = run_trajectory(off_flags, params, t_end)
    on_flags = dict(flags)
    on_flags["neutral_hot_birth_drift"] = True
    sim_on, result_on = run_trajectory(on_flags, params, t_end)

    saved = [
        name
        for name in ("hot_n_flight", "hot_flux_z")
        if isinstance(getattr(result_on, name, None), np.ndarray)
    ]
    check(
        "hot_n_flight and hot_flux_z survive the trajectory round trip",
        len(saved) == 2,
        f"saved rows: {saved}",
    )
    check(
        "both rows read exactly zero with the flag off",
        float(np.abs(np.asarray(result_off.hot_n_flight)).max()) == 0.0
        and float(np.abs(np.asarray(result_off.hot_flux_z)).max()) == 0.0,
        "gated on the drift kernel, as documented",
    )

    n_flight = np.asarray(sim_on._hot_channel_diagnostics["hot_n_flight"])
    flux = np.asarray(sim_on._hot_channel_diagnostics["hot_flux_z"])
    live = n_flight > 0.0
    speed = np.abs(flux[live]) / n_flight[live]
    from cablp.solvers._sim1d.physics.hot_neutrals import hot_thermal_speed

    v_hot = hot_thermal_speed(sim_on.derived.Ti, sim_on._ion_mass_g)
    ceiling = float(np.max(v_hot) + np.max(np.abs(sim_on.derived.u)))
    check(
        "mean hot axial velocity stays under v_hot + |u_i|",
        float(np.max(speed)) <= ceiling,
        f"max |flux/n_flight| = {np.max(speed):.4e} cm/s, ceiling "
        f"{ceiling:.4e} cm/s",
    )

    nn_hot = np.asarray(sim_on._hot_channel_diagnostics["nn_hot"])
    Vp = np.asarray(sim_on._geometry.plasma_volume_cm3, dtype=float)
    local = float(np.sum(nn_hot * Vp))
    nonlocal_total = float(np.sum(n_flight * Vp))
    ratio = nonlocal_total / max(local, 1e-300)
    check(
        "residence-resolved inventory agrees with the local standing population",
        0.5 < ratio < 2.0,
        f"sum(hot_n_flight*Vp)/sum(nn_hot*Vp) = {ratio:.6f}",
    )
    mean_speed = float(np.sum(flux * Vp) / max(np.sum(n_flight * Vp), 1e-300))
    print(f"       column-mean directed hot velocity {mean_speed:+.4e} cm/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--t-end", type=float, default=4.0e-4)
    args = parser.parse_args()

    print("verify_hbd_momentum -- directed hot births: kernel and momentum")
    print(f"  nx = {args.nx}, t_end = {args.t_end * 1e3:.3f} ms, "
          f"direction samples = {BALLISTIC_DIRECTION_SAMPLES}")
    print()
    section_kernel(args.nx)
    print()
    section_guard(args.nx)
    print()
    section_bitexact(args.nx, args.t_end)
    print()
    section_momentum(args.nx, args.t_end)
    print()
    section_streaming(args.nx, args.t_end)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
