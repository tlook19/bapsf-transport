"""Run the conducting-phase window, optionally with the coverage closure on.

Applies ``covbuild_conducting_phase.toml`` as a DELTA over the campaign stance
(``compare_sim1d_es1.PARAM_OVERRIDES`` / ``FLAG_OVERRIDES``) and runs it,
printing the breakdown instant so the window can be checked against the
measured >= 4.5 ms conducting phase.

Shakedown instrument for the coverage-closure build: direction only, never
scored. ``--coverage f0[,r]`` turns the closure on for the A/B arm.
"""

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, ProgressPrinter1D, default_config
from cablp.solvers._sim1d.results.io import save_result_hdf5

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES
from run_mechanism_ladder import ES_OPERATING


DELTA_TOML = Path(__file__).resolve().parent / "covbuild_conducting_phase.toml"


def build_config(nx, coverage=None, extra=None):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    op = ES_OPERATING[1]
    params.update({
        "nx": nx,
        "V_bank": op["V_bank"],
        "cathode_solver_model": "current_driven",
        "beam_deposition_model": "csda",
        "beam_anomalous_model": "quasilinear",
        "cathode_emission_profile": "gaussian",
        "cathode_warming_model": "power_balance",
        "T_s": op["Ts_standby_K"],
        "cathode_Ts_base_K": op["Ts_standby_K"],
        "cathode_heat_capacity_J_per_K": 120.0,
        "cathode_emissivity": 0.7,
        "phi_wf": 2.869,
        "cathode_surface_model": "ads_des",
        "cathode_phiwf_clean_eV": 2.809,
        "cathode_cleaning_sigma_cm2": 3.5e-16,
        "cathode_cleaning_E_th_eV": 20.0,
        "Te_birth_ionization": "floor",
        "gas_puff_mode": "square",
        "cathode_sample_smoothing": "presheath",
    })
    delta = tomllib.loads(DELTA_TOML.read_text())
    params.update(delta.get("params", {}))
    flags.update(delta.get("flags", {}))
    if coverage is not None:
        f0, r = coverage
        params["coverage_initial_fraction"] = f0
        params["coverage_growth_rate_per_s"] = r
        flags["coverage_closure"] = True
        # The coverage deficit partitions nn alone, so the solver refuses a
        # single mean En over a concentrated gas. neutral_energy (and the hot
        # internal wall that requires it) became config defaults at the R2a
        # fold-in, so the coverage arm has to name the layout it composes with.
        flags["neutral_energy"] = False
        flags["neutral_hot_internal_wall"] = False
    if extra:
        params.update(extra)
    return params, flags


def _deposition_profile_split(sim, result):
    """Print the axial split of the beam's deposited power."""
    diag = getattr(result, "cathode_diagnostics", None) or {}
    banks = [
        "beam_heat_coulomb_W",
        "beam_heat_anomalous_W",
        "beam_heat_secondary_W",
        "beam_heat_terminal_W",
    ]
    present = [b for b in banks if b in diag]
    if not present:
        print("deposition split: no CSDA beam-heat banks saved")
        return
    heat = sum(np.asarray(diag[b], dtype=float) for b in present)
    if heat.ndim != 2:
        print("deposition split: unexpected bank shape")
        return
    geom = sim.geometry
    z = np.cumsum(np.asarray(geom.length_cm, dtype=float)) - 0.5 * np.asarray(
        geom.length_cm, dtype=float
    )
    z_mid = 0.5 * (z[0] + z[-1])
    beyond = z > z_mid
    times = np.asarray(result.time, dtype=float)
    total = heat.sum(axis=1)
    live = total > 0.0
    print(f"deposition split: mid-machine at z = {z_mid:.1f} cm")
    if not np.any(live):
        print("  beam deposited no power over the window")
        return
    frac = np.zeros_like(total)
    frac[live] = heat[live][:, beyond].sum(axis=1) / total[live]
    print(
        f"  window-mean fraction beyond mid-machine: "
        f"{float(np.mean(frac[live])):.6f}"
    )
    for want in (0.0, 0.25, 0.5, 0.75, 1.0):
        i = min(int(want * (times.size - 1)), times.size - 1)
        print(
            f"  t={times[i] * 1e3:9.4f} ms  beyond_mid={frac[i]:.6f}  "
            f"total_beam_heat={total[i]:.6e} W"
        )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=120)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--coverage", default=None,
                   help="f_cov0[,r] -- turns the coverage closure on")
    p.add_argument("--extra", nargs="*", default=(),
                   help="additional k=v param overrides (JSON-parsed)")
    p.add_argument("--save-h5", required=True)
    args = p.parse_args(argv)

    coverage = None
    if args.coverage is not None:
        parts = args.coverage.split(",")
        coverage = (float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0)
    extra = {}
    for kv in args.extra:
        k, v = kv.split("=", 1)
        try:
            extra[k] = json.loads(v)
        except json.JSONDecodeError:
            extra[k] = v

    params, flags = build_config(args.nx, coverage=coverage, extra=extra)
    sim = LAPDSim1D(params, flags)
    sim.start_simulation(
        t_end=args.t_end,
        max_steps=args.max_steps,
        progress_tracker=_progress,
        progress_interval_s=0.0,
    )
    result = sim.get_results()
    save_result_hdf5(args.save_h5, result, params=params, flags=flags)

    # Where the beam actually put its energy. Under the coverage closure's
    # two-medium split the reservoir arm is supposed to reach past the source
    # region, so the share deposited beyond mid-machine is the direct readout
    # of whether it does.
    _deposition_profile_split(sim, result)

    times = np.asarray(result.time, dtype=float)
    breakdown = None
    events = getattr(result, "phase_events", None) or {}
    for when, phase, reason in zip(
        np.asarray(events.get("time", []), dtype=float),
        events.get("phase", []),
        events.get("reason", []),
    ):
        print(f"phase event: {phase!r} ({reason!r}) at {when * 1e3:.4f} ms")
        if str(phase) == "main_discharge" and breakdown is None:
            breakdown = float(when)
    print(f"window: 0 -> {times[-1] * 1e3:.4f} ms, {times.size} saves")
    if breakdown is not None:
        print(f"conducting phase contained: {breakdown * 1e3:.4f} ms")
    I_loop = np.asarray(
        result.cathode_diagnostics["circuit_I_loop"], dtype=float
    )
    print(f"I_loop: max {I_loop.max():.4g} A, final {I_loop[-1]:.4g} A")
    for threshold in (2.0, 132.0, 1000.0):
        hit = np.flatnonzero(I_loop >= threshold)
        when = f"{times[hit[0]] * 1e3:.4f} ms" if hit.size else "never"
        print(f"I_loop first reaches {threshold:g} A at {when}")
    if coverage is not None:
        f_cov = np.asarray(
            result.cathode_diagnostics["coverage_fraction"], dtype=float
        )
        print(
            f"f_cov trace (column mean): start {f_cov[0]:.6f}, "
            f"end {f_cov[-1]:.6f}, reaches 0.5 at "
            + (
                f"{times[np.flatnonzero(f_cov >= 0.5)[0]] * 1e3:.4f} ms"
                if np.any(f_cov >= 0.5)
                else "never"
            )
        )
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            i = min(int(frac * (times.size - 1)), times.size - 1)
            print(
                f"  t={times[i] * 1e3:9.4f} ms  f_cov={f_cov[i]:.6f}  "
                f"nn_deficit_max="
                f"{result.cathode_diagnostics['coverage_nn_deficit_max'][i]:.6e}"
            )
        _coverage_profile_read(sim, result)
    print(f"saved {args.save_h5}")


def _coverage_profile_read(sim, result):
    """Print f_cov(z, t) snapshots and the front-lengthening direction read.

    Under v2 the coverage field is z-resolved and driven by the LOCAL beam
    ionization, so the question the snapshots answer is whether downstream
    cells' coverage grows LATER than near-source ones -- a front lengthening in
    kind. Direction only; nothing here is a score.
    """
    diag = getattr(result, "cathode_diagnostics", None) or {}
    profile = diag.get("coverage_fraction_profile")
    if profile is None:
        print("f_cov(z) profile: not saved in this result")
        return
    profile = np.asarray(profile, dtype=float)
    if profile.ndim != 2:
        print("f_cov(z) profile: unexpected shape", profile.shape)
        return
    times = np.asarray(result.time, dtype=float)
    geom = sim.geometry
    length = np.asarray(geom.length_cm, dtype=float)
    z = np.cumsum(length) - 0.5 * length
    live = np.flatnonzero(np.asarray(geom.plasma_active, dtype=bool))
    print("f_cov(z, t) snapshots  [plasma-active cells only]")
    picks = [min(int(f * (times.size - 1)), times.size - 1)
             for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for i in picks:
        row = profile[i][live]
        # Six evenly spaced positions along the live column, so the axial
        # shape is readable at a glance without dumping nx numbers per frame.
        idx = np.linspace(0, row.size - 1, 6).round().astype(int)
        cells = "  ".join(
            f"z={z[live][j]:6.0f}:{row[j]:.6f}" for j in idx
        )
        print(f"  t={times[i] * 1e3:9.4f} ms  {cells}")
    # The front read. The threshold is a quarter of the run's OWN largest
    # relative growth, so it is meaningful whatever the window's absolute
    # growth turns out to be (a short window grows f_cov by a fraction of a
    # percent, and a fixed 10% bar would simply never be crossed). A monotone
    # increase of the crossing time with z is coverage growth arriving later
    # downstream -- a front lengthening in kind.
    f0 = profile[0][live]
    rel = profile[:, live] / f0 - 1.0
    peak = float(np.max(rel))
    if peak <= 0.0:
        print("f_cov(z) did not grow over this window; no front read")
        return
    threshold = 0.25 * peak
    print(
        f"relative growth f_cov(z, t_end)/f_cov(z, 0) - 1: "
        f"max {peak:.6e}, min {float(np.min(rel[-1])):.6e}"
    )
    idx = np.linspace(0, f0.size - 1, 6).round().astype(int)
    print("  final relative growth by position:")
    for j in idx:
        print(f"    z={z[live][j]:6.0f} cm  {rel[-1][j]:.6e}")
    print(f"first time f_cov(z) grows by {threshold:.6e} (0.25 x the peak):")
    onset = []
    for j in range(f0.size):
        hit = np.flatnonzero(rel[:, j] >= threshold)
        onset.append(times[hit[0]] if hit.size else np.nan)
    onset = np.asarray(onset, dtype=float)
    for j in idx:
        when = "never" if not np.isfinite(onset[j]) else f"{onset[j] * 1e3:.4f} ms"
        print(f"    z={z[live][j]:6.0f} cm  onset={when}")
    finite = np.isfinite(onset)
    if finite.sum() >= 2:
        slope = np.polyfit(z[live][finite], onset[finite] * 1e3, 1)[0]
        print(
            f"  onset-vs-z slope: {slope:+.6e} ms/cm "
            f"({'LATER downstream' if slope > 0 else 'EARLIER downstream'}; "
            f"{int(finite.sum())}/{f0.size} cells crossed)"
        )
    else:
        print(
            f"  onset-vs-z slope: only {int(finite.sum())}/{f0.size} cells "
            "crossed; no slope"
        )


# Step-gated only: interval_fraction above 1 can never come due, so the
# cadence is purely `interval_steps`. The solver's own progress_interval_s is
# 0 above, so every accepted step reaches the tracker and the step gate is the
# only one that decides.
_progress = ProgressPrinter1D(interval_fraction=2.0, interval_steps=20000)


if __name__ == "__main__":
    main()
