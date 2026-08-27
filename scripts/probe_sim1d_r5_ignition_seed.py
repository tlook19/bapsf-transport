"""R5 ES1 ignition-seed isolation probe (2026-07-25; rule re-registered 2026-08-04).

HISTORY -- the 2026-07-25 ignition diagnosis (SUPERSEDED; kept as the record of
what this probe was originally built to answer, NOT as a live target). The
repaired-stance ES config sat in ``pre_breakdown`` at ``source_I_tot`` ~ 2.76 A,
and the diagnosis compared, at t = 0.2 ms and nx = 120, a historical/golden
collector density n_col ~ 9.8e8 (ignites) against a repaired-ES n_col ~ 4.0e8.
That reference pair is retired for two independent reasons:

  * It was CONFOUNDED. The failing run had not equilibrated the neutral
    inventory -- it started from the bare ``nn0`` seed, which is precisely the
    condition the ``no_equil`` variant below exists to mimic. The
    9.8e8-vs-4.0e8 gap therefore mixed stance effects with a fill difference,
    and the two were never separated.
  * It is MOOT. The current stance ignites at the sequencing-predicted time,
    so there is no ignition failure left to attribute.

PRE-REGISTERED ISOLATION (this script). Metric: collector-cell plasma density
n_col at t = 0.2 ms (the seed), with source_I_tot and phase as ignition proxies.
From the ES baseline, single-toggle each named suspect and add an all-three
positive control that reverts toward the historical stance:

  baseline      ES stance as-is                          (see ANCHORS below)
  char_off      characteristic_boundary = False          (R3.1 Bohm outflow off)
  birth_legacy  ionization_birth_energy_model = legacy   (hot legacy birth)
  sgp3400       S_gp = 3400 (config default; ES override is 3000)
  all3          char_off + birth_legacy + sgp3400        (positive control)
  no_equil      neutral_equilibration = False            (historical control)

SUPERSEDED 2026-08-21: sccm now MEANS meter-sccm (4.171431e17 particles/s per
sccm, 20 C / 1013 mbar) and the S_gp default moved 3400 -> 3649.84 on the same
rescale, so 3400 is no longer the config default and now ships ~6.85 % less
flux. The arm names and levels above are left AS A RECORD of what this dated
probe ran.

DECISION RULE (RE-REGISTERED 2026-08-04, fixed before the re-measurement runs;
supersedes the retired rule above). Three registered questions:

  PRIMARY -- seed-conditionality of the early-avalanche deposition closure.
    If ANY physically-plausible seed variant CROSSES n_b/n_e = 0.1 before
    ignition -- that is, the QL-cutoff diagnostic reports RELEASE where the
    baseline BINDS, or BINDING where the baseline releases -- then the
    early-avalanche deposition physics is seed-conditional, and the WP-E and
    foot-shape claims must carry a SEED bracket. If no variant crosses, the
    QL-off early avalanche is stated flat.

  SECONDARY -- ignition-time shift per decade of seed at the current stance.
    The loop-gain expectation on record is ~252 us per two decades (delay
    logarithmic in the seed); this measurement confirms or refutes it.

  CONTROL -- ``no_equil`` is the historical-attribution control, and its
    expected outcome is pre-stated: a LARGE n_col collapse under ``no_equil``
    confirms that the 2026-07-25 non-ignition was predominantly the missing
    neutral equilibration rather than a stance effect.

ANCHORS (measured 2026-08-04, nx = 120, t_probe = 0.2 ms):
  baseline n_col   = 2.558e9
  baseline n_b/n_e = 0.106 -- the QL cutoff BINDS on ~95% of the pre-ignition
                     window and never releases on baseline; the ~5% complement
                     is saves carrying no ray at all, not releases.
  char_off and all3 both release the QL cutoff at t = 0.020 ms.

SAMPLING TIME. ``--t-probe`` is model time measured from t = 0, and under the
sequencing stance -- ``tau_neutral_prebreakdown = 0.0``, the config default --
t = 0 is DRIVE-ON: nothing is prepended in front of the plasma clock, so the
metric samples the state 0.2 ms after the drive turns on. A nonzero
``tau_neutral_prebreakdown`` inserts a neutral-only window ahead of drive-on and
translates this axis by exactly that duration (``timing_defaults`` in
core/config.py), so the metric time must move with it to keep naming the same
physical point.

QL-CUTOFF DIAGNOSTIC (secondary output, read-only). The CSDA beam's quasilinear
drag channel is domain-limited: ``quasilinear_relaxation_length_cm`` returns inf
-- no anomalous drag at all -- unless the beam is weak,
``0 < n_b < 0.1 * n_e`` (cablp/cathode/beam_deposition.py). Because the seed sets
n_e, it can decide whether that channel exists during the early avalanche, so
the probe reports when the cutoff releases and how much of the pre-ignition
window sits under it. See ``ql_launch_cutoff`` for the exact quantity, which
cell it is evaluated at, and what it does NOT cover.

This is a diagnostic, not a solver change: no _sim1d/ edit, no gate, no baseline
touch. Run from <repo>/cablp with the fenicsx-env interpreter.
"""

import argparse
import math
import sys
import time as _walltime

import numpy as np

from cablp.cathode.beam_deposition import (
    beam_speed_cm_s,
    quasilinear_relaxation_length_cm,
)
from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.physics.cathode import beam_launch
from cablp.solvers._sim1d.solver import ProgressPrinter1D
from cablp.solvers._sim1d.results.io import save_result_hdf5
from cablp.constants import qe_SI

# Reuse the REAL ES config so the probe cannot drift from the benchmark driver.
from compare_sim1d_es1 import PARAM_OVERRIDES, FLAG_OVERRIDES

VARIANTS = {
    "baseline": (dict(), dict()),
    "char_off": (dict(), {"characteristic_boundary": False}),
    "birth_legacy": ({"ionization_birth_energy_model": "legacy"}, dict()),
    # SUPERSEDED 2026-08-21 (see the module docstring): the two 3400 literals
    # below are pre-changeover sccm, so they now ship ~6.85 % less flux than
    # when the arms ran, and 3400 is no longer the config default. Left AS A
    # RECORD of what this dated probe ran.
    "sgp3400": ({"S_gp": 3400}, dict()),
    "all3": (
        {"ionization_birth_energy_model": "legacy", "S_gp": 3400},
        {"characteristic_boundary": False},
    ),
    # CONTROL for historical attribution (see the re-registered DECISION RULE in
    # the module docstring): skip the 100-cycle neutral pre-equilibration, so
    # the run starts from the bare nn0 seed and builds its fill only from the
    # in-run puff -- exactly the condition the confounded 2026-07-25 diagnosis
    # ran under. Pre-stated expectation: a LARGE n_col collapse here confirms
    # that the 2026-07-25 non-ignition was predominantly the missing neutral
    # equilibration, not a stance effect.
    # NB this control used to ALSO carry the in-run tau_neutral_prebreakdown
    # = 2 ms neutral-only accumulation. That window is gone (the default is now
    # 0.0 -- the machine has no pre-drive window, 2026-08-03), so there is no
    # pre-phase accumulation left to fall back on here.
    "no_equil": (dict(), {"neutral_equilibration": False}),
    # Control: the pre-WIP ES geometry (no end-expansion). The end-expansion
    # geometry (Rcs=40/Lcs=25 + expand-to-1m + collector behind 9 "end" cells)
    # was added to PARAM_OVERRIDES AFTER the ignition diagnosis; this isolates
    # whether that geometry edit itself moved the seed.
    "no_endexp": (dict(), {"end_expansion_geometry": False}),
}


def build_config(variant, nx):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    # run_model() default: knudsen neutral exchange.
    params["neutral_exchange_model"] = "knudsen"
    params["nx"] = nx
    p_extra, f_extra = VARIANTS[variant]
    params.update(p_extra)
    flags.update(f_extra)
    # The end-expansion params are presence-gated on the flag: strip them when
    # the flag is off (else construction raises a loud ValueError, by design).
    if not flags.get("end_expansion_geometry", False):
        for k in ("end_expansion_cells", "end_expansion_machine_radius_cm",
                  "end_expansion_plasma_radius_cm"):
            params.pop(k, None)
    return params, flags


def ql_launch_cutoff(sim, res):
    """Per-save quasilinear-cutoff state at the beam launch cell (READ-ONLY).

    The CSDA module gives the beam an anomalous (quasilinear) drag channel only
    inside the weak-beam domain: ``quasilinear_relaxation_length_cm(E, n_e,
    n_b)`` returns ``inf`` -- i.e. NO anomalous drag -- unless ``n_e > 0`` and
    ``0 < n_b < 0.1 * n_e``. This evaluates that condition, using the module's
    own function so the constant is never restated here, at the LAUNCH cell of
    the source-end ray, where the ray's flux and energy are exactly the cathode
    solve's own recorded ``I_eth_star`` and ``phi_c``::

        n_b = (source_I_eth_star / qe_SI) / (plasma_area_cm2[launch] * v_b(phi_c))

    Nothing is re-solved: every input is read from the saved trajectory and the
    geometry the run already built.

    Returns ``(has_ray, released, nb_over_ne)``, each of length ``n_saves``:

    has_ray     : a source-end CSDA ray was launched on that save (the solve
                  ran, ``beam_csda_active``, and both recorded ray quantities
                  are finite and positive). ``False`` means the QL question is
                  moot on that save, not that the cutoff bound.
    released    : ``has_ray`` AND the cutoff is RELEASED (l_QL finite, so the
                  QL channel is live). ``has_ray and not released`` is the
                  cutoff BINDING.
    nb_over_ne  : n_b / n_e at the launch cell (NaN where ``has_ray`` is
                  False); the cutoff binds at and above 0.1.

    NOT COVERED, by construction: (i) cells downstream of the launch point --
    the ray's energy decays through the march and the per-cell ``E_entry_eV``
    is not among the saved datasets, so only the entry condition is
    recoverable; (ii) the twin/end ray; (iii) the clumped beam
    (``beam_clump_fraction > 0`` launches two rays carrying fractions of
    Gamma0, which lowers each ray's n_b) -- clumping is off in every variant
    this probe builds.
    """
    diag = res.cathode_diagnostics
    launch = beam_launch(sim.geometry, end=0)[0]
    area = float(np.asarray(sim.geometry.plasma_area_cm2, float)[launch])
    phi_c = np.asarray(diag["source_phi_c"], float)
    I_eth_star = np.asarray(diag["source_I_eth_star"], float)
    csda = np.asarray(diag["beam_csda_active"], float)
    ne = np.asarray(res.n, float)[:, launch]

    n_saves = phi_c.size
    has_ray = np.zeros(n_saves, dtype=bool)
    released = np.zeros(n_saves, dtype=bool)
    nb_over_ne = np.full(n_saves, np.nan)
    for k in range(n_saves):
        E0 = float(phi_c[k])
        I0 = float(I_eth_star[k])
        if csda[k] != 1.0 or not (math.isfinite(E0) and E0 > 0.0):
            continue
        if not (math.isfinite(I0) and I0 > 0.0):
            continue
        has_ray[k] = True
        n_b = (I0 / qe_SI) / (area * beam_speed_cm_s(E0))
        ne_k = float(ne[k])
        nb_over_ne[k] = n_b / ne_k if ne_k > 0.0 else np.inf
        released[k] = math.isfinite(
            quasilinear_relaxation_length_cm(E0, ne_k, n_b)
        )
    return has_ray, released, nb_over_ne


def pre_ignition_mask(phases):
    """Boolean mask of the saves BEFORE the first ``main_discharge`` sample.

    All-True when the run never reaches main_discharge, which is the expected
    case for a probe truncated inside the seed window.
    """
    hits = np.flatnonzero(np.asarray(phases, dtype=str) == "main_discharge")
    mask = np.ones(np.asarray(phases).size, dtype=bool)
    if hits.size:
        mask[hits[0]:] = False
    return mask


def probe(variant, nx, t_probe):
    params, flags = build_config(variant, nx)
    sim = LAPDSim1D(params, flags)
    t0 = _walltime.time()
    # start_simulation runs the neutral equilibration, applies it, then runs the
    # plasma -- truncated at t_probe. Exactly the production path, cut short.
    sim.start_simulation(t_end=t_probe, dt=None, operator_split=None, max_steps=None)
    res = sim.get_results()
    wall = _walltime.time() - t0

    roles = np.asarray(sim.geometry.cell_role)
    col = int(np.flatnonzero(roles == "collector")[-1])
    puff = int(np.flatnonzero(roles == "puff")[-1])
    col_mask = roles == "column"

    I = np.asarray(res.cathode_diagnostics["source_I_tot"], float)
    n = np.asarray(res.n, float)
    nn = np.asarray(res.nn, float)
    Te = np.asarray(res.Te, float)
    phases = np.asarray(res.phase, dtype=str)
    t = np.asarray(res.time, float)

    # QL-cutoff state over the pre-ignition window (read-only; see
    # ql_launch_cutoff for exactly what is and is not covered).
    has_ray, released, nb_over_ne = ql_launch_cutoff(sim, res)
    pre = pre_ignition_mask(phases)
    n_pre = int(np.count_nonzero(pre))
    rel_hits = np.flatnonzero(released)
    binding = pre & has_ray & ~released

    return {
        "variant": variant,
        "wall_s": wall,
        "t_end_ms": t[-1] * 1e3,
        "phase": phases[-1],
        "ever_main": bool(np.any(phases == "main_discharge")),
        "I_A": I[-1],
        "I_max_A": float(np.max(I)),
        "n_col": n[-1, col],
        "n_puff": n[-1, puff],
        "n_col_max_cell": float(np.max(n[-1, col_mask])),
        "nn_col": nn[-1, col],
        "Te_col": Te[-1, col],
        # --- QL cutoff at the beam launch cell -------------------------------
        # Earliest save time [ms] at which the QL condition RELEASES (l_QL
        # finite, anomalous channel live); NaN if it never releases in the run.
        "ql_t_release_ms": (
            float(t[rel_hits[0]] * 1e3) if rel_hits.size else float("nan")
        ),
        # Fraction of PRE-IGNITION saves on which a ray was launched and the
        # cutoff BOUND (weak-beam domain violated -> no anomalous drag).
        "ql_cutoff_frac": (
            float(np.count_nonzero(binding)) / n_pre if n_pre else float("nan")
        ),
        # Fraction of pre-ignition saves with NO source ray at all -- the
        # complement that keeps the cutoff fraction readable (a save with no
        # ray is not "under cutoff", the QL question simply does not arise).
        "ql_noray_frac": (
            float(np.count_nonzero(pre & ~has_ray)) / n_pre
            if n_pre
            else float("nan")
        ),
        # n_b / n_e at the launch cell on the last save; the cutoff binds at
        # and above 0.1. NaN when no ray was launched there.
        "ql_nb_over_ne_last": float(nb_over_ne[-1]),
    }


class _DualProgress:
    """Progress callback that tolerates BOTH protocols.

    Equilibration calls callback(fraction: float); the plasma run calls
    callback(progress_obj). Throttled so the plasma phase prints periodically
    with t, phase, dt, and the binding timestep constraint (to spot a crawl).
    """

    def __init__(self, every_steps=800):
        self.every = every_steps
        self._last = -every_steps

    def __call__(self, p):
        # As a progress_TRACKER we receive the SimulationProgress1D object
        # (progress_callback would instead receive a bare float -- see
        # solver._emit_progress).
        if isinstance(p, float):
            return
        if p.step - self._last < self.every and p.fraction < 1.0:
            return
        self._last = p.step
        print(f"  t={p.time*1e3:8.3f}ms {p.phase:<14} step={p.step:>8} "
              f"dt={p.accepted_dt:.2e} cap={p.step_cap} constr={p.active_constraint} "
              f"eta={p.wall_remaining_s:6.0f}s", flush=True)


def trace(variant, nx, out_h5, cached_path=None):
    """Run the ES config to completion with progress, save H5.

    Decisive end-to-end test: does it ignite (phase -> main_discharge), how long,
    and does dt crawl (watch dt sitting at dt_min; since 2026-08-05
    ``active_constraint`` names the bound responsible rather than reading
    "dt_min")? With ``cached_path`` the neutral
    seed is loaded from cache (skips the ~20 s equilibration) -- also a
    correctness check that the cached-seed run matches the live-equilibration one.
    """
    params, flags = build_config(variant, nx)
    if cached_path:
        flags["use_cached_neutral_seed"] = True
        params["neutral_seed_cache_dir"] = cached_path
    sim = LAPDSim1D(params, flags)
    t0 = _walltime.time()
    sim.start_simulation(progress_tracker=_DualProgress(), progress_interval_s=2.0e-5)
    res = sim.get_results()
    wall = _walltime.time() - t0
    phases = np.asarray(res.phase, dtype=str)
    t = np.asarray(res.time, float)
    I = np.asarray(res.cathode_diagnostics["source_I_tot"], float)
    for ph in ("pre_breakdown", "breakdown", "main_discharge", "afterglow"):
        hits = np.flatnonzero(phases == ph)
        if hits.size:
            print(f"# first {ph:<14} at t={t[hits[0]]*1e3:8.3f} ms  I={I[hits[0]]:8.2f} A")
    print(f"# ever main_discharge: {bool(np.any(phases=='main_discharge'))}  "
          f"I_max={np.nanmax(I):.1f} A  t_end={t[-1]*1e3:.3f} ms  wall={wall:.1f} s")
    if out_h5:
        save_result_hdf5(out_h5, res, params=params, flags=flags)
        print(f"# saved {out_h5}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--t-probe", type=float, default=2.0e-4,
                    help="model time to stop at [s], measured from drive-on "
                         "when tau_neutral_prebreakdown = 0 (the default)")
    ap.add_argument("--variants", default="baseline,char_off,birth_legacy,sgp3400,all3")
    ap.add_argument("--trace", metavar="VARIANT", default=None,
                    help="run VARIANT to completion with progress instead of probing")
    ap.add_argument("--out-h5", default=None, help="save the --trace run to this H5")
    ap.add_argument("--cached-seed", default=None,
                    help="load the neutral seed from this .npz (skip equilibration)")
    args = ap.parse_args()

    if args.trace:
        trace(args.trace, args.nx, args.out_h5, cached_path=args.cached_seed)
        return

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"# R5 ignition-seed isolation probe  nx={args.nx}  t_probe={args.t_probe*1e3:.3f} ms")
    print(f"# anchors (measured 2026-08-04): baseline n_col 2.558e9, n_b/n_e 0.106; "
          f"the 9.8e8 / 4.0e8 pair is SUPERSEDED history, not a target")
    print("# QL cutoff read at the beam launch cell: binds when "
          "n_b/n_e >= 0.1 (no anomalous drag)")
    hdr = ("variant", "phase", "main?", "I[A]", "Imax[A]", "n_col", "n_col/base",
           "n_puff", "ncolmx", "nn_col", "Te_col", "wall[s]",
           "qlrel[ms]", "qlcutf", "qlnoray", "nb/ne")
    print(("{:<13}{:<14}{:<6}{:>9}{:>9}{:>11}{:>11}{:>11}{:>11}{:>11}{:>9}{:>9}"
           "{:>11}{:>8}{:>9}{:>11}").format(*hdr))
    sys.stdout.flush()

    base_ncol = None
    for v in variants:
        r = probe(v, args.nx, args.t_probe)
        if v == "baseline" or base_ncol is None:
            base_ncol = r["n_col"]
        ratio = r["n_col"] / base_ncol if base_ncol else float("nan")
        print(("{:<13}{:<14}{:<6}{:>9.2f}{:>9.2f}{:>11.3e}{:>11.2f}"
               "{:>11.3e}{:>11.3e}{:>11.3e}{:>9.3f}{:>9.1f}"
               "{:>11.3f}{:>8.3f}{:>9.3f}{:>11.3e}").format(
            r["variant"], r["phase"], "Y" if r["ever_main"] else "n",
            r["I_A"], r["I_max_A"], r["n_col"], ratio, r["n_puff"],
            r["n_col_max_cell"], r["nn_col"], r["Te_col"], r["wall_s"],
            r["ql_t_release_ms"], r["ql_cutoff_frac"], r["ql_noray_frac"],
            r["ql_nb_over_ne_last"]))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
