"""Compiled vs pure CSDA march -- equivalence, per-call cost, march structure.

The mechanism-level companion to the golden gate for the compiled
``deposit_beam`` march (cost read 2026-08-02, "Kernel 1"). The golden with
``CABLP_COMPILED_KERNELS=1`` is the campaign's bit-exactness VERDICT; this
script says *where* any difference would come from, what one call costs each
way, and whether the march's structure moved.

Three questions, all empirical:

1. **Equivalence, at ``float.hex`` precision.** Every field of
   ``BeamDepositionResult`` compared bit-for-bit (as raw uint64, so signed
   zeros and NaNs are distinguished), over (a) the real production ray
   arguments captured by ``capture_csda_rays.py`` and (b) a randomised
   synthetic sweep across every closure combination. Reported as an exact
   count, never as a tolerance -- a tolerance would hide the answer.
2. **Per-call cost.** Pure vs compiled at each captured production state.
3. **March structure.** Substeps per call and ``deposit_beam`` calls per
   integration step, measured on the pure path over a short production run.
   The compiled path's substep sequence is pinned by question 1 rather than
   counted separately: the loop's exit tests read only ``E``, ``remaining``
   and ``L_tot``, so bit-identical ``E_entry`` and bit-identical banks over
   every cell mean the identical substep sequence -- a divergent march could
   not land on the same bits.

Requires the opt-in, because it needs both paths in one process::

    CABLP_COMPILED_KERNELS=1 python scripts/spike_csda_march.py \\
        --rays scripts/csda_rays_prod.pkl --label compiled_hotpath3

The captured ray files deliberately include the PRE-BREAKDOWN LONG-MFP window
(t ~ 2.01 ms) and that run's WIDEST ray: the cost read flagged this as the
outlier the main discharge never exercises, since production rays die in their
launch cell. The golden fixture is a 69-cell column, so its widest ray spans
~71 cells rather than the read's 260; ``_memo_outlier_state`` therefore adds a
RECONSTRUCTED nx=240-scale version of the same regime (262 cells, E0 =
217.9 eV, a domain-spanning ray of ~366 substeps) so the many-cell,
many-substep corner is exercised at the resolution the read measured it at.
"""

import argparse
import gc
import pickle
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cablp.cathode import beam_deposition as bd  # noqa: E402
from cablp.cathode import kernels as _kernels  # noqa: E402
from cablp.atomic.cross_sections import (  # noqa: E402
    He_beam_excitation_channel_lkup as _exc_lkup,
)

_RESULT_FIELDS = tuple(bd.BeamDepositionResult.__dataclass_fields__)


def _bits(value):
    """Raw uint64 view, so -0.0 != 0.0 and NaN payloads are distinguished."""
    array = np.asarray(value, dtype=np.float64)
    return np.ascontiguousarray(array).view(np.uint64)


def _run_both(kwargs):
    """Return ``(pure_result, compiled_result)`` for one deposit_beam call."""
    saved = bd._CSDA_MARCH
    try:
        bd._CSDA_MARCH = None
        pure = bd.deposit_beam(**kwargs)
        bd._CSDA_MARCH = saved
        compiled = bd.deposit_beam(**kwargs)
    finally:
        bd._CSDA_MARCH = saved
    return pure, compiled


def _compare(pure, compiled):
    """Return the list of field names that are not bit-identical."""
    differing = []
    for field in _RESULT_FIELDS:
        a = getattr(pure, field)
        b = getattr(compiled, field)
        if not np.array_equal(_bits(a), _bits(b)):
            differing.append(field)
    return differing


def _hex_report(pure, compiled, field, buf):
    """Print the first differing element of ``field`` in float.hex form."""
    a = np.atleast_1d(np.asarray(getattr(pure, field), dtype=float))
    b = np.atleast_1d(np.asarray(getattr(compiled, field), dtype=float))
    bad = np.flatnonzero(_bits(a) != _bits(b))
    for index in bad[:4]:
        print(
            "      %s[%d]  pure=%s  compiled=%s"
            % (field, index, float(a[index]).hex(), float(b[index]).hex()),
            file=buf,
        )


def _call_kwargs(state):
    """Rebuild deposit_beam's kwargs from a captured (args, kwargs) pair."""
    positional = (
        "E0_eV", "Gamma0_per_s", "nn", "ne", "Te", "launch", "direction",
        "dz_cm",
    )
    kwargs = dict(zip(positional, state["args"]))
    kwargs.update(state["kwargs"])
    return kwargs


def _memo_outlier_state():
    """The cost read's pre-breakdown long-mfp corner, at nx=240 scale.

    RECONSTRUCTED, not captured: the golden fixture's column is 69 cells, so
    no ray it launches can span 260. The plasma state is a uniform
    pre-breakdown column (n_e ~ 1e10, n_n ~ 1e13, Te ~ 1 eV) at 10 cm
    resolution -- the regime, not a specific saved timestep -- chosen so the
    ray crosses essentially the whole domain (261 of 262 cells) and stops
    there, at ~281 substeps. That is the corner the main-discharge states
    cannot reach -- ~2.5x their substep count and ~3x their per-call cost --
    rather than a bit-for-bit reproduction of the read's own snapshot, whose
    non-uniform column gave 260 cells at 366 substeps.
    """
    cells = 262
    return {
        "label": "memo_outlier_262cell*",
        "t_target_s": 2.01e-3,
        "t_actual_s": float("nan"),
        "args": (
            217.9,
            2.0e18,
            np.full(cells, 1.0e13),
            np.full(cells, 1.0e10),
            np.full(cells, 1.0),
            1,
            1,
            np.full(cells, 10.0),
        ),
        "kwargs": {
            "coulomb_model": "fast_electron",
            "anomalous_model": "quasilinear",
            "beam_area_cm2": 700.0,
        },
    }


def _count_substeps(kwargs):
    """Substeps in one PURE march: the excitation lookup fires once each."""
    counter = {"n": 0}
    real = bd.He_beam_excitation_channel_lkup

    def counting(*args, **kw):
        counter["n"] += 1
        return real(*args, **kw)

    saved = bd._CSDA_MARCH
    try:
        bd._CSDA_MARCH = None
        bd.He_beam_excitation_channel_lkup = counting
        bd.deposit_beam(**kwargs)
    finally:
        bd.He_beam_excitation_channel_lkup = real
        bd._CSDA_MARCH = saved
    return counter["n"]


def _time_call(kwargs, compiled, repeats):
    """Best-of-3 mean seconds per deposit_beam call."""
    saved = bd._CSDA_MARCH
    best = float("inf")
    try:
        bd._CSDA_MARCH = saved if compiled else None
        for _ in range(3):
            gc.disable()
            t0 = perf_counter()
            for _ in range(repeats):
                bd.deposit_beam(**kwargs)
            elapsed = perf_counter() - t0
            gc.enable()
            best = min(best, elapsed / repeats)
    finally:
        bd._CSDA_MARCH = saved
    return best


def _synthetic_sweep(buf, trials, seed):
    """Randomised states across every closure combination."""
    rng = np.random.default_rng(seed)
    cells = 60
    checked = 0
    differing = 0
    for _ in range(trials):
        dz = np.full(cells, 10.0)
        nn = 10.0 ** rng.uniform(11, 15, cells)
        ne = 10.0 ** rng.uniform(9, 13, cells)
        Te = 10.0 ** rng.uniform(-1, 1.9, cells)
        base = dict(
            E0_eV=float(rng.uniform(21.0, 900.0)),
            Gamma0_per_s=10.0 ** rng.uniform(14, 21),
            nn=nn,
            ne=ne,
            Te=Te,
            launch=int(rng.integers(0, cells)),
            direction=int(rng.choice([-1, 1])),
            dz_cm=dz,
            anode_cross_index=int(rng.integers(0, cells)),
            anode_eta=float(rng.choice([0.0, 0.3, 0.75])),
        )
        for coulomb in ("fast_electron", "legacy_tau_ei"):
            for anomalous in ("none", "quasilinear"):
                for product in ("local", "nonlocal"):
                    for tail in ("local", "tail_walk"):
                        if tail == "tail_walk" and anomalous == "none":
                            continue
                        kwargs = dict(
                            base,
                            coulomb_model=coulomb,
                            anomalous_model=anomalous,
                            beam_area_cm2=(
                                700.0 if anomalous == "quasilinear" else None
                            ),
                            product_transport=product,
                            anomalous_transport=tail,
                            tail_energy_eV=(
                                75.0 if tail == "tail_walk" else None
                            ),
                        )
                        pure, compiled = _run_both(kwargs)
                        checked += 1
                        bad = _compare(pure, compiled)
                        if bad:
                            differing += 1
                            print(
                                "  SYNTHETIC DIFF %s/%s/%s/%s: %s"
                                % (coulomb, anomalous, product, tail, bad),
                                file=buf,
                            )
                            for field in bad:
                                _hex_report(pure, compiled, field, buf)
    return checked, differing


def _structure_run(t_end, buf):
    """Substeps/call and calls/step over a short PURE production run."""
    import baseline_sim1d as baseline
    import cablp.solvers._sim1d.physics.cathode as cathode_mod
    from cablp.solvers._sim1d import summarize_result

    counters = {"calls": 0, "substeps": 0}
    real_deposit = cathode_mod.deposit_beam
    real_lkup = bd.He_beam_excitation_channel_lkup

    def counting_deposit(*args, **kwargs):
        counters["calls"] += 1
        return real_deposit(*args, **kwargs)

    def counting_lkup(*args, **kwargs):
        counters["substeps"] += 1
        return real_lkup(*args, **kwargs)

    saved = bd._CSDA_MARCH
    params, flags = baseline.build_baseline_config()
    sim = baseline.LAPDSim1D(params, flags)
    cathode_mod.deposit_beam = counting_deposit
    bd.He_beam_excitation_channel_lkup = counting_lkup
    bd._CSDA_MARCH = None
    try:
        sim.start_simulation(
            t_end=t_end, dt=None, operator_split=None, max_steps=None
        )
    finally:
        cathode_mod.deposit_beam = real_deposit
        bd.He_beam_excitation_channel_lkup = real_lkup
        bd._CSDA_MARCH = saved
    summary = summarize_result(sim.get_results())
    steps = int(summary.steps)
    print("\nMARCH STRUCTURE (pure path, t_end=%.2f ms)" % (t_end * 1e3), file=buf)
    print("  integration steps      : %d" % steps, file=buf)
    print("  deposit_beam calls     : %d" % counters["calls"], file=buf)
    print("  substeps (total)       : %d" % counters["substeps"], file=buf)
    print(
        "  calls per step         : %.3f" % (counters["calls"] / max(steps, 1)),
        file=buf,
    )
    print(
        "  substeps per call      : %.3f"
        % (counters["substeps"] / max(counters["calls"], 1)),
        file=buf,
    )


def _profile_share(t_end, compiled, buf):
    """Cumulative deposit_beam seconds as a fraction of run wall seconds."""
    import baseline_sim1d as baseline
    import cablp.solvers._sim1d.physics.cathode as cathode_mod

    total = {"s": 0.0, "n": 0}
    real_deposit = cathode_mod.deposit_beam

    def timing_deposit(*args, **kwargs):
        t0 = perf_counter()
        out = real_deposit(*args, **kwargs)
        total["s"] += perf_counter() - t0
        total["n"] += 1
        return out

    saved = bd._CSDA_MARCH
    params, flags = baseline.build_baseline_config()
    sim = baseline.LAPDSim1D(params, flags)
    cathode_mod.deposit_beam = timing_deposit
    bd._CSDA_MARCH = saved if compiled else None
    try:
        wall0 = perf_counter()
        sim.start_simulation(
            t_end=t_end, dt=None, operator_split=None, max_steps=None
        )
        wall = perf_counter() - wall0
    finally:
        cathode_mod.deposit_beam = real_deposit
        bd._CSDA_MARCH = saved
    print(
        "  %-9s run wall %8.2f s   deposit_beam %8.2f s   share %5.1f %%"
        "   (%d calls, %.1f us/call)"
        % (
            "compiled" if compiled else "pure",
            wall,
            total["s"],
            100.0 * total["s"] / wall,
            total["n"],
            1e6 * total["s"] / max(total["n"], 1),
        ),
        file=buf,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rays",
        required=True,
        nargs="+",
        help="one or more capture_csda_rays output files",
    )
    parser.add_argument("--label", default="compiled_hotpath3")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--profile-t-end",
        type=float,
        default=4.0e-3,
        help="t_end [s] for the profile-share and structure runs",
    )
    args = parser.parse_args(argv)

    if bd._CSDA_MARCH is None:
        raise SystemExit(
            "the compiled march is not bound; run with "
            "CABLP_COMPILED_KERNELS=1 and build_ext.py --inplace"
        )

    out_path = Path(args.output_dir) / f"{args.label}_csda_equivalence.txt"
    buf = out_path.open("w")
    try:
        print("compiled CSDA march: equivalence and cost", file=buf)
        print("provenance : %s" % _kernels.PROVENANCE, file=buf)
        print("numpy      : %s" % np.__version__, file=buf)
        print("rays       : %s" % " ".join(args.rays), file=buf)

        states = []
        seen = set()
        for path in args.rays:
            with open(path, "rb") as handle:
                captured = pickle.load(handle)
            for state in captured["states"]:
                key = (state["label"], round(state["t_actual_s"], 12))
                if key in seen:
                    continue
                seen.add(key)
                states.append(state)
        states.append(_memo_outlier_state())

        print(
            "\nPRODUCTION RAY STATES (captured live from the golden config; "
            "* = reconstructed)",
            file=buf,
        )
        print(
            "  %-24s %9s %9s %8s %7s %9s %9s"
            % ("label", "t [ms]", "E0 [eV]", "cells", "sub", "pure [us]", "cy [us]"),
            file=buf,
        )
        total_diff = 0
        for state in states:
            kwargs = _call_kwargs(state)
            pure, compiled = _run_both(kwargs)
            bad = _compare(pure, compiled)
            total_diff += len(bad)
            substeps = _count_substeps(kwargs)
            spanned = int(np.count_nonzero(pure.E_entry_eV > 0.0))
            t_pure = _time_call(kwargs, False, args.repeats)
            t_cy = _time_call(kwargs, True, args.repeats)
            print(
                "  %-24s %9.3f %9.2f %8d %7d %9.1f %9.1f   %s"
                % (
                    state["label"],
                    state["t_actual_s"] * 1e3,
                    float(kwargs["E0_eV"]),
                    spanned,
                    substeps,
                    t_pure * 1e6,
                    t_cy * 1e6,
                    "EXACT" if not bad else "DIFF %s" % bad,
                ),
                file=buf,
            )
            if bad:
                for field in bad:
                    _hex_report(pure, compiled, field, buf)
            print(
                "  %-24s speedup %.2fx"
                % ("", t_pure / t_cy if t_cy > 0 else float("nan")),
                file=buf,
            )

        print(
            "\n  production-state fields differing: %d of %d states x %d fields"
            % (total_diff, len(states), len(_RESULT_FIELDS)),
            file=buf,
        )

        print("\nSYNTHETIC CLOSURE SWEEP", file=buf)
        checked, differing = _synthetic_sweep(buf, args.trials, args.seed)
        print(
            "  configurations compared : %d\n  configurations differing: %d"
            % (checked, differing),
            file=buf,
        )

        _structure_run(args.profile_t_end, buf)

        print(
            "\nPROFILE SHARE (t_end=%.2f ms; deposit_beam wrapped with "
            "perf_counter)" % (args.profile_t_end * 1e3),
            file=buf,
        )
        _profile_share(args.profile_t_end, False, buf)
        _profile_share(args.profile_t_end, True, buf)

        verdict = "BIT-EXACT" if (total_diff == 0 and differing == 0) else "DIVERGENT"
        print("\nVERDICT: %s" % verdict, file=buf)
    finally:
        buf.close()
    print(out_path.read_text())
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
