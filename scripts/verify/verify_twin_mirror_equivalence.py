"""Mirror equivalence: the half column against the full two-source column.

``far_end = "mirror"`` ends the column at ``Lm/2`` in a mirror face, standing
in for the symmetric half of a two-source machine whose image source sits at
``z = Lm``. ``TwinCathode`` builds that whole machine: its mesh is the half
column's mesh reflected about ``Lm/2`` (edge for edge on the near half). On a
mirror-symmetric state the two must evolve identically on the half domain,
and this script checks that they do.

Both cases are built from ONE configuration (``default_config()`` plus
:data:`SHARED_OVERRIDES`: the two-source machine ``Lm = 1965.4`` cm, cathode
to image cathode, with ``nx = 118`` far cells per half; the circuit off, cold
fluid neutrals, a plasma-bearing uniform initial state) and differ only in:

* the half column: ``far_end = "mirror"`` and ``S_pump_R = 0`` (the mirror
  has no right pump, and refuses one);
* the full column: ``TwinCathode = True`` and ``S_pump_R = S_pump_L`` (the
  far plenum's pump is the mirror image of the source plenum's). The
  end-side puff ``Twin_S_gp`` equals ``S_gp`` in the shared configuration.

Before marching it checks, and refuses to continue unless it holds exactly,
that the twin's cell edges up to ``Lm/2`` equal the half column's, that its
cell lengths are a palindrome, and that its initial state is mirror-symmetric
(``M`` antisymmetric) with a left half equal to the half column's initial
state. Both are then marched by ``run()``. The half column runs under its
own timestep control, and the timestep proposal its run loop receives at
each step start (``suggest_timestep``: the proposed ``dt``, its raw minimum,
its binding constraint and its dt-floor flag) is recorded. The full column
REPLAYS those proposals: its ``suggest_timestep`` is wrapped, on this
instance only, so that at each recorded step start the run loop reads the
half column's proposal in those four fields and the full column's own in
every other. The run loop's caps (dt growth and its recovery streak, t_end,
phase boundaries, save times) then act on identical inputs and select the
same accepted steps. Nothing else of the run loop changes. The script checks
that both accepted sequences, times and dt, are bit-identical, and refuses
the comparison otherwise. The dt
replay was chosen after the first G2 result, in which the two columns ran
under their own timestep controls and their dt sequences separated from
one ulp; that comparison measured the controllers' roundoff amplification
on a floor-pinned plasma rather than the mirror. The comparison is made at
the saved samples, matched by index, after checking that the two save-time
lattices agree.

Per save it reports the relative L-infinity error over the half domain
(cells ``0 .. cells_half - 1``), ``max|twin - half| / max|half|``, of ``n``,
``Te``, ``Ti`` and ``nn`` (the column neutral density), and, as diagnostics,
the same for the annulus neutral density ``nn_a`` and the full column's own
symmetry defect (left half against the reflected right half).

``--neutral-model kinetic_dvm`` runs both cases with the kinetic neutral
closure (``neutral_model = "kinetic_dvm"``) in place of the fluid one, every
other key unchanged; the half column's far plane is then the DVM's specular
mirror plane. It additionally records, at every neutral tick of the half
column, the DVM ledger's right-end rows -- the gross particle traffic out
through the mirror plane and its same-tick return, their difference (the net
particle flux through the plane), the plane's net energy row
``net_surface_end_R``, the right pump row, the largest entry of the lagged
right-end buffers -- and the tick's particle and energy closure residuals,
and reports the worst of each. They are reported, not gated.

PASS = every one of ``n``, ``Te``, ``Ti``, ``nn`` at or below
:data:`TOLERANCE` at every save. Exit 0 on PASS, 1 on FAIL, 2 when the
precondition checks, the dt replay or the save lattices refuse the
comparison.

Usage::

    python scripts/verify/verify_twin_mirror_equivalence.py --outdir DIR
    python scripts/verify/verify_twin_mirror_equivalence.py --outdir DIR \
        --neutral-model kinetic_dvm

Outputs (under ``--outdir``, which must lie outside the repository):
``twin_mirror_equivalence.tsv`` (the per-save table),
``twin_mirror_dt.tsv`` (both accepted-dt sequences) and
``twin_mirror_equivalence.json`` (summary and verdict); under
``--neutral-model kinetic_dvm`` also ``twin_mirror_dvm_ledger.tsv`` (the
per-tick mirror-plane ledger rows).
"""

import argparse
from dataclasses import replace
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d.core.config import default_config
from cablp.solvers._sim1d.solver import LAPDSim1D

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The gate: relative L-infinity error on the half domain, every save.
TOLERANCE = 1.0e-8
#: The fields the gate decides on, as named on the saved result.
GATE_FIELDS = ("n", "Te", "Ti", "nn")
#: Reported, not gated.
DIAGNOSTIC_FIELDS = ("nn_a",)

#: The shared configuration: default_config() plus these keys.
SHARED_OVERRIDES = {
    "params": {
        # Cold fluid neutrals: the fluid neutral closure without the jets
        # that presume a live cathode.
        "cathode_neutral_jet": False,
        "cathode_jet_surface_debit": False,
        "cathode_jet_energy_convention": "legacy",
        "initial_neutral_state": "fill",
        # The two-source machine: cathode to image cathode, its symmetry
        # plane at Lm/2 = 982.7 cm, with 118 far cells per half.
        "Lm": 1965.4,
        "nx": 118,
        # A plasma-bearing uniform initial state at rest (u0 = 0 keeps it
        # mirror-symmetric).
        "ne0": 1.0e12,
        "Te0": 3.0,
        "Ti0": 1.0,
        "u0": 0.0,
    },
    "flags": {
        "cathode_coupling": False,
        "neutral_momentum": False,
        "neutral_energy": False,
        "neutral_hot_internal_wall": False,
    },
}


#: The ``--neutral-model`` choices; the first is the default.
NEUTRAL_MODELS = ("moment", "kinetic_dvm")


def build_configs(t_dt_save, neutral_model="moment"):
    """Return ``((half_params, half_flags), (twin_params, twin_flags))``."""
    if neutral_model not in NEUTRAL_MODELS:
        raise ValueError(
            f"neutral_model must be one of {list(NEUTRAL_MODELS)} "
            f"(got {neutral_model!r})"
        )
    params, flags = default_config()
    params.update(SHARED_OVERRIDES["params"])
    if neutral_model != "moment":
        params["neutral_model"] = neutral_model
    flags.update(SHARED_OVERRIDES["flags"])
    params["dt_save"] = float(t_dt_save)
    params["t_save_start"] = 0.0
    params["Twin_S_gp"] = params["S_gp"]
    half = (dict(params, far_end="mirror", S_pump_R=0.0), dict(flags))
    twin = (
        dict(params, S_pump_R=params["S_pump_L"]),
        dict(flags, TwinCathode=True),
    )
    return half, twin


def _reflect(values, antisymmetric=False):
    out = np.asarray(values, dtype=float)[..., ::-1]
    return -out if antisymmetric else out


def preconditions(half_sim, twin_sim):
    """Return a list of failed precondition strings (empty = all hold)."""
    failures = []
    hg, tg = half_sim.geometry, twin_sim.geometry
    cells = int(hg.cells)
    if int(tg.cells) != 2 * cells:
        failures.append(f"twin cells {tg.cells} != 2 x half cells {cells}")
        return failures
    if not np.array_equal(tg.z_edges_cm[: cells + 1], hg.z_edges_cm):
        failures.append("twin edges up to Lm/2 differ from the half column's")
    if not np.array_equal(tg.length_cm, tg.length_cm[::-1]):
        failures.append("twin cell lengths are not a palindrome")
    if list(tg.cell_role[:cells]) != list(hg.cell_role):
        failures.append("twin near-half cell roles differ from the half column")
    hs, ts = half_sim.state, twin_sim.state
    for name in ("n", "nn", "nn_a", "M", "Ee", "Ei", "M_n", "En"):
        h, t = getattr(hs, name), getattr(ts, name)
        if (h is None) != (t is None):
            failures.append(f"state field {name} present on one case only")
            continue
        if h is None:
            continue
        if not np.array_equal(t[:cells], h):
            failures.append(f"initial {name}: twin left half != half column")
        if not np.array_equal(
            t[cells:], _reflect(t[:cells], antisymmetric=(name in ("M", "M_n")))
        ):
            failures.append(f"initial {name}: twin state not mirror-symmetric")
    return failures


class _DtRecorder:
    """Progress tracker recording every accepted step's (time, dt)."""

    def __init__(self):
        self.rows = []

    def update(self, progress):
        self.rows.append((float(progress.time), float(progress.accepted_dt)))


#: The timestep-proposal fields the run loop's step selection reads.
CONTROLLER_FIELDS = ("dt", "dt_raw", "active_constraint", "clamped_to_dt_min")


def record_proposals(sim):
    """Record ``sim``'s timestep proposals, keyed by the solver time.

    Wraps ``suggest_timestep`` on this one instance, passing every result
    through unchanged; returns the ``{time: {field: value}}`` dict it fills.
    Inside ``run()`` the proposal is requested once per step, at its start.
    """
    proposals = {}
    original = sim.suggest_timestep

    def suggest_timestep(*args, **kwargs):
        diag = original(*args, **kwargs)
        proposals[float(sim.time)] = {
            name: getattr(diag, name) for name in CONTROLLER_FIELDS
        }
        return diag

    sim.suggest_timestep = suggest_timestep
    return proposals


def replay_proposals(sim, proposals):
    """Make ``sim.run()`` read the recorded proposal at each step start.

    Wraps ``suggest_timestep`` on this one instance: at a recorded time it
    returns the solver's own diagnostics with :data:`CONTROLLER_FIELDS`
    replaced by the recorded values; at any other time it passes them
    through unchanged, and the caller checks the replayed steps afterwards.
    """
    original = sim.suggest_timestep

    def suggest_timestep(*args, **kwargs):
        diag = original(*args, **kwargs)
        recorded = proposals.get(float(sim.time))
        return diag if recorded is None else replace(diag, **recorded)

    sim.suggest_timestep = suggest_timestep


class _MirrorLedgerRecorder:
    """Record the half column's DVM mirror-plane ledger rows at every tick.

    Wraps ``update`` on the one engine instance, passing every ledger through
    unchanged.
    """

    FIELDS = (
        "out_R", "return_R", "net_particles_R", "pump_R",
        "net_energy_R_erg", "energy_out_R_erg", "pending_R_max",
        "particle_distribution_rel", "particle_domain_rel",
        "energy_distribution_rel", "energy_domain_rel",
    )

    def __init__(self, dvm):
        from cablp.solvers._sim1d.physics.kinetic_dvm import (
            ledger_energy_residual,
            ledger_residual,
        )

        self.rows = []
        original = dvm.update

        def update(*args, **kwargs):
            ledger = original(*args, **kwargs)
            energy = ledger["energy"]
            particles = ledger_residual(ledger)
            energies = ledger_energy_residual(ledger)
            self.rows.append({
                "out_R": ledger["loss_end_out_R"],
                "return_R": ledger["birth_end_return_R"],
                "net_particles_R": (
                    ledger["loss_end_out_R"] - ledger["birth_end_return_R"]
                ),
                "pump_R": ledger["loss_pump_R"],
                "net_energy_R_erg": energy["net_surface_end_R"],
                "energy_out_R_erg": energy["loss_end_out_R"],
                "pending_R_max": float(max(
                    np.max(np.abs(dvm.pend_R_c)), np.max(np.abs(dvm.pend_R_a))
                )),
                "particle_distribution_rel": particles["distribution_rel"],
                "particle_domain_rel": particles["domain_rel"],
                "energy_distribution_rel": energies["distribution_rel"],
                "energy_domain_rel": energies["domain_rel"],
            })
            return ledger

        dvm.update = update

    def summary(self):
        """Return ``{field: worst |value|}`` over the ticks, plus the count."""
        out = {"ticks": len(self.rows)}
        for name in self.FIELDS:
            out[f"max_abs_{name}"] = max(
                (abs(float(row[name])) for row in self.rows), default=0.0
            )
        # The net rows relative to the gross traffic they are the difference
        # of, tick by tick.
        out["max_net_particles_over_out_R"] = max(
            (abs(row["net_particles_R"]) / row["out_R"]
             for row in self.rows if row["out_R"] > 0.0),
            default=0.0,
        )
        out["max_net_energy_over_energy_out_R"] = max(
            (abs(row["net_energy_R_erg"]) / row["energy_out_R_erg"]
             for row in self.rows if row["energy_out_R_erg"] > 0.0),
            default=0.0,
        )
        return out


def _rel_linf(values, reference):
    scale = float(np.max(np.abs(reference)))
    diff = float(np.max(np.abs(np.asarray(values) - np.asarray(reference))))
    return diff / scale if scale > 0.0 else diff


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument(
        "--t-end", type=float, default=5.0e-3,
        help="simulated time to march both cases [s] (default 5e-3)",
    )
    ap.add_argument(
        "--dt-save", type=float, default=1.0e-4,
        help="save cadence [s] (default 1e-4)",
    )
    ap.add_argument(
        "--neutral-model", choices=NEUTRAL_MODELS, default=NEUTRAL_MODELS[0],
        help="neutral closure both cases run (default moment); kinetic_dvm "
        "also reports the half column's DVM mirror-plane ledger rows",
    )
    args = ap.parse_args(argv)
    outdir = args.outdir.expanduser().resolve()
    if outdir == REPO_ROOT or REPO_ROOT in outdir.parents:
        ap.error(f"--outdir must lie outside the repository ({REPO_ROOT})")
    if args.t_end < 3.0e-3:
        ap.error("--t-end must be at least 3e-3 s")
    outdir.mkdir(parents=True, exist_ok=True)

    (hp, hf), (tp, tf) = build_configs(args.dt_save, args.neutral_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        half_sim = LAPDSim1D(hp, hf)
        twin_sim = LAPDSim1D(tp, tf)
    cells = int(half_sim.geometry.cells)
    print(
        f"half column: {cells} cells (far_end='mirror'); full column: "
        f"{int(twin_sim.geometry.cells)} cells (TwinCathode)"
        + (
            "" if args.neutral_model == "moment"
            else f"; neutral_model={args.neutral_model!r}"
        )
    )
    failures = preconditions(half_sim, twin_sim)
    for failure in failures:
        print(f"PRECONDITION FAILED: {failure}")
    if failures:
        print("G2 VERDICT: REFUSED (preconditions)")
        return 2
    print(
        "preconditions: twin edges to Lm/2 == half column edges (exact); "
        "twin lengths palindromic; initial state mirror-symmetric with left "
        "half == half column (exact)"
    )

    half_dt, twin_dt = _DtRecorder(), _DtRecorder()
    proposals = record_proposals(half_sim)
    mirror_ledger = (
        _MirrorLedgerRecorder(half_sim._dvm)
        if args.neutral_model == "kinetic_dvm" else None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        half = half_sim.run(
            t_end=args.t_end, progress_tracker=half_dt, progress_interval_s=0.0
        )
        replay_proposals(twin_sim, proposals)
        twin = twin_sim.run(
            t_end=args.t_end, progress_tracker=twin_dt, progress_interval_s=0.0
        )

    # The half column's own mirror-plane ledger needs no comparison, so it is
    # banked before the replay and save-lattice checks can refuse one.
    if mirror_ledger is not None:
        with open(outdir / "twin_mirror_dvm_ledger.tsv", "w") as fh:
            fh.write("tick\t" + "\t".join(mirror_ledger.FIELDS) + "\n")
            for k, row in enumerate(mirror_ledger.rows):
                fh.write(f"{k}\t" + "\t".join(
                    repr(float(row[name])) for name in mirror_ledger.FIELDS
                ) + "\n")
        ledger_summary = mirror_ledger.summary()
        print(
            "DVM mirror plane, worst over the half column's "
            f"{ledger_summary['ticks']} neutral ticks: "
            + ", ".join(
                f"{k} {v:.3e}" for k, v in ledger_summary.items()
                if k != "ticks"
            )
        )

    hdt = np.asarray([row[1] for row in half_dt.rows])
    tdt = np.asarray([row[1] for row in twin_dt.rows])
    dt_identical = hdt.shape == tdt.shape and np.array_equal(hdt, tdt)
    htimes = np.asarray([row[0] for row in half_dt.rows])
    ttimes = np.asarray([row[0] for row in twin_dt.rows])
    steps_identical = dt_identical and np.array_equal(htimes, ttimes)
    with open(outdir / "twin_mirror_dt.tsv", "w") as fh:
        fh.write("step\thalf_time_s\thalf_dt_s\ttwin_time_s\ttwin_dt_s\n")
        for k in range(max(len(half_dt.rows), len(twin_dt.rows))):
            h = half_dt.rows[k] if k < len(half_dt.rows) else (np.nan, np.nan)
            t = twin_dt.rows[k] if k < len(twin_dt.rows) else (np.nan, np.nan)
            fh.write(
                f"{k}\t{h[0]!r}\t{h[1]!r}\t{t[0]!r}\t{t[1]!r}\n"
            )
    common = min(hdt.size, tdt.size)
    dt_max_rel = (
        float(np.max(np.abs(hdt[:common] - tdt[:common]) / hdt[:common]))
        if common else float("nan")
    )
    first_diff = (
        None if dt_identical else next(
            (k for k in range(common) if hdt[k] != tdt[k]), common
        )
    )
    print(
        f"accepted steps: half {hdt.size}, twin {tdt.size} (twin replays the "
        f"half column's dt); dt sequences bit-identical: {dt_identical}; "
        f"step times bit-identical: {steps_identical}"
        + ("" if dt_identical else
           f" (first differing step {first_diff}, max relative dt "
           f"difference over the common steps {dt_max_rel:.3e})")
    )
    if not steps_identical:
        print("G2 VERDICT: REFUSED (the dt replay did not reproduce the steps)")
        return 2

    h_time = np.asarray(half.time, dtype=float)
    t_time = np.asarray(twin.time, dtype=float)
    if h_time.shape != t_time.shape:
        print(
            f"SAVE LATTICES DIFFER: half {h_time.size} saves, twin "
            f"{t_time.size} saves"
        )
        print("G2 VERDICT: REFUSED (save lattices)")
        return 2
    time_offset = float(np.max(np.abs(h_time - t_time))) if h_time.size else 0.0
    times_identical = bool(np.array_equal(h_time, t_time))
    print(
        f"saves: {h_time.size} per case, matched by index; save times "
        f"bit-identical: {times_identical} (max |dt_save offset| "
        f"{time_offset:.3e} s)"
    )

    rows = []
    worst = {name: 0.0 for name in GATE_FIELDS + DIAGNOSTIC_FIELDS}
    worst_sym = {name: 0.0 for name in GATE_FIELDS}
    for k in range(h_time.size):
        row = {"save": k, "time_s": float(h_time[k])}
        for name in GATE_FIELDS + DIAGNOSTIC_FIELDS:
            h = np.asarray(getattr(half, name))[k]
            t = np.asarray(getattr(twin, name))[k]
            err = _rel_linf(t[:cells], h)
            row[name] = err
            worst[name] = max(worst[name], err)
            if name in GATE_FIELDS:
                sym = _rel_linf(t[cells:], _reflect(t[:cells]))
                row[f"sym_{name}"] = sym
                worst_sym[name] = max(worst_sym[name], sym)
        row["pass"] = all(row[name] <= TOLERANCE for name in GATE_FIELDS)
        rows.append(row)

    columns = (
        ["save", "time_s"]
        + list(GATE_FIELDS)
        + list(DIAGNOSTIC_FIELDS)
        + [f"sym_{name}" for name in GATE_FIELDS]
        + ["pass"]
    )
    with open(outdir / "twin_mirror_equivalence.tsv", "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(
                str(row[c]) if c in ("save", "pass") else f"{row[c]:.6e}"
                for c in columns
            ) + "\n")
    header = (
        f"{'save':>4} {'t [ms]':>8} "
        + " ".join(f"{name:>10}" for name in GATE_FIELDS + DIAGNOSTIC_FIELDS)
        + " " + " ".join(f"{'sym_' + name:>10}" for name in GATE_FIELDS)
        + "  pass"
    )
    print(
        "per save: relative L-inf error on the half domain, "
        "max|twin - half| / max|half|; sym_* = the twin's own symmetry defect"
    )
    print(header)
    for row in rows:
        print(
            f"{row['save']:>4} {row['time_s'] * 1e3:>8.4f} "
            + " ".join(
                f"{row[name]:>10.3e}" for name in GATE_FIELDS + DIAGNOSTIC_FIELDS
            )
            + " " + " ".join(
                f"{row['sym_' + name]:>10.3e}" for name in GATE_FIELDS
            )
            + f"  {'PASS' if row['pass'] else 'FAIL'}"
        )

    passed = all(row["pass"] for row in rows) and bool(rows)
    summary = {
        "tolerance": TOLERANCE,
        "gate_fields": list(GATE_FIELDS),
        "diagnostic_fields": list(DIAGNOSTIC_FIELDS),
        "t_end_s": args.t_end,
        "dt_save_s": args.dt_save,
        "half_cells": cells,
        "twin_cells": int(twin_sim.geometry.cells),
        "saves": int(h_time.size),
        "final_time_s": float(h_time[-1]) if h_time.size else None,
        "accepted_steps_half": int(hdt.size),
        "accepted_steps_twin": int(tdt.size),
        "dt_sequences_bit_identical": bool(dt_identical),
        "step_times_bit_identical": bool(steps_identical),
        "dt_control": "twin replays the half column's accepted dt sequence",
        "dt_max_relative_difference": dt_max_rel,
        "save_times_bit_identical": times_identical,
        "worst_rel_linf": worst,
        "worst_symmetry_defect": worst_sym,
        "shared_overrides": SHARED_OVERRIDES,
        "verdict": "PASS" if passed else "FAIL",
    }
    if mirror_ledger is not None:
        summary["neutral_model"] = args.neutral_model
        summary["dvm_mirror_plane"] = ledger_summary
    with open(outdir / "twin_mirror_equivalence.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(
        "worst over all saves: "
        + ", ".join(f"{k} {v:.3e}" for k, v in worst.items())
    )
    print(
        "worst twin symmetry defect: "
        + ", ".join(f"{k} {v:.3e}" for k, v in worst_sym.items())
    )
    print(
        f"G2 VERDICT: {'PASS' if passed else 'FAIL'} (tolerance "
        f"{TOLERANCE:.0e} on {', '.join(GATE_FIELDS)} at all "
        f"{h_time.size} saves to t = {float(h_time[-1]) * 1e3:.3f} ms)"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
