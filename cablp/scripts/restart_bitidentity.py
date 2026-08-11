"""Split-vs-unsplit bit-identity acceptance for the F2/F3 restart machinery.

Runs a window twice:

  UNSPLIT  0 -> t_end in one call
  SPLIT    0 -> t_mid, export a restart payload, rebuild, t_mid -> t_end

and compares every saved frame at a time > t_mid at RAW UINT64 -- the float
bytes, not a tolerance. Frames are matched by their save time; the split run
does not repeat the handoff frame (stage 1's last frame IS that frame), so the
unsplit frame at t_mid is compared against stage 1's and everything after it
against stage 2's.

Two scenarios, both cheap:

  meanfield  a current-driven discharge on the production stance, no coverage
  coverage   the clumpy-plasma closure with the ionizing tail walk live, which
             is what issues the nested walker marches and moves the beam-cross
             continuation cache -- the cache the d1a null measured at ~1.0
             relative in l_b at beam turn-on

Usage:  python scripts/restart_bitidentity.py [--scenario NAME] [--keep-dir DIR]
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config, save_restart_state


#: Fields compared, in a fixed order. The conserved rows plus the derived
#: primitives plus the phase columns: everything the trajectory format carries
#: per frame that is a float array of cell width.
FRAME_FIELDS = (
    "n", "nn", "M", "Ee", "Ei", "u", "Te", "Ti", "pe", "pi", "p", "y",
    "phase_elapsed", "phase_cathode_enabled", "phase_gas_puff_enabled",
    "phase_floating",
)

#: Per-frame cathode diagnostics that MUST be present. Everything the run
#: actually publishes is compared (see ``frames``); this subset is the one the
#: gate refuses to run without, because each is the only place a particular
#: carried member becomes observable -- the circuit V_dis save anchor above
#: all, which touches no state row and would otherwise be untested.
#:
#: An earlier revision of this list named keys that do not exist
#: (``V_dis``, ``I_loop``) and the comparison SKIPPED them silently, which is
#: exactly the failure this assertion now prevents.
REQUIRED_DIAGNOSTIC_KEYS = (
    "source_I_tot",
    "source_phi_c",
    "T_s_surface",
    "circuit_I_loop",
    "circuit_V_dis_step",
    "circuit_V_dis_dt_integral",
)


def scenario_config(name):
    """Return ``(params, flags, t_mid, t_end)`` for a named scenario."""
    params, flags = default_config()
    params.update({
        "phase_transition_mode": "scheduled",
        "tau_neutral_prebreakdown": 0.0,
        "tau_prebreakdown": 0.0,
        "tau_breakdown": 0.0,
        "tau_discharge": 1.0,
        "tau_afterglow": 0.0,
    })
    # Inert for a direct run() (only start_simulation reads it), and cleared so
    # the unsplit and split runs carry byte-identical configs apart from
    # restart_from itself.
    flags["neutral_equilibration"] = False
    if name == "meanfield":
        # CHEAP, and deliberately in the dt-growth-dominated regime: over this
        # window the growth ramp is the active bound on most steps, which is
        # what makes previous_accepted_dt and the recovery streak observable.
        # Later in the discharge surface_loss binds nearly every step and both
        # go inert -- see meanfield_beam, and the control matrix.
        #
        # Saves are also sparse relative to the step: with dt_save capping
        # every step the ramp never binds at all.
        params.update({
            "nx": 24,
            "dt_save": 2.0e-6,
            "dt_growth_recovery_patience": 3,
        })
        return params, flags, 4.0e-6, 8.0e-6
    if name == "meanfield_beam":
        # t_mid sits AFTER beam turn-on so the beam-cross continuation cache is
        # NONZERO at the handoff: before ~2e-4 s it is identically zero and the
        # member the d1a null identified as order-unity goes untested. Mean
        # field, so this reaches the single-medium deposition path the coverage
        # scenario replaces with its two-stream wrapper.
        params.update({"nx": 24, "dt_save": 1.0e-4})
        return params, flags, 3.0e-4, 5.0e-4
    if name == "coverage":
        params.update({
            "nx": 12,
            "dt_save": 5.0e-5,
            "beam_deposition_model": "csda",
            "beam_anomalous_model": "quasilinear",
            "cathode_warming_model": "none",
            "cathode_Ts_base_K": None,
            "cathode_surface_model": "none",
            "cathode_phiwf_clean_eV": None,
            "cathode_cleaning_E_th_eV": None,
            "cathode_sample_smoothing": None,
            "coverage_initial_fraction": 0.3,
            "heating_anomalous_transport": "tail_walk",
            "heating_anomalous_tail_ionization": "on",
        })
        flags["coverage_closure"] = True
        # Beam-live, and f_cov still climbing at the handoff (it saturates at
        # 1.0 by ~3e-4 s), so both coverage members are moving when exported.
        return params, flags, 1.5e-4, 2.5e-4
    raise SystemExit(f"unknown scenario {name!r}")


def raw_bytes(values):
    """Return the raw uint64 view of a float array -- no tolerance anywhere."""
    array = np.ascontiguousarray(np.asarray(values, dtype=float))
    return array.view(np.uint64)


def frames(result):
    """Return ``{save_time_bits: {field: uint64 array}}`` for one result."""
    times = np.asarray(result.time, dtype=float)
    out = {}
    for index, time in enumerate(times):
        record = {}
        for field in FRAME_FIELDS:
            values = getattr(result, field, None)
            if values is None:
                continue
            record[field] = raw_bytes(np.asarray(values)[index])
        diagnostics = getattr(result, "cathode_diagnostics", {}) or {}
        missing = [k for k in REQUIRED_DIAGNOSTIC_KEYS if k not in diagnostics]
        if missing:
            raise SystemExit(
                "cathode_diagnostics is missing required keys "
                f"{missing}; the comparison would silently skip them. "
                f"Available: {sorted(diagnostics)}"
            )
        # EVERY published diagnostic, not a hand-listed subset: a key this
        # script does not know about is still a per-frame float array the
        # restart must reproduce.
        for key, values in diagnostics.items():
            values = np.asarray(values)
            if values.dtype.kind != "f" or values.shape[0] != len(times):
                continue
            record[f"diag:{key}"] = raw_bytes(values[index])
        out[float(time)] = record
    return out


def compare(reference, candidate, label, log):
    """Compare two frame records field by field; return the mismatch list."""
    problems = []
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    if missing:
        problems.append(f"{label}: fields absent from the split run: {missing}")
    if extra:
        problems.append(f"{label}: fields only in the split run: {extra}")
    for field in sorted(set(reference) & set(candidate)):
        a, b = reference[field], candidate[field]
        if a.shape != b.shape:
            problems.append(f"{label}: {field} shape {a.shape} != {b.shape}")
        elif not np.array_equal(a, b):
            differing = int(np.count_nonzero(a != b))
            problems.append(
                f"{label}: {field} differs in {differing}/{a.size} uint64 words"
            )
    return problems


#: Carried payload members perturbed one at a time by --negative-control. A
#: gate that passes with any of these corrupted is not measuring what it
#: claims to, so each must make the comparison FAIL. Keyed by payload section.
NEGATIVE_CONTROLS = {
    "cathode._cathode_beam_cross": ("cathode", "_cathode_beam_cross"),
    "cathode._cathode_x0": ("cathode", "_cathode_x0"),
    "circuit._circuit_I_loop": ("circuit", "_circuit_I_loop"),
    "circuit.V_dis_prev_save_integral": ("circuit", "V_dis_prev_save_integral"),
    "coverage.f": ("coverage", "f"),
    "coverage.deficit": ("coverage", "deficit"),
    "run_loop.previous_accepted_dt": ("run_loop", "previous_accepted_dt"),
    "run_loop.t_last_save": ("run_loop", "t_last_save"),
    "run_loop.dt_growth_capped_streak": ("run_loop", "dt_growth_capped_streak"),
}


#: Controls that are EXPECTED to show no effect, with the evidence for why.
#: A member showing no effect is otherwise a gate defect; listing one here is a
#: claim that it is genuinely inert in that scenario, and each entry has to be
#: backed by a measurement rather than a guess. ``None`` as the scenario means
#: every scenario.
INERT_EXPECTATIONS = {
    (None, "cathode._cathode_x0"):
        "measured inert: the d1a probe (2026-08-11) found ONE distinct beam "
        "result over five seeds spanning a 3300x range, on both the "
        "current-driven and the floating branch. Carried anyway -- the fixed "
        "point is a property of this stance, not a guarantee",
    ("meanfield_beam", "run_loop.dt_growth_capped_streak"):
        "dt_growth_recovery_patience is 0 here, which presence-gates the "
        "recovery branch off entirely (run(), solver.py:4448) -- the streak "
        "cannot matter when nothing reads it; meanfield runs at patience 3 "
        "and its control does break identity",
    ("coverage", "run_loop.dt_growth_capped_streak"):
        "dt_growth_recovery_patience is 0 here, which presence-gates the "
        "recovery branch off entirely (run(), solver.py:4448) -- the streak "
        "cannot matter when nothing reads it; meanfield runs at patience 3 "
        "and its control does break identity",
    ("meanfield", "cathode._cathode_beam_cross"):
        "beam_atten_cross is identically zero until the sheath potential "
        "crosses the ionization threshold (~2e-4 s), so there is nothing to "
        "perturb in this short window; covered by meanfield_beam and coverage",
}


def inert_reason(scenario, control):
    """Return the recorded justification for a no-effect control, or None."""
    return INERT_EXPECTATIONS.get(
        (scenario, control), INERT_EXPECTATIONS.get((None, control))
    )


def engagement_census(result, log):
    """Report what the window actually exercised, so a pass cannot be vacuous."""
    caps = [d.step_cap for d in result.diagnostics]
    census = {cap: caps.count(cap) for cap in sorted(set(caps))}
    diagnostics = getattr(result, "cathode_diagnostics", {}) or {}
    log(f"  step caps: {census}")
    log(f"  steps per save: {int(result.steps)}/{len(result.time)}")
    notes = []
    if census.get("dt_growth", 0) > 0:
        notes.append(f"dt_growth bound {census['dt_growth']} step(s)")
    if "beam_csda_active" in diagnostics:
        active = np.asarray(diagnostics["beam_csda_active"], dtype=float)
        if active.min() < 0.5 < active.max():
            notes.append("beam TURN-ON inside the window")
        elif active.max() > 0.5:
            notes.append("beam live throughout")
    if "coverage_fraction" in diagnostics:
        f_cov = np.asarray(diagnostics["coverage_fraction"], dtype=float)
        notes.append(f"coverage f moved {f_cov.min():.6f} -> {f_cov.max():.6f}")
    for note in notes:
        log(f"  engaged: {note}")
    return census


def perturb(payload_path, control, log):
    """Corrupt one carried member in a written payload, in place."""
    import h5py

    group_name, member = NEGATIVE_CONTROLS[control]
    with h5py.File(payload_path, "r+") as h5:
        group = h5[group_name]
        if member in group:
            values = np.asarray(group[member][()], dtype=float)
            group[member][...] = values * 1.0000001 + 1.0e-30
            log(f"  perturbed dataset {group_name}/{member}")
            return True
        for suffix in ("", "__int", "__none", "__bool"):
            key = f"{member}{suffix}"
            if key in group.attrs:
                if suffix == "__none":
                    log(f"  {group_name}/{member} is None here; control skipped")
                    return False
                old = group.attrs[key]
                group.attrs[key] = (
                    # Integer members are counters compared against a
                    # THRESHOLD (the dt-growth streak against
                    # dt_growth_recovery_patience), so a +1 nudge can sit
                    # entirely below it and prove nothing. Push well past it.
                    int(old) + 64 if suffix == "__int" else float(old) * 1.0000001
                )
                log(f"  perturbed attr {group_name}/{key}: {old} -> "
                    f"{group.attrs[key]}")
                return True
    log(f"  {group_name}/{member} absent in this scenario; control skipped")
    return False


def run_scenario(name, workdir, log, split_at=None, control=None):
    params, flags, t_mid, t_end = scenario_config(name)
    if split_at is not None:
        t_mid = float(split_at)
    log(f"\n===== scenario {name}: t_mid={t_mid:g} s, t_end={t_end:g} s"
        + (f", NEGATIVE CONTROL {control}" if control else "")
        + " =====")

    unsplit = LAPDSim1D(dict(params), dict(flags)).run(t_end=t_end)
    log(f"unsplit: {int(unsplit.steps)} accepted steps, "
        f"{len(unsplit.time)} saved frames")
    engagement_census(unsplit, log)

    # Snap the split point onto the EXACT float of an unsplit save instant.
    #
    # The save lattice is accumulated (next = t_last_save + dt_save), so its
    # points carry float drift: with dt_save=1e-4 the third save lands on
    # 3.0000000000000003e-04, not 3e-04. Asking stage 1 to stop at the nominal
    # 3e-04 stops it one ulp EARLIER than any instant the unsplit run steps to,
    # which re-phases every later save and makes the comparison compare
    # different instants. This is a property of the comparison, not of the
    # restart: in production, stage 2 continues stage 1's own lattice and the
    # question never arises. Snapping is how the test asks a fair question.
    save_times = np.asarray(unsplit.time, dtype=float)
    candidates = save_times[(save_times > 0.0) & (save_times < t_end)]
    if candidates.size == 0:
        raise SystemExit(f"{name}: no interior save to split at")
    snapped = float(candidates[np.argmin(np.abs(candidates - t_mid))])
    if snapped != t_mid:
        log(f"  split point snapped {t_mid!r} -> {snapped!r} "
            "(exact float of an unsplit save instant)")
    t_mid = snapped

    stage1_sim = LAPDSim1D(dict(params), dict(flags))
    stage1 = stage1_sim.run(t_end=t_mid)
    payload = workdir / f"restart_bitidentity_{name}.restart.h5"
    save_restart_state(payload, stage1_sim)
    log(f"stage 1: {int(stage1.steps)} accepted steps, "
        f"{len(stage1.time)} saved frames -> {payload.name}")
    if control is not None and not perturb(payload, control, log):
        return []

    stage2 = LAPDSim1D(
        {**params, "restart_from": str(payload)}, dict(flags)
    ).run(t_end=t_end)
    log(f"stage 2: {int(stage2.steps)} accepted steps, "
        f"{len(stage2.time)} saved frames")

    unsplit_frames = frames(unsplit)
    split_frames = frames(stage1)
    split_frames.update(frames(stage2))

    if int(stage1.steps) + int(stage2.steps) != int(unsplit.steps):
        log(f"  STEP COUNT MISMATCH: {int(stage1.steps)}+{int(stage2.steps)} "
            f"!= {int(unsplit.steps)}")

    after = sorted(t for t in unsplit_frames if t > t_mid)
    at_or_before = sorted(t for t in unsplit_frames if t <= t_mid)
    log(f"frames at/<= t_mid: {len(at_or_before)}; "
        f"frames after t_mid (THE GATE): {len(after)}")

    problems = []
    if not after:
        problems.append(
            f"{name}: no saved frames after t_mid -- the gate would be vacuous"
        )
    for time in sorted(unsplit_frames):
        if time not in split_frames:
            problems.append(f"{name}: split run has no frame at t={time!r}")
            continue
        problems.extend(
            compare(
                unsplit_frames[time],
                split_frames[time],
                f"{name} t={time:.12e}",
                log,
            )
        )
    compared = len(set(unsplit_frames) & set(split_frames))
    log(f"frames compared (whole window, raw uint64): {compared}")
    if control is not None:
        # Inverted: a corrupted payload MUST break the comparison. A control
        # that "passes" means the member is not actually being tested.
        if problems:
            log(f"  RESULT: PASS (control) -- corrupting {control} broke "
                f"identity in {len(problems)} place(s), e.g. {problems[0]}")
            return []
        reason = inert_reason(name, control)
        if reason is not None:
            log(f"  RESULT: INERT (expected) -- {control} changed nothing. "
                f"{reason}")
            return []
        log(f"  RESULT: FAIL (control) -- corrupting {control} changed "
            "NOTHING; the gate does not test this member")
        return [f"{name}: negative control {control} did not break identity"]
    if problems:
        log(f"  RESULT: FAIL -- {len(problems)} mismatch(es)")
        for problem in problems[:40]:
            log(f"    {problem}")
    else:
        log(f"  RESULT: PASS -- every frame raw-byte identical "
            f"({len(after)} after t_mid, {compared} total)")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=None,
                        choices=("meanfield", "meanfield_beam", "coverage"))
    parser.add_argument("--keep-dir", default=None,
                        help="write payloads here instead of a temp dir")
    parser.add_argument("--split-at", type=float, default=None,
                        help="override t_mid (used to probe a split point that "
                             "is NOT a step boundary of the unsplit run)")
    parser.add_argument("--negative-control", action="store_true",
                        help="corrupt each carried member in turn; every one "
                             "must break identity or the gate is vacuous")
    parser.add_argument("--controls", default=None,
                        help="comma-separated subset of NEGATIVE_CONTROLS to "
                             "run, for targeting one member in an expensive "
                             "scenario")
    args = parser.parse_args(argv)
    names = (
        (args.scenario,)
        if args.scenario
        else ("meanfield", "meanfield_beam", "coverage")
    )

    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    if args.keep_dir:
        workdir = Path(args.keep_dir)
        workdir.mkdir(parents=True, exist_ok=True)
        context = None
    else:
        context = tempfile.TemporaryDirectory()
        workdir = Path(context.name)
    try:
        problems = []
        for name in names:
            if args.negative_control:
                selected = sorted(NEGATIVE_CONTROLS)
                if args.controls:
                    selected = [c.strip() for c in args.controls.split(",")]
                    unknown = [c for c in selected if c not in NEGATIVE_CONTROLS]
                    if unknown:
                        raise SystemExit(f"unknown control(s): {unknown}")
                for control in selected:
                    problems.extend(
                        run_scenario(name, workdir, log, control=control)
                    )
            else:
                problems.extend(
                    run_scenario(name, workdir, log, split_at=args.split_at)
                )
    finally:
        if context is not None:
            context.cleanup()

    log("")
    if problems:
        log(f"OVERALL: FAIL ({len(problems)} mismatch(es))")
        return 1
    log(f"OVERALL: PASS -- split-vs-unsplit bit-identity holds for "
        f"{', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
