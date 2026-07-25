"""R5.1 gated fluid<->circuit Picard gate suite (SIM1D_MODEL_AUDIT_PLAN R5.1, A11).

The `coupled_circuit_picard` flag re-runs the accepted step with the frozen loop
current updated to the previous iteration's result, when |dI/dt| is large (the
emission knee / fall edge), so fluid + T_s + circuit share one self-consistent
I_loop. This suite validates the mechanism's SAFETY (the convergence claim itself
is `verify_sim1d_r3_a11.py --picard`):

  H1  snapshot/restore round-trip: `_picard_snapshot` -> advance a full step ->
      `_picard_restore` returns EVERY step-mutated attribute bit-identical to the
      pre-step state (the AGENTS.md accepted-step invariant -- rejected Picard
      iterations must not leak state).
  N1  no-op: picard ON with a trigger so large it never fires (one pass) is
      bit-identical to picard OFF over a short trajectory.
  P1  perturbation: picard ON with the production trigger changes the coupled
      state at the knee (the fix is active where |dI/dt| is large).
  G1  construction: default off; the K4a kinetic engine + picard raises loudly.

Usage:  python scripts/verify_sim1d_r5_picard.py
"""
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
from baseline_sim1d import build_baseline_config


def _build(picard=False, tol=None):
    params, flags = build_baseline_config()
    if picard:
        flags = {**flags, "coupled_circuit_picard": True}
        if tol is not None:
            params = {**params, "circuit_picard_tol_rel": tol}
    return LAPDSim1D(params, flags)


def _fingerprint(sim):
    """A comparable copy of every attribute a step mutates."""
    fp = {a: getattr(sim, a) for a in sim._PICARD_DIRECT_ATTRS}
    fp["_y"] = sim._y.copy()
    fp["_cathode_x0"] = np.asarray(sim._cathode_x0, dtype=float).copy()
    fp["_cathode_x0_twin"] = (
        None if sim._cathode_x0_twin is None
        else np.asarray(sim._cathode_x0_twin, dtype=float).copy()
    )
    fp["_cathode_beam_cross"] = (
        None if sim._cathode_beam_cross is None
        else np.asarray(sim._cathode_beam_cross, dtype=float).copy()
    )
    fp["_floor_ledger"] = dict(sim._floor_ledger)
    fp["_cathode_energy_ledger_J"] = dict(sim._cathode_energy_ledger_J)
    fp["_sample_ema"] = (
        None if sim._sample_ema is None
        else {c: list(v) for c, v in sim._sample_ema.items()}
    )
    fp["_cathode_solve"] = sim._cathode_solve
    return fp


def _fp_equal(a, b):
    diffs = []
    for k in a:
        va, vb = a[k], b[k]
        if k == "_cathode_solve":
            if va is not vb:
                diffs.append(k)
            continue
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not np.array_equal(np.asarray(va), np.asarray(vb)):
                diffs.append(k)
        elif isinstance(va, dict):
            if set(va) != set(vb) or any(
                not np.array_equal(np.asarray(va[c]), np.asarray(vb[c]))
                for c in va
            ):
                diffs.append(k)
        elif va is None or vb is None:
            if va is not vb:
                diffs.append(k)
        elif va != vb:
            diffs.append(k)
    return diffs


def gate_h1():
    sim = _build(picard=True)
    sim.start_simulation(t_end=3.2e-3)  # just past breakdown (~3.07 ms), active
    before = _fingerprint(sim)
    snap = sim._picard_snapshot()
    dt = sim.suggest_timestep(include_heat_conduction=False).dt
    # a full coupled step mutates state, then restore must undo it exactly
    sim.advance_one_step(dt=dt)
    sim._picard_restore(snap)
    after = _fingerprint(sim)
    diffs = _fp_equal(before, after)
    ok = not diffs
    return "H1 snapshot/restore round-trip bit-identical", ok, (
        "clean" if ok else f"leaked: {diffs}"
    )


def gate_n1():
    t_end = 3.3e-3
    off = _build(picard=False)
    off.start_simulation(t_end=t_end)
    on = _build(picard=True, tol=1.0e12)  # trigger never fires -> one pass
    on.start_simulation(t_end=t_end)
    dy = float(np.max(np.abs(off._y - on._y)))
    dI = abs(float(off._circuit_I_loop) - float(on._circuit_I_loop))
    ok = dy == 0.0 and dI == 0.0
    return "N1 no-op: picard(never-fire) == picard-off bit-identical", ok, (
        f"max|dy|={dy:.1e}  |dI_loop|={dI:.1e}"
    )


def gate_p1():
    # At the knee the production trigger fires and changes the coupled state.
    t0 = 4.5e-3
    off = _build(picard=False)
    off.start_simulation(t_end=t0)
    on = _build(picard=True)  # production tol 1e-2
    on.start_simulation(t_end=t0)
    dI = abs(float(off._circuit_I_loop) - float(on._circuit_I_loop))
    dTs = abs(float(off._cathode_Ts_K) - float(on._cathode_Ts_K))
    ok = dI > 0.0 or dTs > 0.0
    return "P1 perturbation: picard changes the knee state", ok, (
        f"|dI_loop|={dI:.3e} A  |dT_s|={dTs:.3e} K"
    )


def gate_g1():
    # default off
    off = _build(picard=False)
    default_off = not off._coupled_circuit_picard
    # kinetic + picard must raise
    params, flags = build_baseline_config()
    params = {**params, "neutral_model": "kinetic",
              "neutral_exchange_model": "knudsen"}
    flags = {**flags, "coupled_circuit_picard": True, "neutral_two_zone": True}
    raised = False
    try:
        LAPDSim1D(params, flags)
    except ValueError as e:
        raised = "kinetic" in str(e).lower()
    except Exception:
        raised = False
    ok = default_off and raised
    return "G1 default-off + kinetic-incompat construction guard", ok, (
        f"default_off={default_off}  kinetic_guard_raised={raised}"
    )


def main():
    gates = [gate_g1, gate_h1, gate_n1, gate_p1]
    all_ok = True
    print("R5.1 gated fluid<->circuit Picard gate suite (A11)")
    print("=" * 70)
    for g in gates:
        name, ok, detail = g()
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    print("=" * 70)
    print("R5.1 picard gates:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
