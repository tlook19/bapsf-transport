"""regime_crf P1: is the conducting-phase loop current dt-CONVERGED?

The registered dt-convergence probe for the circuit-ratchet fix. It reruns the
diagnostician's trace scenario -- ``regime_r2_overlap_gate.build_config(20,
True)``, the pre-breakdown tracer-armed conducting stance with
``cathode_circuit_voltage_bound`` ARMED, R_mesh = 0 -- over t in [0, 2e-5] s at
dt = 1e-8, 1e-7, 3e-7, 1e-6, 3e-6 and adaptive, and reports I_loop(t) per arm.

Before the fix this scenario was the defect's fingerprint: I_loop was the
running maximum of the TR stage's explicit overshoot of the sheath capability
wall, so it grew with dt -- 0.9219 A at dt = 1e-8 rising to 156.7 A in a single
2e-5 s adaptive step, a factor of 170 (memo ``regime_dtq_memo.md``, 2026-08-12).

PRE-REGISTERED BINS AND TOLERANCE (fixed before this script was first run):

  PRIMARY METRIC -- I_loop at the window end t = 2e-5 s, the one sample every
  arm carries regardless of its step count, measured as a relative deviation
  from the dt = 1e-8 arm.

  BIN (i) CONVERGED -- every arm, INCLUDING adaptive, within 2% of the
  dt = 1e-8 arm on the primary metric. 2% is chosen against a measured
  quantity, not by taste: the memo's own residual ratchet bias at dt = 1e-8
  was +3.1% (0.9219 A held vs 0.8945 A relaxed with the bound off), so a
  tolerance BELOW that bias is the honest test that the dt dependence is gone
  rather than merely shrunk. Verdict: fixed.

  BIN (ii) STILL dt-DEPENDENT -- any arm outside 2%. Report the residual
  signature (which arms, which direction, whether monotone in dt) and STOP.

  SECONDARY, REPORTED NOT GATING -- the max deviation over the checkpoint grid
  t = 2, 4, ... 20 us, each arm's trace linearly interpolated onto it. The
  endpoint can agree while the paths differ, and that is worth seeing; but an
  arm taking one 2e-5 step has no interior samples to interpolate from, so
  this cannot be the gate.

Also printed per arm: the step count, the step-cap/active-constraint census
(so a binding ``circuit`` bound is visible), and the sheath regime at the end.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/regime_crf_dtconv.py [arm ...]
"""

import csv
import os
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from regime_r2_overlap_gate import build_config  # noqa: E402
from cablp.solvers._sim1d import LAPDSim1D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T_END = 2.0e-5
REFERENCE_ARM = "dt1e-8"
TOLERANCE_REL = 0.02
CHECKPOINTS = np.arange(2.0e-6, 2.0e-5 + 1.0e-12, 2.0e-6)

ARMS = [
    ("adaptive", None),
    ("dt3e-6", 3.0e-6),
    ("dt1e-6", 1.0e-6),
    ("dt3e-7", 3.0e-7),
    ("dt1e-7", 1.0e-7),
    ("dt1e-8", 1.0e-8),
]


def run_arm(label, dt):
    params, flags = build_config(20, True)
    params["dt_save"] = 1.0e-3  # no trajectory saves inside the window
    sim = LAPDSim1D(params, flags)
    rows = []

    class Tracker:
        def update(self, p):
            solve = sim._cathode_solve
            res = None
            if solve is not None and solve.beam_result is not None:
                res = solve.beam_result.result
            rows.append(
                dict(
                    t=p.time,
                    dt=p.accepted_dt,
                    step_cap=p.step_cap,
                    constraint=p.active_constraint,
                    dt_circuit=getattr(p, "dt_circuit", float("nan")),
                    I_loop=float(sim._circuit_I_loop),
                    V_dis_step=float(sim._circuit_V_dis_step),
                    V_b=(float(res.V_b) if res is not None else float("nan")),
                    phi_c=(float(res.phi_c) if res is not None else float("nan")),
                    regime=(str(res.regime) if res is not None else ""),
                )
            )

    sim.run(
        t_end=T_END,
        dt=dt,
        max_steps=2000000,
        progress_tracker=Tracker(),
        progress_interval_s=0.0,
    )

    path = os.path.join(HERE, f"regime_crf_dtconv_{label}.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    census = {}
    for r in rows:
        key = (r["step_cap"], r["constraint"])
        census[key] = census.get(key, 0) + 1
    print(f"== arm {label}  steps={len(rows)}")
    print("   census:", sorted(census.items(), key=lambda kv: -kv[1])[:6])
    last = rows[-1]
    print(f"   end t={last['t']:.6e}  I_loop={last['I_loop']:.6g} A  "
          f"V_dis={last['V_dis_step']:.6g} V  V_b={last['V_b']:.6g} V  "
          f"phi_c={last['phi_c']:.6g} V  regime={last['regime']!r}")
    return rows


def main():
    only = sys.argv[1:] or None
    traces = {}
    for label, dt in ARMS:
        if only and label not in only:
            continue
        traces[label] = run_arm(label, dt)
        print()

    if REFERENCE_ARM not in traces:
        print(f"(reference arm {REFERENCE_ARM} not run; no verdict)")
        return

    t_ref = np.array([r["t"] for r in traces[REFERENCE_ARM]])
    I_ref = np.array([r["I_loop"] for r in traces[REFERENCE_ARM]])
    end_ref = I_ref[-1]
    ref_grid = np.interp(CHECKPOINTS, t_ref, I_ref)

    print("=" * 72)
    print(f"P1 convergence table   reference {REFERENCE_ARM}: "
          f"I_loop(t={T_END:g}) = {end_ref:.6g} A")
    print(f"   pre-registered tolerance: {TOLERANCE_REL:.1%} on the endpoint")
    print(f"{'arm':>10}  {'steps':>7}  {'I_end [A]':>12}  {'dev_end':>10}  "
          f"{'dev_grid(max)':>14}")
    worst = 0.0
    for label, _ in ARMS:
        if label not in traces:
            continue
        rows = traces[label]
        I_end = rows[-1]["I_loop"]
        dev_end = abs(I_end - end_ref) / abs(end_ref)
        t_a = np.array([r["t"] for r in rows])
        I_a = np.array([r["I_loop"] for r in rows])
        grid = np.interp(CHECKPOINTS, t_a, I_a)
        dev_grid = float(np.max(np.abs(grid - ref_grid) / np.abs(ref_grid)))
        worst = max(worst, dev_end)
        print(f"{label:>10}  {len(rows):>7}  {I_end:>12.6g}  {dev_end:>10.3%}  "
              f"{dev_grid:>14.3%}")

    print()
    if worst <= TOLERANCE_REL:
        print(f"BIN (i) CONVERGED: worst endpoint deviation {worst:.3%} "
              f"<= {TOLERANCE_REL:.1%}. The dt dependence is gone.")
    else:
        print(f"BIN (ii) STILL dt-DEPENDENT: worst endpoint deviation "
              f"{worst:.3%} > {TOLERANCE_REL:.1%}.")


if __name__ == "__main__":
    main()
