"""Fit the loop inductance L to the MEASURED current edges (M6 circuit refit).

The original circuit fit (fit_es1_circuit.py) matched the V_dis trace, which
is nearly L-blind on the plateau (dI/dt ~ 0 there); the edges are where L
lives, and the switch hardware paper (Pribyl & Gekelman, RSI 75, 669 (2004))
confirms the fall is a real flyback freewheel, so both edges are honest L
constraints. Method: integrate the loop ODE

    L dI/dt = V0 - Q/C - I*R_comp - V_dis_meas(t)

driving it with the MEASURED discharge-voltage trace (no plasma model at
all), and score the resulting I(t) against the measured current over the
rise (t in [-0.5, 4] ms) and fall (t in [19.9, 22] ms) windows. During the
fall the switch is open and the flyback path carries I through the plasma
alone: L dI/dt = -V_dis_meas(t).

V0, R_comp and C below are the CORRECTED production circuit stance
(2026-08-03): V0 measured pre-shot, R_comp and C from the V0-pinned four-rung
constrained refit. They supersede the near-singular ES1-only free fit
(173.6 V / 5.72 mOhm / 8.9 F) this script previously drove itself with; see
the PARAM_OVERRIDES comment in compare_sim1d_es1.py.

On the bank capacitance: this file used to say C was "hardware-confirmed at
8.4 F within electrolytic tolerance of the fitted 8.9" -- right to name 8.40 F
as the nominal and right to invoke electrolytic tolerance, but wrong to treat
8.40 F as a target to reconcile to. The tolerance is -10/+50% on 700 cans, so
the allowed total is [7.56, 12.60] F and 8.40 F is a near-FLOOR. Nominal
8.40 F, allowed [7.56, 12.60] F, MEASURED 9.5 F.

This script is the instrument that OWNS L (the plateau is nearly L-blind;
the edges are where L lives). Its rise arm moves with the inputs above; its
FALL arm does not -- the flyback branch uses ``emf = -V_meas`` and touches
neither V0, R nor C, so the fall-only answer (~8.2 uH) is the same before and
after the 2026-08-03 circuit correction and is the strongest single number in
the campaign. Production adopted L = 8.1e-6 on 2026-08-03 on the strength of
this arm plus the V0-pinned plateau refit's 8.06 uH; the instruments cannot
discriminate 8.06/8.1/8.23 (+10%-rms band 7.9-8.5 uH).

RETRACTION, recorded at its origin: this script's own ``print`` statement used
to advertise a "hardware box from fall/rise arithmetic: 15-25 uH". That box is
NOT hardware -- there is no geometry, loop-area or busbar estimate of L
anywhere in the campaign. It is ``L = tau_fall * R_load`` with the plasma
treated as a CONSTANT RESISTOR (0.43 ms x 50.6 mOhm = 21.7 uH). The assumption
is measurably false: the measured V_dis collapses 16x within 0.2 ms of the
fall (88.8 -> 15.5 -> 9.3 -> 5.3 V at t = 20.0/20.2/20.5/21.0 ms) while the
current is still near I0. This script was WRITTEN to test that estimate and
REFUTED it on the day it was written (2026-07-21), and the estimate was
retracted the same day -- the text nonetheless survived here and propagated
into the config docstring and PARAM_OVERRIDES, where it was later cited as the
reason to keep L at the superseded 6.6 uH. The edge instrument EXCLUDES
15-25 uH at 4.6-7.1x its minimum residual (and puts 6.6 uH at 2.0x).

Usage:
    python scripts/fit_circuit_edges.py
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
V0 = 177.843
R_COMP = 7.2244e-3
C_BANK = 9.5
T_DRIVE_END_MS = 19.99

RISE_WIN = (-0.5, 4.0)
FALL_WIN = (19.9, 22.0)


def simulate(L_H, t_ms, V_meas):
    """Integrate the loop with the measured V_dis; switch opens at drive end."""
    I = np.zeros_like(t_ms)
    Q = 0.0
    for k in range(1, t_ms.size):
        dt = (t_ms[k] - t_ms[k - 1]) * 1e-3
        if t_ms[k] <= T_DRIVE_END_MS:
            emf = V0 - Q / C_BANK - I[k - 1] * R_COMP - V_meas[k - 1]
        else:
            # flyback freewheel: bank disconnected, plasma load only
            emf = -V_meas[k - 1]
        I[k] = max(I[k - 1] + dt * emf / L_H, 0.0)
        Q += I[k] * dt
    return I


def edge_cost(L_H, t_ms, I_meas, V_meas):
    I_sim = simulate(L_H, t_ms, V_meas)
    cost = 0.0
    for lo, hi in (RISE_WIN, FALL_WIN):
        m = (t_ms >= lo) & (t_ms <= hi)
        cost += float(np.mean((I_sim[m] - I_meas[m]) ** 2))
    return cost, I_sim


def main():
    ov = np.load(HERE / "data/es1_sim1d_overlay.npz")
    t = np.asarray(ov["discharge_time_ms"], dtype=float)
    I = np.asarray(ov["discharge_current_mean_a"], dtype=float)
    V = np.asarray(ov["discharge_voltage_positive_mean_v"], dtype=float)
    m = (t >= -1.0) & (t <= 24.0)
    t, I, V = t[m], I[m], V[m]
    V = np.nan_to_num(V, nan=float(np.nanmedian(V)))

    Ls = np.geomspace(2e-6, 100e-6, 61)
    costs = [edge_cost(L, t, I, V)[0] for L in Ls]
    L_best = float(Ls[int(np.argmin(costs))])
    # refine
    Ls2 = np.linspace(0.5 * L_best, 2.0 * L_best, 61)
    costs2 = [edge_cost(L, t, I, V)[0] for L in Ls2]
    L_best = float(Ls2[int(np.argmin(costs2))])
    cost, I_sim = edge_cost(L_best, t, I, V)

    # The V0-pinned four-rung V_dis refit independently prefers 8.06 uH, but it
    # is a plateau fit and the plateau is nearly L-blind, so the edge answer
    # governs. Production adopted 8.1e-6 on 2026-08-03 (see the L_parasitic_H
    # docstring in core/config.py); the instruments cannot discriminate
    # 8.06/8.1/8.23.
    print(f"edge-fit L = {L_best * 1e6:.1f} uH  (production: 8.1 uH; "
          f"plateau refit: 8.06 uH)")
    for label, (lo, hi) in (("rise", RISE_WIN), ("fall", FALL_WIN)):
        mm = (t >= lo) & (t <= hi)
        rms = np.sqrt(np.mean((I_sim[mm] - I[mm]) ** 2))
        print(f"  {label} window rms: {rms:.0f} A "
              f"(I range {I[mm].min():.0f}-{I[mm].max():.0f} A)")
    # per-edge independent fits, to expose tension between the edges
    for label, keep in (("rise-only", RISE_WIN), ("fall-only", FALL_WIN)):
        def cost_one(L):
            I_s = simulate(L, t, V)
            mm = (t >= keep[0]) & (t <= keep[1])
            return float(np.mean((I_s[mm] - I[mm]) ** 2))
        Lx = np.geomspace(2e-6, 100e-6, 121)
        Lb = float(Lx[int(np.argmin([cost_one(L) for L in Lx]))])
        print(f"  {label:9s} best L = {Lb * 1e6:.1f} uH")


if __name__ == "__main__":
    main()
