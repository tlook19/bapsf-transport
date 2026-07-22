"""Fit the loop inductance L to the MEASURED current edges (M6 circuit refit).

The original circuit fit (fit_es1_circuit.py) matched the V_dis trace, which
is nearly L-blind on the plateau (dI/dt ~ 0 there); the edges are where L
lives, and the switch hardware paper (Pribyl & Gekelman, RSI 75, 669 (2004))
confirms the fall is a real flyback freewheel, so both edges are honest L
constraints. Method: integrate the loop ODE

    L dI/dt = V0 - Q/C - I*R_comp - V_dis_meas(t)

driving it with the MEASURED discharge-voltage trace (no plasma model at
all), and score the resulting I(t) against the measured current over the
rise (t in [-0.5, 4] ms) and fall (t in [19.9, 22] ms) windows. R_comp, C,
and V0 keep their trace-fit values (C is hardware-confirmed at 8.4 F within
electrolytic tolerance of the fitted 8.9). During the fall the switch is
open and the flyback path carries I through the plasma alone:
L dI/dt = -V_dis_meas(t).

Usage:
    python scripts/fit_circuit_edges.py
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V0 = 173.6
R_COMP = 5.72e-3
C_BANK = 8.9
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

    print(f"edge-fit L = {L_best * 1e6:.1f} uH  (current model: 6.6 uH; "
          f"hardware box from fall/rise arithmetic: 15-25 uH)")
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
