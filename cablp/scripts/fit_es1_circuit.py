"""Fit the discharge-circuit constants from the ES1 current/voltage trace.

The Kirchhoff loop ``V_dis(t) = V0 - Q(t)/C - I*R - L*dI/dt`` holds
identically for the true circuit whatever the plasma does, so an ordinary
least-squares fit over the drive phase recovers the Thevenin-equivalent
circuit the plasma actually sees. The plasma physics only chooses the
trajectory along this constraint.

Reference result (2026-07-19, rms residual 0.14 V over 0.3-19.8 ms):

    V0 = 173.6 V   C_eff = 8.9 F   R = 5.7 mOhm   L = 6.6 uH

Hardware caveats this fit adjudicated (see THESIS_NOTES.md section 2):

- C = 4 F (the bank's nominal maximum) is REJECTED: it demands either
  negative resistance or ~1.6 kA of float-supply recharge against a 120 A
  limit. The effective ~8.9 F implies ~7 V of unexplained slow EMF recovery
  (transistor V_CE drift with junction temperature is a candidate).
- R_comp = 10 mOhm (previously inferred from the peak I/V ratio) is
  superseded by the fitted 5.7 mOhm.
- The independent tail measurement (e-fold ~0.6 ms at ~10-18 mOhm loop)
  and the 18 V early-rise sag at measured dI/dt both give L ~ 7-10 uH,
  consistent with the fit.

Usage::

    python scripts/fit_es1_circuit.py
"""

from pathlib import Path

import numpy as np

OVERLAY = Path(__file__).resolve().parent / "data" / "es1_sim1d_overlay.npz"


def load_drive_phase(t_min_s=0.3e-3, t_max_s=19.8e-3):
    d = np.load(OVERLAY, allow_pickle=False)
    t = np.asarray(d["discharge_time_ms"], dtype=float) * 1e-3
    I = np.asarray(d["discharge_current_mean_a"], dtype=float)
    V = np.asarray(d["discharge_voltage_positive_mean_v"], dtype=float)
    m = (t > t_min_s) & (t < t_max_s) & np.isfinite(I) & np.isfinite(V)
    t, I, V = t[m], I[m], V[m]
    Q = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))])
    dIdt = np.gradient(I, t)
    return t, I, V, Q, dIdt


def fit_free(t, I, V, Q, dIdt):
    """Free fit of (V0, 1/C, R, L)."""
    A = np.column_stack([np.ones_like(t), -Q, -I, -dIdt])
    x, *_ = np.linalg.lstsq(A, V, rcond=None)
    resid = V - A @ x
    return {
        "V0_V": float(x[0]),
        "C_F": float(1.0 / x[1]) if x[1] > 1e-9 else np.inf,
        "R_ohm": float(x[2]),
        "L_H": float(x[3]),
        "rms_V": float(resid.std()),
    }


def fit_fixed_C(t, I, V, Q, dIdt, C_F, I_supply_A=0.0):
    """Fit (V0, R, L) at a prescribed capacitance and float-supply current."""
    A = np.column_stack([np.ones_like(t), -I, -dIdt])
    y = V + (Q - I_supply_A * t) / float(C_F)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ x
    return {
        "V0_V": float(x[0]),
        "R_ohm": float(x[1]),
        "L_H": float(x[2]),
        "rms_V": float(resid.std()),
    }


def main():
    t, I, V, Q, dIdt = load_drive_phase()
    free = fit_free(t, I, V, Q, dIdt)
    print(
        f"free fit:      V0={free['V0_V']:.1f} V  C_eff={free['C_F']:.2f} F  "
        f"R={free['R_ohm']*1e3:.2f} mOhm  L={free['L_H']*1e6:.1f} uH  "
        f"rms={free['rms_V']:.2f} V"
    )
    for C, ips, label in ((4.0, 0.0, "C=4 F, no supply"), (4.0, 120.0, "C=4 F, 120 A supply")):
        fx = fit_fixed_C(t, I, V, Q, dIdt, C, ips)
        print(
            f"{label:22s}: V0={fx['V0_V']:.1f} V  R={fx['R_ohm']*1e3:.2f} mOhm  "
            f"L={fx['L_H']*1e6:.1f} uH  rms={fx['rms_V']:.2f} V"
        )
    print(f"\ncharge drawn over drive phase: {Q[-1]:.1f} C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
