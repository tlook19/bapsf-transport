"""Fit the discharge-circuit constants from the ES1 current/voltage trace.

*** SUPERSEDED, 2026-08-03. This script's result is NO LONGER the production ***
*** circuit stance, and its numbers below are kept only as a record of what  ***
*** was replaced. Do not re-adopt them; do not "refresh" them in place.      ***

Why it was superseded: the four-parameter FREE fit performed here is
NEAR-SINGULAR. The design columns ``[1, -Q, -I, -dI/dt]`` are near-collinear
because I is nearly constant over the plateau and Q is nearly linear in t,
giving ``corr(V0, R) = 0.997``. The fit therefore recovers a DEGENERATE
DIRECTION, not the circuit, and its 0.14 V rms is in-sample. Window
sensitivity makes this concrete: R = 5.71 / 4.42 / 4.27 / 1.94 mOhm as the
start of the window moves 0.3 -> 1 -> 5 -> 10 ms. The formal +-0.079 mOhm bar
this fit reports is meaningless.

The defect was INVISIBLE AT ES1 -- it only surfaced on ladder transfer.
Reconstructing measured plateau V_dis, this parameterization leaves residuals
-0.136 / +6.329 / +5.677 / +5.786 V at ES1/2/3/4.

What replaced it: a CONSTRAINED refit with V0 PINNED per rung to its measured
pre-shot reading and C, R, L shared across four rungs (N = 1952, window
0.3-19.8 ms). Conditioning drops 89.3 -> 4.7; residuals become
+0.010 / +0.139 / -0.053 / -0.309 V. Adopted stance: V_bank 177.843 V (ES1),
R_comp 7.2244 mOhm, C_bank_F 9.5 F (jackknife R = 7.213 +- 0.043 mOhm,
C = 9.56 +- 0.66 F). See ``compare_sim1d_es1.PARAM_OVERRIDES`` and the
circuit docstrings in ``cablp/solvers/_sim1d/core/config.py``.

The Kirchhoff loop ``V_dis(t) = V0 - Q(t)/C - I*R - L*dI/dt`` holds
identically for the true circuit whatever the plasma does, so a least-squares
fit over the drive phase constrains the Thevenin-equivalent circuit. The
plasma physics only chooses the trajectory along this constraint. That much
is still true -- what fails is leaving all four parameters free on one rung.

Reference result (2026-07-19, rms residual 0.14 V over 0.3-19.8 ms), SUPERSEDED:

    V0 = 173.6 V   C_eff = 8.9 F   R = 5.7 mOhm   L = 6.6 uH

Hardware caveats this fit was once said to adjudicate, as they now stand:

- C = 4 F is REJECTED, and that conclusion SURVIVES -- now doubly, at
  5 sigma on the fit and by being below the hardware band floor. But
  "the bank's nominal maximum" was wrong twice over: the nominal is 8.40 F
  (10 switches x 2 minibanks x 35 cans = 700 Chemi-Con 36DY at 12,000 uF),
  and with a -10/+50% per-can tolerance the allowed total is
  [7.56, 12.60] F -- so 8.40 F is a near-FLOOR, not a maximum. The
  "~7 V of unexplained slow EMF recovery" that the effective 8.9 F was said
  to imply DISSOLVES: it was an artifact of leaving V0 free here. There was
  never a capacitance anomaly.
- R_comp = 10 mOhm (previously inferred from the peak I/V ratio) was said to
  be superseded by the fitted 5.7 mOhm. That was NOT an adjudication. The
  measured value is 7.22 mOhm, and the discarded 10 mOhm was CLOSER to right
  than the fit that replaced it.
- This fit's L = 6.6 uH is SUPERSEDED, and was the last of the four to be
  retired (2026-08-03). The independent tail measurement (e-fold ~0.6 ms at
  ~10-18 mOhm loop) and the 18 V early-rise sag at measured dI/dt both give
  L ~ 7-10 uH; L is owned by ``fit_circuit_edges.py``, whose fall (flyback
  volt-second) arm gives 7.2-8.4 uH independently of any circuit constant and
  whose rise arm gives 7.6 uH. Production is now 8.1e-6. This bullet used to
  end by deferring instead to a "15-25 uH" box -- contradicting, in the same
  sentence, the two 7-10 uH estimates it had just cited. That box was never
  ``fit_circuit_edges.py``'s answer and was RETRACTED on 2026-07-21 (it assumed
  a constant-R freewheel, which the measured collapsing V_dis falsifies); see
  the retraction recorded in that script's module docstring.

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
