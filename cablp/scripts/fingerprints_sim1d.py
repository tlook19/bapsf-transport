"""Extract the mechanism-campaign fingerprints from a saved sim1d run.

The regime-universal targets (CATHODE_IDRIVEN_PLAN.md, measured ES1-3
overlays at fixed fueling):

  (a) fractional late current ramp, +5 ms -> end of drive  (~+10 %)
  (b) plateau V_dis decline, +5 ms -> end of drive        (~-7..-10 V),
      still evolving at end of drive
  peak timing (~+19.8 ms, end-of-drive class), t90, plateau level

V_dis uses the dt-weighted average differenced from the running
``circuit_V_dis_dt_integral`` (the inductor's view) when the run carries
it; otherwise the loop reconstruction ``V0 - Q/C - I*R - L*dI/dt`` from
the smooth I(t), the same definition the measurement used. Also reports the T_s trajectory,
honest P_cathode_i, and the power-balance energy ledger when present.

Usage::

    python scripts/fingerprints_sim1d.py run.h5 [run2.h5 ...]
"""

import sys

import numpy as np

from cablp.solvers._sim1d import load_result_hdf5


def non_ignited_message(result, caller):
    """Return the NON-IGNITED diagnosis for a run with no main_discharge.

    Every fingerprint is defined relative to the main-discharge origin. A run
    that never reached that phase has no origin, and the old ``times[0]``
    fallback silently reported pre-breakdown noise as drive-phase
    fingerprints. Fail loudly instead, naming the terminal phase and any
    ignition guard that fired.
    """
    phases = np.asarray(getattr(result, "phase", ()), dtype=str)
    terminal = str(phases[-1]) if phases.size else "<no samples>"
    events = getattr(result, "phase_events", None) or {}
    reasons = [str(reason) for reason in np.asarray(events.get("reason", ()))]
    times = np.asarray(events.get("time", ()), dtype=float)
    guards = [
        f"{reason} at t={time:.6e} s"
        for time, reason in zip(times, reasons)
        if reason in {"ignition_stalled", "prebreakdown_timeout"}
    ]
    abort = getattr(result, "ignition_abort", None)
    guard_text = "; ".join(guards) if guards else "no ignition-guard event"
    abort_text = ""
    if abort:
        abort_text = " | ignition_abort: " + " ".join(
            f"{key}={abort[key]}" for key in sorted(abort)
        )
    return (
        f"NON-IGNITED RUN: {caller} found no sample in the 'main_discharge' "
        f"phase, so this run has no discharge origin and CANNOT be "
        f"fingerprinted. Terminal phase: {terminal!r}. Ignition guards: "
        f"{guard_text}{abort_text}"
    )


def _origin_s(result):
    phases = np.asarray(getattr(result, "phase", ()), dtype=str)
    times = np.asarray(result.time, dtype=float)
    hits = np.flatnonzero(phases == "main_discharge")
    if not hits.size:
        raise RuntimeError(
            non_ignited_message(result, "fingerprints_sim1d._origin_s")
        )
    return float(times[hits[0]])


def _drive_end_ms(t_ms, phases):
    in_drive = np.asarray(phases, dtype=str) == "main_discharge"
    hits = np.flatnonzero(in_drive)
    return float(t_ms[hits[-1]]) if hits.size else float(t_ms[-1])


def report(path):
    result = load_result_hdf5(path)
    diag = result.cathode_diagnostics
    params = dict(getattr(result, "params", None) or {})
    t_ms = (np.asarray(result.time, float) - _origin_s(result)) * 1e3
    I = np.asarray(diag["source_I_tot"], float)
    t_end = _drive_end_ms(t_ms, result.phase)

    drive = (t_ms >= 0.0) & (t_ms <= t_end)
    late = (t_ms >= 5.0) & (t_ms <= t_end)
    plateau = (t_ms >= 15.0) & (t_ms <= min(19.5, t_end))
    early = (t_ms >= 1.0) & (t_ms <= 5.0)

    # WP-D arm marker. Printed as a delta only: "local" is the production
    # stance and the config default, so a production artifact's fingerprint
    # output is unchanged and a nonlocal one cannot be mistaken for one.
    bpt = str(params.get("beam_product_transport", "local"))
    bpt_note = "" if bpt == "local" else f" [beam_product_transport={bpt}]"
    print(f"\n=== {path} (drive end +{t_end:.2f} ms){bpt_note} ===")
    Ipk = float(np.max(I[drive]))
    tpk = float(t_ms[drive][np.argmax(I[drive])])
    Iplat = float(np.median(I[plateau])) if plateau.any() else np.nan
    # t90 on the same definition as the scorecard: first crossing of 90 %
    # of the drive-phase peak.
    above = np.flatnonzero(drive & (I >= 0.9 * Ipk))
    t90 = float(t_ms[above[0]]) if above.size else np.nan
    I5 = float(np.interp(5.0, t_ms[drive], I[drive]))
    Iend = float(I[drive][-1])
    print(f"peak {Ipk:.0f} A at +{tpk:.2f} ms | plateau(15-19.5) {Iplat:.0f} A"
          f" | t90 {t90:.2f} ms")
    print(f"(a) late ramp +5ms->end: {100.0 * (Iend / I5 - 1.0):+.1f} %"
          f"  (I(+5) {I5:.0f} -> I(end) {Iend:.0f} A)")

    # --- V_dis, best available honest reading, in preference order:
    # (1) the dt-weighted average from the running \int V_dis dt (bias-free
    # by construction, and vintage-proof: on runs saved before 2026-07-21
    # circuit_V_dis_step held the last-step sample, which is dt-biased --
    # saves land on dt-capped steps that sample the low state of the knee
    # sawtooth, measured ~25 V low on the ES1 plateau; since then the key
    # itself stores this same interval average); (2) the loop
    # reconstruction from the smooth circuit_I_loop state. Per-solve
    # source_I_tot is chattery (L*dI/dt of it fabricates +-10 V).
    tsec_all = np.asarray(result.time, float)
    Vint = np.asarray(
        diag.get("circuit_V_dis_dt_integral", np.zeros_like(I)), float
    )
    if np.any(Vint != 0.0):
        v_label = "V_dis_tavg"
        dt_save = np.diff(tsec_all)
        with np.errstate(invalid="ignore", divide="ignore"):
            V_mid = np.diff(Vint) / dt_save
        # midpoint series -> per-save series (pad front); zero-length save
        # intervals (none expected) would be inf/nan and are masked by the
        # window fits below.
        V = np.concatenate([[V_mid[0] if V_mid.size else 0.0], V_mid])
    else:
        V = np.zeros_like(I)
        v_label = "V_loop_recon"
        V0 = float(params.get("V_bank", 173.6))
        R = float(params.get("R_comp", 5.72e-3))
        L = float(params.get("L_parasitic_H", 6.6e-6))
        C = params.get("C_bank_F")
        tsec = np.asarray(result.time, float)
        Iloop = np.nan_to_num(
            np.asarray(diag.get("circuit_I_loop", np.zeros_like(I)), float),
            nan=0.0,
        )
        # Pre-breakdown saves carry NaN diagnostics (no solve): zero
        # current there is the physical reading for the loop integral.
        Iz = Iloop if np.any(Iloop != 0.0) else np.nan_to_num(I, nan=0.0)
        Q = np.concatenate(
            ([0.0], np.cumsum(0.5 * (Iz[1:] + Iz[:-1]) * np.diff(tsec)))
        )
        dIdt = np.gradient(Iz, tsec)
        V = V0 - (Q / float(C) if C else 0.0) - Iz * R - L * dIdt

    def _vfit(center_ms, half=1.0):
        win = late & (np.abs(t_ms - center_ms) <= half)
        if win.sum() < 4:
            return np.nan
        return float(np.polyval(
            np.polyfit(t_ms[win], V[win], 1), center_ms
        ))

    V5 = _vfit(6.0)
    Vend = _vfit(t_end - 1.0)
    # "Still evolving at end": the fitted slope over the last 2 ms.
    tail = (t_ms >= t_end - 2.0) & (t_ms <= t_end)
    tail_slope = float(np.polyfit(t_ms[tail], V[tail], 1)[0]) if tail.sum() > 3 else np.nan
    print(f"(b) {v_label} +6ms->end-1: {Vend - V5:+.1f} V  ({V5:.1f} -> {Vend:.1f} V)"
          f" | end slope {tail_slope:+.2f} V/ms")

    # --- Mechanism internals.
    Ts = np.asarray(diag.get("T_s_surface", np.zeros_like(I)), float)
    if np.any(Ts > 0.0):
        Ts5 = float(np.interp(5.0, t_ms, Ts))
        ts_slope = (
            float(np.polyfit(t_ms[tail], Ts[tail], 1)[0])
            if tail.sum() > 3 else np.nan
        )
        print(f"T_s: start {Ts[drive][0]:.0f} K, +5ms {Ts5:.0f} K, "
              f"end {Ts[drive][-1]:.0f} K | end slope {ts_slope:+.3f} K/ms")
    th = np.asarray(diag.get("surface_theta", np.ones_like(I)), float)
    if np.any(th < 1.0):
        pe = np.asarray(diag.get("phi_wf_eff", np.zeros_like(I)), float)
        th5 = float(np.interp(5.0, t_ms, th))
        print(f"theta: +5ms {th5:.3f}, end {th[drive][-1]:.3f} | "
              f"phi_eff: start {pe[drive][0]:.3f} -> end {pe[drive][-1]:.3f} eV")
    Pi = np.asarray(diag.get("source_P_cathode_i", np.zeros_like(I)), float)
    if np.any(Pi != 0.0):
        print(f"P_cathode_i: early(1-5ms) {np.median(Pi[early]) / 1e3:.1f} kW, "
              f"plateau {np.median(Pi[plateau]) / 1e3:.1f} kW")
    if "warming_E_ion_J" in diag:
        E = {k: float(np.asarray(diag[f"warming_E_{k}_J"], float)[-1])
             for k in ("heater", "ion", "rad", "emis", "cond")}
        if any(v != 0.0 for v in E.values()):
            net = E["heater"] + E["ion"] - E["rad"] - E["emis"] - E["cond"]
            print(f"ledger [J]: ion {E['ion']:.0f}, cond->substrate {E['cond']:.0f}, "
                  f"emis {E['emis']:.0f}, rad-heater {E['rad'] - E['heater']:.0f}, "
                  f"net-into-skin {net:.0f}")

    # Plateau chatter of the per-solve V_b, for the bridge/annuli A/Bs.
    Vb = np.asarray(diag.get("source_V_b", np.zeros_like(I)), float)
    if plateau.any() and np.any(Vb[plateau] != 0.0):
        vb = Vb[plateau]
        print(f"per-solve V_b plateau: p5/p50/p95 = {np.percentile(vb, 5):.0f}/"
              f"{np.percentile(vb, 50):.0f}/{np.percentile(vb, 95):.0f} V, "
              f"sigma {np.std(vb):.1f} V")
        if v_label == "V_dis_tavg":
            vs = V[plateau][np.isfinite(V[plateau])]
            print(f"V_dis_tavg plateau: p5/p50/p95 = {np.percentile(vs, 5):.0f}/"
                  f"{np.percentile(vs, 50):.0f}/{np.percentile(vs, 95):.0f} V, "
                  f"sigma {np.std(vs):.1f} V")


def main(argv):
    for path in argv:
        report(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
