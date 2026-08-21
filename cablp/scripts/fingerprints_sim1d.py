"""Extract the mechanism-campaign fingerprints from a saved sim1d run.

The regime-universal targets (measured ES1-3 overlays at fixed
fueling):

  (a) fractional late current ramp, +5 ms -> end of drive  (~+10 %)
  (b) plateau V_dis decline, +5 ms -> end of drive        (~-7..-10 V),
      still evolving at end of drive
  peak timing (~+19.8 ms, end-of-drive class), t90, plateau level

V_dis uses the dt-weighted average differenced from the running
``circuit_V_dis_dt_integral`` (the inductor's view) when the run carries
it; otherwise the loop reconstruction ``V0 - Q/C - I*R - L*dI/dt`` from
the smooth I(t), the same definition the measurement used. Also reports the T_s trajectory,
honest P_cathode_i, and the power-balance energy ledger when present.

A trailing block adds the high-precision watch-class rows, tagged ``(i)``,
``(iii)`` and ``(iv)``: the sub-bin interpolated breakdown-trigger crossing
times against the save-quantized phase-label edge; the plateau V_dis mean,
plateau current and gap P_ohmic ledger; and the breakdown-phase Te_max with
its time and cell. Those rows carry more significant figures than the rows
above them ON PURPOSE -- they are read against sub-volt, sub-percent and
sub-kW predictions. Every row above that block is byte-frozen: it is quoted
in stored comparisons, so this tool only ever gains rows, never re-prints
an old one differently.

Usage::

    python scripts/fingerprints_sim1d.py run.h5 [run2.h5 ...]
"""

import sys

import numpy as np

from cablp.solvers._sim1d import load_result_hdf5


def non_ignited_message(result, caller):
    """Return the NON-IGNITED diagnosis for a run with no main_discharge.

    DUPLICATED, deliberately, from ``compare_sim1d_es1.non_ignited_message``:
    this tool stays standalone, and importing the scorer for a 20-line numpy
    helper would hand it the whole driver. Both copies are exercised together
    by smoke_sim1d, so they cannot drift silently -- keep them in step.

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


def _wpe_arm_line(params):
    """Return the one-line WP-E arm header for a run's saved resolved params.

    DUPLICATED, deliberately, from ``compare_sim1d_es1.wpe_arm_line``, on the
    same grounds as ``non_ignited_message`` above: this tool stays standalone.
    Keep the two in step.

    {local, tail_walk} is a declared BRACKET, so the arm is printed ALWAYS,
    not as a delta -- a fingerprint set is incomplete without it, and a
    delta-only label cannot tell "this run was local" from "this artifact
    predates the label". Artifacts written before WP-E carry neither key and
    are labelled "pre-WP-E". The tail energy is read only under
    ``"tail_walk"`` and is marked inert otherwise.
    """
    params = params or {}
    if "heating_anomalous_transport" not in params:
        return (
            "WP-E arm: pre-WP-E (heating_anomalous_transport absent from this "
            "run's saved params -- the QL heating locality closure did not "
            "exist when this artifact was written, so its behaviour is the "
            "'local' arm by construction)"
        )
    transport = str(params["heating_anomalous_transport"])
    tail = params.get("heating_anomalous_tail_energy_eV")
    tail_text = "<absent>" if tail is None else f"{float(tail):g} eV"
    if transport == "local":
        tail_text += " (inert under 'local')"
    return (
        f"WP-E arm: heating_anomalous_transport={transport} | "
        f"heating_anomalous_tail_energy_eV={tail_text}"
    )


def _clamp_notice(requested, limit, kind, site, extent):
    """Return the window bound actually used, ANNOUNCING a clamp that binds.

    Mirrors ``compare_sim1d_es1._clamp_window_bound``: the arithmetic is the
    ``max``/``min`` it replaces, and nothing is printed when the clamp is
    inert. A window silently shortened to fit a short trace reports a number
    that LOOKS like the configured quantity and is not.
    """
    used = max(requested, limit) if kind == "start" else min(requested, limit)
    if used == requested:
        return used
    print(
        f"  CLAMP NOTICE [{site}]: {kind} bound requested {requested:.4g} ms, "
        f"USED {used:.4g} ms -- clamped to the {extent} extent. This window "
        f"is NOT the requested one, so these rows are not comparable to runs "
        f"reported over the full window."
    )
    return used


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

    # WP-D arm marker. Printed as a delta only: "local" is the production
    # stance and the config default, so a production artifact's fingerprint
    # output is unchanged and a nonlocal one cannot be mistaken for one.
    bpt = str(params.get("beam_product_transport", "local"))
    bpt_note = "" if bpt == "local" else f" [beam_product_transport={bpt}]"
    print(f"\n=== {path} (drive end +{t_end:.2f} ms){bpt_note} ===")
    print(_wpe_arm_line(params))

    # The masks are pure computation and were previously built above the
    # header; they are built here so the plateau clamp notice lands UNDER the
    # run it belongs to.
    drive = (t_ms >= 0.0) & (t_ms <= t_end)
    late = (t_ms >= 5.0) & (t_ms <= t_end)
    plateau_end = _clamp_notice(
        19.5, t_end, "end", "plateau (15-19.5 ms)", "drive-phase"
    )
    plateau = (t_ms >= 15.0) & (t_ms <= plateau_end)
    early = (t_ms >= 1.0) & (t_ms <= 5.0)

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
        # Fallbacks for a result whose saved params omit the key. V_bank and
        # R_comp match the current production stance
        # (compare_sim1d_es1.PARAM_OVERRIDES); the L fallback DELIBERATELY
        # DOES NOT -- it is frozen at 6.6e-6, the value that was the config
        # default while the only artifacts that can reach this branch were
        # being produced. In practice none can: results/io.py saves fully
        # RESOLVED params, so every artifact written since the circuit state
        # existed carries L_parasitic_H and takes the params.get hit. This
        # branch is reachable only by artifacts predating the parasitic
        # inductor entirely (before 07119af), which is exactly why tracking
        # the live stance here would be wrong -- it would reconstruct their
        # loop voltage against a circuit they never ran.
        V0 = float(params.get("V_bank", 177.843))
        R = float(params.get("R_comp", 7.2244e-3))
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

    # --- The high-precision watch-class rows. -------------------------------
    #
    # Everything above is byte-frozen: those lines are quoted in stored
    # comparisons across the campaign record, so this block only ever APPENDS.
    # It also carries its own precision, deliberately finer than the rows
    # above, because the quantities here are tested against sub-volt,
    # sub-percent and sub-kW predictions that the older rows round away.
    #
    # NOT EMITTED HERE: knee time and ramp slope. The "knee" of the current
    # trace has no computed definition anywhere in this repo -- it appears
    # only as prose and as a hard-coded ~4.5 ms landmark
    # (verify_sim1d_r3_a11.py, --phase knee) -- so there is nothing to
    # transcribe, and inventing one would make the row's value an artifact of
    # this file rather than a property of the run.

    # (i) Breakdown-trigger crossing times, SUB-BIN. The solver linearly
    # interpolates each threshold crossing between the two consecutive
    # trigger-check samples that bracket it (``_current_threshold_time``,
    # solver.py) and stores the result; that is a solver-step-resolved reading
    # of the current trace. It is NOT the phase-label edge, which is the first
    # SAVE carrying the new label and is therefore quantized to the save
    # cadence -- the two differ by up to one save interval, which is the whole
    # reason this row exists.
    origin_ms = _origin_s(result) * 1.0e3
    for threshold_key, trigger_key in (
        ("I_prebreakdown", "t_prebreakdown_trigger"),
        ("I_breakdown", "t_breakdown_trigger"),
    ):
        t_trigger = float(getattr(result, trigger_key, np.nan))
        if not np.isfinite(t_trigger):
            continue
        threshold = params.get(threshold_key)
        threshold_text = (
            "<absent>" if threshold is None else f"{float(threshold):.6g} A"
        )
        print(f"(i) {trigger_key}: {t_trigger * 1.0e3:.4f} ms absolute, "
              f"{t_trigger * 1.0e3 - origin_ms:+.4f} ms vs the main-discharge "
              f"origin | threshold {threshold_key}={threshold_text}")
    print(f"(i) phase-label edge (first 'main_discharge' save): "
          f"{origin_ms:.4f} ms absolute -- save-cadence quantized, carried as "
          f"the reference the interpolated crossings above are offset from")

    # (iii) The 21q plateau observables, over the SAME plateau window the rows
    # above use (15 ms -> the clamp-checked end; no second clamp notice is
    # printed because no second window is opened).
    if plateau.any():
        vs_full = V[plateau][np.isfinite(V[plateau])]
        v_text = (
            "<no finite samples>" if not vs_full.size
            else f"{np.mean(vs_full):.4f} V"
        )
        print(f"(iii) {v_label} plateau mean: {v_text}")
        Ip = I[plateau][np.isfinite(I[plateau])]
        if Ip.size:
            print(f"(iii) plateau current: mean {np.mean(Ip):.3f} A | "
                  f"median {np.median(Ip):.3f} A")
        # The gap P_ohmic ledger row. ``P_ohmic = I_tot * V_p`` is the
        # circuit's I^2 R_p dissipated in the plasma between cathode and
        # anode (funcs/_cathode_solver.py), and physics/cathode.py deposits
        # ALL of it into that end's gap cells through weights that normalize
        # to one -- so each end's P_ohmic IS its gap booking, and the two are
        # reported separately as well as summed. The twin end is absent on a
        # single-ended run and reads NaN there, which is why it is presence-
        # gated rather than summed through.
        ohmic_total = 0.0
        ohmic_texts = []
        for end_key in ("source_P_ohmic", "end_P_ohmic"):
            series = np.asarray(diag.get(end_key, np.full_like(I, np.nan)), float)
            finite = series[plateau][np.isfinite(series[plateau])]
            if not finite.size:
                ohmic_texts.append(f"{end_key} <absent>")
                continue
            end_mean = float(np.mean(finite))
            ohmic_total += end_mean
            ohmic_texts.append(f"{end_key} {end_mean / 1.0e3:.4f} kW")
        print("(iii) gap P_ohmic plateau mean: "
              f"{' | '.join(ohmic_texts)} | total {ohmic_total / 1.0e3:.4f} kW")

    # (iv) Breakdown-phase Te_max, over the saves the solver itself labelled
    # 'breakdown' (result.phase), reported with the cell it was attained in --
    # the location is the point of the row, since a gap-local and a
    # far-column maximum are different mechanisms.
    phase_labels = np.asarray(getattr(result, "phase", ()), dtype=str)
    breakdown = phase_labels == "breakdown"
    if breakdown.any():
        Te_bd = np.asarray(result.Te, float)[breakdown]
        if np.any(np.isfinite(Te_bd)):
            sample, cell = np.unravel_index(
                int(np.nanargmax(Te_bd)), Te_bd.shape
            )
            print(f"(iv) breakdown-phase Te_max: {Te_bd[sample, cell]:.4f} eV "
                  f"at {t_ms[breakdown][sample]:+.4f} ms "
                  f"(cell {int(cell)} of {Te_bd.shape[1]})")


def main(argv):
    for path in argv:
        report(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
