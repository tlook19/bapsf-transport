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

Row ``(ii)`` reports fast-phase ramp geometry. Every one of its members
carries a value, a declared equivalence band and a status, and NONE of them
can raise: "the feature is absent on this arm" is a reading, not a failure,
and must not take the other rows down with it.

Usage::

    python scripts/fingerprints_sim1d.py run.h5 [run2.h5 ...]
    python scripts/fingerprints_sim1d.py --pair post.h5=pre.h5 post.h5 pre.h5
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
    are labelled "pre-WP-E". The tail energy is read only when the tail is
    WALKED *and* ``heating_anomalous_tail_energy_keying="fixed"``, and is
    marked inert in both of the other cases -- under ``"phi_c"`` keying the
    walked arm's live birth energy is ``f*e*phi_c(t)`` and the printed
    constant is a number the run never touched. The label follows the
    artifact's own saved keying value and is omitted when that key is absent,
    so an artifact predating the keying closure is not mislabelled.
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
    keying = params.get("heating_anomalous_tail_energy_keying")
    if transport == "local":
        tail_text += " (inert under 'local')"
    elif keying is not None and str(keying) != "fixed":
        tail_text += (
            f" [INERT under {keying} keying; live E_tail = f*e*phi_c(t)]"
        )
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


# --- Row (ii): fast-phase ramp structure. ----------------------------------
#
# This operationalizes the campaign's "knee time + ramp slope" as the pair
# (t_ovs, S_ramp). The word "knee" is deliberately NOT used in any field name:
# this codebase already spends it on two unrelated things (the cathode V(I)
# emission knee, physics/cathode.py; the ~4.5 ms cathode-warming landmark,
# verify_sim1d_r3_a11.py), and a third meaning would make all three
# unsearchable.
#
# WHAT IS AND IS NOT CLAIMED. These rows are TRACE GEOMETRY. "Overshoot" and
# "ringback" name the shape of I(t) and nothing else -- whether that shape is
# a circuit ring, breakdown dynamics, or something else is NOT claimed here
# and must not be read out of these numbers. The overshoot/ringback structure
# is VERIFIED ON TWO ARMS ONLY (an ES1-class run and its pre-batch control)
# and is NOT generic: arms that saturate monotonically into the crawl have no
# such feature at all, and for those the existence criterion below reports
# the four members as 'absent'. Recording the absence IS the reading.
#
# The window is anchored on row (i)'s interpolated trigger and reaches only
# FORWARD. That is causal, not cosmetic: at the ES1-class rung the drive
# turn-on edge (the bank slewing into the cold gap at drive start) precedes
# the conductivity threshold crossing and is steeper than the fast-phase
# ramp, so any backward reach re-admits a feature that is not the one being
# measured.
#
# THAT ORDERING IS RUNG-SPECIFIC, and this comment previously asserted it
# without qualification. It is a property of the ES1-class rung, not of the
# instrument. At higher V_bank the turn-on surge is not separated from the
# trigger in the same way: on an ES2-class arm it can PEAK AFTER the
# trigger, and the forward window's slope maximum then lands on the window's
# first samples; on an ES3-class arm it MERGES with the fast-phase ramp and
# there is no isolated maximum at all. In both cases the forward-only window
# reports 'not-contained' or 'absent' rather than a number, and THAT IS THE
# INTENDED READING -- the geometry these six members describe is not present
# on those arms, and a number obtained by reaching backward for it would be
# a different feature wearing the same name.
#
# A fixed save cadence over the window is asserted, not assumed.

# Window forward extent [s]. Sized so an arm whose slope maximum sits late in
# the fast phase keeps several samples of margin inside the right edge; the
# slow crawl's few-hundred-A/ms never competes with the ramp maximum.
RAMP_WINDOW_S = 1.2e-3

# CLASSIFICATION CONSTANTS -- these decide whether a feature is PRESENT and
# well-separated. They are not estimator parameters: no reported VALUE is a
# function of them, so a reader may not tune them to move a measured number.
# GRID_PHASE_FRACTION is the one exception to that scope, and only on the
# BAND side: it sets the reported t_dImax conditioning band and the coarse
# crossing bands, as its own paragraph below states. Each is quoted with the
# margin it was ruled on.
#
#   EDGE_SLOPE_FRACTION -- the slope at each window edge must fall below this
#       fraction of the maximum, so the maximum is a resolved interior
#       feature rather than the shoulder of something outside the window.
#       The bare interior test is NOT sufficient: an arm whose steepest slope
#       sits at the trigger passes "interior" while measuring the wrong
#       feature entirely.
#   OVERSHOOT_MIN_FRACTION -- fractional depth (I_pk - I_min)/I_pk below which
#       the trace is called monotone-saturating and no overshoot is reported.
#       Measured margin is three orders either side: an ES1-class arm shows
#       ~11 %, a saturating arm ~0.03 %.
#   GRID_PHASE_FRACTION -- the measured scale of the grid-phase artifact in
#       the centered-difference slope, as a fraction of the slope maximum.
#       It sets the t_dImax conditioning band below.
RAMP_EDGE_SLOPE_FRACTION = 0.9
RAMP_OVERSHOOT_MIN_FRACTION = 0.01
RAMP_GRID_PHASE_FRACTION = 0.002

# VALUE-SELECTING CONSTANTS -- unlike the block above, these DO move reported
# values, which is why they are disclosed here rather than left inline. Each
# is a forward reach [s] from an already-located landmark, and it selects
# WHICH discrete sample becomes an extremum: widen one and the search can be
# handed a later, larger sample, moving t_ovs/I_ovs or t_dip/I_dip. They are
# not free, though -- an extremum that lands on its own reach edge is reported
# "not-contained" rather than as a number, so a badly sized reach fails loudly
# instead of quietly returning the edge.
#
#   OVERSHOOT_SEARCH_REACH_S -- forward reach from t_dImax inside which the
#       discrete current maximum is taken. It must cover the ringing peak and
#       stop short of the slow plateau climb, which would otherwise supply a
#       later and larger maximum. Measured margin on the ES1-class arms: the
#       peak sits ~70-75 us after t_dImax, about a quarter of the reach.
#   RINGBACK_SEARCH_REACH_S -- forward reach from that peak inside which the
#       discrete minimum is taken. Longer than the peak reach because the dip
#       bottom is quartic-flat and sits well after the peak; still short
#       enough not to re-enter the plateau recovery. Measured margin on the
#       ES1-class arms: the dip sits ~105 us after the peak, again about a
#       quarter of the reach.
RAMP_OVERSHOOT_SEARCH_REACH_S = 0.30e-3
RAMP_RINGBACK_SEARCH_REACH_S = 0.40e-3

# Declared equivalence bands. Measured spreads, not asserted tolerances: a
# member whose pre->post move sits inside its band is reported "unmoved".
RAMP_BAND_T_FLOOR_S = 2.0e-6      # t_dImax floor, and the flat t_ovs band
RAMP_BAND_T_DIP_S = 4.0e-6        # dip bottom is quartic-flat, worst-conditioned
RAMP_BAND_S_RAMP_FRAC = 0.004     # S_ramp, +-0.4 % on every arm
RAMP_BAND_I_FRAC = 0.002          # I_ovs and I_dip, +-0.2 %

# Paired-reference crossing family. Fractions of a plateau reference current
# taken from the PRE-BATCH PARTNER OF THE SAME RUNG, with both members of a
# pair using the same reference -- an unpaired reference lets an amplitude
# move masquerade as a timing move.
RAMP_CROSSING_FRACTIONS = (0.50, 0.60, 0.90)
RAMP_CROSSING_SLOPE_FRACTION = 0.3   # classification: fast-crossing vs coarse
RAMP_CROSSING_BAND_S = 1.0e-6


def _ramp_member(value, band, status, reason=""):
    """One ramp member: a value, its equivalence band, and its status.

    ``status`` is one of ``ok`` (resolved inside the declared band),
    ``coarse`` (resolved, but the band had to be widened by local
    conditioning), ``absent`` (the feature does not exist in this trace, or
    its reference does not exist), and ``not-contained`` (the window or its
    preconditions could not isolate the feature).

    A member NEVER raises. Row (ii) reporting "feature absent" is a reading,
    and it must not be able to take the rest of the fingerprint set down with
    it -- the other rows are independent measurements of the same artifact.
    """
    return {"value": value, "band": band, "status": status, "reason": reason}


def _parabola_vertex(x, y):
    """Return the vertex of the exact quadratic through three points.

    General (non-uniform) 3-point form via divided differences, so the caller
    never has to assume the samples are evenly spaced. Returns ``(None, None)``
    on degenerate curvature -- three collinear points have no vertex, and
    inventing one would report a grid coordinate as a measurement.
    """
    x0, x1, x2 = (float(v) for v in x)
    y0, y1, y2 = (float(v) for v in y)
    d01 = (y1 - y0) / (x1 - x0)
    d12 = (y2 - y1) / (x2 - x1)
    curvature = (d12 - d01) / (x2 - x0)
    if not np.isfinite(curvature) or curvature == 0.0:
        return None, None
    xv = 0.5 * (x0 + x1) - d01 / (2.0 * curvature)
    yv = y0 + d01 * (xv - x0) + curvature * (xv - x0) * (xv - x1)
    return xv, yv


def _ramp_members(t, I, t_trig):
    """Return the six fast-phase ramp members keyed by name.

    ``t`` is ABSOLUTE saved time [s] over the whole trajectory and ``I`` the
    matching ``source_I_tot`` [A]; ``t_trig`` is row (i)'s interpolated
    breakdown-trigger time [s], which is both the causal left edge of the
    search window and the origin every reported time is offset from. The
    save-quantized phase-label edge is deliberately never used here: it hops
    a whole bin and would inject that hop into otherwise sub-bin readings.
    """
    names = ("t_dImax", "S_ramp", "t_ovs", "I_ovs", "t_dip", "I_dip")

    def unresolved(status, reason, keep=None):
        out = {n: _ramp_member(np.nan, np.nan, status, reason) for n in names}
        out.update(keep or {})
        return out

    if not np.isfinite(t_trig):
        return unresolved("absent", "no interpolated breakdown trigger on this run")

    window = np.flatnonzero((t >= t_trig) & (t <= t_trig + RAMP_WINDOW_S))
    if window.size < 7:
        return unresolved(
            "not-contained", f"only {window.size} saves inside the window"
        )
    spacing = np.diff(t[window])
    if not np.all(np.isfinite(spacing)) or spacing.min() <= 0.0:
        return unresolved("not-contained", "save times are not increasing")
    if spacing.max() / spacing.min() - 1.0 > 0.01:
        return unresolved(
            "not-contained",
            f"save cadence is not fixed over the window (min "
            f"{spacing.min() * 1e6:.4g} us, max {spacing.max() * 1e6:.4g} us)",
        )
    h = float(np.mean(spacing))

    # Centered-difference slope, restricted to samples that HAVE both
    # neighbours; the window's own end samples borrow from outside it, which
    # is what makes the edge-slope test below meaningful.
    lo, hi = int(window[0]), int(window[-1])
    idx = np.arange(max(lo, 1), min(hi, t.size - 2) + 1)
    if idx.size < 5:
        return unresolved("not-contained", "too few interior samples for a slope")
    slope = (I[idx + 1] - I[idx - 1]) / (t[idx + 1] - t[idx - 1])
    if not np.all(np.isfinite(slope)):
        return unresolved("not-contained", "non-finite current inside the window")

    k = int(np.argmax(slope))
    s_max = float(slope[k])
    if k < 2 or k > slope.size - 3:
        return unresolved(
            "not-contained",
            f"ramp slope maximum is within 2 samples of a window edge "
            f"(index {k} of {slope.size})",
        )
    edge = RAMP_EDGE_SLOPE_FRACTION * s_max
    if not (slope[0] < edge and slope[-1] < edge):
        return unresolved(
            "not-contained",
            f"window edge slope is not below {RAMP_EDGE_SLOPE_FRACTION:g}*s_max "
            f"(edges {slope[0] / 1e3:.4g}/{slope[-1] / 1e3:.4g} A/ms vs s_max "
            f"{s_max / 1e3:.4g} A/ms) -- the maximum is not an isolated feature",
        )

    # t_dImax conditioning band. A perturbation ds of the three slope samples
    # moves the 3-point vertex by roughly h*ds/|d|, where d is the discrete
    # second difference; ds is taken at the measured grid-phase scale
    # GRID_PHASE_FRACTION*s_max. A sharply peaked slope floors at the declared
    # band; a flat-topped one widens honestly to the width of its own plateau,
    # which is the case the interior and edge tests cannot see.
    d = float(slope[k - 1] - 2.0 * slope[k] + slope[k + 1])
    if not np.isfinite(d) or d == 0.0:
        return unresolved(
            "not-contained", "degenerate slope curvature at the ramp maximum"
        )
    band_t_dImax = max(
        RAMP_BAND_T_FLOOR_S, h * (RAMP_GRID_PHASE_FRACTION * s_max) / abs(d)
    )
    t_dImax, S_ramp = _parabola_vertex(
        t[idx[k - 1:k + 2]], slope[k - 1:k + 2]
    )
    if t_dImax is None:
        return unresolved(
            "not-contained", "degenerate slope curvature at the ramp maximum"
        )
    resolved = {
        "t_dImax": _ramp_member(
            t_dImax, band_t_dImax,
            "ok" if band_t_dImax <= RAMP_BAND_T_FLOOR_S else "coarse",
            "" if band_t_dImax <= RAMP_BAND_T_FLOOR_S
            else f"flat slope maximum (|d|={abs(d) / 1e3:.4g} A/ms)",
        ),
        # The flatness that ruins the TIME leaves the VALUE well determined,
        # so S_ramp keeps its declared band on every arm.
        "S_ramp": _ramp_member(
            S_ramp, RAMP_BAND_S_RAMP_FRAC * abs(S_ramp), "ok"
        ),
    }

    # Overshoot EXISTENCE, on the discrete samples, before any vertex fit:
    # the peak in a fixed reach after the ramp maximum, then the minimum in a
    # fixed reach after that peak.
    peak_win = np.flatnonzero(
        (t >= t_dImax) & (t <= t_dImax + RAMP_OVERSHOOT_SEARCH_REACH_S)
    )
    if peak_win.size < 3:
        return unresolved("not-contained", "overshoot search window too short", resolved)
    jp = int(np.argmax(I[peak_win]))
    g = int(peak_win[jp])
    dip_win = np.flatnonzero(
        (t >= t[g]) & (t <= t[g] + RAMP_RINGBACK_SEARCH_REACH_S)
    )
    if dip_win.size < 3:
        return unresolved("not-contained", "ringback search window too short", resolved)
    jm = int(np.argmin(I[dip_win]))
    m = int(dip_win[jm])
    I_pk, I_min = float(I[g]), float(I[m])
    depth = (I_pk - I_min) / I_pk if I_pk != 0.0 else 0.0
    if not np.isfinite(depth) or depth < RAMP_OVERSHOOT_MIN_FRACTION:
        reason = (
            f"no overshoot: (I_pk-I_min)/I_pk = {100.0 * depth:.4g} % < "
            f"{100.0 * RAMP_OVERSHOOT_MIN_FRACTION:g} % -- the trace saturates "
            f"monotonically here (I_pk {I_pk:.4g} A, I_min {I_min:.4g} A)"
        )
        return unresolved("absent", reason, resolved)

    if jp == 0 or jp == peak_win.size - 1 or g < 1 or g > t.size - 2:
        return unresolved(
            "not-contained", "overshoot maximum is on its window edge", resolved
        )
    if jm == 0 or jm == dip_win.size - 1 or m < 1 or m > t.size - 2:
        return unresolved(
            "not-contained", "ringback minimum is on its window edge", resolved
        )
    t_ovs, I_ovs = _parabola_vertex(t[g - 1:g + 2], I[g - 1:g + 2])
    t_dip, I_dip = _parabola_vertex(t[m - 1:m + 2], I[m - 1:m + 2])
    if t_ovs is None or t_dip is None:
        return unresolved(
            "not-contained", "degenerate curvature at an extremum", resolved
        )
    resolved.update({
        "t_ovs": _ramp_member(t_ovs, RAMP_BAND_T_FLOOR_S, "ok"),
        "I_ovs": _ramp_member(I_ovs, RAMP_BAND_I_FRAC * abs(I_ovs), "ok"),
        "t_dip": _ramp_member(t_dip, RAMP_BAND_T_DIP_S, "ok"),
        "I_dip": _ramp_member(I_dip, RAMP_BAND_I_FRAC * abs(I_dip), "ok"),
    })
    return resolved


def _plateau_reference_A(path, cache, loaded=None):
    """Return the pre-batch partner's plateau mean current [A], or None.

    Read with h5py rather than through ``load_result_hdf5``: only three
    datasets are needed, and a production artifact is multi-GB -- holding a
    second full result alongside the one being reported would put two of them
    in memory at once for a single scalar.
    """
    if path in cache:
        return cache[path]
    try:
        if loaded is not None:
            time_s = np.asarray(loaded.time, float)
            phases = np.asarray(loaded.phase, dtype=str)
            current = np.asarray(
                loaded.cathode_diagnostics["source_I_tot"], float
            )
        else:
            import h5py

            with h5py.File(path, "r") as handle:
                time_s = np.asarray(handle["time"][...], float)
                phases = np.asarray(handle["phase"][...], dtype=str)
                current = np.asarray(
                    handle["cathode_diagnostics"]["source_I_tot"][...], float
                )
        hits = np.flatnonzero(phases == "main_discharge")
        if not hits.size:
            cache[path] = None
            return None
        t_ms = (time_s - float(time_s[hits[0]])) * 1.0e3
        window = (t_ms >= 15.0) & (t_ms <= 19.5)
        values = current[window][np.isfinite(current[window])]
        cache[path] = float(np.mean(values)) if values.size else None
    except (OSError, KeyError, ValueError):
        cache[path] = None
    return cache[path]


def _partner_ramp_members(path, cache, loaded=None):
    """Return the partner's ``(members, t_trig)`` for the pair block, or None.

    Read with h5py rather than through ``load_result_hdf5``, on the same
    grounds as ``_plateau_reference_A`` above: two datasets and one root
    attribute out of a multi-GB artifact, never a second full result held in
    memory beside the one being reported.

    The members come from ``_ramp_members`` -- the SAME estimator the run's
    own row (ii) used, on the same window, constants and bands. There is
    exactly one implementation of these six quantities in this file, so a
    pre->post delta is a difference between two RUNS and can never be a
    difference between two estimators.

    ``None`` means the partner could not be read at all; the caller says so
    rather than falling silent, because a silent pair block is
    indistinguishable from an unmoved one.
    """
    key = ("ramp", path)
    if key in cache:
        return cache[key]
    try:
        if loaded is not None:
            time_s = np.asarray(loaded.time, float)
            current = np.asarray(
                loaded.cathode_diagnostics["source_I_tot"], float
            )
            t_trig = float(getattr(loaded, "t_breakdown_trigger", np.nan))
        else:
            import h5py

            with h5py.File(path, "r") as handle:
                time_s = np.asarray(handle["time"][...], float)
                current = np.asarray(
                    handle["cathode_diagnostics"]["source_I_tot"][...], float
                )
                t_trig = float(
                    handle.attrs.get("t_breakdown_trigger", np.nan)
                )
        cache[key] = (_ramp_members(time_s, current, t_trig), t_trig)
    except (OSError, KeyError, ValueError):
        cache[key] = None
    return cache[key]


def _ramp_crossings(t, I, t_trig, I_ref, S_ramp):
    """Return the paired-reference crossing family as (fraction, member) rows.

    Each row is the FIRST upward crossing at or after ``t_trig`` of
    ``fraction * I_ref``, linearly interpolated on the bracketing interval,
    carrying the local secant slope the interpolation itself used.
    """
    rows = []
    post = np.flatnonzero(t >= t_trig)
    tp, Ip = t[post], I[post]
    fast = (
        RAMP_CROSSING_SLOPE_FRACTION * S_ramp
        if S_ramp is not None and np.isfinite(S_ramp) else None
    )
    for fraction in RAMP_CROSSING_FRACTIONS:
        level = fraction * I_ref
        hits = np.flatnonzero((Ip[:-1] < level) & (Ip[1:] >= level))
        if not hits.size:
            rows.append((fraction, level, np.nan, _ramp_member(
                np.nan, np.nan, "absent",
                "the current never reaches this level after the trigger")))
            continue
        i = int(hits[0])
        span_t = float(tp[i + 1] - tp[i])
        span_I = float(Ip[i + 1] - Ip[i])
        s_loc = span_I / span_t
        t_cross = float(tp[i]) + (level - float(Ip[i])) * span_t / span_I
        if fast is not None and s_loc >= fast:
            rows.append((fraction, level, s_loc, _ramp_member(
                t_cross, RAMP_CROSSING_BAND_S, "ok")))
        else:
            band = (RAMP_GRID_PHASE_FRACTION * I_ref) / s_loc
            why = (
                f"local slope {s_loc / 1e3:.4g} A/ms is below "
                f"{RAMP_CROSSING_SLOPE_FRACTION:g}*S_ramp"
                if fast is not None else "no S_ramp to classify against"
            )
            rows.append((fraction, level, s_loc, _ramp_member(
                t_cross, band, "coarse", why)))
    return rows


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


def report(path, partner=None, reference_cache=None):
    """Print the fingerprint set for one saved run.

    ``partner`` is the pre-batch artifact of the SAME rung at the SAME stance,
    supplying the crossing family's reference current; ``None`` (the default)
    reports that family absent rather than falling back to anything else.
    ``reference_cache`` is shared across a multi-artifact invocation so a
    partner is read once even when both members of a pair are reported.
    """
    reference_cache = {} if reference_cache is None else reference_cache
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

    # (ii) Fast-phase ramp structure, all times offset from t_trig.
    t_trig = float(getattr(result, "t_breakdown_trigger", np.nan))
    members = _ramp_members(tsec_all, I, t_trig)

    # Absolute times are printed to 1e-8 ms = 0.01 ns. That is far finer
    # than anything this instrument resolves, and deliberately so: the
    # absolute column is the one an arm-to-arm delta is taken on, and a
    # coarser column silently caps the precision of every delta derived from
    # it. At the previous 1e-4 ms the printed absolute difference and the
    # printed t_trig-relative difference disagreed by ~7 % on a banked
    # sub-microsecond reading -- pure print artifact, no physics in it.
    def _t_row(name, member):
        if member["status"] in ("ok", "coarse"):
            value = member["value"]
            body = (f"{value * 1e3:.8f} ms absolute "
                    f"({(value - t_trig) * 1e6:+.2f} us vs t_trig) | band "
                    f"+-{member['band'] * 1e6:.2f} us")
        else:
            body = "<not resolved> | band n/a"
        note = f" -- {member['reason']}" if member["reason"] else ""
        print(f"(ii)   {name:<8} {body} [{member['status']}]{note}")

    def _v_row(name, member, unit, scale, fmt):
        if member["status"] in ("ok", "coarse"):
            body = (f"{member['value'] / scale:{fmt}} {unit} | band "
                    f"+-{member['band'] / scale:{fmt}} {unit}")
        else:
            body = "<not resolved> | band n/a"
        note = f" -- {member['reason']}" if member["reason"] else ""
        print(f"(ii)   {name:<8} {body} [{member['status']}]{note}")

    trig_text = (
        "<absent>" if not np.isfinite(t_trig) else f"{t_trig * 1e3:.8f} ms"
    )
    print(f"(ii) fast-phase ramp structure [t_trig {trig_text}, window "
          f"t_trig -> t_trig+{RAMP_WINDOW_S * 1e3:g} ms, geometry only -- no "
          f"mechanism claimed]")
    print("(ii)   TIME BASIS -- every time below is printed on TWO bases: "
          "'ms absolute' is the saved clock, and the parenthesised "
          "'us vs t_trig' is this run's own t_trig subtracted off. An "
          "arm-to-arm delta differs between the two bases by exactly the "
          "move in t_trig, so a quoted delta MUST name its basis. The "
          "absolute column is printed fine enough to re-derive either.")
    _t_row("t_dImax", members["t_dImax"])
    _v_row("S_ramp", members["S_ramp"], "A/ms", 1.0e3, ".4f")
    _t_row("t_ovs", members["t_ovs"])
    _v_row("I_ovs", members["I_ovs"], "A", 1.0, ".4f")
    _t_row("t_dip", members["t_dip"])
    _v_row("I_dip", members["I_dip"], "A", 1.0, ".4f")

    # The paired-reference crossing family. Both members of a pre/post pair
    # read against the SAME reference -- the pre-batch partner of that rung --
    # so an amplitude move cannot masquerade as a timing move. A rung with no
    # same-stance partner reports the family absent and NAMES what is missing:
    # substituting a cross-stance partner would reintroduce exactly the
    # confound the paired form exists to remove.
    I_ref = None if partner is None else _plateau_reference_A(
        partner, reference_cache, loaded=result if partner == path else None
    )
    if I_ref is None:
        reason = (
            "no pre-batch partner supplied for this rung" if partner is None
            else f"partner {partner} carries no usable 15-19.5 ms plateau"
        )
        print(f"(ii)   crossings <not resolved> [absent] -- {reason}; "
              f"S_ramp above is the partner-free timing-adjacent member")
    else:
        S_ramp = (
            members["S_ramp"]["value"]
            if members["S_ramp"]["status"] in ("ok", "coarse") else None
        )
        print(f"(ii)   crossings vs I_ref {I_ref:.4f} A "
              f"(pre-batch partner {partner}):")
        for fraction, level, s_loc, member in _ramp_crossings(
            tsec_all, I, t_trig, I_ref, S_ramp
        ):
            if member["status"] == "absent":
                body = "<not resolved> | band n/a"
            else:
                body = (f"{member['value'] * 1e3:.8f} ms absolute "
                        f"({(member['value'] - t_trig) * 1e6:+.2f} us vs t_trig)"
                        f" | s_loc {s_loc / 1e3:.4f} A/ms | band "
                        f"+-{member['band'] * 1e6:.2f} us")
            note = f" -- {member['reason']}" if member["reason"] else ""
            print(f"(ii)     {fraction:.2f}*I_ref={level:.4f} A  {body} "
                  f"[{member['status']}]{note}")

    # --- (ii-pair) The row-(ii) PAIR DELTAS. --------------------------------
    #
    # ADDITIVE, and gated on a partner: an invocation without ``--pair``
    # prints not one byte of this block, so every row above stays exactly as
    # byte-frozen as it was.
    #
    # The six member values are the ones row (ii) already computed for this
    # run; the partner's come from the same estimator via
    # ``_partner_ramp_members``. Nothing is re-derived here -- this block only
    # differences and formats.
    #
    # WHAT THE VERDICT WORD MEANS, and what it does not. It compares the
    # pre->post difference against the instrument's own declared conditioning
    # band for that member ON THIS RUN. It is a statement about what this
    # instrument can RESOLVE, and it is not a significance test, not a
    # tolerance check and not a gate: a resolved difference may still be
    # physically negligible, and an unresolved one may still be real but
    # below the band. The gate vocabulary is deliberately absent from this
    # block for that reason -- read these rows as "the instrument can/cannot
    # tell these two runs apart on this member", and take the physics
    # question elsewhere.
    if partner is not None:
        pair = _partner_ramp_members(
            partner, reference_cache,
            loaded=result if partner == path else None,
        )
        print(f"(ii-pair) row (ii) pre->post deltas | pre {partner} -> "
              f"post {path}")
        print("(ii-pair)   VERDICT WORDS -- 'moved' = the pre->post "
              "difference is resolved BEYOND this member's declared "
              "conditioning band; 'unmoved' = it falls WITHIN that band. A "
              "resolution statement about the instrument, not a "
              "physics-significance test.")
        if pair is None:
            print(f"(ii-pair)   <no pair reading> -- partner {partner} could "
                  f"not be read for its row (ii) members, so there is no pre "
                  f"side to difference against")
        else:
            pre_members, pre_trig = pair
            print("(ii-pair)   TIME BASIS -- every time below is on the 'vs "
                  "t_trig' basis with EACH RUN'S OWN t_trig subtracted, so a "
                  "time delta here carries no part of the move in t_trig "
                  "itself. The band shown is the POST run's declared band.")

            def _pair_row(name, unit, scale, fmt, on_trig_basis):
                post_m = members[name]
                pre_m = pre_members[name]
                resolved = ("ok", "coarse")
                if (post_m["status"] not in resolved
                        or pre_m["status"] not in resolved):
                    print(f"(ii-pair)   {name:<8} no pair reading "
                          f"({pre_m['status']}/{post_m['status']})")
                    return
                pre_v = float(pre_m["value"]) - (
                    pre_trig if on_trig_basis else 0.0
                )
                post_v = float(post_m["value"]) - (
                    t_trig if on_trig_basis else 0.0
                )
                delta = post_v - pre_v
                band = float(post_m["band"])
                verdict = "unmoved" if abs(delta) <= band else "moved"
                sign = "+" if on_trig_basis else ""
                print(f"(ii-pair)   {name:<8} pre {pre_v / scale:{sign}{fmt}} "
                      f"{unit} | post {post_v / scale:{sign}{fmt}} {unit} | "
                      f"delta {delta / scale:+{fmt}} {unit} | band "
                      f"+-{band / scale:{fmt}} {unit} [{verdict}]")

            _pair_row("t_dImax", "us", 1.0e-6, ".2f", True)
            _pair_row("S_ramp", "A/ms", 1.0e3, ".4f", False)
            _pair_row("t_ovs", "us", 1.0e-6, ".2f", True)
            _pair_row("I_ovs", "A", 1.0, ".4f", False)
            _pair_row("t_dip", "us", 1.0e-6, ".2f", True)
            _pair_row("I_dip", "A", 1.0, ".4f", False)

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
    # ``--pair RUN=PARTNER`` (repeatable) names the pre-batch artifact of the
    # same rung and stance for RUN. It is passed explicitly rather than
    # inferred from filenames: the pairing is a campaign fact about which two
    # runs share a stance, which a public tool cannot read off a path.
    pairs = {}
    paths = []
    pending = False
    for arg in argv:
        if pending:
            run, _, partner = arg.partition("=")
            pairs[run] = partner or None
            pending = False
        elif arg == "--pair":
            pending = True
        elif arg.startswith("--pair="):
            run, _, partner = arg[len("--pair="):].partition("=")
            pairs[run] = partner or None
        else:
            paths.append(arg)
    if pending:
        raise SystemExit("--pair needs an argument of the form RUN=PARTNER")
    reference_cache = {}
    for path in paths:
        report(path, pairs.get(path), reference_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
