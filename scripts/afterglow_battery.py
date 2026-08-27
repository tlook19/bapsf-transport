"""Read-only afterglow decay battery for a saved sim1d run.

Three zero-cost reads over an artifact that already exists, all of them
regenerated with the stage (iii) scorer's OWN fitter and its OWN port/cell
mapping (``compare_sim1d_es1._efold_time_ms`` and ``compare_decay``, imported
here, never re-implemented) so that every number in the (i) ``tau_Isat``
column is the number the scorer prints:

(i)   TAU SPLIT.  The scorer's Isat proxy is ``n * sqrt(Te)`` at the port
      cell, so its log-linear slope decomposes exactly:

          1/tau_Isat  =  1/tau_n  +  1/(2 tau_Te)

      Both channel times are fitted separately, over the same window and the
      same samples, and the identity is printed with its residual.  Because
      OLS is linear in the fitted signal the residual is a MASK check, not a
      physics check: it is ~0 to rounding whenever the three fits see the
      same samples, and non-zero only if one of them dropped a sample the
      others kept.  On the measured side only ``tau_Isat`` exists; the
      overlay's Te trace stops before the fit window opens and the overlay
      carries no measured n(t) inside it at all, so both are reported absent
      with the reason.

(ii)  SYMMETRIC BURST-TRIM.  The first 0.2 ms is removed from BOTH ends of
      the scorer window (20.2-21.3 ms main-discharge clock), which excludes
      the beam-deposition burst at shutoff without moving the window's
      centre, and the trimmed ratios are printed against the full-window
      ones.

(iii) LATE-WINDOW REFIT.  21.0-22.5 ms main-discharge clock, model and
      measured, with its own ratio.

CLOCKS.  Every window is printed on BOTH clocks.  The scorer's convention is
the authority: its windows are on the MAIN-DISCHARGE clock, whose origin is
the first ``main_discharge`` sample (``_main_discharge_origin``), and the run
clock is that plus the origin offset.  A window quoted on the wrong clock
admits full-drive frames into an afterglow fit.

DETERMINACY.  Per the 2026-08-23 convention, a fitted decay is DETERMINED
only when its window log-slope is negative AND more than 3 sigma_OLS from
zero; sigma_OLS is the ordinary least-squares standard error of the slope
over the same samples the fitter used.  An undetermined fit prints ``n.d.``
and never enters a mean.

Usage (from ``<checkout>/cablp``, with ``PYTHONPATH=<checkout>/cablp``)::

    python scripts/afterglow_battery.py scripts/ph_es1.h5 --es 1
    python scripts/afterglow_battery.py RUN.h5 --es 3 --out report.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import load_result_hdf5

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from compare_sim1d_es1 import (  # noqa: E402
    DECAY_WINDOW_MS,
    OVERLAY,
    _decay_observability,
    _efold_time_ms,
    _main_discharge_origin,
    compare_decay,
)

# The three windows, on the MAIN-DISCHARGE clock [ms].
#
# FULL is the scorer's own DECAY_WINDOW_MS, imported rather than restated so
# a re-anchor of the scored window moves this instrument with it.  TRIM_MS is
# the symmetric burst-trim: the beam-deposition burst at shutoff lasts 0.2 ms,
# and removing that much from BOTH ends excludes it without moving the
# window's centre (an asymmetric trim would change which part of the decay is
# weighted, and the trimmed tau would not be comparable to the full-window
# one).  LATE_WINDOW_MS starts a full window-span after shutoff.
TRIM_MS = 0.2
LATE_WINDOW_MS = (21.0, 22.5)

# Determinacy: |slope| must exceed this many OLS standard errors.
DETERMINACY_SIGMA = 3.0


def _slope_stats(t_ms, y, floor=0.0):
    """Return ``(slope, sigma_slope, n_good)`` for the fitter's own samples.

    The good-sample mask is ``_efold_time_ms``'s mask, reproduced exactly so
    the standard error describes the fit whose tau is reported.  ``sigma`` is
    the textbook OLS standard error of the slope,
    ``s_resid / sqrt(sum((t - tbar)^2))`` with ``s_resid^2 = SSR / (n - 2)``.
    Returns NaNs when the fitter itself would refuse the fit (fewer than 8
    good samples) or when the standard error is undefined (fewer than 3).
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(t_ms) & np.isfinite(y) & (y > max(floor, 0.0))
    n = int(np.count_nonzero(good))
    if n < 8:
        return np.nan, np.nan, n
    t_g = t_ms[good]
    ln_y = np.log(y[good])
    slope, intercept = np.polyfit(t_g, ln_y, 1)
    resid = ln_y - (slope * t_g + intercept)
    ss_t = float(np.sum((t_g - t_g.mean()) ** 2))
    if n <= 2 or ss_t <= 0.0:
        return float(slope), np.nan, n
    s_resid = float(np.sqrt(np.sum(resid**2) / (n - 2)))
    return float(slope), s_resid / np.sqrt(ss_t), n


def _fit(t_ms, y, floor=0.0):
    """Return one fit record: the scorer's tau plus its determinacy.

    ``tau`` always comes from ``_efold_time_ms`` -- this function never
    computes a decay time of its own -- and the slope statistics only decide
    whether that tau is DETERMINED.
    """
    tau = _efold_time_ms(t_ms, y, floor=floor)
    slope, sigma, n = _slope_stats(t_ms, y, floor=floor)
    determined = bool(
        np.isfinite(slope)
        and np.isfinite(sigma)
        and sigma > 0.0
        and slope < 0.0
        and abs(slope) > DETERMINACY_SIGMA * sigma
    )
    n_sigma = abs(slope) / sigma if np.isfinite(sigma) and sigma > 0.0 else np.nan
    return {
        "tau_ms": tau,
        "slope": slope,
        "sigma_slope": sigma,
        "n_sigma": n_sigma,
        "n_good": n,
        "determined": determined,
    }


def _tau_text(fit, width=7):
    """Render a fitted tau, or ``n.d.`` when the fit is undetermined."""
    if not fit["determined"] or not np.isfinite(fit["tau_ms"]):
        return f"{'n.d.':>{width}}"
    return f"{fit['tau_ms']:{width}.2f}"


def _sigma_text(fit, width=6):
    if not np.isfinite(fit["n_sigma"]):
        return f"{'--':>{width}}"
    return f"{fit['n_sigma']:{width}.1f}"


def _ratio(a_fit, b_fit):
    """Ratio of two fitted taus, NaN unless BOTH fits are determined."""
    if not (a_fit["determined"] and b_fit["determined"]):
        return np.nan
    if not (np.isfinite(a_fit["tau_ms"]) and np.isfinite(b_fit["tau_ms"])):
        return np.nan
    return a_fit["tau_ms"] / b_fit["tau_ms"]


def _num(value, width=7, digits=2):
    if not np.isfinite(value):
        return f"{'n.d.':>{width}}"
    return f"{value:{width}.{digits}f}"


def _mean_determined(values):
    """Mean over the finite entries only, with the count that fed it."""
    kept = [v for v in values if np.isfinite(v)]
    if not kept:
        return np.nan, 0
    return float(np.mean(kept)), len(kept)


def _exp_noise_floor(isat_row, t_exp):
    """The scorer's measured noise floor: 5x the robust sigma of the last 5 ms.

    Window-independent by construction (it reads the far tail of the trace,
    never the fit window), which is what makes the same floor correct for all
    three windows below.
    """
    tail = isat_row[t_exp >= t_exp.max() - 5.0]
    return 5.0 * 1.4826 * float(np.nanmedian(np.abs(tail - np.nanmedian(tail))))


def _port_cells(result, overlay):
    """Return ``[(port, z_cm, iz, z_cell_cm), ...]`` in the scorer's order.

    The mapping is the scorer's: nearest model cell centre to the port's
    surveyed z.  It is recomputed here only because ``compare_decay`` builds
    it inline; the assertion in ``battery`` proves the two agree by checking
    that a tau fitted through THIS mapping reproduces the scorer's tau.
    """
    z_model = np.asarray(result.z_cm, dtype=float)
    ports = np.asarray(overlay["isat_decay_port"])
    z_ports = {
        int(p): float(z) for p, z in zip(np.asarray(overlay["port"]), overlay["z_cm"])
    }
    out = []
    for p in range(ports.size):
        z = z_ports.get(int(ports[p]))
        if z is None:
            continue
        iz = int(np.argmin(np.abs(z_model - z)))
        out.append((int(ports[p]), z, iz, float(z_model[iz])))
    return out


def _window_fits(result, overlay, origin_s, window_ms):
    """Fit every channel this battery reports, for one window.

    Returns ``(rows, scorer_rows)``: ``rows`` carries the model Isat/n/Te fits
    and the measured Isat fit per port, ``scorer_rows`` is ``compare_decay``'s
    own output for the same window (the authority the model and measured Isat
    taus are asserted against).
    """
    t0, t1 = float(window_ms[0]), float(window_ms[1])
    scorer_rows, _ = compare_decay(result, overlay, window_ms=(t0, t1))

    t_model_ms = (np.asarray(result.time, dtype=float) - origin_s) * 1.0e3
    n_model = np.asarray(result.n, dtype=float)
    te_model = np.asarray(result.Te, dtype=float)
    model_window = (t_model_ms >= t0) & (t_model_ms <= t1)
    t_win = t_model_ms[model_window]

    t_exp = np.asarray(overlay["isat_decay_time_ms"], dtype=float)
    isat = np.asarray(overlay["isat_decay_mean_a"], dtype=float)
    exp_window = (t_exp >= t0) & (t_exp <= t1)

    rows = []
    for idx, (port, z, iz, z_cell) in enumerate(_port_cells(result, overlay)):
        n_t = n_model[model_window, iz]
        te_t = te_model[model_window, iz]
        proxy = n_t * np.sqrt(np.maximum(te_t, 0.0))
        noise = _exp_noise_floor(isat[idx], t_exp)
        rows.append(
            {
                "port": port,
                "z": z,
                "iz": iz,
                "z_cell": z_cell,
                "model_isat": _fit(t_win, proxy),
                "model_n": _fit(t_win, n_t),
                "model_te": _fit(t_win, te_t),
                "exp_isat": _fit(t_exp[exp_window], isat[idx, exp_window], noise),
                "exp_n_good": int(np.count_nonzero(exp_window)),
            }
        )
    return rows, scorer_rows


def _assert_matches_scorer(rows, scorer_rows, window_ms, out):
    """Assert this battery's Isat taus ARE the scorer's, to the printed digit.

    The comparison is on the printed 2-decimal rendering (what the campaign
    quotes) and, when both are finite, on the raw float as well: an
    instrument that agreed only after rounding would be a different fit.
    """
    label = f"{window_ms[0]:.1f}-{window_ms[1]:.1f} ms"
    for row, ref in zip(rows, scorer_rows):
        assert row["port"] == ref["port"], (
            f"port order diverged from compare_decay at {label}: "
            f"{row['port']} vs {ref['port']}"
        )
        for key, ref_key, side in (
            ("model_isat", "tau_model_ms", "model"),
            ("exp_isat", "tau_exp_ms", "measured"),
        ):
            mine = row[key]["tau_ms"]
            theirs = float(ref[ref_key])
            same_nan = (not np.isfinite(mine)) and (not np.isfinite(theirs))
            assert same_nan or mine == theirs, (
                f"{side} tau_Isat at port {row['port']}, window {label}: "
                f"battery {mine!r} != compare_decay {theirs!r} -- the port/cell "
                "mapping or the fit window diverged from the scorer's"
            )
    out(
        f"  ASSERTED: every tau_Isat above is compare_decay's own value for "
        f"the {label} window,"
    )
    out(
        "  model and measured, at both ports' full float precision "
        "(not merely to the printed digit)."
    )


def _clock_line(name, window_ms, origin_ms):
    t0, t1 = float(window_ms[0]), float(window_ms[1])
    return (
        f"  {name:<11} main-discharge {t0:6.3f} - {t1:6.3f} ms"
        f"  |  run {t0 + origin_ms:6.3f} - {t1 + origin_ms:6.3f} ms"
        f"  |  span {t1 - t0:.3f} ms"
    )


def _report_split(rows, window_ms, out):
    t0, t1 = window_ms
    out("")
    out(f"--- (i) tau split, full window {t0:.1f}-{t1:.1f} ms (main-discharge) ---")
    out("  MODEL, at the port cell.  tau [ms]; 'n.sig' is |slope|/sigma_OLS,")
    out(f"  and a fit is DETERMINED at n.sig > {DETERMINACY_SIGMA:.0f} "
        "(undetermined prints n.d.).")
    out("  identity: 1/tau_Isat  vs  1/tau_n + 1/(2 tau_Te)   [1/ms]")
    header = (
        f"{'port':>5} {'z[cm]':>7} {'cell':>5} {'z_cell':>8} "
        f"{'tau_Isat':>8} {'n.sig':>6} {'tau_n':>7} {'n.sig':>6} "
        f"{'tau_Te':>7} {'n.sig':>6} {'1/t_I':>7} {'1/t_n+1/2t_Te':>14} "
        f"{'resid':>9}"
    )
    out(header)
    out("-" * len(header))
    for r in rows:
        f_i, f_n, f_te = r["model_isat"], r["model_n"], r["model_te"]
        lhs = 1.0 / f_i["tau_ms"] if np.isfinite(f_i["tau_ms"]) else np.nan
        rhs = np.nan
        if np.isfinite(f_n["tau_ms"]) and np.isfinite(f_te["tau_ms"]):
            rhs = 1.0 / f_n["tau_ms"] + 1.0 / (2.0 * f_te["tau_ms"])
        resid = (rhs - lhs) / lhs * 100.0 if np.isfinite(lhs) and lhs != 0.0 else np.nan
        resid_text = f"{resid:8.2e}%" if np.isfinite(resid) else f"{'n.d.':>9}"
        out(
            f"{r['port']:>5} {r['z']:7.0f} {r['iz']:>5} {r['z_cell']:8.1f} "
            f"{_tau_text(f_i, 8)} {_sigma_text(f_i)} {_tau_text(f_n)} "
            f"{_sigma_text(f_n)} {_tau_text(f_te)} {_sigma_text(f_te)} "
            f"{_num(lhs)} {_num(rhs, 14, 4)} {resid_text}"
        )
    out(
        "  the residual is a MASK check (OLS is linear in ln y, so the "
        "identity is exact"
    )
    out("  when the three fits see the same samples), not a physics check.")


def _report_measured_split(rows, scorer_rows, te_end_ms, n_end_ms, window_ms, out):
    t0, t1 = window_ms
    span = t1 - t0
    out("")
    out("  MEASURED, same window.  D_exp = 1 - exp(-span/tau_exp) is the")
    out("  fraction the measured trace actually decayed inside the window.")
    header = (
        f"{'port':>5} {'z[cm]':>7} {'tau_Isat':>8} {'n.sig':>6} "
        f"{'D_exp[%]':>9} {'N':>6}  {'tau_Te':<34} {'tau_n':<20}"
    )
    out(header)
    out("-" * len(header))
    te_text = f"n/a (overlay Te ends at {te_end_ms:.1f} ms)"
    n_text = "n/a (no measured n(t) in window)"
    for r, ref in zip(rows, scorer_rows):
        _, d_exp = _decay_observability(ref["tau_exp_ms"], span)
        d_text = f"{100.0 * d_exp:9.2f}" if np.isfinite(d_exp) else f"{'n.d.':>9}"
        out(
            f"{r['port']:>5} {r['z']:7.0f} {_tau_text(r['exp_isat'], 8)} "
            f"{_sigma_text(r['exp_isat'])} {d_text} {r['exp_n_good']:>6}  "
            f"{te_text:<34} {n_text:<20}"
        )
    out(
        f"  measured tau_Te: the overlay's Te trace (te_time_ms) ends at "
        f"{te_end_ms:.1f} ms on the"
    )
    out(
        f"    main-discharge clock, {t0 - te_end_ms:.1f} ms BEFORE this window "
        "opens -- there is no"
    )
    out("    measured Te sample inside any window in this report.")
    out(
        f"  measured tau_n: the overlay carries no n(t) trace in the "
        f"afterglow at all"
    )
    out(
        f"    (density_time_ms ends at {n_end_ms:.3f} ms, at 0.5 ms cadence); "
        "a measured"
    )
    out("    density decay time is NOT available and none is inferred here.")


def _report_window_isat(title, rows, scorer_rows, window_ms, base_rows, out):
    """Print one alternate-window Isat refit against the full-window fit."""
    t0, t1 = window_ms
    span = t1 - t0
    out("")
    out(f"--- {title} ---")
    header = (
        f"{'port':>5} {'z[cm]':>7} {'tau_model':>9} {'n.sig':>6} "
        f"{'tau_exp':>8} {'n.sig':>6} {'D_exp[%]':>9} {'ratio':>7} "
        f"{'tau_m/full':>10} {'tau_e/full':>10} {'ratio/full':>10}"
    )
    out(header)
    out("-" * len(header))
    ratios = []
    for r, ref, base in zip(rows, scorer_rows, base_rows):
        f_m, f_e = r["model_isat"], r["exp_isat"]
        b_m, b_e = base["model_isat"], base["exp_isat"]
        ratio = _ratio(f_m, f_e)
        ratios.append(ratio)
        base_ratio = _ratio(b_m, b_e)
        rel = ratio / base_ratio if np.isfinite(ratio) and np.isfinite(base_ratio) else np.nan
        _, d_exp = _decay_observability(ref["tau_exp_ms"], span)
        d_text = f"{100.0 * d_exp:9.2f}" if np.isfinite(d_exp) else f"{'n.d.':>9}"
        out(
            f"{r['port']:>5} {r['z']:7.0f} {_tau_text(f_m, 9)} "
            f"{_sigma_text(f_m)} {_tau_text(f_e, 8)} {_sigma_text(f_e)} "
            f"{d_text} {_num(ratio)} {_num(_ratio(f_m, b_m), 10)} "
            f"{_num(_ratio(f_e, b_e), 10)} {_num(rel, 10)}"
        )
    mean, kept = _mean_determined(ratios)
    if kept:
        out(
            f"  mean tau_model/tau_exp: {mean:.2f}   over {kept} of "
            f"{len(ratios)} port(s) determined on BOTH sides"
        )
    else:
        out(
            f"  mean tau_model/tau_exp: n.d. -- no port is determined on both "
            f"sides in this window"
        )
    return ratios


def battery(h5_path, es=1, out=print):
    """Print the whole battery for one saved run."""
    overlay_path = (
        OVERLAY if es == 1 else OVERLAY.parent / f"es{es}_sim1d_overlay.npz"
    )
    overlay = np.load(overlay_path, allow_pickle=False)
    result = load_result_hdf5(h5_path)

    origin_s = _main_discharge_origin(result)
    origin_ms = origin_s * 1.0e3
    events = getattr(result, "phase_events", None) or {}
    phases = [str(p) for p in np.asarray(events.get("phase", []))]
    times_ms = np.asarray(events.get("time", []), dtype=float) * 1.0e3
    shutoff_run_ms = np.nan
    if "afterglow" in phases:
        shutoff_run_ms = float(times_ms[phases.index("afterglow")])

    trim = (DECAY_WINDOW_MS[0] + TRIM_MS, DECAY_WINDOW_MS[1] - TRIM_MS)

    out(f"=== afterglow battery: {h5_path} (ES{es}) ===")
    out(f"overlay: {overlay_path}")
    out("")
    out("CLOCKS.  The scorer's windows are on the MAIN-DISCHARGE clock, whose")
    out("origin is the first 'main_discharge' sample.  run = main-discharge + origin.")
    out(
        f"  origin (first main_discharge sample): run clock {origin_ms:.4f} ms"
    )
    for phase, t_ms in zip(phases, times_ms):
        out(
            f"  phase_events: {phase:<14} run {t_ms:9.4f} ms  |  "
            f"main-discharge {t_ms - origin_ms:9.4f} ms"
        )
    if np.isfinite(shutoff_run_ms):
        out(
            f"  main-discharge SHUTOFF: run {shutoff_run_ms:.4f} ms  |  "
            f"main-discharge {shutoff_run_ms - origin_ms:.4f} ms"
        )
    t_model_ms = (np.asarray(result.time, dtype=float) - origin_s) * 1.0e3
    out(
        f"  trace span: main-discharge {t_model_ms.min():.3f} - "
        f"{t_model_ms.max():.3f} ms  |  run {t_model_ms.min() + origin_ms:.3f} - "
        f"{t_model_ms.max() + origin_ms:.3f} ms"
    )
    out("")
    out("WINDOWS (both clocks):")
    out(_clock_line("full", DECAY_WINDOW_MS, origin_ms))
    out(_clock_line("burst-trim", trim, origin_ms))
    out(_clock_line("late", LATE_WINDOW_MS, origin_ms))
    out(
        f"  burst-trim removes {TRIM_MS:.1f} ms from BOTH ends of the full "
        "window (symmetric:"
    )
    out("  the window centre does not move).")
    out("")
    out(
        f"DETERMINACY.  A fit is DETERMINED when its window log-slope is "
        f"negative and more"
    )
    out(
        f"than {DETERMINACY_SIGMA:.0f} sigma_OLS from zero.  Undetermined fits "
        "print n.d. and enter no mean."
    )

    te_end_ms = float(np.asarray(overlay["te_time_ms"], dtype=float).max())
    n_end_ms = float(np.asarray(overlay["density_time_ms"], dtype=float).max())

    full_rows, full_ref = _window_fits(result, overlay, origin_s, DECAY_WINDOW_MS)
    _report_split(full_rows, DECAY_WINDOW_MS, out)
    _assert_matches_scorer(full_rows, full_ref, DECAY_WINDOW_MS, out)
    _report_measured_split(
        full_rows, full_ref, te_end_ms, n_end_ms, DECAY_WINDOW_MS, out
    )

    out("")
    out(
        f"--- (i) full-window Isat ratios, for reference "
        f"({DECAY_WINDOW_MS[0]:.1f}-{DECAY_WINDOW_MS[1]:.1f} ms) ---"
    )
    base_ratios = []
    for r in full_rows:
        base_ratios.append(_ratio(r["model_isat"], r["exp_isat"]))
    header = f"{'port':>5} {'tau_model':>9} {'tau_exp':>8} {'ratio':>7}"
    out(header)
    out("-" * len(header))
    for r, ratio in zip(full_rows, base_ratios):
        out(
            f"{r['port']:>5} {_tau_text(r['model_isat'], 9)} "
            f"{_tau_text(r['exp_isat'], 8)} {_num(ratio)}"
        )
    mean, kept = _mean_determined(base_ratios)
    out(
        f"  mean tau_model/tau_exp: {mean:.2f}   over {kept} of "
        f"{len(base_ratios)} port(s)"
        if kept
        else "  mean tau_model/tau_exp: n.d."
    )

    trim_rows, trim_ref = _window_fits(result, overlay, origin_s, trim)
    _report_window_isat(
        f"(ii) symmetric burst-trim, {trim[0]:.1f}-{trim[1]:.1f} ms "
        f"(main-discharge) = {trim[0] + origin_ms:.3f}-{trim[1] + origin_ms:.3f} ms (run)",
        trim_rows,
        trim_ref,
        trim,
        full_rows,
        out,
    )
    _assert_matches_scorer(trim_rows, trim_ref, trim, out)

    late_rows, late_ref = _window_fits(result, overlay, origin_s, LATE_WINDOW_MS)
    _report_window_isat(
        f"(iii) late-window refit, {LATE_WINDOW_MS[0]:.1f}-{LATE_WINDOW_MS[1]:.1f} ms "
        f"(main-discharge) = {LATE_WINDOW_MS[0] + origin_ms:.3f}-"
        f"{LATE_WINDOW_MS[1] + origin_ms:.3f} ms (run)",
        late_rows,
        late_ref,
        LATE_WINDOW_MS,
        full_rows,
        out,
    )
    _assert_matches_scorer(late_rows, late_ref, LATE_WINDOW_MS, out)

    out("")
    out("--- (i) tau split in the alternate windows, model only ---")
    for name, rows_w, win in (
        ("burst-trim", trim_rows, trim),
        ("late", late_rows, LATE_WINDOW_MS),
    ):
        out(f"  {name} ({win[0]:.1f}-{win[1]:.1f} ms main-discharge):")
        header = (
            f"{'port':>7} {'tau_Isat':>8} {'n.sig':>6} {'tau_n':>7} "
            f"{'n.sig':>6} {'tau_Te':>7} {'n.sig':>6}"
        )
        out(header)
        out("-" * len(header))
        for r in rows_w:
            out(
                f"{r['port']:>7} {_tau_text(r['model_isat'], 8)} "
                f"{_sigma_text(r['model_isat'])} {_tau_text(r['model_n'])} "
                f"{_sigma_text(r['model_n'])} {_tau_text(r['model_te'])} "
                f"{_sigma_text(r['model_te'])}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path, help="saved sim1d run to read")
    parser.add_argument(
        "--es",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="which experiment-set overlay to read the measured side from",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="also write the report to this file"
    )
    args = parser.parse_args(argv)

    lines = []

    def emit(text=""):
        lines.append(str(text))
        print(text)

    battery(args.h5, es=args.es, out=emit)
    if args.out is not None:
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
