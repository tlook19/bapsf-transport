"""L2 registration arms: plateau-window noise metrics vs the reference arm.

SCRATCH ANALYSIS SCRIPT (untracked). Read-only over saved artifacts; solves
nothing and writes no h5.

For each of {reference, arm1, arm2}, on the MAIN-DISCHARGE clock over the
plateau window 15.0-19.5 ms:

  (a) V_dis  -- quadratic fit in t over the window, residual = trace - fit,
                reported as p5, p95 and the p5-p95 span.
  (b) I_tot  -- same quadratic detrend, residual sigma (population std).
  (c) accepted-step count whose times fall inside the window.

Then arm/reference ratios for all three.

CLOCK AND TRACE DEFINITIONS ARE COPIED FROM scripts/fingerprints_sim1d.py so
the numbers sit on the same clock as every other drive-side reading:

  origin  = time of the FIRST SAVED sample whose phase is 'main_discharge'
            (fingerprints_sim1d._origin_s).
  V_dis   = V_dis_tavg, the dt-weighted interval average differenced from the
            running cathode_diagnostics/circuit_V_dis_dt_integral, front-
            padded to the per-save series. This is the dt-INTEGRATED circuit
            voltage, NOT the instantaneous per-solve source_V_b.
  I_tot   = cathode_diagnostics/source_I_tot (per-save).
  steps   = diagnostics/time, the ACCEPTED-step clock (full length, not the
            subsampled save clock), shifted by the same origin.

Datasets are read individually with h5py rather than through
load_result_hdf5: these artifacts are ~2.4 GB each and only five arrays are
needed.

Usage::

    python scripts/l2arm_noise_metrics.py LABEL=PATH [LABEL=PATH ...]

The FIRST pair is the reference; ratios are taken against it.
"""

import sys

import h5py
import numpy as np

WIN_LO_MS = 15.0
WIN_HI_MS = 19.5
FIT_DEGREE = 2


def _decode(values):
    return np.asarray(
        [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]
    )


def read_traces(path):
    """Return the five arrays this analysis needs, on the raw solver clock."""
    with h5py.File(path, "r") as h5:
        attrs = {
            key: h5.attrs[key]
            for key in ("steps", "run_status", "final_time", "compiled_kernels",
                        "t_breakdown_trigger", "t_prebreakdown_trigger")
            if key in h5.attrs
        }
        t_save = np.asarray(h5["time"][:], float)
        phase = _decode(h5["phase"][:])
        diag = h5["cathode_diagnostics"]
        I = np.asarray(diag["source_I_tot"][:], float)
        Vint = np.asarray(diag["circuit_V_dis_dt_integral"][:], float)
        t_step = np.asarray(h5["diagnostics/time"][:], float)
        ev = h5["phase_events"]
        events = list(
            zip(np.asarray(ev["time"][:], float), _decode(ev["phase"][:]),
                _decode(ev["reason"][:]))
        )
    return t_save, phase, I, Vint, t_step, attrs, events


def origin_s(t_save, phase):
    hits = np.flatnonzero(phase == "main_discharge")
    if not hits.size:
        raise RuntimeError(
            "NON-IGNITED RUN: no saved sample in the 'main_discharge' phase, "
            "so this run has no discharge origin and cannot be windowed."
        )
    return float(t_save[hits[0]])


def v_dis_tavg(t_save, Vint, I):
    """dt-weighted V_dis, exactly as fingerprints_sim1d builds it."""
    if not np.any(Vint != 0.0):
        raise RuntimeError(
            "this artifact carries no circuit_V_dis_dt_integral, so the "
            "dt-integrated V_dis is unavailable; the loop reconstruction is "
            "deliberately NOT substituted here (it is a different quantity)"
        )
    dt_save = np.diff(t_save)
    with np.errstate(invalid="ignore", divide="ignore"):
        V_mid = np.diff(Vint) / dt_save
    front = V_mid[0] if V_mid.size else 0.0
    return np.concatenate([[front], V_mid])


def detrend_residual(t_ms, y, mask):
    """Quadratic fit in t over the window; return (residual, fit coeffs)."""
    tw, yw = t_ms[mask], y[mask]
    good = np.isfinite(tw) & np.isfinite(yw)
    tw, yw = tw[good], yw[good]
    if tw.size < FIT_DEGREE + 2:
        raise RuntimeError(
            f"only {tw.size} finite samples in the window -- too few for a "
            f"degree-{FIT_DEGREE} fit"
        )
    coeffs = np.polyfit(tw, yw, FIT_DEGREE)
    return yw - np.polyval(coeffs, tw), coeffs, tw.size


def drive_end_ms(t_ms, phase):
    hits = np.flatnonzero(phase == "main_discharge")
    return float(t_ms[hits[-1]]) if hits.size else float(t_ms[-1])


def measure(label, path):
    t_save, phase, I, Vint, t_step, attrs, events = read_traces(path)
    t0 = origin_s(t_save, phase)
    t_ms = (t_save - t0) * 1e3
    ts_ms = (t_step - t0) * 1e3
    V = v_dis_tavg(t_save, Vint, I)

    win = (t_ms >= WIN_LO_MS) & (t_ms <= WIN_HI_MS)
    t_end = drive_end_ms(t_ms, phase)
    drive = (t_ms >= 0.0) & (t_ms <= t_end)

    vres, vcoef, nv = detrend_residual(t_ms, V, win)
    ires, icoef, ni = detrend_residual(t_ms, I, win)
    steps_in_win = int(np.count_nonzero((ts_ms >= WIN_LO_MS)
                                        & (ts_ms <= WIN_HI_MS)))

    Ipk = float(np.max(I[drive]))
    tpk = float(t_ms[drive][np.argmax(I[drive])])
    above = np.flatnonzero(drive & (I >= 0.9 * Ipk))
    t90 = float(t_ms[above[0]]) if above.size else float("nan")

    return {
        "label": label,
        "path": path,
        "attrs": attrs,
        "events": events,
        "t_origin_s": t0,
        "drive_end_ms": t_end,
        "n_saves_in_win": nv,
        "V_p5": float(np.percentile(vres, 5)),
        "V_p95": float(np.percentile(vres, 95)),
        "V_span": float(np.percentile(vres, 95) - np.percentile(vres, 5)),
        "V_sigma": float(np.std(vres)),
        "V_fit": vcoef,
        "I_sigma": float(np.std(ires)),
        "I_p5": float(np.percentile(ires, 5)),
        "I_p95": float(np.percentile(ires, 95)),
        "I_fit": icoef,
        "steps_in_win": steps_in_win,
        "V_plateau_mean": float(np.mean(V[win])),
        "V_plateau_median": float(np.median(V[win])),
        "I_plateau_mean": float(np.mean(I[win])),
        "I_plateau_median": float(np.median(I[win])),
        "I_peak": Ipk,
        "t_peak_ms": tpk,
        "t90_ms": t90,
    }


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    pairs = []
    for item in argv:
        label, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"expected LABEL=PATH, got {item!r}")
        pairs.append((label, path))

    rows = [measure(label, path) for label, path in pairs]
    ref = rows[0]

    print(f"plateau window {WIN_LO_MS}-{WIN_HI_MS} ms on the main-discharge "
          f"clock | detrend: degree-{FIT_DEGREE} polynomial in t")
    print(f"V_dis = V_dis_tavg (dt-integrated circuit voltage), "
          f"I_tot = source_I_tot")
    print(f"reference for all ratios: {ref['label']}\n")

    print("=== PER-RUN PROVENANCE ===")
    for r in rows:
        a = r["attrs"]
        print(f"{r['label']}: {r['path']}")
        print(f"  run_status={a.get('run_status')} steps={a.get('steps')} "
              f"final_time={float(a.get('final_time', float('nan'))):.9f} s "
              f"compiled_kernels={a.get('compiled_kernels')}")
        print(f"  main-discharge origin {r['t_origin_s']:.9f} s | "
              f"drive end +{r['drive_end_ms']:.3f} ms | "
              f"saves in window {r['n_saves_in_win']}")
        for t, ph, why in r["events"]:
            print(f"    phase_event {t:.9f} s -> {ph} ({why})")
    print()

    print("=== NOISE METRICS ===")
    hdr = (f"{'run':<10} {'V res p5':>10} {'V res p95':>10} {'V span':>9} "
           f"{'V sigma':>9} {'I res sig':>10} {'acc steps':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<10} {r['V_p5']:>10.4f} {r['V_p95']:>10.4f} "
              f"{r['V_span']:>9.4f} {r['V_sigma']:>9.4f} "
              f"{r['I_sigma']:>10.4f} {r['steps_in_win']:>10d}")
    print("  (V in volts, I in amperes; span = p95 - p5)")
    print()

    print("=== RATIOS vs %s (tripwire: each <= 1.5) ===" % ref["label"])
    hdr2 = (f"{'run':<10} {'V span':>10} {'I res sig':>10} {'acc steps':>10}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in rows[1:]:
        print(f"{r['label']:<10} "
              f"{r['V_span'] / ref['V_span']:>10.4f} "
              f"{r['I_sigma'] / ref['I_sigma']:>10.4f} "
              f"{r['steps_in_win'] / ref['steps_in_win']:>10.4f}")
    print()

    print("=== HEADLINE DRIVE NUMBERS ===")
    hdr3 = (f"{'run':<10} {'V_dis mean':>11} {'V_dis med':>10} "
            f"{'I_tot mean':>11} {'I_tot med':>10} {'I_tot peak':>11} "
            f"{'t_peak ms':>10} {'t90 ms':>8}")
    print(hdr3)
    print("-" * len(hdr3))
    for r in rows:
        print(f"{r['label']:<10} {r['V_plateau_mean']:>11.4f} "
              f"{r['V_plateau_median']:>10.4f} {r['I_plateau_mean']:>11.2f} "
              f"{r['I_plateau_median']:>10.2f} {r['I_peak']:>11.2f} "
              f"{r['t_peak_ms']:>10.3f} {r['t90_ms']:>8.3f}")
    print("  (V_dis mean/median over the plateau window; I_tot peak and "
          "t_peak/t90 over the FULL drive phase)")
    print()

    print("=== IGNITION / BREAKDOWN TIMING (raw solver clock) ===")
    for r in rows:
        a = r["attrs"]
        tpb = a.get("t_prebreakdown_trigger")
        tbd = a.get("t_breakdown_trigger")
        print(f"{r['label']:<10} t_prebreakdown_trigger="
              f"{float(tpb):.9f} s  t_breakdown_trigger={float(tbd):.9f} s  "
              f"main_discharge origin={r['t_origin_s']:.9f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
