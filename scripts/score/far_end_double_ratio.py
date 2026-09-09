"""Far-end AREA-FREE double ratio D = (X50/X41) / (X11/X21) over the drive
plateau, for one saved run against one experiment set's overlay.

WHY A DOUBLE RATIO. Each port row of the overlay carries a probe-area
calibration. The far-end pair (p50 referred to p41) and the near pair (p11
referred to p21) share ONE probe's area calibration -- p11 and p50 are the two
rows the exporter's ``probe_a_factor`` bracket is built from (its lower bound
is n41/raw50 and its upper bound is n21/raw11) -- so any common multiplicative
error k on those two rows enters the numerator of BOTH sub-ratios and divides
out of D exactly:

    D = ((k X50) / X41) / ((k X11) / X21) = (X50/X41) / (X11/X21)

D is therefore free of the probe-A area factor by construction, and a
model-vs-measured comparison of D is a statement about the far-end SHAPE alone,
not about any level either side carries. The cancellation is asserted
numerically by the ``far-end-double-ratio-area-cancels`` case of
scripts/gates/smoke_sim1d.py, which scales the p11 and p50 areas (and the
already area-normalized p11/p50 rows) by a common factor and requires every D
to be unchanged to 1e-12.

THREE METRICS, on both sides:

  (a) core-band density   -- measured ``density_mean_cm3``  vs model n
  (b) flux-tube density   -- measured ``density_ftavg_cm3`` vs model n
  (c) J = n sqrt(Te)      -- the probes' own ion-saturation current per unit
                             area, measured in its TWO area-normalized
                             conventions (the upstream face alone, and the
                             flow-symmetrized two-face geomean) vs model
                             n sqrt(Te)

(a) and (b) differ on the MEASURED side only -- the radial convention -- so
they share a model D; the 1D model carries one radial cell spanning exactly the
flux-tube radius, which is why the flux-tube row needs no model-side change
(the same reasoning the scored ``n_ft`` row of compare_sim1d_es1 states). (c)
carries two measured conventions and one model D: the upstream face keeps the
Mach-probe flow factor, the geomean cancels it to first order in M, so the two
bracket the flow-artifact treatment rather than agreeing.

Every reading is the plateau mean built by ``compare_sim1d_es1``'s own reader,
so each port's samples, mask and interpolation are the ones the scored rows
use. Nothing here is scored: this module enters no total, no mean and no
verdict of the scorer, and the scorer does not import it.

Read-only over a saved artifact. Usage:

    python scripts/score/far_end_double_ratio.py --from-h5 RUN.h5 --es 1
"""

import argparse
import textwrap
from pathlib import Path

import numpy as np

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from cablp.solvers._sim1d import load_result_hdf5  # noqa: E402

import compare_sim1d_es1 as _cmp  # noqa: E402

#: Overlay directory, taken from the scorer so the two never disagree on which
#: overlay set an ``--es`` selects.
OVERLAY_DIR = _cmp.OVERLAY.parent

#: The two ordered (denominator, numerator) port pairs D is built from. The FAR
#: pair reads the far end against the last mid-machine port; the NEAR pair is
#: the reference pair sharing the far pair's probe-area calibration, which is
#: what makes their quotient area-free.
FAR_PAIR = (41, 50)
NEAR_PAIR = (21, 11)

#: Every port a row needs, in z order.
DOUBLE_RATIO_PORTS = (11, 21, 41, 50)

#: Pre-registered decision threshold on D_model / D_measured. Printed with the
#: rows; this module states which bin the readings fall in and interprets
#: nothing further.
DOUBLE_RATIO_BIN = 1.10

#: Overlay keys the J metrics read. The upstream face is area-normalized here
#: by the geomean family's own paired area (column 0), exactly as the scorer's
#: plateau z-trend row does, and the column-0-is-upstream invariant is asserted
#: per port rather than assumed -- the exporter is known to flip the pairing's
#: segment order port by port.
J_OVERLAY_KEYS = (
    "isat_ftavg_geomean_a_per_cm2",
    "isat_ftavg_geomean_time_ms",
    "isat_ftavg_geomean_port",
    "isat_ftavg_geomean_area_cm2",
    "isat_ftavg_geomean_pairing",
    "isat_ftavg_upstream_a",
    "isat_ftavg_upstream_port",
    "isat_ftavg_upstream_source_channel",
)

#: Overlay keys the density metrics read, beyond the shared z-lookup keys
#: below (``port``, ``z_cm``).
DENSITY_OVERLAY_KEYS = (
    "density_time_ms",
    "density_mean_cm3",
    "density_ftavg_cm3",
)

#: The z-lookup keys every metric family needs -- ``z_by_port`` is built once,
#: shared across both families, before any per-family gating runs. A missing
#: key here is checked as an all-or-nothing prerequisite: unlike a
#: per-family field (which withholds only that family's rows), a missing
#: shared key would otherwise raise building ``z_by_port`` rather than
#: skipping cleanly.
SHARED_OVERLAY_KEYS = ("port", "z_cm")

#: One metric row: (id, printed label, model comparand, measured convention
#: line). ``model`` is "n" for the density metrics and "Isat" for the J
#: metrics, naming which model field the row's model side is built from.
METRIC_SPECS = (
    ("n", "(a) core-band density n", "n",
     "measured density_mean_cm3 [cm^-3]"),
    ("n_ft", "(b) flux-tube density n_ft", "n",
     "measured density_ftavg_cm3 [cm^-3]"),
    ("J_upstream", "(c) J = n sqrt(Te), upstream face", "Isat",
     "measured isat_ftavg_upstream_a / isat_ftavg_geomean_area_cm2[:, 0] "
     "[A cm^-2]"),
    ("J_geomean", "(c) J = n sqrt(Te), two-face geomean", "Isat",
     "measured isat_ftavg_geomean_a_per_cm2 [A cm^-2]"),
)

#: The two overlay key families a metric's measured side is read from,
#: keyed by the family name used in the gating messages.
FAMILY_OVERLAY_KEYS = {
    "density": DENSITY_OVERLAY_KEYS,
    "J": J_OVERLAY_KEYS,
}

#: Which family each ``METRIC_SPECS`` entry belongs to, for per-family
#: gating -- a missing key in one family must not withhold the other's rows.
METRIC_FAMILY = {
    "n": "density",
    "n_ft": "density",
    "J_upstream": "J",
    "J_geomean": "J",
}


def model_comparand(result, model_field):
    """Return the model 2-D (time, z) array a metric's model side reads.

    ``"Isat"`` is the scorer's Isat proxy ``n * sqrt(max(Te, 0))``; any other
    name is the result field of that name. Raises ``AttributeError`` on a
    result that carries no such field rather than substituting one.
    """
    if model_field == "Isat":
        return np.asarray(result.n, dtype=float) * np.sqrt(
            np.maximum(np.asarray(result.Te, dtype=float), 0.0)
        )
    return np.asarray(getattr(result, model_field), dtype=float)


def measured_traces(overlay, metric_id):
    """Return ``(t_exp, {port: trace})`` for one metric's measured side.

    The J-upstream metric is the only constructed one: the overlay exports the
    upstream face as a raw current, and this divides it by the same paired area
    the geomean was built from. Every other metric reads one exported field.
    """
    if metric_id == "J_geomean":
        t_exp = np.asarray(
            overlay["isat_ftavg_geomean_time_ms"], dtype=float
        )
        values = np.asarray(
            overlay["isat_ftavg_geomean_a_per_cm2"], dtype=float
        )
        ports = [int(p) for p in np.asarray(overlay["isat_ftavg_geomean_port"])]
        return t_exp, {p: values[i] for i, p in enumerate(ports)}
    if metric_id == "J_upstream":
        t_exp = np.asarray(
            overlay["isat_ftavg_geomean_time_ms"], dtype=float
        )
        areas = np.asarray(
            overlay["isat_ftavg_geomean_area_cm2"], dtype=float
        )
        geo_ports = [
            int(p) for p in np.asarray(overlay["isat_ftavg_geomean_port"])
        ]
        upstream = np.asarray(overlay["isat_ftavg_upstream_a"], dtype=float)
        up_ports = [
            int(p) for p in np.asarray(overlay["isat_ftavg_upstream_port"])
        ]
        _cmp._assert_upstream_is_area_column0(
            "isat_ftavg",
            geo_ports,
            np.asarray(overlay["isat_ftavg_geomean_pairing"]),
            areas,
            up_ports,
            np.asarray(overlay["isat_ftavg_upstream_source_channel"]),
        )
        traces = {}
        for i, port in enumerate(up_ports):
            if port not in geo_ports:
                continue
            traces[port] = upstream[i] / areas[geo_ports.index(port), 0]
        return t_exp, traces
    key = "density_mean_cm3" if metric_id == "n" else _cmp.FTAVG_DENSITY_KEY
    t_exp = np.asarray(overlay["density_time_ms"], dtype=float)
    values = np.asarray(overlay[key], dtype=float)
    ports = [int(p) for p in np.asarray(overlay["port"])]
    return t_exp, {p: values[i] for i, p in enumerate(ports)}


def double_ratio_rows(result, overlay, window_ms=None):
    """Return ``(rows, skip_reason)``: one row per ``METRIC_SPECS`` entry.

    Each row carries the measured and model far ratio (p50/p41), near ratio
    (p11/p21), their quotient D on each side, and ``D_model / D_measured``.
    A metric whose overlay fields or ports are not all present is skipped with
    its own reason rather than silently dropped; ``skip_reason`` is a printable
    sentence and ``rows`` is empty only when NO metric could be formed. The
    shared z-lookup keys (``port``, ``z_cm``) are an ALL-OR-NOTHING
    prerequisite checked first -- every metric needs them, so a missing one
    skips every row with one reason rather than a per-family skip. Past that,
    the density and J (Isat) overlay key families are gated INDEPENDENTLY: a
    family with all its keys present is read, and a family with a missing key
    is skipped on its own, naming the missing keys -- one family's absence
    never withholds the other's metrics.

    ``window_ms`` defaults to the scorer's plateau window, so the readings are
    commensurate with the scored plateau rows.
    """
    window = _cmp.PLATEAU_MS if window_ms is None else window_ms
    missing_shared = _cmp._missing_overlay_keys(overlay, SHARED_OVERLAY_KEYS)
    if missing_shared:
        return [], (
            f"this overlay (schema v{_cmp._overlay_vintage(overlay)}) "
            "carries no " + ", ".join(missing_shared) + " -- the far-end "
            "double ratio's z-lookup is shared by every metric family, so a "
            "missing shared key withholds all of them"
        )
    missing_by_family = {
        family: _cmp._missing_overlay_keys(overlay, keys)
        for family, keys in FAMILY_OVERLAY_KEYS.items()
    }
    skipped = [
        f"{family} family: this overlay (schema v"
        f"{_cmp._overlay_vintage(overlay)}) carries no " + ", ".join(missing)
        + f" -- the far-end double ratio's {family} metrics need this "
        "family's overlay fields over the same four ports, and a metric "
        "built from a substitute field would not be the one this instrument "
        "names"
        for family, missing in missing_by_family.items()
        if missing
    ]
    z_by_port = {
        int(p): float(z)
        for p, z in zip(
            np.asarray(overlay["port"]),
            np.asarray(overlay["z_cm"], dtype=float),
        )
    }
    origin = _cmp._main_discharge_origin(result)
    t_model_ms = (np.asarray(result.time, dtype=float) - origin) * 1.0e3
    z_model = np.asarray(result.z_cm, dtype=float)

    rows = []
    for metric_id, label, model_field, convention in METRIC_SPECS:
        if missing_by_family[METRIC_FAMILY[metric_id]]:
            continue
        t_exp, traces = measured_traces(overlay, metric_id)
        model_2d = model_comparand(result, model_field)
        readings = {}
        for port in DOUBLE_RATIO_PORTS:
            if port not in traces or port not in z_by_port:
                continue
            iz = int(np.argmin(np.abs(z_model - z_by_port[port])))
            reading = _cmp._plateau_port_reading(
                t_exp, traces[port], None, model_2d, iz, t_model_ms, window
            )
            if reading is None:
                continue
            reading["z"] = z_by_port[port]
            readings[port] = reading
        absent = [p for p in DOUBLE_RATIO_PORTS if p not in readings]
        if absent:
            skipped.append(
                f"{label}: no plateau reading at "
                + ", ".join(f"p{p}" for p in absent)
                + " -- the double ratio needs all four ports"
            )
            continue
        far_lo, far_hi = FAR_PAIR
        near_lo, near_hi = NEAR_PAIR
        far_exp = readings[far_hi]["exp"] / readings[far_lo]["exp"]
        near_exp = readings[near_hi]["exp"] / readings[near_lo]["exp"]
        far_model = readings[far_hi]["model"] / readings[far_lo]["model"]
        near_model = readings[near_hi]["model"] / readings[near_lo]["model"]
        d_exp = far_exp / near_exp
        d_model = far_model / near_model
        rows.append(
            {
                "metric": metric_id,
                "label": label,
                "model_field": model_field,
                "convention": convention,
                "far_pair": FAR_PAIR,
                "near_pair": NEAR_PAIR,
                "far_ratio_exp": float(far_exp),
                "near_ratio_exp": float(near_exp),
                "far_ratio_model": float(far_model),
                "near_ratio_model": float(near_model),
                "D_measured": float(d_exp),
                "D_model": float(d_model),
                "D_ratio": float(d_model / d_exp),
                "n_samples": int(
                    min(readings[p]["n_samples"] for p in DOUBLE_RATIO_PORTS)
                ),
                "levels": {
                    p: {
                        "z": readings[p]["z"],
                        "exp": readings[p]["exp"],
                        "model": readings[p]["model"],
                    }
                    for p in DOUBLE_RATIO_PORTS
                },
            }
        )
    if not rows:
        return [], "; ".join(skipped)
    return rows, ("; ".join(skipped) if skipped else None)


def _legend_lines(overlay, window):
    """Return the printable legend: pairing, window and per-metric convention.

    Everything factual here is READ FROM THE OVERLAY -- the face ruling, the
    per-port pairing and the probe-A bracket are the exporter's own statements,
    quoted rather than restated, so this legend cannot drift from the file it
    describes.
    """
    lines = []
    lines.append(
        f"window: {window[0]:g}-{window[1]:g} ms on the main-discharge clock "
        "(the scorer's PLATEAU_MS drive plateau)"
    )
    lines.append(
        f"double ratio: D = (X{FAR_PAIR[1]}/X{FAR_PAIR[0]}) / "
        f"(X{NEAR_PAIR[1]}/X{NEAR_PAIR[0]}), each X a plateau-window mean"
    )
    if all(key in overlay for key in J_OVERLAY_KEYS):
        lines.append("")
        lines.append("port rows and the faces behind them:")
        geo_ports = [
            int(p) for p in np.asarray(overlay["isat_ftavg_geomean_port"])
        ]
        pairing = np.asarray(overlay["isat_ftavg_geomean_pairing"])
        areas = np.asarray(overlay["isat_ftavg_geomean_area_cm2"], dtype=float)
        channels = np.asarray(overlay["isat_ftavg_upstream_source_channel"])
        up_ports = [
            int(p) for p in np.asarray(overlay["isat_ftavg_upstream_port"])
        ]
        channel_by_port = {int(p): str(c) for p, c in zip(up_ports, channels)}
        for port in DOUBLE_RATIO_PORTS:
            if port not in geo_ports:
                continue
            i = geo_ports.index(port)
            lines.append(
                f"  p{port:<3d} upstream channel {channel_by_port.get(port, '?')}"
                f"  area(upstream) {areas[i, 0]:.6f} cm2"
                f"  area(downstream) {areas[i, 1]:.6f} cm2"
            )
            lines.append(f"        pairing: {str(pairing[i])}")
    lines.append("")
    lines.append("metric conventions:")
    for _, label, model_field, convention in METRIC_SPECS:
        model_line = (
            "model n * sqrt(max(Te, 0)) at the port cell"
            if model_field == "Isat"
            else "model n at the port cell"
        )
        lines.append(f"  {label}")
        lines.append(f"      {convention}; {model_line}")
    lines.append("")
    for key in ("density_mean_convention", "ftavg_convention",
                "isat_ftavg_upstream_face", "isat_ftavg_geomean_definition"):
        if key not in overlay:
            continue
        lines.append(f"  [{key}]")
        lines.extend(
            textwrap.wrap(
                str(overlay[key]), width=96,
                initial_indent="      ", subsequent_indent="      ",
            )
        )
    if "probe_a_factor" in overlay:
        lines.append("")
        lines.append(
            "  probe-A area factor "
            f"{float(overlay['probe_a_factor']):.6f} "
            f"[{float(overlay['probe_a_factor_lower_bound']):.6f}, "
            f"{float(overlay['probe_a_factor_upper_bound']):.6f}] applies to "
            f"the p{NEAR_PAIR[1]} and p{FAR_PAIR[1]} rows and CANCELS in D: it "
            "multiplies the numerator of both sub-ratios"
        )
    return lines


def report_double_ratio(label, rows, skip_reason, overlay, window):
    """Print the legend, the D table and the pre-registered bin verdict."""
    print()
    print(f"=== far-end area-free double ratio: {label} ===")
    if not rows:
        print(f"  (no metric formed) {skip_reason}")
        return
    for line in _legend_lines(overlay, window):
        print(line)
    print()
    if skip_reason:
        print(f"  partial: {skip_reason}")
        print()
    header = (
        f"{'metric':38s} {'meas far':>9s} {'meas near':>9s} {'D_meas':>8s} "
        f"{'mod far':>9s} {'mod near':>9s} {'D_model':>8s} "
        f"{'D_mod/D_meas':>12s} {'nsamp':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label']:38s} {row['far_ratio_exp']:9.4f} "
            f"{row['near_ratio_exp']:9.4f} {row['D_measured']:8.4f} "
            f"{row['far_ratio_model']:9.4f} {row['near_ratio_model']:9.4f} "
            f"{row['D_model']:8.4f} {row['D_ratio']:12.4f} "
            f"{row['n_samples']:6d}"
        )
    print()
    print("plateau-mean levels behind the ratios (measured | model):")
    for row in rows:
        cells = "  ".join(
            f"p{p} {row['levels'][p]['exp']:.4g}|{row['levels'][p]['model']:.4g}"
            for p in DOUBLE_RATIO_PORTS
        )
        print(f"  {row['label']:38s} {cells}")
    print()
    ratios = [row["D_ratio"] for row in rows]
    print(f"PRE-REGISTERED BINS (threshold D_model/D_measured = "
          f"{DOUBLE_RATIO_BIN:.2f}):")
    print(
        f"  > {DOUBLE_RATIO_BIN:.2f} in EVERY metric -> the far-end residual "
        "is area-free; the x1.1-1.9 bracket is adopted"
    )
    print(
        f"  <= {DOUBLE_RATIO_BIN:.2f} in ANY metric -> the ratified x1.23 "
        "far-end ratio is withdrawn to convention- and area-conditional"
    )
    below = [row["label"] for row in rows if row["D_ratio"] <= DOUBLE_RATIO_BIN]
    if below:
        print(
            f"  BIN HIT: <= {DOUBLE_RATIO_BIN:.2f} in "
            + "; ".join(below)
            + f"  (min {min(ratios):.4f}, max {max(ratios):.4f})"
        )
    else:
        print(
            f"  BIN HIT: > {DOUBLE_RATIO_BIN:.2f} in every metric "
            f"(min {min(ratios):.4f}, max {max(ratios):.4f})"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-h5", type=Path, required=True,
        help="saved run to read the model side from; this instrument never "
             "runs a model, so there is no configuration to name here -- the "
             "artifact carries its own",
    )
    parser.add_argument(
        "--es", type=int, default=1, choices=(1, 2, 3, 4),
        help="experiment set whose overlay supplies the measured side",
    )
    parser.add_argument(
        "--window", type=float, nargs=2, default=None, metavar=("T0", "T1"),
        help="plateau window on the main-discharge clock [ms] "
             f"(default {_cmp.PLATEAU_MS[0]:g} {_cmp.PLATEAU_MS[1]:g})",
    )
    args = parser.parse_args(argv)

    window = _cmp.PLATEAU_MS if args.window is None else tuple(args.window)
    overlay_path = OVERLAY_DIR / f"es{args.es}_sim1d_overlay.npz"
    overlay = np.load(overlay_path, allow_pickle=False)
    result = load_result_hdf5(args.from_h5)
    rows, skip_reason = double_ratio_rows(result, overlay, window_ms=window)
    label = f"ES{args.es} vs {args.from_h5}"
    report_double_ratio(label, rows, skip_reason, overlay, window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
