#!/usr/bin/env python
"""Report the directed neutral flow and ion drift at the ES1 ports.

Reads one or more saved sim1d result HDF5s and prints, for each ES1 probe
port, the plateau-window mean of the directed neutral flow decomposed into
its two channels, plus the ion drift, all in km/s with ``+`` toward the far
end of the machine (increasing ``z``).

Quantities
----------
``u_cold``
    Directed flow of the cold (thermalized) neutral population, computed as
    ``mean(M_n / m_He) / mean(nn)`` over the window -- the neutral momentum
    density divided by the helium mass, per cold neutral. **This is NOT the
    ``u_n`` dataset.** ``u_n`` is the chamber-mean neutral wind, a different
    quantity with a different normalization; using it here silently answers
    a different question. ``M_n`` is CGS momentum density (g cm^-2 s^-1),
    ``nn`` is cm^-3, so ``M_n / m_He / nn`` is cm/s.

``f_hot``
    Flight-weighted hot fraction, ``mean(hot_n_flight) / (mean(nn) +
    mean(hot_n_flight))`` -- the share of the local neutral population that
    is in hot flight. **Derived from the ``hot_n_flight`` dataset, not read
    from the saved ``f_hot`` dataset**, which carries the birth-side hot
    fraction and is not the same number.

``u_hot``
    Directed flow of the hot-flight population, ``mean(hot_flux_z) /
    mean(hot_n_flight)``. Reported as ``0`` when the run carries no hot
    channel (``hot_n_flight`` absent or identically zero), which is the
    correct reading for a cold-only control arm.

``u_eff``
    The density-weighted directed flow over both neutral channels,
    ``mean(M_n / m_He + hot_flux_z) / (mean(nn) + mean(hot_n_flight))``.
    For a run with no hot channel this reduces exactly to ``u_cold``.

``u_i``
    Ion drift, the window mean of the ``u`` dataset.

Runs with no hot channel are read with the hot terms taken as zero, so cold
control arms and hot-birth arms are directly comparable.

Averaging convention
--------------------
Every velocity is a RATIO OF WINDOW MEANS -- the window-mean flux over the
window-mean density -- never a mean of per-sample ratios. This is what makes
the decomposition an exact identity,

    u_eff == (1 - f_hot) * u_cold + f_hot * u_hot

to machine precision. Averaging the per-sample ratios instead breaks that
identity (by up to ~3e-3 km/s on a hot-birth arm), because a ratio of means
is density-weighted across the window while a mean of ratios is not.

Summary lines
-------------
After the per-port table: the ``p41 -> p50`` span, whether the flow is
monotone toward the end (``u_eff(p50) > u_eff(p41)``), and ``u_eff`` as a
percentage of the RESIDUAL references ``+1.8 km/s`` at p41 and ``+5.4 km/s``
at p50.

**Those two numbers are RESIDUALS, not the machine's flow, and must never be
quoted as a measured velocity.** Each is a measured-minus-model DIFFERENCE,
taken against the era model's ion drift; the bins that used them were
registered against the required INCREMENT, which is what makes a residual the
right reference there. The machine's ABSOLUTE ion flow is a different and
much larger quantity -- roughly 5 to 10.7 km/s across the ports -- so reading
the percentages below as "fraction of the measured flow" overstates the
model's agreement by a factor of about 2 to 5.

Raises
------
``ValueError``
    If the requested time window selects no samples in a run, or if any
    reported port quantity is non-finite (a NaN or Inf ledger is refused
    rather than printed, since a NaN reads as a plausible blank in a table).

Usage::

    python scripts/flow_read_sim1d.py run.h5 [run2.h5 ...]
    python scripts/flow_read_sim1d.py run.h5 --window 15.0 19.5
"""

import argparse

import h5py
import numpy as np

from cablp.vars._cons import m_He_cgs

# ES1 probe ports and their axial positions [cm].
PORTS = {"p11": 470.1, "p21": 789.5, "p29": 1045.2, "p41": 1428.5, "p50": 1716.1}

# RESIDUAL references at the two far ports [km/s] -- measured MINUS the era
# model's ion drift, NOT the machine's flow. The absolute measured ion flow is
# ~5 to 10.7 km/s across the ports; quoting these as a velocity overstates
# agreement several-fold. The name below is historical; the quantity is a
# residual.
MEASURED = {"p41": 1.8, "p50": 5.4}

# Plateau window [ms]: the drive-phase interval the port comparisons use.
DEFAULT_WINDOW_MS = (15.0, 19.5)

CM_S_PER_KM_S = 1e5


def read_flow(path, window_ms=DEFAULT_WINDOW_MS):
    """Return the per-port flow ledger for one run, keyed by port name.

    ``path`` is a sim1d result HDF5; ``window_ms`` is the ``(lo, hi)``
    averaging window in milliseconds, inclusive at both ends. Each value is
    a dict of ``u_cold``, ``f_hot``, ``u_hot``, ``u_eff`` and ``u_i``, with
    every velocity in km/s and positive toward increasing ``z``.
    """
    with h5py.File(path, "r") as handle:
        time = handle["time"][:]
        z_cm = handle["geometry/z_cm"][:]
        nn = handle["nn"][:]
        M_n = handle["M_n"][:]
        u_ion = handle["u"][:]
        # Each hot dataset is probed independently: a run may carry one
        # without the other, and a missing one reads as an absent channel.
        hot_n = (
            handle["hot_n_flight"][:]
            if "hot_n_flight" in handle
            else np.zeros_like(nn)
        )
        hot_flux = (
            handle["hot_flux_z"][:] if "hot_flux_z" in handle else np.zeros_like(nn)
        )

    lo_s, hi_s = window_ms[0] * 1e-3, window_ms[1] * 1e-3
    window = (time >= lo_s) & (time <= hi_s)
    if not window.any():
        raise ValueError(
            f"{path}: the window {window_ms[0]:g}-{window_ms[1]:g} ms selects no "
            f"samples; the run spans {time[0] * 1e3:.3f}-{time[-1] * 1e3:.3f} ms "
            f"over {time.size} samples. Refusing to report on an empty window."
        )

    ledger = {}
    for name, z_port in PORTS.items():
        index = int(np.argmin(np.abs(z_cm - z_port)))
        nn_mean = nn[window, index].mean()
        hot_mean = hot_n[window, index].mean()
        total_mean = nn_mean + hot_mean
        if not (nn_mean > 0.0 and total_mean > 0.0):
            raise ValueError(
                f"{path}: port {name} (z={z_port} cm) has non-positive neutral "
                f"density over the window (nn={nn_mean:.6e}, "
                f"hot_n_flight={hot_mean:.6e} cm^-3); the flow normalization is "
                f"undefined. Refusing to report."
            )
        cold_flux_mean = (M_n[window, index] / m_He_cgs).mean()
        entry = {
            "u_cold": cold_flux_mean / nn_mean / CM_S_PER_KM_S,
            "f_hot": hot_mean / total_mean,
            "u_hot": (
                hot_flux[window, index].mean() / hot_mean / CM_S_PER_KM_S
                if hot_mean > 0.0
                else 0.0
            ),
            "u_eff": (
                (M_n[window, index] / m_He_cgs + hot_flux[window, index]).mean()
                / total_mean
                / CM_S_PER_KM_S
            ),
            "u_i": u_ion[window, index].mean() / CM_S_PER_KM_S,
        }
        bad = sorted(key for key, value in entry.items() if not np.isfinite(value))
        if bad:
            raise ValueError(
                f"{path}: port {name} (z={z_port} cm) produced a non-finite "
                f"ledger for {', '.join(bad)}. Refusing to report a NaN/Inf "
                f"flow, which is indistinguishable from a blank in the table."
            )
        ledger[name] = entry
    return ledger


def report(path, ledger, window_ms):
    """Print the per-port table and the span/monotone/percent summary."""
    print(f"--- {path}, plateau {window_ms[0]:g}-{window_ms[1]:g} ms, km/s ---")
    for name, entry in ledger.items():
        print(
            f"  {name}: u_cold={entry['u_cold']:+.3f}  "
            f"f_hot(flight)={entry['f_hot']:.3f}  "
            f"u_hot={entry['u_hot']:+.3f}  u_eff={entry['u_eff']:+.3f}  "
            f"u_i={entry['u_i']:+.3f}"
        )
    u41 = ledger["p41"]["u_eff"]
    u50 = ledger["p50"]["u_eff"]
    print(
        f"  span p41->p50: {u50 - u41:+.3f} km/s | "
        f"monotone(u50>u41): {u50 > u41} | "
        f"vs RESIDUAL ref +{MEASURED['p41']}/+{MEASURED['p50']} km/s "
        f"(NOT measured flow): "
        f"p41 {u41 / MEASURED['p41'] * 100:.0f}%  "
        f"p50 {u50 / MEASURED['p50'] * 100:.0f}%"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Report plateau-window directed neutral flow (cold + hot-flight "
            "channels) and ion drift at the ES1 ports, from saved sim1d runs."
        )
    )
    parser.add_argument("runs", nargs="+", help="sim1d result HDF5 path(s)")
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        default=list(DEFAULT_WINDOW_MS),
        metavar=("T0_MS", "T1_MS"),
        help="averaging window in ms, inclusive (default: 15.0 19.5)",
    )
    args = parser.parse_args(argv)

    window_ms = (args.window[0], args.window[1])
    if not window_ms[1] > window_ms[0]:
        raise ValueError(
            f"--window must be increasing; got {window_ms[0]:g} {window_ms[1]:g} ms."
        )
    for path in args.runs:
        report(path, read_flow(path, window_ms), window_ms)


if __name__ == "__main__":
    main()
