"""Render the ES1 afterglow BOTH-FACE comparison figure for a saved sim1d run.

Slide-style comparison-to-data companion to stage (iii) of
``compare_sim1d_es1.py``: the same window, the same log-linear fit, the same
overlay product. Plotting only -- it never runs LAPDSim1D and never scores.

Two panels' worth of content:

  (1) per ES1 port, the measured current DENSITIES on the two Mach-probe faces
      at x = 0 -- J_up = isat_decay_mean_a / A_up (the 'i_sweep' channel) and
      J_dn = isat_decay_dn_mean_a / A_dn (the 'isat' channel, which collects in
      the probe body's flow shadow) -- together with the overlay's
      area-normalized geometric mean sqrt(J_up*J_dn), all in A cm^-2, and the
      model Isat proxy n*sqrt(Te) NORMALIZED to the geomean at the window
      start. The proxy carries no absolute unit against the probe, so only its
      SHAPE is being compared and the normalization says so explicitly;
  (2) the tau(z) bracket: per port the measured e-fold time as the interval
      [tau_dn .. tau_up] with the flow-cancelled geomean marked inside it, and
      the model tau beside it. The two faces carry the Chung flow factor with
      opposite sign, so the measurement is the bracket, not either face.

The figure is drawn from the overlay's own keys and skips loudly, without
plotting, on an overlay vintage that carries no second face.

Usage:
  python scripts/plot_es1_afterglow_faces.py \
      --from-h5 scripts/m1_arm2_es1.h5 \
      --output scripts/m1_arm2_es1_afterglow_faces.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cablp.solvers._sim1d import load_result_hdf5

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_sim1d_es1 import (  # noqa: E402
    DECAY_FACE_KEYS,
    DECAY_WINDOW_MS,
    _main_discharge_origin,
    _missing_overlay_keys,
    compare_decay,
    compare_decay_faces,
)

# Context drawn either side of the scored window, so the reader can see the
# window sitting on the decay rather than a floating 1.5 ms of curve.
CONTEXT_MS = (0.5, 1.0)
FACE_STYLE = {
    "up": ("tab:blue", "-", "J_up (upstream face)"),
    "dn": ("tab:orange", "-", "J_dn (downstream face)"),
    "geo": ("k", "-", "geomean sqrt(J_up*J_dn)"),
}


def _overlay_path(es):
    name = "es1_sim1d_overlay.npz" if es == 1 else f"es{es}_sim1d_overlay.npz"
    return SCRIPT_DIR / "data" / name


def _model_proxy(result, z_cm):
    """Return ``(t_ms, n*sqrt(Te))`` at the model cell nearest ``z_cm``."""
    origin = _main_discharge_origin(result)
    t_ms = (np.asarray(result.time, dtype=float) - origin) * 1.0e3
    z_model = np.asarray(result.z_cm, dtype=float)
    iz = int(np.argmin(np.abs(z_model - float(z_cm))))
    n = np.asarray(result.n, dtype=float)[:, iz]
    te = np.asarray(result.Te, dtype=float)[:, iz]
    return t_ms, n * np.sqrt(np.maximum(te, 0.0))


def _draw_port_panel(ax, port_row, overlay, result, window):
    """Draw one port's two faces, their geomean, and the normalized proxy."""
    t0, t1 = float(window[0]), float(window[1])
    t_exp = np.asarray(overlay["isat_decay_time_ms"], dtype=float)
    ports = list(np.asarray(overlay["isat_decay_port"], dtype=int))
    p = ports.index(int(port_row["port"]))
    traces = {
        "up": np.asarray(overlay["isat_decay_mean_a"], dtype=float)[p]
        / port_row["area_up_cm2"],
        "dn": np.asarray(overlay["isat_decay_dn_mean_a"], dtype=float)[p]
        / port_row["area_dn_cm2"],
        "geo": np.asarray(overlay["isat_decay_geomean_a_per_cm2"], dtype=float)[p],
    }
    lo, hi = t0 - CONTEXT_MS[0], t1 + CONTEXT_MS[1]
    shown = (t_exp >= lo) & (t_exp <= hi)
    for key, trace in traces.items():
        color, style, label = FACE_STYLE[key]
        ax.plot(t_exp[shown], trace[shown], style, color=color, lw=1.3, label=label)

    t_model, proxy = _model_proxy(result, port_row["z"])
    geo_at_start = float(np.interp(t0, t_exp, traces["geo"]))
    proxy_at_start = float(np.interp(t0, t_model, proxy))
    if np.isfinite(geo_at_start) and proxy_at_start > 0.0:
        scale = geo_at_start / proxy_at_start
        m_shown = (t_model >= lo) & (t_model <= hi)
        ax.plot(
            t_model[m_shown],
            proxy[m_shown] * scale,
            "--",
            color="tab:red",
            lw=1.5,
            label="model n*sqrt(Te), normalized at window start",
        )
    ax.axvspan(t0, t1, color="0.85", zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_title(f"p{port_row['port']}  z = {port_row['z']:.0f} cm", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", alpha=0.25)


def _draw_bracket_panel(ax, face_rows):
    """Draw the measured [tau_dn .. tau_up] bracket and the model tau vs z."""
    z_all = [r["z"] for r in face_rows]
    pad = 0.08 * (max(z_all) - min(z_all)) if len(z_all) > 1 else 100.0
    for i, r in enumerate(face_rows):
        z = r["z"]
        lo = min(r["tau_dn_ms"], r["tau_up_ms"])
        hi = max(r["tau_dn_ms"], r["tau_up_ms"])
        ax.annotate(
            f"p{r['port']}",
            (z, hi),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color="0.35",
        )
        ax.vlines(z, lo, hi, color="0.35", lw=6, alpha=0.45,
                  label="measured bracket [tau_dn .. tau_up]" if i == 0 else None)
        ax.plot(z, r["tau_up_ms"], "_", color="tab:blue", ms=16, mew=2,
                label="tau_up" if i == 0 else None)
        ax.plot(z, r["tau_dn_ms"], "_", color="tab:orange", ms=16, mew=2,
                label="tau_dn" if i == 0 else None)
        ax.plot(z, r["tau_geo_ms"], "o", color="k", ms=6,
                label="tau_geo (flow-cancelled)" if i == 0 else None)
        ax.plot(z, r["tau_model_ms"], "D", color="tab:red", ms=6,
                label="model n*sqrt(Te)" if i == 0 else None)
    ax.set_xlim(min(z_all) - pad, max(z_all) + pad)
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("e-fold time [ms]")
    ax.set_title(
        "afterglow e-fold time: measured face bracket vs the model proxy",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper left")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-h5", required=True, help="saved sim1d result")
    parser.add_argument("--output", required=True, help="figure file to write")
    parser.add_argument("--es", type=int, default=1, help="overlay rung")
    parser.add_argument(
        "--decay-window",
        type=float,
        nargs=2,
        default=DECAY_WINDOW_MS,
        metavar=("T0_MS", "T1_MS"),
        help="stage (iii) fit window [ms] on the main-discharge clock",
    )
    args = parser.parse_args(argv)

    overlay = np.load(_overlay_path(args.es), allow_pickle=False)
    missing = _missing_overlay_keys(overlay, DECAY_FACE_KEYS)
    if missing:
        print(
            "SKIPPED: this overlay carries no "
            + ", ".join(missing)
            + " -- the second probe face and the geomean were promoted with a "
            "later overlay vintage, so there is no both-face figure to draw."
        )
        return 1

    result = load_result_hdf5(args.from_h5)
    decay_rows, window = compare_decay(
        result, overlay, window_ms=tuple(args.decay_window)
    )
    face_rows, skip_reason = compare_decay_faces(overlay, decay_rows, window_ms=window)
    if skip_reason is not None:
        print(f"SKIPPED: {skip_reason}.")
        return 1

    n_ports = len(face_rows)
    fig = plt.figure(figsize=(3.1 * n_ports, 7.4))
    grid = fig.add_gridspec(2, n_ports, height_ratios=(1.0, 0.9), hspace=0.52)
    for i, row in enumerate(face_rows):
        ax = fig.add_subplot(grid[0, i])
        _draw_port_panel(ax, row, overlay, result, window)
        if i == 0:
            ax.set_ylabel("J [A cm$^{-2}$]")
        if i == n_ports // 2:
            ax.set_xlabel("t [ms], main-discharge clock", fontsize=9)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=8,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.495),
        frameon=False,
    )
    _draw_bracket_panel(fig.add_subplot(grid[1, :]), face_rows)
    fig.suptitle(
        f"ES{args.es} afterglow, both Mach-probe faces at x = 0 -- "
        f"{Path(args.from_h5).name}, window {window[0]:.1f}-{window[1]:.1f} ms",
        fontsize=11,
    )
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
