"""Demo figure for the discharge measured-spread band (read-only).

Renders the ES discharge current and voltage overlay against a spread-carrying
overlay NPZ, at full range and zoomed on the breakdown transition, so the
+/-sd shot envelope is legible where it is large. The zoom is the point of the
band: the ensemble mean is smooth through breakdown because shot-to-shot
breakdown-timing jitter is averaged across it, and the sd is what that jitter
actually looks like.

This is a demo artifact, not a campaign instrument. The production renderer is
`plot_es1_validation.py`, which draws the same bands in its panel (1) row.

Usage:
  python scripts/ovshade_demo_figure.py --from-h5 RUN.h5 --overlay CAND.npz \
      --es 1 --out scripts/ovshade_es1_discharge_demo.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cablp.solvers._sim1d import load_result_hdf5
from compare_sim1d_es1 import _main_discharge_origin

ZOOM_MS = (0.0, 2.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-h5", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--es", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = load_result_hdf5(args.from_h5)
    ov = np.load(args.overlay)
    diag = r.cathode_diagnostics

    origin = _main_discharge_origin(r)
    t_ms = (np.asarray(r.time, float) - origin) * 1e3
    I = np.asarray(diag["source_I_tot"], float)
    Vint = np.asarray(diag.get("circuit_V_dis_dt_integral", np.zeros_like(I)), float)
    tsec = np.asarray(r.time, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        Vmid = np.diff(Vint) / np.diff(tsec)
    Vdis = np.concatenate([[Vmid[0]], Vmid]) if Vmid.size else np.zeros_like(I)

    dt_ms = np.asarray(ov["discharge_time_ms"], float)
    n_tr = int(ov["discharge_n_traces"])
    channels = (
        ("current", "discharge_current_mean_a", "discharge_current_sd_a",
         "discharge_current_sem_a", "I [A]", I),
        ("voltage", "discharge_voltage_positive_mean_v", "discharge_voltage_sd_v",
         "discharge_voltage_sem_v", "V_dis [V]", Vdis),
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, (name, k_mean, k_sd, k_sem, ylab, model) in enumerate(channels):
        mean = np.asarray(ov[k_mean], float)
        sem = np.asarray(ov[k_sem], float)
        sd = np.asarray(ov[k_sd], float) if k_sd in ov else None
        for col, xlim in enumerate((None, ZOOM_MS)):
            ax = axes[row, col]
            if sd is not None:
                ax.fill_between(dt_ms, mean - sd, mean + sd, color="tab:orange",
                                alpha=0.22, lw=0,
                                label=f"measured +/-sd ({n_tr} shots)")
            ax.fill_between(dt_ms, mean - sem, mean + sem, color="gray",
                            alpha=0.45, lw=0, label="measured +/-SEM")
            ax.plot(dt_ms, mean, "k-", lw=1.0, label="measured mean")
            ax.plot(t_ms, model, "b-", lw=1.2, label="model")
            ax.set_xlim(0, 22) if xlim is None else ax.set_xlim(*xlim)
            ax.set_xlabel("t [ms] (main-discharge)")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            span = "full discharge" if xlim is None else "breakdown transition"
            ax.set_title(f"ES{args.es} discharge {name} - {span}")
            if col == 0:
                ax.legend(fontsize=8, loc="lower right")

    # Zoom the current panel onto the ensemble, not the model, so the band is
    # not flattened when the model sits far from the measured trace.
    zoom_sel = (dt_ms >= ZOOM_MS[0]) & (dt_ms <= ZOOM_MS[1])
    cmean = np.asarray(ov["discharge_current_mean_a"], float)[zoom_sel]
    csd = np.asarray(ov["discharge_current_sd_a"], float)[zoom_sel]
    axes[0, 1].set_ylim(min(0.0, (cmean - csd).min()) * 1.05,
                        (cmean + csd).max() * 1.08)

    fig.suptitle(
        f"ES{args.es} discharge overlay with the measured shot-to-shot band  |  "
        f"{Path(args.from_h5).name}  |  {Path(args.overlay).name}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140)
    print(f"# wrote {args.out}")

    peak = int(np.argmax(csd))
    print(f"# current sd over {ZOOM_MS[0]}-{ZOOM_MS[1]} ms: max {csd.max():.1f} A "
          f"at t={dt_ms[zoom_sel][peak]:.3f} ms, "
          f"{100 * csd.max() / cmean[peak]:.1f}% of the mean there")


if __name__ == "__main__":
    main()
