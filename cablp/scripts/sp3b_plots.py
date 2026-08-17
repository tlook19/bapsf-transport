#!/usr/bin/env python
"""Leg-B (sp3 shaped-nn0) verdict figures: four corners vs the anchored point.

Reads the five score files and result HDF5s in this directory; writes PNGs to
sp3b_plots/. Encoding: color = dt_foot (gray REF, blue 2 ms, orange 4.5 ms),
linestyle = kernel (solid diffusive, dashed ballistic) — identity is doubly
encoded, never color-alone.
"""
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = Path(__file__).resolve().parent
OUT = S / "sp3b_plots"
OUT.mkdir(exist_ok=True)

ARMS = {
    "REF (anchored point)": ("t22g5_change", "#6B7280", "-", 2.4),
    "2 ms, diffusive": ("sp3b_2ms_diff", "#2563EB", "-", 1.8),
    "2 ms, ballistic": ("sp3b_2ms_ball", "#2563EB", "--", 1.8),
    "4.5 ms, diffusive": ("sp3b_4p5ms_diff", "#EA580C", "-", 1.8),
    "4.5 ms, ballistic": ("sp3b_4p5ms_ball", "#EA580C", "--", 1.8),
}

ROW = re.compile(
    r"^\s*(Te|n|Isat)\s+(\d+)\s+(\d+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


def parse_scores(tag):
    rows = {}
    for line in (S / f"{tag}_scores.txt").read_text().splitlines():
        m = ROW.match(line)
        if m:
            f, port, z, model, meas, ratio, rms, dev = m.groups()
            rows[(f, int(z))] = dict(
                model=float(model), meas=float(meas), ratio=float(ratio),
                dev=float(dev),
            )
    if not rows:
        raise SystemExit(f"no score rows parsed from {tag}_scores.txt")
    return rows


scores = {label: parse_scores(tag) for label, (tag, *_,) in ARMS.items()}
PORTS = sorted({z for (f, z) in next(iter(scores.values()))})

# ---------------------------------------------------------------- fig 1: ratios
fields = [("n", r"$n_e$"), ("Te", r"$T_e$"), ("Isat", r"$I_{\rm sat}$")]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharex=True)
for ax, (f, label) in zip(axes, fields):
    ax.axhline(1.0, color="#111827", lw=1.0, zorder=1)
    ax.axhspan(0.95, 1.05, color="#11182714", zorder=0)
    for name, (tag, color, ls, lw) in ARMS.items():
        y = [scores[name][(f, z)]["ratio"] for z in PORTS]
        ax.plot(PORTS, y, color=color, ls=ls, lw=lw, marker="o", ms=5,
                mfc=color, mec="white", mew=0.8, zorder=3)
    ax.set_title(f"{label}  model / measured", fontsize=11)
    ax.set_xlabel("port z [cm]")
    ax.grid(color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_xticks(PORTS)
    ax.tick_params(labelsize=8.5)
axes[0].set_ylabel("ratio")
# direct labels on the leftmost panel, legend for all
handles = [
    plt.Line2D([], [], color=c, ls=ls, lw=lw, marker="o", ms=5, mec="white",
               label=n)
    for n, (t, c, ls, lw) in ARMS.items()
]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, 1.06))
fig.suptitle("Leg B — foot-inventory IC vs anchored point (ES1, plateau window)",
             y=1.14, fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "sp3b_port_ratios.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 2: mid-band nn lifetime
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for name, (tag, color, ls, lw) in ARMS.items():
    with h5py.File(S / f"{tag}_arm.h5") as fh:
        t = fh["time"][:] * 1e3
        z = fh["geometry/z_cm"][:]
        nn = fh["nn"][:]
    m = (z >= 700) & (z <= 1100)
    ax.plot(t, nn[:, m].mean(axis=1), color=color, ls=ls, lw=lw, label=name)
ax.axvspan(15.0, 19.5, color="#11182710", zorder=0)
ax.text(17.2, ax.get_ylim()[0], "", fontsize=8)
ax.set_yscale("log")
ax.set_xlabel("t [ms]")
ax.set_ylabel(r"mid-band $\langle n_n \rangle$ (700–1100 cm) [cm$^{-3}$]")
ax.set_title("Inventory lifetime: the foot reservoir feeds the mid-machine "
             "all discharge", fontsize=11)
ax.legend(frameon=False, fontsize=8.5)
ax.grid(color="#E5E7EB", lw=0.6, which="both", zorder=0)
ax.annotate("plateau\nwindow", xy=(17.2, ax.get_ylim()[1] * 0.55),
            ha="center", fontsize=8.5, color="#4B5563")
fig.tight_layout()
fig.savefig(OUT / "sp3b_midband_nn_lifetime.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------- fig 3: plateau axial profiles
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
for ax, (f, label, key, scale) in zip(
    axes,
    [("n", r"$n_e$ [cm$^{-3}$]", "n", 1.0),
     ("Te", r"$T_e$ [eV]", "Te", 1.0)],
):
    for name, (tag, color, ls, lw) in ARMS.items():
        with h5py.File(S / f"{tag}_arm.h5") as fh:
            t = fh["time"][:] * 1e3
            z = fh["geometry/z_cm"][:]
            act = fh["geometry/plasma_active"][:].astype(bool)
            prof = fh[key][:]
        w = (t >= 15.0) & (t <= 19.5)
        ax.plot(z[act], prof[w][:, act].mean(axis=0), color=color, ls=ls,
                lw=lw)
    meas_z = PORTS
    meas = [scores["REF (anchored point)"][(f, z)]["meas"] for z in PORTS]
    ax.plot(meas_z, meas, "s", color="#111827", ms=6, mfc="white", mew=1.4,
            label="measured (ES1 ports)", zorder=5)
    ax.set_xlabel("z [cm]")
    ax.set_ylabel(label)
    ax.grid(color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_xlim(0, 2000)
axes[0].legend(handles=handles + [
    plt.Line2D([], [], color="#111827", marker="s", ls="none", mfc="white",
               mew=1.4, label="measured (ES1 ports)")],
    frameon=False, fontsize=8)
fig.suptitle("Plateau-window axial profiles (15–19.5 ms mean)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "sp3b_plateau_profiles.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("wrote", *[p.name for p in sorted(OUT.glob("sp3b_*.png"))])
