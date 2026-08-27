#!/usr/bin/env python
"""Leg-C verdict figure: u_eff and ion drift port profiles vs the measured flow."""
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = Path(__file__).resolve().parent
PORTS = {"p11": 470.1, "p21": 789.5, "p29": 1045.2, "p41": 1428.5, "p50": 1716.1}
# SUPERSEDED 2026-08-21: the unified helium mass is cablp.constants
# .m_He_cgs = 6.6464790809e-24 g (Ar(4He)*u, CODATA 2022). The literal
# below is 707 ppm low and is left AS A RECORD of what this dated script ran.
M_HE = 4 * 1.6605e-24

ARMS = {
    "background (foot IC)": ("sp3b_4p5ms_diff", "#6B7280", "-"),
    "W1: wind two-zone": ("lcw1", "#2563EB", "-"),
    "C0: hot-birth alone": ("lcc0", "#EA580C", "--"),
    "C1: composite": ("lcc1", "#EA580C", "-"),
}


def read(tag):
    with h5py.File(S / f"{tag}_arm.h5") as f:
        t = f["time"][:]
        z = f["geometry/z_cm"][:]
        nn = f["nn"][:]
        M_n = f["M_n"][:]
        u_i = f["u"][:]
        hnf = f["hot_n_flight"][:] if "hot_n_flight" in f else np.zeros_like(nn)
        hfz = f["hot_flux_z"][:] if "hot_flux_z" in f else np.zeros_like(nn)
    w = (t >= 0.015) & (t <= 0.0195)
    ue, ui = [], []
    for zp in PORTS.values():
        i = int(np.argmin(np.abs(z - zp)))
        nn_m = nn[w, i].mean()
        hnf_m = hnf[w, i].mean()
        ue.append(((M_n[w, i] / M_HE + hfz[w, i]).mean() / (nn_m + hnf_m)) / 1e5)
        ui.append(u_i[w, i].mean() / 1e5)
    return np.array(ue), np.array(ui)


zs = np.array(list(PORTS.values()))
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
for name, (tag, color, ls) in ARMS.items():
    ue, ui = read(tag)
    axes[0].plot(zs, ue, color=color, ls=ls, lw=1.9, marker="o", ms=5,
                 mec="white", mew=0.8, label=name)
    axes[1].plot(zs, ui, color=color, ls=ls, lw=1.9, marker="o", ms=5,
                 mec="white", mew=0.8, label=name)
for ax in axes:
    ax.plot([1428.5, 1716.1], [1.8, 5.4], "s", color="#111827", ms=7,
            mfc="white", mew=1.5, label="measured flow (p41/p50)")
    ax.annotate("", xy=(1716.1, 5.4), xytext=(1428.5, 1.8),
                arrowprops=dict(arrowstyle="->", color="#111827", lw=1.2))
    ax.set_xlabel("port z [cm]")
    ax.grid(color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_xticks(zs)
    ax.tick_params(labelsize=8.5)
axes[0].set_ylabel("u_eff (neutral, density-weighted) [km/s]")
axes[0].set_title("Neutral flow: every arm is flat where the machine rises",
                  fontsize=10.5)
axes[1].set_ylabel("ion drift u$_i$ [km/s]")
axes[1].set_title("Ion drift: the model FALLS toward the end — the inherited shape",
                  fontsize=10.5)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Leg C — the A/B verdict: shape null at every arm; "
             "the remainder is the ion flow profile", fontsize=12)
fig.tight_layout()
out = S / "sp3b_plots" / "lc_flow_verdict.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("wrote", out)
