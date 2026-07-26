"""Render ES1 validation figures (model vs measured) for a saved sim1d run.

Plotting-only campaign instrument for the R5 ES1 refit visual inspection. Does
not run LAPDSim1D or touch the shared scorer. Produces the required validation
panels with the adopted sigma_tot error bars:

  (1) discharge current I(t) and voltage V_dis(t) vs the ES overlay (SEM bands);
  (2) Te(z) and ne(z) at t=15 and 19 ms, model line vs measured port points
      (sigma_tot bars), with vertical dashed lines at the probe/port locations;
  (3) Isat(z) = n*sqrt(Te) (systematics-robust) model vs measured ports, plus
      nn(z)/Ti(z) profiles.

Model V_dis is the dt-integrated circuit voltage (the inductor's view, the
honest smooth trace). Times are on the main-discharge clock (t=0 at discharge
start), matching the overlay's *_time_ms axes.

Usage:
  python scripts/plot_es1_validation.py --from-h5 scripts/es1_r5_ts1840.h5 \
      --es 1 --out scripts/es1_r5_validation_ts1840.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cablp.solvers._sim1d import load_result_hdf5
from compare_sim1d_es1 import (
    _main_discharge_origin,
    _sigma_sys,
    TE_SYS_FRAC,
    TE_SYS_FLOOR_EV,
    N_CAL_FRAC,
)

SCRIPT_DIR = Path(__file__).resolve().parent
T_SLICES_MS = (15.0, 19.0)
SLICE_COLORS = {15.0: "tab:blue", 19.0: "tab:red"}


def _overlay_path(es):
    name = "es1_sim1d_overlay.npz" if es == 1 else f"es{es}_sim1d_overlay.npz"
    return SCRIPT_DIR / "data" / name


def _interp_port_slice(t_axis, values_2d, t_ms):
    """Interpolate each port's measured time series onto t_ms. values_2d[port,t]."""
    return np.array([np.interp(t_ms, t_axis, values_2d[p]) for p in range(values_2d.shape[0])])


def _n_sigma_tot(n_meas, te_meas, n_sem):
    """sigma_tot for density: SEM (+) sqrt((0.5 sigma_Te/Te)^2 + cal^2) propagated."""
    sig_te = TE_SYS_FRAC * np.abs(te_meas) + TE_SYS_FLOOR_EV
    frac = np.sqrt((0.5 * sig_te / np.maximum(te_meas, 1e-9)) ** 2 + N_CAL_FRAC ** 2)
    sys = np.abs(n_meas) * frac
    return np.sqrt(np.asarray(n_sem) ** 2 + sys ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-h5", required=True)
    ap.add_argument("--es", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r = load_result_hdf5(args.from_h5)
    ov = np.load(_overlay_path(args.es))
    diag = r.cathode_diagnostics

    origin = _main_discharge_origin(r)
    t_ms = (np.asarray(r.time, float) - origin) * 1e3
    z = np.asarray(r.z_cm, float)
    n = np.asarray(r.n, float)
    Te = np.asarray(r.Te, float)
    nn = np.asarray(r.nn, float)
    Ti = np.asarray(r.Ti, float)
    I = np.asarray(diag["source_I_tot"], float)
    # instantaneous V_dis from the dt-integrated circuit voltage
    Vint = np.asarray(diag.get("circuit_V_dis_dt_integral", np.zeros_like(I)), float)
    tsec = np.asarray(r.time, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        Vmid = np.diff(Vint) / np.diff(tsec)
    Vdis = np.concatenate([[Vmid[0]], Vmid]) if Vmid.size else np.zeros_like(I)

    zc = np.asarray(ov["z_cm"], float)
    ports = np.asarray(ov["port"])

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle(f"ES{args.es} validation: {Path(args.from_h5).name}", fontsize=12)

    # (1a) discharge current
    ax = axes[0, 0]
    dt_ms = np.asarray(ov["discharge_time_ms"], float)
    dI = np.asarray(ov["discharge_current_mean_a"], float)
    dIs = np.asarray(ov["discharge_current_sem_a"], float)
    ax.fill_between(dt_ms, dI - dIs, dI + dIs, color="gray", alpha=0.3, label="meas SEM")
    ax.plot(dt_ms, dI, "k-", lw=1, label="measured")
    ax.plot(t_ms, I, "b-", lw=1.3, label="model")
    ax.set_xlim(0, 22); ax.set_xlabel("t [ms] (main-discharge)"); ax.set_ylabel("I [A]")
    ax.set_title("discharge current"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1b) discharge voltage
    ax = axes[0, 1]
    dV = np.asarray(ov["discharge_voltage_positive_mean_v"], float)
    dVs = np.asarray(ov["discharge_voltage_sem_v"], float)
    ax.fill_between(dt_ms, dV - dVs, dV + dVs, color="gray", alpha=0.3)
    ax.plot(dt_ms, dV, "k-", lw=1, label="measured")
    ax.plot(t_ms, Vdis, "b-", lw=1.0, label="model V_dis")
    ax.set_xlim(0, 22); ax.set_ylim(0, max(220, np.nanmax(dV) * 1.2))
    ax.set_xlabel("t [ms]"); ax.set_ylabel("V_dis [V]")
    ax.set_title("discharge voltage"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    te_t = np.asarray(ov["te_time_ms"], float)
    te_m = np.asarray(ov["te_mean_ev"], float)
    te_s = np.asarray(ov["te_sem_ev"], float)
    de_t = np.asarray(ov["density_time_ms"], float)
    de_m = np.asarray(ov["density_mean_cm3"], float)
    de_s = np.asarray(ov["density_total_sem_cm3"], float)

    # (2a) Te(z) at slices, (2b) ne(z) at slices, (3a) Isat(z)
    ax_te, ax_ne, ax_is = axes[1, 0], axes[1, 1], axes[2, 0]
    for tsl in T_SLICES_MS:
        c = SLICE_COLORS[tsl]
        it = int(np.argmin(np.abs(t_ms - tsl)))
        ax_te.plot(z, Te[it], "-", color=c, lw=1.3, label=f"model {tsl:.0f} ms")
        ax_ne.plot(z, n[it], "-", color=c, lw=1.3, label=f"model {tsl:.0f} ms")
        # measured port points at this slice
        te_p = _interp_port_slice(te_t, te_m, tsl)
        te_ps = _interp_port_slice(te_t, te_s, tsl)
        ne_p = _interp_port_slice(de_t, de_m, tsl)
        ne_ps = _interp_port_slice(de_t, de_s, tsl)
        te_tot = np.sqrt(te_ps ** 2 + (_sigma_sys("Te", te_p)) ** 2)
        ne_tot = _n_sigma_tot(ne_p, te_p, ne_ps)
        ax_te.errorbar(zc, te_p, yerr=te_tot, fmt="o", color=c, ms=5, capsize=3)
        ax_ne.errorbar(zc, ne_p, yerr=ne_tot, fmt="o", color=c, ms=5, capsize=3)
        # Isat = n sqrt(Te)
        isat_model = n[it] * np.sqrt(np.maximum(Te[it], 0))
        isat_meas = ne_p * np.sqrt(np.maximum(te_p, 0))
        ax_is.plot(z, isat_model, "-", color=c, lw=1.3, label=f"model {tsl:.0f} ms")
        ax_is.plot(zc, isat_meas, "o", color=c, ms=5)
    for ax, ttl, yl in ((ax_te, "Te(z)", "Te [eV]"), (ax_ne, "ne(z)", "n [cm^-3]"),
                        (ax_is, "Isat(z) = n*sqrt(Te)", "n*sqrt(Te)")):
        for zp in zc:
            ax.axvline(zp, color="k", ls="--", lw=0.6, alpha=0.4)
        ax.set_xlabel("z [cm]"); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (3b) nn(z) and Ti(z) profiles
    ax = axes[2, 1]
    for tsl in T_SLICES_MS:
        c = SLICE_COLORS[tsl]
        it = int(np.argmin(np.abs(t_ms - tsl)))
        ax.plot(z, nn[it], "-", color=c, lw=1.3, label=f"nn {tsl:.0f} ms")
        ax.plot(z, Ti[it] * 1e12, ":", color=c, lw=1.0, label=f"Ti*1e12 {tsl:.0f} ms")
    for zp in zc:
        ax.axvline(zp, color="k", ls="--", lw=0.6, alpha=0.4)
    ax.set_yscale("log"); ax.set_xlabel("z [cm]"); ax.set_ylabel("nn [cm^-3] / Ti[eV]*1e12")
    ax.set_title("nn(z), Ti(z)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = args.out or str(Path(args.from_h5).with_suffix("")) + "_validation.png"
    fig.savefig(out, dpi=110)
    print(f"# wrote {out}")


if __name__ == "__main__":
    main()
