"""Render ES1 validation figures (model vs measured) for a saved sim1d run.

Plotting-only campaign instrument for the R5 ES1 refit visual inspection. Does
not run LAPDSim1D or touch the shared scorer. Produces the required validation
panels with the adopted sigma_tot error bars:

  (1) discharge current I(t) and voltage V_dis(t) vs the ES overlay. Where the
      overlay carries the measured ensemble spread (schema v8+), the wide band
      is the shot-to-shot +/-sd ENVELOPE and the narrow one the +/-SEM of the
      mean; on older overlays only the SEM band is drawn, unchanged;
  (2) Te(z) and ne(z) at t=15 and 19 ms, model line vs measured port points
      (sigma_tot bars), with vertical dashed lines at the probe/port locations;
  (3) Isat(z) = n*sqrt(Te) (systematics-robust) model vs measured ports, plus
      nn(z)/Ti(z) profiles;
  (4) per-port Te(t) and ne(t) time series, model line vs measured mean with a
      sigma_tot band, one colour per ES port.

THE COMPARAND. The measured points are the COLUMN convention -- the whole
column's inventory expressed on the model's tube (``density_column_cm3``, and
``te_column_ev``, the density-weighted mean over the same extent) -- drawn as
FILLED circles with their own sems, and carrying the band and the dashed trace
in the time-series panels. Plasma outside the tube got there by cross-field
transport a 1D model does not represent, which is why the column's plasma is
the comparand and not the part of it inside the tube. The other two
conventions are drawn beside it as HOLLOW markers, each with its own style --
the FLUX-TUBE area mean (``te_ftavg_ev``, ``density_ftavg_cm3``) as hollow
circles, the legacy CORE-BAND line cut (``te_mean_ev``, ``density_mean_cm3``)
as hollow squares -- and as thin dotted traces in the time-series panels, so
the three are read together and none is quoted unlabelled.

A column point whose ``te_column_prior_weight`` exceeds one half is MAJORITY
PRIOR: more than half the density-weighted quadrature behind it is the repo's
scrape-off-layer prior rather than a measurement of that port. Those points
are drawn with the '~' marker, the same mark the scorer's table gives them,
and are ineligible for a bin verdict there. On an overlay vintage carrying no
column fields the figure falls back to the flux-tube series as the filled
primary, and to the core band alone if it carries neither, and says which in
the legend.

Model V_dis is the dt-integrated circuit voltage (the inductor's view, the
honest smooth trace). Times are on the main-discharge clock (t=0 at discharge
start), matching the overlay's *_time_ms axes.

The z-profile panels draw the plasma quantities (Te, ne, Isat, Ti) only over
plasma-live cells, blanking the plasma-dead roles behind the cathode face;
nn is drawn over the full domain, where the plenum reservoir is physical.

Usage:
  python scripts/score/plot_es1_validation.py --from-h5 scripts/es1_r5_ts1840.h5 \
      --es 1 --out scripts/es1_r5_validation_ts1840.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cablp.solvers._sim1d import load_result_hdf5
from cablp.solvers._sim1d.core.geometry import PLASMA_DEAD_ROLES
# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from compare_sim1d_es1 import (
    _main_discharge_origin,
    _sigma_sys,
    COLUMN_PRIOR_MAJORITY_WEIGHT,
    COLUMN_PRIOR_WEIGHT_KEY,
    TE_SYS_FRAC,
    TE_SYS_FLOOR_EV,
    N_CAL_FRAC,
)

SCRIPT_DIR = Path(__file__).resolve().parents[1]
T_SLICES_MS = (15.0, 19.0)
SLICE_COLORS = {15.0: "tab:blue", 19.0: "tab:red"}

#: Marker for a PRIMARY (filled) measured point whose column Te average is
#: MAJORITY PRIOR -- more than half of its density-weighted quadrature is the
#: repo's scrape-off-layer prior rather than a measurement of that port. The
#: mathtext tilde is the same mark the scorer's table gives such a row, so the
#: figure and the table say the same thing with the same character.
PRIOR_MAJORITY_MARKER = r"$\sim$"
MEASURED_MARKER = "o"

#: Hollow marker styles for the two non-primary conventions, so the three are
#: distinguishable without reading the legend twice.
FTAVG_MARKER = "o"
CORE_MARKER = "s"


def _overlay_path(es):
    name = "es1_sim1d_overlay.npz" if es == 1 else f"es{es}_sim1d_overlay.npz"
    return SCRIPT_DIR / "data" / name


def _spread_band(ov, key, mean):
    """Return the measured ensemble sd for `key`, or None on older overlays.

    Presence-gated: overlays written before the discharge-spread export carry
    no sd field, and on those the discharge panels keep their SEM-only
    rendering exactly as before. A length that does not match the mean trace
    raises rather than shading the band against the wrong samples.
    """
    if key not in ov:
        return None
    sd = np.asarray(ov[key], float)
    if sd.shape != mean.shape:
        raise ValueError(
            f"overlay {key} has shape {sd.shape}, expected the mean trace's "
            f"{mean.shape}; a mismatched length would shade the band against "
            "the wrong samples"
        )
    return sd


def _plasma_live_mask(result):
    """Return the per-cell plasma-live boolean, or None on results without roles.

    Membership in ``PLASMA_DEAD_ROLES`` is the authoritative test, so twin and
    end wall geometries -- whose dead cells are not a contiguous z<0 block --
    stay correct. Results saved before ``cell_role`` was written carry no roles
    and are left unmasked.
    """
    roles = np.asarray(getattr(result, "cell_role", ()), dtype=object)
    if roles.size == 0:
        return None
    return np.array([str(role) not in PLASMA_DEAD_ROLES for role in roles], dtype=bool)


def _blank_dead(values, live):
    """Return `values` with the plasma-dead cells replaced by NaN (gaps, not lines)."""
    if live is None:
        return values
    return np.where(live, np.asarray(values, float), np.nan)


def _interp_port_slice(t_axis, values_2d, t_ms):
    """Interpolate each port's measured time series onto t_ms. values_2d[port,t]."""
    return np.array([np.interp(t_ms, t_axis, values_2d[p]) for p in range(values_2d.shape[0])])


def _n_sigma_tot(n_meas, te_meas, n_sem):
    """sigma_tot for density: SEM (+) sqrt((0.5 sigma_Te/Te)^2 + cal^2) propagated."""
    sig_te = TE_SYS_FRAC * np.abs(te_meas) + TE_SYS_FLOOR_EV
    frac = np.sqrt((0.5 * sig_te / np.maximum(te_meas, 1e-9)) ** 2 + N_CAL_FRAC ** 2)
    sys = np.abs(n_meas) * frac
    return np.sqrt(np.asarray(n_sem) ** 2 + sys ** 2)


#: The three measured radial conventions, in the order they are preferred as
#: the FILLED primary series. Each entry is
#: ``(name, label, te_time_key, te_key, te_sem_key, density_key)``; the
#: density SEM is convention-specific and is built by ``_density_sem`` below.
CONVENTIONS = (
    ("column", "column",
     "te_time_ms", "te_column_ev", "te_column_sem_ev", "density_column_cm3"),
    ("flux-tube", "flux-tube",
     "te_ftavg_time_ms", "te_ftavg_ev", "te_ftavg_sem_ev",
     "density_ftavg_cm3"),
    ("core-band", "core-band",
     "te_time_ms", "te_mean_ev", "te_sem_ev", "density_mean_cm3"),
)

#: Hollow marker per non-primary convention.
CONVENTION_MARKERS = {
    "column": MEASURED_MARKER,
    "flux-tube": FTAVG_MARKER,
    "core-band": CORE_MARKER,
}


def _density_sem(ov, name, de_m):
    """Return the density SEM for one convention, exactly as the scorer does.

    The column product exports its OWN SEM (``density_column_sem_cm3``); the
    core band exports ``density_total_sem_cm3``; the flux tube exports
    neither, so the core-band FRACTIONAL error is carried across the
    convention, which is what the scorer does for it and what makes the two
    figures agree.
    """
    if name == "column" and "density_column_sem_cm3" in ov:
        return np.asarray(ov["density_column_sem_cm3"], float)
    core_sem = np.asarray(ov["density_total_sem_cm3"], float)
    if name == "core-band":
        return core_sem
    legacy = np.asarray(ov["density_mean_cm3"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return core_sem * np.where(legacy != 0.0, de_m / legacy, np.nan)


def _convention_series(ov, name):
    """Return one convention's measured series, or ``None`` on an overlay
    vintage that does not carry it.

    Presence-gated on the whole set at once: a vintage carrying some of a
    convention's fields but not others would be drawn half under one
    convention and half under another, which is exactly the mislabelling the
    three-convention rendering exists to prevent.
    """
    spec = next(c for c in CONVENTIONS if c[0] == name)
    _, label, te_t_key, te_key, te_sem_key, de_key = spec
    keys = (te_t_key, te_key, te_sem_key, de_key, "density_total_sem_cm3",
            "density_mean_cm3", "density_time_ms")
    if any(k not in ov for k in keys):
        return None
    de_m = np.asarray(ov[de_key], float)
    return {
        "name": name,
        "label": label,
        "te_t": np.asarray(ov[te_t_key], float),
        "te_m": np.asarray(ov[te_key], float),
        "te_s": np.asarray(ov[te_sem_key], float),
        "de_t": np.asarray(ov["density_time_ms"], float),
        "de_m": de_m,
        "de_s": _density_sem(ov, name, de_m),
    }


def _prior_majority(ov, n_ports):
    """Return the per-port MAJORITY-PRIOR flag for the column T_e rows.

    True where the port's mean ``te_column_prior_weight`` exceeds
    ``COLUMN_PRIOR_MAJORITY_WEIGHT``: more than half the row's
    density-weighted quadrature is carried by cells beyond that port's trust
    radius, so the point is substantially a statement about the repo's
    scrape-off-layer prior. The mean is taken over the PRODUCT's whole time
    base -- a figure has no scored window -- so it can differ from the
    scorer's row-level share when the model under-covers the measurement.
    All-False on a vintage carrying no such field.
    """
    if COLUMN_PRIOR_WEIGHT_KEY not in ov:
        return np.zeros(n_ports, dtype=bool)
    w = np.asarray(ov[COLUMN_PRIOR_WEIGHT_KEY], float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(w, axis=1) > COLUMN_PRIOR_MAJORITY_WEIGHT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-h5", required=True)
    ap.add_argument("--es", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--overlay",
        default=None,
        help="overlay NPZ to render against; default is the promoted "
             "data/es{N}_sim1d_overlay.npz",
    )
    args = ap.parse_args()

    r = load_result_hdf5(args.from_h5)
    ov = np.load(args.overlay or _overlay_path(args.es))
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
    live = _plasma_live_mask(r)

    fig, axes = plt.subplots(4, 2, figsize=(13, 17))
    fig.suptitle(f"ES{args.es} validation: {Path(args.from_h5).name}", fontsize=12)

    # (1a) discharge current
    ax = axes[0, 0]
    dt_ms = np.asarray(ov["discharge_time_ms"], float)
    dI = np.asarray(ov["discharge_current_mean_a"], float)
    dIs = np.asarray(ov["discharge_current_sem_a"], float)
    dIsd = _spread_band(ov, "discharge_current_sd_a", dI)
    if dIsd is not None:
        ax.fill_between(dt_ms, dI - dIsd, dI + dIsd, color="tab:orange", alpha=0.22,
                        lw=0, label="meas shot sd")
    ax.fill_between(dt_ms, dI - dIs, dI + dIs, color="gray", alpha=0.3, label="meas SEM")
    ax.plot(dt_ms, dI, "k-", lw=1, label="measured")
    ax.plot(t_ms, I, "b-", lw=1.3, label="model")
    ax.set_xlim(0, 22); ax.set_xlabel("t [ms] (main-discharge)"); ax.set_ylabel("I [A]")
    ax.set_title("discharge current"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1b) discharge voltage
    ax = axes[0, 1]
    dV = np.asarray(ov["discharge_voltage_positive_mean_v"], float)
    dVs = np.asarray(ov["discharge_voltage_sem_v"], float)
    dVsd = _spread_band(ov, "discharge_voltage_sd_v", dV)
    if dVsd is not None:
        ax.fill_between(dt_ms, dV - dVsd, dV + dVsd, color="tab:orange", alpha=0.22,
                        lw=0, label="meas shot sd")
    ax.fill_between(dt_ms, dV - dVs, dV + dVs, color="gray", alpha=0.3)
    ax.plot(dt_ms, dV, "k-", lw=1, label="measured")
    ax.plot(t_ms, Vdis, "b-", lw=1.0, label="model V_dis")
    ax.set_xlim(0, 22); ax.set_ylim(0, max(220, np.nanmax(dV) * 1.2))
    ax.set_xlabel("t [ms]"); ax.set_ylabel("V_dis [V]")
    ax.set_title("discharge voltage"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # The three measured conventions, in preference order. The first one this
    # overlay carries is the FILLED primary; the rest are drawn hollow. An
    # older vintage that carries only the core band therefore renders it
    # filled rather than as though its primary convention had gone missing.
    series = [s for s in (_convention_series(ov, name)
                          for name, *_ in CONVENTIONS) if s is not None]
    if not series:
        raise ValueError(
            "overlay carries none of the three measured radial conventions "
            f"({', '.join(name for name, *_ in CONVENTIONS)}); there is "
            "nothing to plot the model against"
        )
    primary, alts = series[0], series[1:]
    prior_major = _prior_majority(ov, primary["te_m"].shape[0])

    # (2a) Te(z) at slices, (2b) ne(z) at slices, (3a) Isat(z)
    ax_te, ax_ne, ax_is = axes[1, 0], axes[1, 1], axes[2, 0]
    for tsl in T_SLICES_MS:
        c = SLICE_COLORS[tsl]
        it = int(np.argmin(np.abs(t_ms - tsl)))
        ax_te.plot(z, _blank_dead(Te[it], live), "-", color=c, lw=1.3, label=f"model {tsl:.0f} ms")
        ax_ne.plot(z, _blank_dead(n[it], live), "-", color=c, lw=1.3, label=f"model {tsl:.0f} ms")
        isat_model = _blank_dead(n[it] * np.sqrt(np.maximum(Te[it], 0)), live)
        ax_is.plot(z, isat_model, "-", color=c, lw=1.3, label=f"model {tsl:.0f} ms")

        def _slice(s, _tsl=tsl):
            """This convention's (Te, sigma_tot_Te, n, sigma_tot_n) at a slice."""
            te_p = _interp_port_slice(s["te_t"], s["te_m"], _tsl)
            te_ps = _interp_port_slice(s["te_t"], s["te_s"], _tsl)
            ne_p = _interp_port_slice(s["de_t"], s["de_m"], _tsl)
            ne_ps = _interp_port_slice(s["de_t"], s["de_s"], _tsl)
            return (
                te_p,
                np.sqrt(te_ps ** 2 + (_sigma_sys("Te", te_p)) ** 2),
                ne_p,
                _n_sigma_tot(ne_p, te_p, ne_ps),
            )

        # The non-primary conventions first, HOLLOW, each with its own marker.
        for s in alts:
            marker = CONVENTION_MARKERS[s["name"]]
            te_p, te_tot, ne_p, ne_tot = _slice(s)
            ax_te.errorbar(zc, te_p, yerr=te_tot, marker=marker, ls="none",
                           mfc="none", color=c, ms=6, capsize=2, lw=0.8,
                           alpha=0.75)
            ax_ne.errorbar(zc, ne_p, yerr=ne_tot, marker=marker, ls="none",
                           mfc="none", color=c, ms=6, capsize=2, lw=0.8,
                           alpha=0.75)
            ax_is.plot(zc, ne_p * np.sqrt(np.maximum(te_p, 0)), marker=marker,
                       ls="none", mfc="none", color=c, ms=6, alpha=0.75)
        # The PRIMARY, filled, with its own sems; majority-prior ports carry
        # the '~' marker the scorer's table gives them.
        te_p, te_tot, ne_p, ne_tot = _slice(primary)
        pisat = ne_p * np.sqrt(np.maximum(te_p, 0))
        for mask, marker, ms in (
            (~prior_major, MEASURED_MARKER, 5),
            (prior_major, PRIOR_MAJORITY_MARKER, 10),
        ):
            if not np.any(mask):
                continue
            # marker=/ls= rather than fmt=: the majority-prior mark is a
            # mathtext marker, which a format string cannot express.
            ax_te.errorbar(zc[mask], te_p[mask], yerr=te_tot[mask],
                           marker=marker, ls="none", color=c, ms=ms,
                           capsize=3)
            ax_ne.errorbar(zc[mask], ne_p[mask], yerr=ne_tot[mask],
                           marker=marker, ls="none", color=c, ms=ms,
                           capsize=3)
            ax_is.plot(zc[mask], pisat[mask], marker=marker, ls="none",
                       color=c, ms=ms)
    # Kept SHORT and wrapped: the legend title sits inside the panel, and a
    # one-line sentence here runs off the axes at every figure width.
    legend_note = (
        f"meas: filled={primary['label']}\n"
        + ", ".join(
            f"hollow {CONVENTION_MARKERS[s['name']]}={s['label']}"
            for s in alts
        )
        + (", '~'=majority prior" if np.any(prior_major) else "")
    )
    for ax, ttl, yl in ((ax_te, "Te(z)", "Te [eV]"), (ax_ne, "ne(z)", "n [cm^-3]"),
                        (ax_is, "Isat(z) = n*sqrt(Te)", "n*sqrt(Te)")):
        for zp in zc:
            ax.axvline(zp, color="k", ls="--", lw=0.6, alpha=0.4)
        ax.set_xlabel("z [cm]"); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.legend(fontsize=7, title=legend_note, title_fontsize=6)
        ax.grid(alpha=0.3)

    # (3b) nn(z) and Ti(z) profiles
    ax = axes[2, 1]
    for tsl in T_SLICES_MS:
        c = SLICE_COLORS[tsl]
        it = int(np.argmin(np.abs(t_ms - tsl)))
        ax.plot(z, nn[it], "-", color=c, lw=1.3, label=f"nn {tsl:.0f} ms")
        ax.plot(z, _blank_dead(Ti[it] * 1e12, live), ":", color=c, lw=1.0,
                label=f"Ti*1e12 {tsl:.0f} ms")
    for zp in zc:
        ax.axvline(zp, color="k", ls="--", lw=0.6, alpha=0.4)
    ax.set_yscale("log"); ax.set_xlabel("z [cm]"); ax.set_ylabel("nn [cm^-3] / Ti[eV]*1e12")
    ax.set_title("nn(z), Ti(z)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (4) per-port time series: Te(t) and ne(t), model line vs measured
    # mean with a sigma_tot band, one colour per ES port.
    ax_tet, ax_net = axes[3, 0], axes[3, 1]
    port_colors = plt.cm.viridis(np.linspace(0, 0.9, len(zc)))
    for p, (zp, port) in enumerate(zip(zc, ports)):
        c = port_colors[p]
        iz = int(np.argmin(np.abs(z - zp)))
        lbl = f"p{port} z{zp:.0f}"
        # The PRIMARY series carries the band and the dashed trace, on its own
        # Te clock; the other conventions are drawn thin and dotted beside it
        # for comparison and carry no band.
        pte_sys = TE_SYS_FRAC * np.abs(primary["te_m"][p]) + TE_SYS_FLOOR_EV
        pte_band = np.sqrt(primary["te_s"][p] ** 2 + pte_sys ** 2)
        ax_tet.fill_between(primary["te_t"], primary["te_m"][p] - pte_band,
                            primary["te_m"][p] + pte_band, color=c, alpha=0.15)
        ax_tet.plot(primary["te_t"], primary["te_m"][p], "--", color=c, lw=1.0)
        pte_on_de = np.interp(primary["de_t"], primary["te_t"],
                              primary["te_m"][p])
        pne_band = _n_sigma_tot(primary["de_m"][p], pte_on_de,
                                primary["de_s"][p])
        ax_net.fill_between(primary["de_t"], primary["de_m"][p] - pne_band,
                            primary["de_m"][p] + pne_band, color=c, alpha=0.15)
        ax_net.plot(primary["de_t"], primary["de_m"][p], "--", color=c, lw=1.0)
        for s, style in zip(alts, (":", "-.")):
            ax_tet.plot(s["te_t"], s["te_m"][p], style, color=c, lw=0.7,
                        alpha=0.7)
            ax_net.plot(s["de_t"], s["de_m"][p], style, color=c, lw=0.7,
                        alpha=0.7)
        ax_tet.plot(t_ms, Te[:, iz], "-", color=c, lw=1.4, label=lbl)
        ax_net.plot(t_ms, n[:, iz], "-", color=c, lw=1.4, label=lbl)
    series_note = (
        f"solid=model, dashed=meas {primary['label']} (+band)\n"
        + ", ".join(
            f"{style}=meas {s['label']}"
            for s, style in zip(alts, ("dotted", "dash-dot"))
        )
    )
    for ax, ttl, yl in ((ax_tet, "Te(t) per port", "Te [eV]"),
                        (ax_net, "ne(t) per port", "n [cm^-3]")):
        ax.set_xlim(0, 22); ax.set_xlabel("t [ms] (main-discharge)")
        ax.set_ylabel(yl); ax.set_title(ttl)
        ax.legend(fontsize=7, ncol=2, title=series_note, title_fontsize=6)
        ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = args.out or str(Path(args.from_h5).with_suffix("")) + "_validation.png"
    fig.savefig(out, dpi=110)
    print(f"# wrote {out}")


if __name__ == "__main__":
    main()
