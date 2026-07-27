"""Publication/slide power-loss and efficiency figures on the R3.2 closed ledger.

Reads a saved LAPDSim1D HDF5 result and renders two figures from the
current-resolved circuit power ledger (R3.2 / audit A16, see
``~/bapsf/docs/manuscripts/evidence/circuit_power_balance_r32.md``):

* ``<stem>_power``      -- the P_load ledger as a signed stacked area (every
  field-work line the bank funds), the measured closure residual
  |P_load - sum(ledger)| on its own axis, and the plasma-book volumetric
  loss channels (radiation, ionization cost, sheath/surface sinks).
* ``<stem>_efficiency`` -- time-resolved fraction of P_load per channel
  group, plus the energy-integrated share of each group over the
  main-discharge window (the per-run scalar summary, also printed as
  ``key=value`` lines for later per-ES-set comparison).

The ledger is RECONSTRUCTED from saved scalars only -- nothing is assumed:

    P_load = I_tot*V_b = I_tot*phi_c + I_tot*V_p - I_tot*phi_a
    cathode per species (Kirchhoff I_tot = I_eth* + I_i - I_e_ret):
      beam emission   I_eth**phi_c = P_prim + P_beam_bypass
      ion collection  I_i*phi_c
      returning e-   -I_e_ret*phi_c,  I_e_ret = I_eth* + I_i - I_tot
    gap:              P_ohmic (= I_tot*V_p)
    anode (net-current ladder value; per-species anode is R4):  -I_tot*phi_a

so the closure P_load == sum(lines) is verified and its residual displayed,
not assumed. In the bracket-capped regime V_b is clamped off the ladder and
the residual is nonzero BY DESIGN (reported, never hidden).

Styling comes from the house ``figstyle`` module (journal profile: Okabe-Ito,
column sizing, vector PDF; slide profile via ``--profile slide``). Each figure
is written as PDF (vector) + PNG preview. No titles are baked into the
figures -- context belongs in the caption or slide title.

Usage:
    python scripts/plot_sim1d_power.py --from-h5 scripts/RUN.h5
        [--output-dir DIR] [--profile journal|slide]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# House style: the central figstyle module used by the slides/manuscript
# templates. Required -- this script exists to produce house-style output.
FIGSTYLE_DIR = Path.home() / "bapsf" / "docs" / "figures"
sys.path.insert(0, str(FIGSTYLE_DIR))
try:
    import figstyle as fs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"house figstyle module not found in {FIGSTYLE_DIR}: {exc}"
    ) from exc

# =============================================================================
# CHANNEL GROUPING (delivered vs loss) -- the "useful power" definition.
#
# This dict is THE configuration point: the efficiency figure and the printed
# per-run scalars are computed from it, never hardcoded in plot logic. Each
# group lists reconstructed ledger channels (defined in build_ledger) and a
# role:
#   "delivered" -- power entering the plasma heating book (the useful power)
#   "loss"      -- power the load consumes that never heats the plasma
#   "recovered" -- field work returned to the circuit (negative lines; not
#                  dissipated in the load at all)
# The groups partition the ledger exactly: sum over all groups == P_load.
# =============================================================================
CHANNEL_GROUPS = {
    "delivered to plasma": {
        "channels": ("P_prim", "P_ohmic"),
        "role": "delivered",
    },
    "cathode surface (ion sheath fall)": {
        "channels": ("P_cathode_ion_phi",),
        "role": "loss",
    },
    "beam bypass (uncoupled primaries)": {
        "channels": ("P_beam_bypass",),
        "role": "loss",
    },
    "returned to circuit (anode + returning e-)": {
        "channels": ("P_anode_field", "P_electron_return"),
        "role": "recovered",
    },
}

# Ledger channel display order, labels, and colors (Okabe-Ito, print-safe).
LEDGER_STYLE = {
    "P_prim": ("beam primaries into column", "#0072B2"),
    "P_ohmic": ("bulk ohmic", "#56B4E9"),
    "P_cathode_ion_phi": ("cathode ion sheath fall", "#D55E00"),
    "P_beam_bypass": ("beam bypass", "#E69F00"),
    "P_anode_field": ("anode field work (net)", "#009E73"),
    "P_electron_return": ("returning-electron recovery", "#CC79A7"),
}

GROUP_COLORS = {
    "delivered to plasma": "#0072B2",
    "cathode surface (ion sheath fall)": "#D55E00",
    "beam bypass (uncoupled primaries)": "#E69F00",
    "returned to circuit (anode + returning e-)": "#009E73",
}

# Plasma-book volumetric loss channels: volume-integrated (electron + ion)
# energy terms [W/cm^3 * cm^3]. All are sinks in this model; magnitudes are
# drawn on a log axis and channels that never exceed PLASMA_FLOOR_W are
# dropped from the legend as inactive.
PLASMA_LOSS_CHANNELS = {
    "e-i cooling (line radiation)": ("electron_ion_cooling",),
    "e-n cooling": ("electron_neutral_cooling",),
    "ionization cost": ("ionization_energy_cost",),
    "recombination radiation": ("recombination_rad_loss", "recombination_3b_loss"),
    "beam excitation radiation": ("beam_excitation_radiation",),
    "cathode/anode sheath (plasma book)": ("cathode_surface_loss",),
    "anode collection": ("anode_collection",),
    "end/surface loss": ("surface_loss", "characteristic_boundary"),
}
PLASMA_FLOOR_W = 10.0  # legend/axis floor for the loss panel [W]

# Fractions are blanked where P_load falls below this fraction of its
# main-discharge median (the ratio is meaningless with no drive).
EFF_PLOAD_FLOOR_FRAC = 0.05

SIDES = ("source", "end")  # cathode boundary prefixes; inactive sides are NaN


def _boxcar(y, width):
    """Centered boxcar mean over `width` samples (NaN-safe); width<=1 -> raw."""
    if width <= 1:
        return y
    kernel = np.ones(width)
    mask = np.isfinite(y)
    filled = np.where(mask, y, 0.0)
    num = np.convolve(filled, kernel, mode="same")
    den = np.convolve(mask.astype(float), kernel, mode="same")
    out = np.divide(num, den, out=np.full_like(y, np.nan), where=den > 0)
    return np.where(mask, out, np.nan)


# =============================================================================
# Loading and ledger reconstruction
# =============================================================================
def _sum_sides(cd, name, n):
    """Sum a cathode scalar across active sides; NaN (side not solving) -> 0."""
    total = np.zeros(n)
    for side in SIDES:
        key = f"{side}_{name}"
        if key in cd:
            total += np.nan_to_num(np.asarray(cd[key], dtype=float), nan=0.0)
    return total


def load_run(path):
    """Read the channels this script uses from a sim1d-hdf5-v1 result."""
    data = {}
    with h5py.File(path, "r") as f:
        t = np.asarray(f["time"], dtype=float)
        n = len(t)
        data["time_ms"] = t * 1e3
        data["phase"] = np.asarray(f["phase"]).astype(str)
        pe = f["phase_events"]
        data["phase_events"] = list(
            zip(np.asarray(pe["phase"]).astype(str), np.asarray(pe["time"], dtype=float))
        )
        cd = f["cathode_diagnostics"]
        for name in (
            "P_load", "P_wall", "P_comp", "P_prim", "P_ohmic",
            "I_tot", "I_i", "I_eth_star", "phi_c", "phi_a", "V_b", "V_p",
        ):
            data[name] = _sum_sides(cd, name, n)
        # Active frames: any side produced a finite circuit solution.
        active = np.zeros(n, dtype=bool)
        for side in SIDES:
            key = f"{side}_P_load"
            if key in cd:
                active |= np.isfinite(np.asarray(cd[key], dtype=float))
        data["active"] = active
        data["collector_surface_W"] = np.nan_to_num(
            np.asarray(cd["collector_surface_power_W"], dtype=float), nan=0.0
        ) if "collector_surface_power_W" in cd else np.zeros(n)
        # Volume-integrated plasma-book channels (electron + ion books).
        Vp = np.asarray(f["geometry/plasma_volume_cm3"], dtype=float)
        plasma = {}
        for label, terms in PLASMA_LOSS_CHANNELS.items():
            total = np.zeros(n)
            for term in terms:
                for book in ("electron_energy_terms_W_cm3", "ion_energy_terms_W_cm3"):
                    if book in f and term in f[book]:
                        total += np.sum(
                            np.asarray(f[book][term], dtype=float) * Vp[None, :], axis=1
                        )
            plasma[label] = total
        data["plasma_losses"] = plasma
    return data


def build_ledger(d):
    """Reconstruct the R3.2 P_load ledger lines [W] from saved scalars.

    Every line is exact from saved data; no solver re-run and no free
    parameters (the bypass line is I_eth**phi_c - P_prim, so the eta*bypass
    factor is never needed; the returning-electron current comes from the
    cathode Kirchhoff relation I_e_ret = I_eth* + I_i - I_tot).
    """
    ledger = {
        "P_prim": d["P_prim"],
        "P_ohmic": d["P_ohmic"],
        "P_cathode_ion_phi": d["I_i"] * d["phi_c"],
        "P_beam_bypass": d["I_eth_star"] * d["phi_c"] - d["P_prim"],
        "P_anode_field": -d["I_tot"] * d["phi_a"],
        "P_electron_return": -(d["I_eth_star"] + d["I_i"] - d["I_tot"]) * d["phi_c"],
    }
    residual = d["P_load"] - sum(ledger.values())
    return ledger, residual


def group_series(ledger):
    """Sum ledger channels into the configured groups [W]."""
    return {
        name: sum(ledger[ch] for ch in spec["channels"])
        for name, spec in CHANNEL_GROUPS.items()
    }


# =============================================================================
# Figure helpers
# =============================================================================
def _phase_lines(ax, phase_events, t0_ms):
    for name, t_s in phase_events:
        t_ms = t_s * 1e3
        if name in ("main_discharge", "afterglow") and t_ms >= t0_ms:
            ax.axvline(t_ms, color="0.75", lw=0.6, ls=(0, (2, 2)), zorder=1)


def _signed_stack(ax, t, series, styles):
    """Stack signed channels: positive parts up, negative parts down."""
    pos = np.zeros_like(t)
    neg = np.zeros_like(t)
    for key, y in series.items():
        label, color = styles[key]
        yp = np.clip(y, 0.0, None)
        yn = np.clip(y, None, 0.0)
        ax.fill_between(t, pos, pos + yp, color=color, lw=0.0, alpha=0.85,
                        label=label, zorder=2)
        ax.fill_between(t, neg, neg + yn, color=color, lw=0.0, alpha=0.85, zorder=2)
        pos = pos + yp
        neg = neg + yn


def make_power_figure(d, ledger, residual, profile, smooth):
    """Ledger stack + closure residual + plasma-book losses (shared t axis)."""
    act = d["active"]
    t = d["time_ms"]
    t0 = t[act][0] if act.any() else t[0]
    kW = 1e-3
    journal = profile == "journal"

    if journal:
        figsize = (fs.JOURNAL_WIDTHS["aip_double"], 5.6)
    else:
        figsize = (12.9, 8.6)
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.6, 0.8, 1.7]},
    )
    ax_led, ax_res, ax_pla = axes

    # --- panel A: the P_load ledger, signed stacked area + P_load overlay ---
    series = {
        k: _boxcar(np.where(act, v, np.nan), smooth) * kW
        for k, v in ledger.items()
    }
    _signed_stack(ax_led, t, series, LEDGER_STYLE)
    ax_led.plot(t, _boxcar(np.where(act, d["P_load"], np.nan), smooth) * kW,
                color="#1A1A1A", lw=1.0 if journal else 2.2,
                label=r"$P_\mathrm{load} = I_\mathrm{tot} V_b$", zorder=4)
    ax_led.plot(t, _boxcar(np.where(act, d["P_wall"], np.nan), smooth) * kW,
                color="#444A52", lw=0.8 if journal else 1.6, ls=(0, (4, 2)),
                label=r"$P_\mathrm{wall}$ $(= P_\mathrm{load} + P_\mathrm{comp})$",
                zorder=4)
    ax_led.axhline(0.0, color="0.6", lw=0.6, zorder=3)
    ax_led.set_ylabel("power [kW]")
    # Legend above the figure (reserved space) so it never collides with data.
    handles, labels = ax_led.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3,
               handlelength=1.2, columnspacing=1.0, handletextpad=0.5,
               fontsize="small")
    _phase_lines(ax_led, d["phase_events"], t0)

    # --- panel B: the measured closure residual (dots; exact zeros drop out) ---
    res = np.where(act, np.abs(residual), np.nan)
    ax_res.plot(t, res, ls="none", marker=".", ms=1.5 if journal else 3.0,
                color="#0072B2", alpha=0.6, rasterized=True)
    ax_res.set_yscale("log")
    rmax = np.nanmax(res) if act.any() else np.nan
    ax_res.set_ylim(1e-13, max(1e-8, rmax * 30.0))
    ax_res.set_ylabel("closure\nresidual [W]")
    ax_res.annotate(
        f"max {rmax:.1e} W  ({rmax / np.max(d['P_load'][act]):.1e} of "
        r"$P_\mathrm{load}$)",
        xy=(0.99, 0.06), xycoords="axes fraction", ha="right", va="bottom",
        color="#444A52", fontsize="small")
    _phase_lines(ax_res, d["phase_events"], t0)

    # --- panel C: plasma-book volumetric losses (magnitudes, log axis) ---
    cycle = [c for c in fs.palette("journal" if journal else "slide")
             if c.lower() not in ("#000000", "#1a1a1a")]
    losses = dict(d["plasma_losses"])
    losses["collector surface"] = -d["collector_surface_W"]
    xlim = (max(0.0, t0 - 0.5), t[-1])
    in_x = (t >= xlim[0]) & (t <= xlim[1])
    top = PLASMA_FLOOR_W
    ci = 0
    for label, y in losses.items():
        mag = np.abs(np.minimum(_boxcar(y, smooth), 0.0))  # sinks, magnitude
        if np.nanmax(mag[in_x]) < PLASMA_FLOOR_W:
            continue  # inactive channel for this run
        top = max(top, np.nanmax(mag[in_x]))
        # More channels than palette colors: dash the second cycle pass so no
        # two channels share an identical line.
        ls = "-" if ci < len(cycle) else (0, (3, 1.5))
        ax_pla.plot(t, mag * kW, color=cycle[ci % len(cycle)], ls=ls,
                    lw=1.0 if journal else 2.2, label=label)
        ci += 1
    ax_pla.set_yscale("log")
    ax_pla.set_ylim(PLASMA_FLOOR_W * kW, 4.0 * top * kW)
    ax_pla.set_ylabel("plasma-book\nloss [kW]")
    ax_pla.set_xlabel("time [ms]")
    # Legend in its own column to the right of the panel.
    ax_pla.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1,
                  handlelength=1.2, borderaxespad=0.0, fontsize="small")
    _phase_lines(ax_pla, d["phase_events"], t0)

    ax_led.set_xlim(*xlim)
    if smooth > 1:
        dt_ms = float(np.median(np.diff(t)))
        ax_led.annotate(f"{smooth * dt_ms:.2f} ms boxcar", xy=(0.02, 0.97),
                        xycoords="axes fraction", ha="left", va="top",
                        color="#444A52", fontsize="small")
    return fig


def make_efficiency_figure(d, ledger, window, profile, smooth):
    """Fraction of P_load per group vs time + energy-integrated shares."""
    act = d["active"]
    t = d["time_ms"]
    t0 = t[act][0] if act.any() else t[0]
    groups = group_series(ledger)
    P_load = d["P_load"]
    journal = profile == "journal"

    sel = window & act
    floor = EFF_PLOAD_FLOOR_FRAC * np.median(P_load[sel]) if sel.any() else np.inf
    # Numerator and denominator get the SAME boxcar so the ratio is the ratio
    # of the plotted (smoothed) series, not a mixed raw/smoothed quantity.
    P_load_s = _boxcar(np.where(act, P_load, np.nan), smooth)
    denom = np.where(act & (P_load > floor), P_load_s, np.nan)

    if journal:
        figsize = (fs.JOURNAL_WIDTHS["aip_double"], 3.0)
    else:
        figsize = (12.9, 5.6)
    fig, (ax_t, ax_b) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True,
        gridspec_kw={"width_ratios": [1.75, 1.0]},
    )

    # --- panel A: time-resolved group fraction of P_load ---
    lo, hi = 0.0, 100.0
    for name, y in groups.items():
        role = CHANNEL_GROUPS[name]["role"]
        frac = 100.0 * _boxcar(np.where(act, y, np.nan), smooth) / denom
        ax_t.plot(t, frac, color=GROUP_COLORS[name],
                  lw=(1.8 if role == "delivered" else 1.0) if journal
                  else (3.6 if role == "delivered" else 2.0),
                  label=name)
        if sel.any():  # scale the axis from the in-window values only, so
            fw = frac[sel]  # startup/ramp-down blowups clip out of view
            lo = min(lo, np.nanmin(fw))
            hi = max(hi, np.nanmax(fw))
    ax_t.axhline(0.0, color="0.6", lw=0.6)
    ax_t.axhline(100.0, color="0.8", lw=0.6, ls=(0, (2, 2)))
    ax_t.set_xlim(left=max(0.0, t0 - 0.5), right=t[-1])
    ax_t.set_ylim(lo - 12.0, hi + 12.0)
    ax_t.set_xlabel("time [ms]")
    ax_t.set_ylabel(r"share of $P_\mathrm{load}$ [%]")
    # Legend above the figure (reserved space) so it never collides with data.
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2,
               handlelength=1.2, columnspacing=1.0, handletextpad=0.5,
               fontsize="small")
    if smooth > 1:
        dt_ms = float(np.median(np.diff(t)))
        ax_t.annotate(f"{smooth * dt_ms:.2f} ms boxcar", xy=(0.99, 0.02),
                      xycoords="axes fraction", ha="right", va="bottom",
                      color="#444A52", fontsize="small")
    _phase_lines(ax_t, d["phase_events"], t0)

    # --- panel B: energy-integrated share over the main-discharge window ---
    t_s = d["time_ms"] * 1e-3
    E_load = np.trapezoid(np.where(sel, P_load, 0.0), t_s)
    shares, labels, colors = [], [], []
    for name, y in groups.items():
        E = np.trapezoid(np.where(sel, y, 0.0), t_s)
        shares.append(100.0 * E / E_load)
        labels.append(name.split(" (")[0])
        colors.append(GROUP_COLORS[name])
    ypos = np.arange(len(shares))[::-1]
    ax_b.barh(ypos, shares, color=colors, height=0.62)
    ax_b.axvline(0.0, color="0.4", lw=0.8)
    for yp, s in zip(ypos, shares):
        # Positive bars: label just past the bar end. Negative bars: label to
        # the right of the zero line, clear of the bar and the tick labels.
        ax_b.annotate(f"{s:.1f}%", xy=(max(s, 0.0), yp), xytext=(4, 0),
                      textcoords="offset points", ha="left", va="center")
    ax_b.set_yticks(ypos, labels)
    ax_b.set_xlabel("share of $E_\\mathrm{load}$ [%]\n(main discharge)")
    lo_b, hi_b = min(0.0, min(shares)), max(shares)
    ax_b.set_xlim(lo_b - 0.06 * (hi_b - lo_b), hi_b + 0.24 * (hi_b - lo_b))
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    return fig


def save_figure(fig, out_dir, stem, profile):
    """PDF (vector) + PNG preview with deterministic names."""
    paths = []
    for ext, dpi in (("pdf", None), ("png", 200)):
        path = Path(out_dir) / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi if dpi else fs.PROFILES[profile]["dpi"])
        paths.append(path)
    plt.close(fig)
    return paths


# =============================================================================
# Main
# =============================================================================
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--from-h5", required=True, help="saved sim1d HDF5 result")
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).resolve().parent),
        help="directory for the figure files (default: this scripts/ dir)",
    )
    parser.add_argument(
        "--profile", choices=("journal", "slide"), default="journal",
        help="house figstyle profile (default: journal)",
    )
    parser.add_argument(
        "--smooth-ms", type=float, default=0.2,
        help="boxcar width [ms] applied to the PLOTTED series only, and "
        "annotated on the figure (the per-solve circuit scalars carry real "
        "high-frequency ripple that buries the plateau lines); the closure "
        "residual and every printed scalar always use the raw data. "
        "Pass 0 for raw plots (default: 0.2)",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.from_h5)
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Deterministic names from the input stem; the non-default profile gets a
    # suffix so the two profiles never overwrite each other.
    stem = in_path.stem + ("" if args.profile == "journal" else "_slide")

    d = load_run(in_path)
    ledger, residual = build_ledger(d)
    act = d["active"]
    if not act.any():
        raise SystemExit("no active cathode-circuit frames in this run; "
                         "nothing to plot")

    # --- verify and report the closure (never assumed) ---
    r = np.abs(residual[act])
    P_max = np.max(d["P_load"][act])
    print(f"closure_residual_max_W={np.max(r):.3e}")
    print(f"closure_residual_median_W={np.median(r):.3e}")
    print(f"closure_residual_rel={np.max(r) / P_max:.3e}")

    # --- per-run efficiency scalars over the main-discharge window ---
    window = d["phase"] == "main_discharge"
    sel = window & act
    t_s = d["time_ms"] * 1e-3
    E_load = np.trapezoid(np.where(sel, d["P_load"], 0.0), t_s)
    if sel.any():
        w = (t_s[sel][0] * 1e3, t_s[sel][-1] * 1e3)
        print(f"window=main_discharge t0_ms={w[0]:.3f} t1_ms={w[1]:.3f}")
    print(f"E_load_kJ={E_load / 1e3:.3f}")
    delivered_frac = None
    for name, y in group_series(ledger).items():
        E = np.trapezoid(np.where(sel, y, 0.0), t_s)
        key = name.split(" (")[0].replace(" ", "_").replace("-", "_")
        print(f"group_{key}_E_kJ={E / 1e3:.3f} frac={E / E_load:.4f}")
        if CHANNEL_GROUPS[name]["role"] == "delivered":
            delivered_frac = E / E_load
    print(f"efficiency_delivered_frac={delivered_frac:.4f}")

    # --- figures ---
    dt_ms = float(np.median(np.diff(d["time_ms"])))
    smooth = max(1, int(round(args.smooth_ms / dt_ms))) if args.smooth_ms > 0 else 1
    fs.apply(args.profile)
    written = []
    fig = make_power_figure(d, ledger, residual, args.profile, smooth)
    written += save_figure(fig, out_dir, f"{stem}_power", args.profile)
    fig = make_efficiency_figure(d, ledger, window, args.profile, smooth)
    written += save_figure(fig, out_dir, f"{stem}_efficiency", args.profile)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
