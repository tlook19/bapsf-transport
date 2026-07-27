"""Publication/slide power-loss and efficiency figures on the R3.2 closed ledger.

Reads a saved LAPDSim1D HDF5 result and renders two figures from the
current-resolved circuit power ledger (R3.2 / audit A16, see
``~/bapsf/docs/manuscripts/evidence/circuit_power_balance_r32.md``):

* ``<stem>_power``      -- the gross-outflow ledger stack with the net P_load
  line and the hatched round-trip (returned-to-circuit) slice, the measured
  closure residual |P_load - sum(ledger)| on its own axis, and the
  plasma-book volumetric loss groups (radiation, ionization cost,
  surface/sheath sinks, e-n cooling).
* ``<stem>_efficiency`` -- two bar charts of the NET (all-positive) power
  components over the main-discharge window: shares of E_wall (includes the
  compliance dissipation P_comp) and shares of E_load. The same numbers are
  printed as ``key=value`` lines for later per-ES-set comparison.
* ``<stem>_breakout``   -- percent contributions WITHIN the radiation group
  and WITHIN the surface/sheath group.

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
# CHANNEL GROUPING -- the NET (all-positive) power components.
#
# This dict is THE configuration point: the efficiency bar charts and the
# printed per-run scalars are computed from it, never hardcoded in plot
# logic. Each component lists reconstructed ledger channels (defined in
# build_ledger); the components partition P_load exactly.
#
# NET bookkeeping (Tom, 2026-07-27): the round-trip/returned field work is
# folded back into its source component so every displayed component is a
# positive net flow and nothing is negative:
#   * the returning-electron recovery (-I_e_ret*phi_c) nets against the
#     cathode sheath cost (the returning electrons hand cathode-sheath field
#     energy back at the cathode);
#   * the anode return, taken per the potential ladder (-I_tot*phi_a), nets
#     against the DELIVERED component -- the carriers that hand energy back
#     at the anode drew it from the plasma column, which the delivered
#     channels feed; the ion sheath fall and the bypass deposit on surfaces
#     and cannot come back.
# The round trip itself is still displayed on the POWER figure (hatched
# slice) and printed as returned_* scalars for the record.
# =============================================================================
CHANNEL_GROUPS = {
    "delivered to plasma (net of anode return)": {
        "channels": ("P_prim", "P_ohmic", "P_anode_field"),
        "role": "delivered",
    },
    "cathode sheath cost (net of returning e-)": {
        "channels": ("P_cathode_ion_phi", "P_electron_return"),
        "role": "loss",
    },
    "beam bypass (uncoupled primaries)": {
        "channels": ("P_beam_bypass",),
        "role": "loss",
    },
}
# Compliance/mesh resistor dissipation: a component of E_wall only
# (P_wall = P_load + P_comp); drawn from the saved P_comp scalar.
COMP_LABEL = "compliance resistor (P_comp)"
COMP_COLOR = "#999999"

# Positive outflow channels: stack order (bottom -> top), labels, colors.
LEDGER_STYLE = {
    "P_prim": ("beam primaries into column", "#0072B2"),
    "P_ohmic": ("bulk ohmic", "#56B4E9"),
    "P_cathode_ion_phi": ("cathode ion sheath fall", "#D55E00"),
    "P_beam_bypass": ("beam bypass", "#E69F00"),
}
# The round-trip overlay (gross-stack top down to the P_load line).
RETURNED_LABEL = ("returned to circuit (round trip: out via cathode sheath,\n"
                  "back via anode + returning electrons)")
RETURNED_COLOR = "#009E73"
RETURNED_HATCH = "////"

GROUP_COLORS = {
    "delivered to plasma (net of anode return)": "#0072B2",
    "cathode sheath cost (net of returning e-)": "#D55E00",
    "beam bypass (uncoupled primaries)": "#E69F00",
}

# Plasma-book volumetric loss groups: volume-integrated (electron + ion)
# energy terms [W/cm^3 * cm^3], summed per semantic group (signed sum, then
# clipped to the sink part). Each group carries an explicit color AND dash
# pattern so adjacent lines stay distinguishable in grayscale. Groups that
# never exceed PLASMA_FLOOR_W are dropped as inactive.
PLASMA_LOSS_GROUPS = {
    "surface/sheath losses": {
        # cathode+anode sheath deposits, end/characteristic outflow, and the
        # floating collector surface line (scalar channel).
        "terms": ("cathode_surface_loss", "anode_collection", "surface_loss",
                  "characteristic_boundary"),
        "collector_scalar": True,
        "color": "#D55E00", "ls": "-",
    },
    "radiation": {
        # line radiation (e-i cooling) + recombination + beam excitation
        "terms": ("electron_ion_cooling", "recombination_rad_loss",
                  "recombination_3b_loss", "beam_excitation_radiation"),
        "collector_scalar": False,
        "color": "#0072B2", "ls": (0, (4, 1.6)),
    },
    "ionization cost": {
        "terms": ("ionization_energy_cost",),
        "collector_scalar": False,
        "color": "#009E73", "ls": (0, (6, 1.4, 1.4, 1.4)),
    },
    "e-n cooling": {
        "terms": ("electron_neutral_cooling",),
        "collector_scalar": False,
        "color": "#CC79A7", "ls": (0, (1.2, 1.2)),
    },
}
PLASMA_FLOOR_W = 10.0  # activity/axis floor for the loss panel [W]

# WITHIN-group percent breakouts (the <stem>_breakout figure): each entry is
# a sub-component of one plasma-book group; percentages are of the group's
# own energy-integrated total over the main-discharge window.
RADIATION_BREAKOUT = {
    "line radiation (e-i cooling)": {
        "terms": ("electron_ion_cooling",), "color": "#0072B2"},
    "recombination radiation": {
        "terms": ("recombination_rad_loss", "recombination_3b_loss"),
        "color": "#56B4E9"},
    "beam excitation radiation": {
        "terms": ("beam_excitation_radiation",), "color": "#CC79A7"},
}
SURFACE_BREAKOUT = {
    # cathode_surface_loss is the cathode-boundary fluid deposit;
    # anode_collection is the anode-mesh interception sink (R4);
    # the end/characteristic outflow + floating collector are the far end.
    "cathode boundary": {
        "terms": ("cathode_surface_loss",), "color": "#D55E00"},
    "anode (mesh collection)": {
        "terms": ("anode_collection",), "color": "#E69F00"},
    "collector / end": {
        "terms": ("surface_loss", "characteristic_boundary"),
        "collector_scalar": True, "color": "#009E73"},
}

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
        # Volume-integrated plasma-book terms (electron + ion books) for every
        # term any configured group or breakout references; groups sum these
        # (signed) and may fold in the collector surface scalar as a sink.
        Vp = np.asarray(f["geometry/plasma_volume_cm3"], dtype=float)
        wanted = set()
        for cfg in (PLASMA_LOSS_GROUPS, RADIATION_BREAKOUT, SURFACE_BREAKOUT):
            for spec in cfg.values():
                wanted.update(spec["terms"])
        terms = {}
        for term in sorted(wanted):
            total = np.zeros(n)
            for book in ("electron_energy_terms_W_cm3", "ion_energy_terms_W_cm3"):
                if book in f and term in f[book]:
                    total += np.sum(
                        np.asarray(f[book][term], dtype=float) * Vp[None, :], axis=1
                    )
            terms[term] = total
        data["plasma_terms"] = terms
    return data


def plasma_group_sum(d, spec):
    """Signed sum of a plasma-book group's terms [W] (+ collector if declared)."""
    total = sum(d["plasma_terms"][term] for term in spec["terms"])
    if spec.get("collector_scalar"):
        total = total - d["collector_surface_W"]
    return total


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
    """Sum ledger channels into the configured NET groups [W]."""
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


def _positive_stack(ax, t, series, styles):
    """Stack strictly-upward bands; returns the running (gross) top."""
    top = np.zeros_like(t)
    for key, y in series.items():
        label, color = styles[key]
        yp = np.clip(y, 0.0, None)
        ax.fill_between(t, top, top + yp, color=color, lw=0.0, alpha=0.85,
                        label=label, zorder=2)
        top = top + yp
    return top


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

    # --- panel A: gross-outflow stack, P_load overlay, hatched round trip ---
    # The four positive outflow lines stack to the gross circuit outflow
    # P_gross = P_load + P_returned; the P_load line sits below the stack top
    # by exactly the returned power, and that gap is hatched: gross outflow
    # above, net P_load line, hatched slice = energy returned to the circuit.
    series = {
        k: _boxcar(np.where(act, ledger[k], np.nan), smooth) * kW
        for k in LEDGER_STYLE
    }
    gross_top = _positive_stack(ax_led, t, series, LEDGER_STYLE)
    P_load_s = _boxcar(np.where(act, d["P_load"], np.nan), smooth) * kW
    ax_led.fill_between(t, P_load_s, gross_top, facecolor="none",
                        edgecolor=RETURNED_COLOR, hatch=RETURNED_HATCH,
                        lw=0.0, label=RETURNED_LABEL, zorder=3)
    ax_led.plot(t, P_load_s, color="#1A1A1A", lw=1.2 if journal else 2.4,
                label=r"$P_\mathrm{load} = I_\mathrm{tot} V_b$ (net)", zorder=4)
    ax_led.plot(t, _boxcar(np.where(act, d["P_wall"], np.nan), smooth) * kW,
                color="#444A52", lw=0.8 if journal else 1.6, ls=(0, (4, 2)),
                label=r"$P_\mathrm{wall}$ $(= P_\mathrm{load} + P_\mathrm{comp})$",
                zorder=4)
    ax_led.set_ylim(bottom=0.0)
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

    # --- panel C: plasma-book loss groups (magnitudes, log axis) ---
    # Few semantic lines (config: PLASMA_LOSS_GROUPS), each with its own color
    # AND dash pattern (grayscale-safe). Direct right-edge labels were tried
    # and rejected: radiation and ionization cost sit within ~5% of each other
    # on this run, so adjacent labels cannot be tied to their lines; the
    # compact legend lives in the empty decade at lower right instead.
    xlim = (max(0.0, t0 - 0.5), t[-1])
    in_x = (t >= xlim[0]) & (t <= xlim[1])
    top = PLASMA_FLOOR_W
    for label, spec in PLASMA_LOSS_GROUPS.items():
        y = plasma_group_sum(d, spec)
        mag = np.abs(np.minimum(_boxcar(y, smooth), 0.0))  # sink part, magnitude
        if np.nanmax(mag[in_x]) < PLASMA_FLOOR_W:
            continue  # inactive group for this run
        top = max(top, np.nanmax(mag[in_x]))
        ax_pla.plot(t, mag * kW, color=spec["color"], ls=spec["ls"],
                    lw=1.1 if journal else 2.4, label=label)
    ax_pla.set_yscale("log")
    ax_pla.set_ylim(PLASMA_FLOOR_W * kW * 1e-1, 4.0 * top * kW)
    ax_pla.legend(loc="lower right", ncol=2, handlelength=1.8,
                  columnspacing=1.0, fontsize="small", borderaxespad=0.4)
    ax_pla.set_ylabel("plasma-book\nloss [kW]")
    ax_pla.set_xlabel("time [ms]")
    _phase_lines(ax_pla, d["phase_events"], t0)

    ax_led.set_xlim(*xlim)
    return fig


def _share_barh(ax, items):
    """Horizontal percent bars: items = [(label, share_pct, color), ...]."""
    ypos = np.arange(len(items))[::-1]
    for yp, (label, s, color) in zip(ypos, items):
        ax.barh(yp, s, height=0.62, facecolor=color)
        ax.annotate(f"{s:.1f}%", xy=(max(s, 0.0), yp), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center")
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_yticks(ypos, [label for label, _, _ in items])
    hi = max(s for _, s, _ in items)
    lo = min(0.0, min(s for _, s, _ in items))
    ax.set_xlim(lo, hi + 0.24 * (hi - lo))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_efficiency_figure(d, ledger, window, profile):
    """Two bar charts of the NET power components (main-discharge energies).

    Left: shares of E_wall = int(I_tot*V_bank)dt -- this denominator includes
    the compliance/mesh resistor dissipation, so P_comp appears as an
    explicit component. Right: the same components as shares of
    E_load = int(I_tot*V_b)dt (no P_comp component). All components are
    positive NET flows (round trip folded back -- see CHANNEL_GROUPS) and
    each chart sums to 100% of its denominator.
    """
    act = d["active"]
    sel = window & act
    t_s = d["time_ms"] * 1e-3
    journal = profile == "journal"

    def E(y):
        return np.trapezoid(np.where(sel, y, 0.0), t_s)

    E_load = E(d["P_load"])
    E_wall = E(d["P_wall"])
    E_comp = E(d["P_comp"])
    comps = [(name.split(" (")[0], E(y), GROUP_COLORS[name])
             for name, y in group_series(ledger).items()]

    if journal:
        figsize = (fs.JOURNAL_WIDTHS["aip_double"], 2.4)
    else:
        figsize = (12.9, 4.6)
    fig, (ax_w, ax_l) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True)

    wall_items = [(lab, 100.0 * Ei / E_wall, c) for lab, Ei, c in comps]
    wall_items.append((COMP_LABEL.split(" (")[0], 100.0 * E_comp / E_wall,
                       COMP_COLOR))
    _share_barh(ax_w, wall_items)
    ax_w.set_xlabel("share of $E_\\mathrm{wall}$ [%]\n(main discharge)")

    load_items = [(lab, 100.0 * Ei / E_load, c) for lab, Ei, c in comps]
    _share_barh(ax_l, load_items)
    ax_l.set_xlabel("share of $E_\\mathrm{load}$ [%]\n(main discharge)")
    return fig


def make_breakout_figure(d, window, profile):
    """Percent contributions WITHIN the radiation and surface/sheath groups.

    Each panel normalizes its sub-components (energy-integrated sink
    magnitudes over the main-discharge window) by the group's own total.
    """
    sel = window
    t_s = d["time_ms"] * 1e-3
    journal = profile == "journal"

    def E_sink(spec):
        y = plasma_group_sum(d, spec)
        mag = np.abs(np.minimum(y, 0.0))  # sink part only
        return np.trapezoid(np.where(sel, mag, 0.0), t_s)

    if journal:
        figsize = (fs.JOURNAL_WIDTHS["aip_double"], 2.2)
    else:
        figsize = (12.9, 4.2)
    fig, (ax_r, ax_s) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True)

    for ax, breakout, xlabel in (
        (ax_r, RADIATION_BREAKOUT, "% of radiation losses\n(main discharge)"),
        (ax_s, SURFACE_BREAKOUT, "% of surface/sheath losses\n(main discharge)"),
    ):
        energies = {label: E_sink(spec) for label, spec in breakout.items()}
        total = sum(energies.values())
        items = [(label, 100.0 * Ei / total if total > 0 else 0.0,
                  breakout[label]["color"]) for label, Ei in energies.items()]
        _share_barh(ax, items)
        ax.set_xlabel(xlabel)
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
    if sel.any():
        w = (t_s[sel][0] * 1e3, t_s[sel][-1] * 1e3)
        print(f"window=main_discharge t0_ms={w[0]:.3f} t1_ms={w[1]:.3f}")
    print(f"smooth_ms={args.smooth_ms}")

    def E(y):
        return np.trapezoid(np.where(sel, y, 0.0), t_s)

    # Denominator definitions (kept off-figure by request):
    print("E_wall_def=integral(P_wall)dt=integral(I_tot*V_bank)dt over the "
          "main-discharge window; includes the compliance/mesh resistor "
          "dissipation P_comp")
    print("E_load_def=integral(P_load)dt=integral(I_tot*V_b)dt over the "
          "main-discharge window; power across the plasma load, no P_comp")
    E_load = E(d["P_load"])
    E_wall = E(d["P_wall"])
    E_comp = E(d["P_comp"])
    print(f"E_wall_kJ={E_wall / 1e3:.3f}")
    print(f"E_load_kJ={E_load / 1e3:.3f}")
    print(f"E_comp_kJ={E_comp / 1e3:.3f} frac_wall={E_comp / E_wall:.4f}")
    # NET all-positive components (round trip folded back; see CHANNEL_GROUPS).
    delivered_frac = None
    for name, y in group_series(ledger).items():
        Ei = E(y)
        key = name.split(" (")[0].replace(" ", "_").replace("-", "_")
        print(f"net_{key}_E_kJ={Ei / 1e3:.3f} frac_load={Ei / E_load:.4f} "
              f"frac_wall={Ei / E_wall:.4f}")
        if CHANNEL_GROUPS[name]["role"] == "delivered":
            delivered_frac = Ei / E_load
    print(f"efficiency_delivered_net_frac_load={delivered_frac:.4f}")
    # Round-trip record (folded into the net components above, displayed only
    # as the hatched slice on the power figure):
    E_anode_ret = E(-ledger["P_anode_field"])
    E_eret = E(-ledger["P_electron_return"])
    print(f"returned_total_kJ={(E_anode_ret + E_eret) / 1e3:.3f}")
    print(f"returned_anode_kJ={E_anode_ret / 1e3:.3f}")
    print(f"returned_electron_kJ={E_eret / 1e3:.3f}")

    # --- figures ---
    dt_ms = float(np.median(np.diff(d["time_ms"])))
    smooth = max(1, int(round(args.smooth_ms / dt_ms))) if args.smooth_ms > 0 else 1
    fs.apply(args.profile)
    written = []
    fig = make_power_figure(d, ledger, residual, args.profile, smooth)
    written += save_figure(fig, out_dir, f"{stem}_power", args.profile)
    fig = make_efficiency_figure(d, ledger, window, args.profile)
    written += save_figure(fig, out_dir, f"{stem}_efficiency", args.profile)
    fig = make_breakout_figure(d, window, args.profile)
    written += save_figure(fig, out_dir, f"{stem}_breakout", args.profile)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
