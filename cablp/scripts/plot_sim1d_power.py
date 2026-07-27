"""Publication/slide power-loss and efficiency figures on the R3.2 closed ledger.

Reads a saved LAPDSim1D HDF5 result and renders two figures from the
current-resolved circuit power ledger (R3.2 / audit A16, see
``~/bapsf/docs/manuscripts/evidence/circuit_power_balance_r32.md``):

* ``<stem>_power``      -- the gross-outflow ledger stack with the net P_load
  line and the hatched round-trip (returned-to-circuit) slice, the measured
  closure residual |P_load - sum(ledger)| on its own axis, and the
  plasma-book volumetric loss groups (line radiation, other radiation,
  ionization cost, surface/sheath sinks, e-n cooling).
* ``<stem>_efficiency`` -- a three-stage cascade over the main-discharge
  window: source -> absorbed beam power and its fate -> thermal budget.
  Stage 1: the GROSS (unnetted) circuit components against two denominators,
  shares of E_wall (includes the compliance dissipation P_comp) and shares
  of E_load, with the round trip drawn as its own hatched NEGATIVE slice so
  the bars reconcile to the denominator exactly. Stage 2: the fate of the
  absorbed beam power int(P_prim)dt by collision outcome (thermal /
  ionization consumption / prompt radiation), with the unreconciled
  CSDA-vs-circuit remainder as an explicit bar and the ohmic delivery shown
  alongside. Stage 3: the plasma-book budget of the thermal energy, with the
  anode convected-2Te return-current channel broken out and a residual bar
  closing the budget. The same numbers are printed as ``key=value`` lines
  for later per-ES-set comparison.
* ``<stem>_breakout``   -- percent contributions WITHIN the electron cooling
  channels (with the Coulomb e->i equilibration shown for scale but excluded
  from the loss total), WITHIN the radiation group, and WITHIN the
  surface/sheath group.

A channel-to-``b_Q*``-name dictionary for the treacherously named cooling
channels (``ei_exchange`` / ``electron_ion_cooling`` /
``electron_neutral_cooling``) is kept as a comment block above
``PLASMA_LOSS_GROUPS`` -- read it before touching the cooling groups.

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
# STAGE-1 CHANNEL GROUPING -- the GROSS (unnetted) circuit components.
#
# v3 amendment (Tom, 2026-07-27) REPLACES the v2 net/folded design: every
# component here is the raw ledger flow, nothing is folded into anything
# else, and the round trip is its OWN explicitly labeled slice. It is drawn
# as a NEGATIVE, hatched bar (consistent with the power figure's hatched
# round-trip slice) because it RETURNS energy to the circuit, so the bars
# reconcile to their denominator by construction and algebraically sum to
# exactly 100%:
#
#   delivered + cathode ion sheath + bypass [+ P_comp] + circulation = E_denom
#
# with circulation = P_anode_field + P_electron_return (both already negative
# in build_ledger, i.e. -(I_tot*phi_a + I_e_ret*phi_c)). The identity is
# checked and printed per run in main().
#
# This dict is THE configuration point: the stage-1 bars and the printed
# per-run scalars are computed from it, never hardcoded in plot logic.
# =============================================================================
GROSS_GROUPS = {
    "beam power absorbed": {
        # P_prim = (1 - eta*beam_bypass_fraction) * I_eth_star * phi_c, the
        # collisionally coupled fraction of the emitted beam (cathode solver
        # _cathode_solver.py). Stage 2 decomposes exactly this component.
        "channels": ("P_prim",),
        "role": "absorbed",
        "color": "#0072B2",
    },
    "bulk ohmic": {
        "channels": ("P_ohmic",),
        "role": "ohmic",
        "color": "#56B4E9",
    },
    "cathode ion sheath": {
        "channels": ("P_cathode_ion_phi",),
        "role": "loss",
        "color": "#D55E00",
    },
    # Labels are hard-wrapped: unwrapped, these are wider than a half-width
    # journal panel and constrained_layout collapses the axes to zero.
    "beam bypass (to surfaces,\nnever couples to plasma)": {
        "channels": ("P_beam_bypass",),
        "role": "loss",
        "color": "#E69F00",
    },
}
# The round trip, broken out rather than netted or hidden. Both channels are
# negative in build_ledger; the anode term dominates (the returning-electron
# recovery is ~0.04% of E_load on es1_r5_f01_ag26ms) and both are printed
# separately by main() so the lumping is never silent.
CIRCULATION_LABEL = "anode circulation —\nreturned to circuit"
CIRCULATION_CHANNELS = ("P_anode_field", "P_electron_return")
CIRCULATION_COLOR = "#009E73"
CIRCULATION_HATCH = "////"
# Compliance/mesh resistor dissipation: a component of E_wall only
# (P_wall = P_load + P_comp); drawn from the saved P_comp scalar.
COMP_LABEL = "compliance resistor"
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

# =============================================================================
# STAGE-2 -- the FATE OF THE ABSORBED BEAM POWER, by collision outcome.
#
# Denominator: the stage-1 "beam power absorbed" energy int(P_prim)dt. The
# CSDA/Beer-Lambert deposition splits each primary's energy into disjoint
# banks (funcs/_beam_deposition.py), which physics/cathode.py books as three
# separate plasma-book channels:
#   beam_power_deposition     Ee > 0  Coulomb + anomalous drag + secondary
#                                     electrons + sub-threshold residual,
#                                     i.e. THERMAL delivery to the electrons
#   beam_ionization_cost      Ee < 0  I_ion per beam-driven ionization; with
#                                     the conservative birth convention the
#                                     newborn electron carries no energy, so
#                                     this channel IS the consumption
#   beam_excitation_radiation Ee < 0  ~21.2 eV per excitation event, radiated
#                                     promptly as He I light
#
# IDENTITY USED (checked and printed by main(), never assumed):
#
#   int(P_prim)dt = thermal + ionization_cost + radiated + residual
#
# On es1_r5_f01_ag26ms this does NOT close: the residual is -16.5% of P_prim,
# because the fluid's CSDA ray absorbs MORE than the circuit books as
# coupled. The per-ray bank closes on its own terms
# (Gamma0*E0 = heating + radiated + cost + anode_intercepted + transmitted,
# _beam_deposition.py), but the circuit's bypass -- a Beer-Lambert gap
# survival exp(-L_cath/l_b) -- and the CSDA ray's own attenuation are
# independent calculations and disagree by that amount. The residual is drawn
# as an explicit labeled bar rather than absorbed anywhere: it is a finding,
# not a plotting choice.
# =============================================================================
# Labels are single-line: stage 2 gets the full figure width, so wrapping
# them only crowds the rows vertically at slide scale.
BEAM_FATE = {
    "delivered as thermal (to electron pool)": {
        "terms": ("beam_power_deposition",), "color": "#0072B2"},
    "consumed in ionization (beam $I_\\mathrm{ion}$ cost)": {
        "terms": ("beam_ionization_cost",), "color": "#009E73"},
    "immediately radiated (beam excitation)": {
        "terms": ("beam_excitation_radiation",), "color": "#CC79A7"},
}
BEAM_RESIDUAL_LABEL = "unreconciled (CSDA > circuit $P_\\mathrm{prim}$)"
BEAM_RESIDUAL_COLOR = "#888888"
BEAM_RESIDUAL_HATCH = "xxx"
# Shown alongside (excluded from the P_prim partition) so the TOTAL thermal
# input to the plasma is readable off the same panel: the ohmic channel is
# pure thermal delivery and is what stage 3 adds to the beam thermal bank.
OHMIC_REFERENCE_LABEL = "bulk ohmic (pure thermal, not beam)"
OHMIC_REFERENCE_COLOR = "#56B4E9"
OHMIC_REFERENCE_HATCH = "///"

# =============================================================================
# STAGE-3 -- the plasma-book budget OF THE THERMAL ENERGY.
#
# Denominator: the total thermal input established by stage 2,
#   E_thermal = int(beam_power_deposition)dt + int(P_ohmic)dt
# i.e. the beam's thermal bank (NOT P_prim, which stage 2 shows the fluid
# does not agree with) plus the ohmic delivery. The three stages then read as
# a cascade:  source -> absorbed & its fate -> where the thermal energy goes.
#
# anode_collection is BROKEN OUT of the surface/sheath group here (Tom: it is
# the real plasma-side cost of the return current). It is the Bohm-flux
# collection at the anode mesh -- physics/sources.py :: anode_collection_rhs,
# "Mass, momentum and thermal energy leave together as at any wall" -- i.e.
# the convected ~2Te-per-pair sink over the mesh solid fraction eta, NOT the
# beam-mesh interception (see the bypass-termination note in main()).
#
# "ionization cost (bulk)" is the BULK reaction channel and is deliberately
# distinct from stage 2's beam ionization cost: different channels, different
# rate paths (reactions.py SCD vs the beam ray's own events).
#
# These groups are volumetric sinks and need not exhaust the thermal input:
# the remainder (internal energy actually retained, plus any book not listed)
# is an explicit residual bar so the budget visibly closes to 100%.
# =============================================================================
THERMAL_BUDGET = {
    "surface/sheath (cathode + end)": {
        "terms": ("cathode_surface_loss", "surface_loss",
                  "characteristic_boundary"),
        "collector_scalar": True, "color": "#D55E00",
    },
    "anode convected 2Te (return current)": {
        "terms": ("anode_collection",),
        "collector_scalar": False, "color": "#E69F00",
    },
    "line radiation (icool)": {
        "terms": ("electron_ion_cooling",),
        "collector_scalar": False, "color": "#0072B2",
    },
    "other radiation": {
        "terms": ("recombination_rad_loss", "recombination_3b_loss",
                  "beam_excitation_radiation"),
        "collector_scalar": False, "color": "#56B4E9",
    },
    "ionization cost (bulk)": {
        "terms": ("ionization_energy_cost",),
        "collector_scalar": False, "color": "#009E73",
    },
    "e-n cooling": {
        "terms": ("electron_neutral_cooling",),
        "collector_scalar": False, "color": "#CC79A7",
    },
}
RESIDUAL_LABEL = "stored / other (residual)"
RESIDUAL_COLOR = "#B0B0B0"

# =============================================================================
# CHANNEL -> Q-NAME DICTIONARY (read off the RHS code, 2026-07-27).
#
# The saved-channel names and the b_Q* scale-factor names do NOT line up, and
# the collision is exactly in the "e-i" abbreviation. Settled by reading
# physics/energy.py, physics/reactions.py, physics/cathode.py and
# funcs/_heat.py; the b_Q* keys all live in the input_dict template
# (core/config.py), the icool/ncool switches in input_flags:
#
#   ei_exchange              <- b_Qie   [input_dict]   (no flag)
#       energy.py :: electron_ion_exchange_rhs -> _heat.py :: Q_ie,
#       the Braginskii COULOMB (elastic) thermal equilibration
#       Q = 3 n (Te - Ti) / tau_e / (m_i/m_e), positive when Te > Ti.
#       Books Ee = -q AND Ei = +q, i.e. an INTERNAL TRANSFER within the
#       plasma, not a loss from it. NB summing the electron and ion books
#       cancels this channel to EXACTLY zero -- it is only meaningful read
#       from the electron book alone (see plasma_terms_electron).
#
#   electron_ion_cooling     <- b_Qei   [input_dict]   flag `icool`
#       energy.py :: electron_cooling_rhs_terms, want_qei branch.
#       adas : b_Qei * plt2 * n_e * n_e -- OPEN-ADAS PLT at z1 = 2, the LINE
#              RADIATED POWER of He+ (plus prb1, recombination +
#              bremsstrahlung radiated power, only when `icool_recomb` is on).
#       janev: b_Qei * IAEA_exp4(Te, aHeII) * n_e * n_e (He II inelastic).
#       Books Ee only (sink; Ei = 0) -- the energy LEAVES the plasma as light.
#       Despite the name this is NOT electron-ion equilibration: the "e-i"
#       names the collision PARTNER (the ion), not an exchange with it.
#
#   electron_neutral_cooling <- b_Qen   [input_dict]   flag `ncool`
#       energy.py :: electron_cooling_rhs_terms, want_qen branch.
#       adas : b_Qen * plt1 * n_e * n_n -- PLT at z1 = 1, the LINE RADIATED
#              POWER of neutral He; radiation-only, hence consistent with the
#              separate ionization_energy_cost term.
#       janev: b_Qen * IAEA_exp1(Te, aHeI) * n_e * n_n -- this fit ALREADY
#              CONTAINS the ionization potential, so together with the
#              separate ionization_energy_cost term it double-counts.
#       Books Ee only (sink; Ei = 0).
#
# Consequence for the production `atomic_rate_model = "adas"` stance: BOTH
# cooling channels are line radiation and differ only in the collision
# partner (He+ vs He0); b_Qie alone is the Coulomb equilibration. The
# "e-n cooling" line below is therefore also radiated light, kept under its
# historical name.
#
# Supporting channels used by the breakouts:
#   recombination_rad_loss / recombination_3b_loss (reactions.py ::
#       _recombination_loss) remove the recombining pair's THERMAL energy,
#       Ee = -3/2 Te S_rec and Ei = -3/2 Ti S_rec -- NOT photon emission
#       ("rad" names the radiative-recombination process, not the sink).
#       Under "adas" b_rec_3b is inert (ACD already contains three-body), so
#       recombination_3b_loss is identically zero there.
#   beam_excitation_radiation (cathode.py) books Ee only: the ~21.2 eV per
#       beam-impact excitation event, radiated promptly as He I light.
# =============================================================================

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
    # Promoted to its own line (v3): this is the single channel
    # electron_ion_cooling <- b_Qei, He+ line radiation. It carried ~99.7% of
    # the old lumped "radiation" group, and the old label "(e-i cooling)"
    # collided with the Coulomb-equilibration naming (b_Qie / ei_exchange).
    "line radiation (icool)": {
        "terms": ("electron_ion_cooling",),
        "collector_scalar": False,
        "color": "#0072B2", "ls": (0, (4, 1.6)),
    },
    # What is left of the old "radiation" group once icool is promoted out:
    # 0.3% combined on es1_r5_f01_ag26ms, so it is one folded line.
    "other radiation": {
        "terms": ("recombination_rad_loss", "recombination_3b_loss",
                  "beam_excitation_radiation"),
        "collector_scalar": False,
        "color": "#56B4E9", "ls": (0, (5, 1.2, 1.2, 1.2, 1.2, 1.2)),
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
    "line radiation (icool)": {
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
# Electron-book cooling breakout (v3, third panel of the breakout figure):
# the individual electron cooling channels as percent of THEIR OWN sum.
# Read from the ELECTRON BOOK ONLY -- per the dictionary above, ei_exchange
# cancels to exactly zero when the two books are summed, and the
# recombination channels' ion-book part is ion thermal energy, not an
# electron loss. The ionization cost is deliberately not here: it is a
# separate plasma-book group, not a cooling channel.
COOLING_BREAKOUT = {
    "line radiation (icool)": {
        "terms": ("electron_ion_cooling",), "color": "#0072B2"},
    "e-n cooling (ncool)": {
        "terms": ("electron_neutral_cooling",), "color": "#CC79A7"},
    "recombination (rad + 3b)": {
        "terms": ("recombination_rad_loss", "recombination_3b_loss"),
        "color": "#56B4E9"},
    "beam excitation radiation": {
        "terms": ("beam_excitation_radiation",), "color": "#E69F00"},
}
# Drawn alongside COOLING_BREAKOUT for scale but NEVER summed into its
# denominator: the Coulomb equilibration moves energy from the electron book
# to the ion book and never leaves the plasma. Hatched + gray + below a
# divider so it cannot be misread as one of the loss bars.
COOLING_REFERENCE = {
    # Wrapped: as one line this tick label is the widest text in the figure
    # and squeezes the bottom-row panels into their own x-labels on `slide`.
    "label": "internal transfer e$\\rightarrow$i\n(not a loss)",
    "terms": ("ei_exchange",),
    "color": "#666666",
    "hatch": "////",
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
        # Volume-integrated plasma-book terms for every term any configured
        # group or breakout references. Two views are kept:
        #   plasma_terms          -- electron + ion books (the plasma total)
        #   plasma_terms_electron -- electron book only
        # Both are needed: the electron-only view is the correct one for the
        # cooling breakout, because ei_exchange books -q to Ee and +q to Ei
        # and so cancels to EXACTLY zero in the summed view.
        Vp = np.asarray(f["geometry/plasma_volume_cm3"], dtype=float)
        wanted = set()
        for cfg in (PLASMA_LOSS_GROUPS, RADIATION_BREAKOUT, SURFACE_BREAKOUT,
                    COOLING_BREAKOUT, THERMAL_BUDGET, BEAM_FATE):
            for spec in cfg.values():
                wanted.update(spec["terms"])
        wanted.update(COOLING_REFERENCE["terms"])
        terms = {}
        terms_e = {}
        for term in sorted(wanted):
            total = np.zeros(n)
            electron = np.zeros(n)
            for book in ("electron_energy_terms_W_cm3", "ion_energy_terms_W_cm3"):
                if book in f and term in f[book]:
                    contrib = np.sum(
                        np.asarray(f[book][term], dtype=float) * Vp[None, :], axis=1
                    )
                    total += contrib
                    if book.startswith("electron"):
                        electron += contrib
            terms[term] = total
            terms_e[term] = electron
        data["plasma_terms"] = terms
        data["plasma_terms_electron"] = terms_e
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


def _scalar_key(label):
    """Console `key=value` name from a (possibly hard-wrapped) figure label."""
    flat = label.replace("\n", " ").split(" (")[0].strip()
    return flat.replace(" ", "_").replace("-", "_").replace("/", "_")


def gross_series(ledger):
    """Sum ledger channels into the configured GROSS stage-1 groups [W]."""
    return {
        name: sum(ledger[ch] for ch in spec["channels"])
        for name, spec in GROSS_GROUPS.items()
    }


def circulation_series(ledger):
    """The round-trip return [W], negative (energy handed back to the circuit)."""
    return sum(ledger[ch] for ch in CIRCULATION_CHANNELS)




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
    # One extra decade of headroom below the floor: with icool promoted out
    # of the radiation group there are five lines, so the legend is three
    # rows and needs a clear band under the faint "other radiation" trace.
    ax_pla.set_ylim(PLASMA_FLOOR_W * kW * 1e-2, 4.0 * top * kW)
    ax_pla.legend(loc="lower right", ncol=2, handlelength=1.8,
                  columnspacing=1.0, fontsize="small", borderaxespad=0.4,
                  framealpha=1.0, facecolor="white", edgecolor="0.8")
    ax_pla.set_ylabel("plasma-book\nloss [kW]")
    ax_pla.set_xlabel("time [ms]")
    _phase_lines(ax_pla, d["phase_events"], t0)

    ax_led.set_xlim(*xlim)
    return fig


def _share_barh(ax, items, n_excluded=0):
    """Horizontal percent bars: items = [(label, share_pct, color[, hatch]), ...].

    A 4th tuple element draws the bar hatched-and-outlined instead of solid.
    The last ``n_excluded`` items are drawn for scale but are NOT part of the
    total the percentages normalize; a dashed divider separates them.
    """
    ypos = np.arange(len(items))[::-1]
    for yp, item in zip(ypos, items):
        label, s, color = item[:3]
        hatch = item[3] if len(item) > 3 else None
        if hatch:
            ax.barh(yp, s, height=0.62, facecolor="none", edgecolor=color,
                    hatch=hatch, lw=0.8)
        else:
            ax.barh(yp, s, height=0.62, facecolor=color)
        # Sub-0.1% channels are real (the beam ionization/excitation banks);
        # printing them as a flat "0.0%" would misreport them as absent.
        txt = f"{s:.2f}%" if 0.0 < abs(s) < 0.1 else f"{s:.1f}%"
        ax.annotate(txt, xy=(max(s, 0.0), yp), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center")
    if n_excluded:
        ax.axhline(float(ypos[-n_excluded]) + 0.5, color="0.6", lw=0.7,
                   ls=(0, (3, 2)))
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_yticks(ypos, [item[0] for item in items])
    hi = max(item[1] for item in items)
    lo = min(0.0, min(item[1] for item in items))
    ax.set_xlim(lo, hi + 0.30 * (hi - lo))
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def beam_fate_energies(d, E):
    """Stage-2 banks [J]: (thermal, cost, radiated) magnitudes from P_prim."""
    return {
        label: abs(E(sum(d["plasma_terms"][t] for t in spec["terms"])))
        for label, spec in BEAM_FATE.items()
    }


def make_efficiency_figure(d, ledger, window, profile):
    """Three-stage cascade: source -> absorbed & its fate -> thermal budget.

    STAGE 1 (top row, two denominators): the GROSS circuit components, with
    no netting or folding. Left is shares of E_wall = int(I_tot*V_bank)dt,
    which includes the compliance/mesh resistor dissipation so P_comp is an
    explicit component; right is the same components as shares of
    E_load = int(I_tot*V_b)dt. The round trip is its own hatched NEGATIVE
    slice (it returns energy to the circuit), so each chart's bars sum
    algebraically to exactly 100% of its denominator.

    STAGE 2 (bottom left): the fate of the absorbed beam power int(P_prim)dt
    by collision outcome -- thermal / ionization consumption / prompt
    radiation -- with the unreconciled CSDA-vs-circuit remainder as its own
    labeled bar, and the ohmic delivery alongside (excluded from the
    partition) so the total thermal input is readable.

    STAGE 3 (bottom right): where that thermal energy goes -- the plasma-book
    loss groups with the anode convected-2Te return-current channel broken
    out, plus an explicit residual so the budget closes.
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
    gross = gross_series(ledger)
    comps = [(name, E(y), GROSS_GROUPS[name]["color"])
             for name, y in gross.items()]
    E_circ = E(circulation_series(ledger))  # negative: handed back
    roles = {GROSS_GROUPS[name]["role"]: Ei for name, Ei, _ in comps}
    E_prim, E_ohmic = roles["absorbed"], roles["ohmic"]

    fate = beam_fate_energies(d, E)
    E_beam_thermal = fate[next(iter(BEAM_FATE))]  # first entry = thermal bank
    E_thermal = E_beam_thermal + E_ohmic

    # Stage 1's two denominators pair naturally side by side (their labels are
    # short); stages 2 and 3 take the full width, which is what keeps their
    # long channel names and x-labels from colliding at either profile.
    if journal:
        figsize = (fs.JOURNAL_WIDTHS["aip_double"], 8.2)
    else:
        figsize = (12.9, 13.2)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.9, 1.15])
    ax_w = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])
    ax_t = fig.add_subplot(gs[2, :])

    # --- stage 1: gross components + the circulation slice, two denominators
    for ax, denom, with_comp, xlabel in (
        (ax_w, E_wall, True,
         "stage 1 — % of $E_\\mathrm{wall}$\n(bank source)"),
        (ax_l, E_load, False,
         "stage 1 — % of $E_\\mathrm{load}$\n(load source)"),
    ):
        items = [(lab, 100.0 * Ei / denom, c) for lab, Ei, c in comps]
        if with_comp:
            items.append((COMP_LABEL, 100.0 * E_comp / denom, COMP_COLOR))
        items.append((CIRCULATION_LABEL, 100.0 * E_circ / denom,
                      CIRCULATION_COLOR, CIRCULATION_HATCH))
        _share_barh(ax, items)
        ax.set_xlabel(xlabel)

    # --- stage 2: fate of the absorbed beam power (partition of P_prim) -----
    items = [(label, 100.0 * Ei / E_prim, BEAM_FATE[label]["color"])
             for label, Ei in fate.items()]
    beam_resid = E_prim - sum(fate.values())
    items.append((BEAM_RESIDUAL_LABEL, 100.0 * beam_resid / E_prim,
                  BEAM_RESIDUAL_COLOR, BEAM_RESIDUAL_HATCH))
    items.append((OHMIC_REFERENCE_LABEL, 100.0 * E_ohmic / E_prim,
                  OHMIC_REFERENCE_COLOR, OHMIC_REFERENCE_HATCH))
    _share_barh(ax_b, items, n_excluded=1)
    ax_b.set_xlabel("stage 2 — % of ABSORBED beam power\n"
                    "(fate by collision outcome)")

    # --- stage 3: the plasma-book budget of the thermal energy --------------
    budget = []
    for label, spec in THERMAL_BUDGET.items():
        y = plasma_group_sum(d, spec)
        mag = np.abs(np.minimum(y, 0.0))  # sink part only
        budget.append((label, E(mag), spec["color"]))
    accounted = sum(Ei for _, Ei, _ in budget)
    budget.append((RESIDUAL_LABEL, E_thermal - accounted, RESIDUAL_COLOR))
    items = [(lab, 100.0 * Ei / E_thermal, c) for lab, Ei, c in budget]
    _share_barh(ax_t, items)
    ax_t.set_xlabel("stage 3 — % of THERMAL input\n"
                    "(where the thermal energy goes)")
    return fig


def make_breakout_figure(d, window, profile):
    """Percent breakouts WITHIN the cooling, radiation and surface groups.

    Top panel (v3): the electron-book cooling channels individually, each as
    a percent of the summed electron cooling losses, with the Coulomb
    equilibration ei_exchange drawn alongside for scale but excluded from
    that total (it is an internal e->i transfer, not a loss -- see the
    channel dictionary at the top of this file).

    Bottom panels: sub-components of the radiation and surface/sheath groups,
    each normalized by its own group total. All energies are integrated sink
    magnitudes over the main-discharge window.
    """
    sel = window
    t_s = d["time_ms"] * 1e-3
    journal = profile == "journal"

    def E_sink(spec):
        y = plasma_group_sum(d, spec)
        mag = np.abs(np.minimum(y, 0.0))  # sink part only
        return np.trapezoid(np.where(sel, mag, 0.0), t_s)

    def E_sink_electron(terms):
        """Electron-book-only integrated sink magnitude [J] for `terms`."""
        y = sum(d["plasma_terms_electron"][term] for term in terms)
        mag = np.abs(np.minimum(y, 0.0))
        return np.trapezoid(np.where(sel, mag, 0.0), t_s)

    if journal:
        figsize = (fs.JOURNAL_WIDTHS["aip_double"], 4.3)
    else:
        figsize = (12.9, 7.4)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0])
    ax_c = fig.add_subplot(gs[0, :])
    ax_r = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[1, 1])

    # --- cooling breakout: percent of the summed electron cooling losses ---
    cool_E = {label: E_sink_electron(spec["terms"])
              for label, spec in COOLING_BREAKOUT.items()}
    cool_total = sum(cool_E.values())
    cool_items = [(label, 100.0 * Ei / cool_total if cool_total > 0 else 0.0,
                   COOLING_BREAKOUT[label]["color"])
                  for label, Ei in cool_E.items()]
    ref_E = E_sink_electron(COOLING_REFERENCE["terms"])
    cool_items.append((
        COOLING_REFERENCE["label"],
        100.0 * ref_E / cool_total if cool_total > 0 else 0.0,
        COOLING_REFERENCE["color"], COOLING_REFERENCE["hatch"],
    ))
    _share_barh(ax_c, cool_items, n_excluded=1)
    ax_c.set_xlabel("% of summed electron cooling losses\n(main discharge)")

    for ax, breakout, xlabel in (
        (ax_r, RADIATION_BREAKOUT, "% of radiation losses\n(main discharge)"),
        # Wrapped after "surface/sheath": as one line this x-label is wider
        # than its own half-width axes and clips at the figure edge on `slide`.
        (ax_s, SURFACE_BREAKOUT, "% of surface/sheath\nlosses (main discharge)"),
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
    # --- STAGE 1: gross components, nothing netted (see GROSS_GROUPS) -------
    roles = {}
    E_gross_sum = 0.0
    for name, y in gross_series(ledger).items():
        Ei = E(y)
        E_gross_sum += Ei
        roles[GROSS_GROUPS[name]["role"]] = Ei
        key = _scalar_key(name)
        print(f"gross_{key}_E_kJ={Ei / 1e3:.3f} frac_load={Ei / E_load:.4f} "
              f"frac_wall={Ei / E_wall:.4f}")
    E_prim, E_ohmic = roles["absorbed"], roles["ohmic"]
    # Round-trip record, now an explicit stage-1 slice rather than folded away.
    E_circ = E(circulation_series(ledger))  # negative: handed back
    E_anode_ret = E(-ledger["P_anode_field"])
    E_eret = E(-ledger["P_electron_return"])
    print(f"circulation_total_kJ={-E_circ / 1e3:.3f}")
    print(f"circulation_anode_kJ={E_anode_ret / 1e3:.3f}")
    print(f"circulation_returning_electron_kJ={E_eret / 1e3:.3f}")
    print(f"efficiency_beam_absorbed_frac_load={E_prim / E_load:.4f}")
    # Reconciliation identity, stated and CHECKED (never assumed):
    #   sum(gross components) [+ P_comp] + circulation = E_denominator
    print("stage1_identity=sum(gross components)+circulation=E_load; "
          "with P_comp added the same sum=E_wall")
    r_load = E_gross_sum + E_circ - E_load
    r_wall = E_gross_sum + E_comp + E_circ - E_wall
    print(f"stage1_reconcile_load_kJ={r_load / 1e3:.3e} "
          f"rel={abs(r_load) / E_load:.3e}")
    print(f"stage1_reconcile_wall_kJ={r_wall / 1e3:.3e} "
          f"rel={abs(r_wall) / E_wall:.3e}")

    # --- STAGE 2: fate of the absorbed beam power, by collision outcome -----
    # Identity: int(P_prim)dt = thermal + ionization_cost + radiated + residual
    print("stage2_identity=int(P_prim)dt=beam_thermal+beam_ionization_cost"
          "+beam_radiated+residual")
    fate = beam_fate_energies(d, E)
    for label, Ei in fate.items():
        print(f"beam_fate_{_scalar_key(label)}_kJ={Ei / 1e3:.3f} "
              f"frac_P_prim={Ei / E_prim:.4f}")
    beam_resid = E_prim - sum(fate.values())
    print(f"beam_fate_residual_kJ={beam_resid / 1e3:.3f} "
          f"frac_P_prim={beam_resid / E_prim:.4f}")
    E_beam_thermal = fate[next(iter(BEAM_FATE))]
    if abs(beam_resid) > 0.01 * E_prim:
        print("beam_fate_WARNING=stage-2 identity does NOT close: the CSDA "
              "fluid deposition and the circuit's Beer-Lambert bypass are "
              "independent calculations and disagree by the residual above "
              "(reported, not fixed here)")

    # --- STAGE 3: plasma-book budget of the thermal energy ------------------
    E_thermal = E_beam_thermal + E_ohmic
    print(f"E_thermal_kJ={E_thermal / 1e3:.3f} "
          f"(beam_thermal={E_beam_thermal / 1e3:.3f} + "
          f"ohmic={E_ohmic / 1e3:.3f})")
    accounted = 0.0
    for label, spec in THERMAL_BUDGET.items():
        y = plasma_group_sum(d, spec)
        Ei = E(np.abs(np.minimum(y, 0.0)))
        accounted += Ei
        print(f"thermal_budget_{_scalar_key(label)}_kJ={Ei / 1e3:.3f} "
              f"frac_thermal={Ei / E_thermal:.4f}")
    resid = E_thermal - accounted
    print(f"thermal_budget_residual_kJ={resid / 1e3:.3f} "
          f"frac_thermal={resid / E_thermal:.4f}")

    # --- Where the bypass primaries terminate (v3 amendment item 2) ---------
    # Traced through funcs/_cathode_solver.py, physics/cathode.py and
    # funcs/_beam_deposition.py: the bypass is CIRCUIT-booked only. The A15
    # anode-mesh interception removes anode_eta*gamma*E from the ray into
    # BeamDepositionResult.anode_intercepted_erg_s, explicitly NOT into
    # plasma_heating_erg_s -- and that field has no consumer anywhere in
    # cablp/solvers (only the twin-cathode summation in cathode.py:871-872),
    # no RHS channel and no saved diagnostic. The far-end `transmitted_flux`
    # is likewise never deposited on the collector/end books. So the bypass
    # primaries' terminal deposit is UNBOOKED on the plasma/surface side.
    print("bypass_termination=circuit_booked_only; A15 interception -> "
          "BeamDepositionResult.anode_intercepted_erg_s has no RHS consumer "
          "and no saved diagnostic; far-end transmitted_flux is not deposited "
          "on the collector/end books (reported, not fixed here)")

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
