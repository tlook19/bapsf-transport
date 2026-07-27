"""Publication/slide power-loss and efficiency figures on the R3.2 closed ledger.

Reads a saved LAPDSim1D HDF5 result and renders two figures from the
current-resolved circuit power ledger (R3.2 / audit A16, see
``~/bapsf/docs/manuscripts/evidence/circuit_power_balance_r32.md``):

* ``<stem>_power``      -- the gross-outflow ledger stack with the net P_load
  line and the hatched round-trip (returned-to-circuit) slice, the measured
  closure residual |P_load - sum(ledger)| on its own axis, and the
  plasma-book volumetric loss groups (line radiation from ions and from
  neutrals, other radiation, ionization cost, surface/sheath sinks).
* ``<stem>_efficiency`` -- a three-stage cascade: source -> absorbed beam
  power and its fate -> thermal budget.
  Stage 1: the GROSS (unnetted) circuit components against two denominators,
  shares of E_wall (includes the compliance dissipation P_comp) and shares
  of E_load, with the round trip drawn as its own hatched NEGATIVE slice so
  the bars reconcile to the denominator exactly. Stage 2: the fate of the
  absorbed beam power int(P_prim)dt by collision outcome (thermal /
  ionization consumption / prompt radiation), with the unreconciled
  CSDA-vs-circuit remainder as an explicit bar and the ohmic delivery shown
  alongside. Stage 3: the plasma-book budget of the thermal energy, with the
  anode convected-3/2-T return-current channel broken out and a residual bar
  closing the budget. The same numbers are printed as ``key=value`` lines
  for later per-ES-set comparison.

  v4 amendment (Tom, 2026-07-27): stages 1 and 3 keep the main-discharge
  window, but stage 2 is evaluated at BEAM TURN-ON -- window (a),
  plasma launch -> breakdown -- because that is when the absorption fate
  matters; the whole-discharge integral averages the turn-on transient away.
* ``<stem>_beam_windows`` -- the same stage-2 decomposition over all three
  discharge windows ((a) launch -> breakdown, (b) breakdown -> current knee,
  (c) knee -> drive end), so the drift of the unreconciled CSDA-vs-circuit
  remainder across the discharge is readable at a glance. The knee is
  detected on the discharge-current trace by ``KNEE_DETECT``.
* ``<stem>_beam_two_windows`` -- (v5) the ALTERNATE simple split of the same
  stage-2 decomposition: beam turn-on (launch -> knee, i.e. windows (a) and
  (b) merged) against the rest of the discharge (knee -> drive end). Both
  this and the three-window figure are produced; the knee is the one
  ``KNEE_DETECT`` boundary, regrouped, not a second detection.
* ``<stem>_losses``     -- three separately normalized loss graphs (v4), in
  v5 panel order COMBINED / electron / ion: the COMBINED plasma losses, where
  the antisymmetric Coulomb booking cancels identically and the channel
  therefore does not appear; the ELECTRON-book losses (the Coulomb e->i
  transfer included, since it is one of the main Te sinks); and the ION-book
  losses (with the same Coulomb transfer shown as the heating-source context
  bar, excluded from the loss total).
* ``<stem>_breakout``   -- percent contributions WITHIN the radiation group
  and WITHIN the surface/sheath group.

A channel-to-``b_Q*``-name dictionary for the treacherously named cooling
channels (``ei_exchange`` / ``electron_ion_cooling`` /
``electron_neutral_cooling``) is kept as a comment block above
``PLASMA_LOSS_GROUPS`` -- read it before touching the cooling groups. Per
that dictionary the two ``*_cooling`` channels are BOTH line radiation and
differ only in the collision partner, so v4 labels them as the parallel pair
"line radiation (ions)" / "line radiation (neutrals)" on every figure; the
code channel names survive in the config comments only.

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

v5 amendment (Tom, 2026-07-27): the ``_losses`` panels are reordered
COMBINED / electron / ion; EVERY bar chart in every figure is sorted by
descending magnitude (context bars below a divider keep their configured
order); the stage-1 and stage-3 axis labels name the main-discharge window
explicitly; the alternate two-window beam chart above is added. Three labels
that mis-stated the physics were corrected against the RHS code -- see the
"v5 LABEL AUDIT" block below for the receipts. No channel membership, no
grouping and no arithmetic changed, so every previously plotted number is
unchanged.

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
import json
import re
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
    # v5: NOT "delivered as thermal". _beam_ionization_sources builds this
    # channel as plasma_heating + radiated + ionization_cost + the ohmic gap
    # booking, so it is the WHOLE CSDA deposit and already contains the two
    # bars below it (v5 label audit item 4). The overlap is 0.05% of the
    # channel on es1_r5_f01_ag26ms, so the bars are still readable as a
    # partition -- but the label no longer claims one.
    "CSDA ray deposit (heating + cost + radiated + ohmic)": {
        "terms": ("beam_power_deposition",), "color": "#0072B2"},
    "consumed in ionization (beam $I_\\mathrm{ion}$ cost)": {
        "terms": ("beam_ionization_cost",), "color": "#009E73"},
    "immediately radiated (beam excitation)": {
        "terms": ("beam_excitation_radiation",), "color": "#CC79A7"},
}
# -----------------------------------------------------------------------------
# STAGE-2 THERMAL BREAKOUT (2026-07-27) -- what is actually INSIDE the
# "delivered as thermal" bar.
#
# `beam_power_deposition` is a LUMP. Read off physics/cathode.py
# `_beam_ionization_sources`, its Ee row is
#
#   beam_power_deposition = (plasma_heating + radiated + ionization_cost)/Vp
#                           + P_ohmic deposited over the cathode-anode gap
#
# and `funcs/_beam_deposition.py` in turn splits `plasma_heating` into four
# physically distinct deliveries, which the solver now saves per cell
# (`cathode_diagnostics/beam_heat_*_W`, marker `beam_csda_active`):
#
#   Coulomb drag      continuous slowing-down on the bulk electrons
#   anomalous drag    quasilinear beam-plasma relaxation (0 unless selected)
#   event residue     <W_sec> per ionization, i.e. the energy dumped into the
#                     newborn secondary ABOVE the I_ion potential cost -- the
#                     "premature dump"
#   terminal dump     the sub-threshold residual banked whole where the
#                     primary's energy crosses E_stop (end of range)
#
# The last two groups below are the radiated and ionization-cost banks, which
# ride INSIDE this bar and are ALSO drawn as their own stage-2 bars: the RHS
# books them positive here and subtracts them again through the separate
# `beam_ionization_cost` / `beam_excitation_radiation` sinks. Showing them
# inside the breakout is deliberate -- the partition is exact only with them,
# and the overlap is a property of the booking, not of this figure.
#
# Each spec names exactly one source:
#   "diag"   per-cell CSDA channel power [W] summed over the column
#   "terms"  volume-integrated plasma-book terms (sink magnitude)
#   "ledger" a reconstructed circuit ledger line
# -----------------------------------------------------------------------------
BEAM_THERMAL_CHANNELS = {
    "Coulomb drag on bulk electrons (continuous)": {
        "diag": ("beam_heat_coulomb_W",), "color": "#0072B2"},
    "anomalous (quasilinear) drag": {
        "diag": ("beam_heat_anomalous_W",), "color": "#56B4E9"},
    "inelastic-event residue (secondary birth above $I_\\mathrm{ion}$)": {
        "diag": ("beam_heat_secondary_W",), "color": "#E69F00"},
    "primary terminal dump (end of range / sub-threshold)": {
        "diag": ("beam_heat_terminal_W",), "color": "#D55E00"},
    "bulk ohmic booked into this term (gap deposition)": {
        "ledger": ("P_ohmic",), "color": "#009E73"},
    "ionization cost carried inside this term (also its own bar)": {
        "terms": ("beam_ionization_cost",), "color": "#CC79A7"},
    "excitation radiation carried inside this term (also its own bar)": {
        "terms": ("beam_excitation_radiation",), "color": "#F0E442"},
}
# Written into the bar labels so a breakout figure is never mistaken for the
# lumped one, and printed when the breakout is unavailable.
BEAM_THERMAL_LUMPED_LABEL = next(iter(BEAM_FATE))
BEAM_RESIDUAL_LABEL = "unreconciled (CSDA > circuit $P_\\mathrm{prim}$)"
BEAM_RESIDUAL_COLOR = "#888888"
BEAM_RESIDUAL_HATCH = "xxx"
# Shown alongside (excluded from the P_prim partition) so the TOTAL thermal
# input to the plasma is readable off the same panel: the ohmic channel is
# pure thermal delivery and is what stage 3 adds to the beam thermal bank.
OHMIC_REFERENCE_LABEL = "bulk ohmic (pure thermal; already inside the CSDA bar)"
OHMIC_REFERENCE_COLOR = "#56B4E9"
OHMIC_REFERENCE_HATCH = "///"

# -----------------------------------------------------------------------------
# STAGE-2 TIME WINDOWS (v4, Tom 2026-07-27).
#
# The whole-discharge stage-2 integral is misleading: absorption fate matters
# at BEAM TURN-ON, and averaging over 19.5 ms of plateau hides the transient.
# Stage 2 is therefore evaluated over three disjoint windows spanning launch
# to drive end:
#
#   (a) launch -> breakdown   first active circuit frame (= plasma launch,
#                             `launch_plasma_after_equilibration`) to the
#                             saved `t_breakdown_trigger` attribute
#   (b) breakdown -> knee     the current rise
#   (c) knee -> drive end     the plateau; drive end = t_breakdown +
#                             tau_discharge (params_json), i.e. the standard
#                             "discharge clock + t_breakdown offset"
#                             convention used across scripts/
#
# The MAIN efficiency figure displays window (a) only (labeled with its
# span); all three are printed and drawn side by side in <stem>_beam_windows.
#
# KNEE DETECTOR -- THE configuration point, nothing about it is hardcoded in
# plot logic. Definition (chosen 2026-07-27, subject to Tom's confirmation):
#
#   plateau level  = median of the boxcar-smoothed discharge current I_tot
#                    over the trailing `plateau_window_frac` of
#                    [t_breakdown, drive end], active frames only
#   knee time      = the FIRST active sample at or after t_breakdown where
#                    the smoothed I_tot reaches `knee_frac` * plateau level
#
# `plateau_window_frac = 1.0` is the literal "plateau median after
# breakdown"; shrinking it toward 0 measures the level on the true plateau
# instead (which raises the level, and so delays the knee). Smoothing is
# applied so a single rippled sample cannot set the knee; on
# es1_r5_f01_ag26ms the knee moves by 0.01 ms between raw and smoothed.
# -----------------------------------------------------------------------------
KNEE_DETECT = {
    "smooth_ms": 0.2,           # boxcar applied to I_tot before detection
    "plateau_window_frac": 1.0,  # trailing fraction of [breakdown, drive end]
    "knee_frac": 0.90,          # fraction of the plateau level defining the knee
}
WINDOW_KEYS = ("a_launch_to_breakdown", "b_breakdown_to_knee",
               "c_knee_to_drive_end")

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
# the convected 3/2 Te (electrons) + 3/2 Ti (ions) per-pair sink over the mesh
# solid fraction eta, NOT the beam-mesh interception (see the
# bypass-termination note in main()). v5 corrected "~2Te" here: the RHS books
# 1.5*Te and 1.5*Ti, and the anode's 2Te ELECTRON SHEATH power is a different
# channel that rides in cathode_surface_loss (v5 label audit, item 1 and 2).
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
    "surface/sheath (cathode + anode + end)": {
        "terms": ("cathode_surface_loss", "surface_loss",
                  "characteristic_boundary"),
        "collector_scalar": True, "color": "#D55E00",
    },
    "anode mesh convected 3/2 T (return current)": {
        "terms": ("anode_collection",),
        "collector_scalar": False, "color": "#E69F00",
    },
    # v4: the two *_cooling channels are the parallel line-radiation pair
    # (electron_ion_cooling <- b_Qei is He+ light, electron_neutral_cooling
    # <- b_Qen is He0 light); see the channel dictionary below.
    "line radiation (ions)": {
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
    "line radiation (neutrals)": {
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
# partner (He+ vs He0); b_Qie alone is the Coulomb equilibration.
#
# v4 (Tom, 2026-07-27) ACTS on that consequence: the two channels are labeled
# as a parallel pair on every figure, legend and breakout --
#
#     electron_ion_cooling      ->  "line radiation (ions)"
#     electron_neutral_cooling  ->  "line radiation (neutrals)"
#
# -- retiring the v3 labels "line radiation (icool)" and "e-n cooling", which
# named the flag and the historical abbreviation rather than the physics. The
# code channel names live on in these config comments (and only here), and
# ei_exchange is labeled "Coulomb transfer e->i (Qie)" so the three are never
# again confusable on a figure.
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

# =============================================================================
# v5 LABEL AUDIT (Tom, 2026-07-27) -- read off the RHS code, receipts below.
# Three v4 labels MIS-STATED the physics. Only the labels changed here; no
# channel membership, no grouping and no arithmetic was touched.
#
# 1. "anode convected 2Te" -> "anode mesh convected 3/2 T".
#    physics/sources.py :: anode_collection_rhs returns
#        Ee = -1.5 * ev_to_erg * derived.Te * plasma_loss_rate
#        Ei = -1.5 * ev_to_erg * derived.Ti * plasma_loss_rate
#    i.e. 3/2 Te per electron and 3/2 Ti per ion -- the convected INTERNAL
#    energy of the collected pair, never 2Te. Measured on es1_r5_f01_ag26ms
#    (main discharge): electron -0.2933 kJ, ion -0.0206 kJ; the ~14x split is
#    the Te/Ti ratio, exactly as 3/2 T predicts and 2Te does not.
#
# 2. "surface/sheath (cathode + end)" -> "(cathode + anode + end)" on the
#    ELECTRON and COMBINED books. The group's `cathode_surface_loss` channel
#    is NOT cathode-only: physics/cathode.py :: _deposit_electrode_power lands
#    BOTH P_cathode_e_thermal AND P_anode_e_thermal into electron_power_loss_W
#    (the anode share split across the two mesh-flanking cells), and
#    cathode_source_terms books that whole sum as Ee. So the anode mesh's 2Te
#    ELECTRON sheath power lives in the sheath group while the anode's 3/2 T
#    CONVECTED loss is the separate `anode_collection` line -- two different
#    channels at the same electrode, NOT a double count of one quantity.
#    The ION panel keeps "(cathode + end)": in this stance
#    cathode_surface_loss books nothing to the ion book at all (receipt 3).
#
# 3. Ion-book surface content. cathode_source_terms gates its particle sink on
#    `face_absorbs` -- with resolved absorbing faces (production) dN_loss stays
#    zero, so its Ei = -1.5 Ti * plasma_loss_rate is IDENTICALLY ZERO and the
#    channel is electron-only. Measured: cathode_surface_loss ion = 0.0000 kJ.
#    NOTHING 2Te-like is ever charged to the ion book. The only 2Te terms in
#    the code are both ELECTRON-book: the electrode sheath above, and
#    sources.py:621 (collector, `sheath_energy_routing`, off in this run).
#    The Te-flavoured ion quantity that exists -- P_cathode_i_thermal =
#    I_i * (Te/2) in _cathode_solver_idriven.py -- is a CIRCUIT-side audit
#    scalar with no RHS consumer anywhere in cablp/solvers.
#
# 4. Stage 2's "delivered as thermal" -> "CSDA ray deposit". physics/cathode.py
#    :: _beam_ionization_sources builds the channel as
#        plasma_heating + radiated + ionization_cost   (+ the ohmic gap term)
#    so `beam_power_deposition` is the WHOLE deposit, not the thermal part:
#    it already contains the other two stage-2 bars and P_ohmic. Measured over
#    the main discharge: of the 5.1334 kJ channel, 0.0107 kJ (0.21%) is
#    P_ohmic, 0.0018 kJ (0.035%) is the ionization cost and 0.0008 kJ (0.016%)
#    the radiated bank. The three stage-2 bars are therefore NOT disjoint --
#    numerically negligible here (0.05%), but the label must not claim a
#    partition the channel does not provide. Reported, not repaired: fixing it
#    would change the plotted numbers, which is out of scope for v5.
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
    "line radiation (ions)": {
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
    # The other half of the line-radiation pair: electron_neutral_cooling
    # <- b_Qen, He0 light (v3 called this "e-n cooling").
    "line radiation (neutrals)": {
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
    # Membership is UNCHANGED from v3 (only the label is new): this panel
    # normalizes the same "radiation" group the power figure draws, and
    # "line radiation (neutrals)" is a separate group there by v3
    # construction, so it is deliberately not folded in here.
    "line radiation (ions)": {
        "terms": ("electron_ion_cooling",), "color": "#0072B2"},
    "recombination radiation": {
        "terms": ("recombination_rad_loss", "recombination_3b_loss"),
        "color": "#56B4E9"},
    "beam excitation radiation": {
        "terms": ("beam_excitation_radiation",), "color": "#CC79A7"},
}
SURFACE_BREAKOUT = {
    # v5: cathode_surface_loss is the ELECTRODE SHEATH electron power, 2Te per
    # electron, and it covers the cathode disc AND the anode mesh together
    # (P_cathode_e_thermal + P_anode_e_thermal; v5 label audit item 2) -- it is
    # not a cathode-only "boundary" line, which is what the v4 label implied.
    # anode_collection is the separate 3/2 T convected collection sink (R4);
    # the end/characteristic outflow + floating collector are the far end.
    # Kept short: this is a HALF-width panel, and a longer y-tick label pushes
    # the axes right until the x-label clips at the figure edge on `slide`
    # (measured with "cathode + anode sheath (2Te, electrons)").
    "cathode + anode sheath (2Te)": {
        "terms": ("cathode_surface_loss",), "color": "#D55E00"},
    "anode mesh collection (3/2 T)": {
        "terms": ("anode_collection",), "color": "#E69F00"},
    "collector / end": {
        "terms": ("surface_loss", "characteristic_boundary"),
        "collector_scalar": True, "color": "#009E73"},
}
# =============================================================================
# THE THREE NORMALIZED LOSS GRAPHS (v4, Tom 2026-07-27) -- <stem>_losses.
#
# These REPLACE the v3 single "cooling breakout" panel. Each graph is a
# separate normalization of the SAME saved terms read from a different book,
# and the point of showing all three is the Coulomb channel's booking:
#
#   ei_exchange books Ee = -q AND Ei = +q (energy.py ::
#   electron_ion_exchange_rhs). It is therefore
#     * a genuine SINK of the electron book -- one of the main Te sinks, and
#       so it is INSIDE the electron graph's normalization;
#     * a genuine SOURCE of the ion book -- shown there as context below the
#       divider, EXCLUDED from the ion loss total (a loss-only denominator);
#     * identically ZERO in the summed book -- so it cannot and does not
#       appear in the combined graph. main() prints the cancellation as a
#       measured number rather than asserting it.
#
# BOOK SPLIT CAVEAT: the floating-collector surface line is a saved SCALAR
# (collector_surface_power_W) with no per-book split, so `collector_scalar`
# is declared on the COMBINED graph only -- keeping that graph the complete
# plasma-side ledger, consistent with PLASMA_LOSS_GROUPS and THERMAL_BUDGET.
# electron + ion therefore reconcile to combined only up to that one line
# (0.007 kJ, ~0.4% of the combined surface group on es1_r5_f01_ag26ms);
# main() prints the reconciliation so the gap is never silent.
#
# `pressure_work` is deliberately in none of the three: it is reversible
# compressional work against the flow, not a sink, and v3 excluded it too.
# =============================================================================
_COULOMB_E_LABEL = "Coulomb transfer e$\\rightarrow$i (Qie)"
_COULOMB_I_LABEL = "Coulomb transfer e$\\rightarrow$i (Qie)\n— heating SOURCE, not a loss"
_LOSS_COLORS = {
    "surface": "#D55E00", "anode": "#E69F00", "line_ion": "#0072B2",
    "line_neutral": "#CC79A7", "ioniz": "#009E73", "coulomb": "#666666",
    "in_coll": "#8E63A6", "other_rad": "#56B4E9",
}
# (i) ELECTRON book. ei_exchange is INSIDE the normalization.
ELECTRON_LOSS_GROUPS = {
    # Includes the anode mesh's 2Te electron sheath power, which rides in
    # cathode_surface_loss via P_anode_e_thermal (v5 label audit item 2).
    "surface/sheath 2Te (cathode + anode + end)": {
        "terms": ("cathode_surface_loss", "surface_loss",
                  "characteristic_boundary"),
        "color": _LOSS_COLORS["surface"]},
    "line radiation (ions)": {
        "terms": ("electron_ion_cooling",), "color": _LOSS_COLORS["line_ion"]},
    "ionization cost (bulk)": {
        "terms": ("ionization_energy_cost",), "color": _LOSS_COLORS["ioniz"]},
    "line radiation (neutrals)": {
        "terms": ("electron_neutral_cooling",),
        "color": _LOSS_COLORS["line_neutral"]},
    _COULOMB_E_LABEL: {
        "terms": ("ei_exchange",), "color": _LOSS_COLORS["coulomb"]},
    # Plain "Te"/"Ti" rather than inline LaTeX: _scalar_key drops $...$, so the
    # electron and ion variants would otherwise collapse to the same key.
    "anode mesh convected 3/2 Te (return current)": {
        "terms": ("anode_collection",), "color": _LOSS_COLORS["anode"]},
    "other radiation": {
        "terms": ("recombination_rad_loss", "recombination_3b_loss",
                  "beam_excitation_radiation"),
        "color": _LOSS_COLORS["other_rad"]},
}
# (ii) ION book. The ion-neutral collisional channel dominates; ei_exchange
# is the excluded context bar (see ION_LOSS_REFERENCE).
ION_LOSS_GROUPS = {
    "ion–neutral collisional (cx + elastic)": {
        "terms": ("ion_neutral_collision", "ion_charge_exchange",
                  "ion_neutral_drag", "ion_neutral_frictional_heating",
                  "ion_neutral_thermalization"),
        "color": _LOSS_COLORS["in_coll"]},
    # Ion-book only: cathode_surface_loss contributes NOTHING here (its Ei is
    # identically zero with resolved absorbing faces -- v5 label audit item 3),
    # so this group is the characteristic-boundary outflow at the cathode and
    # collector faces. No anode content, hence no anode in the label.
    "surface/sheath (cathode + end)": {
        "terms": ("cathode_surface_loss", "surface_loss",
                  "characteristic_boundary"),
        "color": _LOSS_COLORS["surface"]},
    "anode mesh convected 3/2 Ti (return current)": {
        "terms": ("anode_collection",), "color": _LOSS_COLORS["anode"]},
    # beam_excitation_radiation books Ee only, so the ion side of "other
    # radiation" is the recombination pair alone.
    "other radiation": {
        "terms": ("recombination_rad_loss", "recombination_3b_loss"),
        "color": _LOSS_COLORS["other_rad"]},
}
ION_LOSS_REFERENCE = {
    "label": _COULOMB_I_LABEL,
    "terms": ("ei_exchange",),
    # ei_exchange books +q on the ion side, so this bar is the SOURCE part;
    # clipping it to the sink part (the default) would render it as 0.0%.
    "part": "source",
    "color": _LOSS_COLORS["coulomb"],
    "hatch": "////",
}
# (iii) COMBINED book (electron + ion). ei_exchange cancels identically and
# is absent by construction, so every bar here is energy that actually LEFT
# the plasma.
COMBINED_LOSS_GROUPS = {
    "surface/sheath (cathode + anode + end)": {
        "terms": ("cathode_surface_loss", "surface_loss",
                  "characteristic_boundary"),
        "collector_scalar": True, "color": _LOSS_COLORS["surface"]},
    "line radiation (ions)": {
        "terms": ("electron_ion_cooling",), "color": _LOSS_COLORS["line_ion"]},
    "ionization cost (bulk)": {
        "terms": ("ionization_energy_cost",), "color": _LOSS_COLORS["ioniz"]},
    "line radiation (neutrals)": {
        "terms": ("electron_neutral_cooling",),
        "color": _LOSS_COLORS["line_neutral"]},
    "ion–neutral collisional (cx + elastic)": {
        "terms": ("ion_neutral_collision", "ion_charge_exchange",
                  "ion_neutral_drag", "ion_neutral_frictional_heating",
                  "ion_neutral_thermalization"),
        "color": _LOSS_COLORS["in_coll"]},
    "anode mesh convected 3/2 T (return current)": {
        "terms": ("anode_collection",), "color": _LOSS_COLORS["anode"]},
    "other radiation": {
        "terms": ("recombination_rad_loss", "recombination_3b_loss",
                  "beam_excitation_radiation"),
        "color": _LOSS_COLORS["other_rad"]},
}
# The three graphs in drawing order: (console key, book, groups, reference
# spec or None, x-axis label). `reference` bars are drawn below a divider and
# are NOT part of the normalization.
#
# Keep each x-label LINE under ~46 characters: the y tick labels eat the left
# third of the panel, so a longer line is wider than its own axes and clips
# at the figure edge on `slide` (measured, not guessed).
#
# v5 PANEL ORDER (Tom, 2026-07-27): COMBINED first, then electron, then ion.
# The combined book is the only one of the three that is a complete statement
# of energy that actually LEFT the plasma, so it reads as the headline and the
# two per-book decompositions below it explain how that total is split.
LOSS_GRAPHS = (
    ("combined", "total", COMBINED_LOSS_GROUPS, None,
     "% of COMBINED plasma losses  (main discharge)\n"
     "Coulomb transfer cancels between the books"),
    ("electron", "electron", ELECTRON_LOSS_GROUPS, None,
     "% of ELECTRON-book losses  (main discharge)\n"
     "Coulomb transfer INCLUDED — a real $T_e$ sink"),
    ("ion", "ion", ION_LOSS_GROUPS, ION_LOSS_REFERENCE,
     "% of ION-book losses  (main discharge)\n"
     "Coulomb transfer excluded from the total"),
)

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
        # Discharge-clock anchors for the stage-2 windows (v4). The abs-time
        # convention is the one used across scripts/: discharge clock zero is
        # `t_breakdown_trigger`, and the drive ends tau_discharge later.
        data["t_breakdown_ms"] = 1e3 * float(
            f.attrs.get("t_breakdown_trigger", np.nan))
        params = {}
        if "params_json" in f.attrs:
            params = json.loads(f.attrs["params_json"])
        tau = params.get("tau_discharge", np.nan)
        data["tau_discharge_ms"] = 1e3 * float(
            tau if tau is not None else np.nan)
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
        # --- CSDA channel breakout (2026-07-27) -----------------------------
        # Per-cell channel power [W] -> column totals [W]. Absent on runs
        # saved before the instrumentation, and present-but-zero on runs whose
        # deposition did not go through the CSDA module, so the marker (not
        # mere key presence) decides whether the breakout is meaningful.
        channels = {}
        for name in ("beam_heat_coulomb_W", "beam_heat_anomalous_W",
                     "beam_heat_secondary_W", "beam_heat_terminal_W"):
            if name in cd:
                channels[name] = np.nan_to_num(
                    np.asarray(cd[name], dtype=float), nan=0.0
                ).sum(axis=1)
        data["beam_channels"] = channels
        data["beam_csda_active"] = (
            np.nan_to_num(np.asarray(cd["beam_csda_active"], dtype=float),
                          nan=0.0)
            if "beam_csda_active" in cd else np.zeros(n)
        )
        data["have_beam_channels"] = bool(
            len(channels) == 4 and np.any(data["beam_csda_active"] > 0.0)
        )
        # Exit ledger of the ray: intercepted at the anode mesh, and streaming
        # out of the far end. Neither has an RHS consumer (see main()).
        for name in ("beam_anode_intercepted_W", "beam_transmitted_W",
                     "beam_transmitted_flux_per_s"):
            data[name] = _sum_sides(cd, name, n)
        data["have_beam_exit_ledger"] = any(
            f"{side}_beam_anode_intercepted_W" in cd for side in SIDES
        )
        # Volume-integrated plasma-book terms for every term any configured
        # group or breakout references. Three views are kept:
        #   plasma_terms          -- electron + ion books (the plasma total)
        #   plasma_terms_electron -- electron book only
        #   plasma_terms_ion      -- ion book only
        # All three are needed for the v4 loss graphs: ei_exchange books -q to
        # Ee and +q to Ei, so it is a sink in the electron view, a source in
        # the ion view, and EXACTLY zero in the summed view.
        Vp = np.asarray(f["geometry/plasma_volume_cm3"], dtype=float)
        wanted = set()
        for cfg in (PLASMA_LOSS_GROUPS, RADIATION_BREAKOUT, SURFACE_BREAKOUT,
                    THERMAL_BUDGET, BEAM_FATE, BEAM_THERMAL_CHANNELS,
                    ELECTRON_LOSS_GROUPS, ION_LOSS_GROUPS,
                    COMBINED_LOSS_GROUPS):
            for spec in cfg.values():
                # BEAM_THERMAL_CHANNELS specs may source from "diag"/"ledger"
                # instead, so "terms" is optional here.
                wanted.update(spec.get("terms", ()))
        wanted.update(ION_LOSS_REFERENCE["terms"])
        terms = {}
        terms_e = {}
        terms_i = {}
        for term in sorted(wanted):
            total = np.zeros(n)
            electron = np.zeros(n)
            ion = np.zeros(n)
            for book in ("electron_energy_terms_W_cm3", "ion_energy_terms_W_cm3"):
                if book in f and term in f[book]:
                    contrib = np.sum(
                        np.asarray(f[book][term], dtype=float) * Vp[None, :], axis=1
                    )
                    total += contrib
                    if book.startswith("electron"):
                        electron += contrib
                    else:
                        ion += contrib
            terms[term] = total
            terms_e[term] = electron
            terms_i[term] = ion
        data["plasma_terms"] = terms
        data["plasma_terms_electron"] = terms_e
        data["plasma_terms_ion"] = terms_i
    return data


_BOOK_VIEW = {
    "total": "plasma_terms",
    "electron": "plasma_terms_electron",
    "ion": "plasma_terms_ion",
}


def plasma_group_sum(d, spec, book="total"):
    """Signed sum of a plasma-book group's terms [W] (+ collector if declared).

    `book` selects the electron-only, ion-only or summed view. The saved
    collector surface line has no per-book split, so `collector_scalar` is
    honored on the summed view only (it is declared only there).
    """
    view = d[_BOOK_VIEW[book]]
    total = sum(view[term] for term in spec["terms"])
    if spec.get("collector_scalar") and book == "total":
        total = total - d["collector_surface_W"]
    return total


def loss_group_energy(d, spec, book, sel, t_s):
    """Integrated one-signed magnitude [J] of one group in one book.

    Signed group sum first, then clipped -- the same convention the v3 groups
    use, and the reason the Coulomb channel vanishes from the combined book
    rather than appearing twice with opposite signs. ``spec["part"] ==
    "source"`` takes the POSITIVE part instead of the sink part; the ion
    book's Coulomb context bar needs it, since ei_exchange books +q there and
    would otherwise clip to exactly zero.
    """
    y = plasma_group_sum(d, spec, book)
    if spec.get("part") == "source":
        mag = np.clip(y, 0.0, None)
    else:
        mag = np.abs(np.minimum(y, 0.0))
    return np.trapezoid(np.where(sel, mag, 0.0), t_s)


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
    """Console `key=value` name from a (possibly hard-wrapped) figure label.

    v4 keeps the PARENTHETICAL. v3 truncated the label at " (", which the
    new parallel line-radiation naming makes ambiguous: "line radiation
    (ions)" and "line radiation (neutrals)" would both collapse to
    `line_radiation` and silently overwrite each other in the printed
    scalars. Inline LaTeX is dropped first so keys stay plain identifiers.
    """
    flat = re.sub(r"\$[^$]*\$", " ", label.replace("\n", " "))
    return re.sub(r"[^0-9A-Za-z]+", "_", flat).strip("_")


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
# Stage-2 discharge windows (v4)
# =============================================================================
def find_current_knee(d, t_bd_ms, t_end_ms):
    """Knee time [ms] on the discharge-current trace, per ``KNEE_DETECT``.

    Returns ``(t_knee_ms, plateau_A, threshold_A)``. The definition is stated
    in full in the KNEE_DETECT comment block; briefly, the knee is the first
    active sample at or after breakdown whose boxcar-smoothed I_tot reaches
    ``knee_frac`` of the plateau median. Falls back to the drive end (with a
    printed warning) if the threshold is never reached.
    """
    t = d["time_ms"]
    act = d["active"]
    dt_ms = float(np.median(np.diff(t)))
    width = max(1, int(round(KNEE_DETECT["smooth_ms"] / dt_ms)))
    I = _boxcar(np.where(act, d["I_tot"], np.nan), width)

    frac = float(KNEE_DETECT["plateau_window_frac"])
    plateau_lo = t_end_ms - frac * (t_end_ms - t_bd_ms)
    in_plateau = act & (t >= plateau_lo) & (t <= t_end_ms)
    if not in_plateau.any():
        raise SystemExit("knee detection: no active frames in the plateau "
                         f"window [{plateau_lo:.3f}, {t_end_ms:.3f}] ms")
    plateau = float(np.nanmedian(I[in_plateau]))
    threshold = KNEE_DETECT["knee_frac"] * plateau

    reached = act & (t >= t_bd_ms) & (t <= t_end_ms) & (I >= threshold)
    if not reached.any():
        print("knee_WARNING=discharge current never reaches "
              f"{KNEE_DETECT['knee_frac']:.2f} of the plateau median; "
              "knee falls back to the drive end")
        return t_end_ms, plateau, threshold
    return float(t[reached][0]), plateau, threshold


def discharge_windows(d):
    """The three stage-2 windows as ``(key, label, lo_ms, hi_ms, mask)``.

    (a) plasma launch -> breakdown, (b) breakdown -> current knee,
    (c) knee -> drive end. "Plasma launch" is taken as the first ACTIVE
    circuit frame: the cathode circuit only solves once the plasma exists, so
    that sample is the launch under `launch_plasma_after_equilibration`.
    """
    t = d["time_ms"]
    act = d["active"]
    t_launch = float(t[act][0])

    t_bd = d["t_breakdown_ms"]
    if not np.isfinite(t_bd):
        # Fall back to the recorded phase transition if the attribute is
        # absent/NaN (e.g. a run that never triggered breakdown).
        events = [1e3 * ts for name, ts in d["phase_events"]
                  if name == "main_discharge"]
        if not events:
            raise SystemExit("no t_breakdown_trigger attribute and no "
                             "main_discharge phase event: cannot build the "
                             "stage-2 windows")
        t_bd = float(events[0])

    t_end = t_bd + d["tau_discharge_ms"]
    if not np.isfinite(t_end):
        events = [1e3 * ts for name, ts in d["phase_events"]
                  if name == "afterglow"]
        t_end = float(events[0]) if events else float(t[act][-1])
    t_end = min(t_end, float(t[act][-1]))

    t_knee, plateau, threshold = find_current_knee(d, t_bd, t_end)
    d["knee_ms"] = t_knee
    d["knee_plateau_A"] = plateau
    d["knee_threshold_A"] = threshold
    d["t_breakdown_used_ms"] = t_bd
    d["t_drive_end_ms"] = t_end
    d["t_launch_ms"] = t_launch

    bounds = ((t_launch, t_bd), (t_bd, t_knee), (t_knee, t_end))
    labels = ("(a) launch → breakdown", "(b) breakdown → knee",
              "(c) knee → drive end")
    out = []
    for key, label, (lo, hi) in zip(WINDOW_KEYS, labels, bounds):
        mask = act & (t >= lo) & (t <= hi)
        out.append((key, label, lo, hi, mask))
    return out


def two_window_split(windows):
    """The ALTERNATE simple two-window split (v5, Tom 2026-07-27).

    Tom may prefer this over the three-window version, so both are produced.
    It merges (a) and (b) into a single "beam turn-on" window running from
    plasma launch to the current knee, and keeps (c) as "the rest of the
    discharge":

        turn-on   launch -> knee     (the whole rise: pre-breakdown fill,
                                      breakdown, and the current ramp)
        rest      knee  -> drive end (the plateau)

    The knee is the same ``KNEE_DETECT`` boundary the three-window figure
    uses, so the two charts are two groupings of ONE detection, not two
    different detections. Returned in the same
    ``(key, label, lo_ms, hi_ms, mask)`` shape as ``discharge_windows`` so
    ``make_beam_windows_figure`` renders either without a special case.
    """
    (_, _, a_lo, _, a_mask), (_, _, _, b_hi, b_mask), c = windows
    _, _, c_lo, c_hi, c_mask = c
    # Labels are kept SHORT: make_beam_windows_figure appends the "[lo - hi
    # ms]" span to the same line, and on the `slide` profile a label much past
    # ~45 characters is wider than its own axes and clips at the figure edge
    # (measured -- "rest of discharge: knee → drive end" did exactly that).
    return [
        ("turn_on_launch_to_knee", "beam turn-on: launch → knee",
         a_lo, b_hi, a_mask | b_mask),
        ("rest_knee_to_drive_end", "rest: knee → drive end",
         c_lo, c_hi, c_mask),
    ]




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

    v5 (Tom, 2026-07-27): every bar chart in every figure is sorted by
    DESCENDING MAGNITUDE. This is the single rendering choke point, so the
    ordering is applied here once rather than at each call site and the
    configuration dicts keep their semantic (documentation) order. |value| is
    the sort key, not the signed value, so the negative slices that are real
    and large -- stage 1's round-trip circulation, stage 2's unreconciled
    CSDA remainder -- rank by how much of the denominator they move rather
    than sinking to the bottom. The trailing ``n_excluded`` context bars are
    NOT sorted against the loss bars: they stay in their configured order
    below the divider, since they are a different kind of quantity.
    """
    n_sorted = len(items) - n_excluded
    items = (sorted(items[:n_sorted], key=lambda it: -abs(it[1]))
             + list(items[n_sorted:]))
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


def beam_channel_series(d, spec):
    """Series [W] for one BEAM_THERMAL_CHANNELS spec, whichever source it names."""
    if "diag" in spec:
        return sum(d["beam_channels"][k] for k in spec["diag"])
    if "ledger" in spec:
        return sum(d[k] for k in spec["ledger"])
    return sum(d["plasma_terms"][t] for t in spec["terms"])


def beam_thermal_channel_energies(d, E):
    """Breakout of the stage-2 thermal bar [J], or ``None`` on old/non-CSDA runs.

    The channels partition ``int(beam_power_deposition)dt`` exactly (see
    BEAM_THERMAL_CHANNELS); the caller checks that closure rather than
    assuming it.
    """
    if not d["have_beam_channels"]:
        return None
    return {
        label: abs(E(beam_channel_series(d, spec)))
        for label, spec in BEAM_THERMAL_CHANNELS.items()
    }


def beam_fate_items(d, ledger, sel, t_s):
    """Stage-2 bar items over an arbitrary window mask.

    Returns ``(items, E_prim, E_ohmic, E_resid)`` where `items` is ready for
    ``_share_barh`` with ONE trailing excluded entry (the ohmic reference).

    When the run carries the CSDA channel instrumentation, the single
    "delivered as thermal" bar is replaced by its BEAM_THERMAL_CHANNELS
    breakout; the bars it replaces sum to the same energy, so the P_prim
    partition, the residual and every other bar are unchanged. Runs without
    it keep the lumped bar (main() prints the reason once).
    """
    def E(y):
        return np.trapezoid(np.where(sel, y, 0.0), t_s)

    E_prim = E(ledger["P_prim"])
    E_ohmic = E(ledger["P_ohmic"])
    fate = beam_fate_energies(d, E)
    channels = beam_thermal_channel_energies(d, E)
    items = []
    for label, Ei in fate.items():
        if label == BEAM_THERMAL_LUMPED_LABEL and channels is not None:
            items += [
                (f"{sub} [thermal]", 100.0 * Esub / E_prim,
                 BEAM_THERMAL_CHANNELS[sub]["color"])
                for sub, Esub in channels.items()
            ]
            continue
        items.append((label, 100.0 * Ei / E_prim, BEAM_FATE[label]["color"]))
    E_resid = E_prim - sum(fate.values())
    items.append((BEAM_RESIDUAL_LABEL, 100.0 * E_resid / E_prim,
                  BEAM_RESIDUAL_COLOR, BEAM_RESIDUAL_HATCH))
    items.append((OHMIC_REFERENCE_LABEL, 100.0 * E_ohmic / E_prim,
                  OHMIC_REFERENCE_COLOR, OHMIC_REFERENCE_HATCH))
    return items, E_prim, E_ohmic, E_resid


def make_efficiency_figure(d, ledger, window, stage2, profile):
    """Three-stage cascade: source -> absorbed & its fate -> thermal budget.

    STAGE 1 (top row, two denominators): the GROSS circuit components, with
    no netting or folding. Left is shares of E_wall = int(I_tot*V_bank)dt,
    which includes the compliance/mesh resistor dissipation so P_comp is an
    explicit component; right is the same components as shares of
    E_load = int(I_tot*V_b)dt. The round trip is its own hatched NEGATIVE
    slice (it returns energy to the circuit), so each chart's bars sum
    algebraically to exactly 100% of its denominator.

    STAGE 2 (middle): the fate of the absorbed beam power int(P_prim)dt by
    collision outcome -- thermal / ionization consumption / prompt radiation
    -- with the unreconciled CSDA-vs-circuit remainder as its own labeled
    bar, and the ohmic delivery alongside (excluded from the partition) so
    the total thermal input is readable. On instrumented runs the thermal
    entry is drawn as its BEAM_THERMAL_CHANNELS breakout (Coulomb drag /
    anomalous drag / inelastic-event residue / terminal dump, plus the ohmic
    and cost/radiation banks that ride inside the same term) instead of one
    lumped bar; the bars it replaces sum to the same energy.

    v4: stage 2 is integrated over `stage2` -- window (a), plasma launch to
    breakdown -- NOT the main-discharge window, because absorption fate
    matters at beam turn-on and the whole-discharge integral averages the
    transient away. The window's span is written into the axis label, and
    <stem>_beam_windows shows all three windows side by side.

    STAGE 3 (bottom): where that thermal energy goes -- the plasma-book
    loss groups with the anode convected-3/2-T return-current channel broken
    out, plus an explicit residual so the budget closes.
    """
    act = d["active"]
    sel = window & act
    t_s = d["time_ms"] * 1e-3
    journal = profile == "journal"
    _, s2_label, s2_lo, s2_hi, s2_mask = stage2

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

    # Stage 3's denominator stays on the MAIN-DISCHARGE window (it is the
    # budget of the thermal energy the discharge actually delivered); only
    # stage 2's own partition moves to the turn-on window.
    fate = beam_fate_energies(d, E)
    E_beam_thermal = fate[next(iter(BEAM_FATE))]  # first entry = thermal bank
    E_thermal = E_beam_thermal + E_ohmic

    # Stage 1's two denominators pair naturally side by side (their labels are
    # short); stages 2 and 3 take the full width, which is what keeps their
    # long channel names and x-labels from colliding at either profile.
    # The breakout adds rows to stage 2, so its panel and the figure grow with
    # the bar count instead of squeezing the same box.
    items_b, _, _, _ = beam_fate_items(d, ledger, s2_mask, t_s)
    extra_rows = max(0, len(items_b) - 5)
    if journal:
        figsize = (fs.JOURNAL_WIDTHS["aip_double"], 8.2 + 0.30 * extra_rows)
    else:
        figsize = (12.9, 13.2 + 0.45 * extra_rows)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 0.9 + 0.18 * extra_rows, 1.15]
    )
    ax_w = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])
    ax_t = fig.add_subplot(gs[2, :])

    # --- stage 1: gross components + the circulation slice, two denominators
    # v5 (Tom, v4 review flag): stages 1 and 3 say "(main discharge)" on the
    # axis. Only stage 2 moved to the turn-on window in v4, and without the
    # window named on every panel the figure reads as if all three shared one.
    # These are HALF-WIDTH panels whose long y-tick labels ("beam bypass (to
    # surfaces, ...)") push the axes right, so the centered x-label overruns
    # the figure edge easily -- the same failure the stage-2 and breakout
    # labels are already wrapped against. Each line is kept under ~15
    # characters, which is what the `slide` profile's larger type needs; the
    # 3-line and 2-line variants both clipped on slide when measured, so the
    # denominator, its role and the window each get their own line.
    for ax, denom, with_comp, xlabel in (
        (ax_w, E_wall, True,
         "stage 1 — % of\n$E_\\mathrm{wall}$\n(bank source)\nmain discharge"),
        (ax_l, E_load, False,
         "stage 1 — % of\n$E_\\mathrm{load}$\n(load source)\nmain discharge"),
    ):
        items = [(lab, 100.0 * Ei / denom, c) for lab, Ei, c in comps]
        if with_comp:
            items.append((COMP_LABEL, 100.0 * E_comp / denom, COMP_COLOR))
        items.append((CIRCULATION_LABEL, 100.0 * E_circ / denom,
                      CIRCULATION_COLOR, CIRCULATION_HATCH))
        _share_barh(ax, items)
        ax.set_xlabel(xlabel)

    # --- stage 2: fate of the absorbed beam power (partition of P_prim) -----
    # Integrated over the TURN-ON window only (v4); see the docstring.
    _share_barh(ax_b, items_b, n_excluded=1)
    thermal_note = ("thermal bar broken out by CSDA channel"
                    if d["have_beam_channels"] else "thermal bar lumped")
    ax_b.set_xlabel("stage 2 — % of ABSORBED beam power\n"
                    "(fate by collision outcome, at beam turn-on)\n"
                    f"{thermal_note}\n"
                    f"window {s2_label}   [{s2_lo:.2f} – {s2_hi:.2f} ms]")

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
                    "(where the thermal energy goes, main discharge)")
    return fig


def make_beam_windows_figure(d, ledger, windows, profile):
    """Stage-2 decomposition over a list of discharge windows, stacked.

    One ``_share_barh`` panel per window, same channels and colors as the
    efficiency figure's stage 2, each normalized by its OWN window's
    int(P_prim)dt. The comparison is the point: the unreconciled
    CSDA-vs-circuit remainder is largest at turn-on and shrinks as the
    discharge settles, which the single whole-discharge integral hides.

    v5: the panel count follows ``len(windows)``, so this renders both the
    three-window figure and the alternate two-window split
    (``two_window_split``) with no special case. Panel height is held fixed
    per window rather than dividing a fixed figure height, so the bars keep
    the same scale in both.

    On instrumented runs each panel carries the CSDA thermal breakout, so the
    window comparison also shows how the deposition MECHANISM shifts across
    the discharge (drag vs event residue vs terminal dump), not just how much
    lands as heat. The breakout adds rows to every panel, so the per-window
    panel height grows with the bar count (lumped runs keep the v5 heights).
    """
    t_s = d["time_ms"] * 1e-3
    journal = profile == "journal"
    n_win = len(windows)
    per_window = [beam_fate_items(d, ledger, mask, t_s)
                  for (_, _, _, _, mask) in windows]
    extra_rows = max(0, max(len(it[0]) for it in per_window) - 5)
    per_panel = 2.2 if journal else 3.47
    per_panel += (0.28 if journal else 0.42) * extra_rows
    if journal:
        figsize = (fs.JOURNAL_WIDTHS["aip_double"], per_panel * n_win)
    else:
        figsize = (12.9, per_panel * n_win)
    fig, axes = plt.subplots(n_win, 1, figsize=figsize,
                             constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (_, label, lo, hi, _), (items, E_prim, _, _) in zip(
        axes, windows, per_window
    ):
        _share_barh(ax, items, n_excluded=1)
        # Two lines: as one, this label is wider than its own axes and clips
        # at the figure edge on `slide`.
        ax.set_xlabel(
            f"{label}   [{lo:.2f} – {hi:.2f} ms]\n"
            f"% of $E_\\mathrm{{prim}}$ = {E_prim / 1e3:.3f} kJ")
    return fig


def make_losses_figure(d, window, profile):
    """The three normalized loss graphs (v4): electron, ion, combined books.

    Each panel normalizes by its OWN book's summed loss total over the
    main-discharge window, so the three are read as three separate
    accountings rather than one split. The Coulomb transfer ei_exchange is
    inside the electron total, drawn as an excluded context bar on the ion
    panel, and absent from the combined panel where it cancels identically
    (see the LOSS_GRAPHS comment block).
    """
    sel = window
    t_s = d["time_ms"] * 1e-3
    journal = profile == "journal"

    if journal:
        figsize = (fs.JOURNAL_WIDTHS["aip_double"], 7.6)
    else:
        figsize = (12.9, 12.4)
    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    for ax, (_, book, groups, reference, xlabel) in zip(axes, LOSS_GRAPHS):
        energies = {label: loss_group_energy(d, spec, book, sel, t_s)
                    for label, spec in groups.items()}
        total = sum(energies.values())
        items = [(label, 100.0 * Ei / total if total > 0 else 0.0,
                  groups[label]["color"]) for label, Ei in energies.items()]
        n_excluded = 0
        if reference is not None:
            ref = loss_group_energy(d, reference, book, sel, t_s)
            items.append((reference["label"],
                          100.0 * ref / total if total > 0 else 0.0,
                          reference["color"], reference["hatch"]))
            n_excluded = 1
        _share_barh(ax, items, n_excluded=n_excluded)
        ax.set_xlabel(xlabel)
    return fig


def make_breakout_figure(d, window, profile):
    """Percent breakouts WITHIN the radiation and surface/sheath groups.

    Sub-components of each group, normalized by its own group total; all
    energies are integrated sink magnitudes over the main-discharge window.
    (The v3 electron-cooling panel that used to sit above these is superseded
    by the three loss graphs in <stem>_losses.)
    """
    sel = window
    t_s = d["time_ms"] * 1e-3
    journal = profile == "journal"

    def E_sink(spec):
        y = plasma_group_sum(d, spec)
        mag = np.abs(np.minimum(y, 0.0))  # sink part only
        return np.trapezoid(np.where(sel, mag, 0.0), t_s)

    if journal:
        figsize = (fs.JOURNAL_WIDTHS["aip_double"], 2.6)
    else:
        figsize = (12.9, 4.2)
    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=figsize,
                                     constrained_layout=True)

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
        help="boxcar width [ms] applied to the PLOTTED time series of the "
        "power figure only (the per-solve circuit scalars carry real "
        "high-frequency ripple that buries the plateau lines); the closure "
        "residual and every printed scalar always use the raw data, and the "
        "bar figures are always integrated from the raw data. The knee "
        "detector has its own independent smoothing in KNEE_DETECT. "
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
    # Stage-2 windows (v4); also records the knee scalars onto `d`.
    windows = discharge_windows(d)
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

    # --- STAGE 2 THERMAL BREAKOUT: what is inside beam_power_deposition -----
    print("beam_thermal_breakout_def=beam_power_deposition="
          "coulomb_drag+anomalous_drag+event_residue+terminal_dump"
          "+ionization_cost+excitation_radiation+P_ohmic; the cost and "
          "radiation banks ride inside this term AND are subtracted again by "
          "their own sinks, so they appear twice in stage 2 by construction")
    if not d["have_beam_channels"]:
        print("beam_thermal_breakout=UNAVAILABLE (no CSDA channel "
              "instrumentation in this file: either it predates the saved "
              "cathode_diagnostics/beam_heat_*_W channels, or its deposition "
              "did not run through the CSDA module); stage 2 keeps the "
              "LUMPED thermal bar")
    else:
        chan = beam_thermal_channel_energies(d, E)
        for label, Ei in chan.items():
            print(f"beam_thermal_{_scalar_key(label)}_kJ={Ei / 1e3:.4f} "
                  f"frac_thermal={Ei / E_beam_thermal:.4f} "
                  f"frac_P_prim={Ei / E_prim:.4f}")
        # The partition is CHECKED against the term it claims to decompose.
        chan_resid = E_beam_thermal - sum(chan.values())
        print(f"beam_thermal_breakout_closure_kJ={chan_resid / 1e3:.3e} "
              f"rel={abs(chan_resid) / E_beam_thermal:.3e}")
        if abs(chan_resid) > 0.01 * E_beam_thermal:
            # MEASURED, not assumed. The channels are the CSDA module's own
            # banks for the cached cathode solve; the term is the RHS row the
            # fluid integrated. They are built from the same quantities, so a
            # systematic offset is a real disagreement between the diagnostic
            # solve and the deposited RHS, not a plotting artifact -- reported
            # here rather than normalized away.
            print("beam_thermal_breakout_WARNING=channel sum does NOT "
                  f"reproduce beam_power_deposition (channels/term="
                  f"{sum(chan.values()) / E_beam_thermal:.4f}); the breakout "
                  "bars are the module's banks, the lumped bar is the RHS row "
                  "(reported, not fixed here)")
        # Per-window mechanism shift: the number the turn-on window is for.
        for key, _, lo, hi, mask in windows:
            def Ew(y, m=mask):
                return np.trapezoid(np.where(m, y, 0.0), t_s)

            wchan = beam_thermal_channel_energies(d, Ew)
            wtot = sum(wchan.values())
            print(f"beam_thermal_window_{key}=[{lo:.4f},{hi:.4f}]ms "
                  f"E_thermal_kJ={wtot / 1e3:.4f}")
            for label, Ei in wchan.items():
                print(f"  beam_thermal_window_{key}_"
                      f"{_scalar_key(label)}_pct={100.0 * Ei / wtot:.2f}")

    # --- STAGE 2 BY WINDOW (v4): the whole-discharge integral above is the
    # main-discharge average and hides the turn-on transient. -----------------
    print("stage2_windows_def=(a) plasma launch (first active circuit frame) "
          "-> t_breakdown_trigger; (b) breakdown -> current knee; "
          "(c) knee -> drive end (= t_breakdown + tau_discharge)")
    print(f"knee_def=first active sample at/after breakdown where the "
          f"{KNEE_DETECT['smooth_ms']:.2f} ms boxcar-smoothed I_tot reaches "
          f"{KNEE_DETECT['knee_frac']:.2f} of the median I_tot over the "
          f"trailing {KNEE_DETECT['plateau_window_frac']:.2f} of "
          f"[breakdown, drive end]")
    print(f"knee_plateau_A={d['knee_plateau_A']:.1f} "
          f"knee_threshold_A={d['knee_threshold_A']:.1f}")
    print(f"t_launch_ms={d['t_launch_ms']:.4f} "
          f"t_breakdown_ms={d['t_breakdown_used_ms']:.4f} "
          f"t_knee_ms={d['knee_ms']:.4f} "
          f"t_drive_end_ms={d['t_drive_end_ms']:.4f}")
    for key, label, lo, hi, mask in windows:
        items, Ew_prim, Ew_ohmic, Ew_resid = beam_fate_items(
            d, ledger, mask, t_s)
        print(f"window_{key}=[{lo:.4f},{hi:.4f}]ms span_ms={hi - lo:.4f} "
              f"E_prim_kJ={Ew_prim / 1e3:.4f} E_ohmic_kJ={Ew_ohmic / 1e3:.4f}")
        for item in items[:-2]:
            print(f"  window_{key}_{_scalar_key(item[0])}_pct={item[1]:.2f}")
        print(f"  window_{key}_residual_kJ={Ew_resid / 1e3:.4f} "
              f"pct={100.0 * Ew_resid / Ew_prim:.2f}")

    # --- ALTERNATE two-window split (v5): turn-on vs the rest ---------------
    two = two_window_split(windows)
    print("two_window_def=(turn-on) plasma launch -> current knee, i.e. windows "
          "(a)+(b) merged; (rest) knee -> drive end, i.e. window (c). Same "
          "KNEE_DETECT boundary as the three-window split, regrouped")
    for key, label, lo, hi, mask in two:
        items, Ew_prim, Ew_ohmic, Ew_resid = beam_fate_items(
            d, ledger, mask, t_s)
        print(f"two_window_{key}=[{lo:.4f},{hi:.4f}]ms span_ms={hi - lo:.4f} "
              f"E_prim_kJ={Ew_prim / 1e3:.4f} E_ohmic_kJ={Ew_ohmic / 1e3:.4f}")
        for item in items[:-2]:
            print(f"  two_window_{key}_{_scalar_key(item[0])}_pct={item[1]:.2f}")
        print(f"  two_window_{key}_residual_kJ={Ew_resid / 1e3:.4f} "
              f"pct={100.0 * Ew_resid / Ew_prim:.2f}")

    # --- v4 loss graphs: shares per book over the main-discharge window ------
    print("loss_graphs_def=electron book (Coulomb transfer INSIDE the "
          "normalization), ion book (Coulomb transfer excluded, shown as the "
          "heating-source context bar), combined book (Coulomb transfer "
          "cancels identically); pressure_work excluded from all three")
    for gkey, book, groups, reference, _ in LOSS_GRAPHS:
        energies = {label: loss_group_energy(d, spec, book, sel, t_s)
                    for label, spec in groups.items()}
        total = sum(energies.values())
        print(f"loss_{gkey}_total_kJ={total / 1e3:.4f}")
        for label, Ei in energies.items():
            print(f"  loss_{gkey}_{_scalar_key(label)}_kJ={Ei / 1e3:.4f} "
                  f"pct={100.0 * Ei / total:.2f}")
        if reference is not None:
            ref = loss_group_energy(d, reference, book, sel, t_s)
            print(f"  loss_{gkey}_reference_{_scalar_key(reference['label'])}"
                  f"_kJ={ref / 1e3:.4f} pct_of_total={100.0 * ref / total:.2f} "
                  "(EXCLUDED from the total)")
    # The combined book's Coulomb cancellation is MEASURED, not asserted.
    ei_cancel = E(d["plasma_terms"]["ei_exchange"])
    ei_electron = E(np.abs(np.minimum(
        d["plasma_terms_electron"]["ei_exchange"], 0.0)))
    print(f"loss_combined_ei_exchange_cancellation_kJ={ei_cancel / 1e3:.3e} "
          f"rel_to_electron_book={abs(ei_cancel) / ei_electron:.3e}")
    # electron + ion reconcile to combined up to the un-split collector line.
    E_collector = E(np.abs(d["collector_surface_W"]))
    print(f"loss_collector_line_kJ={E_collector / 1e3:.4f} "
          "(saved scalar, no per-book split; carried in the COMBINED graph "
          "only, so electron+ion reconcile to combined up to this line)")

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
    # plasma_heating_erg_s -- and that field still has no RHS consumer. The
    # far-end `transmitted_flux` is likewise never deposited on the
    # collector/end books. So the bypass primaries' terminal deposit remains
    # UNBOOKED on the plasma/surface side. Both are now SAVED, so the size of
    # the unbooked stream is on record rather than inferred.
    print("bypass_termination=circuit_booked_only; A15 interception -> "
          "BeamDepositionResult.anode_intercepted_erg_s still has no RHS "
          "consumer; far-end transmitted_flux is not deposited on the "
          "collector/end books (reported, not fixed here)")
    if not d["have_beam_exit_ledger"]:
        print("bypass_scalars=UNAVAILABLE (this file predates the saved "
              "beam exit ledger)")
    else:
        E_intercept = E(d["beam_anode_intercepted_W"])
        E_transmit = E(d["beam_transmitted_W"])
        print(f"bypass_anode_intercepted_kJ={E_intercept / 1e3:.4f} "
              f"frac_P_prim={E_intercept / E_prim:.4f} (leaves the ray, "
              "lands on the anode mesh, booked nowhere in the plasma)")
        print(f"bypass_transmitted_kJ={E_transmit / 1e3:.4f} "
              f"frac_P_prim={E_transmit / E_prim:.4f} (primaries streaming "
              "out of the far end, never deposited)")
        for key, _, lo, hi, mask in windows:
            def Ew(y, m=mask):
                return np.trapezoid(np.where(m, y, 0.0), t_s)

            print(f"  bypass_window_{key}=[{lo:.4f},{hi:.4f}]ms "
                  f"anode_intercepted_kJ="
                  f"{Ew(d['beam_anode_intercepted_W']) / 1e3:.4f} "
                  f"transmitted_kJ="
                  f"{Ew(d['beam_transmitted_W']) / 1e3:.4f}")

    # --- figures ---
    dt_ms = float(np.median(np.diff(d["time_ms"])))
    smooth = max(1, int(round(args.smooth_ms / dt_ms))) if args.smooth_ms > 0 else 1
    fs.apply(args.profile)
    written = []
    fig = make_power_figure(d, ledger, residual, args.profile, smooth)
    written += save_figure(fig, out_dir, f"{stem}_power", args.profile)
    # Stage 2 of the main efficiency figure shows window (a) only.
    fig = make_efficiency_figure(d, ledger, window, windows[0], args.profile)
    written += save_figure(fig, out_dir, f"{stem}_efficiency", args.profile)
    fig = make_beam_windows_figure(d, ledger, windows, args.profile)
    written += save_figure(fig, out_dir, f"{stem}_beam_windows", args.profile)
    # v5: the alternate simple split, produced alongside the three-window one.
    fig = make_beam_windows_figure(d, ledger, two, args.profile)
    written += save_figure(fig, out_dir, f"{stem}_beam_two_windows",
                           args.profile)
    fig = make_losses_figure(d, window, args.profile)
    written += save_figure(fig, out_dir, f"{stem}_losses", args.profile)
    fig = make_breakout_figure(d, window, args.profile)
    written += save_figure(fig, out_dir, f"{stem}_breakout", args.profile)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
