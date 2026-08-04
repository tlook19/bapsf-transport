"""Compare a sim1d run against the ES1 experimental overlay.

The point of the model is to reproduce measured behaviour, so the meaningful
benchmark is the data, not a previous version of the model. This samples the
simulation at the five ES1 probe locations and the experimental time base, and
reports the deviation in electron temperature and density against the measured
means and their standard errors.

Comparisons are made on the main-discharge clock, matching the notebook: model
time is shifted so t = 0 is the start of the main discharge, which is what the
experimental ``*_time_ms`` axes are referenced to.

Three comparison stages, in tuning order (each scored independently):

(i)   peak discharge current -- model ``source_I_tot`` vs the measured
      discharge-trace peak. Peak only: the breakdown *rate* is shaped by
      physics the model does not carry, so trace RMS is deliberately not a
      target.
(ii)  bulk Te and density at the five ES1 ports (the original comparison).
(iii) afterglow decay -- per-port e-folding time of the model Isat proxy
      ``n * sqrt(Te)`` against the measured Isat decay traces, over a fit
      window on the main-discharge clock. The default run carries only
      ``tau_afterglow = 5 ms`` past discharge end; use ``--tau-afterglow``
      to extend it toward the measured 27.5 ms tail.

Separately, ``--beta-collapse`` runs the simulation-informed sweep-bias
diagnostic (HYPOTHESIS ON RECORD 2026-07-22 plus its two addenda) over
a set of saved reference runs: per
(port, ES rung, window) residual vectors (dlnTe, dln n) are decomposed
against the (1, -1/2) sweep-inversion manifold, per-rung probe-area
drift (pure n direction), and whatever z/regime structure is left (model
error). Kinetic (k4*) runs are scored in the plateau window only; the
afterglow lives on the Isat decay traces. When the overlay carries the
raw inter-sweep drive Isat (``isat_drive_*``, schema v4) the area guard
runs in its within-shot native form and a model-free sweep-chain consistency check is added.
beta and a_{p,r} are hypothesis-test outputs, never data corrections.

Usage::

    python scripts/compare_sim1d_es1.py                      # resolved + knudsen
    python scripts/compare_sim1d_es1.py --nx 185
    python scripts/compare_sim1d_es1.py --save-h5 run.h5     # keep the run
    python scripts/compare_sim1d_es1.py --from-h5 run.h5     # re-score, no run
    python scripts/compare_sim1d_es1.py --tau-afterglow 0.0275 \
        --decay-window 20.5 40.0
    python scripts/compare_sim1d_es1.py --beta-collapse      # canonical run set
    python scripts/compare_sim1d_es1.py --beta-collapse run_es1.h5 run_es2.h5
"""

import argparse
import re
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import (
    LAPDSim1D,
    default_config,
    load_result_hdf5,
)
from cablp.solvers._sim1d.results.io import save_result_hdf5

OVERLAY = Path(__file__).resolve().parent / "data" / "es1_sim1d_overlay.npz"

PARAM_OVERRIDES = {
    # DISCHARGE CIRCUIT -- corrected stance, 2026-08-03 (Tom's call). These
    # mirror the config defaults EXACTLY (core/config.py active_defaults); the
    # duplication is deliberate -- this dict is the campaign stance record, and
    # dropping the pins would change resolution order for the other drivers.
    #
    # What this replaces: the previous values came from a FREE 4-parameter fit
    # to the ES1 trace alone (scripts/fit_es1_circuit.py, "0.14 V rms"). That
    # fit is NEAR-SINGULAR -- corr(V0, R) = 0.997, and R swings 1.9-5.7 mOhm
    # with the fit window, so the quoted +-0.079 mOhm was meaningless and the
    # 0.14 V rms was in-sample. The error was INVISIBLE AT ES1 and appeared
    # only on LADDER TRANSFER: reconstructing measured plateau V_dis, the old
    # parameterization gave residuals -0.136/+6.329/+5.677/+5.786 V at
    # ES1/2/3/4; the corrected one gives +0.010/+0.139/-0.053/-0.309 V.
    #
    # The replacement is a CONSTRAINED refit: V0 PINNED per rung at its measured
    # pre-shot reading, with C, R, L shared across four rungs (N = 1952, window
    # 0.3-19.8 ms). Conditioning fell 89.3 -> 4.7. Jackknife bars:
    # R = 7.213 +- 0.043 mOhm, C = 9.56 +- 0.66 F, L = 6.7 +- 2.5 uH.
    #
    # V_bank: MEASURED pre-shot open-circuit bank voltage at ES1, same probe
    #   channel as V_dis. +-0.03 V SEM; +-1.2% instrumental MULTIPLICATIVE
    #   systematic, unresolved between supply regulation and probe gain. This is
    #   NOT the 180 V supply setpoint (config.py's default), which is a
    #   different quantity. Per-rung values live in run_mechanism_ladder.
    # R_comp: MEASURED across four rungs (2965-4411 A) agreeing to 1.8%.
    # C_bank_F: MEASURED, and HARDWARE-BOUNDED to [7.56, 12.60] F by the bank
    #   spec (700 Chemi-Con 36DY cans, nominal 8.40 F, tolerance -10/+50%).
    #   The old "nominal bank <= 4 F" caveat is RETRACTED, not restated: it
    #   miscounted the bank by 2x and then read a near-FLOOR nominal as a
    #   ceiling. 8.9 F was inside tolerance all along -- no historical run is
    #   invalidated. See the C_bank_F docstring in core/config.py.
    # R_comp and C_bank_F are ONE joint fit and must move together.
    # L_parasitic_H: 6.6e-6 -> 8.1e-6 (2026-08-03), DERIVED from measurement,
    #   bracket 7.6-8.4 uH. Two instruments with disjoint windows and no shared
    #   fitted parameters agree at ~8 uH: the flyback volt-second balance over
    #   the fall (7.2-8.4 uH; INVARIANT to this circuit correction, since the
    #   fall branch of fit_circuit_edges.py uses emf = -V_meas and touches no
    #   circuit constant) and the rise-edge ODE fit (7.6 uH at 38 A rms). The
    #   plateau refit's own point estimate is 8.06 uH. The "15-25 uH box" that
    #   this comment previously deferred to is RETRACTED: it is L = tau_fall *
    #   R_load with the plasma as a CONSTANT RESISTOR, falsified by the measured
    #   V_dis collapsing 16x within 0.2 ms of the fall, and retracted 2026-07-21
    #   by the very script it was attributed to. 6.6e-6 was the orphaned fourth
    #   member of the same retracted free fit as the old R_comp/C_bank_F, not a
    #   considered stance. Honest limit: 6.6e-6 is still inside the refit's
    #   jackknife bar (6.7 +- 2.5 uH) -- it has no evidence behind it, but it is
    #   not excluded. L is inert for every sigma-scored row; consequences are
    #   confined to the unscored reported fingerprints (t90 +0.05..0.11 ms,
    #   ignition +0.02..0.07 ms, both toward the measurement). This restores the
    #   value the golden fixture has pinned all along (baseline_sim1d.py:104),
    #   so the golden is bit-exact across the change.
    "V_bank": 177.843,
    "R_comp": 7.2244e-3,
    "L_parasitic_H": 8.1e-6,
    "C_bank_F": 9.5,
    # NB the constant-T_s era ended here (f=0.1 stance promotion, 2026-07-27):
    # the retired "T_s": 273.15 + 1725 pin is gone. T_s is now only the INITIAL
    # surface temperature -- cathode_warming_model="power_balance" (a config
    # default) evolves it from cathode_Ts_base_K -- and its config default is
    # the identical 1998.15 K, so dropping the pin changes nothing numerically.
    # Neutral-equilibration puff width, MEASURED (Tom, 2026-07-29, boxed).
    # The ES1-4 total gas-puff pulse width was ~25 ms: operator practice is to
    # fire the valve, wait out the machine breakdown delay (~4-6 ms), hold
    # 20 ms from 1 kA, and round up. Refinable from the V_dis traces, not
    # fitted. Without this the equilibration inherits tau_discharge (20 ms) as
    # its per-cycle puff window -- a double duty with no physical basis. This
    # changes the SCORER's runs (the equilibrated seed rises ~x1.25 in
    # delivered fuel) and NOT the golden, which pins the key back to None.
    "equilibration_gas_puff_on_s": 25e-3,
    "S_gp": 3000,
    "S_gp_decay_target": 2000,
    "tau_gp_pulse_duration": 1e-3,
    "tau_gp_decay_duration": 5e-3,
    # Ion-neutral closure: R5 STANCE FLIP (2026-07-25) -- the ad-hoc constant
    # drag / cx_derived stance (b=0.5, constant, cx_derived, thermalization) is
    # RETIRED in favour of the R4.3 Phelps moment operator
    # (ion_neutral_moment_closure, now the config.py production default;
    # first-principles drag+CX+thermal, no knob). The legacy drag keys are
    # DEPRECATED and no longer set here; the historical golden pins them back
    # (baseline_sim1d BASELINE_PARAM_OVERRIDES) to stay bit-exact.
    # ADAS GCR rates (see cablp/vars/adas/README.md): effective ionization/
    # recombination and radiation-only cooling, consistent with the separate
    # ionization-cost term. b_Q* = 1 is meaningful under this model.
    "atomic_rate_model": "adas",
    # Beam-driven neutral excitation: 1.0 books the 2^1P channel alone, the
    # extra 0.4 approximates the rest of the singlet manifold. Radiates ~21 eV
    # per event as He I light and shortens the beam deposition length.
    "b_beam_excitation": 1.4,
    "b_Qei": 1,
    "b_Qen": 1,
    "b_Qcx": 1,
    "Rp": 15.0,
    "R_cath": 15.0,
    "implicit_heat_scheme": "tr_bdf2",
    "operator_splitting": "strang",
    "heat_picard_iterations": 2,
    "heat_picard_tol": 1e-10,
    # ES production machine geometry (R5 ES1 tuning pass, 2026-07-25, Tom's
    # decision; provisional pending the 2D model). End vessel expands to a 1 m
    # machine (neutral) radius over 10 cells with NO plasma flare (plasma stays
    # at Rp=15); plenum-choke obstruction Rcs=40/Lcs=25 (no support rods); no
    # baffles; collector length unchanged (100 cm default).
    "end_expansion_cells": 10,
    "end_expansion_machine_radius_cm": 100.0,
    "end_expansion_plasma_radius_cm": 15.0,
    "Rcs": 40.0,
    "Lcs": 25.0,
    "Rsup": 0.0,
    # --- f=0.1 PRODUCTION STANCE (promoted 2026-07-27) --------------------
    # Enumerated by config-diffing es1_r5_f01_rev20ms.h5 against
    # default_config() + these overrides, so what lands here is exactly the
    # stack that run carried. Its RUN-COST settings (tau_afterglow=6 ms,
    # max_steps_action="stop", density_dt_fraction=0.5) are deliberately NOT
    # promoted -- they buy runtime, not physics, and belong on the command
    # line of the run that wants them.
    #
    # Cathode power balance, co-tuned with S_gp at the ES1 rung: the
    # skin->substrate conduction is the one fitted knob, frozen after ES1.
    "cathode_conduction_W_per_K": 8000.0,
    # --- CATHODE CALIBRATION REPARAMETERIZED (Tom, 2026-07-29) ------------
    # The retired stance carried the calibration on the standby temperature
    # ("cathode_Ts_base_K": 1840.0 here, 70 K below the measurement). That
    # mislabelled a MEASURED quantity as a fit: the ES1 standby is the
    # Fig-10 digitized 1910 K (ES_OPERATING[1]["Ts_standby_K"], also the
    # config.py default), so the pin is GONE and the stance inherits it.
    #
    # The same calibration now sits on C_R, which the cathode literature
    # already treats as an EFFECTIVE emission constant (surface state,
    # patch fields, non-ideal emitting fraction), not the 120 A/cm^2/K^2
    # Richardson-Dushman universal. Its value is DERIVED, not refitted, by
    # matching the emission at the operating point in the code's own
    # expression J = C_R T^2 exp(-e phi/(kB T)):
    #
    #   J(C_R_eff, T + dT) = J(29.0, T)
    #   =>  C_R_eff = 29.0 * (T/(T+dT))^2 * exp(-(e phi/kB)(1/T - 1/(T+dT)))
    #
    #   T   = 1859.02 K   plateau T_s_surface, mean over 15-19.5 ms on the
    #                     main-discharge clock of the reference production
    #                     run es1_prod_25ms_nx240.h5
    #   dT  = +70 K       the base-temperature move (1910 - 1840), taken to
    #                     propagate ~1:1 into the plateau at fixed heater
    #                     power (the warming balance's restoring term is
    #                     G_cond*(T_s - T_base), so a rigid base shift
    #                     translates the operating point). ASSUMPTION -- the
    #                     revalidation run is what tests it.
    #   phi = 2.809 eV    the work function the emission actually evaluates
    #                     at the plateau: cathode_phiwf_clean_eV, since the
    #                     ads_des surface is fully cleaned there
    #                     (recorded phi_wf_eff = 2.809, theta ~ 1e-19).
    #                     The uncleaned shot-start phi_wf = 2.869 would give
    #                     14.06, a 1.4% difference well inside the residual
    #                     below.
    #
    # => C_R_eff = 14.2546 -> 14.25 adopted (14.3 to 3 s.f.; the extra digit
    #    keeps the point-emission match at 0.03%, inside the 0.1% the
    #    derivation was pre-registered to hit).
    #
    # This is a stance promotion, NOT a flagged feature: nothing here is
    # bit-exact with the retired stance. The match is exact only at the
    # plateau point; the flat direction (~103 K per e-fold of emission,
    # recorded) is not perfectly flat, so residual dynamics shifts of order
    # the ~10% ramp-gain slope across it are ACCEPTED and revalidated by
    # run, not tuned away. Standby emission is not matched exactly either
    # (1910 K on C_R_eff vs 1840 K on 29.0 emits 1.4% more at phi = 2.809,
    # 2.8% at 2.869) -- the ln J curvature between 1840 K and the 1859 K
    # matching point, i.e. the same flat-direction residual seen off the
    # operating point.
    "C_R": 14.25,
    # Beam deposition smoothed over 50 cm. The CSDA range profile is sharp on
    # the mesh scale; this spreads it over the physical straggling width so
    # the deposited power does not follow cell edges.
    "beam_deposition_smoothing_cm": 50.0,
    # Free-streaming cap on the parallel electron heat flux (the flag below).
    # f=0.1 is the flux-limiter coefficient this stance is NAMED for; it
    # combines harmonically (Cowie-McKee) with the Braginskii flux at
    # heat_flux_limiter_exponent=1, which is already the config default.
    "heat_flux_limiter_f": 0.1,
    # Fixed-cell-size source region (7a, approved 2026-07-27), enumerated from
    # es1_r5_srcgrid_shakedown.h5 and cross-checked against
    # es1_r5_srcgrid_nx240.h5 -- the two agree on every key except nx, which is
    # what makes the pair a clean resolution study. The 100 cm column in front
    # of the anode is meshed at exactly 10 cm regardless of nx, so refining nx
    # refines only the FAR column and no longer moves the source cells or the
    # puff cell underneath the source terms. Interim geometry, pending the 2D
    # model. Presence-gated BOTH ways against the flag below, so these three
    # must always travel together.
    "source_region_length_cm": 100.0,
    "source_region_dz_cm": 10.0,
    # NB nx_gap is NOT promoted: both artifacts ran it at 5, which is already
    # the config.py default, so it never appears in the delta.
}
FLAG_OVERRIDES = {
    # R5 stance flip: the legacy ion-neutral thermalization arm is subsumed by
    # the Phelps moment operator (config.py default); no longer set here.
    "ion_neutral_drag_cx_only": False,
    # R5 ES1 tuning pass: the end-vessel expansion geometry above.
    "end_expansion_geometry": True,
    # f=0.1 PRODUCTION STANCE (2026-07-27): the electron heat-flux limiter is
    # ON in production, at heat_flux_limiter_f=0.1 above.
    "electron_heat_flux_limit": True,
    # Fixed-cell-size source region (7a): pairs with source_region_length_cm /
    # source_region_dz_cm above; the geometry raises loudly if either side is
    # set without the other.
    "source_fixed_grid": True,
}


# WP-D beam product transport. "local" is the
# production stance and the config.py default, so it is deliberately absent
# from PARAM_OVERRIDES; "nonlocal" is an A/B arm that must travel with the run
# it scored. Reported as a delta only -- a production (local) artifact scores
# byte-identically to its recorded _scores.txt, and a nonlocal one says so.
BEAM_PRODUCT_TRANSPORT_DEFAULT = "local"


def beam_product_transport_note(params):
    """Return a label suffix naming a non-default beam_product_transport.

    Empty string on the production stance (and on a run too old to carry the
    key), so this only ever ADDS a line where the stance actually differs.
    """
    value = str(
        (params or {}).get(
            "beam_product_transport", BEAM_PRODUCT_TRANSPORT_DEFAULT
        )
    )
    if value == BEAM_PRODUCT_TRANSPORT_DEFAULT:
        return ""
    return f" [beam_product_transport={value}]"


# --- WP-E QL heating locality (heating_anomalous_transport) ---------------
# Unlike the WP-D note above, this label is printed ALWAYS rather than as a
# delta. {local, tail_walk} is a declared BRACKET, not a default plus a
# variant: the config docstring says outright that "a result must state which
# one it used", so a scored number is incomplete without its arm. A
# delta-only label also cannot distinguish "this run was local" from "this
# artifact predates the label", which is exactly the ambiguity the pre-WP-E
# case below exists to remove.
WPE_TRANSPORT_KEY = "heating_anomalous_transport"
WPE_TAIL_ENERGY_KEY = "heating_anomalous_tail_energy_eV"
WPE_TRANSPORT_DEFAULT = "local"


def wpe_arm_line(params):
    """Return the one-line WP-E arm header for a run's saved resolved params.

    ``params`` is the artifact's own resolved parameter dict (``results/io``
    saves them fully resolved), so this reports what the run ACTUALLY carried
    rather than what the current stance would have given it.

    ``heating_anomalous_tail_energy_eV`` is read only under ``"tail_walk"``
    and is labelled inert otherwise, matching its config docstring. An
    artifact written before WP-E existed carries neither key and is labelled
    "pre-WP-E" -- not silently reported as the default arm, and not a crash.

    DUPLICATED, deliberately, in ``fingerprints_sim1d._wpe_arm_line``: that
    tool is standalone by design, on the same grounds as
    ``non_ignited_message`` below. Keep the two in step.
    """
    params = params or {}
    if WPE_TRANSPORT_KEY not in params:
        return (
            f"WP-E arm: pre-WP-E ({WPE_TRANSPORT_KEY} absent from this run's "
            "saved params -- the QL heating locality closure did not exist "
            "when this artifact was written, so its behaviour is the "
            f"{WPE_TRANSPORT_DEFAULT!r} arm by construction)"
        )
    transport = str(params[WPE_TRANSPORT_KEY])
    tail = params.get(WPE_TAIL_ENERGY_KEY)
    tail_text = "<absent>" if tail is None else f"{float(tail):g} eV"
    if transport == WPE_TRANSPORT_DEFAULT:
        tail_text += " (inert under 'local')"
    return (
        f"WP-E arm: {WPE_TRANSPORT_KEY}={transport} | "
        f"{WPE_TAIL_ENERGY_KEY}={tail_text}"
    )


def _clamp_window_bound(requested, limit, kind, site, extent):
    """Return the window bound actually used, ANNOUNCING a clamp that binds.

    ``kind`` is ``"start"`` (the bound is clamped UP to ``limit``) or
    ``"end"`` (clamped DOWN). The arithmetic is exactly the ``max``/``min``
    it replaces, so nothing numeric moves; the point is that a clamp which
    silently shortens a scoring window produces a number that LOOKS like the
    configured quantity and is not -- the same failure ``short_afterglow_message``
    hard-fails on further down.

    A NOTICE rather than an exception is right here: unlike stage (iii),
    these sites are short-trajectory accommodations on diagnostic paths where
    a partial window is still worth reading, provided the reader knows the
    window moved. Nothing is printed when the clamp is inert, so a run whose
    trace covers the requested window scores with no extra output at all.
    """
    used = max(requested, limit) if kind == "start" else min(requested, limit)
    if used == requested:
        return used
    print(
        f"  CLAMP NOTICE [{site}]: {kind} bound requested {requested:.4g} ms, "
        f"USED {used:.4g} ms -- clamped to the {extent} extent. This window "
        f"is NOT the requested one, so these rows are not comparable to runs "
        f"scored over the full window."
    )
    return used


def non_ignited_message(result, caller):
    """Return the NON-IGNITED diagnosis for a run with no main_discharge.

    DUPLICATED, deliberately, in ``fingerprints_sim1d.non_ignited_message``:
    that tool is standalone by design and importing this module for a 20-line
    numpy helper would hand it the whole scorer driver. Both copies are
    exercised together by smoke_sim1d, so they cannot drift silently -- keep
    them in step.

    Every scoring stage is defined relative to the main-discharge origin. A
    run that never reached that phase has no origin, and the old ``times[0]``
    fallback silently scored the pre-breakdown instant as t=0 -- i.e. scored
    garbage against the measurements. Fail loudly instead, and say what the
    run actually did: its terminal phase and any ignition guard that fired.
    """
    phases = np.asarray(getattr(result, "phase", ()), dtype=str)
    terminal = str(phases[-1]) if phases.size else "<no samples>"
    events = getattr(result, "phase_events", None) or {}
    reasons = [str(reason) for reason in np.asarray(events.get("reason", ()))]
    times = np.asarray(events.get("time", ()), dtype=float)
    guards = [
        f"{reason} at t={time:.6e} s"
        for time, reason in zip(times, reasons)
        if reason in {"ignition_stalled", "prebreakdown_timeout"}
    ]
    abort = getattr(result, "ignition_abort", None)
    guard_text = "; ".join(guards) if guards else "no ignition-guard event"
    abort_text = ""
    if abort:
        abort_text = " | ignition_abort: " + " ".join(
            f"{key}={abort[key]}" for key in sorted(abort)
        )
    return (
        f"NON-IGNITED RUN: {caller} found no sample in the 'main_discharge' "
        f"phase, so this run has no discharge origin and CANNOT be scored. "
        f"Terminal phase: {terminal!r}. Ignition guards: {guard_text}"
        f"{abort_text}"
    )


def _main_discharge_origin(result):
    """Return the model time [s] at which the main discharge begins.

    Raises on a run that never ignited -- see ``non_ignited_message``.
    """
    phases = np.asarray(getattr(result, "phase", ()), dtype=str)
    times = np.asarray(result.time, dtype=float)
    hits = np.flatnonzero(phases == "main_discharge")
    if not hits.size:
        raise RuntimeError(
            non_ignited_message(result, "compare_sim1d_es1._main_discharge_origin")
        )
    return float(times[hits[0]])


# Production axial resolution (stance promotion, 2026-07-27). This is a
# DRIVER-level default, deliberately not a config.py default: the golden
# baseline is a regression scaffold, not a production claim, and pins its own
# nx (baseline_sim1d BASELINE_PARAM_OVERRIDES) so the reviewer gate keeps its
# runtime.
PRODUCTION_NX = 240


def run_model(
    nx=PRODUCTION_NX,
    exchange_model="knudsen",
    extra=None,
    drag_closure=None,
    Rp_model=None,
    flags_extra=None,
    t_end=None,
):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    if flags_extra:
        flags.update(flags_extra)
    params["neutral_exchange_model"] = exchange_model
    if nx is not None:
        params["nx"] = nx
    # A/B instrument for the drag-closure gate (M4):
    # swap the drag closure without touching the rest of the production
    # config. "constant" is PARAM_OVERRIDES as-is (the calibrated 0.5);
    # "slip" is the entrainment closure; "neutral_momentum" evolves M_n with
    # the honest b = 1 (the field replaces the compensation constant).
    if drag_closure == "slip":
        params["ion_neutral_drag_model"] = "slip"
        params["b_ion_neutral_drag"] = 1.0
    elif drag_closure == "neutral_momentum":
        params["ion_neutral_drag_model"] = "constant"
        params["b_ion_neutral_drag"] = 1.0
        flags["neutral_momentum"] = True
    elif drag_closure == "neutral_momentum_two_zone":
        params["ion_neutral_drag_model"] = "constant"
        params["b_ion_neutral_drag"] = 1.0
        params["neutral_momentum_radial"] = "two_zone"
        flags["neutral_momentum"] = True
    elif drag_closure not in (None, "constant"):
        raise ValueError(f"unknown drag_closure {drag_closure!r}")
    # A/B instrument for M1: profile-integrated
    # cathode-anode gap resistance vs the historical single-sample R_p.
    # With the production Rp == R_cath the geometric component vanishes, so
    # this isolates the Te-profile effect on V_dis(t).
    if Rp_model is not None:
        params["cathode_Rp_model"] = Rp_model
    if extra:
        params.update(extra)
    sim = LAPDSim1D(params, flags)
    # t_end=None (default) keeps the historical dynamic end time derived from
    # the tau_* budget; an explicit t_end caps run cost WITHOUT deforming the
    # hardware drive length (the loop terminates at t_end regardless of
    # tau_discharge), as run_floorfix_g3g4.py already relies on.
    sim.start_simulation(t_end=t_end, dt=None, operator_split=None, max_steps=None)
    return sim.get_results(), sim.geometry, params, flags


# --- Measurement error model (adopted 2026-07-22, conservative per Tom:
# "assume experimental errors can be on the large side").  Shot-to-shot SEM
# measures precision only; the sweep systematics dominate:
#   sigma_Te,sys = 0.25*Te + 0.20 eV   (fit-window, EEDF tail, sheath
#       expansion, magnetization, fluctuation smearing, surface drift --
#       added generously; Te < 1 eV is flagged SEMI-QUANTITATIVE, where the
#       transition spans < 1 V and the budget approaches order unity)
#   sigma_n,sys  = n * sqrt((0.5*sigma_Te/Te)^2 + 0.10^2)   (the c_s
#       inversion propagates half the fractional Te error, anti-correlated;
#       10 % interferometer calibration + transfer)
# Deviations are reported against sigma_tot = sqrt(SEM^2 + sigma_sys^2).
# NB the dominant biases push LP Te HIGH and hence inverted n LOW -- the
# model-hot / model-underdense residuals are, if anything, understated.
# The "Isat" rows compare in I_sat space (n*sqrt(Te), both sides), where
# the sweep inversion cancels identically -- the systematics-robust
# magnitude/shape observable (the stage-(iii) tau metric already lives
# there by design).
TE_SYS_FRAC = 0.25
TE_SYS_FLOOR_EV = 0.20
TE_SEMIQUANT_EV = 1.0
N_CAL_FRAC = 0.10

# PRIMARY semi-quantitative criterion: the measured fit-window refit SPREAD,
# carried per port as `te_window_spread_frac` by schema-v6+ overlays. It is
# the fractional variation of the fitted Te across the analysis re-fit windows
# -- a DIRECT measurement of how well the sweep pins Te at that port -- so it
# supersedes `Te < 1 eV`, which was only a proxy for the same doubt. The old
# criterion is retained as the SECONDARY label: the two are a union, so no row
# that was flagged before can lose its mark.
#
# A NaN spread is UNDETERMINED, never "not flagged": the row keeps whatever
# the Te-criterion said and is marked explicitly. This is load-bearing --
# the ES3 port-50 refit failed and ES4 has no refit product at all, and port
# 50 is the worst-scoring row in the file.
#
# REGISTERED (Tom, 2026-08-04). 0.50 is read off the measured separation
# rather than fitted: the stable ports sit at 10-25 % and the unstable ones at
# 70-167 %, with nothing in between, so any threshold in the gap selects the
# same set.
TE_SPREAD_SEMIQUANT_FRAC = 0.50  # registered (Tom, 2026-08-04)
TE_SPREAD_FIELD = "te_window_spread_frac"

# Half-strength propagation of the Te spread onto the n rows, against the SAME
# registered threshold. n ~ Isat/sqrt(Te), so a fractional Te spread S maps to
# a fractional n spread ~ S/2. The n rows are therefore flagged when
# S/2 > TE_SPREAD_SEMIQUANT_FRAC, not when S itself exceeds it. A NaN Te
# spread is inherited as UNDETERMINED exactly as on the Te rows. The Isat rows
# take no spread criterion at all: n*sqrt(Te) is the sweep-inversion-cancelling
# observable, which is the whole point of that row.
N_SPREAD_FROM_TE_SPREAD = 0.5


def _sigma_sys(field, exp_values):
    exp_values = np.asarray(exp_values, dtype=float)
    if field == "Te":
        return TE_SYS_FRAC * np.abs(exp_values) + TE_SYS_FLOOR_EV
    if field == "n":
        # 0.5 * sigma_Te/Te with sigma_Te evaluated at the measured Te is
        # applied by the caller (needs the Te trace); this is the
        # calibration part only.
        return N_CAL_FRAC * np.abs(exp_values)
    return np.zeros_like(exp_values)


def compare(result, geometry, overlay):
    """Return per-port deviation of model Te and density from the ES1 means."""
    z_probe = np.asarray(overlay["z_cm"], dtype=float)
    ports = np.asarray(overlay["port"])
    origin = _main_discharge_origin(result)
    t_model_ms = (np.asarray(result.time, dtype=float) - origin) * 1.0e3
    z_model = np.asarray(result.z_cm, dtype=float)

    # Interpolate measured Te onto each field's own time base for the
    # systematic-error propagation (n's sigma_sys needs Te) and the
    # I_sat-space synthesis.
    te_t = np.asarray(overlay["te_time_ms"], dtype=float)
    te_mean_2d = np.asarray(overlay["te_mean_ev"], dtype=float)

    # Presence gate: pre-v6 overlays carry no spread field, and on those the
    # Te < 1 eV criterion stays the sole flag with byte-identical rendering.
    spread_frac = None
    if TE_SPREAD_FIELD in overlay:
        spread_frac = np.asarray(overlay[TE_SPREAD_FIELD], dtype=float)
        if spread_frac.shape != ports.shape:
            raise ValueError(
                f"overlay {TE_SPREAD_FIELD} has shape {spread_frac.shape}, "
                f"expected one entry per port {ports.shape}; a mismatched "
                "length would attribute a port's spread to the wrong row"
            )

    rows = []
    # Stage (ii) has no configured window: its comparison domain is the
    # experimental time base intersected with the model's coverage. That
    # intersection is still a silent window change when the model
    # under-covers -- two runs whose traces end at different times average
    # over different windows -- so announce it, once per distinct time base.
    coverage_announced = set()
    for field, t_key, mean_key, sem_key, unit in (
        ("Te", "te_time_ms", "te_mean_ev", "te_sem_ev", "eV"),
        ("n", "density_time_ms", "density_mean_cm3", "density_total_sem_cm3", "cm^-3"),
        # I_sat space: n*sqrt(Te) on both sides -- the sweep inversion
        # cancels identically on the measured side, so this row carries
        # only SEM + the interferometer calibration.
        ("Isat", "density_time_ms", "density_mean_cm3", "density_total_sem_cm3", "a.u."),
    ):
        t_exp = np.asarray(overlay[t_key], dtype=float)
        mean = np.asarray(overlay[mean_key], dtype=float)
        sem = np.asarray(overlay[sem_key], dtype=float)
        if field == "Isat":
            model_2d = np.asarray(result.n, dtype=float) * np.sqrt(
                np.maximum(np.asarray(result.Te, dtype=float), 0.0)
            )
        else:
            model_2d = np.asarray(getattr(result, field), dtype=float)
        # Only compare where the experiment has data and the model has run.
        if t_key not in coverage_announced:
            coverage_announced.add(t_key)
            _clamp_window_bound(
                float(t_exp.min()), float(t_model_ms.min()), "start",
                f"stage (ii) / {t_key}", "model trace",
            )
            _clamp_window_bound(
                float(t_exp.max()), float(t_model_ms.max()), "end",
                f"stage (ii) / {t_key}", "model trace",
            )
        window = (t_exp >= t_model_ms.min()) & (t_exp <= t_model_ms.max())
        for p, (z, port) in enumerate(zip(z_probe, ports)):
            iz = int(np.argmin(np.abs(z_model - z)))
            model_t = np.interp(t_exp[window], t_model_ms, model_2d[:, iz])
            exp_t = mean[p, window]
            sem_t = sem[p, window]
            te_exp_t = np.interp(t_exp[window], te_t, te_mean_2d[p])
            te_safe = np.maximum(np.abs(te_exp_t), 1e-3)
            if field == "Isat":
                exp_t = exp_t * np.sqrt(te_safe)
                # SEM propagated; systematics: calibration only (the
                # c_s inversion cancels in n*sqrt(Te)).
                sem_t = sem_t * np.sqrt(te_safe)
                sys_t = N_CAL_FRAC * np.abs(exp_t)
            elif field == "Te":
                sys_t = _sigma_sys("Te", exp_t)
            else:
                sig_te = _sigma_sys("Te", te_exp_t)
                sys_t = np.abs(exp_t) * np.sqrt(
                    (0.5 * sig_te / te_safe) ** 2 + N_CAL_FRAC**2
                )
            err_tot = np.sqrt(sem_t**2 + sys_t**2)
            good = np.isfinite(exp_t) & np.isfinite(model_t) & (exp_t != 0.0)
            if not np.any(good):
                continue
            ratio = float(np.mean(model_t[good] / exp_t[good]))
            rel = float(np.sqrt(np.mean(((model_t - exp_t)[good] / exp_t[good]) ** 2)))
            sigma = float(np.mean(np.abs((model_t - exp_t)[good] / err_tot[good])))
            # Secondary criterion (unchanged): the measured Te regime. It
            # still covers the n rows, which inherit the doubt through the
            # sqrt(Te) sweep inversion.
            te_low = field in ("Te", "n") and float(
                np.mean(te_exp_t[good])
            ) < TE_SEMIQUANT_EV
            # Primary criterion: the refit-window spread. The Te rows take it
            # at full strength; the n rows at half, since n ~ Isat/sqrt(Te)
            # maps a fractional Te spread S onto a fractional n spread ~ S/2.
            # Both are tested against the same registered threshold.
            spread_applies = spread_frac is not None and field in ("Te", "n")
            spread_te = float(spread_frac[p]) if spread_applies else np.nan
            spread_eff = spread_te * (
                N_SPREAD_FROM_TE_SPREAD if field == "n" else 1.0
            )
            spread_high = bool(
                np.isfinite(spread_eff) and spread_eff > TE_SPREAD_SEMIQUANT_FRAC
            )
            # UNDETERMINED: spread was expected for this row but is NaN. The
            # verdict falls back to te_low; it is never silently cleared. The
            # n rows inherit the Te row's NaN, since they inherit its spread.
            spread_undetermined = bool(
                spread_applies and not np.isfinite(spread_te)
            )
            rows.append(
                {
                    "field": field,
                    "unit": unit,
                    "port": str(port),
                    "z": float(z),
                    "model": float(np.mean(model_t[good])),
                    "exp": float(np.mean(exp_t[good])),
                    "ratio": ratio,
                    "rms_rel": rel,
                    "sigma": sigma,
                    # Union of the two criteria, so re-basing cannot un-flag.
                    "semiquant": bool(te_low or spread_high),
                    "semiquant_te": bool(te_low),
                    "semiquant_spread": spread_high,
                    "spread_undetermined": spread_undetermined,
                    "spread_gated": spread_frac is not None,
                }
            )
    return rows


def compare_peak_current(result, overlay):
    """Return the stage (i) figure of merit: model vs measured peak current."""
    diag = getattr(result, "cathode_diagnostics", None) or {}
    I_model = np.asarray(diag.get("source_I_tot", ()), dtype=float)
    t_model_ms = (
        np.asarray(result.time, dtype=float) - _main_discharge_origin(result)
    ) * 1.0e3
    t_exp = np.asarray(overlay["discharge_time_ms"], dtype=float)
    I_exp = np.asarray(overlay["discharge_current_mean_a"], dtype=float)
    sem_exp = np.asarray(overlay["discharge_current_sem_a"], dtype=float)

    out = {"model_peak_a": np.nan, "model_peak_t_ms": np.nan}
    if I_model.size and np.any(np.isfinite(I_model)):
        i_peak = int(np.nanargmax(I_model))
        out["model_peak_a"] = float(I_model[i_peak])
        out["model_peak_t_ms"] = float(t_model_ms[i_peak])
    j_peak = int(np.nanargmax(I_exp))
    out["exp_peak_a"] = float(I_exp[j_peak])
    out["exp_peak_t_ms"] = float(t_exp[j_peak])
    out["exp_peak_sem_a"] = float(sem_exp[j_peak])
    out["ratio"] = out["model_peak_a"] / out["exp_peak_a"]

    # Late-window plateau (15-19.5 ms): the model's early transient carries a
    # known V_dis(t)-trajectory artifact, so the
    # established current scale is better read from the end of the drive.
    late = (15.0, 19.5)
    m_model = (t_model_ms >= late[0]) & (t_model_ms <= late[1])
    m_exp = (t_exp >= late[0]) & (t_exp <= late[1])
    out["model_late_a"] = (
        float(np.nanmean(I_model[m_model])) if np.any(m_model) else np.nan
    )
    out["exp_late_a"] = float(np.nanmean(I_exp[m_exp])) if np.any(m_exp) else np.nan
    out["late_ratio"] = out["model_late_a"] / out["exp_late_a"]
    return out


def _efold_time_ms(t_ms, y, floor=0.0):
    """Return the log-linear e-folding decay time [ms] of ``y`` over ``t_ms``.

    Positive for a decaying signal. NaN when fewer than 8 samples survive the
    positivity/noise-floor mask, or when the fitted slope is not a decay.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(t_ms) & np.isfinite(y) & (y > max(floor, 0.0))
    if np.count_nonzero(good) < 8:
        return np.nan
    slope = np.polyfit(t_ms[good], np.log(y[good]), 1)[0]
    return -1.0 / slope if slope < 0.0 else np.nan


# Stage (iii) fit window on the main-discharge clock [ms]. The discharge ends
# at trigger + tau_discharge = 20 ms, so this is the first 1.5 ms OF THE
# AFTERGLOW -- the early, transport-dominated decay that both the model and
# the Isat traces actually resolve. It matches DECAY_WINDOW_MS = (0.0, 1.5) in
# the afterglow decay figure published from these runs. The two windows are
# kept deliberately identical so the scored number and the plotted number
# are the same number; changing one without the other silently desyncs them. Moved here from the historical (20.5, 25.0)
# (Tom, 2026-07-29): that band started 0.5 ms late and ran 5 ms out, so it
# scored tail structure rather than the decay the figure is about.
DECAY_WINDOW_MS = (20.0, 21.5)


def short_afterglow_message(result, window_ms, t_model_ms):
    """Return the SHORT-AFTERGLOW diagnosis for a run that cannot be scored.

    Same philosophy as ``non_ignited_message`` and the retired ``times[0]``
    fallback: a clamp that silently shortens a scoring window produces a
    number that LOOKS like the scored quantity and is not. ``min(t1,
    t_model_ms.max())`` did exactly that -- a run whose afterglow ended before
    the stage (iii) window closed got its e-fold time fitted over whatever
    fraction of the window it happened to cover, and the fit was reported with
    no indication that it answered a different question from every other run
    in the campaign. Two runs scored against DIFFERENT windows are not
    comparable, and comparability across runs is the whole point of stage
    (iii).

    Fail instead, and say what would have to change: the configured window,
    what the run actually covers, and the ``tau_afterglow`` that decides it.
    """
    t0, t1 = float(window_ms[0]), float(window_ms[1])
    t_model_ms = np.asarray(t_model_ms, dtype=float)
    model_start, model_end = float(t_model_ms.min()), float(t_model_ms.max())
    params = getattr(result, "params", None) or {}
    tau = params.get("tau_afterglow")
    tau_text = (
        f"{float(tau):.6g} s ({float(tau) * 1.0e3:.4g} ms)"
        if tau is not None
        else "<absent from this run's params>"
    )
    # What the window demands of the afterglow: the window closes t1 ms after
    # the main-discharge origin, and the discharge itself occupies
    # tau_discharge of that, so the afterglow must run for the remainder.
    tau_discharge = params.get("tau_discharge")
    needed_text = ""
    if tau_discharge is not None:
        needed_ms = t1 - float(tau_discharge) * 1.0e3
        needed_text = (
            f" With tau_discharge={float(tau_discharge):.6g} s, closing this "
            f"window needs tau_afterglow >= {needed_ms * 1.0e-3:.6g} s "
            f"({needed_ms:.4g} ms)."
        )
    return (
        f"SHORT AFTERGLOW: the stage (iii) decay window is "
        f"({t0:.4g}, {t1:.4g}) ms on the main-discharge clock, but this run's "
        f"trace only spans ({model_start:.4g}, {model_end:.4g}) ms -- it ends "
        f"{t1 - model_end:.4g} ms before the window closes, so the decay fit "
        f"CANNOT be computed over the campaign's window and this run cannot "
        f"be scored on stage (iii). Run tau_afterglow={tau_text}."
        f"{needed_text} Extend the run, or score it with an explicit "
        f"--decay-window that its trace covers -- but a window that differs "
        f"from {DECAY_WINDOW_MS} is not comparable to the campaign's other "
        f"stage (iii) numbers."
    )


def compare_decay(result, overlay, window_ms=DECAY_WINDOW_MS):
    """Return per-port stage (iii) rows: model vs measured Isat e-fold times.

    The model Isat proxy is ``n * sqrt(Te)`` at the port cell (the Bohm-flux
    scaling; constants cancel in an e-folding time). Both signals get the same
    log-linear fit over the same window. The experimental noise floor is
    estimated from the final 5 ms of each decay trace (5x its robust sigma).

    Raises ``RuntimeError`` when the model trace ends before the window closes
    -- see ``short_afterglow_message``. The window is never silently shortened
    to fit the run.
    """
    t0, t1 = float(window_ms[0]), float(window_ms[1])
    origin = _main_discharge_origin(result)
    t_model_ms = (np.asarray(result.time, dtype=float) - origin) * 1.0e3
    z_model = np.asarray(result.z_cm, dtype=float)
    n_model = np.asarray(result.n, dtype=float)
    Te_model = np.asarray(result.Te, dtype=float)

    t_exp = np.asarray(overlay["isat_decay_time_ms"], dtype=float)
    isat = np.asarray(overlay["isat_decay_mean_a"], dtype=float)
    ports = np.asarray(overlay["isat_decay_port"])
    z_ports = {
        int(p): float(z)
        for p, z in zip(np.asarray(overlay["port"]), overlay["z_cm"])
    }

    model_end_ms = float(t_model_ms.max())
    if model_end_ms < t1:
        raise RuntimeError(short_afterglow_message(result, (t0, t1), t_model_ms))
    rows = []
    for p in range(ports.size):
        z = z_ports.get(int(ports[p]))
        if z is None:
            continue
        exp_window = (t_exp >= t0) & (t_exp <= t1)
        tail = isat[p, t_exp >= t_exp.max() - 5.0]
        noise = 5.0 * 1.4826 * np.nanmedian(np.abs(tail - np.nanmedian(tail)))
        tau_exp = _efold_time_ms(t_exp[exp_window], isat[p, exp_window], noise)

        iz = int(np.argmin(np.abs(z_model - z)))
        model_window = (t_model_ms >= t0) & (t_model_ms <= t1)
        proxy = n_model[model_window, iz] * np.sqrt(
            np.maximum(Te_model[model_window, iz], 0.0)
        )
        tau_model = _efold_time_ms(t_model_ms[model_window], proxy)

        rows.append(
            {
                "port": int(ports[p]),
                "z": z,
                "tau_exp_ms": tau_exp,
                "tau_model_ms": tau_model,
                "ratio": tau_model / tau_exp if np.isfinite(tau_exp) else np.nan,
            }
        )
    return rows, (t0, t1)


# --- beta-collapse diagnostic (HYPOTHESIS ON RECORD 2026-07-22 + two
# addenda).  Where the model agrees
# in Isat space (n*sqrt(Te), sweep-inversion systematics cancel), it is
# licensed as a reference to test whether the residual Te/n disagreements
# are a sweep-analysis systematic: a Te bias beta forces measured
# n = true/sqrt(beta), so residuals must collapse onto the (1, -1/2)
# manifold in the (dlnTe, dln n) plane.  Three identifiable components:
#   (1) sweep bias: moves ALONG (1, -1/2); condition-dependent
#       fingerprints discriminate the pathology (~Te => fit-window/EEDF
#       tail; ~Isat magnitude => circuit series R; ~density => sheath
#       expansion; day-to-day => probe surface);
#   (2) per-rung probe-area drift a_{p,r} (rotation/deposition): PURE n
#       direction, constant within a rung -- the sweep shape is area-free
#       so Te is never touched.  GUARD: an area nuisance must shift
#       plateau and afterglow identically (nothing rotates or deposits in
#       25 ms);
#   (3) model error: whatever carries physics z/regime structure.
# First-addendum consequences baked in below: each probe's area was
# individually calibrated against the interferometer WITH sweep Te, so a
# constant Te bias is invisible (absorbed into A_p) -- the diagnostic
# constrains beta(z,t)/beta_cal and per-rung area JUMPS only, and the
# cleanest lever arm is the ES ladder at fixed port (same probe, baked-in
# bias cancels, large Te/n swing).  Hierarchy: n axial shape is
# interferometer-anchored (solid); Isat axial shape carries the per-port
# beta_cal spread (demoted); within-shot time structure is fully robust.
#
# Error algebra (from the measurement error model above): the Te
# systematic and its anti-correlated n propagation lie ALONG the
# manifold by construction, so
#   ln beta_hat = dlnTe            carries SEM-only noise, while
#   ln a_hat    = dln n + dlnTe/2  (the manifold-perpendicular, pure-n
#       component) carries n SEM (+) interferometer cal (+) Te SEM/2.
# The sweep-space Isat license budget adds the probe-area transfer term
# from the first addendum (~0.5 * fractional Te bias at calibration) on
# top of SEM + cal -- it is NOT just SEM + 10%.
#
# Coverage: swept Te/n exist only in the drive plateau; the afterglow is
# the Isat decay trace (absolute amps, unknown per-port unit constant vs
# the model proxy).  The area guard is therefore CROSS-RUNG per port:
# the per-rung jump of ln a_hat (plateau) must equal the per-rung jump of
# the afterglow Isat residual (per-port constants cancel when centered
# across rungs).  Because (1, -1/2) and the n direction span the plane,
# that guard residual IS the collapse test after the afterglow-pinned
# area correction -- one number, both roles.
#
# beta and a_{p,r} are HYPOTHESIS-TEST outputs, never data corrections,
# pending independent corroboration (cheapest: re-fit raw sweeps with
# varied windows).  Kinetic (k4*) runs are licensed in the plateau only
# (quasi-static kinetic neutrals; afterglow unscored).

BETA_PLATEAU_MS = (15.0, 19.5)
BETA_AFTERGLOW_MS = (20.5, 25.0)
BETA_LICENSE_SIGMA = 2.0
BETA_ISAT_AREA_FRAC = 0.5 * TE_SYS_FRAC  # addendum (ii): probe-area term
BETA_CANONICAL_FAMILIES = ("2zbase", "2z", "k4t")
BETA_CANONICAL_PATTERN = "es1_nx120_m6_sq3400_{family}_es{rung}.h5"


def _parse_beta_run(spec):
    """Parse a --beta-collapse run spec into (path, rung, kind).

    Rung and kind are read from the file name (``..._es<N>.h5``; any
    ``_k4*`` token means kinetic) and can be overridden with
    ``PATH:es=N`` / ``PATH:kind=kinetic|moment``.
    """
    parts = str(spec).split(":")
    path = Path(parts[0])
    rung = None
    kind = None
    for token in parts[1:]:
        if token.startswith("es="):
            rung = int(token[3:])
        elif token.startswith("kind="):
            kind = token[5:]
        else:
            raise ValueError(f"unknown beta-collapse run option {token!r}")
    stem = path.stem
    if rung is None:
        hits = re.findall(r"_es(\d+)", stem)
        if not hits:
            raise ValueError(
                f"cannot read the ES rung from {path.name!r}; pass PATH:es=N"
            )
        rung = int(hits[-1])
    if kind is None:
        kind = "kinetic" if re.search(r"_k4[a-z0-9]*(_|$)", stem) else "moment"
    if kind not in ("kinetic", "moment"):
        raise ValueError(f"unknown beta-collapse run kind {kind!r}")
    return path, rung, kind


def _beta_sweep_points(result, overlay, window_ms):
    """Per-port (dlnTe, dln n, dlnIsat) window means with log-sigmas.

    Window means of the pointwise log residual ln(exp/model); the
    systematics are coherent within a window, so log-sigmas are window
    means of the fractional errors (no sqrt(N) reduction).
    """
    t0, t1 = float(window_ms[0]), float(window_ms[1])
    origin = _main_discharge_origin(result)
    t_model_ms = (np.asarray(result.time, dtype=float) - origin) * 1.0e3
    z_model = np.asarray(result.z_cm, dtype=float)
    te_2d = np.asarray(result.Te, dtype=float)
    n_2d = np.asarray(result.n, dtype=float)
    t1 = _clamp_window_bound(
        t1, float(t_model_ms.max()), "end",
        "beta-collapse plateau / _beta_sweep_points", "model trace",
    )
    # Hoisted out of the port loop, where it was recomputed identically per
    # port: same value, one notice instead of five.
    t0_used = _clamp_window_bound(
        t0, float(t_model_ms.min()), "start",
        "beta-collapse plateau / _beta_sweep_points", "model trace",
    )

    te_t = np.asarray(overlay["te_time_ms"], dtype=float)
    te_mean = np.asarray(overlay["te_mean_ev"], dtype=float)
    te_sem = np.asarray(overlay["te_sem_ev"], dtype=float)
    n_t = np.asarray(overlay["density_time_ms"], dtype=float)
    n_mean = np.asarray(overlay["density_mean_cm3"], dtype=float)
    n_sem = np.asarray(overlay["density_total_sem_cm3"], dtype=float)

    points = []
    for p, (z, port) in enumerate(zip(overlay["z_cm"], overlay["port"])):
        iz = int(np.argmin(np.abs(z_model - z)))

        w_te = (te_t >= t0_used) & (te_t <= t1)
        te_exp = te_mean[p, w_te]
        te_model = np.interp(te_t[w_te], t_model_ms, te_2d[:, iz])
        good_te = (
            np.isfinite(te_exp) & np.isfinite(te_model)
            & (te_exp > 0) & (te_model > 0)
        )
        w_n = (n_t >= t0_used) & (n_t <= t1)
        n_exp = n_mean[p, w_n]
        n_model = np.interp(n_t[w_n], t_model_ms, n_2d[:, iz])
        te_model_n = np.interp(n_t[w_n], t_model_ms, te_2d[:, iz])
        te_exp_n = np.interp(n_t[w_n], te_t, te_mean[p])
        good_n = (
            np.isfinite(n_exp) & np.isfinite(n_model)
            & (n_exp > 0) & (n_model > 0)
            & (te_exp_n > 0) & (te_model_n > 0)
        )
        if np.count_nonzero(good_te) < 2 or np.count_nonzero(good_n) < 2:
            continue

        dlnte = float(np.mean(np.log(te_exp[good_te] / te_model[good_te])))
        s_te_sem = float(np.mean(te_sem[p, w_te][good_te] / te_exp[good_te]))
        s_te_sys = float(
            np.mean(_sigma_sys("Te", te_exp[good_te]) / te_exp[good_te])
        )
        dlnn = float(np.mean(np.log(n_exp[good_n] / n_model[good_n])))
        s_n_sem = float(np.mean(n_sem[p, w_n][good_n] / n_exp[good_n]))

        # Sweep-space Isat (n*sqrt(Te) both sides): the inversion cancels.
        dlnisat = float(
            np.mean(
                np.log(n_exp[good_n] * np.sqrt(te_exp_n[good_n]))
                - np.log(n_model[good_n] * np.sqrt(te_model_n[good_n]))
            )
        )
        s_isat = float(
            np.sqrt(
                s_n_sem**2
                + (0.5 * s_te_sem) ** 2
                + N_CAL_FRAC**2
                + BETA_ISAT_AREA_FRAC**2
            )
        )
        points.append(
            {
                "port": int(port),
                "z": float(z),
                "dlnte": dlnte,
                "s_te_sem": s_te_sem,
                "s_te_tot": float(np.hypot(s_te_sem, s_te_sys)),
                "dlnn": dlnn,
                "s_n_sem": s_n_sem,
                "dlnisat": dlnisat,
                "s_isat": s_isat,
                "licensed": bool(abs(dlnisat) <= BETA_LICENSE_SIGMA * s_isat),
                # decomposition (see error algebra note above)
                "ln_beta": dlnte,
                "s_beta": s_te_sem,
                "ln_a": dlnn + 0.5 * dlnte,
                "s_a": float(
                    np.sqrt(s_n_sem**2 + N_CAL_FRAC**2 + (0.5 * s_te_sem) ** 2)
                ),
                # Condition variables for the fingerprints, evaluated on
                # the licensed REFERENCE side: beta_hat contains
                # lnTe_exp, so correlating against measured conditions
                # would be self-correlated by construction.
                "te_ref": float(np.mean(te_model[good_te])),
                "n_ref": float(np.mean(n_model[good_n])),
                "isat_ref": float(
                    np.mean(n_model[good_n] * np.sqrt(te_model_n[good_n]))
                ),
            }
        )
    return points


def _beta_trace_points(
    result, overlay, window_ms, prefix="isat_decay", min_samples=8
):
    """Per-port raw-Isat log residual ln(exp/model proxy) over a window.

    ``prefix`` selects the trace family: ``isat_decay`` (afterglow,
    schema v2) or ``isat_drive`` (inter-sweep dead-time cells during the
    drive, schema v4 -- added to the overlay by the external data-analysis
    exporter, not by this repo).  Both are in amps from the swept
    probe's own channel with NO area factor anywhere in the chain, and
    the model proxy n*sqrt(Te) is not in amps, so each value carries an
    unknown per-port unit constant.  Within one rung the constant is
    identical for drive and decay (same channel, zero offsets and shot
    ensemble -- the exporter must guarantee this), so the WITHIN-SHOT
    difference is meaningful in absolute terms; across rungs at fixed
    port, differences are meaningful too (same probe/hardware).
    Returns None when the trace family is not in the overlay.
    """
    if f"{prefix}_time_ms" not in overlay:
        return None
    t0, t1 = float(window_ms[0]), float(window_ms[1])
    origin = _main_discharge_origin(result)
    t_model_ms = (np.asarray(result.time, dtype=float) - origin) * 1.0e3
    z_model = np.asarray(result.z_cm, dtype=float)
    n_2d = np.asarray(result.n, dtype=float)
    te_2d = np.asarray(result.Te, dtype=float)
    t1 = _clamp_window_bound(
        t1, float(t_model_ms.max()), "end",
        f"beta-collapse {prefix} / _beta_trace_points", "model trace",
    )

    t_exp = np.asarray(overlay[f"{prefix}_time_ms"], dtype=float)
    isat = np.asarray(overlay[f"{prefix}_mean_a"], dtype=float)
    sem = np.asarray(overlay[f"{prefix}_sem_a"], dtype=float)
    ports = np.asarray(overlay[f"{prefix}_port"])
    z_ports = {
        int(p): float(z)
        for p, z in zip(np.asarray(overlay["port"]), overlay["z_cm"])
    }

    points = []
    for p in range(ports.size):
        z = z_ports.get(int(ports[p]))
        if z is None:
            continue
        iz = int(np.argmin(np.abs(z_model - z)))
        window = (t_exp >= t0) & (t_exp <= t1)
        exp_t = isat[p, window]
        sem_t = sem[p, window]
        model_t = np.interp(
            t_exp[window],
            t_model_ms,
            n_2d[:, iz] * np.sqrt(np.maximum(te_2d[:, iz], 0.0)),
        )
        good = (
            np.isfinite(exp_t) & np.isfinite(model_t)
            & (exp_t > 0) & (model_t > 0)
        )
        if np.count_nonzero(good) < min_samples:
            continue
        ln_ratio = np.log(exp_t[good] / model_t[good])
        t_good = t_exp[window][good]
        # Slope of ln(exp/model) across the window [1/ms]: a pure
        # area/unit offset is flat; a model decay-rate (tau) error is
        # sloped and leaks into the window mean -- the within-shot guard
        # reports it so tau-confounded failures are identifiable.
        slope = (
            float(np.polyfit(t_good, ln_ratio, 1)[0])
            if np.ptp(t_good) > 0
            else np.nan
        )
        points.append(
            {
                "port": int(ports[p]),
                "z": z,
                "ln_r": float(np.mean(ln_ratio)),
                "s_r": float(np.mean(sem_t[good] / exp_t[good])),
                "dln_r_dt": slope,
            }
        )
    return points


def _beta_sweep_vs_raw(overlay, window_ms):
    """MODEL-FREE sweep-chain consistency: ln(n*sqrt(Te)) - ln(raw Isat).

    If the sweep analysis were exactly the algebraic Isat inversion,
    n_exp*sqrt(Te_exp) would equal Isat_raw/(const_p) with const_p a
    per-port constant (geometry, e, sqrt(M), the calibrated A_p) -- so
    q_p = <ln(n_exp*sqrt(Te_exp)) - ln(Isat_raw)> over the plateau
    window must be RUNG-INDEPENDENT at fixed port.  Cross-rung structure
    in q_p is a condition-dependent pathology of the sweep chain itself
    (sheath expansion, fit window, circuit), diagnosed with no model in
    the loop -- independent corroboration in the pre-registered sense.
    Returns None when the raw drive trace is absent (schema < v3).
    """
    if "isat_drive_time_ms" not in overlay:
        return None
    t0, t1 = float(window_ms[0]), float(window_ms[1])
    t_raw = np.asarray(overlay["isat_drive_time_ms"], dtype=float)
    raw = np.asarray(overlay["isat_drive_mean_a"], dtype=float)
    raw_sem = np.asarray(overlay["isat_drive_sem_a"], dtype=float)
    raw_ports = list(np.asarray(overlay["isat_drive_port"], dtype=int))
    te_t = np.asarray(overlay["te_time_ms"], dtype=float)
    te_mean = np.asarray(overlay["te_mean_ev"], dtype=float)
    te_sem = np.asarray(overlay["te_sem_ev"], dtype=float)
    n_t = np.asarray(overlay["density_time_ms"], dtype=float)
    n_mean = np.asarray(overlay["density_mean_cm3"], dtype=float)
    n_sem = np.asarray(overlay["density_total_sem_cm3"], dtype=float)

    # This one clamps against the MEASURED raw-Isat trace, not a model
    # trajectory -- same silent-window failure, different extent. Hoisted out
    # of the port loop (loop-invariant) so a binding clamp says so once.
    t0_used = _clamp_window_bound(
        t0, float(t_raw.min()), "start",
        "beta-collapse plateau / _beta_sweep_vs_raw", "raw drive Isat trace",
    )
    t1_used = _clamp_window_bound(
        t1, float(t_raw.max()), "end",
        "beta-collapse plateau / _beta_sweep_vs_raw", "raw drive Isat trace",
    )

    points = []
    for p, port in enumerate(np.asarray(overlay["port"], dtype=int)):
        if port not in raw_ports:
            continue
        pr = raw_ports.index(port)
        window = (n_t >= t0_used) & (n_t <= t1_used)
        n_exp = n_mean[p, window]
        te_exp = np.interp(n_t[window], te_t, te_mean[p])
        te_sem_w = np.interp(n_t[window], te_t, te_sem[p])
        raw_w = np.interp(n_t[window], t_raw, raw[pr])
        raw_sem_w = np.interp(n_t[window], t_raw, raw_sem[pr])
        good = (
            np.isfinite(n_exp) & np.isfinite(te_exp) & np.isfinite(raw_w)
            & (n_exp > 0) & (te_exp > 0) & (raw_w > 0)
        )
        if np.count_nonzero(good) < 3:
            continue
        q = float(
            np.mean(
                np.log(n_exp[good] * np.sqrt(te_exp[good]))
                - np.log(raw_w[good])
            )
        )
        s_q = float(
            np.sqrt(
                np.mean(n_sem[p, window][good] / n_exp[good]) ** 2
                + (0.5 * np.mean(te_sem_w[good] / te_exp[good])) ** 2
                + np.mean(raw_sem_w[good] / raw_w[good]) ** 2
            )
        )
        points.append({"port": int(port), "q": q, "s_q": s_q})
    return points


def _centered(values):
    values = np.asarray(values, dtype=float)
    return values - np.mean(values)


def _beta_fingerprints(points):
    """Pearson r of centered ln beta_hat against the condition variables.

    Points must already be centered per (family, port) -- the ES-ladder-
    at-fixed-port lever: everything constant (probe, area, baked-in
    beta_cal) cancels, so only condition-dependent pathologies survive.
    """
    out = []
    lb = np.asarray([p["c_ln_beta"] for p in points], dtype=float)
    if lb.size < 3 or np.allclose(lb.std(), 0.0):
        return out
    for key, pathology in (
        ("te_ref", "fit-window / EEDF tail"),
        ("isat_ref", "circuit series resistance"),
        ("n_ref", "sheath expansion"),
    ):
        x = np.asarray([p["c_" + key] for p in points], dtype=float)
        if np.allclose(x.std(), 0.0):
            continue
        r = float(np.corrcoef(x, lb)[0, 1])
        out.append((key, pathology, r, lb.size))
    return out


def _report_beta_collapse(runs, svr_by_rung=None):
    print(
        "\n=== beta-collapse diagnostic "
        "(hypothesis on record 2026-07-22 + addenda) ==="
    )
    print(
        "  beta and a_{p,r} are hypothesis-test outputs, NEVER data\n"
        "  corrections, pending independent corroboration (re-fit raw\n"
        "  sweeps, varied windows). Constant Te bias is invisible (absorbed\n"
        "  by the per-port area calibration): this constrains\n"
        "  beta(z,t)/beta_cal and per-rung area jumps only.\n"
        "  Hierarchy: n axial shape interferometer-anchored (solid); Isat\n"
        "  axial shape carries per-port beta_cal spread (demoted);\n"
        "  within-shot taus fully robust (stage (iii))."
    )

    # --- Table 1: plateau sweep-window residual decomposition -------------
    print(
        "\n--- plateau sweep residuals per (run, rung, port); "
        "lic = Isat row within "
        f"{BETA_LICENSE_SIGMA:.0f} sigma (SEM (+) cal (+) area budget) ---"
    )
    header = (
        f"{'run':>8} {'rung':>4} {'port':>4} {'z':>6} "
        f"{'dlnTe':>7} {'dlnn':>7} {'dlnIsat':>8} {'+/-':>5} {'lic':>3} "
        f"{'ln_beta':>8} {'+/-':>5} {'ln_a':>7} {'+/-':>5}"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        for pt in run["sweep"]:
            print(
                f"{run['short']:>8} {run['rung']:>4} {pt['port']:>4} "
                f"{pt['z']:6.0f} {pt['dlnte']:7.3f} {pt['dlnn']:7.3f} "
                f"{pt['dlnisat']:8.3f} {pt['s_isat']:5.3f} "
                f"{'y' if pt['licensed'] else 'n':>3} "
                f"{pt['ln_beta']:8.3f} {pt['s_beta']:5.3f} "
                f"{pt['ln_a']:7.3f} {pt['s_a']:5.3f}"
            )
        if run["kind"] == "kinetic":
            print(f"{run['short']:>8}      (kinetic: afterglow unscored)")
    print(
        "  ln_beta = dlnTe (SEM-only noise: the Te systematic IS beta);\n"
        "  ln_a = dlnn + dlnTe/2, the manifold-perpendicular pure-n\n"
        "  component (area drift + model n error). Cross-port structure in\n"
        "  ln_a tests the model n axial shape (interferometer-anchored);\n"
        "  cross-port structure in ln_beta mixes model Te error with\n"
        "  per-port beta_cal spread and is NOT interpreted alone."
    )
    for run in runs:
        sw = run["sweep"]
        if len(sw) >= 2:
            lnb = np.asarray([p["ln_beta"] for p in sw])
            lna = np.asarray([p["ln_a"] for p in sw])
            print(
                f"  {run['short']} es{run['rung']}: cross-port spread "
                f"ln_beta std {lnb.std():.3f}, ln_a std {lna.std():.3f} "
                f"(vs typical sigma "
                f"{np.mean([p['s_beta'] for p in sw]):.3f} / "
                f"{np.mean([p['s_a'] for p in sw]):.3f})"
            )

    # --- Table 2: afterglow Isat residuals (moment runs) ------------------
    aft_runs = [r for r in runs if r["aft"]]
    if aft_runs:
        print(
            "\n--- afterglow Isat residuals ln(exp/model proxy) "
            "(per-port unit constant arbitrary; cross-rung differences "
            "meaningful) ---"
        )
        header = f"{'run':>8} {'rung':>4} " + " ".join(
            f"p{pt['port']:>2}:{'ln_r':>6}" for pt in aft_runs[0]["aft"]
        )
        print(header)
        for run in aft_runs:
            row = f"{run['short']:>8} {run['rung']:>4} "
            row += " ".join(
                f"{pt['ln_r']:9.3f}+/-{pt['s_r']:.3f}" for pt in run["aft"]
            )
            print(row)

    # --- Within-shot guard (addendum-2 native form) ------------------------
    # Raw drive Isat and the decay trace come from the SAME channel, zero
    # offsets and shot ensemble (exporter contract, 7h), so the per-port
    # unit constant cancels exactly in g_shot = ln r(drive) - ln r(aft):
    # an area nuisance a_{p,r} must give g_shot = 0 within errors, per
    # rung, with no ladder needed.  Any condition dependence -- sweep
    # pathology or model regime error -- fails it.
    ws_runs = [r for r in runs if r.get("drive") and r.get("aft")]
    if ws_runs:
        print(
            "\n--- guard (within-shot, native form): raw drive Isat vs "
            "afterglow, per (run, rung, port) ---"
        )
        header = (
            f"{'run':>8} {'rung':>4} {'port':>5} {'ln_r_drv':>8} "
            f"{'ln_r_aft':>8} {'g_shot':>7} {'+/-':>5} {'g/sig':>6} "
            f"{'aftslope':>8}  verdict"
        )
        print(header)
        print("-" * len(header))
        for run in ws_runs:
            aft = {pt["port"]: pt for pt in run["aft"]}
            lic = {pt["port"]: pt["licensed"] for pt in run["sweep"]}
            for pt in run["drive"]:
                a_pt = aft.get(pt["port"])
                if a_pt is None:
                    continue
                g = float(pt["ln_r"] - a_pt["ln_r"])
                s_g = float(np.hypot(pt["s_r"], a_pt["s_r"]))
                nsig = abs(g) / s_g if s_g > 0 else np.inf
                verdict = "PASS" if nsig <= 2.0 else "FAIL"
                # A sloped afterglow residual means the model decay rate
                # is off (stage (iii) tau error) and its accumulated
                # drift, not an area/condition offset, may carry g_shot:
                # flag rows where the drift over the window covers g.
                drift = abs(a_pt.get("dln_r_dt", 0.0)) * (
                    BETA_AFTERGLOW_MS[1] - BETA_AFTERGLOW_MS[0]
                )
                if verdict == "FAIL" and drift >= abs(g):
                    verdict = "FAIL (tau-confounded)"
                if not lic.get(pt["port"], False):
                    verdict += " (unlicensed)"
                print(
                    f"{run['short']:>8} {run['rung']:>4} {pt['port']:>5} "
                    f"{pt['ln_r']:8.3f} {a_pt['ln_r']:8.3f} "
                    f"{g:7.3f} {s_g:5.3f} {nsig:6.1f} "
                    f"{a_pt['dln_r_dt']:8.3f}  {verdict}"
                )
        print(
            "  aftslope = d ln(exp/model)/dt [1/ms] in the afterglow: a\n"
            "  pure area/unit offset is flat; a nonzero slope is the model\n"
            "  decay-rate error leaking into the window mean, and FAILs\n"
            "  whose g is covered by that drift are marked tau-confounded."
        )
    else:
        print(
            "\n--- guard (within-shot): awaiting raw drive Isat "
            "(isat_drive_*, overlay schema v4; exporter brief 7h) -- "
            "falling back to the cross-rung form ---"
        )

    # --- Model-free sweep-chain consistency (raw vs sweep-synth Isat) -----
    svr_by_rung = {
        rung: pts for rung, pts in (svr_by_rung or {}).items() if pts
    }
    if svr_by_rung:
        print(
            "\n--- sweep-chain consistency, MODEL-FREE: "
            "q = <ln(n_exp*sqrt(Te_exp)) - ln(Isat_raw)>, plateau ---"
        )
        print(
            "  q_p is a per-port constant if the sweep analysis is\n"
            "  algebraically consistent; cross-rung structure in q_p is a\n"
            "  condition-dependent sweep pathology with NO model in the\n"
            "  loop (per-port constants cancel when centered)."
        )
        rungs = sorted(svr_by_rung)
        ports = sorted(
            {pt["port"] for pts in svr_by_rung.values() for pt in pts}
        )
        for port in ports:
            vals = {
                rung: next(
                    (p for p in svr_by_rung[rung] if p["port"] == port), None
                )
                for rung in rungs
            }
            have = [r for r in rungs if vals[r] is not None]
            if len(have) < 2:
                continue
            c_q = _centered([vals[r]["q"] for r in have])
            txt = ", ".join(
                f"es{r}: {cq:+.3f}+/-{vals[r]['s_q']:.3f}"
                for r, cq in zip(have, c_q)
            )
            worst = float(
                np.max([abs(cq) / vals[r]["s_q"] for r, cq in zip(have, c_q)])
            )
            verdict = "consistent" if worst <= 2.0 else "CONDITION-DEPENDENT"
            print(f"  port {port}: centered q ({txt}) -> {verdict}")

    # --- Guard / collapse test: ES ladder at fixed port -------------------
    # Group moment runs by family; need >= 2 rungs for the centered test.
    families = {}
    for run in runs:
        families.setdefault((run["family"], run["kind"]), []).append(run)
    tested_any = False
    fingerprint_points = []
    for (family, kind), members in sorted(families.items()):
        members = sorted(members, key=lambda r: r["rung"])
        short = members[0]["short"]
        # Collect (port -> per-rung plateau/afterglow values).
        per_port = {}
        for run in members:
            aft = {pt["port"]: pt for pt in run["aft"] or []}
            for pt in run["sweep"]:
                a_pt = aft.get(pt["port"])
                per_port.setdefault(pt["port"], []).append(
                    (run["rung"], pt, a_pt)
                )
        # Fingerprint points: centered per (family, port) across rungs,
        # licensed plateau points only.
        for port, entries in per_port.items():
            lic = [(r, pt) for r, pt, _ in entries if pt["licensed"]]
            if len(lic) >= 2:
                for key in ("ln_beta",):
                    vals = _centered([pt[key] for _, pt in lic])
                    for (rung, pt), v in zip(lic, vals):
                        rec = {"c_ln_beta": v, "rung": rung, "port": port}
                        for ckey in ("te_ref", "isat_ref", "n_ref"):
                            rec["c_" + ckey] = np.log(pt[ckey]) - np.mean(
                                [np.log(q[ckey]) for _, q in lic]
                            )
                        fingerprint_points.append(rec)
        if kind == "kinetic":
            continue
        complete = {
            port: entries
            for port, entries in per_port.items()
            if len(entries) >= 2 and all(a is not None for _, _, a in entries)
        }
        if not complete:
            continue
        tested_any = True
        print(
            f"\n--- guard / collapse (cross-rung form): {short} ladder at "
            "fixed port (centered across rungs) ---"
        )
        print(
            "  g = centered ln_a(plateau) - centered ln_r(afterglow): an\n"
            "  area jump a_{p,r} must shift both identically, so g ~ 0\n"
            "  within errors PASSES (sweep bias + area drift suffice);\n"
            "  after the afterglow-pinned area correction g is ALSO the\n"
            "  collapse residual perpendicular to the (1,-1/2) manifold --\n"
            "  z/regime structure in g is the model-error component."
        )
        header = (
            f"{'port':>5} {'rung':>4} {'c_ln_a':>7} {'c_ln_r':>7} "
            f"{'g':>7} {'+/-':>5} {'g/sig':>6}  verdict"
        )
        print(header)
        print("-" * len(header))
        for port in sorted(complete):
            entries = sorted(complete[port], key=lambda e: e[0])
            c_a = _centered([pt["ln_a"] for _, pt, _ in entries])
            c_r = _centered([a["ln_r"] for _, _, a in entries])
            for (rung, pt, a_pt), ca, cr in zip(entries, c_a, c_r):
                g = float(ca - cr)
                s_g = float(np.hypot(pt["s_a"], a_pt["s_r"]))
                nsig = abs(g) / s_g if s_g > 0 else np.inf
                verdict = "PASS" if nsig <= 2.0 else "FAIL"
                if not pt["licensed"]:
                    verdict += " (unlicensed)"
                print(
                    f"{port:>5} {rung:>4} {ca:7.3f} {cr:7.3f} "
                    f"{g:7.3f} {s_g:5.3f} {nsig:6.1f}  {verdict}"
                )
    if not tested_any:
        print(
            "\n--- guard / collapse: skipped (needs a moment-run family on "
            ">= 2 ES rungs with afterglow coverage) ---"
        )

    # --- Beta fingerprints -------------------------------------------------
    print("\n--- beta fingerprints (centered per family x port, licensed) ---")
    fps = _beta_fingerprints(fingerprint_points)
    if not fps:
        print("  skipped: needs >= 3 licensed cross-rung points")
    else:
        for key, pathology, r, npts in fps:
            print(
                f"  corr(ln beta, ln {key:<8}) = {r:+5.2f}  (N={npts})"
                f"  -> {pathology}"
            )
        # The ES ladder moves Te, n and Isat together; when the condition
        # variables are themselves collinear the fingerprints cannot
        # separate the pathologies -- say so rather than invite
        # over-reading.
        conds = np.asarray(
            [
                [p["c_te_ref"], p["c_isat_ref"], p["c_n_ref"]]
                for p in fingerprint_points
            ]
        )
        if conds.shape[0] >= 3:
            cc = np.corrcoef(conds.T)
            pair_min = float(
                np.min(np.abs(cc[np.triu_indices_from(cc, k=1)]))
            )
            if pair_min > 0.8:
                print(
                    "  CAVEAT: the condition variables are mutually "
                    f"collinear across the ladder (min |corr| {pair_min:.2f});"
                    "\n  the fingerprints above do NOT discriminate the "
                    "pathologies. Discrimination needs variation that\n"
                    "  breaks the ladder degeneracy: varied fit-window "
                    "re-fits of the raw sweeps (the pre-registered\n"
                    "  corroboration path) or within-window time structure."
                )
        rungs = sorted({p["rung"] for p in fingerprint_points})
        if len(rungs) >= 2:
            means = [
                (
                    rung,
                    float(
                        np.mean(
                            [
                                p["c_ln_beta"]
                                for p in fingerprint_points
                                if p["rung"] == rung
                            ]
                        )
                    ),
                )
                for rung in rungs
            ]
            txt = ", ".join(f"es{r}: {m:+.3f}" for r, m in means)
            print(f"  per-rung mean centered ln beta ({txt}) -> day-to-day/"
                  "surface if structured")


def beta_collapse_main(specs, plateau_ms, afterglow_ms):
    script_dir = Path(__file__).resolve().parent
    if not specs:
        specs = []
        for family in BETA_CANONICAL_FAMILIES:
            for rung in (1, 2, 3):
                cand = script_dir / BETA_CANONICAL_PATTERN.format(
                    family=family, rung=rung
                )
                if cand.exists():
                    specs.append(str(cand))
        if not specs:
            print("beta-collapse: no canonical runs found in scripts/")
            return 1
    runs = []
    svr_by_rung = {}
    for spec in specs:
        path, rung, kind = _parse_beta_run(spec)
        if not path.exists():
            print(f"beta-collapse: missing run {path}")
            return 1
        overlay = np.load(
            OVERLAY.parent / f"es{rung}_sim1d_overlay.npz", allow_pickle=False
        )
        result = load_result_hdf5(path)
        family = re.sub(r"_es\d+$", "", path.stem)
        runs.append(
            {
                "path": path,
                "family": family,
                "short": family.rsplit("_", 1)[-1],
                "rung": rung,
                "kind": kind,
                "sweep": _beta_sweep_points(result, overlay, plateau_ms),
                # kinetic afterglow unscored (quasi-static kinetic neutrals)
                "aft": (
                    None
                    if kind == "kinetic"
                    else _beta_trace_points(
                        result, overlay, afterglow_ms, "isat_decay"
                    )
                ),
                # Raw inter-sweep Isat during the drive (schema v4, no
                # area factor; None until the exporter lands, 7h brief).
                # Coarse cell cadence, so a low sample floor.
                "drive": _beta_trace_points(
                    result, overlay, plateau_ms, "isat_drive", min_samples=3
                ),
            }
        )
        svr_by_rung.setdefault(
            rung, _beta_sweep_vs_raw(overlay, plateau_ms)
        )
    print(
        "beta-collapse reference runs: "
        + ", ".join(
            f"{r['short']} es{r['rung']} ({r['kind']})" for r in runs
        )
    )
    print(
        f"  plateau window {plateau_ms[0]:.1f}-{plateau_ms[1]:.1f} ms, "
        f"afterglow window {afterglow_ms[0]:.1f}-{afterglow_ms[1]:.1f} ms"
    )
    _report_beta_collapse(runs, svr_by_rung)
    return 0


def _report_peak_current(peak):
    print("\n--- stage (i): discharge current ---")
    print(
        f"  peak:    model {peak['model_peak_a']:8.4g} A at {peak['model_peak_t_ms']:+6.2f} ms"
        f" | measured {peak['exp_peak_a']:8.4g} +/- {peak['exp_peak_sem_a']:.2g} A"
        f" at {peak['exp_peak_t_ms']:+6.2f} ms | ratio {peak['ratio']:.3f}"
    )
    print(
        f"  plateau: model {peak['model_late_a']:8.4g} A (15-19.5 ms mean)"
        f" | measured {peak['exp_late_a']:8.4g} A | ratio {peak['late_ratio']:.3f}"
    )


def _report_decay(rows, window):
    print(
        f"\n--- stage (iii): Isat decay e-fold times, window "
        f"{window[0]:.1f}-{window[1]:.1f} ms ---"
    )
    header = f"{'port':>6} {'z [cm]':>8} {'tau_model':>10} {'tau_exp':>9} {'ratio':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['port']:>6} {r['z']:8.0f} {r['tau_model_ms']:9.2f}ms "
            f"{r['tau_exp_ms']:8.2f}ms {r['ratio']:7.2f}"
        )
    ratios = [r["ratio"] for r in rows if np.isfinite(r["ratio"])]
    if ratios:
        print(f"  mean tau_model/tau_exp: {np.mean(ratios):.2f}")


def _report(label, rows):
    print("\n--- stage (ii): bulk Te / density at the ES1 ports ---")
    spread_gated = any(r.get("spread_gated") for r in rows)
    if spread_gated:
        pct = 100.0 * TE_SPREAD_SEMIQUANT_FRAC
        print("  (sigma = |dev|/sigma_tot, SEM (+) sweep systematics;")
        print(f"   semi-quantitative marks: '~' Te refit-window spread > {pct:.0f}%")
        print("   [primary; n rows at half strength, n ~ Isat/sqrt(Te)],")
        print("   '*' measured Te < 1 eV [secondary], '?' spread unavailable")
        print("   -> UNDETERMINED, the row keeps its Te-criterion verdict)")
    else:
        print("  (sigma = |dev|/sigma_tot, SEM (+) sweep systematics; '~' marks")
        print("   semi-quantitative rows where measured Te < 1 eV)")
    header = (
        f"{'field':>5} {'port':>6} {'z [cm]':>8} {'model':>11} {'measured':>11} "
        f"{'ratio':>7} {'rms rel':>8} {'|dev|/sig':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("spread_gated"):
            marks = ""
            if r.get("semiquant_spread"):
                marks += "~"
            if r.get("semiquant_te"):
                marks += "*"
            if r.get("spread_undetermined"):
                marks += "?"
        else:
            marks = "~" if r.get("semiquant") else ""
        # Min-width 1 keeps the un-marked and single-marked rows rendering
        # exactly as they did before the spread criterion existed.
        print(
            f"{r['field']:>5} {r['port']:>6} {r['z']:8.0f} {r['model']:11.4g} "
            f"{r['exp']:11.4g} {r['ratio']:7.2f} {r['rms_rel']:8.2f} "
            f"{r['sigma']:9.1f}{marks:<1}"
        )
    for field in ("Te", "n", "Isat"):
        sub = [r for r in rows if r["field"] == field]
        if sub:
            print(
                f"  {field}: mean ratio {np.mean([r['ratio'] for r in sub]):.2f}, "
                f"mean rms rel {np.mean([r['rms_rel'] for r in sub]):.2f}, "
                f"mean |dev|/sig {np.mean([r['sigma'] for r in sub]):.1f}"
            )
        if len(sub) >= 2:
            # Axial-gradient figure of merit: far-port / near-port ratio.
            # 1.00 means the model's axial falloff matches the measured one
            # regardless of overall magnitude.
            first, last = sub[0], sub[-1]
            grad_model = last["model"] / first["model"]
            grad_exp = last["exp"] / first["exp"]
            print(
                f"  {field} axial gradient (z={last['z']:.0f}/{first['z']:.0f}): "
                f"model {grad_model:.2f} vs measured {grad_exp:.2f} "
                f"(ratio {grad_model / grad_exp:.2f})"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=PRODUCTION_NX)
    parser.add_argument(
        "--exchange-model", default="knudsen", choices=("knudsen", "constant")
    )
    parser.add_argument(
        "--tau-afterglow",
        type=float,
        default=None,
        metavar="S",
        help="override the afterglow duration [s] to cover more of the decay",
    )
    parser.add_argument(
        "--decay-window",
        type=float,
        nargs=2,
        default=DECAY_WINDOW_MS,
        metavar=("T0", "T1"),
        help="stage (iii) fit window on the main-discharge clock [ms]",
    )
    parser.add_argument(
        "--save-h5",
        type=Path,
        default=None,
        help="save the run result for later re-scoring",
    )
    parser.add_argument(
        "--from-h5",
        type=Path,
        default=None,
        help="score a saved result instead of running the model",
    )
    parser.add_argument(
        "--drag-closure",
        default=None,
        choices=(
            "constant",
            "slip",
            "neutral_momentum",
            "neutral_momentum_two_zone",
        ),
        help=(
            "swap the ion-neutral drag closure for the gate-#2 A/B: "
            "constant (production 0.5), slip (entrainment closure, b=1), "
            "neutral_momentum (evolved M_n wind, b=1), or "
            "neutral_momentum_two_zone (M_n wind + two-zone radial closure)"
        ),
    )
    parser.add_argument(
        "--Rp-model",
        default=None,
        choices=("sample", "resolved_gap"),
        help=(
            "cathode gap-resistance model for the M1 A/B: "
            "sample (historical one-cell "
            "Spitzer) or resolved_gap (profile-integrated over the gap)"
        ),
    )
    parser.add_argument(
        "--beam-excitation",
        default=None,
        choices=("scalar14", "manifold"),
        help=(
            "beam excitation channel for the WP-A A/B "
            "(A3): scalar14 (production 2p_scalar "
            "with the historical b=1.4 estimate) or manifold (measured "
            "Ralchenko singlet sum, b=1.0)"
        ),
    )
    parser.add_argument(
        "--beam-deposition",
        default=None,
        choices=("beer_lambert", "csda", "csda_ql"),
        help=(
            "beam deposition model for the WP-B B3 A/B: "
            "beer_lambert (historical "
            "single-event absorption), csda (slowing-down module, classical "
            "fast-electron Coulomb), or csda_ql (csda + quasilinear "
            "beam-plasma drag)"
        ),
    )
    parser.add_argument(
        "--beam-product-transport",
        default=None,
        choices=("local", "nonlocal"),
        help=(
            "beam product transport for the WP-D A/B: "
            "local (production stance and "
            "config default -- products thermalize where they are born) or "
            "nonlocal (products walk, and the escape ledger is live). "
            "nonlocal requires the CSDA deposition module and raises at "
            "construction under --beam-deposition beer_lambert"
        ),
    )
    parser.add_argument(
        "--es",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help=(
            "which experiment-set overlay to score against "
            "(data/es{N}_sim1d_overlay.npz; ES1-3 share fueling and differ "
            "only in heater current and bank voltage — the drive-side "
            "ladder). NB the model config must match the "
            "campaign's operating point; this flag only selects the data."
        ),
    )
    parser.add_argument(
        "--beta-collapse",
        nargs="*",
        default=None,
        metavar="RUN.h5[:es=N][:kind=kinetic|moment]",
        help=(
            "run the sweep-bias beta-collapse diagnostic over saved "
            "reference runs (hypothesis on record + addenda) instead "
            "of scoring a single run; rung/kind parse from the file name. "
            "With no arguments, picks up the canonical "
            "es1_nx120_m6_sq3400_{2zbase,2z,k4t}_es{1,2,3}.h5 set as "
            "available. Kinetic runs are scored in the plateau only."
        ),
    )
    parser.add_argument(
        "--beta-plateau",
        type=float,
        nargs=2,
        default=BETA_PLATEAU_MS,
        metavar=("T0", "T1"),
        help="beta-collapse plateau window on the main-discharge clock [ms]",
    )
    parser.add_argument(
        "--beta-afterglow",
        type=float,
        nargs=2,
        default=BETA_AFTERGLOW_MS,
        metavar=("T0", "T1"),
        help="beta-collapse afterglow window on the main-discharge clock [ms]",
    )
    args = parser.parse_args(argv)

    if args.beta_collapse is not None:
        return beta_collapse_main(
            args.beta_collapse, args.beta_plateau, args.beta_afterglow
        )

    overlay_path = (
        OVERLAY
        if args.es == 1
        else OVERLAY.parent / f"es{args.es}_sim1d_overlay.npz"
    )
    overlay = np.load(overlay_path, allow_pickle=False)
    if args.from_h5 is not None:
        result = load_result_hdf5(args.from_h5)
        geometry = None
        scored_params = getattr(result, "params", None)
        label = f"saved run {args.from_h5}"
        # The artifact's own config is the authority on which arm it is; a
        # nonlocal run must not be scored under a production-stance label.
        label += beam_product_transport_note(getattr(result, "params", None))
    else:
        label = f"resolved ({args.exchange_model}, nx={args.nx or 'default'})"
        if args.drag_closure is not None:
            label += f" [drag={args.drag_closure}]"
        if args.Rp_model is not None:
            label += f" [Rp={args.Rp_model}]"
        if args.beam_excitation is not None:
            label += f" [beam_exc={args.beam_excitation}]"
        extra = {}
        if args.tau_afterglow is not None:
            extra["tau_afterglow"] = args.tau_afterglow
        # A/B instrument for A3: the measured singlet
        # manifold vs the retired 1.4 estimate. "scalar14" is PARAM_OVERRIDES
        # as-is; "manifold" swaps the cross-section set and drops b to the
        # pure-multiplier benchmark value.
        if args.beam_excitation == "manifold":
            extra["beam_excitation_model"] = "manifold"
            extra["b_beam_excitation"] = 1.0
        if args.beam_deposition is not None:
            label += f" [dep={args.beam_deposition}]"
            extra["beam_deposition_model"] = (
                "csda" if args.beam_deposition.startswith("csda")
                else "beer_lambert"
            )
            if args.beam_deposition == "csda_ql":
                extra["beam_anomalous_model"] = "quasilinear"
        # WP-D arm. Lands in `extra`, which run_model applies LAST, so it wins
        # over PARAM_OVERRIDES; PARAM_OVERRIDES itself never sets the key, so
        # omitting the flag reproduces the production stance exactly.
        if args.beam_product_transport is not None:
            extra["beam_product_transport"] = args.beam_product_transport
            label += beam_product_transport_note(extra)
        # No BreakdownError handler here: this driver never sets
        # prebreakdown_timeout_action, so it always runs the "switch_open"
        # default, under which a failed breakdown ends as an OPENED SWITCH
        # (a scorable non-ignited trajectory that _main_discharge_origin
        # rejects by name) and never as an exception. The dead handler used
        # to imply the "raise" mode was reachable from here; it is not.
        result, geometry, params, flags = run_model(
            nx=args.nx,
            exchange_model=args.exchange_model,
            extra=extra,
            drag_closure=args.drag_closure,
            Rp_model=args.Rp_model,
        )
        scored_params = params
        if args.save_h5 is not None:
            save_result_hdf5(args.save_h5, result, params=params, flags=flags)
            print(f"saved result to {args.save_h5}")
    print(f"\n=== {label} ===")
    print(wpe_arm_line(scored_params))
    _report_peak_current(compare_peak_current(result, overlay))
    _report(label, compare(result, geometry, overlay))
    decay_rows, window = compare_decay(result, overlay, window_ms=args.decay_window)
    _report_decay(decay_rows, window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
