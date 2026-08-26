"""K6 spot check: the frozen K5a arm is bit-identical under the K6 default.

Rebuilds the config of the frozen K5a calibration arm
(``scripts/es1_k5a_shot1_nx240.cmd``: ``run_m6_point.py --es 1 --sgp 3000
--nx 240 --two-zone`` with its ``--extra`` set, C_R 12.96 and the bounded-chord
annulus), advances a bounded number of steps, and prints SHA-256 digests of the
raw little-endian float64 bytes of every piece of state the arm carries. Two
checkouts that print the same digests carried the same trajectory to the last
bit.

The default ``heating_anomalous_tail_ionization = "off"`` is what this holds
fixed. K6 adds a config key, four result fields, three cathode diagnostics and
a branch inside the WP-E tail walk; none of it may reach a run that did not ask
for it. A checkout that has the channel and a checkout that has never heard of
it must agree here exactly. (This arm does not even run the tail walk -- its
``heating_anomalous_transport`` is the default ``"local"`` -- which is the
point: the K6 branch must be unreachable from two selectors away, not merely
inert one selector away.)

Mirrors ``scripts/k5_frozen_bitexact.py``, which does the same job for the
annulus selector; the EXTRA block below is shot 1's rather than shot 0's.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/k6_frozen_bitexact.py --steps 400
"""

import argparse
import hashlib

import numpy as np

from run_mechanism_ladder import ES_OPERATING

from cablp.solvers._sim1d import LAPDSim1D, default_config


# --------------------------------------------------------------------------
# FROZEN SNAPSHOT of compare_sim1d_es1.PARAM_OVERRIDES / FLAG_OVERRIDES, taken
# 2026-08-26 at the tip that froze it. This file used to IMPORT those two
# dicts LIVE, which is not a freeze at all: several PARAM_OVERRIDES entries
# mirror scripts/stances/g1atrim.toml (S_gp, C_R, b_beam_excitation, the two
# cathode power-balance areals, equilibration_gas_puff_on_s), so every stance
# re-point silently rebased this bank -- demonstrated by the arm's cell count
# moving 262 -> 252 across the L2 geometry rebaseline with no edit to this
# file. A bank whose configuration drifts underneath it cannot certify that
# two checkouts carried the same trajectory.
#
# The values are kept here verbatim, exactly the way EXTRA below already is.
# The provenance commentary for each one lives with the live dicts in
# scripts/compare_sim1d_es1.py and in scripts/production_stance_provenance.md;
# it is deliberately NOT duplicated here, because this block is a dated
# snapshot and that commentary is not.
PARAM_OVERRIDES = {
    "V_bank": 177.843,
    "R_comp": 0.0072244,
    "L_parasitic_H": 8.1e-06,
    "C_bank_F": 9.5,
    "equilibration_gas_puff_on_s": 0.025,
    "S_gp": 9010.0,
    "tau_gp_pulse_duration": 0.001,
    "tau_gp_decay_duration": 0.005,
    "atomic_rate_model": "adas",
    "b_beam_excitation": 1.4,
    "b_Qei": 1,
    "b_Qen": 1,
    "b_Qcx": 1,
    "Rp": 18.415,
    "R_cath": 18.415,
    "implicit_heat_scheme": "tr_bdf2",
    "operator_splitting": "strang",
    "heat_picard_iterations": 2,
    "heat_picard_tol": 1e-10,
    "Rsup": 0.0,
    "cathode_conduction_W_per_K": 12058.0,
    "cathode_heat_capacity_J_per_K": 181.0,
    "C_R": 8.76,
    "beam_deposition_smoothing_cm": 50.0,
}
FLAG_OVERRIDES = {
    "ion_neutral_drag_cx_only": False,
}
# --------------------------------------------------------------------------

# The frozen arm's argv, transcribed from es1_k5a_shot1_nx240.cmd. Kept here
# verbatim so the check does not depend on a driver a later pass may re-default.
EXTRA = {
    "cathode_solver_model": "current_driven",
    "beam_deposition_model": "csda",
    "beam_anomalous_model": "quasilinear",
    "cathode_emission_profile": "gaussian",
    "cathode_warming_model": "power_balance",
    "cathode_heat_capacity_J_per_K": 120.0,
    "cathode_emissivity": 0.7,
    "phi_wf": 2.869,
    "cathode_surface_model": "ads_des",
    "cathode_phiwf_clean_eV": 2.809,
    "cathode_cleaning_sigma_cm2": 3.5e-16,
    "cathode_cleaning_E_th_eV": 20.0,
    "cathode_sample_smoothing": "presheath",
    "gas_puff_mode": "square",
    # SUPERSEDED 2026-08-21: sccm now MEANS meter-sccm (4.171431e17 particles/s
    # per sccm, 20 C / 1013 mbar), so this literal ships ~6.85 % less flux than
    # it did when the arm ran. Left AS A RECORD of what this dated script ran.
    "S_gp": 3000.0,
    "nx": 240,
    "T_s": 1998.15,
    "Te_birth_ionization": "local",
    "tau_afterglow": 0.006,
    "neutral_model": "kinetic_dvm",
    "neutral_kinetic_dvm_cadence_s": 1.0e-5,
    "neutral_kinetic_dvm_nvz": 96,
    "neutral_kinetic_dvm_nvp": 32,
    "neutral_kinetic_dvm_exchange": "cauchy_chord",
    "neutral_kinetic_dvm_annulus_flights": "bounded_chord",
    "C_R": 12.96,
}


def build():
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    flags["neutral_two_zone"] = True
    params["neutral_exchange_model"] = "knudsen"
    op = ES_OPERATING[1]
    params["V_bank"] = op["V_bank"]
    params["cathode_Ts_base_K"] = op["Ts_standby_K"]
    params.update(EXTRA)
    return LAPDSim1D(params, flags)


def digest(*arrays):
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
        h.update(a.tobytes())
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--report-every", type=int, default=100)
    args = p.parse_args(argv)

    sim = build()
    # getattr: this same file is run against a checkout that predates the
    # selector, which is the whole point of the comparison.
    print(
        f"cells={sim.geometry.cells} arm={sim._dvm is not None} "
        f"tail_ionization="
        f"{sim._input_dict.get('heating_anomalous_tail_ionization', 'off')!r} "
        f"anom_transport="
        f"{sim._input_dict.get('heating_anomalous_transport', 'local')!r}"
    )
    for step in range(1, args.steps + 1):
        sim.advance_one_step()
        if step % args.report_every == 0 or step == args.steps:
            dvm = sim._dvm
            print(
                f"step {step:6d}  t={sim.time:.9e}  updates={dvm.updates:6d}\n"
                f"    y      {digest(sim._y)}\n"
                f"    f_c    {digest(dvm.f_c)}\n"
                f"    f_a    {digest(dvm.f_a)}\n"
                f"    pend   {digest(dvm.pend_L_c, dvm.pend_R_c, dvm.pend_L_a, dvm.pend_R_a)}\n"
                f"    xfer   {digest(dvm.M_transfer, dvm.Ei_transfer, dvm.S_transfer, dvm.Tn_col_eV)}\n"
                f"    debt   {digest(dvm.M_debt, dvm.Ei_debt, dvm.M_applied_cum, dvm.Ei_applied_cum, dvm.M_booked_cum, dvm.Ei_booked_cum)}",
                flush=True,
            )
    print("k6 frozen-arm digests printed")


if __name__ == "__main__":
    main()
