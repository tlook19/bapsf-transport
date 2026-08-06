"""K5 spot check: the frozen DVM arm is bit-identical under the rate default.

Rebuilds the config of the frozen production arm
(``scripts/es1_k3a_cal2_nx240.cmd``: ``run_m6_point.py --es 1 --sgp 3000
--nx 240 --two-zone`` with its ``--extra`` set), advances a bounded number of
steps, and prints SHA-256 digests of the raw little-endian float64 bytes of
every piece of state the arm carries. Two checkouts that print the same
digests carried the same trajectory to the last bit.

The default ``neutral_kinetic_dvm_annulus_flights = "rates"`` is what this
holds fixed: the bounded-chord annulus must be reachable ONLY through that
selector, so a checkout that has it and a checkout that has never heard of it
must agree here exactly.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/k5_frozen_bitexact.py --steps 2500
"""

import argparse
import hashlib

import numpy as np

from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES
from run_mechanism_ladder import ES_OPERATING

from cablp.solvers._sim1d import LAPDSim1D, default_config


# The frozen arm's argv, transcribed: run_m6_point's own ES-1 block plus the
# --extra set the .cmd carries. Kept here verbatim so the check does not
# depend on a driver that a later pass may re-default.
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
    "C_R": 12.46,
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
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--report-every", type=int, default=500)
    args = p.parse_args(argv)

    sim = build()
    # getattr: this same file is run against a checkout that predates the
    # selector, which is the whole point of the comparison.
    print(f"cells={sim.geometry.cells} arm={sim._dvm is not None} "
          f"flights={getattr(sim._dvm, 'annulus_flights', 'rates')!r}")
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
    print("k5 frozen-arm digests printed")


if __name__ == "__main__":
    main()
