"""Attribution check: the handshake fix moves the DVM arm and nothing else.

Runs two configurations and prints SHA-256 digests of the raw float64 state
bytes, in the manner of ``k6_frozen_bitexact.py``:

  no-dvm   the K6 frozen arm's config with the tail walk and tail ionization
           ON but the moment neutral model -- i.e. everything the fix does
           NOT touch, including the K5/K6 machinery two other branches work
           in. A base checkout and a fixed checkout MUST print the same
           digests here.
  dvm      the same config with the kinetic_dvm arm engaged. The digests
           MOVE here by design: the arm now debits the count the plasma
           booked instead of its own post-march tally.

Run it against this checkout and against the base commit's package (point
PYTHONPATH at each) and diff the two outputs.
"""

import argparse
import hashlib

import numpy as np

from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES
from run_mechanism_ladder import ES_OPERATING

from cablp.solvers._sim1d import LAPDSim1D, default_config


COMMON = {
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
    "C_R": 12.96,
    "heating_anomalous_transport": "tail_walk",
    "heating_anomalous_tail_ionization": "on",
}
DVM_ONLY = {
    "neutral_model": "kinetic_dvm",
    "neutral_kinetic_dvm_cadence_s": 1.0e-5,
    "neutral_kinetic_dvm_nvz": 96,
    "neutral_kinetic_dvm_nvp": 32,
    "neutral_kinetic_dvm_exchange": "cauchy_chord",
    "neutral_kinetic_dvm_annulus_flights": "bounded_chord",
}


def build(with_dvm):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    flags["neutral_two_zone"] = True
    params["neutral_exchange_model"] = "knudsen"
    op = ES_OPERATING[1]
    params["V_bank"] = op["V_bank"]
    params["cathode_Ts_base_K"] = op["Ts_standby_K"]
    params.update(COMMON)
    if with_dvm:
        params.update(DVM_ONLY)
    return LAPDSim1D(params, flags)


def digest(*arrays):
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(np.asarray(arr, dtype="<f8")).tobytes())
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--report-every", type=int, default=200)
    args = p.parse_args(argv)

    for label, with_dvm in (("no-dvm", False), ("dvm", True)):
        sim = build(with_dvm)
        print(f"--- {label}: arm={sim._dvm is not None} cells={sim.geometry.cells}")
        for step in range(1, args.steps + 1):
            sim.advance_one_step()
            if step % args.report_every == 0 or step == args.steps:
                line = (
                    f"{label} step {step:6d} t={sim.time:.9e}"
                    f"  y {digest(sim._y)}"
                )
                if sim._dvm is not None:
                    line += (
                        f"\n{label} step {step:6d}"
                        f"  f_c {digest(sim._dvm.f_c)}"
                        f"  f_a {digest(sim._dvm.f_a)}"
                    )
                print(line, flush=True)


if __name__ == "__main__":
    main()
