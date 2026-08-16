"""sprobe: raw-byte bit-exactness probe for the ad-hoc probe neutral source.

The headline house gate for new physics: with the feature ABSENT the solver
must reproduce the pre-branch trajectory to the last bit. "Absent" here is
literal -- the flag is never named and none of the ten parameters is ever set,
so every config below is expressible on the BASE checkout too and the same
source file runs unchanged on both.

Four arms, none of them a config the smoke suite uses: two cathode/deposition
stances crossed with two neutral structures (single zone, and the two-zone
closure with an evolved neutral wind). Each prints SHA-256 over the raw state
vector bytes plus the circuit current and the clock. Run under the BASE
checkout's PYTHONPATH and the BRANCH checkout's PYTHONPATH; every line must be
identical.

Usage (from either checkout's cablp, PYTHONPATH set to that same cablp):
    python sprobe_bitexact_probe.py
"""
import hashlib

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config


def run(tag, deposition, two_zone):
    params, flags = default_config()
    params.update({
        "nx": 20,
        "beam_deposition_model": deposition,
        "cathode_solver_model": "current_driven",
        "cathode_emission_profile": "gaussian",
        "cathode_warming_model": "none",
        "cathode_Ts_base_K": None,
        "cathode_surface_model": "none",
        "cathode_phiwf_clean_eV": None,
        "cathode_cleaning_E_th_eV": None,
        "cathode_sample_smoothing": None,
    })
    if deposition == "csda":
        params["beam_anomalous_model"] = "quasilinear"
    flags = dict(flags)
    flags["neutral_equilibration"] = False
    if two_zone:
        # The structure the probe's zone selector would reach, exercised with
        # the probe absent: two neutral fields and an evolved wind.
        params["neutral_exchange_model"] = "knudsen"
        flags["neutral_two_zone"] = True
        flags["neutral_momentum"] = True
    sim = LAPDSim1D(params, flags)
    for _ in range(10):
        sim.advance_one_step(dt=2.0e-9)
    y = np.asarray(sim._y, dtype=float)
    print(
        f"{tag}: y_sha256={hashlib.sha256(y.tobytes()).hexdigest()} "
        f"n_terms={len(sim.rhs_terms())} "
        f"I_loop={float(sim._circuit_I_loop).hex()} "
        f"t={float(sim._time).hex()}"
    )


for _deposition in ("beer_lambert", "csda"):
    for _two_zone in (False, True):
        run(
            f"{_deposition}/{'two_zone' if _two_zone else 'one_zone'}",
            _deposition,
            _two_zone,
        )
