"""covreview: independent raw-byte bit-exactness probe for the coverage closure.

Runs a config the coder's smoke did NOT use (nx=20, current_driven cathode,
gaussian emission, csda+quasilinear on one arm; default beer_lambert on the
other) for a fixed number of fixed-dt steps and prints SHA-256 over the raw
state-vector bytes plus the circuit current. Run under the BASE checkout's
PYTHONPATH and the BRANCH checkout's PYTHONPATH; the lines must be identical.

Modes:
  off        flag off (runs on base AND branch)
  reduction  coverage_closure on with f_cov0=1, r=0 (branch only; must match
             the off lines bit-for-bit)
"""
import hashlib
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config

mode = sys.argv[1] if len(sys.argv) > 1 else "off"


def run(tag, deposition, coverage_on):
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
    if coverage_on:
        params["coverage_initial_fraction"] = 1.0
        params["coverage_growth_rate_per_s"] = 0.0
        flags["coverage_closure"] = True
    sim = LAPDSim1D(params, flags)
    for _ in range(8):
        sim.advance_one_step(dt=2.0e-9)
    y = np.asarray(sim._y, dtype=float)
    print(
        f"{tag}: y_sha256={hashlib.sha256(y.tobytes()).hexdigest()} "
        f"I_loop={float(sim._circuit_I_loop).hex()} "
        f"t={sim._time.hex() if hasattr(sim._time, 'hex') else float(sim._time).hex()}"
    )


if mode == "off":
    run("off/beer_lambert", "beer_lambert", False)
    run("off/csda", "csda", False)
elif mode == "reduction":
    run("red/csda", "csda", True)
else:
    raise SystemExit(f"unknown mode {mode!r}")
