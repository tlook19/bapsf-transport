"""Microbenchmark the compiled Tier A cathode unit against the pure path.

Run it twice, once per arm, and diff the two reports::

    python scripts/bench_cathode_kernels.py
    CABLP_COMPILED_KERNELS=1 python scripts/bench_cathode_kernels.py

Two levels are timed:

``solve_idriven``
    the cathode sheath solve itself -- the bracket ladder plus brentq, whose
    ~50-100 residual evaluations per call were Python round-trips through
    ``_emission_state`` and its per-annulus loop. This is where the compiled
    unit is supposed to pay.
``solve_cathode_boundary``
    the enclosing solver call, which also runs the CSDA beam deposition. The
    ratio here is the one that matters for a run's wall clock, and it is
    necessarily smaller: only part of it is the cathode solve.

Each timing is the MINIMUM over repeats of a batch, the usual defence against
the machine's other work; the reported number is per call.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from cablp.funcs import _kernels
from cablp.funcs._cathode_solver import DeviceConfig, PlasmaState
from cablp.funcs._cathode_solver_idriven import solve_idriven
from cablp.solvers._sim1d import LAPDSim1D, default_config


def _annuli(Ts_peak, n=10, R_cath=18.5, sigma=0.6):
    edges = np.linspace(0.0, R_cath, n + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    Ts = Ts_peak * np.exp(-0.5 * (r_mid / (sigma * R_cath)) ** 2)
    frac = np.clip(1.0 - (r_mid / R_cath) ** 3, 0.0, 1.0)
    return (
        tuple(float(t) for t in Ts),
        tuple(float(a) for a in area),
        tuple(float(f) for f in frac),
    )


def _config(Ts_peak=1910.0):
    Ts_k, area_k, frac_k = _annuli(Ts_peak)
    return DeviceConfig(
        A_c=math.pi * 18.5**2,
        mu=4.0,
        V_bank=100.0,
        T_s=Ts_peak,
        phi_wf=2.6,
        C_R=14.25,
        R_cath=18.5,
        L_cath=50.0,
        emission_Ts_K=Ts_k,
        emission_area_cm2=area_k,
        emission_plasma_frac=frac_k,
    )


#: (label, T_e [eV], n_e [cm^-3], I_tot [A]) -- operating points spanning the
#: three regimes the ladder can land in, chosen against this benchmark's own
#: 1910 K / C_R 14.25 device (I_eth = 304 A).
STATES = (
    ("virtual_cathode", 0.5, 1.0e10, 10.0),
    ("virtual_cathode2", 8.0, 5.0e11, 100.0),
    ("classical", 3.0, 2.0e12, 500.0),
    ("capability_ltd", 3.0, 2.0e12, 1500.0),
)


def _min_per_call(fn, batch, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(batch):
            fn()
        best = min(best, (time.perf_counter() - t0) / batch)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--boundary-batch", type=int, default=200)
    args = parser.parse_args()

    print(f"kernels: {_kernels.PROVENANCE}")
    cfg = _config()
    for label, T_e, n_e, I_tot in STATES:
        plasma = PlasmaState(T_e=T_e, n_e=n_e, n_n=1.0e13, sigma_b=1.0e-16)
        res = solve_idriven(cfg, plasma, I_tot, schottky=True, bridge=False)
        per_call = _min_per_call(
            lambda: solve_idriven(
                cfg, plasma, I_tot, schottky=True, bridge=False
            ),
            args.batch,
            args.repeats,
        )
        print(
            f"solve_idriven {label:>14s}: {per_call * 1e6:9.3f} us/call "
            f"regime={res.regime} phi_c={res.phi_c!r}"
        )

    # Same construction the smoke suite's csda block uses, so the boundary
    # solve actually has an active cathode ray to deposit.
    params, flags = default_config()
    params = dict(params)
    flags = dict(flags)
    params["phase_transition_mode"] = "scheduled"
    params["gas_puff_mode"] = "decay_after_breakdown"
    params["gas_puff_profile"] = "cell"
    params["beam_deposition_model"] = "csda"
    flags["neutral_prebreakdown"] = False
    flags["cathode_coupling"] = True
    sim = LAPDSim1D(params, flags)
    sim._circuit_I_loop = 3000.0
    probe = sim.solve_cathode_boundary()
    assert probe.beam_deposition is not None and probe.beam_deposition[0]
    per_call = _min_per_call(
        sim.solve_cathode_boundary, args.boundary_batch, 3
    )
    print(f"solve_cathode_boundary       : {per_call * 1e6:9.3f} us/call")


if __name__ == "__main__":
    raise SystemExit(main())
