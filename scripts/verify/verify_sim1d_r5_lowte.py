"""Low-Te atomic-package consistency gate suite.

`adas_low_te_extension` extends acd (recombination rate) and prb1
(recombination radiated power) below the 0.2 eV ADF11 edge via the in-repo
radiative + 3-body shape.

  C1  package consistency: below the edge, acd and prb1 share the SAME extension
      factor (ext/clamp identical); above the edge both are inert.
  P1  scope/no-op: the electron cooling requests no prb1, so the extension
      does not touch cooling; the flag defaults off.

Usage:  python scripts/verify/verify_sim1d_r5_lowte.py
"""
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.state import conservative_from_primitives
from cablp.atomic.adas import he_rates

CLEAN_PARAMS = {
    "ne0": 1e12, "nn0": 1e13, "Te0": 15.0, "Ti0": 2.0, "u0": 0.0,
    "gas_puff_enabled": False, "pump_enabled": False,
    "phase_transition_mode": "scheduled",
    "tau_neutral_prebreakdown": 0.0, "tau_prebreakdown": 0.0,
    "tau_breakdown": 0.0, "tau_discharge": 1.0, "tau_afterglow": 0.0,
}
CLEAN_FLAGS = {"Plasma": True, "cathode_coupling": False, "debug_checks": False}


def _sim(low_te=False):
    params, flags = default_config()
    params.update(CLEAN_PARAMS)
    params["nx"] = 60
    params["adas_low_te_extension"] = low_te
    flags.update(CLEAN_FLAGS)
    return LAPDSim1D(params, flags)


def _state(sim, Te_val):
    cells = sim._geometry.cells
    return conservative_from_primitives(
        n=np.full(cells, 5.0e12), nn=np.full(cells, 1.0e13),
        u=np.zeros(cells), Te=np.full(cells, Te_val),
        Ti=np.full(cells, 2.0), ion_mass_g=sim._ion_mass_g,
    )


def gate_c1():
    # sub-edge (0.1-0.2 eV, above the 0.1 Te floor) and above-edge samples
    ne = np.full(3, 1e12)
    Te_below = np.array([0.12, 0.15, 0.18])
    fac = {}
    for q in ("acd", "prb1"):
        c = he_rates(ne, Te_below, [q], low_te_extension=False)[q]
        e = he_rates(ne, Te_below, [q], low_te_extension=True)[q]
        fac[q] = e / c
    same = np.max(np.abs(fac["acd"] - fac["prb1"]) / np.abs(fac["acd"]))
    Te_above = np.array([1.0, 5.0])
    inert = all(
        np.allclose(
            he_rates(np.full(2, 1e12), Te_above, [q], low_te_extension=False)[q],
            he_rates(np.full(2, 1e12), Te_above, [q], low_te_extension=True)[q],
        )
        for q in ("acd", "prb1")
    )
    ok = same < 1e-12 and inert and np.max(fac["acd"]) > 1.0
    return "C1 package consistency: acd/prb1 share the sub-edge extension", ok, (
        f"max|acd_fac - prb1_fac|/acd_fac = {same:.1e}; above-edge inert={inert}"
    )


def gate_p1():
    default_off = not bool(default_config()[0].get("adas_low_te_extension", False))
    # prb1 is never requested by the cooling, so the extension cannot touch
    # it -> extension on == off.
    on = _sim(low_te=True)
    off = _sim(low_te=False)
    st = _state(on, 0.15)
    e_on = on.electron_cooling_rhs_terms(state=st)["electron_ion_cooling"].Ee
    e_off = off.electron_cooling_rhs_terms(state=st)["electron_ion_cooling"].Ee
    ok = default_off and np.array_equal(e_on, e_off)
    return "P1 scope: no cooling effect; default off", ok, (
        f"default_off={default_off}  cooling identical={np.array_equal(e_on, e_off)}"
    )


def main():
    gates = [gate_c1, gate_p1]
    all_ok = True
    print("R5.3 low-Te atomic-package consistency gate suite (A18)")
    print("=" * 70)
    for g in gates:
        name, ok, detail = g()
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    print("=" * 70)
    print("R5.3 low-Te gates:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
