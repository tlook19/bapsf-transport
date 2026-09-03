"""R4.2 ionization-birth-moment gate suite (audit A14).

Pre-registered gates for the unified ionization birth energy moments. The audit
finding A14: bulk ADAS ionization books electron birth energy
``Ee += 3/2 Te_birth S_ion`` with ``Te_birth_ionization="local"`` (= Te), adding
+43.1 kW that cancels 92% of the separate ionization-potential cost. Creating an
electron does not create ``3 Te/2`` of kinetic energy; the beam already books the
defensible ``Ee = 0`` convention. The repair (the selector
``ionization_birth_energy_model="conservative"``, which is now the shipped
default; it was introduced default-off as ``"legacy"``) reconciles bulk to that
convention -- the new electron is born cold, so ``Te`` falls by dilution -- and
books the ion mass-loading relative-drift **mixing energy**
``1/2 m (u_i - u_n)^2 S_ion`` to ``Ei`` explicitly, so ion total energy closes to
the energy carried in by the consumed neutral rather than losing the drift energy
through the bulk kinetic derivative.

Closed-domain ion-energy identity (per cell). With the birth term supplying
``dn = S_ion``, ``dM = m u_n S_ion`` and the reconstructed bulk kinetic change
``dK = u_i dM - 1/2 m u_i^2 dn``, the ion total-energy change closes to the
consumed neutral's energy::

    dEi + dK == 3/2 Ti_birth S_ion + 1/2 m u_n^2 S_ion

exactly (to machine precision), while the electron birth contributes zero.

This is a pure-RHS verification harness on ``reaction_rhs_terms`` (no time
advance), mirroring the R2/R3/R4.1 gate style. It is not a campaign run.

Gates:
  A1  off-path: "legacy" == the unset default, byte-for-byte (ionization_birth)
  A2  conservative electron birth Ee is exactly zero (no 3Te/2 creation)
  A3  particle & momentum rows unchanged (only the energy booking changes)
  A4  ion total-energy closure to machine precision (the mixing-energy identity),
      with a nonzero neutral wind (u_i != u_n) and at rest (u_n = 0)

Usage:
    python scripts/verify/verify_sim1d_r4_ionization_birth.py
"""
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.state import (
    conservative_from_primitives,
    derive_state,
)
from cablp.solvers._sim1d.physics.sources import neutral_wind_velocity
from cablp.constants import ev_to_erg

CLEAN_PARAMS = {
    "ne0": 1e12, "nn0": 1e13, "Te0": 15.0, "Ti0": 2.0, "u0": 0.0,
    "gas_puff_enabled": False, "pump_enabled": False,
    "atomic_rate_model": "adas",
    "phase_transition_mode": "scheduled",
    "tau_neutral_prebreakdown": 0.0, "tau_prebreakdown": 0.0,
    "tau_breakdown": 0.0, "tau_discharge": 1.0, "tau_afterglow": 0.0,
    "adaptive_retries_enabled": False, "dt_growth_enabled": False,
    "dt_min": 1e-16, "dt_max": 1.0,
    "max_density_step_fraction": 0.0, "max_neutral_step_fraction": 0.0,
    "max_energy_step_fraction": 0.0,
}
CLEAN_FLAGS = {
    "Plasma": True, "implicit_heat_conduction": True,
    "neutral_prebreakdown": False, "neutral_equilibration": False,
    "launch_plasma_after_equilibration": False,
    "cathode_coupling": False, "debug_checks": False,
    "neutral_momentum": True,  # so u_n can differ from u_i
}


def make_sim(model=None):
    params, flags = default_config()
    params.update(CLEAN_PARAMS)
    params["nx"] = 60
    if model is not None:
        params["ionization_birth_energy_model"] = model
    flags.update(CLEAN_FLAGS)
    return LAPDSim1D(params, flags)


def make_state(sim, u_i, u_n):
    """A uniform active-plasma state with ion drift u_i and neutral wind u_n."""
    cells = sim._geometry.cells
    n = np.full(cells, 1.0e12)
    nn = np.full(cells, 1.0e13)
    Te = np.full(cells, 15.0)
    Ti = np.full(cells, 2.0)
    u = np.full(cells, float(u_i))
    un = np.full(cells, float(u_n))
    return conservative_from_primitives(
        n=n, nn=nn, u=u, Te=Te, Ti=Ti, ion_mass_g=sim._ion_mass_g, un=un,
    )


def _active(sim):
    return np.asarray(sim._geometry.plasma_active, dtype=bool)


def gate_a1():
    default_sim = make_sim(model=None)
    legacy_sim = make_sim(model="legacy")
    st = make_state(default_sim, u_i=3.0e5, u_n=1.0e5)
    d = default_sim.reaction_rhs_terms(state=st)["ionization_birth"]
    l = legacy_sim.reaction_rhs_terms(state=st)["ionization_birth"]
    ok = all(
        np.array_equal(getattr(d, f), getattr(l, f))
        for f in ("n", "nn", "M", "Ee", "Ei")
    )
    ok = ok and np.array_equal(
        np.asarray(d.M_n), np.asarray(l.M_n)
    )
    return "A1 off-path: 'legacy' == unset default byte-identical", ok, (
        f"max|dEe|={np.max(np.abs(d.Ee - l.Ee)):.1e}"
    )


def gate_a2():
    sim = make_sim(model="conservative")
    legacy = make_sim(model="legacy")
    st = make_state(sim, u_i=3.0e5, u_n=1.0e5)
    ion_c = sim.reaction_rhs_terms(state=st)["ionization_birth"]
    ion_l = legacy.reaction_rhs_terms(state=st)["ionization_birth"]
    act = _active(sim)
    ee_zero = np.all(ion_c.Ee == 0.0)
    legacy_nonzero = np.any(ion_l.Ee[act] != 0.0)  # the +43.1 kW-type term
    ok = bool(ee_zero and legacy_nonzero)
    return "A2 conservative electron Ee-birth == 0 (legacy nonzero)", ok, (
        f"max|Ee_c|={np.max(np.abs(ion_c.Ee)):.1e} "
        f"sum Ee_l[act]={float(np.sum(ion_l.Ee[act])):.3e} erg/cm^3/s"
    )


def gate_a3():
    sim = make_sim(model="conservative")
    legacy = make_sim(model="legacy")
    st = make_state(sim, u_i=3.0e5, u_n=1.0e5)
    ion_c = sim.reaction_rhs_terms(state=st)["ionization_birth"]
    ion_l = legacy.reaction_rhs_terms(state=st)["ionization_birth"]
    ok = all(
        np.array_equal(getattr(ion_c, f), getattr(ion_l, f))
        for f in ("n", "nn", "M")
    ) and np.array_equal(np.asarray(ion_c.M_n), np.asarray(ion_l.M_n))
    return "A3 particle & momentum rows unchanged by the energy model", ok, (
        f"max|dM|={np.max(np.abs(ion_c.M - ion_l.M)):.1e}"
    )


def _closure_residual(sim, legacy, u_i, u_n):
    st = make_state(sim, u_i=u_i, u_n=u_n)
    der = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    u = der.u
    un = neutral_wind_velocity(
        st, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry,
    )
    ion_c = sim.reaction_rhs_terms(state=st)["ionization_birth"]
    ion_l = legacy.reaction_rhs_terms(state=st)["ionization_birth"]
    m = sim._ion_mass_g
    S = ion_c.n  # ionization_birth.n == S_ion
    dK = u * ion_c.M - 0.5 * m * u**2 * ion_c.n
    # 3/2 Ti_birth S is exactly the legacy ion internal birth (no mixing).
    lhs = ion_c.Ei + dK
    rhs = ion_l.Ei + 0.5 * m * un**2 * S
    act = _active(sim)
    denom = np.maximum(np.abs(rhs[act]), 1e-300)
    return np.max(np.abs((lhs - rhs)[act]) / denom), float(np.max(np.abs(S[act])))


def gate_a4():
    sim = make_sim(model="conservative")
    legacy = make_sim(model="legacy")
    rel_wind, S_wind = _closure_residual(sim, legacy, u_i=3.0e5, u_n=1.0e5)
    rel_rest, _ = _closure_residual(sim, legacy, u_i=3.0e5, u_n=0.0)
    ok = rel_wind < 1e-12 and rel_rest < 1e-12 and S_wind > 0.0
    return "A4 ion total-energy closure to machine precision", ok, (
        f"rel(u_n=1e5)={rel_wind:.2e} rel(u_n=0)={rel_rest:.2e} "
        f"max S_ion={S_wind:.3e}"
    )


def main():
    gates = [gate_a1, gate_a2, gate_a3, gate_a4]
    all_ok = True
    print("R4.2 ionization-birth-moment gate suite (A14)")
    print("=" * 72)
    for g in gates:
        name, ok, detail = g()
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    print("=" * 72)
    print("R4.2 ionization-birth gates:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
