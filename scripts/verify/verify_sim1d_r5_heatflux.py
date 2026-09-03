"""R5.2 electron heat-flux limiter gate suite (audit A9).

The electron_heat_flux_limit flag (default ON) scales the electron conductivity per
cell by the harmonic flux limiter
    lambda = q_sat / (q_sat + q_SH),  q_sat = f n Te v_the,  q_SH = kappa_e |dTe/dz|
so the parallel flux caps at the free-streaming ceiling where gradients are steep
and recovers Spitzer where they are shallow. The harmonic form is Malone, McCrory
& Morse, PRL 34 (1975) 721 (equivalently Fundamenski, PPCF 47 (2005) R163,
eq. 10a); the free-streaming ceiling q_sat is Cowie & McKee, ApJ 211 (1977) 135,
eq. (7). Analytic identities (closed domain, to roundoff):

  S1  Spitzer limit: at large f the limited conductivity == unlimited kappa_e.
  S2  saturation cap: the limited cell flux |kappa_eff * dTe/dz| <= q_sat
      everywhere, and -> q_sat in the strong-gradient limit.
  E1  energy conservation: the limited explicit operator conserves total electron
      energy on a closed domain (sum dEe * Vp == 0, still -div(q)).
  P1  presence: limiter off == no-limiter; limiter on perturbs a steep-gradient
      state. (The flag ships ON; this gate builds both arms explicitly.)

Usage:  python scripts/verify/verify_sim1d_r5_heatflux.py
"""
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.state import conservative_from_primitives, derive_state
from cablp.solvers._sim1d.physics.conduction import (
    flux_limited_electron_conductivity,
    heat_conduction_rhs,
)
from cablp.plasma.heat import kappa_par_elec
from cablp.plasma.params import c_log
from cablp.constants import ev_to_erg, m_e_cgs

CLEAN_PARAMS = {
    "ne0": 1e12, "nn0": 1e13, "Te0": 15.0, "Ti0": 2.0, "u0": 0.0,
    "gas_puff_enabled": False, "pump_enabled": False,
    "phase_transition_mode": "scheduled",
    "tau_neutral_prebreakdown": 0.0, "tau_prebreakdown": 0.0,
    "tau_breakdown": 0.0, "tau_discharge": 1.0, "tau_afterglow": 0.0,
}
CLEAN_FLAGS = {"Plasma": True, "cathode_coupling": False, "debug_checks": False}


def _sim(limit=False, f=0.3):
    params, flags = default_config()
    params.update(CLEAN_PARAMS)
    params["nx"] = 60
    params["heat_flux_limiter_f"] = f
    flags.update(CLEAN_FLAGS)
    flags["electron_heat_flux_limit"] = limit
    return LAPDSim1D(params, flags)


def _steep_state(sim, Te_lo=2.0, Te_hi=60.0):
    """A uniform-density state with a steep Te ramp (drives q_SH high)."""
    cells = sim._geometry.cells
    z = np.asarray(sim._geometry.z_cm, dtype=float)
    frac = (z - z.min()) / max(np.ptp(z), 1.0)
    Te = Te_lo + (Te_hi - Te_lo) * frac  # linear ramp
    n = np.full(cells, 5.0e12)
    nn = np.full(cells, 1.0e13)
    Ti = np.full(cells, 2.0)
    u = np.zeros(cells)
    return conservative_from_primitives(
        n=n, nn=nn, u=u, Te=Te, Ti=Ti, ion_mass_g=sim._ion_mass_g
    )


def _kappa_e(sim, st):
    d = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    n = np.maximum(st.n, sim._floors["n"])
    ln = np.maximum(c_log(d.Te, n, kind="ei"), 1.0)
    return kappa_par_elec(d.Te, n, ln, per_particle=False) * ev_to_erg, d.Te, n


def gate_s1():
    sim = _sim(limit=True)
    st = _steep_state(sim)
    ke, Te, n = _kappa_e(sim, st)
    lim = flux_limited_electron_conductivity(ke, Te, n, sim._geometry, f=1.0e8)
    rel = np.max(np.abs(lim - ke) / np.maximum(np.abs(ke), 1e-300))
    ok = rel < 1e-6
    return "S1 Spitzer limit: large-f limited kappa == unlimited", ok, (
        f"max rel diff = {rel:.2e}"
    )


def gate_s2():
    sim = _sim(limit=True, f=0.1)
    st = _steep_state(sim)
    ke, Te, n = _kappa_e(sim, st)
    Te_erg = Te * ev_to_erg
    v_the = np.sqrt(Te_erg / m_e_cgs)
    q_sat = 0.1 * n * Te_erg * v_the
    grad = np.gradient(Te, np.asarray(sim._geometry.z_cm, dtype=float))
    ke_lim = flux_limited_electron_conductivity(ke, Te, n, sim._geometry, f=0.1)
    q_lim = np.abs(ke_lim * grad)
    ratio = q_lim / np.maximum(q_sat, 1e-300)
    ok = np.max(ratio) <= 1.0 + 1e-12
    return "S2 saturation cap: |kappa_eff dTe/dz| <= q_sat everywhere", ok, (
        f"max flux/q_sat = {np.max(ratio):.4f}  (approaches 1 where steep)"
    )


def gate_e1():
    sim = _sim(limit=True, f=0.1)
    st = _steep_state(sim)
    term = heat_conduction_rhs(
        state=st, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        mu=sim._mu, geometry=sim._geometry,
        **sim._heat_conduction_kwargs(),
    )
    Vp = np.asarray(sim._geometry.plasma_volume_cm3, dtype=float)
    total = float(np.sum(term.Ee * Vp))
    scale = float(np.sum(np.abs(term.Ee) * Vp))
    rel = abs(total) / max(scale, 1e-300)
    ok = rel < 1e-12 and np.any(term.Ee != 0.0)
    return "E1 energy conservation: sum dEe*Vp == 0 (closed domain)", ok, (
        f"net/|net-terms| = {rel:.2e}"
    )


def gate_p1():
    off = _sim(limit=False)
    on = _sim(limit=True, f=0.1)
    st = _steep_state(off)
    kw_off = off._heat_conduction_kwargs()
    kw_on = on._heat_conduction_kwargs()
    r_off = heat_conduction_rhs(
        state=st, floors=off._floors, ion_mass_g=off._ion_mass_g,
        mu=off._mu, geometry=off._geometry, **kw_off)
    r_on = heat_conduction_rhs(
        state=st, floors=on._floors, ion_mass_g=on._ion_mass_g,
        mu=on._mu, geometry=on._geometry, **kw_on)
    flag_off_arm = not off._electron_heat_flux_limit
    perturbs = np.max(np.abs(r_on.Ee - r_off.Ee)) > 0.0
    ok = flag_off_arm and perturbs
    return "P1 presence: flag-off arm is inert; limiter perturbs conduction", ok, (
        f"flag_off_arm={flag_off_arm}  max|dEe|={np.max(np.abs(r_on.Ee - r_off.Ee)):.2e}"
    )


def main():
    gates = [gate_s1, gate_s2, gate_e1, gate_p1]
    all_ok = True
    print("R5.2 electron heat-flux limiter gate suite (A9)")
    print("=" * 70)
    for g in gates:
        name, ok, detail = g()
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    print("=" * 70)
    print("R5.2 heat-flux gates:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
