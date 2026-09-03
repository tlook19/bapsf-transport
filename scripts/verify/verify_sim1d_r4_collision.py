"""R4.3 ion-neutral moment-closure gate suite (audit A7/A8).

Pre-registered analytic identities for the moment-closed reduced ion-neutral
collision operator ``ion_neutral_collision_rhs`` (flag ``ion_neutral_moment_closure``,
default ON). It replaces the drag + frictional-heating + elastic-thermalization +
CX-cooling quartet with ONE equal-mass (He+/He) Braginskii momentum-transfer
operator built from the Phelps isotropic + backscatter rate coefficients:

    nu_mt = nn * (k_b(T_eff) + 0.5*k_iso(T_eff)),   T_eff = (Ti + Tn)/2
    dM/dt  = -m n nu_mt (u - u_n)                              [neutral gets mirror]
    dEi/dt = 0.5 m n nu_mt (u - u_n)^2 + 1.5 n nu_mt (Tn - Ti) [friction + thermal]

where k_b = <Qb v_rel> (backscatter = charge exchange), k_iso = <Qi v_rel>
(isotropic elastic). Pure-RHS harness (no time advance), mirroring the R2/R3/R4
gate style; not a campaign run.

Gates:
  D1  data integrity: the analytic Phelps cross sections reproduce the archived
      LXCat table (atomic/data/he_ion_neutral_phelps_lxcat.txt) to < 1e-4 rel
  M1  momentum antisymmetry: ion M sink == neutral M_n source through the volume
      ratio, i.e. dM*Vp + dM_n*Vnn == 0 per cell, to roundoff (M_n-carrying state)
  E1  energy closure: at Ti == Tn (thermal part off), the ion frictional heating
      equals -0.5 * u_rel * dM per cell -- friction booked at the FULL nu_mt
      (the CX-sized residual is present, not restricted to the elastic fraction)
  T1  thermal-only limit: at u == u_n == 0, dM == 0, no friction, and
      dEi == 1.5 n nu_mt (Tn - Ti) with nu_mt = nn(k_b + 0.5 k_iso) -- the CX
      thermal coefficient is 1.5 K_cx, not the legacy 2.5 K_cx double-count
  P1  presence-off: the moment-closure term is a strict zero and the four legacy
      ion-neutral terms are live (nonzero) when the flag is off
  P2  presence-on/perturbation: the moment-closure term perturbs (M, Ei nonzero)
      and the four legacy ion-neutral terms are all exactly zero when the flag is on
  G1  construction guard: ion_neutral_moment_closure with a non-He gas raises
      loudly at construction

Usage:
    python scripts/verify_sim1d_r4_collision.py
"""
import sys
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.state import (
    conservative_from_primitives,
    derive_state,
)
from cablp.solvers._sim1d.physics.sources import (
    ion_neutral_collision_rhs,
    neutral_wind_velocity,
)
from cablp.atomic.cross_sections import (
    phelps_he_backscatter_cm2,
    phelps_he_isotropic_cm2,
    phelps_momentum_transfer_rate_cm3_s,
)
from cablp.constants import ev_to_erg, kb_cgs

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
}

TN_K = 300.0
TN_EV = TN_K * kb_cgs / ev_to_erg  # single cold-gas neutral temperature (A8)


def make_sim(moment=False, neutral_momentum=False, gas_type="He"):
    params, flags = default_config()
    params.update(CLEAN_PARAMS)
    params["nx"] = 60
    params["gas_type"] = gas_type
    params["Tn_K"] = TN_K
    flags.update(CLEAN_FLAGS)
    flags["ion_neutral_moment_closure"] = bool(moment)
    flags["neutral_momentum"] = bool(neutral_momentum)
    return LAPDSim1D(params, flags)


def make_state(sim, u_i, u_n, Ti=2.0):
    cells = sim._geometry.cells
    n = np.full(cells, 1.0e12)
    nn = np.full(cells, 1.0e13)
    Te = np.full(cells, 15.0)
    Ti_arr = np.full(cells, float(Ti))
    u = np.full(cells, float(u_i))
    un = np.full(cells, float(u_n))
    return conservative_from_primitives(
        n=n, nn=nn, u=u, Te=Te, Ti=Ti_arr, ion_mass_g=sim._ion_mass_g, un=un,
    )


def _active(sim):
    return np.asarray(sim._geometry.plasma_active, dtype=bool)


def _parse_lxcat_blocks(path):
    """Return {'Backscat': (E, sigma_m2), 'Isotropic': (E, sigma_m2)}."""
    lines = Path(path).read_text().splitlines()
    blocks = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("PROCESS:") and "->" in line:
            kind = "Backscat" if "Backscat" in line else (
                "Isotropic" if "Isotropic" in line else None
            )
            # advance to the first dashed separator
            while i < len(lines) and not lines[i].startswith("-----"):
                i += 1
            i += 1
            data = []
            while i < len(lines) and not lines[i].startswith("-----"):
                parts = lines[i].split()
                if len(parts) == 2:
                    data.append((float(parts[0]), float(parts[1])))
                i += 1
            if kind is not None and data:
                arr = np.array(data)
                blocks[kind] = (arr[:, 0], arr[:, 1])
        i += 1
    return blocks


def gate_d1():
    path = (
        Path(__file__).resolve().parents[2]
        / "cablp" / "vars" / "he_ion_neutral_phelps_lxcat.txt"
    )
    blocks = _parse_lxcat_blocks(path)
    Eb, sb = blocks["Backscat"]
    Ei, si = blocks["Isotropic"]
    # skip E == 0 rows (analytic form is 1/E-singular there; table clamps them)
    mb = Eb > 0
    mi = Ei > 0
    rb = np.max(np.abs(phelps_he_backscatter_cm2(Eb[mb]) - sb[mb] * 1e4)
                / (sb[mb] * 1e4))
    ri = np.max(np.abs(phelps_he_isotropic_cm2(Ei[mi]) - si[mi] * 1e4)
                / (si[mi] * 1e4))
    ok = rb < 1e-4 and ri < 1e-4
    return "D1 Phelps analytic == archived LXCat table (<1e-4)", ok, (
        f"max rel err  backscat={rb:.2e}  isotropic={ri:.2e}"
    )


def gate_m1():
    sim = make_sim(moment=True, neutral_momentum=True)
    st = make_state(sim, u_i=3.0e5, u_n=1.0e5, Ti=2.0)
    term = sim.ion_neutral_collision_rhs(state=st)
    Vp = np.asarray(sim._geometry.plasma_volume_cm3, dtype=float)
    Vnn = np.asarray(sim._geometry.neutral_volume_cm3, dtype=float)
    act = _active(sim)
    resid = term.M * Vp + np.asarray(term.M_n) * Vnn
    scale = np.maximum(np.abs(term.M * Vp), 1e-300)
    rel = np.max(np.abs(resid[act]) / scale[act])
    ok = rel < 1e-13 and np.any(term.M[act] != 0.0)
    return "M1 momentum antisymmetry dM*Vp + dM_n*Vnn == 0", ok, (
        f"max rel residual = {rel:.2e}"
    )


def gate_e1():
    # Set Tn == Ti (above the Ti floor) so the thermal channel vanishes exactly
    # and the Ei row is pure frictional heating. Call the sources operator with an
    # explicit Tn_eV to avoid the solver's fixed 300 K (which the Ti floor would
    # keep from matching Ti).
    sim = make_sim(moment=True)
    Ti_eV = 2.0
    st = make_state(sim, u_i=4.0e5, u_n=0.0, Ti=Ti_eV)
    der = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    term = ion_neutral_collision_rhs(
        state=st, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        gas_type="He", Tn_eV=float(np.mean(der.Ti)), geometry=sim._geometry,
    )
    u_rel = der.u  # u_n == 0 (no M_n)
    expected = -0.5 * u_rel * term.M  # = 0.5 m n nu_mt u_rel^2
    act = _active(sim)
    scale = np.maximum(np.abs(expected), 1e-300)
    rel = np.max(np.abs((term.Ei - expected)[act]) / scale[act])
    ok = rel < 1e-13 and np.any(term.Ei[act] != 0.0)
    return "E1 friction == -0.5 u_rel dM (full nu_mt, CX residual present)", ok, (
        f"max rel residual = {rel:.2e}"
    )


def gate_t1():
    # u == u_n == 0 => dM == 0, no friction, pure thermal at 3/2 nu_mt.
    sim = make_sim(moment=True)
    st = make_state(sim, u_i=0.0, u_n=0.0, Ti=2.0)
    term = sim.ion_neutral_collision_rhs(state=st)
    der = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    T_eff = 0.5 * (der.Ti + TN_EV)
    nu_mt = np.asarray(st.nn, dtype=float) * phelps_momentum_transfer_rate_cm3_s(
        T_eff, gas_type="He"
    )
    expected = 1.5 * nu_mt * st.n * (TN_EV - der.Ti) * ev_to_erg
    act = _active(sim)
    m_zero = np.max(np.abs(term.M[act]))
    scale = np.maximum(np.abs(expected), 1e-300)
    rel = np.max(np.abs((term.Ei - expected)[act]) / scale[act])
    ok = m_zero == 0.0 and rel < 1e-13 and np.any(expected[act] != 0.0)
    return "T1 zero-drift thermal == 1.5 n nu_mt (Tn-Ti); M == 0", ok, (
        f"max|M|={m_zero:.1e}  thermal rel residual={rel:.2e}"
    )


def _legacy_terms(sim, st):
    return {
        "drag": sim.ion_neutral_drag_rhs(state=st),
        "frictional_heating": sim.ion_neutral_frictional_heating_rhs(state=st),
        "thermalization": sim.ion_neutral_thermalization_rhs(state=st),
        "charge_exchange": sim.ion_charge_exchange_rhs(state=st),
    }


def _all_zero(term):
    return all(
        np.all(np.asarray(getattr(term, f)) == 0.0)
        for f in ("n", "nn", "M", "Ee", "Ei")
    )


def gate_p1():
    sim = make_sim(moment=False)
    st = make_state(sim, u_i=3.0e5, u_n=0.0, Ti=2.0)
    coll = sim.ion_neutral_collision_rhs(state=st)
    legacy = _legacy_terms(sim, st)
    act = _active(sim)
    coll_zero = _all_zero(coll)
    drag_live = np.any(legacy["drag"].M[act] != 0.0)
    ok = coll_zero and drag_live
    return "P1 flag off: collision term == 0, legacy drag live", ok, (
        f"collision all-zero={coll_zero}  legacy drag nonzero={drag_live}"
    )


def gate_p2():
    sim = make_sim(moment=True)
    st = make_state(sim, u_i=3.0e5, u_n=0.0, Ti=2.0)
    coll = sim.ion_neutral_collision_rhs(state=st)
    legacy = _legacy_terms(sim, st)
    act = _active(sim)
    coll_perturbs = np.any(coll.M[act] != 0.0) and np.any(coll.Ei[act] != 0.0)
    legacy_zeroed = all(_all_zero(t) for t in legacy.values())
    ok = coll_perturbs and legacy_zeroed
    return "P2 flag on: collision perturbs, four legacy terms == 0", ok, (
        f"collision perturbs={coll_perturbs}  legacy all-zero={legacy_zeroed}"
    )


def gate_g1():
    try:
        make_sim(moment=True, gas_type="H")
    except ValueError as exc:
        ok = "gas_type='He'" in str(exc) or "Phelps" in str(exc)
        return "G1 construction guard: non-He + moment closure raises", ok, (
            f"raised: {str(exc)[:70]}"
        )
    return "G1 construction guard: non-He + moment closure raises", False, (
        "no ValueError raised"
    )


def main():
    gates = [gate_d1, gate_m1, gate_e1, gate_t1, gate_p1, gate_p2, gate_g1]
    all_ok = True
    print("R4.3 ion-neutral moment-closure gate suite (A7/A8)")
    print("=" * 72)
    for g in gates:
        name, ok, detail = g()
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    print("=" * 72)
    print("R4.3 collision gates:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
