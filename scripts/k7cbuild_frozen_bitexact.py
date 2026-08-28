"""K7c spot check: every PREVIOUSLY-RUNNABLE tail arm is bit-identical.

K7c widens the EII table-edge refusal into an edge-inclusive guard with a ULP
tolerance, so a tail energy sitting within ~1e-12 relative of the table top
clamps to the last node instead of refusing. The clean property to demonstrate
is the K7b one: the change may only touch configurations the previous code
REFUSED, so anything that already ran must reproduce its trajectory to the last
bit -- including an arm that evaluates the widened guard on a LIVE, per-frame
E_tail at every cathode solve.

Three arms are run:

- ``tw`` -- ``heating_anomalous_transport = "tail_walk"`` at the shipped 75 eV
  rung, energy-only, legacy-pinned;
- ``twion`` -- the same plus ``heating_anomalous_tail_ionization = "on"``, so
  the fixed-rung construction-time copy of the guard is exercised;
- ``twion_phic`` -- the SHIPPED K7 closure (phi_c keying at the default
  bracket arm f = 0.25, reflecting cathode face) with the ionizing channel on.
  This is the arm that matters here: its ``E_tail = 0.25*phi_c(t)`` is rebuilt
  every solve, so the edited guard is re-evaluated on a live value hundreds of
  times per run and must decide identically every time.

A checkout that predates K7 does not have the two legacy keys; ``resolve_config``
refuses unknown keys, so they are added only when this checkout HAS them
(``config_manifest``). The ``twion_phic`` arm needs them and is skipped on such
a checkout.

Otherwise this mirrors ``k7bbuild_frozen_bitexact.py`` and
``k6_frozen_bitexact.py``: the frozen K5a shot-1 calibration arm, a bounded
number of steps, SHA-256 over the raw little-endian float64 bytes of every
piece of state the arm carries.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/k7cbuild_frozen_bitexact.py --steps 400 --arm twion_phic
"""

import argparse
import hashlib

import numpy as np

from run_mechanism_ladder import ES_OPERATING

from cablp.solvers._sim1d import LAPDSim1D, config_manifest, default_config


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

# The frozen arm's argv, transcribed from es1_k5a_shot1_nx240.cmd, exactly as
# k6_frozen_bitexact.py carries it.
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

ARMS = {
    "tw": {"heating_anomalous_transport": "tail_walk"},
    "twion": {
        "heating_anomalous_transport": "tail_walk",
        "heating_anomalous_tail_ionization": "on",
    },
    # The shipped K7 closure: keying="phi_c" and cathode_boundary="reflect"
    # are the defaults, so this arm names only the walk, the bracket arm and
    # the ionizing channel -- and takes NO legacy pins.
    "twion_phic": {
        "heating_anomalous_transport": "tail_walk",
        "heating_anomalous_tail_phi_c_fraction": 0.25,
        "heating_anomalous_tail_ionization": "on",
    },
}

# Arms that carry the pre-K7 closure explicitly.
LEGACY_ARMS = ("tw", "twion")

# Named only where the checkout knows them. On a pre-K7 checkout these keys do
# not exist and the legacy arms above ARE the shipped closure.
LEGACY_PINS = {
    "heating_anomalous_tail_energy_keying": "fixed",
    "heating_anomalous_tail_cathode_boundary": "escape",
}


def build(arm):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    flags["neutral_two_zone"] = True
    params["neutral_exchange_model"] = "knudsen"
    op = ES_OPERATING[1]
    params["V_bank"] = op["V_bank"]
    params["cathode_Ts_base_K"] = op["Ts_standby_K"]
    params.update(EXTRA)
    known = set(config_manifest()["parameters"])
    pinned = {}
    if arm in LEGACY_ARMS:
        pinned = {k: v for k, v in LEGACY_PINS.items() if k in known}
    elif not set(ARMS[arm]) <= known:
        raise SystemExit(
            f"arm {arm!r} needs K7 keys this checkout does not have: "
            f"{sorted(set(ARMS[arm]) - known)}"
        )
    params.update(ARMS[arm])
    params.update(pinned)
    return LAPDSim1D(params, flags), pinned


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
    p.add_argument("--arm", choices=sorted(ARMS), default="tw")
    args = p.parse_args(argv)

    sim, pinned = build(args.arm)
    print(
        f"arm={args.arm} cells={sim.geometry.cells} dvm={sim._dvm is not None} "
        f"anom_transport="
        f"{sim._input_dict.get('heating_anomalous_transport', 'local')!r} "
        f"tail_ionization="
        f"{sim._input_dict.get('heating_anomalous_tail_ionization', 'off')!r} "
        f"E_tail="
        f"{sim._input_dict.get('heating_anomalous_tail_energy_eV', 75.0)!r} "
        f"keying="
        f"{sim._input_dict.get('heating_anomalous_tail_energy_keying', 'n/a')!r} "
        f"f_phi_c="
        f"{sim._input_dict.get('heating_anomalous_tail_phi_c_fraction', 'n/a')!r}"
        f"\nlegacy pins applied: "
        f"{pinned if pinned else '(none -- shipped-closure arm)'}"
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
    print(f"k7c frozen-arm digests printed (arm={args.arm})")


if __name__ == "__main__":
    main()
