"""Handshake-fix acceptance probe: booked-vs-removed at the DVM ionization tick.

The post-fix re-run of the diagnostician's ``ghostinflow_dvm_probe.py``
(2026-08-06), on the same K6d-configured short shot, with the cadence as an
argument so one script covers both rungs. ``TransientDVM.update`` is wrapped
and every tick records THREE numbers, not two:

  proxy_booked  = sum(nu_ion * nn_col_before * V_col) * dt   [particles]
  counted_booked= sum(ion_counts)                            [particles]
  removed       = ledger["loss_ionization"]                  [particles]

``proxy_booked`` is the ORIGINAL probe's metric: the plasma's booking
estimated from the tick-START rate held over the whole tick. ``counted_booked``
is what the plasma actually accumulated as ionization over the tick, which is
what the fixed handshake hands the arm and what the arm must debit exactly.

So the two gaps say different things and both belong in the record:

  removed vs counted_booked -- CONSERVATION. Must be machine zero at every
    cadence; a nonzero value is particle creation in the coupled system.
  removed vs proxy_booked -- the tick-start-rate proxy's own error, i.e. how
    far the ionization RATE moved across one tick. First order in the cadence
    by construction and NOT claimed to be removed by the fix.

Writes a scratch h5 with a distinct name; touches no campaign artifact.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cablp.solvers._sim1d.physics import kinetic_dvm as KD

TALLY = {
    "ticks": 0,
    "proxy_booked": 0.0,
    "counted_booked": 0.0,
    "removed": 0.0,
    "debt": 0.0,
    "limited_ticks": 0,
    "inv_delta": 0.0,
    "births": 0.0,
    "losses": 0.0,
    "ledger_resid": 0.0,
    "puff": 0.0,
    "cath": 0.0,
    "coll": 0.0,
    "rec": 0.0,
    "anode": 0.0,
    "pump": 0.0,
    "endout": 0.0,
}
PER_TICK = []

_orig = KD.TransientDVM.update


def patched(self, dt, *, n_i, Ti_eV, u_i, nu_ion, ion_counts=None,
            sources=None, T_s_K=None):
    nn_before = np.asarray(self.column_density(), dtype=float)
    V_col = np.asarray(self.V_col, dtype=float)
    proxy = float(
        np.sum(np.asarray(nu_ion, dtype=float) * nn_before * V_col) * dt
    )
    counted = 0.0 if ion_counts is None else float(np.sum(ion_counts))
    led = _orig(self, dt, n_i=n_i, Ti_eV=Ti_eV, u_i=u_i, nu_ion=nu_ion,
                ion_counts=ion_counts, sources=sources, T_s_K=T_s_K)
    births = sum(v for k, v in led.items() if k.startswith("birth_"))
    losses = sum(v for k, v in led.items()
                 if k.startswith("loss_") and not k.startswith("loss_pump"))
    inv_d = led["inventory_after"] - led["inventory_before"]
    TALLY["ticks"] += 1
    TALLY["proxy_booked"] += proxy
    TALLY["counted_booked"] += counted
    TALLY["removed"] += led["loss_ionization"]
    TALLY["debt"] = led["ion_debt_carried"]
    TALLY["limited_ticks"] += 1 if led["ion_limited_cells"] else 0
    TALLY["inv_delta"] += inv_d
    TALLY["births"] += births
    TALLY["losses"] += losses
    TALLY["ledger_resid"] += (inv_d - (births - losses))
    TALLY["puff"] += led["birth_puff"]
    TALLY["cath"] += led["birth_cathode_face"]
    TALLY["coll"] += led["birth_collector_face"]
    TALLY["rec"] += led["birth_recombination"]
    TALLY["anode"] += led["birth_anode"]
    TALLY["pump"] += led["loss_pump_L"] + led["loss_pump_R"]
    TALLY["endout"] += led["loss_end_out_L"] + led["loss_end_out_R"]
    if TALLY["ticks"] % 25 == 0:
        PER_TICK.append((TALLY["ticks"], proxy, counted,
                         led["loss_ionization"], led["ion_debt_carried"]))
    return led


KD.TransientDVM.update = patched

import run_m6_point  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence", default="1.0e-5")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.cadence.replace(".", "p").replace("-", "m")
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, f"hshakefix_probe_k6d_short_{tag}.h5")
    argv = [
        "--es", "1", "--sgp", "5200", "--nx", "240", "--two-zone",
        "--extra", "T_s=1998.15", "Te_birth_ionization=local",
        "tau_afterglow=0.0005", "tau_discharge=0.003",
        "neutral_model=kinetic_dvm",
        f"neutral_kinetic_dvm_cadence_s={args.cadence}",
        "neutral_kinetic_dvm_nvz=96", "neutral_kinetic_dvm_nvp=32",
        "neutral_kinetic_dvm_exchange=cauchy_chord", "C_R=12.96",
        "neutral_kinetic_dvm_annulus_flights=bounded_chord",
        "heating_anomalous_transport=tail_walk",
        "heating_anomalous_tail_ionization=on",
        "--save-h5", out,
    ]
    run_m6_point.main(argv)

    print("\n===== hshakefix DVM handshake tally (cadence %s s) ====="
          % args.cadence)
    print("ticks                          %d" % TALLY["ticks"])
    print("plasma-booked (counted)        %.6e particles"
          % TALLY["counted_booked"])
    print("plasma-booked (tick-start rate)%.6e particles"
          % TALLY["proxy_booked"])
    print("DVM-removed   ionization       %.6e particles" % TALLY["removed"])
    cgap = TALLY["counted_booked"] - TALLY["removed"]
    pgap = TALLY["proxy_booked"] - TALLY["removed"]
    print("CONSERVATION GAP (counted)     %.6e  = %.3e of booked"
          % (cgap, cgap / max(TALLY["counted_booked"], 1e-300)))
    print("rate-proxy gap (tick-start)    %.6e  = %.4f of booked"
          % (pgap, pgap / max(TALLY["proxy_booked"], 1e-300)))
    print("outstanding ion debt           %.6e particles" % TALLY["debt"])
    print("ticks with a limited cell      %d" % TALLY["limited_ticks"])
    print("puff booked into DVM           %.6e" % TALLY["puff"])
    print("CONSERVATION GAP / puff        %.4e"
          % (cgap / max(TALLY["puff"], 1e-300)))
    print("rate-proxy gap / puff          %.4f"
          % (pgap / max(TALLY["puff"], 1e-300)))
    print("cathode_face returns           %.6e" % TALLY["cath"])
    print("collector_face returns         %.6e" % TALLY["coll"])
    print("recombination returns          %.6e" % TALLY["rec"])
    print("anode returns                  %.6e" % TALLY["anode"])
    print("end-plane outflow              %.6e   pumped(stick) %.6e "
          "(stick frac %.4f)"
          % (TALLY["endout"], TALLY["pump"],
             TALLY["pump"] / max(TALLY["endout"], 1e-300)))
    print("DVM internal ledger residual   %.6e  (rel to inventory delta %.3e)"
          % (TALLY["ledger_resid"],
             TALLY["ledger_resid"] / max(abs(TALLY["inv_delta"]), 1e-300)))
    print("inventory delta over run       %.6e" % TALLY["inv_delta"])
    print("\ntick sample (n, proxy_booked, counted_booked, removed, debt):")
    for row in PER_TICK[::10]:
        print("  %6d  %.4e  %.4e  %.4e  %.4e" % row)


if __name__ == "__main__":
    main()
