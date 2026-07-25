"""R4.1 anode-mesh beam-interception gate suite (SIM1D_MODEL_AUDIT_PLAN R4, A15).

Pre-registered gates for the CSDA anode-mesh interception event. The audit finding
A15: ``funcs/_beam_deposition.deposit_beam`` launches the full emitted flux
``Gamma0 = I_eth_star/e`` through the whole axial ray, so the fluid deposits the
entire emitted beam (~470 kW on the settled artifact) while the circuit books only
the ``(1 - eta*beam_bypass_fraction)`` fraction into the plasma. The ~164 kW
difference is the long-mfp beam the anode mesh intercepts; the fluid must stop
depositing it. The repair adds a single interception event at the anode-face
crossing: the mesh solid fraction ``eta`` of the flux still streaming at that face
is removed (booked to the anode, not the plasma) and only ``(1 - eta)`` transmits
downstream, with the reduced flux carried through all subsequent deposition.

This is a pure-function verification harness on ``deposit_beam`` (no solver state),
mirroring the R2/R3 gate style. It is not a campaign run.

Gates:
  B1  off-path: no interception args == anode_eta=0 == the historical call,
      byte-for-byte (the presence-gate off path is a strict no-op)
  B2  per-ray energy conservation WITH interception, to roundoff:
      Gamma0*E0 == heating + radiated + cost + anode_intercepted
                   + transmitted_flux*transmitted_energy_eV
  B3  flux/particle interception: pre-anode cells are byte-identical to the
      eta=0 run; post-anode heating and ionization events scale by exactly
      (1 - eta); transmitted flux == (1 - eta)*Gamma0 for a surviving ray
  B4  interception booking: anode_intercepted == eta * Gamma0 * E_entry[cross]
      (remove eta at the exact anode face, at the primary's remaining energy)
  B5  gap-absorbed ray: a beam that stops before the anode face intercepts
      nothing and equals the eta=0 run (only the surviving long-mfp beam is
      intercepted -- the consistency with the circuit's eta*bypass structure)

Usage:
    python scripts/verify_sim1d_r4_beam_interception.py
"""
import sys

import numpy as np

from cablp.funcs._beam_deposition import deposit_beam, _ERG_PER_EV


def _ray(nn_val, cells=10):
    """A uniform column ray: launch at cell 0, +z, 5 cm cells."""
    nn = np.full(cells, float(nn_val))
    ne = np.full(cells, 5.0e12)
    Te = np.full(cells, 8.0)
    dz = np.full(cells, 5.0)
    return nn, ne, Te, dz


E0 = 200.0
CROSS = 4  # anode face crossing: interception before entering cell 4
ETA = 0.3


def _energy_in(res, Gamma0):
    return (
        res.plasma_heating_erg_s.sum()
        + res.radiated_erg_s.sum()
        + res.ionization_cost_erg_s.sum()
        + float(res.anode_intercepted_erg_s)
        + res.transmitted_flux * res.transmitted_energy_eV * _ERG_PER_EV
    )


def gate_b1():
    # Survives the gap (low nn), so an eta=0 interception must be a no-op and the
    # default (no interception args) call must be byte-identical.
    nn, ne, Te, dz = _ray(1.0e12)
    Gamma0 = 1.0e19
    base = deposit_beam(E0, Gamma0, nn, ne, Te, 0, 1, dz)
    eta0 = deposit_beam(
        E0, Gamma0, nn, ne, Te, 0, 1, dz,
        anode_cross_index=CROSS, anode_eta=0.0,
    )
    fields = [
        "ionization_events", "excitation_events", "plasma_heating_erg_s",
        "radiated_erg_s", "ionization_cost_erg_s", "E_entry_eV",
    ]
    ok = all(
        np.array_equal(getattr(base, f), getattr(eta0, f)) for f in fields
    )
    ok = ok and base.transmitted_flux == eta0.transmitted_flux
    ok = ok and base.transmitted_energy_eV == eta0.transmitted_energy_eV
    ok = ok and float(eta0.anode_intercepted_erg_s) == 0.0
    return "B1 off-path: eta=0/no-args byte-identical, zero interception", ok, (
        f"anode_intercepted(eta=0)={float(eta0.anode_intercepted_erg_s):.3e}"
    )


def gate_b2():
    nn, ne, Te, dz = _ray(1.0e12)
    Gamma0 = 1.0e19
    res = deposit_beam(
        E0, Gamma0, nn, ne, Te, 0, 1, dz,
        anode_cross_index=CROSS, anode_eta=ETA,
    )
    total_in = Gamma0 * E0 * _ERG_PER_EV
    total_out = _energy_in(res, Gamma0)
    rel = abs(total_out - total_in) / total_in
    ok = rel < 1e-12 and float(res.anode_intercepted_erg_s) > 0.0
    return "B2 per-ray energy conserves with interception", ok, (
        f"in={total_in:.6e} out={total_out:.6e} rel={rel:.2e}"
    )


def gate_b3():
    nn, ne, Te, dz = _ray(1.0e12)
    Gamma0 = 1.0e19
    ref = deposit_beam(
        E0, Gamma0, nn, ne, Te, 0, 1, dz,
        anode_cross_index=CROSS, anode_eta=0.0,
    )
    cut = deposit_beam(
        E0, Gamma0, nn, ne, Te, 0, 1, dz,
        anode_cross_index=CROSS, anode_eta=ETA,
    )
    # Pre-anode cells (0..CROSS-1) byte-identical.
    pre_ok = all(
        np.array_equal(ref.plasma_heating_erg_s[:CROSS],
                       cut.plasma_heating_erg_s[:CROSS])
        and np.array_equal(ref.ionization_events[:CROSS],
                           cut.ionization_events[:CROSS])
        for _ in (0,)
    )
    # Post-anode cells scale by (1 - eta).
    post = slice(CROSS, None)
    heat_ratio = np.divide(
        cut.plasma_heating_erg_s[post], ref.plasma_heating_erg_s[post],
        out=np.full(ref.plasma_heating_erg_s[post].shape, np.nan),
        where=ref.plasma_heating_erg_s[post] != 0.0,
    )
    ion_ratio = np.divide(
        cut.ionization_events[post], ref.ionization_events[post],
        out=np.full(ref.ionization_events[post].shape, np.nan),
        where=ref.ionization_events[post] != 0.0,
    )
    finite = np.isfinite(heat_ratio)
    scale_ok = (
        np.allclose(heat_ratio[finite], 1.0 - ETA, rtol=1e-12, atol=0.0)
        and np.allclose(ion_ratio[np.isfinite(ion_ratio)], 1.0 - ETA,
                        rtol=1e-12, atol=0.0)
    )
    flux_ok = abs(cut.transmitted_flux - (1.0 - ETA) * Gamma0) <= 1e-3 * Gamma0
    ok = pre_ok and scale_ok and flux_ok
    return "B3 pre-anode identical; post-anode x(1-eta); flux x(1-eta)", ok, (
        f"transmitted={cut.transmitted_flux:.4e} "
        f"expected={(1.0-ETA)*Gamma0:.4e}"
    )


def gate_b4():
    nn, ne, Te, dz = _ray(1.0e12)
    Gamma0 = 1.0e19
    ref = deposit_beam(
        E0, Gamma0, nn, ne, Te, 0, 1, dz,
        anode_cross_index=CROSS, anode_eta=0.0,
    )
    cut = deposit_beam(
        E0, Gamma0, nn, ne, Te, 0, 1, dz,
        anode_cross_index=CROSS, anode_eta=ETA,
    )
    E_at_anode = float(ref.E_entry_eV[CROSS])  # primary energy entering the face
    expected = ETA * Gamma0 * E_at_anode * _ERG_PER_EV
    got = float(cut.anode_intercepted_erg_s)
    rel = abs(got - expected) / expected
    ok = rel < 1e-12 and E_at_anode > 0.0
    return "B4 interception == eta*Gamma0*E_entry[cross] at the anode face", ok, (
        f"E_at_anode={E_at_anode:.4f} eV got={got:.6e} exp={expected:.6e} "
        f"rel={rel:.2e}"
    )


def gate_b5():
    # Dense gas: the beam stops before the anode face, so nothing is intercepted.
    nn, ne, Te, dz = _ray(5.0e16)
    Gamma0 = 1.0e19
    ref = deposit_beam(
        E0, Gamma0, nn, ne, Te, 0, 1, dz,
        anode_cross_index=CROSS, anode_eta=0.0,
    )
    cut = deposit_beam(
        E0, Gamma0, nn, ne, Te, 0, 1, dz,
        anode_cross_index=CROSS, anode_eta=ETA,
    )
    absorbed_before = float(ref.E_entry_eV[CROSS]) == 0.0
    no_intercept = float(cut.anode_intercepted_erg_s) == 0.0
    identical = np.array_equal(
        ref.plasma_heating_erg_s, cut.plasma_heating_erg_s
    )
    ok = absorbed_before and no_intercept and identical
    return "B5 gap-absorbed ray: nothing intercepted, equals eta=0", ok, (
        f"E_entry[cross]={float(ref.E_entry_eV[CROSS]):.3e} "
        f"anode_intercepted={float(cut.anode_intercepted_erg_s):.3e}"
    )


def main():
    gates = [gate_b1, gate_b2, gate_b3, gate_b4, gate_b5]
    all_ok = True
    print("R4.1 anode-mesh beam-interception gate suite (A15)")
    print("=" * 72)
    for g in gates:
        name, ok, detail = g()
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    print("=" * 72)
    print("R4.1 interception gates:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
