"""Step-1 scoping for the cathode neutral-jet hypothesis.

Reads the saved M5a' reference matrices (no physics change; the only model
import is the run-constant topology helper that resolves which cell a boundary
face books into) and computes, per ladder rung:

  (b) recycle flux vs puff; the jet velocity per channel -- fast backscatter
      v_back = sqrt(2 R_E (phi_c + Ti) / m) with (R_N, R_E) literature-BOXED
      (Eckstein/Thomas-class He reflection data; see the bracket below), the
      effusive channel at the map-derived standby T_s (per-particle directed
      momentum of a cosine-law effusive flux, <p>/m = sqrt(pi k T_s / 2 m));
  (c) the jet momentum flux against the column-integrated drag momentum
      exchange read from the saved rhs_terms (ion-side sink of
      `ion_neutral_drag`, the constant-0.5 closure of these references).

Pre-registered gate: if the jet momentum source is < 10 % of the near-source
drag momentum exchange at boxed (R_N, R_E), the hypothesis is immaterial.

(R_N, R_E) provenance / the box
-------------------------------
The cathode is LaB6; backscattering at 60-200 eV is set by the first few
monolayers, and the honest uncertainty is the surface termination (B-rich vs
La), which dwarfs the fit-formula uncertainty:

  - He -> heavy target (La Z=57/M=139; W-class): reduced energy
    eps ~ 0.004-0.014 over the ladder's per-ion energies; Eckstein-class
    data (Data Compendium for PSI; Borovikov et al. JNM 2014 MD for He->W
    at <= 100 eV) put R_N ~ 0.5-0.7, R_E ~ 0.2-0.4.
  - He -> B (Z=5/M=10.8): eps ~ 0.07-0.22; light-target reflection is far
    weaker: R_N ~ 0.05-0.15, R_E ~ 0.02-0.06.

The scoping bracket below evaluates the OPTIMISTIC (La-like), MID (the
nominal R_N 0.5 / R_E 0.2), and PESSIMISTIC (B-like) corners; the gate verdict is
taken at the pessimistic corner.

Usage:
    python scripts/scope_cathode_jet.py
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mc_neutrals import (  # noqa: E402
    BOUNDARY_ROWS,
    assert_recycle_channel_live,
    boundary_recycle_row,
    two_zone_puff_row_from_config,
)

from cablp.constants import m_He_cgs  # noqa: E402
from cablp.solvers._sim1d.core.geometry import (  # noqa: E402
    absorbing_live_cells_by_role,
    build_geometry,
)

E_CHARGE = 1.602176634e-19        # C
EV_TO_ERG = 1.602176634e-12
K_B_ERG = 1.380649e-16
# Helium mass: imported, never re-derived. cablp.constants is THE
# definition point (Ar(4He)*u, CODATA 2022); the hand-made
# 4.002602 * 1.66053907e-24 product this replaced was 0.31 ppm low.
M_HE_G = m_He_cgs

REFS = {
    "ES1": "scripts/es1_nx120_m5ap_es1.h5",
    "ES2": "scripts/es1_nx120_m5ap_ladder_es2.h5",
    "ES3": "scripts/es1_nx120_m5ap_ladder_es3.h5",
}

# (label, R_N, R_E) -- the literature box (docstring above).
RN_RE_BOX = [
    ("pessimistic (B-like)", 0.10, 0.04),
    ("mid (nominal)", 0.50, 0.20),
    ("optimistic (La-like)", 0.60, 0.35),
]

PLATEAU_MS = (5.0, 19.5)          # relative to the breakdown trigger
GATE_FRACTION = 0.10

# Near-source windows for the drag comparison, as z-extent [cm] of column
# cells measured from the cathode face.
WINDOWS_CM = (150.0, 300.0, 500.0, np.inf)


def plateau_mask(f):
    t0 = float(f.attrs["t_breakdown_trigger"])
    t = f["time"][:] - t0
    lo, hi = PLATEAU_MS
    return (t >= lo * 1e-3) & (t <= hi * 1e-3)


def scope_one(path):
    with h5py.File(path, "r") as f:
        mask = plateau_mask(f)
        cd = f["cathode_diagnostics"]
        avg = lambda ds: float(np.mean(np.asarray(ds)[mask]))

        I_i = avg(cd["source_I_i"])                    # A
        P_cath_i = avg(cd["source_P_cathode_i"])       # W
        phi_c = avg(cd["source_phi_c"])                # V
        T_s = avg(cd["T_s_surface"])                   # K
        I_loop = avg(cd["circuit_I_loop"])             # A

        roles = [
            r.decode() if isinstance(r, bytes) else str(r)
            for r in f["geometry/cell_role"][:]
        ]
        cathode_cell = roles.index("cathode")
        z = f["geometry/z_cm"][:]
        Vp = f["geometry/plasma_volume_cm3"][:]
        Vm = f["geometry/neutral_volume_cm3"][:]
        column = np.array([r == "column" for r in roles])
        z_cathode_face = z[cathode_cell] - 0.5 * f["geometry/length_cm"][cathode_cell]

        Ti_cath = float(np.mean(f["Ti"][:, cathode_cell][mask]))  # eV

        # Recycle flux cross-check: the boundary term's neutral gain at the
        # cathode face's LIVE cell [1/cm^3/s] * neutral volume -> atoms/s.
        # Which row carries it is stance-dependent (boundary_recycle_row); the
        # live cell is resolved by role, not by position, because an
        # obstruction cell displaces it inward.
        row, stance = boundary_recycle_row(f)
        params = json.loads(f.attrs["params_json"])
        raw_flags = f.attrs.get("flags_json")
        flags = json.loads(raw_flags) if raw_flags is not None else {}
        by_role = absorbing_live_cells_by_role(build_geometry(params, flags))
        recycle_cell = (
            int(by_role["cathode"][0]) if "cathode" in by_role else cathode_cell
        )
        # Volume convention, exactly as for the puff below: under the two-zone
        # closure the boundary term's nn row is the COLUMN density rate and
        # integrates on Vp, so the flat * Vm read over-counted this face by
        # Vm/Vp; under end_recycle_to_annulus the collector faces' share moves
        # to nn_a on the annulus (Vm - Vp), which the cathode face does not
        # use but which is consumed anyway so the read stays correct per cell.
        # The stance is keyed off the STATE (nn_a), not off per-term row
        # presence: a two-zone artifact predating the per-term annulus rows
        # still has its nn row on Vp, and keying off the row restored the Vm
        # over-count there. Reduces to nn * Vm on a single-zone state.
        two_zone = "nn_a" in f
        ba_nn = np.asarray(f[f"rhs_terms/{row}/nn"][:, recycle_cell])[mask]
        if not two_zone:
            ba_gain = ba_nn * Vm[recycle_cell]
        elif f"rhs_terms/{row}/nn_a" in f:
            ba_nn_a = np.asarray(
                f[f"rhs_terms/{row}/nn_a"][:, recycle_cell]
            )[mask]
            ba_gain = ba_nn * Vp[recycle_cell] + ba_nn_a * max(
                Vm[recycle_cell] - Vp[recycle_cell], 0.0
            )
        else:
            ba_gain = ba_nn * Vp[recycle_cell]
        recycle_ba = float(np.mean(ba_gain))
        removal_any = sum(
            -np.mean(np.asarray(f[f"rhs_terms/{name}/n"][:, recycle_cell])[mask])
            * Vp[recycle_cell]
            for name in BOUNDARY_ROWS
            if f"rhs_terms/{name}/n" in f
        )
        assert_recycle_channel_live(
            recycle_ba,
            removal_any,
            row=row,
            stance=stance,
            path=str(path),
            window_ms=PLATEAU_MS,
        )

        # Puff rate: positive part of neutral_sources' nn term (puff source;
        # the pump sink is the negative part), volume-integrated. Under the
        # two-zone closure the puff is booked into the ANNULUS row (nn_a) --
        # the pipe enters at the wall -- so the nn row alone is pump-only and
        # reads zero puff; each zone is integrated on its own volume, which
        # reduces to nn * Vm on a single-zone state. On a two-zone artifact
        # that saved no annulus row the puff is absent from the ledger
        # ENTIRELY, and a zero here would be a phantom rather than a physical
        # zero (it divides the reported recycle/puff ratio), so it comes from
        # the same config derivation the TPMC loader uses -- which refuses
        # loudly rather than guessing outside the waveform it certifies.
        ns_nn = np.asarray(f["rhs_terms/neutral_sources/nn"])[mask]
        if not two_zone:
            puff_cell = np.clip(ns_nn, 0.0, None) * Vm
        elif "rhs_terms/neutral_sources/nn_a" in f:
            ns_nn_a = np.asarray(f["rhs_terms/neutral_sources/nn_a"])[mask]
            puff_cell = np.clip(ns_nn, 0.0, None) * Vp + np.clip(
                ns_nn_a, 0.0, None
            ) * np.maximum(Vm - Vp, 0.0)
        else:
            puff_cell = np.clip(ns_nn, 0.0, None) * Vp
        puff = float(np.mean(np.sum(puff_cell, axis=1)))
        if two_zone and "rhs_terms/neutral_sources/nn_a" not in f:
            derived_row, derived_provenance = two_zone_puff_row_from_config(
                f, params, flags, f["time"][:], mask, Vm,
                np.maximum(Vm - Vp, 0.0),
            )
            puff += float(derived_row.sum())
            if derived_provenance is not None:
                print(
                    f"[scope_cathode_jet] {derived_provenance}, for {path}.",
                    file=sys.stderr,
                )

        # Drag momentum exchange: ion-side sink of ion_neutral_drag [g/cm^2/s^2]
        # * plasma volume -> dyn, per cell, plateau-averaged.
        drag_M = np.asarray(f["rhs_terms/ion_neutral_drag/M"])[mask]
        drag_dyn_cell = np.mean(np.abs(drag_M) * Vp[None, :], axis=0)

        windows = {}
        for w in WINDOWS_CM:
            sel = column & ((z - z_cathode_face) <= w)
            key = "full column" if not np.isfinite(w) else f"z < {w:.0f} cm"
            windows[key] = float(np.sum(drag_dyn_cell[sel]))

    Gamma = I_i / E_CHARGE                             # atoms/s
    E_ion = P_cath_i / I_i                             # eV per ion (honest)
    v_eff = np.sqrt(np.pi * K_B_ERG * T_s / (2.0 * M_HE_G))  # cm/s

    corners = []
    for label, R_N, R_E in RN_RE_BOX:
        v_back = np.sqrt(2.0 * R_E * (phi_c + Ti_cath) * EV_TO_ERG / M_HE_G)
        # Variant with the per-particle convention R_E/R_N and the honest
        # per-ion energy (reported, not gated on).
        v_back_alt = np.sqrt(2.0 * (R_E / R_N) * E_ion * EV_TO_ERG / M_HE_G)
        F_jet = Gamma * M_HE_G * (R_N * v_back + (1.0 - R_N) * v_eff)
        F_jet_alt = Gamma * M_HE_G * (R_N * v_back_alt + (1.0 - R_N) * v_eff)
        corners.append(
            dict(label=label, R_N=R_N, R_E=R_E,
                 v_back_km_s=v_back / 1e5, v_back_alt_km_s=v_back_alt / 1e5,
                 F_jet_dyn=F_jet, F_jet_alt_dyn=F_jet_alt)
        )

    return dict(
        I_i_A=I_i, I_loop_A=I_loop, P_cathode_i_W=P_cath_i,
        E_ion_eV=E_ion, phi_c_V=phi_c, Ti_cathode_eV=Ti_cath, T_s_K=T_s,
        Gamma_per_s=Gamma, recycle_ba_per_s=recycle_ba, puff_per_s=puff,
        v_eff_km_s=v_eff / 1e5, corners=corners, drag_windows_dyn=windows,
    )


def main():
    out = {}
    for rung, path in REFS.items():
        r = scope_one(path)
        out[rung] = r
        print(f"\n=== {rung}  ({path}) ===")
        print(f"  I_i {r['I_i_A']:.1f} A  (loop {r['I_loop_A']:.1f} A)   "
              f"P_cathode_i {r['P_cathode_i_W'] / 1e3:.1f} kW")
        print(f"  per-ion energy E = P/I_i = {r['E_ion_eV']:.1f} eV   "
              f"phi_c {r['phi_c_V']:.1f} V   Ti(cath) {r['Ti_cathode_eV']:.2f} eV   "
              f"T_s {r['T_s_K']:.0f} K")
        print(f"  recycle flux I_i/e = {r['Gamma_per_s']:.3e} /s   "
              f"[boundary_absorption cross-check {r['recycle_ba_per_s']:.3e} /s]")
        print(f"  puff rate {r['puff_per_s']:.3e} /s   "
              f"recycle/puff = {r['Gamma_per_s'] / r['puff_per_s']:.2f}")
        print(f"  effusive channel: v_eff = {r['v_eff_km_s']:.2f} km/s")
        for c in r["corners"]:
            print(f"  [{c['label']:>22s}] R_N {c['R_N']:.2f} R_E {c['R_E']:.2f}  "
                  f"v_back {c['v_back_km_s']:.1f} km/s "
                  f"(alt {c['v_back_alt_km_s']:.1f})  "
                  f"F_jet {c['F_jet_dyn']:.3e} dyn (alt {c['F_jet_alt_dyn']:.3e})")
        print("  drag momentum exchange (plateau avg, |ion-side sink|):")
        for k, v in r["drag_windows_dyn"].items():
            print(f"    {k:>12s}: {v:.3e} dyn")
        pess = r["corners"][0]
        for k, v in r["drag_windows_dyn"].items():
            frac = pess["F_jet_dyn"] / v if v > 0 else np.inf
            print(f"  gate @ pessimistic corner vs {k}: "
                  f"jet/drag = {frac:.2f} "
                  f"{'PASS' if frac >= GATE_FRACTION else 'FAIL'} (gate 0.10)")

    with open("scripts/scope_cathode_jet.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nsaved scripts/scope_cathode_jet.json")


if __name__ == "__main__":
    main()
