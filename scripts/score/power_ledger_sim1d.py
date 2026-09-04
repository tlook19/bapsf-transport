"""Windowed power ledger over a saved sim1d run: two registered windows, every
channel volume-integrated and phase-tagged, plus the circuit lines, the stored
energy and tau_E, and a per-port breakdown.

READ-ONLY. Nothing is fitted and nothing is written back to the artifact.

WINDOWS. Two, reported separately and never merged: the DRIVE plateau
(default 15.25-19.75 ms) and the AFTERGLOW (default 20.5-24.5 ms). Both are
RUN-CLOCK windows (raw `time`); the breakdown-relative equivalent is printed
beside each.

VOLUMES. Per-cell rhs rows are CGS energy densities (erg cm^-3 s^-1) and are
integrated on the volume their state row is booked on, then divided by 1e7 for
W. The plasma rows `Ee`/`Ei` are on `plasma_volume_cm3`. The neutral row `En`
shares `nn`'s book, which depends on the zone stance: the column volume
V_col = plasma_volume_cm3 under `neutral_two_zone` (the artifact carries an
`nn_a` row), and the chamber volume V_m = neutral_volume_cm3 otherwise. The
annulus carries no energy row in the ratified annulus-cold closure; should one
ever be saved it goes on V_ann = neutral_volume_cm3 - plasma_volume_cm3, the
volume its `nn_a` partner is booked on. The choice made for the artifact in
hand is printed in the header. Sign convention: POSITIVE = power INTO that
fluid.

TAGS. Every channel carries a static phase tag from `CHANNEL_PHASE` below,
derived from what the channel physically is rather than from what a particular
run happened to do. The tag is what makes the under-coupled-vs-over-lossy
question decidable channel by channel: an afterglow-window comparison is
structurally blind to DRIVE-ONLY channels, so a discrepancy carried by those
channels can never be adjudicated in the afterglow.

TAU_E. Stored energy W = 3/2 (pe + pi) . V_p summed over cells, window mean;
tau_E = W / P_coupled with P_coupled = the beam power deposition rows (Ee + Ei,
volume-integrated) ALONE. That row already contains the gap-weighted ohmic
dissipation -- both deposition paths add `ohmic_weights * P_ohmic` into the gap
cells before returning -- so adding `source_P_ohmic` to it, as this script once
did, counted the ohmic term twice. `source_P_ohmic` is still reported, in the
circuit block, and printed beneath P_coupled as the share of it the beam row
carries. The definition is printed in the output header so a quoted tau_E can
never be read against a different denominator.

    power_ledger_sim1d.py RUN.h5 [--drive LO HI] [--afterglow LO HI]
    power_ledger_sim1d.py --selftest [RUN.h5]

Degrades gracefully: an artifact without `rhs_terms` still reports its circuit,
warming and stored-energy lines and says so where the channel table would be;
an artifact without `cathode_diagnostics` reports its channel table and says so
where the circuit lines would be; an energy row a channel does not carry is
omitted rather than entered as a zero; and a channel with no entry in
`CHANNEL_PHASE` is tabulated as UNTAGGED and named in a closing note rather
than being silently absorbed into a subtotal.
"""
import argparse
import json
import os

import h5py
import numpy as np

ERG_PER_J = 1.0e7

DRIVE_WINDOW_MS = (15.25, 19.75)
AFTERGLOW_WINDOW_MS = (20.5, 24.5)

#: ES1 probe ports, cell selected by nearest cell center.
PORTS_Z_CM = {11: 470.05, 21: 789.55, 29: 1045.15, 41: 1428.55, 50: 1716.1}

#: Energy rows read from each rhs_terms channel.
ENERGY_ROWS = ("Ee", "Ei", "En", "En_a")

#: Static phase tag per rhs_terms channel, keyed by channel name, valued
#: (tag, one-line statement of what the channel is).  Tags:
#:   DRIVE-ONLY       the channel's driver is the discharge itself (primary
#:                    beam, cathode emission, electrode circuit, gas puff); it
#:                    goes to zero when the drive ends.
#:   BOTH             the channel is driven by local plasma/neutral state and
#:                    runs in both phases.
#:   AFTERGLOW-ACTIVE the channel is negligible against the drive-phase terms
#:                    and carries the decay after the drive ends.
CHANNEL_PHASE = {
    "anode_collection":
        ("BOTH",
         "energy carried out of the plasma by the Bohm-flux current the anode "
         "mesh collects; the flux is local, not circuit-gated"),
    "beam_excitation_radiation":
        ("DRIVE-ONLY",
         "electron energy radiated away by primary-beam impact excitation; "
         "zero once the cathode solve is disabled"),
    "beam_ionization_birth":
        ("DRIVE-ONLY",
         "particle and energy birth from primary-beam ionization of the "
         "neutral gas"),
    "beam_ionization_cost":
        ("DRIVE-ONLY",
         "ionization potential paid out of the electron fluid for "
         "beam-driven ionization"),
    "beam_power_deposition":
        ("DRIVE-ONLY",
         "primary-beam energy deposited in the background fluids along the "
         "beam path"),
    "boundary_absorption":
        ("BOTH",
         "plasma energy absorbed at the absorbing end faces at the local "
         "Bohm/sheath flux"),
    "cathode_jet_neutral_energy":
        ("DRIVE-ONLY",
         "enthalpy launched into the cold gas with the cathode recycle jet; "
         "rides the cathode solve, so a residual survives the floating "
         "afterglow and it vanishes at post_afterglow"),
    "cathode_surface_loss":
        ("DRIVE-ONLY",
         "the CATHODE half of the sheath-resolved electrode solve: the "
         "cathode surface's particle, momentum and ion-thermal loss, plus "
         "the cathode's own electron sheath power -- milliwatts in "
         "discharge, because the cathode sheath repels plasma electrons. "
         "NB on a PRE-SPLIT artifact this row also carries the anode "
         "electron sheath share and is ~100% anode; the presence of "
         "anode_e_sheath_loss tells the two generations apart"),
    "anode_e_sheath_loss":
        ("DRIVE-ONLY",
         "the ANODE electron sheath deposit (Ee only), landed at the "
         "anode-flanking cells under the Bohm split weights; this is the "
         "~10^5 W channel that used to hide inside cathode_surface_loss. "
         "Absent from pre-split artifacts"),
    "characteristic_boundary":
        ("BOTH",
         "energy leaving through the characteristic ghost-cell boundary at "
         "the local Bohm flux"),
    "ei_exchange":
        ("BOTH",
         "collisional electron-ion temperature equilibration at the local "
         "n, Te, Ti"),
    "electron_ion_cooling":
        ("BOTH",
         "electron energy spent on the inelastic cooling channels of the "
         "rate tables"),
    "electron_neutral_cooling":
        ("BOTH",
         "electron energy lost to neutrals by excitation and elastic "
         "collisions"),
    "flux_tube_geometry":
        ("BOTH",
         "energy bookkeeping of the varying flux-tube cross section; pure "
         "geometry, ungated outside the neutral-only phases"),
    "gas_puff_local_ionization":
        ("DRIVE-ONLY",
         "electron energy spent ionizing the local gas-puff load; follows "
         "the gas-puff phase switch, which under the square waveform stays "
         "on through the afterglow for the closing tail"),
    "heat_conduction":
        ("BOTH",
         "parallel heat conduction on the local temperature gradients; the "
         "explicit row only, the implicit substep books its transport "
         "outside rhs_terms"),
    "hyperbolic_dissipation_heating":
        ("BOTH",
         "Rusanov numerical kinetic-energy dissipation deposited into the "
         "ion internal energy; the pressure half of the old combined "
         "correction now rides pressure_work"),
    "hyperbolic_energy_correction":
        ("BOTH",
         "the COMBINED correction row, in artifacts written before it was "
         "split: the dissipation deposit above plus the energy-consistent "
         "re-discretization of pressure work"),
    "ion_charge_exchange":
        ("BOTH",
         "ion energy exchanged with neutrals through charge exchange; zero "
         "when the ion-neutral moment closure supersedes it"),
    "ion_neutral_collision":
        ("BOTH",
         "elastic ion-neutral friction and thermalization under the moment "
         "closure, heating the cold gas at the ion-neutral slip"),
    "ion_neutral_drag":
        ("BOTH",
         "energy associated with the ion-neutral momentum drag; zero under "
         "the moment closure"),
    "ion_neutral_frictional_heating":
        ("BOTH",
         "heating from the ion-neutral velocity difference; zero under the "
         "moment closure"),
    "ion_neutral_thermalization":
        ("BOTH",
         "ion-neutral temperature relaxation; zero under the moment closure"),
    "ionization_birth":
        ("BOTH",
         "energy carried by particles born in thermal electron-impact "
         "ionization, and the cold gas debited at its own energy"),
    "ionization_energy_cost":
        ("BOTH",
         "ionization potential paid out of the electron fluid for thermal "
         "ionization"),
    "neutral_cx_channel":
        ("BOTH",
         "cold gas debited when charge exchange converts a cold atom into a "
         "fast one, plus the charge-exchange share of the frictional heating"),
    "neutral_energy_wall":
        ("BOTH",
         "free-molecular accommodation of neutral energy at the vessel wall"),
    "neutral_exchange":
        ("BOTH",
         "Knudsen inter-cell neutral diffusion carrying the donor cell's "
         "enthalpy per atom; runs in every phase"),
    "neutral_hot_channel":
        ("BOTH",
         "energy returned to the fluids by the hot charge-exchange-born "
         "ballistic neutrals when their flights land"),
    "neutral_momentum_radial":
        ("BOTH",
         "energy bookkeeping of the radial neutral momentum channel"),
    "neutral_momentum_wall":
        ("BOTH",
         "energy bookkeeping of the neutral wind's wall momentum sink"),
    "neutral_probe_source":
        ("BOTH",
         "prescribed probe neutral source, born at the wall temperature"),
    "neutral_sources":
        ("BOTH",
         "prescribed puff and pump neutral sources; the pump arm is ungated "
         "while the puff arm follows the gas-puff phase switch"),
    "neutral_wind_advection":
        ("BOTH",
         "neutral energy advected by the neutral wind, with its pressure "
         "work"),
    "neutral_zone_exchange":
        ("BOTH",
         "free-molecular column/annulus mixing: gas leaves the column at its "
         "own enthalpy and returns from the annulus at the wall temperature"),
    "plasma_advective_flux":
        ("BOTH",
         "energy advected by the plasma flow across cell faces"),
    "plasma_front_flux":
        ("BOTH",
         "energy carried across the propagating plasma front"),
    "pressure_work":
        ("BOTH",
         "pdV work done by the plasma flow"),
    "recombination_3b_loss":
        ("AFTERGLOW-ACTIVE",
         "three-body recombination, ungated but steep in 1/Te, so it is "
         "negligible against the drive terms and grows as the plasma cools"),
    "recombination_energy_return":
        ("AFTERGLOW-ACTIVE",
         "the (3/2) k Ti the recombining ion hands to the neutral it "
         "becomes; follows the recombination rate"),
    "recombination_rad_loss":
        ("AFTERGLOW-ACTIVE",
         "energy radiated away in radiative recombination; follows the same "
         "cooling-dominated rate"),
    "surface_loss":
        ("BOTH",
         "structurally retained row of the saved rhs_terms layout; the live "
         "solver zeroes it in both branches"),
}

#: Founding drive-window numbers, measured 2026-08-19 on the artifact named
#: below at the registered drive window, keyed by the quantity the --selftest
#: mode re-measures and carried as (name, reference, significant figures
#: compared).  Four figures where the founding read gave four; the cathode
#: line is held to three.
SELFTEST_ARTIFACT = "g1a_foot45_cr6p94.h5"
SELFTEST_WINDOW_MS = (15.25, 19.75)
SELFTEST_REFERENCE = (
    ("cathode_jet_neutral_energy/En", 22.008, 4),
    ("warming_E_ion_J slope", 184.058, 4),
    ("beam_power_deposition/Ee", 322.3, 4),
    # The electrode electron sheath pair, summed: this reference predates the
    # per-electrode split, so it must be read as the PAIR to stay comparable
    # across both artifact generations (the second row is absent, and so
    # contributes zero, on a pre-split artifact).
    ("electrode e-sheath pair/Ee", -96.1, 3),
)


def tag_of(channel):
    """Phase tag for a channel name; UNTAGGED when it is not in the table."""
    return CHANNEL_PHASE.get(channel, ("UNTAGGED", ""))[0]


def sigfig(value, digits=4):
    """`value` rounded to `digits` significant figures (0.0 maps to 0.0)."""
    if value == 0.0 or not np.isfinite(value):
        return float(value)
    return float(f"%.{digits - 1}e" % value)


def volume_book(f, Vp, Vm):
    """Per-row cell volumes for this artifact.

    `Ee`/`Ei` are plasma-column rows.  `En` shares `nn`'s book: the column
    volume under the two-zone stance (an `nn_a` row is present), the chamber
    volume otherwise.  An annulus row goes on V_ann = Vm - Vp.
    """
    two_zone = "nn_a" in f
    return {
        "Ee": Vp,
        "Ei": Vp,
        "En": Vp if two_zone else Vm,
        "En_a": np.maximum(Vm - Vp, 0.0),
    }, two_zone


def window_frames(t_ms, lo, hi):
    """Boolean mask and (first, last) saved-frame indices in [lo, hi] ms."""
    mask = (t_ms >= lo) & (t_ms <= hi)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(
            f"no saved frames in the window [{lo}, {hi}] ms "
            f"(artifact covers {t_ms[0]:.4f}..{t_ms[-1]:.4f} ms)"
        )
    return mask, int(idx[0]), int(idx[-1])


def integrate_rows(f, i0, i1, vols):
    """Window-mean volume-integrated power [kW] per (channel, row), and the
    list of channels present in the artifact."""
    if "rhs_terms" not in f:
        return {}, []
    channels = sorted(f["rhs_terms"].keys())
    table = {}
    for channel in channels:
        for name in ENERGY_ROWS:
            key = f"rhs_terms/{channel}/{name}"
            if key not in f:
                continue
            mean = np.mean(f[key][i0:i1 + 1], axis=0)
            table[(channel, name)] = (
                float(mean.dot(vols[name])) / ERG_PER_J / 1e3)
    return table, channels


def channel_table(table, floor_kW):
    """Channel rows by descending |P|, split into shown and suppressed."""
    rows = sorted(table.items(), key=lambda kv: -abs(kv[1]))
    shown = [(k, v) for k, v in rows if abs(v) >= floor_kW]
    hidden = [(k, v) for k, v in rows if abs(v) < floor_kW]
    return shown, hidden


def diagnostic_mean(dg, key, mask):
    """Window mean of a scalar cathode diagnostic, or nan when absent."""
    if dg is None or key not in dg:
        return float("nan")
    return float(np.mean(dg[key][:][mask]))


def counter_slope(dg, key, i0, i1, dt_s):
    """Endpoint slope [W] of a cumulative energy counter over the window."""
    if dg is None or key not in dg:
        return float("nan")
    d = dg[key][:]
    return (float(d[i1]) - float(d[i0])) / dt_s


def stored_energy_J(f, i0, i1, Vp):
    """Window-mean stored plasma energy 3/2 (pe + pi) . Vp [J]."""
    if "pe" not in f or "pi" not in f:
        return float("nan")
    pe = np.mean(f["pe"][i0:i1 + 1], axis=0)
    pi = np.mean(f["pi"][i0:i1 + 1], axis=0)
    return float(1.5 * (pe + pi).dot(Vp)) / ERG_PER_J


def report_window(f, label, lo, hi, geom, port_top):
    """Print the ledger for one window; return its (channel, row) table."""
    t_ms = geom["t_ms"]
    mask, i0, i1 = window_frames(t_ms, lo, hi)
    t0_ms = geom["t0_ms"]
    dt_s = (t_ms[i1] - t_ms[i0]) * 1e-3
    Vp = geom["Vp"]
    dg = f["cathode_diagnostics"] if "cathode_diagnostics" in f else None

    phases = "n/a"
    if "phase" in f:
        seen = []
        for p in f["phase"][:][mask]:
            p = p.decode() if isinstance(p, bytes) else str(p)
            if p not in seen:
                seen.append(p)
        phases = ", ".join(seen)

    print()
    print("=" * 88)
    print(f"WINDOW {label}   run clock [{lo}, {hi}] ms "
          f"= breakdown-relative [{lo - t0_ms:.4f}, {hi - t0_ms:.4f}] ms")
    print(f"  frames {i0}..{i1}  ({int(mask.sum())} saves, "
          f"t {t_ms[i0]:.4f}..{t_ms[i1]:.4f} ms, span {dt_s * 1e3:.4f} ms)")
    print(f"  phases in window: {phases}")
    print("=" * 88)

    table, channels = integrate_rows(f, i0, i1, geom["vols"])
    print("\n--- CHANNEL TABLE [kW], window mean, volume-integrated, "
          "sorted by |P| (positive = INTO the fluid) ---")
    if not table:
        print("  rhs_terms ABSENT from this artifact -- channel table skipped")
    else:
        shown, hidden = channel_table(table, 1.0e-6)
        print(f"{'channel':<32}{'row':>5}{'tag':>19}{'P [kW]':>16}")
        subtotal = {}
        for (channel, name), value in shown:
            tag = tag_of(channel)
            subtotal[tag] = subtotal.get(tag, 0.0) + value
            print(f"{channel:<32}{name:>5}{tag:>19}{value:>16.5f}")
        for (channel, name), value in hidden:
            tag = tag_of(channel)
            subtotal[tag] = subtotal.get(tag, 0.0) + value
        print(f"  ({len(hidden)} rows below 1e-6 kW not listed; "
              "their tag subtotals still include them)")
        print()
        for tag in ("DRIVE-ONLY", "BOTH", "AFTERGLOW-ACTIVE", "UNTAGGED"):
            if tag in subtotal:
                print(f"{'  subtotal ' + tag:<56}{subtotal[tag]:>16.5f}  kW")
        untagged = sorted({c for c in channels if c not in CHANNEL_PHASE})
        if untagged:
            print(f"  NOTE untagged channels present in the artifact: "
                  f"{untagged}")
        if label == "AFTERGLOW":
            live = sorted(((c, n, v) for (c, n), v in table.items()
                           if tag_of(c) == "DRIVE-ONLY"
                           and abs(v) >= 1.0e-6),
                          key=lambda r: -abs(r[2]))
            if live:
                print("  WARNING: DRIVE-ONLY channels are NONZERO in this "
                      "afterglow window (the inductive tail keeps the "
                      "cathode solve live):")
                for c, n, v in live:
                    print(f"    {c}/{n:<4} {v:>16.5f}  kW")
                print("    an afterglow dichotomy read is NOT clean on these "
                      "channels in this window; move the window past the "
                      "tail or discount them explicitly.")

    print("\n--- CIRCUIT AND SOURCE DIAGNOSTICS, window mean ---")
    if dg is None:
        print("  cathode_diagnostics ABSENT from this artifact")
    else:
        I_loop = diagnostic_mean(dg, "circuit_I_loop", mask)
        V_dis = diagnostic_mean(dg, "circuit_V_dis_step", mask)
        print(f"{'circuit_I_loop':<44}{I_loop:>16.5f}  A")
        print(f"{'circuit_V_dis_step':<44}{V_dis:>16.5f}  V")
        print(f"{'I.V INPUT = I_loop * V_dis_step':<44}"
              f"{I_loop * V_dis / 1e3:>16.5f}  kW")
        for key in sorted(k for k in dg.keys() if k.startswith("source_P_")):
            print(f"{'  ' + key:<44}"
                  f"{diagnostic_mean(dg, key, mask) / 1e3:>16.5f}  kW")

        print("\n--- WARMING COUNTER SLOPES over the window ---")
        for key in sorted(k for k in dg.keys()
                          if k.startswith("warming_E_") and k.endswith("_J")):
            print(f"{'  ' + key:<44}"
                  f"{counter_slope(dg, key, i0, i1, dt_s) / 1e3:>16.5f}  kW")

    print("\n--- STORED ENERGY AND tau_E ---")
    W_J = stored_energy_J(f, i0, i1, Vp)
    # P_coupled is the beam deposition row ALONE. That row already carries the
    # gap-weighted ohmic dissipation: both deposition paths in
    # cablp/solvers/_sim1d/physics/cathode.py add
    # ohmic_weights * solver_result.P_ohmic into the gap cells of the same
    # density they return. Adding source_P_ohmic on top booked the ohmic term
    # twice and inflated the tau_E denominator. The circuit's own P_ohmic is
    # still reported above, in the circuit block, where it belongs.
    P_coupled = (table.get(("beam_power_deposition", "Ee"), 0.0)
                 + table.get(("beam_power_deposition", "Ei"), 0.0))
    ohmic = diagnostic_mean(dg, "source_P_ohmic", mask) / 1e3
    print(f"{'W = 3/2 (pe + pi) . Vp, window mean':<44}{W_J:>16.5f}  J")
    print(f"{'P_coupled = beam_power_deposition (Ee + Ei)':<44}"
          f"{P_coupled:>16.5f}  kW")
    print(f"{'  of which gap ohmic (source_P_ohmic)':<44}"
          f"{ohmic:>16.5f}  kW")
    if P_coupled != 0.0:
        print(f"{'tau_E = W / P_coupled':<44}"
              f"{W_J / P_coupled:>16.6f}  ms")
    else:
        # The window couples no power at all, so W / P_coupled has no value.
        # The row itself is withheld rather than printed as NaN: the header
        # says tau_E is not reported where the beam is off, and a printed row
        # contradicts that before the reader reaches the NOTE below.
        # Said out loud: a bare NaN in this row reads as a broken artifact,
        # and the ohmic line above is a CIRCUIT quantity that this window's
        # deposition row does not carry.
        print("  NOTE: nothing is deposited in this window -- the beam row is "
              "zero, so there is no\n  denominator and tau_E is undefined "
              "here. The ohmic line above is the circuit's own\n  "
              "dissipation, which the deposition row does not carry once the "
              "cathode stops depositing.")

    print(f"\n--- PER-PORT Ee CHANNEL DENSITIES [W cm^-3], top {port_top} "
          "by |value|, window mean ---")
    if "rhs_terms" not in f:
        print("  rhs_terms ABSENT from this artifact -- port block skipped")
        return table
    z = geom["z"]
    roles = geom["roles"]
    dens = {}
    for channel in channels:
        key = f"rhs_terms/{channel}/Ee"
        if key in f:
            dens[channel] = np.mean(f[key][i0:i1 + 1], axis=0) / ERG_PER_J
    n_w = np.mean(f["n"][i0:i1 + 1], axis=0) if "n" in f else None
    Te_w = np.mean(f["Te"][i0:i1 + 1], axis=0) if "Te" in f else None
    for port, z_want in PORTS_Z_CM.items():
        i = int(np.argmin(np.abs(z - z_want)))
        head = (f"port {port:<3} z_want {z_want:>8.2f}  cell {i:>4} "
                f"z {z[i]:>8.2f}  role {roles[i]}")
        if n_w is not None and Te_w is not None:
            head += f"  n {n_w[i]:.4e} cm^-3  Te {Te_w[i]:.4f} eV"
        print(f"\n  {head}")
        local = sorted(((abs(v[i]), c, float(v[i])) for c, v in dens.items()),
                       key=lambda r: -r[0])[:port_top]
        for rank, (_, channel, value) in enumerate(local, start=1):
            print(f"    {rank}. {channel:<32}{tag_of(channel):>19}"
                  f"{value:>16.6e}")
    return table


def load_geometry(f):
    """Grid, time base and volume books shared by both windows."""
    Vp = f["geometry/plasma_volume_cm3"][:]
    Vm = f["geometry/neutral_volume_cm3"][:]
    vols, two_zone = volume_book(f, Vp, Vm)
    return {
        "t_ms": f["time"][:] * 1e3,
        "t0_ms": float(f.attrs["t_breakdown_trigger"]) * 1e3,
        "Vp": Vp,
        "Vm": Vm,
        "vols": vols,
        "two_zone": two_zone,
        "z": f["geometry/z_cm"][:],
        "roles": [r.decode() if isinstance(r, bytes) else str(r)
                  for r in f["geometry/cell_role"][:]],
    }


def print_header(f, path, geom, drive, afterglow):
    """Artifact identity, stance, volume books and the tau_E definition."""
    flags = json.loads(f.attrs.get("flags_json", "{}"))
    params = json.loads(f.attrs.get("params_json", "{}"))
    Vp, Vm = geom["Vp"], geom["Vm"]
    print("=== WINDOWED POWER LEDGER ===")
    print(f"artifact : {path}")
    print(f"run      : run_status={f.attrs.get('run_status')!s} "
          f"steps={f.attrs.get('steps')!s} "
          f"compiled_kernels={f.attrs.get('compiled_kernels')!s} "
          f"saves={f['time'].shape[0]}")
    print(f"stance   : neutral_two_zone={flags.get('neutral_two_zone')}, "
          f"neutral_energy={flags.get('neutral_energy')}, "
          f"neutral_hot_internal_wall="
          f"{flags.get('neutral_hot_internal_wall')}, "
          f"cathode_neutral_jet={params.get('cathode_neutral_jet')}, "
          f"C_R={params.get('C_R')}")
    print(f"grid     : {Vp.size} cells, V_p total {Vp.sum():.6e} cm^3, "
          f"V_m total {Vm.sum():.6e} cm^3, "
          f"V_ann total {np.maximum(Vm - Vp, 0.0).sum():.6e} cm^3")
    en_book = ("V_col = plasma_volume_cm3 (two-zone: nn_a present)"
               if geom["two_zone"] else
               "V_m = neutral_volume_cm3 (one-zone: no nn_a row)")
    print(f"volumes  : Ee/Ei rows x plasma_volume_cm3 ; En rows x {en_book} ; "
          "any *_a row x V_ann")
    print("units    : rhs rows are erg cm^-3 s^-1; powers are kW, "
          "POSITIVE = into that fluid")
    print("tau_E    : W = 3/2 (pe + pi) . V_p (window mean) divided by "
          "P_coupled := beam_power_deposition")
    print("           (Ee + Ei, volume-integrated) ALONE, which already "
          "carries the gap ohmic; both windows")
    print("           use this same definition. tau_E is a bookkeeping ratio "
          "against that denominator;")
    print("           where the beam is off there is no denominator and it is "
          "not reported")
    print(f"windows  : DRIVE {drive[0]}-{drive[1]} ms, "
          f"AFTERGLOW {afterglow[0]}-{afterglow[1]} ms (run clock)")
    print("tags     : DRIVE-ONLY / BOTH / AFTERGLOW-ACTIVE, static and "
          "physical (see CHANNEL_PHASE in this file)")


def run_report(path, drive, afterglow, port_top):
    """Full two-window report for one artifact."""
    with h5py.File(path, "r") as f:
        geom = load_geometry(f)
        print_header(f, path, geom, drive, afterglow)
        report_window(f, "DRIVE", drive[0], drive[1], geom, port_top)
        report_window(f, "AFTERGLOW", afterglow[0], afterglow[1], geom,
                      port_top)


def selftest(path):
    """Re-measure the founding drive-window numbers and hard-assert them."""
    with h5py.File(path, "r") as f:
        geom = load_geometry(f)
        lo, hi = SELFTEST_WINDOW_MS
        _, i0, i1 = window_frames(geom["t_ms"], lo, hi)
        dt_s = (geom["t_ms"][i1] - geom["t_ms"][i0]) * 1e-3
        table, _ = integrate_rows(f, i0, i1, geom["vols"])
        dg = f["cathode_diagnostics"]
        measured = {
            "cathode_jet_neutral_energy/En":
                table[("cathode_jet_neutral_energy", "En")],
            "warming_E_ion_J slope":
                counter_slope(dg, "warming_E_ion_J", i0, i1, dt_s) / 1e3,
            "beam_power_deposition/Ee":
                table[("beam_power_deposition", "Ee")],
            "electrode e-sheath pair/Ee":
                table[("cathode_surface_loss", "Ee")]
                + table.get(("anode_e_sheath_loss", "Ee"), 0.0),
        }
    print("=== SELFTEST: founding drive-window numbers of record ===")
    print(f"artifact : {path}")
    print(f"window   : run clock [{lo}, {hi}] ms, frames {i0}..{i1}")
    print(f"{'quantity':<36}{'measured [kW]':>18}{'reference [kW]':>18}"
          f"{'s.f.':>6}{'verdict':>10}")
    failures = []
    for name, reference, digits in SELFTEST_REFERENCE:
        value = measured[name]
        ok = sigfig(value, digits) == sigfig(reference, digits)
        if not ok:
            failures.append((name, value, reference))
        print(f"{name:<36}{value:>18.5f}{reference:>18.5f}{digits:>6}"
              f"{('MATCH' if ok else 'DIFFER'):>10}")
    assert not failures, (
        "power ledger selftest FAILED against the founding numbers: "
        + "; ".join(f"{n}: measured {v!r} vs reference {r!r}"
                    for n, v, r in failures)
    )
    print("SELFTEST PASSED")


def main():
    parser = argparse.ArgumentParser(
        description="Windowed, phase-tagged power ledger over a saved sim1d "
                    "run (read-only).")
    parser.add_argument("run", nargs="?",
                        help="sim1d HDF5 result; optional with --selftest, "
                             f"which otherwise reads {SELFTEST_ARTIFACT} "
                             "beside this script")
    parser.add_argument("--drive", nargs=2, type=float,
                        metavar=("LO_MS", "HI_MS"), default=DRIVE_WINDOW_MS,
                        help="drive plateau window, run clock [ms]")
    parser.add_argument("--afterglow", nargs=2, type=float,
                        metavar=("LO_MS", "HI_MS"),
                        default=AFTERGLOW_WINDOW_MS,
                        help="afterglow window, run clock [ms]")
    parser.add_argument("--port-top", type=int, default=6,
                        help="channels listed per ES1 port")
    parser.add_argument("--selftest", action="store_true",
                        help="re-measure the founding numbers and assert them")
    args = parser.parse_args()

    if args.selftest:
        path = args.run or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), SELFTEST_ARTIFACT)
        selftest(path)
        return
    if args.run is None:
        parser.error("a run artifact is required (or use --selftest)")
    run_report(args.run, tuple(args.drive), tuple(args.afterglow),
               args.port_top)


if __name__ == "__main__":
    main()
