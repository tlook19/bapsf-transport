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

CIRCUIT PROJECTION. `circuit_projection_dropped` [W], in the circuit block,
PRESENCE-GATED on `cathode_circuit_project_over_wall`: the window rate
(endpoint slope) of the cumulative `circuit_projection_energy_J` counter, the
inductor energy a projection event drops (`0.5 * L * (I_held**2 -
I_root**2)`, non-negative by construction) rather than hands to any plasma
channel. A named ledger row, not a residual, so the loop's I.V input and its
`source_P_*` sinks do not carry it as an unexplained gap. In the DRIVE
window only, printed beside a PRE-REGISTERED clause: the rate must sit under
0.1% of the window-mean `source_P_prim`. Absent (not zero) on a run that
never armed the key.
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

#: READ-SIDE ALIASES, retired label -> current label. The model's far face is
#: the LAPD chamber's END WALL, and the role, keys and rows that carried the
#: old ``collector`` name were renamed. A trajectory saved before the rename
#: holds the old names in its groups, its params block and its
#: ``geometry/cell_role``, so every read below resolves through this table
#: and reports the CURRENT label whatever the artifact spells it. It is a
#: read alias only: nothing here is ever written.
LEGACY_LABEL_ALIASES = {
    # the saved cell role
    "collector": "end_wall",
    # saved rhs_terms channel / cathode_diagnostics dataset names
    "collector_e_sheath_climb": "end_wall_e_sheath_climb",
    "collector_surface_power_W": "end_wall_surface_power_W",
    # saved dvm_particle_ledger dataset names
    "birth_collector_face": "birth_end_wall_face",
    "birth_collector_jet": "birth_end_wall_jet",
    "energy_birth_collector_face": "energy_birth_end_wall_face",
    "energy_birth_collector_jet": "energy_birth_end_wall_jet",
    # the DVM CHANNEL names those rows are derived from, and the internal
    # per-attempt booking record. Neither is a dataset name on its own; both
    # are carried so a caller holding a channel name read off an old
    # artifact's row can resolve it here rather than re-deriving the map.
    "collector_face": "end_wall_face",
    "collector_jet": "end_wall_jet",
    "collector_jet_energy_booking": "end_wall_jet_energy_booking",
    # saved params/flags keys
    "collector_length_cm": "end_wall_length_cm",
    "collector_sheath_full_debit": "end_wall_sheath_full_debit",
    "neutral_kinetic_dvm_collector_jet": "neutral_kinetic_dvm_end_wall_jet",
    "neutral_kinetic_dvm_collector_jet_R_N":
        "neutral_kinetic_dvm_end_wall_jet_R_N",
    "neutral_kinetic_dvm_collector_jet_R_E":
        "neutral_kinetic_dvm_end_wall_jet_R_E",
    "neutral_kinetic_dvm_collector_jet_T_launch_eV":
        "neutral_kinetic_dvm_end_wall_jet_T_launch_eV",
    "neutral_kinetic_dvm_collector_jet_sheath_Te_multiple":
        "neutral_kinetic_dvm_end_wall_jet_sheath_Te_multiple",
}

#: The same table read the other way, current label -> the retired label a
#: pre-rename artifact stores it under.
LEGACY_LABELS_BY_CURRENT = {
    new: old for old, new in LEGACY_LABEL_ALIASES.items()
}


def current_label(name):
    """Return ``name``'s current spelling, mapping a retired one."""
    return LEGACY_LABEL_ALIASES.get(name, name)


def legacy_get(mapping, key, default=None):
    """``mapping[key]``, falling back to ``key``'s retired spelling."""
    if key in mapping:
        return mapping[key]
    old = LEGACY_LABELS_BY_CURRENT.get(key)
    if old is not None and old in mapping:
        return mapping[old]
    return default

#: The four rhs_terms channels the two END-FACE keys add, in the order the
#: per-window block reports them: `end_wall_sheath_full_debit`'s one row
#: first, then `cathode_face_full_debit`'s emitting-face three.  Each row is
#: PRESENCE-GATED on ITS OWN key, so a run may carry the end wall row alone,
#: the three cathode rows alone, all four, or none -- absence here means
#: "never booked", never "booked zero".
END_SHEATH_ROWS = (
    "end_wall_e_sheath_climb",
    "cathode_e_emitted_enthalpy",
    "cathode_e_emitted_fall",
    "cathode_e_collected_climb",
)

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
    "end_wall_e_sheath_climb":
        ("BOTH",
         "END-FACE SHEATH CLOSURE (Ee only): the sheath fall the end wall's "
         "collected electrons climbed, taken from the electron store and "
         "handed to the ions. With the 2 Te of characteristic_boundary the "
         "end wall debit is the sheath-edge (2 + Lambda_eff) Te per "
         "collected electron. Present only on a run with "
         "end_wall_sheath_full_debit armed"),
    "cathode_e_emitted_enthalpy":
        ("BOTH",
         "END-FACE SHEATH CLOSURE (Ee only), HEATING: the 2 k_B T_s the "
         "released electrons carry into the plasma off the emitting surface, "
         "at the space-charge-released current. BOTH, not DRIVE-ONLY: the "
         "emitting surface is still hot and still releasing current into the "
         "floating afterglow, so this row runs in both windows by "
         "construction. Present only on a run with cathode_face_full_debit "
         "armed"),
    "cathode_e_emitted_fall":
        ("AFTERGLOW-ACTIVE",
         "END-FACE SHEATH CLOSURE (Ee only), HEATING: the part of the "
         "cathode fall the released electrons drop through that the beam "
         "row does not already carry. AFTERGLOW-ACTIVE BY CONSTRUCTION: it "
         "is identically zero while phi_c_minus = 0, so it is exactly zero "
         "through the drive and nonzero only in the virtual-cathode regime "
         "the afterglow reaches. Present only on a run with "
         "cathode_face_full_debit armed"),
    "cathode_e_collected_climb":
        ("AFTERGLOW-ACTIVE",
         "END-FACE SHEATH CLOSURE (Ee only), COOLING: the barrier the "
         "returning plasma electrons climbed at the cathode, charged to "
         "their own store -- the anode's plasma-pays convention at the other "
         "electrode. AFTERGLOW-ACTIVE: the discharge-phase cathode sheath "
         "repels plasma electrons, so the returning current is microamps "
         "there and the row is negligible against the drive-phase terms; it "
         "carries real power only once the barrier collapses. Present only "
         "on a run with cathode_face_full_debit armed"),
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
    "parallel_momentum_sink":
        ("BOTH",
         "the imposed parallel momentum sink of a RESPONSE-MAP arm: it "
         "carries no energy row of its own, so it appears here only when "
         "an artifact records it, and its power is the heating row below"),
    "parallel_momentum_sink_heating":
        ("BOTH",
         "the frictional work of that imposed sink, deposited whole into "
         "the ion internal energy. NO PHYSICAL OWNER: a run carrying this "
         "row is a response map, not a physical arm, and its ledger is not "
         "comparable channel-for-channel with a stance run's"),
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

#: Root the founding artifact is read from. ``scripts/`` holds code only and
#: every run artifact lives outside the repository under this root (see
#: scripts/README.md, "Run artifacts do not live here"). It is written the way
#: every artifact-reading script here writes it and expanded against the
#: caller's home, so no machine's path is in the source and the root can be
#: pointed elsewhere by pointing HOME elsewhere.
ARTIFACTS_ROOT = "~/bapsf/artifacts"

#: Founding drive-window numbers, measured 2026-08-19 on the artifact named
#: below at the registered drive window, keyed by the quantity the --selftest
#: mode re-measures and carried as (name, reference, significant figures
#: compared).  Four figures where the founding read gave four; the cathode
#: line is held to three.
#:
#: The artifact path is RELATIVE to :data:`ARTIFACTS_ROOT`: this run predates
#: the artifacts rehome, so it sits in the collected loose set, which is where
#: a pointer of the older ``scripts/<artifact>`` form resolves.
SELFTEST_ARTIFACT = "scripts_loose_2026-09-03/runs/g1a_foot45_cr6p94.h5"
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
    # Channel names are resolved through the retired-label table on read, so
    # a pre-rename artifact reports its rows under their current names.
    stored = sorted(f["rhs_terms"].keys())
    channels = [current_label(channel) for channel in stored]
    table = {}
    for channel, name_in_file in zip(channels, stored):
        for name in ENERGY_ROWS:
            key = f"rhs_terms/{name_in_file}/{name}"
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
    """NaN-aware window mean of a scalar cathode diagnostic.

    ``nan`` when the artifact carries no such row, and when every frame in
    the window is ``nan`` -- both mean "not measured here" and neither is
    zero. Otherwise the mean over the FINITE frames only.

    NaN frames are not an anomaly in this export, they are the convention: a
    frame with no cathode solve carries NaN across the whole block, and a
    prescribed arm's frames carry NaN in the rows only the calibrated
    emission model computes. A plain mean therefore returned ``nan`` for a
    window that straddles a phase boundary or a drive hand-off, which is
    every window worth reporting on such a run. The count of frames each
    mean was taken over is in the regime line of the window header.
    """
    if dg is None or key not in dg:
        return float("nan")
    values = np.asarray(dg[key][:][mask], dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def regime_counts(dg, mask):
    """Return ``(prescribed, floating, total)`` frame counts in the window.

    Read off ``source_regime``, the tag the cathode solve itself returns:
    ``"prescribed"`` is the measured-drive solve, and every other tag --
    including ``"none"`` on a frame with no solve -- is counted as floating,
    i.e. as a frame the model's own circuit set the drive on. A run that
    hands off part-way carries both, which is exactly what makes a bare
    window mean of a cathode row ambiguous. ``(0, 0, 0)`` where the artifact
    has no such row.
    """
    if dg is None or "source_regime" not in dg:
        return 0, 0, 0
    tags = dg["source_regime"][:][mask]
    tags = [t.decode() if isinstance(t, bytes) else str(t) for t in tags]
    prescribed = sum(1 for t in tags if t == "prescribed")
    return prescribed, len(tags) - prescribed, len(tags)


def clamped_counts(dg, mask):
    """Return ``(clamped, total)`` cathode-clamped save counts in the window.

    A cathode solve whose root sits above the composed ceiling is clamped to
    the ceiling and tagged ``regime = "capability_limited"``, and nothing
    raises -- so a window that spent frames on the ceiling looks exactly like
    one that did not unless the frames are counted. Read off ``source_regime``.

    THIS ROW AND THE RUN-LEVEL CENSUS COUNT DIFFERENT POPULATIONS. The
    predicate is the same tag, but this row counts SAVES whose saved
    ``source_regime`` reads ``capability_limited``, while the run-level census
    (the ``cathode_clamped_solves`` / ``cathode_total_solves`` root
    attributes) counts accepted SOLVES. A run performs many solves between two
    saves, so a clamp that lasts microseconds falls between save frames and is
    invisible here while the census sees it: measured on a reference probe
    run, the census counts tens of clamped solves out of thousands while this
    row reads ZERO clamped saves. That is why both are reported, and it is why
    a 0-of-N row is NOT evidence that the run never clamped -- the root
    attributes are the evidence for that, and this row is only about the
    frames a reader can actually inspect.

    ``(None, None)`` where the artifact carries no such row: a file written
    before the census existed still loads and still reports every other row,
    and the caller prints "n/a" rather than a zero that would read as "the
    clamp never fired here".
    """
    if dg is None or "source_regime" not in dg:
        return None, None
    tags = dg["source_regime"][:][mask]
    tags = [t.decode() if isinstance(t, bytes) else str(t) for t in tags]
    clamped = sum(1 for t in tags if t == "capability_limited")
    return clamped, len(tags)


def dvm_flow_power(f, key, i0, i1, dt_s):
    """Window-mean power [W] of a per-frame DVM ledger FLOW row [erg].

    The rows of ``dvm_particle_ledger`` are per-tick quantities summed over
    the ticks each save frame covers, so the energy delivered BETWEEN frames
    ``i0`` and ``i1`` is the sum over frames ``i0 + 1 .. i1`` -- frame
    ``i0``'s own row covers ticks before the window opened and is excluded.
    ``nan`` where the artifact carries no such row, which on a moment run is
    every one of them.
    """
    path = f"dvm_particle_ledger/{key}"
    if path not in f:
        # A pre-rename artifact stores this row under its retired name.
        old_key = LEGACY_LABELS_BY_CURRENT.get(key)
        path = f"dvm_particle_ledger/{old_key}" if old_key else path
    if path not in f:
        return float("nan")
    return float(np.sum(f[path][i0 + 1:i1 + 1])) / ERG_PER_J / dt_s


def counter_slope(dg, key, i0, i1, dt_s):
    """Endpoint slope [W] of a cumulative energy counter over the window."""
    if dg is None or key not in dg:
        return float("nan")
    d = dg[key][:]
    return (float(d[i1]) - float(d[i0])) / dt_s


def projection_dropped_W(dg, i0, i1, dt_s):
    """Window dropped-energy rate [W] of the over-wall projection counter.

    PRESENCE-GATED on ``circuit_projection_energy_J``: returns ``None`` on an
    artifact that carries no such row (the run was not armed with
    ``cathode_circuit_project_over_wall``) -- never ``0.0``, which would read
    as "armed and idle" rather than "not computed". Otherwise the endpoint
    slope (:func:`counter_slope`) of the cumulative counter over the window,
    the same convention as the WARMING COUNTER SLOPES block.

    SIGN: positive = energy LEAVING the circuit loop. The counter accumulates
    ``0.5 * L * (I_held**2 - I_root**2)`` at each projection event, where
    ``I_root`` is a bisection root bracketed below ``I_held`` (never above
    it), so the increment -- and this rate -- is non-negative by
    construction. That matches every other row in the circuit block: each
    ``source_P_*`` sink is likewise a positive power flowing OUT of the loop
    into its named channel, and this row is that same kind of sink, named for
    the inductor energy a projection event discards rather than hands to any
    plasma channel. Without a named row for it, the loop's I.V input and its
    source_P_* sinks would carry that energy as an unexplained gap.
    """
    if dg is None or "circuit_projection_energy_J" not in dg:
        return None
    return counter_slope(dg, "circuit_projection_energy_J", i0, i1, dt_s)


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
    n_prescribed, n_floating, n_regime = regime_counts(dg, mask)
    if n_regime:
        print(f"  regime frames: {n_prescribed} prescribed / "
              f"{n_floating} floating (from source_regime)")
    n_clamped, n_clamp_frames = clamped_counts(dg, mask)
    if n_clamped is None:
        print("  cathode clamp frames: n/a -- this artifact carries no "
              "source_regime row")
    else:
        share = n_clamped / n_clamp_frames if n_clamp_frames else float("nan")
        print(f"  cathode clamp frames: {n_clamped} of {n_clamp_frames} "
              f"saves ({share:.4f}) at the composed ceiling "
              "(source_regime == capability_limited)")
        if not n_clamped:
            print("    SAVES, not solves: a clamp between two save frames is "
                  "invisible here -- the run-level count is the file's")
            print("    cathode_clamped_solves / cathode_total_solves root "
                  "attributes, and a zero here is not evidence of none")
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

    print("\n--- END-FACE SHEATH CLOSURE (end_wall_sheath_full_debit, "
          "cathode_face_full_debit) [kW], window mean ---")
    if not table:
        print("  n/a -- rhs_terms ABSENT from this artifact")
    else:
        present = [r for r in END_SHEATH_ROWS if (r, "Ee") in table]
        if not present:
            print("  n/a -- this run armed neither end-face key (the four "
                  "rows are absent, which is not the same as zero)")
        else:
            missing = [r for r in END_SHEATH_ROWS if r not in present]
            total = 0.0
            for row in present:
                value = table[(row, "Ee")]
                total += value
                print(f"{'  ' + row:<44}{value:>16.5f}  kW")
            print(f"{'  NET (all four rows)':<44}{total:>16.5f}  kW")
            if missing:
                print(f"  NB rows absent from this artifact: {missing} -- the "
                      "net above is over the rows present, not the closure "
                      "(the two end-face keys arm independently, so a "
                      "one-key run is missing the other key's rows by "
                      "construction)")
            print("  the end wall row is the sheath fall its collected "
                  "electrons climbed; read it WITH characteristic_boundary,\n"
                  "  which carries the same face's 2 Te. The three cathode "
                  "rows are the emitting face's own channels and are\n"
                  "  additional to cathode_surface_loss, not a re-cut of it.")

    # The emitted enthalpy's OTHER placement (cathode_enthalpy_on_beam). This
    # is NOT an rhs_terms row and must not be added into the net above: it is
    # the part of beam_power_deposition that IS the emitted electrons' launch
    # enthalpy, carried by the beam wherever the released electrons are the
    # primary beam. It is the reading that explains a zero
    # cathode_e_emitted_enthalpy row on an armed run -- the enthalpy did not
    # vanish, it moved from the cathode-adjacent cell into the column.
    # ABSENT (not zero) on a run that did not arm the key.
    if dg is not None and "source_P_emitted_enthalpy_on_beam" in dg:
        on_beam = diagnostic_mean(
            dg, "source_P_emitted_enthalpy_on_beam", mask
        )
        launch_V = diagnostic_mean(dg, "source_beam_launch_enthalpy_V", mask)
        print()
        print("--- EMITTED ENTHALPY CARRIED ON THE BEAM "
              "(cathode_enthalpy_on_beam), window mean ---")
        print(f"{'  cathode_e_emitted_enthalpy_on_beam':<44}"
              f"{on_beam / 1e3:>16.5f}  kW")
        print(f"{'  beam launch enthalpy':<44}{launch_V:>16.5f}  V")
        print("  rides beam_power_deposition, NOT the end-face net above; it "
              "is the full 2 k_B T_s\n"
              "  at the emitted current the march launches, and it is zero on "
              "a frame whose solve\n"
              "  kept the enthalpy on the cathode-adjacent cell (a virtual "
              "cathode, or no beam).")

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
        # Over-wall projection loss, PRESENCE-GATED on
        # cathode_circuit_project_over_wall -- see projection_dropped_W.
        # A LEDGER ROW, not a residual: the dropped inductor energy is named
        # here so the loop's I.V input and its source_P_* sinks above do not
        # carry it as an unexplained gap.
        dropped_W = projection_dropped_W(dg, i0, i1, dt_s)
        if dropped_W is not None:
            print(f"{'  circuit_projection_dropped':<44}"
                  f"{dropped_W / 1e3:>16.5f}  kW")
            if label == "DRIVE":
                # PRE-REGISTERED clause (the O6-B3 advisor read): the
                # plateau-mean dropped rate must sit under 0.1% of the
                # plateau-mean primary-beam power -- a genuine loss, not a
                # material share of what the drive puts in. Gated to the
                # DRIVE window alone: "plateau" names that window in this
                # script (see the module docstring), and the AFTERGLOW
                # window's source_P_prim is ~0 by construction, which would
                # make the ratio meaningless there.
                P_prim_W = diagnostic_mean(dg, "source_P_prim", mask)
                if np.isfinite(P_prim_W) and P_prim_W != 0.0:
                    share_pct = abs(dropped_W) / P_prim_W * 100.0
                    ok = share_pct < 0.1
                    print(f"    [{'PASS' if ok else 'FAIL'}] PRE-REGISTERED "
                          "clause: plateau-mean circuit_projection_dropped "
                          f"< 0.1% of plateau-mean source_P_prim "
                          f"({share_pct:.4f}%)")
                else:
                    print("    clause n/a -- plateau-mean source_P_prim is "
                          "zero or not finite in this window")
        if n_prescribed:
            # WHAT THE RESIDUAL IS on a prescribed frame. Nothing is
            # re-booked here: the row is what the circuit already computed
            # and this only says what it measures, because on the
            # current-driven solve the same row is a closure check that is
            # ~0 by construction and a reader carries that reading across.
            #
            # The prescribed mode books the emitted current as
            # I_eth* = max(I - I_i, 0) with no returning-electron term, so
            # the ledger it is checked against is short exactly that term.
            # ALWAYS: source_P_load_residual = -phi_c *
            # source_I_cathode_kirchhoff_residual. WHERE THE FLOOR IS
            # INACTIVE (I > I_i, the operating regime of a real rung) that
            # is the returning-electron field work and equals
            # source_P_cathode_e_phi above. Where the measured current sits
            # BELOW the Bohm ion current the floor binds, the Kirchhoff
            # residual carries the whole shortfall instead, and the row is
            # orders larger than the field work -- so the two are printed
            # together rather than one being called the other.
            print("  NOTE prescribed frames in this window: "
                  "source_P_load_residual is the returning-electron field "
                  "work\n  (= source_P_cathode_e_phi above) wherever the "
                  "emitted-current floor I_eth* = max(I - I_i, 0) is\n  "
                  "inactive; in general it is -phi_c * "
                  "source_I_cathode_kirchhoff_residual, which is how a "
                  "floored\n  frame shows up. Compare the two rows above "
                  "before reading it as the field work.")

        print("\n--- WARMING COUNTER SLOPES over the window ---")
        for key in sorted(k for k in dg.keys()
                          if k.startswith("warming_E_") and k.endswith("_J")):
            print(f"{'  ' + key:<44}"
                  f"{counter_slope(dg, key, i0, i1, dt_s) / 1e3:>16.5f}  kW")

    print("\n--- KINETIC JET INJECTION over the window ---")
    if "dvm_particle_ledger" not in f:
        print("  dvm_particle_ledger ABSENT from this artifact -- a moment "
              "run launches no kinetic jet")
    else:
        # The energy the end wall jet put INTO the gas: the atoms it launched
        # times the launch energy each carried, as the engine booked it at the
        # birth site. It is not an rhs_terms channel -- the jet is a neutral
        # birth, not a fluid source row -- so it is invisible in the channel
        # table above and has to be read from the jet's own ledger.
        for jet_key, jet_label in (
            ("energy_birth_end_wall_jet", "end wall jet injected"),
            ("energy_birth_cathode_jet", "cathode jet injected"),
            ("energy_birth_anode_jet", "anode jet injected"),
        ):
            print(f"{'  ' + jet_label:<44}"
                  f"{dvm_flow_power(f, jet_key, i0, i1, dt_s) / 1e3:>16.5f}"
                  "  kW")

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
        key = f"rhs_terms/{LEGACY_LABELS_BY_CURRENT.get(channel, channel)}/Ee"
        if key not in f:
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
        "roles": [current_label(r.decode() if isinstance(r, bytes) else str(r))
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
    # The kinetic arm's own jet selectors, which the stance line above does
    # not carry: two arms can share every fluid setting and differ only in
    # which surfaces launch an energetic recycle stream, and the injection
    # rows below are meaningless without knowing which ones were armed.
    print(f"dvm jets : neutral_model={params.get('neutral_model')}, "
          f"cathode={params.get('neutral_kinetic_dvm_cathode_jet')}, "
          f"anode={params.get('neutral_kinetic_dvm_anode_jet')}, "
          f"end_wall="
          f"{legacy_get(params, 'neutral_kinetic_dvm_end_wall_jet')}")
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


def selftest_artifact_path():
    """Absolute path of the founding --selftest artifact.

    :data:`SELFTEST_ARTIFACT` under :data:`ARTIFACTS_ROOT`, with the root
    expanded against the caller's home. Returns the path whether or not a
    file is there; the caller reports its absence.
    """
    return os.path.join(os.path.expanduser(ARTIFACTS_ROOT), SELFTEST_ARTIFACT)


def selftest(path):
    """Re-measure the founding drive-window numbers and hard-assert them."""
    with h5py.File(path, "r") as f:
        geom = load_geometry(f)
        lo, hi = SELFTEST_WINDOW_MS
        _, i0, i1 = window_frames(geom["t_ms"], lo, hi)
        dt_s = (geom["t_ms"][i1] - geom["t_ms"][i0]) * 1e-3
        table, _ = integrate_rows(f, i0, i1, geom["vols"])
        dg = f["cathode_diagnostics"]
        # PRESENCE-GATE check for circuit_projection_dropped: the founding
        # artifact predates cathode_circuit_project_over_wall and carries no
        # circuit_projection_energy_J row, so this is the one artifact on
        # hand that can exercise the absent branch -- projection_dropped_W
        # must read as ABSENT (None), never a measured 0.0 that would claim
        # the run was armed and idle.
        assert "circuit_projection_energy_J" not in dg, (
            "power ledger selftest: the founding artifact now carries "
            "circuit_projection_energy_J -- the presence-gate check below "
            "needs an unarmed artifact to exercise the absent branch"
        )
        assert projection_dropped_W(dg, i0, i1, dt_s) is None, (
            "power ledger selftest: circuit_projection_dropped must read "
            "ABSENT (None) on an artifact without circuit_projection_energy_J"
        )
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
    print(f"{'circuit_projection_dropped presence-gate':<52}"
          f"{'ABSENT (unarmed artifact)':>18}")
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
                             "which otherwise reads the founding artifact "
                             f"{SELFTEST_ARTIFACT} under the artifacts root "
                             f"{ARTIFACTS_ROOT}")
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
        path = args.run
        if path is None:
            path = selftest_artifact_path()
            if not os.path.exists(path):
                parser.error(
                    f"--selftest found no artifact at {path}. That path is "
                    f"the founding artifact {SELFTEST_ARTIFACT} resolved "
                    f"under the artifacts root {ARTIFACTS_ROOT}, which is "
                    "where run artifacts live -- this script's own directory "
                    "holds code only, so there is nothing to fall back to "
                    "beside it. Point at another copy by naming it: "
                    "power_ledger_sim1d.py --selftest RUN.h5"
                )
        selftest(path)
        return
    if args.run is None:
        parser.error("a run artifact is required (or use --selftest)")
    run_report(args.run, tuple(args.drive), tuple(args.afterglow),
               args.port_top)


if __name__ == "__main__":
    main()
