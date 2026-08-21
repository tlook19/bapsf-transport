"""Cached neutral-equilibration seed (R5 ES1 tuning pass, 2026-07-26).

The optional neutral pre-equilibration (``neutral_equilibration`` flag) runs a
100-cycle puff/off neutral-only accumulation before the plasma launch, and injects
ONLY the equilibrated neutral density profile into the plasma run's initial state
(``_apply_neutral_equilibration_result`` seeds ``nn``/``nn_a``; n, M, Ee, Ei, M_n
stay at the fresh IC). That equilibration is deterministic given the neutral-flow
configuration and costs ~1 minute every run, so its result can be cached and
reused until the neutral-flow configuration changes.

Tom's rule (2026-07-26): re-equilibrate when — and only when — puffing, pumping,
or any machine geometry that affects neutral flow changes; also when switching to
a mode with neutral momentum, kinetic neutrals, or two zones. Circuit, cathode
power-balance, atomic-rate, and plasma-numerics changes must NOT invalidate the
seed (so the ES1 circuit/cathode refit iteration keeps the cached seed).

The signature is FAIL-CLOSED: it hashes the ENTIRE (params, flags) config EXCEPT an
explicit allowlist of keys that are provably inert to a neutral-only equilibration
(which runs with ``Plasma=False`` and ``cathode_coupling=False``). Any key not on
the allowlist — including any newly added key — invalidates the cache by default.
This makes it impossible to silently reuse a stale seed after adding a new
neutral-affecting knob: the cost of a missed categorisation is an unnecessary
re-equilibration (safe), never a wrong seed.

A miss (missing file) or a signature mismatch raises a loud ``ValueError`` naming
the cause and the rebuild command (house rule: fail loud on misconfiguration, no
silent fallback). Rebuild with ``scripts/build_neutral_seed_cache.py``.
"""

import hashlib
import json

import numpy as np

# v2 (item 37 duty fix, 2026-07-29): EVERY v1 seed was equilibrated with the
# defective puff duty -- the phase lookup and the phase-boundary schedule
# disagreed about the puff-off instant, so the puff ran one extra step in some
# cycles and the seed records an over-fuelled fill (measured +12.0% delivered
# ON-time at the production 20 ms schedule). A v1 seed replayed under the fixed
# solver would silently reproduce the OLD duty, so the format is bumped: v1
# files are rejected (fixed-path mode raises, database mode re-equilibrates and
# overwrites) rather than reused.
CACHE_FORMAT = "sim1d-neutral-seed-v2"

# Keys PROVABLY inert to a neutral-only equilibration (Plasma=False,
# cathode_coupling=False): circuit, cathode/beam/emission/surface, atomic-rate
# and cooling scales, plasma initial condition, plasma-run numerics, plasma-phase
# timing, and output cadence. Everything ELSE (all geometry, puffing, pumping,
# neutral transport/IC, neutral-momentum/two-zone/kinetic selectors, and the
# equilibration schedule) stays in the signature. Keep this list conservative:
# when in doubt, leave a key OUT of this set so it stays in the hash.
INERT_PARAM_KEYS = frozenset({
    # --- discharge circuit ---
    "V_bank", "R_comp", "R_comp_partition", "R_mesh_ohm", "L_parasitic_H",
    "C_bank_F", "C_R", "eta", "I_prebreakdown", "I_breakdown",
    # --- plasma-phase timing (NOT the equilibration schedule) ---
    # tau_discharge was on this list and should NOT have been (item 37,
    # 2026-07-29): it IS the equilibration's per-cycle puff-ON window whenever
    # equilibration_gas_puff_on_s is unset, so two tau_discharge families
    # equilibrate to DIFFERENT fills while hashing to the same signature --
    # exactly the silent stale-seed reuse this signature exists to prevent.
    # It stays in the hash unconditionally (fail-closed: when in doubt, leave
    # a key OUT of this set); the cost is one unnecessary re-equilibration for
    # configs that do override the window. Its override,
    # equilibration_gas_puff_on_s, is likewise absent from this list and so is
    # hashed -- setting it MUST re-key, it changes the fill directly.
    "tau_prebreakdown", "tau_breakdown", "tau_afterglow",
    "tau_neutral_prebreakdown", "phase_transition_mode",
    # --- cathode / emission / warming / surface (no cathode during equil) ---
    "T_s", "L_cath", "R_cath", "phi_wf",
    "cathode_Ts_base_K", "cathode_Ts_fwhm_cm",
    "cathode_cleaning_E_th_eV", "cathode_cleaning_sigma_cm2",
    "cathode_conduction_W_per_K", "cathode_emission_annuli",
    "cathode_emission_profile", "cathode_emissivity",
    "cathode_heat_capacity_J_per_K", "cathode_model", "cathode_phi_c_cap_V",
    "cathode_phiwf_clean_eV", "cathode_sample_smoothing",
    "cathode_solver_model", "cathode_surface_model", "cathode_warming_model",
    "cathode_Rp_model",
    # --- beam deposition / excitation (no plasma/beam during equil) ---
    "beam_anomalous_model", "ql_relaxation_coeff",
    "beam_coulomb_model", "beam_deposition_model",
    "beam_excitation_energy_eV", "beam_excitation_model", "b_beam_excitation",
    # --- atomic-rate / cooling / plasma-physics scales (no plasma during equil) ---
    "atomic_rate_model", "b_ioniz", "b_rec_rad", "b_rec_3b", "b_Qie", "b_Qei",
    "b_Qen", "b_Qcx", "b_Qei_Te_exp", "b_Qen_Te_exp", "b_Q_Te_ref_eV",
    "b_epara", "b_ipara", "recombination_energy_return",
    "sigma_in_model", "sigma_in_cm2", "ionization_birth_energy_model",
    "Te_birth_ionization", "Ti_birth_ionization",
    "b_ion_neutral_drag", "b_ion_neutral_thermalization", "b_slip_entrainment",
    "ion_neutral_drag_model", "D_amb", "D_amb_model", "heat_flux_limiter_f",
    "b_presheath_length", "b_anode_collection", "b_anode_advective_block",
    "b_surface_loss", "alpha_isat", "alpha_front", "front_flux_model",
    "adas_low_te_extension",
    # --- plasma initial condition + plasma floors ---
    "ne0", "Te0", "Ti0", "u0", "ne_floor", "Te_floor", "Ti_floor",
    # nn0 is the DIRECT-RUN neutral fill. run_neutral_equilibration pins its
    # inner sim's start at 1e8 regardless of it, so nn0 cannot reach a seed and
    # cannot change one. This is the one neutral-side key that is provably
    # inert; every other neutral knob stays in the hash.
    #
    # Its shaped counterparts nn0_profile / nn0_annulus_profile are
    # deliberately NOT listed here, and neither is the neutral_initial_profile
    # flag in INERT_FLAG_KEYS. They are inert for a stronger reason than nn0
    # is -- the flag REFUSES neutral_equilibration at construction, so an
    # armed profile can never reach this cache at all -- but the fail-closed
    # rule above says a key leaves the hash only when it must, and these three
    # sit at their None/False defaults on every config that can be cached.
    # They contribute a constant to the signature, which rotates the hash once
    # (a cold recompute, bit-exact results) and is stable thereafter.
    "nn0",
    # The prescribed per-cell geometry keys (plasma_radius_profile_cm,
    # machine_radius_profile_cm, plasma_area_max_vessel_fraction,
    # neutral_annulus_volume_fraction_min) are deliberately NOT listed here,
    # and neither is prescribed_area_geometry in INERT_FLAG_KEYS -- unlike
    # end_recycle_to_annulus below, which IS exempt. That flag changes only two
    # plasma boundary terms an equilibration never evaluates; these change the
    # GEOMETRY. The column volume Vp, the vessel volume Vm, the annulus volume
    # V_ann = Vm - Vp, the zone exchange conductance (~ Rp dz) and the
    # free-molecular face conductances (~ the hydraulic radius) are all read by
    # the neutral-only equilibration, whose whole content is where the gas
    # settles, so a seed equilibrated on one machine is simply wrong for
    # another. They must re-key, and being absent from this allowlist is what
    # makes them re-key -- the fail-closed rule at the top, working as
    # intended.
    # --- plasma-run numerics (equil uses fixed neutral_equilibration_dt) ---
    "cfl", "density_dt_fraction", "drag_dt_fraction", "heat_dt_fraction",
    "implicit_heat_scheme", "operator_splitting", "heat_picard_iterations",
    "heat_picard_tol", "ln_lambda_min", "max_density_step_fraction",
    "max_energy_step_fraction", "circuit_picard_max_iter", "circuit_picard_tol_rel",
    "hyperbolic_wave_speed", "dt_growth_enabled", "dt_growth_factor",
    "dt_reject_factor", "adaptive_retries_enabled",
    # --- output cadence (run_neutral_equilibration overrides these) ---
    "dt_save", "t_save_start", "max_output_steps",
    # --- the cache path itself is not seed content ---
    "neutral_seed_cache_dir",
})

INERT_FLAG_KEYS = frozenset({
    # plasma / cathode / circuit / numerics toggles inert to neutral-only equil,
    # plus the cache-control flags themselves (they select the seed source, not
    # its content).
    "Plasma", "cathode_coupling", "active_plasma_topology",
    "beam_anode_interception", "cathode_emission_bridge", "cathode_schottky",
    "characteristic_boundary", "coupled_circuit_picard", "cx",
    "electron_heat_flux_limit", "heat_conduction", "hyperbolic_energy_consistent",
    "icool", "icool_recomb", "implicit_heat_conduction", "ion_neutral_drag",
    "ion_neutral_drag_cx_only", "ion_neutral_moment_closure",
    "ion_neutral_thermalization", "ionization_energy_cost", "ncool",
    "raw_stage_validation",
    "debug_checks",
    # end_recycle_to_annulus changes ONLY the two plasma-terminating boundary
    # terms (boundary_absorption / characteristic_boundary), and an
    # equilibration cannot reach either. run_neutral_equilibration pins
    # Plasma=False on its inner sim, and with Plasma off rhs_terms takes the
    # neutral-only branch, which returns _zero_rhs_state() for BOTH of those
    # terms; the implicit neutral-only stepper that actually advances that
    # phase assembles exchange, pump and puff alone and never calls them. No
    # plasma => no boundary recycle => nothing for the routing to route. This
    # is the same argument that already exempts characteristic_boundary above,
    # and it is what keeps a default-config flag addition from rotating every
    # cached seed in the database.
    "end_recycle_to_annulus",
    # cache-control + equilibration-trigger flags (not seed content)
    "neutral_equilibration", "launch_plasma_after_equilibration",
    "use_cached_neutral_seed",
})


def _canonical(value):
    """Return a JSON-serialisable, hash-stable form of a config value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # repr round-trips float exactly and is stable across platforms.
        return repr(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_canonical(v) for v in np.asarray(value).tolist()]
    if isinstance(value, (np.floating,)):
        return repr(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    return repr(value)


def _signature_payload(params, flags):
    kept_params = {
        k: _canonical(v) for k, v in params.items() if k not in INERT_PARAM_KEYS
    }
    kept_flags = {
        k: _canonical(v) for k, v in flags.items() if k not in INERT_FLAG_KEYS
    }
    return {"params": kept_params, "flags": kept_flags}


def neutral_seed_signature(params, flags):
    """Return a stable hex signature over the neutral-flow configuration."""
    payload = _signature_payload(params, flags)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def save_neutral_seed(path, nn, nn_a, params, flags, meta=None):
    """Write the equilibrated neutral seed and its signature to ``path`` (.npz)."""
    signature = neutral_seed_signature(params, flags)
    payload = _signature_payload(params, flags)
    meta_json = json.dumps(meta or {}, sort_keys=True)
    np.savez(
        path,
        format=CACHE_FORMAT,
        signature=signature,
        signature_payload=json.dumps(payload, sort_keys=True),
        meta=meta_json,
        nn=np.asarray(nn, dtype=float),
        nn_a=(np.asarray([]) if nn_a is None else np.asarray(nn_a, dtype=float)),
        has_nn_a=bool(nn_a is not None),
    )
    return signature


def load_neutral_seed(path, params, flags, expected_cells=None):
    """Load and validate the cached seed; raise a loud ValueError on any miss.

    Returns ``(nn, nn_a_or_None)``.
    """
    import os

    rebuild = (
        "Rebuild it for this neutral-flow configuration with:\n"
        "  python scripts/build_neutral_seed_cache.py --out "
        f"{path}\n"
        "(re-equilibration is required whenever puffing, pumping, neutral-flow "
        "geometry, or the neutral-momentum/two-zone/kinetic mode changes)."
    )
    if not os.path.exists(path):
        raise ValueError(
            f"neutral seed cache not found: {path}\n{rebuild}"
        )
    want = neutral_seed_signature(params, flags)
    with np.load(path, allow_pickle=False) as data:
        fmt = str(data["format"])
        if fmt != CACHE_FORMAT:
            raise ValueError(
                f"neutral seed cache {path} has format {fmt!r}, "
                f"expected {CACHE_FORMAT!r}.\n{rebuild}"
            )
        have = str(data["signature"])
        if have != want:
            raise ValueError(
                f"neutral seed cache {path} is STALE for this configuration: "
                f"stored signature {have[:16]}... != current {want[:16]}...\n"
                "The neutral-flow configuration (puffing / pumping / neutral-flow "
                "geometry / neutral-momentum / two-zone / kinetic mode) changed.\n"
                f"{rebuild}"
            )
        nn = np.asarray(data["nn"], dtype=float).copy()
        has_nn_a = bool(data["has_nn_a"])
        nn_a = np.asarray(data["nn_a"], dtype=float).copy() if has_nn_a else None
    if expected_cells is not None and nn.shape[0] != int(expected_cells):
        raise ValueError(
            f"neutral seed cache {path} has {nn.shape[0]} cells, geometry has "
            f"{int(expected_cells)} (nx / geometry changed).\n{rebuild}"
        )
    return nn, nn_a


# --- Database mode: a directory of seeds keyed by the neutral-flow signature ---
# Each distinct neutral-flow configuration (machine geometry / puffing / pumping /
# neutral physics) gets its own entry, auto-populated on first use. A "miss" is a
# new config (a new fill rate to record), NOT a misconfiguration, so it builds and
# stores rather than raising -- the directory becomes a browsable fill-rate table.

def seed_db_path(cache_dir, params, flags):
    """Return the database file path for this config's neutral-flow signature."""
    import os

    sig = neutral_seed_signature(params, flags)
    return os.path.join(str(cache_dir), f"neutral_seed_{sig[:16]}.npz")


def try_load_neutral_seed(path, params, flags, expected_cells=None):
    """Return ``(nn, nn_a)`` if a valid seed is at ``path``, else ``None``.

    Non-raising database lookup: a missing file, format/signature mismatch, or
    cell-count mismatch all return ``None`` so the caller can (re-)build and store.
    """
    import os

    if not os.path.exists(path):
        return None
    want = neutral_seed_signature(params, flags)
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["format"]) != CACHE_FORMAT:
                return None
            if str(data["signature"]) != want:
                return None
            nn = np.asarray(data["nn"], dtype=float).copy()
            has_nn_a = bool(data["has_nn_a"])
            nn_a = (
                np.asarray(data["nn_a"], dtype=float).copy() if has_nn_a else None
            )
    except (OSError, ValueError, KeyError):
        return None
    if expected_cells is not None and nn.shape[0] != int(expected_cells):
        return None
    return nn, nn_a


def fill_rate_meta(params, nn):
    """Fill-rate summary stored with a DB entry (for the browsable table)."""
    nn = np.asarray(nn, dtype=float)
    keys = (
        "S_gp", "Twin_S_gp", "gas_puff_mode", "gas_puff_profile", "S_pump_L",
        # nn0 is deliberately absent: it is the direct-run fill, not the
        # equilibration's start (which is pinned at 1e8), so recording it here
        # would mislabel the entry's provenance.
        "S_pump_R", "gas_type", "Tn_K", "nx",
        "neutral_equilibration_cycles",
    )
    meta = {k: params.get(k) for k in keys}
    meta.update(
        cells=int(nn.shape[0]),
        mean_nn=float(np.mean(nn)),
        min_nn=float(np.min(nn)),
        max_nn=float(np.max(nn)),
    )
    return meta
