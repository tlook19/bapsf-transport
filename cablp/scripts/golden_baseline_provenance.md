# Provenance of the golden baseline pins (`baseline_sim1d.BASELINE_*_OVERRIDES`)

`scripts/baseline_sim1d.py` builds the configuration behind the committed
regression fixture `scripts/baselines/production_discharge.npz`. It layers its
own pins on top of the production stance it imports, so this note only covers
what it changes. For the imported layer see
`scripts/production_stance_provenance.md`; for parameter meanings see the
docstrings in `cablp/solvers/_sim1d/core/config.py`, and for the shipped
defaults see `cablp/solvers/_sim1d/core/config_defaults_provenance.md`.

## What kind of configuration this is

**The golden is a regression scaffold, not a physical claim.** Almost every pin
below exists for one reason: the committed fixture was captured at a particular
historical configuration, and the anchor must not drift when a live default
moves. These pins are therefore neither MEASURED nor FITTED — they are
*recorded historical settings*, and their provenance is "this is what the
fixture ran at".

Two rules follow, and they are the whole point of the file:

- **The fixture is never recaptured to make a changed default look unchanged.**
  A repaired physical default gets a pin here, not a new NPZ.
- **A physical default must never be quietly changed to make an old result
  reproduce.** If a default moves, the pin records the old value explicitly.

`scripts/baselines/` is off limits to routine work.

## The one pin that is not historical

**`L_parasitic_H = 8.1e-6`.** This fixture has pinned 8.1e-6 since it was
captured, which is why it is bit-exact across the circuit correction that moved
the production stance from 6.6e-6 to 8.1e-6. It was the pre-regression config
default, and the fixture carried it through the interval in which the config
default had been overwritten to 6.6e-6 by an unrelated hunk. The production
stance now finally matches it. Value class: DERIVED from measurement, bracket
7.6-8.4 uH — see the defaults provenance note for the two instruments.

## Historical pins, by reason

**Resolution.** `nx = 60` — the value the fixture was captured at (the config
default at capture time). The campaign drivers promoted their own default to
240; the gate must not inherit it, because quadrupling the cell count would
multiply every reviewer gate's runtime.

**Historical checkpoint seed.** `Te0 = 0.1`, `Ti0 = 0.1`, `Ti_floor = 0.1`. The
live defaults are deliberately different.

**Pre-repair physics, pinned to their old values** so the trajectory stays
bit-exact while the live defaults carry the repairs:

| key | pinned | live default |
|---|---|---|
| `ionization_birth_energy_model` | `"legacy"` | `"conservative"` |
| `hyperbolic_wave_speed` | `"isothermal"` | `"adiabatic"` |
| `Te_birth_ionization` | `"local"` | (inert under `"conservative"`) |
| `hyperbolic_energy_consistent` | `False` | `True` |
| `characteristic_boundary` | `False` | `True` |
| `ion_neutral_moment_closure` | `False` | `True` |
| `beam_anode_interception` | `False` | `True` |
| `front_flux` | `True` | `False` |
| `active_plasma_topology`, `raw_stage_validation` | `False` | `True` |

**Superseded ion-neutral closure**, pinned because the fixture ran the ad-hoc
drag stance that the Phelps moment operator replaced: `b_ion_neutral_drag = 0.5`
(a FITTED slip constant), `ion_neutral_drag_model = "constant"`,
`sigma_in_model = "cx_derived"`, `b_ion_neutral_thermalization = 1.0`,
`ion_neutral_thermalization = True`.

**Neutral flow and geometry as captured:** `S_pump_L = 2000` (the live default
now matches `S_pump_R`), `gas_puff_profile = "cell"` (live default
`"cosine_pipe"`), `equilibration_gas_puff_on_s = None` (so the equilibration
inherits `tau_discharge`, rather than the measured 25 ms puff width the
production stance adopted), and `Rcs = Lcs = Rsup = 0.0` with
`end_expansion_geometry = False` — the fixture was captured at the plain
geometry defaults, 67 cells. `build_baseline_config` pops the inherited
`end_expansion_*` parameters when the flag is off, because they are
presence-gated and would otherwise raise.

**Anti-drift pins on default-off closures.** `beam_product_transport = "local"`
and `heating_anomalous_transport = "local"` are already the config defaults;
they are pinned anyway so a future promotion to `"nonlocal"` / `"tail_walk"`
cannot silently move the anchor.

## Values the fixture shares with the machine

A few pins are real quantities rather than historical accidents, taken from the
per-setting operating points in `run_mechanism_ladder.ES_OPERATING` (see
`scripts/ladder_operating_provenance.md`):

- `V_bank` — the MEASURED pre-shot open-circuit bank voltage.
- `T_s` and `cathode_Ts_base_K` — the MEASURED standby surface temperature.

The remaining cathode pins (`phi_wf = 2.869`, `cathode_phiwf_clean_eV = 2.809`,
`cathode_cleaning_sigma_cm2 = 3.5e-16`, `cathode_cleaning_E_th_eV = 20.0`,
`cathode_heat_capacity_J_per_K = 120.0`, `cathode_conduction_W_per_K = 1200.0`,
`cathode_emissivity = 0.7`, `S_gp = 3400`, `gas_puff_mode = "square"`,
`cathode_sample_smoothing = "presheath"`, `neutral_exchange_model = "knudsen"`)
are the values the fixture was captured with; their provenance classes are in
`cablp/solvers/_sim1d/core/config_defaults_provenance.md`. Note
`cathode_conduction_W_per_K` here is the capture-time value and differs from the
production stance's fitted one.

`BASELINE_RUN_KWARGS` are all `None`, i.e. the solver's own run defaults:
adaptive dt, dynamic current-trigger end time, unlimited steps.

## Recapture record

**2026-08-09 — returned-root sheath-ceiling fix (AUTHORIZED recapture, Tom,
2026-08-09).** The current-driven sheath solve enforced `cathode_phi_c_cap_V`
only at the bracket ladder's doubling grid points, never on the root it
returned; the fixture's own ignition foot contained 34 such escaped solves
(net phi_c up to 1.9669× the 1000 V cap — 1966.89 V returned at the cap),
i.e. the committed trajectory certified the defect. The fix (commit
`8a09363`, this branch) tests the located J-root against the cap and routes
an at-or-above-cap root through the pre-existing ceiling branch, so the foot
solves move BY DESIGN and both goldens fail against the old fixture with
`max_rel=2.000e+00`, `time_max_abs=1.113e-06 s`, character-identical on the
pure and compiled paths. Recaptured with the script's own
`--capture` at the fixed code. **The pin set is unchanged:** zero
added, removed, or changed keys in `BASELINE_PARAM_OVERRIDES` /
`BASELINE_FLAG_OVERRIDES` (the fix touches neither `baseline_sim1d.py` nor
`core/config.py`), and the sidecar params/flags diff against the previous
capture shows zero changed values — the 18 param keys and 1 flag key newly
recorded are config defaults added since the 2026-08-03 capture, already in
effect for every verify since (verify rebuilds the config from live code),
recorded at their unchanged live defaults. saves stays 2545, cells stays 72;
steps 41054 → 40975 on the corrected foot.
