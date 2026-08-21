# Provenance of the golden baseline pins (`baseline_sim1d.BASELINE_*_OVERRIDES`)

**Recaptured 2026-08-21 (the pre-Tuesday physics batch; the 2026-08-20
stance-update wave and the thread-24 R2b re-anchor onto the stance preceded it
— all three under the reviewed-recapture protocol, see the recapture record
below).** The committed regression fixture
`scripts/baselines/production_discharge.npz` is captured at **the stance of
record, re-cut to the gate mesh** — `default_config()` plus the committed stance
file `scripts/stances/g1atrim.toml`, minus that stance's mesh-sized package,
plus `nx = 60`.

**The previous fixture is retired.** It held the 2026-07-22 operating point
behind ~30 explicit pins and is reproducible only at the anchor tag
`pre-refactor-2026-08-20`, with the environment lockfile recorded against that
tag. Its pin table is in this file's git history; do not reconstruct it here.

`scripts/baseline_sim1d.py` builds the configuration. It imports no campaign
driver — `compare_sim1d_es1` and `run_mechanism_ladder` stay unimported — and
reads the stance through `stance_config`, the loader for that committed
artifact. For parameter meanings see the docstrings in
`cablp/solvers/_sim1d/core/config.py`; for the stance values and their
provenance see `production_stance_provenance.md`; for everything inherited from
the defaults see `cablp/solvers/_sim1d/core/config_defaults_provenance.md`.

## Why the stance, and not the shipped defaults

**The shipped defaults are NOT the production package.** An earlier draft of
this note said they were, on the strength of the R2a/R2b fold-ins. That was
wrong, and the correction matters: those folds moved the *neutral closure
family* and the *measured machine* into `default_config()`, but the **operating
point stayed in the stance file** — the effective Richardson constant, the bank
voltage, the cathode thermal pair, the emission profile, the puff level and its
25 ms equilibration window, the afterglow length.

Captured at bare defaults, the fixture gated an **unrepresentative corner**:
breakdown was marginal, the anode–cathode gap never filled, and the anode sheath
power drained a near-empty flanking cell on a ~100 ns e-fold. The adaptive dt was
consequently pinned near `3e-8 s` by the `surface_loss` limiter and never
recovered, which is why that draft needed a 40,000-step cost cap and could only
reach `t = 1.9e-3 s`. **The stiffness was a symptom of the wrong configuration,
not a property of the model.** On the stance the discharge ignites properly, the
limiter relaxes as the column fills, and the full cycle is affordable.

**The trade this makes, stated plainly: EDITING THE STANCE FILE BREAKS THIS GATE
until the fixture is recaptured.** That is intended — the fixture tracks the
configuration the campaign actually runs. It is not a licence to recapture
casually; a recapture stays a reviewed, authorized, recorded event.

## The re-cut: what could not travel to `nx = 60`

Four stance params are per-cell arrays sized to the stance's own **280-cell**
mesh (1 plenum + 5 gap + 5 fixed source + 268 far column + 1 collector). They
cannot be applied at `nx = 60` (a 72-cell mesh), and they are **not resampled**:

| dropped | why not interpolated |
|---|---|
| `plasma_radius_profile_cm`, `machine_radius_profile_cm` | Built offline by `scripts/g1_build_profiles.py` from a measured field census. The vessel profile is a **staircase** (40 / 50 / 76.2 cm bores plus annulus-equivalent radii over the cathode box); interpolating it would smear the steps into a bore the machine does not have, and the cathode-box annulus areas are exact measured quantities. |
| `nn0_profile`, `nn0_annulus_profile` | An equilibrated 4.5 ms ballistic foot computed **for that mesh**. Resampling changes the neutral inventory and the near-source structure, so the result is a new initial condition, not the stance's. |

The package is dropped **whole**, together with the two flags that require it
(`prescribed_area_geometry`, `neutral_initial_profile`). Half-applying it — a
prescribed geometry carrying a default fill — would be a hybrid corner of exactly
the kind this re-anchor exists to eliminate.

**Everything mesh-independent still travels**, which is every scalar
operating-point key plus the baffles (`neutral_baffle_positions_cm` /
`_clear_radii_cm` are physical cm, not per-cell, so `neutral_baffles` stays on).
Because the shaped foot is gone and the solver refuses a profile and an
equilibration together, `neutral_equilibration` returns to `True` — the
equilibrated seed refills the machine at the stance's own 25 ms puff window,
which is the scalar substitute for the foot and is why the gap fills.

**What the gate therefore does NOT exercise:** the measured flare, the vessel
staircase, and the shaped initial fill. Those live in the campaign arms, not
here. The fixture is not bit-comparable to any campaign arm and was never meant
to be.

## The pin table

| pin | value | why it is run shape, not physics |
|---|---|---|
| `nx` | `60` | Axial resolution of the far column: a pure cost knob. The campaign runs 268; a reviewer pays for this gate on the candidate branch and again post-merge. Pinned rather than inherited so a future default-`nx` change cannot multiply every gate's runtime silently. |
| `max_steps_action` | `"raise"` | Deliberately overrides the stance's `"stop"`. For a campaign arm a step cap is a budget and a truncated arm is still data; here the cap is a tripwire, and tripping it should be loud. |
| `max_steps` (run kwarg) | `150000` | **A tripwire, not a run length** — ~1.9× the measured 80,416 steps. It exists so a change that quietly destroys the timestep fails fast instead of running for hours. If it fires, the question is what happened to `dt`, not what happened to the trajectory. Sized at 2× deliberately: a backstop with a few percent of headroom is not a backstop, it is a second cost cap waiting to truncate the gate. |

`BASELINE_FLAG_OVERRIDES` carries one entry, `neutral_equilibration = True`, for
the reason given in the re-cut section above.

`t_end`, `dt` and `operator_split` are `None`, i.e. the solver's own run
defaults: adaptive dt, the shipped operator split, and the dynamic
current-trigger end time — **which the fixture now reaches**, so it covers the
whole cycle rather than a truncated foot.

### What the fixture costs and covers (measured at capture)

| quantity | value |
|---|---|
| steps | 80,416 |
| wall, single lane | 1047 s / 1034 s over the two captures (~17.5 / 17.2 min) |
| saves | 2,627 |
| `final_time` | 2.625885e-02 s (the dynamic `t_end`, reached) |
| trajectory | `y[2627, 576]` = 8 fields × 72 cells |
| phase census (saves) | 9 `pre_breakdown`, 17 `breakdown`, 2000 `main_discharge`, 600 `afterglow`, 1 `post_afterglow` |
| save cadence | 10 us — the finest timing shift this fixture can resolve |

*(Figures above are the 2026-08-21 physics-batch capture, the SECOND one — the m_He ruling landed after the first and forced a re-recapture. The two captures
were bit-identical but not equal in wall time; the spread is scheduling, not
trajectory, and the smaller figure is the cleaner lane.)*

**The gate is ~2× the wall time of the fixture it replaced** (~8–9 min), not the
same — the pre-capture projection of 40–45k steps was taken from the
early-discharge `dt` (6.4e-7 at t = 1.4e-3 s) and the timestep does not hold
that value through the plateau; the measured mean is 3.1e-7. What the extra cost
buys is the whole cycle at a representative operating point, including the
plateau and the afterglow, instead of 8 % of the discharge at a corner the
campaign never runs. Flagged rather than silently absorbed: a reviewer runs this
twice per merge.

## Disclosed closure stress (diagnosis side-finding, not fixed here)

The anode sheath power is booked **volumetrically into two flanking cells**, and
that booking diverges as those cells empty — the ~100 ns e-fold seen at bare
defaults. It is visible only OFF the stance, where the gap never fills; on the
stance the cells carry plasma and the term is well behaved. Recorded as a known
closure stress, deliberately not addressed by this pass.

## Recapture record

**2026-08-21 — the pre-Tuesday PHYSICS BATCH (AUTHORIZED recapture; the
authorization is the batch brief itself, member 3 by Tom's 21as ruling).** All
FIVE members landed in one recapture event, which is why the movement below is
JOINT and no single member may be credited with any of it. Unlike the
2026-08-20 wave, this one moves CODE as well as defaults: the parallel Spitzer
conductivity is no longer a frozen coefficient, and the helium mass changed.

| # | member | what moved |
|---|---|---|
| 1 | pumping | `S_pump_L`, `S_pump_R` `2900.0` -> `3000.0` L/s (Davis-pinned elbow transmission) |
| 2 | anode recycle | `anode_neutral_jet` + `neutral_mesh_accommodation` ARMED at the stance; `anode_jet_R_N` `0.5` -> `0.63`, `anode_jet_R_E` `0.25` -> `0.41`; NEW key `anode_jet_energy_convention` (ships `None`, stance declares `"total_reflected"`) |
| 4 | fueling units | `SCCM_TO_PARTICLES_PER_S` `4.477962e17` -> `4.171431e17` (0 °C -> flow-meter 20 °C / 1013 mbar), with the three-class migration; `S_gp`/`Twin_S_gp` defaults `3400` -> `3649.84`, `S_gp_decay_target` `1500` -> `1610.23`; the stance's inert `S_gp_decay_target` line dropped |
| 5 | conductivity | `sigma_par = 14.6 Te^1.5` -> `(1.96/(1.03e-2 lnLambda(Te,n))) Te^1.5` in both sheath solvers and every `sigma_par` consumer; NEW selector `cathode_lnL_model` |
| 3 | helium mass | `ion_mass_g` `6.6464731e-24` -> `6.6464790809e-24` g (Ar(4He)*u), and the two other hand-made spellings in the repo collapsed onto that one definition point |

Member 3 was ruled after the batch's FIRST recapture (it was reported back as a
blocker: neither repo spelling was citable). It moves `ion_mass_g` repo-wide,
so it forced a SECOND recapture, and the fixture recorded here is that second
one. The first capture is superseded and is not the gate.

Values, classes and brackets: `config_defaults_provenance.md` (the `S_pump`,
anode-jet, `cathode_lnL_model` and sccm-changeover entries were all rewritten
in the same pass) and `production_stance_provenance.md` (the arm itself and the
`S_gp` meter-class restatement).

**Delta discipline — what was proven before anything was recaptured.** A
no-solve resolved-config diff of the branch against the base commit, in BOTH
columns (bare `default_config()` and `default_config()` + `g1atrim` + `nx=60`),
showed exactly the config members above and nothing else: 9 deltas in the
default column, 10 in the golden column, 0 in either flags column. (The
helium mass is not a config key, so it does not appear in that diff; it is
pinned by a smoke literal instead — see `config_defaults_provenance.md`.) `S_gp = 9010`
verified UNCHANGED (it is METER-CLASS). `pump_elbow_conductance_lps` stayed
`None` — the double-count trap the `S_pump` entry documents.
`config_snapshots.json` was regenerated in the same pass; `parameter_count`
248 -> 250 and `flag_count` 47 unchanged, the two new keys being the whole of
the manifest delta.

**Capture evidence.** Recaptured twice from clean separate processes to
temporary paths and compared BEFORE installing: NPZ and JSON sidecar both
byte-identical (`sha256` of the NPZ
`857b9e0b6b31c2de36d5cfe24a8fb0023c16ec4b68a9111c5eb89be4a9cc47d1`, of the
sidecar `ffab0aa9b7d1702ef2c21f4905c9db383fd4be8d8478121a8f6566c3c45f2435`),
and raw-bitwise identical at `uint64` over all three arrays (`y`, `time`,
`phase`; 0 differing elements). Wall 1047 s and 1034 s (~17.5 / 17.2 min) on
one lane, run strictly serially. `--verify` prints `exact=True` on the pure
path.

**What moved in the trajectory, and why it is the physics.** Steps
84,276 -> 80,416; saves 2,626 -> 2,627; `final_time`
2.624091e-02 -> 2.625885e-02 s; phase census `breakdown` 16 -> 17, everything
else unchanged. (The helium mass alone accounts for 100 of those steps and
~2e-11 s of the end time — a 0.9 ppm perturbation that the adaptive timestep
amplifies into a different step count while leaving every reported figure
below unchanged at the quoted precision.) The fueling changeover dominates the fill: `nn` at `t = 0`
falls 11.4 % (the meter convention delivers 6.85 % fewer particles per
configured sccm, and the stance's 9010 was deliberately NOT rescaled), and the
column follows — plateau mean `n` −10.0 %, `n_max` −9.9 %.

**`Te_max` (−12.4 %) is a breakdown-phase transient attained in the FAR COLUMN
(cell 65 → 58, i.e. 90 % → 81 % of the mesh), not the gap; it moves jointly
with the fill and two saves later (save 13 → 15, +20 µs); no member is
credited. Plateau Te moves the OTHER way, +8.8 %.** An earlier draft of this
entry credited that fall to the conductivity member as its "signature". That
was wrong, and it is corrected here rather than quietly deleted, because a
wrong attribution that has already been read is worth more as a correction
than as an absence.

*Localization, verified on all three fixtures rather than taken on report:*
pre-batch save 13 / cell 65; batch-before-`m_He` save 15 / cell 59; final
save 15 / cell 58 — all three in the `breakdown` phase and all three in the
far column. (The advisor's 21at reading of "cell 59" was taken on the
before-`m_He` capture and is exact for it; the final fixture sits one cell
further in. "One save later" measures as TWO saves, 13 → 15, on both.)

**Plateau Te rises, and that is a DEBIT.** The model already runs hot in `Te`
against the measurement, so +8.8 % over the plateau (+10.5 % at mid-machine)
makes the model-hot residual WORSE at this fixture. It is recorded as a debit
and deliberately not netted against the density improvement.

The `I_sat`-class metric `n*sqrt(Te)` moves only −3.2 % over the plateau
(−4.7 % mid, +0.7 % far column), so most of the density fall is offset in the
systematics-robust combination. Health stayed sane and finite throughout;
`Ti_max` +0.9 %.

**Endward steepening.** The far column falls further than the mean: plateau
mean `n` −10.0 % against −12.8 % over the outer quarter of the mesh (−9.4 % at
mid-machine). The profile does not translate down uniformly; it steepens
toward the far end.

**Afterglow** (sub-eV `Te` is SEMI-QUANTITATIVE by the campaign's own
measurement policy — reported, not scored): mean `n` −8.6 %, mean `nn`
−8.1 %, mean `Te` 0.726 → 0.844 eV.

**Timing, stated at the fixture's resolution and no finer.** The
main-discharge onset moves +10 µs LATER and the afterglow onset with it. The
save cadence is 10 µs, so that is ONE save bin — the fixture cannot resolve the
predicted competition between a later breakdown (less fuel) and an earlier
conductivity knee (order tens of µs) any more sharply than this, and the
observed shift must not be read as a measurement of either member alone.

**The old and new trajectories are not comparable point-by-point** — this is a
configuration AND code change, not a repair, and no bit-level comparison
between them is meaningful.

**2026-08-20 — stance-update wave: the S_pump and cathode-jet re-cuts
(AUTHORIZED recapture, Tom; CAMPAIGN_LOG 2026-08-20xx/as/ax).** **The first
recapture forced by the GOLDEN-AT-STANCE consequence** — the standing trade
declared in the R2b entry below, exercised for the first time. Two shipped
defaults moved, so the fixture moved with them; no code path, pin, or run
shape changed.

| key | old | new |
|---|---|---|
| `S_pump_L`, `S_pump_R` | `4000` | `2900.0` L/s |
| `cathode_jet_R_N` | `0.5` | `0.34` |
| `cathode_jet_R_E` | `0.2` | `0.18` |

Values, classes and brackets: `config_defaults_provenance.md` (both entries were
rewritten in the same pass; `S_pump` moved ASSUMED -> DERIVED, the cathode jet
pair moved MEASURED -> ASSUMED mid-box).

**Delta discipline — what was proven before anything was recaptured.** A
no-solve config diff of the rebuilt baseline against the OUTGOING fixture's own
sidecar showed **exactly those four keys and nothing else**, and a four-case A/B
showed the same four across every config-complete driver
(`production_golden`, `compare_sim1d_es1`, `run_m6_point_es1_sgp3400_defaults`,
`run_mechanism_ladder_es1_defaults`) with no flag moving in any of them.
`pump_elbow_conductance_lps` stayed `None` — the double-count trap the S_pump
entry documents. The `config_snapshots.json` fixture was regenerated in the same
pass for the same reason and under the same proof: substituting the four old
defaults back into the current manifest reproduces the previously committed
`manifest_sha256` bit-for-bit, so those four are provably its only deltas, and
`parameter_count` (261) / `flag_count` (51) are unchanged.

**Capture evidence.** Recaptured twice from clean separate processes to
temporary paths and compared BEFORE installing: NPZ and JSON sidecar both
byte-identical (`sha256` of the NPZ
`970006067c3cd6fbf406112fe26a4738c5c54f4aa783b07271d121dcc4a24571`), and
raw-bitwise identical at `uint64` over all three arrays (`y`, `time`, `phase`;
0 differing elements). `--verify` prints `exact=True` on the pure and the
compiled path, the latter with `KERNEL_ID` probed in-process.

**What moved in the trajectory, and why it is the physics.** The pumping re-cut
raises the equilibrated fill (~+36 % centrally), so the machine breaks down
sooner and runs denser: `pre_breakdown` saves 13 -> 9, `breakdown` 20 -> 16,
`n_max` 2.14e13 -> 2.92e13 cm^-3, steps 75,615 -> 84,276, `final_time`
2.632261e-02 -> 2.624091e-02 s. Health stayed sane and finite throughout:
`Te_max` 28.9 -> 29.5 eV, `Ti_max` 8.36 -> 7.85 eV. **The old and new
trajectories are not comparable point-by-point** — this is a configuration
change, not a repair, and no bit-level comparison between them is meaningful.

**2026-08-20 — R2b re-anchor onto the stance of record (AUTHORIZED recapture,
Tom, 2026-08-20; thread-24 R2b, GOLDEN-AT-STANCE ratified after the dt-collapse
diagnosis).** The fixture moved from the ~30-pin 2026-07-22 operating point to
the stance re-cut at `nx = 60`. This is a wholesale change of configuration, not
a repair: the new and old trajectories are unrelated and no comparison between
them is meaningful. The pass folded `heat_flux_limiter_f` (`0.3 -> 0.1`) into the
shipped defaults, removed every campaign-driver import from `baseline_sim1d.py`,
replaced the pin table, and recaptured twice from clean separate processes with
byte-identical results.

Health of the recaptured trajectory, as a check that the stance does what the
diagnosis said it would: `Te_max = 28.9 eV`, `Ti_max = 8.36 eV`,
`n_max = 2.14e13 cm^-3`, finite throughout. The bare-defaults draft reached
`Te_max = 474 eV` — the flanking-cell runaway the diagnosis identified.

*An intermediate draft of this pass captured at bare `default_config()` and was
corrected before merge.* It is recorded here because the failure is instructive:
"the folds made the defaults the production package" was a plausible reading of
the R2a work and it was false, and the fixture it produced looked healthy —
`exact=True` on both paths, byte-identical captures — while gating a corner the
campaign never runs. Bit-exactness gates certify reproducibility, not
representativeness; nothing in the gate could have caught this, and the
dt-collapse diagnosis is what did.

*A latent sidecar bug this pass surfaced and fixed:* `capture()` derived the
sidecar's `cells` as `y.shape[1] // 5`, assuming five packed fields per cell.
That divisor is only right for the cold single-zone neutral layout the retired
fixture ran; under the shipped closure the packing is EIGHT fields. `run_baseline`
now reports the cell count from the solver's own geometry and the sidecar records
`fields_per_cell` alongside it, so no fixed divisor is assumed.

*On the `heat_flux_limiter_f` fold:* it was held back from R2a on the expectation
that folding it would move the golden. It would not have — the R1
stance-decoupling pass had already pinned `0.1` as a literal in
`BASELINE_PARAM_OVERRIDES`, which made the old fixture immune to that default.
Verified: the resolved OLD golden config is byte-identical across the fold.

**2026-08-09 — returned-root sheath-ceiling fix (AUTHORIZED recapture, Tom,
2026-08-09).** *Applies to the retired fixture, retained as record.* The
current-driven sheath solve enforced `cathode_phi_c_cap_V` only at the bracket
ladder's doubling grid points, never on the root it returned; the fixture's own
ignition foot contained 34 such escaped solves (net phi_c up to 1.9669× the
1000 V cap — 1966.89 V returned at the cap), i.e. the committed trajectory
certified the defect. The fix (commit `8a09363`) tests the located J-root
against the cap and routes an at-or-above-cap root through the pre-existing
ceiling branch, so the foot solves moved BY DESIGN and both goldens failed
against the old fixture with `max_rel=2.000e+00`,
`time_max_abs=1.113e-06 s`, character-identical on the pure and compiled paths.
Recaptured with the script's own `--capture` at the fixed code. The pin set was
unchanged: zero added, removed, or changed keys, and the sidecar params/flags
diff against the previous capture showed zero changed values — the 18 param keys
and 1 flag key newly recorded were config defaults added since the 2026-08-03
capture, already in effect for every verify since, recorded at their unchanged
live defaults. saves stayed 2545, cells stayed 72; steps 41054 → 40975 on the
corrected foot.
