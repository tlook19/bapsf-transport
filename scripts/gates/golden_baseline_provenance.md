# Provenance of the golden baseline pins (`baseline_sim1d.BASELINE_*_OVERRIDES`)

**Recaptured 2026-09-02 (THE KINETIC STANCE EVENT — `g1atrim.toml` adopted the
transient discrete-velocity neutral closure at wall accommodation 0.40, with
the `C_R = 9.30` drive calibration promoted from arm override to stance value,
after the ES1 program on the measured field profile; the neutral-equilibration
pre-solve was fixed in the same event. A stance change and a golden re-anchor
are one event, so this is the recapture that pays for it.** It moves the
TRAJECTORY, the config identity, and the packed state's SHAPE: the fixture
carries 6 fields per cell instead of 8, because the closure retires the evolved
neutral momentum and energy rows. The recapture record's newest entry carries
the moved numbers; the preceding recaptures are summarized next.

**Recaptured 2026-08-28 (THE R3-TIP RECAPTURE — the one anchor event that
rotates every golden reference together, closing the six-member R3
reimplementation window; the stance file `g1atrim.toml` and
`default_config()` are BOTH untouched, and so is the whole config surface.
What moved is the VALUE OF PHYSICAL CONSTANTS IN CODE.** That is a shape this
file has not recorded before: every previous recapture moved a configuration
key, so its digest `config_identity` rotated. This one does not — the
`config_identity` is bit-identical across the event at
`ec8ba03d…` while the trajectory digest moves, which is the exact signature of
a change that is real physics and no configuration at all.) The recapture
record's newest entry carries the moved numbers; the preceding recaptures are
summarized next.

**Recaptured 2026-08-26 (`[afterglow-dt-cost]` ADOPTION — the exemption
hysteresis band and the accelerated dt-growth re-approach flipped to armed
config defaults; a stance change and a golden re-anchor in one event, see the
recapture record below). The stance file `g1atrim.toml` is UNTOUCHED by that
event: what moved is `default_config()`, which this fixture layers under the
stance.** The recapture record's newest entry carries the moved numbers; the
preceding recapture is summarized next.

**Recaptured 2026-08-25 (THE STANCE EVENT — the `plateau_multigroup`
anomalous-heating closure adopted with its `C_R` re-trim, the
`anode_sheath_full_debit` and `beam_deposition_in_heat_substep` flags armed, and
the `V_bank` stance row deleted; the 2026-08-24 tube-beamed injection row, the
CAD-span machine geometry with its ray-clip exactness fix, the 2026-08-23
conserving ionization birth with its own `C_R` re-trim, the `heat_flux_limiter_f`
re-cut to 0.45, the pre-Tuesday physics batch, the 2026-08-20 stance-update wave
and the thread-24 R2b re-anchor onto the stance preceded it — all eight under the
reviewed-recapture protocol, see the recapture record below). This is the first
recapture since the digest reference was introduced in which the TRAJECTORY
digest moves rather than only the config identity.** The committed
regression fixture
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
**A recapture regenerates the short-horizon digest reference
`scripts/baselines/golden_digest_4k.json` in the SAME event**, with
`python scripts/golden_digest_gate.py --capture`: it pins this same
configuration and is invalidated by exactly the events that invalidate the NPZ,
so the two rotate together or the digest gate starts failing for a reason that
has nothing to do with the code under review.

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
| `max_steps` (run kwarg) | `150000` | **A tripwire, not a run length** — ~2.65x the 56,605 steps measured at the 2026-09-02 kinetic stance. It exists so a change that quietly destroys the timestep fails fast instead of running for hours; if it fires, the question is what happened to `dt`, not what happened to the trajectory. The value was SIZED at ~2x deliberately, against the 76,631 steps the fixture ran at the R2b re-anchor (2026-08-20) — a backstop with a few percent of headroom is not a backstop, it is a second cost cap waiting to truncate the gate. The margin has moved with the fixture ever since WITHOUT the cap moving: ~1.6x at the 94,044 steps of the 2026-08-25 stance event, ~2.4x at the 62,612 of the 2026-08-26 adoption and the 2026-08-28 R3-tip recapture (which moved the count by ONE step), ~2.65x now. Re-sizing it remains a golden-touching change rather than a maintenance edit. |
| digest horizon (`baselines/golden_digest_4k.json`) | first `4000` accepted steps | The companion fixture for `scripts/golden_digest_gate.py`, which folds the packed state into a running SHA-256 after EVERY accepted step of this same configuration. The horizon is a cost knob, not physics: 4,000 steps is **8 min 06 s** on the pure path (2026-09-02 kinetic recapture, clean lane) against the full gate's 26 min 56 s capture / 26 min 50 s verify COMPILED — the pure full-gate figure is not measured at this stance. Over the steps it covers the digest is the STRONGER check, because the golden certifies only what reaches a save. (Earlier figures, each true at its own stance: 7 min 57 s pure against 16 min 27 s compiled / 38 min 41 s pure at the 2026-08-28 R3-tip recapture; 11 min 33 s / 16 min 16 s / 45 min 55 s at the 2026-08-26 adoption; ~2.5 min against a ~17 min gate before the 2026-08-25 adaptive-`dt` change.) That gate runs at `max_steps_action = "stop"` — the cap is its run length, not a tripwire — which changes what happens AT the cap and nothing before it. |

`BASELINE_FLAG_OVERRIDES` carries one entry, `neutral_equilibration = True`, for
the reason given in the re-cut section above.

`t_end`, `dt` and `operator_split` are `None`, i.e. the solver's own run
defaults: adaptive dt, the shipped operator split, and the dynamic
current-trigger end time — **which the fixture now reaches**, so it covers the
whole cycle rather than a truncated foot.

### What the fixture costs and covers (measured at capture)

| quantity | value |
|---|---|
| steps | 56,605 |
| wall, single lane | **26 min 56 s** CAPTURE compiled, **26 min 50 s** VERIFY compiled. Measured 2026-09-02 on a CLEAN lane, strictly serially, nothing else running. **No PURE full-gate figure is measured at this stance, and none is carried forward from the last one** — the pure path is bit-exact against compiled by construction, so what is missing is a wall time and nothing about the fixture; anyone who needs it must measure it. |
| saves | 2,620 |
| `final_time` | 2.618754e-02 s (the dynamic `t_end`, reached) |
| trajectory | `y[2620, 432]` = **6** fields × 72 cells |
| phase census (saves) | 8 `pre_breakdown`, 11 `breakdown`, 2000 `main_discharge`, 600 `afterglow`, 1 `post_afterglow` |
| save cadence | 10 us — the finest timing shift this fixture can resolve |
| 4,000-step digest horizon | **8 min 06 s** pure (the reference capture) |

**Six fields per cell, not eight.** The kinetic closure retires the evolved
neutral momentum and energy rows from the packed state, so the trajectory
narrows from 576 to 432 columns. That is a SHAPE change, and `--verify` reports
a shape mismatch before it reports a trajectory mismatch, so a stale fixture
against this configuration fails loudly rather than subtly.

**The step count fell 62,612 -> 56,605 (-9.6 %) and the wall time ROSE.** The
compiled capture and verify are ~27 min each against the previous stance's
16 min 27 s compiled verify: the per-accepted-step cost is higher under the
kinetic closure, which is unsurprising — the DVM advances a distribution on a
64 x 24 velocity mesh every 3.125 us tick, work the fluid closure did not do.
The size of that per-step cost is NOT measured here and is not claimed; the two
figures are single measurements on one lane, and a controlled comparison would
have to run both closures at one stance.

*(The paragraphs below this table are the record of EARLIER events and their
figures are those events'. The 2026-08-28 R3-tip figures were: 62,612 steps;
16 min 27 s VERIFY compiled, 38 min 41 s VERIFY pure, 37 min 56 s / 38 min 36 s
CAPTURE pure; `final_time` 2.618681e-02 s; `y[2620, 576]`; 7 min 57 s pure at
the digest horizon. `steps`, `saves` and
`final_time` are read from the committed sidecar
`scripts/baselines/production_discharge.json`, which is regenerated at every
recapture and is the authority for them, and the table above now carries the
2026-09-02 values. The pure CAPTURE fell 45 min 55 s →
37 min 56 s and the pure 4,000-step digest 11 min 33 s → 7 min 57 s across
this event, on comparably clean lanes; the R3 lane march is the obvious
candidate and the step count barely moved, but neither figure is a controlled
measurement of it and the saving is not claimed as one.)*

**Wall time did not fall in proportion to the step count, and the comparison
that would explain it is not available.** The adoption removed 33.4 % of the
accepted steps, but the pure capture came in at 45 min 55 s against the
previous capture's recorded 46 min 38 s and the compiled verify at 16 min 16 s
against 19 min 14 s. The two figures are not comparable measurements: the
2026-08-25 pair was measured on CONTENDED lanes and recorded as an upper
bound, this pair on a clean one. What can be said without a controlled
re-measurement is that the per-accepted-step cost is higher at the armed
defaults. **Why is not established here** — rejected attempts per accepted
step are the obvious candidate and were not counted — and neither the size of
the effect nor the true wall-time saving is measured. Anyone who needs that
saving as a number must measure both stances on one lane.

**At the 2026-08-20 R2b re-anchor the gate came out ~2× the wall time of the
fixture it replaced** (~8–9 min), not the same — the pre-capture projection of
40–45k steps was taken from the early-discharge `dt` (6.4e-7 at t = 1.4e-3 s)
and the timestep does not hold that value through the plateau; the mean measured
at that capture was 3.1e-7. What the extra cost buys is the whole cycle at a
representative operating point, including the plateau and the afterglow, instead
of 8 % of the discharge at a corner the campaign never runs. Flagged rather than
silently absorbed: it is a cost paid by everyone who runs the full golden.

*(Every figure in the paragraph above is R2b's, over its 76,631-step capture,
and is kept because it is the record of that re-anchor's decision — the fixture
has since moved to 94,044 steps, see the table above. It also used to end "a
reviewer runs this twice per merge", which is no longer how the gate is run:
the golden's cadence is set by the review tiering, not by this note, and most
merges do not run it at all.)*

## Disclosed closure stress (diagnosis side-finding, not fixed here)

The anode sheath power is booked **volumetrically into two flanking cells**, and
that booking diverges as those cells empty — the ~100 ns e-fold seen at bare
defaults. It is visible only OFF the stance, where the gap never fills; on the
stance the cells carry plasma and the term is well behaved. Recorded as a known
closure stress, deliberately not addressed by this pass.

## Recapture record

**2026-09-03 — THE SPEED-UP EVENT: four bit-moving performance members
re-anchored in one recapture (AUTHORIZED; the Tier C continuity pair passed all
seven of its pre-registered gates at the production operating point before this
member was dispatched).** The members are a REIMPLEMENTATION programme at fixed
physics intent — every one of them computes the same model by a cheaper route —
and they were bundled so the golden re-anchors once. In merge order:

| member | branch commit | what it changed |
|---|---|---|
| `[wr-diffuse-vectorize]` | `aa2cc81` | the diffuse cylindrical-wall return caches the grid-only factors of its 64-node cosine quadrature on the velocity grid. BIT-EXACT: the 4k digest is unmoved at this member. |
| `[wr-secant-solve]` | `df560c8` | the same wall's re-emission temperature is solved by a secant in `ln s` on a SEPARABLY contracted mean energy instead of a bracketed bisection on assembled spectra — 54 residual evaluations per solve to 4–6. The bisection is retained as the deterministic fallback and is still the path that raises on a saturated target. |
| `[equilibration-banded-solve]` | `9e61247` | both implicit neutral steppers assemble and solve in LAPACK banded storage instead of a dense `numpy.linalg.solve` — tridiagonal single-zone, pentadiagonal two-zone in the interleaved `(column cell, annulus cell)` ordering — and `NEUTRAL_STEPPER_ID` enters the neutral-seed signature. |
| `[hyperbolic-correction-row-split]` | `b486373` | the KEP energy correction is booked in TWO ledger rows: its pressure re-discretization folds into `pressure_work`, and the Rusanov numerical dissipation becomes `hyperbolic_dissipation_heating`. This re-associates the RHS fold. |

**Nothing in the CONFIGURATION moved.** `g1atrim.toml` is untouched;
`default_config()` is untouched; no config key was added, removed or re-valued.
This is the same shape of event as the 2026-08-28 R3-tip recapture and the
opposite of the 2026-09-02 stance event: the trajectory digest rotates while the
config identity does not.

| quantity | before | after |
|---|---|---|
| `production_discharge.npz` `sha256` | `cdd706d3301e85871ae6e0406bd31cf373ac46f1f38cc00d2c704961513704fc` | `c286d2f415058c20966d85d6d15c4355887eefd6f548c49c371d9bd64f1578c4` |
| `production_discharge.json` `sha256` | `3e1cd3514a9425cd4fc6d24f74cb86da1644c8f3d83afbf0ea18250157d792e9` | `dd10e5c763f777c62655780afc98d07d8a3e5422760f38170d9bcad6c83a0191` |
| `golden_digest_4k.json` `sha256` | `7f5bd37630feec56ef3e5b366ce0aed429f29eab359ee27cae9e733658b7699c` | `a38d5a72459d2979fc95000e9fd183d72a3f5eaf94083d2ac91c6176cea6dac4` |
| `golden_digest_4k.json` `digest` | `b883916aae9b6aca6b2f501f6418a943363abed29ce9f19394d2dfc7275fc086` | `4c0e105b922e67308595e2cbdd628d9f4dcde2e1686d00225cb69bacec8643ea` |
| `golden_digest_4k.json` `config_identity` | `ea042038a5e01230c11a66f1cf429099fc914633febd2fd6e9721b6b2626c965` | **UNCHANGED** — `ea042038…` |
| `golden_digest_4k.json` `final_time` | `0.0004563240693165767` | `0.00045632279608921294` |
| `golden_digest_4k.json` `fields_per_cell` / `cells` / `steps` | `6` / `72` / `4000` | UNCHANGED |
| sidecar `steps` | `56,605` | **`56,392`** (−213, −0.38 %) |
| sidecar `final_time` | `2.618754254253e-02` s | `2.618754254937e-02` s — the dynamic `t_end`, still reached |
| sidecar `saves` / `cells` / `fields_per_cell` | `2620` / `72` / `6` | UNCHANGED |
| sidecar `params` / `flags` key counts | `256` / `53` | UNCHANGED — 0 added, 0 removed, **0 re-valued** |

**The fixture's shape and cycle coverage are unchanged, and that was measured
rather than assumed.** `y` is `[2620, 432]` before and after; the `phase` array
is byte-identical; the phase census is the same `8 / 11 / 2000 / 600 / 1`
(pre-breakdown / breakdown / main discharge / afterglow / post-afterglow). The
`time` array differs in exactly **1 of 2620** samples — the last one (index 2619), where the
dynamic `t_end` lands 6.8 ps later — so the save schedule the fixture pins is
otherwise identical. `y` differs in 1,121,076 of 1,131,840 raw `uint64`, which
is what a bit-moving member set over 56,392 steps produces.

**What the trajectory movement is, stated honestly.** Over the whole fixture the
relative move of `y` has median 1.4e-4, 99th percentile 3.6e-2, and a maximum of
538 at one afterglow momentum entry that crosses zero (8.935e-12 → −4.802e-09
against a field scale of 1.4e-6 at that save — a denominator artifact, not a
physics move). This is why the golden is a REGRESSION SCAFFOLD and not a
physical claim: bits move from the first step and 56,392 steps of a nonlinear
run amplify them. **The physics verdict is the Tier C continuity pair
(2026-09-03), not this fixture.**

**Physics verdict — the Tier C continuity pair, 2026-09-03**, measured at the
PRODUCTION ES1 operating point on the compiled path, parent `e67b24b` against
tip `b486373`. Seven of seven pre-registered gates PASS, with two disclosures:

| gate | result |
|---|---|
| 1 stage-(ii) `dev/sigma` | 98-line transcript identical except three `Te` means at the third decimal (8.726/4.862/3.270 → 8.725/4.861/3.269 eV); every ratio and `|dev|/sigma` identical to print; 0 printed entries moved |
| 2 plateau current | 2955.726 → 2955.721 A, −1.7e-6 (bar 0.1 %) |
| 3 stage-(iii) `tau` per port | identical to print, 5 of 5 |
| 4 `t90` / peak time | 2.71 → 2.70 ms / +19.98 ms unchanged (bar ≤ 0.5 ms) |
| 5 ledger closure + rows (DRIVE window, partition-mapped) | handshake closure `7.227e-19` IDENTICAL; the split maps exactly — parent `pressure_work` Ee −15.50773 + hyperbolic 8.95521 = −6.55252 against tip `pressure_work` Ee −6.55255, and Ei −8.86590 + 7.11878 = −1.74712 against tip −2.72277 + dissipation 0.97565 = −1.74712; mapped rows 4.6e-6 (bar ≤ 1e-3). Disclosed: worst other row `neutral_kinetic_dvm_coupling` −20.09078 → −20.10632, 7.7e-4 — the secant solve is the one numerics change on the DVM path |
| 6 floor census | `floor_ledger` Ee 1.9787e7 → 1.9520e7 erg (−1.3 %), Ei 3.119e4 → 1.565e4 erg, particles 0 both; negligible energies, no regime change. Disclosed: the Ei floor injection halves, 3.1 → 1.6 mJ over the run |
| 7 steps / saves / config | 49374 → 49464 steps (+0.18 %, bar ≤ 1 %); saves 2613 identical; params+flags diff EMPTY. Disclosed: the tip has `relax_limited_steps = 1`, `limited_cells = 5` against the parent's 0 — one event, at the cadence an earlier retrim arm also showed |

Health at the production point (scaffold, ungated): `n_max` 1.794406e13 →
1.794591e13, `nn_max` 9.6997e13 → 9.6929e13, `Te_max` 30.05799 → 30.05800 eV,
`Ti_max` 7.4516 unchanged, plasma-inventory drift 2448.45 → 2448.47,
thermal-energy drift 2878.1 → 2823.4.

**Sidecar health, before and after this recapture.** `finite: true` both;
`samples` 2620 both.

| row | before | after |
|---|---|---|
| `Te_max` | `26.51924703839592` | `26.51924703839118` — identical to 12 significant digits |
| `Te_min` / `Ti_min` | `0.1` / `0.02585` | UNCHANGED (the floors) |
| `Ti_max` | `7.318796883522123` | `7.318876364837494` |
| `n_max` | `33088144428064.066` | `33081315555355.5` (−2.1e-4 relative) |
| `n_min` | `851849034.5862272` | `851849034.5862641` |
| `nn_max` | `135809829921687.27` | `135755895284095.28` (−4.0e-4 relative) |
| `nn_min` | `234627346500.57492` | `234548993631.36646` |
| `plasma_inventory_relative_drift` | `4584.0697421545065` | `4584.836734431819` |
| `neutral_inventory_relative_drift` | `1.1049204968281352` | `1.1049859150084276` |
| `thermal_energy_relative_drift` | `4676.843333625317` | `4668.466395970876` |
| `total_particle_inventory_relative_drift` | `1.1937872576987203` | `1.1938675471050408` |

**The INITIAL CONDITION moved, and exactly one member moved it.** The digest
gate's step-0 checkpoint — the state the stepping starts from, which is the
equilibrated neutral seed — goes `2dcb7c3e…` → `5ecba233…`. Recorded per member
from clean captures at each commit, the chain is:

| member | step-0 checkpoint | 4k digest |
|---|---|---|
| parent `e67b24b` / `[wr-diffuse-vectorize]` | `2dcb7c3e…` | `b883916aae9b6aca…` |
| `[wr-secant-solve]` | `2dcb7c3e…` | `99255f0bafd5bbdd…` |
| `[equilibration-banded-solve]` | **`5ecba233…`** | `35616c8ea2af819a…` |
| `[hyperbolic-correction-row-split]` | `5ecba233…` | `4c0e105b922e6730…` |

The seed is produced by the implicit neutral stepper, so the banded solve is the
only member that can move it, and it is the only member that does. This is the
rotation `NEUTRAL_STEPPER_ID` was added to make visible from the seed cache's
own side: every seed equilibrated by the dense stepper now keys apart from one
this stepper would write, and stored seed databases must be rebuilt with
`scripts/build_neutral_seed_cache.py`. The per-member digests above are recorded
so a future regression in this window can be bisected without re-deriving them.

**Capture evidence.** The NPZ fixture was captured TWICE from clean separate
processes to temporary paths, strictly serially per the serial-golden rule, and
compared BEFORE either was installed: `phase` identical at raw bytes, `time`
**0 differing of 2,620** and `y` **0 differing of 1,131,840** at `uint64`, both
NPZ `sha256` `c286d2f4…` and both JSON sidecars byte-identical
(`sha256` `dd10e5c7…`). The digest reference was regenerated in the same event
as the protocol requires, also twice from clean separate processes, byte-identical
(`sha256` `a38d5a72…` both).

Gates at this tip, single lane, strictly serial, nothing else running:

| gate | result |
|---|---|
| CAPTURE, compiled | `baseline captured: ... saves=2620, cells=72, fields=6, steps=56392, final_time=2.618754e-02 s` (`kernels=cython/_cathode_kernels_cy/tierA+csda`, probed in-process), twice, byte-identical |
| COMPILED golden | `baseline verify OK: saves=2620, exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s (rtol=1.0e-09, atol=0.0e+00)`, **16 min 17 s** (`kernels=cython/_cathode_kernels_cy/tierA+csda`, probed in-process) |
| 4k digest reference, pure | regenerated in this event, `kernel_provenance: "pure"` |
| PURE 4k digest leg | `digest gate OK: steps=4000, digest=4c0e105b922e67308595e2cbdd628d9f4dcde2e1686d00225cb69bacec8643ea, exact=True` (`digest gate kernels=pure`) |
| COMPILED 4k digest leg | `digest gate OK: steps=4000, digest=4c0e105b922e67308595e2cbdd628d9f4dcde2e1686d00225cb69bacec8643ea, exact=True` (`digest gate kernels=cython/_cathode_kernels_cy/tierA+csda`) |
| smoke | exit 0, all five compiled-kernel equivalence blocks LIVE |
| `config_snapshots.json` | **NOT regenerated, and that is a verified result rather than an omission.** `scripts/audit_sim1d_configs.py` verifies the committed artifact (`cablp/solvers/_sim1d/config_snapshots.json`, `sha256` `75aba027…`) CLEAN at this tip: `params=256, flags=53, cases=4`, all four case digests reproducing, `production_golden` still `0f284816…` and `manifest_sha256` still `fccc1248…`. No config key was added, removed or re-valued — the same fact the unchanged `config_identity` states from the other side, measured by a second independent instrument. |

**KERNEL PATH — the NPZ was captured COMPILED, as at the 2026-09-02 stance
event, and a PURE full-gate verify was NOT run here.** What that costs is a
wall-time number and nothing else, because pure-vs-compiled bit-exactness is
RE-EVIDENCED at this tip rather than assumed: the 4k digest gate passes on BOTH
paths against the same PURE-captured reference, printing the identical digest
with kernel provenance probed in-process each time. Anyone who needs a pure
full-gate figure at this tip must measure it rather than read an earlier row.

**Wall time, and the point of the event.** The production ES1 arm falls from
**80 min to 31 min (2.6×)** across this window, measured in the same contention
window on both sides. Uncontended figures were NOT measured and no per-member
attribution of the run-level number was taken; the per-member microbenchmarks
live with their own members. The registered component measurements were: the
diffuse wall return 11.8768 → 0.9144 ms per call at the stance velocity grid
(13.0×), and one implicit neutral step 6899.27 → 126.99 µs at `nx = 268`
(54.3×), each min-wall over 9 repeats on a single lane.


**2026-09-02 — THE KINETIC STANCE EVENT. RECAPTURED, on the compiled path.**
`scripts/stances/g1atrim.toml` adopted `neutral_model = "kinetic_dvm"` — the
transient discrete-velocity neutral closure — at wall accommodation 0.40 on a
64 x 24 velocity mesh with a 3.125 us neutral clock, both DVM directed-recycle
jets armed under the current latch, the baffles acting on the kinetic annulus,
and `C_R = 9.30` promoted from arm override to the stance value. The decision
followed the ES1 program on the measured field profile; the stance-side
provenance — every key, its class and its honest bar, and what may be claimed
from the closure family — is in `scripts/production_stance_provenance.md` and
is not restated here. A stance change and a golden re-anchor are one event,
which is what this recapture pays for.

**The fixture's SHAPE moved, which is new for this record.** The closure
retires the evolved neutral momentum and energy rows, so the packed state
carries 6 fields per cell instead of 8 and the trajectory narrows
`y[2620, 576]` -> `y[2620, 432]`. A stale fixture against this configuration
therefore fails on the shape check before the trajectory comparison is even
reached.

| reference | before | after |
|---|---|---|
| `production_discharge.npz` `sha256` | `2c02ccd882261a0b01e0a1e8f0e313113b8431fadd526fa9d4859694e5704306` | `cdd706d3301e85871ae6e0406bd31cf373ac46f1f38cc00d2c704961513704fc` |
| `production_discharge.json` `sha256` | `93a41a1c4fc4010fc3c0d57c6527f1de7e5b8c21d145a9b791eeca445068c9f0` | `3e1cd3514a9425cd4fc6d24f74cb86da1644c8f3d83afbf0ea18250157d792e9` |
| `golden_digest_4k.json` `sha256` | `71150c0483357f1adace777b8250fad02883c5a048190b889c2211e6c38bf46c` | `7f5bd37630feec56ef3e5b366ce0aed429f29eab359ee27cae9e733658b7699c` |
| `golden_digest_4k.json` `config_identity` | `ed352561a09c442c0e9ee5a592f3ce55744cb25d0110af9774a60eee10217d37` | `ea042038a5e01230c11a66f1cf429099fc914633febd2fd6e9721b6b2626c965` |
| `golden_digest_4k.json` `digest` | `cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2` | `b883916aae9b6aca6b2f501f6418a943363abed29ce9f19394d2dfc7275fc086` |
| `golden_digest_4k.json` `final_time` | `0.0004714055010197914` | `0.0004563240693165767` |
| `golden_digest_4k.json` `fields_per_cell` | `8` | `6` |
| sidecar `steps` | `62,612` | `56,605` |
| sidecar `final_time` | `2.618681e-02` s | `2.618754e-02` s |
| sidecar `saves` / `cells` | `2620` / `72` | UNCHANGED |
| sidecar `params` / `flags` key counts | `256` / `53` | UNCHANGED — 0 added, 0 removed |

**Exactly nineteen configuration values moved, and every one is a stance key.**
Read off the committed sidecar's own `params`/`flags` before and after:

- `params` (15): `neutral_model`, `C_R`, `cathode_neutral_jet`,
  `cathode_jet_surface_debit`, `cathode_jet_energy_convention`,
  `anode_neutral_jet`, `anode_jet_energy_convention`,
  `neutral_mesh_accommodation`, `neutral_kinetic_dvm_cathode_jet`,
  `neutral_kinetic_dvm_anode_jet`, `neutral_kinetic_dvm_cadence_s`,
  `neutral_kinetic_dvm_nvz`, `neutral_kinetic_dvm_nvp`,
  `neutral_jet_arm_current_A`, `neutral_jet_disarm_current_A`;
- `flags` (4): `neutral_momentum`, `neutral_energy`,
  `neutral_hot_internal_wall`, `neutral_kinetic_dvm_baffles`.

The stance also NAMES `neutral_kinetic_dvm_accommodation = 0.40`,
`neutral_kinetic_dvm_wall_reflection = "diffuse_elastic"` and
`neutral_two_zone = true`, and they are absent from the list above BECAUSE THEY
ALREADY EQUALLED THEIR CONFIG DEFAULTS. They are written into the stance as
class-1 declarations — the closure is exactly those choices — and they move no
resolved value, which is why the recapture cannot see them. The same is true of
the seven `[models.neutral_closure]` members that were already at the value the
selection requires.

**A SOLVER FIX RODE IN THE SAME EVENT, and it is bit-exact for every
configuration that constructed before it.** `run_neutral_equilibration` builds
an inner neutrals-only `LAPDSim1D` and already cleared `Plasma`,
`cathode_coupling` and the seed-cache controls on its own copy of the config;
it did not clear the two DVM jets, so the golden's own re-cut — which arms
`neutral_equilibration` because the shaped foot cannot travel to `nx = 60` —
was REFUSED at the inner construction the moment the stance selected
`kinetic_dvm`. The pre-solve has no plasma and no cathode solve, so neither jet
has a collected ion flux to split or a sheath potential to launch against, and
both are additionally latched to an arm current a `Plasma = False` run never
reaches: the guard was firing on a state where the thing it protects cannot
happen. The only configurations the fix touches are ones that RAISED, so no
equilibrated seed, cache signature or trajectory that existed before it can
move. It is covered from now on by the smoke case
`golden-baseline-config-constructs`, which builds this fixture's config through
`baseline_sim1d.build_baseline_config()` — so it tracks the stance of record
with no pin of its own — and runs the pre-solve to `t_end = 0.0`, exercising the
inner construction at zero steps. Constructing the outer sim alone would not
have caught it, and did not: the outer sim constructed fine throughout.

Gates at this tip, single lane, strictly serial, nothing else running:

| gate | result |
|---|---|
| CAPTURE, compiled | `baseline captured: ... saves=2620, cells=72, fields=6, steps=56605, final_time=2.618754e-02 s`, **26 min 56 s** (`kernels=cython/_cathode_kernels_cy/tierA+csda`, probed in-process) |
| COMPILED golden | `baseline verify OK: saves=2620, exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s`, **26 min 50 s** |
| 4k digest reference, pure | captured in the same event, **8 min 06 s** (`kernels=pure`) |
| PURE 4k digest leg | `digest gate OK: steps=4000, digest=b883916aae9b6aca6b2f501f6418a943363abed29ce9f19394d2dfc7275fc086, exact=True` |
| smoke | exit 0 at 120 cases, all five compiled-kernel equivalence blocks LIVE |
| `config_snapshots.json` | rotated with the stance (four case digests; `manifest_sha256`, `parameter_count` 256 and `flag_count` 53 all UNCHANGED). Re-derived after the solver fix and byte-identical, so the fix moved no config digest — it is runtime behaviour on a copy, not config resolution. |

**Not measured at this stance, and not carried forward from the last one: every
PURE figure for the full gate.** The capture and verify above are compiled. The
pure path is bit-exact against compiled by construction, so the fixture is not
in doubt; what is missing is only a wall-time number, and anyone who needs one
must measure it rather than read the previous stance's row.

**2026-09-01 — NON-RECAPTURE STANCE EVENT: `g1atrim.toml`'s
`plasma_radius_profile_cm` rebuilt from the measured field. NOTHING WAS
RECAPTURED, AND THE GOLDEN DID NOT MOVE.** This entry exists for the same
reason the 2026-08-30 block-form entry below does: the stance file is a golden
INPUT, this record is where a reader looks when it changes, and a future
`git log` on `scripts/stances/g1atrim.toml` showing a 60-line diff on this date
must not read as an unrecorded stance change. Merge `a248ae0`
([msi-field-profile], ruled 2026-09-01 by Tom); the stance-side provenance,
including the honest bar and the unresolved coil-location disagreement, is in
`scripts/production_stance_provenance.md` and is not restated here.

**Exactly one key moved**, by an independent TOML-level walk at review: 65 keys
before and after, 0 added, 0 removed, `input_dict.plasma_radius_profile_cm` the
only change, 58 of its 280 entries differing (indices 222 onward — the profile
is held exactly flat at 18.415 cm below that). `machine_radius_profile_cm` is
byte-equal across the event.

**The golden is INERT BY CONSTRUCTION, and was measured so rather than
assumed.** `baseline_sim1d.py` pops the mesh-sized profile keys from the stance
before building the baseline config, and the golden runs
`prescribed_area_geometry = False` at `nx = 60`, so the changed array is never
read on that route. Verified three independent ways at the merge tip:

| check | result |
|---|---|
| config identity, digest-gate expression | `ed352561a09c442c0e9ee5a592f3ce55744cb25d0110af9774a60eee10217d37` — UNMOVED, matches the committed reference |
| COMPILED golden | `baseline verify OK: saves=2620, exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s` (`KERNEL_ID` `cython/_cathode_kernels_cy/tierA+csda`, asserted in-process before the run) |
| PURE 4k digest leg | `digest gate OK: steps=4000, digest=cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2, exact=True` |

`scripts/baselines/` is untouched — no `--capture`, no rotation. Both golden
references and the NPZ are byte-unchanged.

**ROUTE IDENTITY MOVED, and NO ROTATION IS OWED.** A stance change is supposed
to move the routes that carry the stance, and it did — but
`scripts/declm_route_identity.py` builds both sides on the fly and compares
them, pinning no committed reference file and no hash constant. Its exit 1 is
the tool REPORTING a stance diff, which is what it exists to do, not a gate
failing against a pin. There is therefore nothing to rotate, and this event
follows the 2026-08-30 block-form precedent, which likewise filed a record here
and rotated nothing. The new values of record, verified at the merge tip:

| route | before | after |
|---|---|---|
| `golden` | `3ac1dc1be6201c8bb48e1e2fe6c488427b52f434d0f44d0a5f1a8976bd3cfb8a` | UNCHANGED |
| `default`, `b0c`, `k2_dvm` | — | UNCHANGED |
| `ka1c` | `235d1c52ccaba6a28af8415eea9fb024fa3648f6db6e3a490af466fd9eba6780` | `0ed7d4c412f28cd11a5eba1f616213753c2975fcbe95ed13d871458e428bcb49` |
| `m6_es1` | `d51fc59f8f09719dd3ee9a33fd3e22a4376bfcc957a3bdcc32d0b9ab7de9625a` | `fd7d119023fe12e74f5a00823a428b0343551b7527c4ec25b9e22ab02ff22df6` |
| `stance_g1atrim` | `266c9d7621ea8a3fb40fa8ef938d8458356fe002b2133ee53777e4da1ad93fd5` | `a78a26fdf16c341c0aeb68bb7eaca420c71aeec8bde16b2d1e9a51d5f1cd3bdd` |

Each of the three moved routes moves by exactly `params:plasma_radius_profile_cm`
and no other key, checked per-key at review.

**The flat rule is a z-threshold, not a per-cell tolerance, and that is
load-bearing.** The profile is held exactly at `Rp` upstream of the first
sustained departure rather than wherever `B_hat` happens to sit within a
tolerance. A per-cell rule would have flared the source and plenum cells
(`B_hat` ~ 0.91 at z = 0) and desynced them from the SCALAR `Rp` the cathode,
puff and anode-flanking sites read. Verified at review that the cathode (cell
1), plenum (0), gap (2–5), puff (9) and anode-flanking (5, 6) cells all read
18.415 EXACTLY, so the scalar and vector cannot disagree.

Gates at the merge tip, each with an in-process import-provenance assertion:
smoke exit 0 at 119 cases with all five compiled-kernel equivalence blocks
LIVE; `verify_sim1d_k2_dvm.py` 121/121; `verify_sim1d_edt.py` 31/31;
`verify_sim1d_r3_boundary.py` exit 0; `declm_block_gate.py` 35 checks, 0
failed; `sgfs_census.py --assert-clean` (at commit 48be9a4, retired 2026-09-03) PASS; `batch11_restart_citations.py` (at commit 48be9a4, retired 2026-09-03)
58 cites PASS; `m1_verdict_invariance.py --self-test` 10 cases, 0 failures.

**2026-08-31 — IDENTITY-ONLY ROTATION (the [m1-a1-arming] cathode-jet arming
keys). NOTHING WAS RECAPTURED.** Merge `03389f7` added TWO `input_dict` keys —
`neutral_jet_arm_current_A = 0.0` and `neutral_jet_disarm_current_A = 0.0` —
both inert at their shipped values: `arm = 0` declares NO arming criterion, the
resolver returns its presence gate false, and the latch those keys describe is
never constructed, never evaluated and never reachable by any consumer. The
golden therefore keeps the pre-member behaviour bit for bit and only its config
identity moves. The rotation commit `d0b6f03` regenerated BOTH references in the
same event, the reviewer authoring it from each reference's own expression:

| reference | before | after |
|---|---|---|
| `golden_digest_4k.json` `config_identity` | `c22691e5a4f0ecf75edf7db2bca8b43851e3ee8eb25749974e49d0cc194475c8` | `ed352561a09c442c0e9ee5a592f3ce55744cb25d0110af9774a60eee10217d37` |
| `golden_digest_4k.json` checkpoints 0/1000/2000/3000/4000, final digest, `steps`, `cells`, `fields_per_cell`, `checkpoint_interval`, `final_time` | — | UNCHANGED (final digest `cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2`, `final_time` 0.0004714055010197914) |
| `production_discharge.json` params | 254 keys | 256 keys: the two ADDED at 0.0/0.0, 0 removed, 0 moved |
| `production_discharge.json` flags | 53 keys | UNCHANGED at 53 — both keys are `input_dict`; `saves` 2620, `cells` 72, `fields_per_cell` 8 and the 16-field `summary` all CARRIED |
| sidecar `sha256` | `8776b13dcfc349e2e2adf12ce429ecdd0ee40a15a0c5fdbab50e5c0dc0ddda64` | `93a41a1c4fc4010fc3c0d57c6527f1de7e5b8c21d145a9b791eeca445068c9f0` |
| digest reference `sha256` | `f5fe1bb29853ff194a0b06457d9fe8da46c1da4ce29f3faa8ce2e91fea6cdcbf` | `71150c0483357f1adace777b8250fad02883c5a048190b889c2211e6c38bf46c` |
| NPZ `sha256` | `2c02ccd882261a0b01e0a1e8f0e313113b8431fadd526fa9d4859694e5704306` | UNCHANGED (the file is not rewritten) |

Proof that the move is the two added keys and nothing else, verified at the
merge tip before anything was written: a **STRIP-2 CONTROL** removing both keys
and recomputing through the gate's OWN expression reproduces `c22691e5…` bit for
bit, and a **WRONG-VALUE CONTROL** declaring `arm = 50 A` instead of the inert
`0.0` gives `e4b519f2…`, so the control discriminates rather than merely
agreeing. Field-level: exactly one field moved in each reference
(`config_identity`; `params`), with 0 removed, 0 flags touched and 0 surviving
values changed. `config_snapshots.json` regenerated at the merge tip is
byte-identical to the merged file (`params=256, flags=53, cases=4`), and route
identity across all four representative routes moves by exactly these two
additions.

**`verify_sim1d_edt.py`'s G1 pin needed NO treatment at this rotation**, unlike
the previous one. That gate measures the golden identity against a FIXED pre-edt
baseline, so any key added downstream of it must be stripped for the control to
reach that baseline; the member extended G1's strip inventory to cover these two
keys rather than re-baselining the pin. Keeping the baseline fixed is what
preserves the property the gate exists to pin — that the baseline is the config
with the edt keys REMOVED, not merely some earlier config — at the cost of
requiring that inventory to grow with every future config addition. G1 passes at
the rotated tip.

Gates at the rotated tip `d0b6f03`, each run with an in-process import-provenance
assertion — `cablp.__file__` resolving inside this checkout, printed with
`KERNEL_ID` in the same process as the gate, and for the compiled leg the
compiled path asserted BEFORE the run: COMPILED golden `baseline verify OK:
saves=2620, exact=True, max_rel=0.000e+00, max_abs=0.000e+00,
time_max_abs=0.000e+00 s` (`KERNEL_ID` `cython/_cathode_kernels_cy/tierA+csda`);
PURE 4k digest leg `digest gate OK: steps=4000,
digest=cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2,
exact=True` against the ROTATED reference, `kernels=pure`; smoke exit 0 at 119
cases with all five compiled-kernel equivalence blocks LIVE;
`verify_sim1d_k2_dvm.py` 119/119 including the seven new JA latch gates;
`verify_sim1d_edt.py` 31/31; `verify_sim1d_r3_boundary.py` exit 0 (`boundary
unit gates: OK`); `declm_block_gate.py` 35 checks, 0 failed; `sgfs_census.py
--assert-clean` PASS; `batch11_restart_citations.py` 52 cites PASS;
`m1_verdict_invariance.py --self-test` 7 of 7.

The stance file `g1atrim.toml` is untouched by this event; the golden's
trajectory is the 2026-08-28 capture's, unchanged.

**2026-08-31 — IDENTITY-ONLY ROTATION, BY REMOVAL (the
[legacy-boundary-retirement] keys). NOTHING WAS RECAPTURED.** Merge `1fc05c9`
retired two `input` keys and their whole code chains — `characteristic_boundary`
(an `input_flags` key, shipped default **True**) and
`neutral_kinetic_dvm_tn_feedback` (an `input_dict` key, shipped default
`False`). What the retirement deleted is the NON-DEFAULT path in both cases, so
the golden — which runs the defaults — is untouched: this is the first rotation
since D3 in which the config surface SHRINKS, and the sidecar therefore LOSES
fields rather than gaining them. The rotation commit `3e2d5a9` regenerated
THREE pinned identities in one event, the reviewer authoring it from each
pin's own expression:

| reference | before | after |
|---|---|---|
| `golden_digest_4k.json` `config_identity` | `c5f7f3f7347ce2d62fd354d73d472fb62f4f17cdef53fa664619ac2dfc74b760` | `c22691e5a4f0ecf75edf7db2bca8b43851e3ee8eb25749974e49d0cc194475c8` |
| `golden_digest_4k.json` checkpoints 0/1000/2000/3000/4000, final digest, `steps`, `cells`, `fields_per_cell`, `checkpoint_interval`, `final_time` | — | UNCHANGED (final digest `cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2`, `final_time` 0.0004714055010197914) |
| `production_discharge.json` params | 255 keys | 254 keys: `neutral_kinetic_dvm_tn_feedback` REMOVED, 0 added, 0 survivor values moved |
| `production_discharge.json` flags | 54 keys | 53 keys: `characteristic_boundary` REMOVED, 0 added, 0 survivor values moved; `saves` 2620, `cells` 72, `fields_per_cell` 8, the 16-field `summary`, `description` and `result_format` CARRIED |
| `verify_sim1d_edt.py` `BASE_CONFIG_IDENTITY` | `21a9b4764df68bc9c201d5ea11589223358bd9ca19d2801f82ac7bd75db632c3` | `7f2eadcb0b0610fa1ab6c8cd4fe174d61227ce1e2973e1d146cbbb1e91993d87` |
| sidecar `sha256` | `9046adf879fa343d0c9a092ad30d1bfea544aecf0c49fea139630132336f4eaa` | `8776b13dcfc349e2e2adf12ce429ecdd0ee40a15a0c5fdbab50e5c0dc0ddda64` |
| digest reference `sha256` | `e041c30b03fc6dd9309cceea86b34630b9ca6ff4a847d9b77e0850b7d4d8e0be` | `f5fe1bb29853ff194a0b06457d9fe8da46c1da4ce29f3faa8ce2e91fea6cdcbf` |
| NPZ `sha256` | `2c02ccd882261a0b01e0a1e8f0e313113b8431fadd526fa9d4859694e5704306` | UNCHANGED (md5 `fd8ac896ccba10c66a7c18ec609ec48e` before and after; the file is not rewritten) |

**A third pin rotates here, which is new.** `verify_sim1d_edt.py`'s G1 strip
control pins the golden identity WITHOUT the electron-drift member's three
keys. That quantity moves for the same reason the two golden references do, so
leaving it behind would have failed G1 at this tip for a reason that has
nothing to do with the electron-drift member. It is computed through the strip
control G1 itself runs, not through either golden expression — the three
expressions legitimately carry three different identities, and a control
computed through the wrong one matches none of them.

Proof that the move is the two retired keys and nothing else, all verified at
the merge tip BEFORE anything was written. A **RESTORE-2 CONTROL** puts both
keys back at the defaults they carried at the base commit
(`characteristic_boundary=True`, `neutral_kinetic_dvm_tn_feedback=False`) and
recomputes through the gate's OWN expression: it reproduces `c5f7f3f7…`
bit for bit. A **WRONG-VALUE CONTROL** restoring the flag at `False` instead
gives `894afe9c…`, so the control discriminates rather than merely agreeing.
The same restore, applied on top of the edt strip-3, reproduces `21a9b476…`
exactly. For the sidecar the load-bearing half of a REMOVAL rotation is the
SURVIVOR check rather than the count: a removal that also perturbed a surviving
key would show up as a smaller diff, not a larger one, so the tool refuses to
write unless `gained == []`, `lost` is exactly the entitled pair, and no
surviving key's value moved. Its guard was exercised against a NEGATIVE
CONTROL — an injected `b_presheath_length` 2.0 → 1.0 — which it refused with
`REFUSING TO WRITE: the manifest moved beyond the two retired keys` and exit 1.
Route identity across all four representative routes moves by exactly these two
removals, and `config_snapshots.json` regenerated at the merge tip is
byte-identical to the merged file (`params=254, flags=53, cases=4`).

**The deletion was dead code, which is the cardinal claim of a retirement
pass, and it was verified independently rather than inferred from the golden.**
The four electron-energy electrode channels
(`beam_power_deposition + beam_ionization_cost + cathode_surface_loss +
anode_e_sheath_loss`) were evaluated at the base commit with
`characteristic_boundary=True` — the surviving path, explicitly selected —
and at this tip where the flag no longer exists. The rows are BIT-IDENTICAL
between the two (row hash `f16b421b89d0e7be` on both, `cathode_surface_loss`
min −1.135568e-05, `anode_e_sheath_loss` min −4.608156e-07). Selecting the
retired path at base instead gives a different answer (`cathode_surface_loss`
max +2.560624e-05, four cells with a net POSITIVE electron deposit). So the
retirement removed a path rather than changing one.

Gates at the rotated tip `3e2d5a9`, each run with an in-process
import-provenance assertion — `cablp.__file__` resolving inside this checkout,
printed with `KERNEL_ID` in the same process as the gate, and for the compiled
leg the compiled path asserted BEFORE the run: COMPILED golden `baseline verify
OK: saves=2620, exact=True, max_rel=0.000e+00, max_abs=0.000e+00,
time_max_abs=0.000e+00 s` (`KERNEL_ID`
`cython/_cathode_kernels_cy/tierA+csda`); PURE 4k digest leg `digest gate OK:
steps=4000, digest=cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2,
exact=True` against the ROTATED reference, `kernels=pure`; smoke exit 0 at 119
cases with the registry byte-identical to base and all five compiled-kernel
equivalence blocks LIVE; `verify_sim1d_k2_dvm.py` 112/112 (G9 retired — both
keys it set are now refused by the generic unknown-key check, so it would have
passed coincidentally rather than off the specific guard it was written for);
`verify_sim1d_edt.py` 31/31 (26 plus the five arms the bundled G12 widen adds);
`declm_block_gate.py` 35 checks, 0 failed; `sgfs_census.py --assert-clean`
PASS; `batch11_restart_citations.py` 52 cites PASS;
`m1_verdict_invariance.py --self-test` 7 of 7.

The stance file `g1atrim.toml` is untouched by this event; the golden's
trajectory is the 2026-08-28 capture's, unchanged.

**2026-08-31 — IDENTITY-ONLY ROTATION (the [ue-pressure-work] drift-transport
keys). NOTHING WAS RECAPTURED.** Merge `935dba3` ([ue-pressure-work], the
electron drift-transport and EMF-work operator) added ONE `input_flags` key and
TWO `input_dict` keys to the package surface — `electron_drift_transport =
False`, `electron_drift_charge_death = "cell_1"`,
`electron_drift_anode_handshake = "sheath_row_closes_all"` — all default-inert:
the operator is off, and both convention keys are REFUSED at a non-default value
while it is off, so the golden cannot reach them. The config identity therefore
moved for the three added keys and for nothing else; the rotation commit
`b3d201d` regenerated BOTH references in one event, the reviewer authoring it
from each reference's own expression:

| reference | before | after |
|---|---|---|
| `golden_digest_4k.json` `config_identity` | `21a9b4764df68bc9c201d5ea11589223358bd9ca19d2801f82ac7bd75db632c3` | `c5f7f3f7347ce2d62fd354d73d472fb62f4f17cdef53fa664619ac2dfc74b760` |
| `golden_digest_4k.json` checkpoints 0/1000/2000/3000/4000, final digest, `steps`, `cells`, `fields_per_cell`, `checkpoint_interval`, `final_time` | — | UNCHANGED (final digest `cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2`, `final_time` 0.0004714055010197914) |
| `production_discharge.json` params | 253 keys | 255 keys: the two ADDED, 0 removed, 0 moved |
| `production_discharge.json` flags | 53 keys | 54 keys: the one ADDED, 0 removed, 0 moved; `saves` 2620, `cells` 72, `fields_per_cell` 8, the 16-field `summary`, `description` and `result_format` all CARRIED |
| sidecar `sha256` | `f7bd286409b15945da3f21c1a8d672bde7876d08dad81593700beef2673b3f12` | `9046adf879fa343d0c9a092ad30d1bfea544aecf0c49fea139630132336f4eaa` |
| digest reference `sha256` | `e05b5dad3029bd65af41205f76ff80fa5f213815b23a9167fe9363afd56fe877` | `e041c30b03fc6dd9309cceea86b34630b9ca6ff4a847d9b77e0850b7d4d8e0be` |
| NPZ `sha256` | `2c02ccd882261a0b01e0a1e8f0e313113b8431fadd526fa9d4859694e5704306` | UNCHANGED (md5 `fd8ac896ccba10c66a7c18ec609ec48e` before and after; the file is not rewritten) |

Proof that the move is the three keys and nothing else: a STRIP-3 CONTROL at the
merge tip removes `params['electron_drift_charge_death']`,
`params['electron_drift_anode_handshake']` and
`flags['electron_drift_transport']`, recomputes through the gate's OWN
expression (`config_identity(*build_baseline_config(DIGEST_PARAM_OVERRIDES))`)
and reproduces `21a9b476…` bit for bit (`params 255 -> 253`, `flags 54 -> 53`);
and a full 4,000-step pure trajectory at the merge tip agreed with the committed
reference at every checkpoint BEFORE the rotation was written. The rotation tool
refuses to write unless each non-identity field is proven unmoved field by field
first, so the on-disk diff is exactly four lines — one `config_identity` swap and
three key insertions. The sidecar params and flags were regenerated
programmatically (`build_baseline_config()` -> `json.dumps(..., sort_keys=True)`),
never hand-edited.

The event adds keys in BOTH namespaces: one FLAGS key (the arming flag) and two
PARAMS keys (the declared conventions it gates), which is why this rotation moves
`params` and `flags` together where B6 moved only `flags` and B4 only `params`.
Route identity across the four representative routes (`production_golden`,
`compare_sim1d_es1`, `run_m6_point_es1_sgp3649_defaults`,
`run_mechanism_ladder_es1_defaults`) moves by exactly these three lines each,
with no other added, removed or value-moved key on any route, and each route's
digest with the three keys stripped MATCHES its base-commit digest.
`config_snapshots.json` was regenerated at the merge tip and is byte-identical to
the merged file (`params=255, flags=54, cases=4`).

Gates at the rotated tip `b3d201d` (this entry is the markdown-only commit on top
of it), each run with an in-process import-provenance assertion — `cablp.__file__`
resolving inside this checkout, printed with `KERNEL_ID` in the same process as
the gate, and for the compiled leg the compiled path ASSERTED before the run
rather than read after it: COMPILED golden `baseline verify OK: saves=2620,
exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s
(rtol=1.0e-09, atol=0.0e+00)` (`KERNEL_ID`
`cython/_cathode_kernels_cy/tierA+csda`); PURE 4k digest leg `digest gate OK:
steps=4000, digest=cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2,
exact=True` against the ROTATED reference, `kernels=pure`; smoke exit 0 with all
five compiled-kernel equivalence blocks LIVE (meanfield, coverage, landau,
emitting_area, initial_profile — bit-identical final state on each);
`verify_sim1d_k2_dvm.py` 113/113; `verify_sim1d_edt.py` 26/26 (the new member's
own suite); `declm_block_gate.py` 35 checks, 0 failed; `sgfs_census.py
--assert-clean` PASS; `batch11_restart_citations.py` 52 cites PASS;
`m1_verdict_invariance.py --self-test` 7 of 7.

The stance file `g1atrim.toml` is untouched by this event; the golden's
trajectory is the 2026-08-28 capture's, unchanged.

**2026-08-31 — IDENTITY-ONLY ROTATION (the B6 baffle-interception key). NOTHING
WAS RECAPTURED.** Merge `a8590e3` ([dvm-b6-baffle-interception], the annular
baffles acting on the transient DVM's annulus) added ONE `input_flags` key to
the package surface — `neutral_kinetic_dvm_baffles = False` — default-inert:
the channel is off, and it is read only under `neutral_model = "kinetic_dvm"`,
which the golden does not run. The config identity therefore moved for the
added key and for nothing else; the rotation commit `7a03cd9` regenerated BOTH
references in one event, the reviewer authoring it from each reference's own
expression:

| reference | before | after |
|---|---|---|
| `golden_digest_4k.json` `config_identity` | `8974b3ec46a944947e6f080ef48e973ecaaf51163e979d14b874fdb02f57563c` | `21a9b4764df68bc9c201d5ea11589223358bd9ca19d2801f82ac7bd75db632c3` |
| `golden_digest_4k.json` checkpoints 0/1000/2000/3000/4000, final digest, `steps`, `cells`, `fields_per_cell`, `checkpoint_interval`, `final_time` | — | UNCHANGED (final digest `cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2`, `final_time` 0.0004714055010197914) |
| `production_discharge.json` flags | 52 keys | 53 keys: the one ADDED, 0 removed, 0 moved; `params` 253 unchanged; `saves` 2620, `summary`, `cells` 72, `description`, `result_format` CARRIED |
| sidecar `sha256` | `c1cd0f639d5b8b3cd93ea0ca5476a5af0734ed62935f88324ea9d6dd79fb3f0c` | `f7bd286409b15945da3f21c1a8d672bde7876d08dad81593700beef2673b3f12` |
| digest reference `sha256` | `f7403cbc777780e120a50bd3cec5d56037f8c14f528888b60c16a8c2f5bffa64` | `e05b5dad3029bd65af41205f76ff80fa5f213815b23a9167fe9363afd56fe877` |
| NPZ `sha256` | `2c02ccd882261a0b01e0a1e8f0e313113b8431fadd526fa9d4859694e5704306` | UNCHANGED (md5 `fd8ac896ccba10c66a7c18ec609ec48e` before and after; the file is not rewritten) |

Proof that the move is the one key and nothing else: a STRIP-1 CONTROL at the
merge tip removes `flags['neutral_kinetic_dvm_baffles']`, recomputes through
the gate's OWN expression
(`config_identity(*build_baseline_config(DIGEST_PARAM_OVERRIDES))`) and
reproduces `8974b3ec…` bit for bit (`flags 53 -> 52`, `params 253` untouched);
and every checkpoint agreed with the committed reference before the rotation
was written. The rotation tool refuses to write unless each non-identity field
is proven unmoved field by field first, so the on-disk diff is exactly two
lines — one `config_identity` swap and one `"neutral_kinetic_dvm_baffles":
false` insertion. The sidecar flags were regenerated programmatically
(`build_baseline_config()` -> `_json_safe` -> `json.dumps(..., sort_keys=True)`),
never hand-edited.

The KEY is a FLAGS key, not a params key: it is filed beside the fluid
`neutral_baffles` it gates on rather than beside the two DVM jets, which is why
this rotation moves `flags` where the B4 rotation moved `params`. Route
identity across all seven representative routes moves by exactly one line each
(`+ flags:neutral_kinetic_dvm_baffles = False`), with no other added, removed
or value-moved key on any route.

Gates at the rotated tip `7a03cd9` (this entry is the markdown-only commit on
top of it), each run with an in-process import-provenance assertion —
`cablp.__file__` resolving inside this checkout, printed with `KERNEL_ID` in
the same process as the gate: COMPILED golden `baseline verify OK: saves=2620,
exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s`
(`KERNEL_ID` `cython/_cathode_kernels_cy/tierA+csda` attested in-process, and
asserted before the run rather than read after it); PURE 4k digest leg
`digest gate OK: steps=4000,
digest=cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2,
exact=True` against the ROTATED reference, `kernels=pure`; smoke exit 0 with
all five compiled-kernel equivalence blocks LIVE (meanfield, coverage, landau,
emitting_area, initial_profile — bit-identical final state on each);
`verify_sim1d_k2_dvm.py` 113/113 including the three new B6 gates and G30–G32;
`declm_block_gate.py` 35 checks, 0 failed; `sgfs_census.py --assert-clean`
PASS; `batch11_restart_citations.py` 52 cites PASS.

The stance file `g1atrim.toml` is untouched by this event; the golden's
trajectory is the 2026-08-28 capture's, unchanged.

**2026-08-30 — IDENTITY-ONLY ROTATION (the B4 anode-jet keys). NOTHING WAS
RECAPTURED.** Merge `e0beed5` ([dvm-b4-anode-jet], the DVM anode recycle
triple) added four `input_dict` keys to the package surface —
`neutral_kinetic_dvm_anode_jet = False`, `neutral_kinetic_dvm_anode_jet_R_N =
0.63`, `neutral_kinetic_dvm_anode_jet_R_E = 0.41`,
`neutral_kinetic_dvm_anode_jet_T_launch_eV = None` — all default-inert (the
channel is off, and read only under `neutral_model = "kinetic_dvm"`, which the
golden does not run). The config identity therefore moved for the added keys
and for nothing else; the rotation commit `141ff2c` regenerated BOTH references
in one event, the reviewer authoring it from each reference's own expression:

| reference | before | after |
|---|---|---|
| `golden_digest_4k.json` `config_identity` | `fcc61568a6b11673110cc22feaacf2e8b6f496e97a24fc0d302802d9378509de` | `8974b3ec46a944947e6f080ef48e973ecaaf51163e979d14b874fdb02f57563c` |
| `golden_digest_4k.json` checkpoints 0/1000/2000/3000/4000, final digest, `steps`, `final_time` | — | UNCHANGED (final digest `cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2`) |
| `production_discharge.json` params | 249 keys | 253 keys: the four ADDED, 0 removed, 0 moved; `flags` 52 unchanged; `saves` 2620, `summary.final_time` 0.026186806122473337, `summary.steps` 62612 CARRIED |
| sidecar `sha256` | `18d065b70d12a5187b3e240bb2f0eae1aec0a09b1c0076954c1cbe8e7a6c2176` | `c1cd0f639d5b8b3cd93ea0ca5476a5af0734ed62935f88324ea9d6dd79fb3f0c` |
| digest reference `sha256` | — | `f7403cbc777780e120a50bd3cec5d56037f8c14f528888b60c16a8c2f5bffa64` |
| NPZ `sha256` | `2c02ccd882261a0b01e0a1e8f0e313113b8431fadd526fa9d4859694e5704306` | UNCHANGED (md5 `fd8ac896ccba10c66a7c18ec609ec48e` before and after) |

Proof that the move is the four keys and nothing else: a strip-four-keys
control at the merge tip reproduces `fcc61568…` bit-for-bit through the gate's
own expression (`params 253 -> 249`), and every checkpoint agrees with the
committed reference before the rotation. The sidecar params were regenerated
programmatically (`build_baseline_config()` -> `_json_safe` -> `json.dumps(...,
sort_keys=True)`), never hand-edited. Gates at the rotated tip: compiled golden
`saves=2620, exact=True, max_rel=0.000e+00` (`KERNEL_ID`
`cython/_cathode_kernels_cy/tierA+csda` attested in-process); pure 4k digest
`exact=True` against the rotated reference; smoke exit 0 with the five
equivalence blocks live. The stance file `g1atrim.toml` is untouched by this
event; the golden's trajectory is the 2026-08-28 capture's, unchanged.

**2026-08-30 — NON-RECAPTURE STANCE-FILE EVENT: `g1atrim.toml` migrated to
declaration-block FORM. NOTHING WAS RECAPTURED.** This entry exists because the
stance file is a golden input and this record is where a reader looks when it
changes; it is filed here so that a future `git log` on
`scripts/stances/g1atrim.toml` showing a large diff on this date does not read
as an unrecorded stance change.

**No value moved.** The file was rewritten from a flat delta into two flat
tables plus four `[models.*]` declaration blocks — `beam_tail_closure` (22
members), `cathode_surface_recycle` (6), `anode_surface_recycle` (5) and
`initial_neutral_state` (12). A block is an INVENTORY, so it states members the
flat stance never named; every one of those was written AT ITS `default_config()`
VALUE, which is what makes the projection inert. The two neutral-closure
families are undeclarable at this stance (their selectors are not engaged: it
runs `neutral_model = "moment"`, `neutral_momentum_radial = "uniform"`) and stay
implicit. See `cablp/solvers/_sim1d/CONFIG_DECLARATIONS.md` for the form.

**No fixture, sidecar or digest reference was touched.** `scripts/baselines/` is
untouched; there was no `--capture`; `steps`, `saves`, `final_time`, the NPZ and
sidecar `sha256` and both digest hashes all stand as the 2026-08-28 entry below
records them.

MEASURED EVIDENCE (all at the migration tip, each with in-process import
provenance asserting the worktree and, where the compiled path is claimed,
`KERNEL_ID`):

| gate | quantity | result |
|---|---|---|
| G0 `g1atrim_blockform_delta_check.py` (at commit 48be9a4, retired 2026-09-03) | the stance DELTA before vs after | 35 base entries preserved, 0 violations; 32 keys added, 0 not at their config default |
| G1 `declm_route_identity.py` | resolved surface of 7 representative routes | **ALL ROUTES IDENTICAL** — `golden` `413c8d0c…`, `stance_g1atrim` `261d5469…`, `m6_es1` `c084b91b…`, `default` `cbf51a31…`, `ka1c` `7d1d7287…`, `k2_dvm`/`b0c` `e8bd6263…` |
| G3 `golden_digest_gate.py` (pure) | golden config identity + every accepted step to 4,000 | `config_identity=fcc61568a6b11673110cc22feaacf2e8b6f496e97a24fc0d302802d9378509de` UNCHANGED; `digest gate OK: steps=4000, digest=cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2, exact=True` |
| G4 `baseline_sim1d.py --verify` (compiled) | the full saved trajectory | see the transcript line below |
| G5 `smoke_sim1d.py` | assertion suite, built extension | exit 0, five compiled-kernel-equivalence cases `ok … final state bit-identical` |

The G1 row is the load-bearing one: the `golden` route IS
`build_baseline_config()`, so its digest holding fixed says the golden's own
configuration did not move, independently of running the golden.

G4 transcript (compiled path, `CABLP_COMPILED_KERNELS=1`):

```
[gaterun] cablp.__file__ = <worktree>/cablp/__init__.py
[gaterun] KERNEL_ID = cython/_cathode_kernels_cy/tierA+csda
provenance: kernels=cython/_cathode_kernels_cy/tierA+csda
baseline verify OK: saves=2620, exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s (rtol=1.0e-09, atol=0.0e+00)
```

`saves=2620` is the sidecar's own count, unchanged. `max_rel` and `max_abs` are
both exactly zero: the trajectory is bit-identical, not merely within tolerance.

**2026-08-28 — THE R3-TIP RECAPTURE: the six-member R3 reimplementation window
closed in one anchor event (AUTHORIZED recapture; the R3 build set was ratified
member by member across 2026-08-27 and the Tier C continuity pair passed all
seven of its pre-registered gates before this member was dispatched).** The R3
build set is a REIMPLEMENTATION programme at fixed physics intent, and its
members are, in merge order:

| member | staging commit | what it changed |
|---|---|---|
| R3 sub-event 1 — constants unification | `5e253da` | six truncated physical constants unified UP to full CODATA |
| R3 sub-event 1b — `Q_ie` mass ratio | `89a7c9c` | `H_e_mass_ratio` retired for a DERIVED `He_e_mass_ratio`; `Q_ie` divides by the true helium-4/electron ratio (coefficient ×1.0066473) |
| smoke-eqfix | `39cb395` | smoke-suite case attribution and child pinning; no solver float |
| small batch 1 | `508462f` | flatten stragglers, golden transcript provenance, afterglow H retirement |
| 1d `vbar` mass unify + H quarantine | `fed56dd` | the last neutral-side mass-number reconstruction removed (the two `vbar` sites genuinely disagreed 0.330 % before it) |
| small batch 2 | `0144808` | seven inert/dead-code rows, incl. the `A_R318` single-inserted-duplicate repair |
| R3 sub-event 2 — lane march | `e95bb21` | `deposit_beam`'s tail legs marched as numpy lanes with an exact-FMA reconstruction; BITWISE-0 against the scalar route |
| `[fma-underflow-fence]` | `80ef40d` | the lower Dekker fence `FMA_ARRAY_MIN_ABS`; structurally inert |

plus THIS member (anchor hardening + the recapture itself), which added the
`fma_array` signed-zero docstring caveat and the array-path leg of
`scripts/interp_bitexact_gate.py`, and changed no arithmetic.

**Nothing in the CONFIGURATION moved, and that is the defining feature of this
event.** `g1atrim.toml` is untouched; `default_config()` is untouched; no
config key was added, removed or re-valued. The consequences are worth stating
plainly, because they invert a pattern every previous entry in this record
shares:

| quantity | before | after |
|---|---|---|
| `steps` | 62,613 | **62,612** (−1) |
| `saves` | 2,620 | **2,620** (unchanged) |
| `final_time` | 2.618682e-02 s | **2.618681e-02 s** — the dynamic `t_end`, still reached |
| NPZ `sha256` | `77c0543cbbee6038756b79fcb904c8849c57c6aab9929bc100a0872c7f6ed93d` | `2c02ccd882261a0b01e0a1e8f0e313113b8431fadd526fa9d4859694e5704306` |
| sidecar `sha256` | `18587f7190bafcf10d56b500407401e2528c3df6ec378798cbf86160abd2a87c` | `18d065b70d12a5187b3e240bb2f0eae1aec0a09b1c0076954c1cbe8e7a6c2176` |
| 4k trajectory digest | `3f2424e3b9f6954a736c84041c2737fb2624778043b1fab028c725ed1cc4991f` | `cb54b74a34cbee055612d404abb44ba4522bea11316044556fa43c83a75b2ae2` |
| digest `config_identity` | `ec8ba03db7e061948a8d438ed027747cccaef05cf8171225e259dd25e9871157` | **UNCHANGED** — `ec8ba03d…` |

**The trajectory digest moves while the config identity does NOT, and that
asymmetry is the load-bearing evidence, not a curiosity.** Every recapture
before this one moved a configuration key, so both halves rotated together (or,
in three of them, only the config identity did over an unchanged trajectory).
Here the pair separates for the first time, and it separates in exactly the
direction a constants reimplementation predicts: the configuration the campaign
runs is bit-identical, and the floats the model computes from it are not.
A reader who sees only the moved NPZ `sha256` should read this row before
concluding a stance changed. It did not.

**`config_snapshots.json` was NOT regenerated, and that is a verified result
rather than an omission.** `scripts/audit_sim1d_configs.py` verifies the
committed artifact (`cablp/solvers/_sim1d/config_snapshots.json`, `sha256`
`b8ecf6c1…`) CLEAN at this tip: `params=257, flags=51, cases=4`, all four case
digests reproducing, `production_golden` still `21409d1a…` and
`manifest_sha256` still `d2fd9e26…`. `parameter_count` and `flag_count` did
not move because R3 added and removed no config key — the same fact the
unchanged `config_identity` states from the other side, measured by a second
independent instrument.

**Health, and the continuity read at the gate.** The sidecar reports
`finite: true`, `Te_max` 27.897 → **27.805 eV**, `Ti_max` 8.2080 → 8.2077 eV,
`n_max` 1.41548e13 → **1.41964e13 cm^-3**, `nn_max` 3.54163e13 → 3.53955e13
cm^-3. The phase census (8 / 11 / 2000 / 600 / 1) and the save count are both
unchanged, so the fixture covers exactly the same cycle it did before. The
150,000-step tripwire did not fire and its ~2.4× margin is unmoved. This is a
health check, not a physics verdict: the physics verdict is the Tier C
continuity pair (measured 2026-08-27), which measured this window at the
production operating point against the R2-tip parent `bc0a515` and passed all
seven gates — all 15 printed stage-(ii) dev/σ entries identical between arms,
plateau −0.0027 %, `t90` Δ 0.00 ms, ledger closure identical to all printed
digits.

**Capture evidence.** The NPZ fixture was captured TWICE from clean separate
processes to temporary paths, strictly serially per the serial-golden rule, and
compared BEFORE either was installed: NPZ and JSON sidecar byte-identical (the
two `sha256` values above) and raw-bitwise identical over all three arrays —
`phase` identical at raw bytes, `time` **0 differing of 2,620** and `y`
**0 differing of 1,509,120** at `uint64`. The digest reference was regenerated
in the same event as the protocol requires, also twice from clean separate
processes, byte-identical (`sha256` `3a032eae…` both). `--verify` prints:

```
baseline verify OK: saves=2620, exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s (rtol=1.0e-09, atol=0.0e+00)
```

**KERNEL PATH — the fixture was captured PURE and verified on BOTH paths, which
is the stronger form and a return to how recaptures were taken before
2026-08-25.** That event captured the NPZ compiled for cost reasons and
disclosed the departure; here the pure capture cost 37 min 56 s, comfortably
inside the harness lifetime, so no such trade was needed. Both `--verify` runs
print `exact=True, max_rel=0.000e+00` against the pure-captured fixture, with
kernel provenance probed IN-PROCESS on each (`pure`, and
`cython/_cathode_kernels_cy/tierA+csda` under `CABLP_COMPILED_KERNELS=1`). The
digest reference was likewise captured pure and reads `kernel_provenance:
"pure"`, with the pure digest gate printing
`digest gate OK: steps=4000, digest=cb54b74a…, exact=True` against it. So
pure-vs-compiled bit-exactness is RE-EVIDENCED at this tip on the full cycle
rather than assumed — which matters more than usual here, because the lane
march is the first member to give the two paths structurally different
inner loops (lane legs 13,424 pure / 0 compiled).

**The corpus fixture rotated one member earlier and is NOT part of this
event.** `scripts/data/deposit_beam_reference.npz` was re-captured at unified
constants during R3 sub-event 1 (`77b010e1…` → `e70396e5…`, Tom's 2026-08-27
ruling); it is untouched here and `deposit_beam_reference.py --verify` reads
506 entries, 0 differing of 206,218 at this tip.

**Two anchors, and where the line now falls.** Pre-R3 BIT-LEVEL claims are
HISTORICAL as of this recapture and are reproducible only at
`r2-tip-2026-08-27`, with its environment lockfile. Score-level physics
transfers across the boundary — that is what the Tier C pair established and is
the only sense in which results carry across it. This is the same two-anchor
relationship the 2026-08-16 desktop migration created, for the same reason: the
floats changed underneath a model whose physics did not.

**Gates run at this recapture** (worktree branch `worktree-agent-a8eaac447400d6525`,
all from the worktree's own root with `PYTHONPATH` pointed at it and
`cablp.__file__` asserted in-process): `scripts/smoke_sim1d.py` exit 0 with the
compiled-equivalence block live; `deposit_beam_reference.py --verify` 0
differing; `interp_bitexact_gate.py` GATE OK on BOTH legs — the scalar
`tw`/`twion` arms 0 differing and the new array-path corpus leg 0 flipped of
17,024 walkers with 210 of 364 batches genuinely lane-marched;
`baseline_sim1d.py --verify` `exact=True` pure AND compiled;
`golden_digest_gate.py` `exact=True` pure against the newly captured reference.

**2026-08-26 — `[afterglow-dt-cost]` ADOPTION: the exemption hysteresis band
and the accelerated dt-growth re-approach flipped to armed config defaults
(AUTHORIZED recapture; Tom-ruled adoption after all three pre-registered gates
passed — registration `AFTERGLOW_DT_COST_REGISTRATION_2026-08-26`, A3
adjudication `AFTERGLOW_DT_COST_A3_ADVISORY_2026-08-26`, both in the local
docs repo).** Unlike every recapture before it, **the stance file did not
move**: the two keys are `default_config()` defaults, and the fixture picks
them up because it layers the stance over the shipped defaults. The event is
still a stance change and a golden re-anchor, for the reason the protocol
cares about — the configuration the campaign runs changed.

| key | old | new | class |
|---|---|---|---|
| `surface_loss_floor_exempt_exit_rtol` (config default) | `0.0` (band off) | `0.1` | **DERIVED** — A/B-selected numerics knob |
| `dt_growth_recovery_patience` (config default) | `0` (accelerator off) | `4` | **DERIVED** — A/B-selected numerics knob |

Values, classes, the corrected two-prong A3 bar and the near-floor resolution
bracket the band buys: `production_stance_provenance.md`. The claim under test
was COST, not correctness; `dt_growth_recovery_factor` did not move but is now
live at its own default, which the deprecation register records.

**What moved in the fixture.**

| quantity | before | after |
|---|---|---|
| `steps` | 94,044 | **62,613** (−31,431, −33.4 %) |
| `saves` | 2,620 | **2,620** (unchanged) |
| `final_time` | 2.618731e-02 s | **2.618682e-02 s** — the dynamic `t_end`, still reached |
| NPZ `sha256` | `60e11c6fa64ced1050e57feae788f7e0eb4f1cd9b7e072bc483e3695002a9b76` | `77c0543cbbee6038756b79fcb904c8849c57c6aab9929bc100a0872c7f6ed93d` |
| sidecar `sha256` | `074a682519fc9a4459b34829b549dc955ee3966b666af8763bdd775a0f9d22d5` | `18587f7190bafcf10d56b500407401e2528c3df6ec378798cbf86160abd2a87c` |
| 4k trajectory digest | `f1461a03f2677146cbd927a614e3fc575a2d0dfe8a9d1eaeed54ebe872ce4b80` | `3f2424e3b9f6954a736c84041c2737fb2624778043b1fab028c725ed1cc4991f` |
| digest `config_identity` | `567adf6b48bc8d0f3e92bd2818857e17ee5bc0308bfc9d28c18eeaef9e744b48` | `c56bd67dd782fde57ddc272abb00d271898d35fc9eb138d5f72acca46dec154d` |

**Both digest pairs move, and that is the authorized outcome**: arming the band
changes which cells the drain bound reads from the first step it binds, so this
is a trajectory-moving event, not a config-identity-only rotation. The digest
reference was recaptured with PURE kernels, matching the provenance the
committed reference records.

**Health, and the continuity read at the gate.** The sidecar reports
`finite: true`, `Te_max` 28.38 → **27.90 eV**, `Ti_max` 8.211 → 8.208 eV,
`n_max` 1.41557e13 → 1.41548e13 cm^-3. The phase census
(8 / 11 / 2000 / 600 / 1) and the save count are both unchanged, so the
fixture covers exactly the same cycle it did before at a third fewer steps. That is the
golden-side echo of the A2 continuity gate the runner passed at the production
point; it is a health check here, not a second physics verdict.

**The 150,000-step tripwire did not fire, and its margin improved without the
cap moving** — 1.6× → 2.4× headroom. The pin table row records the new ratio.

**Gates run at this recapture** (worktree `agent/dtc-adoption`, all from the
worktree's own `cablp/` with `PYTHONPATH` pointed at it):
`scripts/smoke_sim1d.py` exit 0; `baseline_sim1d.py --verify` COMPILED
(`CABLP_COMPILED_KERNELS=1`, kernel provenance probed in-process as
`cython/_cathode_kernels_cy/tierA+csda`) printing
`exact=True, max_rel=0.000e+00`; `golden_digest_gate.py` printing
`exact=True` against the newly captured reference on the pure path. The
fixture was CAPTURED pure and VERIFIED compiled, so pure-vs-compiled
bit-exactness is re-evidenced by this recapture rather than assumed.

**2026-08-25 — THE STANCE EVENT: the `plateau_multigroup` anomalous-heating
closure adopted, with its `C_R` re-trim, two armed physics flags and the
`V_bank` half of the twice-deferred 23b R4 deletion (AUTHORIZED recapture;
ratified 2026-08-25 (Tom), as amended the same day, plus the same session's
rulings that ADOPTED the selector and ARMED the stb fix).** Five
stance members moved in one event, and they are not five independent choices:
the selector is the physics change and the `C_R` re-trim is its consequence —
adopting the spectrum moves the drive, so the one drive-side fit knob was
re-trimmed ONCE, at ES1, drive-band target only, to put the plateau ratio back
on 1.000.

| key | old | new | class |
|---|---|---|---|
| `heating_anomalous_transport` (stance) | `"tail_walk"` | `"plateau_multigroup"` | selector — **DERIVED**, zero fitted parameters |
| `C_R` (stance) | `7.36` | `8.76` | **FITTED** — the ONE drive-side fit knob |
| `anode_sheath_full_debit` (stance flags) | *absent*, default `False` | `true` | ARM of a booking correction |
| `beam_deposition_in_heat_substep` (stance flags) | *absent*, default `False` | `true` | ARM of the `[stage-te-bias]` class-1 fix |
| `V_bank` (stance) | `177.843` | *row deleted* | rung-owned measurement; the row duplicated it |

Values, classes and the knob-assignment argument: `production_stance_provenance.md`.
The `T_s` half of the R4 deletion was NOT executed — see the note at the end of
this record.

**What moved in the fixture.**

| quantity | before | after |
|---|---|---|
| `steps` | 70,408 | **94,044** (+23,636) |
| `saves` | 2,625 | **2,620** (−5) |
| `final_time` | 2.623728e-02 s | **2.618731e-02 s** — the dynamic `t_end`, still reached |
| NPZ `sha256` | `d53100197b76def84e4f59d13be8a40aa6b7b8733482917570787e4c8e8c77f9` | `60e11c6fa64ced1050e57feae788f7e0eb4f1cd9b7e072bc483e3695002a9b76` |
| sidecar `sha256` | `50a3ba19d1f7dbb7fe09b643738290281d0f340cea22267ed16c43a490b54988` | `074a682519fc9a4459b34829b549dc955ee3966b666af8763bdd775a0f9d22d5` |
| 4k trajectory digest | `e50f58c9bcb0468a8834b92c9d49d3896afd2125235eebd022c65fba4747fa5a` | `f1461a03f2677146cbd927a614e3fc575a2d0dfe8a9d1eaeed54ebe872ce4b80` |
| digest `config_identity` | `5bacfee769e5226c7d1d902abdb6a2e8b5a8302e629f74bb9541b0d1f6b8ec9b` | `f95eedbbb1a76e103f92bd58e00fc8f6d401c409d828a1095f67f2d195c61310` |

**BOTH digest pairs move, and that is the expected and authorized outcome** —
unlike the three preceding rotations, which were config-identity-only moves over
an unchanged `e50f58c9` trajectory, this event changes physics. The sidecar
reports `finite: true` with `Te_max` 28.38 eV. The 150,000-step tripwire did NOT
fire, which is the load-bearing negative: the extra cost below is per-step, not
a timestep collapse.

**Run cost — the fixture got substantially more expensive, and a reviewer must
budget for it.** AT CAPTURE the digest horizon's 4,000 steps cost **19 min 36 s
pure** against ~2.5 min at the previous stance (~8.8×), and the full fixture now
runs 94,044 steps instead of 70,408. The adaptive `dt` is the mechanism: the
digest's `final_time` over its fixed 4,000 steps falls 1.5529e-03 → 4.5446e-04
s. This is a real consequence of the adopted closure, not a defect, but it
changed what the gate costs at every future merge.

**CORRECTION, same day — the `~2.8 h` pure full capture this paragraph
originally gave was an EXTRAPOLATION, and it was WRONG.** No pure full capture
had completed when it was written: it was projected from the two attempts killed
at ~60 and ~65 minutes recorded below, so there was no finished run underneath
it. The pure full golden was subsequently RUN TO COMPLETION at **1 h 21 min
42 s**, `exact=True`; and then at **46 min 38 s**, `exact=True`, once
`mg-pure-vectorize` (campaign @ `4ef3a37`) made the CSDA march's table lookups
1.75× faster. Against the 19 min 14 s compiled capture the pure path therefore
costs **2.42×**, not the near-order-of-magnitude the extrapolation implied. Two
consequences for reading this section: the AT-CAPTURE figures above stand as
measured and are not restated — but the `~8.8×` digest ratio is a
pre-`mg-pure-vectorize` number describing a stance-to-stance change, and does
not describe what the gate costs now (the pure 4,000-step digest gate measures
10 min 46 s on a clean lane, 2026-08-26). Every `~2.8 h` in this file was that
one extrapolation propagated, and is corrected here and at both other sites.

**Capture evidence.** The NPZ fixture was recaptured TWICE from clean separate
processes to temporary paths, run strictly serially per the serial-golden rule,
and compared BEFORE either was installed: NPZ and JSON sidecar byte-identical
(the two `sha256` values above), and raw-bitwise identical over all three arrays
— `phase` identical at raw bytes, `time` **0 differing of 2,620** and `y`
**0 differing of 1,509,120** at `uint64`. The digest reference was regenerated in
the same event as the protocol requires, also twice from clean separate
processes, byte-identical. `--verify` prints:

```
baseline verify OK: saves=2620, exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s (rtol=1.0e-09, atol=0.0e+00)
```

**KERNEL PATH — a disclosed departure from how previous recaptures were taken.**
The two NPZ captures were taken on the **COMPILED** path; the digest reference
was captured **PURE**. The reason is cost, not preference: two successive pure
attempts were killed at ~60 and ~65 minutes with nothing written, exceeding the
agent harness's background-process lifetime. (This sentence originally priced
that pure capture at `~2.8 h`. That figure was extrapolated from these same two
kills and is WRONG — see the correction above; the pure full golden measures
1 h 21 min 42 s, and 46 min 38 s post-`mg-pure-vectorize`. The kills were real
and the decision to capture compiled stands on them; only the number attached to
them was invented.)
The licence for using the compiled path is that pure and compiled are bit-exact
BY CONSTRUCTION of the source on linux-64 (both paths fuse their scalar lerp
explicitly via `math.fma` in `cablp/numerics/interp.py`; certified
2026-08-17), and
that this was **proven again at THIS stance** before any fixture was captured:
capturing the digest reference on both paths gave the identical final digest, all
five identical checkpoints, identical `config_identity`, `steps` and
`final_time`, with `KERNEL_ID` (`cython/_cathode_kernels_cy/tierA+csda`) probed
IN-PROCESS so the compiled run provably loaded the compiled path. The NPZ and its
sidecar carry no kernel-provenance field, so a compiled capture of them is
indistinguishable from a pure one; the digest reference DOES carry that field,
which is why it was captured pure and still reads `kernel_provenance: "pure"`
exactly as the outgoing reference did. The independent leg closing the loop is
the digest gate run on the PURE path against that pure-captured reference:
`digest gate OK: steps=4000, digest=f1461a03…, exact=True`.

**`config_snapshots.json` was regenerated in this event, and the cause is
proven rather than asserted.** All four cases moved. Restoring the stance file
from `43e7870` reproduces all four **committed** digests exactly — the
stripped-key reproduction — and `parameter_count` (256), `flag_count` (51) and
`manifest_sha256` are unchanged, so the config SURFACE did not move and no key
was added or removed. The per-case deltas are minimal and legible:
`production_golden` moves on all five stance members, because it is the only case
that applies the stance file; the three campaign cases (`compare_sim1d_es1`,
`run_m6_point_es1_sgp3649_defaults`, `run_mechanism_ladder_es1_defaults`) move on
**`C_R` alone**, since that is the only changed key `compare_sim1d_es1`'s
stance-populated `PARAM_OVERRIDES` carries — `V_bank` is supplied per rung by
those drivers, which is exactly why its deletion is invisible to them.

**The `T_s` half of the 23b R4 deletion was NOT executed, and is deferred with
its reason on record.** Deleting `V_bank` and `T_s` together was the registered
intent, on the reading that both merely restated config defaults. That reading is
correct for `V_bank` and for `T_s` *against `default_config()`* — which is why
deleting `T_s` is invisible to THIS fixture, whose `T_s` resolves to the config
default `1998.15` either way. It is NOT correct on the campaign route:
`run_m6_point.py:216` supplies `T_s` from the ES rung's `Ts_standby_K`
(ES1 = 1910.0) and the stance row SUPERSEDES it, so deleting the row would move
`T_s` 1998.15 → 1910.0 on every campaign arm. That was measured as a real second
delta in the pre-flight against `scripts/mgcr1_confirm.h5` and disappeared the
moment the row was restored. Whether the move is physically inert is unsettled:
under `cathode_warming_model = "power_balance"` the evolving surface temperature
is seeded from `cathode_Ts_base_K`, not `T_s` (`solver.py:1917`), and every live
read of `T_s` is guarded by that evolving value — but `solver.py:10925` reads it
UNGUARDED into the kinetic background, dead here only because this stance runs
`neutral_model = "moment"` and live under `"kinetic_dvm"`. The adjudication is
assigned to the DVM program.


**2026-08-24 — the TUBE-BEAMED INJECTION ROW adopted, with the shaped foot
fill regenerated under it (AUTHORIZED recapture; Tom's `[porf-dvm-consistency]`
ruling, which ruled the two changes ONE event).** The stance moved the fluid
puff off the `cosine_pipe` deposition envelope and onto the CAD-derived
`"orifice"` injection row, and rebuilt the shaped initial fill on that row at
the current geometry. They are one event because the old fill was ALSO stale
against the geometry the stance already carried, so rebuilding it and changing
the row could not be separated without shipping a fill that matched neither.

| key | old | new | class |
|---|---|---|---|
| `gas_puff_profile` (stance `g1atrim.toml`) | `"cosine_pipe"` | `"orifice"` | selector — the row is **DERIVED** (CAD port + Clausing tube beaming) |
| `gas_puff_orifice_id_cm` (stance; new config key, default `None`) | *absent* | `3.95` | **DERIVED**, the MIDPOINT of the ruled [3.8, 4.1] cm hardware bracket |
| `gas_puff_orifice_length_cm` (stance; new config key, default `None`) | *absent* | `22.0` | **DERIVED**, the one-sided lower bound L ≥ 22 cm |
| `nn0_profile`, `nn0_annulus_profile` (stance) | the 2026-08-19 `g1afix_foot45.npz` fill | the `g1aporf_foot45.npz` fill | **DERIVED** from a model run, regenerated at this tip |

Values, classes, brackets and the closure disclosures:
`production_stance_provenance.md` (the stance rows) and
`config_defaults_provenance.md` (the two new config keys). The derivation is
`cablp/solvers/_sim1d/physics/puff_orifice.py`; `MODEL.md`'s fueling section
carries the disclosure that a kinetic first-flight row is being read as the
fluid deposition row.

**What reaches THIS fixture, and what does not.** The re-cut drops the stance's
mesh-sized package, so **the regenerated fill does not travel to the golden at
all** — `nn0_profile` and `nn0_annulus_profile` are dropped whole and the
gate runs the equilibrated seed instead. The three PUFF keys are scalars and do
travel. So the fixture moves for one reason only: the puff's axial row. It also
runs that row on a DIFFERENT flight than the campaign does, because dropping
the prescribed radii puts the vessel wall at 50 cm instead of the measured
40 cm; the row is correspondingly wider here (5–95 % span 87.05 cm vs 59.81 cm
on the campaign mesh, `scripts/g1aporf_rowcensus.txt`). That is the re-cut
behaving as documented, not a second change.

**Delta discipline — proven BEFORE anything was recaptured.** The code change
that adds the profile is bit-inert with the profile off, which is what licensed
it to ride this recapture:

| arm | result |
|---|---|
| the `"orifice"` code path alone, stance untouched (still `cosine_pipe`) | `baseline verify OK: saves=2625, exact=True, max_rel=0.000e+00` on the FULL pure golden — **bit-inert with the feature off** |
| the same, compiled | `CABLP_COMPILED_KERNELS=1` FULL golden `exact=True` |
| the shared builder's orifice row vs `scripts/puff_orifice.py` on the same inputs | **bit-for-bit identical** (raw bytes), asserted by the `gas-puff-orifice-profile` smoke case — one derivation, not two |
| total inflow under the new row | conserved to `0.000e+00` relative on the golden mesh, `1.110e-16` on the campaign mesh |
| neutral seed cache | all four candidate signatures distinct — the profile and EACH new key re-key the fail-closed hash on their own, so no stale equilibrated seed can be served (`scripts/g1aporf_seedcache.txt`); no salt needed |

`config_snapshots.json` was regenerated **twice** in this pass, and the two
regenerations say different things. After the CODE change alone:
`parameter_count` 252 → **254** (the two new keys, both `None` by default),
`flag_count` unchanged at 48, and every one of the four config-complete driver
cases reproduces its previously committed digest EXACTLY when those two keys
are deleted from the resolved params — so the config surface moved by the keys
alone. After the STANCE edit: **only `production_golden` moved**, which is the
only case that applies the stance file, with the counts unchanged again.

**What moved in the fixture.**

| quantity | before | after |
|---|---|---|
| `steps` | 71,287 | **70,408** (−879) |
| `saves` | 2,625 | **2,625** (unchanged) |
| `final_time` | 2.623652e-02 s | **2.623728e-02 s** — the dynamic `t_end`, still reached |
| NPZ `sha256` | `e2cceae7b999bb9e89b6cd56a3dec1d40c77efc956e970a659cc10f5b55e1439` | `d53100197b76def84e4f59d13be8a40aa6b7b8733482917570787e4c8e8c77f9` |
| sidecar `sha256` | `dbfb813bc5784afc3c71611cddcd4195874b2e440edaf72b3b245affc44c03b0` | `50a3ba19d1f7dbb7fe09b643738290281d0f340cea22267ed16c43a490b54988` |
| 4k digest | `d28b3ca8e49b0d5bed2dda7882997becd404f26ce063f76e94ecad53ea57eccd` | `e50f58c9bcb0468a8834b92c9d49d3896afd2125235eebd022c65fba4747fa5a` |
| digest `config_identity` | `91e19ac5a7eb11c21ce0c38ab36cb60f948c420edc8ae0a1642e80095cb0eec6` | `b5315d5c931e5404febef3133b0949348e31c406b28f80cb1d4b6b87418ed46a` |

**Rotation at merge (reviewer, 2026-08-24): config-identity only.** Merging this
recapture onto an `agent-staging` that already carried the
`neutral_kinetic_dvm_transfer_hold` key (the `agent/dvm-exp-hold` rotation
record below) changed the resolved-config identity again, so the digest
reference was re-captured at the MERGED tree — twice, from clean separate
processes, byte-identical — and installed with `config_identity`
`b5315d5c931e5404febef3133b0949348e31c406b28f80cb1d4b6b87418ed46a` →
`ba5a8ed54f291a08046fea902e0c1dbefab8a0f84a0dab996c326872ecf81f24`. The
trajectory digest is UNCHANGED from this recapture
(`e50f58c9bcb0468a8834b92c9d49d3896afd2125235eebd022c65fba4747fa5a`, all five
checkpoints, `steps = 4000` and `final_time` equal); the NPZ fixture and
sidecar are untouched by the merge. `config_snapshots.json` was regenerated at
the merged tree in the same step (253 → 255 params: the two orifice keys on
top of the exp-hold key).

**The cause, stated plainly: the puff's axial row, and nothing else.** The
`"orifice"` row concentrates the same total inflow into a much narrower
footprint — on this mesh 5–95 % span 87.05 cm against the superseded
`cosine_pipe` envelope's 187.8 cm — so the near-source neutral density is
higher and the far column lower from the first step. The trajectory is
different, not degraded: the run still ignites, reaches the dynamic `t_end`,
holds the same 2,625 saves, and the sidecar reports `finite: true` with
`Te_max` 21.43 eV.

**A behavioural consequence worth knowing.** Unlike the length-weighted fluid
shapes, the orifice row is NOT masked to `_PUFF_ELIGIBLE_ROLES` — it deposits
where the ray optics lands it. On THIS mesh that puts **7.24 %** of the row's
mass in the cathode–anode gap, the plenum and the cathode cell (3.87 % on the
campaign mesh, whose measured 40 cm wall shortens the flight);
`scripts/g1aporf_rowcensus.txt` has the per-role census on both meshes. Total
inflow is conserved exactly either way, and the `kinetic_dvm` annulus-starvation
check finds no starved support cell on either mesh.

**Capture evidence.** Recaptured twice from clean separate processes to
temporary paths and compared BEFORE installing either, run strictly serially
per the serial-golden rule: NPZ and JSON sidecar both byte-identical (the two
`sha256` values above), and raw-bitwise identical over all three arrays —
`phase` identical at raw bytes, `time` **0 differing of 2,625** and `y`
**0 differing of 1,512,000** at `uint64`. The digest reference
`baselines/golden_digest_4k.json` was regenerated in the same event, as the
protocol requires. `--verify` prints:

```
baseline verify OK: saves=2625, exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s (rtol=1.0e-09, atol=0.0e+00)
```


**2026-08-24 — DIGEST-REFERENCE ROTATION ONLY, config-identity cause (the
`neutral_kinetic_dvm_transfer_hold` key added; NPZ fixture UNTOUCHED).** The
merge of `agent/dvm-exp-hold` (agent-staging `deeb8a7`, tree `edd0c07`) added
one `input_dict` key, `neutral_kinetic_dvm_transfer_hold` (default `None`,
resolved to `"exponential"` only under `neutral_model = "kinetic_dvm"`; the
golden runs the moment path and never enters the code). The short-horizon
digest gate hashes the resolved config, so the reference's config identity
moved while nothing in the trajectory did. This is NOT a fixture recapture:
`scripts/baselines/production_discharge.npz` and its sidecar are unchanged, the
FULL pure golden printed `baseline verify OK: saves=2625, exact=True,
max_rel=0.000e+00` against the standing fixture on the candidate (`da08ce1`)
and at the merged tip, and the compiled golden printed the same on the
candidate (`PROVENANCE cython/_cathode_kernels_cy/tierA+csda`).

**Evidence that the trajectory is bit-identical.** The gate run on `da08ce1`
before rotation reported `exact=False` with exactly ONE failure line — the
config identity — after walking every checkpoint and the final digest (the
gate does not short-circuit on identity). The new reference was captured
twice from clean separate processes at the merged tip (`deeb8a7`, main
checkout, `PYTHONPATH` pinned) and the two captures are byte-identical;
against the outgoing reference the only differing key is `config_identity`,
with all 5 checkpoints, `steps = 4000`, `final_time` and the final digest
equal.

| | outgoing | incoming |
|---|---|---|
| config identity | `91e19ac5a7eb11c21ce0c38ab36cb60f948c420edc8ae0a1642e80095cb0eec6` | `814b175fa688ac77bce36b439438a23a3ac3b367d3265a6e688d01f45de45326` |
| final digest (4,000 steps) | `d28b3ca8e49b0d5bed2dda7882997becd404f26ce063f76e94ecad53ea57eccd` | `d28b3ca8e49b0d5bed2dda7882997becd404f26ce063f76e94ecad53ea57eccd` (unchanged) |
| checkpoints | 5 | 5, all equal |
| NPZ fixture / sidecar | unchanged | unchanged |

Reviewed and executed by the fable-reviewer under the orchestrator's
instruction for this merge; the digest gate printed `exact=True` after the
rotation (transcript in the review report). Authority for the key itself:
`core/config_defaults_provenance.md`, `neutral_kinetic_dvm_transfer_hold`.


**2026-08-24 — the CAD-SPAN MACHINE GEOMETRY adopted, with the ray-clip
EXACTNESS FIX it forced (AUTHORIZED recapture; ruled 2026-08-23 and
2026-08-24 (Tom)).** Four config keys moved and one physics-path
function was corrected. The keys are not four independent choices: three of
them are ONE measured distance and the region that rides it.

| key | old | new | class |
|---|---|---|---|
| `cathode_anode_gap_cm` (config default) | `50.0` | `53.25` | **MEASURED**, CAD-span midpoint of 0.531–0.534 m |
| `L_cath` (config default) | `50.0` | `53.25` | **MEASURED**, the SAME distance, same reduction |
| `source_region_length_cm` (config default) | `100.0` | `103.25` | consequence — the fixed source region is defined from the anode face outward, so its far end rides the gap; span (50 cm) and cell count (5) unchanged |
| `neutral_baffle_positions_cm` (stance `g1atrim.toml`) | `[342.6]` | `[342.65]` | **MEASURED**, CAD-span midpoint of 3.401–3.452 m; the `342.6` was that same midpoint stale-rounded |

Values, classes and honest bars: `config_defaults_provenance.md` (the gap and
the new `L_cath` entry — that key had shipped with NO provenance row of its
own until this event) and `production_stance_provenance.md` (the baffle
entry, and the source-region entry re-cut to say the offset is measured and
only the span is assumed). The stance's two per-cell radius profiles were
REBUILT against the new mesh in the same event
(`scripts/g1_build_profiles.py`), because the gap moves every cell
downstream of the cathode face; the 280-cell census is unchanged.

**The code change, and why it is part of this event rather than a follow-up.**
`_clip_ray_length` (`physics/cathode.py`) built the beam ray's per-cell path
by decrementing a running remainder. At the new gap that leaves
`53.25 − 5 × 10.65 = 3.55e-15` cm, which put a non-zero `dz` on the
ANODE-CROSSING cell. Anode-mesh interception scales the deposition ray's flux
there and the probe's not at all, so the two parted company and the item-35
gap ledger opened by **35.8 % of emitted beam power (tolerance 5 %)** — the
solver's own tripwire fired on the first capture attempt, which was
therefore HELD and discarded rather than installed. The function now
ACCUMULATES FORWARD along the ray, which reproduces the same left-to-right
sum the mesh uses to place its faces, so a clip ending on a face lands on it
EXACTLY with no tolerance; a mesh-scale snap (`_CLIP_FACE_SNAP_REL = 1e-12`,
≈0.1 pm at a 10.65 cm cell) closes the residual case where `L_cath` and the
accumulated face differ by rounding rather than coinciding. The previous gap
survived only because `50.0/5 == 10.0` is exact in binary — the defect was
always latent.

**Delta discipline — every arm run BEFORE anything was recaptured.** Each is
the base tree (`4396dad`) plus exactly ONE delta, digested against the
then-committed `golden_digest_4k.json` (short-horizon, 4,000 steps):

| arm | result |
|---|---|
| base control | `exact=True`, digest `ada72fd1…` — reproduces the committed reference |
| `Te_birth_ionization` line REMOVED from the stance | `exact=True`, digest `ada72fd1…` — **bit-inert**; see the withdrawal note below |
| baffle `342.6 -> 342.65` | config identity moved, **no divergent checkpoint and no final-digest difference — the trajectory is bit-identical**. The baffle applies at the nearest cell FACE and both values snap to face 18 at `nx = 60` and face 43 at `nx = 268`. This delta is DOCUMENTARY at both shipped meshes. |
| gap `50.0 -> 53.25` (+ the ruled source-region shift) | `exact=False`, first divergent checkpoint **step 0** |
| `L_cath` `50.0 -> 53.25` ALONE | `exact=False`, first divergent checkpoint step 1000, **and it fires the item-35 tripwire** — with the mesh gap still at 50 cm the clip runs 3.25 cm PAST the anode face. Not a physically meaningful configuration: this arm is an ISOLATION ARTIFACT, and it is also the direct evidence that the two keys cannot be varied separately. |
| the exactness FIX alone, at BASE geometry | `exact=True` on the digest AND `baseline verify OK: saves=2626, exact=True, max_rel=0.000e+00` on the FULL golden against the then-committed fixture — **bit-inert at base**, which is what licensed it to ride this recapture |

A no-solve resolved-config diff of the branch against `4396dad` shows
**exactly the three geometry params in all four config-complete driver cases
(`production_golden`, `compare_sim1d_es1`,
`run_m6_point_es1_sgp3649_defaults`, `run_mechanism_ladder_es1_defaults`),
plus the baffle in `production_golden` alone** — which is the only case that
applies the stance — and **0 flag deltas anywhere**. `parameter_count` (252)
and `flag_count` (48) are UNCHANGED: no key was added or removed, only values
moved. `config_snapshots.json` was regenerated in the same pass under that
proof.

**A delta that was WITHDRAWN, and why it is recorded here.** The event was
briefed to also delete `Te_birth_ionization = "local"` from the stance file
as a silent-inert accretion. It is inert at THIS fixture — the digest arm
above proves it bit-for-bit, because the config default is `"local"` — but it
is NOT inert where the stance is actually consumed: `run_m6_point.py` carries
its own `ELECTRON_BIRTH_POLICY = "floor"`, and the stance line is exactly
what overrides it. With the line removed, `preflight_diffcfg.py --stance
g1atrim m6` reports `!! CHANGED Te_birth_ionization: 'local' -> 'floor'` and
`PRE-FLIGHT: FAIL`; with it restored, `stance supersedes this driver's
default Te_birth_ionization: 'floor' -> 'local'` and `PRE-FLIGHT: PASS`.
Removing it would have silently changed the electron birth policy of every
`run_m6_point --stance g1atrim` campaign arm. **The removal was withdrawn;
the line stays.** The driver-side fix is queued separately, and the
bit-inertness arm above is the evidence it will need.

**Capture evidence.** Recaptured twice from clean separate processes to
temporary paths and compared BEFORE installing either: NPZ and JSON sidecar
both byte-identical (`sha256` of the NPZ
`e2cceae7b999bb9e89b6cd56a3dec1d40c77efc956e970a659cc10f5b55e1439`, of the
sidecar
`dbfb813bc5784afc3c71611cddcd4195874b2e440edaf72b3b245affc44c03b0`), and
raw-bitwise identical over all three arrays (`y` and `time` at `uint64`,
`phase` at raw bytes; **0 differing elements of 1,512,000** in `y`). Run
strictly serially per the serial-golden rule. Neither capture emitted the
item-35 tripwire. `--verify` prints:

```
baseline verify OK: saves=2625, exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s (rtol=1.0e-09, atol=0.0e+00)
```

The companion digest reference was regenerated in the same event:
`scripts/baselines/golden_digest_4k.json`, steps 4000, digest
`d28b3ca8e49b0d5bed2dda7882997becd404f26ce063f76e94ecad53ea57eccd`,
config identity `91e19ac5a7eb11c2…`.

The outgoing fixture was NPZ
`99020956d804450388abc104519c236e6e69988a1cb3f37304c8fe3dafe6d2a4`
(sidecar `1efa4e635bea0bc7efe5aa95659a9f2dbea687a845fbf5d95fc91381faa2dc7f`;
saves 2,626, steps 76,631).

**What moved in the trajectory — jointly attributed to the geometry, and NOT
to the code fix.** This is worth stating precisely, because the fix was
adopted on the strength of a tripwire and it would be easy to over-credit it:
the HELD pre-fix capture at this same geometry and the installed post-fix
capture are **bit-identical — 0 differing raw-`uint64` elements in `y` and in
`time`.** The 35.8 % ledger non-closure was real and is exactly what the
tripwire exists to catch, but at this operating point it did not reach the
saved state: the extra 3.55e-15 cm of path deposited nothing the fixture can
resolve. The fix is therefore a DIAGNOSTIC-INTEGRITY correction here, and a
correctness guard on any geometry where the sliver would land somewhere the
beam is still live. The hold stands as correct procedure regardless — a
fixture whose own capture emits a 35.8 % conservation warning is not
installable, and that the state happened to be unaffected could only be known
by producing the fix and comparing.

Steps 76,631 -> 71,287 (−6.97 %); saves 2,626 -> **2,625**; `final_time`
2.624039e-02 -> 2.623652e-02 s (−0.01 %, well under one 10 µs save bin).
Phase census: `breakdown` 16 -> 15, everything else unchanged.

| summary scalar | old | new | change |
|---|---|---|---|
| `Te_max` | 20.422 eV | 21.336 eV | +4.47 % |
| `Ti_max` | 9.9170 | 9.9803 | +0.64 % |
| `n_max` | 2.48883e13 | 2.48795e13 | −0.04 % |
| `n_min` | 8.41592e8 | 8.39857e8 | −0.21 % |
| `nn_max` | 3.98703e13 | 3.97930e13 | −0.19 % |
| `nn_min` | 8.62161e10 | 8.71940e10 | +1.13 % |
| `neutral_inventory_relative_drift` | 1.19215 | 1.19207 | −0.01 % |
| `total_particle_inventory_relative_drift` | 1.25671 | 1.25648 | −0.02 % |
| `plasma_inventory_relative_drift` | 3349.8 | 3342.1 | −0.23 % |
| `thermal_energy_relative_drift` | 3016.5 | 3001.1 | −0.51 % |

`Te_min`/`Ti_min` sit on their floors and are unchanged; health stayed finite
throughout. The direction is unremarkable for a 6.5 % longer cathode-anode
gap at fixed drive: a longer gap raises the gap resistance
`R_p = L_cath/(pi R_cath^2 sigma_par)` and the beam's path before the anode,
and `Te_max` rises with it. **No reading of whether any of this is an
improvement is offered here, and none should be taken from a regression
fixture.**

**The old and new trajectories are not comparable point-by-point** — this is a
configuration change, not a repair, and no bit-level comparison between them
is meaningful.

**2026-08-23 — the CONSERVING IONIZATION BIRTH adopted as the default,
jointly with the `C_R` re-trim it forced (AUTHORIZED recapture; adopted
2026-08-23 (Tom)).** **TWO keys moved, and this
event is JOINTLY ATTRIBUTED to both of them.** Nothing below may be credited
to either key alone: the birth booking changed the plasma load and the drive
knob was re-trimmed against that change in the same event, so they are not
separable at this fixture. No code path, pin, run shape or driver changed.

| key | old | new | class |
|---|---|---|---|
| `Ti_birth_ionization` (config default) | `"floor"` | `"neutral"` | **DERIVED (conservation)** |
| `C_R` (stance `g1atrim.toml`) | `7.09` | `7.36` | **FITTED**, one-knob drive-band re-trim |

Values, classes and honest bars: `config_defaults_provenance.md` (the new
`Ti_birth_ionization` entry) and `production_stance_provenance.md` (the `C_R`
entry, rewritten). The stance file also gained an explicit
`Ti_birth_ionization = "neutral"` line — a class-1 declaration of a
physics-bearing selection that happens to equal the config default, which does
not change any resolved config but does change the stance artifact.

**What the birth change is.** One ionization event was booked twice and the
two bookings only agreed at `Tn = 300 K`: the `En` side removes the local
`(3/2) k Tn` per consumed atom while the `Ei` side added
`(3/2) k T_birth` at the ion floor. Under the cathode neutral jet the
source-region gas runs near 11.6 eV, so the pair was deleting **~9.7 kW at
plateau** on the stance arm (9250 W bulk + 427 W beam, `ph_es1.h5`) — a live
non-conservation inside the previous fixture, named nowhere. `"neutral"`
closes it; the residual is now carried explicitly by the
`ionization_birth_thermal_deficit_*_W_cm3` diagnostic rows, which read zero to
roundoff here.

**Delta discipline — what was proven before anything was recaptured.**
Substituting the THREE declared config-surface changes back into the live tree
(the default flip, the stance `C_R`, and the stance's new declaration line)
reproduces the previously committed `config_snapshots.json` **bit-for-bit**,
across all four config-complete driver cases (`production_golden`,
`compare_sim1d_es1`, `run_m6_point_es1_sgp3649_defaults`,
`run_mechanism_ladder_es1_defaults`) and the machine-readable default manifest.
They are therefore provably its whole delta. `parameter_count` (252) and
`flag_count` (48) are UNCHANGED — no key was added; only values moved.
Independently, the incoming fixture's own sidecar diffed against the outgoing
one shows exactly `C_R: 7.09 -> 7.36` and
`Ti_birth_ionization: 'floor' -> 'neutral'` and nothing else, in either
namespace. `config_snapshots.json` was regenerated in the same pass under that
proof.

*An intermediate run of the proof is worth recording because it caught
something: reverting only the two VALUES left the `production_golden` case
still differing, because the stance's new declaration line was itself a third
change to the stance artifact. The proof was re-run with all three reverted and
then matched. A delta proof that is allowed to pass with one of your own edits
unaccounted for is not a proof.*

**Capture evidence.** Recaptured twice from clean separate processes to
temporary paths and compared BEFORE installing either: NPZ and JSON sidecar
both byte-identical (`sha256` of the NPZ
`99020956d804450388abc104519c236e6e69988a1cb3f37304c8fe3dafe6d2a4`, of the
sidecar `1efa4e635bea0bc7efe5aa95659a9f2dbea687a845fbf5d95fc91381faa2dc7f`),
and raw-bitwise identical over all three arrays (`y` and `time` at `uint64`,
`phase` at raw bytes; 0 differing elements in each). 1006 s and 1004 s, run
strictly serially per the serial-golden rule — but on CONTENDED lanes (other
agents' gates were running), so those are upper bounds on the wall cost, not a
clean-lane measurement. `--verify` prints:

```
baseline verify OK: saves=2626, exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s (rtol=1.0e-09, atol=0.0e+00)
```

The outgoing fixture was NPZ
`3e5120012a11ff88f4da5653ba4fdbb195136c00aa2fa1445bd955a3d33c3e15`
(sidecar `f623cc860ad97ead24679ab8711baeecaed672d7d40e5144fd579d1025ae0cdb`;
saves 2,628, steps 79,348).

**What moved in the trajectory — JOINTLY, no single-key attribution.** Steps
79,348 -> 76,631 (−3.42 %); saves 2,628 -> **2,626**; `final_time`
2.626672e-02 -> 2.624039e-02 s (−0.10 %, about two 10 µs save bins). Phase
census: `breakdown` 18 -> 16, everything else unchanged (9 `pre_breakdown`,
2000 `main_discharge`, 600 `afterglow`, 1 `post_afterglow`).

| summary scalar | old | new | change |
|---|---|---|---|
| `Te_max` | 20.844 eV | 20.422 eV | −2.03 % |
| `Ti_max` | 7.9136 | 9.9170 | **+25.32 %** |
| `n_max` | 2.50888e13 | 2.48883e13 | −0.80 % |
| `n_min` | 8.41607e8 | 8.41592e8 | −0.00 % |
| `nn_max` | 3.87819e13 | 3.98703e13 | +2.81 % |
| `nn_min` | 1.01013e11 | 8.62161e10 | −14.65 % |
| `neutral_inventory_relative_drift` | 1.19355 | 1.19215 | −0.12 % |
| `total_particle_inventory_relative_drift` | 1.25934 | 1.25671 | −0.21 % |
| `plasma_inventory_relative_drift` | 3413.4 | 3349.8 | −1.86 % |
| `thermal_energy_relative_drift` | 2993.9 | 3016.5 | +0.76 % |

`Te_min`/`Ti_min` sit on their floors and are unchanged; health stayed finite
throughout.

**The movement was DISCLOSED before the capture and is recorded, not
adjudicated here.** `Ti_max` **+25.3 %** is the expected direction and the
expected size: the birth change hands the ions the `(3/2) k Tn` the neutral gas
gives up, and the ~26 % rise in source-region `Ti` measured on the campaign
arms (5.32 -> 6.83 eV, cells 1–5, 10–20 ms mean, `ph_es1` -> `tbn2_es1`) is the
same effect at the campaign mesh. `nn_min` falling 14.7 % and `nn_max` rising
2.8 % is the neutral field keeping the energy it used to lose. Whether any of
this is an improvement is not a question this fixture answers, and no such
reading is offered here.

**The old and new trajectories are not comparable point-by-point** — this is a
configuration change, not a repair, and no bit-level comparison between them is
meaningful.

**2026-08-21 — the `heat_flux_limiter_f` RE-CUT (AUTHORIZED recapture).** A
SINGLE config default moved; no code path, pin, run shape or driver changed.

| key | old | new | class |
|---|---|---|---|
| `heat_flux_limiter_f` | `0.1` | `0.45` | **ASSUMED -> BOXED (literature)**, bracket [0.32, 1.5] |

Value, class, bracket and the free-streaming CONVENTION that makes `0.45`
meaningful: `config_defaults_provenance.md` and
`production_stance_provenance.md`, both rewritten in the same change set. It
was **not** chosen by scoring — the scored `f` family is flat above `f ~ 0.3`,
which is the evidence that the data exerted no pull. It is not a fit.

*An `S_pump` re-cut to 2900 L/s rode the first draft of this event and was
WITHDRAWN before it landed: the pinned-transmission derivation already in
`config_defaults_provenance.md` (`P = 0.303`, Davis 1960 Table II, per-pump
`S_eff` ~ 1510) is the better-sourced leg and is what discharged the open
citation on the unattributable 1.33 D equivalent-length convention, so `3000`
stands and the `S_pump` entry is unchanged. The withdrawn draft's fixture is
not the gate and its movement figures are void.*

**Delta discipline — what was proven before anything was recaptured.** A
no-solve resolved-config diff of the branch against its base commit
(`49b73c3`), in BOTH columns (bare `default_config()` and
`default_config()` + `g1atrim`), showed **exactly ONE param and nothing
else — 1 delta in each column, 0 in either flags column.** The same single key,
and only it, moved in all FOUR config-complete driver cases
(`production_golden`, `compare_sim1d_es1`,
`run_m6_point_es1_sgp3649_defaults`, `run_mechanism_ladder_es1_defaults`),
with 0 flags moving in any of them; and the machine-readable default manifest
diff is likewise that one key alone. `config_snapshots.json` was regenerated in
the same pass under that proof; `parameter_count` (252) and `flag_count` (48)
are unchanged, so the one value is the whole of the `manifest_sha256` delta.

*Reading the sidecar diff against the OUTGOING fixture shows three params and
one flag, not one: the extra two params (`cathode_jet_hot_carrier`,
`neutral_wall_partition_sigma_hehe_cm2`) and the flag
(`neutral_wall_momentum_partition`) are keys that did not EXIST when that
fixture was captured — they arrived with the thread-23 hot-carrier and
entrainment wall-partition merges, ship default-off/`None`, and were gated
bit-exact-off on their own branches. The one-delta proof above is against the
base commit and is the one that governs.*

**Capture evidence.** Recaptured twice from clean separate processes to
temporary paths and compared BEFORE installing either: NPZ and JSON sidecar
both byte-identical (`sha256` of the NPZ
`3e5120012a11ff88f4da5653ba4fdbb195136c00aa2fa1445bd955a3d33c3e15`, of the
sidecar `f623cc860ad97ead24679ab8711baeecaed672d7d40e5144fd579d1025ae0cdb`),
and raw-bitwise identical over all three arrays (`y` and `time` at `uint64`,
`phase` at raw bytes; 0 differing elements in each). ~13 min per capture, one
lane, run strictly serially per the serial-golden rule. `--verify` prints:

```
baseline verify OK: saves=2628, exact=True, max_rel=0.000e+00, max_abs=0.000e+00, time_max_abs=0.000e+00 s (rtol=1.0e-09, atol=0.0e+00)
```

The outgoing fixture was NPZ
`857b9e0b6b31c2de36d5cfe24a8fb0023c16ec4b68a9111c5eb89be4a9cc47d1`
(saves 2,627, steps 80,416).

**What moved in the trajectory.** Because ONE key moved, all of this is
attributable to `f` alone — there is no joint-attribution caveat on this
event. Steps 80,416 -> 79,348 (−1.33 %); saves 2,627 -> **2,628**;
`final_time` 2.625885e-02 -> 2.626672e-02 s (+0.03 %, under one 10 µs save
bin).

| summary scalar | old | new | change |
|---|---|---|---|
| `Te_max` | 25.810 eV | 20.844 eV | **−19.2 %** |
| `Ti_max` | 7.9210 | 7.9136 | −0.09 % |
| `n_max` | 2.62791e13 | 2.50888e13 | −4.53 % |
| `n_min` | 8.41902e8 | 8.41607e8 | −0.04 % |
| `nn_max` | 3.90059e13 | 3.87819e13 | −0.57 % |
| `nn_min` | 1.08381e11 | 1.01013e11 | −6.80 % |
| `neutral_inventory_relative_drift` | 1.1927 | 1.1935 | +0.07 % |
| `total_particle_inventory_relative_drift` | 1.2593 | 1.2593 | −0.00 % |
| `plasma_inventory_relative_drift` | 3457.6 | 3413.4 | −1.28 % |
| `thermal_energy_relative_drift` | 2983.7 | 2993.9 | +0.34 % |

`Te_min`/`Ti_min` sit on their floors and are unchanged; health stayed finite
throughout.

**The movement was DISCLOSED before the capture and is recorded, not
adjudicated here.** Raising `f` `0.1 -> 0.45` relaxes the electron heat-flux
cap by 4.5x. The limiter binds mainly during BREAKDOWN — where `lambda_ei`
exceeds the machine length and Spitzer-Harm is invalid — and is largely inert
in the collisional discharge phase, so the trajectory was EXPECTED to move and
the re-anchor exists to absorb that. `Te_max` is a breakdown-phase transient
and is the observable that moved most. Whether any of this is an improvement is
not a question this fixture answers, and no such reading is offered.

**The old and new trajectories are not comparable point-by-point** — this is a
configuration change, not a repair, and no bit-level comparison between them is
meaningful.

**2026-08-21 — the pre-Tuesday PHYSICS BATCH (AUTHORIZED recapture; the
authorization is the batch brief itself, member 3 by Tom's ruling of
2026-08-21).** All
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
far column. (The advisor's 2026-08-21 reading of "cell 59" was taken on the
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
(AUTHORIZED recapture; adopted 2026-08-20 (Tom)).** **The first
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
