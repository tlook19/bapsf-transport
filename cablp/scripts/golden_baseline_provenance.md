# Provenance of the golden baseline pins (`baseline_sim1d.BASELINE_*_OVERRIDES`)

**Recaptured 2026-08-24 (the TUBE-BEAMED INJECTION ROW adopted — the stance's
gas puff moved off the `cosine_pipe` deposition envelope onto the CAD-derived
`"orifice"` row, with the shaped initial fill regenerated under it in the same
event; the CAD-span machine geometry with its ray-clip exactness fix, the
2026-08-23 conserving ionization birth with its `C_R` re-trim, the
`heat_flux_limiter_f` re-cut to 0.45, the pre-Tuesday physics batch, the
2026-08-20 stance-update wave and the thread-24 R2b re-anchor onto the stance
preceded it — all seven under the reviewed-recapture protocol, see the
recapture record below).** The committed
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
| `max_steps` (run kwarg) | `150000` | **A tripwire, not a run length** — ~2.0× the measured 76,631 steps. It exists so a change that quietly destroys the timestep fails fast instead of running for hours. If it fires, the question is what happened to `dt`, not what happened to the trajectory. Sized at 2× deliberately: a backstop with a few percent of headroom is not a backstop, it is a second cost cap waiting to truncate the gate. |
| digest horizon (`baselines/golden_digest_4k.json`) | first `4000` accepted steps | The companion fixture for `scripts/golden_digest_gate.py`, which folds the packed state into a running SHA-256 after EVERY accepted step of this same configuration. The horizon is a cost knob, not physics: 4,000 steps is ~2.5 min against the full gate's ~17, and over the steps it covers it is the STRONGER check, because the golden certifies only what reaches a save. That gate runs at `max_steps_action = "stop"` — the cap is its run length, not a tripwire — which changes what happens AT the cap and nothing before it. |

`BASELINE_FLAG_OVERRIDES` carries one entry, `neutral_equilibration = True`, for
the reason given in the re-cut section above.

`t_end`, `dt` and `operator_split` are `None`, i.e. the solver's own run
defaults: adaptive dt, the shipped operator split, and the dynamic
current-trigger end time — **which the fixture now reaches**, so it covers the
whole cycle rather than a truncated foot.

### What the fixture costs and covers (measured at capture)

| quantity | value |
|---|---|
| steps | 70,408 |
| wall, single lane | ~15 min per capture, the two run strictly serially |
| saves | 2,625 |
| `final_time` | 2.623728e-02 s (the dynamic `t_end`, reached) |
| trajectory | `y[2625, 576]` = 8 fields × 72 cells |
| phase census (saves) | 9 `pre_breakdown`, 15 `breakdown`, 2000 `main_discharge`, 600 `afterglow`, 1 `post_afterglow` |
| save cadence | 10 us — the finest timing shift this fixture can resolve |

*(Figures above are the 2026-08-24 tube-beamed-row capture; `steps`, `saves` and
`final_time` are read from the committed sidecar
`scripts/baselines/production_discharge.json`, which is regenerated at every
recapture and is the authority for them. The two captures were
byte-identical but not equal in wall time; the spread is scheduling, not
trajectory, and both lanes here were CONTENDED — other agents' gates were
running — so these wall figures are an upper bound, not a clean-lane
measurement.)*

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
EXACTNESS FIX it forced (AUTHORIZED recapture; Tom's rulings 2026-08-23bh,
2026-08-24, campaign log 24n).** Four config keys moved and one physics-path
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
jointly with the `C_R` re-trim it forced (AUTHORIZED recapture, Tom's
decision 5; CAMPAIGN_LOG 2026-08-23k/23r/23s).** **TWO keys moved, and this
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
