# `scripts/` — the sim1d tooling, sorted by what each file is FOR

Ten directories, no loose files. The seven code directories below are named
for the question a reader arrives with — *is this a check, a driver, a scorer,
a stance input, an instrument?* — and the three fixture directories are fixed
by the golden protocol and do not move. Every script runs from the repository
root as `python scripts/<dir>/<name>.py`; scripts import each other by bare
module name, and each one that does carries a short block putting the seven
code directories on `sys.path`, so the layout costs the caller nothing.

**Every run names a configuration** (the "no default plasma" ruling,
2026-09-03). `default_config()` is the template of keys and their classes, not
a plasma anyone runs; `stances/g1atrim.toml` is the LAPD reference
configuration a run starts from; and an alternate the campaign runs against it
is a DERIVED configuration — a committed file naming a `base` plus the deltas
that move it, `stances/examples/g1atrim_fluid_comparator.toml` being the worked
one. `stance/stance_config.py` resolves both forms and returns, with the
`(params, flags)`, the lineage a run writes into its HDF5: the configuration's
name, its base chain, each file's sha256, its delta keys and the resolved
identity. **No entry point that builds a solver has a bare mode.** Every driver in
`run/`, the scorer's own run route in `score/compare_sim1d_es1.py`, and
`gates/audit_sim1d_equilibration_duty.py` each take `--config`/`--stance` or an
explicit `--no-stance`, so an artifact can always say which configuration
produced it. One deliberate exception, which names a
configuration without being asked: `run/capture_phase3_rhs.py` runs one locked
recipe and takes the reference configuration's name from it. Scoring an
existing artifact (`compare_sim1d_es1.py --from-h5`) names nothing on purpose:
it reads the configuration out of the file it scores. The form, its refusals and the lineage fields are
`cablp/solvers/_sim1d/CONFIG_DECLARATIONS.md`.

**`gates/`** — the checks that must pass before anything merges, and the
fixtures they read. `smoke_sim1d.py` is the assertion suite every solver
change runs; `baseline_sim1d.py` is the production golden and
`golden_digest_gate.py` its short-horizon complement; `interp_bitexact_gate.py`,
`interp_fused_reference.py`, `deposit_beam_reference.py` and
`restart_bitidentity.py` pin arithmetic and restart identity;
`audit_sim1d_configs.py`, `declm_block_gate.py` and `declm_route_identity.py`
pin the configuration surface, with `audit_sim1d_configs_delta.py` as the
rotation record that says which snapshot case moved and in which resolved
values; `preflight_diffcfg.py` is the no-solve config
diff every campaign arm runs before spending compute. A file belongs here when
a merge is blocked by its verdict.

**`run/`** — the drivers that build a `LAPDSim1D` and run it.
`run_m6_point.py` is the config-complete campaign driver, `run_sim1d.py` the
plain one, `run_mechanism_ladder.py` the ladder; the rest build the inputs a
run needs (`build_neutral_seed_cache.py`, `eqmap_make.py`) or measure the run
itself (`profile_sim1d.py`, `sweep_sim1d_stability.py`). A file belongs here
when its job is to *produce a trajectory*.

**`score/`** — measurement of a saved run against the experiment.
`compare_sim1d_es1.py` is the scorer of record and `fingerprints_sim1d.py` the
drive-side transfer check; the plotters render comparison-to-data figures, and
the radiation and power-ledger tools read a trajectory and report physics from
it. A file belongs here when it *consumes* an h5 and says how the model did.

**`stance/`** — everything that decides what the operating point IS.
`stance_config.py` loads the committed stance; `g1_build_profiles.py`,
`build_msi_field_profile.py`, `sp3_build_nn0.py`, `puff_orifice.py` and the
coil-field solvers build the per-cell profiles and rows the stance names; the
circuit fits pin the drive constants. A file belongs here when
changing it would change what the production configuration means.

**`atomic/`** — cross sections, rate tables and the ADAS comparisons. Table
generators (`generate_eii_tables.py`, `generate_he_ion_rate_table.py`) write
into `cablp/atomic/data/`; the rest check the packaged data against its
sources. A file belongs here when its subject is atomic data rather than the
solver.

**`verify/`** — the per-build acceptance instruments. Every
`verify_sim1d_*.py` is the registered gate of one build (its cases are cited
by name in the campaign record), alongside the reference-corpus builders and
the bit-inertness A/B harnesses. These differ from `gates/` in cadence, not in
rigor: a `gates/` check runs on every merge, a `verify/` instrument runs for
the build that owns it and stays runnable afterwards so its verdict can be
re-derived.

**`kinetic/`** — the neutral-closure instruments that stand outside the
solver: `mc_neutrals.py` (frozen-field TPMC) and `kn2zone.py` (the
deterministic two-zone model), with their comparison harnesses. They exist to
be checked against each other and against the in-solver closure on the same
background.

## The three fixed directories

`baselines/`, `data/` and `stances/` do **not** move and are not to be
reorganized. The golden protocol names those paths — `baseline_sim1d.py`
reads `baselines/production_discharge.npz`, `stance_config.py` reads
`stances/g1atrim.toml`, the reference corpora live in `data/` — and a fixture
whose path moves is a fixture whose provenance has to be re-established.
`baselines/` in particular is never touched outside the reviewed-recapture
protocol. `stances/examples/` holds derived configurations and is an ADDITION
to that directory rather than a reorganization of it: the committed
configuration set `stance_config.available_stances()` offers by name is
`stances/*.toml` and nothing below it, so an example is reached by path and can
never be mistaken for a base.

## Run artifacts do not live here

`scripts/` holds **code only**. Every run artifact — `.h5`, `.log`, `.cmd`,
`.npz`, `.prof`, probe transcripts, figures — is written to or copied into
`~/bapsf/artifacts/<campaign-or-event>/`. An untracked artifact found under
`scripts/` is a defect to move, not a convention. Artifacts produced before
2026-09-03 were collected into `~/bapsf/artifacts/scripts_loose_2026-09-03/`,
which is where a log pointer of the form `scripts/<artifact>` from before that
date resolves.

**In-repo notes therefore SUMMARISE the evidence rather than cite the run file
(2026-09-05).** Because run artifacts live outside this repository, a reader of
this PUBLIC repo cannot obtain one, so a note that rests its claim on a bare
file name states nothing that reader can check. Notes instead carry the numbers
and the derivation in the prose, and identify the run that produced them by
CONFIGURATION, rung, date and tip — "the ES1 reference run at the `g1atrim`
reference configuration, taken at commit `d0e9748` (2026-09-04), 49,415
accepted steps" — rather than by file name. A committed fixture under
`data/` or `baselines/` is obtainable and is still named directly; so is any
committed script. The point is not to hide the artifact but to make the note
survive without it.

**The same holds for where a configured VALUE came from (2026-09-05).** Each
scalar's class -- MEASURED, DERIVED, FITTED or ASSUMED -- its honest bar and
the measurement or fit behind it are recorded outside this repository, for the
same reason: that record rests on measurement memos and run artifacts a reader
of this PUBLIC repo cannot obtain, so a pointer to it states nothing that
reader can follow. What stays here is what a reader can check. The docstrings
in `cablp/solvers/_sim1d/core/config.py` say what each key MEANS -- its units,
sign convention, valid range, which term consumes it, what it raises and which
flag gates it -- and say it without reference to which number we picked; the
configuration files under `stances/` carry the values themselves; the committed
fixtures under `data/` and `baselines/` carry the arithmetic and the
trajectories a gate compares against; and `baselines/production_discharge.json`,
regenerated at every recapture, is the in-repo authority for what the golden
was captured at. Where a docstring or comment would only have pointed at the
outside record, it now says nothing further rather than pointing.
