# `scripts/` — the sim1d tooling, sorted by what each file is FOR

Ten directories, no loose files. The seven code directories below are named
for the question a reader arrives with — *is this a check, a driver, a scorer,
a stance input, an instrument?* — and the three fixture directories are fixed
by the golden protocol and do not move. Every script runs from the repository
root as `python scripts/<dir>/<name>.py`; scripts import each other by bare
module name, and each one that does carries a short block putting the seven
code directories on `sys.path`, so the layout costs the caller nothing.

**Every run names a configuration** (the "no default plasma" convention).
`default_config()` is the template of keys and their classes, not a plasma
anyone runs; `stances/g1atrim.toml` is the LAPD reference configuration a run
starts from; and an alternate the campaign runs against it is a DERIVED
configuration — a committed file naming a `base` plus the deltas that move it,
`stances/examples/g1atrim_fluid_comparator.toml` being the worked one.
`stance/stance_config.py` resolves both forms and returns, with the `(params,
flags)`, the lineage a run writes into its HDF5: the configuration's name, its
base chain, each file's sha256, its delta keys and the resolved identity. **No
entry point that builds a solver has a bare mode.** Every driver in `run/`, the
scorer's own run route in `score/compare_sim1d_es1.py`, and
`gates/audit_sim1d_equilibration_duty.py` each take `--config`/`--stance` or an
explicit `--no-stance`, so an artifact can always say which configuration
produced it. That value takes either form everywhere it is accepted: a committed
configuration NAME in `stances/`, or the PATH of a configuration file, derived
or not — so a derived configuration runs from where it lives, and the lineage
recorded is the same either way. One deliberate exception, which names a
configuration without being asked: `run/capture_phase3_rhs.py` runs one locked
recipe and takes the reference configuration's name from it. Scoring an existing
artifact (`compare_sim1d_es1.py --from-h5`) names nothing on purpose: it reads
the configuration out of the file it scores. The form, its refusals and the
lineage fields are `cablp/solvers/_sim1d/CONFIG_DECLARATIONS.md`.

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
a merge is blocked by its verdict. Two audits here are not gates:
`audit_sim1d_equilibration_duty.py` and `audit_sim1d_floor_activation.py`
measure and assert nothing, so their exit status carries no verdict.

`smoke_sim1d.py` is the entry point of the smoke suite, and it keeps the
command line: no arguments for the full gate, `--list`, `--only <case>` and
`--trace`. The suite itself is the `gates/smoke/` package. Each case module
holds the cases of one subsystem, and `smoke/__init__.py` fixes the run order
in `_CASE_ORDER`. A new case goes in its subsystem's module and at the end of
`_CASE_ORDER`. The modules:

- `_harness.py`: the case registry and `@_case`, the runner and command line, the historical-stance pins, the fixtures, and the helpers that cases in more than one module share.
- `configuration.py`: configuration files, derived configurations, key namespaces and construction refusals.
- `geometry.py`: the axial grid, area profiles, obstructions and the mirror-field loader.
- `end_wall.py`: the end-wall sheath, its face fluxes and its retired names.
- `circuit_cathode.py`: the cathode sheath solve, the discharge circuit, the anode and the electrode sample.
- `beam.py`: the primary beam, CSDA deposition, the walked hot tail and the quasilinear relaxation closure.
- `neutrals.py`: neutral state, gas puff, fill, equilibration and the neutral closures.
- `atomic_rates.py`: atomic rate models and cross sections.
- `dvm.py`: the transient discrete-velocity neutral model and its exports.
- `phases.py`: breakdown, ignition, the prescribed drive and the afterglow tail hand-off.
- `numerics.py`: time integration, timestep bounds, heat conduction, fluid operators and the implicit sinks.
- `results_io.py`: results, HDF5 I/O, restart, capture artifacts, and the scoring and comparison tools.
- `compiled.py`: compiled-kernel equivalence against the pure path.

`result_bitdiff.py` is the full-result bit-diff gate: the golden compares
`time`, `y` and `phase`, and this gate compares every group, dataset and
attribute of a saved `sim1d-hdf5-v1` result at raw bytes (so `-0.0` against
`+0.0` is a difference), with an explicit allow-list for the fields stamped
from the clock or the execution. Run it on a change that could move saved
output while leaving the state trajectory bit-identical: diagnostics, ledgers,
RHS term rows, result I/O. `matrix` runs a fixed set of short runs of
committed configurations under two code trees (a revision or a directory),
each leg in its own process on its own tree, and compares each pair;
`--self-test` proves the allow-list complete and the comparator able to fail.

    python scripts/gates/result_bitdiff.py compare A.h5 B.h5
    python scripts/gates/result_bitdiff.py matrix --base <rev-or-tree> \
        --head <rev-or-tree> --outdir <dir outside the repo> [--compiled]
    python scripts/gates/result_bitdiff.py --self-test --outdir <dir outside the repo>

`ledger_check.py` reads a result file's conservation receipt and decides
whether every account closes. It is written from the conservation laws in
`MODEL.md` and the result file's layout alone and imports nothing from the
solver, so it checks the receipt with arithmetic the solver did not supply.
The solver does not write a receipt yet; `--self-test` builds synthetic files
with planted closures and planted breaks and checks each verdict.

    python scripts/gates/ledger_check.py RUN.h5 [--stage particles|energy|momentum ...] [--margin M]
    python scripts/gates/ledger_check.py --self-test

The receipt (`receipt-v1`) is a `receipt/` group: attrs `schema`, `cadence`
(`save` or `step`) and `stages_present`; `interval_t0`, `interval_t1` and
`interval_steps` per interval; `entries/<term>/<quantity>`, the
volume-integrated amount the term moved in each interval from its `debit`
account to its `credit` account (attrs `debit`, `credit`, `site`, `units`),
with its required companion `entries/<term>/<quantity>_gross`, the summed
magnitudes of the contributions that made it; `state/<name>`, inventories the
saved fields do not hold; and `census/<term>` with a `status` of `entered`,
`zero` or `not_tracked` and a `reason`. Quantities are `particles`,
`momentum` and `energy_e`, `energy_i`, `energy_k`, `energy_n` (units
`particles`, `g cm/s`, `erg`), grouped into the stages `particles`, `energy`
and `momentum`. An exchange computed at two code sites is booked as two
ordinary entries through a clearing account `exchange:<name>`, each touching
only the state its own site changed; the clearing account's inventory is
`state/exchange_<name>` (a declared carried debt) or zero, so its closure is
the agreement of the two sites. Per stage the checker tests that each
inventoried account's change, clearing accounts included, equals its entries
in minus out, and that the stage's summed change equals what crossed its
boundary (which adds information only where an account is not tracked). It
then tests every entry's units and gross and the census, including that a
`zero` or `not_tracked` term books nothing. The bar is a roundoff bound
`margin * count * 2**-53 * gross`, built from the entries' gross magnitudes,
the inventories' summand magnitudes and the operations that produced each
comparison, never from the net change.
The account lists, the inventory assumptions and the bound's derivation are in
the module docstring. Exit 0 pass, 1 a failure or census gap, 2 the check
could not run; a file with no receipt is exit 2, never a pass.

`reference_coverage.py` records which `cablp/` lines the golden route executes on each kernel route (`capture`) and classifies each hunk of a diff as reached, unreached or import-only against those maps (`check`); it is under evaluation and gates nothing.

**`run/`** — the drivers that build a `LAPDSim1D` and run it.
`run_m6_point.py` is the config-complete campaign driver, `run_sim1d.py` the
plain one, `run_mechanism_ladder.py` the ladder, `run_closure_ladder.py` the
closure ladder (the kinetic reference, its parameter bands and the fluid
closures, generated, t0-registered, run and tabulated per rung); the rest build the inputs a
run needs (`build_neutral_seed_cache.py`, `eqmap_make.py`) or measure the run
itself (`profile_sim1d.py`, `sweep_sim1d_stability.py`). A file belongs here
when its job is to *produce a trajectory*.

**`score/`** — measurement of a saved run against the experiment.
`compare_sim1d_es1.py` is the scorer of record and `fingerprints_sim1d.py` the
drive-side regression check (in the prescribed-drive mode it also prints the
response rows an imposed drive does not fix); the plotters render
comparison-to-data figures, and the radiation and power-ledger tools read a
trajectory and report physics from it. A file belongs here when it *consumes*
an h5 and says how the model did.

**`stance/`** — everything that decides what the operating point IS.
`stance_config.py` resolves a configuration by committed name or by file
path; `g1_build_profiles.py`,
`build_msi_field_profile.py`, `sp3_build_nn0.py`, `puff_orifice.py` and the
coil-field solvers build the per-cell profiles and rows the stance names; the
circuit fits pin the drive constants. The three row builders take `--stance
NAME_OR_PATH` (a committed configuration name or a configuration file path),
which sizes their rows to, and registers them on, that configuration's own
mesh; with no `--stance` each builds the reference rows exactly as before.
`g1_build_profiles.py` and `build_msi_field_profile.py` then evaluate the same
rules at the named mesh's cell centres (a half column ending in a mirror face
carries the reference rows' values over `[0, Lm/2]`), and `sp3_build_nn0.py`
builds on the named configuration whole, taking its `nx`, asserting a mirror
face zero-flux and refusing a far pump or a matrix kernel there. A file
belongs here when changing it would change what the production configuration
means.

**`atomic/`** — cross sections, rate tables and the ADAS comparisons. Table
generators (`generate_eii_tables.py`, `generate_he_ion_rate_table.py`) write
into `cablp/atomic/data/`; the rest check the packaged data against its
sources. A file belongs here when its subject is atomic data rather than the
solver.

**`verify/`** — the per-build acceptance instruments. Every
`verify_sim1d_*.py` is the registered gate of one build (its cases are cited
by name in the campaign record), alongside the wall-return reference corpus's
builder, verifier and bench (`build_wall_return_reference.py`,
`verify_wall_return_reference.py`, `bench_wall_return.py`) and the DVM
mirror-plane corpus's capture/verify script (`dvm_mirror_plane_reference.py`,
pinning `scripts/data/dvm_mirror_plane_reference.npz`) and the CSDA module's
mirror-branch corpus's (`deposit_beam_mirror_reference.py`, pinning
`scripts/data/deposit_beam_mirror_reference.npz`). The rest split
into four kinds: read-only audits/censuses of a saved run or build
(`audit_sim1d_afterglow_ion_channel.py`, `census_afterglow_tail_handoff.py`,
`t23c_pairwise_audit.py`, and the geometric zone-exchange measurement
`k2_dvm_exchange_measure.py`; of these, the ion-channel audit, the census
without `--assert-no-handoff-before-ms` and the zone-exchange measurement
measure and assert nothing, as does `bench_wall_return.py`, so their exit
status carries no verdict); a lane-equivalence check
(`r3lane_equivalence.py`); one fixed-point/underflow fence
(`r3fma_underflow_fence.py`); and one-build acceptance instruments not named
`verify_sim1d_*.py` (`k2_dvm_exchange_acceptance.py`,
`verify_beam_deposition.py`,
`verify_phase3_source_capture.py`,
`verify_twin_mirror_equivalence.py` — the `far_end = "mirror"` half column
marched against the full two-source `TwinCathode` column on a symmetric
state (fluid neutrals by default, the kinetic closure with
`--neutral-model kinetic_dvm`),
`verify_fill_spreading.py` — the initial fill's spreading members. The
registered member is a finite-volume Knudsen diffusion of the foot inventory
from the puff row, with a continuous source over the foot, gap-coupled
through the anode mesh, on the equilibrated base; its coefficient carries
three named members, and the matrix kernels the builder also offers are
retained to reproduce earlier rows. The in-repo gates are density continuity
across a bore step against the length-weighted route as negative control, the
equilibrium and free-space limits, propagator reciprocity in volume, and
convergence, with `--stance NAME_OR_PATH` choosing the configuration the
production legs run on and adding a mirror-face gate (inventory ledger, zero
mirror-face flux, the builder's refusals) when that configuration ends in a
mirror face; three optional modes read data from outside the repo — a
two-leg row bit-identity check, and comparisons against a banked
test-particle record and against one on the production geometry). These
differ from `gates/` in cadence, not
in rigor: a `gates/` check runs on every merge, a `verify/` instrument runs
for the build that owns it and stays runnable afterwards so its verdict can be
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
never be mistaken for a base. `stances/examples/g1atrim_es4_fill_linear.toml`
moves only `S_gp`, to the ES4 rung's piezo-drive fill.

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
