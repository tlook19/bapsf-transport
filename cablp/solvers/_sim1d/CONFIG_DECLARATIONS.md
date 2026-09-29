# Declaration blocks — the config surface's model-family form

A **declaration block** states one model family's COMPLETE membership in one
place. It is an input form only: blocks are projected onto the same two flat
namespaces (`input_dict`, `input_flags`) the solver has always read, so
adopting one changes no value and moves no trajectory.

Read this alongside `core/config.py` (which owns the keys and their defaults),
`core/model_families.py` (which owns the family data) and
`core/model_declarations.py` (which owns the form and its refusals). Values
themselves live in `core/config.py` and in the configuration files under
`scripts/stances/`; a value's class and the measurement behind it are recorded
outside this repository. Neither belongs here.

## Why the form exists

Every model-specific key lives in one flat top-level namespace, so a config
carries every family's keys at all times — including the families it did not
select, at values that are meaningless or that the selected model refuses
outright. Nothing in the flat form says which keys belong together. Two
consequences, both measured rather than supposed:

- A selection is assembled one key at a time and verified by running into the
  guards one refusal at a time — engage, read the refusal, clear the key it
  names, run again. `core/model_families.py` flattened that cascade for the two
  neutral-closure families; a block removes the shape that produces it.
- An off-arm key still sits in the namespace, so guarding against it takes a
  hand-rolled per-key check, and a family member left undeclared reaches a run
  as whatever the package default happens to be that week.

## The form

```toml
[models.cathode_surface_recycle]
cathode_neutral_jet           = true
cathode_jet_R_N               = 0.34
cathode_jet_R_E               = 0.18
cathode_jet_energy_convention = "total_reflected"
cathode_jet_surface_debit     = true
cathode_jet_hot_carrier       = false
```

Three properties, each answering a different half of the ruling:

**Explicit regardless of value.** Every member is written, including members
sitting at their config default. A block is an INVENTORY, not a delta: reading
it tells you the whole decision, and it cannot go stale against a default that
moves underneath it. A missing member is a refusal, never an inherited value.

**Namespace-free.** `cathode_neutral_jet` is an `input_dict` key and
`use_cached_neutral_seed` is an `input_flags` key; a block states neither fact.
The family membership carries the namespace and the resolver files each member
where it belongs. The driver-side hazard — a key filed into the wrong namespace
— cannot be expressed in a block at all.

**One owner per key.** Two blocks may not claim the same key, and a member the
caller also *chose* flat, at a different value, is refused.

### `none_valued`

TOML has no null literal, so a member whose declared value is `None` is named
in the block's `none_valued` array instead of carrying a value:

```toml
[models.anode_surface_recycle]
anode_neutral_jet           = false
anode_jet_R_N               = 0.63
anode_jet_R_E               = 0.41
neutral_mesh_accommodation  = false
none_valued                 = ["anode_jet_energy_convention"]
```

The committed stance files already use this convention at file scope, so a
reader meets it once. The Python API passes `None` directly and never needs it.

## Where a block may be written

| route | how |
|---|---|
| TOML config | `[models.<family>]` tables, read by `core/config.load_config` |
| committed stance | `[models.<family>]` tables, read by `scripts/stance/stance_config.py`; projected into the stance's own delta at load, so every existing consumer of `Stance.params`/`.flags` reads them unchanged, and `Stance.models` keeps the block as written |
| Python | `LAPDSim1D(input_dict, input_flags, input_models)`, or `resolve_config(params, flags, models)` |

The campaign driver reaches blocks through `--stance`; it grew no new flag.

**The stance of record, `scripts/stances/g1atrim.toml`, is written in block form
since 2026-08-30**: three families declared — `neutral_closure`,
`beam_tail_closure` and `initial_neutral_state` — with the rest of the stance
staying flat.

`neutral_closure` became declarable there at the kinetic stance event
(2026-09-02), when that stance adopted `neutral_model = "kinetic_dvm"`. Its
membership CLAIMS seven of the eleven keys the stance's former
`cathode_surface_recycle` and `anode_surface_recycle` blocks declared, and two
blocks may not claim one key, so those two blocks were removed there in the
same event; the four keys they carried that `neutral_closure` does not own
(`cathode_jet_R_N`/`_R_E`, `anode_jet_R_N`/`_R_E`) are the fluid jets' surface
pairs, inert with those jets off, and are left at their config defaults. That
subsumption is the overlap rule doing what it is for, not a loss of coverage:
the selection that owns those keys is the one that states them.

## How it resolves

```
[models.<family>] blocks ─┐
                          ├─> resolve_config ─> flat (params, flags)
[params] / [flags] flat ──┘         │
                                    └─> resolve_model_families
                                    └─> every construction guard, unchanged
```

`resolve_config` validates and projects the blocks, then merges the caller's
flat overrides onto the templates. `resolve_model_families` then runs as it
always has: a config that arrived as a block has already stated every member,
so the family resolver finds nothing to resolve and nothing to refuse; a flat
config still gets its cascade flattened. Every single-key guard below is still
reached, by both routes.

**A block route and a flat route resolve to the identical surface, byte for
byte.** That is the migration's whole claim, and it is measured per family by
`scripts/gates/declm_block_gate.py` — on NON-DEFAULT values, so a member the
projection dropped cannot hide behind both arms falling back to the same
default — and per representative route by `scripts/gates/declm_route_identity.py`.

That harness runs SEVEN routes over SIX distinct surfaces: default, golden,
stance, campaign driver, the 13-member kinetic command line, and the k2_dvm
fixture. The seventh, `b0c`, is not a seventh surface —
`verify_sim1d_b0c_cadence` imports `arm_config` from `verify_sim1d_k2_dvm`
rather than restating it, so the two routes resolve to the same digest by
construction. It is kept deliberately, as a check that the consumer path still
reaches that fixture.

### The flat-conflict rule, and why it is not co-presence

Most callers here hand the solver a COMPLETE config dict — `default_config()`
with overrides applied, a resolved stance, a fixture's `arm_config()` — so
every member is present flat whether anyone chose it or not. Refusing mere
co-presence would make blocks unusable from exactly the entry points that need
them. The test is instead the one `model_families` already uses for "did the
caller choose this?":

- flat value **equals the template default** → inherited, not chosen; the block
  wins, silently;
- flat value **equals the block's own** → consistent; it stands;
- otherwise → the key is answered TWICE, differently, and is refused loudly
  with both values.

Inside a TOML `[params]`/`[flags]` table, where only chosen keys are written at
all, a member stated both ways is therefore always either redundant or a
conflict — which is the form the migration aims at. The tolerance exists for
the full-dict callers.

## The families

Membership is MEASURED — read off what the guards couple, not off what a name
suggests. `core/model_families.py` is authoritative; the counts below are a
reader's index.

| family | members | selector | notes |
|---|---|---|---|
| `neutral_closure` | 11 | `neutral_model = "kinetic_dvm"` | the selection plus the 10 keys it forces |
| `beam_tail_closure` | 12 | — | beam deposition, the anomalous channel, the walked tail |
| `cathode_surface_recycle` | 6 | — | the cathode surface's directed-recycle channel |
| `anode_surface_recycle` | 5 | — | the anode mesh's channel; `neutral_mesh_accommodation` is a member here |
| `initial_neutral_state` | 10 | — | two mutually exclusive routes (below) |

A family with a **selector** may only be declared when that selector is at its
engaging value: declaring the membership of a family you are not selecting
would claim a decision this config is not making.

Families **overlap** — the jet keys are members of the surface-recycle
families and of the DVM set that forbids them. Overlap is why two blocks claiming one key is
refused rather than merged: the two families disagree about which decision owns
the key, and only the caller can settle it.

### `initial_neutral_state` — one selector and the restart route

The key `initial_neutral_state` is itself the selector of how the neutral
initial condition is built: `"equilibrate"` runs the puff/off accumulation and
then the plasma run, `"equilibrate_only"` stops after the accumulation,
`"fill"` starts from the scalar `nn0`, and `"profile"` starts from the shaped
per-cell `nn0_profile`. Its values are mutually exclusive by construction, so
the only pairwise refusal left is the selector against `restart_from`, which
replaces the whole initial condition. `use_cached_neutral_seed` is **not** a
route: it requires `initial_neutral_state = "equilibrate"` and its dispatch is
a hit/miss branch inside the equilibration path, so it is a modifier of that
value.

## Refusals

All are `ValueError` at construction, naming the offender and carrying the
remedy. There are TEN, and `scripts/gates/declm_block_gate.py` exercises every one.
The owning function is named so the table and the code can be checked against
each other — this note is the KB schema source for the form, so a row missing
here is a gap in the schema, not just in the prose.

| case | refusal | owner |
|---|---|---|
| unknown family | names it, lists the declarable families | `resolve_declaration_blocks` |
| a block that is not a table | names the type it got; states there is no shorthand form | `_check_block_is_a_table` |
| `none_valued` not an array of key names | names what it got, plus the full inventory | `_split_none_valued` |
| a member both valued and in `none_valued` | names the key and the value it carries | `_split_none_valued` |
| a key the family does not own | names it and its owner — another family, an `input_dict`/`input_flags` key no family owns (with the flat table to state it in), or no template at all — plus the full inventory | `_check_membership` |
| **incomplete membership** | names every missing member, plus the full inventory | `_check_membership` |
| a block for an unselected family | names the selector, its given value and its engaging value | `_check_selector_engaged` |
| two exclusive routes armed | names both routes, each one's WHY and each one's off value | `_check_routes` |
| two blocks claiming one key | names the key and both blocks | `resolve_declaration_blocks` |
| a member also chosen flat, differently | names the key and both values | `_refuse_flat_conflicts` |

## What a block does not do

It does not change any value, and it does not validate physics. Every presence
gate, domain check and coupling guard runs afterwards, unchanged, and remains
the authority on what each edge means.

## Derived configurations — a base and what it moves

*(The "no default plasma" convention: `default_config()` is a template of keys
and their classes, never an implied plasma; every run names a
configuration.)*

Every run names a configuration. `default_config()` is the TEMPLATE of keys and
their classes — never an implied plasma — and the configuration a run names is
a committed file: `scripts/stances/g1atrim.toml` is the LAPD reference
configuration, and the alternates the campaign runs against it are DERIVED from
it. A derived configuration is a first-class object, not a command line, which
is what lets an arm be identified from its artifact rather than from a shell
history.

A configuration file may declare a base and the deltas that move it:

```toml
base = "g1atrim"

[input_dict]
neutral_model = "moment"

[input_flags]
neutral_momentum = true
```

`base` is a committed file NAME, without path or suffix, resolved in
`scripts/stances/` — a base is a committed configuration by definition, even
when the deriving file lives elsewhere. The worked example is
`scripts/stances/examples/g1atrim_fluid_comparator.toml`.

Resolution, and it is the whole contract:

```
default_config() → the base chain, oldest base first → this file's deltas
                 → the driver's nx / mesh package
```

A driver layers only its mesh on top; everything that decides what the plasma
IS comes from the named file. Deltas are validated exactly as a base
configuration's keys are — the unknown-key refusal, the wrong-namespace
refusal, and every declaration-block refusal in the table above. A block in a
derived file replaces its base's block for that family, and a family the file
does not select stays undeclarable there, so a derived file that DE-selects a
family states the freed keys flat. That is the selector rule read from the
other side, and it is why the fluid comparator's deltas are flat.

Two refusals belong to the derived form alone:

| refusal | message names | why |
|---|---|---|
| a delta restating its base's resolved value | every restated delta — a flat key with both values, a block by family name — and the waiver | a delta must MOVE something: a line that repeats its base reads as a decision, changes nothing, and stops agreeing with the base silently the first time the base moves |
| a base chain deeper than three files, or a cycle | the whole chain, in order | past that depth a value cannot be traced to the file that chose it by reading, only by running the loader |

`allow_restated = true` at the top of a file waives the first, for a file that
pins a value deliberately against its base drifting.

**The unit of that check is the delta the file WROTE, and a declaration block
is one delta, not a handful.** A block is an INVENTORY of a family's complete
membership, written out regardless of value, so an individual member agreeing
with the base is the form working and is not restatement — a complete block
with one member moved is a legitimate delta, whatever the other members say.
What the block as a whole must still do is move something: a block whose EVERY
member equals the base's resolved value re-declares a decision the base already
made, which is the same fault one flat line commits, and it is refused by
family name rather than by listing members that are individually blameless. A
flat key is checked on its own, as before; a key inside a block is checked only
through its block, because a member also stated flat is already refused by the
declaration form.

### The lineage a run records

A load returns the resolved `(params, flags)` and a `ConfigurationLineage`
(`core/config.py`): the configuration's `name`, its `base_chain` nearest base
first, the `file_sha256` of every file in that chain, the `delta_keys` this
file declares (names only — values live in the recorded config), and the
`identity`, which is `config_identity` over the resolved configuration. That
identity is the same sha256 `scripts/gates/audit_sim1d_configs.py` pins its
reviewed snapshots with, so "the same configuration" means one thing
everywhere.

`LAPDSim1D(..., configuration=<lineage>)` stores it. Nothing reads it: a run
with a lineage and a run without are bit-identical, and
`LAPDSim1D(default_config())` stays constructible, because the golden builder,
the smoke suite and the unit instruments name no committed file and must not
borrow one's name. `results/io.py` writes `configuration_name` on every file
(`"<unnamed>"` for such a run) and the other four attributes only when a
lineage exists. Reading is presence-gated attribute by attribute: a file
written before 2026-09-03 reports `None` for each, meaning "this file does not
say" — never "unnamed", and never an identity reconstructed from `params_json`.

### The half column and its fill chain

`scripts/stances/examples/g1atrim_twin_half_base.toml` is the reference
configuration cut at the mid-plane mirror face: the HALF COLUMN of a
two-source machine, and the base its shaped initial fill is built on. It lives
in `examples/` and names `base = "g1atrim"`, which resolves in
`scripts/stances/` as every base does. It moves:

| key | value | why |
|---|---|---|
| `far_end` | `"mirror"` | the mesh stops at `z = Lm/2` in a closed mirror face |
| `Lm` | `1965.4` | the mirror configuration's own length, cathode to image cathode, so the plane sits at 982.7 cm |
| `nx` | `118` | the far-column cells between the fixed source region and the plane |
| `S_pump_R` | `0.0` | a half column has no end wall for the right pump |
| `anode_tail_booking` | `"emission_fraction"` | the walked tail constructs at a mirror only under it |
| `plasma_radius_profile_cm`, `machine_radius_profile_cm` | 129 entries each | the reference rows on the half mesh |
| `[models.initial_neutral_state]` | the equilibrate route | the reference's shaped fill rows are sized to its own mesh |

The mesh is 129 cells (1 plenum, 5 gap, 5 fixed source, 118 far column) with
no end wall cell. The reference's neutral baffle keys are inherited unchanged.

The per-cell rows are the reference rows' own values over `[0, Lm/2]`.
The reference configuration's two geometry builders take `--stance
NAME_OR_PATH`, which sizes their output to that configuration's own mesh; with
no `--stance` they build the reference rows exactly as before. Run against
this file they evaluate the same rules at its cell centres, with no re-fit and
no new measurement: `g1_build_profiles.py` gives the bore staircase and the
cathode-box stages, and `build_msi_field_profile.py` gives the measured-field
flux tube, whose flat hold reaches past the plane, so every plasma radius is
the column's 18.415 cm. Every entry equals the reference row's value in the
reference cell containing the half mesh's cell centre.

**The fill chain** repeats the reference configuration's on this geometry:

1. The two radius rows, from the two builders above with `--stance` naming
   this file.
2. An equilibrated BASE run of this file. The equilibrate route runs the
   puff/off accumulation at the reference's 27 ms puff window and seeds the
   plasma run's `t = 0` frame from it, so a plasma run as short as the solver
   allows is enough (`scripts/run/run_sim1d.py --config <this file> --t-end
   2e-5 --output <base.h5>`). `"equilibrate_only"` does not serve here: it
   saves the accumulation's own start frame at `t = 0`. The seed cache is
   off, so the cache is neither read nor written.
3. `scripts/stance/sp3_build_nn0.py --stance <this file> --sgp 9010
   --base-from-h5 <base.h5>`, which builds the registered Knudsen member at
   the registered foot on this file's mesh and writes the `nn0_profile` /
   `nn0_annulus_profile` rows.

At a mirror face the fill builder's wall-limited operator is zero-flux at the
plane by construction. The builder asserts it (the mirror face is the mesh's
last face, the cell beside it is carried, and no operator conductance sits on
it) and records it in its ledger. It refuses a far pump (`S_pump_R != 0`) and
the two matrix kernels, whose column normalization is not an image fold
through the plane. `scripts/verify/verify_fill_spreading.py --stance <this
file>` runs its production legs on this mesh and adds the mirror-face gate:
the inventory ledger of the registered member on the real lobe, the zero
mirror-face flux, and the two refusals.
