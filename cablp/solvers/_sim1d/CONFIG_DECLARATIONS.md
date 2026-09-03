# Declaration blocks — the config surface's model-family form

A **declaration block** states one model family's COMPLETE membership in one
place. It is an input form only: blocks are projected onto the same two flat
namespaces (`input_dict`, `input_flags`) the solver has always read, so
adopting one changes no value and moves no trajectory.

Read this alongside `core/config.py` (which owns the keys and their defaults),
`core/model_families.py` (which owns the family data) and
`core/model_declarations.py` (which owns the form and its refusals). Values and
their provenance live in the `*_provenance.md` notes, never here.

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
`neutral_equilibration` is an `input_flags` key; a block states neither fact.
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
`neutral_radial_closure` stays UNDECLARABLE there — its selector
`neutral_momentum_radial` is `"uniform"`, not `"kinetic_two_moment"`.

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
| `neutral_closure` | 14 | `neutral_model = "kinetic_dvm"` | the selection plus the 13 keys it forces |
| `neutral_radial_closure` | 8 | `neutral_momentum_radial = "kinetic_two_moment"` | plus the two keys that have no reading without it |
| `beam_tail_closure` | 22 | — | beam deposition, the anomalous channel, the walked tail |
| `cathode_surface_recycle` | 6 | — | the cathode surface's directed-recycle channel |
| `anode_surface_recycle` | 5 | — | the anode mesh's channel; `neutral_mesh_accommodation` is a member here |
| `initial_neutral_state` | 12 | — | three mutually exclusive routes (below) |

A family with a **selector** may only be declared when that selector is at its
engaging value: declaring the membership of a family you are not selecting
would claim a decision this config is not making.

Families **overlap** — `neutral_momentum_radial` is the two-moment selector and
also a key the DVM selection forces; the jet keys are members of family B and
of the DVM set that forbids them. Overlap is why two blocks claiming one key is
refused rather than merged: the two families disagree about which decision owns
the key, and only the caller can settle it.

### `initial_neutral_state` — three routes, not four

The routes are `equilibrate`, `profile` and `restart`, and at most one may be
armed. `use_cached_neutral_seed` is **not** a fourth route: it REQUIRES
`neutral_equilibration` and its dispatch is a hit/miss branch inside the
equilibration path, so it is a MODIFIER of `equilibrate`, not an alternative to
it. Three routes have three pairs, and the code carries exactly three direct
pairwise refusals. (The 2026-08-23 census recorded four mutually exclusive
routes and six pairwise exclusions; the code says otherwise, and collapsing the
four-route reading into one selector would have made `cached_seed` and
`equilibrate` mutually exclusive — a behaviour change.)

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

*(Adopted 2026-09-03, the "no default plasma" ruling.)*

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
| a flat delta restating its base's resolved value | every restated key, both values, and the waiver | a delta must MOVE something: a line that repeats its base reads as a decision, changes nothing, and stops agreeing with the base silently the first time the base moves |
| a base chain deeper than three files, or a cycle | the whole chain, in order | past that depth a value cannot be traced to the file that chose it by reading, only by running the loader |

`allow_restated = true` at the top of a file waives the first, for a file that
pins a value deliberately against its base drifting. Declaration-block members
are exempt from it by construction: a block is an INVENTORY of a family's
complete membership, written out regardless of value, so a member that agrees
with the base is the form working.

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
