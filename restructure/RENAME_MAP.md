# The R2 rename map — flatten + carve

**Status: DRAFT, reviewer-gated. Nothing here has moved.** This document is a
proposal for how every tracked path changes across the R2 mechanical
restructure. It changes no file, and the tree it describes is the tree at
`5caa8ce2cb5cfc3d89270d915a15213f1aec7fe5` (255 tracked files).

> **RE-STAMPED 2026-08-26 to the R2 cut revision (Q7).** The map was authored
> against `c018d925a90dcd86314866b8555fe42d9f22753d` and is now pinned to
> `5caa8ce2cb5cfc3d89270d915a15213f1aec7fe5`, the tip of `agent-staging` and
> of `campaign`. That window is **21 modified files and nothing else** — no
> adds, no deletes, no renames — so the file inventory is still 255, both
> prefix rules still hold, and all 244 `covers` entries regenerate from
> `git ls-files` at the new base **byte-identically**. What did move is
> re-derived here: every `file:line` citation in this document and in
> `FLATTEN_EXECUTION_NOTES.md` was re-resolved against the new base by its
> quoted symbol or string, and one content-edit row was **withdrawn** because
> the string it cited no longer exists (§2.1, `_adas.py`). Two import counts
> moved (§7). Re-verify any citation by its quoted symbol or string, which is
> stable, rather than by its line number, which is not.

Every line falls into one of two classes and they are marked differently:

| Marking | Meaning |
|---|---|
| *(unmarked)* | Follows directly from a ruling already made (R0.1 flatten, the R0.2 carve slate, the R0.4 retirements). |
| **PROPOSED** | A judgment call this map makes because the ruling does not reach it. Tom or the reviewer decides; nothing downstream may treat it as settled. |
| **RULED** | Was PROPOSED here and has since been decided. Currently: P-3 and P-4 (Tom, 26dz) and open question Q1 (Sol, 26dz). |

**Amendment pass, 26dz.** Two ruling sets have landed since the first draft:
Tom decided Q6 (the cathode module names — §3.4, P-3/P-4), and Sol assented to
Q1's schema fields with a semantic amendment (a prefix row is a compact mapping
MACRO, not a claim that the directory is a KB entity — §8, Q1). Both are folded
in here and in the manifest and validator.

Companion documents in this directory:

- `manifests/delta_flatten.DRAFT.json` — the flatten commit's delta manifest.
- `manifests/validate_manifest.py` — the manifest validator.
- `FLATTEN_EXECUTION_NOTES.md` — the ordered command sequence and gate plan.

---

## 1. The two mechanical rules

R2 is staged: **the flatten is one commit, the carve is a later one.** Almost
every path change is one of two prefix substitutions, and the value of stating
them as rules is that a reviewer can check a rule once instead of 244 lines.

**Rule F1 — the package flattens.**

```
cablp/cablp/<anything>   →   cablp/<anything>
```

64 tracked files. No exceptions, no renames inside the subtree. The
**importable name is unchanged**: the package was `cablp` and remains `cablp`;
only the directory that contains it moves. Consequently **not one `from
cablp…` statement and not one intra-package relative import changes at the
flatten** — the 470 absolute import sites and the 117 relative ones are all
invariant. That is the single most useful fact about this commit.

**Rule F2 — the sim1d script directory takes over the vacated root slot.**

```
cablp/scripts/<anything>   →   scripts/<anything>
```

180 tracked files, `baselines/`, `data/` and `stances/` travelling intact.
**PROPOSED** — see §5.3 for why this destination and not another, and §5.3 for
the one alternative.

**Rule C — the carve** (a later commit; §3 gives it per file, since it is a
real rename of every member and no single prefix covers it).

---

## 2. The flatten, file by file

### 2.1 Covered by rule F1 — the package (64 files)

`cablp/cablp/` → `cablp/`, applied to:

```
__init__.py
funcs/__init__.py                       funcs/_adas.py
funcs/_beam_deposition.py               funcs/_cathode_kernels_cy.pyx
funcs/_cathode_solver.py                funcs/_cathode_solver_idriven.py
funcs/_cross.py                         funcs/_fits.py
funcs/_heat.py                          funcs/_interp.py
funcs/_kernels.py                       funcs/_plasmaparams.py
solvers/__init__.py
solvers/_sim1d/MODEL.md                 solvers/_sim1d/NUMERICS.md
solvers/_sim1d/RESTART.md               solvers/_sim1d/__init__.py
solvers/_sim1d/config_snapshots.json
solvers/_sim1d/core/          (13 files: __init__, config, config_defaults_provenance.md,
                               deprecations, geometry, ignition, integrator,
                               model_families, neutral_seed_cache, options, state,
                               timestep, validation)
solvers/_sim1d/physics/       (15 files: __init__, cathode, conduction, energy, flux,
                               hot_neutrals, jet_carrier, kinetic_dvm, kinetic_neutrals,
                               mirror_field, neutrals, puff_orifice, reactions, sources,
                               tracer)
solvers/_sim1d/results/       (6 files: __init__, compat, health, io, phase3_capture,
                               restart)
solvers/_sim1d/solver.py
vars/__init__.py    vars/_coeff.py    vars/_cons.py    vars/_nn_table.py
vars/adas/README.md
vars/h_eii_cross.csv    vars/he_eii_cross.csv
vars/he_ion_neutral_phelps_lxcat.txt    vars/he_ion_rate.csv    vars/nn_table.csv
```

The manifest's `prefix_rule.covers` list is generated from `git ls-files`, so
it is the authoritative enumeration; the above is the human-readable form.

Two files in this subtree get a **content edit** riding the move, because
they carry the old path as a *string*:

| File (new path) | Line | What it says |
|---|---|---|
| `cablp/solvers/_sim1d/results/phase3_capture.py` | 441 | `producer_anchor` prefix `"cablp/cablp/solvers/_sim1d/solver.py:"` — has its own `surface_change` row; see §6.3 |
| `cablp/vars/adas/README.md` | 17 | Prose naming the download directory: ``directory (`cablp/cablp/vars/adas/`)`` |

**Withdrawn at the 2026-08-26 re-stamp: `cablp/funcs/_adas.py:36` was a third
row here and is no longer one.** At the old base its missing-data message read
`"be fetched by hand into cablp/cablp/vars/adas/ before the "` — repo-relative,
and therefore wrong after the flatten. Commit `f04d8a8`
(`[adas-error-path-anchor]`, message text only, no behaviour) normalized it to
`cablp/vars/adas/`, the **CABLP-relative** anchor that the next line's README
pointer already used. That anchor is *invariant across the flatten*, because
the working directory the message is read from moves in the same commit:
`<repo>/cablp` + `cablp/vars/adas/` and `<repo>` + `cablp/vars/adas/` name the
same directory before and after. `_adas.py` therefore moves **byte-identical**;
do not re-add an edit to it.

### 2.2 Covered by rule F2 — the sim1d scripts (180 files)

`cablp/scripts/` → `scripts/`. Structure preserved:

- 161 `*.py` drivers, gates, instruments and audits (including
  `baseline_sim1d.py`, `smoke_sim1d.py`, `golden_digest_gate.py`).
- 4 `*.md` provenance notes (`golden_baseline_provenance.md`,
  `production_stance_provenance.md`, `ladder_operating_provenance.md`,
  `pec_band_fractions.md`).
- `covbuild_conducting_phase.toml`, `sim1d_run_and_plot.ipynb`.
- `baselines/` (7), `data/` (5), `stances/` (1).

Content edits riding this move are enumerated in §6.

### 2.3 Explicit rows — the project-root files (5)

| Old | New | Note |
|---|---|---|
| `cablp/pyproject.toml` | `pyproject.toml` | Its `packages = [{include = "cablp"}]`, `include = [{path = "cablp/funcs/*.so"}]` and `script = "build_ext.py"` are all *project-root*-relative and are therefore **invariant** — no edit. |
| `cablp/poetry.lock` | `poetry.lock` | Travels with pyproject. |
| `cablp/build_ext.py` | `build_ext.py` | `HERE = Path(__file__).resolve().parent` and `PYX = "cablp/funcs/_cathode_kernels_cy.pyx"` are project-root-relative: **invariant** at the flatten. The `Extension("cablp.funcs._cathode_kernels_cy", …)` name is import-qualified and also invariant here — it changes at the *carve*. |
| `cablp/generate_eii_tables.py` | `scripts/generate_eii_tables.py` | **PROPOSED** — see P-8. |
| `cablp/generate_he_ion_rate_table.py` | `scripts/generate_he_ion_rate_table.py` | **PROPOSED** — see P-8. |

### 2.4 Unmoved at the flatten (6 repo-root files)

`.gitattributes`, `.gitignore` (content edit, §5.7), `.vscode/settings.json`,
`LICENSE`, `log.txt`, and — deleted, not moved — `scripts/test_adapt.ipynb`.

`log.txt` is tracked at the repository root and matches no `.gitignore`
pattern (`*.log` does not match `.txt`). It is not a restructure question and
this map leaves it alone, but the reviewer may want to ask whether it belongs
in the tree at all.

### 2.5 Deleted at the flatten (1)

`scripts/test_adapt.ipynb` — the R0.4 retirement (26dj). Deleting it is a
**precondition** for rule F2, not merely concurrent with it: `cablp/scripts/`
cannot move onto `scripts/` while a tracked file occupies the destination.

---

## 3. The carve, module by module

Applied *after* the flatten, so the "old" column is the post-flatten path.
`funcs/` and `vars/` cease to exist; nothing is left behind in either.

Module basenames become **public** (the leading underscore goes) per R0.2.
The one deliberate exception is the Cython extension — see P-6.

### 3.1 `cablp/constants` — the single-source constants (R0.2)

| Old (post-flatten) | New | Note |
|---|---|---|
| `cablp/vars/_cons.py` | `cablp/constants.py` | **PROPOSED shape** (P-1): a plain module, not a package. |

52 import statements (`cablp.vars._cons`) rewrite to `cablp.constants`. Note the R3
constants-unification pass lands **inside this file** — the divergences
(`m_e_cgs`, `m_p_cgs`, `I_ion`, the inline `qe_SI` at `solver.py:5742`) are
census item 3 and are numerics-moving; **none of them may be touched at R2.**

### 3.2 `cablp/atomic` — cross sections, fits, coefficients, ADAS

| Old (post-flatten) | New | Note |
|---|---|---|
| `cablp/funcs/_cross.py` | `cablp/atomic/cross_sections.py` | 34 import statements. |
| `cablp/funcs/_fits.py` | `cablp/atomic/fits.py` | 4 import statements. |
| `cablp/vars/_coeff.py` | `cablp/atomic/coefficients.py` | 9 import statements. **The slate names this `funcs/_coeff`; the file is at `vars/_coeff.py`.** The destination is unambiguous, only the source path in the ruling is off by one directory. Flagged, not silently corrected — see Q2. |
| `cablp/funcs/_adas.py` | `cablp/atomic/adas.py` | 15 import statements. The slate's "(+ adf11/ADAS access)" is exactly this file. |
| `cablp/vars/adas/README.md` | `cablp/atomic/data/adas/README.md` | **PROPOSED** (P-2). |
| `cablp/vars/h_eii_cross.csv` | `cablp/atomic/data/h_eii_cross.csv` | **PROPOSED** (P-2). |
| `cablp/vars/he_eii_cross.csv` | `cablp/atomic/data/he_eii_cross.csv` | **PROPOSED** (P-2). |
| `cablp/vars/he_ion_rate.csv` | `cablp/atomic/data/he_ion_rate.csv` | **PROPOSED** (P-2). |
| `cablp/vars/he_ion_neutral_phelps_lxcat.txt` | `cablp/atomic/data/he_ion_neutral_phelps_lxcat.txt` | **PROPOSED** (P-2). |
| — | `cablp/atomic/__init__.py` | `add` row. |

Two `__file__`-derived data-directory anchors move with the data and each
needs one edit:

- `_cross.py:38` `_VARS_DIR = Path(__file__).parent.parent / "vars"` →
  `Path(__file__).parent / "data"`.
- `_adas.py:27` `ADAS_DIR = Path(__file__).resolve().parent.parent / "vars" / "adas"` →
  `Path(__file__).resolve().parent / "data" / "adas"`.

Both are **invariant at the flatten** and change **only at the carve** — the
distinction matters because it keeps the flatten a pure move.

**The ADAS `.dat` files are untracked and no `git mv` will carry them.** The
39 adf11/adf15 masters under `vars/adas/` are gitignored by licence
(`.gitignore:69`); only `README.md` is tracked. They must be hand-moved on
disk, in every checkout and every live worktree, at the flatten *and* again at
the carve, and verified against the README's checksum table afterwards. A
checkout that misses this fails every `atomic_rate_model = "adas"` gate with
`RuntimeError: OPEN-ADAS data file not found` — observed in this branch's own
worktree before the files were copied in. `FLATTEN_EXECUTION_NOTES.md` §0.5
carries the procedure.

### 3.3 `cablp/plasma` — plasma parameters and heat

| Old (post-flatten) | New | Note |
|---|---|---|
| `cablp/funcs/_plasmaparams.py` | `cablp/plasma/params.py` | 9 import statements. This is the exact mapping the manifest-schema document uses in its worked example, so it is settled by that document's own illustration. |
| `cablp/funcs/_heat.py` | `cablp/plasma/heat.py` | 5 import statements. |
| — | `cablp/plasma/__init__.py` | `add` row. |

### 3.4 `cablp/cathode` — the sheath solvers, the beam, the kernels

| Old (post-flatten) | New | Note |
|---|---|---|
| `cablp/funcs/_cathode_solver.py` | `cablp/cathode/circuit.py` | 12 import statements. **RULED** (P-3, Tom 26dz). |
| `cablp/funcs/_beam_deposition.py` | `cablp/cathode/beam_deposition.py` | 15 import statements. 3,874 lines; R3's reimplementation target. |
| `cablp/funcs/_cathode_solver_idriven.py` | `cablp/cathode/circuit_idriven.py` | 7 import statements. **RULED** (P-4, Tom 26dz). |
| `cablp/funcs/_kernels.py` | `cablp/cathode/kernels.py` | 8 import statements. **PROPOSED** (P-5). |
| `cablp/funcs/_cathode_kernels_cy.pyx` | `cablp/cathode/_cathode_kernels_cy.pyx` | 7 textual references (importlib strings + prose; not `import` statements). **PROPOSED** (P-6): basename deliberately *not* made public. |
| — | `cablp/cathode/__init__.py` | `add` row. |

### 3.5 `cablp/numerics` — the fused interpolation

| Old (post-flatten) | New | Note |
|---|---|---|
| `cablp/funcs/_interp.py` | `cablp/numerics/interp.py` | 2 import statements. **PROPOSED shape** (P-7): a package, unlike `constants`. |
| — | `cablp/numerics/__init__.py` | `add` row. |

`_interp.py` imports only `math` and `bisect`. It is the most isolated module
in the package and the one whose bit-exactness certification (`math.fma`,
`scripts/data/interp_fused_reference.npz`) is load-bearing — the carve must
not so much as reflow it.

### 3.6 Deleted at the carve / retirement commit

| Path (post-flatten) | Disposition |
|---|---|
| `cablp/funcs/__init__.py` | **delete**, `retired_no_successor`. A star-import aggregator (`from ._cross import *`, …) for a package that ceases to exist. Every one of the 17 `from cablp.funcs import X` sites imports a *submodule*, never an aggregated name, so nothing consumes what it re-exports. |
| `cablp/vars/__init__.py` | **delete**, same reasoning (`from ._coeff import *`, `from ._cons import *`). |
| `cablp/vars/_nn_table.py` | **delete**, R0.4 retirement. Frozen, ungenerable (its generator is the untracked `scripts/generate_nn_table.py`), one consumer (`core/config.py:3` `lookup_nn0`), and production short-circuits it. **Not golden-inert in the source sense**: `core/config.py` must drop the import and the `resolve_nn0` branch that calls it, which is a solver-file edit inside R2. See Q3. |
| `cablp/vars/nn_table.csv` | **delete**, the data half of the same retirement. |
| `cablp/solvers/_sim1d/results/compat.py` | **delete**, R0.4 retirement (sim3 aliases for a solver removed at D2). Three consumers: `results/io.py:16,380`, `solver.py:205,10153`, and — **the one that bites** — `scripts/smoke_sim1d.py:8074-8077`, which asserts on `sim3_compat_units` and `sim3_compat_notes`. See §6.4 and Q3. |
| `cablp/__init__.py` | **NOT deleted** — `surface_change` row: `submodules = ["funcs", "vars", "solvers"]` becomes `["atomic", "cathode", "constants", "numerics", "plasma", "solvers"]`. |

### 3.7 Hydrogen arms — KEEP

R0.5: the H arms of `_cross`/`_fits`/`_coeff` are **not** selected for
retirement and travel with their modules unchanged
(`cablp/atomic/cross_sections.py`, `fits.py`, `coefficients.py`;
`cablp/atomic/data/h_eii_cross.csv`). The corrupt-H-table question and
quarantine-vs-remove are R3's. **No H row appears in any R2 manifest**, which
is the correct signal that R2 did not touch them.

---

## 4. Unlisted-member proposals (the judgment calls)

Each is flagged **PROPOSED**; the reviewer or Tom rules.

**P-1 · `cablp/constants.py`, a module, not a package.**
`vars/_cons.py` is 43 lines of flat constant bindings with no natural
sub-structure. The import-qualified symbols are `cablp.constants.qe_SI` etc.
**either way**, so promoting it to a package later costs zero import-site
edits — which is exactly why the cheaper form should be chosen first.
*Alternative:* `cablp/constants/__init__.py`, if the R3 unification is
expected to split it by unit system (SI / CGS / mpmath).

**P-2 · `cablp/atomic/data/` for the five `vars/` data files and the ADAS
directory.** `vars/` is dissolved by the carve, so the data must go
*somewhere*, and every one of these files is consumed by exactly one carve
destination (`_cross.py` reads the three CSVs and the LXCat text; `_adas.py`
reads `adas/`). Putting them under the package that reads them keeps the
`__file__`-relative anchor one level deep instead of two.
*Alternative:* a top-level `cablp/data/` shared across packages — worth
considering only if a later package needs to read the same tables, which
nothing currently does.

**P-3 · `cablp/cathode/circuit.py`** for `_cathode_solver.py` — **RULED by
Tom, 26dz.** Both of this map's original candidates were wrong. The module
does not merely solve a sheath: it solves the full cathode/anode/bank
**circuit** — the Thevenin load line off `V_bank`, the `R_comp` partition,
the anode sheath `phi_a`, and `I_tot` — so `sheath.py` named a part for the
whole, and `solver.py` named nothing at all while colliding with
`cablp/solvers/_sim1d/solver.py`. `circuit.py` is what the module is, and it
retires the collision as a side effect.

**P-4 · `cablp/cathode/circuit_idriven.py`** for
`_cathode_solver_idriven.py` — **RULED by Tom, 26dz**, with P-3. The pairing
is load-bearing and now reads correctly: this module inverts the *same
circuit* formulation, taking the inductor-integrated loop current as the
independent variable, and it imports every physical piece from its partner
rather than restating it. The two names must continue to move together.

**P-5 · `cablp/cathode/kernels.py`** for `_kernels.py`. It is the opt-in
selector that binds exactly one compiled module,
`cablp.funcs._cathode_kernels_cy`, whose own `KERNEL_ID` is
`cython/_cathode_kernels_cy/tierA+csda` — cathode tier A plus the CSDA march,
both cathode-domain. Keeping selector and extension in the same package is
what makes the `_MODULE` string readable. *Counter-argument the reviewer
should weigh:* `_kernels.py` is really build/deployment infrastructure
(`CABLP_COMPILED_KERNELS`, `PROVENANCE`) consumed by `results/io.py` and
`solver.py` for artifact metadata, not by cathode physics; that framing would
put it in `cablp/numerics/` or a new `cablp/build/`. I prefer `cathode`
because the coupling to the one extension is total and a package boundary
should follow the coupling.

**P-6 · `cablp/cathode/_cathode_kernels_cy.pyx` — basename deliberately
unchanged, underscore retained.** This is the one place where "private names
become public" should NOT be applied. Renaming the extension module changes
its importable name, the built `.so` filename, and — if the `KERNEL_ID` string
were touched with it — the `compiled_kernels` provenance value recorded in
**every artifact ever produced on the compiled path**. Freezing the basename
keeps `KERNEL_ID` at `cython/_cathode_kernels_cy/tierA+csda` across R2, so
artifacts from before and after the restructure stay directly comparable. The
package path in `build_ext.py:59` and `_kernels.py:39`
(`"cablp.funcs._cathode_kernels_cy"` → `"cablp.cathode._cathode_kernels_cy"`)
still changes; the *basename* does not.

**P-7 · `cablp/numerics/` as a package** (unlike `constants`, P-1).
`_interp.py` is its only member today, but numerics is the one carve
destination R3 is *known* to grow: the deposit_beam lane-march reimplementation
is numerical-method code with no cathode-physics content. A package now costs
one empty `__init__.py`; a module now costs a second rename later.

**P-8 · `scripts/generate_eii_tables.py` and
`scripts/generate_he_ion_rate_table.py`.** Both are one-shot generators for
committed data tables, with no importer. `scripts/` is where this repository
keeps exactly that class of thing. *Cost, stated plainly:* both anchor on
`Path(__file__).parent` meaning "the project root"
(`generate_eii_tables.py:18,24`; `generate_he_ion_rate_table.py:18`), so each
needs a second `.parent` — three content edits total inside a commit whose
whole discipline is "pure moves". *Zero-edit alternative:* leave both at the
repository root, where `Path(__file__).parent` keeps meaning the project root
and the rows become pure prefix moves. If the reviewer prefers a
content-edit-free flatten, take the alternative and revisit at the carve.

**P-9 · `cablp/funcs/__init__.py` and `cablp/vars/__init__.py` are deleted,
not re-homed.** Both are star-import aggregators. R0.3 says the cross-package
private-name fiction dies with the rename, and every real consumer already
imports the submodule directly (17 sites, all of the form
`from cablp.funcs import <module>`). The new package `__init__.py` files
should be **empty or docstring-only**, not new aggregators — otherwise the
two-tier surface is re-created under new names on day one.

---

## 5. What happens to everything that is not a package module

### 5.1 `build_ext.py` and the Cython extension source

`build_ext.py` moves to the repository root (`cablp/build_ext.py` →
`build_ext.py`) and is **content-invariant at the flatten**: `HERE`, `PYX` and
the `Extension` name are project-root-relative or import-qualified, and the
project root *is* the repository root afterwards. The documented invocation
`python build_ext.py --inplace` is then run from the repository root instead of
from `cablp/` — a one-line change in CLAUDE.md and in `_kernels.py`'s
RuntimeError text (`"…from the cablp/ directory…"`, `_kernels.py:73`).

At the **carve**, three coordinated edits: `build_ext.py:59`'s Extension name,
`build_ext.py`'s `PYX` path, and `_kernels.py:39`'s `_MODULE`. The
`-ffp-contract=off` flag and every other compile argument are untouched — they
are the bit-exactness contract.

### 5.2 `pyproject.toml` / `poetry.lock`

Both move to the repository root. `pyproject.toml` needs **no content edit at
the flatten** (§2.3). At the **carve** it needs one: `include = [{path =
"cablp/funcs/*.so", format = "wheel"}]` becomes `cablp/cathode/*.so`. Missing
that edit produces a platform-tagged wheel with no extension in it — silently,
which is why the comment above that line exists.

Installation stays `pip install -e . --no-deps --no-build-isolation` (never
`poetry install` into the conda env, per the 2026-08-16 bring-up trap), run
from the repository root rather than from `cablp/`.

### 5.3 `cablp/scripts/` — the sim1d script directory

**PROPOSED: `cablp/scripts/` → `scripts/`.** The reasoning chain:

1. After rule F1, the path `cablp/scripts/` is *inside the package*. Leaving
   180 driver scripts there would make them a shipped subpackage. Not an
   option.
2. The repository-root `scripts/` slot is vacated by the 26dj legacy archival
   (its one tracked file is deleted; the untracked notebooks, `lecroydaq.py`
   and `data/` are disk-archived outside the repository), and the same ruling
   **drops the `.gitignore` `scripts/*` rule** — which only makes sense if
   something tracked is about to live there.
3. CLAUDE.md's "two script directories" section is slated to update in the R2
   docs pass. After this move there is **one**, which is the simplification the
   ruling anticipates.
4. It is the choice that *preserves* the most path arithmetic. Every
   `SCRIPT_DIR`-relative anchor (the golden's `baselines/`, the digest gate's
   reference) is invariant, and `smoke_sim1d.py`'s
   `cwd=Path(__file__).resolve().parents[1]` + `"scripts/run_sim1d.py"` pair
   keeps resolving because both halves shift together. §6 lists the residue
   that does not.

*Alternative:* `tools/` at the repository root, if Tom wants the root
`scripts/` name permanently retired along with its contents. Costs the same
edits; gains nothing except the absence of a name collision with history.

### 5.4 The notebooks

`cablp/scripts/sim1d_run_and_plot.ipynb` → `scripts/sim1d_run_and_plot.ipynb`
under rule F2. It builds from `default_config()` and writes `sim1d_run.h5`
*alongside itself* (gitignored), so its behaviour is location-independent and
it needs no edit. The `.gitignore` `*.h5` rule is unanchored and keeps
matching.

`scripts/test_adapt.ipynb` (repository root) is **deleted** — R0.4.

The untracked root-`scripts/` notebooks (`cross_sections.ipynb`,
`rates_H.ipynb`, `rates_He.ipynb`, `langmuir2.ipynb`, `rk4.ipynb`,
`debug_solver.ipynb`, `op.ipynb`, `inter.ipynb`, `mach2.ipynb`, …), plus
`lecroydaq.py`, `generate_nn_table.py` and `data/`, are **disk-archived
outside the repository** before the move. Two of them are load-bearing for
decisions already recorded and the archive must not be treated as a deletion:

- `cross_sections.ipynb` is the sole consumer of `a_11s_double` in
  `vars/_coeff.py`, which carries an explicit in-repo comment saying to delete
  that row *with that notebook's disposition, not before*.
- `generate_nn_table.py` is the generator whose untracked status is the
  evidence for `_nn_table` being "ungenerable" in the R0.4 retirement.

### 5.5 `.gitattributes` and nbstripout

**No change.** `.gitattributes` is three unanchored patterns (`*.ipynb
filter=nbstripout`, `*.zpln filter=nbstripout`, `*.ipynb diff=ipynb`) that
match by extension anywhere in the tree, so the notebook keeps its clean
filter across the move. The filter *definition* lives in `.git/config`, which
is not versioned and is per-clone; the flatten does not touch it. One thing to
verify at execution time and nowhere else: run `git config --get
filter.nbstripout.clean` in the checkout doing the move, because a checkout
without the filter installed would commit notebook outputs — an existing trap,
not a new one.

### 5.6 `MODEL.md`, `NUMERICS.md`, `RESTART.md`, provenance notes

All move under rule F1 or F2 with no content change from the *flatten* itself:

| Old | New |
|---|---|
| `cablp/cablp/solvers/_sim1d/MODEL.md` | `cablp/solvers/_sim1d/MODEL.md` |
| `cablp/cablp/solvers/_sim1d/NUMERICS.md` | `cablp/solvers/_sim1d/NUMERICS.md` |
| `cablp/cablp/solvers/_sim1d/RESTART.md` | `cablp/solvers/_sim1d/RESTART.md` |
| `cablp/cablp/solvers/_sim1d/core/config_defaults_provenance.md` | `cablp/solvers/_sim1d/core/config_defaults_provenance.md` |
| `cablp/scripts/golden_baseline_provenance.md` | `scripts/golden_baseline_provenance.md` |
| `cablp/scripts/production_stance_provenance.md` | `scripts/production_stance_provenance.md` |
| `cablp/scripts/ladder_operating_provenance.md` | `scripts/ladder_operating_provenance.md` |
| `cablp/scripts/pec_band_fractions.md` | `scripts/pec_band_fractions.md` |

R2 requires these stay *identifiable* after the move, and they do — each keeps
its filename and its position relative to what it documents. Two content
references need the docs pass, not the move: `NUMERICS.md:693` mentions
`results/compat.py` (which the retirement deletes), and
`config_defaults_provenance.md:308` mentions `vars/nn_table.csv` (likewise) —
both at the *carve/retirement* commit, not at the flatten.

### 5.7 `.gitignore`

Ten pattern lines change. This is a *modification*, not a move, so it carries
no manifest row (the coverage check reads added/removed paths only) — it is
documented here and in the execution notes instead.

| Line | Now | Becomes |
|---|---|---|
| 1 | `scripts/*` | **dropped** (26dj) |
| 10 | `cablp/cablp/funcs/*.c` | `cablp/funcs/*.c` — and at the *carve*, `cablp/cathode/*.c` |
| 16 | `cablp/dist/` | `dist/` |
| 34–36 | `cablp/scripts/*.npz`, `*.json`, `*.png` | `scripts/*.npz`, `*.json`, `*.png` |
| 49–55 | `cablp/scripts/*.txt`, `*.cmd`, `*.exit`, `*.head`, `*.start`, `*.time`, `*.wall` | `scripts/…` |
| 58 | `cablp/scripts/neutral_seed_db/` | `scripts/neutral_seed_db/` |
| 65, 69 | `cablp/cablp/vars/adas/README.md` (prose), `cablp/cablp/vars/adas/*.dat` | `cablp/vars/adas/…` — and at the *carve*, `cablp/atomic/data/adas/*.dat` |

**Two ordering hazards.** (a) Line 1 must be dropped in the *same commit* as
rule F2, or the 180 moved scripts are invisible to `git add`. (b) Lines 34–36
must be re-anchored in the same commit, or `scripts/*.json` stops being
ignored and roughly 2,500 untracked run-record artifacts in the working
directory become stageable — the exact noise the D-3 ruling removed. Both are
in the execution sequence.

### 5.8 The golden baseline paths

`scripts/baselines/` must remain gate-valid, and under rule F2 it does, for a
reason worth stating precisely: **the golden's own path arithmetic is
`SCRIPT_DIR`-relative, and `SCRIPT_DIR` moves with the fixture.**

- `baseline_sim1d.py:66` `SCRIPT_DIR = Path(__file__).resolve().parent`
- `baseline_sim1d.py:67` `DEFAULT_BASELINE = SCRIPT_DIR / "baselines" / "production_discharge.npz"`
- `baseline_sim1d.py:73` `sys.path.insert(0, str(SCRIPT_DIR))` → `from stance_config import load_stance`
- `golden_digest_gate.py:65,66,79` — the same three moves, for `baselines/golden_digest_4k.json`

All four are invariant. The stance file `scripts/stances/g1atrim.toml` travels
with them, and the committed sidecar `scripts/baselines/production_discharge.json`
is data, not code. **The golden's config identity does not change at the
flatten** — which is what makes `exact=True` the correct expectation rather
than a hope.

§6 is the list of things that are *not* invariant.

---

## 6. Path breakage: what actually breaks at the flatten

Everything below is a **functional** break unless marked as prose. All line
numbers are at `5caa8ce`, and all paths are the *pre*-flatten paths.

### 6.1 Repository-root anchors computed by counting parents — these BREAK

The flatten removes one directory level between a script and the repository
root, so every `parents[N]` that reaches the repository root is off by one
afterwards.

| File:line | Now | Must become | Consequence if missed |
|---|---|---|---|
| `cablp/scripts/capture_phase3_rhs.py:27` | `REPOSITORY_ROOT = SCRIPT_DIR.parents[1]` | `SCRIPT_DIR.parents[0]` | Every `git` call in the module runs one directory *above* the repository. |
| `cablp/scripts/verify_phase3_source_capture.py:520` | `repo_root = Path(__file__).resolve().parents[2]` | `parents[1]` | `check_constructor_order_and_cli_import` reads the solver source from outside the repository and raises. |
| `cablp/scripts/stage3_observability_check.py:54` | `cwd=pathlib.Path(__file__).resolve().parents[2]` | `parents[1]` | `git show <ref>:<path>` runs outside the repository. |

**Two `parents[N]` sites that do NOT break, and it matters that they are
listed as verified rather than assumed:**

- `cablp/scripts/smoke_sim1d.py:11143` and `:11177` —
  `cwd=Path(__file__).resolve().parents[1]` paired with the argument
  `"scripts/run_sim1d.py"` / `"scripts/plot_sim1d_run.py"`. Both halves shift
  by exactly one level: `<root>/cablp` + `scripts/…` becomes `<root>` +
  `scripts/…`. **Invariant.**
- `cablp/scripts/audit_sim1d_configs.py:28` — `Path(__file__).resolve().parents[1]
  / "cablp" / "solvers" / "_sim1d" / "config_snapshots.json"`. Today that is
  `<root>/cablp/cablp/solvers/…`; afterwards `<root>/cablp/solvers/…`.
  **Invariant, by exactly the same cancellation.** (It breaks at the *carve*
  only if `config_snapshots.json` moves, which it does not.)

### 6.2 Repository-relative path *strings* — these BREAK

`capture_phase3_rhs.py` computes the phase-3 provenance census from a list of
repository-relative paths and a `git ls-tree` pathspec. Every one is wrong
after the flatten:

| File:line | Now |
|---|---|
| `capture_phase3_rhs.py:29` | `PRODUCER_PATH = "cablp/scripts/capture_phase3_rhs.py"` |
| `capture_phase3_rhs.py:36` | `"cablp/cablp/solvers/_sim1d/core/config.py"` |
| `capture_phase3_rhs.py:37` | `"cablp/cablp/solvers/_sim1d/core/state.py"` |
| `capture_phase3_rhs.py:38` | `"cablp/cablp/solvers/_sim1d/results/io.py"` |
| `capture_phase3_rhs.py:39` | `"cablp/cablp/solvers/_sim1d/results/phase3_capture.py"` |
| `capture_phase3_rhs.py:40` | `"cablp/cablp/solvers/_sim1d/solver.py"` |
| `capture_phase3_rhs.py:41` | `"cablp/scripts/baseline_sim1d.py"` |
| `capture_phase3_rhs.py:42` | `"cablp/scripts/golden_digest_gate.py"` |
| `capture_phase3_rhs.py:43` | `"cablp/scripts/stances/g1atrim.toml"` |
| `capture_phase3_rhs.py:44` | `"cablp/scripts/baselines/golden_digest_4k.json"` |
| `capture_phase3_rhs.py:45` | `"cablp/scripts/baselines/production_discharge.json"` |
| `capture_phase3_rhs.py:128` | `environment_lock={"path": "cablp/poetry.lock", …}` |
| `capture_phase3_rhs.py:186` | `_git("ls-tree", "-r", "--name-only", "HEAD", "--", "cablp/cablp")` |
| `verify_phase3_source_capture.py:126,317,332,348` | `… / "cablp/scripts/baselines/phase3_rhs"` |
| `verify_phase3_source_capture.py:137,280,360` | `producer_path="cablp/scripts/capture_phase3_rhs.py"` |
| `verify_phase3_source_capture.py:151,290,370` | `environment_lock={"path": "cablp/poetry.lock", …}` |
| `verify_phase3_source_capture.py:226` | asserts `producer_path == "cablp/scripts/capture_phase3_rhs.py"` |
| `verify_phase3_source_capture.py:229` | asserts the `producer_anchor` prefix `"cablp/cablp/solvers/_sim1d/solver.py:"` |
| `verify_phase3_source_capture.py:477` | `repo_root / "cablp/cablp/solvers/_sim1d/solver.py"` |
| `verify_phase3_source_capture.py:501` | `repo_root / "cablp/scripts/capture_phase3_rhs.py"` |
| `stage3_observability_check.py:35` | `REPO_PATH = "cablp/scripts/compare_sim1d_es1.py"` |
| `smoke_sim1d.py:22623` | `_p3_out = _p3_root / "cablp/scripts/baselines/phase3_rhs"` |
| `smoke_sim1d.py:22630` | `producer_path="cablp/scripts/capture_phase3_rhs.py"` |
| `smoke_sim1d.py:22641` | `environment_lock={"path": "cablp/poetry.lock", …}` |

`smoke_sim1d.py`'s three are synthetic fixture strings, not filesystem reads —
they must still be updated so the fixture keeps describing a real repository
shape, and the smoke suite is a gate, so they are not optional.

### 6.3 The one generated locator — `producer_anchor`

`cablp/cablp/solvers/_sim1d/results/phase3_capture.py:441` emits
`"cablp/cablp/solvers/_sim1d/solver.py:" + "LAPDSim1D._trajectory_result -> …"`
into every phase-3 qualification record. This is not a comment: it is the
locator downstream consumers compare against, and it changes at the flatten.
It carries its own `surface_change` row in the manifest.

**And the committed record must NOT be rewritten to match.**
`cablp/scripts/baselines/phase3_rhs/77d675a4-6852-45ff-9211-1c3cb6e74572.provenance.json`
holds 100 occurrences of the old prefix. It describes a capture taken at a
revision where that path *was* correct; editing it would falsify a provenance
record to make a path lookup convenient. It moves under rule F2 with its
contents byte-identical. See Q4.

### 6.4 Not a path break, but a gate break at the retirement commit

`cablp/scripts/smoke_sim1d.py:8074-8077` asserts on `sim3_compat_units` and
`sim3_compat_notes` — attributes attached by the `results/compat.py` that R0.4
retires. The golden fixture is unaffected (`production_discharge.npz` holds
exactly `phase`, `time`, `y`), so the *baseline* is inert; the *smoke suite* is
not. Deleting `compat.py` requires deleting those four assertions in the same
commit. Same shape for `_nn_table`: `core/config.py:3` imports `lookup_nn0` and
must lose the import and its call site.

### 6.5 Prose-only mentions (correct at the docs pass, not load-bearing)

`build_ext.py:5,6`; `_cathode_kernels_cy.pyx:85`;
`physics/cathode.py:780,1213`; `physics/hot_neutrals.py:33`;
`physics/reactions.py:52`; `interp_bitexact_gate.py:34`;
`interp_fused_reference.py:39`; `mc_neutrals.py:1373`;
`pec_band_fractions.py:89,98,111,223`; `profile_sim1d.py:64`;
`vars/adas/README.md:17`; `_kernels.py:73` ("from the cablp/
directory"). Most are `cablp.funcs.X`-style *import* references that break at
the **carve**, not the flatten. (`_adas.py:36` was listed here until the
2026-08-26 re-stamp; at this base it names no `cablp/cablp/` path — see §2.1.)

Two deserve a second look because they are `git show` recipes against
**historical** revisions:

- `t23c_config_snapshot_delta.py:16` — `git show <base>:cablp/cablp/solvers/_sim1d/config_snapshots.json`
- `mirror_fieldmap_bitexact_structural.py:54` — the same recipe

At a `<base>` older than the flatten, the **old** path is the correct one.
Blind-rewriting these makes the documented recipe fail against exactly the
revisions it exists to compare with. Q5.

---

## 7. Import-churn scope, by commit

The whole tree holds **470** statements matching
`^\s*(from|import)\s+cablp` across tracked `*.py` and `*.pyx`: 279 target
`cablp.solvers.*`, 190 target `cablp.funcs.*` or `cablp.vars.*`, and one is a
bare `import cablp`. (469/278 at the pre-re-stamp base; `smoke_sim1d.py` gained
one `from cablp.solvers._sim1d.solver import SURFACE_LOSS_FLOOR_EXEMPT_RTOL` in
the window. **The carve's 190 is unchanged**, and so is every per-module tally
below — the new statement targets `cablp.solvers.*`, which neither commit
rewrites.)

| Commit | Absolute `cablp.*` import statements changed | Relative intra-package imports changed |
|---|---|---|
| **Flatten** | **0** | **0** |
| **Carve** | **190** | **13 rewritten + 7 deleted** |

**Flatten: zero.** The package's importable name is `cablp` before and after;
only its containing directory moves. Nothing that resolves a module name is
affected, at any depth.

**Carve: all 190.** They divide as 174 dotted-module imports plus 16
package-level `from cablp.funcs import <module>` forms:

```
52  cablp.vars._cons              →  cablp.constants
34  cablp.funcs._cross            →  cablp.atomic.cross_sections
15  cablp.funcs._beam_deposition  →  cablp.cathode.beam_deposition
15  cablp.funcs._adas             →  cablp.atomic.adas
12  cablp.funcs._cathode_solver   →  cablp.cathode.circuit         (P-3)
 9  cablp.vars._coeff             →  cablp.atomic.coefficients     (Q2)
 9  cablp.funcs._plasmaparams     →  cablp.plasma.params
 8  cablp.funcs._kernels          →  cablp.cathode.kernels         (P-5)
 7  cablp.funcs._cathode_solver_idriven → cablp.cathode.circuit_idriven (P-4)
 5  cablp.funcs._heat             →  cablp.plasma.heat
 4  cablp.funcs._fits             →  cablp.atomic.fits
 2  cablp.funcs._interp           →  cablp.numerics.interp
 2  cablp.vars._nn_table          →  DELETED with the module (R0.4)
---
174 dotted + 16 package-level = 190
```

All 16 package-level forms are `from cablp.funcs import <module>` —
`smoke_sim1d.py` ×8, `spike_csda_march.py` ×2, and one each in
`capfix_frozen_census.py`, `interp_bitexact_gate.py`,
`interp_fused_reference.py`, `phicspike_frozen_sweep.py`,
`spike_cython_kernels.py` and `tailion_estimate.py`. **Every one imports a
submodule; not one imports a name aggregated by `funcs/__init__.py`**, which
is the evidence P-9 rests on. The 470th statement is the bare `import cablp`
at `fnb3_closure.py:48`, which is invariant under both commits.

**Relative imports.** 13 statements are rewritten and 7 are deleted with the
two aggregator `__init__.py` files:

| Statement (current) | Becomes | Kind |
|---|---|---|
| `_beam_deposition.py:585` `from ._cathode_solver import _c_log_ei` | `from .circuit import _c_log_ei` | stays same-package |
| `_beam_deposition.py:594` `from ._kernels import COMPILED_KERNELS…` | `from .kernels import …` | stays same-package |
| `_heat.py:3` `from ._plasmaparams import (…)` | `from .params import (…)` | stays same-package |
| `_beam_deposition.py:586` `from ._cross import (…)` | `from ..atomic.cross_sections import (…)` | becomes cross-package |
| `_heat.py:2` `from ._cross import charge_ex_react` | `from ..atomic.cross_sections import charge_ex_react` | becomes cross-package |
| `_cross.py:6` `from ._interp import interp_scalar_fused…` | `from ..numerics.interp import …` | becomes cross-package |
| `_adas.py:25` `from ..vars._cons import qe_SI` | `from ..constants import qe_SI` | re-rooted |
| `_cross.py:7` `from ..vars._cons import (…)` | `from ..constants import (…)` | re-rooted |
| `_fits.py:3` `from ..vars._cons import qe_SI` | `from ..constants import qe_SI` | re-rooted |
| `_heat.py:1` `from ..vars._cons import H_e_mass_ratio` | `from ..constants import H_e_mass_ratio` | re-rooted |
| `_cross.py:252,271,300` `from ..vars._coeff import …` (×3, function-local) | `from .coefficients import …` | becomes same-package |
| `funcs/__init__.py:1,2,3,12,13` | **deleted** (P-9) | — |
| `vars/__init__.py:1,2` | **deleted** (P-9) | — |

Note the direction of travel: three imports that are cross-package today
(`_cross` → `_coeff` reaching from `funcs/` into `vars/`) become
*intra*-package once both land in `cablp/atomic/`. That is the carve doing its
job, and it is the cheapest available evidence that the slate's grouping
matches the real coupling.

The 117 relative imports inside `solvers/_sim1d/` are untouched by both
commits.

---

## 8. Open questions for the reviewer

**Q1 — RESOLVED (Sol, 26dz): assent with a semantic amendment.** Both fields
are adopted into the schema doc, with the reading corrected: **a prefix row is
a compact mapping MACRO, not an assertion that the directory is a KB entity.**
Consequences already applied to these artifacts:

- The package row no longer hangs the macro off a representative `cablp`
  module locator. Both ends are now `anchor_kind: "directory"`, carrying
  `path` alone — no `symbol`, `signature` or `line_hint`, because inventing an
  import-qualified name for `scripts/` would be misleading.
- `old.path`/`new.path` equal their prefixes minus the trailing slash;
  `prefix_rule` is legal only on `move`/`move+rename`; `directory` is legal
  only with a `prefix_rule`.
- `covers` are base-revision old paths: nonempty, sorted, unique, tracked at
  `base_revision`, each strictly under `old_prefix`. Each derived destination
  is exactly `new_prefix + old_path.removeprefix(old_prefix)` and must exist
  at `new_revision`.
- No covered path may take a conflicting mapping from another prefix or file
  row; a finer symbol row on the *same* path pair is a legal override (that is
  what the `phase3_capture.py` `surface_change` row is).
- Coverage is now "an explicit file/module row **or** membership in exactly
  one prefix rule".
- `proposed_continuity: "same_entity"` **vectorizes** over the covered pairs.
  It proposes nothing about the directory, and Codex confirms or rejects per
  expanded file — the directory row does not force an all-or-nothing verdict.

The validator enforces all of it mechanically and gains a
`--emit-expanded` diagnostic printing the canonical per-file expansion,
labelled derived and non-authoritative. Authority:
`SOL_MANIFEST_SCHEMA_FIELDS_ANSWER_2026-08-26.md`.

**Q2 — `_coeff` lives in `vars/`, not `funcs/`.** R0.2 reads
"`funcs/_cross` + `_fits` + `_coeff` … → `cablp/atomic`". The file is
`cablp/cablp/vars/_coeff.py`. The *destination* is unambiguous and this map
assigns it to `cablp/atomic/coefficients.py`; flagging it rather than
silently correcting it, because a ruling that names a wrong source path may
have been written from a wrong mental model of what else `vars/` contains.

**Q3 — the two retirements are not source-inert, and R2 forbids semantic
change.** Both `results/compat.py` and `vars/_nn_table` require *edits to
solver-tree files* in the retirement commit: `results/io.py:16,380` and
`solver.py:205,10153` lose the compat call, `core/config.py:3` loses
`lookup_nn0` and its call site, and `smoke_sim1d.py:8074-8077` loses four
assertions. The golden is inert to all of it (the fixture holds only
`phase`/`time`/`y`), and production already short-circuits `nn_table` — but
"delete a module and the branch that called it" is a larger act than "move a
file", and R2's contract is pure moves/renames/import churn. Should the
retirements be their own commit with its own golden run and an explicit
statement that the deleted branches were unreachable at the stance? This map
assumes yes.

**Q4 — the committed phase-3 provenance record.** It carries 100 occurrences
of `cablp/cablp/…` describing a capture at a revision where that was correct.
This map says: move it, do not rewrite it, and accept that a re-verification
of *that* artifact against post-flatten code will report an anchor mismatch
unless the verifier is taught the flatten boundary. Confirm that is the
intended reading of HISTORY INVIOLABLE.

**Q5 — `git show <historical-base>:<path>` recipes.** Two scripts document
recipes that are only correct against pre-flatten revisions. Options: leave
them (correct for old bases, wrong for new), update them (vice versa), or make
them branch on whether the base predates the flatten commit. The third is real
work and beyond "pure moves"; this map recommends leaving them and adding one
sentence naming the flatten commit as the boundary.

**Q6 — RESOLVED (Tom, 26dz): `cablp/cathode/circuit.py` and
`circuit_idriven.py`.** Neither of this map's candidates survived. The module
solves the cathode/anode/bank **circuit** — the Thevenin load line off
`V_bank`, the `R_comp` partition, `phi_a`, `I_tot` — so `sheath.py` named a
part for the whole and `solver.py` named nothing while colliding with
`cablp/solvers/_sim1d/solver.py`. P-3 and P-4 are RULED, not PROPOSED, and
§3.4 and §7 carry the new names.

**Q7 — the golden re-anchor ordering. The `base_revision` half is DONE
(2026-08-26); the `new_revision` half is still open.**

`base_revision` was **pinned** in the draft to
`c018d925a90dcd86314866b8555fe42d9f22753d` — the tip this map was authored
against — rather than read from `HEAD`, which advances as this branch commits.
It is now re-stamped to `5caa8ce2cb5cfc3d89270d915a15213f1aec7fe5`, the R2 cut
revision (`agent-staging` == `campaign`, pushed), and the `covers` lists were
regenerated from `git ls-files` at that revision: **244 entries, byte-identical
to the previous lists**, because the window is 21 modified files and nothing
else. All 253 locators in the manifest (244 `covers` + 9 row `old` paths) were
re-resolved one by one with `git cat-file` at the new base; none was missing.
That they were unchanged is a fact about this window, not a property of the
map — regenerate again at any later re-cut.

A golden re-anchor **did land inside that window**: the sidecar
`scripts/baselines/production_discharge.json` moves from `steps: 94044` to
`steps: 62613` (the `dt_growth_recovery_patience` change), with
`saves` unchanged at **2620** — so `saves: 2620` remains the count of record at
the new base, and the digests quoted at commit time are the re-anchored ones.
**Whether that is the re-anchor this question was registered against is for the
reviewer to confirm**, not for this map to assume.

Still open, and still the reason this manifest is a DRAFT: it carries
`new_revision: "TBD-at-commit"` and `golden_gate.result: "TBD-at-commit"`, both
of which are filled in only when the flatten commit exists. Confirm the
ordering before the flatten branch is cut, because a flatten rebased across a
re-anchor invalidates the transcript, not the map.
