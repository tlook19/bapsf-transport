# Flatten execution notes — the first R2 commit

**Status: DRAFT, reviewer-gated. Nothing here has been executed.** This is the
ordered recipe for the commit that `restructure/manifests/delta_flatten.DRAFT.json`
describes, together with what must be true before it starts and what must be
re-measured after it lands.

> **Citations are pinned to `5caa8ce2cb5cfc3d89270d915a15213f1aec7fe5`, the R2
> cut revision (`agent-staging` == `campaign`).** Re-stamped 2026-08-26 from
> `c018d925a90dcd86314866b8555fe42d9f22753d` per `RENAME_MAP.md` Q7; every line
> citation below was re-derived at the new base by its quoted string. That
> window is 21 modified files with no adds, deletes or renames, so the move
> sequence, the file counts and the `.gitignore` work are all unaffected. One
> §3.5 edit was **withdrawn** (`_adas.py`, see `RENAME_MAP.md` §2.1) and the
> import totals in §4 moved. Locate every §3 edit by its quoted string, not by
> its line number, and re-derive the line numbers again if the flatten branch
> is cut from anything other than `5caa8ce`.

Read `RENAME_MAP.md` first — this document assumes its rules F1 and F2 and its
open questions Q1–Q7. Rule F2's destination (`cablp/scripts/` → `scripts/`) is
**PROPOSED and unratified**; if Tom rules otherwise, step 3 changes and nothing
else does.

---

## 0. Preconditions

Every one of these is a stop condition, not a warning.

1. **The golden re-anchor has landed** and its digest, `saves` count and
   transcript are the ones this commit will be gated against. The flatten is
   registered to follow the pending re-anchor (Q7). A flatten rebased *across*
   a re-anchor invalidates its transcript. **A re-anchor landed between the old
   and the new base** (`production_discharge.json`: `steps` 94044 → 62613,
   `saves` unchanged at 2620); the reviewer confirms whether that is the one
   Q7 names before treating this precondition as met.
2. **The legacy root `scripts/` archival is done in the main checkout** —
   this is the one step that cannot happen in a worktree, because the material
   is untracked and does not exist in a fresh worktree. In the main checkout at
   `/home/trloo/bapsf/bapsf-transport/scripts/` there are roughly 30 MB of
   untracked notebooks (`cross_sections.ipynb`, `rates_H.ipynb`,
   `rates_He.ipynb`, `langmuir2.ipynb`, `rk4.ipynb`, `debug_solver.ipynb`,
   `op.ipynb`, `inter.ipynb`, `mach2.ipynb`, …), `lecroydaq.py`,
   `generate_nn_table.py`, LaTeX build residue, `data/` and `plots/`. They are
   **disk-archived outside the repository**, not deleted:
   `generate_nn_table.py` is the evidence that `_nn_table` is ungenerable
   (R0.4), and `cross_sections.ipynb` is the sole consumer of `a_11s_double`,
   whose in-repo comment says to delete it *with that notebook's disposition,
   not before*.
3. **The nbstripout clean filter is installed in the checkout doing the
   commit.** `git config --get filter.nbstripout.clean` must print a command.
   The filter lives in `.git/config`, which is not versioned and is per-clone
   and per-worktree; a checkout without it will commit the notebook's ~2 MB of
   rendered outputs. `sim1d_run_and_plot.ipynb` moves in this commit, so the
   filter runs on it.
4. **A clean `git status`** apart from the intended work. The move touches 250
   tracked paths; a stray edit inside that set is unreviewable.
5. **The untracked OPEN-ADAS `.dat` files are present in whichever checkout
   runs the gates, and are relocated by hand when the tree moves.** The 38
   adf11/adf15 files under `cablp/cablp/vars/adas/` are gitignored by licence
   (`.gitignore:69`), so they exist only on disk. Consequences the mover must
   plan for:
   - **A fresh worktree does not have them**, and the smoke suite fails at the
     first `atomic_rate_model = "adas"` reaction call with
     `RuntimeError: OPEN-ADAS data file not found: …/vars/adas/scd96_he.dat`.
     Measured in this branch's own worktree, 2026-08-26. `cp <main
     checkout>/cablp/cablp/vars/adas/*.dat <worktree>/cablp/cablp/vars/adas/`
     fixes it; it is a bring-up step, never a code change.
   - **`git mv` will not carry them.** After step 4 the directory on disk is
     `cablp/vars/adas/`, and the `.dat` files must be moved there by hand in
     the same operation, in every checkout and every live worktree, or every
     ADAS-path gate in that checkout breaks with the message above. The same
     applies again, harder, at the **carve**, where P-2 moves the directory to
     `cablp/atomic/data/adas/`.
   - `README.md` in that directory **is** tracked and moves with the tree
     normally; only the `.dat` files need hand-carrying. The README's checksum
     table is what makes a mis-carried copy detectable, so verify against it
     after moving rather than trusting the copy.
6. **The compiled extension is rebuilt or removed.** A stale `.so` sitting at
   `cablp/cablp/funcs/_cathode_kernels_cy*.so` is gitignored and will not move
   with the tree. It is not a correctness hazard for the flatten (the package
   path is unchanged, so it will still import if it lands beside the moved
   source), but the reviewer's compiled golden must run against a `.so` built
   from the post-flatten tree, so rebuild after step 8.

---

## 1. The ordered sequence

Run from the repository root. Steps 2 and 3 are the two that must not be
reordered.

```bash
# 1. branch
git switch -c agent/r2-flatten agent-staging

# 2. clear the destination BEFORE the move.
#    R0.4 retirement; also a hard precondition for step 3, because
#    `git mv cablp/scripts scripts` refuses a non-empty destination.
git rm scripts/test_adapt.ipynb
rmdir scripts            # only succeeds once the untracked legacy content is
                         # archived out (precondition 2); if it fails, STOP

# 3. rule F2 -- the sim1d scripts take the vacated slot  (180 tracked files)
git mv cablp/scripts scripts

# 4. rule F1 -- the package flattens  (64 tracked files)
#    Done via a temporary name because `git mv cablp/cablp cablp` is a
#    move-onto-my-own-parent and git will not do it.
git mv cablp/cablp _cablp_pkg
git mv cablp/pyproject.toml pyproject.toml
git mv cablp/poetry.lock poetry.lock
git mv cablp/build_ext.py build_ext.py
git mv cablp/generate_eii_tables.py scripts/generate_eii_tables.py
git mv cablp/generate_he_ion_rate_table.py scripts/generate_he_ion_rate_table.py
rmdir cablp              # must now be empty of TRACKED files; if untracked
                         # build residue (dist/, __pycache__, *.so) remains,
                         # remove or relocate it deliberately, never with -rf
git mv _cablp_pkg cablp

# 5. .gitignore -- IN THIS COMMIT, not a follow-up (see section 2)
$EDITOR .gitignore

# 6. the content edits (section 3)
$EDITOR <the files listed in section 3>

# 7. stage and inspect BEFORE committing
git add -A
git status --short
git diff --cached --stat
git diff --cached -M --name-status | grep -v '^R' || echo "pure renames only"

# 8. rebuild the extension against the moved tree
python build_ext.py --inplace
```

**Why the temporary name in step 4.** `cablp/cablp` cannot be moved directly
onto `cablp`. The two-step through `_cablp_pkg` keeps every operation a real
`git mv` so the index records renames, which is what makes step 7's review
tractable and what lets the manifest's coverage check see a clean
delete/add pair per file.

**Why `git mv` at all, when git does not store renames.** Two reasons that are
not cosmetic: it keeps the working tree and index in agreement at every
intermediate step, and it makes `git diff -M --name-status` a usable review
instrument. The manifest deliberately does **not** consume rename detection as
evidence (the adopted contract forbids it); detection is a review convenience
here and a coverage cross-check in the validator, nothing more.

---

## 2. The `.gitignore` edit, and why it is in this commit

Ten pattern lines change (`RENAME_MAP.md` §5.7 has the table). Two of them are
ordering hazards rather than tidiness:

- **Line 1, `scripts/*`, must be dropped in this commit.** Otherwise the 180
  scripts moved in step 3 are invisible to `git add` and the commit silently
  loses them.
- **Lines 34–36 (`cablp/scripts/*.npz`, `*.json`, `*.png`) must be re-anchored
  to `scripts/…` in this commit.** Otherwise `scripts/*.json` stops being
  ignored and the working directory's run-record artifacts — roughly 2,500 of
  them in the main checkout — become stageable. That is precisely the noise the
  D-3 ruling (2026-08-20) removed, and `git add -A` in step 7 would sweep it
  in.

The same applies to lines 49–55 (`*.txt`, `*.cmd`, `*.exit`, `*.head`,
`*.start`, `*.time`, `*.wall`) and line 58 (`neutral_seed_db/`).

`.gitignore` is a **modification, not a move**, so it carries no manifest row:
the coverage check reads added and removed paths only. That is correct
behaviour, and it is the reason this note exists — the edit has no other
enforcement.

---

## 3. Content edits inside the flatten

249 tracked paths move and 1 is deleted. **237 of the 249 move byte-identical.**
The 12 that do not are listed below, with every changed line. Nothing else in
any moved file may change, and the AST-identity check applies to the prose
edits in §3.5 in its two-tier form (tier B, since they rewrite user-facing
strings).

### 3.1 Functional — repository-root anchors that are now off by one (3 edits)

| File (post-move) | Line | Change |
|---|---|---|
| `scripts/capture_phase3_rhs.py` | 27 | `REPOSITORY_ROOT = SCRIPT_DIR.parents[1]` → `SCRIPT_DIR.parents[0]` |
| `scripts/verify_phase3_source_capture.py` | 520 | `parents[2]` → `parents[1]` |
| `scripts/stage3_observability_check.py` | 54 | `parents[2]` → `parents[1]` |

### 3.2 Functional — repository-relative path strings (32 sites, 5 files)

All are enumerated with line numbers in `RENAME_MAP.md` §6.2. Summarised:

- `scripts/capture_phase3_rhs.py` — `PRODUCER_PATH` (:29), the eleven
  `PRODUCER_INPUT_PATHS` entries (:36–45), the `environment_lock` path (:128),
  and the `git ls-tree … -- cablp/cablp` pathspec (:186), plus three docstring
  mentions (:177, :181, and the `cablp/cablp` at :175).
- `scripts/verify_phase3_source_capture.py` — fixture paths and assertions at
  :126, :137, :151, :226, :229, :280, :290, :317, :332, :348, :360, :370,
  :477, :501.
- `scripts/stage3_observability_check.py` — `REPO_PATH` (:35).
- `scripts/smoke_sim1d.py` — the synthetic phase-3 fixture strings at :22623,
  :22630, :22641. **These are a gate**: they are fixture data rather than
  filesystem reads, but leaving them describing a repository shape that no
  longer exists makes the fixture a lie.
- `scripts/pec_band_fractions.py` — the mirrored fetch-instruction string
  (:98).

### 3.3 Functional — the one generated locator (1 site)

`cablp/solvers/_sim1d/results/phase3_capture.py:441` — the `producer_anchor`
prefix `"cablp/cablp/solvers/_sim1d/solver.py:"`. It has its own
`surface_change` row in the manifest, and its assertion partner is
`verify_phase3_source_capture.py:229` (§3.2). **The committed record
`scripts/baselines/phase3_rhs/77d675a4-….provenance.json` is NOT edited** —
100 occurrences of the old prefix stay exactly as they are, because that file
describes a capture at a revision where the old path was correct (Q4).

### 3.4 Functional — only if P-8's `scripts/` destination is taken (3 edits)

| File (post-move) | Line | Change |
|---|---|---|
| `scripts/generate_eii_tables.py` | 18 | `sys.path.insert(0, str(Path(__file__).parent))` → `.parent.parent` |
| `scripts/generate_eii_tables.py` | 24 | `OUT_DIR = Path(__file__).parent / "cablp" / "vars"` → `.parent.parent / …` |
| `scripts/generate_he_ion_rate_table.py` | 18 | `OUT_PATH = Path(__file__).parent / "cablp" / "vars" / "he_ion_rate.csv"` → `.parent.parent / …` |

Taking the zero-edit alternative (both files stay at the repository root)
removes this subsection and two manifest rows change destination.

### 3.5 Prose (no behaviour)

`cablp/vars/adas/README.md:17`, `cablp/funcs/_kernels.py:74` ("from the
`cablp/` directory" in the RuntimeError text), `build_ext.py:5,6`,
`scripts/profile_sim1d.py:64`.

**Withdrawn at the 2026-08-26 re-stamp:** `cablp/funcs/_adas.py:36` was listed
here. At this base its fetch-instruction message already reads
`cablp/vars/adas/` (commit `f04d8a8`), a CABLP-relative anchor that stays
correct across the flatten because the reader's working directory moves with
it. `_adas.py` moves byte-identical — `RENAME_MAP.md` §2.1 has the argument.

**Deliberately NOT edited:** `scripts/t23c_config_snapshot_delta.py:16` and
`scripts/mirror_fieldmap_bitexact_structural.py:54`, both of which document
`git show <base>:cablp/cablp/…` recipes. At a `<base>` older than this commit
the old path is the correct one. **Q5 is RULED (review, 26dw): they stay
unedited, and this commit is the named boundary.**

---

## 4. Import-churn scope

**Zero.** Not one of the 470 `from cablp…` / `import cablp…` statements
changes, and not one of the 117 intra-package relative imports changes.

The reason is worth stating because it is the whole argument for doing the
flatten first: the package's *importable* name is `cablp` before and after —
only the directory containing it moves — so no module-name resolution is
affected at any depth. The 190 `cablp.funcs.*` / `cablp.vars.*` statements
change at the **carve**, in a later commit, and that is where the import churn
lives (`RENAME_MAP.md` §7).

The measurements, reproducible from the repository root:

```bash
# 470 -- every absolute cablp import statement in the tree
git grep -n -E '^[[:space:]]*(from|import)[[:space:]]+cablp' -- '*.py' '*.pyx' | wc -l

# 279 cablp.solvers.* / 127 cablp.funcs.* / 63 cablp.vars.*  (+1 bare `import cablp`)
git grep -h -o -E '(from|import) cablp\.[A-Za-z0-9_.]+' -- '*.py' '*.pyx' \
  | awk -F. '{print $2}' | sort | uniq -c

# 190 -- the statements the CARVE rewrites (funcs/vars targets only)
git grep -n -E '^[[:space:]]*(from|import)[[:space:]]+cablp\.(funcs|vars)' \
  -- '*.py' '*.pyx' | wc -l

# 117 -- relative imports inside the package
git grep -n -E '^[[:space:]]*from[[:space:]]+\.' -- 'cablp/cablp/*' | wc -l
```

### Path assumptions, which are the real churn

Six `Path(__file__)`-derived anchors in the moved tree reach for the
repository root or the project root. Three break and are edited (§3.1); three
do not, and are listed so nobody re-derives them under time pressure:

| Site | Verdict |
|---|---|
| `scripts/smoke_sim1d.py:11143`, `:11177` — `cwd=…parents[1]` paired with the argument `"scripts/run_sim1d.py"` / `"scripts/plot_sim1d_run.py"` | **Invariant.** Both halves shift one level together: `<root>/cablp` + `scripts/…` becomes `<root>` + `scripts/…`. |
| `scripts/audit_sim1d_configs.py:28` — `…parents[1] / "cablp" / "solvers" / "_sim1d" / "config_snapshots.json"` | **Invariant.** `<root>/cablp/cablp/solvers/…` becomes `<root>/cablp/solvers/…`; the two changes cancel exactly. |
| `scripts/baseline_sim1d.py:66,67,73` and `scripts/golden_digest_gate.py:65,66,79` — `SCRIPT_DIR`-relative `baselines/` and `sys.path` | **Invariant.** The fixture and the stance loader move with the script. This is what keeps the golden gate-valid. |

---

## 5. Gate plan

**The requirement: `exact=True` through this commit.** R2 is bit-exact by
construction, and this commit is the strongest case of that — the golden's
config identity is untouched, since `default_config()` is the same source
text, the stance file `scripts/stances/g1atrim.toml` is the same bytes, and
`nx=60` is the same override. Any digest movement here is a defect, never a
re-anchor.

### On the coder's branch, before handoff

```bash
cd <worktree>
PYTHONPATH=<worktree> python scripts/smoke_sim1d.py
```

Exit 0 required. Note the invocation: after the flatten there is **no
`cablp/` project directory to `cd` into** — every command that CLAUDE.md
writes as `cd cablp && python scripts/…` becomes `python scripts/…` from the
repository root, and `PYTHONPATH` points at the worktree root rather than at
`<worktree>/cablp`. Getting this wrong resolves the editable install to the
main checkout and silently tests the wrong code.

The smoke suite is the right gate for the coder here because it is the one
that actually exercises the moved paths: `smoke_sim1d.py` subprocesses
`scripts/run_sim1d.py` and `scripts/plot_sim1d_run.py` with an explicit `cwd`
(§4), imports the stance loader through a mutated `sys.path`, and carries the
phase-3 fixture strings edited in §3.2. A path mistake in this commit shows up
there, not in the baseline.

### At the reviewer

1. **Compiled golden, post-merge on `agent-staging`:**
   `CABLP_COMPILED_KERNELS=1 python scripts/baseline_sim1d.py --verify` →
   must print `exact=True`. Rebuild the extension first
   (`python build_ext.py --inplace`) and probe `KERNEL_ID`
   (`cython/_cathode_kernels_cy/tierA+csda`) — the flatten does not change the
   extension's importable name, so a `KERNEL_ID` of `None` means the build
   went missing in the move, which is exactly the failure this commit could
   plausibly cause.
2. **Pure-path 4k-step digest leg** per the 2026-08-25 amendment.
3. **The manifest, validated:**
   ```bash
   python restructure/manifests/validate_manifest.py \
       restructure/manifests/delta_<shortsha>.json
   ```
   With `new_revision` filled in with the real commit sha, all six checks run;
   check (4) cross-checks the row coverage against `git diff --name-status
   --no-renames` and will fail if any moved or deleted path lacks a row. In
   DRAFT form (as committed on this branch) checks (3) and (4) and the new end
   of check (2) report SKIP, and the validator says so rather than claiming a
   pass.
4. **Orphan scan** for artifacts left in the worktree.

### What is NOT run here

The full pure golden. It is reserved for ANCHOR events (recaptures, numpy
upgrades, migrations) per CLAUDE.md. The flatten is adjacent to an anchor
event but is not one: it changes no arithmetic, and the pending re-anchor that
precedes it (Q7) is where the full protocol belongs.

### Digest re-quoting

Because the flatten lands **after** the pending golden re-anchor, the digest
and `saves` count quoted in the commit's manifest are the re-anchored ones. At
the re-stamped base the sidecar `scripts/baselines/production_discharge.json`
reads `saves: 2620` (unchanged across the re-anchor in this window, which moved
`steps` 94044 → 62613) — read it from the sidecar at commit time rather than
from here. The DRAFT manifest therefore carries
`new_revision: "TBD-at-commit"` and `golden_gate: {"result":
"TBD-at-commit", "saves": null}`. Both are filled in at commit time, in the
same edit that renames the file from `delta_flatten.DRAFT.json` to
`delta_<shortsha>.json`.

---

## 6. Follow-ups this commit deliberately does not do

- **The docs pass.** CLAUDE.md's "two script directories" section, its
  `cd cablp` invocations and its path references update in the R2 docs pass
  (26dj), not here. Keeping the flatten's diff to moves plus content edits in
  12 files is what makes it reviewable.
- **The carve.** Rule C, its 190 import-statement rewrites and its 13 relative
  rewrites are a separate commit with a separate manifest.
- **The retirements.** `results/compat.py`, `vars/_nn_table` and
  `vars/nn_table.csv` are a separate commit — and, per Q3, one that is *not*
  a pure move: it deletes call sites in `results/io.py`, `solver.py`,
  `core/config.py` and four assertions in `smoke_sim1d.py:8074-8077`. Only
  `scripts/test_adapt.ipynb` rides the flatten, because rule F2 cannot proceed
  around it.
- **`log.txt`.** Tracked at the repository root, matching no `.gitignore`
  pattern. Untouched here; worth a question, not a silent deletion.
