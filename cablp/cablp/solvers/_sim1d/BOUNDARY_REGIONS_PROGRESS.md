# 1D Source Boundary — Progress Tracker

Lightweight status for the work described in
[`BOUNDARY_REGIONS_PLAN.md`](BOUNDARY_REGIONS_PLAN.md). Branch: **`1D_source`**.
Update the checkboxes, the "Current focus" block, and the Decisions log as part of
each milestone (do it *in the same commit* as the milestone's code).

## Current focus

- **Milestone:** M3 — plasma faces & anode obstruction.
- **Next action:** plan M3 — make `flux.py` consume the `plasma_open` array instead
  of hard-coding walls at the two external faces ([flux.py:64-67](physics/flux.py)),
  apply `heat_transmission` in `conduction.py` (walls already zero-flux by
  construction, so this must stay bit-exact), confirm inert plenum plasma with
  `audit_sim1d_floor_activation.py`, and model the anode as a partial obstruction
  (collection-sink model, `heat_transmission = 1-eta`). Forces §11 **#5** (anode as
  face vs cell) — carried over from M2 — and touches **#7** (subsonic regime).
- **Blocked on:** nothing. Acceptance gate: `scripts/baseline_sim1d.py --verify`
  stays bit-exact (switch off) and `smoke_sim1d.py` stays green.

## Milestones

Each milestone ends with a commit, a passing `smoke_sim1d.py`, and (from M1 on)
the §13 legacy-equivalence assertion still holding.

- [x] **M0 — Baseline & scaffolding.** Plan + this tracker + resume prompt
  committed. Golden baseline captured and committed *before* touching code
  (§12.0, §13): `scripts/baseline_sim1d.py` (`--capture`/`--verify`) plus the
  fixture `scripts/baselines/notebook_discharge.npz` (+ `.json` sidecar). Baseline
  = the production notebook config (implicit heat, tr_bdf2 + Strang + Picard,
  cathode on) run to its dynamic current-trigger end (2793 saves, 16084 steps,
  final_time = 2.7913e-2 s). Fresh re-run is **bit-exact**, and `smoke_sim1d.py`
  passes at baseline.
- [x] **M1 — Geometry schema** behind the `resolved_boundaries` master switch
  (default off). Typed segments; `cell_role` load-bearing; per-cell/face `area` +
  `hydraulic_radius`; face-property arrays `plasma_open` / `neutral_conductance` /
  `heat_transmission`. Legacy mode reproduces today (assert golden equivalence).
  (§3, §13) — `Sim1DGeometry` extended with `neutral_hydraulic_radius_cm`,
  `neutral_face_hydraulic_radius_cm`, `plasma_open`, `heat_transmission`,
  `neutral_face_conductance_cm3_s`. `build_geometry(input_dict, flags)` dispatches
  to `_build_legacy_geometry` (numeric lines verbatim) or `_build_resolved_geometry`
  (typed segments, collector default / twin-mirror per decision 4). `cells` now =
  `length_cm.size`. Golden `--verify` **bit-exact** with switch off; smoke covers
  both legacy schema defaults and resolved single/twin invariants. Operators not
  yet rewired (M2+).
- [x] **M2 — Neutral conductances** from the schema: generalized Clausing (area +
  hydraulic radius), prescribed apertures (cathode obstruction, anode mesh), pump
  relocated to the plenum with an effective speed, puff moved to its cell. (§4) —
  Clausing now reads the carried face area + hydraulic radius (with a NaN-sentinel
  direct-conductance escape hatch). Obstruction is a **real annular cell** (decision
  1), so its throttle comes from its own geometry, not a prescribed face aperture;
  `Rsup` blocks plenum volume only. Puff/pump anchored by role in **both** the
  explicit RHS and the implicit neutral matrix. Effective pump speed
  `1/S_eff = 1/S_pump + 1/C_elbow` on plenum pumps only. Anode mesh throttle
  deferred to M3/M4 (decision 3). Golden `--verify` **bit-exact**; smoke extended.
- [ ] **M3 — Plasma faces & anode obstruction.** Generalize reflecting faces to the
  `plasma_open` array (interior cathode face); `heat_transmission = 0` at walls;
  confirm inert plenum plasma. Anode as a partial obstruction (collection-sink
  model) with `heat_transmission = (1−eta)`. (§5)
- [ ] **M4 — Term relocations via roles.** Surface neutralization, cathode/anode
  source terms (incl. the bilateral anode neutralization split), beam anchoring,
  ohmic — all anchored by `cell_role`, not `[0]`/`[-1]`. (§6, §8)
- [ ] **M5 — Cathode solver split sampling (level a).** Distinct anode/cathode
  `(n, Te)` → distinct `I_i`; `P_cathode_e` / `P_anode_e` land at their own cells.
  Collapses to `I_i_a = 2·eta·I_i` when the sample cells coincide. (§7)
- [ ] **M6 — Validation & sensitivity.** Smoke + order tests, legacy-equivalence
  assertions, sensitivity sweep (`Lcs`, `Rcs`, `Rsup`, `eta`, pump speed),
  re-baseline, update `manuscript/THESIS_NOTES.md`. (§10)

## Decisions log

Resolve these as the milestones reach them; record the choice + one-line reason.
Open items are plan §11.

| # | decision | status | choice / note |
|---|---|---|---|
| 1 | `Lcs` obstruction: face vs cell | **resolved** | **cell** — a real `Lcs`-long annular duct between plenum and cathode, so its gas inventory reaches the pump (`Lcs`~25 is comparable to the ~30 cm cell size). Omitted entirely when `Lcs<=0`, which is the legacy limit. (M2) |
| 2 | cathode solver level (a) now / (b) path-integral later | open | (a) at M5; (b) deferred |
| 3 | cathode ion loss: volumetric vs Bohm face | open | volumetric recommended (M4) |
| 4 | end default: collector vs mirror/twin | **resolved** | single-cathode collector default; the existing `TwinCathode` flag switches the far end to the symmetric plenum/cathode/anode mirror (no redundant knob). (M1) |
| 5 | anode as face vs cell | open | linked to #6; decide at M3/M4. M2 deliberately deferred the anode *neutral* throttle too, so the mesh is still neutral-transparent — the puff→cathode back-path is not yet throttled. |
| — | pump mapping in resolved single-cathode | **resolved** | each end keeps its pump: `S_pump_L`→plenum (elbow-effective), `S_pump_R`→collector. Preserves today's total pumping speed; set `S_pump_R=0` for plenum-only. (M2) |
| — | neutral face aperture: mean vs restricting | **resolved** | **restricting (min)** of the two adjacent cells — a conductance between a wide and a narrow duct is set by the narrow one; mean would under-throttle the annulus. Bit-identical to the old mean whenever adjacent radii match. (M2) |
| 6 | asymmetric anode sheath | open | investigate at M5 |
| 7 | anode obstruction in subsonic regime | open | revisit if flow is subsonic at anode |
| — | single code path vs duplicate legacy path | **resolved** | single role/face-driven **operator** path (no flag branch); geometry construction uses two builders behind the switch so the legacy builder stays numerically verbatim and keeps the golden bit-exact. (M1) |
| — | golden-baseline form (M0 tooling) | **resolved** | notebook production config (implicit tr_bdf2 + Strang + Picard, cathode on), full packed-`y` trajectory to the dynamic current-trigger end, stored as NPZ. Chosen over a compressed-timing or cathode-off run (both weaker: floor-collapsed / miss the cathode/anode terms M4–M5 relocate). Implicit split makes the real-timescale run cheap (~65 s). |

## Notes / scratch

_(Running notes: surprises, dead ends, things the next session should know.)_

- **Golden baseline is bit-deterministic.** A fresh `--verify` reproduced the
  captured trajectory exactly (`max_rel=0`, `max_abs=0`). So M1+ equivalence is a
  hard target: a pure role-/face-driven refactor on legacy inputs should stay
  bit-exact; any nonzero `max_rel` means real arithmetic changed and needs
  explaining, not just tolerating.
- **Baseline end time is dynamic, not `default_t_end`.** With
  `phase_transition_mode="current"` (default) and no explicit `t_end`, `run()`
  shortens `t_end` when the discharge-current trigger fires (`solver.py` ~1043,
  1171), so the run ends at 2.79e-2 s, not the nominal 0.077 s. Deterministic
  (state-driven), so fine for a fixture — but if M1's refactor shifts the trigger
  step, trajectory length changes and `--verify` reports a shape mismatch rather
  than a value diff. Watch for that.
- **Fixture is 6.2 MB** (NPZ, full float64 packed trajectory). `*.h5`/`*.hdf5`
  are gitignored but `*.npz` is not, so it commits cleanly. If repo size matters,
  it can be slimmed to strided checkpoints + final state; kept full for now since
  the run is bit-exact and full compare is the sharpest check.
- **Keep `baseline_sim1d.py` in sync with `sim1d_run_and_plot.ipynb` cell 3.** The
  baseline config is a hand-copy of the notebook overrides. A deliberate
  re-baseline (re-`--capture`, reviewed) is the only correct way to change it.
- **M1 schema fields carry legacy defaults so the OFF path is bit-exact and the
  future array-driven operators are drop-in.** `plasma_open` = False only at the
  two external ends (legacy) or also at plenum↔cathode faces (resolved);
  `heat_transmission` 1 interior / 0 at plasma walls; `neutral_face_conductance_cm3_s`
  = NaN sentinel meaning "derive Clausing" (M2 populates apertures);
  `neutral_face_hydraulic_radius_cm[1:-1]` = today's `R_face`. No operator reads these
  yet — M2 (neutrals), M3 (flux/conduction) do.
- **Resolved geometry is structurally valid but physically provisional.** Per-role
  cell lengths (`plenum/cathode/anode/collector_length_cm`) and the obstruction/rod
  knobs (`Rcs`/`Lcs`/`Rsup`, default 0 = legacy limit) are placeholders; apertures
  are not yet applied (all faces full-bore). Turning the switch on today runs the
  *unrewired* index-based operators on the resolved grid → physically wrong, which is
  why it stays off. `cells` = `length_cm.size`; nothing reads `geometry.nx` directly.
- **Degenerate-resolved == legacy is NOT yet a test.** Resolved adds cells, so it
  can't be bit-identical to the lump until operators stop indexing `[0]`/`[-1]`
  (M3+). M1's guarantee is the switch-OFF golden bit-exactness only.
- **Puff/pump are anchored in TWO places — keep them in sync.** The explicit RHS
  (`physics/neutrals.py:neutral_source_sink_rhs`) and the implicit backward-Euler
  neutral matrix (`solver.py`, `_neutral_backward_euler_step`) both place the puff
  and pump terms. M2 relocated both by role; editing only one silently desyncs the
  neutral-equilibration path from the explicit one. Same trap likely applies to the
  M4 term relocations.
- **Neutral face apertures are now the restricting (min) of adjacent cells, not the
  mean.** Bit-identical wherever adjacent radii match, so the golden is unaffected —
  but a legacy config that sets `source_Rm`/`end_Rm` different from `Rm` will see
  its two end-face conductances change. Nothing in-repo does that. A true series
  combination of half-cell conductances would be the refinement if this matters.
- **The obstruction cell only exists when `Lcs > 0`.** At the default `Lcs = 0` the
  resolved layout is unchanged from M1, so all M1 resolved assertions still hold;
  `Rcs` alone does nothing without a duct length.
- **The anode is still neutral-transparent.** M2 deferred the mesh throttle
  (decision 3), so the central physics payoff — puffed gas leaking backward past the
  anode to the pump (§1) — is NOT yet modeled. M3/M4 must deliver it alongside §11 #5.

---

## Resuming in a new session

Paste this as the opening prompt for a fresh Claude Code session:

> We're implementing the resolved 1D source/end boundary for `LAPDSim1D`, on
> branch `1D_source`. **First read
> `cablp/cablp/solvers/_sim1d/BOUNDARY_REGIONS_PLAN.md` (the full design) and
> `BOUNDARY_REGIONS_PROGRESS.md` (milestone tracker) in that directory.** Then
> continue the next unchecked milestone in the tracker.
>
> Rules of the road:
> - Work one milestone at a time; end each with a commit that also updates the
>   tracker (check the box, log any decision).
> - Everything stays reversible (plan §13): the `resolved_boundaries` master switch
>   defaults **off** and must reproduce today's lumped behavior; keep the golden
>   baseline assertion passing.
> - After any `_sim1d/` change run the smoke test:
>   `conda activate fenicsx-env && cd cablp && python scripts/smoke_sim1d.py`
>   (exit 0 = pass).
> - Surface trade-offs on the open decisions (tracker "Decisions log") rather than
>   silently picking; this is thesis-facing code.
>
> Tell me which milestone you're starting and your plan for it before writing code.
