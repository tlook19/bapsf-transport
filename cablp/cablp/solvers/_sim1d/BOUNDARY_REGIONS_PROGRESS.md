# 1D Source Boundary — Progress Tracker

Lightweight status for the work described in
[`BOUNDARY_REGIONS_PLAN.md`](BOUNDARY_REGIONS_PLAN.md). Branch: **`1D_source`**.
Update the checkboxes, the "Current focus" block, and the Decisions log as part of
each milestone (do it *in the same commit* as the milestone's code).

## Current focus

- **Milestone:** M1 — geometry schema behind the `resolved_boundaries` master switch.
- **Next action:** plan M1 (typed segments, load-bearing `cell_role`, per-cell/face
  `area` + `hydraulic_radius`, face-property arrays) and surface the schema decisions
  it forces before writing code. The M0 golden baseline is now the acceptance gate:
  `python scripts/baseline_sim1d.py --verify` must stay green (currently bit-exact).
- **Blocked on:** nothing.

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
- [ ] **M1 — Geometry schema** behind the `resolved_boundaries` master switch
  (default off). Typed segments; `cell_role` load-bearing; per-cell/face `area` +
  `hydraulic_radius`; face-property arrays `plasma_open` / `neutral_conductance` /
  `heat_transmission`. Legacy mode reproduces today (assert golden equivalence).
  (§3, §13)
- [ ] **M2 — Neutral conductances** from the schema: generalized Clausing (area +
  hydraulic radius), prescribed apertures (cathode obstruction, anode mesh), pump
  relocated to the plenum with an effective speed, puff moved to its cell. (§4)
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
| 1 | `Lcs` obstruction: face vs cell | open | decide at M2, from `Lcs` vs cell size |
| 2 | cathode solver level (a) now / (b) path-integral later | open | (a) at M5; (b) deferred |
| 3 | cathode ion loss: volumetric vs Bohm face | open | volumetric recommended (M4) |
| 4 | end default: collector vs mirror/twin | open | single-cathode collector default |
| 5 | anode as face vs cell | open | linked to #6; decide at M3/M4 |
| 6 | asymmetric anode sheath | open | investigate at M5 |
| 7 | anode obstruction in subsonic regime | open | revisit if flow is subsonic at anode |
| — | single code path vs duplicate legacy path | open | plan §13 recommends single path |
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
