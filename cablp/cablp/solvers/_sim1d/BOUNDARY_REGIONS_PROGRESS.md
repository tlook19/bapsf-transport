# 1D Source Boundary — Progress Tracker

Lightweight status for the work described in
[`BOUNDARY_REGIONS_PLAN.md`](BOUNDARY_REGIONS_PLAN.md). Branch: **`1D_source`**.
Update the checkboxes, the "Current focus" block, and the Decisions log as part of
each milestone (do it *in the same commit* as the milestone's code).

## Current focus

- **Milestone:** M4 — term relocations via roles.
- **Next action:** plan M4 — anchor surface neutralization, the cathode surface
  terms (ion neutralization, sheath electron loss, ohmic) and the beam by
  `cell_role` / face helpers rather than `[0]`/`[-1]` (§6, §8). `cathode_adjacent_cells`
  and `anode_flanking_cells` are already in place. Forces §11 **#3** (cathode ion
  loss: volumetric term vs Bohm-flux face). Watch the two-anchoring-sites trap that
  bit M2. Note the anode's bilateral neutralization is **already handled** by M3's
  interception sink, so M4 must not add it again.
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
- [x] **M3 — Plasma faces & anode obstruction.** Generalize reflecting faces to the
  `plasma_open` array (interior cathode face); `heat_transmission = 0` at walls;
  confirm inert plenum plasma. Anode as a partial obstruction (collection-sink
  model) with `heat_transmission = (1−eta)`. (§5) — `flux.py` now imposes walls from
  `plasma_open` (pressure taken from the live cell) and scales faces by
  `plasma_transmission`; `conduction.py` applies `heat_transmission` at all three
  face-conductance sites (explicit flux, implicit tridiagonal, dt bound). The anode
  throttles heat and neutrals by `(1-eta)` while its advective face stays open, and
  `sources.anode_collection_rhs` removes plasma at the **Bohm sheath flux** on both
  mesh faces (decision 7, revised). `P_ohmic` now spreads along the gap weighted by
  `Te^-3/2` instead of piling into one cell. Golden `--verify` **bit-exact**; smoke
  extended; first full resolved discharge run and audited.
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
| 5 | anode as face vs cell | **resolved** | **face**. Fixing the cathode surface at `z=0` and the anode at `z=50` makes both *surfaces*, which have a position but no length. The mesh throttle and the bilateral neutralization split get the face they wanted; the sheath samples the cells flanking each face, which is what §7's bilateral treatment needs anyway. Cell roles `cathode`/`gap`/`puff` carry the term anchoring; `cathode_face_indices`/`anode_face_indices` carry the surfaces. (geometry rework, pre-M3) |
| — | pump mapping in resolved single-cathode | **resolved** | each end keeps its pump: `S_pump_L`→plenum (elbow-effective), `S_pump_R`→collector. Preserves today's total pumping speed; set `S_pump_R=0` for plenum-only. (M2) |
| — | neutral face aperture: mean vs restricting | **resolved** | **restricting (min)** of the two adjacent cells — a conductance between a wide and a narrow duct is set by the narrow one; mean would under-throttle the annulus. Bit-identical to the old mean whenever adjacent radii match. (M2) |
| — | resolved machine coordinates | **resolved** | cathode surface fixed at `z=0`, anode at `z=cathode_anode_gap_cm`=50. Two cell counts: `nx_gap`=5 across the gap (10 cm cells), `nx`=60 from anode to collector (1850 cm ⇒ 30.83 cm cells) — the old 100 cm source lump splits into the 50 cm gap plus 50 cm added to the column. `Lm` spans cathode→far end; plenum/obstruction sit at **negative z**, so total mesh > `Lm`. **Legacy geometry deliberately frozen** at 100/1800/100 so the golden stays bit-exact. (geometry rework, pre-M3) |
| 6 | asymmetric anode sheath | open | investigate at M5 |
| 7 | anode obstruction in subsonic regime | **resolved (revised)** | **Bohm collection only.** A sheath forms on every mesh wire, so ions arrive at `exp(-0.5)*n*c_s` set by the sheath, *not* by the bulk drift — a mesh in stagnant plasma still collects. The advective face therefore stays **open** (shrinking it too would remove the same particles twice, §5); `b_anode_advective_block` (default 0) can dial the `(1-eta)` reduction back in for a study, but it reflects rather than absorbs. Each mesh face is evaluated against its own side's `n`/`Te`, so collection is asymmetric — this also settles the *particle* half of #6. Supersedes the first pass, which used the intercepted directed flux `eta*n*u` and under-removed whenever the flow was subsonic. (M3, revised) |
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
  anode to the pump (§1) — is NOT yet modeled. §11 #5 is now settled (anode = face),
  so M3 has the face it needs: set the anode face's neutral aperture to
  `(1-eta)·πRm²` and its `heat_transmission` to `(1-eta)`.
- **Resolved geometry now uses machine coordinates** (cathode surface at `z=0`,
  anode at `z=50`). Consequences to remember:
  - `z_edges_cm` is **negative** behind the cathode. Anything assuming `z` starts at
    0 (plot scripts, `z`-indexed post-processing) needs checking before resolved
    results are plotted. Legacy is unaffected (`z_edges_cm[0] == 0`).
  - `length_cm.sum() != Lm` in resolved mode — the plenum and obstruction are extra.
    The invariant is `z_edges_cm[cathode_face] == 0` and the far end `== Lm`.
  - The 10 cm gap cells are the **smallest in the mesh**, so they set the explicit
    CFL (`center_distance_cm`, `timestep.py`). Resolved runs will take roughly 3x
    the steps of legacy at `nx_gap=5`; raising `nx_gap` scales that cost directly.
    The legacy golden verify is unaffected.
  - `cathode_length_cm`/`anode_length_cm` were **removed** — surfaces have no
    length. Anything referencing them is stale.
- **⚠ M5 must NOT re-add the anode particle sink.** `physics/sources.anode_collection_rhs`
  already removes `2*eta*I_i_a` worth of plasma — as two Bohm half-terms, each
  sampled on its own side of the mesh. M5's remaining job at the anode is the
  *sheath* (`P_anode_e`) and the circuit current, and that current should be the
  one this term already implies, not a second independent estimate.
- **Ion collection at a surface is set by the sheath, not the drift.** The first M3
  pass removed the intercepted *directed* flux `eta*n*u`, which under-removes as
  `u -> 0` (a mesh in stagnant plasma still collects at the Bohm flux), picks a
  single donor by flux sign when a real mesh collects from both sides at once, and
  ties collection to a bulk quantity the sheath makes irrelevant. Replaced by Bohm
  collection reusing `_cell_surface_particle_loss`, the same primitive the cathode
  and collector walls use. Smoke asserts the collection is unchanged when the bulk
  flow is switched off — the property the old model got wrong.
- **The anode collection is conservative and asserted:** each face's collected ions
  reappear as neutrals *on the side they came from* (a wire blocks the path to the
  other side, and the mesh throttles neutral flow between them, §7). Particle
  inventory rate is 0.
- **`P_ohmic` is spread along the cathode–anode gap, weighted by `Te^-3/2`.**
  `I^2 R_p` is dissipated in the plasma *between* the electrodes, and with uniform
  current density the power per unit length follows the local Spitzer resistivity,
  so it concentrates where the gap is coldest. `gap_cell_indices()` returns the gap
  cells (resolved) or the single source/end cell (legacy), so legacy deposition is
  bit-identical — a one-cell gap normalizes to exactly 1.0. This is a partial step
  toward §7 level (b), which replaces the single `R_p` with a path integral of
  resistivity over the same cells.
- **First full resolved discharge ran successfully** (`audit_sim1d_floor_activation.py
  --resolved`, backward_euler): final thermal 2.19e5 erg, Te floor clips 6.0% of
  cell-visits injecting +0.02% of final thermal, Ti 6.6% injecting +0.15%. Stable,
  not yet validated — that is M6.
- **Plenum plasma is inert but its `n` floor DOES bind** (7826 clips, all in cell 0
  at z=-50, **zero** energy injected; Ti never clips there at all). §5 predicted the
  floors would never bind. They do, benignly: with no source and recombination still
  running, plenum density decays below the floor and is pinned there — the floor
  holding the plenum at "plasma sits at the floor", not the cathode face leaking.
  It is a tiny unphysical *particle* source; worth a look if plenum inventory ever
  matters to the pump balance.
- **The audit's conduction verdict is vacuous under backward_euler** (its maximum
  principle forbids clipping). Re-run `--resolved --scheme tr_bdf2` to say anything
  about the scheme the production baseline actually uses.

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
