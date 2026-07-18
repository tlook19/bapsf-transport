# 1D Source Boundary — Progress Tracker

Lightweight status for the work described in
[`BOUNDARY_REGIONS_PLAN.md`](BOUNDARY_REGIONS_PLAN.md). Branch: **`1D_source`**.
Update the checkboxes, the "Current focus" block, and the Decisions log as part of
each milestone (do it *in the same commit* as the milestone's code).

## Current focus

- **Milestone:** all of M0-M6 complete. `resolved_boundaries` still defaults OFF.
- **Next action:** a decision, not code — whether resolved becomes the production
  default. That needs publication gates #4 and #5 in `THESIS_NOTES.md` closed
  first: is the ~2x peak `Te` vs legacy physical, and is the few-percent gap-mesh
  uncertainty acceptable or to be converged away. Only then does a re-baseline
  (re-`--capture` against a resolved config) make sense.
- **Remaining known work:** §11 #6 per-face anode sheath (particle half done,
  circuit still solves one `phi_a`); §7 level (b) resistivity path integral over
  the gap; and the pre-existing `b_Q*` / `b_ion_neutral_drag` gates, which are
  independent of this branch.
- **Blocked on:** nothing. Both gates green: `scripts/baseline_sim1d.py --verify`
  bit-exact with the switch off, `smoke_sim1d.py` passes.

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
- [x] **M4 — Term relocations via roles.** Surface neutralization, cathode/anode
  source terms (incl. the bilateral anode neutralization split), beam anchoring,
  ohmic — all anchored by `cell_role`, not `[0]`/`[-1]`. (§6, §8)
  - [x] **M4a — absorbing Bohm surfaces.** `plasma_absorbing` face array (empty in
    legacy); cathode surfaces and the collector drain at the Bohm flux with sonic
    outflow momentum via `sources.boundary_absorption_rhs`, applied one-sidedly.
    The volumetric `surface_neutralization` and the cathode circuit's particle loss
    are superseded wherever a face absorbs, so nothing is neutralized twice. Ohmic
    gap distribution landed earlier. Golden **bit-exact**; resolved floor clips fell
    ~150x.
  - [x] **M4b — remaining relocations.** `cathode_sample_indices()` moves the
    circuit's plasma sample off cell `[0]` (the plenum in resolved geometry) to the
    cathode-adjacent cell; `P_cathode_e` lands at the cathode and `P_anode_e` is
    split across the anode-flanking cells by each face's Bohm collection, retiring
    the TODO in `physics/cathode.py`; `beam_launch()` + a `direction` argument
    generalize `beam_absorption_weights`, which previously hard-rejected any launch
    index but `0`/`-1`, so the beam starts at the cathode surface and deposits
    nothing behind it. The cathode/anode `(1 + 2*eta)` split needed no code: M4a's
    absorbing-face guard already suppresses the volumetric loss in resolved, where
    `anode_collection_rhs` owns the `2*eta`. Golden **bit-exact**; smoke extended.
    (M4b initially broke resolved breakdown by moving the beam read off the index
    the solver writes to; fixed immediately after — see the notes.)
- [x] **M5 — Cathode solver split sampling (level a).** Distinct anode/cathode
  `(n, Te)` → distinct `I_i`; `P_cathode_e` / `P_anode_e` land at their own cells.
  Collapses to `I_i_a = 2·eta·I_i` when the sample cells coincide. (§7) —
  `solve()`/`solve_beam_system` take optional `cathode_current_A`,
  `anode_current_A`, `anode_T_e`; all default to `None`, so `_sim3` and legacy are
  bit-for-bit unchanged. The anode drop is rescaled by `tau_a = Te_anode/Te_cathode`
  in the residual (the loop equation is nondimensionalized in *cathode* `Te`, but
  `phi_a` is a real voltage), and `phi_a` plus the anode powers are evaluated at
  `Te_anode`. `I_i_a` is now the **same Bohm collection the fluid removes**, shared
  rather than re-derived, so circuit and fluid agree exactly (ratio 1.000000, was
  23.5). `SolverResult` gained `I_i_a` for diagnostics.
- [x] **M6 — Validation & sensitivity.** Smoke + order tests, legacy-equivalence
  assertions, sensitivity sweep (`Lcs`, `Rcs`, `Rsup`, `eta`, pump speed),
  re-baseline, update `manuscript/THESIS_NOTES.md`. (§10) — new
  `scripts/sweep_sim1d_resolved.py` (`--convergence`, `--sensitivity`). Order
  unchanged (2.00 TR-BDF2 / 1.99 CN / 1.00 BE control, floors inert). Golden
  bit-exact throughout. `THESIS_NOTES.md` §3 gained a resolved-boundary section
  with the legacy-vs-resolved deltas, the sensitivity ranking and the mesh
  uncertainty; two new publication gates (#4, #5) added. **No re-baseline
  performed** — the golden is the *legacy* trajectory and is still bit-exact, so
  there is nothing to re-capture until resolved becomes the production default,
  which is a separate decision (see notes).

## Decisions log

Resolve these as the milestones reach them; record the choice + one-line reason.
Open items are plan §11.

| # | decision | status | choice / note |
|---|---|---|---|
| 1 | `Lcs` obstruction: face vs cell | **resolved** | **cell** — a real `Lcs`-long annular duct between plenum and cathode, so its gas inventory reaches the pump (`Lcs`~25 is comparable to the ~30 cm cell size). Omitted entirely when `Lcs<=0`, which is the legacy limit. (M2) |
| 2 | cathode solver level (a) now / (b) path-integral later | **resolved (a)** | level (a) done at M5: the anode sheath is driven by anode-local `(n, Te)`. Level (b) — replacing the single `R_p` from `L_cath` with a resistivity path integral over the resolved gap — remains deferred; `gap_cell_indices()` already returns the cells it would need, and `P_ohmic` is already distributed over them by `Te^-3/2`. |
| 3 | cathode ion loss: volumetric vs Bohm face | **resolved** | **Bohm-flux absorbing face**, at the cathode *and* the collector, with **sonic outflow momentum** (leaves at `c_s` into the surface, not at the cell's drift `u`) so the loss drives flow toward the wall instead of deleting stationary plasma. Resolved cathode uses 1.0x face area; legacy keeps its 2.0 scale and its volumetric terms untouched. Implemented one-sidedly, not as a face flux — see notes. (M4a) |
| 4 | end default: collector vs mirror/twin | **resolved** | single-cathode collector default; the existing `TwinCathode` flag switches the far end to the symmetric plenum/cathode/anode mirror (no redundant knob). (M1) |
| 5 | anode as face vs cell | **resolved** | **face**. Fixing the cathode surface at `z=0` and the anode at `z=50` makes both *surfaces*, which have a position but no length. The mesh throttle and the bilateral neutralization split get the face they wanted; the sheath samples the cells flanking each face, which is what §7's bilateral treatment needs anyway. Cell roles `cathode`/`gap`/`puff` carry the term anchoring; `cathode_face_indices`/`anode_face_indices` carry the surfaces. (geometry rework, pre-M3) |
| — | pump mapping in resolved single-cathode | **resolved** | each end keeps its pump: `S_pump_L`→plenum (elbow-effective), `S_pump_R`→collector. Preserves today's total pumping speed; set `S_pump_R=0` for plenum-only. (M2) |
| — | neutral face aperture: mean vs restricting | **resolved** | **restricting (min)** of the two adjacent cells — a conductance between a wide and a narrow duct is set by the narrow one; mean would under-throttle the annulus. Bit-identical to the old mean whenever adjacent radii match. (M2) |
| — | resolved machine coordinates | **resolved** | cathode surface fixed at `z=0`, anode at `z=cathode_anode_gap_cm`=50. Two cell counts: `nx_gap`=5 across the gap (10 cm cells), `nx`=60 from anode to collector (1850 cm ⇒ 30.83 cm cells) — the old 100 cm source lump splits into the 50 cm gap plus 50 cm added to the column. `Lm` spans cathode→far end; plenum/obstruction sit at **negative z**, so total mesh > `Lm`. **Legacy geometry deliberately frozen** at 100/1800/100 so the golden stays bit-exact. (geometry rework, pre-M3) |
| 6 | asymmetric anode sheath | **partly addressed** | the *particle* half is done: each mesh face collects against its own side's plasma, and `P_anode_e` is split by that same weighting. The circuit still solves **one** `phi_a`, from a collection-weighted `Te_anode`. A genuinely per-face sheath (two `phi_a`) remains open — revisit if the gap and column sides diverge strongly in the M6 sweep. |
| 7 | anode obstruction in subsonic regime | **resolved (revised)** | **Bohm collection only.** A sheath forms on every mesh wire, so ions arrive at `exp(-0.5)*n*c_s` set by the sheath, *not* by the bulk drift — a mesh in stagnant plasma still collects. The advective face therefore stays **open** (shrinking it too would remove the same particles twice, §5); `b_anode_advective_block` (default 0) can dial the `(1-eta)` reduction back in for a study, but it reflects rather than absorbs. Each mesh face is evaluated against its own side's `n`/`Te`, so collection is asymmetric — this also settles the *particle* half of #6. Supersedes the first pass, which used the intercepted directed flux `eta*n*u` and under-removed whenever the flow was subsonic. (M3, revised) |
| — | single code path vs duplicate legacy path | **resolved** | single role/face-driven **operator** path (no flag branch); geometry construction uses two builders behind the switch so the legacy builder stays numerically verbatim and keeps the golden bit-exact. (M1) |
| — | golden-baseline form (M0 tooling) | **resolved** | notebook production config (implicit tr_bdf2 + Strang + Picard, cathode on), full packed-`y` trajectory to the dynamic current-trigger end, stored as NPZ. Chosen over a compressed-timing or cathode-off run (both weaker: floor-collapsed / miss the cathode/anode terms M4–M5 relocate). Implicit split makes the real-timescale run cheap (~65 s). |

## Notes / scratch

- **M6 headline: the resolved model is a different model, not a fix.** Peak `Te`
  51.6 eV vs legacy 24.5 eV (2.1x), final thermal -13%, discharge end +4.5%. That
  cannot be presented as a correction to previously published numbers.
- **`eta` and the pump path are strong new knobs** — `eta` 0.358->0.6 costs 49% of
  the final thermal energy, a pump elbow 42%. Comparable in leverage to the
  existing `b_Q*` factors, which is a caution: a model with two more strong knobs
  fits data more easily and is correspondingly weaker as evidence.
- **`eta = 0` was a broken reversibility path** and the sweep found it. The anode
  sheath relation is singular at zero anode current (an anode that collects
  nothing cannot close the circuit), so §13's "eta=0 -> transparent" limit holds
  for the neutral/heat throttles only. Now raises an explanatory error instead of
  an internal `-inf`. Pre-existing, not introduced by this branch.
- **No re-baseline was done, deliberately.** The golden fixture is the *legacy*
  trajectory and it is still bit-exact, so there is nothing to re-capture. A
  re-baseline only becomes meaningful if resolved is promoted to the production
  default.

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
- **~~Plenum plasma is inert but its `n` floor DOES bind~~ — FIXED in M4a, and my
  M3 diagnosis was wrong.** The 7826 `n` clips in cell 0 were not recombination
  draining a source-free cell. They were the *mis-anchored cathode term*: with
  `resolved_boundaries` on, cell `[0]` is the plenum, and `cathode_source_terms`
  was still depositing its `[0]`-anchored particle loss there, draining a
  plasma-dead cell onto its floor. Disabling that volumetric loss wherever an
  absorbing face handles it removed the cause — the resolved run now has **zero**
  `n` clips. A reminder that a floor binding is often a symptom of a misplaced
  term, not of the floor being needed.
- **An interior absorbing face cannot be a face flux.** The flux array telescopes,
  so a cathode surface expressed as a face flux hands the plasma it removes to the
  plenum *behind* it instead of out of the domain, and kicks that plasma-dead cell
  with sonic momentum while its density sits on the floor. `boundary_absorption_rhs`
  therefore applies the loss one-sidedly to the live cell; closed faces stay
  wall-like for both neighbours. The smoke conservation assertion is what caught
  this — keep it.
- **~~The resolved run no longer reaches breakdown~~ — RESOLVED, and the first
  diagnosis was wrong.** The cause was a bug introduced in M4b, not physics:
  `solve_beam_system` hard-coded `Te[0]/ne[0]/nn[0]` and *wrote* its beam
  quantities (`v_beam`, `n_beam`, `beam_cross`, `l_b`, `p_beam`) at index 0, while
  M4b moved `beam_launch()` to read the cathode-adjacent cell. So `beam_cross[launch]`
  was 0, the early return in `_beam_ionization_profile` fired, and **resolved mode
  had no beam ionization and no beam power at all**. With no igniter the discharge
  could not bootstrap. Fixed by giving `solve_beam_system` explicit
  `cathode_index`/`twin_index` (defaulting to 0/-1, so `_sim3` and legacy are
  untouched). Resolved now ignites normally: I_tot 2514 A at 4 ms, peaking ~2690 A
  vs legacy's ~2980 A.
- **⚠ Lesson: do not build a physics narrative on one failed run.** The first
  diagnosis blamed a mesh-dependent Bohm drain and concluded legacy breakdown was
  "partly a numerical artifact". The 1/cell_length drain scaling is real arithmetic
  (3222/s at 100 cm vs 9389/s at 10 cm) but it was *not* the operative cause, and
  resolved breaks down fine at 10 cm cells. Check for a plumbing bug before
  concluding the physics is mesh-limited.
- **Latent trap this exposed:** the cathode solver both *samples* and *writes* at
  the same index. Any future change to where the beam is read from must move the
  solver's index with it, or the beam silently vanishes rather than erroring.
- **Presheath-attenuated sheath factor (`presheath_alpha`).** `alpha_isat` =
  `exp(-1/2)` is the Boltzmann drop across the *whole* presheath, so it is only
  valid applied to the presheath-*entrance* density. A cell buried inside the
  presheath has already undergone part of that drop, so the factor applied to a
  local sample is `alpha_isat ** (min(L_cell, L_ps) / L_ps)`, with
  `L_ps = c_s / nu_in` computed from existing machinery. Limits:
  presheath fits inside the cell -> full `exp(-1/2)`; presheath much longer than
  the cell -> no correction (the cell already sits at the sheath edge). Measured
  alphas at 10 cm cells: 0.6065 (dense/hot, L_ps=5.4 cm), 0.8046 (L_ps=23 cm),
  0.9962 (cold/rarefied, L_ps=1300 cm). `b_presheath_length=0` recovers the old
  constant. Self-limiting (factor <= 1, so no flux cap is needed) and
  self-consistently mesh-independent in principle: refine the cell and the local
  density falls along the same Boltzmann profile the exponent compensates for.
  **Measured at M6, that claim holds only for what it targets** — see the gap
  refinement note.
- **⚠ Gap refinement (M6): the presheath correction helps the near-cathode
  quantities it targets, and only those.** `sweep_sim1d_resolved.py --convergence`
  over `nx_gap` = 5/10/20 (10 / 5 / 2.5 cm gap cells), max relative spread:

  | metric | correction ON | historical constant |
  |---|---|---|
  | `Te_max` | **7.1%** | 15.9% |
  | `n_max` | **1.5%** | 3.2% |
  | `final_time` | 1.3% | **0.8%** |
  | `thermal` | 5.0% | **3.7%** |

  So it roughly halves the mesh sensitivity of the near-wall quantities (and the
  OFF sweep shows the classic tell: `Te_max` sits at ~75 then collapses to 63.9 at
  the finest mesh), but it does *not* improve the global integrals and slightly
  worsens them. The earlier "self-consistently mesh-independent" claim was too
  broad. **Neither configuration is converged below a few percent at these
  resolutions**, so resolved results carry a few-percent mesh uncertainty at the
  `nx_gap = 5` default — state that alongside any resolved number.
  Cost scales as expected with the CFL: 21k / 37k / 69k steps, 90 / 154 / 281 s.
- **The anode is NOT presheath-attenuated, by the same rule rather than an
  exception.** A mesh's presheath is *geometric* — set by the wire spacing,
  sub-millimetre — not collisional, so it always fits inside a cell, the fraction
  is 1, and the full `exp(-1/2)` applies. Verified: anode collection is bit-identical
  at `b_presheath_length` 0 and 1 (-2.4381e16 both), while the cathode drain moves
  by the predicted 1.327x.
- **~~OPEN for M5: the circuit's anode current is ~23x off the fluid's~~ — FIXED in M5.** The
  circuit still assumes `I_i_a = 2*eta*I_i` scaled from the *cathode* cell, while
  `anode_collection_rhs` computes the real thing from anode-local plasma. On a
  representative resolved state (depleted cathode cell, dense gap) that is
  8.49 A assumed vs 199.9 A actual — a factor 23.5. This is exactly what §7 level
  (a) exists to fix, and it is now the largest known inconsistency in resolved mode.
- **~~OPEN for M5: circuit/fluid agreement on the cathode current~~ — the hook is in place.** M4a established that the absorbing face and the circuit compute the
  same Bohm flux. The fluid drain now carries `presheath_alpha`, but the circuit's
  `solve()` still uses the bare `exp(-1/2)` on the raw cell density, so they differ
  by the attenuation factor (1.327x in the test case). Restore this when M5
  restructures the sampling — the cleanest route is an explicit sample override on
  `solve_beam_system`, alongside the `cathode_index` parameter added there.
- **The sonic BC made the resolved run markedly healthier, not less stable.** Floor
  clips fell from 6.0%/6.6% of cell-visits (Te/Ti) to 0.041%/0.131%, injected energy
  from +0.02%/+0.15% to +0.0000%, cells clipped from 57/67 and 60/67 to 1/67 and
  23/67, and the step count *fell* (150276 -> 113771 state calls), so the feared CFL
  tightening did not materialize. Expected in hindsight: the old form deleted plasma
  that was not flowing anywhere, locally depleting cells onto the floor.
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
