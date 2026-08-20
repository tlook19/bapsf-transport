# Provenance of the golden baseline pins (`baseline_sim1d.BASELINE_*_OVERRIDES`)

**Recaptured 2026-08-20 (thread-24 R2b, reviewed-recapture protocol).** The
committed regression fixture `scripts/baselines/production_discharge.npz` is now
captured at **the shipped LAPDSim1D defaults** — `default_config()`, which after
the R2a and R2b fold-ins IS the g1atrim production package — plus the run-shape
overrides tabulated below and nothing else.

**The previous fixture is retired.** It held the 2026-07-22 operating point
behind ~30 explicit pins and is reproducible only at the anchor tag
`pre-refactor-2026-08-20` with its environment lockfile under `notes/env/`. Its
pin table is in this file's git history; do not reconstruct it here.

`scripts/baseline_sim1d.py` builds the configuration. It imports nothing from
the campaign drivers, so this note is complete on its own: for parameter
meanings see the docstrings in `cablp/solvers/_sim1d/core/config.py`, and for
every value the fixture inherits see
`cablp/solvers/_sim1d/core/config_defaults_provenance.md`.

## What kind of configuration this is

**The golden is a regression scaffold, not a physical claim** — it asserts that
the solver still computes what it computed, not that what it computes is right.
What changed at R2b is which configuration it scaffolds: it now tracks the
shipped defaults instead of a frozen historical point.

That swap is the whole reason the pin table collapsed from ~30 to one. Under the
old arrangement every default the campaign moved had to be pinned back, so the
table grew monotonically and the fixture drifted further from anything the
campaign ran — by the end it was pinning a machine length, a fueling waveform, a
neutral closure and a circuit that no production run had used for weeks. Pinning
a physics value here creates a SECOND stance that no one maintains. So:

- **A pin must be a MESH or COST choice.** Physics and stance values come from
  `default_config()`, unrestated.
- **The fixture is never recaptured to make a changed default look unchanged.**
  A recapture is a reviewed event with its own entry in the record below, and it
  says what moved and why.
- **A default must never be quietly changed to make an old fixture reproduce.**

`scripts/baselines/` is off limits to routine work. Recapture is the single
sanctioned exception and requires explicit authorization.

## The pin table

**Three run-shape choices, down from ~30 pins.** Two are `input_dict` entries;
the third is a run control.

| pin | value | why it is run shape, not physics |
|---|---|---|
| `nx` | `60` | Axial resolution of the far column: a pure cost knob. The campaign runs 268; a reviewer pays for this gate on the candidate branch and again post-merge, so the anchor runs the coarse mesh. Pinned rather than inherited because a future default-`nx` change would otherwise multiply every gate's runtime silently — the one way a shipped default can damage this fixture without changing any physics. |
| `max_steps_action` | `"stop"` | Consequence of the cap below, not an independent choice. The shipped default is `"raise"` because for a campaign arm a step cap means the run failed to finish; for this gate the cap IS the run length, so reaching it is the success path. The production stance carries the same value for the same reason. |
| `max_steps` (run kwarg) | `40000` | **The cost knob.** See below. |

`BASELINE_FLAG_OVERRIDES` — **empty by construction.** Every flag the production
package needs is a shipped default; a flag pinned here would be a stance choice,
and stance choices are exactly what this table no longer carries.

The remaining `BASELINE_RUN_KWARGS` (`t_end`, `dt`, `operator_split`) are `None`,
i.e. the solver's own run defaults: adaptive dt, the shipped operator split, and
a dynamic current-trigger end time that this fixture never reaches.

### Why the cap, and what it costs in coverage

**MEASURED 2026-08-20 on the shipped defaults at `nx = 60`:** the adaptive dt is
held near `3e-8 s` by the `surface_loss` limiter through the ignition foot and
does not recover — it fell `4.7e-7 -> 4.9e-8 -> 2.0e-8` over the first 15,000
steps. Running to the dynamic `t_end` (which resolves to `2.530938e-02 s`) would
take **~4 hours**, against ~8 minutes for the fixture this replaced, and a
reviewer runs the golden twice per merge. The retired fixture was cheap because
its 2026-07-22 operating point was a much less stiff configuration; tracking the
shipped defaults buys fidelity to what the campaign runs and pays for it in
stiffness.

The cap is a **step count rather than a `t_end`** deliberately: a step cap bounds
what a reviewer pays even if a future change shrinks the adaptive dt, whereas a
duration cap would let that same change lengthen the gate without bound. 40,000
steps sizes the gate at roughly the wall time of the fixture it replaced.

**The coverage consequence, stated plainly.** At 40,000 steps the trajectory
reaches `t ~ 1e-3 s`: the pre-breakdown foot, breakdown, and the first ~0.7 ms of
the discharge. It does **not** reach the discharge plateau or the afterglow. So
this gate certifies construction, the neutral equilibration, the geometry, the
cathode solve, beam deposition, the neutral closure, conduction, the circuit and
ignition — and it certifies nothing about late-time behaviour. A bit-exactness
gate only certifies what reaches its saved state; a change that touches only the
plateau or the afterglow can pass this golden without ever being exercised by it,
and needs its own evidence.

## Recapture record

**2026-08-20 — R2b re-anchor onto the shipped defaults (AUTHORIZED recapture,
Tom, 2026-08-20; thread-24 R2b).** The fixture moved from the ~30-pin 2026-07-22
operating point to `default_config()` + `nx = 60`. This is a wholesale change of
configuration, not a repair: the new and old trajectories are unrelated and no
comparison between them is meaningful. What the pass did, in order — folded
`heat_flux_limiter_f` (`0.3 -> 0.1`) into the shipped defaults so the limiter's
flag and coefficient live in one place; made `baseline_sim1d.py` self-contained
so no stance edit can reach the anchor; replaced the pin table; capped the run
at 40,000 steps after measuring the uncapped cost at ~4 hours; recaptured.
Captured twice from clean separate processes and verified byte-identical.

*A note on the fold that rode this pass:* `heat_flux_limiter_f` was held back
from R2a on the expectation that folding it would move the golden. It would not
have — the R1 stance-decoupling pass had already pinned `0.1` as a literal in
`BASELINE_PARAM_OVERRIDES`, which made the old fixture immune to that default.
Verified at R2b: the resolved OLD golden config is byte-identical across the
fold. The key still landed here, but as an ordinary value-neutral fold rather
than as the one that moved the anchor.

**2026-08-09 — returned-root sheath-ceiling fix (AUTHORIZED recapture, Tom,
2026-08-09).** *Applies to the retired fixture, retained as record.* The
current-driven sheath solve enforced `cathode_phi_c_cap_V` only at the bracket
ladder's doubling grid points, never on the root it returned; the fixture's own
ignition foot contained 34 such escaped solves (net phi_c up to 1.9669× the
1000 V cap — 1966.89 V returned at the cap), i.e. the committed trajectory
certified the defect. The fix (commit `8a09363`) tests the located J-root
against the cap and routes an at-or-above-cap root through the pre-existing
ceiling branch, so the foot solves move BY DESIGN and both goldens failed
against the old fixture with `max_rel=2.000e+00`,
`time_max_abs=1.113e-06 s`, character-identical on the pure and compiled paths.
Recaptured with the script's own `--capture` at the fixed code. The pin set was
unchanged: zero added, removed, or changed keys, and the sidecar params/flags
diff against the previous capture showed zero changed values — the 18 param keys
and 1 flag key newly recorded were config defaults added since the 2026-08-03
capture, already in effect for every verify since, recorded at their unchanged
live defaults. saves stayed 2545, cells stayed 72; steps 41054 → 40975 on the
corrected foot.
