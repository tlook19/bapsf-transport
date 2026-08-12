# Restart: exporting an end state and resuming from it

`LAPDSim1D` can write its complete evolving state to a **restart payload** and
construct a later solver whose initial condition is that state. The intended
use is a two-stage run — a conducting-phase configuration whose end state
replaces the equilibrated neutral seed as a second stage's IC — but the
machinery is generic.

The correctness standard is **continuation bit-identity**: running `0 -> t_end`
in one call and running `0 -> t_mid`, exporting, restarting, `t_mid -> t_end`
must produce raw-byte-identical saved frames after `t_mid`. Anything weaker
would make a two-stage result incomparable with a single-stage one, which is
the whole point of the handoff.

## API

```python
from cablp.solvers._sim1d import LAPDSim1D, save_restart_state

sim = LAPDSim1D(params, flags)
sim.run(t_end=t_mid)
save_restart_state("stage1.restart.h5", sim)

resumed = LAPDSim1D({**params, "restart_from": "stage1.restart.h5"}, flags)
resumed.run(t_end=t_end)
```

`restart_from` defaults to `None`. With it unset nothing in this document is
reached: no payload is opened, no attribute is overwritten, and the solver's
behaviour is byte-identical to a build that lacks the feature.

The payload is a self-describing HDF5 file with its own format string
(`sim1d-restart-v1`), independent of the trajectory format `sim1d-hdf5-v1`.
Trajectory files are unchanged, and a restart payload is not a trajectory: it
carries one instant, not a history.

## Why an enumeration was necessary

A restart is only as honest as its inventory of evolving state. The solver
carries substantially more than the conserved fields: continuation caches that
seed the next nonlinear solve, latched phase triggers, accumulators, a ring
buffer, and — crucially — **run-loop controller state that lives in local
variables of `run()` rather than on the instance at all**.

Two of these are order-unity, not last-bit:

* `_cathode_beam_cross` is the previous solve's beam attenuation cross-section
  and seeds the next one. Measured (d1a, 2026-08-11): at beam turn-on, two
  paired solves differing only in this cache produce beam coupling lengths
  `l_b` that differ by ~1.0 *relative*. Dropping it does not perturb a restart,
  it relocates the beam.
* `t_last_save` (a `run()` local) sets the save cadence, and a save is not
  passive: `_trajectory_snapshot` calls `rhs_terms`, which issues a
  `solve_cathode_boundary(update_cache=True)` and **rewrites the cathode
  continuation cache**. Save times are therefore part of the trajectory, and a
  restart that re-aligns the save lattice changes the physics it saves.

The same measurement priced `_cathode_x0` (the sheath solve's warm-start seed)
at exactly zero: five seeds spanning a 3300x range give one distinct beam
result on both the current-driven and floating branches. It is carried anyway —
free, and the fixed point is not guaranteed to survive a stance change.

## The inventory

`CARRIED` — written to the payload and restored.
`DERIVABLE` — reconstructed from config/geometry at construction; not stored.
`DROPPED` — deliberately not carried; justified per row.
`REFUSED` — its presence makes the whole restart raise at construction.

### Conserved state

| state | site | class |
|---|---|---|
| `_y` (packed `n, nn, M, Ee, Ei` + optional `M_n, nn_a, M_n_a`) | `solver.py:3437` (`_set_state_vector` via accept) | CARRIED |
| `_state`, `_derived` | `solver.py:8575-8577` | DERIVABLE (unpacked from `_y`) |
| `_time` | `solver.py:3441` | CARRIED |
| `_floors`, `_ion_mass_g`, `_mu`, `_geometry` | construction | DERIVABLE |

The packed vector is restored **exactly**, not through the flooring path — the
same reasoning `_picard_restore` records at `solver.py:3811`: re-flooring an
already-floored accepted state is not guaranteed idempotent.

### Cathode continuation caches

| state | site | class |
|---|---|---|
| `_cathode_x0` | `solver.py:5578` | CARRIED |
| `_cathode_x0_twin` | `solver.py:5579` | CARRIED |
| `_cathode_beam_cross` | `solver.py:5581` | CARRIED |
| `_cathode_solve` | `solver.py:5577` | DROPPED — see below |

These are written by `solve_cathode_boundary(update_cache=True)`, which is
reached from three consumers: the SSPRK2 stages, `_update_current_phase_triggers`
(`solver.py:6534`) and `_trajectory_snapshot` -> `rhs_terms` (`solver.py:2488`).

`_cathode_solve` is the full solve *result object* from the last update. It is
read on the accept path (`solver.py:3508`) for the warming and coverage surface
updates and by the diagnostic snapshot. It is **not** carried: it is a derived
product of the state and the three caches above, it holds nested namespaces
with no serialisation contract, and — decisively — the accept path that reads
it always runs after a fresh solve within the same step. The restart's first
step re-establishes it before any consumer sees it. Its transient absence is
therefore observable only if a restart were resumed into a diagnostic read
before the first step, which the load path forbids (see *The leading save*).

### Circuit

| state | site | class |
|---|---|---|
| `_circuit_I_loop` | `solver.py:3704`, `3752`, `3845` | CARRIED |
| `_circuit_I_prev` | `solver.py:3519`, `3524` | CARRIED |
| `_circuit_V_cap` | `solver.py:3713`, `3758` | CARRIED |
| `_circuit_V_dis_step` | `solver.py:3705`, `3753` | CARRIED |
| `_circuit_V_dis_time_integral` | `solver.py:3754` | CARRIED |
| `_circuit_V_dis_prev_save` | `solver.py:7474` | CARRIED |

`_circuit_V_dis_prev_save` is the `(t, integral)` anchor from which each save
reconstructs the dt-averaged discharge voltage. It is mutated **once per
trajectory save**, and its value appears in the saved cathode diagnostics, so
it is load-bearing for frame identity even though it never touches the state.

### Cathode surface

| state | site | class |
|---|---|---|
| `_cathode_Ts_K` | `solver.py:3614` | CARRIED |
| `_cathode_theta` | `solver.py:3691` | CARRIED |
| `_cathode_energy_ledger_J` | `solver.py:3618-3623` | CARRIED |
| `_cathode_warming_model`, `_cathode_surface_ion_retention` | construction | DERIVABLE |

### Coverage closure (v2)

| state | site | class |
|---|---|---|
| `_coverage_f` (z-resolved) | `solver.py:1822` | CARRIED |
| `_coverage_deficit` (z-resolved `D`) | `solver.py:1928` | CARRIED |
| `_coverage_burn_accum` | `solver.py:1865`, armed `2894` | DROPPED — per-attempt |
| `_coverage_reservoir_burn_accum` | `solver.py:1861`, armed `2897` | DROPPED — per-attempt |
| `_coverage_w_accum` | `solver.py:1841`, armed `2900` | DROPPED — per-attempt |
| `_coverage_burn_weight` | `solver.py:2903` | DROPPED — per-attempt |
| `_coverage_reservoir_debit` | `solver.py:2411`, `2516` | DROPPED — per-RHS-evaluation |
| `_coverage_r`, `_coverage_tau_s`, `_coverage` | construction | DERIVABLE |

The five dropped members are armed by `_attempt_step` and cleared at
`solver.py:2963-2973`; the code states the invariant directly at
`solver.py:1594-1601` ("armed by `_attempt_step` and dropped with the attempt").
They are `None` at every point a restart can be taken, so dropping them is not
an approximation — there is nothing there.

### Ad-hoc probe neutral source

| state | site | class |
|---|---|---|
| `_probe` (amplitude, weights, waveform, zone) | construction | DERIVABLE |

**No payload member.** The instrument carries no evolving state at all: its
whole content is resolved from config at construction, and its per-step
waveform average is computed from the absolute clock and the step window,
both of which the restart already reproduces. It is listed here so its absence
reads as a decision rather than an omission.

### Phase triggers and the current-threshold interpolant

| state | site | class |
|---|---|---|
| `_t_prebreakdown_trigger` | `solver.py:6554` | CARRIED |
| `_t_breakdown_trigger` | `solver.py:6556`, `6598` | CARRIED |
| `_last_current_trigger_time` | `solver.py:6362` | CARRIED |
| `_last_current_trigger_I_tot` | `solver.py:6363` | CARRIED |
| `_current_trigger_samples` | `solver.py:6364` | CARRIED |
| `_t_ignition_abort` | `solver.py:6384` | CARRIED |
| `_ignition_abort_reason` | `solver.py:6385` | CARRIED |
| `_ignition_abort_context` | `solver.py:6386` | DROPPED — rebuilt diagnostic |
| `_ignition_abort_threshold_name` | `solver.py:6457` | CARRIED |
| `_run_start_for_phase_events` | `solver.py:4006` | CARRIED (diagnostic) |

The last-sample pair is not diagnostic: `_current_threshold_time`
(`solver.py:6340`) interpolates the threshold crossing between it and the
current sample, and the resulting trigger time becomes a phase boundary that
**caps the timestep** (`solver.py:4158-4169`). A restart that reset it would
place breakdown at a different instant.

`_run_start_for_phase_events` feeds only the `phase_events` diagnostic list.
It is carried so a resumed run reports events from the original origin rather
than from the restart instant; it does not affect stepping.

`_ignition_abort_context` is the one member of this table that is DROPPED, and
deliberately so. It is absent from `_RESTART_TRIGGER_ATTRS` (`solver.py:3855`)
and `_apply_restart_payload` then sets it to `None` outright
(`solver.py:4039`), with the reason stated in the comment above that line: the
context is a diagnostic record of a switch-open that has *already* happened,
while the abort reason, time and threshold name — all three carried — are what
the wind-down reads, and the context is rebuilt by the next guard evaluation.
Carrying it would restore a snapshot of guard internals that the next
evaluation overwrites anyway.

### Ignition stall monitor

| state | site | class |
|---|---|---|
| `IgnitionMonitor._samples` ring buffer | `ignition.py:244` via `solver.py:6951` | CARRIED |
| `IgnitionMonitor._stalled` latch | `ignition.py:255` | CARRIED |
| `_last_ignition_record` | `solver.py:6988` | DROPPED — rebuilt diagnostic |
| `window_s`, `rate_window_s`, `min_samples` | construction | DERIVABLE |

Not diagnostic: a trip calls `_open_ignition_switch` (`solver.py:6989`), which
sets `_t_ignition_abort` and shortens `t_end`. The buffer is fed once per
trajectory save, so it is coupled to the save lattice the run-loop block below
preserves.

`_last_ignition_record` is the exception and is DROPPED, alongside
`_ignition_abort_context` and for the same reason: `_apply_restart_payload`
resets it to `None` (`solver.py:4040`). It holds the most recent monitor
*report*, rebuilt in full at the next monitor evaluation (`solver.py:7310`)
from the ring buffer and the RHS terms; its only readers assemble abort
artifacts (`solver.py:6760`, `6816`). The latch and the samples, which do decide
whether a trip fires, are carried above.

### Electrode sample EMA

| state | site | class |
|---|---|---|
| `_sample_ema` (`{cell: [n, Te]}`) | `solver.py:6003` | CARRIED |
| `_sample_smoothing`, `_sample_smooth_cells` | construction | DERIVABLE |

Three consumers read the smoothed sample (`solver.py:6008` is the only
substitution site): the RHS/beam sheath solve, the accepted-state surface
re-solve at `solver.py:3542`, and the surface coverage update. The EMA is
seeded from the initial state at construction, so an uncarried restart would
restart the average — a first-order error in every sheath solve that follows.

### Floor ledger

| state | site | class |
|---|---|---|
| `_floor_ledger` | `solver.py:2771` | CARRIED |

Accumulator; appears in the result. No feedback into the state.

### Picard counters

| state | site | class |
|---|---|---|
| `_picard_extra_solves` | `solver.py:3846` | DROPPED |
| `_picard_triggered_steps` | `solver.py:3859` | DROPPED |

Counters only, read nowhere in the solver, and — decisively — they exist only
when `coupled_circuit_picard` is on (`solver.py:811-812` sits inside that
branch), so carrying them unconditionally would make the payload's own shape
depend on a non-structural flag. Each stage counts its own.

### Run-loop controller state (LOCAL to `run()`, not instance attributes)

This is the block a naive restart misses entirely, because none of it is an
attribute to grep for.

| local | site | class |
|---|---|---|
| `previous_accepted_dt` | `solver.py:4002`, set `4219`, read `4149` | CARRIED |
| `t_last_save` | `solver.py:4001`, set `4065`/`4231`, read `4031`/`4041` | CARRIED |
| `dt_growth_capped_streak` | `solver.py:4100`, set `4216-4218` | CARRIED |
| `consecutive_dt_min_clamps` | `solver.py:4085`, set `4125`/`4132` | CARRIED |
| `len(saved)` | `solver.py:3998` | CARRIED (as a frame-count offset) |
| `steps` | `solver.py:4083` | CARRIED (as an offset for the accepted-step guard) |
| `run_start` | `solver.py:4004` | CARRIED (via `_run_start_for_phase_events`) |
| `ignition_wall_clock_start` | `solver.py:4092` | DROPPED — see below |
| `progress_wall_start`, `last_progress_time`, `force_progress` | `solver.py:4005`, `4081-4082` | DROPPED — progress reporting only |
| `saved`, `diagnostics`, `timestep_rejection_events` | `solver.py:3998-4000` | DROPPED — each stage owns its own trajectory |

`previous_accepted_dt` is the dt-growth ramp's anchor: without it the first
step after a restart is not growth-capped and takes whatever the physics bound
allows, which changes that step's dt and every subsequent one.
`dt_growth_capped_streak` is the recovery hysteresis (the streak-update block in
`run()`, `solver.py:4519-4528`), asymmetric by design and therefore not
reconstructible from the state.

`ignition_wall_clock_start` is DROPPED because wall clock is a property of the
process, not of the trajectory. A two-stage run genuinely gets two wall-clock
budgets, and there is no honest alternative. The accepted-**step** cap is
carried as an offset, so the work-done budget does transfer. This is the one
place where a restarted run can differ from an unsplit one, it is disclosed
here, and it is inert at the shipped defaults (`ignition_wall_clock_cap_s` is
`0.0`, i.e. off).

### The leading save

`run()` opens with a leading `if should_save(self._time)` (`solver.py:4063`).
On a restarted run that instant was already saved by the producing stage, and
the save is not passive — it would issue a second cache-mutating
`solve_cathode_boundary` at the same instant. The restart therefore
**suppresses the leading save**, so exactly one save and one cache write happen
at the handoff instant across the pair, in the same order as an unsplit run.
The consequence is that a stage-2 trajectory does not repeat the handoff
frame; stage 1's last frame is that frame.

### Warn-once latches

| state | site | class |
|---|---|---|
| `_beam_gap_ledger_warned` | `solver.py:5627`, reset `solver.py:3968` | DROPPED |
| `_dvm_ion_shortfall_warned` | `solver.py:8546` | DROPPED (arm is REFUSED) |

`_beam_gap_ledger_warned` is reset by `run()` itself at entry, so carrying it
would be overwritten anyway. The effect of dropping it is that a warning
already emitted before the handoff may be emitted once more after it. No
trajectory consequence.

### Result handles

`_last_result`, `_last_neutral_equilibration_result`,
`_last_neutral_equilibration_summary`, `_run_via_start_simulation`
(`solver.py:4265`, `4404-4405`, `4573`): DROPPED. They hold the previous call's
return value and feed nothing in the stepping path.

### Refused subsystems

| subsystem | gate | why refused |
|---|---|---|
| K4a kinetic response functions (`_kinetic`) | `neutral_model == "kinetic"` | `solver.py:709-728`: frozen velocity grid, per-channel response functions, refresh clocks. Not serialised. |
| K2a transient DVM (`_dvm`, `_dvm_*`) | `neutral_model == "kinetic_dvm"` | `solver.py:2328`: a full distribution function plus ion-debt ledger and a neutral clock (`solver.py:8496-8560`). Not serialised. |
| neutral equilibration | `neutral_equilibration` flag | `start_simulation()` would run the puff/off accumulation and **overwrite** the restored IC (`solver.py:4550-4571`). A restart *is* the seed. |

Each raises `ValueError` at construction when combined with `restart_from`.
They are refusals, not partial loads: the alternative — carrying the fluid
state and silently reseeding a kinetic distribution — would produce a run that
looks like a continuation and is not one.

## How the inventory is checked

`scripts/restart_bitidentity.py` runs a window unsplit and split and compares
every saved frame after the handoff at **raw uint64** — the float bytes, with no
tolerance anywhere — across the conserved rows, the derived primitives, and
every per-frame cathode diagnostic the run publishes.

Passing that is necessary but not sufficient: a comparison can be green because
nothing it claims to test was exercised. So the script also runs a **negative
control** per carried member (`--negative-control`): it corrupts one member in
a written payload and requires the comparison to FAIL. A member whose
corruption changes nothing is either untested or genuinely inert, and the
script refuses to guess between those — the inert ones are listed with their
evidence in `INERT_EXPECTATIONS`, and anything else is reported as a defect.

This is what caught the two mistakes that would otherwise have shipped a
green-but-hollow gate: a diagnostic key list naming keys that do not exist
(compared nothing, silently), and a split point taken at a nominal time rather
than the exact float of a save instant.

## What "bit-identical" does and does not promise

A restart reproduces, exactly, the continuation of **the step sequence stage 1
actually took**. It reproduces an unsplit run's frames when — and only when —
the handoff instant is one the unsplit run also steps to exactly.

That is not a limitation in practice: every save instant IS a step boundary,
because `next_save_time_after` caps the adaptive step to land on it
(`solver.py:4170-4177`), and an export is naturally taken at the end of a
stage. It does matter to anyone comparing a split run against an unsplit one,
because the save lattice is **accumulated** (`next = t_last_save + dt_save`)
and therefore carries float drift: at `dt_save = 1e-4` the third save lands on
`3.0000000000000003e-04`, not `3e-04`. Stopping stage 1 at the nominal `3e-04`
stops it one ulp before any instant the unsplit run visits, which re-phases
every later save and compares different instants. `scripts/restart_bitidentity.py`
snaps its split point onto the exact float of an unsplit save for this reason,
and says so when it does.

Members whose carry is **inert at the shipped defaults** — real, carried, but
which a default-configured acceptance run cannot distinguish from dropping:

* `dt_growth_capped_streak`, unless `dt_growth_recovery_patience > 0`
  (default `0` presence-gates the whole branch in `run()`: neither the
  widened-ceiling read at `solver.py:4448` nor the streak update at
  `solver.py:4519` runs). The
  acceptance harness raises it so the carry is exercised.
* `_cathode_x0`, whose consequence the d1a probe measured at exactly zero over
  a 3300x seed span. Carried regardless: the fixed point is a property of the
  current stance, not a guarantee.
* `_cathode_beam_cross`, which is identically zero until the sheath potential
  crosses the ionization threshold — roughly `2e-4 s` into the production
  stance. An acceptance window shorter than that cannot test it, which is why
  both harness scenarios hand off in the beam-live regime instead.

## Compatibility refusal

The payload records the producing run's full resolved `params`/`flags`, the
cell count, and the packed state-field layout. On load the following must match
the constructed solver exactly, or the load raises:

* cell count and packed state-vector length,
* the packed state field names (which optional rows exist),
* the **structural** config keys — the ones that decide what the payload's
  members mean rather than merely how big a number is:
  flags `coverage_closure`, `neutral_momentum`, `neutral_two_zone`,
  `TwinCathode`, `Plasma`, `cathode_coupling`; params `neutral_model`,
  `cathode_warming_model`, `cathode_surface_model`,
  `cathode_sample_smoothing`, `phase_transition_mode`.

Every other config key is free to differ — that is what makes a two-stage
hybrid possible — and the producing run's full config is retained in the
payload so any difference is auditable after the fact.
