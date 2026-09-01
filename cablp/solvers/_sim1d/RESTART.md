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
| `_y` (packed `n, nn, M, Ee, Ei` + optional `M_n, nn_a, M_n_a, En`) | `_accept_step_attempt` -> `_set_state_vector` (`solver.py`) | CARRIED |
| `_state`, `_derived` | `_set_state_vector` (`solver.py`) | DERIVABLE (unpacked from `_y`) |
| `_time` | `_accept_step_attempt` (`solver.py`) | CARRIED |
| `_floors`, `_ion_mass_g`, `_mu`, `_geometry` | construction | DERIVABLE |

The packed vector is restored **exactly**, not through the flooring path — the
same reasoning `_picard_restore` records in `solver.py`: re-flooring an
already-floored accepted state is not guaranteed idempotent.

### Cathode continuation caches

| state | site | class |
|---|---|---|
| `_cathode_x0` | `solve_cathode_boundary` (`solver.py`) | CARRIED |
| `_cathode_x0_twin` | `solve_cathode_boundary` (`solver.py`) | CARRIED |
| `_cathode_beam_cross` | `solve_cathode_boundary` (`solver.py`) | CARRIED |
| `_cathode_tail_anode_I` | `solve_cathode_boundary` (`solver.py`) | CARRIED |
| `_cathode_solve` | `solve_cathode_boundary` (`solver.py`) | DROPPED — see below |

`_cathode_tail_anode_I` is the A2a anode tail cull's one-step lag: the current
the anode collected directly from the QL tail walkers on the last accepted
step, read by the NEXT step's sheath solve. It is carried on its own payload
key rather than inside the strict inventory loop, so a payload written before
the cull existed still loads and restores it to `0.0` — which is exactly what
an unarmed run carries at every instant.

These are written by `solve_cathode_boundary(update_cache=True)`, which is
reached from three consumers: the SSPRK2 stages, `_update_current_phase_triggers`
(`solver.py`) and `_trajectory_snapshot` -> `rhs_terms` (`solver.py`).

`_cathode_solve` is the full solve *result object* from the last update. It is
read on the accept path (`_accept_step_attempt` in `solver.py`) for the warming and coverage surface
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
| `_circuit_I_loop` | `_accept_step_attempt` (`solver.py`), `_accept_step_with_picard` | CARRIED |
| `_circuit_I_prev` | `_accept_step_attempt` (`solver.py`) | CARRIED |
| `_circuit_V_cap` | `_accept_step_attempt` (`solver.py`) | CARRIED |
| `_circuit_V_dis_step` | `_accept_step_attempt` (`solver.py`) | CARRIED |
| `_circuit_V_dis_time_integral` | `_accept_step_attempt` (`solver.py`) | CARRIED |
| `_circuit_V_dis_prev_save` | `_cathode_diagnostic_snapshot` (`solver.py`) | CARRIED |

`_circuit_V_dis_prev_save` is the `(t, integral)` anchor from which each save
reconstructs the dt-averaged discharge voltage. It is mutated **once per
trajectory save**, and its value appears in the saved cathode diagnostics, so
it is load-bearing for frame identity even though it never touches the state.

### Vessel common-mode node (`regime_vessel_node`)

| state | site | class |
|---|---|---|
| `_vessel_V_cm` | `_vessel_advance`, from `_accept_step_attempt` | CARRIED (armed only) |
| `_vessel_charge_ledger_C` | `_vessel_advance` | CARRIED (armed only) |
| `_vessel_wall_currents_A` | `_vessel_advance` | DROPPED — re-read next step |
| `_vessel` (resolved constants) | construction | DERIVABLE |

These ride the **`circuit` group**, written and read only when the node is
armed, so a payload from a run without the node is structurally what it always
was and no format version moved. `_vessel_wall_currents_A` is the last step's
`(I_e, I_i, I_leak)` triple and exists for the diagnostics alone; the next
accepted step re-reads all three from the state before using them, so carrying
it would change nothing.

`regime_vessel_node` is a **structural flag key**: resuming across a change of
arming would either drop an evolved potential or leave one unread, so the
compatibility check refuses instead.

### Cathode surface

| state | site | class |
|---|---|---|
| `_cathode_Ts_K` | `_accept_step_attempt` (`solver.py`) | CARRIED |
| `_cathode_theta` | `_accept_step_attempt` (`solver.py`) | CARRIED |
| `_cathode_f_em` | `_advance_emitting_area_fraction`, from `_accept_step_attempt` | CARRIED (armed only) |
| `_cathode_energy_ledger_J` | `_accept_step_attempt` (`solver.py`) | CARRIED |
| `_anode_energy_ledger_J` | `_accept_step_attempt`, B4 anode jet | CARRIED (armed only) |
| `_cathode_warming_model`, `_cathode_surface_ion_retention` | construction | DERIVABLE |

`_anode_energy_ledger_J` is the anode mesh's own cumulative surface energy book
(`ion_incident`, `backscatter`), and exists only while
`neutral_kinetic_dvm_anode_jet` is armed. Like the cathode ledger it is
CARRIED rather than derivable — it is an integral over accepted steps — and
like `_cathode_f_em` it is PRESENCE-GATED, so a payload from a run without the
channel carries no such rows and payloads written before the channel existed
stay readable.

`_cathode_f_em` is the emitting-area closure's whole state — the lit fraction
of the cathode face. It rides the **`cathode` group** and is written and read
only when `cathode_emitting_area` is armed, so a payload from a run without the
closure is structurally what it always was and payloads written before the
closure existed stay readable. It is one scalar, but it multiplies every
annulus's emission, so dropping it would relocate the discharge current rather
than perturb it; the `emitting_area` scenario of
`scripts/restart_bitidentity.py` exports it mid-climb and its negative control
breaks identity. `cathode_emitting_area` is a **structural flag key**: resuming
across a change of arming would either drop an evolved fraction or leave an
armed closure sitting at its seed, so the compatibility check refuses instead.

### Cathode-jet arming latch

| state | site | class |
|---|---|---|
| `_jet_armed` | `_update_jet_arming_latch` (`solver.py`) | CARRIED (armed only) |
| `_jet_arming_censored_steps` | `_accept_step_attempt` (`solver.py`) | CARRIED (armed only) |
| `_jet_arming_transitions` | `_update_jet_arming_latch` (`solver.py`) | CARRIED (armed only) |
| `_jet_arming_last_transition_s` | `_update_jet_arming_latch` (`solver.py`) | CARRIED (armed only) |
| `_jet_arm_current_A`, `_jet_disarm_current_A`, `_jet_arming_active` | construction | DERIVABLE |

`_jet_armed` is order-unity, not last-bit. It decides whether the next step's
cathode jets launch at all AND whether the cathode surface is debited for
them — one latch state read by both sides — so dropping it does not perturb a
restart, it **re-censors a channel the producing run had already brought into
existence**, from the resumed run's first step until the booked ion current
climbs back across `neutral_jet_arm_current_A`. The three census members carry
alongside it so a resumed run reports one run's arming history rather than two.

They ride the **`cathode` group** on their own keys rather than joining
`_RESTART_CATHODE_ATTRS` (`solver.py`) and its strict loop, and they are
**PRESENCE-GATED** on a criterion being declared, exactly like `_cathode_f_em`
and the vessel node. At the shipped inert default (`arm = 0`) the latch is
permanently armed and the counters permanently at their seed, so no rows are
written and the payload is byte-unchanged — this member is absent there, not
carried-but-inert.

**The arming keys are deliberately NOT structural**, and the two asymmetric
resumes are handled rather than refused:

* resuming with NO criterion from a payload that has the rows ignores them.
  With no criterion the jets are unconditionally live and nothing reads the
  latch; restoring it would only mislabel a run summary.
* resuming WITH a criterion from a payload that has no rows — which is exactly
  what a payload written before this carriage looks like — **LOADS**. It does
  not raise. `_apply_restart_payload` (`solver.py`) keeps the constructor's
  disarmed seed and **warns** that it did, naming the consequence: until the
  current re-crosses the arm threshold the jets launch nothing, the surface is
  debited nothing, and the continuation is not bit-identical.

The format string did not move: these are presence-gated additions to an
existing group, the same shape `_cathode_f_em` and the anode ledger took, so
every payload already on disk stays readable under `sim1d-restart-v1`.

Its negative control does **not** live in `scripts/restart_bitidentity.py`:
none of that harness's four scenarios declares an arming criterion, so all
four would report the rows absent and skip. The control is `gate_ja8` in
`scripts/verify_sim1d_k2_dvm.py`, which exports a handoff at which the latch
is armed with a nonzero census, resumes it, and then resumes the SAME payload
with the four rows deleted — the pre-carriage shape — and requires that leg to
come up disarmed, warn, and diverge.

### Coverage closure (v2)

| state | site | class |
|---|---|---|
| `_coverage_f` (z-resolved) | `_advance_coverage_fraction` (`solver.py`) | CARRIED |
| `_coverage_deficit` (z-resolved `D`) | `_advance_coverage_deficit` (`solver.py`) | CARRIED |
| `_coverage_burn_accum` | `_accumulate_coverage_burn` (`solver.py`), armed in `_attempt_step` | DROPPED — per-attempt |
| `_coverage_reservoir_burn_accum` | `_accumulate_coverage_burn` (`solver.py`), armed in `_attempt_step` | DROPPED — per-attempt |
| `_coverage_w_accum` | `_accumulate_coverage_burn` (`solver.py`), armed in `_attempt_step` | DROPPED — per-attempt |
| `_coverage_burn_weight` | `_attempt_step` (`solver.py`) | DROPPED — per-attempt |
| `_coverage_reservoir_debit` | `rhs_terms` (`solver.py`) | DROPPED — per-RHS-evaluation |
| `_coverage_r`, `_coverage_tau_s`, `_coverage` | construction | DERIVABLE |

The five dropped members are armed by `_attempt_step` and cleared at
`floor_with_ledger` (`solver.py`); the code states the invariant directly at
`solver.py` ("armed by `_attempt_step` and dropped with the attempt").
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
| `_t_prebreakdown_trigger` | `_update_current_phase_triggers` (`solver.py`) | CARRIED |
| `_t_breakdown_trigger` | `_update_current_phase_triggers` (`solver.py`) | CARRIED |
| `_last_current_trigger_time` | `_record_current_trigger_sample` (`solver.py`) | CARRIED |
| `_last_current_trigger_I_tot` | `_record_current_trigger_sample` (`solver.py`) | CARRIED |
| `_current_trigger_samples` | `_record_current_trigger_sample` (`solver.py`) | CARRIED |
| `_t_ignition_abort` | `_open_ignition_switch` (`solver.py`) | CARRIED |
| `_ignition_abort_reason` | `_open_ignition_switch` (`solver.py`) | CARRIED |
| `_ignition_abort_context` | `_open_ignition_switch` (`solver.py`) | DROPPED — rebuilt diagnostic |
| `_ignition_abort_threshold_name` | `_prebreakdown_timeout_switch_open` (`solver.py`) | CARRIED |
| `_run_start_for_phase_events` | `run()` (`solver.py`) | CARRIED (diagnostic) |

The last-sample pair is not diagnostic: `_current_threshold_time`
(`solver.py`) interpolates the threshold crossing between it and the
current sample, and the resulting trigger time becomes a phase boundary that
**caps the timestep** (`cap_step` in `run()`, `solver.py`). A restart that reset it would
place breakdown at a different instant.

`_run_start_for_phase_events` feeds only the `phase_events` diagnostic list.
It is carried so a resumed run reports events from the original origin rather
than from the restart instant; it does not affect stepping.

`_ignition_abort_context` is the one member of this table that is DROPPED, and
deliberately so. It is absent from `_RESTART_TRIGGER_ATTRS` (`solver.py`)
and `_apply_restart_payload` then sets it to `None` outright
(`solver.py`), with the reason stated in the comment above it: the
context is a diagnostic record of a switch-open that has *already* happened,
while the abort reason, time and threshold name — all three carried — are what
the wind-down reads, and the context is rebuilt by the next guard evaluation.
Carrying it would restore a snapshot of guard internals that the next
evaluation overwrites anyway.

### Ignition stall monitor

| state | site | class |
|---|---|---|
| `IgnitionMonitor._samples` ring buffer | `IgnitionMonitor.record` (`ignition.py`) via `_ignition_diagnostic_snapshot` (`solver.py`) | CARRIED |
| `IgnitionMonitor._stalled` latch | `IgnitionMonitor.record` (`ignition.py`) | CARRIED |
| `_last_ignition_record` | `_ignition_diagnostic_snapshot` (`solver.py`) | DROPPED — rebuilt diagnostic |
| `window_s`, `rate_window_s`, `min_samples` | construction | DERIVABLE |

Not diagnostic: a trip calls `_open_ignition_switch` (`solver.py`), which
sets `_t_ignition_abort` and shortens `t_end`. The buffer is fed once per
trajectory save, so it is coupled to the save lattice the run-loop block below
preserves.

`_last_ignition_record` is the exception and is DROPPED, alongside
`_ignition_abort_context` and for the same reason: `_apply_restart_payload`
resets it to `None` (`solver.py`). It holds the most recent monitor
*report*, rebuilt in full at the next monitor evaluation (`solver.py`)
from the ring buffer and the RHS terms; its only readers assemble abort
artifacts (`solver.py`). The latch and the samples, which do decide
whether a trip fires, are carried above.

### Electrode sample EMA

| state | site | class |
|---|---|---|
| `_sample_ema` (`{cell: [n, Te]}`) | `_update_sample_smoothing` (`solver.py`) | CARRIED |
| `_sample_smoothing`, `_sample_smooth_cells` | construction | DERIVABLE |

Three consumers read the smoothed sample (`_smoothed_sample_state` in `solver.py` is the only
substitution site): the RHS/beam sheath solve, the accepted-state surface
re-solve at `_accept_step_attempt` (`solver.py`), and the surface coverage update. The EMA is
seeded from the initial state at construction, so an uncarried restart would
restart the average — a first-order error in every sheath solve that follows.

### Floor ledger

| state | site | class |
|---|---|---|
| `_floor_ledger` | `_accumulate_floor_ledger` (`solver.py`) | CARRIED |

Accumulator; appears in the result. No feedback into the state.

### Surface-loss floor-exempt latch

| state | site | class |
|---|---|---|
| `_surface_loss_floor_exempt_latch` | allocated in `__init__`, advanced by `suggest_timestep` via `plasma_source_timestep` | DROPPED |

Per-cell, per-energy-channel (`"Ee"`/`"Ei"`) memory of the `surface_loss`
floor-exempt verdict: which cells are currently inside the two-threshold
re-admission band and holding their previous verdict. It exists only when
`surface_loss_floor_exempt_exit_rtol` is nonzero, which is the shipped
default, and it is `None` on the knife-edge path.

It is DROPPED, and the consequence is load-bearing: **a resumed run starts
with every cell un-exempt**, so the first `suggest_timestep` after a restart
re-derives each verdict from the inner entry threshold alone. A cell that
stage 1 was holding exempt inside the band is therefore re-admitted to the
drain bound on the resumed run's first evaluation, which collapses that step's
`surface_loss` candidate and, through the ramp, every step after it.
**Continuation bit-identity is therefore NOT guaranteed at the shipped
defaults.** It survives only a handoff at which no cell is sitting inside the
band on a held exemption — true of any window before the floors are reached,
and false in general once a floor-pinned afterglow is running. It is
unconditional only on an arm that sets `surface_loss_floor_exempt_exit_rtol`
to `0.0` (or turns `surface_loss_floor_exempt` off), where no latch exists and
the exemption test is recomputed from the current margin every call anyway.

Carrying it would be a structural change to the payload rather than one more
scalar: the latch is per-cell and per-channel, so it would have to be
compatibility-refused like the packed state fields, and the band's whole
purpose is to damp float residue rather than to define the trajectory. The
honest disclosure is this row.

Measured, so the disclosure is not merely theoretical in the other direction
either: at the armed defaults the `meanfield` and `meanfield_beam` acceptance
scenarios both still hand off raw-byte identical (2026-08-26), the latter with
`surface_loss` binding 2,169 of its 2,190 steps. Neither window reaches a
floor-pinned afterglow, which is the regime the paragraph above is about.

### Picard counters

| state | site | class |
|---|---|---|
| `_picard_extra_solves` | `_accept_step_with_picard` (`solver.py`) | DROPPED |
| `_picard_triggered_steps` | `_accept_step_with_picard` (`solver.py`) | DROPPED |

Counters only, read nowhere in the solver, and — decisively — they exist only
when `coupled_circuit_picard` is on (`solver.py` sits inside that
branch), so carrying them unconditionally would make the payload's own shape
depend on a non-structural flag. Each stage counts its own.

### Run-loop controller state (LOCAL to `run()`, not instance attributes)

This is the block a naive restart misses entirely, because none of it is an
attribute to grep for.

| local | site | class |
|---|---|---|
| `previous_accepted_dt` | `run()` (`solver.py`), set and read inside the step loop | CARRIED |
| `t_last_save` | `run()` (`solver.py`), set and read inside the step loop | CARRIED |
| `dt_growth_capped_streak` | `run()` (`solver.py`), set inside the step loop | CARRIED |
| `consecutive_dt_min_clamps` | `run()` (`solver.py`), set inside the step loop | CARRIED |
| `len(saved)` | `run()` (`solver.py`) | CARRIED (as a frame-count offset) |
| `steps` | `run()` (`solver.py`) | CARRIED (as an offset for the accepted-step guard) |
| `run_start` | `run()` (`solver.py`) | CARRIED (via `_run_start_for_phase_events`) |
| `ignition_wall_clock_start` | `run()` (`solver.py`) | DROPPED — see below |
| `progress_wall_start`, `last_progress_time`, `force_progress` | `run()` (`solver.py`) | DROPPED — progress reporting only |
| `saved`, `diagnostics`, `timestep_rejection_events` | `run()` (`solver.py`) | DROPPED — each stage owns its own trajectory |

`previous_accepted_dt` is the dt-growth ramp's anchor: without it the first
step after a restart is not growth-capped and takes whatever the physics bound
allows, which changes that step's dt and every subsequent one.
`dt_growth_capped_streak` is the recovery hysteresis (the streak-update block in
`run()`, `solver.py`), asymmetric by design and therefore not
reconstructible from the state.

`ignition_wall_clock_start` is DROPPED because wall clock is a property of the
process, not of the trajectory. A two-stage run genuinely gets two wall-clock
budgets, and there is no honest alternative. The accepted-**step** cap is
carried as an offset, so the work-done budget does transfer. This is the one
place where a restarted run can differ from an unsplit one, it is disclosed
here, and it is inert at the shipped defaults (`ignition_wall_clock_cap_s` is
`0.0`, i.e. off).

### The leading save

`run()` opens with a leading `if should_save(self._time)` (`solver.py`, guarded by `resume is None`).
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
| `_beam_gap_ledger_warned` | `_warn_beam_gap_ledger` (`solver.py`), reset in `run()` (`solver.py`) | DROPPED |
| `_dvm_ion_shortfall_warned` | `_dvm_advance` (`solver.py`) | DROPPED (arm is REFUSED) |

`_beam_gap_ledger_warned` is reset by `run()` itself at entry, so carrying it
would be overwritten anyway. The effect of dropping it is that a warning
already emitted before the handoff may be emitted once more after it. No
trajectory consequence.

### Result handles

`_last_result`, `_last_neutral_equilibration_result`,
`_last_neutral_equilibration_summary`, `_run_via_start_simulation`
(`solver.py`): DROPPED. They hold the previous call's
return value and feed nothing in the stepping path.

### Refused subsystems

| subsystem | gate | why refused |
|---|---|---|
| K4a kinetic response functions (`_kinetic`) | `neutral_model == "kinetic"` | `solver.py`: frozen velocity grid, per-channel response functions, refresh clocks. Not serialised. |
| K2a transient DVM (`_dvm`, `_dvm_*`) | `neutral_model == "kinetic_dvm"` | `solver.py`: a full distribution function plus ion-debt ledger and a neutral clock (`solver.py`). Not serialised. |
| neutral equilibration | `neutral_equilibration` flag | `start_simulation()` would run the puff/off accumulation and **overwrite** the restored IC (`start_simulation`, `solver.py`). A restart *is* the seed. |

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
the handoff instant is one the unsplit run also steps to exactly, and — at the
shipped defaults — when no cell is holding an exemption inside the
`surface_loss` re-admission band at that instant (the DROPPED
`_surface_loss_floor_exempt_latch` row above states the consequence in full).

That is not a limitation in practice: every save instant IS a step boundary,
because `next_save_time_after` caps the adaptive step to land on it
(`cap_step` in `run()`, `solver.py`), and an export is naturally taken at the end of a
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

* `_cathode_x0`, whose consequence the d1a probe measured at exactly zero over
  a 3300x seed span. Carried regardless: the fixed point is a property of the
  current stance, not a guarantee.
* `_cathode_beam_cross`, which is identically zero until the sheath potential
  crosses the ionization threshold — roughly `2e-4 s` into the production
  stance. An acceptance window shorter than that cannot test it, which is why
  both harness scenarios hand off in the beam-live regime instead.

`dt_growth_capped_streak` was on this list while `dt_growth_recovery_patience`
defaulted to `0` and presence-gated the whole branch in `run()`. The shipped
patience is nonzero, so the widened-ceiling read and the streak update both run
on a default-configured acceptance run and the carry is now exercised without
the harness having to raise the key itself.

## Compatibility refusal

The payload records the producing run's full resolved `params`/`flags`, the
cell count, and the packed state-field layout. On load the following must match
the constructed solver exactly, or the load raises:

* cell count and packed state-vector length,
* the packed state field names (which optional rows exist),
* the **structural** config keys — the ones that decide what the payload's
  members mean rather than merely how big a number is:
  flags `coverage_closure`, `neutral_momentum`, `neutral_two_zone`,
  `neutral_energy`, `TwinCathode`, `Plasma`, `cathode_coupling`,
  `cathode_emitting_area`, `regime_vessel_node`;
  params `neutral_model`,
  `cathode_warming_model`, `cathode_surface_model`,
  `cathode_sample_smoothing`, `phase_transition_mode`.

Every other config key is free to differ — that is what makes a two-stage
hybrid possible — and the producing run's full config is retained in the
payload so any difference is auditable after the fact.
