"""A2a gates: lag discipline, construction refusals, and the namespace probe.

G3  LAG DISCIPLINE. ``_cathode_tail_anode_I`` carries the anode's direct tail
    current from one ACCEPTED step to the next sheath solve. The rule the
    campaign holds every lagged member to is that a REJECTED attempt must not
    move it. Three statements, on an ARMED run whose lag is already non-zero:

      G3a  a read-only solve (``update_cache=False``) leaves it untouched --
           that is the mode every rejected attempt's internal solves run in
           when they are served, and the mode the solve memo is keyed for;
      G3b  a full step ATTEMPT followed by the step-cache restore leaves it at
           the value the attempt started from, even though the attempt's own
           SSPRK2 stages write the cache with ``update_cache=True``;
      G3c  the Picard snapshot/restore round trip leaves it likewise.

    Each is printed with the before/after values, not merely asserted.

G4  CONSTRUCTION REFUSALS at the solver boundary (the module's own copies are
    gated by scripts/a2a_cull_conservation.py): range, ``eta_E > R_e``, the
    rider without the cull, the rider under the energy-only walk, the cull
    without a walked tail, and the cull without the primary interception.

G5  NAMESPACE PROBE. ``input_dict`` and ``input_flags`` are separate
    namespaces; a key filed in the wrong one must raise at construction rather
    than going silently inert. Probed in BOTH directions for the new family.

Run from the worktree root with PYTHONPATH=<worktree>.
"""

import sys
from pathlib import Path

import numpy as np

import cablp
from cablp.solvers._sim1d import LAPDSim1D

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from baseline_sim1d import build_baseline_config  # noqa: E402

print(f"cablp.__file__ = {cablp.__file__}")
print()

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          f"{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


ARMED_FLAGS = {"beam_tail_anode_interception": True}


def armed_config(**param_extra):
    params, flags = build_baseline_config()
    params = dict(params)
    params["max_steps_action"] = "stop"
    params.update(param_extra)
    flags = dict(flags)
    flags.update(ARMED_FLAGS)
    return params, flags


def step_once(sim):
    split = sim._flags.get("implicit_heat_conduction", False)
    diag = sim.suggest_timestep(include_heat_conduction=not split)

    def generate():
        attempt, retries, reason, events = sim._attempt_step_with_retries(
            dt=diag.dt, operator_split=None, diag=diag,
        )
        return attempt, (retries, reason, events)

    return sim._accept_step_with_picard(generate)


print("=" * 74)
print("G3  LAG DISCIPLINE")
print("=" * 74)
params, flags = armed_config()
sim = LAPDSim1D(input_dict=params, input_flags=flags)
print(f"  cells = {sim.geometry.cells}; cull armed = "
      f"{sim._flags['beam_tail_anode_interception']}")

MAX_WARMUP = 4000
lag = 0.0
for k in range(1, MAX_WARMUP + 1):
    step_once(sim)
    lag = float(sim._cathode_tail_anode_I)
    if lag != 0.0:
        print(f"  lag became non-zero at accepted step {k}: {lag!r} A "
              f"(t = {sim.time:.6e} s)")
        break
else:
    print(f"  lag still 0.0 after {MAX_WARMUP} accepted steps -- the probe "
          f"cannot distinguish 'unmoved' from 'never set'")
    FAIL.append("lag warm-up")

L0 = float(sim._cathode_tail_anode_I)

# G3a -- a read-only solve.
_ = sim.solve_cathode_boundary(time=sim.time, update_cache=False)
L_ro = float(sim._cathode_tail_anode_I)
print(f"  G3a before {L0!r} -> after read-only solve {L_ro!r}")
check("G3a read-only solve leaves the lag unmoved", L_ro == L0)

# G3b -- a full ATTEMPT (never accepted), then the step-cache restore. The
# attempt's own stages write the cathode caches with update_cache=True, so the
# in-flight value is expected to move; what must hold is that the restore puts
# it back exactly.
# A SENTINEL is planted first, so the probe can tell "the value came back"
# apart from "nothing ever wrote it". ``_attempt_step`` returns a candidate
# WITHOUT committing caches: it takes the step cache's own snapshot and
# restores it, which is why the sentinel survives even though the attempt's
# internal stage solves run with ``update_cache=True`` and DO write the lag.
# Both halves are measured here rather than assumed -- the cache-writing
# solves are counted, and the negative control below shows the sentinel is
# clobbered the moment the member is dropped from the snapshot.
SENTINEL = -7654321.0
CACHE_WRITES = {"n": 0}
_real_solve = type(sim).solve_cathode_boundary


def _counted_solve(self, *a, **kw):
    if kw.get("update_cache", False):
        CACHE_WRITES["n"] += 1
    return _real_solve(self, *a, **kw)


type(sim).solve_cathode_boundary = _counted_solve
sim._cathode_tail_anode_I = SENTINEL
diag = sim.suggest_timestep(
    include_heat_conduction=not sim._flags.get(
        "implicit_heat_conduction", False
    )
)
attempt = sim._attempt_step(dt=diag.dt, operator_split=None)
L_after = float(sim._cathode_tail_anode_I)
writes = CACHE_WRITES["n"]
print(f"  G3b sentinel {SENTINEL!r} -> after a rejected attempt {L_after!r} "
      f"({writes} cache-writing solves ran inside it)")
check("G3b the attempt DID run cache-writing solves (else this is vacuous)",
      writes > 0, f"{writes} solves with update_cache=True")
check("G3b a rejected attempt leaves the lag exactly where it was",
      L_after == SENTINEL)

# NEGATIVE CONTROL for G3b: drop the member from the step-cache snapshot, as
# it would be if this build had wired the lag without touching that pair, and
# show the sentinel does NOT survive. Without this the pass above could be
# reporting a value nothing ever wrote.
_real_snap = type(sim)._step_cache_snapshot
_real_restore = type(sim)._restore_step_cache


def _snap_without_lag(self):
    snap = _real_snap(self)
    del snap.cathode_tail_anode_I
    return snap


def _restore_without_lag(self, snapshot):
    snapshot.cathode_tail_anode_I = float(self._cathode_tail_anode_I)
    return _real_restore(self, snapshot)


type(sim)._step_cache_snapshot = _snap_without_lag
type(sim)._restore_step_cache = _restore_without_lag
sim._cathode_tail_anode_I = SENTINEL
diag = sim.suggest_timestep(
    include_heat_conduction=not sim._flags.get(
        "implicit_heat_conduction", False
    )
)
sim._attempt_step(dt=diag.dt, operator_split=None)
L_control = float(sim._cathode_tail_anode_I)
type(sim)._step_cache_snapshot = _real_snap
type(sim)._restore_step_cache = _real_restore
type(sim).solve_cathode_boundary = _real_solve
print(f"  G3b negative control (member dropped from the snapshot): sentinel "
      f"{SENTINEL!r} -> {L_control!r}")
check("G3b negative control: without the snapshot entry the attempt DOES "
      "leak into the lag", L_control != SENTINEL)
sim._cathode_tail_anode_I = L0

# G3c -- the Picard snapshot/restore round trip.
psnap = sim._picard_snapshot()
sim._cathode_tail_anode_I = L0 + 12345.0
sim._picard_restore(psnap)
L_picard = float(sim._cathode_tail_anode_I)
print(f"  G3c before {L0!r} -> perturbed {L0 + 12345.0!r} -> after Picard "
      f"restore {L_picard!r}")
check("G3c the Picard restore returns the lag exactly", L_picard == L0)

# The restart payload round trip, for the same member.
payload = sim.restart_payload()
carried = float(payload["cathode"]["_cathode_tail_anode_I"])
print(f"  restart payload carries {carried!r}")
check("the restart payload carries the lag", carried == L0)
legacy = {**payload, "cathode": {
    k: v for k, v in payload["cathode"].items()
    if k != "_cathode_tail_anode_I"
}}
sim._apply_restart_payload(legacy)
print(f"  a payload written BEFORE the cull existed restores it to "
      f"{float(sim._cathode_tail_anode_I)!r}")
check("a pre-A2a restart payload still loads, at 0.0",
      float(sim._cathode_tail_anode_I) == 0.0)

print()
print("=" * 74)
print("G4  CONSTRUCTION REFUSALS at the solver boundary")
print("=" * 74)


def refuses(label, fragment, params, flags):
    try:
        LAPDSim1D(input_dict=params, input_flags=flags)
    except ValueError as exc:
        ok = fragment in str(exc)
        check(label, ok,
              "" if ok else f"message did not name {fragment!r}: {exc}")
        return
    check(label, False, "no ValueError raised")


p, f = armed_config()
refuses("R_e outside [0, 1]", "must be in [0, 1]",
        {**p, "beam_tail_anode_reflected_particles": 1.5}, f)
refuses("eta_E outside [0, 1]", "must be in [0, 1]",
        {**p, "beam_tail_anode_reflected_particles": 0.5,
         "beam_tail_anode_reflected_energy": 1.2}, f)
refuses("eta_E > R_e", "must not exceed",
        {**p, "beam_tail_anode_reflected_particles": 0.10,
         "beam_tail_anode_reflected_energy": 0.26}, f)
p_off, f_off = build_baseline_config()
refuses("rider without the cull armed", "ride on the anode tail cull",
        {**dict(p_off), "beam_tail_anode_reflected_particles": 0.37,
         "beam_tail_anode_reflected_energy": 0.26}, dict(f_off))
refuses("rider under the energy-only walk", "requires "
        "heating_anomalous_tail_ionization='on'",
        {**p, "heating_anomalous_tail_ionization": "off",
         "beam_tail_anode_reflected_particles": 0.37,
         "beam_tail_anode_reflected_energy": 0.26}, f)
refuses("cull with no walked tail", "needs a walked tail",
        {**p, "heating_anomalous_transport": "local",
         "heating_anomalous_tail_ionization": "off"}, f)
refuses("cull without the primary interception",
        "requires beam_anode_interception",
        p, {**f, "beam_anode_interception": False})

print()
print("=" * 74)
print("G5  NAMESPACE PROBE -- both directions")
print("=" * 74)
UNKNOWN = "unknown LAPDSim1D configuration keys"
refuses("a FLAGS key filed in input_dict", UNKNOWN,
        {**dict(p_off), "beam_tail_anode_interception": True}, dict(f_off))
refuses("a PARAMS key filed in input_flags", UNKNOWN,
        dict(p_off),
        {**dict(f_off), "beam_tail_anode_reflected_particles": 0.37})
refuses("the other PARAMS key filed in input_flags", UNKNOWN,
        dict(p_off),
        {**dict(f_off), "beam_tail_anode_reflected_energy": 0.26})

print()
print("=" * 74)
print("ARMED RUN SANITY -- the cull and the lag are actually live")
print("=" * 74)
p2, f2 = armed_config()
p2["beam_tail_anode_reflected_particles"] = 0.37
p2["beam_tail_anode_reflected_energy"] = 0.26
sim2 = LAPDSim1D(input_dict=p2, input_flags=f2)
for _ in range(400):
    step_once(sim2)
dep = sim2._cathode_solve.beam_deposition
tot_c = tot_r = 0.0
for d in (dep or {}).values():
    if d is None:
        continue
    tot_c += float(d.tail_anode_culled_erg_s)
    tot_r += float(d.tail_anode_returned_erg_s)
print(f"  box-top arm after 400 steps: culled {tot_c:.6e} erg/s, "
      f"returned {tot_r:.6e} erg/s")
print(f"  lag current {float(sim2._cathode_tail_anode_I):.6e} A")
check("the box-top arm culls and returns", tot_c > 0.0 and tot_r > 0.0)
check("the lag is non-zero on the armed run",
      float(sim2._cathode_tail_anode_I) != 0.0)

print()
if FAIL:
    print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
    sys.exit(1)
print("ALL A2a LAG AND REFUSAL GATES PASS")
