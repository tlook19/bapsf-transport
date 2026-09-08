"""Measure how often sim1d's numerical floors actually bind, and what they inject.

The saved trajectory cannot answer this. ``derive_state`` floors Te/Ti on every
read (``core/state.py``), so ``result.Te`` is already floored by the time it
reaches HDF5: a post-hoc pass can see that Te *equals* the floor but not whether
it was clipped up to it, nor how much energy that clip injected. Both floors are
therefore instrumented at runtime here.

Two distinct clip sites are tracked:

``state``
    ``core/state.apply_state_floors`` -- applied at every SSPRK2 stage and after
    each operator-split step. Injected energy is measured exactly, as the change
    in the conservative Ee/Ei the call actually made.

``conduction``
    ``physics/conduction._implicit_species_energy`` -- the ``np.maximum(
    temperature, temperature_floor)`` after the implicit tridiagonal solve. This
    is the clip that would launder Crank-Nicolson ringing into an energy source,
    so it is the one that decides whether a non-backward-Euler theta is safe.

**This audit names the configuration it measures.** How often a floor binds is
a property OF a configuration, so there is no bare mode: the command line
carries ``--stance <name>`` or an explicit ``--no-stance``, and the header
prints the name, the base chain and the resolved ``config_identity`` so a
reading can be compared with another one. ``default_config()`` is the template
of keys, never an implied plasma.

Usage::

    # the golden gate's own configuration (the reference stance at nx = 60)
    python scripts/gates/audit_sim1d_floor_activation.py --stance g1atrim \
        --golden-route
    # the named configuration at ITS OWN mesh, short shakedown
    python scripts/gates/audit_sim1d_floor_activation.py --stance g1atrim \
        --t-end 2e-3
    # no configuration: default_config() alone, recorded as unnamed
    python scripts/gates/audit_sim1d_floor_activation.py --no-stance
"""

import argparse
import sys

import numpy as np

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

import cablp.solvers._sim1d.core.state as state_mod  # noqa: E402
import cablp.solvers._sim1d.physics.conduction as conduction_mod  # noqa: E402
from cablp.solvers._sim1d import (  # noqa: E402
    LAPDSim1D,
    ProgressPrinter1D,
    config_identity,
    default_config,
)
from cablp.solvers._sim1d.physics.conduction import (  # noqa: E402
    IMPLICIT_HEAT_SCHEMES,
)
from cablp.constants import ev_to_erg  # noqa: E402

# The golden gate's own layering, imported rather than restated: the reference
# configuration minus its mesh-sized package, plus the gate's run-shape pins.
# Re-deriving that treatment here would be a second copy of it, free to drift
# from the one the golden actually runs.
from baseline_sim1d import (  # noqa: E402
    PRODUCTION_STANCE,
    baseline_lineage,
    build_baseline_config,
)
from stance_config import available_stances, load_configuration  # noqa: E402


# A value sitting exactly on its floor round-trips through
# conservative<->primitive conversion to within a few ULP, so a strict ``<``
# comparison fires on floating-point dust and reports a clip that injects no
# energy. Only deficits deeper than this relative tolerance are counted as
# material clips; shallower ones are counted as "resting on the floor", which is
# benign and physically expected wherever the solution is genuinely cold.
FLOOR_RTOL = 1e-9

# Injected energy below this fraction of the column thermal energy is reported
# as negligible. A clip count on its own decides nothing -- what matters is how
# much energy the clipping launders into the solution.
NEGLIGIBLE_ENERGY_FRACTION = 1e-6


class FloorRecorder:
    """Accumulate floor-activation counts and injected energy per clip site."""

    def __init__(self, cells, scheme="backward_euler"):
        self.cells = cells
        self.scheme = scheme
        self.time_getter = lambda: np.nan
        self.sites = {}
        # implicit_heat_conduction_step calls _implicit_species_energy for the
        # electrons first, then the ions. Te_floor and Ti_floor are equal under
        # the default config, so the floor value cannot identify the species --
        # call parity is what distinguishes them.
        self.cond_parity = 0

    def _site(self, name):
        if name not in self.sites:
            self.sites[name] = {
                "calls": 0,
                "fields": {},
                "energy_erg": {},
                "cell_hits": {},
                "first_time": np.nan,
                "last_time": np.nan,
            }
        return self.sites[name]

    def record(self, site_name, field, mask, energy_erg=0.0, resting=None):
        site = self._site(site_name)
        hits = int(np.count_nonzero(mask))
        site["fields"][field] = site["fields"].get(field, 0) + hits
        site["energy_erg"][field] = site["energy_erg"].get(field, 0.0) + float(
            energy_erg
        )
        if resting is not None:
            site.setdefault("resting", {})
            site["resting"][field] = site["resting"].get(field, 0) + int(
                np.count_nonzero(resting)
            )
        if hits:
            cell_hits = site["cell_hits"].setdefault(
                field, np.zeros(self.cells, dtype=np.int64)
            )
            cell_hits += np.asarray(mask, dtype=np.int64)
            t = float(self.time_getter())
            if not np.isfinite(site["first_time"]):
                site["first_time"] = t
            site["last_time"] = t

    def bump_calls(self, site_name):
        self._site(site_name)["calls"] += 1


def install_probes(recorder):
    """Monkeypatch the two clip sites; return a restore callable."""
    orig_apply = state_mod.apply_state_floors
    orig_species = conduction_mod._implicit_species_energy

    def probed_apply_state_floors(state, floors, ion_mass_g):
        recorder.bump_calls("state")
        n_safe = np.maximum(np.asarray(state.n, dtype=float), floors["n"])
        raw_Te = (2.0 / 3.0) * np.asarray(state.Ee, dtype=float) / (n_safe * ev_to_erg)
        raw_Ti = (2.0 / 3.0) * np.asarray(state.Ei, dtype=float) / (n_safe * ev_to_erg)

        out = orig_apply(state, floors=floors, ion_mass_g=ion_mass_g)

        # Exact injected energy: what the call actually did to conservative Ee/Ei.
        dEe = np.asarray(out.Ee, dtype=float) - np.asarray(state.Ee, dtype=float)
        dEi = np.asarray(out.Ei, dtype=float) - np.asarray(state.Ei, dtype=float)
        te_lo, ti_lo = floors["Te"] * (1.0 - FLOOR_RTOL), floors["Ti"] * (
            1.0 - FLOOR_RTOL
        )
        recorder.record(
            "state",
            "Te",
            raw_Te < te_lo,
            np.sum(dEe),
            resting=(raw_Te >= te_lo) & (raw_Te <= floors["Te"] * (1.0 + FLOOR_RTOL)),
        )
        recorder.record(
            "state",
            "Ti",
            raw_Ti < ti_lo,
            np.sum(dEi),
            resting=(raw_Ti >= ti_lo) & (raw_Ti <= floors["Ti"] * (1.0 + FLOOR_RTOL)),
        )
        recorder.record(
            "state",
            "n",
            np.asarray(state.n, dtype=float) < floors["n"] * (1.0 - FLOOR_RTOL),
        )
        recorder.record(
            "state",
            "nn",
            np.asarray(state.nn, dtype=float) < floors["nn"] * (1.0 - FLOOR_RTOL),
        )
        return out

    def probed_implicit_species_energy(
        energy, capacity, temperature_floor, conductivity, geometry, dt, **kwargs
    ):
        # The pre-clip temperature is recovered by calling the real solve with
        # the floor pushed to -inf, so this probe never duplicates the library's
        # discretization and cannot drift from it. The only place the floor
        # enters that solve other than the final clip is the theta<1 explicit
        # half, as max(energy/capacity, floor); the state reaching this step is
        # always floored by the preceding SSPRK2 stage, so energy/capacity is
        # already >= floor and the -inf call computes an identical right-hand
        # side. **kwargs forwards theta (and anything added later) untouched.
        recorder.bump_calls("conduction")
        unclipped = orig_species(
            energy=energy,
            capacity=capacity,
            temperature_floor=-np.inf,
            conductivity=conductivity,
            geometry=geometry,
            dt=dt,
            **kwargs,
        )
        raw_T = np.asarray(unclipped, dtype=float) / capacity

        lo = temperature_floor * (1.0 - FLOOR_RTOL)
        clipped = raw_T < lo
        injected = (
            float(np.sum(capacity[clipped] * (temperature_floor - raw_T[clipped])))
            if np.any(clipped)
            else 0.0
        )
        field = "Te" if recorder.cond_parity % 2 == 0 else "Ti"
        recorder.cond_parity += 1
        recorder.record(
            "conduction",
            field,
            clipped,
            injected,
            resting=(raw_T >= lo)
            & (raw_T <= temperature_floor * (1.0 + FLOOR_RTOL)),
        )

        # Bit-passive by construction: capacity is positive, so multiplication
        # by it is order-preserving and this is elementwise identical to the
        # library's own capacity * np.maximum(T, floor) -- while dividing out
        # capacity and multiplying it back would round twice and inject ULP
        # perturbations into every conduction substep of an instrumented run.
        return np.maximum(unclipped, capacity * temperature_floor)

    state_mod.apply_state_floors = probed_apply_state_floors
    conduction_mod._implicit_species_energy = probed_implicit_species_energy

    def restore():
        state_mod.apply_state_floors = orig_apply
        conduction_mod._implicit_species_energy = orig_species

    return restore


def report(recorder, sim, result):
    z = np.asarray(sim._geometry.z_cm, dtype=float)
    thermal = float(
        np.sum(
            (np.asarray(result.Ee, dtype=float)[-1] + np.asarray(result.Ei, dtype=float)[-1])
            * np.asarray(sim._geometry.plasma_volume_cm3, dtype=float)
        )
    )
    print("\n" + "=" * 78)
    print("FLOOR ACTIVATION AUDIT")
    print("=" * 78)
    print(f"final thermal energy in column : {thermal:.4e} erg")

    for site_name in ("state", "conduction"):
        site = recorder.sites.get(site_name)
        print(f"\n--- site: {site_name} ---")
        if site is None or site["calls"] == 0:
            print("  never invoked")
            continue
        calls = site["calls"]
        print(f"  calls: {calls}")
        resting = site.get("resting", {})
        for field in sorted(resting):
            visits = calls * recorder.cells
            if field in ("Te", "Ti") and visits:
                pct = 100.0 * resting[field] / visits
                print(f"  {field}: resting on floor (no energy injected) {pct:6.2f}% of cell-visits")
        any_hit = False
        for field, hits in sorted(site["fields"].items()):
            if hits == 0:
                continue
            any_hit = True
            energy = site["energy_erg"].get(field, 0.0)
            cell_hits = site["cell_hits"].get(field)
            frac = 100.0 * hits / (calls * recorder.cells)
            print(f"  {field}: {hits} cell-clips ({frac:.3f}% of cell-visits)")
            print(
                f"      injected energy : {energy:+.4e} erg"
                f"  ({100.0 * energy / thermal:+.4f}% of final thermal)"
                if thermal
                else f"      injected energy : {energy:+.4e} erg"
            )
            print(
                f"      active window   : t = {site['first_time']:.4e} .. "
                f"{site['last_time']:.4e} s"
            )
            if cell_hits is not None:
                hot = np.nonzero(cell_hits)[0]
                print(
                    f"      cells clipped   : {hot.size}/{recorder.cells}"
                    f"  (z = {z[hot.min()]:.0f} .. {z[hot.max()]:.0f} cm)"
                )
                busiest = np.argsort(cell_hits)[::-1][:5]
                pretty = ", ".join(
                    f"cell {int(c)} (z={z[c]:.0f}cm): {int(cell_hits[c])}"
                    for c in busiest
                    if cell_hits[c] > 0
                )
                print(f"      busiest cells   : {pretty}")
        if not any_hit:
            print("  no field ever materially clipped")

    cond = recorder.sites.get("conduction")
    scheme = recorder.scheme
    print("\n" + "-" * 78)
    print(f"VERDICT  (implicit_heat_scheme = {scheme!r})")
    if cond is None or cond["calls"] == 0:
        print("  Conduction clip never ran (implicit heat path inactive?).")
    elif sum(cond["fields"].values()) == 0:
        if scheme == "backward_euler":
            # (C + dt*K) is an M-matrix with row sums equal to capacity, and
            # K*1 = 0, so T_new >= min(T_old) >= floor. A zero count here is a
            # theorem, not evidence about any other scheme.
            print("  No clips -- but backward Euler CANNOT clip: its discrete")
            print("  maximum principle guarantees T_new >= min(T_old) >= floor.")
            print("  This confirms the matrix assembly is correct, and says")
            print("  NOTHING about Crank-Nicolson. Re-run with --scheme")
            print("  crank_nicolson for a result that bears on the theta choice.")
        else:
            print(f"  Conduction floor never binds under {scheme!r}.")
            print("  Ringing (if any) stays above the floor: no energy laundered.")
    else:
        clips = sum(cond["fields"].values())
        injected = sum(cond["energy_erg"].values())
        frac = abs(injected) / thermal if thermal else float("inf")
        print(f"  Conduction floor binds under {scheme!r}: {clips} clips in")
        print(f"  {cond['calls']} solves, injecting {injected:+.2e} erg")
        print(f"  = {frac:.1e} of the column thermal energy.")
        # A clip count alone says nothing about whether it matters: what the
        # theta choice hinges on is how much energy the clipping launders.
        if frac < NEGLIGIBLE_ENERGY_FRACTION:
            print("  That is negligible. Ringing is being clipped, but the energy")
            print("  it injects is far too small to affect the solution.")
        else:
            print("  That is large enough to matter. Prefer a larger theta, or")
            print("  tr_bdf2, which is L-stable and does not sustain ringing.")
    print("-" * 78)


def build_audit_config(args):
    """Return ``(params, flags, lineage)`` for the configuration to audit.

    Three routes, one of which the caller has already been required to name:

    ``--stance NAME --golden-route``
        :func:`baseline_sim1d.build_baseline_config` verbatim -- the golden
        gate's own layering, which is the named configuration minus its
        mesh-sized package plus the gate's run-shape pins. This is what lets
        the audit measure the configuration the golden actually runs, and it
        is imported rather than restated so the two cannot diverge. ``--nx``
        layers a different mesh on that same treatment; without it the gate's
        own ``nx`` stands and the lineage identity is the golden's.
    ``--stance NAME``
        the named configuration resolved by the stance loader, at ITS OWN
        mesh -- the mesh-sized package travels with it.
    ``--no-stance``
        ``default_config()`` alone. The template of keys is not a plasma, so
        the lineage is ``None``: this run records itself as unnamed rather
        than borrowing a name it did not use.

    ``lineage`` is ``None`` only on the ``--no-stance`` route.
    """
    if args.golden_route:
        overrides = {"nx": int(args.nx)} if args.nx is not None else None
        params, flags = build_baseline_config(overrides)
        return params, flags, baseline_lineage(params, flags)
    if args.no_stance:
        params, flags = default_config()
        return params, flags, None
    params, flags, lineage = load_configuration(args.stance)
    return params, flags, lineage


def print_header(params, flags, lineage, scheme_override):
    """Print WHICH configuration this reading is about, before the run starts.

    A floor-activation count is only comparable against another count taken on
    the same configuration, so the identity is printed at the top rather than
    left to the command line to remember.

    The name and the base chain are facts about the FILE and come from the
    lineage; the identity is restated over the pair this process actually
    constructs, so a ``--scheme``, ``--resolved`` or ``--nx`` on the command
    line moves it. With none of those on the golden route it is the golden's
    own. ``lineage`` is ``None`` for a run that named no configuration; the
    identity is printed either way, because it is a fact about the resolved
    pair and not about a name.
    """
    identity = config_identity(params, flags)
    if lineage is None:
        name, chain = "(none named)", ""
    else:
        name = lineage.name
        chain = " <- ".join(lineage.base_chain)
    print("=" * 78)
    print("FLOOR ACTIVATION AUDIT -- configuration")
    print("=" * 78)
    print(f"configuration    : {name}")
    print(f"base chain       : {chain or '(none -- base configuration)'}")
    print(f"config_identity  : {identity}")
    print(f"nx               : {params['nx']}")
    print(f"neutral_model    : {params['neutral_model']!r}")
    print(
        f"implicit_heat_scheme : {params['implicit_heat_scheme']!r}"
        + (
            f"  (OVERRIDDEN on the command line: --scheme {scheme_override})"
            if scheme_override is not None
            else "  (the configuration's own)"
        )
    )
    print(f"operator_splitting   : {params['operator_splitting']!r}")
    print(f"resolved_boundaries  : {bool(flags['resolved_boundaries'])}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-end", type=float, default=None, help="final time [s]")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--scheme",
        default=None,
        choices=sorted(IMPLICIT_HEAT_SCHEMES),
        help="override implicit_heat_scheme for the conduction substep. "
             "Default: the configuration's own value, whatever it names. An "
             "explicit value is announced as an override in the header.",
    )
    parser.add_argument(
        "--golden-route",
        action="store_true",
        help="apply the golden gate's own mesh treatment to the named "
             "configuration (its mesh-sized package dropped whole, plus the "
             "gate's run-shape pins), so this audit measures the "
             f"configuration the golden runs. Requires --stance "
             f"{PRODUCTION_STANCE}.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=None,
        help="axial cell count, layered on the --golden-route treatment. "
             "Without --golden-route there is no mesh treatment to layer it "
             "on, so it is refused there rather than half-applied.",
    )
    parser.add_argument(
        "--resolved",
        action="store_true",
        help=(
            "enable the resolved_boundaries geometry. Use this to check that the "
            "plasma-dead plenum behind the cathode stays inert: its cells sit at "
            "the floor by construction, and floor clips there would mean the "
            "reflecting cathode face is leaking."
        ),
    )
    stance_group = parser.add_mutually_exclusive_group()
    stance_group.add_argument(
        "--stance", metavar="NAME_OR_PATH", default=None,
        help="configuration this audit measures: a committed configuration "
             "name in scripts/stances/, or the path of a configuration file "
             "(derived or not). Available: "
             + (", ".join(available_stances()) or "(none committed)"),
    )
    stance_group.add_argument(
        "--no-stance", action="store_true",
        help="acknowledge that this audit names no configuration and measures "
             "default_config() plus the overrides on this command line",
    )
    args = parser.parse_args(argv)

    # Every run entry point names its configuration: default_config() is the
    # template of keys and never an implied plasma, so a floor reading taken
    # against it must say so instead of being reported as "the" answer.
    if args.stance is None and not args.no_stance:
        parser.error(
            "name the configuration package. Pass --stance <name> to measure "
            "a committed configuration (available: "
            f"{', '.join(available_stances()) or '(none committed)'}), or "
            "--no-stance to acknowledge that this audit names none and "
            "measures default_config() plus the overrides on this command "
            "line. default_config() is the template of keys, not a plasma."
        )
    if args.golden_route and args.stance != PRODUCTION_STANCE:
        parser.error(
            "--golden-route is the golden gate's layering of "
            f"{PRODUCTION_STANCE!r} and is defined for no other "
            f"configuration. Pass --stance {PRODUCTION_STANCE} with it, or "
            "drop --golden-route to measure the configuration you named at "
            "its own mesh."
        )
    if args.nx is not None and not args.golden_route:
        parser.error(
            "--nx layers a mesh on the --golden-route treatment. Without that "
            "treatment the named configuration's mesh-sized package is still "
            "in place and a different nx would half-apply it, which the "
            "solver refuses; pass --golden-route, or drop --nx."
        )

    params, flags, lineage = build_audit_config(args)
    if args.resolved:
        flags["resolved_boundaries"] = True
    if args.scheme is not None:
        params["implicit_heat_scheme"] = args.scheme
    scheme = params["implicit_heat_scheme"]
    print_header(params, flags, lineage, args.scheme)

    sim = LAPDSim1D(params, flags)
    recorder = FloorRecorder(cells=sim._geometry.cells, scheme=scheme)
    recorder.time_getter = lambda: sim._time
    restore = install_probes(recorder)
    try:
        sim.start_simulation(
            t_end=args.t_end,
            max_steps=args.max_steps,
            progress_tracker=None if args.quiet else ProgressPrinter1D(),
        )
        result = sim.get_results()
    finally:
        restore()

    report(recorder, sim, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
