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

Usage:
    python scripts/audit_sim1d_floor_activation.py                 # notebook production config
    python scripts/audit_sim1d_floor_activation.py --t-end 2e-3    # short shakedown
"""

import argparse
import sys

import numpy as np

import cablp.solvers._sim1d.core.state as state_mod
import cablp.solvers._sim1d.physics.conduction as conduction_mod
from cablp.solvers._sim1d import LAPDSim1D, ProgressPrinter1D, default_config
from cablp.solvers._sim1d.physics.conduction import IMPLICIT_HEAT_SCHEME_THETA
from cablp.vars._cons import ev_to_erg

# Overrides mirroring scripts/sim1d_run_and_plot.ipynb (the production run).
PARAM_OVERRIDES = {
    "V_bank": 180.0,
    "T_s": 273.15 + 1700,
    "S_gp": 4000,
    "S_gp_decay_target": 1500,
    "tau_gp_pulse_duration": 1e-3,
    "tau_gp_decay_duration": 5e-3,
    "b_ion_neutral_drag": 0.5,
    "b_Qei": 0.5,
    "b_Qen": 0.5,
    "b_Qcx": 0.5,
    "Rp": 15.0,
    "R_cath": 15.0,
    "R_comp": 0.010,
}
FLAG_OVERRIDES = {
    # The notebook lists ion_neutral_drag_cx_only under its parameter overrides,
    # but it is an input_flags key, so it lands in input_dict and is ignored.
    # Keeping the flag at its template default preserves the notebook's actual
    # behaviour rather than the behaviour its parameter block implies.
}


# A value sitting exactly on its floor round-trips through
# conservative<->primitive conversion to within a few ULP, so a strict ``<``
# comparison fires on floating-point dust and reports a clip that injects no
# energy. Only deficits deeper than this relative tolerance are counted as
# material clips; shallower ones are counted as "resting on the floor", which is
# benign and physically expected wherever the solution is genuinely cold.
FLOOR_RTOL = 1e-9


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

        return capacity * np.maximum(raw_T, temperature_floor)

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
        print(f"  Conduction floor DOES bind under {scheme!r}: ringing is being")
        print("  clipped into a spurious energy source. Prefer a larger theta.")
    print("-" * 78)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-end", type=float, default=None, help="final time [s]")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--scheme",
        default="backward_euler",
        choices=sorted(IMPLICIT_HEAT_SCHEME_THETA),
        help="implicit_heat_scheme for the conduction substep",
    )
    args = parser.parse_args(argv)

    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    params["implicit_heat_scheme"] = args.scheme

    sim = LAPDSim1D(params, flags)
    recorder = FloorRecorder(cells=sim._geometry.cells, scheme=args.scheme)
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
