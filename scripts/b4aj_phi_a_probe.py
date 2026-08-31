"""[B4] Measure phi_a + Ti at the anode cells over the arm's opening window.

The measurement behind the branch's ONE FAILED pre-registered gate. The B4
registration pins the incident per-ion energy at the fluid channel's own clamp,
``max(phi_a + Ti, 0)`` -- and the fluid anode spec
(``solver._anode_jet_spec``) passes ``phi_a`` through RAW, where the fluid
cathode spec clamps ``phi_c`` at zero first. That asymmetry is faithfully
mirrored on both sides of this member, and on the anode side it makes the
clamped sum reachably ZERO: before breakdown the anode sheath is
electron-attracting.

This prints, per accepted step over the arm's own 800-step window, the anode
sheath potential, the ion temperature at each flanking cell of the mesh, their
clamped sum, and the counted collection rate -- so "the incident energy is
exactly zero while the mesh is still collecting" is a measurement rather than
an inference from a traceback.

It touches nothing this branch changes and reads no B4 configuration key: it
runs the arm with the channel OFF and reads the same two quantities the
channel would have been handed.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/b4aj_phi_a_probe.py [--steps 800]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import cablp  # noqa: E402
from cablp.cathode.kernels import KERNEL_ID  # noqa: E402
from cablp.solvers._sim1d import LAPDSim1D  # noqa: E402
from cablp.solvers._sim1d.core.state import derive_state  # noqa: E402
from cablp.solvers._sim1d.solver import LAPDSim1D as _Solver  # noqa: E402

from b4aj_bitinert_ab import KINETIC_EXTRA, WINDOW_STEPS  # noqa: E402
from b5cj_bitinert_ab import build_arm, step_once  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=WINDOW_STEPS)
    parser.add_argument("--every", type=int, default=25)
    args = parser.parse_args(argv)

    print("[B4] anode sheath probe over the arm's opening window")
    print("=" * 78)
    print(f"cablp.__file__ = {cablp.__file__}")
    print(f"KERNEL_ID      = {KERNEL_ID}")
    print(f"cadence        = {KINETIC_EXTRA['neutral_kinetic_dvm_cadence_s']} s")
    print("=" * 78)

    params, flags = build_arm("kinetic_dvm", dict(KINETIC_EXTRA))
    sim = LAPDSim1D(input_dict=params, input_flags=flags)

    original = _Solver._dvm_source_channel_rows

    def spy(self, terms):
        rows = original(self, terms)
        self._b4aj_anode_row = np.asarray(rows["anode"], dtype=float).copy()
        return rows

    _Solver._dvm_source_channel_rows = spy
    first_zero = None
    last_zero = None
    try:
        print(f"{'step':>5} {'t [s]':>13} {'phi_a [V]':>11}  per anode cell: "
              "Ti [eV], max(phi_a+Ti,0) [eV], counted rate [1/s]")
        for step in range(1, args.steps + 1):
            step_once(sim)
            report = step <= 12 or step % args.every == 0
            row = getattr(sim, "_b4aj_anode_row", None)
            if row is None:
                continue
            state = sim.state
            solve = sim.solve_cathode_boundary(
                state=state, time=sim.time, update_cache=False
            )
            phi_a = float("nan")
            if solve is not None and solve.beam_result is not None:
                phi_a = float(solve.beam_result.result.phi_a)
            Ti = derive_state(
                state, floors=sim._floors, ion_mass_g=sim._ion_mass_g
            ).Ti
            cells = np.flatnonzero(row)
            clamped = [
                max(phi_a + float(Ti[c]), 0.0) for c in cells
            ]
            if cells.size and max(clamped) == 0.0:
                first_zero = step if first_zero is None else first_zero
                last_zero = step
            if report:
                text = "  ".join(
                    f"c{int(c)}: Ti={float(Ti[c]):.4g} "
                    f"inc={clamped[i]:.4g} N={float(row[c]):.3e}"
                    for i, c in enumerate(cells)
                )
                print(f"{step:5d} {sim.time:13.6e} {phi_a:11.5g}  {text}")
    finally:
        _Solver._dvm_source_channel_rows = original

    print("=" * 78)
    if first_zero is None:
        print("no step in this window carried a zero clamped incident energy")
    else:
        print(
            "ZERO CLAMPED INCIDENT ENERGY on a NON-EMPTY counted anode stream: "
            f"accepted steps {first_zero} through {last_zero}. A neutral tick "
            "falling entirely inside that span hands the anode jet a counted "
            "stream with no committed energy, and the launch spectrum raises."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
