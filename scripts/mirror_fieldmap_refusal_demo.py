#!/usr/bin/env python3
"""Print every refusal the mirror field-map loader raises, verbatim.

The loader's contract is that it either resolves a real solved field onto a
real mesh, or it fails LOUDLY naming what was wrong. It never silently
extrapolates, never guesses an end-coil case, and never quietly hands back a
field it did not read. This enumerates the refusals so the messages themselves
can be reviewed as the documentation they are, and so a later change that
softens one is visible as a diff.

Every case here RAISES by design. Exit 0 means every case raised the expected
type; a case that returns a profile is the failure.

There is no ``flux_tube_mirror`` flag and no ``mirror_field_*`` configuration
key to demo, deliberately: the fluid mirror force is already in the model as
the quasi-1D ``p dA/dz`` source (``physics.sources.flux_tube_geometry_rhs``,
armed by ``prescribed_area_geometry``), so a flag gating a ``-mu grad B``
sibling would gate a term that must never be built -- an inert control by
construction. The last two demos assert that absence rather than a refusal
message.

Run from ``<checkout>/cablp`` with ``PYTHONPATH`` set to that directory. The
field map defaults to ``scripts/lapd_end_field_Rp18p415.npz``, which is an
ignored run artifact; pass ``--map`` to point elsewhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

from cablp.solvers._sim1d import config_manifest, default_config  # noqa: E402
from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402
from cablp.solvers._sim1d.physics import mirror_field as mirror_field_mod  # noqa: E402
from cablp.solvers._sim1d.physics.mirror_field import (  # noqa: E402
    load_mirror_field,
)

DEFAULT_MAP = _HERE / "lapd_end_field_Rp18p415.npz"

WITHDRAWN_KEYS = (
    "flux_tube_mirror",
    "mirror_field_map_path",
    "mirror_field_case",
    "mirror_field_interior_fill_gauss",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", default=str(DEFAULT_MAP))
    parser.add_argument("--output", default=None, help="also write the report here")
    args = parser.parse_args(argv)
    map_path = args.map

    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    failures = []

    params, flags = default_config()
    geometry = build_geometry(params, flags)
    long_params = dict(params)
    long_params["Lm"] = 2400.0
    long_geometry = build_geometry(long_params, flags)

    def show(label, expected, **kwargs):
        emit("=" * 72)
        emit(f"DEMO: {label}")
        try:
            load_mirror_field(**kwargs)
        except expected as exc:
            emit(f"{type(exc).__name__}: {exc}")
        except BaseException as exc:  # noqa: BLE001 -- reporting, not handling
            emit(f"UNEXPECTED {type(exc).__name__}: {exc}")
            failures.append(label)
        else:
            emit("NO RAISE -- this is the failure this script exists to catch")
            failures.append(label)
        emit()

    emit("mirror field-map loader: every refusal, verbatim")
    emit(f"field map: {map_path}")
    emit(f"mesh:      default_config(), {geometry.cells} cells, "
         f"z in [{geometry.z_edges_cm[0]:.2f}, {geometry.z_edges_cm[-1]:.2f}] cm")
    emit()

    show(
        "A. an unknown end-coil case",
        ValueError,
        map_path=map_path, case="droopmin", geometry=geometry,
    )
    show(
        "B. an undeclared end-coil case (case=None)",
        ValueError,
        map_path=map_path, case=None, geometry=geometry,
    )
    show(
        "C. the end-coil case omitted entirely (no default reading exists)",
        TypeError,
        map_path=map_path, geometry=geometry,
    )
    show(
        "D. the map path omitted entirely (the solver models no field)",
        TypeError,
        case="droop_min", geometry=geometry,
    )
    show(
        "E. a map path that names nothing readable",
        ValueError,
        map_path="/nonexistent/field.npz", case="off", geometry=geometry,
    )
    show(
        "F. a non-positive interior fill",
        ValueError,
        map_path=map_path, case="off", geometry=geometry,
        interior_fill_gauss=-1.0,
    )
    show(
        "G. a non-finite interior fill",
        ValueError,
        map_path=map_path, case="off", geometry=geometry,
        interior_fill_gauss=float("nan"),
    )
    show(
        "H. a map anchored on a different column radius than the caller's Rp",
        ValueError,
        map_path=map_path, case="droop_min", geometry=geometry,
        plasma_radius_cm=15.0,
    )
    show(
        "I. a mesh that runs past the map's high edge",
        ValueError,
        map_path=map_path, case="droop_min", geometry=long_geometry,
    )

    emit("=" * 72)
    emit("DEMO: J. no mirror force term exists to be called")
    for name in ("mirror_force_rhs", "MIRROR_FORCE_PENDING"):
        present = hasattr(mirror_field_mod, name)
        emit(f"  physics.mirror_field.{name}: "
             f"{'PRESENT -- FAIL' if present else 'absent'}")
        if present:
            failures.append(f"J:{name}")
    emit("  The fluid mirror force is sources.flux_tube_geometry_rhs, armed by")
    emit("  prescribed_area_geometry. At A proportional to 1/B it IS the")
    emit("  isotropic average of -mu grad_par B, so a sibling here would")
    emit("  double-count it by 100 %, exactly.")
    emit()

    emit("=" * 72)
    emit("DEMO: K. no mirror configuration key exists to be misfiled")
    manifest = config_manifest()
    for key in WITHDRAWN_KEYS:
        in_params = key in manifest["parameters"]
        in_flags = key in manifest["flags"]
        emit(f"  {key:38s} parameters={str(in_params):5s} "
             f"flags={str(in_flags):5s}")
        if in_params or in_flags:
            failures.append(f"K:{key}")
    emit("  A flag gating a term that must never be built can never do")
    emit("  anything, and a silent inert control is forbidden -- so the loader")
    emit("  is a library function driven by its own arguments instead.")
    emit()

    emit("=" * 72)
    if failures:
        emit(f"RESULT: FAILED -- {len(failures)} case(s) did not refuse: "
             f"{failures}")
    else:
        emit("RESULT: every case refused loudly at the call site, and no "
             "mirror force")
        emit("term or configuration key exists to be armed.")

    if args.output:
        Path(args.output).write_text("\n".join(lines) + "\n")
        print(f"\nreport written to {Path(args.output).resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
