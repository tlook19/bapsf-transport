"""fab -- NO-SOLVE geometry characterisation of the antenna-baffle surrogate.

DISCLOSURE: the baffle is a deliberately crude AZIMUTHALLY-SYMMETRIC surrogate
for an ASYMMETRIC antenna array around LAPD port 27. A real antenna array
blocks part of the azimuth over a couple of ports; this instrument (and the
run it characterises) replaces that by a full annular iris. The surrogate can
only bound the axial-choke effect; it cannot represent azimuthal bypass, which
in reality lets gas around the obstruction.

Read-only. Builds the geometry the driver would build, with and without the
baffle set, and reports exactly what the baffle changes: realized mesh faces,
the annulus open area, and the two-zone annulus Knudsen conductance. Nothing
is solved and nothing is written to the campaign state.
"""

import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
for _entry in (str(_SCRIPTS.parent), str(_SCRIPTS)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402
from cablp.solvers._sim1d.physics.neutrals import (  # noqa: E402
    two_zone_knudsen_coefficients,
)

import h5py  # noqa: E402

REF = _SCRIPTS / "fa4j_arm.h5"
POSITIONS = [941.25, 981.25, 1021.25]
RADII = [30.0, 30.0, 30.0]


def main():
    with h5py.File(REF, "r") as f:
        params = json.loads(f.attrs["params_json"])
        flags = json.loads(f.attrs["flags_json"])

    base_geo = build_geometry(dict(params), dict(flags))

    arm_params = dict(params)
    arm_params["neutral_baffle_positions_cm"] = list(POSITIONS)
    arm_params["neutral_baffle_clear_radii_cm"] = list(RADII)
    arm_flags = dict(flags)
    arm_flags["neutral_baffles"] = True
    arm_geo = build_geometry(arm_params, arm_flags)

    print("=== CONSTRUCTION ===")
    print("  baffle geometry CONSTRUCTED without error")
    faces = np.asarray(arm_geo.neutral_baffle_face_indices, dtype=int)
    radii = np.asarray(arm_geo.neutral_baffle_clear_radius_cm, dtype=float)
    print(f"  realized faces: {faces.tolist()}")
    print(f"  realized clear radii [cm]: {radii.tolist()}")

    # Reconstruct face edges from cell centres and lengths.
    z = np.asarray(base_geo.z_cm, dtype=float)
    L = np.asarray(base_geo.length_cm, dtype=float)
    edges = np.empty(z.size + 1)
    edges[0] = z[0] - 0.5 * L[0]
    for i in range(z.size):
        edges[i + 1] = edges[i] + L[i]

    print()
    print("=== REQUESTED -> REALIZED (nearest interior face, NUMERICS.md) ===")
    print(f"  {'requested':>10} {'face':>5} {'z_face':>9} {'offset':>8}")
    for req, face in zip(POSITIONS, faces):
        print(
            f"  {req:>10.2f} {int(face):>5d} {edges[int(face)]:>9.3f} "
            f"{edges[int(face)] - req:>+8.3f}"
        )
    span = edges[int(faces[-1])] - edges[int(faces[0])]
    print(f"  realized span: {span:.2f} cm (requested 80.00 cm)")

    print()
    print("=== AREAS AT THE BAFFLE FACES [cm^2] ===")
    Rm = np.asarray(base_geo.Rm_cm, dtype=float)
    Rp = np.asarray(base_geo.Rp_cm, dtype=float)
    a_base = np.asarray(base_geo.neutral_face_area_cm2, dtype=float)
    a_arm = np.asarray(arm_geo.neutral_face_area_cm2, dtype=float)
    pf_base = np.asarray(base_geo.plasma_face_area_cm2, dtype=float)
    pf_arm = np.asarray(arm_geo.plasma_face_area_cm2, dtype=float)
    for face, clear in zip(faces, radii):
        i = int(face)
        rp = max(Rp[i - 1], Rp[i])
        ann_base = a_base[i] - np.pi * rp**2
        ann_arm = a_arm[i] - np.pi * rp**2
        print(
            f"  face {i:>3d}: neutral_face_area {a_base[i]:>9.1f} -> "
            f"{a_arm[i]:>9.1f}   annulus {ann_base:>9.1f} -> {ann_arm:>9.1f} "
            f"({100.0 * ann_arm / ann_base:>5.1f} % open)"
        )
        print(
            f"            blocked ring r={clear:.1f}->{Rm[i]:.1f} cm, "
            f"area {np.pi * (Rm[i] ** 2 - clear**2):>9.1f}"
        )
    print(
        "  plasma_face_area_cm2 identical everywhere: "
        f"{np.array_equal(pf_base, pf_arm)}"
    )
    print(
        "  neutral_face_area_cm2 differs ONLY at the baffle faces: "
        f"{sorted(np.flatnonzero(a_base != a_arm).tolist()) == sorted(faces.tolist())}"
    )

    print()
    print("=== TWO-ZONE ANNULUS KNUDSEN CONDUCTANCE [cm^3/s] ===")
    model = params.get("neutral_transport_model", "knudsen")
    Tn = float(params.get("Tn_K", 300.0))
    mu = float(params.get("mu_neutral", 4.0))
    const = float(params.get("neutral_exchange_coeff_cm3_s", 0.0))
    clausing = float(params.get("neutral_clausing_scale", 1.0))
    kwargs = dict(Tn_K=Tn, mu_neutral=mu, clausing_scale=clausing)
    col_b, ann_b = two_zone_knudsen_coefficients(base_geo, **kwargs)
    col_a, ann_a = two_zone_knudsen_coefficients(arm_geo, **kwargs)
    print(f"  model={model!r} Tn_K={Tn} mu={mu} clausing_scale={clausing}")
    print(f"  COLUMN coefficients bit-identical: {np.array_equal(col_b, col_a)}")
    changed = np.flatnonzero(ann_b != ann_a)
    print(f"  ANNULUS coefficients changed at interior indices: {changed.tolist()}")
    print(f"  {'face':>5} {'ann_base':>13} {'ann_baffled':>13} {'ratio':>8}")
    for face in faces:
        i = int(face) - 1
        print(
            f"  {int(face):>5d} {ann_b[i]:>13.4e} {ann_a[i]:>13.4e} "
            f"{ann_a[i] / ann_b[i]:>8.4f}"
        )
    print()
    print("  SERIES NOTE: each iris adds 1/C_orifice to its OWN face; the")
    print("  three faces are three independent series resistances along z.")
    print("  The code models zero-thickness apertures, NOT a narrowed")
    print("  finite-length tube (physics/neutrals.py:82-84), so the 80 cm")
    print("  LENGTH is representable only as N irises in series. N=3 was")
    print("  pre-registered before the run; a denser series would throttle")
    print("  more. The N-dependence is a disclosed modelling choice.")


if __name__ == "__main__":
    main()
