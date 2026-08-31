"""Negative control: the extended (c) loop FAILS against the pre-fix freeze set.

Reproduces 4f1e20b's behaviour by freezing only the four arrays that version
froze, then running the same five-component probe the fixed instrument runs.
cathode_face_indices must accept the write -- which is the F2 defect, and is
what the old one-component probe could not see.
"""
import sys

sys.path.insert(0, "scripts")
import numpy as np
from cablp.solvers._sim1d.core.config import default_config
from cablp.solvers._sim1d.core.geometry import build_geometry

params, flags = default_config()
params = dict(params)
params["nx"] = 60
geom = build_geometry(params, flags)

# The PRE-FIX freeze set (four of five).
for values in (geom.z_cm, geom.length_cm, geom.z_edges_cm, geom.plasma_active):
    values.flags.writeable = False

accepted = []
for name in ("z_cm", "length_cm", "z_edges_cm", "plasma_active",
             "cathode_face_indices"):
    v = getattr(geom, name)
    try:
        v[0] = v[0]
    except ValueError:
        print(f"  {name}: refused (guarded)")
    else:
        print(f"  {name}: ACCEPTED the write  <-- unguarded")
        accepted.append(name)

print()
print(f"pre-fix freeze set leaves {len(accepted)} component(s) writeable: {accepted}")
print("=> the five-component (c) loop FAILS on the pre-fix code, as it must;")
print("   the old one-component probe (z_cm only) PASSED it.")
