"""Where the orifice row's mass lands, by cell role, on both shipped meshes.

The orifice row is deliberately NOT masked to _PUFF_ELIGIBLE_ROLES -- it lands
where the ray optics puts it. This measures how much that decision actually
moves, on the two meshes the repo ships: the golden's re-cut nx=60 mesh (which
drops the stance's prescribed radii, so the flight runs from Rm = 50 cm) and
the campaign's nx=268 mesh (measured radii, Rm = 40 cm at the port).
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")

import puff_orifice as po  # noqa: E402
from baseline_sim1d import build_baseline_config  # noqa: E402
from stance_config import load_stance  # noqa: E402

from cablp.solvers._sim1d import default_config  # noqa: E402
from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402
from cablp.solvers._sim1d.physics.neutrals import (  # noqa: E402
    _PUFF_ELIGIBLE_ROLES,
    gas_puff_rate_profile,
    puff_particles_per_s,
)

ARM = {
    "gas_puff_profile": "orifice",
    "gas_puff_orifice_id_cm": 3.95,
    "gas_puff_orifice_length_cm": 22.0,
}


def stance_mesh():
    p, f = default_config()
    s = load_stance("g1atrim")
    p.update(s.params)
    f.update(s.flags)
    p.update(ARM)
    return p, f


print("=== g1aporf_rowcensus: where the orifice row lands, by cell role ===")
print("row: feed pipe d = 3.95 cm (bracket midpoint), L = 22.0 cm, "
      "Gamma = 5.5696\n")

for label, (p, f) in (
    ("GOLDEN  nx=60, prescribed radii DROPPED",
     build_baseline_config(param_overrides=ARM)),
    ("CAMPAIGN nx=268, measured radii", stance_mesh()),
):
    g = build_geometry(p, f)
    z0 = float(p["gas_puff_z_cm"])
    i = int(np.searchsorted(g.z_edges_cm, z0) - 1)
    row, meta = po.launch_row(
        g.z_edges_cm, pipe_id_cm=3.95, aspect_ratio=22.0 / 3.95,
        r_wall_cm=float(g.Rm_cm[i]), r_edge_cm=float(g.Rp_cm[i]), z_port_cm=z0,
    )
    span, lo, hi = po.mass_span(row, g.z_edges_cm)
    cos = gas_puff_rate_profile(g, p["S_gp"], p["gas_puff_valves"],
                                profile="cosine_pipe", z_cm=z0,
                                throw_cm=p["gas_puff_throw_cm"])
    cw = cos * np.asarray(g.neutral_volume_cm3, float)
    cspan, clo, chi = po.mass_span(cw / cw.sum(), g.z_edges_cm)

    print(f"--- {label}")
    print(f"    cells {g.cells}, port cell {i} (role {str(g.cell_role[i])!r}), "
          f"flight Rm = {g.Rm_cm[i]:g} -> Rp = {g.Rp_cm[i]:g} cm at z = {z0:g} cm")
    print(f"    orifice 5-95% span {span:8.4g} cm [{lo:.4g}, {hi:.4g}]   "
          f"perigee-placed {meta['missed_fraction']:.6f}, "
          f"off-grid {meta['clipped_fraction']:.3e}")
    print(f"    cosine  5-95% span {cspan:8.4g} cm [{clo:.4g}, {chi:.4g}]   "
          f"(the superseded fluid envelope, same mesh)")
    nz = np.flatnonzero(row > 0.0)
    by_role = {}
    for k in nz:
        r = str(np.asarray(g.cell_role)[k])
        by_role[r] = by_role.get(r, 0.0) + float(row[k])
    ineligible = 0.0
    for r, m in sorted(by_role.items(), key=lambda kv: -kv[1]):
        ok = r in _PUFF_ELIGIBLE_ROLES
        if not ok:
            ineligible += m
        print(f"      {r:12s} {100.0 * m:8.4f}%"
              f"{'' if ok else '   <-- outside _PUFF_ELIGIBLE_ROLES'}")
    print(f"    TOTAL on roles the length-weighted fluid shapes would EXCLUDE: "
          f"{100.0 * ineligible:.4f}%")
    rate = gas_puff_rate_profile(
        g, p["S_gp"], p["gas_puff_valves"], profile="orifice", z_cm=z0,
        orifice_id_cm=3.95, orifice_length_cm=22.0)
    tot = puff_particles_per_s(p["S_gp"], p["gas_puff_valves"],
                               float(p.get("gas_puff_delivery_fraction", 1.0)))
    got = float(np.sum(rate * g.neutral_volume_cm3))
    print(f"    inflow conserved: {got:.9e} /s vs throughput {tot:.9e} /s, "
          f"rel {abs(got / tot - 1.0):.3e}")
    V_ann = np.asarray(g.Rm_cm, float) ** 2 - np.asarray(g.Rp_cm, float) ** 2
    starved = int(np.count_nonzero((row > 0.0) & (V_ann <= 0.0)))
    print(f"    support cells with Rm == Rp (kinetic_dvm annulus refusal): "
          f"{starved}\n")

print("READING: the row is a ray-optics landing distribution and is applied")
print("UNMASKED by design, so a grazing ray that reaches the column inside the")
print("cathode-anode gap deposits there. The mass involved is small and the")
print("total inflow is conserved exactly either way; masking would have moved")
print("that fuel somewhere the geometry did not put it. Disclosed, not fixed.")
