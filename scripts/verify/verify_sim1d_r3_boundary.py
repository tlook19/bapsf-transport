"""R3.1 characteristic ghost-cell Bohm outflow: pre-registered unit gates.

Static gates for the ghost-cell Bohm outflow boundary (audit A1/A16): a
one-sided ghost-cell KEP/Rusanov flux against the Bohm outflow state
(n_se = n*presheath_alpha, u = c_s into the wall, Te, Ti). These gates check the
boundary in isolation on a controlled state; the *dynamics* (u -> c_s
established, the settled boundary as a net energy sink) are validated by the
separate short startup run (verify_sim1d_r3_boundary_startup.py), NOT here.

Gates:
  G1 both outward normals drain their live cell (Bohm particle SINK, source-left
     and collector-right), and only the live cell (the plenum is untouched);
  G2 the particle sink is the sonic flux ~ n * c_s * A to the flux's KEP
     dissipation band (the ghost n_se = alpha*n plus Rusanov upwinding);
  G3 restoring momentum: at the A1 ANOMALY state (interior flowing AWAY from each
     wall) the boundary's momentum flux is well below the reflecting closed-wall
     pressure p_live, so the interior is pulled toward the wall -- the mechanism
     that drives u -> c_s;
  G4 at the physical Bohm outflow state (u = outward*c_s) the boundary is a net
     energy sink (electron internal + ion internal + reconstructed kinetic < 0),
     i.e. NOT the A1 +18.5 kW kinetic source.

RETIRED 2026-08-31 (Tom), with the legacy volumetric absorber they compared
against: G5 (off-path presence and flag perturbation) in full, and G3's
"reconstructed-kinetic source smaller than the old volumetric sink's" half.
Neither is constructible now that the absorber and its flag are gone -- G3's
restoring-momentum statement, which is measured against the reflecting-wall
pressure rather than against the other operator, is unaffected and still runs.

The SETTLED-window net sink and u -> c_s establishment are the RUN gate
(verify_sim1d_r3_boundary_startup.py), NOT this static probe.

Usage:  python scripts/verify/verify_sim1d_r3_boundary.py
"""
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.state import conservative_from_primitives
from cablp.solvers._sim1d.physics.flux import ion_sound_speed
from cablp.solvers._sim1d.physics.sources import (
    presheath_alpha,
    presheath_length_cm,
)
# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES

ERG_TO_W = 1e-7


def _resolved_sim(extra_flags=None):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    params["nx"] = 120
    if extra_flags:
        flags.update(extra_flags)
    return LAPDSim1D(params, flags)


def _uniform_state(sim, n0, Te0, Ti0, u_edge_frac):
    """Uniform plasma with the two edge cells set to ``u_edge_frac * outward c_s``.

    ``u_edge_frac = 1`` is the physical Bohm outflow (into each wall);
    ``u_edge_frac = -1`` is the A1 anomaly (flowing away from each wall).
    """
    geo = sim.geometry
    cells = geo.cells
    mu = sim._mu
    n = np.full(cells, n0)
    Te = np.full(cells, Te0)
    Ti = np.full(cells, Ti0)
    u = np.zeros(cells)
    cs = ion_sound_speed(Te0, mu)
    edges = {}
    for face in np.flatnonzero(np.asarray(geo.plasma_absorbing, bool)):
        live = int(geo.plasma_face_live_cell[face])
        if live < 0:
            continue
        outward = -1.0 if live == face else 1.0
        u[live] = u_edge_frac * outward * cs
        edges[int(face)] = live
    state = conservative_from_primitives(
        n=n, nn=np.full(cells, 1.0e12), u=u, Te=Te, Ti=Ti,
        ion_mass_g=sim.ion_mass_g,
    )
    return state, edges, cs


def _net_power(term, geo):
    """Net booked power [W] of a boundary term: electron + ion internal +
    reconstructed kinetic, integrated over the plasma volume."""
    Vp = np.asarray(geo.plasma_volume_cm3)
    return {
        "electron": float(np.sum(term.Ee * Vp) * ERG_TO_W),
        "ion": float(np.sum(term.Ei * Vp) * ERG_TO_W),
    }


def main():
    ok = True
    sim = _resolved_sim()
    geo = sim.geometry
    mu = sim._mu
    m_i = sim.ion_mass_g
    Vp = np.asarray(geo.plasma_volume_cm3)
    n0, Te0, Ti0 = 5.0e12, 4.0, 1.0

    # --- G1/G2: Bohm particle sink on both normals, plenum untouched -------
    state, edges, cs = _uniform_state(sim, n0, Te0, Ti0, u_edge_frac=1.0)
    # Call the BOUND method, not the module function: it threads the resolved
    # run constants (gas_type, b_presheath_length, the cathode jet spec, the
    # wave-speed and energy-consistency selectors, alpha_isat and
    # b_surface_loss) exactly as the production callers do, so this probe
    # cannot drift from the operator it is gating. The former raw call
    # hardcoded alpha_isat = exp(-0.5) and b_surface_loss = 1.0, which are the
    # values the resolved bundle carries at this configuration.
    ch = sim.characteristic_boundary_rhs(state=state)
    dead = ~np.asarray(geo.plasma_active, bool)
    g1 = all(ch.n[live] < 0.0 for live in edges.values())
    g1 &= all(ch.nn[live] > 0.0 for live in edges.values())  # neutral return
    g1_plenum = bool(np.all(ch.n[dead] == 0.0) and np.all(ch.M[dead] == 0.0))
    print(f"G1 both normals drain live cell, neutral return : {g1}")
    print(f"G1 plenum (plasma-dead) untouched               : {g1_plenum}")

    g2 = True
    for face, live in edges.items():
        # The removal rate is the sonic Bohm flux at the sheath-edge density,
        # broadened by the Rusanov/KEP dissipation across the ghost jump. Compare
        # to the interior sonic flux n c_s A (band, not equality: the dissipation
        # term is a genuine part of the numerical flux, not an error).
        sonic = n0 * cs * float(geo.plasma_face_area_cm2[face])
        actual = -ch.n[live] * Vp[live]  # particles/s removed
        ratio = actual / sonic
        g2 &= 0.6 < ratio < 1.6 and actual > 0.0
        print(f"G2 face {face:3d}: Bohm sink {actual:.3e} vs n c_s A {sonic:.3e}"
              f"  (ratio {ratio:.3f})")

    # --- G3: restoring momentum at the A1 anomaly state -------------------
    from cablp.solvers._sim1d.core.state import derive_state
    anom_state, anom_edges, _ = _uniform_state(
        sim, n0, Te0, Ti0, u_edge_frac=-1.0
    )  # interior flowing AWAY from each wall (the A1 pathology)
    dv_a = derive_state(anom_state, floors=sim._floors, ion_mass_g=m_i)
    ch_a = sim.characteristic_boundary_rhs(state=anom_state)
    g3 = True
    for face, live in anom_edges.items():
        p_live = float(dv_a.p[live])
        # Physical +z momentum flux the ghost puts on the face (undo the one-
        # sided divergence sign to recover the face flux magnitude).
        signL = 1.0 if live == face else -1.0
        f_M = ch_a.M[live] * Vp[live] / float(geo.plasma_face_area_cm2[face]) / signL
        u_l = float(dv_a.u[live])
        dK_new = float((u_l * ch_a.M[live] - 0.5 * m_i * u_l**2 * ch_a.n[live])
                       * Vp[live] * ERG_TO_W)
        restoring = f_M < 0.5 * p_live  # far below the reflecting wall it replaces
        g3 &= restoring
        print(f"G3 face {face:3d}: anomaly F_M {f_M:.3e} vs p_live {p_live:.3e} "
              f"(ratio {f_M/p_live:.3f}<0.5); reconstructed KE {dK_new:+.2e} W")

    # --- G4: net energy sink at the physical Bohm outflow state -----------
    def reconstructed_kinetic(term, u):
        dK = (u * term.M - 0.5 * m_i * u**2 * term.n) * Vp * ERG_TO_W
        return float(np.sum(dK))

    u_bohm = derive_state(state, floors=sim._floors, ion_mass_g=m_i).u
    P = _net_power(ch, geo)
    P_kin = reconstructed_kinetic(ch, u_bohm)
    net = P["electron"] + P["ion"] + P_kin
    g4 = net < 0.0
    print(f"G4 Bohm-outflow state [W]: e {P['electron']:+.3e}  i {P['ion']:+.3e}"
          f"  kinetic {P_kin:+.3e}  NET {net:+.3e}  (sink: {g4})")

    # G5 (off-path presence + flag perturbation) was RETIRED 2026-08-31 (Tom)
    # with the flag it switched: there is no off path to be present, and no
    # second operator to perturb away from.

    ok = g1 and g1_plenum and g2 and g3 and g4
    print("\nboundary unit gates:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
