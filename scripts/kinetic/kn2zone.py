"""KN2Zone: deterministic kinetic two-zone neutrals (K1).

The synthesis instrument between the moment two-zone model
and the TPMC (`mc_neutrals.py`): two distribution functions on a shared
velocity grid,

    f_c(z, v_z, v_perp)   column   (r < Rp)
    f_a(z, v_z, v_perp)   annulus  (Rp < r < Rm)

with velocity-DEPENDENT radial reduction -- zone exchange and wall collision
rates scale with v_perp, so grazing atoms (small v_perp) stream axially with
few interactions: the duct tail the Fickian moment closure cannot represent
is representable here by construction.

Rates (flux-equivalent forms; angle-averaging a 2D in-plane isotropic
distribution gives <v_perp> = (pi/4) vbar, so each reproduces the
free-molecular flux rate exactly at the Maxwellian zeroth moment -- the
moment model's K_r and the M5 factors are these operators' zeroth moments,
a consistency check, not an input):

    nu_x  (column -> annulus)  = 2 v_perp / (pi Rp)
    nu_x' (annulus -> column)  = 2 v_perp Rp / (pi (Rm^2 - Rp^2))
    nu_w  (annulus wall)       = 2 v_perp Rm / (pi (Rm^2 - Rp^2))

Steady state by generation iteration (KN1D's algorithm class): each
generation is an implicit-upwind march in z per velocity bin with the 2x2
zone coupling solved per cell; the collided fluxes (CX in the column at the
local ion Maxwellian -- the relay; wall accommodation in the annulus at
300 K; end-wall thermal re-emission with pump sticking; anode-mesh
interception) re-enter as the next generation's sources. Ionization
absorbs. Atomic data and the source ledger are the run's own, via
`mc_neutrals.load_background` -- disagreement between instruments is method
error, never input error.

Discrete shifted Maxwellians use a moment-exact projection: analytic bin
masses, then a two-basis linear compensation (KN1DPy
`create_shifted_maxwellian` algorithm, reimplemented; LGPL clone consulted
in scratch only) so the numerically evaluated drift and energy moments hit
their targets to rounding.

Usage:
    python scripts/kn2zone.py RUN.h5 [--window 5 19.5] [--nvz 80 --nvp 24]
        [--truncate 1e-3] [--max-gen 400] [--out PREFIX]
        [--puff-orifice {wide,narrow}]
    python scripts/kn2zone.py --selftest
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mc_neutrals import (  # noqa: E402
    EV,
    KB,
    M_HE,
    PUFF_ORIFICE_ENDPOINTS,
    T_WALL_K,
    load_background,
)


from cablp.solvers._sim1d.physics.kinetic_neutrals import (  # noqa: E402
    KN2Zone,
    KN2ZoneJump,
    VGrid,
    bin_edges,
    puff_launch_bins,
    stretched_axis,
    stretched_positive_axis,
)


def build_hop_kernels(kn):
    """Return the K2 hop-matrix closure pieces from a ``KN2ZoneJump``.

    All from geometry + the 300 K wall spectrum (no free parameters):

      C_hop  (nz, nz)  symmetric pairwise annulus conductances [cm^3/s];
                       net annulus flow i->j = C_hop_ij (n_i - n_j).
      T_in   (nz, nz)  annulus->column transfer [cm^3/s]: flow into column
                       cell j per unit annulus density in i. Columns are
                       normalized so sum_i T_in_ij equals the local K_r_j
                       exactly -- uniform equal densities stay stationary,
                       and the total reduces to the moment model's K_r
                       identically (Rm * F = Rp).

    End-clipped flights fold into the end cells (the solver's ends
    re-emit in place; pumping remains the separate named sink), keeping
    both kernels conservative on the domain.
    """
    g = kn.g
    nz = kn.nz
    wall_spec = kn.M_wall
    P_ww = np.zeros((nz, nz))
    P_in = np.zeros((nz, nz))
    dummy_t = np.zeros(nz)
    dummy_tv = np.zeros(nz)
    for i in range(nz):
        for P, chord, frac in (
            (P_ww, kn.c_ww, 1.0 - kn.F_inner[i]),
            (P_in, kn.c_wi, kn.F_inner[i]),
        ):
            if frac <= 0:
                continue
            launch = np.zeros((nz, g.nvz, g.nvp))
            launch[i] = frac * wall_spec
            land, end_L, end_R = kn._fly(launch, chord, dummy_t, dummy_tv)
            row = land.sum(axis=(1, 2))
            row[0] += float(end_L.sum())
            row[-1] += float(end_R.sum())
            total = row.sum()
            if total > 0:
                # row sums to the branch mass (frac) up to numeric leakage;
                # renormalize so the kernels are exactly conservative.
                P[i] = row * (frac / total)
    vbar = np.sqrt(8.0 * KB * T_WALL_K / (np.pi * M_HE))
    gamma = 0.25 * vbar * 2.0 * np.pi * kn.Rm * kn.dz  # wall flux per density
    C = gamma[:, None] * P_ww  # includes the (1-F) branch mass already
    C_hop = 0.5 * (C + C.T)  # enforce reciprocity exactly
    np.fill_diagonal(C_hop, 0.0)
    T_in = gamma[:, None] * P_in
    # column normalization: sum_i T_in_ij == K_r_j exactly
    K_r = 0.25 * vbar * 2.0 * np.pi * kn.Rp * kn.dz
    col_sum = T_in.sum(axis=0)
    scale = np.where(col_sum > 0, K_r / np.maximum(col_sum, 1e-300), 0.0)
    T_in = T_in * scale[None, :]
    return C_hop, T_in, K_r


def moment_hop_steady(kn, C_hop, T_in, K_r):
    """Solve the frozen-plasma steady moment system with the hop closure.

    Unknowns [nn (column), nn_a]; column axial transport keeps its Fickian
    conductances (collisional gas), the annulus uses the hop matrix; the
    exchange is K_r (column->annulus, local) vs T_in (annulus->column,
    kernel-spread); ionization is the column sink; ledger sources routed
    as in the solver (faces/vol_rec -> column, puff -> annulus); pumping
    at the chamber rate on both zones at the end cells (TPMC sticking
    equivalent). Returns (nn_col, nn_ann).
    """
    nz = kn.nz
    bgs = kn.bg["sources"]
    A = np.zeros((2 * nz, 2 * nz))
    b = np.zeros(2 * nz)
    V_col, V_ann = kn.V_col, kn.V_ann
    # column axial: Fickian tube conductances (2/3 vbar Rp A / dz)
    vbar = np.sqrt(8.0 * KB * T_WALL_K / (np.pi * M_HE))
    Rp_f = 0.5 * (kn.Rp[:-1] + kn.Rp[1:])
    A_col_f = np.pi * np.minimum(kn.Rp[:-1], kn.Rp[1:]) ** 2
    dzc = 0.5 * (kn.dz[:-1] + kn.dz[1:])
    C_col = (2.0 / 3.0) * vbar * Rp_f * A_col_f / dzc
    if 0 <= kn.mesh_face - 1 < C_col.size:
        C_col[kn.mesh_face - 1] *= kn.transparency
    for f, c in enumerate(C_col):
        A[f, f] -= c
        A[f, f + 1] += c
        A[f + 1, f + 1] -= c
        A[f + 1, f] += c
    # annulus hops
    for i in range(nz):
        for j in range(nz):
            c = C_hop[i, j]
            if c <= 0:
                continue
            A[nz + i, nz + i] -= c
            A[nz + i, nz + j] += c
    # exchange: column -K_r n_c +T_in^T n_a ; annulus +K_r n_c (local) ...
    for i in range(nz):
        A[i, i] -= K_r[i]
        A[nz + i, i] += K_r[i]
        for j in range(nz):
            t = T_in[i, j]
            if t <= 0:
                continue
            A[j, nz + i] += t
            A[nz + i, nz + i] -= t
    # ionization sink (column)
    for i in range(nz):
        A[i, i] -= kn.nu_ion[i] * V_col[i]
    # pumping: chamber-rate sink on both zones at the end cells
    Vm = V_col + V_ann
    for idx, S in ((0, kn.bg["S_pump_L"]), (nz - 1, kn.bg["S_pump_R"])):
        rate = S * 1e3 / Vm[idx]
        A[idx, idx] -= rate * V_col[idx]
        A[nz + idx, nz + idx] -= rate * V_ann[idx]
    # sources [atoms/s]
    b[0] -= bgs.get("cathode_face", 0.0)
    b[nz - 1] -= bgs.get("collector_face", 0.0)
    rec = kn.bg.get("rec_cell")
    if rec is not None:
        scale = bgs.get("vol_rec", rec.sum()) / max(rec.sum(), 1e-300)
        b[:nz] -= scale * rec
    for name, off in (("anode_left", -1), ("anode_right", 0)):
        j = min(max(kn.mesh_face + off, 0), nz - 1)
        b[j] -= bgs.get(name, 0.0)
    # Same placement as the kinetic engine's, from the same helper: the two
    # halves of this instrument are compared against each other, so a
    # difference in where the fuel enters would read as closure error.
    for iz, rate in puff_launch_bins(bgs, kn.z_edges, nz):
        b[nz + iz] -= rate
    x = np.linalg.solve(A, b)
    return x[:nz], x[nz:]


# ---------------------------------------------------------------- self-tests


def selftest():
    ok = True
    # 1. Moment-exact shifted Maxwellians across the temperature range.
    g = VGrid(4e6, 4e6, 80, 24, 0.25 * np.sqrt(KB * T_WALL_K / M_HE))
    for T, u in ((0.0259, 0.0), (0.0259, 3e4), (1.0, 5e4), (3.0, -2e5)):
        f = g.maxwellian(T, u)
        s2 = T * EV / M_HE
        n = float(f.sum())
        m1 = float((f * g.VZ).sum())
        m2 = float((f * g.V2).sum())
        e_t = u**2 + 3.0 * s2
        if not (
            abs(n - 1) < 1e-12
            and abs(m1 - u) <= 1e-8 * max(abs(u), np.sqrt(s2))
            and abs(m2 - e_t) <= 1e-8 * e_t
        ):
            print(f"FAIL maxwellian moments T={T} u={u}: "
                  f"n-1={n-1:.2e} du={m1-u:.3e} dE/E={(m2-e_t)/e_t:.2e}")
            ok = False
    print("shifted-Maxwellian moments:", "ok" if ok else "FAIL")

    # 2. Streaming-only attenuation: single-bin beam through uniform
    # absorber matches the discrete closed form (1 + nu dz/vz)^-i, and the
    # closed form converges to exp(-nu z / vz).
    nz, nu, dz = 40, 5.0e3, 10.0
    bg = _slab_background(nz=nz, dz=dz, nu_ion=nu)
    kn = KN2Zone(bg, nvz=16, nvp=6, verbose=False, max_gen=1)
    g = kn.g
    ivz = int(np.argmin(np.abs(g.vz - 2.0e5)))
    ivp = 0
    Fin = np.zeros((g.nvz, g.nvp))
    Fin[ivz, ivp] = 1.0
    kn.nux[:] = 0.0
    kn.nuxp[:] = 0.0
    kn.nuw[:] = 0.0
    Fc, _, _, _ = kn.sweep(
        np.zeros((nz, g.nvz, g.nvp)), np.zeros((nz, g.nvz, g.nvp)),
        Fin, np.zeros_like(Fin), np.zeros_like(Fin), np.zeros_like(Fin),
    )
    vz = g.vz[ivz]
    expected = (1.0 + nu * dz / vz) ** -(np.arange(nz) + 1)
    got = Fc[:, ivz, ivp]
    err = np.max(np.abs(got - expected) / expected)
    print(f"streaming attenuation vs discrete closed form: "
          f"max rel {err:.2e}", "ok" if err < 1e-12 else "FAIL")
    ok = ok and err < 1e-12

    # 3. Closed box: no sinks, reflective ends, walls + zone exchange only.
    # The generation-iterated equilibrium must be axially flat with EQUAL
    # zone densities (detailed balance of nu_x / nu_x') -- the kinetic
    # counterpart of the TPMC's closed-box test.
    bg2 = _slab_background(nz=12, dz=30.0, nu_ion=0.0)
    kn2 = KN2Zone(bg2, nvz=24, nvp=8, verbose=False, max_gen=4000,
                  truncate=1e-7)
    kn2.s_L = 0.0
    kn2.s_R = 0.0
    src = dict(kn2.bg["sources"])
    kn2.bg["sources"] = {
        "puff": 1.0e20, "puff_z": 0.5 * 12 * 30.0,
        "cathode_face": 0.0, "collector_face": 0.0,
    }
    res = kn2.solve()
    # normalized flatness and zone balance over the interior
    nc, na = res["nn_col"], res["nn_ann"]
    flat = np.max(np.abs(na / na.mean() - 1.0))
    balance = np.max(np.abs(nc / na - 1.0))
    print(f"closed box: annulus flatness {flat:.3f}, "
          f"col/ann balance dev {balance:.3f}",
          "ok" if flat < 0.05 and balance < 0.05 else "FAIL (loose gate)")
    ok = ok and flat < 0.05 and balance < 0.05
    kn2.bg["sources"] = src
    return 0 if ok else 1


def _slab_background(nz, dz, nu_ion):
    z_edges = np.arange(nz + 1) * dz
    return {
        "z_edges": z_edges,
        "Rp": np.full(nz, 15.0),
        "Rm": np.full(nz, 50.0),
        "nu_ion": np.full(nz, nu_ion),
        "nu_cx": np.zeros(nz),
        "Ti": np.full(nz, 0.05),
        "u": np.zeros(nz),
        "T_s": T_WALL_K,
        "S_pump_L": 0.0,
        "S_pump_R": 0.0,
        "eta": 0.0,
        "mesh_edge": -999,
        "sources": {},
    }


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--window", nargs=2, type=float, default=(5.0, 19.5))
    ap.add_argument("--nvz", type=int, default=80)
    ap.add_argument("--nvp", type=int, default=24)
    ap.add_argument("--zone-rates", choices=("cauchy", "flux"),
                    default="flux")
    ap.add_argument("--kernel", choices=("rate", "jump"), default="rate",
                    help="annulus transport: exponential rates or "
                         "bounded-chord jumps (K1b)")
    ap.add_argument("--moment-hop", action="store_true",
                    help="K2 offline gate: solve the frozen-plasma steady "
                         "MOMENT system with the hop-matrix closure and "
                         "compare against the kinetic solution")
    ap.add_argument("--truncate", type=float, default=1e-3)
    ap.add_argument("--max-gen", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--puff-orifice", choices=PUFF_ORIFICE_ENDPOINTS, default=None,
        help="place the puff by the CAD-derived tube-beamed injection row "
             "instead of the run's own gas_puff_profile row, at the named "
             "endpoint of the one-sided feed-line bracket (default: unset, "
             "the run's own row)",
    )
    args = ap.parse_args(argv)

    if args.selftest:
        sys.exit(selftest())
    if args.moment_hop and args.kernel != "jump":
        ap.error("--moment-hop requires --kernel jump (the K1b instrument)")
    if args.run is None:
        ap.error("RUN.h5 required unless --selftest")

    bg = load_background(
        args.run, tuple(args.window), puff_orifice=args.puff_orifice
    )
    bg["zone_rates"] = args.zone_rates
    cls = KN2ZoneJump if args.kernel == "jump" else KN2Zone
    kn = cls(bg, nvz=args.nvz, nvp=args.nvp, truncate=args.truncate,
             max_gen=args.max_gen)
    res = kn.solve()
    print(f"converged in {res['generations']} generations")

    ze = bg["z_edges"]
    zc = 0.5 * (ze[:-1] + ze[1:])
    print(f"\n{'z[cm]':>7} {'nn_col':>10} {'nn_ann':>10} "
          f"{'un_col[km/s]':>12} {'un_ann':>8}")
    for i in range(0, zc.size, max(1, zc.size // 18)):
        print(f"{zc[i]:7.0f} {res['nn_col'][i]:10.3g} "
              f"{res['nn_ann'][i]:10.3g} {res['un_col'][i]/1e5:12.2f} "
              f"{res['un_ann'][i]/1e5:8.2f}")

    if args.moment_hop:
        C_hop, T_in, K_r = build_hop_kernels(kn)
        m_col, m_ann = moment_hop_steady(kn, C_hop, T_in, K_r)
        print("\n=== K2 offline gate: hop-matrix MOMENT model vs K1b kinetic ===")
        print(f"{'z[cm]':>7} {'ann_moment':>11} {'ann_kinetic':>11} {'ratio':>6} "
              f"{'col ratio':>9}")
        for i in range(0, zc.size, max(1, zc.size // 16)):
            print(f"{zc[i]:7.0f} {m_ann[i]:11.3g} {res['nn_ann'][i]:11.3g} "
                  f"{m_ann[i] / max(res['nn_ann'][i], 1e-3):6.2f} "
                  f"{m_col[i] / max(res['nn_col'][i], 1e-3):9.2f}")
        mid = (zc >= 500.0) & (zc <= 1000.0)
        print("mid-machine moment/kinetic: ann %.2f  col %.2f" % (
            m_ann[mid].mean() / res["nn_ann"][mid].mean(),
            m_col[mid].mean() / res["nn_col"][mid].mean()))

    out = args.out or (Path(args.run).stem + "_kn2zone")
    np.savez(
        Path(args.run).parent / f"{out}.npz",
        z=zc, nn_col=res["nn_col"], nn_ann=res["nn_ann"],
        un_col=res["un_col"], un_ann=res["un_ann"],
        generations=res["generations"],
    )
    print(f"\nsaved {out}.npz")


if __name__ == "__main__":
    main()
