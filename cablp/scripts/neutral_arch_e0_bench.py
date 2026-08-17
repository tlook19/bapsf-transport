"""E0: frozen-background COST microbenchmark for the kinetic neutral architecture.

Measures the per-unit cost of the three candidate kinetic-neutral
architectures (deterministic velocity grid / micro-macro deviational MC /
full-particle MC) on ONE shared frozen background, so the K1 architecture
choice rests on measured arithmetic rather than estimates.

**This is a measurement instrument, not a physics instrument.** It runs no
campaign point, fits nothing, and compares no closure against data. Accuracy
and cadence trades are E1; physics comparison is E2. Nothing here touches the
solver: the background is a saved run read read-only, and every kernel timed
lives in this file or in the already-shipped `physics/kinetic_neutrals.py`.

All arms share the SAME background: the same saved run, the same plateau
window (so the same elapsed time), the same ADAS/CX atomic rates, the same
source ledger, and the same (nvz, nvp) velocity grid. Disagreement between
arms is therefore an architecture cost difference and nothing else.

What is measured (the E0 list)
------------------------------
1. construction cost -- velocity grid, DVM operators, jump kernels, the
   compiled flight engine, and the MC particle population;
2. one transient DVM update (explicit upwind + collision on the full
   (nz, nvz, nvp) state) and one DVM generation sweep;
3. persistent-MC events per second and memory;
4. macro-plus-correction tally cost (three-moment deposition);
5. wall-energy tally cost;
6. source-channel count dependence;
7. the correction-particle population a three-moment macro control variate
   would need, estimated from the measured deviational mass fraction.

MC kernel choice: batch-synchronous **pure-numpy vectorization**, not a
compiled extension. The population is persistent (particles live across
substeps and absorbed histories are respawned from the ledger, holding the
count fixed), and each iteration resolves one flight segment for every live
particle with whole-array operations -- the same shape as `mc_neutrals.py`,
which is how a coupled MC would actually be written in this package. The
production profile's call-overhead finding does not transfer: that result is
about scalar Python called ~6e4 times per step, whereas this loop does O(20)
array operations over 1e5+ particles, so the per-event cost is dominated by
memory traffic that a Cython transcription would not remove. Adding a second
extension module was therefore rejected as build risk (it would touch
`build_ext.py` and the shipped `_cathode_kernels_cy` build) for no measurement
fidelity.

Usage (single command, reruns end to end):

    PYTHONPATH=<checkout>/cablp python scripts/neutral_arch_e0_bench.py \
        --run scripts/es1_kn2z_promoted_nx240.h5 --out-dir scripts
"""

import argparse
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mc_neutrals import (  # noqa: E402
    EV,
    KB,
    M_HE,
    RAY_EPS_CM,
    T_WALL_K,
    cosine_emit,
    load_background,
    maxwellian,
    wall_emit_inward,
)

from cablp.solvers._sim1d.physics.kinetic_neutrals import (  # noqa: E402
    KineticEngineFast,
    KN2Zone,
    KN2ZoneJump,
    VGrid,
)

# Production-run scale factors used ONLY in the projection arithmetic of the
# summary. They are inputs quoted from the nx=240 production profile
# (es1_prod_profile_nx240_ANALYSIS.md, 2026-07-30), not measurements of this
# bench, and are labelled as such wherever they appear.
PROD_STEPS = 95_483
PROD_SIM_TIME_S = 26.55e-3  # 0.25 + 0.34 + 20.00 + 5.96 ms, per the phase table
PROD_WALL_S = 1398.0  # unprofiled reference run
E4_TARGET_S = (350.0, 400.0)
E4_TRADE_S = (400.0, 699.0)


# ------------------------------------------------------------------ timing


class Timed:
    """One repeated measurement: median plus spread, and the raw samples."""

    def __init__(self, name, samples, unit="s", note=""):
        self.name = name
        self.samples = list(samples)
        self.unit = unit
        self.note = note

    @property
    def median(self):
        return statistics.median(self.samples)

    @property
    def lo(self):
        return min(self.samples)

    @property
    def hi(self):
        return max(self.samples)

    @property
    def spread(self):
        """Full-range spread relative to the median."""
        m = self.median
        return (self.hi - self.lo) / m if m else float("nan")

    def line(self):
        return (
            f"{self.name:<56s} median {self.median:12.6g} {self.unit:<6s}"
            f" [min {self.lo:11.6g}, max {self.hi:11.6g}]"
            f" n={len(self.samples)} spread {100 * self.spread:6.1f}%"
            + (f"  # {self.note}" if self.note else "")
        )


def repeat(name, fn, n, unit="s", note=""):
    """Call ``fn`` ``n`` times, timing each call. One untimed warm-up first."""
    fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return Timed(name, samples, unit=unit, note=note)


def machine_note():
    load1, load5, load15 = os.getloadavg()
    return (
        f"load average {load1:.2f} / {load5:.2f} / {load15:.2f} "
        f"(1/5/15 min), {os.cpu_count()} logical cores"
    )


def rss_bytes():
    """Peak resident set size in bytes (macOS reports ru_maxrss in bytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024


# ------------------------------------------------------------------ DVM arm


def transient_state(kn):
    """Allocate a transient DVM state: column and annulus distributions."""
    g = kn.g
    shape = (kn.nz, g.nvz, g.nvp)
    return np.zeros(shape), np.zeros(shape)


def transient_update(kn, Fc, Fa, dt, work):
    """One explicit transient DVM update of (Fc, Fa) -- the per-step cost.

    First-order upwind advection in z plus the local collision operator:
    ionization absorbs, CX relaxes the column to the local ion Maxwellian,
    the zone operators exchange column <-> annulus, and the annulus wall
    rate re-emits at the 300 K cosine-wall spectrum. Fully vectorized over
    the whole (nz, nvz, nvp) state -- this is the operation a time-dependent
    DVM would perform once per solver substep.
    """
    g = kn.g
    vzp, vzm, inv_dz = work["vzp"], work["vzm"], work["inv_dz"]
    up_c, dn_c, up_a, dn_a = work["up_c"], work["dn_c"], work["up_a"], work["dn_a"]
    up_c[0] = 0.0
    up_c[1:] = Fc[:-1]
    dn_c[-1] = 0.0
    dn_c[:-1] = Fc[1:]
    up_a[0] = 0.0
    up_a[1:] = Fa[:-1]
    dn_a[-1] = 0.0
    dn_a[:-1] = Fa[1:]
    adv_c = -(vzp * (Fc - up_c) + vzm * (dn_c - Fc)) * inv_dz
    adv_a = -(vzp * (Fa - up_a) + vzm * (dn_a - Fa)) * inv_dz
    nux = kn.nux[:, None, :]
    nuxp = kn.nuxp[:, None, :]
    nuw = kn.nuw[:, None, :]
    sink_c = (kn.nu_ion + kn.nu_cx)[:, None, None]
    n_c = Fc.sum(axis=(1, 2))
    n_a = Fa.sum(axis=(1, 2))
    coll_c = (
        -(sink_c + nux) * Fc
        + nux * Fa
        + (kn.nu_cx * n_c)[:, None, None] * kn.M_cx
    )
    wall_rate = (Fa * nuw).sum(axis=(1, 2))
    coll_a = (
        -(nuxp + nuw) * Fa
        + nuxp * Fc
        + wall_rate[:, None, None] * kn.M_wall[None, :, :]
    )
    Fc += dt * (adv_c + coll_c)
    Fa += dt * (adv_a + coll_a)
    np.maximum(Fc, 0.0, out=Fc)
    np.maximum(Fa, 0.0, out=Fa)
    return float(n_a.sum())


def assemble_sources(kn, sources):
    """Build the per-bin volume sources and boundary inflows from a ledger.

    Mirrors the prologue of ``KN2Zone.solve`` -- the only part of the DVM
    cost that depends on how many source channels the ledger carries.
    """
    g = kn.g
    nz = kn.nz
    Sc = np.zeros((nz, g.nvz, g.nvp))
    Sa = np.zeros((nz, g.nvz, g.nvp))
    inflows = []
    for name, T, sign, area in (
        ("cathode_face", kn.bg["T_s"], +1, kn.A_col[0]),
        ("collector_face", T_WALL_K, -1, kn.A_col[-1]),
    ):
        rate = sources.get(name, 0.0)
        spec = g.half_flux_spectrum(T, sign)
        with np.errstate(divide="ignore", invalid="ignore"):
            dens = np.where(np.abs(g.VZ) > 0, spec / (np.abs(g.VZ) * area), 0.0)
        inflows.append(rate * dens)
    if sources.get("puff", 0.0) > 0:
        iz = int(np.searchsorted(kn.bg["z_edges"], sources["puff_z"]) - 1)
        iz = min(max(iz, 0), nz - 1)
        Sa[iz] += sources["puff"] / kn.V_ann[iz] * kn.M_wall
    if sources.get("vol_rec", 0.0) > 0:
        rec = kn.bg.get("rec_cell")
        if rec is not None and rec.sum() > 0:
            scale = sources["vol_rec"] / rec.sum()
            for i in range(nz):
                if rec[i] > 0:
                    Sc[i] += scale * rec[i] / kn.V_col[i] * kn.M_cx[i]
    for name, sign in (("anode_left", -1), ("anode_right", +1)):
        rate = sources.get(name, 0.0)
        if rate <= 0:
            continue
        j = kn.mesh_face - 1 if sign < 0 else kn.mesh_face
        j = min(max(j, 0), nz - 1)
        Sc[j] += rate / kn.V_col[j] * g.half_flux_spectrum(T_WALL_K, sign)
    return Sc, Sa, inflows


def transient_work(kn):
    g = kn.g
    shape = (kn.nz, g.nvz, g.nvp)
    return {
        "vzp": np.maximum(g.VZ, 0.0)[None, :, :],
        "vzm": np.minimum(g.VZ, 0.0)[None, :, :],
        "inv_dz": (1.0 / kn.dz)[:, None, None],
        "up_c": np.empty(shape),
        "dn_c": np.empty(shape),
        "up_a": np.empty(shape),
        "dn_a": np.empty(shape),
    }


def dvm_arm(bg, args, out):
    """Run and record the deterministic-velocity-grid measurements."""
    rep = args.repeats
    Ti_max = float(np.max(bg["Ti"]))
    u_max = float(np.max(np.abs(bg["u"])))
    vmax = 4.0 * np.sqrt(max(Ti_max, 0.5) * EV / M_HE) + 1.5 * u_max
    v_fine = 0.25 * np.sqrt(KB * T_WALL_K / M_HE)

    res = []
    res.append(
        repeat(
            "construct: velocity grid VGrid(nvz,nvp)",
            lambda: VGrid(vmax, vmax, args.nvz, args.nvp, v_fine),
            rep,
            note="distribution-grid construction",
        )
    )
    res.append(
        repeat(
            "construct: DVM operators KN2Zone(rate kernel)",
            lambda: KN2Zone(bg, nvz=args.nvz, nvp=args.nvp, verbose=False),
            rep,
            note="grid + per-cell shifted Maxwellians + zone rates",
        )
    )
    res.append(
        repeat(
            "construct: DVM operators KN2ZoneJump(chord kernel)",
            lambda: KN2ZoneJump(bg, nvz=args.nvz, nvp=args.nvp, verbose=False),
            rep,
            note="adds the per-cell chord classes",
        )
    )

    kn = KN2Zone(bg, nvz=args.nvz, nvp=args.nvp, verbose=False)
    jump = KN2ZoneJump(bg, nvz=args.nvz, nvp=args.nvp, verbose=False)
    res.append(
        repeat(
            "construct: compiled flight engine KineticEngineFast",
            lambda: KineticEngineFast(jump),
            max(3, rep // 2),
            note="one-time flight-kernel compile (2 x nz _fly passes)",
        )
    )

    g = kn.g
    nz = kn.nz
    shape = (nz, g.nvz, g.nvp)
    Sc = np.zeros(shape)
    Sa = np.zeros(shape)
    Zin = np.zeros((g.nvz, g.nvp))
    res.append(
        repeat(
            "update: DVM generation sweep (implicit upwind march)",
            lambda: kn.sweep(Sc, Sa, Zin, Zin, Zin, Zin),
            rep,
            note="one generation, both directions, 2x2 zone coupling",
        )
    )

    Fc, Fa = transient_state(kn)
    Fc += 1.0e6 * kn.M_wall[None, :, :]
    Fa += 1.0e6 * kn.M_wall[None, :, :]
    work = transient_work(kn)
    dt = float(0.2 * np.min(kn.dz) / np.max(np.abs(g.vz)))
    res.append(
        repeat(
            "update: transient DVM step (explicit upwind + collision)",
            lambda: transient_update(kn, Fc, Fa, dt, work),
            rep,
            note=f"dt={dt:.3e} s (0.2 CFL); the per-substep cost",
        )
    )

    # Steady solve: the quasi-static refresh unit, and the source of the
    # converged distribution the control-variate estimate is measured on.
    sol = kn.solve()
    res.append(
        repeat(
            "solve: DVM steady generation iteration to truncate=1e-3",
            kn.solve,
            max(3, rep // 3),
            note=f"{sol['generations']} generations per solve",
        )
    )

    # Source-channel count dependence. The sweep itself is channel-blind
    # (its cost is set by the grid), so what is measured is the ledger-driven
    # source assembly, which is the only channel-dependent DVM cost.
    channel_rows = []
    names = [
        k
        for k in ("puff", "anode_right", "anode_left", "vol_rec",
                  "cathode_face", "collector_face")
        if bg["sources"].get(k, 0.0) > 0
    ]
    full_sources = dict(bg["sources"])
    for k in range(1, len(names) + 1):
        sub = {n: full_sources[n] for n in names[:k]}
        sub["puff_z"] = full_sources.get("puff_z", 0.0)
        t_build = repeat(
            f"  channels={k}: DVM source assembly (per generation)",
            lambda s=sub: assemble_sources(kn, s),
            max(3, rep // 2),
        )
        channel_rows.append((k, names[:k], t_build))

    mem = {
        "state Fc+Fa": 2 * np.prod(shape) * 8,
        "operators M_cx": np.prod(shape) * 8,
        "transient scratch (4 arrays)": 4 * np.prod(shape) * 8,
        "flight engine matrices (nz x nz x 6)": 6 * nz * nz * 8,
    }

    write_dvm(out, args, bg, kn, res, channel_rows, mem, sol)
    return kn, sol, res


def write_dvm(path, args, bg, kn, res, channel_rows, mem, sol):
    L = []
    L.append("E0 kinetic-neutral architecture bench -- DVM arm")
    L.append("=" * 78)
    L.extend(header_lines(args, bg, kn))
    L.append("")
    L.append("MEASUREMENTS (each timed repeatedly; one untimed warm-up first)")
    L.append("-" * 78)
    for r in res:
        L.append(r.line())
    L.append("")
    L.append("SOURCE-CHANNEL COUNT DEPENDENCE")
    L.append("-" * 78)
    L.append("The sweep is channel-blind (its cost is set by the grid and nz),")
    L.append("so the channel-dependent DVM cost is the ledger source assembly:")
    for k, chans, t in channel_rows:
        L.append(t.line())
        L.append(f"{'':48s}channels: {', '.join(chans)}")
    L.append("")
    L.append("MEMORY (analytic, float64)")
    L.append("-" * 78)
    total = 0
    for k, v in mem.items():
        total += v
        L.append(f"{k:<46s} {v / 1e6:12.3f} MB")
    L.append(f"{'TOTAL DVM resident state':<46s} {total / 1e6:12.3f} MB")
    L.append("")
    L.append("STEADY SOLUTION (used by the control-variate estimate)")
    L.append("-" * 78)
    L.append(f"generations to truncate=1e-3: {sol['generations']}")
    L.append(
        f"column density range [cm^-3]: {sol['nn_col'].min():.4g} "
        f"to {sol['nn_col'].max():.4g}"
    )
    L.append(
        f"annulus density range [cm^-3]: {sol['nn_ann'].min():.4g} "
        f"to {sol['nn_ann'].max():.4g}"
    )
    Path(path).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------- MC arm


class PersistentMC:
    """Persistent-population vectorized MC over the frozen background.

    Fixed particle count: histories absorbed by ionization or pumping are
    respawned from the source ledger in the same iteration, so the population
    is stationary and the kernel measures the steady per-event cost of a
    solver-coupled MC rather than a launch-to-extinction test-particle run.

    One ``step()`` resolves exactly one flight segment for every live
    particle: distance to the next z-edge, to the radial wall, to the column
    surface, and to the next null collision; the minimum wins; the event is
    applied. Events per second is therefore ``n_particles / step_wall_time``.
    """

    def __init__(self, bg, n_particles, rng, channels=None):
        self.bg = bg
        self.rng = rng
        self.N = int(n_particles)
        ze = bg["z_edges"]
        self.ze = ze
        self.ncell = ze.size - 1
        self.Rp = bg["Rp"]
        self.Rm = bg["Rm"]
        self.nu_ion = bg["nu_ion"]
        self.nu_cx = bg["nu_cx"]
        self.nu_tot = self.nu_ion + self.nu_cx
        self.nu_max = float(self.nu_tot.max())
        self.mesh_edge = bg["mesh_edge"]
        self.transparency = 1.0 - bg["eta"]
        vbar = np.sqrt(8.0 * KB * T_WALL_K / (np.pi * M_HE))
        A_end = np.pi * self.Rm[-1] ** 2
        self.s_R = bg["S_pump_R"] * 1e3 / (A_end * vbar / 4.0)
        self.s_L = bg["S_pump_L"] * 1e3 / (A_end * vbar / 4.0)
        src = bg["sources"]
        order = ("puff", "anode_right", "anode_left", "vol_rec",
                 "cathode_face", "collector_face")
        avail = [k for k in order if src.get(k, 0.0) > 0]
        self.names = avail if channels is None else avail[:channels]
        if not self.names:
            raise ValueError("no active source channels in the ledger")
        rates = np.array([src[k] for k in self.names])
        self.rates = rates
        self.frac = rates / rates.sum()
        self.w_scale = rates.sum() / self.N  # atoms/s carried per history
        self.V_col = np.pi * self.Rp**2 * np.diff(ze)
        self.V_ann = np.pi * (self.Rm**2 - self.Rp**2) * np.diff(ze)
        self.pos = np.zeros((self.N, 3))
        self.vel = np.zeros((self.N, 3))
        self.wgt = np.full(self.N, self.w_scale)
        self.dt_last = np.zeros(self.N)
        self.icell = np.zeros(self.N, dtype=np.intp)
        self.spawn(np.arange(self.N))
        self.n_events = 0
        self.dt_sum = 0.0
        # On-wall wall-root clamps (see step()); counted so the clamp cannot
        # silently become a bias.
        self.n_wall_clamp = 0

    # -- persistent state footprint, bytes per particle
    @property
    def bytes_per_particle(self):
        return sum(
            a.itemsize * (a.size // self.N)
            for a in (self.pos, self.vel, self.wgt, self.dt_last, self.icell)
        )

    def spawn(self, idx):
        """(Re)launch histories ``idx`` from the source ledger."""
        if idx.size == 0:
            return
        rng = self.rng
        pick = rng.choice(len(self.names), size=idx.size, p=self.frac)
        for k, name in enumerate(self.names):
            sel = idx[pick == k]
            if sel.size == 0:
                continue
            p, v = self._launch(name, sel.size)
            self.pos[sel] = p
            self.vel[sel] = v
            self.wgt[sel] = self.w_scale

    def _launch(self, name, N):
        rng = self.rng
        bg = self.bg
        src = bg["sources"]
        ze, Rp, Rm = self.ze, self.Rp, self.Rm
        pos = np.zeros((N, 3))
        if name == "puff":
            zc = src["puff_z"]
            ic = int(np.searchsorted(ze, zc) - 1)
            th = rng.random(N) * 2 * np.pi
            pos[:, 0] = Rm[ic] * 0.999 * np.cos(th)
            pos[:, 1] = Rm[ic] * 0.999 * np.sin(th)
            pos[:, 2] = zc
            vel = wall_emit_inward(rng, pos[:, 0], pos[:, 1], T_WALL_K)
        elif name in ("cathode_face", "collector_face"):
            at_start = name == "cathode_face"
            rad = (bg["R_cath"] if at_start else Rp[-1]) * np.sqrt(rng.random(N))
            th = rng.random(N) * 2 * np.pi
            pos[:, 0] = rad * np.cos(th)
            pos[:, 1] = rad * np.sin(th)
            pos[:, 2] = 1e-6 if at_start else ze[-1] - 1e-6
            sign = 1.0 if at_start else -1.0
            vel = cosine_emit(
                rng, N, bg["T_s"] if at_start else T_WALL_K, sign
            )
        elif name == "vol_rec":
            w = bg["rec_cell"] / bg["rec_cell"].sum()
            ic = rng.choice(w.size, size=N, p=w)
            rad = Rp[ic] * np.sqrt(rng.random(N))
            th = rng.random(N) * 2 * np.pi
            pos[:, 0] = rad * np.cos(th)
            pos[:, 1] = rad * np.sin(th)
            pos[:, 2] = ze[ic] + rng.random(N) * (ze[ic + 1] - ze[ic])
            vel = maxwellian(rng, N, bg["Ti"][ic], bg["u"][ic])
        elif name in ("anode_left", "anode_right"):
            left = name == "anode_left"
            ic = self.mesh_edge - 1 if left else self.mesh_edge
            rad = Rp[ic] * np.sqrt(rng.random(N))
            th = rng.random(N) * 2 * np.pi
            pos[:, 0] = rad * np.cos(th)
            pos[:, 1] = rad * np.sin(th)
            pos[:, 2] = ze[self.mesh_edge] + (-1e-6 if left else 1e-6)
            vel = cosine_emit(rng, N, T_WALL_K, -1.0 if left else 1.0)
        else:
            raise ValueError(name)
        return pos, vel

    def step(self):
        """Advance every particle by one flight segment and resolve its event."""
        rng = self.rng
        pos, vel = self.pos, self.vel
        ze, Rp, Rm, ncell = self.ze, self.Rp, self.Rm, self.ncell
        N = self.N
        speed = np.maximum(np.linalg.norm(vel, axis=1), 1.0)
        icell = np.clip(np.searchsorted(ze, pos[:, 2]) - 1, 0, ncell - 1)
        self.icell = icell
        with np.errstate(divide="ignore", invalid="ignore"):
            d_z = np.where(
                vel[:, 2] > 0,
                (ze[icell + 1] - pos[:, 2]) / vel[:, 2],
                np.where(vel[:, 2] < 0, (ze[icell] - pos[:, 2]) / vel[:, 2],
                         np.inf),
            ) * speed
        vxy2 = vel[:, 0] ** 2 + vel[:, 1] ** 2
        b = pos[:, 0] * vel[:, 0] + pos[:, 1] * vel[:, 1]
        r2 = pos[:, 0] ** 2 + pos[:, 1] ** 2
        Rw = Rm[icell]
        disc = b**2 + vxy2 * (Rw**2 - r2)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_wall = (-b + np.sqrt(np.maximum(disc, 0.0))) / np.where(
                vxy2 > 0, vxy2, np.inf
            )
        d_wall = np.where(vxy2 > 0, t_wall * speed, np.inf)
        # On-wall degenerate, mirroring mc_neutrals.run_mc (which carries the
        # full derivation and the tripwire this bench does not): an event that
        # ends a segment within the ray overshoot of the wall is advanced
        # THROUGH it whenever the wall handler below is skipped, and the ray
        # is then at most RAY_EPS_CM outside its own cell's Rm, where both
        # wall roots go negative. Such a ray is ON the wall -- clamp the root
        # to zero so the wall handler takes it. Gating on the RADIAL excess
        # keeps a genuine escape (cm and beyond) negative.
        on_wall = (r2 > Rw**2) & ((np.sqrt(r2) - Rw) <= RAY_EPS_CM)
        clamp = on_wall & (d_wall < 0.0)
        if clamp.any():
            self.n_wall_clamp += int(clamp.sum())
            d_wall = np.where(clamp, 0.0, d_wall)
        Rp_here = Rp[icell]
        disc_p = b**2 + vxy2 * (Rp_here**2 - r2)
        sq_p = np.sqrt(np.maximum(disc_p, 0.0))
        inside = r2 < Rp_here**2
        with np.errstate(divide="ignore", invalid="ignore"):
            t_exit = (-b + sq_p) / np.where(vxy2 > 0, vxy2, np.inf)
            t_enter = (-b - sq_p) / np.where(vxy2 > 0, vxy2, np.inf)
        t_rp = np.where(inside, t_exit, np.where(t_enter > 0, t_enter, np.inf))
        d_rp = np.where(
            (vxy2 > 0) & (disc_p > 0) & (t_rp > 1e-12), t_rp * speed, np.inf
        )
        d_coll = -np.log(rng.random(N)) * speed / self.nu_max
        d = np.minimum(np.minimum(d_z, d_wall), np.minimum(d_coll, d_rp))
        d = np.minimum(d, 1e6)
        dt = d / speed
        self.dt_last = dt
        self.r2 = r2
        pos += vel * dt[:, None]
        pos += (vel / speed[:, None]) * RAY_EPS_CM
        dead = np.zeros(N, dtype=bool)

        hit_c = d_coll <= np.minimum(np.minimum(d_z, d_wall), d_rp)
        if hit_c.any():
            ic = icell[hit_c]
            real = rng.random(int(hit_c.sum())) < (
                self.nu_tot[ic] / self.nu_max
            ) * (r2[hit_c] < Rp[ic] ** 2)
            idx = np.flatnonzero(hit_c)[real]
            if idx.size:
                ii = icell[idx]
                ionz = rng.random(idx.size) < self.nu_ion[ii] / self.nu_tot[ii]
                dead[idx[ionz]] = True
                cx = idx[~ionz]
                if cx.size:
                    jj = icell[cx]
                    vel[cx] = maxwellian(
                        rng, cx.size, self.bg["Ti"][jj], self.bg["u"][jj]
                    )

        hit_w = (~hit_c) & (d_wall <= np.minimum(d_z, d_rp))
        self.hit_wall = hit_w
        if hit_w.any():
            idx = np.flatnonzero(hit_w)
            r_now = np.sqrt(pos[idx, 0] ** 2 + pos[idx, 1] ** 2)
            shrink = (Rm[icell[idx]] * 0.9999) / np.maximum(r_now, 1e-9)
            pos[idx, 0] *= shrink
            pos[idx, 1] *= shrink
            vel[idx] = wall_emit_inward(rng, pos[idx, 0], pos[idx, 1], None)

        hit_z = (~hit_c) & (~hit_w) & (d_z <= d_rp)
        self.hit_end = np.zeros(N, dtype=bool)
        if hit_z.any():
            idx = np.flatnonzero(hit_z)
            zdir = np.sign(vel[idx, 2])
            edge = np.where(zdir > 0, icell[idx] + 1, icell[idx])
            for at_end, sign, s_stick, T_emit in (
                (edge == 0, 1.0, self.s_L, self.bg["T_s"]),
                (edge == ncell, -1.0, self.s_R, T_WALL_K),
            ):
                eidx = idx[at_end]
                if eidx.size == 0:
                    continue
                self.hit_end[eidx] = True
                stick = rng.random(eidx.size) < s_stick
                dead[eidx[stick]] = True
                keep = eidx[~stick]
                if keep.size:
                    vel[keep] = cosine_emit(rng, keep.size, T_emit, sign)
                    pos[keep, 2] = np.clip(pos[keep, 2], 1e-6, ze[-1] - 1e-6)
            midx = idx[(edge == self.mesh_edge) & (edge != 0) & (edge != ncell)]
            if midx.size:
                blocked = rng.random(midx.size) > self.transparency
                bidx = midx[blocked]
                if bidx.size:
                    sign = -np.sign(vel[bidx, 2])
                    vel[bidx] = cosine_emit(rng, bidx.size, T_WALL_K, sign)
                    pos[bidx, 2] = ze[self.mesh_edge] + sign * 1e-6
            # annular step face: where Rm narrows across an interior z-edge,
            # the part of the crossing plane with Rm(dest) < r <= Rm(src) is
            # vessel wall, not an opening. Unhandled, the ray passes THROUGH
            # the wall and thereafter sits at r > Rm of its own cell, where
            # both wall-intersection roots go negative and the flight length
            # turns negative -- the transport diverges instead of degrading.
            interior = (edge > 0) & (edge < ncell)
            e = idx[interior]
            if e.size:
                zdir_i = zdir[interior]
                dest = np.where(zdir_i > 0, edge[interior], edge[interior] - 1)
                # r_step, not r_e: harmless in this method (no closures), but
                # mc_neutrals' mirror of this block has an r_e PARAMETER its
                # launch() closure reads, so the two stay identical.
                r_step = np.sqrt(pos[e, 0] ** 2 + pos[e, 1] ** 2)
                step = r_step > Rm[dest]
                h = e[step]
                if h.size:
                    sgn = -zdir_i[step]
                    vel[h] = cosine_emit(rng, h.size, T_WALL_K, sgn)
                    pos[h, 2] += sgn * 1e-6
        self.spawn(np.flatnonzero(dead))
        self.n_events += N
        self.dt_sum += float(dt.sum())
        return N

    # -- tallies, timed separately from transport

    def tally_macro3(self, acc):
        """Three-moment (n, n*u_z, n*E) track-length deposition, per zone.

        The macro-plus-correction tally: exactly the moments a fluid macro
        solver would need handed back from the particle correction. Uses
        ``np.add.at``, the convention of the existing TPMC instrument.
        """
        w_dt = self.wgt * self.dt_last
        in_col = self.r2 < self.Rp[self.icell] ** 2
        zone = np.where(in_col, 0, 1)
        vz = self.vel[:, 2]
        v2 = (self.vel * self.vel).sum(axis=1)
        np.add.at(acc[0], (self.icell, zone), w_dt)
        np.add.at(acc[1], (self.icell, zone), w_dt * vz)
        np.add.at(acc[2], (self.icell, zone), w_dt * v2)

    def tally_macro3_bincount(self, acc):
        """Same three moments via ``np.bincount`` on a flattened index.

        Measured separately because ``np.add.at`` is an unbuffered ufunc
        path and is the slower of the two by a wide margin; quoting only
        one of them would misstate the architecture's tally floor.
        """
        w_dt = self.wgt * self.dt_last
        in_col = self.r2 < self.Rp[self.icell] ** 2
        flat = self.icell * 2 + (~in_col)
        m = 2 * self.ncell
        vz = self.vel[:, 2]
        v2 = (self.vel * self.vel).sum(axis=1)
        acc[0] += np.bincount(flat, weights=w_dt, minlength=m).reshape(-1, 2)
        acc[1] += np.bincount(
            flat, weights=w_dt * vz, minlength=m
        ).reshape(-1, 2)
        acc[2] += np.bincount(
            flat, weights=w_dt * v2, minlength=m
        ).reshape(-1, 2)

    def tally_wall_energy(self, acc):
        """Kinetic energy delivered to the radial wall and the end walls."""
        hit = self.hit_wall | self.hit_end
        if not hit.any():
            return
        idx = np.flatnonzero(hit)
        e = 0.5 * M_HE * (self.vel[idx] * self.vel[idx]).sum(axis=1)
        np.add.at(acc, self.icell[idx], self.wgt[idx] * e)


def mc_memory_probe(args):
    """Measure peak RSS vs population size in clean subprocesses.

    In-process peak RSS is useless here (it is a high-water mark already set
    by the DVM arm and the HDF5 read), so each point is a fresh interpreter
    that builds one population, advances it two iterations, and reports its
    own peak. A least-squares line through the points separates the
    interpreter floor from the per-particle cost INCLUDING the batch
    kernel's transient scratch, which is the number that sizes the
    architecture.
    """
    script = str(Path(__file__).resolve())
    points = []
    for N in (args.mc_particles, 4 * args.mc_particles, 12 * args.mc_particles):
        cmd = [sys.executable, script, "--mem-probe", str(N),
               "--run", args.run, "--window", str(args.window[0]),
               str(args.window[1]), "--seed", str(args.seed)]
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            env={**os.environ},
        )
        points.append((N, int(out.stdout.strip().split()[-1])))
    Ns = np.array([p[0] for p in points], dtype=float)
    Bs = np.array([p[1] for p in points], dtype=float)
    slope, intercept = np.polyfit(Ns, Bs, 1)
    return points, slope, intercept


def mc_arm(bg, args, out):
    rep = args.repeats
    res = []
    rng = np.random.default_rng(args.seed)

    res.append(
        repeat(
            "construct: MC population (alloc + ledger launch)",
            lambda: PersistentMC(bg, args.mc_particles,
                                 np.random.default_rng(args.seed)),
            max(3, rep // 2),
            note=f"N={args.mc_particles}",
        )
    )
    mc = PersistentMC(bg, args.mc_particles, rng)

    # settle the population before timing steady-state event throughput, then
    # reset the event counters so the flight statistics describe the settled
    # population rather than the launch transient
    for _ in range(args.mc_warm):
        mc.step()
    mc.n_events = 0
    mc.dt_sum = 0.0

    t_step = repeat(
        "transport: one persistent-MC event iteration (all particles)",
        mc.step,
        rep,
        note=f"1 event per particle per call, N={mc.N}",
    )
    res.append(t_step)
    ev_per_s = mc.N / t_step.median
    res.append(
        Timed(
            "throughput: persistent-MC events per second",
            [mc.N / s for s in t_step.samples],
            unit="ev/s",
            note="derived from the iteration timings above",
        )
    )
    mean_dt = mc.dt_sum / max(mc.n_events, 1)

    acc3 = np.zeros((3, mc.ncell, 2))
    res.append(
        repeat(
            "tally: macro-plus-correction 3-moment (np.add.at)",
            lambda: mc.tally_macro3(acc3),
            rep,
            note="n, n*u_z, n*v^2 per cell per zone",
        )
    )
    acc3b = np.zeros((3, mc.ncell, 2))
    res.append(
        repeat(
            "tally: macro-plus-correction 3-moment (np.bincount)",
            lambda: mc.tally_macro3_bincount(acc3b),
            rep,
            note="same three moments, buffered index path",
        )
    )
    accw = np.zeros(mc.ncell)
    res.append(
        repeat(
            "tally: wall-energy deposition (radial + end walls)",
            lambda: mc.tally_wall_energy(accw),
            rep,
            note=f"{int((mc.hit_wall | mc.hit_end).sum())} wall hits in the "
                 "timed batch",
        )
    )

    # source-channel count dependence
    n_avail = len(
        [k for k in ("puff", "anode_right", "anode_left", "vol_rec",
                     "cathode_face", "collector_face")
         if bg["sources"].get(k, 0.0) > 0]
    )
    channel_rows = []
    for k in range(1, n_avail + 1):
        mck = PersistentMC(bg, args.mc_particles,
                           np.random.default_rng(args.seed), channels=k)
        for _ in range(max(2, args.mc_warm // 4)):
            mck.step()
        tk = repeat(
            f"  channels={k}: one persistent-MC event iteration",
            mck.step,
            max(3, rep // 2),
        )
        ck = repeat(
            f"  channels={k}: population construct (alloc + launch)",
            lambda kk=k: PersistentMC(bg, args.mc_particles,
                                      np.random.default_rng(args.seed),
                                      channels=kk),
            3,
        )
        channel_rows.append((k, list(mck.names), tk, ck))

    mem_points, mem_slope, mem_floor = mc_memory_probe(args)

    write_mc(out, args, bg, mc, res, channel_rows,
             (mem_points, mem_slope, mem_floor), mean_dt, ev_per_s)
    return mc, res, mean_dt, ev_per_s, mem_slope


def write_mc(path, args, bg, mc, res, channel_rows, mem, mean_dt,
             ev_per_s):
    mem_points, mem_slope, mem_floor = mem
    L = []
    L.append("E0 kinetic-neutral architecture bench -- persistent-MC arm")
    L.append("=" * 78)
    L.extend(header_lines(args, bg, None))
    L.append(f"MC kernel                : pure-numpy batch-synchronous, "
             f"persistent population")
    L.append(f"MC particles             : {mc.N}")
    L.append(f"MC warm-up iterations    : {args.mc_warm}")
    L.append(f"active source channels   : {', '.join(mc.names)}")
    L.append("")
    L.append("MEASUREMENTS (each timed repeatedly; one untimed warm-up first)")
    L.append("-" * 78)
    for r in res:
        L.append(r.line())
    L.append("")
    L.append("EVENT STATISTICS")
    L.append("-" * 78)
    L.append(f"events resolved in the timed run : {mc.n_events}")
    L.append(f"mean flight-segment duration     : {mean_dt:.6g} s")
    L.append(f"event rate per particle          : {1.0 / mean_dt:.6g} 1/s")
    L.append(f"events per second (median)       : {ev_per_s:.6g}")
    L.append("")
    L.append("MEMORY")
    L.append("-" * 78)
    bpp = mc.bytes_per_particle
    L.append(f"persistent state per particle    : {bpp} B "
             "(pos3 + vel3 + weight + dt + cell index)")
    L.append(f"analytic population footprint    : "
             f"{bpp * mc.N / 1e6:.3f} MB at N={mc.N}")
    L.append("")
    L.append("peak RSS vs N, one clean subprocess per point:")
    for N, b in mem_points:
        L.append(f"  N={N:>9,d}   peak RSS {b / 1e6:10.3f} MB")
    L.append(f"least-squares slope              : {mem_slope:.1f} B/particle "
             "(persistent state + batch-kernel transient scratch)")
    L.append(f"least-squares intercept          : {mem_floor / 1e6:.1f} MB "
             "(interpreter + numpy/h5py + frozen background)")
    L.append("")
    for n in (1e4, 1e5, 1e6, 1e7):
        L.append(f"  projected at N={n:.0e}: persistent "
                 f"{bpp * n / 1e6:10.3f} MB, measured-slope total "
                 f"{mem_slope * n / 1e6:10.3f} MB")
    L.append("")
    L.append("SOURCE-CHANNEL COUNT DEPENDENCE")
    L.append("-" * 78)
    for k, chans, tk, ck in channel_rows:
        L.append(tk.line())
        L.append(ck.line())
        L.append(f"{'':48s}channels: {', '.join(chans)}")
    Path(path).write_text("\n".join(L) + "\n")


# --------------------------------------------- three-moment control variate


def control_variate_estimate(kn, sol):
    """Deviational mass fraction of the converged DVM distribution.

    A micro-macro scheme carries a macro state (n, u, T) and represents only
    the remainder ``df = f - M[n,u,T]`` with particles. The per-cell
    deviational mass fraction

        alpha = sum |f - M| / sum f

    is what sets how many correction particles are needed: an estimator
    carrying absolute deviational mass ``alpha * n`` has standard error
    ``alpha * n / sqrt(N_corr)``, so matching a direct-MC relative error
    ``eps`` on the density needs ``N_corr ~ (alpha / eps)^2`` -- i.e.
    ``alpha^2`` times the direct-MC population. Returns per-cell alpha for
    both zones.
    """
    g = kn.g
    out = {}
    for key, F in (("col", sol["Fc"]), ("ann", sol["Fa"])):
        n = F.sum(axis=(1, 2))
        alpha = np.zeros(kn.nz)
        T_par = np.zeros(kn.nz)
        T_perp = np.zeros(kn.nz)
        for i in range(kn.nz):
            if n[i] <= 0:
                continue
            f = F[i]
            u = float((f * g.VZ).sum() / n[i])
            e = float((f * ((g.VZ - u) ** 2 + g.VP**2)).sum() / n[i])
            T_eV = max(M_HE * (e / 3.0) / EV, 1e-6)
            M = n[i] * g.maxwellian(T_eV, u)
            alpha[i] = float(np.abs(f - M).sum() / n[i])
            T_par[i] = M_HE * float((f * (g.VZ - u) ** 2).sum() / n[i]) / EV
            T_perp[i] = (
                M_HE * float((f * g.VP**2).sum() / n[i]) / (2.0 * EV)
            )
        out[key] = alpha
        out[key + "_Tpar"] = T_par
        out[key + "_Tperp"] = T_perp
    return out


# ------------------------------------------------------------------ summary


def header_lines(args, bg, kn):
    nz = bg["z_edges"].size - 1
    L = [
        f"background               : {args.run}",
        f"plateau window [ms]      : {args.window[0]} to {args.window[1]}",
        f"axial cells (nz)         : {nz}",
        f"velocity grid            : nvz={args.nvz} x nvp={args.nvp} "
        f"= {args.nvz * args.nvp} bins",
        f"seed                     : {args.seed}",
        f"repeats                  : {args.repeats}",
        f"machine                  : {platform.platform()}",
        f"python                   : {platform.python_version()}, "
        f"numpy {np.__version__}",
        f"concurrent load          : {args.load_note}",
        f"command                  : {args.cmdline}",
    ]
    if kn is not None:
        L.append(f"vmax [cm/s]              : {kn.g.vz.max():.4g}")
    return L


def write_summary(path, args, bg, dvm_res, mc_res, mean_dt, ev_per_s, alpha,
                  mc, mem_slope):
    def get(res, key):
        for r in res:
            if r.name.startswith(key):
                return r
        raise KeyError(key)

    t_transient = get(dvm_res, "update: transient DVM step")
    t_sweep = get(dvm_res, "update: DVM generation sweep")
    t_solve = get(dvm_res, "solve: DVM steady")
    t_ctor_dvm = get(dvm_res, "construct: DVM operators KN2Zone")
    t_ctor_fast = get(dvm_res, "construct: compiled flight engine")
    t_step = get(mc_res, "transport: one persistent-MC event iteration")
    t_m3 = get(mc_res, "tally: macro-plus-correction 3-moment (np.add.at)")
    t_m3b = get(mc_res, "tally: macro-plus-correction 3-moment (np.bincount)")
    t_we = get(mc_res, "tally: wall-energy")
    t_ctor_mc = get(mc_res, "construct: MC population")

    a_col = alpha["col"]
    a_ann = alpha["ann"]
    zc = 0.5 * (bg["z_edges"][:-1] + bg["z_edges"][1:])
    mid = (zc >= 500.0) & (zc <= 1000.0)
    nz = bg["z_edges"].size - 1

    L = []
    L.append("# E0 -- kinetic neutral architecture cost bench (numbers only)")
    L.append("")
    L.append(f"Background `{Path(args.run).name}`, plateau window "
             f"{args.window[0]}-{args.window[1]} ms, nz="
             f"{bg['z_edges'].size - 1}, velocity grid "
             f"{args.nvz}x{args.nvp}. Seed {args.seed}. "
             f"{args.repeats} repeats per measurement.")
    L.append("")
    L.append(f"Machine: {platform.platform()}, {os.cpu_count()} logical "
             f"cores; Python {platform.python_version()}, numpy "
             f"{np.__version__}. Concurrent load: {args.load_note}.")
    L.append("")
    L.append("Full command (reruns end to end):")
    L.append("")
    L.append("```")
    L.append(args.cmdline)
    L.append("```")
    L.append("")
    L.append("## 1. Measured costs")
    L.append("")
    L.append("| quantity | median | min | max | spread |")
    L.append("|---|---|---|---|---|")
    for r in (t_ctor_dvm, t_ctor_fast, t_sweep, t_transient, t_solve,
              t_ctor_mc, t_step, t_m3, t_m3b, t_we):
        L.append(
            f"| {r.name.strip()} | {r.median:.6g} {r.unit} | "
            f"{r.lo:.6g} | {r.hi:.6g} | {100 * r.spread:.1f}% |"
        )
    L.append("")
    L.append(f"Persistent-MC throughput: **{ev_per_s:.4g} events/s** at "
             f"N={mc.N} (one event = one resolved flight segment).")
    L.append(f"Persistent state: **{mc.bytes_per_particle} B/particle** "
             f"= {mc.bytes_per_particle * mc.N / 1e6:.3f} MB at N={mc.N}; "
             f"measured peak-RSS slope over clean subprocesses "
             f"**{mem_slope:.0f} B/particle** (persistent state plus the "
             f"batch kernel's transient scratch).")
    L.append(f"Mean flight-segment duration: **{mean_dt:.4g} s**, i.e. "
             f"**{1.0 / mean_dt:.4g} events per particle per simulated "
             f"second**.")
    L.append("")
    L.append("## 2. Three-moment control-variate correction population")
    L.append("")
    L.append("Deviational mass fraction `alpha = sum|f - M[n,u,T]| / sum f` "
             "measured on the converged DVM distribution:")
    L.append("")
    L.append("| zone | mid-machine mean (z 500-1000 cm) | domain max | "
             "domain min |")
    L.append("|---|---|---|---|")
    L.append(f"| column | {a_col[mid].mean():.4f} | {a_col.max():.4f} | "
             f"{a_col.min():.4f} |")
    L.append(f"| annulus | {a_ann[mid].mean():.4f} | {a_ann.max():.4f} | "
             f"{a_ann.min():.4f} |")
    L.append("")
    L.append("`alpha` is bounded by 2 (disjoint `f` and `M`). Sampled "
             "profile, with the measured temperature anisotropy of the same "
             "cells (`T_par` from `(v_z - u)^2`, `T_perp` from `v_perp^2/2`; "
             "300 K = 0.02585 eV):")
    L.append("")
    L.append("| z [cm] | alpha col | T_par col [eV] | T_perp col [eV] | "
             "alpha ann | T_par ann [eV] | T_perp ann [eV] |")
    L.append("|---|---|---|---|---|---|---|")
    for i in range(0, nz, max(1, nz // 10)):
        L.append(
            f"| {zc[i]:.0f} | {a_col[i]:.3f} | {alpha['col_Tpar'][i]:.4f} | "
            f"{alpha['col_Tperp'][i]:.4f} | {a_ann[i]:.3f} | "
            f"{alpha['ann_Tpar'][i]:.4f} | {alpha['ann_Tperp'][i]:.4f} |"
        )
    L.append("")
    L.append("Arithmetic (labelled as such, not a measurement): a deviational "
             "estimator carrying absolute deviational mass `alpha*n` has "
             "standard error `alpha*n/sqrt(N_corr)`, so a target per-cell "
             "relative error `eps` needs `N_corr = (alpha/eps)^2` correction "
             "particles per cell, against `N_direct = 1/eps^2` for a "
             "full-particle representation of the same cell -- a factor "
             "`alpha^2`.")
    L.append("")
    L.append("| eps (per-cell rel. error) | N_direct/cell | N_corr/cell, "
             "column (mid / max alpha) | N_corr/cell, annulus "
             "(mid / max alpha) |")
    L.append("|---|---|---|---|")
    for eps in (0.01, 0.02, 0.05):
        L.append(
            f"| {eps:.0%} | {1 / eps**2:,.0f} | "
            f"{(a_col[mid].mean() / eps) ** 2:,.0f} / "
            f"{(a_col.max() / eps) ** 2:,.0f} | "
            f"{(a_ann[mid].mean() / eps) ** 2:,.0f} / "
            f"{(a_ann.max() / eps) ** 2:,.0f} |"
        )
    L.append("")

    def n_corr_total(eps):
        """Exact per-cell sum over both zones of (alpha/eps)^2."""
        return float(((a_col / eps) ** 2).sum() + ((a_ann / eps) ** 2).sum())

    L.append(f"Domain totals summed cell by cell over both zones "
             f"({nz} cells x 2):")
    L.append("")
    L.append("| eps | N_direct (domain) | N_corr (domain) | ratio |")
    L.append("|---|---|---|---|")
    for eps in (0.01, 0.02, 0.05):
        nd = 2 * nz / eps**2
        ncv = n_corr_total(eps)
        L.append(f"| {eps:.0%} | {nd:,.0f} | {ncv:,.0f} | {ncv / nd:.2f} |")
    L.append("")
    L.append("Caveat on the counting, stated because it bounds the numbers "
             "above: these are INDEPENDENT samples per cell. A persistent "
             "population's successive events on the same particle are "
             "correlated, so the live-population requirement is bounded "
             "BELOW by these figures; the effective-sample-size factor is "
             "not measured here (it is an accuracy/cadence question, i.e. "
             "E1). The alpha^2 variance ratio between the two "
             "representations is unaffected by that factor.")
    L.append("")
    L.append("## 3. Projection against the E4 production gate")
    L.append("")
    L.append("**These are projections, not measurements**: the measured "
             "per-unit costs above multiplied by production-run scale "
             "factors quoted from `es1_prod_profile_nx240_ANALYSIS.md` "
             f"(2026-07-30): {PROD_STEPS:,} accepted steps, "
             f"{PROD_SIM_TIME_S * 1e3:.2f} ms simulated, {PROD_WALL_S:.0f} s "
             "unprofiled reference wall time. No cadence or accuracy trade "
             "is applied (that is E1).")
    L.append("")
    L.append(f"E4 bands: target {E4_TARGET_S[0]:.0f}-{E4_TARGET_S[1]:.0f} s "
             f"added; {E4_TRADE_S[0]:.0f}-{E4_TRADE_S[1]:.0f} s = explicit "
             f"trade decision; >{E4_TRADE_S[1]:.0f} s = stop.")
    L.append("")

    def band(x):
        if x < E4_TARGET_S[0]:
            return "below target band"
        if x <= E4_TARGET_S[1]:
            return "TARGET"
        if x <= E4_TRADE_S[1]:
            return "TRADE"
        return "STOP"

    L.append("### 3a. DVM, transient (one update per solver step)")
    L.append("")
    L.append(f"`added_s = {t_transient.median:.6g} s/update x N_updates`")
    L.append("")
    L.append("| N_updates | added wall [s] | E4 band |")
    L.append("|---|---|---|")
    for n in (PROD_STEPS, PROD_STEPS // 10, PROD_STEPS // 100,
              PROD_STEPS // 1000):
        x = t_transient.median * n
        L.append(f"| {n:,} | {x:,.1f} | {band(x)} |")
    for lim, lab in ((E4_TARGET_S[1], "400 s"), (E4_TRADE_S[1], "699 s")):
        L.append(f"| **{lim / t_transient.median:,.0f}** (budget at {lab}) | "
                 f"{lim:,.0f} | boundary |")
    L.append("")
    L.append("### 3b. DVM, quasi-static refresh (one steady solve per refresh)")
    L.append("")
    L.append(f"`added_s = {t_solve.median:.6g} s/solve x N_refresh` "
             f"(plus a one-off {t_ctor_fast.median:.4g} s flight-engine "
             f"compile if the jump kernel is used)")
    L.append("")
    L.append("| N_refresh | added wall [s] | E4 band |")
    L.append("|---|---|---|")
    for n in (10, 50, 100, 200, 500):
        x = t_solve.median * n
        L.append(f"| {n} | {x:,.1f} | {band(x)} |")
    for lim, lab in ((E4_TARGET_S[1], "400 s"), (E4_TRADE_S[1], "699 s")):
        L.append(f"| **{lim / t_solve.median:,.1f}** (budget at {lab}) | "
                 f"{lim:,.0f} | boundary |")
    L.append("")
    L.append("### 3c. Persistent MC (micro-macro correction, and full-particle)")
    L.append("")
    per_ev = 1.0 / ev_per_s
    tally_per_ev = (min(t_m3.median, t_m3b.median) + t_we.median) / mc.N
    L.append(f"`added_s = N_part x (1/{mean_dt:.4g} s) x "
             f"{PROD_SIM_TIME_S:.5g} s x ({per_ev:.4g} + {tally_per_ev:.4g}) "
             f"s/event`, i.e. `N_part x "
             f"{(1.0 / mean_dt) * PROD_SIM_TIME_S * (per_ev + tally_per_ev):.6g}"
             f" s` -- transport plus both tallies, every event tallied, "
             f"taking the faster of the two 3-moment deposition paths.")
    L.append("")
    L.append("| N_part | added wall [s] | memory [MB, measured slope] | "
             "E4 band |")
    L.append("|---|---|---|---|")
    coef = (1.0 / mean_dt) * PROD_SIM_TIME_S * (per_ev + tally_per_ev)
    for n in (1e3, 1e4, 3e4, 1e5, 3e5, 1e6):
        x = coef * n
        L.append(f"| {n:,.0f} | {x:,.1f} | "
                 f"{mem_slope * n / 1e6:.1f} | {band(x)} |")
    for lim, lab in ((E4_TARGET_S[1], "400 s"), (E4_TRADE_S[1], "699 s")):
        L.append(f"| **{lim / coef:,.0f}** (budget at {lab}) | {lim:,.0f} | "
                 f"{mem_slope * (lim / coef) / 1e6:.1f} | "
                 f"boundary |")
    L.append("")
    L.append("Mapping the section-2 populations onto that coefficient "
             "(eps = 2% per cell, whole domain):")
    L.append("")
    L.append("| representation | N_part | added wall [s] | E4 band |")
    L.append("|---|---|---|---|")
    for lab, npart in (
        ("full-particle reference (N_direct)", 2 * nz / 0.02**2),
        ("micro-macro, 3-moment control variate (N_corr)",
         n_corr_total(0.02)),
    ):
        x = coef * npart
        L.append(f"| {lab} | {npart:,.0f} | {x:,.1f} | {band(x)} |")
    L.append("")
    L.append("## 4. Source-channel count dependence")
    L.append("")
    L.append("Per-arm tables are in `neutral_arch_e0_dvm_nx240.txt` and "
             "`neutral_arch_e0_mc_nx240.txt`. The ledger of this background "
             f"carries {len(mc.names)} channels with nonzero rate "
             f"({', '.join(mc.names)}).")
    L.append("")
    Path(path).write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E0 frozen-background cost bench for the kinetic "
                    "neutral architecture decision."
    )
    ap.add_argument(
        "--run",
        default=str(Path(__file__).resolve().parent
                    / "es1_kn2z_promoted_nx240.h5"),
        help="saved nx=240 production background (read in place)",
    )
    ap.add_argument("--window", nargs=2, type=float, default=(5.0, 19.5),
                    help="plateau-average window [ms]")
    ap.add_argument("--nvz", type=int, default=48)
    ap.add_argument("--nvp", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--mc-particles", type=int, default=200_000)
    ap.add_argument("--mc-warm", type=int, default=400,
                    help="untimed event iterations before throughput timing")
    ap.add_argument("--out-dir", default="scripts")
    ap.add_argument("--tag", default="nx240")
    ap.add_argument("--mem-probe", type=int, default=0,
                    help="internal: build one population of this size, step "
                         "it twice, print peak RSS in bytes, and exit")
    args = ap.parse_args(argv)

    if args.mem_probe:
        bg = load_background(args.run, tuple(args.window))
        m = PersistentMC(bg, args.mem_probe, np.random.default_rng(args.seed))
        m.step()
        m.step()
        print(f"peak_rss_bytes {rss_bytes()}")
        return 0

    args.cmdline = " ".join(
        [f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}",
         sys.executable, str(Path(sys.argv[0]))] + list(sys.argv[1:])
    )
    args.load_note = machine_note()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dvm_path = out_dir / f"neutral_arch_e0_dvm_{args.tag}.txt"
    mc_path = out_dir / f"neutral_arch_e0_mc_{args.tag}.txt"
    sum_path = out_dir / "neutral_arch_e0_summary.md"

    print(f"E0 bench: {args.load_note}")
    bg = load_background(args.run, tuple(args.window))
    print(f"background loaded: nz={bg['z_edges'].size - 1}, "
          f"channels={[k for k, v in bg['sources'].items() if k != 'puff_z' and v > 0]}")

    print("DVM arm ...")
    kn, sol, dvm_res = dvm_arm(bg, args, dvm_path)
    print(f"  wrote {dvm_path}")

    print("persistent-MC arm ...")
    mc, mc_res, mean_dt, ev_per_s, mem_slope = mc_arm(bg, args, mc_path)
    print(f"  wrote {mc_path}")

    print("three-moment control-variate estimate ...")
    alpha = control_variate_estimate(kn, sol)

    write_summary(sum_path, args, bg, dvm_res, mc_res, mean_dt, ev_per_s,
                  alpha, mc, mem_slope)
    print(f"  wrote {sum_path}")
    print(f"final load: {machine_note()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
