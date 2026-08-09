"""Frozen-field test-particle Monte Carlo for LAPD neutrals (TPMC).

Adjudicates the solver's neutral closures against a kinetic reference on the
SAME plasma background: reads a saved sim1d run, plateau-averages its plasma
fields, and transports test atoms through them with the model's own atomic
data (ADAS SCD ionization, the CX table) -- so any disagreement with the
solver's nn / u_n is closure error, not input error.

Physics: axisymmetric cylinder r < Rm(z), z in [0, Lm]; plasma column
r < Rp(z) carries the 1D fields (n, Te, Ti, u_i). Free-molecular neutrals
(no neutral-neutral collisions -- Kn >~ 1): free flight + null-collision
events. Events: electron-impact ionization (absorb), resonant CX (resample
velocity from the local ion Maxwellian + drift: the relay). Boundaries:
diffuse 300 K re-emission at the radial wall and collector; the anode mesh
plane intercepts with probability 1 - T (T = 1 - eta) and re-emits on the
incident side; the cathode disc re-emits either thermally (the solver's
at-rest convention) or as the directed jet (--jet); end pumps are sticking
probabilities s = S_pump / (A vbar / 4). Sources and their absolute rates
come from the run's own ledger (puff cell, cathode/collector faces, anode
mesh), so tallies are absolute densities.

Track-length estimators per z-cell, split column / annulus: nn and mean
axial drift, i.e. exactly the quantities the two-zone closure and the M_n
wind claim to predict.

Simplifications (documented): elastic (Langevin) scattering folded into CX
resampling is omitted -- CX dominates momentum transfer for He+/He; radial
plasma profile is the 1D model's own top-hat; no plenum volume behind the
cathode plane (its pump becomes a z=0 annulus sticking coefficient).

Usage:
    python scripts/mc_neutrals.py RUN.h5 [-n 200000] [--jet {none,cathode,both}]
        [--window 5 19.5] [--seed 1] [--out PREFIX]
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cablp.funcs._adas import he_rates
from cablp.funcs._cross import charge_ex_react
from cablp.solvers._sim1d.core.geometry import (
    absorbing_live_cells_by_role,
    build_geometry,
)

EV = 1.602176634e-12
KB = 1.380649e-16
M_HE = 4.002602 * 1.66053907e-24
E_CHARGE = 1.602176634e-19
T_WALL_K = 300.0

# Ray overshoot [cm]. After every segment the ray is advanced this far along
# its own direction so that no boundary (a z-edge, the Rp surface, the vessel
# wall) can alias into a zero-length loop. It is also the width of the on-wall
# band in run_mc, and for the same reason: the overshoot is the ONLY way a ray
# ends up outside its own cell's radial wall by a hair, so an excess larger
# than this is a real escape and not a boundary artifact.
RAY_EPS_CM = 1e-7

# The two mutually exclusive rows the solver books the plasma-terminating
# boundary under. Exactly one of them is live on any given run.
BOUNDARY_ROWS = ("boundary_absorption", "characteristic_boundary")


def vt_cm_s(T_eV):
    return np.sqrt(T_eV * EV / M_HE)


def boundary_recycle_row(f):
    """Name the ``rhs_terms`` row carrying this run's boundary recycle.

    Returns ``(row_name, stance)``, ``stance`` being the saved
    ``characteristic_boundary`` flag.

    Under ``characteristic_boundary`` -- the shipped production stance since
    R5 -- the solver zeroes the WHOLE ``boundary_absorption`` state, plasma
    removal and neutral return ``nn`` alike, and books both under
    ``characteristic_boundary``. An offline reader that hardcodes the legacy
    row therefore gets an identically-zero channel on every production run and
    silently drops the end-wall return from its source menu. Pre-R5 artifacts
    carry no ``characteristic_boundary`` key in ``flags_json`` (and usually no
    such row at all) and keep the legacy row.
    """
    raw = f.attrs.get("flags_json")
    flags = json.loads(raw) if raw is not None else {}
    stance = bool(flags.get("characteristic_boundary", False))
    return ("characteristic_boundary" if stance else "boundary_absorption"), stance


def assert_recycle_channel_live(recycle, removal, *, row, stance, path, window_ms):
    """Raise unless the selected recycle channel carries the boundary return.

    ``removal`` is the plasma removal booked by EITHER boundary row, so a
    channel read from the wrong row is caught rather than quietly contributing
    nothing to the source menu.
    """
    if np.any(recycle) or not np.any(removal):
        return
    raise ValueError(
        f"boundary recycle channel is identically zero over "
        f"{window_ms[0]}-{window_ms[1]} ms while the plasma-removal row is "
        f"nonzero, for {path}.\n"
        f"  stance: characteristic_boundary={stance}\n"
        f"  row read: rhs_terms/{row}\n"
        "  likely cause: the run books its boundary physics under the OTHER "
        f"row of {BOUNDARY_ROWS} -- under characteristic_boundary the solver "
        "zeroes the entire boundary_absorption state (including the neutral "
        "return nn) and books it under characteristic_boundary. Refusing to "
        "run source-starved: the end-wall return would silently vanish from "
        "the source menu."
    )


def _puff_peak_cell(ns, roles):
    """Index of the puff row's peak cell, ties broken toward the puff cell.

    A distributed puff profile is symmetric about the valve, so the two cells
    straddling it carry EQUAL weight and ``np.argmax`` alone picks whichever
    comes first in the array -- a plain column cell one cell upstream of the
    role-tagged puff cell. Ties are resolved toward the ``puff`` role.
    """
    peak = ns.max()
    if peak <= 0.0:
        return int(np.argmax(ns))
    tied = np.flatnonzero(ns >= peak)
    for i in tied:
        if roles[i] == "puff":
            return int(i)
    return int(tied[0])


def load_background(path, window_ms):
    with h5py.File(path, "r") as f:
        t0 = float(f.attrs["t_breakdown_trigger"])
        t = (f["time"][:] - t0) * 1e3
        m = (t >= window_ms[0]) & (t <= window_ms[1])
        g = f["geometry"]
        roles = [
            r.decode() if isinstance(r, bytes) else str(r)
            for r in g["cell_role"][:]
        ]
        # Domain: cathode face (first non-plenum cell edge) to the far end.
        first = roles.index("cathode") if "cathode" in roles else 0
        length = g["length_cm"][:]
        z_lo = np.concatenate(([0.0], np.cumsum(length)))  # provisional
        # rebuild absolute edges from z centers
        zc = g["z_cm"][:]
        edges = np.concatenate((zc - 0.5 * length, [zc[-1] + 0.5 * length[-1]]))
        sel = slice(first, len(roles))
        bg = {
            "z_edges": edges[first : len(roles) + 1] - edges[first],
            "Rp": g["Rp_cm"][:][sel],
            "Rm": g["Rm_cm"][:][sel],
            "roles": roles[first:],
            "Vp": g["plasma_volume_cm3"][:][sel],
            "Vm": g["neutral_volume_cm3"][:][sel],
            "n": np.mean(f["n"][:][m], axis=0)[sel],
            "Te": np.mean(f["Te"][:][m], axis=0)[sel],
            "Ti": np.mean(f["Ti"][:][m], axis=0)[sel],
            "u": np.mean(f["u"][:][m], axis=0)[sel],
            "nn_model": np.mean(f["nn"][:][m], axis=0)[sel],
        }
        if "u_n" in f:
            bg["un_model"] = np.mean(f["u_n"][:][m], axis=0)[sel]
        if "nn_a" in f:
            # Two-zone run: the nn dataset is the
            # COLUMN density and nn_a the annulus -- exactly the TPMC's
            # per-zone tallies. nn_model is rebuilt as the chamber mean for
            # the headline table; the per-zone comparison prints separately.
            bg["nna_model"] = np.mean(f["nn_a"][:][m], axis=0)[sel]
            bg["nncol_model"] = bg["nn_model"]
            Vp_sel = g["plasma_volume_cm3"][:][sel]
            Vm_sel = g["neutral_volume_cm3"][:][sel]
            Va_sel = np.maximum(Vm_sel - Vp_sel, 0.0)
            bg["nn_model"] = (
                bg["nncol_model"] * Vp_sel + bg["nna_model"] * Va_sel
            ) / Vm_sel
        Vm_full = g["neutral_volume_cm3"][:]
        Vp_full = g["plasma_volume_cm3"][:]
        # The boundary row is stance-dependent (see boundary_recycle_row).
        row, stance = boundary_recycle_row(f)
        # Plasma removal as booked by EITHER row: the guard's reference, so a
        # channel read from the wrong row cannot pass as a genuinely empty one.
        removal_any = sum(
            -np.mean(f[f"rhs_terms/{name}/n"][:][m], axis=0) * Vp_full
            for name in BOUNDARY_ROWS
            if f"rhs_terms/{name}/n" in f
        )
        ba = np.mean(f[f"rhs_terms/{row}/nn"][:][m], axis=0) * Vm_full
        an = np.mean(f["rhs_terms/anode_collection/nn"][:][m], axis=0) * Vm_full
        ns = np.mean(
            np.clip(f["rhs_terms/neutral_sources/nn"][:][m], 0.0, None), axis=0
        ) * Vm_full
        if not np.any(ba) and "nn_a" in f:
            # K4a kinetic run: the neutral ledger rows are superseded
            # (zeroed) -- rebuild the source menu from the PLASMA-side
            # rows, which keep their exact forms: the recycle source is
            # the boundary plasma loss, the anode source its collection,
            # and the puff comes from the configured waveform. The recycle
            # row is the stance's row here too -- reading the legacy one is
            # the same defect one level down.
            ba = -np.mean(
                f[f"rhs_terms/{row}/n"][:][m], axis=0
            ) * Vp_full
            an = -np.mean(
                f["rhs_terms/anode_collection/n"][:][m], axis=0
            ) * Vp_full
            params_k = __import__("json").loads(f.attrs["params_json"])
            sccm = float(params_k.get("S_gp", 0.0))
            valves = float(params_k.get("gas_puff_valves", 2))
            ns = np.zeros_like(ba)
            # square waveform at plateau: full flow into the puff cell
            roles_full = [
                r.decode() if isinstance(r, bytes) else str(r)
                for r in g["cell_role"][:]
            ]
            puff_idx = (
                roles_full.index("puff") if "puff" in roles_full else 0
            )
            ns[puff_idx] = 4.477962e17 * sccm * valves
        assert_recycle_channel_live(
            ba,
            removal_any,
            row=row,
            stance=stance,
            path=str(path),
            window_ms=window_ms,
        )
        # Volume-recombination birth (n^2 * ACD via the run's own ledger --
        # identical to recomputing from the frozen fields, and closed by
        # construction): an nn gain everywhere the plasma recombines. The
        # plenum cell's share falls outside the TPMC domain (no plenum
        # volume; documented simplification) and is dropped from the menu.
        rec = np.zeros(len(roles))
        for term in ("recombination_rad_loss", "recombination_3b_loss"):
            key = f"rhs_terms/{term}/nn"
            if key in f and np.any(f[key][:][m]):
                rec += np.mean(
                    np.clip(f[key][:][m], 0.0, None), axis=0
                ) * Vm_full
            elif f"rhs_terms/{term}/n" in f:
                rec += np.mean(
                    np.clip(-f[f"rhs_terms/{term}/n"][:][m], 0.0, None),
                    axis=0,
                ) * Vp_full
        cd = f["cathode_diagnostics"]
        phi_c = float(np.nanmean(cd["source_phi_c"][:][m]))
        T_s = float(np.mean(cd["T_s_surface"][:][m]))
        params = __import__("json").loads(f.attrs["params_json"])
        raw_flags = f.attrs.get("flags_json")
        flags = json.loads(raw_flags) if raw_flags is not None else {}
    # Per-face cells by ROLE: the live cell against an absorbing face is where
    # the boundary term books its removal and its neutral return, and it is not
    # at a fixed offset from the array ends (an obstruction cell pushes the
    # cathode's live cell one further in). Legacy geometry declares no
    # absorbing faces at all -- there the boundary term is volumetric and the
    # end cells are the only meaningful attribution.
    by_role = absorbing_live_cells_by_role(build_geometry(params, flags))
    if by_role:
        missing = [r for r in ("cathode", "collector") if r not in by_role]
        if missing:
            raise ValueError(
                f"no plasma-absorbing live cell with role(s) {missing}; "
                f"absorbing faces resolve to {by_role}. The recycle ledger "
                "cannot be attributed per face."
            )
        cath_cell = int(by_role["cathode"][0])
        coll_cell = int(by_role["collector"][-1])
    else:
        cath_cell = roles.index("cathode")
        coll_cell = len(roles) - 1
    anode_cells = [i for i, r in enumerate(roles) if r == "gap"][-1:]  # gap side
    bg["sources"] = {
        "cathode_face": float(ba[cath_cell]),
        "collector_face": float(ba[coll_cell]),
        "anode_left": float(an[an.nonzero()[0][0]]) if an.any() else 0.0,
        "anode_right": float(an[an.nonzero()[0][-1]]) if an.any() else 0.0,
        # The puff is a DISTRIBUTION over cells, not a point: the solver's
        # 'gaussian' and 'cosine_pipe' profiles spread the inflow over every
        # eligible main-chamber cell (normalized to conserve it exactly), and
        # only the legacy 'cell' profile is a single cell. Carry the whole row
        # and let the launcher sample it, exactly as vol_rec does. The rate and
        # the weights are read from the SAME in-domain slice so they cannot
        # disagree (any share upstream of the cathode face is outside the TPMC
        # domain, as for vol_rec).
        "puff": float(ns[first:].sum()),
        # Representative SINGLE-cell z, kept for the point-injection consumers
        # of this loader (kn2zone, the E0 bench, the E2 DVM comparison), whose
        # own discretizations inject the puff into one z-bin. run_mc no longer
        # uses it. np.argmax alone resolved a tie by array order, which on a
        # cosine_pipe run put the source in a plain column cell one cell
        # UPSTREAM of the role-tagged puff cell; prefer the puff cell whenever
        # it is among the maxima.
        "puff_z": float(zc[_puff_peak_cell(ns, roles)] - edges[first]),
        "vol_rec": float(rec[first:].sum()),
    }
    bg["puff_cell"] = ns[first:]
    bg["rec_cell"] = rec[first:]
    bg["phi_c"] = phi_c
    bg["T_s"] = T_s
    bg["eta"] = float(params.get("eta", 0.358))
    bg["S_pump_L"] = float(params.get("S_pump_L", 2000.0))
    bg["S_pump_R"] = float(params.get("S_pump_R", 4000.0))
    bg["R_cath"] = float(params.get("R_cath", 15.0))
    # anode mesh plane: boundary between gap and puff cells
    gap_last = max(i for i, r in enumerate(bg["roles"]) if r == "gap")
    bg["mesh_edge"] = gap_last + 1  # index into z_edges
    # collision rates per cell (column only)
    n_safe = np.maximum(bg["n"], 1e6)
    rates = he_rates(n_safe, np.maximum(bg["Te"], 0.2), ("scd",))
    bg["nu_ion"] = bg["n"] * rates["scd"]
    bg["nu_cx"] = bg["n"] * charge_ex_react(np.maximum(bg["Ti"], 0.05), "He")
    return bg


def cosine_emit(rng, N, T_K, sign_z):
    """Diffuse (cosine-flux) emission from a z-normal surface at T_K."""
    vt = np.sqrt(KB * T_K / M_HE)
    vz = sign_z * vt * np.sqrt(-2.0 * np.log(rng.random(N)))
    vx = rng.normal(0.0, vt, N)
    vy = rng.normal(0.0, vt, N)
    return np.column_stack((vx, vy, vz))


def wall_emit_inward(rng, x, y, T_K):
    """Diffuse emission from the radial wall, normal pointing inward."""
    N = x.size
    vt = np.sqrt(KB * T_WALL_K / M_HE) if T_K is None else np.sqrt(KB * T_K / M_HE)
    r = np.sqrt(x**2 + y**2)
    nx, ny = -x / r, -y / r  # inward normal
    vn = vt * np.sqrt(-2.0 * np.log(rng.random(N)))
    vt1 = rng.normal(0.0, vt, N)  # tangential in-plane (t = (-ny, nx))
    vz = rng.normal(0.0, vt, N)
    vx = vn * nx - vt1 * ny
    vy = vn * ny + vt1 * nx
    return np.column_stack((vx, vy, vz))


def maxwellian(rng, N, Ti_eV, u_drift):
    s = vt_cm_s(np.maximum(Ti_eV, 0.02))
    v = rng.normal(0.0, 1.0, (N, 3)) * s[:, None]
    v[:, 2] += u_drift
    return v


def run_mc(bg, n_particles, jet, rng, r_n=(0.5, 0.5), r_e=(0.2, 0.25),
           max_iter=20000, report_times_s=()):
    ze = bg["z_edges"]
    ncell = ze.size - 1
    Rp, Rm = bg["Rp"], bg["Rm"]
    nu_ion, nu_cx = bg["nu_ion"], bg["nu_cx"]
    nu_tot = nu_ion + nu_cx
    nu_max = float(nu_tot.max())
    mesh_edge = bg["mesh_edge"]
    transparency = 1.0 - bg["eta"]
    vbar = np.sqrt(8.0 * KB * T_WALL_K / (np.pi * M_HE))
    A_end = np.pi * Rm[-1] ** 2
    s_R = bg["S_pump_R"] * 1e3 / (A_end * vbar / 4.0)
    s_L = bg["S_pump_L"] * 1e3 / (A_end * vbar / 4.0)

    # ---- source menu: (rate, launcher) ----
    src = bg["sources"]
    T_s, phi_c = bg["T_s"], bg["phi_c"]
    R_cath = bg["R_cath"]

    def launch(name, N):
        pos = np.zeros((N, 3))
        if name == "puff":
            # Sample the launch cell from the run's own per-cell puff row, then
            # uniformly within that cell. Entry is still at the chamber wall
            # pointing inward -- the physical pipe outlet the 'cosine_pipe'
            # profile models -- so only the AXIAL spread changes.
            w_cell = bg["puff_cell"] / bg["puff_cell"].sum()
            icell = rng.choice(w_cell.size, size=N, p=w_cell)
            th = rng.random(N) * 2 * np.pi
            pos[:, 0] = Rm[icell] * 0.999 * np.cos(th)
            pos[:, 1] = Rm[icell] * 0.999 * np.sin(th)
            pos[:, 2] = ze[icell] + rng.random(N) * (ze[icell + 1] - ze[icell])
            vel = wall_emit_inward(rng, pos[:, 0], pos[:, 1], T_WALL_K)
        elif name in ("cathode_face", "collector_face"):
            at_start = name == "cathode_face"
            rad = (R_cath if at_start else Rp[-1]) * np.sqrt(rng.random(N))
            th = rng.random(N) * 2 * np.pi
            pos[:, 0] = rad * np.cos(th)
            pos[:, 1] = rad * np.sin(th)
            pos[:, 2] = 1e-6 if at_start else ze[-1] - 1e-6
            sign = 1.0 if at_start else -1.0
            if at_start and jet in ("cathode", "both"):
                RN, RE = r_n[0], r_e[0]
                fast = rng.random(N) < RN
                v_back = np.sqrt(2.0 * RE * (max(phi_c, 0.0) + 1.0) * EV / M_HE)
                vel = cosine_emit(rng, N, T_s, sign)
                sc = np.where(
                    fast,
                    v_back / np.maximum(np.linalg.norm(vel, axis=1), 1.0),
                    1.0,
                )
                vel = vel * sc[:, None]
            else:
                vel = cosine_emit(rng, N, T_s if at_start else T_WALL_K, sign)
        elif name == "vol_rec":
            # Recombination birth: in-column, at the local ion Maxwellian +
            # drift (the recombined ion hands its momentum over -- the same
            # convention as the solver's handover and the CX resample here).
            w_cell = bg["rec_cell"] / bg["rec_cell"].sum()
            icell = rng.choice(w_cell.size, size=N, p=w_cell)
            rad = Rp[icell] * np.sqrt(rng.random(N))
            th = rng.random(N) * 2 * np.pi
            pos[:, 0] = rad * np.cos(th)
            pos[:, 1] = rad * np.sin(th)
            pos[:, 2] = ze[icell] + rng.random(N) * (ze[icell + 1] - ze[icell])
            vel = maxwellian(rng, N, bg["Ti"][icell], bg["u"][icell])
        elif name in ("anode_left", "anode_right"):
            left = name == "anode_left"
            icell = mesh_edge - 1 if left else mesh_edge
            rad = Rp[icell] * np.sqrt(rng.random(N))
            th = rng.random(N) * 2 * np.pi
            pos[:, 0] = rad * np.cos(th)
            pos[:, 1] = rad * np.sin(th)
            pos[:, 2] = ze[mesh_edge] + (-1e-6 if left else 1e-6)
            sign = -1.0 if left else 1.0
            if jet == "both":
                RN, RE = r_n[1], r_e[1]
                fast = rng.random(N) < RN
                # phi_a ~ from the solve would be better; use 0.4*phi_c class
                v_back = np.sqrt(2.0 * RE * (0.45 * max(phi_c, 0.0)) * EV / M_HE)
                vel = cosine_emit(rng, N, T_WALL_K, sign)
                sc = np.where(
                    fast,
                    v_back / np.maximum(np.linalg.norm(vel, axis=1), 1.0),
                    1.0,
                )
                vel = vel * sc[:, None]
            else:
                vel = cosine_emit(rng, N, T_WALL_K, sign)
        else:
            raise ValueError(name)
        return pos, vel

    names = [k for k in ("puff", "cathode_face", "collector_face",
                         "anode_left", "anode_right", "vol_rec")
             if src.get(k, 0.0) > 0]
    rates = np.array([src[k] for k in names])
    frac = rates / rates.sum()
    counts = np.maximum((frac * n_particles).astype(int), 1)
    w_each = rates / counts  # atoms/s per history

    # tallies
    tal_t = np.zeros((ncell, 2))       # residence [atom-s per s] col/ann
    tal_tv = np.zeros((ncell, 2))      # sum w*dt*vz
    tal_ion = np.zeros(ncell)          # ionization sink [atoms/s]
    # Time-dependent buildup tallies (K0): for
    # stationary sources switched on into an EMPTY box at t = 0, the density
    # at time T is exactly the steady residence tally restricted to
    # particle age < T -- so each segment contributes
    # wgt * clip(T - age, 0, dt) to the report-time-T tally. Exact (no age
    # binning error); the steady tally is the T -> inf member.
    report_times = np.asarray(report_times_s, dtype=float)
    tal_t_time = np.zeros((report_times.size, ncell, 2))
    lost = {"ion": 0.0, "pump": 0.0, "stuck": 0.0}
    # On-wall wall-root clamps (see the guard in the step below). Reported so
    # the clamp cannot silently become a bias: it is a roundoff-scale event
    # and its count belongs in the run's own output.
    n_wall_clamp = 0

    for name, N, w in zip(names, counts, w_each):
        pos, vel = launch(name, int(N))
        wgt = np.full(int(N), w)
        age = np.zeros(int(N))
        for _ in range(max_iter):
            n_act = wgt.size
            if n_act == 0:
                break
            speed = np.linalg.norm(vel, axis=1)
            speed = np.maximum(speed, 1.0)
            icell = np.clip(np.searchsorted(ze, pos[:, 2]) - 1, 0, ncell - 1)
            # distance to next z-edge along vz
            with np.errstate(divide="ignore"):
                d_z = np.where(
                    vel[:, 2] > 0,
                    (ze[icell + 1] - pos[:, 2]) / vel[:, 2],
                    np.where(
                        vel[:, 2] < 0,
                        (ze[icell] - pos[:, 2]) / vel[:, 2],
                        np.inf,
                    ),
                ) * speed  # convert time to path length
            # distance to radial wall |xy + t*vxy| = Rm(icell)
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
            # On-wall degenerate. Only the wall handler below pulls a ray back
            # inside the vessel, and it is skipped whenever another event won
            # the step -- so an event that ends a segment within the ray
            # overshoot of the wall (in the case this guard was written for, a
            # null collision 2.30737 cm along the ray, ~9e-8 cm short of the
            # wall) is advanced THROUGH it, leaving the ray at most RAY_EPS_CM
            # outside its own cell's Rm. There (Rw^2 - r2) < 0 turns both roots
            # negative and the backward root wins the minimum below. Such a
            # ray is ON the wall: its flight length is zero and its next event
            # is the wall itself, so clamp the root to zero and let the wall
            # handler take it. The gate is the RADIAL excess, which the
            # overshoot bounds -- not the size of d, which a grazing ray
            # inflates by 1/cos -- so a ray that genuinely punched through a
            # step face (whole cm to 1e18 cm outside) keeps its negative d and
            # is still refused by the tripwire.
            on_wall = (r2 > Rw**2) & ((np.sqrt(r2) - Rw) <= RAY_EPS_CM)
            clamp = on_wall & (d_wall < 0.0)
            if clamp.any():
                n_wall_clamp += int(clamp.sum())
                d_wall = np.where(clamp, 0.0, d_wall)
            # distance to the column surface r = Rp (both directions), so no
            # segment ever spans the column boundary -- otherwise a chord
            # through the column would skip collision testing (a transparent
            # column artifactually inflates annulus lifetimes).
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
            # null-collision distance
            d_coll = -np.log(rng.random(n_act)) * speed / nu_max
            d = np.minimum(np.minimum(d_z, d_wall), np.minimum(d_coll, d_rp))
            d = np.minimum(d, 1e6)
            if np.any(d < 0.0):
                # A negative flight length is never physical: it means a ray is
                # standing at r > Rm(icell), where (Rw^2 - r2) < 0 drives both
                # wall-intersection roots negative and the backward root wins
                # the minimum. Tallying it accumulates NEGATIVE residence and
                # marches the particle backwards without bound, so the
                # estimator diverges rather than degrading. Fail loudly here
                # instead: the tally below is unrecoverable once fed.
                bad = np.flatnonzero(d < 0.0)
                j = bad[0]
                raise ValueError(
                    f"negative flight length in the neutral ray tracer: "
                    f"{bad.size} of {n_act} histories, min d={d[bad].min():.6g} "
                    f"cm (source '{name}').\n"
                    f"  first offender: cell {icell[j]}, "
                    f"r={np.sqrt(r2[j]):.6g} cm vs Rm={Rm[icell[j]]:.6g} cm "
                    f"(excess {np.sqrt(r2[j]) - Rm[icell[j]]:.6g} cm, "
                    f"overshoot {RAY_EPS_CM:g} cm)\n"
                    "  cause: the ray sits outside the vessel wall of its own "
                    "cell by MORE than the ray overshoot, which happens when a "
                    "z-crossing into a NARROWER section is not intercepted by "
                    "the annular step face. (Excesses within the overshoot are "
                    "the on-wall degenerate and are clamped above, not "
                    "refused.)"
                )
            dt = d / speed
            # tally the segment (entirely inside icell)
            in_col = r2 < Rp[icell] ** 2  # start-of-segment zone (approx)
            zone = np.where(in_col, 0, 1)
            np.add.at(tal_t, (icell, zone), wgt * dt)
            np.add.at(tal_tv, (icell, zone), wgt * dt * vel[:, 2])
            if report_times.size:
                min_age = float(age.min())
                for k, T in enumerate(report_times):
                    if T <= min_age:
                        continue  # every particle already older than T
                    w_dt = wgt * np.clip(T - age, 0.0, dt)
                    np.add.at(tal_t_time[k], (icell, zone), w_dt)
                age = age + dt
            # advance; overshoot 0.1 um along the ray so no boundary (z-edge
            # or the Rp surface) can alias into zero-length loops
            pos = pos + vel * (dt[:, None] * 1.0)
            pos = pos + (vel / speed[:, None]) * RAY_EPS_CM
            kill = np.zeros(n_act, dtype=bool)
            # --- collision events
            hit_c = d_coll <= np.minimum(np.minimum(d_z, d_wall), d_rp)
            if hit_c.any():
                ic = icell[hit_c]
                real = rng.random(hit_c.sum()) < (nu_tot[ic] / nu_max) * (
                    r2[hit_c] < Rp[ic] ** 2
                )
                idx = np.flatnonzero(hit_c)[real]
                if idx.size:
                    ii = icell[idx]
                    ionz = rng.random(idx.size) < nu_ion[ii] / nu_tot[ii]
                    ion_idx = idx[ionz]
                    np.add.at(tal_ion, icell[ion_idx], wgt[ion_idx])
                    lost["ion"] += float(wgt[ion_idx].sum())
                    kill[ion_idx] = True
                    cx_idx = idx[~ionz]
                    if cx_idx.size:
                        ii = icell[cx_idx]
                        vel[cx_idx] = maxwellian(
                            rng, cx_idx.size, bg["Ti"][ii], bg["u"][ii]
                        )
            # --- radial wall
            hit_w = (~hit_c) & (d_wall <= np.minimum(d_z, d_rp))
            if hit_w.any():
                idx = np.flatnonzero(hit_w)
                r_now = np.sqrt(pos[idx, 0] ** 2 + pos[idx, 1] ** 2)
                shrink = (Rm[icell[idx]] * 0.9999) / np.maximum(r_now, 1e-9)
                pos[idx, 0] *= shrink
                pos[idx, 1] *= shrink
                vel[idx] = wall_emit_inward(rng, pos[idx, 0], pos[idx, 1], None)
            # --- z-edge crossings: ends, mesh, and the annular step face
            # (Rp-surface crossings need no handler: the segment split plus
            # the ray overshoot is the whole event)
            hit_z = (~hit_c) & (~hit_w) & (d_z <= d_rp)
            if hit_z.any():
                idx = np.flatnonzero(hit_z)
                zdir = np.sign(vel[idx, 2])
                edge = np.where(zdir > 0, icell[idx] + 1, icell[idx])
                # ends
                at_L = edge == 0
                at_R = edge == ncell
                atm = edge == mesh_edge
                # pump sticking at ends, else diffuse re-emit
                for at_end, sign, s_stick, T_emit in (
                    (at_L, 1.0, s_L, T_s),
                    (at_R, -1.0, s_R, T_WALL_K),
                ):
                    eidx = idx[at_end]
                    if eidx.size == 0:
                        continue
                    stick = rng.random(eidx.size) < s_stick
                    kill[eidx[stick]] = True
                    lost["pump"] += float(wgt[eidx[stick]].sum())
                    keep = eidx[~stick]
                    if keep.size:
                        vel[keep] = cosine_emit(rng, keep.size, T_emit, sign)
                        pos[keep, 2] = np.clip(
                            pos[keep, 2], 1e-6, ze[-1] - 1e-6
                        )
                # mesh interception
                midx = idx[atm & ~at_L & ~at_R]
                if midx.size:
                    blocked = rng.random(midx.size) > transparency
                    bidx = midx[blocked]
                    if bidx.size:
                        sign = -np.sign(vel[bidx, 2])
                        vel[bidx] = cosine_emit(rng, bidx.size, T_WALL_K, sign)
                        pos[bidx, 2] = ze[mesh_edge] + sign * 1e-6
                # annular step face: where Rm narrows across an interior
                # z-edge, the part of the crossing plane with
                # Rm(dest) < r <= Rm(src) is a real z-normal annulus of
                # vessel wall, not an opening. Without this the ray passes
                # THROUGH the wall and is left outside its cell's radius --
                # the divergent-estimator failure the tripwire above names.
                # Diffuse re-emission back into the cell it came from, at the
                # radial wall's convention (full accommodation, 300 K).
                interior = (edge > 0) & (edge < ncell)
                e = idx[interior]
                if e.size:
                    zdir_i = zdir[interior]
                    dest = np.where(zdir_i > 0, edge[interior],
                                    edge[interior] - 1)
                    # NOT r_e: that is a run_mc PARAMETER (the jet
                    # fast-fraction energy pair) which the launch() closure
                    # reads from this scope, so binding it here would feed
                    # launch() an array of radii on every later source.
                    r_step = np.sqrt(pos[e, 0] ** 2 + pos[e, 1] ** 2)
                    step = r_step > Rm[dest]
                    h = e[step]
                    if h.size:
                        sgn = -zdir_i[step]
                        vel[h] = cosine_emit(rng, h.size, T_WALL_K, sgn)
                        pos[h, 2] += sgn * 1e-6
            alive = ~kill
            pos, vel, wgt = pos[alive], vel[alive], wgt[alive]
            age = age[alive]
        else:
            # max_iter exhausted: report separately -- a nonzero fraction
            # here means the transport is under-resolved, not pumped.
            lost["stuck"] += float(wgt.sum())

    V_col = np.pi * Rp**2 * np.diff(ze)
    V_ann = np.pi * (Rm**2 - Rp**2) * np.diff(ze)
    nn_col = tal_t[:, 0] / np.maximum(V_col, 1e-9)
    nn_ann = tal_t[:, 1] / np.maximum(V_ann, 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        un_col = np.where(tal_t[:, 0] > 0, tal_tv[:, 0] / tal_t[:, 0], 0.0)
        un_ann = np.where(tal_t[:, 1] > 0, tal_tv[:, 1] / tal_t[:, 1], 0.0)
    nn_mean = (tal_t.sum(axis=1)) / (V_col + V_ann)
    un_mean = np.where(
        tal_t.sum(axis=1) > 0, tal_tv.sum(axis=1) / tal_t.sum(axis=1), 0.0
    )
    out = {
        "nn_col": nn_col, "nn_ann": nn_ann, "nn_mean": nn_mean,
        "un_col": un_col, "un_ann": un_ann, "un_mean": un_mean,
        "S_ion": tal_ion, "lost": lost, "rates": dict(zip(names, rates)),
        "n_wall_clamp": n_wall_clamp,
    }
    if report_times.size:
        out["report_times_s"] = report_times
        out["nn_col_t"] = tal_t_time[:, :, 0] / np.maximum(V_col, 1e-9)
        out["nn_ann_t"] = tal_t_time[:, :, 1] / np.maximum(V_ann, 1e-9)
        out["nn_mean_t"] = tal_t_time.sum(axis=2) / (V_col + V_ann)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("-n", "--n-particles", type=int, default=200000)
    ap.add_argument("--jet", choices=("none", "cathode", "both"),
                    default="none")
    ap.add_argument("--no-vol-rec", action="store_true",
                    help="drop the volume-recombination birth source")
    ap.add_argument("--report-ms", default="1,2,3,5,8,12,17,25,40",
                    help="comma-separated buildup report times [ms] "
                         "(K0); empty string "
                         "disables the time-dependent tallies")
    ap.add_argument("--window", nargs=2, type=float, default=(5.0, 19.5))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    bg = load_background(args.run, tuple(args.window))
    if args.no_vol_rec:
        bg["sources"]["vol_rec"] = 0.0
    report_times = tuple(
        float(x) * 1e-3 for x in args.report_ms.split(",") if x.strip()
    )
    rng = np.random.default_rng(args.seed)
    res = run_mc(bg, args.n_particles, args.jet, rng,
                 report_times_s=report_times)

    tot = sum(res["rates"].values())
    print(f"sources [atoms/s]: " + ", ".join(
        f"{k}={v:.3g}" for k, v in res["rates"].items()))
    print(f"sinks: ionization {res['lost']['ion']:.3g}, "
          f"pump {res['lost']['pump']:.3g}, "
          f"stuck {res['lost']['stuck']:.3g}, total {tot:.3g} "
          f"(closure {sum(res['lost'].values()) / tot:.3f})")
    print(f"on-wall wall-root clamps: {res['n_wall_clamp']}")

    ze = bg["z_edges"]
    zc = 0.5 * (ze[:-1] + ze[1:])
    print(f"\n{'z[cm]':>7} {'nn_model':>10} {'nn_MC':>10} {'ratio':>6} "
          f"{'col/ann':>8} {'un_MC[km/s]':>11} {'un_model':>9}")
    un_model = bg.get("un_model", np.full_like(zc, np.nan))
    for i in range(0, zc.size, max(1, zc.size // 18)):
        ca = res["nn_col"][i] / max(res["nn_ann"][i], 1e-3)
        print(f"{zc[i]:7.0f} {bg['nn_model'][i]:10.3g} "
              f"{res['nn_mean'][i]:10.3g} "
              f"{res['nn_mean'][i] / max(bg['nn_model'][i], 1e-3):6.2f} "
              f"{ca:8.2f} {res['un_mean'][i] / 1e5:11.2f} "
              f"{un_model[i] / 1e5 if np.isfinite(un_model[i]) else np.nan:9.2f}")

    if "nna_model" in bg:
        # Per-zone comparison: the model's split fields against the MC's
        # per-zone tallies -- the M4 gate.
        print(f"\n{'z[cm]':>7} {'col_model':>10} {'col_MC':>10} {'r_col':>7} "
              f"{'ann_model':>10} {'ann_MC':>10} {'r_ann':>7}")
        for i in range(0, zc.size, max(1, zc.size // 18)):
            print(f"{zc[i]:7.0f} {bg['nncol_model'][i]:10.3g} "
                  f"{res['nn_col'][i]:10.3g} "
                  f"{res['nn_col'][i] / max(bg['nncol_model'][i], 1e-3):7.2f} "
                  f"{bg['nna_model'][i]:10.3g} {res['nn_ann'][i]:10.3g} "
                  f"{res['nn_ann'][i] / max(bg['nna_model'][i], 1e-3):7.2f}")

    if "report_times_s" in res:
        # K0 deliverable: the annulus reservoir's
        # buildup from an empty start against the ~20 ms drive. The steady
        # tallies are the infinite-time limit and an UPPER BOUND for
        # in-shot conditions; closure gates should compare like-for-like
        # at the model's own time.
        mid = (zc >= 500.0) & (zc <= 1000.0)
        ann_steady = float(np.mean(res["nn_ann"][mid]))
        col_steady = float(np.mean(res["nn_col"][mid]))
        print("\nK0 buildup (mid-machine z=500-1000 mean; fraction of "
              "steady):")
        print(f"{'t[ms]':>6} {'nn_ann':>10} {'f_ann':>6} "
              f"{'nn_col':>10} {'f_col':>6}")
        for k, T in enumerate(res["report_times_s"]):
            ann_T = float(np.mean(res["nn_ann_t"][k][mid]))
            col_T = float(np.mean(res["nn_col_t"][k][mid]))
            print(f"{T * 1e3:6.1f} {ann_T:10.3g} "
                  f"{ann_T / max(ann_steady, 1e-30):6.3f} "
                  f"{col_T:10.3g} {col_T / max(col_steady, 1e-30):6.3f}")

    # NBL observable (validation target for the two-zone particle channel):
    # peak location, width, and magnitude of the far-end neutral
    # accumulation. Peak location is reported as an observation, not a
    # gate (an off-wall peak was an impression from earlier runs, not a
    # requirement -- Tom, 2026-07-21). The physics content is detachment:
    # the NBL is the layer through which the incoming column plasma cools
    # and recombines, so only a fraction of the column flux reaches the
    # wall as ions (divertor-like physics on LAPD); the vol_rec /
    # collector_face source-rate ratio above is the ledger's own
    # detachment fraction.
    half = zc.size // 2
    for label, prof in (("MC", res["nn_mean"]), ("model", bg["nn_model"])):
        far = prof[half:]
        ipk = half + int(np.argmax(far))
        peak = prof[ipk]
        above = np.flatnonzero(prof[half:] >= 0.5 * peak) + half
        width = ze[above[-1] + 1] - ze[above[0]]
        wall = prof[-1]
        print(f"NBL[{label}]: peak {peak:.3g} at z={zc[ipk]:.0f} "
              f"({'off-wall' if ipk < zc.size - 1 else 'wall cell'}), "
              f"peak/wall {peak / max(wall, 1e-3):.2f}, FWHM {width:.0f} cm")

    out = args.out or (Path(args.run).stem + f"_mc_{args.jet}")
    np.savez(
        Path(args.run).parent / f"{out}.npz",
        z=zc, nn_model=bg["nn_model"], un_model=un_model,
        **{k: bg[k] for k in ("nncol_model", "nna_model") if k in bg},
        **{
            k: v for k, v in res.items() if isinstance(v, np.ndarray)
        },
    )
    print(f"\nsaved {out}.npz")


if __name__ == "__main__":
    main()
