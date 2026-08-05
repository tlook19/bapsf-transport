"""E1: multirate-convergence and velocity-grid adequacy for the transient DVM.

**A measurement instrument, not a physics instrument.** It fits nothing,
tunes nothing, and recommends nothing: it reports how far each candidate
neutral clock and each candidate velocity grid sits from a finer anchor, so
that the cadence and grid decisions can be made on numbers. Per the standing
rule of the evaluation package, cadence is never argued from runtime --
Part 1 reports ACCURACY only, and the one cost table in this file (Part 2)
is a property of the velocity grid, not of the clock.

Everything is driven through the SHIPPED ``TransientDVM`` operator
(``physics/kinetic_dvm.py``). Nothing here re-implements the velocity grid,
the implicit march, the collision closure or the ledger; the only arithmetic
this file owns is (a) reading a frozen background, (b) forming moments of
the operator's own public state, and (c) inverting the operator's own birth
additions to recover the marched (post-substep-A) distribution, which is
what the wall-energy readouts need and which is checked against the
operator's particle ledger on every single update.

Part 1 -- multirate convergence
-------------------------------
On the frozen nx=240 background, the DVM is advanced at neutral clocks of
5 / 10 / 25 / 50 / 100 us through three intervals: a rapid source transient
(breakdown burnout), a representative discharge interval (plateau) and an
afterglow interval. The 5 us arm is the ANCHOR; every readout is reported as
a relative deviation from it, so the 10 us clock the K2a build provisionally
sized against has a measured convergence direction rather than an assumed
one.

The coupling convention is production's, taken from ``_dvm_advance``: the
plasma coefficients are those at the END of the neutral tick, applied across
the whole tick. That is what makes the coarse clock lag -- between ticks the
frozen ``nu_ion`` does not follow the plasma's own ionization booking -- and
the transient interval is where that lag is expected to show.

All arms of one interval start from a single shared kinetic state: the fluid
neutral densities at the start of a burn-in window, seeded as a wall-
temperature Maxwellian and then relaxed through the burn-in at the anchor
cadence. One snapshot, restored per arm, so the arms differ ONLY in clock.

Part 2 -- velocity-grid adequacy
--------------------------------
The K2a build measured the production 48x12 sinh-stretched grid sitting
~2.4e-2 from the continuum Maxwellian at the closed-box collision fixed
point, refining to 8.4e-3 at 96x32 (gate L4). This part quantifies that per
moment: on three representative background cells (cold near-wall,
mid-column with the CX tail, hot source-adjacent) the DVM is relaxed to a
local fixed point on a closed uniform tube carrying that cell's plasma
condition, at 48x12 / 64x16 / 96x32 / 128x48, and the density, momentum,
T_par, T_perp and plasma source integrals are compared against the 128x48
anchor. The local problem is closed by balancing the one non-conservative
channel: ionization is matched, tick by tick, by a recombination-channel
rebirth of exactly equal rate, so the box has a genuine steady state
instead of draining. Per-update COST is measured on the production nx=240
geometry at each grid (median plus spread, the E0 pattern), so the grid
decision has both axes.

Usage (single command, reruns end to end):

    PYTHONPATH=<checkout>/cablp python scripts/neutral_arch_e1_study.py \
        --run scripts/es1_kn2z_promoted_nx240.h5 --out-dir scripts
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cablp.solvers._sim1d import LAPDSim1D  # noqa: E402
from cablp.solvers._sim1d.core.state import derive_state  # noqa: E402
from cablp.solvers._sim1d.physics.kinetic_dvm import (  # noqa: E402
    TransientDVM,
    ledger_residual,
)
from cablp.solvers._sim1d.physics.kinetic_neutrals import (  # noqa: E402
    EV,
    KB,
    M_HE,
    T_WALL_K,
)

# --- Part 1 configuration -------------------------------------------------
# The anchor is finer than the 10 us the K2a build provisionally sized
# against, so 10 us is measured rather than assumed. Every coarser clock is
# an integer multiple of the anchor, which is what lets all arms sample the
# SAME background table with no interpolation of the derived quantities.
ANCHOR_S = 5.0e-6
CADENCES_S = (5.0e-6, 1.0e-5, 2.5e-5, 5.0e-5, 1.0e-4)
BURN_IN_S = 5.0e-4

# (label, start_ms, end_ms, description). Windows are integer multiples of
# the coarsest clock (100 us) so every arm lands exactly on the interval end.
INTERVALS = (
    (
        "transient",
        2.10,
        2.60,
        "rapid source transient: breakdown burnout, nn falls ~8.6x",
    ),
    (
        "discharge",
        12.00,
        13.00,
        "representative discharge interval: main-discharge plateau",
    ),
    (
        "afterglow",
        22.40,
        23.40,
        "afterglow interval: Te collapse and neutral refill",
    ),
)

# Hot-tail thresholds [eV] of TOTAL kinetic energy per atom. The 300 K wall
# population has a mean kinetic energy of 1.5 kT = 3.88e-2 eV, so both
# thresholds isolate the charge-exchange-fed tail from the thermal bulk.
HOT_TAIL_EV = (0.5, 2.0)

# --- Part 2 configuration -------------------------------------------------
VGRIDS = ((48, 12), (64, 16), (96, 32), (128, 48))
VGRID_ANCHOR = (128, 48)
VGRID_TUBE_CELLS = 12
VGRID_DT_S = 2.5e-5
VGRID_UPDATES = 400
VGRID_SETTLE_TAIL = 40
COST_REPEATS = 7

ROUNDOFF_REL = 1.0e-9


# ------------------------------------------------------------------ moments


def _kinetic_energy_per_bin(g):
    """Total kinetic energy of one atom in each velocity bin [erg]."""
    return 0.5 * M_HE * g.V2


def zone_moments(f, volumes, g):
    """Return (particles, axial momentum [g cm/s], kinetic energy [erg])."""
    counts = f * np.asarray(volumes, dtype=float)[:, None, None]
    return (
        float(counts.sum()),
        float(M_HE * (counts * g.VZ[None, :, :]).sum()),
        float((counts * _kinetic_energy_per_bin(g)[None, :, :]).sum()),
    )


def hot_tail_inventory(f, volumes, g, threshold_eV):
    """Particles in bins whose kinetic energy exceeds ``threshold_eV``."""
    mask = _kinetic_energy_per_bin(g) > threshold_eV * EV
    counts = f * np.asarray(volumes, dtype=float)[:, None, None]
    return float((counts * mask[None, :, :]).sum())


def directional_temperatures_eV(f, g):
    """Return per-cell (T_par, T_perp) [eV] of a distribution.

    ``T_par = m <(v_z - u)^2>`` (one degree of freedom) and
    ``T_perp = m <v_perp^2> / 2`` (two degrees of freedom), so that
    ``(T_par + 2 T_perp)/3`` is the scalar temperature the engine reports.
    """
    n = f.sum(axis=(1, 2))
    safe = np.maximum(n, 1e-300)
    u = (f * g.VZ[None, :, :]).sum(axis=(1, 2)) / safe
    c_par2 = (g.VZ[None, :, :] - u[:, None, None]) ** 2
    t_par = M_HE * (f * c_par2).sum(axis=(1, 2)) / safe / EV
    t_perp = (
        0.5 * M_HE * (f * (g.VP**2)[None, :, :]).sum(axis=(1, 2)) / safe / EV
    )
    return np.where(n > 0.0, t_par, 0.0), np.where(n > 0.0, t_perp, 0.0)


# ----------------------------------------------- marched-state reconstruction


class MarchedStateError(RuntimeError):
    """Raised when the reconstruction disagrees with the engine's ledger."""


def _outgoing_weight(g, sign):
    """``|v_z|`` on one half of the v_z axis, zero on the other."""
    w = np.zeros((g.nvz, g.nvp))
    sel = g.vz > 0 if sign > 0 else g.vz < 0
    w[sel, :] = np.abs(g.vz[sel])[:, None]
    return w


def reconstruct_marched(dvm, dt, known, nu_c_react, ledger):
    """Recover the post-substep-A distributions from the public post-state.

    ``update`` adds every birth to the marched state in place, so the
    marched state is not exposed. It is however recoverable in closed form,
    because each birth is a KNOWN spectrum scaled by a linear functional of
    the marched state itself:

    * column: ``f_c = f_m + a M_i + b M_wall``, with ``a`` the charge
      exchange plus elastic rebirth (a rate moment of ``f_m``) and ``b``
      the anode-mesh re-emission (an outgoing-flux moment of ``f_m``, and
      nonzero only in the two cells flanking the mesh face);
    * annulus: ``f_a = f_m + (c + d) M_wall``, with ``c`` the radial-wall
      re-emission and ``d`` the mesh re-emission.

    Substituting the definitions gives a 2x2 linear system per cell, solved
    here exactly. ``known_*`` are the birth contributions that do not depend
    on the marched state (the external source ledger), already in density
    units.

    Requires unit accommodation: below unity the wall re-emits a
    bin-preserving reflected fraction, which merges with the marched state
    and makes the inversion singular. The caller is expected to run the
    production stance, where the accommodation is 1.

    Every reconstruction is checked against the engine's own particle
    ledger before it is used; a disagreement raises ``MarchedStateError``
    rather than being reported.
    """
    if abs(dvm.accommodation - 1.0) > 0.0:
        raise MarchedStateError(
            "the marched-state reconstruction requires accommodation = 1 "
            f"(got {dvm.accommodation})"
        )
    g = dvm.g
    nz = dvm.nz
    inv_vc = np.where(dvm.V_col > 0.0, 1.0 / np.maximum(dvm.V_col, 1e-300), 0.0)
    inv_va = np.where(dvm.V_ann > 0.0, 1.0 / np.maximum(dvm.V_ann, 1e-300), 0.0)

    Y_c = dvm.f_c - known.column
    Y_a = dvm.f_a - known.annulus

    # Ion Maxwellians, the same array substep B built its births from.
    M_i = known.M_i
    M_wall = np.broadcast_to(dvm.M_wall, (nz, g.nvz, g.nvp))

    # Mesh coupling: only the two cells flanking the mesh face see it, and
    # each sees one half of the v_z axis (the half that crosses the face
    # toward it). Transcribed from ``_march``: forward sweep blocks into
    # cell mesh_face - 1, backward sweep into cell mesh_face.
    k_c = np.zeros(nz)
    k_a = np.zeros(nz)
    w_out = np.zeros((nz, g.nvz, g.nvp))
    fi = dvm.mesh_face
    if 0 <= fi <= nz and dvm.transparency < 1.0:
        blocked = 1.0 - dvm.transparency
        if 0 <= fi - 1 < nz:
            k_c[fi - 1] = blocked * dvm.face_c[fi] * dt * inv_vc[fi - 1]
            k_a[fi - 1] = blocked * dvm.face_a[fi] * dt * inv_va[fi - 1]
            w_out[fi - 1] = _outgoing_weight(g, +1)
        if 0 <= fi < nz:
            k_c[fi] = blocked * dvm.face_c[fi] * dt * inv_vc[fi]
            k_a[fi] = blocked * dvm.face_a[fi] * dt * inv_va[fi]
            w_out[fi] = _outgoing_weight(g, -1)

    def solve_pair(Y, w1, s1, w2, s2, k):
        """Solve the per-cell 2x2 for the two birth coefficients.

        Every argument is already broadcast to ``(nz, nvz, nvp)``; ``w1``
        and ``w2`` are the two linear functionals, ``s1`` and ``s2`` the two
        birth spectra, ``k`` the per-cell prefactor of the second channel.
        """
        a11 = 1.0 + dt * (w1 * s1).sum(axis=(1, 2))
        a12 = dt * (w1 * s2).sum(axis=(1, 2))
        a21 = k * (w2 * s1).sum(axis=(1, 2))
        a22 = 1.0 + k * (w2 * s2).sum(axis=(1, 2))
        r1 = dt * (w1 * Y).sum(axis=(1, 2))
        r2 = k * (w2 * Y).sum(axis=(1, 2))
        det = a11 * a22 - a12 * a21
        return (r1 * a22 - a12 * r2) / det, (a11 * r2 - a21 * r1) / det

    # Column: spectrum 1 is the local ion Maxwellian (cx + elastic rebirth),
    # spectrum 2 the wall spectrum (anode-mesh re-emission).
    a_c, b_c = solve_pair(Y_c, nu_c_react, M_i, w_out, M_wall, k_c)
    f_c_m = Y_c - a_c[:, None, None] * M_i - b_c[:, None, None] * dvm.M_wall

    # Annulus: both births carry the wall spectrum, but the two functionals
    # (the radial-wall rate moment and the mesh outgoing-flux moment) are
    # independent, so the system is still non-singular.
    nuw = np.broadcast_to(dvm.nuw[:, None, :], (nz, g.nvz, g.nvp))
    c_a, d_a = solve_pair(Y_a, nuw, M_wall, w_out, M_wall, k_a)
    f_a_m = Y_a - (c_a + d_a)[:, None, None] * dvm.M_wall

    # --- ledger checks: six independent statements about the recovery.
    N_wall = c_a * dvm.V_ann
    N_ce = a_c * dvm.V_col
    N_mesh = b_c * dvm.V_col + d_a * dvm.V_ann
    N_ion = dt * (
        known.nu_ion[:, None, None] * f_c_m * dvm.V_col[:, None, None]
    ).sum()
    out = {}
    for zone, f_m, face in (
        ("c", f_c_m, dvm.face_c),
        ("a", f_a_m, dvm.face_a),
    ):
        for end, cell, sign in ((-1, 0, -1), (+1, nz - 1, +1)):
            out[(zone, end)] = (
                f_m[cell] * _outgoing_weight(g, sign) * face[0 if end < 0 else -1] * dt
            )
    out_L = float(out[("c", -1)].sum() + out[("a", -1)].sum())
    out_R = float(out[("c", +1)].sum() + out[("a", +1)].sum())

    checks = {
        "wall": (float(N_wall.sum()), ledger["loss_wall"]),
        "cx_elastic": (
            float(N_ce.sum()),
            ledger["birth_cx"] + ledger["birth_elastic"],
        ),
        "mesh": (float(N_mesh.sum()), ledger["loss_mesh_blocked"]),
        "ionization": (float(N_ion), ledger["loss_ionization"]),
        "end_out_L": (out_L, ledger["loss_end_out_L"]),
        "end_out_R": (out_R, ledger["loss_end_out_R"]),
    }
    worst = 0.0
    worst_name = "none"
    scale = max(abs(ledger["inventory_before"]), 1e-300)
    for name, (got, want) in checks.items():
        rel = abs(got - want) / max(abs(want), 1e-6 * scale, 1e-300)
        if rel > worst:
            worst, worst_name = rel, name
    if worst > ROUNDOFF_REL:
        raise MarchedStateError(
            f"marched-state reconstruction disagrees with the ledger: "
            f"{worst_name} off by {worst:.3e}"
        )
    return SimpleNamespace(
        f_c=f_c_m,
        f_a=f_a_m,
        N_wall=N_wall,
        N_mesh_c=b_c * dvm.V_col,
        N_mesh_a=d_a * dvm.V_ann,
        out=out,
        check_worst=worst,
        check_worst_name=worst_name,
    )


def surface_energy_tallies(dvm, march, dt):
    """Return the per-update surface energy budget [erg].

    Incident energy is the kinetic energy the marched distribution delivers
    to each surface; returned energy is what that surface re-emits. The
    radial wall and the anode mesh re-emit at the wall spectrum, so their
    returned energy is the tallied particle count times the mean energy of
    that spectrum. The end walls return through the engine's own pending
    buffers, which are public, so their returned energy is read directly
    from the buffers the update just filled; the pumped share leaves the
    domain and is reported separately rather than as deposition.
    """
    g = dvm.g
    e_bin = _kinetic_energy_per_bin(g)
    e_wall_spec = float((dvm.M_wall * e_bin).sum())

    nuw = dvm.nuw[:, None, :]
    incident_wall = float(
        (nuw * march.f_a * dt * dvm.V_ann[:, None, None] * e_bin[None, :, :]).sum()
    )
    returned_wall = float(march.N_wall.sum()) * e_wall_spec

    # The mesh tally is a flux-weighted count per cell, not a per-bin array,
    # so its incident energy is formed here from the same outgoing-flux
    # moment of the marched state that produced the count.
    incident_mesh = 0.0
    fi = dvm.mesh_face
    nz = dvm.nz
    if 0 <= fi <= nz and dvm.transparency < 1.0:
        blocked = 1.0 - dvm.transparency
        for cell, sign, face_c, face_a in (
            (fi - 1, +1, dvm.face_c[fi], dvm.face_a[fi]),
            (fi, -1, dvm.face_c[fi], dvm.face_a[fi]),
        ):
            if not 0 <= cell < nz:
                continue
            w = _outgoing_weight(g, sign)
            incident_mesh += blocked * dt * (
                float((march.f_c[cell] * w * e_bin).sum()) * face_c
                + float((march.f_a[cell] * w * e_bin).sum()) * face_a
            )
    returned_mesh = (
        float(march.N_mesh_c.sum() + march.N_mesh_a.sum()) * e_wall_spec
    )

    incident_end = 0.0
    pumped_end = 0.0
    for (zone, end), counts in march.out.items():
        e_out = float((counts * e_bin).sum())
        incident_end += e_out
        pumped_end += (dvm.s_L if end < 0 else dvm.s_R) * e_out
    returned_end = float(
        sum(
            (buf * e_bin).sum()
            for buf in (
                dvm.pend_L_c,
                dvm.pend_R_c,
                dvm.pend_L_a,
                dvm.pend_R_a,
            )
        )
    )
    return {
        "wall_incident": incident_wall,
        "wall_returned": returned_wall,
        "wall_deposited": incident_wall - returned_wall,
        "mesh_incident": incident_mesh,
        "mesh_returned": returned_mesh,
        "mesh_deposited": incident_mesh - returned_mesh,
        "end_incident": incident_end,
        "end_returned": returned_end,
        "end_pumped": pumped_end,
        "end_deposited": incident_end - pumped_end - returned_end,
    }


# ---------------------------------------------------------------- background


class Background:
    """Frozen plasma trajectory, sampled on the anchor clock.

    The saved run holds the packed state every ~10 us; the anchor clock is
    5 us, so the packed state is interpolated LINEARLY in time between saved
    samples and the derived quantities (``nu_ion`` and the source ledger) are
    then computed from the interpolated state by the SOLVER'S OWN methods --
    ``_dvm_ionization_frequency`` and ``_kinetic_channel_rates``, the same
    two calls production's ``_dvm_advance`` makes. Nothing about the source
    ledger is re-derived here.

    The table is built once per interval and shared by every cadence arm, so
    all arms see the identical background function of time.
    """

    def __init__(self, path, t_start_s, t_end_s, step_s):
        self.step_s = float(step_s)
        self.t0 = float(t_start_s)
        n = int(round((t_end_s - t_start_s) / step_s))
        if abs(n * step_s - (t_end_s - t_start_s)) > 1e-15:
            raise ValueError("background window is not a multiple of the step")
        self.times = t_start_s + step_s * np.arange(n + 1)
        with h5py.File(path, "r") as f:
            params = json.loads(f.attrs["params_json"])
            flags = json.loads(f.attrs["flags_json"])
            t_saved = f["time"][:]
            y_saved = f["y"]
            Ts_saved = f["cathode_diagnostics"]["T_s_surface"][:]
            lo = max(int(np.searchsorted(t_saved, self.times[0]) - 2), 0)
            hi = min(
                int(np.searchsorted(t_saved, self.times[-1]) + 2),
                t_saved.size,
            )
            t_win = t_saved[lo:hi]
            y_win = y_saved[lo:hi, :]
            Ts_win = Ts_saved[lo:hi]
        self.params = params
        self.flags = flags
        self.sim = LAPDSim1D(dict(params), dict(flags))
        self.geometry = self.sim.geometry

        self.n_i = []
        self.Ti_eV = []
        self.u_i = []
        self.nu_ion = []
        self.sources = []
        self.T_s_K = []
        self.nn_col = []
        self.nn_ann = []
        for t in self.times:
            # Linear interpolation of the PACKED state between the two
            # bracketing saved samples: one well-defined background function
            # of time, sampled identically by every cadence arm.
            j = int(np.clip(np.searchsorted(t_win, t) - 1, 0, t_win.size - 2))
            span = t_win[j + 1] - t_win[j]
            w = 0.0 if span <= 0.0 else (t - t_win[j]) / span
            y = (1.0 - w) * y_win[j] + w * y_win[j + 1]
            self.sim._time = float(t)
            self.sim._y = y
            self.sim._state = self.sim._unpack(y)
            self.sim._derived = derive_state(
                self.sim._state, self.sim._floors, self.sim._ion_mass_g
            )
            state = self.sim.state
            derived = self.sim.derived
            self.n_i.append(np.asarray(state.n, dtype=float).copy())
            self.Ti_eV.append(np.asarray(derived.Ti, dtype=float).copy())
            self.u_i.append(np.asarray(derived.u, dtype=float).copy())
            self.nu_ion.append(
                self.sim._dvm_ionization_frequency(state, derived).copy()
            )
            rates = self.sim._kinetic_channel_rates(state, derived, float(t))
            self.sources.append(
                {
                    "puff": np.asarray(rates["puff"], dtype=float).copy(),
                    "recombination": np.asarray(rates["rec"], dtype=float).copy(),
                    "anode": np.asarray(rates["anode"], dtype=float).copy(),
                    "cathode_face": float(rates["cath"]),
                    "collector_face": float(rates["coll"]),
                }
            )
            self.T_s_K.append(float(np.interp(t, t_win, Ts_win)))
            self.nn_col.append(np.asarray(state.nn, dtype=float).copy())
            self.nn_ann.append(np.asarray(state.nn_a, dtype=float).copy())

    def index(self, t):
        k = int(round((t - self.t0) / self.step_s))
        if not 0 <= k < self.times.size:
            raise IndexError(f"background has no sample at t={t}")
        return k


def build_production_dvm(sim, nvz=48, nvp=12):
    """Build a TransientDVM with production's own configuration choices.

    Mirrors ``LAPDSim1D._configure_kinetic_dvm`` and reuses the solver's own
    pump-sticking helper, so the transparency, mesh face and end sticking
    are production's rather than this file's.
    """
    anode_faces = np.asarray(
        getattr(sim.geometry, "anode_face_indices", ()), dtype=int
    )
    return TransientDVM(
        geometry=sim.geometry,
        nvz=nvz,
        nvp=nvp,
        accommodation=float(
            sim._input_dict.get("neutral_kinetic_dvm_accommodation", 1.0)
        ),
        elastic_model=str(
            sim._input_dict.get("neutral_kinetic_dvm_elastic", "phelps_iso")
        ),
        transparency=1.0 - float(sim._input_dict.get("eta", 0.358)),
        mesh_face=int(anode_faces[0]) if anode_faces.size else -999,
        s_L=sim._dvm_end_sticking("S_pump_L"),
        s_R=sim._dvm_end_sticking("S_pump_R"),
    )


def ion_maxwellians(dvm, Ti_eV, u_i):
    """Per-cell local ion Maxwellians, exactly as substep B builds them."""
    g = dvm.g
    M_i = np.empty((dvm.nz, g.nvz, g.nvp))
    Ti = np.asarray(Ti_eV, dtype=float)
    u = np.asarray(u_i, dtype=float)
    for i in range(dvm.nz):
        M_i[i] = g.maxwellian(max(float(Ti[i]), 0.02), float(u[i]))
    return M_i


def known_births(dvm, dt, src, M_i, T_s_K, nu_ion):
    """Return the birth contributions that do not depend on the march.

    These are the external ledger channels: puff, volume recombination,
    anode collection and the two end faces. Built with the engine's own
    spectra so the reconstruction subtracts exactly what ``update`` added.
    """
    g = dvm.g
    nz = dvm.nz
    inv_vc = np.where(dvm.V_col > 0.0, 1.0 / np.maximum(dvm.V_col, 1e-300), 0.0)
    inv_va = np.where(dvm.V_ann > 0.0, 1.0 / np.maximum(dvm.V_ann, 1e-300), 0.0)
    column = np.zeros((nz, g.nvz, g.nvp))
    annulus = np.zeros((nz, g.nvz, g.nvp))
    rec = np.asarray(src["recombination"], dtype=float) * dt
    anode = np.asarray(src["anode"], dtype=float) * dt
    puff = np.asarray(src["puff"], dtype=float) * dt
    column += (rec * inv_vc)[:, None, None] * M_i
    column += (anode * inv_vc)[:, None, None] * dvm.M_wall[None, :, :]
    annulus += (puff * inv_va)[:, None, None] * dvm.M_cold
    cath = float(src["cathode_face"]) * dt
    coll = float(src["collector_face"]) * dt
    if cath:
        column[0] += cath * inv_vc[0] * g.half_flux_spectrum(T_s_K, +1)
    if coll:
        column[-1] += coll * inv_vc[-1] * g.half_flux_spectrum(dvm.T_wall_K, -1)
    return SimpleNamespace(
        column=column,
        annulus=annulus,
        M_i=M_i,
        nu_ion=np.asarray(nu_ion, dtype=float),
    )


# ------------------------------------------------------------------- part 1


def run_cadence_arm(dvm, bg, seed_snapshot, t0, t1, cadence_s, collect):
    """Advance one clock across an interval and accumulate every readout."""
    dvm.restore(seed_snapshot)
    n_ticks = int(round((t1 - t0) / cadence_s))
    if abs(n_ticks * cadence_s - (t1 - t0)) > 1e-15:
        raise ValueError("interval is not a multiple of the cadence")
    acc = {
        "wall_deposited": 0.0,
        "wall_returned": 0.0,
        "wall_incident": 0.0,
        "mesh_deposited": 0.0,
        "mesh_returned": 0.0,
        "end_deposited": 0.0,
        "end_returned": 0.0,
        "end_incident": 0.0,
        "end_pumped": 0.0,
        "S_plasma": 0.0,
        "M_plasma": 0.0,
        "Ei_plasma": 0.0,
        "ionized": 0.0,
        "check_worst": 0.0,
        "ledger_worst": 0.0,
    }
    for j in range(n_ticks):
        k = bg.index(t0 + (j + 1) * cadence_s)
        src = bg.sources[k]
        ledger = dvm.update(
            cadence_s,
            n_i=bg.n_i[k],
            Ti_eV=bg.Ti_eV[k],
            u_i=bg.u_i[k],
            nu_ion=bg.nu_ion[k],
            sources=src,
            T_s_K=bg.T_s_K[k],
        )
        res = ledger_residual(ledger)
        acc["ledger_worst"] = max(
            acc["ledger_worst"],
            abs(res["distribution_rel"]),
            abs(res["domain_rel"]),
        )
        if collect:
            nu_cx, nu_el = dvm.collision_frequencies(
                bg.n_i[k], bg.Ti_eV[k], bg.u_i[k]
            )
            known = known_births(
                dvm,
                cadence_s,
                src,
                ion_maxwellians(dvm, bg.Ti_eV[k], bg.u_i[k]),
                bg.T_s_K[k],
                bg.nu_ion[k],
            )
            march = reconstruct_marched(
                dvm, cadence_s, known, nu_cx + nu_el, ledger
            )
            tal = surface_energy_tallies(dvm, march, cadence_s)
            for key in (
                "wall_deposited",
                "wall_returned",
                "wall_incident",
                "mesh_deposited",
                "mesh_returned",
                "end_deposited",
                "end_returned",
                "end_incident",
                "end_pumped",
            ):
                acc[key] += tal[key]
            acc["check_worst"] = max(acc["check_worst"], march.check_worst)
        acc["ionized"] += ledger["loss_ionization"]
        acc["S_plasma"] += float((dvm.S_transfer * dvm.V_col).sum()) * cadence_s
        acc["M_plasma"] += float((dvm.M_transfer * dvm.V_col).sum()) * cadence_s
        acc["Ei_plasma"] += float((dvm.Ei_transfer * dvm.V_col).sum()) * cadence_s
    return acc


def interval_readouts(dvm, acc, g):
    """Assemble the spec's readout list from the final state and the sums."""
    n_col, p_col, e_col = zone_moments(dvm.f_c, dvm.V_col, g)
    n_ann, p_ann, e_ann = zone_moments(dvm.f_a, dvm.V_ann, g)
    out = {
        "column particle inventory [atoms]": n_col,
        "annulus particle inventory [atoms]": n_ann,
        "neutral axial momentum [g cm/s]": p_col + p_ann,
        "neutral kinetic energy [erg]": e_col + e_ann,
        "radial-wall energy deposited [erg]": acc["wall_deposited"],
        "radial-wall energy returned [erg]": acc["wall_returned"],
        "anode-mesh energy deposited [erg]": acc["mesh_deposited"],
        "end-wall energy deposited [erg]": acc["end_deposited"],
        "end-wall energy returned [erg]": acc["end_returned"],
        "plasma particle source integral [atoms]": acc["S_plasma"],
        "plasma momentum source integral [g cm/s]": acc["M_plasma"],
        "plasma energy source integral [erg]": acc["Ei_plasma"],
    }
    for th in HOT_TAIL_EV:
        out[f"hot-tail inventory > {th} eV [atoms]"] = hot_tail_inventory(
            dvm.f_c, dvm.V_col, g, th
        )
    return out


def run_part1(path, log):
    """Run the multirate-convergence study over the three intervals."""
    results = {}
    for label, t0_ms, t1_ms, note in INTERVALS:
        t0 = t0_ms * 1e-3
        t1 = t1_ms * 1e-3
        log(f"[part1] {label}: {t0_ms:.3f} -> {t1_ms:.3f} ms ({note})")
        tic = time.perf_counter()
        bg = Background(path, t0 - BURN_IN_S, t1, ANCHOR_S)
        log(f"[part1]   background table: {bg.times.size} samples in "
            f"{time.perf_counter() - tic:.1f} s")

        dvm = build_production_dvm(bg.sim)
        k0 = bg.index(t0 - BURN_IN_S)
        dvm.seed_from_density(
            np.maximum(bg.nn_col[k0], 0.0), np.maximum(bg.nn_ann[k0], 0.0)
        )
        tic = time.perf_counter()
        run_cadence_arm(
            dvm, bg, dvm.snapshot(), t0 - BURN_IN_S, t0, ANCHOR_S, collect=False
        )
        seed = dvm.snapshot()
        log(f"[part1]   burn-in {BURN_IN_S * 1e6:.0f} us at the anchor clock: "
            f"{time.perf_counter() - tic:.1f} s")

        hot_start = {
            th: hot_tail_inventory(dvm.f_c, dvm.V_col, dvm.g, th)
            for th in HOT_TAIL_EV
        }
        arms = {}
        for cad in CADENCES_S:
            tic = time.perf_counter()
            acc = run_cadence_arm(dvm, bg, seed, t0, t1, cad, collect=True)
            read = interval_readouts(dvm, acc, dvm.g)
            for th in HOT_TAIL_EV:
                key = f"hot-tail decay ratio > {th} eV [-]"
                start = hot_start[th]
                read[key] = (
                    read[f"hot-tail inventory > {th} eV [atoms]"] / start
                    if start > 0.0
                    else float("nan")
                )
            read["_nn_col"] = dvm.column_density().copy()
            arms[cad] = {
                "readouts": read,
                "ticks": int(round((t1 - t0) / cad)),
                "check_worst": acc["check_worst"],
                "ledger_worst": acc["ledger_worst"],
                "wall_s": time.perf_counter() - tic,
            }
            log(f"[part1]   clock {cad * 1e6:6.1f} us: "
                f"{arms[cad]['ticks']:4d} ticks, "
                f"ledger {acc['ledger_worst']:.2e}, "
                f"recon {acc['check_worst']:.2e}, "
                f"{arms[cad]['wall_s']:.1f} s")
        results[label] = {
            "window_ms": (t0_ms, t1_ms),
            "note": note,
            "arms": arms,
            "hot_start": hot_start,
        }
    return results


# ------------------------------------------------------------------- part 2


def uniform_tube(nz, dz_cm, Rp_cm, Rm_cm):
    """A strictly uniform coaxial tube carrying one cell's local geometry."""
    dz = np.full(nz, float(dz_cm))
    Rp = np.full(nz, float(Rp_cm))
    Rm = np.full(nz, float(Rm_cm))
    return SimpleNamespace(
        cells=nz,
        length_cm=dz,
        Rp_cm=Rp,
        Rm_cm=Rm,
        plasma_volume_cm3=np.pi * Rp**2 * dz,
        neutral_volume_cm3=np.pi * Rm**2 * dz,
    )


def select_cells(bg, k):
    """Pick the three representative cells by stated deterministic rules.

    The three axes a velocity grid has to span are a cold wall-fed
    population (fine resolution near v = 0), a hot charge-exchange tail
    (wide velocity extent, which then under-resolves the cold core), and a
    large ion drift (an off-centre birth Maxwellian on the v_z axis). The
    rules below pick the extreme of each axis over the plateau background:

    * cold, wall-fed -- the lowest-``Ti`` cell of the ``column`` role;
    * mid-column with the CX tail -- the highest-``Ti`` cell of that role;
    * source-adjacent -- the plasma-active cell of largest ``|u_i|``.

    Selecting on ``n_i`` instead does not discriminate on this background:
    the plateau column density varies by well under a factor of three while
    ``Ti`` varies by 5.6x and ``|u_i|`` by more than an order of magnitude.
    """
    geom = bg.geometry
    roles = [
        r.decode() if isinstance(r, bytes) else str(r)
        for r in np.asarray(geom.cell_role)
    ]
    Ti = bg.Ti_eV[k]
    active = np.asarray(geom.plasma_active, dtype=bool)
    column = np.array(
        [i for i, r in enumerate(roles) if r == "column"], dtype=int
    )
    live = np.flatnonzero(active)
    cold = int(column[np.argmin(Ti[column])])
    tail = int(column[np.argmax(Ti[column])])
    drift = int(live[np.argmax(np.abs(bg.u_i[k][live]))])
    return (
        ("cold wall-fed (lowest-Ti column cell)", cold),
        ("mid-column with the CX tail (highest-Ti column cell)", tail),
        ("source-adjacent (largest |u_i| plasma-active cell)", drift),
    )


def relax_local(bg, k, cell, nvz, nvp, updates=VGRID_UPDATES, dt=VGRID_DT_S):
    """Relax a closed tube at one cell's plasma condition to a fixed point.

    The tube is closed (no pumping, unit accommodation, no mesh), so the one
    non-conservative channel is ionization; it is balanced by re-injecting,
    through the recombination channel, exactly the particle count the
    PREVIOUS update's ledger booked as ionized. That gives the local problem
    a genuine steady state instead of a drain, and it is stable: with the
    loss a fraction ``a < 1`` of the inventory, the resulting lag-one
    recurrence has roots ``1`` and ``-a``. Balancing against the
    pre-update density instead would inject more than the implicit march
    destroys and diverge geometrically, which is what an earlier form of
    this function did.

    The rebirth carries the local ion Maxwellian, which is the spectrum the
    charge-exchange and elastic channels already re-emit at, so the balance
    introduces no spectrum this closure did not already contain. Every
    surface in the tube -- radial wall and both ends -- is a 300 K
    re-emitter, so the fixed point is a mixture of a wall-temperature
    population and the local ion Maxwellian: precisely the mixture the
    production velocity grid has to represent.
    """
    geom = bg.geometry
    dz = float(np.asarray(geom.length_cm)[cell])
    tube = uniform_tube(
        VGRID_TUBE_CELLS,
        dz,
        float(np.asarray(geom.Rp_cm)[cell]),
        float(np.asarray(geom.Rm_cm)[cell]),
    )
    dvm = TransientDVM(geometry=tube, nvz=nvz, nvp=nvp)
    nz = tube.cells
    nn_c = float(bg.nn_col[k][cell])
    nn_a = float(bg.nn_ann[k][cell])
    dvm.seed_from_density(np.full(nz, nn_c), np.full(nz, nn_a))
    n_i = np.full(nz, float(bg.n_i[k][cell]))
    Ti = np.full(nz, float(bg.Ti_eV[k][cell]))
    u_i = np.full(nz, float(bg.u_i[k][cell]))
    nu_ion = np.full(nz, float(bg.nu_ion[k][cell]))

    history = []
    prev_ionized = None
    for _ in range(updates):
        weight = dvm.column_density() * dvm.V_col
        total = float(weight.sum())
        if prev_ionized is None:
            # First tick only: no ledger to balance against yet.
            rec = nu_ion * weight
        else:
            rec = (prev_ionized / dt) * (
                weight / total if total > 0.0 else np.zeros_like(weight)
            )
        ledger = dvm.update(
            dt,
            n_i=n_i,
            Ti_eV=Ti,
            u_i=u_i,
            nu_ion=nu_ion,
            sources={"recombination": rec},
        )
        prev_ionized = ledger["loss_ionization"]
        history.append(local_moments(dvm, ledger, dt))

    # Reported values are means over the last window, and convergence is the
    # window-to-window change. Both exist because the ionization balance is
    # a lag-one recurrence whose second root is a damped alternation: the
    # state moments are unaffected (they settle to ~1e-6 within a window),
    # but any readout formed as a near-cancellation between the loss and the
    # balancing rebirth alternates, and a last-sample readout would report
    # that alternation as non-convergence.
    def window_mean(window):
        return {
            key: float(np.mean([h[key] for h in window])) for key in window[0]
        }

    last = window_mean(history[-VGRID_SETTLE_TAIL:])
    prev = window_mean(history[-2 * VGRID_SETTLE_TAIL : -VGRID_SETTLE_TAIL])
    settle = max(
        abs(last[key] - prev[key]) / max(abs(last[key]), 1e-300)
        for key in last
    )
    out = dict(last)
    out["_settle"] = settle
    return out


def local_moments(dvm, ledger, dt):
    """Moments of the local state, plus the plasma source integrands.

    The particle channel is reported as the GROSS ionization sink the ledger
    booked, not as the net ``S_transfer``: under this study's ionization
    balance the net is the difference of two nearly equal numbers and
    carries only cancellation noise, whereas the gross sink is the zeroth
    moment of ``nu_ion f`` over the marched state and is exactly the
    grid-sensitive quantity the readout is asking about. The momentum and
    energy channels have no such cancellation and are the accumulators the
    solver reads.
    """
    g = dvm.g
    n_c = dvm.column_density()
    n_a = dvm.annulus_density()
    total_c = float((n_c * dvm.V_col).sum())
    total_a = float((n_a * dvm.V_ann).sum())
    t_par, t_perp = directional_temperatures_eV(dvm.f_c, g)
    weights = n_c * dvm.V_col
    wsum = max(float(weights.sum()), 1e-300)
    _, p_col, _ = zone_moments(dvm.f_c, dvm.V_col, g)
    volume = max(float(dvm.V_col.sum()), 1e-300)
    return {
        "column density [cm^-3]": float(n_c.mean()),
        "annulus/column density ratio [-]": (
            total_a / total_c if total_c > 0.0 else float("nan")
        ),
        "axial momentum per atom [g cm/s]": (
            p_col / total_c if total_c > 0.0 else float("nan")
        ),
        "T_par [eV]": float((t_par * weights).sum() / wsum),
        "T_perp [eV]": float((t_perp * weights).sum() / wsum),
        "ionization sink [cm^-3 s^-1]": (
            ledger["loss_ionization"] / (volume * dt)
        ),
        "plasma momentum source [g cm^-2 s^-2]": float(dvm.M_transfer.mean()),
        "plasma energy source [erg cm^-3 s^-1]": float(dvm.Ei_transfer.mean()),
    }


def measure_update_cost(bg, k, nvz, nvp, repeats=COST_REPEATS):
    """Time one production-geometry DVM update at a given velocity grid."""
    dvm = build_production_dvm(bg.sim, nvz=nvz, nvp=nvp)
    dvm.seed_from_density(
        np.maximum(bg.nn_col[k], 0.0), np.maximum(bg.nn_ann[k], 0.0)
    )
    src = bg.sources[k]
    kwargs = dict(
        n_i=bg.n_i[k],
        Ti_eV=bg.Ti_eV[k],
        u_i=bg.u_i[k],
        nu_ion=bg.nu_ion[k],
        sources=src,
        T_s_K=bg.T_s_K[k],
    )
    dvm.update(ANCHOR_S, **kwargs)  # warm-up
    samples = []
    for _ in range(repeats):
        tic = time.perf_counter()
        dvm.update(ANCHOR_S, **kwargs)
        samples.append(time.perf_counter() - tic)
    return {
        "median_ms": 1e3 * statistics.median(samples),
        "min_ms": 1e3 * min(samples),
        "max_ms": 1e3 * max(samples),
        "bins": nvz * nvp,
    }


def run_part2(path, log):
    """Run the velocity-grid adequacy study plus the cost scaling."""
    t_plateau = INTERVALS[1][1] * 1e-3
    bg = Background(path, t_plateau, t_plateau + ANCHOR_S, ANCHOR_S)
    k = bg.index(t_plateau)
    cells = select_cells(bg, k)
    log(f"[part2] representative cells at t={t_plateau * 1e3:.3f} ms:")
    conditions = {}
    for name, idx in cells:
        conditions[name] = {
            "cell": idx,
            "z_cm": float(np.asarray(bg.geometry.z_cm)[idx]),
            "n_i": float(bg.n_i[k][idx]),
            "Ti_eV": float(bg.Ti_eV[k][idx]),
            "u_i": float(bg.u_i[k][idx]),
            "nu_ion": float(bg.nu_ion[k][idx]),
            "nn_col": float(bg.nn_col[k][idx]),
            "nn_ann": float(bg.nn_ann[k][idx]),
        }
        log(f"[part2]   {name}: cell {idx}, z={conditions[name]['z_cm']:.1f} cm, "
            f"n_i={conditions[name]['n_i']:.3e}, Ti={conditions[name]['Ti_eV']:.3f} eV, "
            f"nu_ion={conditions[name]['nu_ion']:.3e} 1/s")

    moments = {}
    for name, idx in cells:
        moments[name] = {}
        for nvz, nvp in VGRIDS:
            tic = time.perf_counter()
            moments[name][(nvz, nvp)] = relax_local(bg, k, idx, nvz, nvp)
            log(f"[part2]   {name} @ {nvz}x{nvp}: settle "
                f"{moments[name][(nvz, nvp)]['_settle']:.2e}, "
                f"{time.perf_counter() - tic:.1f} s")

    costs = {}
    for nvz, nvp in VGRIDS:
        costs[(nvz, nvp)] = measure_update_cost(bg, k, nvz, nvp)
        log(f"[part2]   cost {nvz}x{nvp}: "
            f"{costs[(nvz, nvp)]['median_ms']:.2f} ms")
    return conditions, moments, costs


# ------------------------------------------------------------------ reports


def _rel(value, ref):
    if not np.isfinite(value) or not np.isfinite(ref):
        return float("nan")
    if ref == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return (value - ref) / abs(ref)


def _header(lines, command, title):
    lines.append(title)
    lines.append("=" * 78)
    lines.append(f"accepted command line: {command}")
    lines.append(f"python: {platform.python_version()}  numpy: {np.__version__}")
    lines.append(f"platform: {platform.platform()}")
    lines.append("=" * 78)


def write_cadence_artifact(path, command, results):
    lines = []
    _header(
        lines,
        command,
        "E1 multirate convergence: neutral-clock cadence vs the 5 us anchor",
    )
    lines.append(
        "Anchor clock 5 us. Every number is a relative deviation "
        "(X_cadence - X_anchor)/|X_anchor| of the readout named on the row."
    )
    lines.append(
        "The engine is the shipped TransientDVM on the production nx=240 "
        "geometry; all arms of an interval start from ONE shared kinetic "
        "state (500 us burn-in at the anchor clock) and see ONE shared "
        "background table."
    )
    lines.append(
        "Coupling convention is production's: the plasma coefficients are "
        "those at the END of each neutral tick, applied across the tick."
    )
    lines.append("")
    for label, res in results.items():
        t0, t1 = res["window_ms"]
        lines.append("-" * 78)
        lines.append(f"INTERVAL {label}: {t0:.3f} -> {t1:.3f} ms")
        lines.append(f"  {res['note']}")
        arms = res["arms"]
        cads = [c for c in CADENCES_S if c != ANCHOR_S]
        ref = arms[ANCHOR_S]["readouts"]
        lines.append("")
        lines.append(
            f"  {'readout':<46}{'anchor value':>15}"
            + "".join(f"{c * 1e6:>12.0f} us" for c in cads)
        )
        worst = {c: (0.0, "") for c in cads}
        for key in ref:
            if key.startswith("_"):
                continue
            row = f"  {key:<46}{ref[key]:>15.6e}"
            for c in cads:
                rel = _rel(arms[c]["readouts"][key], ref[key])
                row += f"{rel:>15.3e}"
                if np.isfinite(rel) and abs(rel) > abs(worst[c][0]):
                    worst[c] = (rel, key)
            lines.append(row)
        # profile deviation of the column density
        row = f"  {'column density profile, max |rel dev| [-]':<46}{'':>15}"
        for c in cads:
            a = arms[c]["readouts"]["_nn_col"]
            b = ref["_nn_col"]
            dev = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
            row += f"{dev:>15.3e}"
        lines.append(row)
        lines.append("")
        lines.append(f"  {'ticks in the interval':<46}{arms[ANCHOR_S]['ticks']:>15d}"
                     + "".join(f"{arms[c]['ticks']:>15d}" for c in cads))
        lines.append("  worst relative deviation, by cadence:")
        for c in cads:
            lines.append(
                f"    {c * 1e6:6.1f} us: {worst[c][0]:+.3e}  ({worst[c][1]})"
            )
        lines.append("  integrity, worst over the interval (all arms):")
        for c in CADENCES_S:
            lines.append(
                f"    {c * 1e6:6.1f} us: ledger residual "
                f"{arms[c]['ledger_worst']:.3e}, marched-state "
                f"reconstruction vs ledger {arms[c]['check_worst']:.3e}"
            )
        lines.append("")
    lines.append("=" * 78)
    lines.append(
        "The marched-state reconstruction is checked on EVERY update against "
        "six independent ledger statements (radial wall, cx+elastic, mesh, "
        "ionization, both end outflows); the study aborts if any exceeds "
        f"{ROUNDOFF_REL:.0e}."
    )
    Path(path).write_text("\n".join(lines) + "\n")


def write_vgrid_artifact(path, command, conditions, moments, costs):
    lines = []
    _header(
        lines,
        command,
        "E1 velocity-grid adequacy: moment error and per-update cost vs grid",
    )
    lines.append(
        f"Anchor grid {VGRID_ANCHOR[0]}x{VGRID_ANCHOR[1]}. Moment rows are "
        "relative deviations from the anchor of the relaxed local fixed "
        "point; the local problem is a closed uniform tube "
        f"({VGRID_TUBE_CELLS} cells) carrying the named background cell's "
        "plasma condition, with ionization balanced tick by tick by an "
        "equal-rate recombination rebirth."
    )
    lines.append(
        f"Relaxation: {VGRID_UPDATES} updates at {VGRID_DT_S * 1e6:.0f} us "
        f"({VGRID_UPDATES * VGRID_DT_S * 1e3:.1f} ms); the settle figure is "
        f"the largest relative change over the last {VGRID_SETTLE_TAIL} "
        "updates."
    )
    lines.append("")
    lines.append("-" * 78)
    lines.append("REPRESENTATIVE CELLS (frozen background, plateau)")
    for name, cond in conditions.items():
        lines.append(f"  {name}")
        lines.append(
            f"    cell {cond['cell']}, z = {cond['z_cm']:.1f} cm, "
            f"n_i = {cond['n_i']:.4e} cm^-3, Ti = {cond['Ti_eV']:.4f} eV, "
            f"u_i = {cond['u_i']:.4e} cm/s"
        )
        lines.append(
            f"    nu_ion = {cond['nu_ion']:.4e} 1/s, "
            f"nn_col = {cond['nn_col']:.4e} cm^-3, "
            f"nn_ann = {cond['nn_ann']:.4e} cm^-3"
        )
    lines.append("")
    grids = [g for g in VGRIDS if g != VGRID_ANCHOR]
    for name in conditions:
        lines.append("-" * 78)
        lines.append(f"MOMENT ERROR vs the anchor: {name}")
        ref = moments[name][VGRID_ANCHOR]
        lines.append(
            f"  {'moment':<44}{'anchor value':>15}"
            + "".join(f"{a}x{b:<3}".rjust(15) for a, b in grids)
        )
        for key in ref:
            if key.startswith("_"):
                continue
            row = f"  {key:<44}{ref[key]:>15.6e}"
            for g in grids:
                row += f"{_rel(moments[name][g][key], ref[key]):>15.3e}"
            lines.append(row)
        row = f"  {'settle (last-40-update drift) [-]':<44}{ref['_settle']:>15.3e}"
        for g in grids:
            row += f"{moments[name][g]['_settle']:>15.3e}"
        lines.append(row)
        lines.append("")
    lines.append("-" * 78)
    lines.append(
        "PER-UPDATE COST, production nx=240 geometry "
        f"({COST_REPEATS} repeats after one warm-up)"
    )
    lines.append(
        f"  {'grid':<12}{'bins':>8}{'median [ms]':>14}{'min [ms]':>12}"
        f"{'max [ms]':>12}{'vs 48x12':>12}{'vs bin count':>14}"
    )
    base = costs[(48, 12)]
    for nvz, nvp in VGRIDS:
        c = costs[(nvz, nvp)]
        lines.append(
            f"  {f'{nvz}x{nvp}':<12}{c['bins']:>8}{c['median_ms']:>14.3f}"
            f"{c['min_ms']:>12.3f}{c['max_ms']:>12.3f}"
            f"{c['median_ms'] / base['median_ms']:>12.2f}"
            f"{c['bins'] / base['bins']:>14.2f}"
        )
    lines.append("")
    lines.append("=" * 78)
    Path(path).write_text("\n".join(lines) + "\n")


def write_summary(path, command, results, conditions, moments, costs):
    lines = []
    lines.append("# E1 - multirate convergence and velocity-grid adequacy")
    lines.append("")
    lines.append(f"Accepted command line: `{command}`")
    lines.append("")
    lines.append(
        "Measurement only. No fit, no tune, no recommendation: the cadence "
        "and grid decisions are not made here. Per the evaluation package's "
        "standing rule, cadence is reported on ACCURACY alone -- the cost "
        "table below is a property of the velocity grid, not of the clock."
    )
    lines.append("")
    lines.append("## What was driven")
    lines.append("")
    lines.append(
        "The shipped `TransientDVM` (`physics/kinetic_dvm.py`), built with "
        "production's own configuration (unit accommodation, `phelps_iso` "
        "elastic, transparency `1 - eta`, the anode mesh face, and the end "
        "sticking from `_dvm_end_sticking`), on the frozen nx=240 background "
        "`es1_kn2z_promoted_nx240.h5`. The plasma trajectory and the source "
        "ledger come from the solver's own `_dvm_ionization_frequency` and "
        "`_kinetic_channel_rates`, evaluated on the saved packed state; the "
        "coupling convention is production's (`_dvm_advance`): the plasma "
        "coefficients are those at the END of each neutral tick, applied "
        "across the tick."
    )
    lines.append("")
    lines.append(
        "Wall-energy readouts need the post-march, pre-birth distribution, "
        "which `update` does not expose. It is recovered in closed form by "
        "inverting the birth additions (a per-cell 2x2 linear system), and "
        "the recovery is checked on EVERY update against six independent "
        "statements of the engine's own particle ledger. The study aborts if "
        f"any check exceeds {ROUNDOFF_REL:.0e}; none did."
    )
    lines.append("")
    lines.append("## Part 1 - multirate convergence")
    lines.append("")
    lines.append(
        "Anchor 5 us; arms 10 / 25 / 50 / 100 us. All arms of an interval "
        "start from one shared kinetic state (500 us burn-in at the anchor "
        "clock) and read one shared background table, so they differ only in "
        "clock. Full per-readout tables: "
        "`neutral_arch_e1_cadence_nx240.txt`."
    )
    lines.append("")
    lines.append("### Worst relative deviation from the anchor, by interval")
    lines.append("")
    cads = [c for c in CADENCES_S if c != ANCHOR_S]
    lines.append(
        "| interval | window [ms] | "
        + " | ".join(f"{c * 1e6:.0f} us" for c in cads)
        + " |"
    )
    lines.append("|---|---|" + "---|" * len(cads))
    worst_all = {}
    for label, res in results.items():
        ref = res["arms"][ANCHOR_S]["readouts"]
        cells = []
        for c in cads:
            worst = 0.0
            name = ""
            for key in ref:
                if key.startswith("_"):
                    continue
                rel = _rel(res["arms"][c]["readouts"][key], ref[key])
                if np.isfinite(rel) and abs(rel) > abs(worst):
                    worst, name = rel, key
            worst_all[(label, c)] = (worst, name)
            cells.append(f"{worst:+.2e}")
        t0, t1 = res["window_ms"]
        lines.append(
            f"| {label} | {t0:.2f}-{t1:.2f} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    lines.append("### Which readout carries the worst deviation")
    lines.append("")
    for (label, c), (worst, name) in worst_all.items():
        lines.append(f"- {label}, {c * 1e6:.0f} us: `{name}` at {worst:+.3e}")
    lines.append("")
    lines.append("### Per-readout deviation, all intervals")
    lines.append("")
    for label, res in results.items():
        ref = res["arms"][ANCHOR_S]["readouts"]
        lines.append(f"**{label}** ({res['note']})")
        lines.append("")
        lines.append(
            "| readout | anchor value | "
            + " | ".join(f"{c * 1e6:.0f} us" for c in cads)
            + " |"
        )
        lines.append("|---|---|" + "---|" * len(cads))
        for key in ref:
            if key.startswith("_"):
                continue
            row = f"| {key} | {ref[key]:.6e} | "
            row += " | ".join(
                f"{_rel(res['arms'][c]['readouts'][key], ref[key]):+.3e}"
                for c in cads
            )
            lines.append(row + " |")
        lines.append("")
    lines.append("## Part 2 - velocity-grid adequacy")
    lines.append("")
    lines.append(
        "Relaxed local fixed point on a closed uniform tube carrying each "
        "representative cell's plasma condition, compared against the "
        f"{VGRID_ANCHOR[0]}x{VGRID_ANCHOR[1]} anchor. Full tables: "
        "`neutral_arch_e1_vgrid_nx240.txt`."
    )
    lines.append("")
    grids = [g for g in VGRIDS if g != VGRID_ANCHOR]
    for name in conditions:
        ref = moments[name][VGRID_ANCHOR]
        lines.append(f"**{name}** "
                     f"(cell {conditions[name]['cell']}, "
                     f"n_i = {conditions[name]['n_i']:.3e} cm^-3, "
                     f"Ti = {conditions[name]['Ti_eV']:.3f} eV)")
        lines.append("")
        lines.append(
            "| moment | anchor value | "
            + " | ".join(f"{a}x{b}" for a, b in grids)
            + " |"
        )
        lines.append("|---|---|" + "---|" * len(grids))
        for key in ref:
            if key.startswith("_"):
                continue
            row = f"| {key} | {ref[key]:.6e} | "
            row += " | ".join(
                f"{_rel(moments[name][g][key], ref[key]):+.3e}" for g in grids
            )
            lines.append(row + " |")
        lines.append("")
    lines.append("### Per-update cost vs velocity grid (production nx=240)")
    lines.append("")
    lines.append("| grid | bins | median [ms] | min [ms] | max [ms] | "
                 "x 48x12 cost | x 48x12 bins |")
    lines.append("|---|---|---|---|---|---|---|")
    base = costs[(48, 12)]
    for nvz, nvp in VGRIDS:
        c = costs[(nvz, nvp)]
        lines.append(
            f"| {nvz}x{nvp} | {c['bins']} | {c['median_ms']:.3f} | "
            f"{c['min_ms']:.3f} | {c['max_ms']:.3f} | "
            f"{c['median_ms'] / base['median_ms']:.2f} | "
            f"{c['bins'] / base['bins']:.2f} |"
        )
    lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------- main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        default="scripts/es1_kn2z_promoted_nx240.h5",
        help="frozen nx=240 background run (read-only)",
    )
    parser.add_argument(
        "--out-dir", default="scripts", help="directory for the artifacts"
    )
    parser.add_argument(
        "--part",
        choices=("both", "1", "2"),
        default="both",
        help="which part to run (the summary needs both)",
    )
    args = parser.parse_args(argv)
    command = "scripts/neutral_arch_e1_study.py " + " ".join(
        sys.argv[1:] if argv is None else argv
    )
    command = command.strip()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    log(f"[e1] background: {args.run}")
    results = conditions = moments = costs = None
    t_all = time.perf_counter()
    if args.part in ("both", "1"):
        results = run_part1(args.run, log)
        write_cadence_artifact(
            out_dir / "neutral_arch_e1_cadence_nx240.txt", command, results
        )
        log("[e1] wrote neutral_arch_e1_cadence_nx240.txt")
    if args.part in ("both", "2"):
        conditions, moments, costs = run_part2(args.run, log)
        write_vgrid_artifact(
            out_dir / "neutral_arch_e1_vgrid_nx240.txt",
            command,
            conditions,
            moments,
            costs,
        )
        log("[e1] wrote neutral_arch_e1_vgrid_nx240.txt")
    if args.part == "both":
        write_summary(
            out_dir / "neutral_arch_e1_summary.md",
            command,
            results,
            conditions,
            moments,
            costs,
        )
        log("[e1] wrote neutral_arch_e1_summary.md")
    log(f"[e1] total {time.perf_counter() - t_all:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
