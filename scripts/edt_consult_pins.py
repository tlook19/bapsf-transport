"""Row-level evaluator for the [ue-pressure-work] advisor-consult pins.

**What this is.** A STANDALONE re-derivation of the electron drift-transport +
EMF-work operator from a SAVED sim1d trajectory. It reads only the HDF5 file --
no solver construction, no new solver code -- so it can be run against an
artifact captured at any commit, including one that predates the operator's
implementation. That is what makes it usable as the base-side measurement of
the pre-registered pins (AGENTS.md, "Measure pre-registered pins at BASE before
gating on them"): the pins are properties of a saved state and of the consult's
own algebra, not of the new code, so they are measurable before a line of the
operator exists.

**The operator** (advisor consult 2026-08-26; Braginskii conventions):

    Delta = -div(3/2 T_e Gamma_d) - p_e div(Gamma_d / n) - div(q_u)
            + 0.71 Gamma_d . grad(T_e),

    Gamma_d = (I_tot - I_beam) / (e A),   q_u = 0.71 T_e Gamma_d.

``Gamma_d`` is the THERMAL-electron drift particle flux, a FACE quantity,
positive toward +z (cathode -> anode: the electrons carry the discharge current
that way, so the conventional current runs -z). ``I_tot`` is the model's own
booked loop current and ``I_beam`` the launched beam current still travelling as
a beam at that face. The electron energy equation books its pressure work with
the ION velocity, so the whole of ``Delta`` is absent from the ledger; the
resistive part of the field work (eta j^2) is NOT part of it -- that is already
booked as ``P_ohmic``.

**Discretization.** Conservative finite volume on the solver's own face
convention (`physics/sources.py::velocity_divergence`): face ``k`` lies between
cell ``k-1`` and cell ``k``, face arrays carry ``cells + 1`` entries, a
cell-centred quantity is carried to an interior face by the arithmetic mean and
to a face with only one live neighbour one-sidedly. Because the drift current is
INTERCEPTED at the anode mesh, the operator's faces are carried as a per-cell
(low, high) pair rather than one shared array: the anode face debits the cell
upstream of it and credits nothing downstream, which is what makes the operator
an open-system exchange with the electrodes rather than an internal
redistribution.

**The two declared conventions** (bracket axes, not chosen readings):

``charge_death`` -- where the beam's charge dies, hence which faces still carry
    ``I_beam``. ``cell_1`` (the consult's bracket A) puts the death in the
    cathode cell, so only that cell's low face carries beam current; ``cell_2``
    puts it in the first gap cell, so the cathode cell's high face carries it
    too.

``anode_handshake`` -- what the anode face does with the drift ENTHALPY flux.
    ``export_counts`` books it as an export out of the plasma (the cell upstream
    of the mesh is debited ``3.21 T_e I / e``); ``sheath_row_closes`` holds the
    existing anode sheath row to be the complete face closure for that enthalpy,
    so the operator books no enthalpy export there and would otherwise
    double-count it. The pressure-drift WORK at that face is a volumetric
    compression of the upstream cell, not a face export, and stands under both.

**Usage**::

    python scripts/edt_consult_pins.py --h5 scripts/mgcr1_confirm.h5

    # every arm of the bracket, and the afterglow clause
    python scripts/edt_consult_pins.py --h5 scripts/mgcr1_confirm.h5 --all-arms

The consult's numbers were measured on ``scripts/mgcr1_confirm.h5`` over the
window 0.1-20.1 ms at a mean loop current of 2774 A, and are quoted in the pin
table below as the comparison target. They are the consult's, not re-derived
here.
"""

import argparse
import json
import sys

import h5py
import numpy as np

#: Elementary charge [C]. Only ratio I/e enters, and every power below reduces
#: to (coefficient) x T_e[eV] x I[A] = W exactly, so no other conversion is
#: needed: the eV->erg and erg->W factors cancel against C/e.
_QE_C = 1.602176634e-19

#: The enthalpy-plus-thermal-force coefficient carried by a drift face flux:
#: 3/2 from the internal energy convected, 0.71 from the Braginskii thermal
#: force heat flux q_u. The remaining 1.0 that completes the consult's 3.21
#: boundary coefficient comes from the pressure-drift WORK term by summation by
#: parts, not from a face flux -- which is why the anode handshake toggles 2.21
#: and not 3.21.
_C_FACE = 1.5 + 0.71

#: The consult's boundary coefficient, 5/2 (enthalpy) + 0.71 (thermal force).
_C_BOUNDARY = 2.5 + 0.71

#: The consult's pins, for the report line. Values are the consult's own
#: measurement on mgcr1_confirm.h5 over 0.1-20.1 ms; see the module docstring.
CONSULT_PINS = {
    "cathode_face_handshake_kW": 14.8,
    "compression_cells_2_5_kW": 13.6,
    "cell_1_kW": -50.5,
    "gap_cells_kW": (5.5, 4.5, 2.6, 2.0),
    "gap_sum_export_counts_kW": -35.8,
    "gap_sum_sheath_row_closes_kW": 14.4,
    "W_EMF_kW": (16.3, 11.8),  # bracket A, bracket B
    "inplasma_emf_V": (5.9, 4.2),  # bracket A, bracket B
    "afterglow_kW": (-0.5, -0.6),
}

CHARGE_DEATH_CHOICES = ("cell_1", "cell_2")
ANODE_HANDSHAKE_CHOICES = ("export_counts", "sheath_row_closes")


def _decode(arr):
    return np.array(
        [a.decode() if isinstance(a, bytes) else str(a) for a in arr]
    )


class SavedGeometry:
    """The pieces of the axial grid the operator needs, off the saved file.

    ``plasma_face_area_cm2`` is not written to the trajectory, so it is rebuilt
    here by the solver's own rule (``core/geometry.py::_face_area``: interior
    faces are the arithmetic mean of the adjacent cell areas, external faces
    take the end cell). The rebuild is asserted to be exact where it matters --
    the operator's support -- by requiring the cell areas across it to be
    uniform, which makes the mean exact rather than approximate. A file whose
    source region is flared fails that assertion loudly rather than being
    silently mis-evaluated.
    """

    def __init__(self, h5):
        g = h5["geometry"]
        self.length_cm = g["length_cm"][:]
        self.z_cm = g["z_cm"][:]
        self.plasma_volume_cm3 = g["plasma_volume_cm3"][:]
        self.plasma_active = g["plasma_active"][:].astype(bool)
        self.role = _decode(g["cell_role"][:])
        self.cells = self.length_cm.size
        self.plasma_area_cm2 = self.plasma_volume_cm3 / self.length_cm
        area = self.plasma_area_cm2
        face = np.empty(self.cells + 1, dtype=float)
        face[0] = area[0]
        face[-1] = area[-1]
        face[1:-1] = 0.5 * (area[:-1] + area[1:])
        self.plasma_face_area_cm2 = face

        # The operator's support: the cathode cell and every cell up to the
        # anode mesh. Derived from the saved roles, then cross-checked against
        # the configured cathode length so a role/geometry disagreement is loud.
        cathode = np.flatnonzero(self.role == "cathode")
        if cathode.size != 1:
            raise ValueError(
                "edt_consult_pins expects exactly one cathode cell; found "
                f"{cathode.size} (roles {sorted(set(self.role))})"
            )
        self.cathode_cell = int(cathode[0])
        self.cathode_face = self.cathode_cell  # the cell's LOW face
        column = np.flatnonzero(self.role == "column")
        downstream = column[column > self.cathode_cell]
        if downstream.size == 0:
            raise ValueError("no column cell downstream of the cathode cell")
        self.anode_face = int(downstream[0])  # low face of the first column cell
        self.last_source_cell = self.anode_face - 1

    def check_uniform_area(self):
        lo, hi = self.cathode_cell, self.last_source_cell
        area = self.plasma_area_cm2[lo : hi + 1]
        if not np.allclose(area, area[0], rtol=0.0, atol=0.0):
            raise ValueError(
                "the plasma area is not uniform across the operator's support "
                f"(cells {lo}-{hi}); the face-area rebuild would be an "
                f"approximation. Areas: {area}"
            )

    def check_against_config(self, params):
        """Cross-check the anode face against the configured cathode length."""
        z_edges = np.concatenate(([0.0], np.cumsum(self.length_cm)))
        # The saved z_cm is the CENTRED axis; the plenum sits behind z=0, so the
        # edge grid above is offset by the plenum length. Rebase onto the
        # cathode-cell low face, which the roles put at z = 0.
        z0 = z_edges[self.cathode_face]
        z_anode = z_edges[self.anode_face] - z0
        L_cath = float(params.get("L_cath", np.nan))
        return z_anode, L_cath


def _face_pairs(geom, charge_death, anode_handshake, I_tot, I_beam):
    """Return the per-cell (low, high) drift-current face pairs, in amperes.

    Two arrays are returned for the enthalpy/heat-flux channel and two for the
    pressure-drift-work channel, because the anode handshake separates them: the
    ``sheath_row_closes`` reading holds the existing anode sheath row to be the
    complete closure of the drift ENTHALPY at that face, while the compression
    work on the cell upstream of the mesh is not booked by any existing row and
    stands under both readings.
    """
    cells = geom.cells
    lo_flux = np.zeros(cells, dtype=float)
    hi_flux = np.zeros(cells, dtype=float)
    lo_work = np.zeros(cells, dtype=float)
    hi_work = np.zeros(cells, dtype=float)

    c = geom.cathode_cell
    last = geom.last_source_cell
    # Faces carrying beam current: always the cathode cell's low face; under
    # "cell_2" its high face as well (the charge survives into the next cell).
    beam_faces_through = 1 if charge_death == "cell_1" else 2

    for j in range(c, last + 1):
        # Low face of cell j is face index j; high face is j + 1.
        n_beam_lo = I_beam if (j - c) < beam_faces_through else 0.0
        n_beam_hi = I_beam if (j + 1 - c) < beam_faces_through else 0.0
        lo_flux[j] = I_tot - n_beam_lo
        hi_flux[j] = I_tot - n_beam_hi
        lo_work[j] = lo_flux[j]
        hi_work[j] = hi_flux[j]
    if anode_handshake == "sheath_row_closes":
        hi_flux[last] = 0.0
    return lo_flux, hi_flux, lo_work, hi_work


def _face_mean(values, active, cells):
    """Carry a cell-centred quantity to faces, one-sided at a dead neighbour."""
    face = np.zeros(cells + 1, dtype=float)
    for k in range(cells + 1):
        left = k - 1
        right = k
        lo_ok = left >= 0 and active[left]
        hi_ok = right < cells and active[right]
        if lo_ok and hi_ok:
            face[k] = 0.5 * (values[left] + values[right])
        elif lo_ok:
            face[k] = values[left]
        elif hi_ok:
            face[k] = values[right]
    return face


def evaluate(geom, Te, n, I_tot, I_beam, charge_death, anode_handshake):
    """Return the operator's four rows [W per cell] plus its face bookkeeping.

    ``Te`` [eV] and ``n`` [cm^-3] are one saved instant's profiles; ``I_tot``
    and ``I_beam`` are that instant's currents [A]. Every power below is
    ``coefficient x T_e[eV] x I[A]``, which is watts exactly.

    A non-finite current is read as zero rather than propagated: the cathode
    solve writes NaN into its per-solve currents on a save where no solve ran
    (the last afterglow sample is one), and a step with no current has no drift
    to book.
    """
    I_tot = float(I_tot) if np.isfinite(I_tot) else 0.0
    I_beam = float(I_beam) if np.isfinite(I_beam) else 0.0
    cells = geom.cells
    active = geom.plasma_active
    Te_face = _face_mean(Te, active, cells)
    n_face = _face_mean(n, active, cells)
    lo_flux, hi_flux, lo_work, hi_work = _face_pairs(
        geom, charge_death, anode_handshake, I_tot, I_beam
    )

    idx = np.arange(cells)
    Te_lo = Te_face[idx]
    Te_hi = Te_face[idx + 1]
    n_lo = n_face[idx]
    n_hi = n_face[idx + 1]

    # -div(3/2 T_e Gamma_d): in - out, as a face-power difference.
    enthalpy = 1.5 * (Te_lo * lo_flux - Te_hi * hi_flux)
    # -div(q_u), q_u = 0.71 T_e Gamma_d: the same face structure.
    thermal_flux = 0.71 * (Te_lo * lo_flux - Te_hi * hi_flux)
    # -p_e div(Gamma_d / n): p_e is cell-centred, the divergence is the face
    # pair of the drift VELOCITY Gamma_d/n. A face with no live neighbour
    # carries n_face = 0 and contributes nothing.
    with np.errstate(divide="ignore", invalid="ignore"):
        w_hi = np.where(n_hi > 0.0, hi_work / n_hi, 0.0)
        w_lo = np.where(n_lo > 0.0, lo_work / n_lo, 0.0)
    pressure_work = -Te * n * (w_hi - w_lo)
    # +0.71 Gamma_d . grad(T_e), cell-centred on the same face pair.
    emf_work = 0.71 * 0.5 * (lo_flux + hi_flux) * (Te_hi - Te_lo)

    total = enthalpy + thermal_flux + pressure_work + emf_work

    c, last = geom.cathode_cell, geom.last_source_cell
    # The face powers the identity's boundary term is built from.
    cathode_face_flux_W = _C_FACE * Te_lo[c] * lo_flux[c]
    cathode_face_work_W = Te[c] * n[c] * w_lo[c]
    anode_face_flux_W = _C_FACE * Te_hi[last] * hi_flux[last]
    anode_face_work_W = Te[last] * n[last] * w_hi[last]

    # W_EMF, computed INDEPENDENTLY of the residual so the identity is a real
    # test: the thermal-force half is the emf_work row; the pressure half is the
    # discrete summation-by-parts partner of the pressure-work row, a sum over
    # the faces INTERIOR to the operator's support.
    w_emf_pressure = 0.0
    for k in range(c + 1, last + 1):
        pe_left = Te[k - 1] * n[k - 1]
        pe_right = Te[k] * n[k]
        if n_face[k] > 0.0:
            w_emf_pressure += (lo_work[k] / n_face[k]) * (pe_right - pe_left)
    w_emf = w_emf_pressure + float(emf_work[c : last + 1].sum())

    return {
        "enthalpy_W": enthalpy,
        "thermal_flux_W": thermal_flux,
        "pressure_work_W": pressure_work,
        "emf_work_W": emf_work,
        "total_W": total,
        "cathode_face_flux_W": cathode_face_flux_W,
        "cathode_face_work_W": cathode_face_work_W,
        "anode_face_flux_W": anode_face_flux_W,
        "anode_face_work_W": anode_face_work_W,
        "W_EMF_W": w_emf,
        "W_EMF_pressure_W": w_emf_pressure,
    }


def volume_identity(res, geom):
    """Return (lhs, rhs, relative residual) of the consult's volume identity.

    ``sum(Delta dV) == [3.21 T_e I/e]_in - [3.21 T_e I/e]_out + W_EMF``.

    The boundary term is assembled from the ACTUAL face powers the convention
    selects (2.21 from the enthalpy/heat-flux channel plus 1.00 from the
    pressure-drift work, which together are the consult's 3.21), so the identity
    stays exact under the ``sheath_row_closes`` reading -- where the enthalpy
    export is held closed by the sheath row and the boundary coefficient at the
    anode face is 1.00 rather than 3.21 -- instead of silently failing there.
    """
    c, last = geom.cathode_cell, geom.last_source_cell
    lhs = float(res["total_W"][c : last + 1].sum())
    boundary_in = res["cathode_face_flux_W"] + res["cathode_face_work_W"]
    boundary_out = res["anode_face_flux_W"] + res["anode_face_work_W"]
    rhs = boundary_in - boundary_out + res["W_EMF_W"]
    scale = max(abs(lhs), abs(rhs), 1.0)
    return lhs, rhs, abs(lhs - rhs) / scale


def _window_mean_rows(h5, geom, t_lo, t_hi, charge_death, anode_handshake):
    """Average the operator's rows over every saved sample in a time window."""
    t = h5["time"][:]
    sel = np.flatnonzero((t >= t_lo) & (t <= t_hi))
    if sel.size == 0:
        raise ValueError(f"no saved samples in [{t_lo}, {t_hi}] s")
    Te_all = h5["Te"]
    n_all = h5["n"]
    I_tot_all = h5["cathode_diagnostics/circuit_I_loop"][:]
    I_beam_all = h5["cathode_diagnostics/source_I_eth_star"][:]

    acc = None
    scal = {
        "cathode_face_flux_W": 0.0,
        "cathode_face_work_W": 0.0,
        "anode_face_flux_W": 0.0,
        "anode_face_work_W": 0.0,
        "W_EMF_W": 0.0,
        "W_EMF_pressure_W": 0.0,
    }
    ident_worst = 0.0
    for i in sel:
        res = evaluate(
            geom,
            Te_all[i, :],
            n_all[i, :],
            float(I_tot_all[i]),
            float(I_beam_all[i]),
            charge_death,
            anode_handshake,
        )
        if acc is None:
            acc = {
                k: np.zeros(geom.cells)
                for k in (
                    "enthalpy_W",
                    "thermal_flux_W",
                    "pressure_work_W",
                    "emf_work_W",
                    "total_W",
                )
            }
        for k in acc:
            acc[k] += res[k]
        for k in scal:
            scal[k] += res[k]
        _, _, rel = volume_identity(res, geom)
        ident_worst = max(ident_worst, rel)
    for k in acc:
        acc[k] /= sel.size
    for k in scal:
        scal[k] /= sel.size
    acc.update(scal)
    acc["_samples"] = sel.size
    acc["_identity_worst_rel"] = ident_worst
    acc["_t_lo"] = float(t[sel[0]])
    acc["_t_hi"] = float(t[sel[-1]])
    acc["_I_tot_mean"] = float(np.nanmean(I_tot_all[sel]))
    acc["_I_beam_mean"] = float(np.nanmean(I_beam_all[sel]))
    return acc


def _report_arm(geom, rows, charge_death, anode_handshake, throughput_W):
    c, last = geom.cathode_cell, geom.last_source_cell
    tot = rows["total_W"]
    gap_sum = float(tot[c : last + 1].sum())
    compression = float(rows["pressure_work_W"][c + 1 : last + 1].sum())
    handshake = rows["cathode_face_flux_W"]
    print(
        f"  [arm charge_death={charge_death} anode_handshake={anode_handshake}]"
    )
    per_cell = "  ".join(
        f"cell{j}={tot[j] * 1e-3:+.2f}" for j in range(c, last + 1)
    )
    print(f"    per-cell total kW : {per_cell}")
    print(f"    gap sum           : {gap_sum * 1e-3:+.2f} kW")
    print(
        f"    cathode-face handshake (2.21 Te I/e, face {geom.cathode_face}) : "
        f"{handshake * 1e-3:+.3f} kW"
    )
    print(
        f"    compression piece (pressure-drift work, cells "
        f"{c + 1}-{last})                : {compression * 1e-3:+.3f} kW"
    )
    print(
        f"    W_EMF             : {rows['W_EMF_W'] * 1e-3:+.3f} kW "
        f"(pressure half {rows['W_EMF_pressure_W'] * 1e-3:+.3f} kW)"
    )
    emf_V = (
        rows["W_EMF_W"] / rows["_I_tot_mean"]
        if rows["_I_tot_mean"] != 0.0
        else float("nan")
    )
    print(f"    in-plasma EMF     : {emf_V:+.3f} V")
    print(
        f"    volume identity   : worst relative residual over the window "
        f"{rows['_identity_worst_rel']:.3e}"
    )
    if throughput_W:
        print(
            f"    throughput-normalized: handshake "
            f"{handshake / throughput_W:.4f}, compression "
            f"{compression / throughput_W:.4f} (of P_prim "
            f"{throughput_W * 1e-3:.1f} kW)"
        )
    return {
        "gap_sum_W": gap_sum,
        "compression_W": compression,
        "handshake_W": handshake,
        "W_EMF_W": rows["W_EMF_W"],
        "emf_V": emf_V,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", required=True, help="saved sim1d trajectory")
    ap.add_argument("--t-lo", type=float, default=1.0e-4)
    ap.add_argument("--t-hi", type=float, default=2.01e-2)
    ap.add_argument("--charge-death", default="cell_1",
                    choices=CHARGE_DEATH_CHOICES)
    ap.add_argument("--anode-handshake", default="export_counts",
                    choices=ANODE_HANDSHAKE_CHOICES)
    ap.add_argument("--all-arms", action="store_true",
                    help="evaluate all four bracket arms")
    ap.add_argument("--afterglow-t", type=float, default=2.6e-2,
                    help="instant for the afterglow clause (gate 6)")
    ap.add_argument("--afterglow-window-lo", type=float, default=2.01e-2,
                    help="start of the afterglow WINDOW; the consult's 213 A "
                         "tail is this window's mean loop current, not an "
                         "instantaneous value")
    args = ap.parse_args(argv)

    with h5py.File(args.h5, "r") as h5:
        geom = SavedGeometry(h5)
        geom.check_uniform_area()
        params = json.loads(h5.attrs["params_json"])
        z_anode, L_cath = geom.check_against_config(params)
        print(f"# {args.h5}")
        print(
            f"# cells={geom.cells} cathode_cell={geom.cathode_cell} "
            f"cathode_face={geom.cathode_face} anode_face={geom.anode_face} "
            f"source cells {geom.cathode_cell}-{geom.last_source_cell}"
        )
        print(
            f"# anode face at z={z_anode:.4f} cm vs configured L_cath="
            f"{L_cath:.4f} cm "
            f"({'MATCH' if abs(z_anode - L_cath) < 1e-6 else 'MISMATCH'})"
        )
        cd = h5["cathode_diagnostics"]
        t = h5["time"][:]
        sel = np.flatnonzero((t >= args.t_lo) & (t <= args.t_hi))
        throughput_W = float(np.mean(cd["source_P_prim"][sel]))

        arms = (
            [(a, b) for a in CHARGE_DEATH_CHOICES for b in ANODE_HANDSHAKE_CHOICES]
            if args.all_arms
            else [(args.charge_death, args.anode_handshake)]
        )
        print(
            f"\n# WINDOW {args.t_lo * 1e3:.3f}-{args.t_hi * 1e3:.3f} ms"
        )
        for charge_death, anode_handshake in arms:
            rows = _window_mean_rows(
                h5, geom, args.t_lo, args.t_hi, charge_death, anode_handshake
            )
            if (charge_death, anode_handshake) == arms[0]:
                print(
                    f"# samples={rows['_samples']} "
                    f"t={rows['_t_lo'] * 1e3:.3f}-{rows['_t_hi'] * 1e3:.3f} ms "
                    f"I_tot_mean={rows['_I_tot_mean']:.1f} A "
                    f"I_beam_mean={rows['_I_beam_mean']:.1f} A"
                )
            _report_arm(geom, rows, charge_death, anode_handshake, throughput_W)

        # The afterglow clause (gate 6): reported, never gated on.
        i = int(np.argmin(np.abs(t - args.afterglow_t)))
        print(f"\n# AFTERGLOW CLAUSE at t={t[i] * 1e3:.4f} ms")
        for charge_death, anode_handshake in arms:
            res = evaluate(
                geom,
                h5["Te"][i, :],
                h5["n"][i, :],
                float(cd["circuit_I_loop"][i]),
                float(cd["source_I_eth_star"][i]),
                charge_death,
                anode_handshake,
            )
            c, last = geom.cathode_cell, geom.last_source_cell
            net = float(res["total_W"][c : last + 1].sum())
            print(
                f"  [{charge_death}/{anode_handshake}] net "
                f"{net * 1e-3:+.4f} kW at I_tot="
                f"{float(cd['circuit_I_loop'][i]):.1f} A, I_beam="
                f"{float(cd['source_I_eth_star'][i]):.1f} A"
            )

        # The same clause over the afterglow WINDOW. The consult's "213 A still
        # flows" is this window's MEAN loop current, not the value at any single
        # instant -- the loop rings down from ~2980 A to ~12 A across the first
        # 1.5 ms of afterglow -- so the window mean, not the 26 ms instant, is
        # the reading its -0.5...-0.6 kW belongs to.
        lo, hi = args.afterglow_window_lo, float(t[-1])
        print(
            f"\n# AFTERGLOW WINDOW {lo * 1e3:.3f}-{hi * 1e3:.3f} ms "
            "(the reading the consult's 213 A tail belongs to)"
        )
        for charge_death, anode_handshake in arms:
            rows = _window_mean_rows(h5, geom, lo, hi, charge_death,
                                     anode_handshake)
            c, last = geom.cathode_cell, geom.last_source_cell
            net = float(rows["total_W"][c : last + 1].sum())
            print(
                f"  [{charge_death}/{anode_handshake}] net "
                f"{net * 1e-3:+.4f} kW over {rows['_samples']} samples at "
                f"mean I_tot={rows['_I_tot_mean']:.1f} A, mean "
                f"I_beam={rows['_I_beam_mean']:.1f} A"
            )

        print("\n# CONSULT PINS (advisor consult 2026-08-26), for comparison")
        for k, v in CONSULT_PINS.items():
            print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
