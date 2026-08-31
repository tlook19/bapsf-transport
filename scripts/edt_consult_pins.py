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

**The cathode face** carries no enthalpy or thermal-force flux at all: those
channels would carry the RETURNING thermal-electron current, and the cathode
sheath repels plasma electrons. Its WORK channel rides the model's own face-1
particle flux, so it is the exact partner of the expansion cooling
``pressure_work_rhs`` books there.

**The declared bracket axis**: ``charge_death`` -- where the beam's charge
dies, hence which faces still carry ``I_beam``. ``cell_1`` puts the death in
the cathode cell, so only that cell's low face carries beam current;
``cell_2`` puts it in the first gap cell, so the cathode cell's high face
carries it too.

**The anode reading** ``anode_handshake`` is a RULING, not a bracket axis.
``sheath_row_closes_all`` (default) closes both channels at the mesh face: the
kinetic anode sheath row is the total electron energy flux at the sheath edge
for the thermal population, so any fluid export there double-counts it, and the
beam electrons that reach the mesh directly are outside both bookings.
``sheath_row_closes`` (the 2.21 channel only) and ``export_counts`` (neither)
are retained as disclosed INSTRUMENT arms bounding that double count.

**Usage**::

    python scripts/edt_consult_pins.py --h5 scripts/mgcr1_confirm.h5

    # every arm, both anode instrument arms, and the afterglow clause
    python scripts/edt_consult_pins.py --h5 scripts/mgcr1_confirm.h5 --all-arms

**Provenance of the pins.** The 2026-08-26 consult measured the first set on
``scripts/mgcr1_confirm.h5`` over 0.1-20.1 ms at a mean loop current of 2772 A.
Two of its readings did not survive measurement and were retired by the
advisor adjudication of 2026-08-31: the ``+14.8 kW`` cathode-face handshake
(it rode the circuit's ion current at a face whose electron channel carries
0.3 mA, and its stated rationale was a stale read of a legacy row inert on the
shipped stance since R3.2), and the ``robust +13.6 kW`` compression piece (over
a fixed cell range it is bracket-A-specific, reading -5.3 kW under bracket B).
Its ``16.3/11.8 kW`` W_EMF is reproduced here as the WIDE support leg -- it
includes the mesh face's pressure jump -- and is reported as one end of a
bracket rather than as a disagreement. ``PINS`` below carries both the live
values and the retired ones, because a retired pin that simply vanishes leaves
the next reader re-deriving the same wrong number.
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

#: erg per eV, matching ``cablp.constants.ev_to_erg``. Used only to rebuild
#: what ``pressure_work_rhs`` books at the cathode face, in the solver's units.
_EV_TO_ERG = 1.602176634e-12

#: The enthalpy-plus-thermal-force coefficient carried by a drift face flux:
#: 3/2 from the internal energy convected, 0.71 from the Braginskii thermal
#: force heat flux q_u. The remaining 1.0 that completes the consult's 3.21
#: boundary coefficient comes from the pressure-drift WORK term by summation by
#: parts, not from a face flux -- which is why the anode handshake toggles 2.21
#: and not 3.21.
_C_FACE = 1.5 + 0.71

#: The consult's boundary coefficient, 5/2 (enthalpy) + 0.71 (thermal force).
_C_BOUNDARY = 2.5 + 0.71

#: The pins of record, AMENDED 2026-08-31 by the advisor adjudication and the
#: ratified amendment that followed it. Two of the 2026-08-26 consult's
#: readings did not survive measurement and are recorded here as RETIRED
#: rather than quietly dropped, because a retired pin that vanishes leaves the
#: next reader re-deriving the same wrong number.
PINS = {
    # RETIRED: the consult's +14.8 kW rode the circuit's ion current at the
    # cathode face. The channel there carries the RETURNING thermal-electron
    # current (0.3 mA), so the correct value is zero, and the consult's
    # rationale -- that it cancels a ghost-Bohm booking -- was a stale read of
    # a legacy row that has been inert on the shipped stance since R3.2.
    "cathode_face_enthalpy_kW": 0.0,
    "cathode_face_enthalpy_kW_RETIRED": 14.8,
    # The work channel's partner, which pressure_work_rhs books at that face.
    "cathode_face_work_kW": 4.298,
    # Over the cells strictly DOWNSTREAM of the death cell (bracket A, B).
    # EVERY row below names its anode reading, because the compression row and
    # the afterglow net are DIFFERENT QUANTITIES under the registered closure
    # and under the instrument arms -- the closure hands the mesh face's work
    # term to the sheath row and it lands in these same cells. Quoting one
    # without its reading was a review finding, twice.
    "compression_downstream_kW__CLOSURE": (35.4, 29.9),
    "compression_downstream_kW__export_counts_ARM": (13.6, 8.2),
    # RETIRED: over the fixed cells 2-5 this reads -5.3 kW under bracket B, so
    # the consult's "robust, handshake-independent +13.6 kW" was
    # bracket-A-specific rather than robust.
    "compression_cells_2_5_kW_RETIRED": 13.6,
    # Corrected source-region sums, per anode reading (bracket A, B).
    "gap_sum_kW__CLOSURE": (19.0, 14.5),
    "gap_sum_kW__sheath_row_closes_ARM": (-2.8, -7.3),
    "gap_sum_kW__export_counts_ARM": (-53.0, -57.5),
    # The EMF is a BRACKET over the SUPPORT, not a point, and is
    # current-weighted; it is insensitive to the anode reading.
    "inplasma_emf_V_range": (3.7, 6.2),
    "inplasma_emf_boltzmann_V": 5.7,  # Te ln(n_5/n_1)
    # Afterglow, by reading. The closure and the arms differ in SIGN.
    "afterglow_window_kW__CLOSURE": (0.25, 0.12),
    "afterglow_window_kW__export_counts_ARM": (-0.68, -0.81),
    "afterglow_instant_26ms_W__CLOSURE": (1.6, -1.0),
    # RETIRED: the as-built figures, moved by the 2026-08-31 cathode amendment.
    "afterglow_window_kW_RETIRED": (-0.55, -0.68),
    "afterglow_instant_26ms_kW_RETIRED": -0.017,
}

#: Q2 -- the cathode-face pins of the 2026-08-31 amendment, measured on
#: ``scripts/mgcr1_confirm.h5`` and reported by the Q2 section of this
#: evaluator. They exist because the amendment's clause (iii) rests on them:
#: the cathode-face enthalpy-ZERO premise is a drive-phase statement, and the
#: afterglow window is far outside the regime it was taken in.
#:
#: EVERY Q2b entry carries its CONSTRUCTION in its name and none is quotable
#: without it. ``Te[launch]`` and the drift current covary hard across the
#: ring-down -- 22 of 600 window samples run ``I_tot > I_beam`` at ``Te`` up
#: to 7.96 eV -- so the mean of the per-sample product and the product of the
#: window means differ IN SIGN. That is the same class of trap the EMF column
#: carries (read current-weighted, never as a straight sample average), and
#: it is why a bare "the export is ~N W" sentence is not a reading.
PINS_Q2 = {
    # (i) The circuit's own cathode electron-power row, drive vs afterglow.
    "q2a_source_P_cathode_e_drive_mean_W": 0.0584097,
    "q2a_source_P_cathode_e_afterglow_window_mean_W": 95.3081,
    "q2a_source_P_cathode_e_afterglow_window_peak_W": 2392.13,
    # (ii) The un-booked cathode-return export, 2.21 x Te[launch] x
    # (I_beam - I_tot) / e, over the afterglow window. THREE labelled pins.
    # The CONDITIONAL mean is the physical one: the export is a
    # returning-electron channel, so it is defined on the samples where there
    # are returning electrons to carry it (I_beam > I_tot).
    "q2b_export_conditional_mean_W__RETURNING_SAMPLES_ONLY": 35.0176,
    "q2b_export_unconditional_mean_W__COVARIANCE_DOMINATED": -122.317,
    "q2b_export_product_of_means_W__NOT_A_SAMPLE_MEAN": 49.2947,
    "q2b_export_window_peak_W": 251.547,
    # (iii) The registered closure arm is valid while I_tot >= 0; this is the
    # window minimum that says so.
    "q2c_min_I_tot_over_afterglow_window_A": 10.9039,
}

#: The afterglow window the Q2 pins are taken over [s].
Q2_WINDOW_S = (2.01e-2, 2.61e-2)

#: The enthalpy/heat-flux face coefficient, 1.5 + 0.71 -- the same ``_C_FACE``
#: the operator carries, named here because the Q2b export estimate is built
#: from it.
Q2_EXPORT_COEFF = _C_FACE

CHARGE_DEATH_CHOICES = ("cell_1", "cell_2")

#: ``sheath_row_closes_all`` first: it is the registered closure and the
#: default. The other two are disclosed INSTRUMENT arms bounding the double
#: count at the mesh face, and are not claim-bearing.
ANODE_HANDSHAKE_CHOICES = (
    "sheath_row_closes_all",
    "sheath_row_closes",
    "export_counts",
)


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


def _face_pairs(
    geom, charge_death, anode_handshake, I_tot, I_beam, n, u_face
):
    """Return the per-cell (low, high) drift-current face pairs, in amperes.

    Two arrays for the enthalpy/heat-flux channel and two for the
    pressure-drift-work channel, because the two faces that bound the operator
    treat them differently.

    **The cathode face** (amended 2026-08-31 after the advisor adjudication).
    The enthalpy and thermal-force channels there would carry the RETURNING
    thermal-electron current, and the cathode sheath repels plasma electrons --
    ``P_cathode_e`` is 0.06 W on this artifact, a return current of order
    0.3 mA -- so they are exactly zero. The WORK channel rides the model's own
    face-1 particle flux ``-e n u_face A``, making it the exact partner of the
    expansion cooling ``pressure_work_rhs`` books at that face. The earlier
    ``+14.8 kW`` reading, which rode the circuit's ion current there, is
    RETIRED as measured-wrong: it was an unsourced credit with no physical
    carrier and no ledger partner.

    **The anode face.** ``"sheath_row_closes_all"`` is the registered closure
    (ruled 2026-08-31) and closes BOTH channels: the kinetic anode sheath row
    is the total electron energy flux at the sheath edge for the thermal
    population, so any fluid export there double-counts it. The other two
    values are disclosed instrument arms bounding that double count.
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
    lo_flux[c] = 0.0
    if not (I_tot == 0.0 and I_beam == 0.0):
        # See the solver's own guard: the cathode-face work current encodes a
        # REPELLING sheath and is a driven-state statement, so it is booked
        # only where the model books a current at all.
        lo_work[c] = -_QE_C * n[c] * geom.plasma_face_area_cm2[c] * u_face[c]
    else:
        lo_work[c] = 0.0
    if anode_handshake in ("sheath_row_closes", "sheath_row_closes_all"):
        hi_flux[last] = 0.0
    if anode_handshake == "sheath_row_closes_all":
        hi_work[last] = 0.0
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


def evaluate(geom, Te, n, I_tot, I_beam, charge_death, anode_handshake, u=None):
    """Return the operator's four rows [W per cell] plus its face bookkeeping.

    ``Te`` [eV], ``n`` [cm^-3] and ``u`` [cm/s] are one saved instant's
    profiles; ``I_tot`` and ``I_beam`` are that instant's currents [A]. Every
    power below is ``coefficient x T_e[eV] x I[A]``, which is watts exactly.

    ``u`` is the ion velocity, needed ONLY for the cathode face, where the work
    channel has to ride the same velocity ``pressure_work_rhs`` books on. A
    caller that omits it gets a zero face-1 work current, which is a different
    (and weaker) statement -- so it is passed everywhere it matters.

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
    u_face = _face_mean(
        np.zeros(cells) if u is None else np.asarray(u, dtype=float),
        active,
        cells,
    )
    lo_flux, hi_flux, lo_work, hi_work = _face_pairs(
        geom, charge_death, anode_handshake, I_tot, I_beam, n, u_face
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

    # The SUPPORT-sensitivity leg of the EMF bracket: the same sum extended
    # across the anode mesh face, i.e. including that face's pressure jump.
    # This is the support the advisor identified as the consult's (it is what
    # reproduces the consult's 16.3/11.8 kW), and it is reported as the other
    # end of a bracket rather than as a correction. The DECLARED support stops
    # at the last interior face, because the drift is absorbed at the mesh and
    # never traverses the cell-5-to-6 gradient.
    #
    # It carries the PHYSICAL drift current at that face, not the handshake's
    # closed value: the bracket is about where the operator's support ends, not
    # about which ledger owns the face, and zeroing it under the registered
    # closure would collapse the leg into the other one.
    beam_faces_through = 1 if charge_death == "cell_1" else 2
    anode_beam = I_beam if (last + 1 - c) < beam_faces_through else 0.0
    anode_current = I_tot - anode_beam
    w_emf_pressure_wide = w_emf_pressure
    if n_face[last + 1] > 0.0 and last + 1 < cells:
        w_emf_pressure_wide += (anode_current / n_face[last + 1]) * (
            Te[last + 1] * n[last + 1] - Te[last] * n[last]
        )
    w_emf_wide = w_emf_pressure_wide + float(emf_work[c : last + 1].sum())

    # What ``pressure_work_rhs`` books at the cathode face on the SAME state:
    # -p_e div(u) contributes +p_e A u_face there. The operator's face-1 work
    # term is its exact partner, so the two must sum to roundoff -- that sum is
    # gate 3's second half, and it is the whole reason the work current rides
    # u_face rather than the circuit's ion current.
    pressure_work_face1_W = (
        Te[c] * n[c] * _EV_TO_ERG * geom.plasma_face_area_cm2[c] * u_face[c]
        * 1.0e-7
    )

    return {
        "W_EMF_wide_W": w_emf_wide,
        "anode_face_current_A": anode_current,
        "pressure_work_face1_W": pressure_work_face1_W,
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
    u_all = h5["u"]
    I_tot_all = h5["cathode_diagnostics/circuit_I_loop"][:]
    I_beam_all = h5["cathode_diagnostics/source_I_eth_star"][:]

    acc = None
    scal = {
        "cathode_face_flux_W": 0.0,
        "cathode_face_work_W": 0.0,
        "anode_face_flux_W": 0.0,
        "anode_face_work_W": 0.0,
        "W_EMF_W": 0.0,
        "W_EMF_wide_W": 0.0,
        "W_EMF_pressure_W": 0.0,
        "pressure_work_face1_W": 0.0,
    }
    # The EMF bracket is reported CURRENT-WEIGHTED: sum(W_EMF) / sum(I), not
    # the mean of a per-sample ratio. A per-sample ratio is a small-denominator
    # trap -- the pre-breakdown frames divide a few watts by a few amperes and
    # dominate any straight average with a number that is not a physics
    # reading.
    emf_num = 0.0
    emf_num_wide = 0.0
    emf_den = 0.0
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
            u=u_all[i, :],
        )
        emf_num += res["W_EMF_W"]
        emf_num_wide += res["W_EMF_wide_W"]
        emf_den += (
            float(I_tot_all[i]) if np.isfinite(I_tot_all[i]) else 0.0
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
    # The two ends of the EMF support bracket, current-weighted.
    acc["_emf_V_declared"] = emf_num / emf_den if emf_den else float("nan")
    acc["_emf_V_wide"] = emf_num_wide / emf_den if emf_den else float("nan")
    return acc


def _report_arm(geom, rows, charge_death, anode_handshake, throughput_W):
    c, last = geom.cathode_cell, geom.last_source_cell
    tot = rows["total_W"]
    gap_sum = float(tot[c : last + 1].sum())
    # Gate 4 is over the cells strictly DOWNSTREAM of the death cell, not a
    # fixed cell range: the fixed range was bracket-A-specific, and under
    # "cell_2" it swept in the death cell's own large negative and reported the
    # compression heating as a cooling.
    death_cell = c if charge_death == "cell_1" else c + 1
    compression = float(rows["pressure_work_W"][death_cell + 1 : last + 1].sum())
    handshake = rows["cathode_face_flux_W"]
    cathode_work = rows["cathode_face_work_W"]
    print(
        f"  [arm charge_death={charge_death} anode_handshake={anode_handshake}]"
    )
    per_cell = "  ".join(
        f"cell{j}={tot[j] * 1e-3:+.2f}" for j in range(c, last + 1)
    )
    print(f"    per-cell total kW : {per_cell}")
    print(f"    gap sum           : {gap_sum * 1e-3:+.2f} kW")
    print(
        f"    cathode-face enthalpy+thermal-force (face {geom.cathode_face}) : "
        f"{handshake * 1e-3:+.6f} kW   [expected 0; the +14.8 kW pin is "
        "RETIRED as measured-wrong]"
    )
    print(
        f"    cathode-face work term : {cathode_work * 1e-3:+.4f} kW  "
        "(the partner of pressure_work_rhs's face-1 piece; the two must sum "
        "to roundoff)"
    )
    print(
        f"    compression piece (pressure-drift work, cells "
        f"{death_cell + 1}-{last}, strictly downstream of the death cell): "
        f"{compression * 1e-3:+.3f} kW"
    )
    print(
        f"    W_EMF             : {rows['W_EMF_W'] * 1e-3:+.3f} kW declared "
        f"support / {rows['W_EMF_wide_W'] * 1e-3:+.3f} kW across the mesh face"
    )
    print(
        f"    in-plasma EMF     : {rows['_emf_V_declared']:+.3f} V (faces "
        f"{c + 1}-{last}, declared) .. {rows['_emf_V_wide']:+.3f} V (faces "
        f"{c + 1}-{last + 1}), current-weighted window means"
    )
    print(
        f"    volume identity   : worst relative residual over the window "
        f"{rows['_identity_worst_rel']:.3e}"
    )
    if throughput_W:
        print(
            f"    throughput-normalized: compression "
            f"{compression / throughput_W:.4f} of P_prim "
            f"{throughput_W * 1e-3:.1f} kW"
        )
    return {
        "gap_sum_W": gap_sum,
        "compression_W": compression,
        "handshake_W": handshake,
        "cathode_work_W": cathode_work,
        "W_EMF_W": rows["W_EMF_W"],
        "emf_V": rows["_emf_V_declared"],
        "emf_V_wide": rows["_emf_V_wide"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", required=True, help="saved sim1d trajectory")
    ap.add_argument("--t-lo", type=float, default=1.0e-4)
    ap.add_argument("--t-hi", type=float, default=2.01e-2)
    ap.add_argument("--charge-death", default="cell_1",
                    choices=CHARGE_DEATH_CHOICES)
    ap.add_argument("--anode-handshake", default="sheath_row_closes_all",
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
                u=h5["u"][i, :],
            )
            c, last = geom.cathode_cell, geom.last_source_cell
            net = float(res["total_W"][c : last + 1].sum())
            I_now = float(cd["circuit_I_loop"][i])
            I_b_now = float(cd["source_I_eth_star"][i])
            print(
                f"  [{charge_death}/{anode_handshake}] net "
                f"{net * 1e-3:+.4f} kW at I_tot={I_now:.1f} A, "
                f"I_beam={I_b_now:.1f} A -> Gamma_d current "
                f"{I_now - I_b_now:+.1f} A"
            )

        # The same clause over the afterglow WINDOW. The consult's "213 A still
        # flows" is this window's MEAN loop current, not the value at any single
        # instant -- the loop rings down from ~2980 A to ~12 A across the first
        # 1.5 ms of afterglow -- so the window mean, not the 26 ms instant, is
        # the reading its -0.5...-0.6 kW belongs to.
        lo, hi = args.afterglow_window_lo, float(t[-1])
        print(
            f"\n# AFTERGLOW WINDOW {lo * 1e3:.3f}-{hi * 1e3:.3f} ms "
            "(the reading the 213 A tail belongs to)"
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
                f"I_beam={rows['_I_beam_mean']:.1f} A -> mean Gamma_d current "
                f"{rows['_I_tot_mean'] - rows['_I_beam_mean']:+.1f} A "
                "(NEGATIVE: the emission outlasts the loop current, so the "
                "drift reverses in afterglow)"
            )
        print(
            "  NOTE: the afterglow term is confined to the ~1.5 ms ring-down; "
            "the loop falls from ~2980 A to ~12 A inside it, so the window "
            "mean and the 26 ms instant are two different readings and both "
            "are reported above."
        )

        # ---- Q2: the cathode-face pins of the 2026-08-31 amendment --------
        q2_lo, q2_hi = Q2_WINDOW_S
        drive_sel = np.flatnonzero((t >= args.t_lo) & (t <= args.t_hi))
        q2_sel = np.flatnonzero((t >= q2_lo) & (t <= q2_hi))
        print(
            f"\n# Q2 CATHODE-FACE PINS  (drive {args.t_lo * 1e3:.3f}-"
            f"{args.t_hi * 1e3:.3f} ms, afterglow window "
            f"{q2_lo * 1e3:.3f}-{q2_hi * 1e3:.3f} ms)"
        )
        P_ce = cd["source_P_cathode_e"][:]
        print("  Q2a source_P_cathode_e -- the circuit's own cathode electron")
        print("  power row. The drive-phase mean is the number the")
        print("  cathode-face enthalpy-ZERO premise was taken against; the")
        print("  afterglow mean is what that premise meets in the window.")
        print(
            f"    drive-phase mean          {np.nanmean(P_ce[drive_sel]):.6g} W"
            f"   (N={drive_sel.size})"
        )
        print(
            f"    afterglow-window mean     {np.nanmean(P_ce[q2_sel]):.6g} W"
            f"   (N={q2_sel.size})"
        )
        _pk = int(np.nanargmax(P_ce[q2_sel]))
        print(
            f"    afterglow-window peak     {np.nanmax(P_ce[q2_sel]):.6g} W"
            f"   at t={t[q2_sel][_pk] * 1e3:.4f} ms"
        )
        print(
            f"    ratio afterglow/drive     "
            f"{np.nanmean(P_ce[q2_sel]) / np.nanmean(P_ce[drive_sel]):.6g}"
        )

        I_tot_q2 = cd["circuit_I_loop"][:]
        I_beam_q2 = cd["source_I_eth_star"][:]
        Te_launch = h5["Te"][:, geom.cathode_cell]
        drift = I_beam_q2 - I_tot_q2
        export_W = Q2_EXPORT_COEFF * Te_launch * drift
        returning = drift[q2_sel] > 0.0
        print(
            f"  Q2b un-booked cathode-return export "
            f"{Q2_EXPORT_COEFF} x Te[{geom.cathode_cell}] x "
            f"(I_beam - I_tot) / e."
        )
        print("  THREE readings, and they DISAGREE IN SIGN. Quote none of")
        print("  them without the label: Te[launch] and the drift current")
        print("  covary across the ring-down, so the mean of the product is")
        print("  not the product of the means.")
        print(
            f"    CONDITIONAL mean, returning-electron samples only "
            f"(I_beam > I_tot)  "
            f"{np.nanmean(export_W[q2_sel][returning]):.6g} W"
        )
        print(
            f"      -- the PHYSICAL reading: the channel is defined where "
            f"there are returning"
        )
        print(
            f"         electrons to carry it. {int(returning.sum())} of "
            f"{q2_sel.size} samples; strictly positive"
        )
        print(
            f"         (min {np.nanmin(export_W[q2_sel][returning]):.6g} W, "
            f"max {np.nanmax(export_W[q2_sel][returning]):.6g} W)"
        )
        print(
            f"    UNCONDITIONAL per-sample mean, COVARIANCE-DOMINATED       "
            f"{np.nanmean(export_W[q2_sel]):.6g} W"
        )
        print(
            f"      -- the {int((~returning).sum())} samples with "
            f"I_tot >= I_beam mean "
            f"{np.nanmean(export_W[q2_sel][~returning]):.6g} W and carry it"
        )
        print(
            f"         negative; they sit in the first ~0.1 ms of ring-down "
            f"at Te up to "
            f"{np.nanmax(Te_launch[q2_sel]):.4g} eV"
        )
        print(
            f"    PRODUCT OF WINDOW MEANS, not a sample mean               "
            f"{Q2_EXPORT_COEFF * np.nanmean(Te_launch[q2_sel]) * np.nanmean(drift[q2_sel]):.6g} W"
        )
        _epk = int(np.nanargmax(export_W[q2_sel]))
        print(
            f"    window peak                                              "
            f"{np.nanmax(export_W[q2_sel]):.6g} W   at "
            f"t={t[q2_sel][_epk] * 1e3:.4f} ms"
        )
        print(
            f"  Q2c min I_tot over the window "
            f"{np.nanmin(I_tot_q2[q2_sel]):.6g} A "
            f"({int((I_tot_q2[q2_sel] < 0).sum())} samples below zero) -- the "
            f"registered closure arm is valid while I_tot >= 0, and over this "
            f"window it is."
        )

        print("\n# PINS OF RECORD (amended 2026-08-31), for comparison")
        for k, v in PINS.items():
            tag = "  [RETIRED]" if k.endswith("_RETIRED") else ""
            print(f"    {k}: {v}{tag}")
        print("\n# Q2 PINS OF RECORD (2026-08-31), for comparison")
        for k, v in PINS_Q2.items():
            print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
