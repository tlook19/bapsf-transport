"""Transient deterministic velocity-grid neutral engine (K2a).

The LIVE-transient promotion of the steady generation solver in
``kinetic_neutrals.py``: instead of iterating unit-source responses to a
steady state, this module carries accepted-state distributions

    f_c(z, v_z, v_perp)   column  [cm^-3 per bin, on the column volume]
    f_a(z, v_z, v_perp)   annulus [cm^-3 per bin, on the annulus volume]

and advances them ONE step per neutral-clock tick with the same implicit
upwind march, the same sinh-stretched shared ``VGrid``, the same
moment-exact Maxwellian projection, the same cosine-wall re-emission
spectrum and the same Cauchy-chord zone rates. Nothing here is a second
implementation of the velocity grid or of the transport sweep -- the
operators are imported or transcribed from that module so the offline
instruments and the in-solver arm keep agreeing on inputs.

Time discretization (split implicit; first order, unconditionally stable,
positivity preserving):

  A. transport + every LOSS process, fully implicit -- one backward-Euler
     upwind march per sign of ``v_z``, with the 2x2 column/annulus zone
     coupling solved exactly per (cell, bin) as in ``KN2Zone.sweep``. The
     march diagonal carries ``1/dt`` and the right-hand side ``f^n/dt``;
     every loss channel is then tallied from ``f^{n+1}``, so the tallies
     are the losses the update actually took.
  B. BIRTHS, at masses exactly equal to the substep-A tallies. Charge
     exchange and elastic scattering re-emit their own losses at the local
     ion Maxwellian, the cylindrical wall re-emits its own losses (split
     accommodated/reflected), the anode mesh re-emits what it intercepted,
     and the external ledger (puff, recombination, recycling faces, anode
     rebirths) is added as counted particles.

Splitting births from losses this way is what makes the inventory ledger
EXACT rather than converged: substep A never creates a particle and
substep B creates exactly the number substep A destroyed, per channel, so
the domain total closes to roundoff at every update regardless of dt.

End walls are the one lagged channel. Their returns re-enter as boundary
INFLOW ghost densities rather than as a volume source in the end cell,
because only the inflow form preserves an equilibrium Maxwellian exactly
(the returned flux spectrum divided by ``|v_z| A`` reproduces the volume
Maxwellian bin for bin). The outflow is known only after the march, so
the returning particles are held in a per-end pending buffer and injected
on the next update. The buffer is part of the inventory: closure is
stated over ``sum(f V) + pending``.

Sign and unit conventions: CGS throughout, distributions in cm^-3 per
bin (the bin sum IS the density), ledger entries in PARTICLES (not
rates), momentum tallies in g cm/s, energy tallies in erg. The
plasma-coupling arrays the solver reads are per-second densities on the
PLASMA volume, which is exactly the column volume.
"""

import numpy as np

from cablp.funcs._cross import (
    phelps_he_backscatter_cm2,
    phelps_he_isotropic_cm2,
)

from .kinetic_neutrals import EV, KB, M_HE, T_WALL_K, VGrid
from .neutrals import neutral_zone_volumes


ELASTIC_MODELS = ("phelps_iso", "off")

# Column<->annulus zone-exchange closures. Both are algebraic rates on the
# same (cell, v_perp) index and both impose the antisymmetry through the
# actual geometry volumes; they differ in the mean chord and in how one
# surface encounter is split between the two cylinders. See
# ``TransientDVM`` for the expressions and
# ``scripts/k2_dvm_exchange_measured.txt`` for the measurement.
EXCHANGE_MODELS = ("cauchy_chord", "geometric")

# Rate factor on the isotropic-elastic BGK channel. A full-replacement event
# transfers m (v - u_i), which is twice the isotropic angular average
# ``mu <1 - cos th> g = m g / 2`` at equal mass; halving the collision rate
# restores the correct mean momentum (and energy) transfer per unit time. It
# is the equal-mass reduced-mass ratio ``mu/m = 1/2``, not a fitted number.
# See ``TransientDVM.collision_frequencies``.
ELASTIC_BGK_MOMENTUM_FACTOR = 0.5

# Every channel the ledger books, as (birth, loss) name pairs plus the
# one-sided channels. Named here so the verification script and the smoke
# suite can assert the ledger is complete rather than re-listing them.
LEDGER_LOSS_CHANNELS = (
    "ionization",
    "cx",
    "elastic",
    "wall",
    "mesh_blocked",
    "end_out_L",
    "end_out_R",
)
LEDGER_BIRTH_CHANNELS = (
    "cx",
    "elastic",
    "wall_accommodated",
    "wall_reflected",
    "mesh_reemit",
    "end_return_L",
    "end_return_R",
    "puff",
    "recombination",
    "cathode_face",
    "collector_face",
    "anode",
)
# Losses that leave the modelled system entirely (the rest are internal
# and are paired with a birth of exactly equal mass).
LEDGER_EXTERNAL_LOSSES = ("ionization", "pump_L", "pump_R")
LEDGER_EXTERNAL_BIRTHS = (
    "puff",
    "recombination",
    "cathode_face",
    "collector_face",
    "anode",
)


class TransientDVM:
    """Live transient two-zone velocity-grid neutral state.

    ``accommodation`` is the thermal accommodation coefficient of the
    stainless-steel surfaces: the accommodated fraction is re-emitted
    cosine-distributed at the wall temperature, the remaining fraction is
    reflected at the incident energy. On the axisymmetric ``(v_z,
    v_perp)`` grid a specular reflection off the cylindrical wall reverses
    only the radial component of ``v_perp`` and so is bin-preserving; off
    an end wall it reverses ``v_z``, which is the exact bin mirror of the
    symmetric stretched axis. Both are therefore represented exactly, with
    no re-projection error.

    ``elastic_model`` selects the polarization-elastic channel:
    ``"phelps_iso"`` adds a BGK-like relaxation toward the local ion
    Maxwellian at HALF the Phelps isotropic rate (the isotropic
    momentum-transfer average -- see :meth:`collision_frequencies` for why
    the half is there), ``"off"`` drops it (charge exchange then carries
    all ion-neutral momentum transfer). The elastic channel exists because
    the arm supersedes the fluid ion-neutral collision family wholesale
    and the fluid operator's momentum-transfer cross section is
    ``Qi + 2 Qb``; carrying only ``Qb`` would silently drop the ``Qi``
    half.

    ``exchange_model`` selects the column<->annulus zone-exchange closure,
    i.e. the per-``(cell, v_perp)`` frequencies at which a neutral crosses
    ``r = Rp`` in either direction and strikes ``r = Rm``:

    ``"cauchy_chord"``
        the three-dimensional Cauchy mean chord ``4V/S = 2 (Rm - Rp)``
        evaluated at the perpendicular speed, with the encounter split
        between the two cylinders as ``Rp/Rm : (1 - Rp/Rm)``::

            nu_total = vp / (2 (Rm - Rp))
            nu_a->c  = (Rp/Rm) nu_total        nu_a->wall = (1 - Rp/Rm) nu_total

    ``"geometric"``
        the mean chord of the cell CROSS-SECTION, ``pi A / P``, since the
        crossings of two coaxial cylinders are decided entirely by the
        motion in the ``(x, y)`` plane, with the encounter split between
        the two circles in proportion to their PERIMETERS::

            nu_total = 2 vp / (pi (Rm - Rp))
            nu_a->c  = 2 vp Rp / (pi (Rm^2 - Rp^2))
            nu_a->wall = 2 vp Rm / (pi (Rm^2 - Rp^2))
            nu_c->a  = 2 vp / (pi Rp)

        Averaged over a Maxwellian this ``nu_c->a`` is ``vbar / (2 Rp)``,
        the free-molecular column loss rate the fluid arm's
        :func:`~cablp.solvers._sim1d.physics.neutrals.neutral_zone_exchange_conductance`
        carries.

    Both branches impose ``V_col nu_c->a == V_ann nu_a->c`` through the
    actual cell volumes, so the ledger's zone channel cancels exactly
    either way. Any other value raises.
    """

    def __init__(
        self,
        *,
        geometry,
        nvz=48,
        nvp=12,
        accommodation=1.0,
        elastic_model="phelps_iso",
        exchange_model="cauchy_chord",
        transparency=1.0,
        mesh_face=-999,
        s_L=0.0,
        s_R=0.0,
        T_wall_K=T_WALL_K,
        vmax_cm_s=None,
        Ti_cap_eV=10.0,
        u_cap_cm_s=2.0e6,
        grid=None,
    ):
        if elastic_model not in ELASTIC_MODELS:
            raise ValueError(
                f"elastic_model must be one of {ELASTIC_MODELS} "
                f"(got {elastic_model!r})"
            )
        if exchange_model not in EXCHANGE_MODELS:
            raise ValueError(
                f"exchange_model must be one of {EXCHANGE_MODELS} "
                f"(got {exchange_model!r})"
            )
        self.accommodation = float(accommodation)
        self.elastic_model = str(elastic_model)
        self.exchange_model = str(exchange_model)
        self.T_wall_K = float(T_wall_K)
        self.transparency = float(transparency)
        self.mesh_face = int(mesh_face)
        self.s_L = float(s_L)
        self.s_R = float(s_R)

        self.dz = np.asarray(geometry.length_cm, dtype=float)
        self.nz = self.dz.size
        V_col, V_ann = neutral_zone_volumes(geometry)
        self.V_col = np.asarray(V_col, dtype=float)
        self.V_ann = np.asarray(V_ann, dtype=float)
        self.A_col = self.V_col / self.dz
        self.A_ann = self.V_ann / self.dz
        # Axial transport uses FACE areas, not cell areas. The shipped
        # steady march writes cell i's gain as ``lam_i f_{i-1}`` with
        # ``lam_i = |v_z|/dz_i``, which moves ``|v_z| f A_i dt`` particles
        # in while cell i-1 loses ``|v_z| f A_{i-1} dt`` -- equal only on a
        # constant-area grid. This device's end cells are expanded, so the
        # face form is required for the inventory ledger to close: one
        # area per face, taken as the throat ``min(A_left, A_right)``, the
        # free-molecular choice. Both ends are open (the pumped faces).
        self.face_c = _throat_areas(self.A_col)
        self.face_a = _throat_areas(self.A_ann)
        Rp = np.asarray(geometry.Rp_cm, dtype=float)
        Rm = np.asarray(geometry.Rm_cm, dtype=float)

        if grid is None:
            if vmax_cm_s is None:
                vmax_cm_s = 4.0 * np.sqrt(
                    max(float(Ti_cap_eV), 0.5) * EV / M_HE
                ) + 1.5 * float(u_cap_cm_s)
            v_fine = 0.25 * np.sqrt(KB * self.T_wall_K / M_HE)
            grid = VGrid(float(vmax_cm_s), float(vmax_cm_s), nvz, nvp, v_fine)
        self.g = grid
        g = self.g
        if g.nvz % 2:
            raise ValueError(
                "the DVM velocity grid needs an EVEN v_z bin count: an odd "
                "count places a bin at exactly v_z = 0, which neither "
                "transports nor mirrors under end-wall reflection "
                f"(got nvz={g.nvz})"
            )
        # Exact v_z -> -v_z bin map (the stretched axis is symmetric about
        # zero at half-offsets, so the mirror is a pure index reversal).
        self.mirror = np.arange(g.nvz)[::-1]

        # Zone rates, with the column<->annulus antisymmetry imposed
        # through the ACTUAL geometry volumes so that
        #     V_col * nu_x  ==  V_ann * nu_xp
        # holds to roundoff cell by cell (the particle ledger's zone
        # channel cancels exactly, which the ledger gate checks). Only the
        # mean chord and the surface split differ between the branches.
        vp = g.vp[None, :]
        Rp2 = Rp[:, None]
        Rm2 = Rm[:, None]
        gap = np.maximum(Rm2 - Rp2, 1e-9)
        if self.exchange_model == "geometric":
            # 2D Cauchy chord pi A / P on the cell cross-section; the
            # encounter splits between the two circles by perimeter.
            nu_total = 2.0 * vp / (np.pi * gap)
            self.nuxp = (Rp2 / (Rp2 + Rm2)) * nu_total
            self.nuw = (Rm2 / (Rp2 + Rm2)) * nu_total
        else:
            # Cauchy-chord branch, transcribed from KN2Zone's default:
            # the 3D chord 4V/S = 2 gap, split as Rp/Rm : 1 - Rp/Rm.
            nu_total = vp / (2.0 * gap)
            self.nuxp = (Rp2 / Rm2) * nu_total
            self.nuw = (1.0 - Rp2 / Rm2) * nu_total
        ratio = np.where(
            self.V_col > 0.0, self.V_ann / np.maximum(self.V_col, 1e-300), 0.0
        )
        self.nux = self.nuxp * ratio[:, None]
        # Cells with no annulus exchange with nothing and see no radial wall.
        no_ann = self.V_ann <= 0.0
        self.nux[no_ann] = 0.0
        self.nuxp[no_ann] = 0.0
        self.nuw[no_ann] = 0.0

        self.M_wall = g.wall_emission_spectrum(self.T_wall_K)
        self.M_cold = g.maxwellian(self.T_wall_K * KB / EV, 0.0)

        shape = (self.nz, g.nvz, g.nvp)
        self.f_c = np.zeros(shape)
        self.f_a = np.zeros(shape)
        # Pending end-wall returns, in PARTICLES per bin (inward half only).
        self.pend_L_c = np.zeros((g.nvz, g.nvp))
        self.pend_R_c = np.zeros((g.nvz, g.nvp))
        self.pend_L_a = np.zeros((g.nvz, g.nvp))
        self.pend_R_a = np.zeros((g.nvz, g.nvp))

        # Plasma-coupling accumulators, frozen between neutral updates and
        # read by the solver's RHS. Per-second densities on the PLASMA
        # volume (= the column volume).
        cells = np.zeros(self.nz)
        self.M_transfer = cells.copy()
        self.Ei_transfer = cells.copy()
        self.S_transfer = cells.copy()
        self.Tn_col_eV = np.full(self.nz, self.T_wall_K * KB / EV)
        self.updates = 0
        self.last_ledger = None

    # ------------------------------------------------------------ state

    def seed_from_density(self, nn_col, nn_ann, T_K=None):
        """Seed both distributions as Maxwellians at ``T_K`` (default wall)."""
        if T_K is None:
            spec = self.M_cold
        else:
            spec = self.g.maxwellian(float(T_K) * KB / EV, 0.0)
        self.f_c = np.asarray(nn_col, dtype=float)[:, None, None] * spec
        self.f_a = np.asarray(nn_ann, dtype=float)[:, None, None] * spec
        self.pend_L_c[...] = 0.0
        self.pend_R_c[...] = 0.0
        self.pend_L_a[...] = 0.0
        self.pend_R_a[...] = 0.0

    def snapshot(self):
        """Return a deep copy of every mutable piece of the DVM state."""
        return {
            "f_c": self.f_c.copy(),
            "f_a": self.f_a.copy(),
            "pend_L_c": self.pend_L_c.copy(),
            "pend_R_c": self.pend_R_c.copy(),
            "pend_L_a": self.pend_L_a.copy(),
            "pend_R_a": self.pend_R_a.copy(),
            "M_transfer": self.M_transfer.copy(),
            "Ei_transfer": self.Ei_transfer.copy(),
            "S_transfer": self.S_transfer.copy(),
            "Tn_col_eV": self.Tn_col_eV.copy(),
            "updates": int(self.updates),
        }

    def restore(self, snap):
        """Restore a :meth:`snapshot`."""
        self.f_c = snap["f_c"].copy()
        self.f_a = snap["f_a"].copy()
        self.pend_L_c = snap["pend_L_c"].copy()
        self.pend_R_c = snap["pend_R_c"].copy()
        self.pend_L_a = snap["pend_L_a"].copy()
        self.pend_R_a = snap["pend_R_a"].copy()
        self.M_transfer = snap["M_transfer"].copy()
        self.Ei_transfer = snap["Ei_transfer"].copy()
        self.S_transfer = snap["S_transfer"].copy()
        self.Tn_col_eV = snap["Tn_col_eV"].copy()
        self.updates = int(snap["updates"])

    # ---------------------------------------------------------- moments

    def column_density(self):
        """Column neutral density [cm^-3] -- the zeroth moment of ``f_c``."""
        return self.f_c.sum(axis=(1, 2))

    def annulus_density(self):
        """Annulus neutral density [cm^-3] -- the zeroth moment of ``f_a``."""
        return self.f_a.sum(axis=(1, 2))

    def column_drift(self):
        """Column axial drift ``<v_z>`` [cm/s]."""
        return _drift(self.f_c, self.g)

    def column_temperature_eV(self):
        """Column neutral temperature [eV] from the second central moment.

        ``T_n = (m/3) <|v - u_n|^2>`` with the three-dimensional measure
        (``v_perp`` already carries two degrees of freedom). Always
        computed; whether it FEEDS the fluid rate evaluations is a
        separate switch on the solver side.
        """
        return _temperature_eV(self.f_c, self.g)

    def f_inventory(self):
        """Particles carried by the distributions themselves."""
        return float(
            (self.f_c.sum(axis=(1, 2)) * self.V_col).sum()
            + (self.f_a.sum(axis=(1, 2)) * self.V_ann).sum()
        )

    def pending_inventory(self):
        """Particles held in the lagged end-wall return buffers."""
        return float(
            self.pend_L_c.sum()
            + self.pend_R_c.sum()
            + self.pend_L_a.sum()
            + self.pend_R_a.sum()
        )

    def total_inventory(self):
        """Domain particle inventory including the pending end buffers."""
        return self.f_inventory() + self.pending_inventory()

    # ------------------------------------------------------- the update

    def collision_frequencies(self, n_i, Ti_eV, u_i):
        """Return ``(nu_cx, nu_el)`` [1/s], shape ``(nz, nvz, nvp)``.

        BGK-like form: the loss frequency of a neutral AT velocity ``v``
        against the local ion Maxwellian, evaluated at the mean relative
        speed

            g_eff^2 = |v - u_i|^2 + 8 k T_i / (pi mu),   mu = m_He / 2

        (the standard interpolation between the drift-dominated and
        thermal-dominated limits for an equal-mass pair), with the Phelps
        He+/He backscatter cross section for charge exchange and the
        Phelps isotropic cross section for polarization-elastic
        scattering.

        Both channels are BGK FULL-REPLACEMENT events: the neutral is
        deleted and re-emitted at the local ion Maxwellian, so one event
        transfers the whole ``m (v - u_i)``. That is the correct weight
        for backscatter -- ``mu (1 - cos th) g`` at ``cos th = -1`` and
        ``mu = m/2`` is exactly ``m g`` -- so ``nu_cx`` is the Phelps
        backscatter rate unreduced. It is TWICE the correct weight for
        isotropic scattering, whose angular average ``<1 - cos th> = 1``
        gives ``mu g = m g / 2``. The returned ``nu_el`` therefore carries
        an explicit factor ``ELASTIC_BGK_MOMENTUM_FACTOR = 1/2``, which is
        not a tuning constant but the equal-mass ``mu/m`` ratio; with it
        the arm's effective momentum transfer is ``k_b + 0.5 k_iso``,
        exactly the superseded fluid operator
        ``phelps_momentum_transfer_rate_cm3_s``. The factor scales the
        whole elastic channel, so its energy transfer and its rebirth
        throughput are reduced in the same proportion.
        """
        g = self.g
        n_i = np.asarray(n_i, dtype=float)[:, None, None]
        Ti = np.maximum(np.asarray(Ti_eV, dtype=float), 1e-6)[:, None, None]
        u = np.asarray(u_i, dtype=float)[:, None, None]
        w2 = (g.VZ[None, :, :] - u) ** 2 + (g.VP**2)[None, :, :]
        # mu = m/2 for the symmetric pair, so 8 k T / (pi mu) = 16 k T/(pi m)
        g_eff = np.sqrt(w2 + 16.0 * Ti * EV / (np.pi * M_HE))
        E_rel = np.maximum(0.25 * M_HE * g_eff**2 / EV, 1e-9)
        nu_cx = n_i * phelps_he_backscatter_cm2(E_rel) * g_eff
        if self.elastic_model == "off":
            nu_el = np.zeros_like(nu_cx)
        else:
            nu_el = (
                ELASTIC_BGK_MOMENTUM_FACTOR
                * n_i
                * phelps_he_isotropic_cm2(E_rel)
                * g_eff
            )
        return nu_cx, nu_el

    def _march(self, dt, nu_c_loss, nu_a_loss, inflow_c, inflow_a):
        """Backward-Euler implicit upwind march (substep A).

        ``nu_c_loss`` / ``nu_a_loss`` are the per-(cell, bin) NON-zone loss
        frequencies of each zone; the zone-exchange rates are added here so
        the 2x2 coupling stays exactly antisymmetric. ``inflow_*`` are
        boundary ghost DENSITIES keyed ``(-1, +1)`` by domain end.

        Returns ``(f_c, f_a, mesh_c, mesh_a, out)`` where the mesh arrays
        are intercepted PARTICLES per emitting cell and ``out`` maps
        ``(zone, end)`` to the outgoing particles per bin.
        """
        g = self.g
        nz, nvz, nvp = self.nz, g.nvz, g.nvp
        f_c = np.zeros((nz, nvz, nvp))
        f_a = np.zeros((nz, nvz, nvp))
        mesh_c = np.zeros(nz)
        mesh_a = np.zeros(nz)
        out = {}
        inv_dt = 1.0 / dt
        for direction in (+1, -1):
            if direction > 0:
                order = range(nz)
                sel = g.vz > 0
                end_in, end_out = -1, +1
            else:
                order = range(nz - 1, -1, -1)
                sel = g.vz < 0
                end_in, end_out = +1, -1
            vz = np.abs(g.vz[sel])[:, None]
            F_c_prev = inflow_c[end_in][sel]
            F_a_prev = inflow_a[end_in][sel]
            for i in order:
                # Upstream face carries the inflow, downstream face the
                # outflow; both are throat areas, so what leaves one cell
                # is exactly what the next receives.
                fi = i if direction > 0 else i + 1
                fo = i + 1 if direction > 0 else i
                in_c = vz * self.face_c[fi] / self.V_col[i]
                out_c = vz * self.face_c[fo] / self.V_col[i]
                if self.V_ann[i] > 0.0:
                    in_a = vz * self.face_a[fi] / self.V_ann[i]
                    out_a = vz * self.face_a[fo] / self.V_ann[i]
                else:
                    in_a = np.zeros_like(vz)
                    out_a = np.zeros_like(vz)
                if fi == self.mesh_face:
                    blocked_c = (1.0 - self.transparency) * F_c_prev
                    blocked_a = (1.0 - self.transparency) * F_a_prev
                    j = min(max(i - direction, 0), nz - 1)
                    mesh_c[j] += float(
                        (blocked_c * vz).sum() * self.face_c[fi] * dt
                    )
                    mesh_a[j] += float(
                        (blocked_a * vz).sum() * self.face_a[fi] * dt
                    )
                    F_c_prev = self.transparency * F_c_prev
                    F_a_prev = self.transparency * F_a_prev
                nux = self.nux[i][None, :]
                nuxp = self.nuxp[i][None, :]
                a11 = inv_dt + out_c + nu_c_loss[i][sel] + nux
                a12 = -nux * np.ones_like(vz)
                a21 = -nuxp * np.ones_like(vz)
                a22 = inv_dt + out_a + nu_a_loss[i][sel] + nuxp
                r1 = self.f_c[i][sel] * inv_dt + in_c * F_c_prev
                r2 = self.f_a[i][sel] * inv_dt + in_a * F_a_prev
                det = a11 * a22 - a12 * a21
                fc = (r1 * a22 - a12 * r2) / det
                fa = (a11 * r2 - a21 * r1) / det
                f_c[i][sel] = fc
                f_a[i][sel] = fa
                F_c_prev, F_a_prev = fc, fa
            # The last cell marched empties across the open domain end; its
            # downstream-face loss IS the one-way outgoing flux there.
            last = nz - 1 if direction > 0 else 0
            fo_end = nz if direction > 0 else 0
            out[("c", end_out)] = np.zeros((nvz, nvp))
            out[("a", end_out)] = np.zeros((nvz, nvp))
            out[("c", end_out)][sel] = (
                f_c[last][sel] * vz * self.face_c[fo_end] * dt
            )
            out[("a", end_out)][sel] = (
                f_a[last][sel] * vz * self.face_a[fo_end] * dt
            )
        return f_c, f_a, mesh_c, mesh_a, out

    def update(
        self,
        dt,
        *,
        n_i,
        Ti_eV,
        u_i,
        nu_ion,
        sources=None,
        T_s_K=None,
    ):
        """Advance ``(f_c, f_a)`` by one neutral-clock tick of ``dt`` seconds.

        ``nu_ion`` is the velocity-BLIND ionization frequency per cell
        [1/s] -- the registered channel-1 convention; the solver derives
        it from the ionization the plasma actually books, so the two sides
        remove and create the same particles. ``sources`` holds the
        external ledger in atoms/s: ``puff`` (annulus cells),
        ``recombination`` (column cells), ``cathode_face`` /
        ``collector_face`` (scalars, column ends), ``anode`` (column
        cells). ``T_s_K`` is the live cathode-surface temperature used for
        the cathode-adjacent surfaces (the stated special case); the wall
        temperature is used everywhere else.

        Returns the ledger of this update: every birth and loss channel in
        PARTICLES, plus the inventory before/after, so that

            inventory_after - inventory_before == sum(births) - sum(losses)

        holds to roundoff.
        """
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"the DVM update needs a positive dt (got {dt})")
        g = self.g
        sources = {} if sources is None else sources
        T_s_K = self.T_wall_K if T_s_K is None else float(T_s_K)
        inv_before = self.total_inventory()
        f_before = self.f_inventory()

        nu_ion = np.asarray(nu_ion, dtype=float)
        nu_cx, nu_el = self.collision_frequencies(n_i, Ti_eV, u_i)
        nu_c_loss = nu_ion[:, None, None] + nu_cx + nu_el
        nu_a_loss = self.nuw[:, None, :] * np.ones((self.nz, g.nvz, g.nvp))

        # --- boundary inflow: last update's pending returns, as ghost
        # densities that inject exactly the buffered particle count.
        inflow_c = {
            -1: _ghost_density(self.pend_L_c, self.face_c[0], dt, g),
            +1: _ghost_density(self.pend_R_c, self.face_c[-1], dt, g),
        }
        inflow_a = {
            -1: _ghost_density(self.pend_L_a, self.face_a[0], dt, g),
            +1: _ghost_density(self.pend_R_a, self.face_a[-1], dt, g),
        }
        birth_return_L = float(self.pend_L_c.sum() + self.pend_L_a.sum())
        birth_return_R = float(self.pend_R_c.sum() + self.pend_R_a.sum())

        f_c, f_a, mesh_c, mesh_a, out = self._march(
            dt, nu_c_loss, nu_a_loss, inflow_c, inflow_a
        )

        # --- substep A tallies, in PARTICLES, from the marched state
        vol_c = self.V_col[:, None, None]
        vol_a = self.V_ann[:, None, None]
        L_ion = nu_ion[:, None, None] * f_c * dt * vol_c
        L_cx = nu_cx * f_c * dt * vol_c
        L_el = nu_el * f_c * dt * vol_c
        L_wall = self.nuw[:, None, :] * f_a * dt * vol_a

        # --- substep B: births at exactly the tallied masses
        M_i = np.empty((self.nz, g.nvz, g.nvp))
        Ti_arr = np.asarray(Ti_eV, dtype=float)
        u_arr = np.asarray(u_i, dtype=float)
        for i in range(self.nz):
            M_i[i] = g.maxwellian(max(float(Ti_arr[i]), 0.02), float(u_arr[i]))

        N_cx = L_cx.sum(axis=(1, 2))
        N_el = L_el.sum(axis=(1, 2))
        N_wall = L_wall.sum(axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_vc = np.where(self.V_col > 0.0, 1.0 / self.V_col, 0.0)
            inv_va = np.where(self.V_ann > 0.0, 1.0 / self.V_ann, 0.0)

        birth_cx = (N_cx * inv_vc)[:, None, None] * M_i
        birth_el = (N_el * inv_vc)[:, None, None] * M_i
        f_c += birth_cx + birth_el

        alpha = self.accommodation
        birth_wall_acc = (
            alpha * (N_wall * inv_va)[:, None, None] * self.M_wall[None, :, :]
        )
        # Specular reflection off the cylindrical wall reverses only the
        # radial direction, which this axisymmetric grid does not resolve:
        # the bin is unchanged, so the reflected fraction returns exactly
        # where it left.
        birth_wall_ref = (1.0 - alpha) * L_wall * inv_va[:, None, None]
        f_a += birth_wall_acc + birth_wall_ref

        # Anode-mesh interception re-emits at the wall temperature in the
        # cell it was intercepted from, on both sides of the mesh.
        f_c += (mesh_c * inv_vc)[:, None, None] * self.M_wall[None, :, :]
        f_a += (mesh_a * inv_va)[:, None, None] * self.M_wall[None, :, :]

        # --- external source ledger (counted particles this update)
        puff = np.asarray(sources.get("puff", 0.0), dtype=float) * dt
        rec = np.asarray(sources.get("recombination", 0.0), dtype=float) * dt
        anode = np.asarray(sources.get("anode", 0.0), dtype=float) * dt
        cath = float(sources.get("cathode_face", 0.0)) * dt
        coll = float(sources.get("collector_face", 0.0)) * dt
        if puff.ndim:
            # Registered channel 5: the puff is born as a 300 K Maxwellian
            # at rest -- the zero-momentum convention, as a distribution.
            f_a += (puff * inv_va)[:, None, None] * self.M_cold
        if rec.ndim:
            f_c += (rec * inv_vc)[:, None, None] * M_i
        if anode.ndim:
            f_c += (anode * inv_vc)[:, None, None] * self.M_wall[None, :, :]
        if cath:
            f_c[0] += cath * inv_vc[0] * g.half_flux_spectrum(T_s_K, +1)
        if coll:
            f_c[-1] += coll * inv_vc[-1] * g.half_flux_spectrum(
                self.T_wall_K, -1
            )

        # --- end walls: pump what sticks, buffer what returns
        out_L = float(out[("c", -1)].sum() + out[("a", -1)].sum())
        out_R = float(out[("c", +1)].sum() + out[("a", +1)].sum())
        self.pend_L_c = _end_return(
            out[("c", -1)], self.s_L, alpha, self.mirror,
            g.half_flux_spectrum(T_s_K, +1),
        )
        self.pend_L_a = _end_return(
            out[("a", -1)], self.s_L, alpha, self.mirror,
            g.half_flux_spectrum(T_s_K, +1),
        )
        self.pend_R_c = _end_return(
            out[("c", +1)], self.s_R, alpha, self.mirror,
            g.half_flux_spectrum(self.T_wall_K, -1),
        )
        self.pend_R_a = _end_return(
            out[("a", +1)], self.s_R, alpha, self.mirror,
            g.half_flux_spectrum(self.T_wall_K, -1),
        )

        self.f_c = f_c
        self.f_a = f_a
        inv_after = self.total_inventory()

        # --- plasma coupling: minus the moments of the kinetic operators
        self._book_transfer(dt, L_ion, L_cx, L_el, birth_cx, birth_el, rec,
                            M_i, u_arr)
        self.Tn_col_eV = self.column_temperature_eV()
        self.updates += 1

        ledger = {
            "dt": dt,
            "inventory_before": inv_before,
            "inventory_after": inv_after,
            "f_inventory_before": f_before,
            "f_inventory_after": self.f_inventory(),
            "loss_ionization": float(L_ion.sum()),
            "loss_cx": float(L_cx.sum()),
            "loss_elastic": float(L_el.sum()),
            "loss_wall": float(N_wall.sum()),
            "loss_mesh_blocked": float(mesh_c.sum() + mesh_a.sum()),
            "loss_end_out_L": out_L,
            "loss_end_out_R": out_R,
            "loss_pump_L": self.s_L * out_L,
            "loss_pump_R": self.s_R * out_R,
            "birth_cx": float(N_cx.sum()),
            "birth_elastic": float(N_el.sum()),
            "birth_wall_accommodated": float(alpha * N_wall.sum()),
            "birth_wall_reflected": float((1.0 - alpha) * N_wall.sum()),
            "birth_mesh_reemit": float(mesh_c.sum() + mesh_a.sum()),
            "birth_end_return_L": birth_return_L,
            "birth_end_return_R": birth_return_R,
            "birth_puff": float(puff.sum()),
            "birth_recombination": float(rec.sum()),
            "birth_cathode_face": cath,
            "birth_collector_face": coll,
            "birth_anode": float(anode.sum()),
        }
        self.last_ledger = ledger
        return ledger

    def _book_transfer(self, dt, L_ion, L_cx, L_el, birth_cx, birth_el, rec,
                       M_i, u_i):
        """Book the plasma-side momentum/energy/particle transfer.

        Every entry is MINUS a measured moment of a kinetic operator, so
        the fluid gain and the kinetic loss are antisymmetric to roundoff
        by construction rather than by agreement of two formulas:

        - ionization: the plasma gains the whole momentum and energy of
          the ionized population (registered channel 1, refining the R4.2
          ``(u_n, 300 K)`` booking);
        - charge exchange and elastic scattering: the plasma gains what
          the lost neutrals carried and pays for what the replacement
          neutrals carry away;
        - recombination: the plasma pays for the born neutral.

        The energy moment is a TOTAL kinetic energy; the fluid ``Ei`` row
        is an internal energy, so the bulk term is removed with the same
        decomposition the ``ionization_birth_energy_model="conservative"``
        booking uses, ``d(KE) = u dM - (1/2) m u^2 dN``.
        """
        g = self.g
        VZ = g.VZ[None, :, :]
        V2 = g.V2[None, :, :]
        vol_c = np.maximum(self.V_col, 1e-300)

        def moments(counts):
            return (
                counts.sum(axis=(1, 2)),
                M_HE * (counts * VZ).sum(axis=(1, 2)),
                0.5 * M_HE * (counts * V2).sum(axis=(1, 2)),
            )

        N_ion, P_ion, E_ion = moments(L_ion)
        _, P_cx_l, E_cx_l = moments(L_cx + L_el)
        # Births are densities per bin; convert back to particles.
        births = (birth_cx + birth_el) * self.V_col[:, None, None]
        _, P_cx_b, E_cx_b = moments(births)
        rec_counts = (
            np.zeros((self.nz, g.nvz, g.nvp))
            if not np.ndim(rec)
            else np.asarray(rec, dtype=float)[:, None, None] * M_i
        )
        N_rec, P_rec, E_rec = moments(rec_counts)

        P = P_ion + (P_cx_l - P_cx_b) - P_rec
        E = E_ion + (E_cx_l - E_cx_b) - E_rec
        S = N_ion - N_rec
        scale = 1.0 / (vol_c * dt)
        self.M_transfer = P * scale
        self.S_transfer = S * scale
        u = np.asarray(u_i, dtype=float)
        self.Ei_transfer = (
            E * scale - u * self.M_transfer + 0.5 * M_HE * u**2 * self.S_transfer
        )


def _throat_areas(cell_areas):
    """Return the ``nz+1`` face areas of a per-cell area profile [cm^2].

    Interior faces take the throat ``min`` of the two cells they join, so
    a narrowing (or a vanishing annulus) throttles the flux from both
    sides identically. The two domain-end faces take their own cell's
    area: both ends are open.
    """
    a = np.asarray(cell_areas, dtype=float)
    faces = np.empty(a.size + 1)
    faces[0] = a[0]
    faces[-1] = a[-1]
    faces[1:-1] = np.minimum(a[:-1], a[1:])
    return faces


def ledger_residual(ledger):
    """Return the ledger's particle-closure residuals.

    ``distribution``: ``Delta(sum f V)`` minus (all births - all losses),
    which is the statement that substep B creates exactly what substep A
    destroyed. ``domain``: ``Delta(inventory incl. pending)`` minus
    (external births - ionization - pumping), the physical closure with
    every internal channel cancelled.

    Both are absolute particle counts. The relative forms divide by the
    throughput PLUS the standing inventory, because each identity is a
    difference of two inventories: on a short neutral tick the throughput
    can be many orders below the inventory, and the floating-point noise
    floor of the statement is then set by the inventory, not by the
    handful of particles that moved. Normalizing by throughput alone would
    report cancellation noise as a conservation error.
    """
    births = sum(
        v for k, v in ledger.items() if k.startswith("birth_")
    )
    losses = sum(
        v
        for k, v in ledger.items()
        if k.startswith("loss_") and not k.startswith("loss_pump_")
    )
    distribution = (
        ledger["f_inventory_after"] - ledger["f_inventory_before"]
    ) - (births - losses)
    external = sum(
        ledger[f"birth_{name}"] for name in LEDGER_EXTERNAL_BIRTHS
    )
    domain = (
        ledger["inventory_after"] - ledger["inventory_before"]
    ) - (
        external
        - ledger["loss_ionization"]
        - ledger["loss_pump_L"]
        - ledger["loss_pump_R"]
    )
    throughput = births + losses + 1e-300
    scale = throughput + abs(ledger["inventory_before"])
    return {
        "distribution": distribution,
        "domain": domain,
        "throughput": throughput,
        "scale": scale,
        "distribution_rel": distribution / scale,
        "domain_rel": domain / scale,
    }


def _ghost_density(pending, area_cm2, dt, g):
    """Convert buffered particles per bin into a boundary ghost density.

    The march injects ``|v_z| F A dt`` particles per bin, so dividing the
    buffered count by exactly that factor injects the buffered count --
    independent of ``dt``, which is what lets the neutral clock change
    cadence without leaking particles.
    """
    if area_cm2 <= 0.0 or not np.any(pending):
        return np.zeros_like(pending)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(
            np.abs(g.VZ) > 0.0,
            pending / (np.abs(g.VZ) * area_cm2 * dt),
            0.0,
        )
    return dens


def _end_return(outgoing, sticking, accommodation, mirror, spectrum):
    """Split an end-wall outflow into the buffered return, per bin.

    The pumped fraction ``sticking`` leaves. Of the rest, the
    accommodated fraction is re-emitted cosine-distributed at the surface
    temperature (``spectrum``, already an inward half-flux distribution),
    and the remainder is reflected at the incident energy, which on the
    symmetric ``v_z`` axis is the exact bin mirror.
    """
    back = (1.0 - float(sticking)) * outgoing
    total = float(back.sum())
    reflected = (1.0 - float(accommodation)) * back[mirror, :]
    accommodated = float(accommodation) * total * spectrum
    return reflected + accommodated


def _drift(f, g):
    n = f.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            n > 0.0,
            (f * g.VZ[None, :, :]).sum(axis=(1, 2)) / np.maximum(n, 1e-300),
            0.0,
        )


def _temperature_eV(f, g):
    n = f.sum(axis=(1, 2))
    u = _drift(f, g)
    c2 = (g.VZ[None, :, :] - u[:, None, None]) ** 2 + (g.VP**2)[None, :, :]
    mean_c2 = np.where(
        n > 0.0, (f * c2).sum(axis=(1, 2)) / np.maximum(n, 1e-300), 0.0
    )
    return M_HE * mean_c2 / (3.0 * EV)
