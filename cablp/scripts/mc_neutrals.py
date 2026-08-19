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

``--fast-reflected`` is a SEPARATE, self-contained mode over the same
background (:func:`run_fast_reflected`): it launches ONLY the cathode jet's
fast reflected lobe, transports it with an energy-resolved Phelps-Qb CX rate,
and reports where that lobe deposits inside the plasma column. It shares no
tally, source or default with the source-menu run above. The shipped ``-n``
default of 200,000 histories puts the binomial error on its headline ``f_dep``
below 0.12 % absolute -- an order of magnitude inside the 1 % the read asks
for -- at any ``f_dep``.

Usage:
    python scripts/mc_neutrals.py RUN.h5 [-n 200000] [--jet {none,cathode,both}]
        [--window 5 19.5] [--seed 1] [--out PREFIX]
    python scripts/mc_neutrals.py RUN.h5 --fast-reflected [-n 200000]
        [--r-e 0.2] [--r-n 0.5] [--window 5 19.5] [--seed 1] [--out PREFIX]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cablp.funcs._adas import he_rates
from cablp.funcs._cross import charge_ex_react, phelps_he_backscatter_cm2
from cablp.solvers._sim1d.core.geometry import (
    absorbing_live_cells_by_role,
    build_geometry,
)
from cablp.solvers._sim1d.physics.neutrals import (
    gas_puff_rate_profile,
    puff_particles_per_s,
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

# Energy range [eV] over which the Phelps He+/He backscatter cross section
# phelps_he_backscatter_cm2 is data-backed: the span of the archived LXCat
# download vars/he_ion_neutral_phelps_lxcat.txt, whose tabulated He+/He
# "Backscat" block runs 0 -> 1.0e4 eV (the analytic form's 1+5/E factor gives
# it a finite ~2.21e-15 cm^2 limit as E -> 0, so the low end is a table
# endpoint and not a singularity). --fast-reflected refuses a launch energy
# outside it rather than extrapolating the fit.
PHELPS_QB_RANGE_EV = (1.0e-4, 1.0e4)

# THE ESCAPE DEFINITION of the --fast-reflected mode, quoted verbatim into both
# its docstring and its output header so the reviewer reads the same sentence
# the code implements. It is the load-bearing choice of the whole read.
FAST_ESCAPE_DISCLOSURE = (
    "A fast atom LEAVES the fast population at its FIRST CROSSING of the local\n"
    "plasma column radius Rp(z) -- the column surface, NOT the vessel wall Rm.\n"
    "Crossing counts in both forms: outward through the cylindrical surface\n"
    "r = Rp(icell), and axially across a z-edge into a cell whose Rp is smaller\n"
    "than the atom's current radius (the column's own annular step). Nothing is\n"
    "reflected and nothing re-enters: the atom is gone from the tally the moment\n"
    "it is outside the column, because the deliverable is the IN-COLUMN\n"
    "deposition fraction the phase-1 kernel's survival factor consumes, and a\n"
    "fast atom outside the column deposits nowhere the kernel can see."
)


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


def assert_two_zone_puff_live(ns, ns_ann, path, window_ms, expected_per_s=0.0):
    """Raise when a two-zone ledger's annulus puff is missing from the menu.

    ``ns`` is the assembled per-cell source row [s^-1] and ``ns_ann`` the
    annulus rate row it was built from (``None`` on a single-zone run, and on
    a two-zone artifact predating the per-term annulus rows).

    Under ``neutral_two_zone`` the gas puff is booked into the ANNULUS row
    (``nn_a``), so a menu assembled from the column row alone reads zero puff
    on every such background and the TPMC runs unfuelled. A nonzero annulus
    source with an empty assembled row is that defect and nothing else: a
    genuinely unfuelled window leaves both zero and does not raise.

    ``expected_per_s`` [s^-1] is the puff rate the background is known to
    carry from evidence OTHER than that row -- the config-derived rate of
    ``two_zone_puff_row_from_config`` on an artifact that saved no annulus
    row. Without it this guard would be structurally inert on exactly the
    artifacts whose puff is hardest to recover, which is where it is most
    needed; with it, a derivation that silently failed to land in the menu
    raises here rather than running the TPMC unfuelled.
    """
    live = expected_per_s > 0.0 or (
        ns_ann is not None and bool(np.any(ns_ann > 0.0))
    )
    if not live or np.any(ns > 0.0):
        return
    raise ValueError(
        f"two-zone background is known to be puffing over "
        f"{window_ms[0]}-{window_ms[1]} ms but the assembled source menu has "
        f"no puff, for {path}.\n"
        f"  annulus row nonzero: {ns_ann is not None and bool(np.any(ns_ann > 0.0))}"
        f"   config-derived rate: {expected_per_s:.6e} /s\n"
        "  likely cause: the menu was read from rhs_terms/neutral_sources/nn "
        "alone. Under the neutral_two_zone closure the puff enters at the "
        "wall and is booked into the nn_a row; the nn row is pump-only, so a "
        "column-only read degrades the puff to nothing. On an artifact that "
        "saved no such row the config derivation is the only route, and it "
        "did not land. Refusing to run source-starved."
    )


def assert_end_recycle_routed_live(ba, ba_ann, path, window_ms):
    """Raise when a routed end recycle is missing from the assembled menu.

    ``ba`` is the assembled per-cell boundary-recycle row [s^-1] and
    ``ba_ann`` the annulus rate row it was built from (``None`` on a run whose
    boundary term carries no ``nn_a`` row, which returns immediately).

    Under ``end_recycle_to_annulus`` the collector faces' recycle is booked
    into the ANNULUS row, so a menu assembled from the column row alone loses
    the end recycle entirely -- and, unlike the all-zero case, it does so
    while the CATHODE row stays nonzero, which is precisely why
    ``assert_recycle_channel_live`` cannot see it.
    """
    if ba_ann is None:
        return
    routed = np.asarray(ba_ann) > 0.0
    if not np.any(routed) or np.any(np.asarray(ba)[routed] > 0.0):
        return
    raise ValueError(
        f"boundary recycle was routed into the annulus over "
        f"{window_ms[0]}-{window_ms[1]} ms but the assembled recycle row is "
        f"zero on every routed cell, for {path}.\n"
        "  likely cause: the menu was read from rhs_terms/<boundary>/nn "
        "alone. Under the end_recycle_to_annulus closure the collector faces "
        "rebirth their stream into the nn_a row on the annulus volume; a "
        "column-only read degrades the end recycle to nothing while leaving "
        "the cathode face intact. Refusing to run source-starved."
    )


def _square_puff_envelope(times, params, flags, t_breakdown_trigger):
    """Return the ``gas_puff_mode="square"`` envelope [1] at ``times`` [s].

    A transcription of the solver's own envelope (the ``"square"`` branch of
    ``LAPDSim1D._effective_gas_puff_sccm``): an erf opening edge anchored on
    the end of the neutral-prebreakdown phase plus ``gas_puff_rise_center_s``,
    an erf closing edge anchored on the main-discharge start plus
    ``tau_discharge`` and ``gas_puff_close_lag_s``, both built with the one
    shared width, clamped at zero where they are configured to overlap.

    Every input is either saved config or the saved ``t_breakdown_trigger``
    attribute, so the envelope is a function of the ARTIFACT and never of
    solver run state. ``math.erf`` is used rather than a vectorized library
    erf because the solver uses ``math.erf``, and this function is only worth
    having if it reproduces the applied waveform bit for bit.
    """
    origin = 0.0
    if flags.get("Plasma", True) and flags.get("neutral_prebreakdown", False):
        origin = max(float(params.get("tau_neutral_prebreakdown", 0.0)), 0.0)
    width = float(params.get("gas_puff_rise_width_s", 5.0e-4))
    t_on = origin + float(params.get("gas_puff_rise_center_s", 5.0e-4))
    t_close = (
        float(t_breakdown_trigger)
        + max(float(params.get("tau_discharge", 0.0)), 0.0)
        + float(params.get("gas_puff_close_lag_s", 5.0e-4))
    )
    t = np.asarray(times, dtype=float)
    rise = 0.5 * (1.0 + np.array([math.erf(x) for x in (t - t_on) / width]))
    fall = 0.5 * (1.0 + np.array([math.erf(x) for x in (t - t_close) / width]))
    return np.maximum(rise - fall, 0.0)


def two_zone_puff_row_from_config(f, params, flags, times, mask, Vm_full, Va_full):
    """Return ``(per-cell puff row [s^-1], provenance)`` derived from the config.

    For the two-zone artifacts written BEFORE the per-term annulus rows
    existed. Under ``neutral_two_zone`` the puff is booked into the ANNULUS --
    the pipe enters at the vessel wall -- so on such an artifact the neutral
    ledger carries no trace of it at all: the column row is pump-only and a
    per-zone read of it returns a GENUINE zero, indistinguishable from an
    unfuelled window. This is the anode fallback's situation one term over,
    with one difference that matters: the anode's total is recoverable from
    the plasma-side row, and the puff's is not recoverable from the ledger at
    ALL. Config is the only surviving evidence, so it is used, and labelled.

    The derivation is the solver's own construction, not a paraphrase of it:
    the per-cell shape comes from ``gas_puff_rate_profile`` called with the
    artifact's resolved config on a geometry rebuilt from that same config
    (and asserted identical to the saved one), and the applied level is that
    profile scaled by the ``"square"`` envelope, which is linear in the
    configured flow. Cells with an annulus take the whole per-cell rate; cells
    without one are already carried by the column row, exactly as the solver
    splits them.

    ``mask`` selects the plateau window; the returned row is the window MEAN,
    the same reduction the ledger rows get, so the menu entries stay
    commensurable.

    Raises ``ValueError`` -- loudly, rather than returning a number nobody can
    check -- on any configuration outside the certified one: a non-``square``
    waveform, a phase-transition mode whose main-discharge start is not the
    saved trigger, a geometry that does not rebuild, or a missing trigger.
    """
    mode = str(params.get("gas_puff_mode", "square"))
    if mode != "square":
        raise ValueError(
            f"UNRECOVERABLE two-zone puff: gas_puff_mode={mode!r}.\n"
            "  The artifact saved no rhs_terms/neutral_sources/nn_a row, so "
            "the puff is absent from the neutral ledger and only the config "
            "can supply it; the config derivation implemented here covers the "
            "'square' waveform alone. Refusing to guess a rate."
        )
    transition = str(params.get("phase_transition_mode", "current"))
    if transition != "current":
        raise ValueError(
            f"UNRECOVERABLE two-zone puff: phase_transition_mode="
            f"{transition!r}.\n"
            "  The square waveform's closing edge is anchored on the "
            "main-discharge start, which is the saved t_breakdown_trigger "
            "only under the 'current' transition mode. Refusing to guess a "
            "closing time."
        )
    if "t_breakdown_trigger" not in f.attrs:
        raise ValueError(
            "UNRECOVERABLE two-zone puff: no t_breakdown_trigger attribute, "
            "so the square waveform's closing edge has no anchor."
        )
    geometry = build_geometry(params, flags)
    Vm_geo = np.asarray(geometry.neutral_volume_cm3, dtype=float)
    if Vm_geo.shape != Vm_full.shape or not np.array_equal(Vm_geo, Vm_full):
        raise ValueError(
            "UNRECOVERABLE two-zone puff: the geometry rebuilt from the saved "
            "config does not reproduce the saved neutral volumes, so the "
            "puff profile cannot be placed on the artifact's own cells."
        )
    profile = gas_puff_rate_profile(
        geometry,
        params.get("S_gp", 0.0),
        params.get("gas_puff_valves", 2),
        profile=str(params.get("gas_puff_profile", "cell")),
        z_cm=params.get("gas_puff_z_cm"),
        sigma_cm=float(params.get("gas_puff_sigma_cm", 50.0)),
        throw_cm=float(params.get("gas_puff_throw_cm", 100.0)),
        end=0,
        delivery_fraction=float(params.get("gas_puff_delivery_fraction", 1.0)),
    )
    if flags.get("TwinCathode", False):
        profile = profile + gas_puff_rate_profile(
            geometry,
            params.get("Twin_S_gp", 0.0),
            params.get("gas_puff_valves", 2),
            profile=str(params.get("gas_puff_profile", "cell")),
            z_cm=params.get("gas_puff_z_cm"),
            sigma_cm=float(params.get("gas_puff_sigma_cm", 50.0)),
            throw_cm=float(params.get("gas_puff_throw_cm", 100.0)),
            end=-1,
            delivery_fraction=float(
                params.get("gas_puff_delivery_fraction", 1.0)
            ),
        )
    envelope = _square_puff_envelope(
        np.asarray(times, dtype=float)[mask],
        params,
        flags,
        f.attrs["t_breakdown_trigger"],
    )
    # The per-sample phase gate the solver applied, as saved. It zeroes the
    # whole puff term when shut, so it multiplies the envelope exactly.
    if "phase_gas_puff_enabled" in f:
        envelope = envelope * np.asarray(f["phase_gas_puff_enabled"][:])[mask]
    scale = float(np.mean(envelope))
    row = np.where(Va_full > 0.0, profile * Vm_full * scale, 0.0)
    total = float(row.sum())
    if total <= 0.0:
        # A genuinely shut valve (gas_puff_enabled off, or a window entirely
        # outside the waveform). Nothing was derived, so nothing is labelled.
        return row, None
    provenance = (
        "puff derived from the resolved config (gas_puff_rate_profile at the "
        "saved S_gp/profile/geometry, scaled by the square waveform envelope "
        f"averaged over the window, {scale:.9f}): two-zone artifact carrying "
        "no rhs_terms/neutral_sources/nn_a row, so the annulus-booked puff is "
        "absent from the neutral ledger entirely and the column row reads a "
        "genuine zero"
    )
    return row, provenance


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
        t_abs = f["time"][:]
        t = (t_abs - t0) * 1e3
        m = (t >= window_ms[0]) & (t <= window_ms[1])
        # Read here rather than at the point of use: the config-derived puff
        # below needs them, and there is only ever one copy.
        params = json.loads(f.attrs["params_json"])
        raw_flags = f.attrs.get("flags_json")
        flags = json.loads(raw_flags) if raw_flags is not None else {}
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
        # VOLUME CONVENTION of the neutral ledger, which the two-zone closure
        # changes wholesale and per-term-row presence does NOT report. On a
        # single-zone run every term's nn row is a chamber-mean density rate
        # and integrates on Vm. On a two-zone run (the state carries nn_a) the
        # nn row is the COLUMN density rate and integrates on Vp, while
        # whatever the term books into the annulus lives in its own nn_a row
        # on Vm - Vp. The stance is a property of the STATE, so it is read
        # from the state dataset: a two-zone run whose term books nothing into
        # the annulus -- or which predates the per-term annulus rows -- still
        # has its nn row on Vp, and keying the convention off the per-term row
        # would silently restore the Vm over-count there.
        two_zone = "nn_a" in f
        Va_full = np.maximum(Vm_full - Vp_full, 0.0)
        # The boundary row is stance-dependent (see boundary_recycle_row).
        row, stance = boundary_recycle_row(f)
        # Plasma removal as booked by EITHER row: the guard's reference, so a
        # channel read from the wrong row cannot pass as a genuinely empty one.
        removal_any = sum(
            -np.mean(f[f"rhs_terms/{name}/n"][:][m], axis=0) * Vp_full
            for name in BOUNDARY_ROWS
            if f"rhs_terms/{name}/n" in f
        )
        # Volume-integrated boundary recycle. Both zones, each on its OWN
        # volume: under neutral_two_zone the boundary term's nn row lives on
        # the column volume Vp, and under the end_recycle_to_annulus closure
        # the COLLECTOR faces' share is booked into nn_a on the annulus
        # (Vm - Vp) instead. A column-only read then drops the entire end
        # recycle -- the same defect the two-zone puff had, one term over --
        # and would leave the cathode row nonzero, so the existing
        # recycle-channel guard would not catch it. Reduces to the single-zone
        # nn * Vm on a single-zone state.
        #
        # The stance comes from two_zone (the STATE), not from row presence: a
        # July-era two-zone artifact saved no per-term annulus rows at all, and
        # keying off the row there restored the Vm over-count -- x Vm/Vp,
        # measured 11.1 on es1_nx120_m6_sq3400_2z_es1.h5. The column-row-on-Vp
        # read IS the state-level truth for face recycle on such an artifact:
        # the boundary term rebirths exactly what it removed, so it reproduces
        # the plasma-side removal -n * Vp cell for cell (verified equal at the
        # cathode cell, the collector cell and in total on that artifact).
        ba_col = np.mean(f[f"rhs_terms/{row}/nn"][:][m], axis=0)
        ba_ann = None
        if f"rhs_terms/{row}/nn_a" in f:
            ba_ann = np.mean(f[f"rhs_terms/{row}/nn_a"][:][m], axis=0)
        if not two_zone:
            ba = ba_col * Vm_full
        elif ba_ann is None:
            ba = ba_col * Vp_full
        else:
            ba = ba_col * Vp_full + ba_ann * np.maximum(Vm_full - Vp_full, 0.0)
        # Volume-integrated anode-mesh collection, both zones each on its OWN
        # volume. Under neutral_two_zone the mesh feeds the ANNULUS wherever
        # there is one (physics/sources.anode_collection_rhs), so the column
        # row is identically zero on every cell flanking the mesh and the old
        # nn * Vm read returned EXACTLY ZERO on every two-zone background --
        # the same defect the two-zone puff had, one term over, and worse,
        # because a vanished channel simply drops out of the menu without a
        # ratio to notice.
        an_col = np.mean(f["rhs_terms/anode_collection/nn"][:][m], axis=0)
        an_ann = None
        if "rhs_terms/anode_collection/nn_a" in f:
            an_ann = np.mean(
                f["rhs_terms/anode_collection/nn_a"][:][m], axis=0
            )
        an_provenance = None
        if not two_zone:
            an = an_col * Vm_full
        elif an_ann is not None:
            an = an_col * Vp_full + an_ann * Va_full
        else:
            # July-era two-zone artifact, saved before the per-term annulus
            # rows existed: the collected stream is nowhere in the neutral
            # ledger at all, so a per-zone read of it would return zero and
            # the menu would silently lose the channel. Derive it from the
            # PLASMA-side row instead -- which is EXACT here, not an
            # approximation: anode_collection books n = -dN_loss / Vp and
            # rebirths exactly dN_loss as neutrals, split between the zones,
            # so -n * Vp IS the total rebirth rate whatever the split was.
            # Same construction as the K4a kinetic fallback below, and
            # labelled in the returned menu so the derivation is never
            # mistaken for a ledger read.
            an = -np.mean(
                f["rhs_terms/anode_collection/n"][:][m], axis=0
            ) * Vp_full
            an_provenance = (
                "anode_collection derived from the plasma-side row (-n * Vp): "
                "two-zone artifact carrying no rhs_terms/anode_collection/nn_a "
                "row, so the annulus booking is absent from the neutral ledger"
            )
            print(
                f"[load_background] {an_provenance}, for {path}.",
                file=sys.stderr,
            )
        # Volume-integrated POSITIVE part of the neutral_sources row -- the
        # puff (the pump is its negative part). Under the two-zone closure the
        # puff is booked into the ANNULUS row (nn_a): the pipe enters at the
        # vessel wall, so the column row nn is pump-only and a menu assembled
        # from it alone loses the puff entirely. Integrate each zone on its
        # own volume, exactly as bg["nn_model"] is rebuilt above -- nn lives
        # on the column volume (Vp) and nn_a on the annulus (Vm - Vp) -- which
        # reduces to the single-zone nn * Vm on a single-zone state. As for
        # the recycle row above, the stance is the STATE's: on a two-zone
        # artifact predating the per-term annulus rows the column row is
        # pump-only and integrates on Vp, and the puff is recovered separately
        # below because nothing in the ledger carries it.
        ns_col = np.mean(
            np.clip(f["rhs_terms/neutral_sources/nn"][:][m], 0.0, None), axis=0
        )
        ns_ann = None
        if "rhs_terms/neutral_sources/nn_a" in f:
            ns_ann = np.mean(
                np.clip(
                    f["rhs_terms/neutral_sources/nn_a"][:][m], 0.0, None
                ),
                axis=0,
            )
        if not two_zone:
            ns = ns_col * Vm_full
        elif ns_ann is None:
            ns = ns_col * Vp_full
        else:
            ns = ns_col * Vp_full + ns_ann * np.maximum(Vm_full - Vp_full, 0.0)
        kinetic_fallback = bool(not np.any(ba) and "nn_a" in f)
        if kinetic_fallback:
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
            ns[puff_idx] = puff_particles_per_s(sccm, valves)
        ns_provenance = None
        ns_expected = 0.0
        if two_zone and ns_ann is None and not kinetic_fallback:
            # July-era two-zone artifact: the puff is booked into the annulus
            # and no annulus row survives, so the assembled row above is a
            # genuine zero and the ledger has nothing else to offer. Derive it
            # from the resolved config, label it, and say so on stderr -- the
            # anode fallback's precedent. The kinetic fallback is excluded
            # because it has already rebuilt the whole menu, its own puff
            # entry included.
            ns_derived, ns_provenance = two_zone_puff_row_from_config(
                f, params, flags, t_abs, m, Vm_full, Va_full
            )
            ns = ns + ns_derived
            ns_expected = float(ns_derived.sum())
            if ns_provenance is not None:
                print(
                    f"[load_background] {ns_provenance}, for {path}.",
                    file=sys.stderr,
                )
        assert_two_zone_puff_live(
            ns,
            ns_ann,
            path=str(path),
            window_ms=window_ms,
            expected_per_s=ns_expected,
        )
        assert_end_recycle_routed_live(
            ba, ba_ann, path=str(path), window_ms=window_ms
        )
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
                # Volume convention as above. Recombination books its birth
                # into the COLUMN on a two-zone state -- physics/reactions.py
                # sets the neutral-row conversion to unity there, because nn
                # IS the column density and the column volume IS Vp -- so the
                # row integrates on Vp and the old * Vm read over-counted the
                # channel by Vm/Vp (~29x at the collector cell on l2a7b). No
                # annulus row is written for these terms today; one is
                # consumed where it exists so the read cannot go stale if that
                # ever changes.
                col = np.mean(np.clip(f[key][:][m], 0.0, None), axis=0)
                if two_zone:
                    rec += col * Vp_full
                    key_a = f"rhs_terms/{term}/nn_a"
                    if key_a in f:
                        rec += np.mean(
                            np.clip(f[key_a][:][m], 0.0, None), axis=0
                        ) * Va_full
                else:
                    rec += col * Vm_full
            elif f"rhs_terms/{term}/n" in f:
                rec += np.mean(
                    np.clip(-f[f"rhs_terms/{term}/n"][:][m], 0.0, None),
                    axis=0,
                ) * Vp_full
        cd = f["cathode_diagnostics"]
        phi_c = float(np.nanmean(cd["source_phi_c"][:][m]))
        T_s = float(np.mean(cd["T_s_surface"][:][m]))
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
    # Present ONLY when a menu entry was not read straight off the neutral
    # ledger, so a consumer that reports it cannot mislabel an ordinary
    # read, and its absence is the ordinary case rather than a default.
    provenance = {}
    if an_provenance is not None:
        provenance["anode_left"] = an_provenance
        provenance["anode_right"] = an_provenance
    if ns_provenance is not None:
        provenance["puff"] = ns_provenance
    if provenance:
        bg["source_provenance"] = provenance
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


def run_fast_reflected(bg, n_particles, rng, r_e=0.2, r_n=0.5, max_iter=20000):
    """Transport ONLY the cathode jet's fast reflected lobe on ``bg``.

    A self-contained read that shares nothing with :func:`run_mc` but the
    background, the geometry helpers and the RNG conventions. Its question is
    narrow: of the backscattered atoms the cathode launches into the machine,
    what fraction deposits INSIDE the plasma column before it leaves, where
    does it deposit, and with what decay length -- the survival factor a
    phase-1 in-column kernel needs.

    SOURCE. One population only: the ``R_N`` reflected fraction of the cathode
    face's own recycle row, launched from a point sampled uniformly on the
    cathode face disc ``r <= R_cath`` at ``z = 0``, with a cosine-law direction
    into the domain (:func:`cosine_emit` with ``sign_z=+1``; the sampled
    DIRECTION is kept and the magnitude replaced) and a single, monoenergetic
    speed. The thermal remainder of the cathode face and all five other
    sources of the full menu are absent by construction.

    LAUNCH ENERGY. The ``"total_reflected"`` convention of
    :func:`cablp.solvers._sim1d.physics.sources.cathode_jet_backscatter_speed`
    -- ``R_E`` is the TOTAL reflected energy fraction, so each of the ``R_N``
    backscattered atoms carries ``R_E/R_N`` of the incident per-particle
    energy ``phi_c + Ti``::

        E_fast = (R_E / R_N) * (phi_c + Ti_cathode)   [eV]

    ``phi_c`` is the window-mean cathode drop the background saved and
    ``Ti_cathode`` the window-mean ion temperature of the first in-domain
    (cathode-face) cell. The legacy per-particle reading of ``R_E`` that
    :func:`run_mc`'s ``--jet`` launcher uses is NOT available here; this mode
    exists to measure the ratified convention.

    CX. The fast atom's charge-exchange frequency is ENERGY-RESOLVED, taken
    from the same Phelps He+/He backscatter cross section the solver's R4.3
    ion-neutral operator uses, in the CENTRE-OF-MASS convention::

        g_eff^2 = |v - u_i|^2 + 8 k T_i / (pi mu),   mu = m_He / 2
        E_rel   = 1/2 mu g_eff^2
        nu_cx   = n_i * Qb(E_rel) * g_eff

    ``Qb`` is :func:`cablp.funcs._cross.phelps_he_backscatter_cm2`, whose
    argument is the RELATIVE collision energy; it is valid over
    ``PHELPS_QB_RANGE_EV`` = 1.0e-4 to 1.0e4 eV (the span of the archived LXCat
    table), and an ``E_rel`` reachable outside that range raises.

    CENTRE-OF-MASS CONVENTION, stated rather than buried. ``Qb`` is tabulated
    against the relative collision energy of the He+/He pair, so its argument
    is ``E_rel = 1/2 mu g^2`` with the reduced mass ``mu = m_He / 2`` -- for
    equal masses and cold, drift-free ions that is HALF the atom's own lab
    kinetic energy, which is what this mode previously passed. The same
    ``g_eff`` also sets the rate: a collision frequency is
    ``n sigma <relative speed>``, not ``n sigma <lab speed>``. ``g_eff`` is the
    standard interpolation between the drift-dominated and thermal-dominated
    limits for an equal-mass pair, so the ion thermal spread is carried rather
    than neglected. The form is transcribed symbol-for-symbol from the repo's
    existing correct consumer,
    :meth:`cablp.solvers._sim1d.physics.kinetic_dvm.TransientDVM.collision_frequencies`
    (``kinetic_dvm.py`` lines 686-689), including its ``T_i`` clamp at 1e-6 eV
    and its ``E_rel`` floor at 1e-9 eV; ``u_i`` is the local ion MEAN drift.
    This replaces :func:`run_mc`'s treatment, which keys CX off the background
    ION TEMPERATURE through :func:`~cablp.funcs._cross.charge_ex_react` and so
    transports a fast atom at a thermal collision rate.

    Electron-impact ionization keeps the background's Te-Maxwellian ADAS SCD
    rate (``bg["nu_ion"]``) unchanged: the electrons ARE the Maxwellian
    species there, so no energy resolution on the neutral is called for.

    Both events END the fast history. Ionization removes the atom outright; a
    CX event hands the fast atom's charge over, and the newborn THERMAL neutral
    it leaves behind is deliberately NOT tracked -- this mode measures the fast
    lobe and nothing else, so the thermal relay is :func:`run_mc`'s business.

    ESCAPE -- the load-bearing definition of the whole read, quoted here and,
    verbatim from ``FAST_ESCAPE_DISCLOSURE``, into the output header:

        A fast atom LEAVES the fast population at its FIRST CROSSING of the
        local plasma column radius Rp(z) -- the column surface, NOT the vessel
        wall Rm. Crossing counts in both forms: outward through the
        cylindrical surface ``r = Rp(icell)``, and axially across a z-edge
        into a cell whose Rp is smaller than the atom's current radius (the
        column's own annular step). Nothing is reflected and nothing
        re-enters: the atom is gone from the tally the moment it is outside
        the column, because the deliverable is the IN-COLUMN deposition
        fraction the phase-1 kernel's survival factor consumes, and a fast
        atom outside the column deposits nowhere the kernel can see.

    Two z-normal planes end a history as well: crossing the far end is
    ``end_loss``, and re-crossing the cathode face is ``source_return``.

    ANODE MESH, disclosed. The mesh plane is the gap/puff z-edge the loader
    resolves from the run's own geometry (``bg["mesh_edge"]``, nominally
    z = 50 cm), and it is OPAQUE with the run's own probability
    ``eta = bg["eta"]``, the same number :func:`run_mc` uses. At an atom's
    first crossing of that plane it is culled with probability ``eta`` into the
    sixth outcome bin ``mesh_intercepted``; survivors continue unchanged. No
    velocity in this mode ever changes -- every event kills the history and
    nothing is re-emitted -- so ``z`` is monotone along a history and the plane
    can be met at most once; that invariant is ASSERTED per crossing rather
    than assumed, and a second crossing raises.

    The mesh cull is applied BEFORE the annular-step branch of the escape test
    at the same z-edge: the mesh is a barrier standing in the crossing plane,
    so an atom meets it before it can be found outside the destination cell's
    column. A culled history is therefore booked ``mesh_intercepted``, never
    ``column_escape``.

    ``f_dep`` is consequently NO LONGER a one-sided upper bound in the mesh
    respect; the residual bias is two-sided. Downward: :func:`run_mc` re-emits
    an intercepted atom thermally on the incident side, and that thermal
    population -- some of which re-enters the column and deposits -- is outside
    this mode's scope, exactly as the newborn thermal neutral of a CX event is,
    so the cull discards it. Upward: ``eta`` is a plane-averaged opacity applied
    without angular resolution, and the fast lobe's cosine-launched incidence
    distribution at the mesh is not the one that average was taken over.

    Returns a dict of per-cell profiles [atoms/s], history counts, the outcome
    split, ``f_dep`` with its binomial error, and the launch-energy record.
    """
    ze = bg["z_edges"]
    ncell = ze.size - 1
    dz = np.diff(ze)
    Rp = bg["Rp"]
    n_bg, u_bg = bg["n"], bg["u"]
    nu_ion = bg["nu_ion"]
    R_cath = bg["R_cath"]
    T_s, phi_c = bg["T_s"], bg["phi_c"]

    r_e = float(r_e)
    r_n = float(r_n)
    if not (0.0 < r_e <= r_n < 1.0):
        raise ValueError(
            "--fast-reflected uses the total_reflected energy convention, "
            "which requires 0 < R_E <= R_N < 1 (each backscattered atom "
            f"carries R_E/R_N of the incident energy); got R_E={r_e}, "
            f"R_N={r_n}."
        )
    Ti_cath = float(bg["Ti"][0])
    E_fast = (r_e / r_n) * max(phi_c + Ti_cath, 0.0)
    v_fast = math.sqrt(2.0 * E_fast * EV / M_HE)
    # The CoM launch energy in the strict cold-ion, drift-free limit: for an
    # equal-mass pair E_rel = 1/2 mu g^2 = 1/4 m_He v^2 is exactly half the
    # lab energy. It is the number that pairs with E_fast in the header; the
    # REALIZED E_rel is the bracket computed just below.
    E_rel_launch = 0.25 * M_HE * v_fast**2 / EV

    # Ion thermal spread and drift, cell by cell, in the reference consumer's
    # own form (kinetic_dvm.collision_frequencies, lines 686-689): mu = m/2 for
    # the symmetric pair, so 8 k T / (pi mu) = 16 k T / (pi m).
    Ti_safe = np.maximum(np.asarray(bg["Ti"], dtype=float), 1e-6)
    th2 = 16.0 * Ti_safe * EV / (np.pi * M_HE)

    # Null-collision majorant, and it is EXACT rather than heuristic because
    # g_eff is bounded on both sides. The atom's speed is |v| = v_fast for its
    # whole life (both event channels kill the history, and nothing is
    # re-emitted), so per cell i
    #
    #     max(v_fast - |u_i|, 0)^2 + th2_i  <=  g_eff^2  <=  (v_fast + |u_i|)^2 + th2_i
    #
    # and E_rel is a monotone function of g_eff. Qb is STRICTLY DECREASING on
    # E > 0 -- d(ln Qb)/dE = -0.15/(E + 5) - 0.25/(1000 + E) < 0, the two
    # falling factors of the Phelps form beating its rising (1 + 5/E)^-0.15
    # exactly -- so Qb(E_rel) <= Qb(E_rel_lo_i). Bounding the two factors
    # separately therefore bounds their product:
    #
    #     nu_cx <= n_i * Qb(E_rel_lo_i) * g_hi_i
    #
    # Asserted per step below all the same -- a broken majorant biases a
    # null-collision estimator silently.
    u_abs = np.abs(u_bg)
    g_lo = np.sqrt(np.maximum(v_fast - u_abs, 0.0) ** 2 + th2)
    g_hi = np.sqrt((v_fast + u_abs) ** 2 + th2)
    E_rel_lo = np.maximum(0.25 * M_HE * g_lo**2 / EV, 1e-9)
    E_rel_hi = np.maximum(0.25 * M_HE * g_hi**2 / EV, 1e-9)
    lo, hi = PHELPS_QB_RANGE_EV
    if not (lo <= float(E_rel_lo.min()) and float(E_rel_hi.max()) <= hi):
        raise ValueError(
            "the fast lobe's reachable CoM collision energy E_rel spans "
            f"{float(E_rel_lo.min()):.6g}-{float(E_rel_hi.max()):.6g} eV "
            f"(launch E_fast={E_fast:.6g} eV lab), which leaves the Phelps "
            f"He+/He backscatter table's range {lo:g}-{hi:g} eV, so the CX "
            "cross section would be an extrapolation of the fit rather than "
            "the archived data. Refusing to run."
        )
    Qb_fast = float(phelps_he_backscatter_cm2(E_rel_launch))
    nu_max = float(
        (nu_ion + n_bg * phelps_he_backscatter_cm2(E_rel_lo) * g_hi).max()
    )

    # The anode mesh plane, from the run's own geometry and config -- the same
    # z-edge index and the same eta run_mc uses, never a literal here.
    mesh_edge = int(bg["mesh_edge"])
    z_mesh = float(ze[mesh_edge])
    eta = float(bg["eta"])
    transparency = 1.0 - eta

    N = int(n_particles)
    rate_fast = r_n * float(bg["sources"]["cathode_face"])
    w = rate_fast / N

    rad = R_cath * np.sqrt(rng.random(N))
    th = rng.random(N) * 2.0 * np.pi
    pos = np.zeros((N, 3))
    pos[:, 0] = rad * np.cos(th)
    pos[:, 1] = rad * np.sin(th)
    pos[:, 2] = 1e-6
    vel = cosine_emit(rng, N, T_s, 1.0)
    vel = vel * (v_fast / np.maximum(np.linalg.norm(vel, axis=1), 1.0))[:, None]

    cnt_cx = np.zeros(ncell)          # first-interaction CX census [histories]
    cnt_ion = np.zeros(ncell)         # first-interaction ionization census
    # Per-history record of whether the mesh plane has already been met, so the
    # at-most-one-crossing invariant is checked rather than trusted. Compacted
    # with pos/vel at the end of every segment.
    crossed_mesh = np.zeros(N, dtype=bool)
    outcome = {
        "cx_deposited": 0,
        "ionization_deposited": 0,
        "column_escape": 0,
        "end_loss": 0,
        "source_return": 0,
        "mesh_intercepted": 0,
    }
    for _ in range(max_iter):
        n_act = pos.shape[0]
        if n_act == 0:
            break
        speed = np.linalg.norm(vel, axis=1)
        icell = np.clip(np.searchsorted(ze, pos[:, 2]) - 1, 0, ncell - 1)
        with np.errstate(divide="ignore"):
            d_z = np.where(
                vel[:, 2] > 0,
                (ze[icell + 1] - pos[:, 2]) / vel[:, 2],
                np.where(
                    vel[:, 2] < 0, (ze[icell] - pos[:, 2]) / vel[:, 2], np.inf
                ),
            ) * speed
        # Outward root of |xy + t vxy| = Rp(icell): the column surface, which
        # here is an ABSORBER rather than a segment split.
        vxy2 = vel[:, 0] ** 2 + vel[:, 1] ** 2
        b = pos[:, 0] * vel[:, 0] + pos[:, 1] * vel[:, 1]
        r2 = pos[:, 0] ** 2 + pos[:, 1] ** 2
        Rp_here = Rp[icell]
        disc = b**2 + vxy2 * (Rp_here**2 - r2)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_rp = (-b + np.sqrt(np.maximum(disc, 0.0))) / np.where(
                vxy2 > 0, vxy2, np.inf
            )
        d_rp = np.where(vxy2 > 0, np.maximum(t_rp, 0.0) * speed, np.inf)
        d_coll = -np.log(rng.random(n_act)) * speed / nu_max
        d = np.minimum(np.minimum(d_z, d_rp), d_coll)
        dt = d / speed
        pos = pos + vel * dt[:, None]
        pos = pos + (vel / speed[:, None]) * RAY_EPS_CM
        kill = np.zeros(n_act, dtype=bool)

        hit_c = d_coll <= np.minimum(d_z, d_rp)
        if hit_c.any():
            idx = np.flatnonzero(hit_c)
            ii = icell[idx]
            # CoM convention, kinetic_dvm.collision_frequencies lines 686-689
            # symbol for symbol: the thermal-spread effective relative speed
            # sets BOTH the cross section's argument and the rate.
            w2 = (
                vel[idx, 0] ** 2
                + vel[idx, 1] ** 2
                + (vel[idx, 2] - u_bg[ii]) ** 2
            )
            g_eff = np.sqrt(w2 + th2[ii])
            E_rel = np.maximum(0.25 * M_HE * g_eff**2 / EV, 1e-9)
            nu_cx = n_bg[ii] * phelps_he_backscatter_cm2(E_rel) * g_eff
            nu_tot = nu_ion[ii] + nu_cx
            if np.any(nu_tot > nu_max):
                raise ValueError(
                    "null-collision majorant violated in the fast-lobe "
                    f"tracer: max nu_tot={float(nu_tot.max()):.6g} /s exceeds "
                    f"nu_max={nu_max:.6g} /s. The estimator would under-count "
                    "collisions; refusing to continue."
                )
            real = rng.random(idx.size) < (nu_tot / nu_max) * (
                r2[idx] < Rp[ii] ** 2
            )
            hit = idx[real]
            if hit.size:
                jj = icell[hit]
                ionz = rng.random(hit.size) < (
                    nu_ion[jj] / nu_tot[real]
                )
                np.add.at(cnt_ion, jj[ionz], 1.0)
                np.add.at(cnt_cx, jj[~ionz], 1.0)
                outcome["ionization_deposited"] += int(ionz.sum())
                outcome["cx_deposited"] += int((~ionz).sum())
                kill[hit] = True

        # Column surface: the escape definition, cylindrical branch.
        esc = (~hit_c) & (d_rp <= d_z)
        outcome["column_escape"] += int(esc.sum())
        kill |= esc

        hit_z = (~hit_c) & (~esc)
        if hit_z.any():
            idx = np.flatnonzero(hit_z)
            zdir = np.sign(vel[idx, 2])
            edge = np.where(zdir > 0, icell[idx] + 1, icell[idx])
            at_R = edge == ncell
            at_L = edge == 0
            outcome["end_loss"] += int(at_R.sum())
            outcome["source_return"] += int(at_L.sum())
            kill[idx[at_R | at_L]] = True
            interior = ~(at_R | at_L)
            # ANODE MESH: an opaque barrier standing IN the crossing plane, so
            # it is resolved before the annular-step test at the same edge --
            # a culled atom is mesh_intercepted, never column_escape.
            at_mesh = interior & (edge == mesh_edge)
            if at_mesh.any():
                m = idx[at_mesh]
                if crossed_mesh[m].any():
                    raise ValueError(
                        f"{int(crossed_mesh[m].sum())} fast histories reached "
                        f"the anode mesh plane z = {z_mesh:.4f} cm a SECOND "
                        "time. Nothing in this mode changes a velocity, so z "
                        "is monotone along a history and the plane is "
                        "reachable at most once; a second crossing means the "
                        "mesh cull was applied more than once to the same "
                        "atom and the eta bookkeeping is wrong. Refusing to "
                        "continue."
                    )
                crossed_mesh[m] = True
                blocked = rng.random(m.size) > transparency
                outcome["mesh_intercepted"] += int(blocked.sum())
                kill[m[blocked]] = True
            # Column surface, ANNULAR-STEP branch: a z-crossing into a section
            # whose column is narrower leaves the atom outside Rp without ever
            # meeting the cylindrical root above.
            e = idx[interior]
            if e.size:
                dest = np.where(
                    zdir[interior] > 0, edge[interior], edge[interior] - 1
                )
                r_step = np.sqrt(pos[e, 0] ** 2 + pos[e, 1] ** 2)
                step = (r_step > Rp[dest]) & ~kill[e]
                outcome["column_escape"] += int(step.sum())
                kill[e[step]] = True

        alive = ~kill
        pos, vel = pos[alive], vel[alive]
        crossed_mesh = crossed_mesh[alive]
    else:
        if pos.shape[0]:
            raise ValueError(
                f"{pos.shape[0]} of {N} fast histories were still in flight "
                f"after max_iter={max_iter} segments. Every outcome of this "
                "mode is a registered channel, so an exhausted history has "
                "nowhere honest to be booked; refusing to report a split with "
                "a silent extra bin."
            )

    n_dep = outcome["cx_deposited"] + outcome["ionization_deposited"]
    f_dep = n_dep / N
    f_dep_err = math.sqrt(max(f_dep * (1.0 - f_dep), 0.0) / N)
    n_first = n_dep
    cx_share = outcome["cx_deposited"] / n_first if n_first else float("nan")
    cx_share_err = (
        math.sqrt(max(cx_share * (1.0 - cx_share), 0.0) / n_first)
        if n_first
        else float("nan")
    )

    dep_cnt = cnt_cx + cnt_ion
    dep_per_cm = dep_cnt / N / dz
    # e-fold of the deposition profile. A log-linear fit is only defined on the
    # profile's FALLING side, so the window opens at the peak -- the end of the
    # rise -- and closes at the last contiguous cell still above peak/e^3 with a
    # nonzero count. The rule and the resulting window are both printed.
    fit = {"ok": False, "e_fold_cm": float("nan"), "z_lo": float("nan"),
           "z_hi": float("nan"), "ncell": 0}
    zc = 0.5 * (ze[:-1] + ze[1:])
    if dep_per_cm.max() > 0.0:
        ipk = int(np.argmax(dep_per_cm))
        floor = dep_per_cm[ipk] * math.exp(-3.0)
        j = ipk
        while j + 1 < ncell and dep_per_cm[j + 1] >= floor and dep_per_cm[j + 1] > 0.0:
            j += 1
        if j - ipk >= 2:
            slope, intercept = np.polyfit(
                zc[ipk : j + 1], np.log(dep_per_cm[ipk : j + 1]), 1
            )
            fit = {
                "ok": bool(slope < 0.0),
                "e_fold_cm": float(-1.0 / slope) if slope < 0.0 else float("inf"),
                "z_lo": float(zc[ipk]),
                "z_hi": float(zc[j]),
                "ncell": int(j - ipk + 1),
            }

    return {
        "z": zc,
        "dz": dz,
        "S_cx_fast": cnt_cx * w,
        "S_ion_fast": cnt_ion * w,
        "cnt_cx": cnt_cx,
        "cnt_ion": cnt_ion,
        "dep_per_cm": dep_per_cm,
        "outcome": outcome,
        "n_launched": N,
        "f_dep": f_dep,
        "f_dep_err": f_dep_err,
        "cx_share": cx_share,
        "cx_share_err": cx_share_err,
        "fit": fit,
        "E_fast_eV": E_fast,
        "E_rel_eV": E_rel_launch,
        "E_rel_lo_eV": float(E_rel_lo.min()),
        "E_rel_hi_eV": float(E_rel_hi.max()),
        "v_fast_cm_s": v_fast,
        "Qb_fast_cm2": Qb_fast,
        "z_mesh_cm": z_mesh,
        "eta": eta,
        "phi_c_V": phi_c,
        "Ti_cathode_eV": Ti_cath,
        "T_s_K": T_s,
        "R_cath_cm": R_cath,
        "r_e": r_e,
        "r_n": r_n,
        "rate_fast_per_s": rate_fast,
        "nu_max_per_s": nu_max,
    }


def report_fast_reflected(res, bg, args):
    """Print the ``--fast-reflected`` read in the house ``*_read.txt`` style."""
    bar = "=" * 78
    print(bar)
    print("P0.3 FAST REFLECTED LOBE -- in-column deposition read")
    print(bar)
    print(f"  background     {args.run}")
    print(f"  window [ms]    {args.window[0]}-{args.window[1]}")
    print(f"  seed           {args.seed}")
    print(f"  launched       {res['n_launched']} histories")
    print()
    print("SOURCE")
    print("  convention     total_reflected  "
          "(cathode_jet_energy_convention; sources.py "
          "cathode_jet_backscatter_speed)")
    print(f"  R_E / R_N      {res['r_e']:.6g} / {res['r_n']:.6g}"
          f"   ->  R_E/R_N = {res['r_e'] / res['r_n']:.6g}")
    print(f"  phi_c [V]      {res['phi_c_V']:.6f}"
          "   (window mean, cathode_diagnostics/source_phi_c)")
    print(f"  Ti_cath [eV]   {res['Ti_cathode_eV']:.6f}"
          "   (window mean, first in-domain cell)")
    print(f"  E_fast [eV]    {res['E_fast_eV']:.6f}"
          "   = (R_E/R_N) * (phi_c + Ti_cath)   [LAB, per atom]")
    print(f"  E_rel [eV]     {res['E_rel_eV']:.6f}"
          "   = 1/2 mu v_fast^2, mu = M_He/2   [CoM, cold drift-free ions]")
    print(f"  v_fast [cm/s]  {res['v_fast_cm_s']:.6e}")
    print(f"  angular        cosine into the domain (cosine_emit, sign_z=+1, "
          f"T_s={res['T_s_K']:.1f} K); direction only, speed set to v_fast")
    print(f"  launch disc    uniform on r <= R_cath = {res['R_cath_cm']:.4f} cm "
          "at z = 0")
    print(f"  source rate    R_N * cathode_face = {res['rate_fast_per_s']:.6e} "
          "atoms/s")
    print("                 (rate scales the tallies only; every deliverable "
          "below is a FRACTION)")
    print()
    print("COLLISION PHYSICS (fast population)")
    print("  CX             nu_cx = n_i * Qb(E_rel) * g_eff, Phelps He+/He "
          "backscatter")
    print("                 g_eff^2 = |v - u_i|^2 + 8 k T_i / (pi mu),   "
          "mu = M_He/2")
    print("                 E_rel   = 1/2 mu g_eff^2")
    print(f"                 Qb validity range {PHELPS_QB_RANGE_EV[0]:g} - "
          f"{PHELPS_QB_RANGE_EV[1]:g} eV (archived LXCat table); the "
          "reachable E_rel")
    print(f"                 spans {res['E_rel_lo_eV']:.6g} - "
          f"{res['E_rel_hi_eV']:.6g} eV over the background and is inside it")
    print(f"                 Qb(E_rel) = {res['Qb_fast_cm2']:.6e} cm^2 "
          "at the drift-free cold-ion launch E_rel")
    print("  CoM convention Qb is tabulated against the RELATIVE collision "
          "energy of the")
    print("                 He+/He pair, so its argument is E_rel = 1/2 mu "
          "g^2 with the")
    print("                 reduced mass mu = M_He/2 -- half the atom's own "
          "lab energy for")
    print("                 equal masses and cold, drift-free ions. The same "
          "g_eff sets the")
    print("                 RATE: a collision frequency is n sigma <relative "
          "speed>, not")
    print("                 n sigma <lab speed>. g_eff carries the ion "
          "thermal spread (the")
    print("                 standard equal-mass interpolation) rather than "
          "neglecting it, and")
    print("                 u_i is the local ion MEAN drift. Transcribed "
          "symbol-for-symbol")
    print("                 from kinetic_dvm.py:686-689 "
          "(TransientDVM.collision_frequencies),")
    print("                 the repo's existing correct consumer, including "
          "its T_i clamp at")
    print("                 1e-6 eV and its E_rel floor at 1e-9 eV.")
    print("  ionization     nu_ion = n_i * SCD(n,Te), the background's "
          "Te-Maxwellian ADAS rate")
    print("                 (unchanged: the ELECTRONS are the Maxwellian "
          "species)")
    print("  A CX event ENDS the fast history. The newborn thermal neutral is "
          "NOT tracked;")
    print("  this mode measures the fast lobe only.")
    print(f"  null-collision majorant nu_max = {res['nu_max_per_s']:.6e} /s")
    print()
    print("ESCAPE DEFINITION (disclosed)")
    for line in FAST_ESCAPE_DISCLOSURE.split("\n"):
        print(f"  {line}")
    print()
    print("ANODE MESH BIN (disclosed)")
    print(f"  The mesh plane is the run's own gap/puff z-edge, z = "
          f"{res['z_mesh_cm']:.4f} cm, and it is")
    print(f"  OPAQUE with the run's own eta = {res['eta']:.6g} (the same "
          "number run_mc uses; both")
    print("  are read from the background, never hardcoded here). At an atom's "
          "FIRST crossing")
    print("  of that plane it is culled with probability eta into the sixth "
          "outcome bin")
    print("  mesh_intercepted; survivors continue unchanged. No velocity in "
          "this mode ever")
    print("  changes, so z is monotone along a history and the plane is "
          "reachable at most")
    print("  once -- asserted per crossing, not assumed. The cull is resolved "
          "BEFORE the")
    print("  annular-step branch of the escape test at the same edge, so a "
          "culled atom is")
    print("  booked mesh_intercepted and never column_escape.")
    print("  f_dep is therefore NO LONGER a one-sided upper bound in the mesh "
          "respect; the")
    print("  residual bias is TWO-SIDED. Downward: run_mc re-emits an "
          "intercepted atom")
    print("  thermally on the incident side, and that thermal population -- "
          "some of which")
    print("  re-enters the column and deposits -- is outside this mode's "
          "scope, exactly as a")
    print("  CX event's newborn neutral is, so the cull discards it. Upward: "
          "eta is a")
    print("  plane-averaged opacity applied without angular resolution, and "
          "the fast lobe's")
    print("  incidence distribution at the mesh is not the one that average "
          "was taken over.")
    print()

    out = res["outcome"]
    N = res["n_launched"]
    print(bar)
    print("OUTCOME SPLIT (fast population, per launched history)")
    print(bar)
    print(f"  {'channel':<22} {'histories':>10} {'fraction':>10}")
    for key in ("cx_deposited", "ionization_deposited", "column_escape",
                "end_loss", "source_return", "mesh_intercepted"):
        print(f"  {key:<22} {out[key]:>10d} {out[key] / N:>10.5f}")
    print(f"  {'TOTAL':<22} {sum(out.values()):>10d} "
          f"{sum(out.values()) / N:>10.5f}")
    print()
    print(f"  f_dep = (CX + ionization) / launched = {res['f_dep']:.5f} "
          f"+/- {res['f_dep_err']:.5f}  (binomial)")
    print(f"  N = {N} gives a binomial error of {res['f_dep_err'] * 100:.3f} % "
          "absolute on f_dep")
    print()
    print("FIRST-INTERACTION CENSUS")
    print("  Every event channel of this mode kills the history, so an "
          "interaction is")
    print("  necessarily a history's FIRST -- the census below is the "
          "first-interaction one.")
    print(f"  CX               {out['cx_deposited']:>10d}")
    print(f"  ionization       {out['ionization_deposited']:>10d}")
    print(f"  CX share         {res['cx_share']:.5f} "
          f"+/- {res['cx_share_err']:.5f}  (binomial)")
    print()

    fit = res["fit"]
    print("DEPOSITION e-FOLD")
    print("  Log-linear fit of ln(deposition per cm) vs z. Window rule: opens "
          "at the")
    print("  profile PEAK (the end of the rise, where a log-linear decay is "
          "defined) and")
    print("  closes at the last contiguous cell still above peak/e^3 with a "
          "nonzero count.")
    if fit["ok"]:
        print(f"  fit window     z = {fit['z_lo']:.1f} - {fit['z_hi']:.1f} cm "
              f"({fit['ncell']} cells)")
        print(f"  e-fold         {fit['e_fold_cm']:.2f} cm "
              f"= {fit['e_fold_cm'] / 100.0:.4f} m")
    else:
        print("  fit window     NOT ESTABLISHED (profile does not fall "
              "log-linearly over a")
        print("                 window of at least 3 cells)")
    print()

    print(bar)
    print("DEPOSITION PROFILE (fast population)")
    print(bar)
    print(f"  {'z[cm]':>8} {'CX[/s]':>12} {'ion[/s]':>12} "
          f"{'dep/launch/cm':>14} {'cum f_dep':>10}")
    dep_cnt = res["cnt_cx"] + res["cnt_ion"]
    cum = np.cumsum(dep_cnt) / N
    zc = res["z"]
    # The lobe deposits over a small leading fraction of the machine, so a
    # fixed stride over all 279 cells prints the empty tail and hides the
    # profile. Walk the cells that carry deposition instead, and account for
    # the remainder in one line. The npz carries every cell regardless.
    live = np.flatnonzero(dep_cnt > 0.0)
    hi = min(int(live[-1]) + 1, zc.size - 1) if live.size else 0
    step = max(1, (hi + 1) // 30)
    for i in range(0, hi + 1, step):
        print(f"  {zc[i]:8.1f} {res['S_cx_fast'][i]:12.4e} "
              f"{res['S_ion_fast'][i]:12.4e} {res['dep_per_cm'][i]:14.6e} "
              f"{cum[i]:10.5f}")
    print(f"  cells beyond z = {zc[hi]:.1f} cm carry "
          f"{res['f_dep'] - cum[hi]:.6f} of the launched population "
          f"(profile printed with stride {step})")
    print()

    prefix = args.out or (Path(args.run).stem + "_fastlobe")
    np.savez(
        Path(args.run).parent / f"{prefix}.npz",
        z=res["z"], dz=res["dz"],
        S_cx_fast=res["S_cx_fast"], S_ion_fast=res["S_ion_fast"],
        cnt_cx=res["cnt_cx"], cnt_ion=res["cnt_ion"],
        dep_per_cm=res["dep_per_cm"],
        Rp=bg["Rp"], n_bg=bg["n"], Te_bg=bg["Te"], Ti_bg=bg["Ti"],
        outcome_keys=np.array(list(res["outcome"].keys())),
        outcome_counts=np.array(list(res["outcome"].values())),
        scalars_keys=np.array([
            "n_launched", "f_dep", "f_dep_err", "cx_share", "cx_share_err",
            "E_fast_eV", "v_fast_cm_s", "Qb_fast_cm2", "phi_c_V",
            "Ti_cathode_eV", "R_cath_cm", "r_e", "r_n", "rate_fast_per_s",
            "nu_max_per_s", "e_fold_cm", "fit_z_lo", "fit_z_hi",
            "E_rel_eV", "E_rel_lo_eV", "E_rel_hi_eV", "z_mesh_cm", "eta",
        ]),
        scalars=np.array([
            res["n_launched"], res["f_dep"], res["f_dep_err"],
            res["cx_share"], res["cx_share_err"], res["E_fast_eV"],
            res["v_fast_cm_s"], res["Qb_fast_cm2"], res["phi_c_V"],
            res["Ti_cathode_eV"], res["R_cath_cm"], res["r_e"], res["r_n"],
            res["rate_fast_per_s"], res["nu_max_per_s"],
            res["fit"]["e_fold_cm"], res["fit"]["z_lo"], res["fit"]["z_hi"],
            res["E_rel_eV"], res["E_rel_lo_eV"], res["E_rel_hi_eV"],
            res["z_mesh_cm"], res["eta"],
        ], dtype=float),
    )
    print(f"saved {prefix}.npz")


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
    ap.add_argument(
        "--fast-reflected", action="store_true",
        help="run the SELF-CONTAINED cathode fast-reflected-lobe deposition "
             "read instead of the full source menu: launches only the R_N "
             "backscattered fraction of the cathode face, monoenergetic at "
             "the total_reflected energy (R_E/R_N)*(phi_c+Ti), with an "
             "energy-resolved Phelps-Qb CX rate, and reports the in-column "
             "deposition profile, outcome split, f_dep and e-fold",
    )
    ap.add_argument(
        "--r-e", type=float, default=0.2,
        help="cathode energy reflection coefficient R_E, read in the "
             "total_reflected convention (--fast-reflected only; default "
             "0.2, the stance box)",
    )
    ap.add_argument(
        "--r-n", type=float, default=0.5,
        help="cathode particle reflection coefficient R_N "
             "(--fast-reflected only; default 0.5, the stance box)",
    )
    args = ap.parse_args(argv)

    bg = load_background(args.run, tuple(args.window))
    if args.fast_reflected:
        # A separate mode end to end: its own launcher, its own collision
        # physics, its own tallies and its own report. It shares no state with
        # the source-menu path below, which is left bit-for-bit as it was.
        res = run_fast_reflected(
            bg,
            args.n_particles,
            np.random.default_rng(args.seed),
            r_e=args.r_e,
            r_n=args.r_n,
        )
        report_fast_reflected(res, bg, args)
        return
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
