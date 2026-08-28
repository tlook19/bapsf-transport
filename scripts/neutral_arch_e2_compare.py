"""E2: matched-time PHYSICS-REFERENCE comparison for the transient DVM arm.

Drives the MERGED in-solver kinetic engine (``physics/kinetic_dvm.TransientDVM``
-- the real operator, not a transcription) and a transient full-particle Monte
Carlo reference over the SAME frozen plasma background, the SAME source ledger,
the SAME geometry and the SAME accommodation, and compares every quantity on the
E2 list at IDENTICAL elapsed times, through one transient (sources on into an
empty box) and one relaxation (sources off) phase.

**This is a measurement instrument.** It fits nothing, tunes nothing and
recommends nothing. Every comparison row carries the reference it was measured
against and the reference's own Monte-Carlo statistical error, so a
deterministic-vs-MC difference can be judged against the noise it sits in.

Reference choice (the K2a constraint)
-------------------------------------
The shipped ``KN2Zone`` two-zone steady march is NON-CONSERVATIVE in the
ANNULUS channel at the area-jump faces of the expanded geometry (a
collisionless annulus stream exits the end expansion at 4.2967x the injected
flux; the column channel and the single-zone variants are unaffected because
their area is uniform). Its annulus channel is therefore NOT used as truth
anywhere in this script. The reference is:

  PRIMARY   ``TransientMC`` below -- a transient, time-resolved, full-particle
            Monte Carlo on the real ``(r, z)`` geometry. Conservation-clean by
            construction: particles are neither created nor destroyed except at
            the named ionization and pump channels, and the vessel wall is
            resolved as an actual surface rather than as a rate.
  SECONDARY time-dependent single-zone KN2ZoneJump variants -- NOT exercised
            here, because every E2 row except ``nn_col`` is a two-zone quantity
            (``nn_ann``, annulus momentum/energy, radial exchange, radial-wall
            deposition) that a uniform-area single-zone variant cannot produce.
            Stated rather than silently skipped.

What is matched, and what is deliberately not
---------------------------------------------
MATCHED (inputs): geometry ``Rp(z), Rm(z), dz``; the plateau-averaged plasma
fields ``n_i, T_i, u_i``; the ADAS ionization frequency; the external source
ledger and every source's registered velocity convention (puff = 300 K
zero-momentum volume Maxwellian in the annulus; recombination = local ion
Maxwellian in the column; anode = the engine's wall-emission spectrum in the
column; faces = cosine half-flux spectra); the accommodation coefficient; the
end-pump sticking coefficients; the anode-mesh transparency and its re-emission
spectrum; the wall and cathode-surface temperatures.

NOT MATCHED, ON PURPOSE (this is what E2 measures): the two instruments'
TRANSPORT and their SURFACE/EXCHANGE operators. The DVM carries Cauchy-chord
zone-exchange and wall rates on a ``(v_z, v_perp)`` grid with an implicit upwind
march and throat-face areas; the MC ray-traces the actual cylinder. Their
collision channels differ too, and the ``--cx-model`` switch separates the two
contributions:

  ``--cx-model kinetic`` (the reference): true two-body event kinematics --
      a partner ion is sampled from the local Maxwellian, the event rate is
      ``n_i sigma(E_rel) g`` at the true relative speed, resonant charge
      exchange swaps identities and isotropic elastic scattering is resolved
      in the centre of mass.
  ``--cx-model bgk``: an exact particle realization of the DVM's OWN collision
      operator (the ``g_eff`` interpolation, full-replacement rebirth at the
      local ion Maxwellian, the 1/2 factor on the isotropic-elastic rate).
      Running both isolates how much of a DVM-vs-reference deviation is the
      collision model and how much is the transport.

One channel is shared by construction and so is NOT tested here: the
anode-mesh interception (and the anode source) re-emit at the engine's
cylindrical wall-emission spectrum on what is geometrically a z-normal plane.
The MC adopts the same convention so the comparison stays about transport;
the exclusion is stated in the summary rather than hidden.

One channel exists only in the reference: the annular STEP face at the end
expansion (``Rm`` 50 -> 100 cm). The DVM's free-molecular throat-face flux form
has no such surface -- it throttles the aperture instead of terminating rays on
the step -- so its step deposition is identically zero and the MC's is reported
as a reference-only measurement.

Artifacts (written to ``--out-dir``)
------------------------------------
``neutral_arch_e2_compare_nx240.txt``  the matched-time comparison tables
``neutral_arch_e2_cx_channel.txt``     the CX/elastic-channel adjudication
``neutral_arch_e2_summary.md``         factual statements only

Usage (single command, reruns end to end):

    PYTHONPATH=<checkout>/cablp python scripts/neutral_arch_e2_compare.py \
        --run scripts/es1_kn2z_promoted_nx240.h5 --out-dir scripts
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mc_neutrals import (  # noqa: E402
    EV,
    KB,
    M_HE,
    T_WALL_K,
    load_background,
)

from cablp.atomic.cross_sections import (  # noqa: E402
    phelps_cx_rate_cm3_s,
    phelps_he_backscatter_cm2,
    phelps_he_isotropic_cm2,
    phelps_iso_rate_cm3_s,
    phelps_momentum_transfer_rate_cm3_s,
)
from cablp.solvers._sim1d.physics.kinetic_dvm import (  # noqa: E402
    ELASTIC_BGK_MOMENTUM_FACTOR,
    TransientDVM,
)
from cablp.solvers._sim1d.physics.kinetic_neutrals import (  # noqa: E402
    ion_thermal_g_eff_floor_cm2_s2,
)

# Return-spectrum energy bin edges [eV]. Spans the 300 K wall population
# (mean cosine-flux energy 2 kT = 0.0517 eV) through the CX-relayed tail at the
# background's peak ion temperature (~9.6 eV).
E_BIN_EDGES_EV = np.array(
    [0.0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, np.inf]
)
N_EBIN = E_BIN_EDGES_EV.size - 1

# Axial regions the aggregate rows are reported over, as (label, z_lo, z_hi) in
# cm. "near-anode" is the mesh/puff neighbourhood, which is the only region the
# shared mesh-spectrum convention touches; "end-expansion" is the only region
# with an annular step face.
REGIONS = (
    ("near-anode  z<100", 0.0, 100.0),
    ("mid-machine 500-1000", 500.0, 1000.0),
    ("far column 1000-1800", 1000.0, 1800.0),
    ("end-expansion z>1800", 1800.0, 1e9),
)

ZONES = ("col", "ann")


# ------------------------------------------------------------------ shared


def build_shared(bg, args):
    """Assemble the inputs BOTH instruments are driven with.

    Everything here is an input, and both arms read these same objects: the
    geometry, the frozen plasma fields, the source ledger in the engine's
    registered form, and the surface parameters. Nothing in this function is a
    modelling choice of either instrument.
    """
    ze = np.asarray(bg["z_edges"], dtype=float)
    dz = np.diff(ze)
    nz = dz.size
    Rp = np.asarray(bg["Rp"], dtype=float)
    Rm = np.asarray(bg["Rm"], dtype=float)
    V_col = np.pi * Rp**2 * dz
    V_mach = np.pi * Rm**2 * dz
    geometry = SimpleNamespace(
        cells=nz,
        length_cm=dz,
        Rp_cm=Rp,
        Rm_cm=Rm,
        plasma_volume_cm3=V_col,
        neutral_volume_cm3=V_mach,
    )

    # End-pump sticking, the shipped TPMC/KN2Zone convention: the pumping speed
    # over the one-way thermal flux through THAT END'S OWN end plane. Both ends
    # previously used A_end = pi Rm[-1]^2, which the shipped instruments also
    # did; they now take their own, matching the solver's _dvm_end_sticking
    # (45e7f3b).
    vbar = np.sqrt(8.0 * KB * T_WALL_K / (np.pi * M_HE))
    A_end_L = np.pi * Rm[0] ** 2
    A_end_R = np.pi * Rm[-1] ** 2
    s_L = float(bg["S_pump_L"]) * 1e3 / (A_end_L * vbar / 4.0)
    s_R = float(bg["S_pump_R"]) * 1e3 / (A_end_R * vbar / 4.0)

    # Source ledger -> the engine's per-cell arrays. Rates in atoms/s.
    src = bg["sources"]
    puff = np.zeros(nz)
    if src.get("puff", 0.0) > 0.0:
        iz = int(np.clip(np.searchsorted(ze, src["puff_z"]) - 1, 0, nz - 1))
        puff[iz] = float(src["puff"])
    rec_cell = np.asarray(bg["rec_cell"], dtype=float)
    rec = np.zeros(nz)
    if rec_cell.sum() > 0.0 and src.get("vol_rec", 0.0) > 0.0:
        rec = rec_cell * (float(src["vol_rec"]) / rec_cell.sum())
    anode = np.zeros(nz)
    mesh_face = int(bg["mesh_edge"])
    for name, off in (("anode_left", -1), ("anode_right", 0)):
        rate = float(src.get(name, 0.0))
        if rate <= 0.0:
            continue
        anode[int(np.clip(mesh_face + off, 0, nz - 1))] += rate
    sources = {
        "puff": puff,
        "recombination": rec,
        "anode": anode,
        "cathode_face": float(src.get("cathode_face", 0.0)),
        "collector_face": float(src.get("collector_face", 0.0)),
    }
    total_rate = (
        puff.sum() + rec.sum() + anode.sum()
        + sources["cathode_face"] + sources["collector_face"]
    )
    if total_rate <= 0.0:
        raise ValueError("the background's source ledger is empty")

    # Common initial condition: the background's OWN saved two-zone neutral
    # profile, laid down as a 300 K Maxwellian at rest in both instruments.
    # An empty vessel is not usable here -- free-molecular filling of a 20 m
    # duct takes far longer than the ~2 ms inventory turnover this schedule
    # resolves, so an empty start would leave every row beyond the source
    # region statistically empty and would compare two instruments' filling
    # fronts rather than their operators on a developed state.
    seed_col = np.asarray(bg.get("nncol_model", bg["nn_model"]), dtype=float)
    seed_ann = np.asarray(
        bg.get("nna_model", np.zeros(nz)), dtype=float
    )
    seed_col = np.maximum(seed_col, 0.0)
    seed_ann = np.where(V_mach - V_col > 0.0, np.maximum(seed_ann, 0.0), 0.0)

    Ti = np.asarray(bg["Ti"], dtype=float)
    u_i = np.asarray(bg["u"], dtype=float)
    plasma = {
        "n_i": np.asarray(bg["n"], dtype=float),
        "Ti_eV": Ti,
        "u_i": u_i,
        "nu_ion": np.asarray(bg["nu_ion"], dtype=float),
    }

    # Velocity-grid extent: cover the hottest ion Maxwellian and the fastest
    # drift in the background, so no rebirth is projected off the grid.
    Ti_cap = float(np.max(Ti)) * 1.05
    u_cap = float(np.max(np.abs(u_i))) * 1.05

    nb = int(round(args.t_end_ms / args.bin_ms))
    if abs(nb * args.bin_ms - args.t_end_ms) > 1e-9:
        raise ValueError("--t-end-ms must be an integer multiple of --bin-ms")
    if abs(round(args.t_switch_ms / args.bin_ms) * args.bin_ms
           - args.t_switch_ms) > 1e-9:
        raise ValueError("--t-switch-ms must be an integer multiple of --bin-ms")

    return {
        "geometry": geometry,
        "z_edges": ze,
        "z_cm": 0.5 * (ze[:-1] + ze[1:]),
        "dz": dz,
        "Rp": Rp,
        "Rm": Rm,
        "nz": nz,
        "V_col": V_col,
        "V_ann": np.maximum(V_mach - V_col, 0.0),
        "plasma": plasma,
        "sources": sources,
        "seed_col": seed_col if args.seed_state else np.zeros(nz),
        "seed_ann": seed_ann if args.seed_state else np.zeros(nz),
        "total_rate": float(total_rate),
        "s_L": s_L,
        "s_R": s_R,
        "transparency": 1.0 - float(bg["eta"]),
        "mesh_face": mesh_face,
        "T_s_K": float(bg["T_s"]),
        "T_wall_K": T_WALL_K,
        "R_cath": float(bg["R_cath"]),
        "Ti_cap_eV": Ti_cap,
        "u_cap_cm_s": u_cap,
        "t_end": args.t_end_ms * 1e-3,
        "t_switch": args.t_switch_ms * 1e-3,
        "bin_s": args.bin_ms * 1e-3,
        "nbin": nb,
    }


def blank_diag(nz, nbin):
    """Allocate the shared diagnostic layout both arms fill."""
    z2 = (nbin, nz)
    d = {}
    for zone in ZONES:
        d[f"n_{zone}"] = np.zeros(z2)      # cm^-3, bin-averaged
        d[f"p_{zone}"] = np.zeros(z2)      # g cm/s per cm^3, bin-averaged
        d[f"e_{zone}"] = np.zeros(z2)      # erg per cm^3, bin-averaged
    d["exch_ca"] = np.zeros(z2)            # atoms/s column -> annulus
    d["exch_ac"] = np.zeros(z2)            # atoms/s annulus -> column
    d["wrad_inc"] = np.zeros(z2)           # erg/s incident on the radial wall
    d["wrad_ret"] = np.zeros(z2)           # erg/s carried back off it
    d["wstep_inc"] = np.zeros(z2)          # erg/s incident on the annular step
    d["wstep_ret"] = np.zeros(z2)
    d["wend_inc"] = np.zeros((nbin, 2))    # erg/s, [L, R]
    d["wend_ret"] = np.zeros((nbin, 2))
    # Return spectra, per phase (0 transient, 1 relaxation): mass [atoms/s
    # bin-averaged] per energy bin, plus the mass and energy totals the mean
    # energy is formed from.
    d["spec_rad_acc"] = np.zeros((2, N_EBIN))
    d["spec_rad_ref"] = np.zeros((2, N_EBIN))
    d["spec_end_acc"] = np.zeros((2, 2, N_EBIN))
    d["spec_end_ref"] = np.zeros((2, 2, N_EBIN))
    d["specE_rad_acc"] = np.zeros(2)       # erg/s carried by that return
    d["specE_rad_ref"] = np.zeros(2)
    d["specE_end_acc"] = np.zeros((2, 2))
    d["specE_end_ref"] = np.zeros((2, 2))
    # second energy moment, so an RMS return energy can be quoted -- a
    # spectrum statistic that does not depend on the histogram binning
    d["specE2_rad_acc"] = np.zeros(2)
    d["specE2_rad_ref"] = np.zeros(2)
    d["specE2_end_acc"] = np.zeros((2, 2))
    d["specE2_end_ref"] = np.zeros((2, 2))
    return d


def ebin_index(E_eV):
    """Return the return-spectrum energy-bin index of an energy in eV."""
    return np.clip(
        np.searchsorted(E_BIN_EDGES_EV, np.asarray(E_eV, dtype=float),
                        side="right") - 1,
        0,
        N_EBIN - 1,
    )


# ------------------------------------------------------------------ DVM arm


def run_dvm(shared, dt, nvz, nvp, accommodation, elastic_model, progress=None):
    """Advance the MERGED ``TransientDVM`` over the schedule and diagnose it.

    The engine is used exactly as shipped. The only instrumentation is a
    non-invasive wrapper on ``_march`` that keeps the values that method
    already returns -- the marched (substep-A) distributions, the mesh
    interception and the end outflows -- so the wall, end and exchange rows
    are the ones the update ACTUALLY took rather than a reconstruction.
    """
    nz = shared["nz"]
    nbin = shared["nbin"]
    bin_s = shared["bin_s"]
    t_end = shared["t_end"]
    t_switch = shared["t_switch"]
    if abs(round(bin_s / dt) * dt - bin_s) > 1e-15:
        raise ValueError("the DVM dt must divide the report bin width exactly")

    dvm = TransientDVM(
        geometry=shared["geometry"],
        nvz=nvz,
        nvp=nvp,
        accommodation=accommodation,
        elastic_model=elastic_model,
        transparency=shared["transparency"],
        mesh_face=shared["mesh_face"],
        s_L=shared["s_L"],
        s_R=shared["s_R"],
        T_wall_K=shared["T_wall_K"],
        Ti_cap_eV=shared["Ti_cap_eV"],
        u_cap_cm_s=shared["u_cap_cm_s"],
    )
    dvm.seed_from_density(shared["seed_col"], shared["seed_ann"])
    g = dvm.g
    captured = {}
    real_march = dvm._march

    def march(*a, **kw):
        out = real_march(*a, **kw)
        captured["res"] = out
        return out

    dvm._march = march

    V2 = g.V2[None, :, :]
    VZ = g.VZ[None, :, :]
    E_bin_eV = 0.5 * M_HE * g.V2 / EV               # (nvz, nvp)
    ei = ebin_index(E_bin_eV)                        # (nvz, nvp)
    spec_L = g.half_flux_spectrum(shared["T_s_K"], +1)
    spec_R = g.half_flux_spectrum(shared["T_wall_K"], -1)
    e_wall_per = 0.5 * M_HE * float((dvm.M_wall * g.V2).sum())
    e_L_per = 0.5 * M_HE * float((spec_L * g.V2).sum())
    e_R_per = 0.5 * M_HE * float((spec_R * g.V2).sum())
    E2_bin = (0.5 * M_HE * g.V2) ** 2
    e2_wall_per = float((dvm.M_wall * E2_bin).sum())
    e2_L_per = float((spec_L * E2_bin).sum())
    e2_R_per = float((spec_R * E2_bin).sum())
    hist_wall = np.bincount(ei.ravel(), weights=dvm.M_wall.ravel(),
                            minlength=N_EBIN)
    hist_L = np.bincount(ei.ravel(), weights=spec_L.ravel(), minlength=N_EBIN)
    hist_R = np.bincount(ei.ravel(), weights=spec_R.ravel(), minlength=N_EBIN)

    diag = blank_diag(nz, nbin)
    alpha = float(accommodation)
    s_L, s_R = shared["s_L"], shared["s_R"]

    def state_moments():
        n_c = dvm.f_c.sum(axis=(1, 2))
        n_a = dvm.f_a.sum(axis=(1, 2))
        p_c = M_HE * (dvm.f_c * VZ).sum(axis=(1, 2))
        p_a = M_HE * (dvm.f_a * VZ).sum(axis=(1, 2))
        e_c = 0.5 * M_HE * (dvm.f_c * V2).sum(axis=(1, 2))
        e_a = 0.5 * M_HE * (dvm.f_a * V2).sum(axis=(1, 2))
        return n_c, n_a, p_c, p_a, e_c, e_a

    prev = state_moments()
    nsteps = int(round(t_end / dt))
    t0_wall = time.perf_counter()
    ledger = {
        "launched": dvm.total_inventory(),
        "ionized": 0.0,
        "pumped": 0.0,
        "external": 0.0,
    }
    for step in range(nsteps):
        t0 = step * dt
        k = min(int(t0 / bin_s), nbin - 1)
        phase = 0 if t0 < t_switch - 1e-15 else 1
        src = shared["sources"] if t0 < t_switch - 1e-15 else None
        led = dvm.update(
            dt,
            sources=src,
            T_s_K=shared["T_s_K"],
            **shared["plasma"],
        )
        ledger["ionized"] += led["loss_ionization"]
        ledger["pumped"] += led["loss_pump_L"] + led["loss_pump_R"]
        ledger["external"] += (
            led["birth_puff"] + led["birth_recombination"]
            + led["birth_anode"] + led["birth_cathode_face"]
            + led["birth_collector_face"]
        )
        # ``_march`` also returns the intercepted mesh ENERGIES (the
        # B0a energy ledger's one in-sweep tally); nothing here reads
        # them.
        f_c_m, f_a_m, mesh_c, mesh_a, out, _mesh_E, _closed = captured["res"]

        # --- state moments, trapezoid over the step (second order in dt, so
        # the bin average is not biased by the update cadence)
        cur = state_moments()
        for key, a, b in (
            ("n_col", prev[0], cur[0]),
            ("n_ann", prev[1], cur[1]),
            ("p_col", prev[2], cur[2]),
            ("p_ann", prev[3], cur[3]),
            ("e_col", prev[4], cur[4]),
            ("e_ann", prev[5], cur[5]),
        ):
            diag[key][k] += 0.5 * (a + b) * dt
        prev = cur

        # --- radial exchange, from the marched state the coupling used
        diag["exch_ca"][k] += (
            (dvm.nux[:, None, :] * f_c_m).sum(axis=(1, 2)) * dvm.V_col * dt
        )
        diag["exch_ac"][k] += (
            (dvm.nuxp[:, None, :] * f_a_m).sum(axis=(1, 2)) * dvm.V_ann * dt
        )

        # --- cylindrical wall: incident and returned energy
        L_wall = dvm.nuw[:, None, :] * f_a_m * dt * dvm.V_ann[:, None, None]
        N_wall = L_wall.sum(axis=(1, 2))
        E_inc = 0.5 * M_HE * (L_wall * V2).sum(axis=(1, 2))
        # Reflection off the cylinder reverses only the (unresolved) radial
        # component, so the reflected fraction returns at its incident energy.
        E_ret = alpha * N_wall * e_wall_per + (1.0 - alpha) * E_inc
        diag["wrad_inc"][k] += E_inc
        diag["wrad_ret"][k] += E_ret

        # --- end walls
        for j, (end, stick, e_per, e2_per, hist) in enumerate((
            (-1, s_L, e_L_per, e2_L_per, hist_L),
            (+1, s_R, e_R_per, e2_R_per, hist_R),
        )):
            tot = out[("c", end)] + out[("a", end)]
            N_out = float(tot.sum())
            E_out = 0.5 * M_HE * float((tot * g.V2).sum())
            back = (1.0 - stick) * tot
            N_back = float(back.sum())
            E_back = 0.5 * M_HE * float((back * g.V2).sum())
            # accommodated: re-emitted at the surface spectrum; reflected: the
            # exact v_z mirror, which leaves |v| and hence the energy unchanged
            E_ret_end = alpha * N_back * e_per + (1.0 - alpha) * E_back
            diag["wend_inc"][k, j] += E_out
            diag["wend_ret"][k, j] += E_ret_end
            diag["spec_end_acc"][phase, j] += alpha * N_back * hist
            diag["specE_end_acc"][phase, j] += alpha * N_back * e_per
            diag["specE2_end_acc"][phase, j] += alpha * N_back * e2_per
            ref = (1.0 - alpha) * back
            diag["spec_end_ref"][phase, j] += np.bincount(
                ei.ravel(), weights=ref.ravel(), minlength=N_EBIN
            )
            diag["specE_end_ref"][phase, j] += (1.0 - alpha) * E_back
            diag["specE2_end_ref"][phase, j] += (1.0 - alpha) * float(
                (back * E2_bin).sum()
            )

        # --- radial-wall return spectra
        N_wall_tot = float(N_wall.sum())
        diag["spec_rad_acc"][phase] += alpha * N_wall_tot * hist_wall
        diag["specE_rad_acc"][phase] += alpha * N_wall_tot * e_wall_per
        diag["specE2_rad_acc"][phase] += alpha * N_wall_tot * e2_wall_per
        ref_w = (1.0 - alpha) * L_wall.sum(axis=0)
        diag["spec_rad_ref"][phase] += np.bincount(
            ei.ravel(), weights=ref_w.ravel(), minlength=N_EBIN
        )
        diag["specE_rad_ref"][phase] += (1.0 - alpha) * float(E_inc.sum())
        diag["specE2_rad_ref"][phase] += (1.0 - alpha) * float(
            (L_wall.sum(axis=0) * E2_bin).sum()
        )

        if progress and (step % progress == 0):
            print(
                f"    DVM step {step + 1}/{nsteps} "
                f"t={1e3 * (t0 + dt):.3f} ms  "
                f"({time.perf_counter() - t0_wall:.1f} s)",
                flush=True,
            )

    ledger["launched"] += ledger["external"]
    ledger["resident"] = dvm.total_inventory()
    finalize_diag(diag, shared)
    diag["_ledger"] = ledger
    return diag, dvm


def finalize_diag(diag, shared):
    """Convert the accumulated (quantity * time) sums into the reported units.

    Everything tallied above is an integral over the report bin: densities and
    momentum/energy densities carry a factor of time, rates carry a count. One
    division by the bin width turns each into the bin AVERAGE, and the zone
    volume turns extensive tallies into densities. Spectra are additionally
    divided by their phase duration.
    """
    bin_s = shared["bin_s"]
    for key in ("n_col", "p_col", "e_col", "n_ann", "p_ann", "e_ann",
                "exch_ca", "exch_ac", "wrad_inc", "wrad_ret",
                "wstep_inc", "wstep_ret", "wend_inc", "wend_ret"):
        diag[key] /= bin_s
    t_sw = shared["t_switch"]
    dur = np.array([t_sw, shared["t_end"] - t_sw])
    dur = np.maximum(dur, 1e-300)
    for key in ("spec_rad_acc", "spec_rad_ref"):
        diag[key] /= dur[:, None]
    for key in ("specE_rad_acc", "specE_rad_ref",
                "specE2_rad_acc", "specE2_rad_ref"):
        diag[key] /= dur
    for key in ("spec_end_acc", "spec_end_ref"):
        diag[key] /= dur[:, None, None]
    for key in ("specE_end_acc", "specE_end_ref",
                "specE2_end_acc", "specE2_end_ref"):
        diag[key] /= dur[:, None]


# ------------------------------------------------------------------- MC arm


def maxwell3(rng, N, T_eV, u_z):
    """Sample N velocities from a drifting He Maxwellian [cm/s]."""
    s = np.sqrt(np.maximum(T_eV, 1e-6) * EV / M_HE)
    v = rng.normal(0.0, 1.0, (N, 3)) * np.atleast_1d(s)[:, None]
    v[:, 2] += u_z
    return v


def cosine_z(rng, N, T_K, sign_z):
    """Cosine (flux-weighted) emission off a z-normal surface at T_K."""
    vt = np.sqrt(KB * np.asarray(T_K, dtype=float) / M_HE)
    vt = np.atleast_1d(vt)
    vz = sign_z * vt * np.sqrt(-2.0 * np.log(rng.random(N)))
    vx = rng.normal(0.0, 1.0, N) * vt
    vy = rng.normal(0.0, 1.0, N) * vt
    return np.column_stack((vx, vy, vz))


def cylinder_spectrum(rng, N, T_K):
    """Sample the engine's cylindrical wall-emission spectrum, in 3D.

    The DVM's ``wall_emission_spectrum`` has a Gaussian ``v_z`` marginal and a
    ``v_perp`` marginal proportional to ``vp^2 exp(-vp^2 / 2 s^2)`` -- the
    Maxwell SPEED law in the two perpendicular directions plus the cosine
    weight along the radial normal. Sampling ``v_perp`` as the norm of three
    standard normals reproduces that marginal exactly; the azimuth of the
    perpendicular vector is uniform, which is what an unresolved azimuth means.
    """
    s = np.sqrt(KB * float(T_K) / M_HE)
    vz = rng.normal(0.0, s, N)
    vp = s * np.linalg.norm(rng.normal(0.0, 1.0, (N, 3)), axis=1)
    th = rng.random(N) * 2.0 * np.pi
    return np.column_stack((vp * np.cos(th), vp * np.sin(th), vz))


def wall_emit_cyl(rng, x, y, T_K):
    """Diffuse cosine re-emission off the cylindrical wall at (x, y)."""
    N = x.size
    s = np.sqrt(KB * float(T_K) / M_HE)
    r = np.maximum(np.sqrt(x**2 + y**2), 1e-12)
    nx, ny = -x / r, -y / r
    vn = s * np.sqrt(-2.0 * np.log(rng.random(N)))
    vt1 = rng.normal(0.0, s, N)
    vz = rng.normal(0.0, s, N)
    return np.column_stack((vn * nx - vt1 * ny, vn * ny + vt1 * nx, vz))


def collision_majorants(shared, mode, dvm_grid_vmax):
    """Return per-cell majorant collision frequencies [1/s] for null collisions.

    The majorant must bound the true rate for EVERY velocity a particle can
    carry, so it is maximized over the whole relative-speed range the grid and
    the drifts admit. Sampled events that exceed it are counted and reported;
    a nonzero count invalidates the null-collision sampling and is a failure,
    not a tolerance.
    """
    pl = shared["plasma"]
    n_i = pl["n_i"]
    Ti = pl["Ti_eV"]
    u_i = pl["u_i"]
    g_max = 3.0 * dvm_grid_vmax
    gg = np.linspace(1.0e3, g_max, 4000)
    E = np.maximum(0.25 * M_HE * gg**2 / EV, 1e-9)
    sig_g = (phelps_he_backscatter_cm2(E) + phelps_he_isotropic_cm2(E)) * gg
    if mode == "kinetic":
        K = float(sig_g.max())
        nu_coll_max = n_i * K
    else:
        # The DVM's own rate, maximized over |v - u_i| on the same range.
        # The thermal floor TRANSCRIBES the operator's own (ion-only, full
        # ion mass -- see the collide() BGK branch); a majorant built on a
        # different g_eff does not bound the rate it is used against.
        w = gg
        g_eff = np.sqrt(
            w[None, :] ** 2
            + ion_thermal_g_eff_floor_cm2_s2(Ti[:, None])
        )
        Ee = np.maximum(0.25 * M_HE * g_eff**2 / EV, 1e-9)
        rate = (
            phelps_he_backscatter_cm2(Ee)
            + ELASTIC_BGK_MOMENTUM_FACTOR * phelps_he_isotropic_cm2(Ee)
        ) * g_eff
        # w = 0 is the low end of the interpolation and can dominate; include it
        g0 = np.sqrt(ion_thermal_g_eff_floor_cm2_s2(Ti))
        E0 = np.maximum(0.25 * M_HE * g0**2 / EV, 1e-9)
        rate0 = (
            phelps_he_backscatter_cm2(E0)
            + ELASTIC_BGK_MOMENTUM_FACTOR * phelps_he_isotropic_cm2(E0)
        ) * g0
        nu_coll_max = n_i * np.maximum(rate.max(axis=1), rate0)
    return nu_coll_max * 1.001, float(g_max)


class TransientMC:
    """Transient, time-resolved, full-particle Monte Carlo reference.

    Sources are switched on into an EMPTY vessel at ``t = 0`` and off at
    ``t_switch``; each history is launched at a stratified birth time inside
    the on-window and carries a fixed weight in atoms, so the tallies are
    absolute. Histories are advanced one flight segment at a time and are
    retired when they are ionized, pumped, or reach ``t_end``.

    Track-length estimators are split exactly across the report bins (a
    segment straddling a bin boundary contributes its overlap to each), so the
    bin averages carry no time-binning error.
    """

    def __init__(self, shared, mode, rng, n_particles, accommodation,
                 elastic_model, dvm_grid_vmax):
        self.sh = shared
        self.mode = mode
        self.rng = rng
        self.alpha = float(accommodation)
        self.elastic = elastic_model != "off"
        self.ze = shared["z_edges"]
        self.nz = shared["nz"]
        self.Rp = shared["Rp"]
        self.Rm = shared["Rm"]
        self.pl = shared["plasma"]
        self.nu_ion = self.pl["nu_ion"]
        self.mesh_face = shared["mesh_face"]
        self.transparency = shared["transparency"]
        self.s_end = (shared["s_L"], shared["s_R"])
        self.T_end = (shared["T_s_K"], shared["T_wall_K"])
        self.bin_s = shared["bin_s"]
        self.nbin = shared["nbin"]
        self.t_end = shared["t_end"]
        self.t_switch = shared["t_switch"]
        self.nu_coll_max, self.g_max = collision_majorants(
            shared, mode, dvm_grid_vmax
        )
        self.nu_maj = self.nu_ion + self.nu_coll_max
        self.majorant_violations = 0
        # Particle ledger, in atoms: what was launched must leave through
        # exactly one of these doors, or still be resident at t_end.
        self.lost_ion = 0.0
        self.lost_pump = 0.0
        self.resident = 0.0
        self.stuck = 0.0
        self.n_segments = 0
        self.diag = blank_diag(self.nz, self.nbin)
        self._launch(int(n_particles))

    # -------------------------------------------------------------- launch

    def _launch(self, n_particles):
        rng = self.rng
        sh = self.sh
        src = sh["sources"]
        # The particle budget is stratified over two kinds of channel: the
        # seeded initial inventory (atoms present at t = 0) and the external
        # sources (atoms injected over the on-window). Allocating in
        # proportion to ATOMS gives every history the same weight, which is
        # the minimum-variance allocation for the additive tallies here.
        menu = []
        seed_atoms_c = float((sh["seed_col"] * sh["V_col"]).sum())
        seed_atoms_a = float((sh["seed_ann"] * sh["V_ann"]).sum())
        if seed_atoms_c > 0.0:
            menu.append(("seed_col", seed_atoms_c, True))
        if seed_atoms_a > 0.0:
            menu.append(("seed_ann", seed_atoms_a, True))
        for name in ("puff", "recombination", "anode"):
            arr = np.asarray(src[name], dtype=float)
            if arr.sum() > 0.0:
                menu.append((name, float(arr.sum()) * self.t_switch, False))
        for name in ("cathode_face", "collector_face"):
            if float(src[name]) > 0.0:
                menu.append((name, float(src[name]) * self.t_switch, False))
        atoms = np.array([a for _, a, _ in menu])
        frac = atoms / atoms.sum()
        counts = np.maximum((frac * n_particles).astype(np.int64), 1)
        pos, vel, wgt, born = [], [], [], []
        for (name, a, is_seed), N in zip(menu, counts):
            p, v = self._launch_channel(name, int(N))
            pos.append(p)
            vel.append(v)
            wgt.append(np.full(int(N), a / int(N)))
            if is_seed:
                born.append(np.zeros(int(N)))
            else:
                # stratified birth times: exact uniform coverage of the window
                u = (np.arange(int(N)) + rng.random(int(N))) / int(N)
                born.append(u * self.t_switch)
        self.pos = np.concatenate(pos)
        self.vel = np.concatenate(vel)
        self.wgt = np.concatenate(wgt)
        self.clock = np.concatenate(born)
        self.channels = [
            (n, a, int(c)) for (n, a, _), c in zip(menu, counts)
        ]
        self.launched_atoms = float(self.wgt.sum())
        self.seed_atoms = seed_atoms_c + seed_atoms_a

    def _in_cell(self, ic, radius):
        """Uniform positions inside cylinders of radius ``radius[ic]``."""
        rng = self.rng
        N = ic.size
        rad = radius[ic] * np.sqrt(rng.random(N))
        th = rng.random(N) * 2.0 * np.pi
        z = self.ze[ic] + rng.random(N) * (self.ze[ic + 1] - self.ze[ic])
        return np.column_stack((rad * np.cos(th), rad * np.sin(th), z))

    def _in_annulus(self, ic):
        rng = self.rng
        N = ic.size
        r2 = self.Rp[ic] ** 2 + rng.random(N) * (
            self.Rm[ic] ** 2 - self.Rp[ic] ** 2
        )
        rad = np.sqrt(r2)
        th = rng.random(N) * 2.0 * np.pi
        z = self.ze[ic] + rng.random(N) * (self.ze[ic + 1] - self.ze[ic])
        return np.column_stack((rad * np.cos(th), rad * np.sin(th), z))

    def _launch_channel(self, name, N):
        """Launch a channel with the ENGINE's registered velocity convention."""
        rng = self.rng
        sh = self.sh
        src = sh["sources"]
        if name in ("seed_col", "seed_ann"):
            col = name == "seed_col"
            dens = sh["seed_col"] if col else sh["seed_ann"]
            vol = sh["V_col"] if col else sh["V_ann"]
            w = np.asarray(dens, dtype=float) * vol
            ic = rng.choice(w.size, size=N, p=w / w.sum())
            pos = (
                self._in_cell(ic, self.Rp) if col else self._in_annulus(ic)
            )
            # the seeded state is the engine's own: a 300 K Maxwellian at rest
            vel = maxwell3(rng, N, np.full(N, KB * T_WALL_K / EV), 0.0)
        elif name == "puff":
            w = np.asarray(src["puff"], dtype=float)
            ic = rng.choice(w.size, size=N, p=w / w.sum())
            pos = self._in_annulus(ic)
            # registered channel 5: born at rest as a 300 K Maxwellian
            vel = maxwell3(rng, N, np.full(N, KB * T_WALL_K / EV), 0.0)
        elif name == "recombination":
            w = np.asarray(src["recombination"], dtype=float)
            ic = rng.choice(w.size, size=N, p=w / w.sum())
            pos = self._in_cell(ic, self.Rp)
            vel = maxwell3(rng, N, self.pl["Ti_eV"][ic], 0.0)
            vel[:, 2] += self.pl["u_i"][ic]
        elif name == "anode":
            w = np.asarray(src["anode"], dtype=float)
            ic = rng.choice(w.size, size=N, p=w / w.sum())
            pos = self._in_cell(ic, self.Rp)
            vel = cylinder_spectrum(rng, N, T_WALL_K)
        elif name in ("cathode_face", "collector_face"):
            left = name == "cathode_face"
            radius = self.sh["R_cath"] if left else self.Rp[-1]
            rad = radius * np.sqrt(rng.random(N))
            th = rng.random(N) * 2.0 * np.pi
            z = np.full(N, 1e-6 if left else self.ze[-1] - 1e-6)
            pos = np.column_stack((rad * np.cos(th), rad * np.sin(th), z))
            vel = cosine_z(
                rng, N, self.T_end[0] if left else T_WALL_K,
                1.0 if left else -1.0,
            )
        else:
            raise ValueError(name)
        return pos, vel

    # ------------------------------------------------------------- tallies

    def _deposit_tracks(self, t0, t1, idx_cz, w, vz, v2):
        """Split the segments' track-length weights exactly across bins."""
        nz, nbin, bw = self.nz, self.nbin, self.bin_s
        k0 = np.clip((t0 / bw).astype(np.int64), 0, nbin - 1)
        k1 = np.clip(((t1 - 1e-15) / bw).astype(np.int64), 0, nbin - 1)
        span = int((k1 - k0).max()) if k0.size else 0
        m_all = idx_cz
        for j in range(span + 1):
            kk = k0 + j
            m = kk <= k1
            if not m.any():
                continue
            kkm = kk[m]
            lo = np.maximum(t0[m], kkm * bw)
            hi = np.minimum(t1[m], (kkm + 1) * bw)
            ov = np.maximum(hi - lo, 0.0)
            flat = kkm * (nz * 2) + m_all[m]
            ww = w[m] * ov
            length = nbin * nz * 2
            self._acc_n += np.bincount(flat, weights=ww, minlength=length)
            self._acc_p += np.bincount(
                flat, weights=ww * (M_HE * vz[m]), minlength=length
            )
            self._acc_e += np.bincount(
                flat, weights=ww * (0.5 * M_HE * v2[m]), minlength=length
            )

    def _tbin(self, t):
        return np.clip((t / self.bin_s).astype(np.int64), 0, self.nbin - 1)

    def _phase(self, t):
        return (t >= self.t_switch).astype(np.int64)

    def _spec_add(self, key_mass, key_energy, phase, w, E_erg, sub=None):
        """Add a return population to a spectrum histogram."""
        eb = ebin_index(E_erg / EV)
        d = self.diag
        key_e2 = key_energy.replace("specE_", "specE2_")
        for ph in (0, 1):
            m = phase == ph
            if not m.any():
                continue
            if sub is None:
                d[key_mass][ph] += np.bincount(
                    eb[m], weights=w[m], minlength=N_EBIN
                )
                d[key_energy][ph] += float((w[m] * E_erg[m]).sum())
                d[key_e2][ph] += float((w[m] * E_erg[m] ** 2).sum())
            else:
                for j in (0, 1):
                    mj = m & (sub == j)
                    if not mj.any():
                        continue
                    d[key_mass][ph, j] += np.bincount(
                        eb[mj], weights=w[mj], minlength=N_EBIN
                    )
                    d[key_energy][ph, j] += float((w[mj] * E_erg[mj]).sum())
                    d[key_e2][ph, j] += float(
                        (w[mj] * E_erg[mj] ** 2).sum()
                    )

    # ---------------------------------------------------------------- run

    def run(self, max_iter=200_000, progress=None):
        nz, nbin = self.nz, self.nbin
        length = nbin * nz * 2
        self._acc_n = np.zeros(length)
        self._acc_p = np.zeros(length)
        self._acc_e = np.zeros(length)
        rng = self.rng
        ze, Rp, Rm = self.ze, self.Rp, self.Rm
        t_wall0 = time.perf_counter()
        it = 0
        while self.wgt.size and it < max_iter:
            it += 1
            self._step(rng, ze, Rp, Rm)
            if progress and it % progress == 0:
                print(
                    f"    MC[{self.mode}] iter {it}: {self.wgt.size} live, "
                    f"{self.n_segments / 1e6:.1f}e6 segments "
                    f"({time.perf_counter() - t_wall0:.1f} s)",
                    flush=True,
                )
        if self.wgt.size:
            self.stuck = float(self.wgt.sum())
        self._finish()
        return self.diag

    def _step(self, rng, ze, Rp, Rm):
        pos, vel, wgt, clock = self.pos, self.vel, self.wgt, self.clock
        N = wgt.size
        nz = self.nz
        v2 = (vel * vel).sum(axis=1)
        speed = np.maximum(np.sqrt(v2), 1.0)
        icell = np.clip(np.searchsorted(ze, pos[:, 2]) - 1, 0, nz - 1)
        r2 = pos[:, 0] ** 2 + pos[:, 1] ** 2
        inside = r2 < Rp[icell] ** 2

        with np.errstate(divide="ignore", invalid="ignore"):
            d_z = np.where(
                vel[:, 2] > 0,
                (ze[icell + 1] - pos[:, 2]) / vel[:, 2],
                np.where(
                    vel[:, 2] < 0, (ze[icell] - pos[:, 2]) / vel[:, 2], np.inf
                ),
            ) * speed
        vxy2 = vel[:, 0] ** 2 + vel[:, 1] ** 2
        b = pos[:, 0] * vel[:, 0] + pos[:, 1] * vel[:, 1]
        Rw = Rm[icell]
        disc = b**2 + vxy2 * (Rw**2 - r2)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_wall = (-b + np.sqrt(np.maximum(disc, 0.0))) / np.where(
                vxy2 > 0, vxy2, np.inf
            )
        d_wall = np.where(vxy2 > 0, t_wall * speed, np.inf)
        Rp_here = Rp[icell]
        disc_p = b**2 + vxy2 * (Rp_here**2 - r2)
        sq_p = np.sqrt(np.maximum(disc_p, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_exit = (-b + sq_p) / np.where(vxy2 > 0, vxy2, np.inf)
            t_enter = (-b - sq_p) / np.where(vxy2 > 0, vxy2, np.inf)
        t_rp = np.where(inside, t_exit, np.where(t_enter > 0, t_enter, np.inf))
        d_rp = np.where(
            (vxy2 > 0) & (disc_p > 0) & (t_rp > 1e-12), t_rp * speed, np.inf
        )
        # Null-collision flight. Collisions live in the column only, and a
        # segment never spans the column surface (d_rp terminates it), so the
        # annulus majorant is exactly zero -- no null events are wasted there.
        nu_here = np.where(inside, self.nu_maj[icell], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_coll = np.where(
                nu_here > 0.0,
                -np.log(rng.random(N)) * speed / np.maximum(nu_here, 1e-300),
                np.inf,
            )

        d_geom = np.minimum(np.minimum(d_z, d_wall), np.minimum(d_coll, d_rp))
        dt_geom = d_geom / speed
        dt_left = self.t_end - clock
        dt = np.minimum(dt_geom, dt_left)
        truncated = dt_left <= dt_geom

        zone = np.where(inside, 0, 1).astype(np.int64)
        self._deposit_tracks(
            clock, clock + dt, icell * 2 + zone, wgt, vel[:, 2], v2
        )
        self.n_segments += N

        pos = pos + vel * dt[:, None]
        pos = pos + (vel / speed[:, None]) * 1e-7
        t_ev = clock + dt
        dead = truncated.copy()
        self.resident += float(wgt[truncated].sum())

        # ---- collision events (column only)
        hit_c = (~truncated) & (d_coll <= np.minimum(
            np.minimum(d_z, d_wall), d_rp
        ))
        if hit_c.any():
            idx = np.flatnonzero(hit_c)
            ii = icell[idx]
            u = rng.random(idx.size) * self.nu_maj[ii]
            ionz = u < self.nu_ion[ii]
            dead[idx[ionz]] = True
            self.lost_ion += float(wgt[idx[ionz]].sum())
            rest = idx[~ionz]
            if rest.size:
                self._collide(rng, rest, icell[rest], vel, u[~ionz])

        # ---- cylindrical wall
        hit_w = (~truncated) & (~hit_c) & (d_wall <= np.minimum(d_z, d_rp))
        if hit_w.any():
            idx = np.flatnonzero(hit_w)
            E_in = 0.5 * M_HE * (vel[idx] * vel[idx]).sum(axis=1)
            r_now = np.sqrt(pos[idx, 0] ** 2 + pos[idx, 1] ** 2)
            shrink = (Rm[icell[idx]] * 0.9999) / np.maximum(r_now, 1e-9)
            pos[idx, 0] *= shrink
            pos[idx, 1] *= shrink
            acc = rng.random(idx.size) < self.alpha
            new = vel[idx].copy()
            ia = idx[acc]
            if ia.size:
                new[acc] = wall_emit_cyl(
                    rng, pos[ia, 0], pos[ia, 1], self.sh["T_wall_K"]
                )
            ir = np.flatnonzero(~acc)
            if ir.size:
                # specular off the cylinder: reverse the radial component
                j = idx[~acc]
                rr = np.maximum(np.sqrt(pos[j, 0] ** 2 + pos[j, 1] ** 2), 1e-12)
                nx, ny = pos[j, 0] / rr, pos[j, 1] / rr
                vn = new[ir, 0] * nx + new[ir, 1] * ny
                new[ir, 0] -= 2.0 * vn * nx
                new[ir, 1] -= 2.0 * vn * ny
            vel[idx] = new
            E_out = 0.5 * M_HE * (new * new).sum(axis=1)
            kk = self._tbin(t_ev[idx])
            np.add.at(self.diag["wrad_inc"], (kk, icell[idx]), wgt[idx] * E_in)
            np.add.at(self.diag["wrad_ret"], (kk, icell[idx]), wgt[idx] * E_out)
            ph = self._phase(t_ev[idx])
            self._spec_add(
                "spec_rad_acc", "specE_rad_acc", ph[acc], wgt[idx][acc],
                E_out[acc],
            )
            self._spec_add(
                "spec_rad_ref", "specE_rad_ref", ph[~acc], wgt[idx][~acc],
                E_out[~acc],
            )

        # ---- column-surface crossings: the radial exchange channel
        hit_rp = (~truncated) & (~hit_c) & (~hit_w) & (d_rp < d_z)
        if hit_rp.any():
            idx = np.flatnonzero(hit_rp)
            kk = self._tbin(t_ev[idx])
            out_going = inside[idx]
            np.add.at(
                self.diag["exch_ca"], (kk[out_going], icell[idx][out_going]),
                wgt[idx][out_going],
            )
            np.add.at(
                self.diag["exch_ac"], (kk[~out_going], icell[idx][~out_going]),
                wgt[idx][~out_going],
            )

        # ---- z-edge crossings: ends, mesh, and the annular step face
        hit_z = (~truncated) & (~hit_c) & (~hit_w) & (~hit_rp)
        if hit_z.any():
            idx = np.flatnonzero(hit_z)
            zdir = np.sign(vel[idx, 2])
            edge = np.where(zdir > 0, icell[idx] + 1, icell[idx])
            self._ends(rng, idx, edge, pos, vel, wgt, t_ev, dead)
            self._mesh(rng, idx, edge, icell, pos, vel)
            self._step_face(rng, idx, edge, zdir, icell, pos, vel, wgt, t_ev)

        alive = ~dead
        self.pos = pos[alive]
        self.vel = vel[alive]
        self.wgt = wgt[alive]
        self.clock = t_ev[alive]

    def _collide(self, rng, idx, ii, vel, u_draw):
        """Resolve a real collision on histories ``idx`` in cells ``ii``."""
        Ti = self.pl["Ti_eV"][ii]
        u_i = self.pl["u_i"][ii]
        n_i = self.pl["n_i"][ii]
        vn = vel[idx]
        if self.mode == "kinetic":
            vi = maxwell3(rng, idx.size, Ti, 0.0)
            vi[:, 2] += u_i
            gvec = vn - vi
            gmag = np.maximum(np.linalg.norm(gvec, axis=1), 1.0)
            E = np.maximum(0.25 * M_HE * gmag**2 / EV, 1e-9)
            Qb = phelps_he_backscatter_cm2(E)
            Qi = phelps_he_isotropic_cm2(E) if self.elastic else np.zeros_like(Qb)
            rate = n_i * (Qb + Qi) * gmag
            self.majorant_violations += int(
                np.count_nonzero(rate > self.nu_coll_max[ii] * 1.0000001)
            )
            real = (u_draw - self.nu_ion[ii]) < rate
            if not real.any():
                return
            sel = np.flatnonzero(real)
            j = idx[sel]
            back = rng.random(sel.size) * (Qb[sel] + Qi[sel]) < Qb[sel]
            new = np.empty((sel.size, 3))
            # resonant charge exchange: the pair swaps identities
            new[back] = vi[sel][back]
            if (~back).any():
                # isotropic elastic, equal masses: the neutral leaves the
                # centre of mass at half the relative speed, isotropically
                k = np.flatnonzero(~back)
                vcm = 0.5 * (vn[sel][k] + vi[sel][k])
                nhat = rng.normal(0.0, 1.0, (k.size, 3))
                nhat /= np.linalg.norm(nhat, axis=1)[:, None]
                new[k] = vcm + 0.5 * gmag[sel][k][:, None] * nhat
            vel[j] = new
        else:
            w2 = (
                (vn[:, 2] - u_i) ** 2 + vn[:, 0] ** 2 + vn[:, 1] ** 2
            )
            # Only the IONS are Maxwellian: vn is this particle's own
            # velocity and is already exact in w2, so the thermal floor
            # carries the FULL ion mass, not the two-Maxwellian reduced mass
            # mu = m/2. (The reduced mass in E below is two-body kinematics
            # and does belong there.)
            g_eff = np.sqrt(w2 + ion_thermal_g_eff_floor_cm2_s2(Ti))
            E = np.maximum(0.25 * M_HE * g_eff**2 / EV, 1e-9)
            nu_cx = n_i * phelps_he_backscatter_cm2(E) * g_eff
            nu_el = (
                ELASTIC_BGK_MOMENTUM_FACTOR
                * n_i
                * phelps_he_isotropic_cm2(E)
                * g_eff
            ) if self.elastic else np.zeros_like(nu_cx)
            rate = nu_cx + nu_el
            self.majorant_violations += int(
                np.count_nonzero(rate > self.nu_coll_max[ii] * 1.0000001)
            )
            real = (u_draw - self.nu_ion[ii]) < rate
            if not real.any():
                return
            sel = np.flatnonzero(real)
            j = idx[sel]
            # both DVM channels are full-replacement rebirths at the local
            # ion Maxwellian
            new = maxwell3(rng, sel.size, Ti[sel], 0.0)
            new[:, 2] += u_i[sel]
            vel[j] = new

    def _ends(self, rng, idx, edge, pos, vel, wgt, t_ev, dead):
        for j, (at, sign) in enumerate(
            ((edge == 0, +1.0), (edge == self.nz, -1.0))
        ):
            e = idx[at]
            if e.size == 0:
                continue
            E_in = 0.5 * M_HE * (vel[e] * vel[e]).sum(axis=1)
            kk = self._tbin(t_ev[e])
            np.add.at(self.diag["wend_inc"], (kk, j), wgt[e] * E_in)
            stick = rng.random(e.size) < self.s_end[j]
            dead[e[stick]] = True
            self.lost_pump += float(wgt[e[stick]].sum())
            keep = e[~stick]
            if keep.size == 0:
                continue
            acc = rng.random(keep.size) < self.alpha
            new = vel[keep].copy()
            if acc.any():
                new[acc] = cosine_z(
                    rng, int(acc.sum()), self.T_end[j], sign
                )
            if (~acc).any():
                # specular off a z-normal wall: the exact v_z mirror
                new[~acc, 2] = -new[~acc, 2]
            vel[keep] = new
            pos[keep, 2] = np.clip(pos[keep, 2], 1e-6, self.ze[-1] - 1e-6)
            E_out = 0.5 * M_HE * (new * new).sum(axis=1)
            np.add.at(
                self.diag["wend_ret"], (self._tbin(t_ev[keep]), j),
                wgt[keep] * E_out,
            )
            ph = self._phase(t_ev[keep])
            sub = np.full(keep.size, j, dtype=np.int64)
            self._spec_add(
                "spec_end_acc", "specE_end_acc", ph[acc], wgt[keep][acc],
                E_out[acc], sub=sub[acc],
            )
            self._spec_add(
                "spec_end_ref", "specE_end_ref", ph[~acc], wgt[keep][~acc],
                E_out[~acc], sub=sub[~acc],
            )

    def _mesh(self, rng, idx, edge, icell, pos, vel):
        """Anode-mesh interception, at the engine's re-emission convention."""
        m = (edge == self.mesh_face) & (edge != 0) & (edge != self.nz)
        e = idx[m]
        if e.size == 0:
            return
        blocked = rng.random(e.size) > self.transparency
        b = e[blocked]
        if b.size == 0:
            return
        # re-emitted into the cell the flux came from, at the engine's
        # cylindrical wall spectrum (the shared convention; see the header)
        back_cell = icell[b]
        pos[b] = self._in_cell(back_cell, self.Rp)
        vel[b] = cylinder_spectrum(rng, b.size, self.sh["T_wall_K"])

    def _step_face(self, rng, idx, edge, zdir, icell, pos, vel, wgt, t_ev):
        """Annular step wall where ``Rm`` narrows across a z-edge.

        A ray leaving the expanded end vessel at ``r > Rm`` of the narrow
        section strikes a real z-normal annulus. The DVM's throat-face flux
        form has no such surface, so this channel is reference-only and is
        reported separately rather than folded into either wall row.
        """
        interior = (edge > 0) & (edge < self.nz)
        e = idx[interior]
        if e.size == 0:
            return
        dest = np.where(zdir[interior] > 0, edge[interior], edge[interior] - 1)
        r = np.sqrt(pos[e, 0] ** 2 + pos[e, 1] ** 2)
        hit = r > self.Rm[dest]
        h = e[hit]
        if h.size == 0:
            return
        sign = -zdir[interior][hit]
        E_in = 0.5 * M_HE * (vel[h] * vel[h]).sum(axis=1)
        acc = rng.random(h.size) < self.alpha
        new = vel[h].copy()
        if acc.any():
            new[acc] = cosine_z(
                rng, int(acc.sum()), self.sh["T_wall_K"], sign[acc]
            )
        if (~acc).any():
            new[~acc, 2] = -new[~acc, 2]
        vel[h] = new
        pos[h, 2] += sign * 1e-6
        E_out = 0.5 * M_HE * (new * new).sum(axis=1)
        kk = self._tbin(t_ev[h])
        np.add.at(self.diag["wstep_inc"], (kk, icell[h]), wgt[h] * E_in)
        np.add.at(self.diag["wstep_ret"], (kk, icell[h]), wgt[h] * E_out)

    def _finish(self):
        nz, nbin = self.nz, self.nbin
        acc = {
            "n": self._acc_n.reshape(nbin, nz, 2),
            "p": self._acc_p.reshape(nbin, nz, 2),
            "e": self._acc_e.reshape(nbin, nz, 2),
        }
        V = np.stack(
            (self.sh["V_col"], np.maximum(self.sh["V_ann"], 1e-300)), axis=1
        )
        for q in ("n", "p", "e"):
            for j, zone in enumerate(ZONES):
                self.diag[f"{q}_{zone}"] = acc[q][:, :, j] / V[None, :, j]
        finalize_diag(self.diag, self.sh)


# ------------------------------------------------------------- MC statistics


def mc_reduce(batches):
    """Return (mean, sem) dicts over independent, identically-seeded batches."""
    mean, sem = {}, {}
    n = len(batches)
    for key in batches[0]:
        stack = np.stack([b[key] for b in batches])
        mean[key] = stack.mean(axis=0)
        sem[key] = (
            stack.std(axis=0, ddof=1) / np.sqrt(n) if n > 1
            else np.zeros_like(stack[0])
        )
    return mean, sem


def run_mc(shared, args, mode, accommodation, elastic_model, dvm_vmax,
           label=""):
    """Run the MC reference as ``--batches`` independent, deterministic legs."""
    batches = []
    meta = []
    for k in range(args.batches):
        seed = args.seed + 1000 * k + (0 if mode == "kinetic" else 500_000)
        rng = np.random.default_rng(seed)
        mc = TransientMC(
            shared, mode, rng, args.particles, accommodation, elastic_model,
            dvm_vmax,
        )
        t0 = time.perf_counter()
        diag = mc.run(progress=args.progress)
        dtw = time.perf_counter() - t0
        batches.append(diag)
        meta.append(
            {
                "seed": seed,
                "wall_s": dtw,
                "segments": mc.n_segments,
                "violations": mc.majorant_violations,
                "lost_ion": mc.lost_ion,
                "lost_pump": mc.lost_pump,
                "resident": mc.resident,
                "stuck": mc.stuck,
                "launched_atoms": mc.launched_atoms,
                "channels": mc.channels,
            }
        )
        print(
            f"  MC[{mode}]{label} batch {k + 1}/{args.batches} seed={seed}: "
            f"{mc.n_segments / 1e6:.2f}e6 segments in {dtw:.1f} s, "
            f"majorant violations {mc.majorant_violations}, "
            f"ledger closure "
            f"{(mc.lost_ion + mc.lost_pump + mc.resident + mc.stuck) / max(mc.launched_atoms, 1e-300):.9f}",
            flush=True,
        )
    mean, sem = mc_reduce(batches)
    return mean, sem, meta


# --------------------------------------------------------------- comparison


def region_masks(z_cm):
    return [(lab, (z_cm >= lo) & (z_cm < hi)) for lab, lo, hi in REGIONS]


def agg(field, mask, weight=None):
    """Volume- (or count-) weighted region aggregate of a (nbin, nz) field."""
    if weight is None:
        return field[:, mask].sum(axis=1)
    w = weight[mask]
    return (field[:, mask] * w[None, :]).sum(axis=1) / max(w.sum(), 1e-300)


def compare_rows(shared, dvm_d, mc_m, mc_s, ref_label):
    """Build every E2 comparison row as region aggregates per time bin.

    Densities and the momentum/energy densities are volume-weighted region
    means (the physically meaningful aggregate of an intensive quantity);
    rates and depositions are region SUMS (extensive). Each row carries the
    reference's own standard error, propagated the same way.
    """
    z = shared["z_cm"]
    V_col = shared["V_col"]
    V_ann = shared["V_ann"]
    rows = []
    for lab, m in region_masks(z):
        for key, weight, kind in (
            ("n_col", V_col, "intensive"),
            ("n_ann", V_ann, "intensive"),
            ("p_col", V_col, "intensive"),
            ("p_ann", V_ann, "intensive"),
            ("e_col", V_col, "intensive"),
            ("e_ann", V_ann, "intensive"),
            ("exch_ca", None, "extensive"),
            ("exch_ac", None, "extensive"),
            ("wrad_inc", None, "extensive"),
            ("wrad_ret", None, "extensive"),
            ("wstep_inc", None, "extensive"),
            ("wstep_ret", None, "extensive"),
        ):
            if not m.any():
                continue
            if weight is not None and weight[m].sum() <= 0.0:
                continue
            d = agg(dvm_d[key], m, weight)
            r = agg(mc_m[key], m, weight)
            # the SEM aggregates in quadrature over cells for sums, and with
            # the same volume weights for the intensive means
            if weight is None:
                e = np.sqrt((mc_s[key][:, m] ** 2).sum(axis=1))
            else:
                w = weight[m] / max(weight[m].sum(), 1e-300)
                e = np.sqrt(((mc_s[key][:, m] * w[None, :]) ** 2).sum(axis=1))
            rows.append((lab, key, kind, d, r, e, ref_label))
    for j, end in enumerate(("end L (cathode)", "end R (collector)")):
        for key in ("wend_inc", "wend_ret"):
            rows.append(
                (end, key, "extensive", dvm_d[key][:, j], mc_m[key][:, j],
                 mc_s[key][:, j], ref_label)
            )
    return rows


def dev_sigma(d, r, e):
    """Deviation of the DVM from the reference in units of the MC error."""
    d = np.asarray(d, dtype=float)
    r = np.asarray(r, dtype=float)
    e = np.asarray(e, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(e > 0.0, (d - r) / np.where(e > 0.0, e, 1.0), np.nan)


def rel_dev(d, r):
    """Relative deviation of the DVM from the reference (NaN if ref is zero)."""
    d = np.asarray(d, dtype=float)
    r = np.asarray(r, dtype=float)
    ok = np.abs(r) > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, (d - r) / np.where(ok, r, 1.0), np.nan)


# ------------------------------------------------------- CX-channel adjudication


def dvm_channel_moments(dvm, Ti, u_i, Tn, u_n):
    """Return the DVM's per-``n_i n_n`` CX+elastic transfer moments.

    Evaluated by exact quadrature on the engine's OWN velocity grid with the
    engine's OWN ``collision_frequencies``, so this is the operator's number,
    not a re-derivation. Returns ``(C_P, C_E, C_P_cx, C_P_el, k_cx, k_el)``
    with ``C_P`` in cm^3/s * g cm/s and ``C_E`` in cm^3/s * erg, both per unit
    ``n_i n_n``, and the ``k`` entries the channel rate coefficients in cm^3/s.
    """
    g = dvm.g
    f = g.maxwellian(float(Tn), float(u_n))            # bin masses, sum 1
    nu_cx, nu_el = dvm.collision_frequencies(
        np.array([1.0]), np.array([Ti]), np.array([u_i])
    )
    nu_cx = nu_cx[0]
    nu_el = nu_el[0]
    dvz = g.VZ - float(u_i)
    e_birth = 0.5 * M_HE * float(u_i) ** 2 + 1.5 * max(float(Ti), 0.02) * EV
    de = 0.5 * M_HE * g.V2 - e_birth
    k_cx = float((f * nu_cx).sum())
    k_el = float((f * nu_el).sum())
    C_P_cx = M_HE * float((f * nu_cx * dvz).sum())
    C_P_el = M_HE * float((f * nu_el * dvz).sum())
    C_E = float((f * (nu_cx + nu_el) * de).sum())
    return C_P_cx + C_P_el, C_E, C_P_cx, C_P_el, k_cx, k_el


def mc_channel_moments(rng, N, Ti, u_i, Tn, u_n, elastic=True, chunk=250_000):
    """Chunked wrapper around :func:`_mc_channel_chunk`, so a large sample
    count costs bounded memory. Sums and sums of squares are accumulated, so
    the reported SEM is that of the whole sample, not of the chunk means."""
    tot = {}
    tot2 = {}
    done = 0
    while done < N:
        n = min(chunk, N - done)
        s, s2 = _mc_channel_chunk(rng, n, Ti, u_i, Tn, u_n, elastic)
        for k, v in s.items():
            tot[k] = tot.get(k, 0.0) + v
            tot2[k] = tot2.get(k, 0.0) + s2[k]
        done += n
    out = {}
    for k, v in tot.items():
        mean = v / N
        var = max(tot2[k] / N - mean**2, 0.0) * N / max(N - 1, 1)
        out[k] = mean
        out[k + "_sem"] = float(np.sqrt(var / N))
    return out


def _mc_channel_chunk(rng, N, Ti, u_i, Tn, u_n, elastic=True):
    """Return the TRUE-kinematics per-``n_i n_n`` transfer moments, with SEM.

    Samples a neutral from its Maxwellian and a partner ion from the local ion
    Maxwellian and evaluates the exact two-body event weights:

      backscatter (resonant CX, identity exchange): the plasma gains
        ``m (v_n - v_i)`` and ``(1/2) m (v_n^2 - v_i^2)`` per event;
      isotropic elastic, equal masses: the isotropic angular average leaves
        exactly HALF of each, ``mu/m = 1/2``.

    The estimator is the event-rate-weighted mean, ``<sigma(E) g X>``, which
    is what a rate coefficient multiplied by a transfer actually is -- no
    linearization and no effective-speed interpolation anywhere.
    """
    s_n = np.sqrt(max(float(Tn), 1e-6) * EV / M_HE)
    s_i = np.sqrt(max(float(Ti), 1e-6) * EV / M_HE)
    vn = rng.normal(0.0, s_n, (N, 3))
    vn[:, 2] += float(u_n)
    vi = rng.normal(0.0, s_i, (N, 3))
    vi[:, 2] += float(u_i)
    gvec = vn - vi
    gmag = np.linalg.norm(gvec, axis=1)
    E = np.maximum(0.25 * M_HE * gmag**2 / EV, 1e-12)
    Qb = phelps_he_backscatter_cm2(E)
    Qi = phelps_he_isotropic_cm2(E) if elastic else np.zeros_like(Qb)
    dP = M_HE * gvec[:, 2]
    dE = 0.5 * M_HE * ((vn * vn).sum(axis=1) - (vi * vi).sum(axis=1))
    sP = (Qb + 0.5 * Qi) * gmag * dP
    sE = (Qb + 0.5 * Qi) * gmag * dE
    terms = {
        "C_P": sP,
        "C_E": sE,
        "C_P_cx": Qb * gmag * dP,
        "C_P_el": 0.5 * Qi * gmag * dP,
        "k_cx": Qb * gmag,
        "k_iso": Qi * gmag,
    }
    s = {k: float(v.sum()) for k, v in terms.items()}
    s2 = {k: float((v * v).sum()) for k, v in terms.items()}
    return s, s2


def fluid_channel_moments(Ti, u_i, Tn, u_n):
    """Return the shipped fluid Phelps moment operator's transfer, per n_i n_n.

    ``sources.ion_neutral_collision_rhs`` books, per unit volume,

        dM/dt  = -m nu_mt n (u - u_n),  nu_mt = nn (k_b + 1/2 k_iso)(T_eff)
        dEi/dt = 1/2 m nu_mt n (u - u_n)^2 + 3/2 nu_mt n (T_n - T_i)

    with ``T_eff = (T_i + T_n)/2``. Dividing by ``n nn`` gives the per-pair
    coefficients compared here. The production path pins ``T_n`` to the fixed
    300 K cold-gas value; evaluating it at the state's actual ``T_n`` is the
    generalized form of the same operator and is labelled as such.
    """
    T_eff = 0.5 * (float(Ti) + float(Tn))
    k_mt = float(phelps_momentum_transfer_rate_cm3_s(T_eff, gas_type="He"))
    du = float(u_i) - float(u_n)
    C_P = -M_HE * k_mt * du
    C_Ei = 0.5 * M_HE * k_mt * du**2 + 1.5 * k_mt * (float(Tn) - float(Ti)) * EV
    return C_P, C_Ei, k_mt


# ------------------------------------------------------------------- output


def header_lines(args, shared, bg, extra=()):
    L = [
        f"background               : {args.run}",
        f"plateau window [ms]      : {args.window[0]} to {args.window[1]}",
        f"axial cells (nz)         : {shared['nz']}",
        f"domain length [cm]       : {shared['z_edges'][-1]:.1f}",
        f"velocity grid            : nvz={args.nvz} x nvp={args.nvp}"
        + (
            f"; {shared['grid_note']}" if "grid_note" in shared else ""
        ),
        f"initial condition        : "
        + (
            "the background's own saved two-zone neutral profile, laid down "
            "as a 300 K Maxwellian at rest in BOTH arms"
            if float(np.sum(shared["seed_col"])) > 0.0
            else "empty vessel"
        ),
        f"seeded inventory [atoms] : "
        f"{float((shared['seed_col'] * shared['V_col']).sum()):.6g} column + "
        f"{float((shared['seed_ann'] * shared['V_ann']).sum()):.6g} annulus",
        f"schedule                 : sources ON 0 to "
        f"{1e3 * shared['t_switch']:.3f} ms, OFF to "
        f"{1e3 * shared['t_end']:.3f} ms; report bins "
        f"{1e3 * shared['bin_s']:.3f} ms ({shared['nbin']} bins)",
        f"DVM neutral clock dt [s] : {args.dvm_dt:.6g}",
        f"MC particles / batch     : {args.particles:,}",
        f"MC batches (seeds)       : {args.batches} from base seed {args.seed}",
        f"accommodation            : {args.accommodation}",
        f"elastic model            : {args.elastic_model}",
        f"total source rate [at/s] : {shared['total_rate']:.6g}",
        f"end sticking s_L / s_R   : {shared['s_L']:.6g} / {shared['s_R']:.6g}",
        f"mesh face / transparency : {shared['mesh_face']} / "
        f"{shared['transparency']:.6g}",
        f"T_wall / T_s [K]         : {shared['T_wall_K']:.1f} / "
        f"{shared['T_s_K']:.2f}",
        f"machine                  : {platform.platform()}",
        f"python / numpy           : {platform.python_version()} / "
        f"{np.__version__}",
        f"command                  : {args.cmdline}",
    ]
    L.extend(extra)
    return L


UNITS = {
    "n_col": "cm^-3", "n_ann": "cm^-3",
    "p_col": "g cm/s /cm^3", "p_ann": "g cm/s /cm^3",
    "e_col": "erg/cm^3", "e_ann": "erg/cm^3",
    "exch_ca": "atoms/s", "exch_ac": "atoms/s",
    "wrad_inc": "erg/s", "wrad_ret": "erg/s",
    "wstep_inc": "erg/s", "wstep_ret": "erg/s",
    "wend_inc": "erg/s", "wend_ret": "erg/s",
}

ROW_TITLES = {
    "n_col": "nn_col   column neutral density",
    "n_ann": "nn_ann   annulus neutral density",
    "p_col": "column axial momentum density",
    "p_ann": "annulus axial momentum density",
    "e_col": "column kinetic energy density",
    "e_ann": "annulus kinetic energy density",
    "exch_ca": "radial exchange, column -> annulus",
    "exch_ac": "radial exchange, annulus -> column",
    "wrad_inc": "radial wall energy incident",
    "wrad_ret": "radial wall energy returned",
    "wstep_inc": "step-face energy incident (reference-only)",
    "wstep_ret": "step-face energy returned (reference-only)",
    "wend_inc": "end wall energy incident",
    "wend_ret": "end wall energy returned",
}


def fmt_bin_table(L, shared, rows, key, ref_label):
    """Print one E2 quantity as a matched-time table over the report bins."""
    bin_s = shared["bin_s"]
    t_sw = shared["t_switch"]
    L.append("")
    L.append(f"### {ROW_TITLES[key]}   [{UNITS[key]}]")
    L.append(f"    reference: {ref_label}")
    L.append(
        f"{'region':<22s} {'t [ms]':>12s} {'phase':>5s} {'DVM':>12s} "
        f"{'reference':>12s} {'MC SEM':>10s} {'dev %':>9s} {'dev/sigma':>10s}"
    )
    L.append("-" * 100)
    for lab, k, kind, d, r, e, _ in rows:
        if k != key:
            continue
        for b in range(shared["nbin"]):
            t0 = b * bin_s
            ph = "tran" if t0 < t_sw - 1e-15 else "relx"
            L.append(
                f"{lab:<22s} "
                f"{1e3 * t0:5.2f}-{1e3 * (t0 + bin_s):<6.2f} {ph:>5s} "
                f"{d[b]:12.5g} {r[b]:12.5g} {e[b]:10.3g} "
                f"{100 * rel_dev(d[b], r[b]):9.2f} "
                f"{dev_sigma(d[b], r[b], e[b]):10.2f}"
            )
        L.append("")


def spectrum_block(L, shared, dvm_d, mc_m, mc_s, ref_label):
    """Accommodated / non-accommodated return spectra, per surface and phase."""
    L.append("")
    L.append("## Return spectra: accommodated and non-accommodated")
    L.append(
        "Mass-normalized energy histograms of the population RETURNED off each "
        "surface, per phase. Absolute return rates [atoms/s] and mean return "
        "energies [eV] head each block; the histogram rows are fractions of "
        "that return."
    )
    L.append(f"reference: {ref_label}")
    edges = E_BIN_EDGES_EV
    labels = [
        f"{edges[i]:.4g}-{edges[i + 1]:.4g}" if np.isfinite(edges[i + 1])
        else f">{edges[i]:.4g}"
        for i in range(N_EBIN)
    ]
    L.append(
        "The DVM's spectrum is its velocity grid's DISCRETE representation, "
        "and each grid bin's whole mass lands in the energy bin of its "
        "centre. On the shipped grid the 300 K wall population is carried by "
        "only a handful of perpendicular bins, so the per-bin FRACTION rows "
        "below carry that binning on top of any real spectral difference. "
        "The mean and RMS return energies do not, and are the grid-robust "
        "statements."
    )
    blocks = [
        ("radial wall, accommodated", "spec_rad_acc", "specE_rad_acc", None),
        ("radial wall, NON-accommodated", "spec_rad_ref", "specE_rad_ref", None),
        ("end L (cathode), accommodated", "spec_end_acc", "specE_end_acc", 0),
        ("end L (cathode), NON-accommodated", "spec_end_ref", "specE_end_ref", 0),
        ("end R (collector), accommodated", "spec_end_acc", "specE_end_acc", 1),
        ("end R (collector), NON-accommodated", "spec_end_ref", "specE_end_ref",
         1),
    ]
    for phase, pname in ((0, "transient (sources on)"),
                         (1, "relaxation (sources off)")):
        L.append("")
        L.append(f"### phase: {pname}")
        for title, mkey, ekey, sub in blocks:
            e2key = ekey.replace("specE_", "specE2_")
            if sub is None:
                dm, rm, rs = (
                    dvm_d[mkey][phase], mc_m[mkey][phase], mc_s[mkey][phase]
                )
                de, re_ = dvm_d[ekey][phase], mc_m[ekey][phase]
                res = mc_s[ekey][phase]
                d2, r2_ = dvm_d[e2key][phase], mc_m[e2key][phase]
                r2s = mc_s[e2key][phase]
            else:
                dm, rm, rs = (
                    dvm_d[mkey][phase, sub], mc_m[mkey][phase, sub],
                    mc_s[mkey][phase, sub],
                )
                de, re_ = dvm_d[ekey][phase, sub], mc_m[ekey][phase, sub]
                res = mc_s[ekey][phase, sub]
                d2, r2_ = dvm_d[e2key][phase, sub], mc_m[e2key][phase, sub]
                r2s = mc_s[e2key][phase, sub]
            dtot, rtot = float(dm.sum()), float(rm.sum())
            rtot_e = float(np.sqrt((rs**2).sum()))
            L.append("")
            L.append(f"-- {title}")
            L.append(
                f"   return rate [atoms/s]:  DVM {dtot:12.5g}   "
                f"reference {rtot:12.5g} +/- {rtot_e:.3g}   "
                f"dev {100 * rel_dev(dtot, rtot):8.2f}%   "
                f"dev/sigma {dev_sigma(dtot, rtot, rtot_e):7.2f}"
            )
            mean_d = de / max(dtot, 1e-300) / EV
            mean_r = re_ / max(rtot, 1e-300) / EV
            mean_r_e = res / max(rtot, 1e-300) / EV
            L.append(
                f"   mean return energy [eV]: DVM {mean_d:12.5g}   "
                f"reference {mean_r:12.5g} +/- {mean_r_e:.3g}   "
                f"dev {100 * rel_dev(mean_d, mean_r):8.2f}%   "
                f"dev/sigma {dev_sigma(mean_d, mean_r, mean_r_e):7.2f}"
            )
            rms_d = np.sqrt(max(d2 / max(dtot, 1e-300), 0.0)) / EV
            rms_r = np.sqrt(max(r2_ / max(rtot, 1e-300), 0.0)) / EV
            rms_r_e = (
                0.5 * r2s / max(r2_, 1e-300) * rms_r if r2_ > 0.0 else 0.0
            )
            L.append(
                f"   RMS return energy [eV]:  DVM {rms_d:12.5g}   "
                f"reference {rms_r:12.5g} +/- {rms_r_e:.3g}   "
                f"dev {100 * rel_dev(rms_d, rms_r):8.2f}%   "
                f"dev/sigma {dev_sigma(rms_d, rms_r, rms_r_e):7.2f}"
            )
            if rtot <= 0.0 and dtot <= 0.0:
                L.append("   (channel inactive at this accommodation)")
                continue
            L.append(
                f"   {'E bin [eV]':>16s} {'DVM frac':>10s} "
                f"{'ref frac':>10s} {'ref SEM':>10s} {'dev/sigma':>10s}"
            )
            for i in range(N_EBIN):
                fd = dm[i] / max(dtot, 1e-300)
                fr = rm[i] / max(rtot, 1e-300)
                fe = rs[i] / max(rtot, 1e-300)
                if fd == 0.0 and fr == 0.0:
                    continue
                L.append(
                    f"   {labels[i]:>16s} {fd:10.5f} {fr:10.5f} "
                    f"{fe:10.3g} {dev_sigma(fd, fr, fe):10.2f}"
                )


def write_compare(path, args, shared, bg, blocks, notes):
    L = []
    L.append("E2 matched-time physics-reference comparison -- transient DVM arm")
    L.append("=" * 100)
    L.extend(header_lines(args, shared, bg))
    L.append("")
    L.append("REFERENCE POLICY")
    L.append("-" * 100)
    L.append(
        "The shipped KN2Zone two-zone steady march is NON-CONSERVATIVE in the "
        "annulus channel at the area-jump faces of this geometry (K2a review); "
        "it is NOT used as truth in any row here. The reference on every row "
        "below is the transient full-particle MC named in that row's header. "
        "The time-dependent single-zone KN2ZoneJump variant is not exercised: "
        "every E2 row except nn_col is a two-zone quantity that a uniform-area "
        "single-zone reference cannot produce."
    )
    for note in notes:
        L.append("")
        L.append(note)
    for title, dvm_d, mc_m, mc_s, ref_label, rows in blocks:
        L.append("")
        L.append("=" * 100)
        L.append(f"# {title}")
        L.append("=" * 100)
        for key in (
            "n_col", "n_ann", "p_col", "p_ann", "e_col", "e_ann",
            "exch_ca", "exch_ac", "wrad_inc", "wrad_ret",
            "wend_inc", "wend_ret", "wstep_inc", "wstep_ret",
        ):
            fmt_bin_table(L, shared, rows, key, ref_label)
        spectrum_block(L, shared, dvm_d, mc_m, mc_s, ref_label)
    Path(path).write_text("\n".join(L) + "\n")


def write_cx(path, args, shared, table, extra_lines):
    L = []
    L.append("E2 CX / elastic channel adjudication -- DVM vs true-kinematics MC")
    L.append("=" * 100)
    L.extend(header_lines(args, shared, None))
    L.append("")
    L.append("WHAT THIS MEASURES")
    L.append("-" * 100)
    L.append(
        "The DVM books ion-neutral momentum and energy transfer as BGK "
        "full-replacement events at a frequency evaluated at ONE effective "
        "relative speed, g_eff^2 = |v - u_i|^2 + 8 k T_i / (pi m_He) -- the "
        "thermal floor carries the FULL ion mass because the neutral velocity "
        "is resolved and enters exactly in |v - u_i|^2, so only the ions are "
        "Maxwellian. That is an "
        "interpolation, not the Maxwellian rate average <sigma(g) g>, and the "
        "two differ because sigma_b(E) is not flat. The MC below samples a "
        "partner ion from the local Maxwellian and evaluates the exact "
        "two-body event weights, so it measures the average the DVM "
        "approximates. The fluid column is the shipped Phelps moment "
        "operator (sources.ion_neutral_collision_rhs), evaluated at "
        "T_eff = (T_i + T_n)/2."
    )
    L.append("")
    L.append(
        "All coefficients are per unit n_i * n_n. C_P is the axial momentum "
        "gained by the plasma [cm^3/s * g cm/s]; C_Ei is the plasma INTERNAL "
        "energy gained [cm^3/s * erg], formed on all three sides with the same "
        "bulk decomposition the engine uses, C_Ei = C_E - u_i C_P. Signs are "
        "as booked into the plasma rows."
    )
    L.extend(extra_lines)
    L.append("")
    L.append("=" * 100)
    L.append("# Momentum channel")
    L.append("=" * 100)
    L.append(
        f"{'state':<34s} {'C_P DVM':>12s} {'C_P MC':>12s} {'MC SEM':>10s} "
        f"{'C_P fluid':>12s} {'DVM/MC':>8s} {'fluid/MC':>9s} "
        f"{'DVM dev/sig':>11s}"
    )
    L.append("-" * 118)
    for r in table:
        L.append(
            f"{r['label']:<34s} {r['P_dvm']:12.5g} {r['P_mc']:12.5g} "
            f"{r['P_mc_sem']:10.3g} {r['P_fluid']:12.5g} "
            f"{r['P_dvm'] / r['P_mc'] if r['P_mc'] else float('nan'):8.4f} "
            f"{r['P_fluid'] / r['P_mc'] if r['P_mc'] else float('nan'):9.4f} "
            f"{dev_sigma(r['P_dvm'], r['P_mc'], r['P_mc_sem']):11.1f}"
        )
    L.append("")
    L.append("=" * 100)
    L.append("# Internal-energy channel")
    L.append("=" * 100)
    L.append(
        f"{'state':<34s} {'C_Ei DVM':>12s} {'C_Ei MC':>12s} {'MC SEM':>10s} "
        f"{'C_Ei fluid':>12s} {'DVM/MC':>8s} {'fluid/MC':>9s} "
        f"{'DVM dev/sig':>11s}"
    )
    L.append("-" * 118)
    for r in table:
        L.append(
            f"{r['label']:<34s} {r['E_dvm']:12.5g} {r['E_mc']:12.5g} "
            f"{r['E_mc_sem']:10.3g} {r['E_fluid']:12.5g} "
            f"{r['E_dvm'] / r['E_mc'] if r['E_mc'] else float('nan'):8.4f} "
            f"{r['E_fluid'] / r['E_mc'] if r['E_mc'] else float('nan'):9.4f} "
            f"{dev_sigma(r['E_dvm'], r['E_mc'], r['E_mc_sem']):11.1f}"
        )
    L.append("")
    L.append(
        "NB the internal-energy column subtracts the bulk term u_i C_P, and "
        "at T_n = T_i with a small relative drift the remainder is a near "
        "cancellation of two much larger numbers: the ratios there are "
        "ratios of residuals and are reported for completeness, not as "
        "fractional errors. The TOTAL kinetic-energy moment below does not "
        "cancel and is the readable form of the same channel."
    )
    L.append("")
    L.append("=" * 100)
    L.append("# Total kinetic-energy channel (no bulk subtraction)")
    L.append("=" * 100)
    L.append(
        f"{'state':<34s} {'C_E DVM':>12s} {'C_E MC':>12s} {'MC SEM':>10s} "
        f"{'DVM/MC':>8s} {'DVM dev/sig':>11s}"
    )
    L.append("-" * 118)
    for r in table:
        L.append(
            f"{r['label']:<34s} {r['Etot_dvm']:12.5g} {r['Etot_mc']:12.5g} "
            f"{r['Etot_mc_sem']:10.3g} "
            f"{r['Etot_dvm'] / r['Etot_mc'] if r['Etot_mc'] else float('nan'):8.4f} "
            f"{dev_sigma(r['Etot_dvm'], r['Etot_mc'], r['Etot_mc_sem']):11.1f}"
        )
    L.append("")
    L.append("=" * 100)
    L.append("# The deliberately-ungated kinetic/fluid drag ratio, decomposed")
    L.append("=" * 100)
    L.append(
        "C_P DVM/fluid is the ratio the elastic-half review reported "
        "(~1.34x). Splitting it against the true-kinematics reference says "
        "how much of it is the DVM's g_eff interpolation running high and how "
        "much is the fluid moment operator running low."
    )
    L.append(
        f"{'state':<34s} {'DVM/fluid':>10s} {'DVM/MC':>10s} {'fluid/MC':>10s}"
    )
    L.append("-" * 118)
    for r in table:
        L.append(
            f"{r['label']:<34s} "
            f"{r['P_dvm'] / r['P_fluid'] if r['P_fluid'] else float('nan'):10.4f} "
            f"{r['P_dvm'] / r['P_mc'] if r['P_mc'] else float('nan'):10.4f} "
            f"{r['P_fluid'] / r['P_mc'] if r['P_mc'] else float('nan'):10.4f}"
        )
    L.append("")
    L.append("=" * 100)
    L.append("# Channel rate coefficients [cm^3/s]")
    L.append("=" * 100)
    L.append(
        f"{'state':<34s} {'k_cx DVM':>12s} {'k_cx MC':>12s} {'MC SEM':>10s} "
        f"{'ratio':>8s} {'k_el DVM':>12s} {'k_el MC(x1/2)':>13s} "
        f"{'ratio':>8s}"
    )
    L.append("-" * 118)
    for r in table:
        L.append(
            f"{r['label']:<34s} {r['kcx_dvm']:12.5g} {r['kcx_mc']:12.5g} "
            f"{r['kcx_mc_sem']:10.3g} "
            f"{r['kcx_dvm'] / r['kcx_mc'] if r['kcx_mc'] else float('nan'):8.4f} "
            f"{r['kel_dvm']:12.5g} {r['kel_mc']:13.5g} "
            f"{r['kel_dvm'] / r['kel_mc'] if r['kel_mc'] else float('nan'):8.4f}"
        )
    Path(path).write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------- main


def sampling_check(dvm, shared, args):
    """Check that the MC samples exactly the spectra the engine books.

    The whole artifact rests on the two arms being fed the same
    distributions, so this projects large MC samples onto the engine's own
    velocity grid and compares the resulting bin masses against the analytic
    bin masses the engine uses. A residual here is either a sampling bug or
    the grid's own discretization -- and the mean-energy column separates
    them, because the engine's ``maxwellian`` projection is moment-compensated
    while its wall and half-flux spectra are raw bin masses.
    """
    g = dvm.g
    rng = np.random.default_rng(args.seed + 31)
    N = int(args.sampling_check)
    T_s = shared["T_s_K"]
    T_w = shared["T_wall_K"]

    def project(v):
        iz = np.clip(np.searchsorted(g.vz_edges, v[:, 2]) - 1, 0, g.nvz - 1)
        vp = np.hypot(v[:, 0], v[:, 1])
        ip = np.clip(np.searchsorted(g.vp_edges, vp) - 1, 0, g.nvp - 1)
        h = np.bincount(iz * g.nvp + ip, minlength=g.nvz * g.nvp)
        return h.reshape(g.nvz, g.nvp) / h.sum()

    cases = (
        ("cylindrical wall re-emission",
         lambda: cylinder_spectrum(rng, N, T_w),
         g.wall_emission_spectrum(T_w)),
        ("end R half-flux (300 K, -z)",
         lambda: cosine_z(rng, N, T_w, -1.0),
         g.half_flux_spectrum(T_w, -1)),
        ("end L half-flux (T_s, +z)",
         lambda: cosine_z(rng, N, T_s, +1.0),
         g.half_flux_spectrum(T_s, +1)),
        ("300 K volume Maxwellian (puff, seed)",
         lambda: maxwell3(rng, N, np.full(N, KB * T_w / EV), 0.0),
         g.maxwellian(KB * T_w / EV, 0.0)),
    )
    L = [
        "REFERENCE SAMPLING CHECK (do both arms see the same distributions?)",
        "-" * 100,
        f"{N:,} samples per case, projected onto the engine's own "
        f"{g.nvz}x{g.nvp} grid and compared with the analytic bin masses the "
        f"engine uses.",
        f"{'distribution':<38s} {'L1 bin-mass diff':>17s} "
        f"{'max |dev|/SEM':>14s} {'<E> MC [eV]':>13s} {'<E> grid [eV]':>14s} "
        f"{'ratio':>8s}",
    ]
    for name, sample, target in cases:
        v = sample()
        h = project(v)
        p = np.maximum(target, 0.0)
        sem = np.sqrt(p * (1.0 - p) / N)
        dev = np.where(sem > 0.0, (h - target) / np.maximum(sem, 1e-30), 0.0)
        eS = float(0.5 * M_HE * (v * v).sum(axis=1).mean() / EV)
        eT = float(0.5 * M_HE * (target * g.V2).sum() / EV)
        L.append(
            f"{name:<38s} {np.abs(h - target).sum():17.5f} "
            f"{np.abs(dev).max():14.2f} {eS:13.6f} {eT:14.6f} "
            f"{eS / eT:8.5f}"
        )
    L.append(
        "The volume Maxwellian's mean energy matches to rounding because the "
        "engine's Maxwellian projection is moment-compensated. The wall and "
        "half-flux spectra are raw analytic bin masses with no compensation, "
        "so their ratio is the shipped grid's own mean-energy discretization "
        "error and it propagates directly into the return-spectrum rows."
    )
    return "\n".join(L)


def ledger_note(dvm_ledger, mc_meta, args):
    """Compare the two arms' whole-window particle ledgers.

    Not an E2 row, but the statement that makes the rows readable: both arms
    started from the same seeded inventory and were fed the same external
    ledger, so a difference in what they IONIZED, PUMPED or still HOLD over
    the window is the integrated form of every per-bin deviation below.
    """
    nb = len(mc_meta)
    def tot(key):
        return sum(m[key] for m in mc_meta) / nb
    mc = {
        "launched": tot("launched_atoms"),
        "ionized": tot("lost_ion"),
        "pumped": tot("lost_pump"),
        "resident": tot("resident"),
        "capped": tot("stuck"),
    }
    d = dvm_ledger
    L = [
        "WHOLE-WINDOW PARTICLE LEDGER (both arms, same inputs)",
        "-" * 100,
        f"{'channel [atoms]':<34s} {'DVM':>16s} {'reference (MC)':>16s} "
        f"{'ratio':>10s}",
    ]
    for label, dk, mk in (
        ("seeded + externally injected", "launched", "launched"),
        ("ionized in the column", "ionized", "ionized"),
        ("pumped at the ends", "pumped", "pumped"),
        ("resident at t_end", "resident", "resident"),
    ):
        dv, mv = float(d[dk]), float(mc[mk])
        L.append(
            f"{label:<34s} {dv:16.6g} {mv:16.6g} "
            f"{dv / mv if mv else float('nan'):10.4f}"
        )
    dsum = d["ionized"] + d["pumped"] + d["resident"]
    msum = mc["ionized"] + mc["pumped"] + mc["resident"] + mc["capped"]
    L.append(
        f"{'closure (out+held)/(in)':<34s} "
        f"{dsum / max(d['launched'], 1e-300):16.10f} "
        f"{msum / max(mc['launched'], 1e-300):16.10f}"
    )
    L.append(
        "The MC column is the mean over its independent batches; the DVM "
        "column is exact bookkeeping from the engine's own per-update ledger."
    )
    return "\n".join(L)


def worst_rows(rows, keys):
    """Return the worst |dev/sigma| and worst |dev %| per quantity."""
    out = {}
    for key in keys:
        best_s = (0.0, None)
        best_r = (0.0, None)
        for lab, k, kind, d, r, e, ref in rows:
            if k != key:
                continue
            for b in range(d.size):
                s = dev_sigma(d[b], r[b], e[b])
                p = rel_dev(d[b], r[b])
                if np.isfinite(s) and abs(s) > abs(best_s[0]):
                    best_s = (s, (lab, b))
                if np.isfinite(p) and abs(p) > abs(best_r[0]):
                    best_r = (p, (lab, b))
        out[key] = (best_s, best_r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E2 matched-time physics-reference comparison for the "
                    "transient DVM neutral arm."
    )
    ap.add_argument(
        "--run",
        default=str(Path(__file__).resolve().parent
                    / "es1_kn2z_promoted_nx240.h5"),
        help="saved nx=240 production background (read in place)",
    )
    ap.add_argument("--window", nargs=2, type=float, default=(5.0, 19.5))
    ap.add_argument("--nvz", type=int, default=48)
    ap.add_argument("--nvp", type=int, default=12)
    ap.add_argument("--dvm-dt", type=float, default=2.5e-5,
                    help="DVM neutral-clock tick [s]; must divide --bin-ms")
    ap.add_argument("--t-end-ms", type=float, default=6.0)
    ap.add_argument("--t-switch-ms", type=float, default=3.0)
    ap.add_argument("--bin-ms", type=float, default=0.5)
    ap.add_argument("--particles", type=int, default=120_000)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--no-seed-state", dest="seed_state", action="store_false",
                    default=True,
                    help="start from an EMPTY vessel instead of the "
                         "background's own saved two-zone neutral profile")
    ap.add_argument("--accommodation", type=float, default=1.0)
    ap.add_argument("--accommodation-split", type=float, default=0.9,
                    help="second configuration, run so the non-accommodated "
                         "return-spectrum rows are populated")
    ap.add_argument("--elastic-model", default="phelps_iso",
                    choices=("phelps_iso", "off"))
    ap.add_argument("--sampling-check", type=int, default=2_000_000,
                    help="samples per distribution in the reference-sampling "
                         "check")
    ap.add_argument("--cx-samples", type=int, default=4_000_000,
                    help="MC samples per state in the CX-channel adjudication")
    ap.add_argument("--dt-refine", action="store_true", default=True,
                    help="also run the DVM at dt/2 as a self-convergence check")
    ap.add_argument("--no-dt-refine", dest="dt_refine", action="store_false")
    ap.add_argument("--no-bgk-arm", dest="bgk_arm", action="store_false",
                    default=True,
                    help="skip the DVM-rate MC arm (kinetic arm only)")
    ap.add_argument("--no-split-arm", dest="split_arm", action="store_false",
                    default=True,
                    help="skip the partial-accommodation configuration")
    ap.add_argument("--progress", type=int, default=0,
                    help="print progress every N iterations (0 = quiet)")
    ap.add_argument("--out-dir", default="scripts")
    ap.add_argument("--tag", default="nx240")
    args = ap.parse_args(argv)

    args.cmdline = " ".join(
        [f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}",
         sys.executable, str(Path(sys.argv[0]))] + list(sys.argv[1:])
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp_path = out_dir / f"neutral_arch_e2_compare_{args.tag}.txt"
    cx_path = out_dir / "neutral_arch_e2_cx_channel.txt"
    sum_path = out_dir / "neutral_arch_e2_summary.md"
    (out_dir / "neutral_arch_e2_compare.cmd").write_text(
        args.cmdline + "\n"
    )

    print(f"E2 compare: loading {args.run}", flush=True)
    bg = load_background(args.run, tuple(args.window))
    shared = build_shared(bg, args)
    print(
        f"  nz={shared['nz']}, L={shared['z_edges'][-1]:.0f} cm, "
        f"total source {shared['total_rate']:.4g} atoms/s",
        flush=True,
    )

    t_all = time.perf_counter()
    notes = []
    blocks = []
    summary = {}

    # ---------------- configuration A: the shipped accommodation
    print(f"DVM arm (alpha={args.accommodation}) ...", flush=True)
    t0 = time.perf_counter()
    dvm_A, dvm_obj = run_dvm(
        shared, args.dvm_dt, args.nvz, args.nvp, args.accommodation,
        args.elastic_model, progress=args.progress,
    )
    dvm_A_wall = time.perf_counter() - t0
    dvm_vmax = float(dvm_obj.g.vz.max())
    s_wall = np.sqrt(KB * T_WALL_K / M_HE)
    shared["grid_note"] = (
        f"vmax {dvm_vmax:.4g} cm/s; the 300 K wall population (thermal "
        f"spread {s_wall:.4g} cm/s) is carried by "
        f"{int(np.count_nonzero(dvm_obj.g.vp < 2.0 * s_wall))} of "
        f"{args.nvp} perpendicular bins and "
        f"{int(np.count_nonzero(np.abs(dvm_obj.g.vz) < 2.0 * s_wall))} of "
        f"{args.nvz} axial bins"
    )
    print(f"  DVM done in {dvm_A_wall:.1f} s (vmax {dvm_vmax:.4g} cm/s)",
          flush=True)

    print("MC reference, true kinematics ...", flush=True)
    mc_kin_m, mc_kin_s, mc_kin_meta = run_mc(
        shared, args, "kinetic", args.accommodation, args.elastic_model,
        dvm_vmax,
    )
    rows_kin = compare_rows(
        shared, dvm_A, mc_kin_m, mc_kin_s,
        "transient full-particle TPMC, TRUE two-body kinematics",
    )
    blocks.append((
        f"Configuration A: accommodation = {args.accommodation}; reference = "
        "true-kinematics transient TPMC",
        dvm_A, mc_kin_m, mc_kin_s,
        "transient full-particle TPMC, TRUE two-body kinematics",
        rows_kin,
    ))
    summary["kin"] = (rows_kin, mc_kin_meta)

    if args.bgk_arm:
        print("MC reference, DVM-BGK collision rates ...", flush=True)
        mc_bgk_m, mc_bgk_s, mc_bgk_meta = run_mc(
            shared, args, "bgk", args.accommodation, args.elastic_model,
            dvm_vmax,
        )
        rows_bgk = compare_rows(
            shared, dvm_A, mc_bgk_m, mc_bgk_s,
            "transient full-particle TPMC, the DVM's OWN BGK collision rates",
        )
        blocks.append((
            f"Configuration A': accommodation = {args.accommodation}; "
            "reference = TPMC carrying the DVM's own collision operator "
            "(isolates transport from collision-model error)",
            dvm_A, mc_bgk_m, mc_bgk_s,
            "transient full-particle TPMC, the DVM's OWN BGK collision rates",
            rows_bgk,
        ))
        summary["bgk"] = (rows_bgk, mc_bgk_meta)

    # ---------------- configuration B: partial accommodation
    if args.split_arm:
        a2 = args.accommodation_split
        print(f"DVM arm (alpha={a2}) ...", flush=True)
        dvm_B, _ = run_dvm(
            shared, args.dvm_dt, args.nvz, args.nvp, a2, args.elastic_model,
            progress=args.progress,
        )
        print("MC reference, true kinematics, partial accommodation ...",
              flush=True)
        mc_B_m, mc_B_s, mc_B_meta = run_mc(
            shared, args, "kinetic", a2, args.elastic_model, dvm_vmax,
            label=f" alpha={a2}",
        )
        rows_B = compare_rows(
            shared, dvm_B, mc_B_m, mc_B_s,
            "transient full-particle TPMC, TRUE two-body kinematics",
        )
        blocks.append((
            f"Configuration B: accommodation = {a2} (populates the "
            "NON-accommodated return-spectrum rows); reference = "
            "true-kinematics transient TPMC",
            dvm_B, mc_B_m, mc_B_s,
            "transient full-particle TPMC, TRUE two-body kinematics",
            rows_B,
        ))
        summary["split"] = (rows_B, mc_B_meta)

    # ---------------- DVM self-convergence in dt
    dt_note = None
    dt_band = {}
    if args.dt_refine:
        print("DVM self-convergence: dt/2 ...", flush=True)
        dvm_h, _ = run_dvm(
            shared, 0.5 * args.dvm_dt, args.nvz, args.nvp, args.accommodation,
            args.elastic_model, progress=args.progress,
        )
        lines = [
            "DVM SELF-CONVERGENCE IN THE NEUTRAL CLOCK",
            "-" * 100,
            f"The same configuration A run at dt = {args.dvm_dt:.4g} s and at "
            f"dt/2. This bounds how much of any DVM-vs-reference deviation "
            f"below is the DVM's own time discretization.",
            f"{'quantity':<12s} {'region':<22s} {'max |dt vs dt/2| %':>20s} "
            f"{'at bin':>7s} {'last bin %':>11s}",
        ]
        # The end-wall rows are indexed by END, not by axial cell, so they
        # are compared below rather than through the region masks.
        for key in ("n_col", "n_ann", "p_col", "p_ann", "e_col", "e_ann",
                    "exch_ca", "exch_ac", "wrad_inc", "wrad_ret"):
            for lab, m in region_masks(shared["z_cm"]):
                if not m.any():
                    continue
                w = (shared["V_col"] if key.endswith("col")
                     else shared["V_ann"] if key.endswith("ann") else None)
                if w is not None and w[m].sum() <= 0.0:
                    continue
                a = agg(dvm_A[key], m, w)
                b = agg(dvm_h[key], m, w)
                r = np.abs(rel_dev(a, b)) * 100.0
                dt_band[key] = max(dt_band.get(key, 0.0), float(np.nanmax(r)))
                lines.append(
                    f"{key:<12s} {lab:<22s} {np.nanmax(r):20.3f} "
                    f"{int(np.nanargmax(r)):7d} {r[-1]:11.3f}"
                )
        for j, end in enumerate(("end L", "end R")):
            for key in ("wend_inc", "wend_ret"):
                r = np.abs(rel_dev(dvm_A[key][:, j], dvm_h[key][:, j])) * 100.0
                dt_band[key] = max(dt_band.get(key, 0.0), float(np.nanmax(r)))
                lines.append(
                    f"{key:<12s} {end:<22s} {np.nanmax(r):20.3f} "
                    f"{int(np.nanargmax(r)):7d} {r[-1]:11.3f}"
                )
        lines.append(
            "The COLUMN rows relax on the local ion-neutral loss time "
            "1/(nu_ion + nu_cx + nu_el), which this background puts at a few "
            "times 1e-5 s -- the same order as the shipped neutral cadence. "
            "Their dt band is therefore large and a DVM-vs-reference "
            "deviation smaller than it is not resolvable at this cadence. "
            "The annulus has no volume loss channel and is well converged."
        )
        dt_note = "\n".join(lines)
        notes.append(dt_note)

    sampling_note = sampling_check(dvm_obj, shared, args)
    notes.append(sampling_note)
    notes.append(ledger_note(dvm_A["_ledger"], summary["kin"][1], args))

    notes.append(
        "\n".join([
            "SHARED CONVENTIONS NOT TESTED HERE (stated, not hidden)",
            "-" * 100,
            "1. The anode-mesh interception and the anode external source both "
            "re-emit at the engine's CYLINDRICAL wall-emission spectrum on "
            "what is geometrically a z-normal plane. The MC adopts the same "
            "convention so this comparison stays about transport; the "
            "convention itself is therefore NOT adjudicated. It acts only in "
            f"the cells adjacent to mesh face {shared['mesh_face']}.",
            "2. The end-pump sticking coefficients use each end's OWN "
            "end-plane area (A_end_L = pi Rm[0]^2, A_end_R = pi Rm[-1]^2), "
            "the shipped instrument convention since 2026-08-26, on both "
            "sides. Before that date both ends took pi Rm[-1]^2 here and in "
            "KN2Zone/TPMC, so numbers from earlier runs of this comparison "
            "are not directly comparable to these.",
            "3. The puff is born as a 300 K zero-momentum volume Maxwellian in "
            "the annulus on both sides (the engine's registered channel 5).",
        ])
    )

    print("writing the comparison tables ...", flush=True)
    write_compare(cmp_path, args, shared, bg, blocks, notes)
    print(f"  wrote {cmp_path}", flush=True)

    # ---------------- CX-channel adjudication
    print("CX-channel adjudication ...", flush=True)
    cx_table, cx_extra = cx_adjudication(shared, args, dvm_obj)
    write_cx(cx_path, args, shared, cx_table, cx_extra)
    print(f"  wrote {cx_path}", flush=True)

    write_summary(sum_path, args, shared, bg, summary, cx_table, dt_note,
                  dt_band, dvm_A_wall, time.perf_counter() - t_all,
                  sampling_note=sampling_note)
    print(f"  wrote {sum_path}", flush=True)
    print(f"total wall {time.perf_counter() - t_all:.1f} s", flush=True)
    return 0


def cx_adjudication(shared, args, dvm):
    """Measure the CX/elastic transfer channel three ways on matched states."""
    pl = shared["plasma"]
    Ti = pl["Ti_eV"]
    u_i = pl["u_i"]
    z = shared["z_cm"]
    # states spanning the background's Ti range, at the cells that carry them
    order = np.argsort(Ti)
    picks = order[np.linspace(0, order.size - 1, 8).astype(int)]
    rng = np.random.default_rng(args.seed + 7)
    Tn_wall = KB * T_WALL_K / EV
    table = []
    for i in picks:
        for tn_label, Tn in (("Tn=300K", Tn_wall), ("Tn=Ti", float(Ti[i]))):
            for un_label, u_n in (("un=0", 0.0),):
                label = (
                    f"z={z[i]:6.0f} Ti={Ti[i]:5.2f} "
                    f"u={u_i[i] / 1e5:6.1f}km/s {tn_label}"
                )
                P_d, E_d, Pcx_d, Pel_d, kcx_d, kel_d = dvm_channel_moments(
                    dvm, float(Ti[i]), float(u_i[i]), Tn, u_n
                )
                mc = mc_channel_moments(
                    rng, args.cx_samples, float(Ti[i]), float(u_i[i]), Tn,
                    u_n, elastic=args.elastic_model != "off",
                )
                P_f, Ei_f, k_mt = fluid_channel_moments(
                    float(Ti[i]), float(u_i[i]), Tn, u_n
                )
                # the engine books internal energy as C_E - u_i C_P
                Ei_d = E_d - float(u_i[i]) * P_d
                Ei_mc = mc["C_E"] - float(u_i[i]) * mc["C_P"]
                Ei_mc_sem = float(
                    np.hypot(mc["C_E_sem"], float(u_i[i]) * mc["C_P_sem"])
                )
                table.append({
                    "label": label,
                    "P_dvm": P_d, "P_mc": mc["C_P"], "P_mc_sem": mc["C_P_sem"],
                    "P_fluid": P_f,
                    "E_dvm": Ei_d, "E_mc": Ei_mc, "E_mc_sem": Ei_mc_sem,
                    "E_fluid": Ei_f,
                    "Etot_dvm": E_d, "Etot_mc": mc["C_E"],
                    "Etot_mc_sem": mc["C_E_sem"],
                    "kcx_dvm": kcx_d, "kcx_mc": mc["k_cx"],
                    "kcx_mc_sem": mc["k_cx_sem"],
                    "kel_dvm": kel_d,
                    "kel_mc": ELASTIC_BGK_MOMENTUM_FACTOR * mc["k_iso"],
                    "k_mt_fluid": k_mt,
                })
    extra = [
        "",
        f"MC samples per state: {args.cx_samples:,} matched pairs, base seed "
        f"{args.seed + 7}. States are the background's own cells, chosen at "
        f"eight equally spaced ranks of T_i (range "
        f"{Ti.min():.3g} to {Ti.max():.3g} eV), each evaluated with a neutral "
        f"Maxwellian at the 300 K wall temperature (the fluid operator's own "
        f"assumption) and at T_n = T_i (the CX-relayed population).",
        "",
        "Reference for every row: the true-kinematics MC. The 'fluid' column "
        "is the shipped moment operator and is quoted, not used as truth.",
    ]
    return table, extra


def write_summary(path, args, shared, bg, summary, cx_table, dt_note, dt_band,
                  dvm_wall, total_wall, sampling_note=None):
    keys = ("n_col", "n_ann", "p_col", "p_ann", "e_col", "e_ann",
            "exch_ca", "exch_ac", "wrad_inc", "wrad_ret", "wend_inc",
            "wend_ret")
    L = []
    L.append("# E2 -- matched-time physics-reference comparison (numbers only)")
    L.append("")
    L.append(
        f"Background `{Path(args.run).name}`, plateau window "
        f"{args.window[0]}-{args.window[1]} ms, nz={shared['nz']}, velocity "
        f"grid {args.nvz}x{args.nvp}, DVM neutral clock "
        f"dt={args.dvm_dt:.4g} s. Both arms start from the same initial "
        f"state: "
        + (
            "the background's own saved two-zone neutral profile "
            f"({float((shared['seed_col'] * shared['V_col']).sum()):.4g} "
            f"column + "
            f"{float((shared['seed_ann'] * shared['V_ann']).sum()):.4g} "
            "annulus atoms), laid down as a 300 K Maxwellian at rest"
            if float(np.sum(shared["seed_col"])) > 0.0
            else "an empty vessel"
        )
        + f". Schedule: sources ON over 0-{1e3 * shared['t_switch']:.2f} ms "
        f"(transient), OFF over "
        f"{1e3 * shared['t_switch']:.2f}-{1e3 * shared['t_end']:.2f} ms "
        f"(relaxation); {shared['nbin']} report bins of "
        f"{1e3 * shared['bin_s']:.2f} ms. Reference: transient full-particle "
        f"MC, {args.batches} independent batches of {args.particles:,} "
        f"histories, base seed {args.seed}."
    )
    L.append("")
    L.append(
        f"Machine: {platform.platform()}, {os.cpu_count()} logical cores; "
        f"Python {platform.python_version()}, numpy {np.__version__}. "
        f"DVM arm {dvm_wall:.1f} s; whole script {total_wall:.1f} s."
    )
    L.append("")
    L.append("Full command (reruns end to end):")
    L.append("")
    L.append("```")
    L.append(args.cmdline)
    L.append("```")
    L.append("")
    L.append("## Reference policy")
    L.append("")
    L.append(
        "The shipped `KN2Zone` two-zone steady march is non-conservative in "
        "the annulus channel at this geometry's area-jump faces (K2a review), "
        "so it is used as truth nowhere here. Every row's reference is the "
        "transient full-particle MC named in that row. The time-dependent "
        "single-zone `KN2ZoneJump` variant is not exercised: every E2 row "
        "except `nn_col` is a two-zone quantity a uniform-area single-zone "
        "reference cannot produce."
    )
    L.append("")
    for tag, title in (
        ("kin", "true-kinematics TPMC"),
        ("bgk", "TPMC carrying the DVM's own BGK collision operator"),
        ("split", "true-kinematics TPMC, partial accommodation"),
    ):
        if tag not in summary:
            continue
        rows, meta = summary[tag]
        L.append(f"## Worst matched-time deviation vs {title}")
        L.append("")
        L.append(
            "| quantity | worst dev/sigma (region, bin) | worst dev % "
            "(region, bin) | DVM dt band % | resolvable? |"
        )
        L.append("|---|---|---|---|---|")
        w = worst_rows(rows, keys)
        for key in keys:
            (s, sw), (p, pw) = w[key]
            sl = f"{s:+.1f} ({sw[0]}, bin {sw[1]})" if sw else "n/a"
            pl_ = f"{100 * p:+.2f}% ({pw[0]}, bin {pw[1]})" if pw else "n/a"
            band = dt_band.get(key)
            bl = f"{band:.1f}%" if band is not None else "n/a"
            if band is None or pw is None:
                verdict = "n/a"
            elif abs(100 * p) > band:
                verdict = "yes"
            else:
                verdict = "NO -- inside the dt band"
            L.append(f"| `{key}` | {sl} | {pl_} | {bl} | {verdict} |")
        L.append("")
        L.append(
            "`DVM dt band %` is the worst change in that quantity between the "
            "shipped neutral cadence and half of it (the block below). A "
            "deviation smaller than the band is not resolvable at this "
            "cadence and is marked so; it is a statement about the cadence, "
            "not evidence that the two arms agree."
        )
        L.append("")
        seg = sum(m["segments"] for m in meta)
        viol = sum(m["violations"] for m in meta)
        lau = sum(m["launched_atoms"] for m in meta)
        ion = sum(m["lost_ion"] for m in meta)
        pmp = sum(m["lost_pump"] for m in meta)
        res = sum(m["resident"] for m in meta)
        stk = sum(m["stuck"] for m in meta)
        L.append(
            f"MC integrity: {seg / 1e6:.1f}e6 flight segments over "
            f"{len(meta)} batches; null-collision majorant violations "
            f"**{viol}** (any nonzero count invalidates the sampling). "
            f"Particle ledger over the window, in atoms: launched "
            f"{lau:.5g} = ionized {ion:.5g} + pumped {pmp:.5g} + resident at "
            f"t_end {res:.5g} + iteration-capped {stk:.5g}; closure ratio "
            f"{(ion + pmp + res + stk) / max(lau, 1e-300):.10f}."
        )
        L.append("")
    L.append("## CX / elastic channel adjudication")
    L.append("")
    L.append(
        "Per unit `n_i n_n`, on the background's own states. `C_P` is the "
        "axial momentum the plasma gains, `C_Ei` its internal-energy gain "
        "with the engine's own bulk decomposition. Reference is the "
        "true-kinematics MC; the fluid column is the shipped Phelps moment "
        "operator, quoted and not used as truth."
    )
    L.append("")
    L.append(
        "| state | C_P DVM/MC | C_P fluid/MC | C_Ei DVM/MC | C_Ei fluid/MC | "
        "k_cx DVM/MC |"
    )
    L.append("|---|---|---|---|---|---|")
    for r in cx_table:
        def rat(a, b):
            return f"{a / b:.4f}" if b else "n/a"
        L.append(
            f"| {r['label']} | {rat(r['P_dvm'], r['P_mc'])} | "
            f"{rat(r['P_fluid'], r['P_mc'])} | {rat(r['E_dvm'], r['E_mc'])} | "
            f"{rat(r['E_fluid'], r['E_mc'])} | "
            f"{rat(r['kcx_dvm'], r['kcx_mc'])} |"
        )
    L.append("")
    pr = [r["P_dvm"] / r["P_mc"] for r in cx_table if r["P_mc"]]
    fr = [r["P_fluid"] / r["P_mc"] for r in cx_table if r["P_mc"]]
    kr = [r["kcx_dvm"] / r["kcx_mc"] for r in cx_table if r["kcx_mc"]]
    dfr = [r["P_dvm"] / r["P_fluid"] for r in cx_table if r["P_fluid"]]
    cold = [r for r in cx_table if "Tn=300K" in r["label"]]
    er = [r["Etot_dvm"] / r["Etot_mc"] for r in cold if r["Etot_mc"]]
    eir = [r["E_dvm"] / r["E_mc"] for r in cold if r["E_mc"]]
    if pr:
        L.append(
            f"Range over the states measured: `C_P` DVM/MC "
            f"{min(pr):.4f} to {max(pr):.4f}; fluid/MC {min(fr):.4f} to "
            f"{max(fr):.4f}; DVM/fluid {min(dfr):.4f} to {max(dfr):.4f}; "
            f"`k_cx` DVM/MC {min(kr):.4f} to {max(kr):.4f}."
        )
        L.append("")
        L.append(
            f"Energy ranges are quoted over the `Tn=300K` states only: "
            f"`C_Ei` DVM/MC {min(eir):.4f} to {max(eir):.4f}, total "
            f"kinetic-energy moment DVM/MC {min(er):.4f} to {max(er):.4f}. "
            f"At `Tn=Ti` the neutral and ion populations are at the same "
            f"temperature and the drift is small, so both energy moments are "
            f"a near cancellation of much larger terms; their ratios in the "
            f"table above are ratios of residuals, not fractional errors, and "
            f"the momentum channel is the readable statement there."
        )
    L.append("")
    if sampling_note:
        L.append("## Reference sampling check")
        L.append("")
        L.append("```")
        L.append(sampling_note)
        L.append("```")
        L.append("")
    if dt_note:
        L.append("## DVM self-convergence in the neutral clock")
        L.append("")
        L.append("```")
        L.append(dt_note)
        L.append("```")
        L.append("")
    L.append("## Conventions shared by both arms (not adjudicated here)")
    L.append("")
    L.append(
        "1. Anode-mesh interception and the anode source both re-emit at the "
        "engine's cylindrical wall-emission spectrum on a geometrically "
        "z-normal plane; the MC adopts the same convention, so this "
        f"comparison does not test it. It acts only near mesh face "
        f"{shared['mesh_face']}."
    )
    L.append(
        "2. End-pump sticking uses each end's OWN end-plane area "
        "(`A_end_L = pi Rm[0]^2`, `A_end_R = pi Rm[-1]^2`) on both sides -- "
        "the shipped instrument convention since 2026-08-26. Before that "
        "date both ends took `pi Rm[-1]^2` here and in KN2Zone/TPMC, so "
        "numbers from earlier runs of this comparison are not directly "
        "comparable to these."
    )
    L.append(
        "3. The puff is born as a 300 K zero-momentum volume Maxwellian in the "
        "annulus on both sides (the engine's registered channel 5)."
    )
    L.append(
        "4. The annular STEP face where `Rm` narrows out of the end expansion "
        "exists only in the reference: the DVM's throat-face flux form "
        "throttles the aperture instead of terminating rays on a surface, so "
        "its `wstep_*` rows are identically zero. The reference's values are "
        "reported as a reference-only measurement."
    )
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
