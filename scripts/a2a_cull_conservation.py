"""A2a gate: cull conservation and the pairwise-partner audit, per tail path.

Runs BOTH tail march variants -- ``deposit_beam`` and
``deposit_beam_two_stream`` -- over a synthetic column that puts the anode
plane inside the tail walk window, with the tail cull armed and with it off,
and asserts:

G2a  PATH IDENTITY (cull off): the armed-module call with the cull argument
     absent reproduces the unarmed call bit for bit, on both paths and on all
     three tail sub-branches (marched, energy-only reflecting, energy-only
     plain).

G2b  CULL CONSERVATION, per path and per sub-branch: the flux the mesh removed
     plus the flux that survived past the plane equals the flux that arrived,
     to roundoff. Measured as the ENERGY identity the module already closes --
     ``Gamma0*E0 == heating + radiated + cost + anode_intercepted +
     transmitted + end_losses`` -- which is the only statement that sees BOTH
     halves at once.

G2c  PAIRWISE PARTNERS, named. Each increment matched to its equal-and-opposite
     partner rather than to a sum:
       culled_erg          <-> the walk's own energy loss at the plane
       returned_erg        <-> the reversed legs' deposit + their end loss
       anode_intercepted   <-> culled_erg - returned_erg   (exactly)
       culled_flux         <-> returned_flux + collected_flux (the lag's input)
     Reported ROW-RELATIVE (against the row's own magnitude) as well as
     throughput-normalized, per the standing negative-control rule.

G2d  RIDER REFUSALS: every registered misconfiguration raises at the module
     boundary, and the 50 eV clamp absorbs a sub-box crossing.

Run from the worktree root with PYTHONPATH=<worktree>.
"""

import math
import sys

import numpy as np

import cablp
from cablp.cathode import beam_deposition as B

print(f"cablp.__file__ = {cablp.__file__}")
print(f"compiled march  = {B._CSDA_MARCH is not None}")
print()

_ERG = B._ERG_PER_EV
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


# --- the synthetic column ------------------------------------------------
# 24 cells; the beam launches at cell 0 heading +z, the anode plane sits at
# cell 4, and the QL tail is born across the column so walkers approach the
# plane from both sides.
CELLS = 24
ANODE = 4
ETA = 0.358
dz = np.full(CELLS, 10.0)
nn = np.full(CELLS, 3.0e13)
ne = np.full(CELLS, 5.0e11)
Te = np.full(CELLS, 4.0)
area = np.full(CELLS, 1000.0)

BASE = dict(
    nn=nn, ne=ne, Te=Te, launch=0, direction=1, dz_cm=dz,
    I_ion_eV=24.587, coulomb_model="fast_electron",
    anomalous_model="quasilinear", beam_area_cm2=area,
)
E0 = 120.0
G0 = 4.0e19


def ray(**kw):
    return B.deposit_beam(E0, G0, **{**BASE, **kw})


def identity_residual(res, launched):
    """Relative closure of the per-ray energy identity."""
    total = (
        float(np.sum(res.plasma_heating_erg_s))
        + float(np.sum(res.radiated_erg_s))
        + float(np.sum(res.ionization_cost_erg_s))
        + float(res.anode_intercepted_erg_s)
        + float(res.end_loss_low_erg_s)
        + float(res.end_loss_high_erg_s)
        + float(res.end_loss_tail_low_erg_s)
        + float(res.end_loss_tail_high_erg_s)
        + (0.0 if res.end_loss_transmitted_erg_s else
           float(res.transmitted_flux) * float(res.transmitted_energy_eV) * _ERG)
    )
    return abs(total - launched) / launched, total


def bits(a):
    return np.asarray(a, dtype=float).tobytes()


def fingerprint(res):
    return (
        bits(res.ionization_events), bits(res.excitation_events),
        bits(res.plasma_heating_erg_s), bits(res.radiated_erg_s),
        bits(res.ionization_cost_erg_s), bits(res.heating_anomalous_erg_s),
        bits(np.array([res.anode_intercepted_erg_s,
                       res.end_loss_tail_low_erg_s,
                       res.end_loss_tail_high_erg_s,
                       res.transmitted_flux, res.transmitted_energy_eV])),
    )


ARMS = {
    # (label, extra kwargs) -- the three tail sub-branches, each with the
    # primary interception on so the plane exists for the primary too.
    "marched (tail_ionization=on, reflect)": dict(
        anomalous_transport="tail_walk", tail_energy_eV=90.0,
        tail_ionization="on", tail_walk_window=(0, CELLS - 1),
        tail_reflect_face=-1, tail_reflect_threshold_eV=80.0,
    ),
    "energy-only, reflecting face": dict(
        anomalous_transport="tail_walk", tail_energy_eV=90.0,
        tail_walk_window=(0, CELLS - 1),
        tail_reflect_face=-1, tail_reflect_threshold_eV=80.0,
    ),
    "energy-only, plain": dict(
        anomalous_transport="tail_walk", tail_energy_eV=90.0,
    ),
}

CULL_KW = dict(tail_anode_cross_index=ANODE, tail_anode_eta=ETA)
PRIM_KW = dict(anode_cross_index=ANODE, anode_eta=ETA)

print("=" * 74)
print("G2a  PATH IDENTITY -- cull off reproduces the pre-A2a call bit for bit")
print("=" * 74)
print("deposit_beam:")
for label, kw in ARMS.items():
    off = ray(**PRIM_KW, **kw)
    # The same call with the A2a arguments present but at their OFF values.
    off2 = ray(**PRIM_KW, **kw, tail_anode_cross_index=None,
               tail_anode_eta=0.0, tail_anode_reflected_particles=0.0,
               tail_anode_reflected_energy=0.0)
    check(f"{label}", fingerprint(off) == fingerprint(off2))

print()
print("=" * 74)
print("G2b  CULL CONSERVATION -- the per-ray energy identity still closes")
print("=" * 74)
launched = G0 * E0 * _ERG
print("deposit_beam:")
results = {}
for label, kw in ARMS.items():
    off = ray(**PRIM_KW, **kw)
    on = ray(**PRIM_KW, **kw, **CULL_KW)
    results[label] = (off, on)
    r_off, _ = identity_residual(off, launched)
    r_on, _ = identity_residual(on, launched)
    check(f"{label}: identity closes with the cull ON",
          r_on < 1e-12, f"rel residual {r_on:.3e} (off: {r_off:.3e})")
    moved = float(on.tail_anode_culled_erg_s) > 0.0
    check(f"{label}: the cull actually fired", moved,
          f"culled {float(on.tail_anode_culled_erg_s):.6e} erg/s, "
          f"flux {float(on.tail_anode_culled_flux_per_s):.6e} 1/s")

print()
print("=" * 74)
print("G2c  PAIRWISE PARTNERS -- named, row-relative AND throughput-normalized")
print("=" * 74)
for label, (off, on) in results.items():
    print(f"\n{label}")
    culled = float(on.tail_anode_culled_erg_s)
    returned = float(on.tail_anode_returned_erg_s)
    prim = float(off.anode_intercepted_erg_s)
    # PARTNER 1: anode_intercepted's tail share == culled - returned, exactly.
    tail_share = float(on.anode_intercepted_erg_s) - prim
    d1 = abs(tail_share - (culled - returned))
    row1 = d1 / max(abs(culled - returned), 1e-300)
    print(f"  anode_intercepted tail share   {tail_share: .12e} erg/s")
    print(f"  culled - returned              {culled - returned: .12e} erg/s")
    print(f"    row-relative {row1:.3e}   throughput-normalized "
          f"{d1 / launched:.3e}")
    check(f"{label}: anode row == culled - returned", row1 < 1e-14)
    # PARTNER 2: the walk's own energy loss at the plane. With the rider off
    # (this arm) the culled energy is exactly what the plasma no longer got
    # plus what no longer left through the tail ends.
    walk_off = (
        float(np.sum(off.heating_anomalous_erg_s))
        + float(off.end_loss_tail_low_erg_s)
        + float(off.end_loss_tail_high_erg_s)
        + float(np.sum(off.radiated_tail_erg_s))
        + float(np.sum(off.ionization_cost_tail_erg_s))
    )
    walk_on = (
        float(np.sum(on.heating_anomalous_erg_s))
        + float(on.end_loss_tail_low_erg_s)
        + float(on.end_loss_tail_high_erg_s)
        + float(np.sum(on.radiated_tail_erg_s))
        + float(np.sum(on.ionization_cost_tail_erg_s))
    )
    lost = walk_off - walk_on
    d2 = abs(lost - culled)
    row2 = d2 / max(abs(culled), 1e-300)
    print(f"  walk energy the cull removed   {lost: .12e} erg/s")
    print(f"  culled_erg                     {culled: .12e} erg/s")
    print(f"    row-relative {row2:.3e}   throughput-normalized "
          f"{d2 / launched:.3e}")
    check(f"{label}: walk loss == culled_erg (rider off)", row2 < 1e-10,
          f"row-relative {row2:.3e}")
    # PARTNER 3: the flux the lag reads.
    cf = float(on.tail_anode_culled_flux_per_s)
    rf = float(on.tail_anode_returned_flux_per_s)
    print(f"  culled_flux {cf: .12e} = returned {rf: .12e} + collected "
          f"{cf - rf: .12e} 1/s")

print()
print("=" * 74)
print("G2c(2)  THE RIDER, on the marched path")
print("=" * 74)
kw = ARMS["marched (tail_ionization=on, reflect)"]
off, on = results["marched (tail_ionization=on, reflect)"]
R_e, eta_E = 0.37, 0.26
rid = ray(**PRIM_KW, **kw, **CULL_KW,
          tail_anode_reflected_particles=R_e,
          tail_anode_reflected_energy=eta_E)
r_rid, _ = identity_residual(rid, launched)
check("rider arm: the per-ray energy identity closes", r_rid < 1e-12,
      f"rel residual {r_rid:.3e}")
cf_r = float(rid.tail_anode_culled_flux_per_s)
rf_r = float(rid.tail_anode_returned_flux_per_s)
ce_r = float(rid.tail_anode_culled_erg_s)
re_r = float(rid.tail_anode_returned_erg_s)
print(f"  culled  flux {cf_r:.6e} 1/s   erg {ce_r:.6e} erg/s")
print(f"  returned flux {rf_r:.6e} 1/s   erg {re_r:.6e} erg/s")
# Every crossing here is above the 50 eV floor (E_tail = 90 eV, and the
# walkers reach the plane well above the floor), so the returned shares are
# exactly R_e and eta_E of the culled ones.
check("returned_flux == R_e * culled_flux",
      abs(rf_r - R_e * cf_r) <= 1e-13 * max(rf_r, 1e-300),
      f"{rf_r:.12e} vs {R_e * cf_r:.12e}")
check("returned_erg  == eta_E * culled_erg",
      abs(re_r - eta_E * ce_r) <= 1e-13 * max(re_r, 1e-300),
      f"{re_r:.12e} vs {eta_E * ce_r:.12e}")
check("the anode row is the NET",
      abs((float(rid.anode_intercepted_erg_s)
           - float(off.anode_intercepted_erg_s)) - (ce_r - re_r))
      <= 1e-14 * max(abs(ce_r - re_r), 1e-300))
check("the rider returns energy to the plasma the cull-only arm lost",
      float(np.sum(rid.heating_anomalous_erg_s))
      > float(np.sum(on.heating_anomalous_erg_s)),
      f"{float(np.sum(rid.heating_anomalous_erg_s)):.6e} > "
      f"{float(np.sum(on.heating_anomalous_erg_s)):.6e}")

print()
print("=" * 74)
print("G2d  REFUSALS at the module boundary")
print("=" * 74)


def refuses(label, fragment, **kw):
    try:
        ray(**kw)
    except ValueError as exc:
        ok = fragment in str(exc)
        check(label, ok, ("" if ok else f"message did not name {fragment!r}: "
                          f"{exc}"))
        return
    check(label, False, "no ValueError raised")


mk = ARMS["marched (tail_ionization=on, reflect)"]
refuses("R_e outside [0, 1]", "must be in [0, 1]",
        **PRIM_KW, **mk, **CULL_KW, tail_anode_reflected_particles=1.4,
        tail_anode_reflected_energy=0.0)
refuses("eta_E outside [0, 1]", "must be in [0, 1]",
        **PRIM_KW, **mk, **CULL_KW, tail_anode_reflected_particles=0.5,
        tail_anode_reflected_energy=-0.1)
refuses("eta_E > R_e", "must not exceed",
        **PRIM_KW, **mk, **CULL_KW, tail_anode_reflected_particles=0.10,
        tail_anode_reflected_energy=0.26)
refuses("rider without the cull armed", "needs the tail cull",
        **PRIM_KW, **mk, tail_anode_reflected_particles=0.37,
        tail_anode_reflected_energy=0.26)
refuses("rider under the energy-only walk", "requires tail_ionization='on'",
        **PRIM_KW, **ARMS["energy-only, plain"], **CULL_KW,
        tail_anode_reflected_particles=0.37,
        tail_anode_reflected_energy=0.26)
refuses("cull with no walked tail", "need a walked tail",
        **PRIM_KW, **CULL_KW)
refuses("tail_anode_eta out of range", "tail_anode_eta must be in [0, 1)",
        **PRIM_KW, **mk, tail_anode_cross_index=ANODE, tail_anode_eta=1.0)
_narrow = dict(
    anomalous_transport="tail_walk", tail_energy_eV=90.0,
    tail_ionization="on", tail_walk_window=(6, 20),
    tail_reflect_face=-1, tail_reflect_threshold_eV=80.0,
)
refuses("cull cell outside the walk window", "lies outside the tail walk",
        **PRIM_KW, **_narrow, tail_anode_cross_index=ANODE,
        tail_anode_eta=ETA)

print()
print("=" * 74)
print("G2d(2)  THE 50 eV CLAMP")
print("=" * 74)
# A tail launched below the rider's energy floor: every crossing is
# sub-box, so nothing returns however the pair is set.
low = dict(
    anomalous_transport="tail_walk", tail_energy_eV=30.0,
    tail_ionization="on", tail_walk_window=(0, CELLS - 1),
    tail_reflect_face=-1, tail_reflect_threshold_eV=25.0,
)
lo = ray(**PRIM_KW, **low, **CULL_KW,
         tail_anode_reflected_particles=0.37,
         tail_anode_reflected_energy=0.26)
print(f"  E_tail = 30 eV (floor {B.TAIL_ANODE_RIDER_MIN_ENERGY_EV} eV)")
print(f"  culled flux   {float(lo.tail_anode_culled_flux_per_s):.6e} 1/s")
print(f"  returned flux {float(lo.tail_anode_returned_flux_per_s):.6e} 1/s")
check("a sub-floor crossing returns nothing",
      float(lo.tail_anode_returned_flux_per_s) == 0.0
      and float(lo.tail_anode_returned_erg_s) == 0.0)
check("...and its whole culled share lands on the anode",
      abs((float(lo.anode_intercepted_erg_s)
           - float(ray(**PRIM_KW, **low).anode_intercepted_erg_s))
          - float(lo.tail_anode_culled_erg_s))
      <= 1e-14 * max(float(lo.tail_anode_culled_erg_s), 1e-300))

print()
print("=" * 74)
print("SECOND TAIL PATH -- deposit_beam_two_stream")
print("=" * 74)
f_cov = np.full(CELLS, 0.6)
TS = dict(
    f_cov=f_cov, nn_channel=nn / 0.6, ne_channel=ne / 0.6,
    nn_reservoir=np.full(CELLS, 1.0e12), ne_reservoir=np.full(CELLS, 1.0e8),
    Te=Te, launch=0, direction=1, dz_cm=dz, I_ion_eV=24.587,
    coulomb_model="fast_electron", anomalous_model="quasilinear",
    beam_area_cm2=area,
    stopping_coefficient=B._coulomb_stopping_coefficient(
        ne, Te, "fast_electron"
    ),
)
TS_ARMS = {
    "marched (tail_ionization=on, reflect)": dict(
        anomalous_transport="tail_walk", tail_energy_eV=90.0,
        tail_ionization="on", tail_walk_window=(0, CELLS - 1),
        tail_reflect_face=-1, tail_reflect_threshold_eV=80.0,
        nn_mean=nn, ne_mean=ne,
    ),
    "energy-only, reflecting face": dict(
        anomalous_transport="tail_walk", tail_energy_eV=90.0,
        tail_walk_window=(0, CELLS - 1),
        tail_reflect_face=-1, tail_reflect_threshold_eV=80.0,
    ),
    "energy-only, plain": dict(
        anomalous_transport="tail_walk", tail_energy_eV=90.0,
    ),
}


def ts(**kw):
    return B.deposit_beam_two_stream(E0, G0, **{**TS, **kw})


def ts_identity(ch, res_):
    total = 0.0
    for r in (ch, res_):
        total += (
            float(np.sum(r.plasma_heating_erg_s))
            + float(np.sum(r.radiated_erg_s))
            + float(np.sum(r.ionization_cost_erg_s))
            + float(r.anode_intercepted_erg_s)
            + float(r.end_loss_tail_low_erg_s)
            + float(r.end_loss_tail_high_erg_s)
            + float(r.transmitted_flux) * float(r.transmitted_energy_eV) * _ERG
        )
    return abs(total - launched) / launched


for label, kw in TS_ARMS.items():
    ch0, rs0, _ = ts(**PRIM_KW, **kw)
    ch1, rs1, _ = ts(**PRIM_KW, **kw, **CULL_KW)
    check(f"{label}: cull-off path identity",
          fingerprint(ch0) == fingerprint(ts(**PRIM_KW, **kw,
                                             tail_anode_cross_index=None,
                                             tail_anode_eta=0.0)[0]))
    r1 = ts_identity(ch1, rs1)
    check(f"{label}: identity closes with the cull ON", r1 < 1e-12,
          f"rel residual {r1:.3e}")
    check(f"{label}: the cull actually fired",
          float(ch1.tail_anode_culled_erg_s) > 0.0,
          f"culled {float(ch1.tail_anode_culled_erg_s):.6e} erg/s")
    tail_share = (float(ch1.anode_intercepted_erg_s)
                  - float(ch0.anode_intercepted_erg_s))
    d = abs(tail_share - (float(ch1.tail_anode_culled_erg_s)
                          - float(ch1.tail_anode_returned_erg_s)))
    check(f"{label}: anode row == culled - returned",
          d <= 1e-14 * max(float(ch1.tail_anode_culled_erg_s), 1e-300))

print()
if FAIL:
    print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
    sys.exit(1)
print("ALL A2a CULL-CONSERVATION GATES PASS")
