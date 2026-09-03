"""Scoping: can ppm-level C/O impurity radiation supply the missing Te sink?

Tests hypothesis (ii) of the Te-excess scoping against real ADAS
data before any model term is written: the stage-(ii) Te excess (~2x at
ES1, worsening down the fixed-fueling ladder while n transfers at ~1.0)
has the shape of a missing n^2-scaling radiative sink, and stainless
outgassing suggests a fixed C/O fraction n_z = f_imp * n_e radiating
n_e * n_z * L_z(Te).

Method (no model code): read a saved reference run, integrate its
volumetric electron-energy budget over the plateau window to get the
required sink (~50 % of gross electron heating for a 2x Te excess), build
L_z(ne, Te) for C and O from the OPEN-ADAS adf11 '96 files in vars/adas
(equilibrium stage balance from SCD/ACD, radiated power from PLT+PRB),
and solve for the f_imp that closes the gap.  A best-case bound uses the
brightest single-stage PLT in the window (~1e-25 W cm^3) in place of the
equilibrium mix, i.e. maximal non-equilibrium enhancement.

Usage:
    python scripts/scope_impurity_radiation.py [run.h5] [Te_excess_ratio] [elements]

``elements`` is a comma list from {o, c, b, w} (default "o,c").  Boron and
tungsten use the '89 Abels-van Maanen series (no '96 exists; older
average-ion-era data, order-of-magnitude at these Te).  Tungsten stands in
as the heavy-element analog for lanthanum, which has no OPEN-ADAS adf11
data in any series.

Verdict on record (2026-07-21, es1_nx120_m5ap_es1.h5): required
f_imp ~ 10 % (O or C, equilibrium L_z at model Te), >= 3-7 % even at the
non-equilibrium ceiling -- the >= 1 % plausibility gate fires and the
campaign stops before implementation.  The original ppm arithmetic assumed
L_z(O) ~ 5e-25 W cm^3 at 5-15 eV; the ADAS GCR equilibrium value there is
1e-26 - 5e-26 W cm^3 (the 5e-25 class is only reached near the O radiation
peak at ~20-40 eV), a factor ~20-50 that the ppm estimate inherited.
"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cablp.atomic.adas import ADAS_DIR, _interp_blend, _interp_coords, read_adf11

PLATEAU_MS = (5.0, 19.5)

# adf11 series year per element: '96 GCR where it exists, '89 otherwise.
SERIES = {
    "he": "96", "c": "96", "o": "96",
    "b": "89", "w": "89", "mo": "89",
    # Stainless species ('89 Abels-van Maanen, added 2026-07-21): the last
    # untested radiators for the stage-(ii) fixed-fraction hypothesis.
    "fe": "89", "cr": "89", "ni": "89",
}


def _load_element(el):
    """Return (log_ne, log_te, tables[cls][z1]) for one element's adf11 set."""
    tables = {}
    axes = None
    for cls in ("scd", "acd", "plt", "prb"):
        log_ne, log_te, stages = read_adf11(ADAS_DIR / f"{cls}{SERIES[el]}_{el}.dat")
        if axes is None:
            axes = (log_ne, log_te)
        elif not (
            np.array_equal(log_ne, axes[0]) and np.array_equal(log_te, axes[1])
        ):
            raise ValueError(f"{cls}96_{el}.dat: grid differs within element set")
        tables[cls] = stages
    return axes[0], axes[1], tables


def lz_equilibrium_Wcm3(el, ne_cm3, Te_eV):
    """Equilibrium radiated-power coefficient L_z [W cm^3] and stage fractions.

    Stage balance at the local (ne, Te): f_z1/f_z1-1 = SCD(z1)/ACD(z1)
    (adf11 convention: SCD(z1) ionizes charge z1-1 -> z1, ACD(z1) recombines
    z1 -> z1-1).  Radiated power per impurity nucleus and electron:
    sum_z1 [ f_{z1-1} * PLT(z1) + f_{z1} * PRB(z1) ].
    """
    log_ne, log_te, tab = _load_element(el)
    Z = max(tab["scd"])
    ne = np.asarray(ne_cm3, dtype=float)
    Te = np.asarray(Te_eV, dtype=float)
    ix, iy, fx, fy = _interp_coords(log_ne, log_te, np.log10(ne), np.log10(Te))

    def ev(cls, z1):
        return 10.0 ** _interp_blend(tab[cls][z1], ix, iy, fx, fy)

    log_f = [np.zeros(np.broadcast(ne, Te).shape)]
    for z1 in range(1, Z + 1):
        log_f.append(log_f[-1] + np.log10(ev("scd", z1)) - np.log10(ev("acd", z1)))
    log_f = np.stack(log_f)
    log_f -= log_f.max(axis=0)
    f = 10.0**log_f
    f /= f.sum(axis=0)

    lz = np.zeros_like(f[0])
    for z1 in range(1, Z + 1):
        lz += f[z1 - 1] * ev("plt", z1) + f[z1] * ev("prb", z1)
    return lz, f


def lz_brightest_stage_Wcm3(el, ne_cm3, Te_eV):
    """Best-case bound: the largest single-stage PLT at each (ne, Te).

    Upper-bounds any non-equilibrium (ionizing / recycling) enhancement:
    no charge-state mix radiates more than every nucleus held in the
    brightest stage.
    """
    log_ne, log_te, tab = _load_element(el)
    ne = np.asarray(ne_cm3, dtype=float)
    Te = np.asarray(Te_eV, dtype=float)
    ix, iy, fx, fy = _interp_coords(log_ne, log_te, np.log10(ne), np.log10(Te))
    stacked = np.stack(
        [
            10.0 ** _interp_blend(tab["plt"][z1], ix, iy, fx, fy)
            for z1 in sorted(tab["plt"])
        ]
    )
    return stacked.max(axis=0)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "es1_nx120_m5ap_es1.h5"
    te_excess = float(sys.argv[2]) if len(sys.argv) > 2 else 2.02
    elements = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("o", "c")
    path = Path(__file__).resolve().parents[1] / run

    with h5py.File(path, "r") as fh:
        t = fh["time"][:]
        phase = fh["phase"][:].astype(str)
        n = fh["n"][:]
        Te = fh["Te"][:]
        vol = fh["geometry/plasma_volume_cm3"][:]
        terms = {
            k: fh[f"electron_energy_terms_W_cm3/{k}"][:]
            for k in fh["electron_energy_terms_W_cm3"]
        }

    origin = t[np.flatnonzero(phase == "main_discharge")[0]]
    tms = (t - origin) * 1.0e3
    win = (tms >= PLATEAU_MS[0]) & (tms <= PLATEAU_MS[1])
    print(
        f"{run}: {win.sum()} saves in the {PLATEAU_MS[0]}-{PLATEAU_MS[1]} ms "
        f"plateau window, column volume {vol.sum():.3g} cm^3"
    )

    print("\n--- electron energy budget, plateau average [kW] ---")
    budget = {}
    for name, arr in sorted(terms.items()):
        p_kw = (arr[win] * vol).sum(axis=1).mean() / 1.0e3
        budget[name] = p_kw
        if abs(p_kw) > 0.5:
            print(f"  {name:35s} {p_kw:+9.1f}")
    gross_heat = sum(p for p in budget.values() if p > 0)
    print(f"  {'gross heating':35s} {gross_heat:+9.1f}")
    print(f"  {'total sinks':35s} {sum(p for p in budget.values() if p < 0):+9.1f}")

    p_req_kw = 0.5 * gross_heat
    print(
        f"\nrequired sink for the ~{te_excess}x Te excess "
        f"(~50 % of gross heating): {p_req_kw:.0f} kW"
    )

    ne_w = np.maximum(n[win], 1.0e6)
    n2_dV = (ne_w**2 * vol).sum(axis=1).mean()
    print(f"time-averaged integral n_e^2 dV = {n2_dV:.3g} cm^-3")

    print("\n--- required f_imp = n_z/n_e (>= 1 % fails plausibility) ---")
    for el in elements:
        for label, scale in (("model Te", 1.0), (f"Te/{te_excess}", 1.0 / te_excess)):
            te_w = np.maximum(Te[win] * scale, 0.2)
            lz, _ = lz_equilibrium_Wcm3(el, ne_w, te_w)
            watts_per_f = (ne_w**2 * lz * vol).sum(axis=1).mean()
            f_req = p_req_kw * 1.0e3 / watts_per_f
            print(
                f"  {el.upper()} [{label:9s}] sum n^2*Lz*dV = "
                f"{watts_per_f:.3g} W per unit f_imp -> "
                f"f_imp = {f_req:.2e} ({f_req * 1.0e6:.0f} ppm)"
            )
        lz_max = lz_brightest_stage_Wcm3(el, ne_w, np.maximum(Te[win], 0.2))
        watts_per_f = (ne_w**2 * lz_max * vol).sum(axis=1).mean()
        f_req = p_req_kw * 1.0e3 / watts_per_f
        print(
            f"  {el.upper()} [brightest-stage ceiling] -> "
            f"f_imp >= {f_req:.2e} ({f_req * 1.0e6:.0f} ppm)"
        )


if __name__ == "__main__":
    main()
