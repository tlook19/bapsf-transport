#!/usr/bin/env python
"""Acceptance tests for the standalone CSDA beam-deposition module (B1).

No solver in the loop. Checks, per closure:

1. Per-ray energy conservation to roundoff:
   Gamma0*E0 = heating + radiated + ionization cost + transmitted.
2. Breakdown conditions (nn = 3e14, ne = 1e10, 150 eV): inelastic events per
   primary ~ the hand estimate (phi_c / <dE per event> ~ 4), and the
   ionizations-per-primary multiple over the Beer-Lambert single-event
   booking (<= 1) — the ~3-5x undercount recorded as item 10.
3. Analytic Coulomb-only null (nn = 0, "fast_electron"): the numeric range
   matches the closed form R = (E0^2 - E_stop^2) / (4 pi e^4 ne lnL).
4. Analytic inelastic-only null (ne = 0): ionizations per primary match the
   quadrature of dN/dE = sigma_i / [sigma_i (I + <W>) + sigma_x E_rad].
5. Closure ordering at production conditions: l_QL << legacy l_bi << the
   classical fast-electron e-fold (~0.1 m / ~1 m / ~35 m) — the three-decade
   span recorded as item 12.

The historical Beer-Lambert profile comparison from the original B1
acceptance is deliberately replaced by nulls 3-4: no continuous slowing-down
model reproduces an exponential absorption profile (BL treats Coulomb drag
as absorption), so the license for the model swap is B3's solver-level null
on main-discharge metrics, not a profile match.
"""

from __future__ import annotations

import math

import numpy as np

from cablp.cathode.beam_deposition import (
    HE_E_STOP_EV,
    HE_I_ION_EV,
    beam_speed_cm_s,
    coulomb_stopping_eV_per_cm,
    deposit_beam,
    he_mean_secondary_energy_eV,
    quasilinear_relaxation_length_cm,
)
from cablp.cathode.circuit import _c_log_ei, _compute_l_b
from cablp.atomic.cross_sections import He_EII_cross_lkup, He_beam_excitation_channel

_ERG_PER_EV = 1.602176634e-12
_E4_CGS = (4.80320425e-10) ** 4

GAMMA0 = 1.0e22  # primaries/s (a few kA)
E0 = 150.0


def uniform_column(cells, dz, nn, ne, Te):
    ones = np.ones(cells)
    return dict(
        nn=nn * ones, ne=ne * ones, Te=Te * ones,
        launch=0, direction=1, dz_cm=dz * ones,
    )


def conservation_error(res):
    total = (
        res.plasma_heating_erg_s.sum()
        + res.radiated_erg_s.sum()
        + res.ionization_cost_erg_s.sum()
        + res.transmitted_flux * res.transmitted_energy_eV * _ERG_PER_EV
    )
    budget = GAMMA0 * E0 * _ERG_PER_EV
    return abs(total - budget) / budget


def main() -> None:
    # --- 1. conservation, across regimes and closures -----------------------
    regimes = {
        "breakdown": dict(cells=200, dz=10.0, nn=3.0e14, ne=1.0e10, Te=1.0),
        "main_discharge": dict(cells=200, dz=10.0, nn=2.0e13, ne=5.0e12, Te=8.0),
        "short_column": dict(cells=20, dz=5.0, nn=1.0e13, ne=1.0e12, Te=4.0),
    }
    print("=== 1. per-ray energy conservation ===")
    worst = 0.0
    for name, r in regimes.items():
        col = uniform_column(**r)
        for coulomb in ("fast_electron", "legacy_tau_ei"):
            for anom, extra in (
                ("none", {}),
                ("quasilinear", {"beam_area_cm2": 700.0}),
            ):
                res = deposit_beam(
                    E0, GAMMA0, coulomb_model=coulomb,
                    anomalous_model=anom, **extra, **col,
                )
                err = conservation_error(res)
                worst = max(worst, err)
                print(f"  {name:14s} {coulomb:14s} anom={anom:11s} "
                      f"rel err {err:.2e}  transmitted "
                      f"{res.transmitted_flux / GAMMA0:.3f} "
                      f"@ {res.transmitted_energy_eV:6.1f} eV")
                assert err < 1.0e-10, (name, coulomb, anom, err)
    print(f"  worst: {worst:.2e}  (bound 1e-10)  OK")

    # --- 2. breakdown ionizations per primary vs Beer-Lambert ---------------
    print("\n=== 2. breakdown-phase event count (item 10) ===")
    col = uniform_column(cells=400, dz=10.0, nn=3.0e14, ne=1.0e10, Te=1.0)
    res = deposit_beam(E0, GAMMA0, **col)
    ion_per_primary = res.ionization_events.sum() / GAMMA0
    exc_per_primary = res.excitation_events.sum() / GAMMA0
    events = ion_per_primary + exc_per_primary
    # Beer-Lambert books at most l_b/l_bn <= 1 ionization per primary.
    sigma_i0 = He_EII_cross_lkup(E0 / HE_I_ION_EV)
    sigma_x0, _ = He_beam_excitation_channel(E0)
    l_bn = 1.0 / (3.0e14 * (sigma_i0 + sigma_x0))
    l_b = _compute_l_b(E0, 1.0, 1.0e10, 3.0e14, sigma_i0 + sigma_x0)
    bl_ionizations = min(l_b / l_bn, 1.0) * (
        sigma_i0 / (sigma_i0 + sigma_x0)
    )
    print(f"  CSDA: {ion_per_primary:.2f} ionizations + "
          f"{exc_per_primary:.2f} excitations per primary "
          f"({events:.2f} events; absorbed = "
          f"{res.transmitted_flux == 0.0})")
    print(f"  Beer-Lambert books {bl_ionizations:.2f} ionizations/primary "
          f"-> undercount factor {ion_per_primary / bl_ionizations:.1f}x")
    assert 2.0 < events < 6.0, events
    assert ion_per_primary / bl_ionizations > 2.0

    # --- 3. Coulomb-only analytic range -------------------------------------
    print("\n=== 3. Coulomb-only range vs closed form ===")
    ne_c, Te_c = 5.0e12, 8.0
    col = uniform_column(cells=40000, dz=1.0, nn=0.0, ne=ne_c, Te=Te_c)
    E_stop = 1.0
    res = deposit_beam(
        E0, GAMMA0, E_stop_eV=E_stop, coulomb_model="fast_electron", **col
    )
    k = 2.0 * math.pi * _E4_CGS * ne_c * _c_log_ei(Te_c, ne_c) / _ERG_PER_EV**2
    R_analytic = (E0**2 - E_stop**2) / (2.0 * k)
    absorbed_cells = np.flatnonzero(res.plasma_heating_erg_s > 0.0)
    R_numeric = float(absorbed_cells[-1]) + 1.0
    rel = abs(R_numeric - R_analytic) / R_analytic
    print(f"  range: numeric {R_numeric:.0f} cm vs analytic "
          f"{R_analytic:.0f} cm (rel {rel:.2e})")
    assert rel < 0.01, rel

    # --- 4. inelastic-only ionization count vs quadrature -------------------
    print("\n=== 4. inelastic-only ionizations vs quadrature ===")
    col = uniform_column(cells=4000, dz=10.0, nn=3.0e14, ne=0.0, Te=1.0)
    res = deposit_beam(E0, GAMMA0, **col)
    ion_numeric = res.ionization_events.sum() / GAMMA0
    E_grid = np.linspace(HE_E_STOP_EV, E0, 20001)
    dN = np.zeros_like(E_grid)
    for i, E in enumerate(E_grid):
        s_i = He_EII_cross_lkup(E / HE_I_ION_EV) if E > HE_I_ION_EV else 0.0
        s_x, e_rad = He_beam_excitation_channel(E)
        denom = s_i * (HE_I_ION_EV + he_mean_secondary_energy_eV(E)) + s_x * e_rad
        dN[i] = s_i / denom if denom > 0.0 else 0.0
    ion_quad = float(np.trapezoid(dN, E_grid))
    rel = abs(ion_numeric - ion_quad) / ion_quad
    print(f"  ionizations/primary: numeric {ion_numeric:.4f} vs quadrature "
          f"{ion_quad:.4f} (rel {rel:.2e})")
    assert rel < 0.005, rel

    # --- 5. closure ordering at production conditions -----------------------
    # Evaluated at the historical flapped energy scale (150 eV) *and* the
    # honest current-driven plateau phi_c ~ 93 V (M4: the V-driven 405 V
    # median was a flapping artifact) — the ladder and
    # its ordering survive at both.
    print("\n=== 5. stopping-length ordering (item 12) ===")
    ladder_bounds = {
        # E_eV: (l_fast low/high, l_legacy low/high) [cm]
        150.0: ((2000.0, 10000.0), (50.0, 300.0)),  # ~30 m / ~1 m
        93.0: ((800.0, 2000.0), (40.0, 150.0)),  # ~12 m / ~0.8 m
    }
    for E_probe, ((f_lo, f_hi), (g_lo, g_hi)) in ladder_bounds.items():
        v_b = beam_speed_cm_s(E_probe)
        n_b = GAMMA0 / (700.0 * v_b)
        l_ql = quasilinear_relaxation_length_cm(E_probe, ne_c, n_b)
        l_legacy = E_probe / coulomb_stopping_eV_per_cm(
            E_probe, ne_c, Te_c, "legacy_tau_ei"
        )
        l_fast = E_probe / coulomb_stopping_eV_per_cm(
            E_probe, ne_c, Te_c, "fast_electron"
        )
        print(f"  E = {E_probe:5.0f} eV: l_QL = {l_ql:5.1f} cm  |  "
              f"legacy l_bi = {l_legacy:5.0f} cm  |  classical e-fold = "
              f"{l_fast:5.0f} cm  (n_b/n_e = {n_b / ne_c:.1e})")
        assert l_ql < l_legacy < l_fast
        assert l_ql < 100.0  # sub-m at beam energies
        assert f_lo < l_fast < f_hi, (E_probe, l_fast)
        assert g_lo < l_legacy < g_hi, (E_probe, l_legacy)

    print("\nbeam deposition acceptance: all checks OK")


if __name__ == "__main__":
    main()
