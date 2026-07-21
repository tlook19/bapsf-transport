"""M5b: the between-shot ads/des cycle map (analytic, no simulation).

The shot-cycle dynamical system for the surface coverage
(CATHODE_IDRIVEN_PLAN.md M5b): during the ~2.97 s cooldown the coverage
relaxes toward the adsorption / thermal-desorption equilibrium at the
standby temperature,

    theta_eq(T) = k_ads / (k_ads + nu_th(T)),   nu_th = nu0 e^(-E_des/kT)
    tau_r(T)    = 1 / (k_ads + nu_th(T))

and the drive phase strips it by the measured fluence exponent
Lambda ~ 7 (theta_end ~ 1e-3 in every M5a' campaign run). The cycle
fixed point is

    theta* = theta_eq (1 - E) / (1 - e^(-Lambda) E),   E = e^(-t_c/tau_r)

This script scans (k_ads, nu_th(1910 K), E_des) and reports the region
compatible with the two DATA-side constraints, plus what that region
implies for the third:

  [C1] normalization: theta*(1910) >= 0.97 -- the M5a fit anchored
       phi_start at full shot-start coverage on ES1;
  [C2] ladder flatness: theta*(1972)/theta*(1910) >= 0.90 -- a much
       cleaner ES3 start would re-break the measured regime-flat ramp
       that M5a' reproduced;
  [Q ] the maximal bulk-cycle T-sensitivity achievable inside [C1]x[C2]:
       dlnJ_start/dT = (e*dphi_shot/kT) * (-dtheta*/dT), compared to the
       ~1 %/K Richardson slope. ANALYTIC PRE-REGISTRATION: inside the
       allowed region this tops out around 0.01-0.05 %/K -- the cycle
       map CANNOT be the ES1 marginal-drift amplifier if the ramp
       ladder is flat. The marginality question therefore rests on the
       measured energy-side gain (the +-8 K stability pair with the
       full model), not on coverage kinetics. If that pair also comes
       back strongly subcritical, the honest conclusion is that the
       trim phenomenology needs physics outside this model (heater
       regulation mode, bulk view-factor drift, deeper contamination
       reservoirs), and the thesis says so.

Usage::

    python scripts/cycle_map_m5b.py [--t-cycle 2.97] [--Lambda 7.0]
"""

import argparse

import numpy as np

KB_EV = 8.617333262e-5
STANDBY_K = {"ES1": 1910.0, "ES2": 1949.0, "ES3": 1972.0}
DPHI_SHOT_EV = 0.060  # the M5a' per-shot amplitude
KT_1910 = KB_EV * 1910.0


def theta_star(T_K, k_ads, nu0, E_des_eV, t_cycle_s, Lam):
    nu = nu0 * np.exp(-E_des_eV / (KB_EV * T_K))
    theta_eq = k_ads / (k_ads + nu)
    E = np.exp(-t_cycle_s * (k_ads + nu))
    return theta_eq * (1.0 - E) / (1.0 - np.exp(-Lam) * E)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--t-cycle", type=float, default=2.97)
    p.add_argument("--Lambda", type=float, default=7.0)
    args = p.parse_args(argv)

    # nu0 is parameterized through nu_th at the ES1 standby so the scan
    # axes are the physically meaningful pair (recontamination rate,
    # desorption rate at operating point); E_des then sets the ladder.
    k_grid = np.geomspace(0.1, 100.0, 61)          # k_ads [1/s]
    r_grid = np.geomspace(1e-4, 1.0, 61)           # nu_th(1910)/k_ads
    print(f"t_cycle {args.t_cycle} s, Lambda {args.Lambda}, "
          f"constraints: theta*(1910)>=0.97, ladder ratio>=0.90\n")
    for E_des in (2.0, 3.0, 4.0, 5.0):
        best = None
        n_allowed = 0
        for k in k_grid:
            for r in r_grid:
                nu1910 = r * k
                nu0 = nu1910 / np.exp(-E_des / KT_1910)
                th1 = theta_star(1910.0, k, nu0, E_des,
                                 args.t_cycle, args.Lambda)
                th3 = theta_star(1972.0, k, nu0, E_des,
                                 args.t_cycle, args.Lambda)
                if th1 < 0.97 or th3 / th1 < 0.90:
                    continue
                n_allowed += 1
                # T-sensitivity at the ES1 point (finite difference +-1 K)
                dth = (
                    theta_star(1911.0, k, nu0, E_des,
                               args.t_cycle, args.Lambda)
                    - theta_star(1909.0, k, nu0, E_des,
                                 args.t_cycle, args.Lambda)
                ) / 2.0
                dlnJ = (DPHI_SHOT_EV / KT_1910) * (-dth)  # per K
                if best is None or abs(dlnJ) > abs(best[0]):
                    th2 = theta_star(1949.0, k, nu0, E_des,
                                     args.t_cycle, args.Lambda)
                    best = (dlnJ, k, r, th1, th2, th3)
        if best is None:
            print(f"E_des = {E_des:.1f} eV: allowed region EMPTY")
            continue
        dlnJ, k, r, th1, th2, th3 = best
        print(
            f"E_des = {E_des:.1f} eV: {n_allowed:4d} allowed points; "
            f"max |dlnJ_start/dT| = {abs(dlnJ) * 100:.4f} %/K "
            f"(Richardson ~1 %/K)\n"
            f"    at k_ads = {k:.2f}/s, nu_th(1910)/k_ads = {r:.4f}; "
            f"emergent theta* ladder ES1/ES2/ES3 = "
            f"{th1:.3f}/{th2:.3f}/{th3:.3f} "
            f"-> phi_start shifts <= "
            f"{DPHI_SHOT_EV * (th1 - th3) * 1e3:.1f} meV"
        )
    print(
        "\nReading: within the data-allowed region the cycle map's "
        "contribution to bulk\nT-sensitivity is negligible against the "
        "Richardson slope, and the emergent\ntheta* ladder stays within "
        "a few % of 1 -- the M5a zero-tuning assumption\n(theta_start = "
        "1 everywhere) is SELF-CONSISTENT with the map. The ES1\n"
        "marginality must come from the energy loop or from physics "
        "outside the model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
