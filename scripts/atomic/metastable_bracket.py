"""Size the metastable / stepwise channel of the He ionization coefficient.

An INSTRUMENT, not a model change: nothing here is wired into the solver, no
rate is retuned (`b = 1` stands), and the production stance keeps the
unresolved `scd96_he.dat` it has always used. What this measures is how much
of the APPLIED ionization coefficient rides on the 2^3S metastable, and where
the ionizing column's own quasi-static metastable population puts the rate
relative to the table the solver reads.

WHAT THE RESOLVED DATA ACTUALLY CARRIES
---------------------------------------
The applied `scd96_he.dat` is ADAS404's UNRESOLVED collapse of the
metastable-resolved '96 helium set. The resolved siblings fetched alongside it
(`scd96r/acd96r/qcd96r/plt96r/prb96r_he.dat`, same producer, same code, same
04/11/99 date) resolve He0 into TWO metastables, not three -- their
metastable-count line reads `2 1 1`, and every block header runs IGRD = 1, 2
for z1 = 1. The second metastable is 1s2s 3S: fitting the low-Te slope of
ln(QCD_1->2 / QCD_2->1) = ln(g2/g1) - dE/Te over the 1-15 eV nodes returns
dE = 19.67-19.78 eV against the 2^3S term energy 19.820 eV, where 2^1S would
require 20.616 eV. The singlet metastable 2^1S is NOT an independent
population in this dataset; it sits inside the collisional-radiative bundle
built on the ground state, which is also why no `xcd96r_he.dat` exists (parent
cross-coupling needs two parent metastables and He+ has one).

THE POPULATION SOLVE
--------------------
Write N1, N2 for the He0 densities in 1^1S and 2^3S, and adopt the ADAS GCR
equations restricted to the two resolved metastables of the neutral stage.
With S_m = SCD_r(IPRT=1, IGRD=m, z1=1), Q_12 = QCD_r(IGRD=1, JGRD=2, z1=1),
Q_21 = QCD_r(IGRD=2, JGRD=1, z1=1), a recombination feed n_e N+ A_2 with
A_2 = ACD_r(IPRT=1, IGRD=2, z1=1), and a metastable transport loss 1/tau_x:

    dN2/dt = n_e N1 Q_12 + n_e N+ A_2
             - n_e N2 Q_21 - n_e N2 S_2 - N2/tau_x  =  0

    r = N2/N1 = (Q_12 + (N+/N1) A_2) / (Q_21 + S_2 + 1/(n_e tau_x))
    SCD_collapse = (S_1 + r S_2) / (1 + r)

the last line being the ionization rate n_e (N1 S_1 + N2 S_2) referred to the
total neutral density N1 + N2, i.e. the normalization the unresolved table
uses. Two closures of that equation are used here:

  SCD_thin    the IONIZING-COLUMN rate: no recombination feed (N+/N1 = 0) and
              no transport loss. This is the physically right closure for the
              LAPD column and is the instrument's deliverable.
  SCD_ionbal  the ionization-BALANCE rate: the recombination feed carried at
              the equilibrium N+/N0 = SCD_u/ACD_u, solved self-consistently.
              Used only as the upper end of the containment gate below.

WHY THE TABLE IS NOT EITHER OF THEM
-----------------------------------
ADAS404 collapses the resolved set with the ADF10 equilibrium metastable
fractions, which are computed in a LOW-LEVEL balance of spontaneous emission
and collisional excitation / de-excitation ONLY -- no ionization loss out of
the metastable and no recombination feed into it. The applied table therefore
sits ABOVE the ionizing-column collapse (which pays the ionization loss out of
2^3S) and BELOW the ionization-balance collapse (which is additionally fed by
recombination). That ordering is the gate:

  (i)  CORONAL CONTROL. At the lowest tabulated density every
       density-proportional term vanishes and the three closures must
       coincide, so SCD_thin must reproduce the table there.
  (ii) CONTAINMENT. At every node of the LAPD box the applied table must lie
       inside [SCD_thin, SCD_ionbal], within a small slack.

SCD_ionbal is deliberately NOT a single comparand: it overshoots the table by
20-49 % above 5 eV inside the box, which is the point -- it is a bracket end,
not a model of ADAS404.

`--calibration-only` runs exactly those two legs at the ADF11 NODES, with no
interpolation on either side, and exits non-zero on a miss.

WHAT IS NOT COMPUTED
--------------------
58.4 nm resonance trapping is NOT applied and cannot be at this level: 2^1P is
not a resolved population of the adf11 set, so a Holstein escape factor has no
coefficient to multiply. Reaching it needs a level-resolved (adf04)
collisional-radiative solve. The 584 nm line-centre optical depth tau_0 is
reported per port so the size of that omission is on the record.

Usage:
    python scripts/atomic/metastable_bracket.py --calibration-only
    python scripts/atomic/metastable_bracket.py --h5 RUN.h5 --out OUT.json
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cablp.atomic.adas import (  # noqa: E402
    ADAS_DIR,
    _interp_blend,
    _interp_coords,
    he_rates,
    read_adf11,
    read_adf11_resolved,
)
from cablp.constants import (  # noqa: E402
    c_cgs,
    kb_cgs,
    m_He_cgs,
    m_e_cgs,
    qe_cgs,
)

# The LAPD box the containment leg is asserted over.
CALIBRATION_NE_CM3 = (1.0e11, 1.0e14)
CALIBRATION_TE_EV = (1.0, 20.0)

# Leg (i): the coronal control, and the temperature span it is asserted over.
CORONAL_TOL = 0.03
CORONAL_TE_EV = (0.7, 1.5e4)

# Leg (ii): how far outside [SCD_thin, SCD_ionbal] the table may sit before
# the ordering is judged violated.
CONTAINMENT_SLACK = 0.02

# He I 1^1S - 2^1P resonance line. The repository carries no archived
# resonance cross-section (`scripts/score/pec_band_fractions.md` names the
# wavelength for a band split and nothing else), so sigma_0 is derived below
# from the oscillator strength. f = 0.2762 is the value the registration
# supplies; the wavelength is the standard vacuum value for the line.
HE_RESONANCE_LAMBDA_CM = 584.334e-8
HE_RESONANCE_F = 0.2762

# The five ES1 probe ports and their axial positions, read from the scoring
# overlay so this instrument and `scripts/score/compare_sim1d_es1.py` cannot
# disagree about where a port is.
OVERLAY = Path(__file__).resolve().parents[1] / "data" / "es1_sim1d_overlay.npz"

RESOLVED_FILES = {
    "scd": "scd96r_he.dat",
    "acd": "acd96r_he.dat",
    "qcd": "qcd96r_he.dat",
}


def load_resolved():
    """Return the resolved He0 coefficient tables on their shared grid.

    Asserts the metastable structure this solve is written for -- two neutral
    metastables, one parent -- and the index-name convention of each class, so
    a rotated OPEN-ADAS revision that changed either raises here rather than
    silently assigning a coefficient to the wrong metastable.
    """
    tables = {}
    axes = None
    for key, fname in RESOLVED_FILES.items():
        log_ne, log_te, counts, blocks, names = read_adf11_resolved(
            ADAS_DIR / fname
        )
        if counts[:2] != (2, 1):
            raise ValueError(
                f"{fname}: metastable counts {counts} -- this solve is "
                "written for two He0 metastables and one He+ parent"
            )
        if axes is None:
            axes = (log_ne, log_te)
        elif not (
            np.array_equal(log_ne, axes[0]) and np.array_equal(log_te, axes[1])
        ):
            raise ValueError(f"{fname}: axes differ from the other resolved files")
        tables[key] = (blocks, names)

    scd_blocks, scd_names = tables["scd"]
    acd_blocks, acd_names = tables["acd"]
    qcd_blocks, qcd_names = tables["qcd"]
    if scd_names != ("IPRT", "IGRD"):
        raise ValueError(f"scd96r: index names {scd_names}, expected IPRT/IGRD")
    if acd_names != ("IPRT", "IGRD"):
        raise ValueError(f"acd96r: index names {acd_names}, expected IPRT/IGRD")
    if qcd_names != ("IGRD", "JGRD"):
        raise ValueError(f"qcd96r: index names {qcd_names}, expected IGRD/JGRD")

    return {
        "log_ne": axes[0],
        "log_te": axes[1],
        # log10 tables, (nte, ndens)
        "S1": scd_blocks[(1, 1, 1)],
        "S2": scd_blocks[(1, 2, 1)],
        "A2": acd_blocks[(1, 2, 1)],
        "Q12": qcd_blocks[(1, 2, 1)],
        "Q21": qcd_blocks[(2, 1, 1)],
    }


def _eval(res, key, ne_cm3, Te_eV):
    """Bilinear lookup of one resolved table, in the units of the file."""
    ix, iy, fx, fy = _interp_coords(
        res["log_ne"], res["log_te"], np.log10(ne_cm3), np.log10(Te_eV)
    )
    return 10.0 ** _interp_blend(res[key], ix, iy, fx, fy)


def metastable_ratio(S2, Q12, Q21, ne_cm3, tau_x_s=None, feed=0.0):
    """Return r = N(2^3S)/N(1^1S) in quasi-static equilibrium.

    ``tau_x_s`` is the metastable transport loss time (``None`` is the
    transport-free limit the tabulated coefficient assumes); ``feed`` is the
    recombination source (N+/N1) * A_2, zero for the ionizing column.
    """
    loss = Q21 + S2
    if tau_x_s is not None:
        loss = loss + 1.0 / (ne_cm3 * tau_x_s)
    return (Q12 + feed) / loss


def scd_collapse(S1, S2, r):
    """Ground-referenced effective ionization coefficient [cm^3/s]."""
    return (S1 + r * S2) / (1.0 + r)


def scd_ionization_balance(S1, S2, A2, Q12, Q21, unresolved_scd, unresolved_acd):
    """Collapse with the recombination feed at ionization equilibrium.

    N+/N0 = SCD_u/ACD_u sets the parent density; the metastable ratio and the
    total neutral density it is referred to are solved together.
    """
    np_over_n0 = unresolved_scd / unresolved_acd
    r = metastable_ratio(S2, Q12, Q21, None)
    for _ in range(300):
        r = metastable_ratio(
            S2, Q12, Q21, None, feed=np_over_n0 * (1.0 + r) * A2
        )
    return r, scd_collapse(S1, S2, r)


def run_calibration(res):
    """Run the two-leg gate at the shared table nodes.

    Returns ``(passed, record, grid)``. Both legs compare node to node with no
    interpolation on either side, so nothing here can be an interpolation
    artifact.
    """
    log_ne_u, log_te_u, stages_u = read_adf11(ADAS_DIR / "scd96_he.dat")
    log_ne_a, log_te_a, stages_a = read_adf11(ADAS_DIR / "acd96_he.dat")
    for grid_ne, grid_te, name in (
        (log_ne_u, log_te_u, "scd96_he.dat"),
        (log_ne_a, log_te_a, "acd96_he.dat"),
    ):
        if not (
            np.array_equal(grid_ne, res["log_ne"])
            and np.array_equal(grid_te, res["log_te"])
        ):
            raise ValueError(
                f"{name} and the resolved files are tabulated on different "
                "grids; the node-to-node gate assumes one shared grid"
            )
    ne = 10.0 ** res["log_ne"]
    Te = 10.0 ** res["log_te"]
    S1 = 10.0 ** res["S1"]
    S2 = 10.0 ** res["S2"]
    A2 = 10.0 ** res["A2"]
    Q12 = 10.0 ** res["Q12"]
    Q21 = 10.0 ** res["Q21"]
    table = 10.0 ** stages_u[1]
    table_acd = 10.0 ** stages_a[1]

    r_thin = metastable_ratio(S2, Q12, Q21, None)
    scd_thin = scd_collapse(S1, S2, r_thin)
    _, scd_ionbal = scd_ionization_balance(
        S1, S2, A2, Q12, Q21, table, table_acd
    )

    # --- leg (i): coronal control -------------------------------------
    band = (Te >= CORONAL_TE_EV[0]) & (Te <= CORONAL_TE_EV[1])
    coronal_dev = scd_thin[:, 0] / table[:, 0] - 1.0
    coronal_worst = float(np.abs(coronal_dev[band]).max())
    coronal_pass = coronal_worst <= CORONAL_TOL

    # --- leg (ii): containment ----------------------------------------
    in_box = (
        (ne[None, :] >= CALIBRATION_NE_CM3[0])
        & (ne[None, :] <= CALIBRATION_NE_CM3[1])
        & (Te[:, None] >= CALIBRATION_TE_EV[0])
        & (Te[:, None] <= CALIBRATION_TE_EV[1])
    ) & np.ones_like(table, dtype=bool)
    above_lower = table >= scd_thin * (1.0 - CONTAINMENT_SLACK)
    below_upper = table <= scd_ionbal * (1.0 + CONTAINMENT_SLACK)
    n_low = int((in_box & ~above_lower).sum())
    n_high = int((in_box & ~below_upper).sum())
    containment_pass = (n_low == 0) and (n_high == 0)

    passed = coronal_pass and containment_pass

    dev_thin = scd_thin / table - 1.0
    dev_ionbal = scd_ionbal / table - 1.0
    rows = []
    for i, t in enumerate(Te):
        for j, n in enumerate(ne):
            if in_box[i, j]:
                rows.append(
                    {
                        "ne_cm3": float(n),
                        "Te_eV": float(t),
                        "scd_table_cm3_s": float(table[i, j]),
                        "scd_thin_cm3_s": float(scd_thin[i, j]),
                        "scd_ionbal_cm3_s": float(scd_ionbal[i, j]),
                        "thin_over_table_minus_1": float(dev_thin[i, j]),
                        "ionbal_over_table_minus_1": float(dev_ionbal[i, j]),
                        "contained": bool(
                            above_lower[i, j] and below_upper[i, j]
                        ),
                        "metastable_fraction": float(
                            r_thin[i, j] / (1.0 + r_thin[i, j])
                        ),
                        "stepwise_share": float(
                            r_thin[i, j]
                            * S2[i, j]
                            / (S1[i, j] + r_thin[i, j] * S2[i, j])
                        ),
                    }
                )

    record = {
        "coronal_control": {
            "ne_cm3": float(ne[0]),
            "Te_band_eV": list(CORONAL_TE_EV),
            "tolerance": CORONAL_TOL,
            "max_abs_deviation_frac": coronal_worst,
            "passed": bool(coronal_pass),
        },
        "containment": {
            "ne_box_cm3": list(CALIBRATION_NE_CM3),
            "Te_box_eV": list(CALIBRATION_TE_EV),
            "slack": CONTAINMENT_SLACK,
            "n_nodes": int(in_box.sum()),
            "n_below_thin": n_low,
            "n_above_ionbal": n_high,
            "passed": bool(containment_pass),
        },
        "passed": bool(passed),
        "nodes": rows,
    }
    grid = (ne, Te, dev_thin, dev_ionbal, r_thin, S1, S2)
    return passed, record, grid


def print_calibration(record, grid):
    ne, Te, dev_thin, dev_ionbal, r_thin, S1, S2 = grid
    jn = [
        j
        for j, n in enumerate(ne)
        if CALIBRATION_NE_CM3[0] <= n <= CALIBRATION_NE_CM3[1]
    ]
    it = [
        i
        for i, t in enumerate(Te)
        if CALIBRATION_TE_EV[0] <= t <= CALIBRATION_TE_EV[1]
    ]

    cc = record["coronal_control"]
    print("GATE (i) CORONAL CONTROL -- SCD_thin / SCD_table - 1 at the lowest")
    print(f"  tabulated density, {cc['Te_band_eV'][0]} eV - "
          f"{cc['Te_band_eV'][1]:.0f} eV, bar {cc['tolerance'] * 100:.0f} %")
    print(f"  ne = {cc['ne_cm3']:.2e} cm^-3")
    for i, t in enumerate(Te):
        if CORONAL_TE_EV[0] <= t <= CORONAL_TE_EV[1] and t <= 30.0:
            print(f"      Te = {t:8.3f} eV : {dev_thin[i, 0] * 100:+7.3f} %")
    print(f"  max |deviation| over the band : "
          f"{cc['max_abs_deviation_frac'] * 100:.3f} %")
    print(f"  LEG (i): {'PASS' if cc['passed'] else 'FAIL'}")

    print()
    ct = record["containment"]
    print("GATE (ii) CONTAINMENT -- the applied table must lie inside")
    print("  [SCD_thin, SCD_ionbal] at every node of the LAPD box, slack "
          f"{ct['slack'] * 100:.0f} %")
    print()
    print("  SCD_thin / SCD_table - 1, in %  (the ionizing-column lower end)")
    print("  Te\\ne " + "".join(f"{ne[j]:10.1e}" for j in jn))
    for i in it:
        print(f"  {Te[i]:6.2f}"
              + "".join(f"{dev_thin[i, j] * 100:10.2f}" for j in jn))
    print()
    print("  SCD_ionbal / SCD_table - 1, in %  (the ionization-balance upper end)")
    print("  Te\\ne " + "".join(f"{ne[j]:10.1e}" for j in jn))
    for i in it:
        print(f"  {Te[i]:6.2f}"
              + "".join(f"{dev_ionbal[i, j] * 100:10.2f}" for j in jn))
    print()
    share = r_thin * S2 / (S1 + r_thin * S2)
    print("  stepwise share of the rate carried by the 2^3S channel, in %")
    print("  Te\\ne " + "".join(f"{ne[j]:10.1e}" for j in jn))
    for i in it:
        print(f"  {Te[i]:6.2f}"
              + "".join(f"{share[i, j] * 100:10.2f}" for j in jn))
    print()
    print(f"  nodes tested            : {ct['n_nodes']}")
    print(f"  table below SCD_thin    : {ct['n_below_thin']}")
    print(f"  table above SCD_ionbal  : {ct['n_above_ionbal']}")
    print(f"  LEG (ii): {'PASS' if ct['passed'] else 'FAIL'}")
    print()
    print(f"CALIBRATION: {'PASS' if record['passed'] else 'FAIL'}")


def resonance_cross_section_cm2(T_gas_K):
    """Doppler line-centre cross-section of He I 584.334 A [cm^2].

    sigma_0 = sqrt(pi) e^2 f / (m_e c dnu_D) with the Doppler width
    dnu_D = (nu_0/c) sqrt(2 k T / M), i.e. the line-centre value of
    (pi e^2 / m_e c) f phi(nu) for a normalized Doppler profile.
    """
    c = float(c_cgs)
    nu0 = c / HE_RESONANCE_LAMBDA_CM
    v_mp = np.sqrt(2.0 * kb_cgs * T_gas_K / m_He_cgs)
    dnu_d = nu0 * v_mp / c
    return (
        np.sqrt(np.pi)
        * qe_cgs**2
        * HE_RESONANCE_F
        / (m_e_cgs * c * dnu_d)
    )


def mean_thermal_speed_cm_s(T_gas_K):
    """Mean (not most-probable) He atom speed sqrt(8kT/pi M) [cm/s]."""
    return np.sqrt(8.0 * kb_cgs * T_gas_K / (np.pi * m_He_cgs))


def read_plateau_state(h5_path, window_ms):
    """Return the plateau-averaged per-cell state of a saved run."""
    with h5py.File(h5_path, "r") as f:
        params = json.loads(f.attrs["params_json"])
        t_ms = f["time"][:] * 1.0e3
        mask = (t_ms >= window_ms[0]) & (t_ms <= window_ms[1])
        if not mask.any():
            raise ValueError(
                f"{h5_path}: no saved frames in {window_ms[0]}-{window_ms[1]} ms"
            )
        state = {
            "z_cm": f["geometry/z_cm"][:],
            "Rp_cm": f["geometry/Rp_cm"][:],
            "plasma_active": f["geometry/plasma_active"][:],
            "cell_role": np.array(
                [
                    r.decode() if isinstance(r, bytes) else str(r)
                    for r in f["geometry/cell_role"][:]
                ]
            ),
            "ne_cm3": f["n"][:][mask].mean(axis=0),
            "Te_eV": f["Te"][:][mask].mean(axis=0),
            "nn_cm3": f["nn"][:][mask].mean(axis=0),
            "n_frames": int(mask.sum()),
            "window_ms": list(window_ms),
            "configuration_name": str(f.attrs.get("configuration_name")),
            "configuration_identity": str(f.attrs.get("configuration_identity")),
            "Tn_K": float(params["Tn_K"]),
        }
    return state


def port_cells(z_model):
    """Map the five ES1 ports onto model cells, as the scorer does."""
    overlay = np.load(OVERLAY, allow_pickle=True)
    ports = np.asarray(overlay["port"], dtype=int)
    z_port = np.asarray(overlay["z_cm"], dtype=float)
    return [
        (int(p), float(z), int(np.argmin(np.abs(z_model - z))))
        for p, z in zip(ports, z_port)
    ]


def build_cell_table(res, state):
    """Per-cell metastable diagnostics over the column."""
    active = state["plasma_active"] & (state["cell_role"] == "column")
    idx = np.flatnonzero(active)
    ne = state["ne_cm3"][idx]
    Te = state["Te_eV"][idx]
    nn = state["nn_cm3"][idx]
    Rp = state["Rp_cm"][idx]
    T_gas = state["Tn_K"]

    S1 = _eval(res, "S1", ne, Te)
    S2 = _eval(res, "S2", ne, Te)
    Q12 = _eval(res, "Q12", ne, Te)
    Q21 = _eval(res, "Q21", ne, Te)

    # The APPLIED coefficient, read exactly as the solver reads it.
    scd_table = np.asarray(he_rates(ne, Te, ("scd",))["scd"], dtype=float)

    r_thin = metastable_ratio(S2, Q12, Q21, None)
    scd_thin = scd_collapse(S1, S2, r_thin)

    # The transport-loss correction: tau_x is the column-crossing time at the
    # local gas temperature -- the diameter 2 Rp at the mean thermal speed,
    # the form that reproduces the registration's stated ~3e-4 s at 300 K
    # (18.415 cm plasma radius gives 2.93e-4 s).
    v_th = mean_thermal_speed_cm_s(T_gas)
    tau_x = 2.0 * Rp / v_th
    r_tx = metastable_ratio(S2, Q12, Q21, ne, tau_x_s=tau_x)
    scd_tx = scd_collapse(S1, S2, r_tx)

    # Scoping diagnostic for the trapping leg, which is NOT computed: the
    # 58.4 nm line-centre optical depth across the plasma radius.
    sigma0 = resonance_cross_section_cm2(T_gas)
    tau0 = nn * sigma0 * Rp

    return {
        "cell": idx,
        "z_cm": state["z_cm"][idx],
        "ne_cm3": ne,
        "Te_eV": Te,
        "nn_cm3": nn,
        "Rp_cm": Rp,
        "S1": S1,
        "S2": S2,
        "r_thin": r_thin,
        "scd_table": scd_table,
        "scd_thin": scd_thin,
        "thin_over_table": scd_thin / scd_table,
        "scd_transport": scd_tx,
        "transport_over_thin": scd_tx / scd_thin,
        "stepwise_share": r_thin * S2 / (S1 + r_thin * S2),
        "tau0_584": tau0,
        "tau_x_s": tau_x,
        "sigma0_cm2": float(sigma0),
        "v_thermal_cm_s": float(v_th),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--h5",
        default=None,
        help="saved reference run to read the plateau state from",
    )
    ap.add_argument(
        "--window",
        default="15.0,19.5",
        help="plateau window in ms as 'start,end' (default 15.0,19.5)",
    )
    ap.add_argument("--out", default=None, help="write the record as JSON here")
    ap.add_argument(
        "--calibration-only",
        action="store_true",
        help="run the calibration gate and nothing else",
    )
    args = ap.parse_args(argv)

    res = load_resolved()
    passed, calibration, grid = run_calibration(res)
    print_calibration(calibration, grid)
    record = {"calibration": calibration}

    if args.calibration_only:
        if args.out:
            Path(args.out).write_text(json.dumps(record, indent=2))
            print(f"\nwrote {args.out}")
        return 0 if passed else 1

    if not args.h5:
        ap.error("--h5 is required unless --calibration-only is given")

    window = tuple(float(x) for x in args.window.split(","))
    state = read_plateau_state(args.h5, window)
    table = build_cell_table(res, state)
    ports = port_cells(state["z_cm"])

    print()
    print(f"PLATEAU STATE  {Path(args.h5).name}")
    print(f"  configuration {state['configuration_name']} "
          f"({state['configuration_identity'][:16]}...)")
    print(f"  window {window[0]}-{window[1]} ms, {state['n_frames']} frames; "
          f"Tn = {state['Tn_K']:.1f} K")
    print(f"  mean He speed = {table['v_thermal_cm_s']:.4e} cm/s; "
          f"tau_x = 2Rp/v = {table['tau_x_s'][0]:.3e} s at Rp = "
          f"{table['Rp_cm'][0]:.3f} cm")
    print(f"  sigma_0(584.334 A, Doppler, {state['Tn_K']:.0f} K) = "
          f"{table['sigma0_cm2']:.4e} cm^2   "
          f"(f = {HE_RESONANCE_F}, derived -- no archived value in-repo)")
    print()
    print("PER-PORT -- the ionizing column's quasi-static rate against the "
          "applied table")
    print("  port    z_cm     ne_cm3    Te_eV   SCD_thin/table-1  stepwise%  "
          "transport   tau0_584")
    port_rows = []
    for p, z_port, cell in ports:
        w = np.flatnonzero(table["cell"] == cell)
        if w.size == 0:
            continue
        k = int(w[0])
        row = {
            "port": p,
            "z_port_cm": z_port,
            "cell": cell,
            "z_cell_cm": float(table["z_cm"][k]),
            "ne_cm3": float(table["ne_cm3"][k]),
            "Te_eV": float(table["Te_eV"][k]),
            "nn_cm3": float(table["nn_cm3"][k]),
            "scd_table_cm3_s": float(table["scd_table"][k]),
            "scd_thin_cm3_s": float(table["scd_thin"][k]),
            "thin_over_table_minus_1": float(table["thin_over_table"][k] - 1.0),
            "stepwise_share": float(table["stepwise_share"][k]),
            "transport_over_thin": float(table["transport_over_thin"][k]),
            "tau0_584": float(table["tau0_584"][k]),
            "metastable_fraction": float(
                table["r_thin"][k] / (1.0 + table["r_thin"][k])
            ),
        }
        port_rows.append(row)
        print(f"  p{p:<4d} {row['z_cell_cm']:8.2f} {row['ne_cm3']:10.3e} "
              f"{row['Te_eV']:7.3f} "
              f"{row['thin_over_table_minus_1'] * 100:15.2f} % "
              f"{row['stepwise_share'] * 100:9.2f} "
              f"{row['transport_over_thin']:11.5f} "
              f"{row['tau0_584']:10.3f}")

    thin_dev = table["thin_over_table"] - 1.0
    print()
    print("OVER THE COLUMN")
    print(f"  cells                     : {table['cell'].size}")
    print(f"  SCD_thin/table - 1        : {thin_dev.min() * 100:+.2f} % .. "
          f"{thin_dev.max() * 100:+.2f} %")
    print(f"  stepwise share            : "
          f"{table['stepwise_share'].min() * 100:.2f} .. "
          f"{table['stepwise_share'].max() * 100:.2f} %")
    print(f"  transport correction      : "
          f"{table['transport_over_thin'].min():.5f} .. "
          f"{table['transport_over_thin'].max():.5f}")
    print(f"  tau0(584 nm)              : {table['tau0_584'].min():.3f} .. "
          f"{table['tau0_584'].max():.3f}")

    print()
    print("WHAT THIS SAYS")
    print("  The applied table is the ADF10 low-level-balance collapse. The "
          "ionizing")
    print("  column pays an ionization loss out of 2^3S that that balance "
          "does not,")
    print("  so its quasi-static rate sits BELOW the table by "
          f"{abs(port_rows[0]['thin_over_table_minus_1']) * 100:.1f} % at p"
          f"{port_rows[0]['port']} rising to "
          f"{abs(port_rows[-1]['thin_over_table_minus_1']) * 100:.1f} % at p"
          f"{port_rows[-1]['port']},")
    print("  tracking the fall in Te along the machine. Metastable transport "
          "loss is a")
    print("  null at column densities (<= "
          f"{(1.0 - table['transport_over_thin'].min()) * 100:.1f} % anywhere, "
          f"<= {(1.0 - min(r['transport_over_thin'] for r in port_rows)) * 100:.2f} "
          "% at the ports):")
    print("  collisional quenching of 2^3S beats its column crossing by about "
          "two orders.")
    print()
    print("  58.4 nm TRAPPING: NOT COMPUTED at adf11 level. 2^1P is not a "
          "resolved")
    print("  population of this dataset, so a Holstein escape factor has no "
          "coefficient")
    print("  to multiply; reaching it needs a level-resolved (adf04) CR "
          "solve. The tau0")
    print("  column above records how far from thin the ports actually are.")

    record["state"] = {
        "h5": str(args.h5),
        "configuration_name": state["configuration_name"],
        "configuration_identity": state["configuration_identity"],
        "window_ms": state["window_ms"],
        "n_frames": state["n_frames"],
        "Tn_K": state["Tn_K"],
        "sigma0_cm2": table["sigma0_cm2"],
        "resonance_lambda_cm": HE_RESONANCE_LAMBDA_CM,
        "resonance_f": HE_RESONANCE_F,
        "v_thermal_cm_s": table["v_thermal_cm_s"],
    }
    record["ports"] = port_rows
    record["column"] = {
        "cell": [int(c) for c in table["cell"]],
        "z_cm": [float(v) for v in table["z_cm"]],
        "ne_cm3": [float(v) for v in table["ne_cm3"]],
        "Te_eV": [float(v) for v in table["Te_eV"]],
        "nn_cm3": [float(v) for v in table["nn_cm3"]],
        "Rp_cm": [float(v) for v in table["Rp_cm"]],
        "tau0_584": [float(v) for v in table["tau0_584"]],
        "tau_x_s": [float(v) for v in table["tau_x_s"]],
        "metastable_fraction": [
            float(v / (1.0 + v)) for v in table["r_thin"]
        ],
        "stepwise_share": [float(v) for v in table["stepwise_share"]],
        "scd_table_cm3_s": [float(v) for v in table["scd_table"]],
        "scd_thin_cm3_s": [float(v) for v in table["scd_thin"]],
        "thin_over_table_minus_1": [
            float(v - 1.0) for v in table["thin_over_table"]
        ],
        "transport_over_thin": [
            float(v) for v in table["transport_over_thin"]
        ],
    }
    record["trapping"] = {
        "computed": False,
        "reason": (
            "2^1P is not a resolved population of the adf11 set, so a Holstein "
            "escape factor has no coefficient to multiply; a level-resolved "
            "(adf04) collisional-radiative solve is required"
        ),
        "tau0_584_by_port": {
            f"p{r['port']}": r["tau0_584"] for r in port_rows
        },
    }

    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2))
        print(f"\nwrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
