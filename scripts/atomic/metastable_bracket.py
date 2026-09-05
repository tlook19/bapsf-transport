"""Bracket the metastable/stepwise channel of the He ionization coefficient.

An INSTRUMENT, not a model change: nothing here is wired into the solver, no
rate is retuned (`b = 1` stands), and the production stance keeps the
unresolved `scd96_he.dat` it has always used. The question asked is how much
of the applied ionization coefficient rides on the 2^3S metastable, and how
far the coefficient could move at the production state once the metastable
population stops being the optically-thin, transport-free quasi-static one
that the tabulated coefficient assumes.

WHAT THE RESOLVED DATA ACTUALLY CARRIES
---------------------------------------
The applied `scd96_he.dat` is ADAS404's UNRESOLVED projection of the
metastable-resolved '96 helium set. The resolved siblings fetched alongside it
(`scd96r/acd96r/qcd96r/plt96r/prb96r_he.dat`, same producer, same code, same
04/11/99 date) resolve He0 into TWO metastables, not three -- their
metastable-count line reads `2 1 1`, and every block header runs IGRD = 1, 2
for z1 = 1. The second metastable is 1s2s 3S: fitting the low-Te slope of
ln(QCD_1->2 / QCD_2->1) = ln(g2/g1) - dE/Te over the 1-15 eV nodes returns
dE = 19.75-19.78 eV against the 2^3S term energy 19.820 eV, where 2^1S would
require 20.616 eV. The singlet metastable 2^1S is NOT an independent
population in this dataset; it sits inside the collisional-radiative bundle
built on the ground state, which is also why no `xcd96r_he.dat` exists (parent
cross-coupling needs two parent metastables and He+ has one).

THE POPULATION SOLVE
--------------------
Write N1, N2 for the He0 densities in 1^1S and 2^3S, and adopt the ADAS GCR
equations restricted to the two resolved metastables of the neutral stage.
With S_m = SCD_r(IPRT=1, IGRD=m, z1=1), Q_12 = QCD_r(IGRD=1, JGRD=2, z1=1),
Q_21 = QCD_r(IGRD=2, JGRD=1, z1=1), and a metastable transport loss 1/tau_x:

    dN2/dt = n_e N1 Q_12 - n_e N2 Q_21 - n_e N2 S_2 - N2/tau_x  =  0

so the metastable-to-ground ratio and the ground-referenced effective
coefficient are

    r  = N2/N1 = Q_12 / (Q_21 + S_2 + 1/(n_e tau_x))
    SCD_eff = (S_1 + r S_2) / (1 + r)

the second line being the ionization rate n_e (N1 S_1 + N2 S_2) referred to
the total neutral density N1 + N2, i.e. the same normalization the unresolved
table uses. The recombination feed n_e N+ ACD_r(->2) is dropped: the LAPD
column is strongly ionizing, and carrying it requires a He+ density the adf11
files do not record (see the calibration verdict).

THE CALIBRATION GATE
--------------------
With tau_x infinite the solve must reproduce the unresolved `scd96_he.dat` at
every shared table node, because both files are the same ADAS404 projection of
the same underlying data. `--calibration-only` runs exactly that comparison
over the LAPD box (n_e 1e11-1e14 cm^-3, T_e 1-20 eV) at the ADF11 NODES, with
no interpolation, and exits non-zero on a miss.

VERDICT ON RECORD (2026-09-05): the gate FAILS. In the coronal limit
(n_e = 5e7 cm^-3) the solve reproduces the unresolved table to <= 1.1 % from
0.5 eV to 15 keV -- the equations, the block assignment and the normalization
are right -- but the deviation grows monotonically with density and reaches
-13.9 % inside the LAPD box, systematically LOW. The missing term is
density-proportional and recombination-shaped: inverting the unresolved table
for the metastable fraction ADAS404 must have used returns a required N+/N1
that tracks the ionization-equilibrium SCD_u/ACD_u to within a smoothly
varying factor 0.03-1.14, so ADAS404 projected at a reference ionization
balance the public adf11 files do not carry. That reference is not
reconstructible from adf11 alone, and inventing one would be tuning.

Because the gate is the instrument that licenses the corrections, this script
STOPS at the state table: it reports the plateau plasma state, the optical
depth that scopes the trapping correction, the metastable fractions and the
transport-loss ratio as EVIDENCE, and withholds the quotable bracket.

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

# The LAPD box the calibration gate is asserted over, and the gate's bar. The
# 5 % bar is the accuracy the ADAS404 projection is expected to hold to; a
# miss means the population solve is wrong, not that the bar is generous.
CALIBRATION_NE_CM3 = (1.0e11, 1.0e14)
CALIBRATION_TE_EV = (1.0, 20.0)
CALIBRATION_TOL = 0.05

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
    qcd_blocks, qcd_names = tables["qcd"]
    if scd_names != ("IPRT", "IGRD"):
        raise ValueError(f"scd96r: index names {scd_names}, expected IPRT/IGRD")
    if qcd_names != ("IGRD", "JGRD"):
        raise ValueError(f"qcd96r: index names {qcd_names}, expected IGRD/JGRD")

    return {
        "log_ne": axes[0],
        "log_te": axes[1],
        # log10 tables, (nte, ndens)
        "S1": scd_blocks[(1, 1, 1)],
        "S2": scd_blocks[(1, 2, 1)],
        "Q12": qcd_blocks[(1, 2, 1)],
        "Q21": qcd_blocks[(2, 1, 1)],
    }


def _eval(res, key, ne_cm3, Te_eV):
    """Bilinear lookup of one resolved table, in the units of the file."""
    ix, iy, fx, fy = _interp_coords(
        res["log_ne"], res["log_te"], np.log10(ne_cm3), np.log10(Te_eV)
    )
    return 10.0 ** _interp_blend(res[key], ix, iy, fx, fy)


def metastable_ratio(S2, Q12, Q21, ne_cm3, tau_x_s=None):
    """Return r = N(2^3S)/N(1^1S) in quasi-static equilibrium.

    ``tau_x_s`` is the metastable transport loss time; ``None`` is the
    transport-free limit the tabulated coefficient assumes.
    """
    loss = Q21 + S2
    if tau_x_s is not None:
        loss = loss + 1.0 / (ne_cm3 * tau_x_s)
    return Q12 / loss


def scd_effective(S1, S2, r):
    """Ground-referenced effective ionization coefficient [cm^3/s]."""
    return (S1 + r * S2) / (1.0 + r)


def run_calibration(res):
    """Compare the transport-free solve to the unresolved table at the nodes.

    Returns ``(passed, record)``. The comparison is node-to-node with no
    interpolation on either side, so nothing here can be an interpolation
    artifact.
    """
    log_ne_u, log_te_u, stages_u = read_adf11(ADAS_DIR / "scd96_he.dat")
    if not (
        np.array_equal(log_ne_u, res["log_ne"])
        and np.array_equal(log_te_u, res["log_te"])
    ):
        raise ValueError(
            "scd96_he.dat and scd96r_he.dat are tabulated on different grids; "
            "the node-to-node calibration assumes one shared grid"
        )
    ne = 10.0 ** res["log_ne"]
    Te = 10.0 ** res["log_te"]
    S1 = 10.0 ** res["S1"]
    S2 = 10.0 ** res["S2"]
    Q12 = 10.0 ** res["Q12"]
    Q21 = 10.0 ** res["Q21"]
    unresolved = 10.0 ** stages_u[1]

    r = metastable_ratio(S2, Q12, Q21, ne[None, :])
    eff = scd_effective(S1, S2, r)
    dev = eff / unresolved - 1.0

    in_box = (
        (ne[None, :] >= CALIBRATION_NE_CM3[0])
        & (ne[None, :] <= CALIBRATION_NE_CM3[1])
        & (Te[:, None] >= CALIBRATION_TE_EV[0])
        & (Te[:, None] <= CALIBRATION_TE_EV[1])
    ) & np.ones_like(dev, dtype=bool)

    worst = float(np.abs(dev[in_box]).max())
    passed = worst <= CALIBRATION_TOL

    # The coronal column is the control: at the lowest tabulated density the
    # recombination feed and every other density-proportional term vanish, so
    # a solve with the right equations must agree there whatever it does in
    # the box. Reported so a failure can be told from a coding error.
    coronal = dev[:, 0]
    coronal_band = (Te >= 0.5) & (Te <= 1.5e4)
    coronal_worst = float(np.abs(coronal[coronal_band]).max())

    rows = []
    for i, t in enumerate(Te):
        for j, n in enumerate(ne):
            if in_box[i, j]:
                rows.append(
                    {
                        "ne_cm3": float(n),
                        "Te_eV": float(t),
                        "scd_unresolved_cm3_s": float(unresolved[i, j]),
                        "scd_eff_qss_cm3_s": float(eff[i, j]),
                        "deviation_frac": float(dev[i, j]),
                        "metastable_fraction": float(
                            r[i, j] / (1.0 + r[i, j])
                        ),
                        "stepwise_share": float(
                            r[i, j] * S2[i, j] / (S1[i, j] + r[i, j] * S2[i, j])
                        ),
                    }
                )

    record = {
        "tolerance": CALIBRATION_TOL,
        "ne_box_cm3": list(CALIBRATION_NE_CM3),
        "Te_box_eV": list(CALIBRATION_TE_EV),
        "n_nodes": int(in_box.sum()),
        "max_abs_deviation_frac": worst,
        "passed": bool(passed),
        "coronal_control_ne_cm3": float(ne[0]),
        "coronal_control_max_abs_deviation_frac": coronal_worst,
        "nodes": rows,
    }
    return passed, record, (ne, Te, dev, r, S1, S2, unresolved)


def print_calibration(record, grid):
    ne, Te, dev, r, S1, S2, unresolved = grid
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
    print("CALIBRATION GATE -- SCD_eff(thin QSS) / SCD_unresolved - 1, in %")
    print("  node-to-node, no interpolation; bar is "
          f"{CALIBRATION_TOL * 100:.0f} % at every node")
    print("  Te\\ne " + "".join(f"{ne[j]:10.1e}" for j in jn))
    for i in it:
        print(f"  {Te[i]:6.2f}" + "".join(f"{dev[i, j] * 100:10.2f}" for j in jn))
    print()
    share = r * S2 / (S1 + r * S2)
    print("  stepwise share of SCD_eff carried by the 2^3S channel, in %")
    print("  Te\\ne " + "".join(f"{ne[j]:10.1e}" for j in jn))
    for i in it:
        print(f"  {Te[i]:6.2f}" + "".join(f"{share[i, j] * 100:10.2f}" for j in jn))
    print()
    print(f"  nodes tested                : {record['n_nodes']}")
    print(f"  max |deviation| in the box  : "
          f"{record['max_abs_deviation_frac'] * 100:.3f} %")
    print(f"  coronal control (ne={record['coronal_control_ne_cm3']:.1e}, "
          f"0.5 eV-15 keV) : "
          f"{record['coronal_control_max_abs_deviation_frac'] * 100:.3f} %")
    print(f"  GATE: {'PASS' if record['passed'] else 'FAIL'}")


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

    r_thin = metastable_ratio(S2, Q12, Q21, ne)
    scd_thin = scd_effective(S1, S2, r_thin)

    # Correction (ii): metastable transport loss. tau_x is the column-crossing
    # time at the local gas temperature -- the diameter 2 Rp at the mean
    # thermal speed, the form that reproduces the registration's stated
    # ~3e-4 s at 300 K (18.415 cm plasma radius gives 2.93e-4 s).
    v_th = mean_thermal_speed_cm_s(T_gas)
    tau_x = 2.0 * Rp / v_th
    r_tx = metastable_ratio(S2, Q12, Q21, ne, tau_x_s=tau_x)
    scd_tx = scd_effective(S1, S2, r_tx)

    # Scoping diagnostic for correction (i), which is NOT applied: the 58.4 nm
    # line-centre optical depth across the plasma radius.
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
        "scd_thin": scd_thin,
        "r_transport": r_tx,
        "scd_transport": scd_tx,
        "ratio_transport": scd_tx / scd_thin,
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
    print(f"  sigma_0(584.334 A, Doppler, {state['Tn_K']:.0f} K) = "
          f"{table['sigma0_cm2']:.4e} cm^2   "
          f"(f = {HE_RESONANCE_F}, derived -- no archived value in-repo)")
    print(f"  mean He speed = {table['v_thermal_cm_s']:.4e} cm/s; "
          f"tau_x = 2Rp/v = {table['tau_x_s'][0]:.3e} s at Rp = "
          f"{table['Rp_cm'][0]:.3f} cm")
    print()
    print("PER-PORT")
    print("  port    z_cm     ne_cm3    Te_eV     nn_cm3   tau0_584   "
          "f(2^3S)  stepwise%  SCD_tx/SCD_thin")
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
            "tau0_584": float(table["tau0_584"][k]),
            "metastable_fraction": float(
                table["r_thin"][k] / (1.0 + table["r_thin"][k])
            ),
            "stepwise_share": float(table["stepwise_share"][k]),
            "scd_thin_cm3_s": float(table["scd_thin"][k]),
            "scd_transport_cm3_s": float(table["scd_transport"][k]),
            "ratio_transport": float(table["ratio_transport"][k]),
        }
        port_rows.append(row)
        print(f"  p{p:<4d} {row['z_cell_cm']:8.2f} {row['ne_cm3']:10.3e} "
              f"{row['Te_eV']:8.3f} {row['nn_cm3']:10.3e} "
              f"{row['tau0_584']:10.3f} {row['metastable_fraction']:9.2e} "
              f"{row['stepwise_share'] * 100:9.2f} "
              f"{row['ratio_transport']:15.5f}")

    print()
    print("OVER THE COLUMN")
    print(f"  cells                  : {table['cell'].size}")
    print(f"  tau0(584 nm)           : {table['tau0_584'].min():.3f} .. "
          f"{table['tau0_584'].max():.3f}")
    print(f"  stepwise share         : "
          f"{table['stepwise_share'].min() * 100:.2f} .. "
          f"{table['stepwise_share'].max() * 100:.2f} %")
    print(f"  SCD_transport/SCD_thin : "
          f"{table['ratio_transport'].min():.5f} .. "
          f"{table['ratio_transport'].max():.5f}")

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
        "scd_thin_cm3_s": [float(v) for v in table["scd_thin"]],
        "scd_transport_cm3_s": [float(v) for v in table["scd_transport"]],
        "ratio_transport": [float(v) for v in table["ratio_transport"]],
    }

    print()
    print("BRACKET: WITHHELD")
    print("  The calibration gate is what licenses a quotable bracket and it "
          "FAILED\n"
          f"  ({calibration['max_abs_deviation_frac'] * 100:.2f} % against a "
          f"{CALIBRATION_TOL * 100:.0f} % bar). The rows above are evidence, "
          "not a claim.")
    print("  Radiation trapping (correction i) is additionally NOT APPLIED: "
          "2^1P is\n"
          "  not a resolved population of the adf11 set, so a Holstein escape "
          "factor has\n"
          "  no coefficient to multiply. The tau0 column scopes how much it "
          "could matter.")

    record["bracket"] = {
        "emitted": False,
        "reason": (
            "calibration gate failed at "
            f"{calibration['max_abs_deviation_frac'] * 100:.2f} % against a "
            f"{CALIBRATION_TOL * 100:.0f} % bar; radiation trapping is not "
            "applicable to adf11-resolved coefficients"
        ),
    }

    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2))
        print(f"\nwrote {args.out}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
