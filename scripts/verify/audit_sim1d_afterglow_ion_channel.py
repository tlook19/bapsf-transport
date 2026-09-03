"""Read-only afterglow ion-channel ledger audit (warm-ion hypothesis).

Tom's hypothesis (2026-07-29): in the MACHINE, mid-column neutral depletion
starves charge exchange, ions stay warm, the e->i drain stalls, and the
mid-machine electron decay slows relative to the model.  This audit reads the
MODEL side of that chain from a saved production artifact — no reruns:

  (a) what dominates the ion energy ledger at each scored port in the
      afterglow (is CX the ion cooling channel, and how fast is it?),
  (b) what dominates the electron ledger (is e->i the drain that sets the
      mid-column Te decay?),
  (c) Ti(z,t), Te-Ti, and the coupled timescales, and
  (d) a no-rerun scaling read: the CX cooling time if the model's port nn
      were divided by the measured mid-column depletion bracket (x9-x20).

Usage:
  python audit_sim1d_afterglow_ion_channel.py --h5 es1_prod_25ms_nx240.h5 \
      [--early-ms 1.0] [--output out.txt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

PORTS_Z_CM = {11: 470.05, 21: 789.55, 29: 1045.15, 41: 1428.55, 50: 1716.1}
EV_PER_ERG = 6.241509e11
QE_ERG_PER_EV = 1.602176634e-12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--early-ms", type=float, default=1.0,
                    help="early-afterglow averaging window length (ms)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out_lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out_lines.append(s)

    with h5py.File(args.h5, "r") as f:
        t = np.asarray(f["time"])
        z = np.asarray(f["geometry/z_cm"])
        Te = np.asarray(f["Te"])
        Ti = np.asarray(f["Ti"])
        n = np.asarray(f["n"])
        nn = np.asarray(f["nn"])
        Ei = np.asarray(f["Ei"])  # erg/cm3
        Ee = np.asarray(f["Ee"])

        phases = [p.decode() for p in f["phase_events/phase"][:]]
        ptimes = np.asarray(f["phase_events/time"])
        t_ag = ptimes[phases.index("afterglow")]
        t_pa = (ptimes[phases.index("post_afterglow")]
                if "post_afterglow" in phases else t[-1])

        ion_terms = {k: np.asarray(v) for k, v in
                     f["ion_energy_terms_W_cm3"].items()}
        ele_terms = {k: np.asarray(v) for k, v in
                     f["electron_energy_terms_W_cm3"].items()}

    port_cells = {p: int(np.argmin(np.abs(z - zc)))
                  for p, zc in PORTS_Z_CM.items()}

    early = (t >= t_ag) & (t <= t_ag + args.early_ms * 1e-3)
    full = (t >= t_ag) & (t <= t_pa)

    emit(f"artifact: {args.h5}")
    emit(f"afterglow: {t_ag*1e3:.3f} -> {t_pa*1e3:.3f} ms; "
         f"early window = first {args.early_ms:.2f} ms "
         f"({early.sum()} saves; full window {full.sum()} saves)")
    emit()

    def budget(terms: dict[str, np.ndarray], cell: int,
               mask: np.ndarray, top: int = 6) -> list[tuple[str, float]]:
        rows = [(k, float(np.mean(v[mask, cell]))) for k, v in terms.items()]
        rows = [(k, val) for k, val in rows if abs(val) > 0.0]
        rows.sort(key=lambda kv: -abs(kv[1]))
        return rows[:top]

    for p, cell in sorted(port_cells.items()):
        emit(f"=== port {p} (z = {PORTS_Z_CM[p]:.1f} cm, cell {cell}) ===")
        for label, mask in (("early", early), ("full", full)):
            Te_m = float(np.mean(Te[mask, cell]))
            Ti_m = float(np.mean(Ti[mask, cell]))
            n_m = float(np.mean(n[mask, cell]))
            nn_m = float(np.mean(nn[mask, cell]))
            emit(f"  [{label}] Te {Te_m:.3f} eV  Ti {Ti_m:.3f} eV  "
                 f"Te-Ti {Te_m-Ti_m:+.3f} eV  n {n_m:.3e}  nn {nn_m:.3e}  "
                 f"nn/n {nn_m/max(n_m,1e-30):.2f}")

        emit("  ion ledger (mean W/cm3, early | full):")
        rows_e = dict(budget(ion_terms, cell, early, top=99))
        rows_f = dict(budget(ion_terms, cell, full, top=99))
        keys = sorted(set(rows_e) | set(rows_f),
                      key=lambda k: -abs(rows_e.get(k, 0.0)))
        for k in keys[:7]:
            emit(f"    {k:32s} {rows_e.get(k,0.0):+.3e} | "
                 f"{rows_f.get(k,0.0):+.3e}")

        emit("  electron ledger (mean W/cm3, early | full):")
        rows_e = dict(budget(ele_terms, cell, early, top=99))
        rows_f = dict(budget(ele_terms, cell, full, top=99))
        keys = sorted(set(rows_e) | set(rows_f),
                      key=lambda k: -abs(rows_e.get(k, 0.0)))
        for k in keys[:7]:
            emit(f"    {k:32s} {rows_e.get(k,0.0):+.3e} | "
                 f"{rows_f.get(k,0.0):+.3e}")

        # Timescales (early window): tau = energy content / |channel power|.
        Ei_m = float(np.mean(Ei[early, cell]))  # erg/cm3
        Ee_m = float(np.mean(Ee[early, cell]))
        p_cx = float(np.mean(ion_terms["ion_charge_exchange"][early, cell]))
        p_ei_i = float(np.mean(ion_terms["ei_exchange"][early, cell]))
        p_ei_e = float(np.mean(ele_terms["ei_exchange"][early, cell]))
        W_per_erg = 1e-7  # 1 erg/s = 1e-7 W
        def tau_ms(E_erg_cm3: float, P_W_cm3: float) -> float:
            if abs(P_W_cm3) < 1e-30:
                return float("inf")
            return (E_erg_cm3 * W_per_erg / abs(P_W_cm3)) * 1e3

        emit(f"  timescales (early): tau_CX(ion) {tau_ms(Ei_m, p_cx):.2f} ms"
             f"   tau_ei(ion gain) {tau_ms(Ei_m, p_ei_i):.2f} ms"
             f"   tau_ei(elec drain) {tau_ms(Ee_m, p_ei_e):.2f} ms")

        # (d) depletion scaling: CX power ~ nn (rate linear in nn at fixed
        # Ti, sigma-v).  If the machine's mid nn is x9-x20 below the model:
        for div in (9.0, 20.0):
            emit(f"    if nn/{div:.0f}: tau_CX -> "
                 f"{tau_ms(Ei_m, p_cx / div):.2f} ms")
        emit()

    # Column view: where is the model's nn trough in the afterglow?
    emit("=== column snapshot at afterglow open + 0.5 ms ===")
    i0 = int(np.argmin(np.abs(t - (t_ag + 0.5e-3))))
    interior = slice(1, -1)
    zz = z[interior]
    emit(f"  t = {t[i0]*1e3:.3f} ms")
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        j = int(np.argmin(np.abs(zz - frac * zz[-1]))) + 1
        emit(f"  z {z[j]:7.1f} cm: nn {nn[i0,j]:.3e}  n {n[i0,j]:.3e}  "
             f"Te {Te[i0,j]:.3f}  Ti {Ti[i0,j]:.3f}")

    if args.output:
        Path(args.output).write_text("\n".join(out_lines) + "\n")
        print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
