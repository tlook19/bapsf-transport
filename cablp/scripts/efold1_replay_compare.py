"""efold1: bit-exactness comparison, covcal_f2_shot1 (HEAD 4e4dd27) vs the
honest-clock replay of the same recipe at HEAD 57ba63e.

Compares at raw uint64 (np.ndarray.view) over every save the replay's shorter
window shares with the original, on every field the e-fold-owner decomposition
(covcal_efold_read.py) actually consumes: the density, the active-cell mean it
forms, every rhs_terms n row, the walker events, and the circuit/source
diagnostics. Read-only; no solve.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent

REF = "covcal_f2_shot1"
NEW = "efold1_replay_shot1"

# Exactly the fields covcal_efold_read.load() reads.
FIELDS_2D = ("n", "nn", "Te")
CDIAG = ("beam_tail_ionization_events_per_s", "beam_tail_power_W",
         "beam_tail_ionization_cost_W", "beam_tail_radiated_W",
         "beam_heat_anomalous_W", "circuit_I_loop", "circuit_V_dis_step",
         "source_I_e", "source_I_eth", "source_I_eth_star", "source_phi_c",
         "source_phi_c_at_cap", "source_P_prim", "source_l_b",
         "source_beam_bypass_fraction", "coverage_fraction")


def bits(a):
    a = np.ascontiguousarray(np.asarray(a, float))
    return a.view(np.uint64)


def cmp(label, a, b, out):
    ea, eb = bits(a), bits(b)
    if ea.shape != eb.shape:
        out.append((label, "SHAPE", f"{ea.shape} vs {eb.shape}"))
        return
    nbad = int((ea != eb).sum())
    if nbad:
        d = np.abs(np.asarray(a, float) - np.asarray(b, float))
        out.append((label, "DIFF", f"{nbad}/{ea.size} words; max |d| = {d.max():.6e}"))
    else:
        out.append((label, "BIT-EXACT", f"{ea.size} words"))


def main():
    pr, pn = HERE / f"{REF}.h5", HERE / f"{NEW}.h5"
    for p in (pr, pn):
        if not p.exists():
            sys.exit(f"ABSENT: {p}")
    with h5py.File(pr, "r") as fr, h5py.File(pn, "r") as fn:
        tr = np.asarray(fr["time"], float)
        tn = np.asarray(fn["time"], float)
        k = tn.size
        print(f"== efold1_replay_compare: {REF} (4e4dd27) vs {NEW} (57ba63e)")
        print(f"   reference saves {tr.size}, replay saves {k}; comparing the "
              f"first {k} shared saves")
        print(f"   reference t[:{k}] = {tr[:k]}")
        print(f"   replay    t[:{k}] = {tn[:k]}")
        print(f"   time lattice bit-exact: "
              f"{bool((bits(tr[:k]) == bits(tn[:k])).all())}")
        pa = json.loads(fr.attrs["params_json"])
        pb = json.loads(fn.attrs["params_json"])
        shared = sorted(set(pa) & set(pb))
        moved = [x for x in shared if pa[x] != pb[x]]
        print(f"   shared param keys {len(shared)}; VALUE MOVES among them: "
              f"{moved if moved else 'NONE'}")
        print(f"   compiled_kernels: ref={fr.attrs.get('compiled_kernels')!r} "
              f"replay={fn.attrs.get('compiled_kernels')!r}")

        out = []
        for nm in FIELDS_2D:
            cmp(nm, fr[nm][:k], fn[nm][:k], out)
        for nm in sorted(fr["rhs_terms"].keys()):
            if nm in fn["rhs_terms"]:
                cmp(f"rhs_terms/{nm}/n", fr["rhs_terms"][nm]["n"][:k],
                    fn["rhs_terms"][nm]["n"][:k], out)
        cmp("total_rhs/n", fr["total_rhs"]["n"][:k], fn["total_rhs"]["n"][:k], out)
        for nm in CDIAG:
            if nm in fr["cathode_diagnostics"] and nm in fn["cathode_diagnostics"]:
                cmp(f"cathode_diagnostics/{nm}", fr["cathode_diagnostics"][nm][:k],
                    fn["cathode_diagnostics"][nm][:k], out)

    print(f"\n   {'field':52s} {'verdict':10s} detail")
    for lab, v, det in out:
        print(f"   {lab:52s} {v:10s} {det}")
    bad = [o for o in out if o[1] != "BIT-EXACT"]
    print(f"\n   {len(out)-len(bad)}/{len(out)} arrays BIT-EXACT at raw uint64")
    print(f"   VERDICT: {'BIT-EXACT ON EVERY COMPARED ARRAY' if not bad else 'DIFFERENCES PRESENT'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
