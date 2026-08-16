"""pd1owner: the late SELF-TRAPPING episode on f100/f050. NO SOLVE.

Post-hoc over pd1owner_heat_*.npz + pd1_endvent_*.npz. Finding: the "walker
in-column thermalization" share (launched - vented) is NOT distributed
en-route slowing -- ~97-99 % of it is a single late episode in which the QL
bank detaches into the far column, the walkers' own deposited heat raises the
local thermalization floor max(1.5*Te, 0.1) ABOVE E_tail = f*phi_c, and every
subsequently launched walker is born below the floor and thermalizes in its
birth cell (vent/launch drops to exactly 0 while launch power peaks). The
bank cell then migrates cathode-ward (31 -> 18 on f100) as the front heats
its neighbours; when bank-cell Te falls back below E_tail/1.5 the venting
resumes in one step. Same floor arithmetic that owns the f=0.25 collapse.
"""
import numpy as np

for arm, f in (("f100", 1.0), ("f050", 0.5)):
    S = np.asarray(np.load(f"scripts/pd1owner_heat_{arm}.npz",
                           allow_pickle=True)["scalars"], float)
    H = np.load(f"scripts/pd1owner_heat_{arm}.npz", allow_pickle=True)
    e = np.load(f"scripts/pd1_endvent_{arm}.npz", allow_pickle=True)
    anom = H["heating_anomalous_erg_s"]
    Te = np.asarray(e["cell_Te"], float)
    t, phic, lau = S[:, 0], S[:, 2], S[:, 3]
    ven = S[:, 4] + S[:, 5]
    walkth = lau - ven
    E_wt = np.trapezoid(walkth, t)
    vl = np.divide(ven, lau, out=np.full(t.size, np.nan), where=lau > 0)
    ep = (vl < 0.5) & np.isfinite(vl)
    idx = np.flatnonzero(ep)
    E_ep = np.trapezoid(np.where(ep, walkth, 0.0), t)
    print(f"\n== {arm}: walker-thermalization integral {E_wt:.4e} erg;"
          f" self-trap steps (vent/launch<0.5): {idx.size}")
    if idx.size:
        print(f"   episode t [{t[idx[0]]:.4e}, {t[idx[-1]]:.4e}] s carries "
              f"{100 * E_ep / E_wt:.1f} % of the channel")
        for i in (idx[0] - 5, idx[0], idx[0] + 15,
                  (idx[0] + idx[-1]) // 2, idx[-1], idx[-1] + 1):
            j = int(np.argmax(anom[i]))
            E = f * phic[i]
            flo = 1.5 * Te[i][j]
            print(f"   step {i:4d} t={t[i]:.4e}: bank cell {j:2d} "
                  f"({anom[i][j]:.2e} erg/s)  Te(bank) {Te[i][j]:6.1f} eV  "
                  f"floor {flo:6.1f} vs E_tail {E:6.1f} eV  "
                  f"-> {'TRAPPED' if flo > E else 'vents':7s} "
                  f"(measured vent/launch {vl[i]:.3f})")
