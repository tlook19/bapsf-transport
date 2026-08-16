"""pd1owner: bin-(ii) completion reads. NO SOLVE.

Post-hoc over the recorded pd1/pd0 artifacts plus the pd1owner_heat_* ledgers:
  (1) e-fold OWNER decomposition of d ln N_col/dt over the build leg, per arm
      (channels are the pd0_endvent instrument's own recorded rows; the
      density convention is N_col = sum(n*Vp over active), volume-integrated,
      DISCLOSED as such -- the covcal instrument used <n>_active. Under this
      stance heating_anomalous_tail_ionization='off', so the beam-impact row
      is PRIMARY CSDA ionization only; walker ionization is structurally 0).
  (2) the disposal-aware HEATING channel table over the build window.
  (3) the escalation estimate: remaining gain if channel X were also removed
      (linear power attribution of the bulk-thermal channel, frozen
      trajectory, sinks held fixed -- an UPPER bound on the remaining bulk
      gain because the true SCD(Te) response is steeper than linear and the
      fixed sinks would then dominate; DISCLOSED estimate, not a run).
  (4) the f-threshold reach arithmetic: the shipped closed-form walk
      (u = W^2 falling linearly at 2*A(ne,Te) per cm, A from
      _coulomb_stopping_coefficient, thermalization floor max(1.5*Te, 0.1),
      cathode face reflecting below e*phi_c) evaluated per accepted step on
      the recorded cell states, birth at the measured QL site (n-argmax cell),
      compared against the MEASURED per-step vented/launched share.
  (5) the f025 venting time structure (which steps vent, and why).
  (6) the Q3 estimate: product-transport-movable heating shares and the
      walked-product reach at their own energies.

Constructs one LAPDSim1D for GEOMETRY ONLY (cell lengths); integrates nothing.
Writes only pd1owner_* artifacts. Read-only w.r.t. the repo.
"""
import json
import numpy as np

from cablp.funcs._beam_deposition import (
    _coulomb_stopping_coefficient,
    coulomb_stopping_eV_per_cm,
    he_mean_secondary_energy_eV,
    landau_branching_fraction,
    HE_E_STOP_EV,
)

ARMS = (
    ("pd1_f100", 1.00, "pd1_endvent_f100.npz", "pd1_ledger_f100.npz",
     "pd1owner_heat_f100.npz"),
    ("pd1_f050", 0.50, "pd1_endvent_f050.npz", "pd1_ledger_f050.npz",
     "pd1owner_heat_f050.npz"),
    ("pd1_f025", 0.25, "pd1_endvent_f025.npz", "pd1_ledger_f025.npz",
     "pd1owner_heat_f025.npz"),
    ("a1_baseline", None, "pd0_endvent_a1.npz", None, None),
)
HERE = "scripts/"


def sec(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def load(stem):
    return np.load(HERE + stem, allow_pickle=True) if stem else None


def main():
    # geometry: cell lengths only, no run
    import warnings
    warnings.simplefilter("ignore")
    from regime_r2_overlap_gate import build_config
    from cablp.solvers._sim1d import LAPDSim1D
    params, flags = build_config(20, False)
    sim = LAPDSim1D(params, flags)
    dz = np.asarray(sim.geometry.length_cm, float)
    del sim
    CATH, COLL = 2, 41
    L_col = float(dz[CATH:COLL + 1].sum())
    print(f"geometry: nx=20 grid, {dz.size} cells, cathode cell {CATH}, "
          f"collector cell {COLL}, cathode->collector path {L_col:.1f} cm")

    data = {}
    for label, f, env, led, heat in ARMS:
        d = {"f": f, "env": load(env), "led": load(led), "heat": load(heat)}
        d["meta"] = json.loads(str(d["env"]["meta"]))
        data[label] = d

    # ---------------- (1) e-fold owner decomposition --------------------
    sec("(1) E-FOLD OWNER DECOMPOSITION, d ln N_col/dt over the build leg")
    print("  gamma_c(t) = rate_c(t)/N_col(t); leg log-gain = trapz(gamma_c dt);")
    print("  channels are the pd0_endvent instrument's recorded rows.")
    for label, d in data.items():
        e = d["env"]
        i0 = int(d["meta"]["i_on"])
        t = np.asarray(e["t"], float)
        N = np.asarray(e["N_col"], float)
        sl = slice(i0, None)
        obs = float(np.log(N[-1] / N[i0]))
        T = t[-1] - t[i0]
        chans = (
            ("beam PRIMARY (CSDA) ionization", np.asarray(e["birth_beam"], float)),
            ("bulk/thermal ionization_birth", np.asarray(e["birth_bulk"], float)),
            ("collector-end vent (sink)", -np.asarray(e["vent_coll_p"], float)),
            ("cathode-face absorption (sink)", -np.asarray(e["vent_cath_p"], float)),
            ("anode-mesh collection (sink)", -np.asarray(e["anode_p"], float)),
            ("recombination (sink)", -np.asarray(e["recomb_p"], float)),
        )
        print(f"\n  --- {label} (leg steps {i0}..{t.size - 1}, "
              f"[{t[i0]:.4g}, {t[-1]:.4g}] s) ---")
        print(f"  {'channel':34s} {'dln N':>9s} {'share':>9s} "
              f"{'tau if alone [us]':>18s}")
        tot = 0.0
        for nm, r in chans:
            g = float(np.trapezoid(r[sl] / N[sl], t[sl]))
            tot += g
            ta = T / g * 1e6 if g != 0.0 else float("inf")
            print(f"  {nm:34s} {g:+9.4f} {100 * g / obs:+8.2f} % {ta:18.2f}")
        print(f"  {'SUM (reconstructed)':34s} {tot:+9.4f} {100 * tot / obs:+8.2f} %")
        print(f"  {'OBSERVED ln(N_end/N_on)':34s} {obs:+9.4f}   "
              f"(closure {100 * (tot - obs) / obs:+.2f} %, advective row not "
              "recorded by this instrument)")

    # ---------------- (2) heating channel table -------------------------
    sec("(2) DISPOSAL-AWARE HEATING CHANNELS, build window (onset->end)")
    tables = {}
    for label, d in data.items():
        if d["heat"] is None:
            continue
        e, S = d["env"], np.asarray(d["heat"]["scalars"], float)
        H = d["heat"]
        i0 = int(d["meta"]["i_on"])
        t = S[:, 0]
        sl = slice(i0, None)

        def integ(y):
            return float(np.trapezoid(np.asarray(y, float)[sl], t[sl]))

        anom = integ(H["heating_anomalous_erg_s"].sum(axis=1))
        coul = integ(H["heating_coulomb_erg_s"].sum(axis=1))
        seco = integ(H["heating_secondary_erg_s"].sum(axis=1))
        term = integ(H["heating_terminal_erg_s"].sum(axis=1))
        rad = integ(H["radiated_erg_s"].sum(axis=1))
        cost = integ(H["ionization_cost_erg_s"].sum(axis=1))
        launched = integ(S[:, 3])
        vented = integ(S[:, 4] + S[:, 5])
        P_QL = anom + vented  # tail ionization/radiation identically 0 here
        retained = P_QL - launched
        walkth = launched - vented
        cst = float(np.trapezoid(np.asarray(e["cst_E"], float)[sl], t[sl]))
        cool = float(np.trapezoid(np.asarray(e["cool_E"], float)[sl], t[sl]))
        heat_net = coul + anom + seco + term
        tables[label] = dict(anom=anom, coul=coul, seco=seco, term=term,
                             retained=retained, walkth=walkth, vented=vented,
                             P_QL=P_QL, heat_net=heat_net, cst=cst, cool=cool)
        print(f"\n  --- {label} ---   [erg over the build window]")
        print(f"  P_QL (extracted bank)            {P_QL:.4e}   "
              f"launched {launched / P_QL:.3f}  vented {vented / P_QL:.3f}")
        print(f"  {'plasma-heating channel':38s} {'erg':>11s} "
              f"{'of net heating':>15s} {'of P_QL':>9s}")
        for nm, v in (("QL retained local (1-f_L share)", retained),
                      ("QL walker in-column thermalization", walkth),
                      ("  [anomalous delivered, sum]", anom),
                      ("primary Coulomb drag (CSDA collisional)", coul),
                      ("primary terminal residual (local)", term),
                      ("secondary <W_sec> (local, product ch.)", seco)):
            print(f"  {nm:38s} {v:11.4e} {100 * v / heat_net:14.2f} % "
                  f"{100 * v / P_QL:8.2f} %")
        print(f"  {'NET BEAM HEATING (sum of 4)':38s} {heat_net:11.4e}")
        print(f"  non-heating for reference: radiated {rad:.3e}  "
              f"ionization_cost {cost:.3e}  cathode_terms {cst:.3e}  "
              f"cooling {cool:.3e}")

    # ---------------- (3) escalation estimate ---------------------------
    sec("(3) WHAT REMAINS IF CHANNEL X WERE ALSO VENTED (estimate, method "
        "stated in the docstring)")
    for label in ("pd1_f100",):
        d, tb = data[label], tables[label]
        e = d["env"]
        i0 = int(d["meta"]["i_on"])
        t = np.asarray(e["t"], float)
        N = np.asarray(e["N_col"], float)
        sl = slice(i0, None)
        T = t[-1] - t[i0]
        g_beam = float(np.trapezoid(
            np.asarray(e["birth_beam"], float)[sl] / N[sl], t[sl])) / T
        g_bulk = float(np.trapezoid(
            np.asarray(e["birth_bulk"], float)[sl] / N[sl], t[sl])) / T
        g_sink = -float(np.trapezoid(
            (np.asarray(e["vent_coll_p"], float)
             + np.asarray(e["vent_cath_p"], float)
             + np.asarray(e["anode_p"], float)
             + np.asarray(e["recomb_p"], float))[sl] / N[sl], t[sl])) / T
        g_tot = g_beam + g_bulk + g_sink
        print(f"\n  {label}: mean leg rates [1/s]: primary {g_beam:.4e}  "
              f"bulk {g_bulk:.4e}  sinks {g_sink:.4e}  net {g_tot:.4e} "
              f"(tau {1e6 / g_tot:.2f} us)")
        loss = -g_sink
        for nm, share in (
            ("QL retained local", tb["retained"] / tb["heat_net"]),
            ("QL walker in-column thermalization", tb["walkth"] / tb["heat_net"]),
            ("BOTH anomalous-delivered channels", tb["anom"] / tb["heat_net"]),
            ("anomalous + terminal", (tb["anom"] + tb["term"]) / tb["heat_net"]),
        ):
            g_rem = g_beam + g_bulk * (1.0 - share) + g_sink
            ratio = g_rem / loss
            print(f"  remove {nm:38s} (heating share {100 * share:5.1f} %): "
                  f"remaining gain {g_rem:.4e} 1/s "
                  f"(tau {1e6 / g_rem if g_rem > 0 else float('inf'):8.2f} us; "
                  f"gain/all-surface-loss ~ {ratio:.2f})")
        print(f"  [all-surface loss rate used for the ratio: {loss:.4e} 1/s "
              "(same-leg mean; the verdict table's medians are the record)]")

    # ---------------- (4) reach arithmetic vs measured venting ----------
    sec("(4) f-THRESHOLD: shipped-walk reach arithmetic vs measured venting")
    print("  Closed form (fast_electron, q=2): W_exit^2 = E_tail^2 - "
        "2*sum(A(ne,Te)*dz) along the path;")
    print("  thermalized where W falls below max(1.5*Te, 0.1); cathode face "
        "reflects below e*phi_c, so")
    print("  EVERY vent is through the collector end (measured: "
        "end_loss_tail_low == 0.0 on all arms).")
    for label in ("pd1_f100", "pd1_f050", "pd1_f025"):
        d = data[label]
        e, S = d["env"], np.asarray(d["heat"]["scalars"], float)
        f = d["f"]
        t = S[:, 0]
        phic = S[:, 2]
        launched = S[:, 3]
        vented = S[:, 4] + S[:, 5]
        n_c = np.asarray(e["cell_n"], float)
        Te_c = np.asarray(e["cell_Te"], float)
        ns = t.size
        pred = np.full(ns, np.nan)
        jstars = np.zeros(ns, int)
        for i in range(ns):
            if launched[i] <= 0 or not np.isfinite(phic[i]):
                continue
            E = f * phic[i]
            ne_i, Te_i = n_c[i], Te_c[i]
            A = _coulomb_stopping_coefficient(
                np.maximum(ne_i[CATH:COLL + 1], 1.0),
                np.maximum(Te_i[CATH:COLL + 1], 0.1), "fast_electron")
            flo = np.maximum(1.5 * Te_i[CATH:COLL + 1], 0.1)
            dzw = dz[CATH:COLL + 1]
            j = int(np.argmax(ne_i)) - CATH
            j = min(max(j, 0), COLL - CATH)
            if E <= flo[j]:
                pred[i] = 0.0
                jstars[i] = j + CATH
                continue

            def walk(path):
                u = E * E
                for k in path:
                    u -= 2.0 * A[k] * dzw[k]
                    if u <= flo[k] * flo[k]:
                        return 0.0
                return np.sqrt(max(u, 0.0)) / E

            w_coll = walk(range(j, COLL - CATH + 1))
            w_refl = walk(list(range(j, -1, -1))
                          + list(range(0, COLL - CATH + 1)))
            pred[i] = 0.5 * (w_coll + w_refl)
            jstars[i] = j + CATH
        meas = np.divide(vented, launched, out=np.full(ns, np.nan),
                         where=launched > 0)
        i0 = int(d["meta"]["i_on"])
        ok = np.isfinite(pred) & np.isfinite(meas) & (np.arange(ns) >= i0)
        print(f"\n  --- {label} (f={f}) --- birth at n-argmax cell "
              f"(median cell {int(np.median(jstars[ok]))})")
        print(f"  {'t [s]':>10s} {'E_tail':>7s} {'ne@j*':>10s} {'Te@j*':>6s} "
              f"{'floor':>6s} {'pred vent/launch':>16s} {'MEASURED':>9s}")
        for i in np.unique(np.linspace(i0, ns - 1, 8).astype(int)):
            j = jstars[i]
            print(f"  {t[i]:10.3e} {f * phic[i]:7.1f} {n_c[i][j]:10.3e} "
                  f"{Te_c[i][j]:6.1f} {max(1.5 * Te_c[i][j], 0.1):6.1f} "
                  f"{pred[i]:16.3f} {meas[i]:9.3f}")
        print(f"  build-window medians: predicted {np.median(pred[ok]):.3f}  "
              f"measured {np.median(meas[ok]):.3f}")
        # reach numbers at the leg endpoints for the report
        for i in (i0, ns - 1):
            E = f * phic[i]
            j = jstars[i]
            ne_j = max(n_c[i][j], 1.0)
            Te_j = max(Te_c[i][j], 0.1)
            A1 = float(_coulomb_stopping_coefficient(
                ne_j, Te_j, "fast_electron")[0])
            flo = max(1.5 * Te_j, 0.1)
            R = max(E * E - flo * flo, 0.0) / (2.0 * A1)
            print(f"    t={t[i]:.3e}: E_tail {E:.1f} eV, ne(j*) {ne_j:.2e}, "
                  f"uniform-state reach {R / 100:.1f} m vs column {L_col / 100:.1f} m "
                  f"(floor {flo:.1f} eV)")

    # ---------------- (5) f025 venting time structure -------------------
    sec("(5) f025: WHICH steps vent (the median-0 vs integral-0.046 anomaly)")
    d = data["pd1_f025"]
    S = np.asarray(d["heat"]["scalars"], float)
    t, phic, launched = S[:, 0], S[:, 2], S[:, 3]
    vented = S[:, 4] + S[:, 5]
    live = launched > 0
    vfrac = np.divide(vented, launched, out=np.zeros(t.size),
                      where=live)
    nz = vfrac > 1e-6
    print(f"  steps with any venting: {int(nz.sum())} of {t.size} "
          f"({100 * nz.sum() / t.size:.1f} % -- hence per-step median 0)")
    if nz.any():
        idx = np.flatnonzero(nz)
        E_int = np.trapezoid(vented, t)
        brk = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[brk + 1]]
        ends = np.r_[idx[brk], idx[-1]]
        print("  contiguous venting EPISODES:")
        for a, b in zip(starts, ends):
            share = np.trapezoid(vented[a:b + 1], t[a:b + 1]) / E_int
            vl = np.max(np.divide(vented[a:b + 1], launched[a:b + 1],
                                  out=np.zeros(b + 1 - a),
                                  where=launched[a:b + 1] > 0))
            print(f"    steps {a:4d}..{b:4d} [{t[a]:.3e},{t[b]:.3e}] s: "
                  f"{100 * share:5.1f} % of all vented energy, E_tail "
                  f"[{0.25 * phic[a:b + 1].min():.1f},"
                  f"{0.25 * phic[a:b + 1].max():.1f}] eV, "
                  f"max vent/launch {vl:.3f}")
        n_c = np.asarray(d["env"]["cell_n"], float)
        for i in (idx[0], idx[len(idx) // 2], idx[-1],
                  min(idx[-1] + 50, t.size - 1)):
            print(f"    t={t[i]:.3e}: E_tail={0.25 * phic[i]:6.1f} eV  "
                  f"n_max={n_c[i].max():.2e}  vent/launch={vfrac[i]:.3f}")

    # ---------------- (6) Q3: product-transport axis --------------------
    sec("(6) Q3: what the product-transport axis could move under the branch")
    tb = tables["pd1_f100"]
    d = data["pd1_f100"]
    e = d["env"]
    S = np.asarray(d["heat"]["scalars"], float)
    i0 = int(d["meta"]["i_on"])
    movable = tb["seco"] + tb["term"]
    print(f"  pd1_f100 build window: transport-movable heating (secondary + "
        f"terminal) = {movable:.4e} erg")
    print(f"    = {100 * movable / tb['heat_net']:.2f} % of net beam heating "
          f"(secondary {100 * tb['seco'] / tb['heat_net']:.2f} %, terminal "
          f"{100 * tb['term'] / tb['heat_net']:.2f} %)")
    # energies of the walked populations, and their reach at leg densities
    t = S[:, 0]
    phic = S[:, 2]
    n_c = np.asarray(e["cell_n"], float)
    Te_c = np.asarray(e["cell_Te"], float)
    for i in (i0, t.size - 1):
        Ep = phic[i]
        Wsec = he_mean_secondary_energy_eV(Ep)
        j = int(np.argmax(n_c[i]))
        ne_j, Te_j = max(n_c[i][j], 1.0), max(Te_c[i][j], 0.1)
        A1 = float(_coulomb_stopping_coefficient(
            ne_j, Te_j, "fast_electron")[0])
        flo = max(1.5 * Te_j, 0.1)
        for nm, W in (("secondary <W_sec>", Wsec),
                      ("terminal residual (<= E_stop)", HE_E_STOP_EV)):
            R = max(W * W - flo * flo, 0.0) / (2.0 * A1)
            print(f"    t={t[i]:.3e} (ne {ne_j:.2e}, floor {flo:.1f} eV): "
                  f"{nm} = {W:.1f} eV -> reach {R / 100:.2f} m "
                  f"vs column {L_col / 100:.1f} m")


if __name__ == "__main__":
    main()
