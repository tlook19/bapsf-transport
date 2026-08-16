"""pd0 READ A/B instrument: end-vent budget + per-cell state record per arm.

WRAPPER around the efold1_traj accepted-step loop (regime_r2_overlap_gate
build_config + recertdiag2_traj.apply_set, sim.run(t_end, max_steps=1), same
exception handling). SINGLE-DELTA ON INSTRUMENTATION ONLY: after each ACCEPTED
step it evaluates, on the accepted state, the named per-channel conservative
RHS terms via the solver's own diagnostic wrappers with an explicitly-passed
cathode solve obtained with update_cache=False -- verified non-mutating
(cathode_x0 / beam_cross unchanged; and the per-step (t, I_loop, n_max) trace
is compared against the arm's recorded npz at the end, printed as the
IDENTITY CONTROL).

Channels recorded per accepted step (particles/s and erg/s, cell-resolved
where it matters):
  * characteristic_boundary (sources.characteristic_boundary_rhs via
    solver.characteristic_boundary_rhs): the Bohm ghost-flux sink at the two
    absorbing faces -- face 2 (cathode disc, live cell 2) and face 42
    (collector, live cell 41). THE END-VENT SINK.
  * anode_collection (sources.anode_collection_rhs): Bohm collection at the
    anode mesh (live cells 6,7 in this geometry).
  * ionization_birth / recombination_* (reaction_rhs_terms)
  * beam_ionization_birth / beam_power_deposition etc. (beam_ionization_rhs_terms)
  * cathode_source_terms .rhs energy rows; electron_cooling_rhs_terms rows.
Also per-cell n, Te, Ti, nn snapshots (READ B feedstock) and phi_c, I_eth*.

Writes only pd0_* artifacts. Read-only w.r.t. the repo.
"""
import argparse, json, sys, time, warnings
import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.core.state import derive_state
from recertdiag2_traj import apply_set

I_WALL_TIME = 1.0e-6
ONSET_FACTOR = 10.0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--t-target", type=float, default=1.0e-4)
    p.add_argument("--set", dest="sets", nargs="*", default=())
    p.add_argument("--npz", required=True)
    p.add_argument("--ref-npz", default=None,
                   help="recorded arm npz for the trace identity control")
    args = p.parse_args(argv)
    warnings.simplefilter("ignore")

    from regime_r2_overlap_gate import build_config
    params, flags = build_config(args.nx, False)
    applied = apply_set(params, flags, args.sets)

    from cablp.funcs import _kernels
    kid = str(getattr(_kernels.COMPILED_KERNELS, "KERNEL_ID",
                      _kernels.COMPILED_KERNELS))
    sim = LAPDSim1D(params, flags)
    g = sim.geometry
    Vp = np.asarray(g.plasma_volume_cm3, dtype=float)
    active = np.asarray(g.plasma_active, bool)
    roles = np.asarray(g.cell_role)
    cells = g.cells
    print(f"== pd0_endvent_traj label={args.label} nx={args.nx} "
          f"t_target={args.t_target:g}")
    print(f"   deltas via --set: {applied}")
    print(f"   kernels: {kid}")
    absorbing = np.flatnonzero(np.asarray(g.plasma_absorbing, bool))
    live = [int(g.plasma_face_live_cell[f]) for f in absorbing]
    print(f"   absorbing faces {list(absorbing)} -> live cells {live} "
          f"roles {[str(roles[c]) for c in live]}")
    cath_cell = live[0]; coll_cell = live[-1]

    rec = {k: [] for k in (
        "t", "dt", "I_loop", "phi_c", "Ieth",
        "vent_coll_p", "vent_cath_p", "vent_coll_E", "vent_cath_E",
        "anode_p", "anode_E",
        "birth_bulk", "birth_beam", "recomb_p",
        "beam_Edep", "cst_E", "cool_E",
        "N_col", "E_col", "N_cath", "n_cath")}
    cellrec = {k: [] for k in ("n", "Te", "Ti", "nn")}
    t_prev = 0.0
    t0 = time.time()
    refusal = None
    while sim._time < args.t_target:
        try:
            sim.run(t_end=args.t_target, max_steps=1)
        except RuntimeError:
            pass
        except Exception as err:
            refusal = f"{type(err).__name__}: {err}"
            break
        t = float(sim._time)
        st = sim.state
        d = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
        solve = sim.solve_cathode_boundary(state=st, time=t, update_cache=False)
        cb = sim.characteristic_boundary_rhs(state=st, cathode_solve=solve, time=t)
        ac = sim.anode_collection_rhs(state=st, cathode_solve=solve, time=t)
        cst = sim.cathode_source_terms(state=st, cathode_solve=solve, time=t).rhs
        rx = sim.reaction_rhs_terms(state=st)
        bt = sim.beam_ionization_rhs_terms(state=st, cathode_solve=solve, time=t)
        cool = sim.electron_cooling_rhs_terms(state=st)

        E0 = G0 = float("nan")
        br = getattr(solve, "beam_result", None)
        res = getattr(br, "result", None)
        if res is not None:
            E0 = float(res.phi_c); G0 = float(res.I_eth_star)

        n = np.asarray(st.n, float)
        Ecol = float(((np.asarray(st.Ee) + np.asarray(st.Ei)) * Vp)[active].sum())
        cbn = np.asarray(cb.n); cbE = np.asarray(cb.Ee) + np.asarray(cb.Ei)
        acn = np.asarray(ac.n); acE = np.asarray(ac.Ee) + np.asarray(ac.Ei)
        rxn = np.asarray(rx["ionization_birth"].n)
        rec_p = (np.asarray(rx["recombination_rad_loss"].n)
                 + np.asarray(rx["recombination_3b_loss"].n))
        btn = np.asarray(bt["beam_ionization_birth"].n)
        btE = np.asarray(bt["beam_power_deposition"].Ee)
        cstE = np.asarray(cst.Ee) + np.asarray(cst.Ei)
        coolE = sum(np.asarray(cool[k].Ee) for k in cool)

        rec["t"].append(t); rec["dt"].append(t - t_prev)
        rec["I_loop"].append(float(sim._circuit_I_loop))
        rec["phi_c"].append(E0); rec["Ieth"].append(G0)
        rec["vent_coll_p"].append(float(-cbn[coll_cell] * Vp[coll_cell]))
        rec["vent_cath_p"].append(float(-cbn[cath_cell] * Vp[cath_cell]))
        rec["vent_coll_E"].append(float(-cbE[coll_cell] * Vp[coll_cell]))
        rec["vent_cath_E"].append(float(-cbE[cath_cell] * Vp[cath_cell]))
        rec["anode_p"].append(float(-(acn * Vp).sum()))
        rec["anode_E"].append(float(-(acE * Vp).sum()))
        rec["birth_bulk"].append(float((rxn * Vp).sum()))
        rec["birth_beam"].append(float((btn * Vp).sum()))
        rec["recomb_p"].append(float(-(rec_p * Vp).sum()))
        rec["beam_Edep"].append(float((btE * Vp).sum()))
        rec["cst_E"].append(float((cstE * Vp).sum()))
        rec["cool_E"].append(float((coolE * Vp).sum()))
        rec["N_col"].append(float((n * Vp)[active].sum()))
        rec["E_col"].append(Ecol)
        rec["N_cath"].append(float(n[cath_cell] * Vp[cath_cell]))
        rec["n_cath"].append(float(n[cath_cell]))
        cellrec["n"].append(n.copy())
        cellrec["Te"].append(np.asarray(d.Te, float).copy())
        cellrec["Ti"].append(np.asarray(d.Ti, float).copy())
        cellrec["nn"].append(np.asarray(st.nn, float).copy())
        t_prev = t
        if len(rec["t"]) > 400000:
            print("   BAILING: >400000 accepted steps"); break

    wall = time.time() - t0
    R = {k: np.asarray(v, float) for k, v in rec.items()}
    C = {k: np.asarray(v, float) for k, v in cellrec.items()}
    ns = R["t"].size
    print(f"   accepted steps: {ns}  ({wall:.0f} s wall)")
    if refusal:
        print(f"   REFUSAL at t={float(sim._time):.6g}: {refusal}")
    if not ns:
        return 0

    # ---- IDENTITY CONTROL against the recorded arm trace -------------------
    if args.ref_npz:
        ref = np.load(args.ref_npz, allow_pickle=True)
        tr = np.asarray(ref["trace"], float)
        m = min(tr.shape[0], ns)
        dt_max = float(np.max(np.abs(tr[:m, 0] - R["t"][:m])))
        dI_max = float(np.max(np.abs(tr[:m, 2] - R["I_loop"][:m])))
        same_steps = tr.shape[0] == ns
        print(f"   IDENTITY CONTROL vs {args.ref_npz}: steps {tr.shape[0]} vs "
              f"{ns} (match={same_steps}); max|dt|={dt_max:.3e} s, "
              f"max|dI_loop|={dI_max:.3e} A "
              f"{'-- IDENTICAL' if same_steps and dt_max == 0.0 and dI_max == 0.0 else '-- DIFFERS'}")

    t = R["t"]
    # onset per the efold1 pre-registered definition
    I = R["I_loop"]
    I_wall = float(np.interp(I_WALL_TIME, t, I))
    on = np.flatnonzero(I >= ONSET_FACTOR * I_wall)
    i_on = int(on[0]) if on.size else 0
    t_on = float(t[i_on]) if on.size else float("nan")
    print(f"   onset (efold1 def): t_on={t_on:.6g} s (step {i_on}), "
          f"I_wall={I_wall:.6g} A")

    def integ(y, i0=0):
        return float(np.trapezoid(y[i0:], t[i0:]))

    for name, i0 in (("FULL WINDOW", 0), ("BUILD WINDOW (onset->end)", i_on)):
        created_bulk = integ(R["birth_bulk"], i0)
        created_beam = integ(R["birth_beam"], i0)
        vent_coll = integ(R["vent_coll_p"], i0)
        vent_cath = integ(R["vent_cath_p"], i0)
        anode = integ(R["anode_p"], i0)
        recomb = integ(R["recomb_p"], i0)
        dN = R["N_col"][-1] - R["N_col"][i0]
        created = created_bulk + created_beam
        booked = vent_coll + vent_cath + anode + recomb + dN
        print(f"\n   -- PARTICLE BUDGET, {name} [{t[i0]:.4g}, {t[-1]:.4g}] s --")
        print(f"   created: bulk-thermal {created_bulk:.4e}  "
              f"beam-impact {created_beam:.4e}  total {created:.4e}")
        print(f"   vented collector-end {vent_coll:.4e} ({vent_coll/created:.3%})")
        print(f"   absorbed cathode-face {vent_cath:.4e} ({vent_cath/created:.3%})")
        print(f"   anode-mesh collection {anode:.4e} ({anode/created:.3%})")
        print(f"   recombination {recomb:.4e} ({recomb/created:.3%})")
        print(f"   accumulated dN_col {dN:.4e} ({dN/created:.3%})")
        print(f"   closure residual (created - booked)/created = "
              f"{(created-booked)/created:.3%}")
        Edep = integ(R["beam_Edep"], i0); EcstI = integ(R["cst_E"], i0)
        Ecool = integ(R["cool_E"], i0)
        Evc = integ(R["vent_coll_E"], i0); Evk = integ(R["vent_cath_E"], i0)
        Ean = integ(R["anode_E"], i0)
        dE = R["E_col"][-1] - R["E_col"][i0]
        print(f"   -- ENERGY [erg], same window: beam_dep {Edep:.4e}  "
              f"cathode_terms {EcstI:.4e}  cooling {Ecool:.4e}")
        print(f"      vent collector {Evc:.4e}  vent cathode-face {Evk:.4e}  "
              f"anode {Ean:.4e}  dE_col {dE:.4e}")

    # gain vs end-loss rate over the build
    Ncol = R["N_col"]
    lam = np.divide(R["vent_coll_p"], Ncol, out=np.zeros_like(Ncol),
                    where=Ncol > 0)
    lam_tot = np.divide(R["vent_coll_p"] + R["vent_cath_p"] + R["anode_p"],
                        Ncol, out=np.zeros_like(Ncol), where=Ncol > 0)
    mask = np.arange(ns) >= i_on
    if mask.sum() >= 2:
        gfit = float(np.polyfit(t[mask], np.log(Ncol[mask]), 1)[0])
    else:
        gfit = float("nan")
    print(f"\n   -- GAIN vs END-LOSS (build window) --")
    print(f"   d ln N_col/dt (LS fit) = {gfit:.6e} 1/s -> tau = "
          f"{1e6/gfit:.4f} us")
    for nm, arr in (("collector-only", lam), ("all-surface", lam_tot)):
        w = arr[mask]
        print(f"   end-loss rate {nm}: onset {arr[i_on]:.4e}  "
              f"median {np.median(w):.4e}  end {arr[-1]:.4e} 1/s")
        print(f"     GAIN/END-LOSS ratio (fit gain / median loss) = "
              f"{gfit/np.median(w):.4g}")

    np.savez(args.npz,
             **{k: v for k, v in R.items()},
             cell_n=C["n"], cell_Te=C["Te"], cell_Ti=C["Ti"], cell_nn=C["nn"],
             active=active, Vp=Vp, roles=np.asarray([str(r) for r in roles]),
             meta=np.array(json.dumps({
                 "label": args.label, "sets": applied, "kid": kid,
                 "t_on": t_on, "i_on": i_on, "wall_s": wall,
                 "refusal": refusal, "cath_cell": cath_cell,
                 "coll_cell": coll_cell})))
    print(f"   saved {args.npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
