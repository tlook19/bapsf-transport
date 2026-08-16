"""pd1owner: per-step DEPOSITION-SPLIT ledger on a pd1 arm (Q1 feedstock).

WRAPPER around the identical accepted-step loop efold1_traj.py /
pd0_endvent_traj.py / pd1_tailledger.py run (regime_r2_overlap_gate
.build_config + recertdiag2_traj.apply_set, ``sim.run(t_end, max_steps=1)``,
same exception handling). SINGLE-DELTA ON INSTRUMENTATION ONLY: after each
accepted step it reads the BeamDepositionResult rows off a cathode solve taken
with update_cache=False (the same non-mutating convention the pd0/pd1
instruments used), and the per-step (t, I_loop) trace is compared against the
arm's recorded efold1 npz as the IDENTITY CONTROL.

Per accepted step it records, summed over ends:
  scalars: t, I_loop, phi_c, tail_power, end_loss_tail_low, end_loss_tail_high,
           end_loss_low, end_loss_high, anode_intercepted
  per-cell rows [erg/s]: heating_coulomb, heating_anomalous, heating_secondary,
           heating_terminal, radiated, ionization_cost, plasma_heating,
           and the tail splits ionization_cost_tail, radiated_tail.

Writes only pd1owner_* artifacts. Read-only w.r.t. the repo.
"""
import argparse, json, sys, time, warnings
import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
from recertdiag2_traj import apply_set


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--t-target", type=float, default=1.0e-4)
    p.add_argument("--set", dest="sets", nargs="*", default=())
    p.add_argument("--npz", required=True)
    p.add_argument("--ref-npz", default=None)
    args = p.parse_args(argv)
    warnings.simplefilter("ignore")

    from regime_r2_overlap_gate import build_config
    params, flags = build_config(args.nx, False)
    applied = apply_set(params, flags, args.sets)

    from cablp.funcs import _kernels
    kid = str(getattr(_kernels.COMPILED_KERNELS, "KERNEL_ID",
                      _kernels.COMPILED_KERNELS))
    sim = LAPDSim1D(params, flags)
    cells = sim.geometry.cells
    print(f"== pd1owner_heatledger label={args.label} nx={args.nx} "
          f"t_target={args.t_target:g}")
    print(f"   deltas via --set: {applied}")
    print(f"   kernels: {kid}")

    ROWS = ("heating_coulomb_erg_s", "heating_anomalous_erg_s",
            "heating_secondary_erg_s", "heating_terminal_erg_s",
            "radiated_erg_s", "ionization_cost_erg_s",
            "plasma_heating_erg_s", "ionization_cost_tail_erg_s",
            "radiated_tail_erg_s")
    SCAL = ("tail_power_erg_s", "end_loss_tail_low_erg_s",
            "end_loss_tail_high_erg_s", "end_loss_low_erg_s",
            "end_loss_high_erg_s", "anode_intercepted_erg_s")
    scalars = []
    cellrows = {k: [] for k in ROWS}
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
        solve = sim.solve_cathode_boundary(state=sim.state, time=t,
                                           update_cache=False)
        E0 = float("nan")
        br = getattr(solve, "beam_result", None)
        res = getattr(br, "result", None)
        if res is not None:
            E0 = float(res.phi_c)
        acc_c = {k: np.zeros(cells) for k in ROWS}
        acc_s = {k: 0.0 for k in SCAL}
        for dep in (getattr(solve, "beam_deposition", None) or {}).values():
            if dep is None:
                continue
            for k in ROWS:
                acc_c[k] += np.asarray(getattr(dep, k), float)
            for k in SCAL:
                acc_s[k] += float(getattr(dep, k))
        scalars.append([t, float(sim._circuit_I_loop), E0]
                       + [acc_s[k] for k in SCAL])
        for k in ROWS:
            cellrows[k].append(acc_c[k])
        if len(scalars) > 400000:
            print("   BAILING: >400000 accepted steps")
            break

    wall = time.time() - t0
    S = np.asarray(scalars, float)
    C = {k: np.asarray(v, float) for k, v in cellrows.items()}
    ns = S.shape[0]
    print(f"   accepted steps: {ns}  ({wall:.0f} s wall)")
    if refusal:
        print(f"   REFUSAL at t={float(sim._time):.6g}: {refusal}")
    if not ns:
        return 0
    t, I = S[:, 0], S[:, 1]

    if args.ref_npz:
        ref = np.load(args.ref_npz, allow_pickle=True)
        tr = np.asarray(ref["trace"], float)
        m = min(tr.shape[0], ns)
        dt_max = float(np.max(np.abs(tr[:m, 0] - t[:m])))
        dI_max = float(np.max(np.abs(tr[:m, 2] - I[:m])))
        same = tr.shape[0] == ns
        print(f"   IDENTITY CONTROL vs {args.ref_npz}: steps {tr.shape[0]} vs "
              f"{ns} (match={same}); max|dt|={dt_max:.3e} s, "
              f"max|dI_loop|={dI_max:.3e} A "
              f"{'-- IDENTICAL' if same and dt_max == 0.0 and dI_max == 0.0 else '-- DIFFERS'}")

    # ledger identity in its shipped form, as a control
    anom = C["heating_anomalous_erg_s"].sum(axis=1)
    ion_t = C["ionization_cost_tail_erg_s"].sum(axis=1)
    rad_t = C["radiated_tail_erg_s"].sum(axis=1)
    led = S[:, 4] + S[:, 5]
    P_QL = anom + ion_t + rad_t + led
    launched = S[:, 3]
    live = P_QL > 0
    resid = np.abs((anom + ion_t + rad_t + led - P_QL)[live]) / P_QL[live]
    print(f"   identity control (trivial by construction here): "
          f"max = {resid.max():.1e}")
    print(f"   end_loss_tail_low (cathode end) total integral = "
          f"{np.trapezoid(S[:, 4], t):.6e} erg  "
          f"[phi_c keying predicts EXACTLY 0.0]")
    print(f"   end_loss_tail_high (collector end) total integral = "
          f"{np.trapezoid(S[:, 5], t):.6e} erg")
    for nm, i in (("tail_power", 3), ("anode_intercepted", 8)):
        print(f"   {nm} integral = {np.trapezoid(S[:, i], t):.6e} erg")
    for k in ("heating_coulomb_erg_s", "heating_anomalous_erg_s",
              "heating_secondary_erg_s", "heating_terminal_erg_s",
              "radiated_erg_s", "ionization_cost_erg_s"):
        print(f"   {k:28s} integral = "
              f"{np.trapezoid(C[k].sum(axis=1), t):.6e} erg")

    np.savez(args.npz, scalars=S,
             **{k: C[k] for k in ROWS},
             meta=np.array(json.dumps(
                 {"label": args.label, "sets": applied, "kid": kid,
                  "wall_s": wall, "refusal": refusal,
                  "scalar_cols": ["t", "I_loop", "phi_c"] + list(SCAL)})))
    print(f"   saved {args.npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
